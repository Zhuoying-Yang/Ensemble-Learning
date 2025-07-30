import os
import gc
import random
import numpy as np
import torch
import h5py
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import entropy, kurtosis
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# -----------------------------
# CONFIGURATION
# -----------------------------
base_path = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data/"
save_dir = "/home/zhuoying/preprocessed_data"
os.makedirs(save_dir, exist_ok=True)

folders = [
    "HUP262b_phaseII", "HUP267_phaseII", "HUP269_phaseII", "HUP270_phaseII",
    "HUP271_phaseII", "HUP272_phaseII", "HUP273_phaseII", "HUP273c_phaseII"
]

fs = 1024
segment_len = 2 * fs
lowcut, highcut = 0.5, 40
three_hours = 3 * 60 * 60 * fs
four_hours = 4 * 60 * 60 * fs
non_seizure_ratio = 3
threshold_max = 1000
gap_threshold = 2 * 60 * fs

# -----------------------------
# FILTER & FEATURE FUNCTIONS
# -----------------------------
def bandpass(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def extract_features(seg, fs=1024):
    features = []
    for ch_data in seg:
        ch_data = np.asarray(ch_data)
        ch_abs = np.abs(ch_data)
        features.extend([
            np.mean(ch_data),
            np.var(ch_data),
            np.max(ch_data),
            np.min(ch_data),
            np.median(ch_data),
            entropy(ch_abs + 1e-8),
            kurtosis(ch_data),
            np.sum(ch_data ** 2),
            ((ch_data[:-1] * ch_data[1:]) < 0).sum(),
        ])
        f, psd = welch(ch_data, fs=fs, nperseg=fs//2)
        def bandpower(low, high):
            return np.trapz(psd[(f >= low) & (f <= high)], f[(f >= low) & (f <= high)])
        features.extend([
            bandpower(0.5, 4), bandpower(4, 8),
            bandpower(8, 13), bandpower(13, 30),
            bandpower(30, 40)
        ])
    return np.array(features, dtype=np.float32)  # flat, no reshape

# -----------------------------
# MAIN EXTRACTION
# -----------------------------
version_suffixes = ["v1", "v2", "v3", "v4"]
merged_val_data, merged_val_labels = [], []
merged_test_data, merged_test_labels = [], []

for folder in folders:
    print(f"\nProcessing {folder}...")
    folder_path = os.path.join(base_path, folder)
    label_path = os.path.join(folder_path, folder + ".mat")

    try:
        mat = loadmat(label_path)
        seizure_starts = sorted([int(row[0].item() * fs) for row in mat.get("tszr", [])])
    except:
        print(f"⚠️ Failed to load label for {folder}")
        continue

    try:
        with h5py.File(os.path.join(folder_path, "LEAR1.mat"), 'r') as f:
            eeg_lear1 = np.squeeze(f[list(f.keys())[0]][()])
        with h5py.File(os.path.join(folder_path, "REAR1.mat"), 'r') as f:
            eeg_rear1 = np.squeeze(f[list(f.keys())[0]][()])
        min_len = min(len(eeg_lear1), len(eeg_rear1))
        eeg_all = np.stack([eeg_lear1[:min_len], eeg_rear1[:min_len]], axis=0)
    except:
        print(f"⚠️ Failed to load EEG files for {folder}")
        continue

    total_len = eeg_all.shape[1]
    grouped_seizure_starts = []
    for s in seizure_starts:
        if not grouped_seizure_starts or s - grouped_seizure_starts[-1] > gap_threshold:
            grouped_seizure_starts.append(s)

    for rep in range(4):
        print(f"→ Dataset version {rep+1}")
        X_rep, y_rep = [], []

        # Seizure segments
        for start in grouped_seizure_starts:
            end = start + 120 * fs
            if end > total_len:
                continue
            raw = eeg_all[:, start:end].copy()
            for ch in range(2):
                if np.isnan(raw[ch]).any():
                    if np.all(np.isnan(raw[ch])): break
                    raw[ch] = np.interp(np.arange(len(raw[ch])),
                                        np.flatnonzero(~np.isnan(raw[ch])),
                                        raw[ch][~np.isnan(raw[ch])])
                raw[ch] -= np.mean(raw[ch])
                raw[ch] = bandpass(raw[ch], lowcut, highcut, fs)
            else:
                for i in range(0, raw.shape[1] - segment_len + 1, segment_len):
                    seg = raw[:, i:i+segment_len]
                    if np.max(np.abs(seg)) <= threshold_max:
                        X_rep.append(extract_features(seg))
                        y_rep.append(1)

        # Preictal segments
        for start in grouped_seizure_starts:
            pre_start = max(0, start - 3 * 60 * fs)
            pre_end = start
            if pre_end <= pre_start or pre_end > total_len:
                continue
            if any(pre_start < s + 120 * fs and pre_end > s for s in seizure_starts):
                continue
            raw = eeg_all[:, pre_start:pre_end].copy()
            for ch in range(2):
                if np.isnan(raw[ch]).any():
                    if np.all(np.isnan(raw[ch])): break
                    raw[ch] = np.interp(np.arange(len(raw[ch])),
                                        np.flatnonzero(~np.isnan(raw[ch])),
                                        raw[ch][~np.isnan(raw[ch])])
                raw[ch] -= np.mean(raw[ch])
                raw[ch] = bandpass(raw[ch], lowcut, highcut, fs)
            else:
                for i in range(0, raw.shape[1] - segment_len + 1, segment_len):
                    seg = raw[:, i:i+segment_len]
                    if np.max(np.abs(seg)) <= threshold_max:
                        X_rep.append(extract_features(seg))
                        y_rep.append(2)

        # Non-seizure segments
        non_count = 0
        tries = 0
        needed = non_seizure_ratio * y_rep.count(1)
        while non_count < needed and tries < 200000:
            i = random.randint(0, total_len - segment_len - 1)
            if all(i < s - three_hours or i > s + four_hours for s in seizure_starts):
                raw = eeg_all[:, i:i+segment_len].copy()
                for ch in range(2):
                    if np.isnan(raw[ch]).any():
                        if np.all(np.isnan(raw[ch])): break
                        raw[ch] = np.interp(np.arange(len(raw[ch])),
                                            np.flatnonzero(~np.isnan(raw[ch])),
                                            raw[ch][~np.isnan(raw[ch])])
                    raw[ch] -= np.mean(raw[ch])
                    raw[ch] = bandpass(raw[ch], lowcut, highcut, fs)
                else:
                    if np.max(np.abs(raw)) <= threshold_max:
                        X_rep.append(extract_features(raw))
                        y_rep.append(0)
                        non_count += 1
            tries += 1

        X = np.array(X_rep, dtype=np.float32)
        y = np.array(y_rep, dtype=np.int64)
        del X_rep, y_rep
        gc.collect()

        # Split and Save
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.08, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1/0.92, stratify=y_trainval, random_state=42)
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        base = f"{folder}_LEAR1_REAR1_v{rep+1}_feature_change_three"
        torch.save((torch.tensor(X_train_sm), torch.tensor(y_train_sm)), f"{save_dir}/train_{base}.pt")
        torch.save((torch.tensor(X_val), torch.tensor(y_val)), f"{save_dir}/val_{base}.pt")
        torch.save((torch.tensor(X_test), torch.tensor(y_test)), f"{save_dir}/test_{base}.pt")

        merged_val_data.append(torch.tensor(X_val))
        merged_val_labels.append(torch.tensor(y_val))
        merged_test_data.append(torch.tensor(X_test))
        merged_test_labels.append(torch.tensor(y_test))

        print(f"Saved {base}")

# Save merged val/test sets
torch.save((torch.cat(merged_val_data), torch.cat(merged_val_labels)), f"{save_dir}/merged_val_feature_change_three.pt")
torch.save((torch.cat(merged_test_data), torch.cat(merged_test_labels)), f"{save_dir}/merged_test_feature_change_three.pt")
print("Saved merged validation and test sets")
