import os
import numpy as np
import torch
import h5py
import gc
import random
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import entropy, kurtosis, skew
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ========== CONFIG ==========
base_path = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data"
save_dir = "/home/zhuoying/preprocessed_data"
os.makedirs(save_dir, exist_ok=True)

folders = [
    "HUP262b_phaseII", "HUP267_phaseII", "HUP269_phaseII",
    "HUP270_phaseII", "HUP271_phaseII", "HUP272_phaseII",
    "HUP273_phaseII", "HUP273c_phaseII"
]
version_suffixes = ["v1", "v2", "v3", "v4"]
fs = 1024
segment_len = 2 * fs
lowcut, highcut = 0.5, 40
threshold_max = 1000
three_hours = 3 * 60 * 60 * fs
four_hours = 4 * 60 * 60 * fs
gap_threshold = 2 * 60 * fs
non_seizure_ratio = 3

# ========== Helpers ==========
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
            np.mean(ch_data), np.var(ch_data),  # std -> var
            np.max(ch_data), np.min(ch_data), np.median(ch_data),
            kurtosis(ch_data),
            np.sum(ch_data**2), ((ch_data[:-1] * ch_data[1:]) < 0).sum()
        ])
        f, psd = welch(ch_data, fs=fs, nperseg=fs//2)
        def bandpower(low, high):
            return np.trapz(psd[(f >= low) & (f <= high)], f[(f >= low) & (f <= high)])
        features.extend([
            bandpower(0.5, 4), bandpower(4, 8), bandpower(8, 13),
            bandpower(13, 30), bandpower(30, 40)
        ])
    return features

# ========== MAIN ==========
merged_val_data, merged_val_labels = [], []
merged_test_data, merged_test_labels = [], []

for rep, version in enumerate(version_suffixes):
    features_all, labels_all = [], []

    for folder in folders:
        print(f"[{version}] Processing {folder}...")
        folder_path = os.path.join(base_path, folder)
        label_path = os.path.join(folder_path, folder + ".mat")

        try:
            mat = loadmat(label_path)
            tszr = mat.get("tszr", [])
            seizure_starts = sorted([int(row[0].item() * fs) for row in tszr])
        except:
            continue

        try:
            with h5py.File(os.path.join(folder_path, "LEAR1.mat"), 'r') as f:
                eeg_lear1 = np.squeeze(f[list(f.keys())[0]][()])
            with h5py.File(os.path.join(folder_path, "REAR1.mat"), 'r') as f:
                eeg_rear1 = np.squeeze(f[list(f.keys())[0]][()])
            min_len = min(len(eeg_lear1), len(eeg_rear1))
            eeg_all = np.stack([eeg_lear1[:min_len], eeg_rear1[:min_len]], axis=0)
        except:
            continue

        total_len = eeg_all.shape[1]
        grouped = []
        for s in seizure_starts:
            if not grouped or s - grouped[-1] > gap_threshold:
                grouped.append(s)

        for start in grouped:
            end = start + 120 * fs
            if end > total_len: continue
            raw = eeg_all[:, start:end].copy()
            for ch in range(2):
                if np.isnan(raw[ch]).any():
                    if np.all(np.isnan(raw[ch])): break
                    raw[ch] = np.interp(np.arange(len(raw[ch])), np.flatnonzero(~np.isnan(raw[ch])), raw[ch][~np.isnan(raw[ch])])
                raw[ch] -= np.mean(raw[ch])
                raw[ch] = bandpass(raw[ch], lowcut, highcut, fs)
            n = raw.shape[1] // segment_len
            for i in range(n):
                seg = raw[:, i * segment_len:(i + 1) * segment_len]
                if np.max(np.abs(seg)) < threshold_max:
                    features_all.append(extract_features(seg))
                    labels_all.append(1)

        for start in grouped:
            pre_start, pre_end = max(0, start - 3 * 60 * fs), start
            if pre_end <= pre_start or pre_end > total_len: continue
            if any(pre_start < s + 120*fs and pre_end > s for s in seizure_starts): continue
            raw = eeg_all[:, pre_start:pre_end].copy()
            for ch in range(2):
                if np.isnan(raw[ch]).any():
                    if np.all(np.isnan(raw[ch])): break
                    raw[ch] = np.interp(np.arange(len(raw[ch])), np.flatnonzero(~np.isnan(raw[ch])), raw[ch][~np.isnan(raw[ch])])
                raw[ch] -= np.mean(raw[ch])
                raw[ch] = bandpass(raw[ch], lowcut, highcut, fs)
            n = raw.shape[1] // segment_len
            for i in range(n):
                seg = raw[:, i * segment_len:(i + 1) * segment_len]
                if np.max(np.abs(seg)) < threshold_max:
                    features_all.append(extract_features(seg))
                    labels_all.append(2)

        n_non = len([l for l in labels_all if l == 1]) * non_seizure_ratio
        non_idxs, attempts = set(), 0
        while len(non_idxs) < n_non and attempts < 200000:
            i = random.randint(0, total_len - segment_len - 1)
            if all(i < s - three_hours or i > s + four_hours for s in seizure_starts):
                non_idxs.add(i)
            attempts += 1
        for i in non_idxs:
            raw = eeg_all[:, i:i + segment_len].copy()
            for ch in range(2):
                if np.isnan(raw[ch]).any():
                    if np.all(np.isnan(raw[ch])): break
                    raw[ch] = np.interp(np.arange(len(raw[ch])), np.flatnonzero(~np.isnan(raw[ch])), raw[ch][~np.isnan(raw[ch])])
                raw[ch] -= np.mean(raw[ch])
                raw[ch] = bandpass(raw[ch], lowcut, highcut, fs)
            if np.max(np.abs(raw)) < threshold_max:
                features_all.append(extract_features(raw))
                labels_all.append(0)

    X = np.array(features_all, dtype=np.float32)
    y = np.array(labels_all, dtype=np.int64)

    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.08, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1/0.92, stratify=y_trainval, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    train_X = torch.tensor(X_train_sm, dtype=torch.float32)
    train_y = torch.tensor(y_train_sm, dtype=torch.long)
    val_X = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.long)
    test_X = torch.tensor(X_test, dtype=torch.float32)
    test_y = torch.tensor(y_test, dtype=torch.long)

    torch.save((train_X, train_y), f"{save_dir}/train_{version}_feature_change1_three.pt")
    torch.save((val_X, val_y), f"{save_dir}/val_{version}_feature_change1_three.pt")
    torch.save((test_X, test_y), f"{save_dir}/test_{version}_feature_change1_three.pt")

    print(f"[{version}] Saved train/val/test with _feature_change1_three suffix.")

    merged_val_data.append(val_X)
    merged_val_labels.append(val_y)
    merged_test_data.append(test_X)
    merged_test_labels.append(test_y)
    gc.collect()

# ========== MERGE AND SAVE ==========
torch.save((torch.cat(merged_val_data, dim=0), torch.cat(merged_val_labels, dim=0)),
           f"{save_dir}/merged_val_feature_change1_three.pt")
torch.save((torch.cat(merged_test_data, dim=0), torch.cat(merged_test_labels, dim=0)),
           f"{save_dir}/merged_test_feature_change1_three.pt")
print("Merged val/test sets saved with _feature_change1_three suffix.")
