import h5py
import numpy as np
import random
import os
import gc
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from scipy.stats import entropy, kurtosis, skew

# -----------------------------
# Configuration
# -----------------------------
base_path = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data/"
folders = [
    "HUP262b_phaseII",
    "HUP267_phaseII",
    "HUP269_phaseII",
    "HUP270_phaseII",
    "HUP271_phaseII",
    "HUP272_phaseII",
    "HUP273_phaseII",
    "HUP273c_phaseII"
]

fs = 1024
segment_len = 2 * fs  # 2 seconds
lowcut, highcut = 0.5, 40
three_hours = 3 * 60 * 60 * fs
four_hours = 4 * 60 * 60 * fs
non_seizure_ratio = 3
threshold_max = 1000  # µV
gap_threshold = 2 * 60 * fs  # 2 minutes

# -----------------------------
# Helpers
# -----------------------------
def bandpass(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def extract_features(seg):
    features = []
    for ch_data in seg:
        features.extend([
            np.mean(ch_data),
            np.std(ch_data),
            entropy(np.abs(ch_data) + 1e-8),
            kurtosis(ch_data),
            skew(ch_data)
        ])
    return features

# -----------------------------
# Main Loop over folders
# -----------------------------
for folder in folders:
    print(f"\nProcessing {folder}...")
    folder_path = os.path.join(base_path, folder)
    label_path = os.path.join(folder_path, folder + ".mat")

    try:
        mat = loadmat(label_path)
        tszr = mat.get("tszr", [])
        seizure_starts = [int(row[0].item() * fs) for row in tszr]
        seizure_starts = sorted(seizure_starts)
        print(f"Found {len(seizure_starts)} annotated events")
    except Exception as e:
        print(f"Failed to load label: {e}")
        continue

    print("Reading LEAR1.mat and REAR1.mat...")
    try:
        with h5py.File(os.path.join(folder_path, "LEAR1.mat"), 'r') as f:
            eeg_lear1 = np.squeeze(f[list(f.keys())[0]][()])
        with h5py.File(os.path.join(folder_path, "REAR1.mat"), 'r') as f:
            eeg_rear1 = np.squeeze(f[list(f.keys())[0]][()])

        min_len = min(len(eeg_lear1), len(eeg_rear1))
        eeg_all = np.stack([eeg_lear1[:min_len], eeg_rear1[:min_len]], axis=0)
    except Exception as e:
        print(f"Failed to load EEG pair: {e}")
        continue

    total_len = eeg_all.shape[1]

    grouped_seizure_starts = []
    for s in seizure_starts:
        if not grouped_seizure_starts or s - grouped_seizure_starts[-1] > gap_threshold:
            grouped_seizure_starts.append(s)
    print(f"Grouped into {len(grouped_seizure_starts)} seizure events")

    for rep in range(4):
        print(f"\nPreparing dataset version {rep+1}/4...")
        features_rep, labels_rep = [], []

        seizure_count = 0
        for start in grouped_seizure_starts:
            end = start + 120 * fs
            if end > total_len:
                continue
            raw = eeg_all[:, start:end].copy()
            valid = True
            for ch in range(2):
                if np.isnan(raw[ch]).any():
                    if np.all(np.isnan(raw[ch])):
                        valid = False
                        break
                    raw[ch] = np.interp(np.arange(len(raw[ch])),
                                        np.flatnonzero(~np.isnan(raw[ch])),
                                        raw[ch][~np.isnan(raw[ch])])
                raw[ch] -= np.mean(raw[ch])
                raw[ch] = bandpass(raw[ch], lowcut, highcut, fs)
            if not valid:
                continue

            n = raw.shape[1] // segment_len
            raw = raw[:, :n * segment_len]
            for seg_idx in range(n):
                seg = raw[:, seg_idx * segment_len: (seg_idx + 1) * segment_len]
                if np.max(np.abs(seg)) > threshold_max:
                    continue
                features = extract_features(seg)
                features_rep.append(features)
                labels_rep.append(1)
                seizure_count += 1

        print(f"Collected {seizure_count} seizure segments")

        preictal_count = 0
        preictal_max_offset = 3 * 60 * fs
        for start in grouped_seizure_starts:
            pre_start = max(0, int(start - preictal_max_offset))
            pre_end = int(start)

            if pre_end <= pre_start or pre_end > total_len:
                continue

            overlaps = False
            for other in seizure_starts:
                other_end = other + 120 * fs
                if pre_end > other and pre_start < other_end:
                    overlaps = True
                    break
            if overlaps:
                continue

            raw = eeg_all[:, pre_start:pre_end].copy()
            valid = True
            for ch in range(2):
                if np.isnan(raw[ch]).any():
                    if np.all(np.isnan(raw[ch])):
                        valid = False
                        break
                    raw[ch] = np.interp(np.arange(len(raw[ch])),
                                        np.flatnonzero(~np.isnan(raw[ch])),
                                        raw[ch][~np.isnan(raw[ch])])
                raw[ch] -= np.mean(raw[ch])
                raw[ch] = bandpass(raw[ch], lowcut, highcut, fs)
            if not valid:
                continue

            n = raw.shape[1] // segment_len
            raw = raw[:, :n * segment_len]
            for seg_idx in range(n):
                seg = raw[:, seg_idx * segment_len: (seg_idx + 1) * segment_len]
                if np.max(np.abs(seg)) > threshold_max:
                    continue
                features = extract_features(seg)
                features_rep.append(features)
                labels_rep.append(2)
                preictal_count += 1

        print(f"Collected {preictal_count} preictal segments")

        n_non = seizure_count * non_seizure_ratio
        non_idxs, attempts = set(), 0
        while len(non_idxs) < n_non and attempts < 200000:
            i = random.randint(0, total_len - segment_len - 1)
            if all(i < s - three_hours or i > s + four_hours for s in seizure_starts):
                non_idxs.add(i)
            attempts += 1

        print(f"Collected {len(non_idxs)} non-seizure candidates after {attempts} attempts")

        nonseizure_count = 0
        for i in non_idxs:
            raw = eeg_all[:, i:i + segment_len].copy()
            valid = True
            for ch in range(2):
                if np.isnan(raw[ch]).any():
                    if np.all(np.isnan(raw[ch])):
                        valid = False
                        break
                    raw[ch] = np.interp(np.arange(len(raw[ch])),
                                        np.flatnonzero(~np.isnan(raw[ch])),
                                        raw[ch][~np.isnan(raw[ch])])
                raw[ch] -= np.mean(raw[ch])
                raw[ch] = bandpass(raw[ch], lowcut, highcut, fs)
            if not valid:
                continue

            if np.max(np.abs(raw)) > threshold_max:
                continue

            features = extract_features(raw)
            features_rep.append(features)
            labels_rep.append(0)
            nonseizure_count += 1

        print(f"Final collected segments — Seizure: {seizure_count}, Preictal: {preictal_count}, Non-seizure: {nonseizure_count}")

        base = f"{folder}_LEAR1_REAR1_v{rep+1}_feature_three"
        np.save(f"{base}_features.npy", np.array(features_rep, dtype=np.float32))
        np.save(f"{base}_labels.npy", np.array(labels_rep, dtype=np.int64))
        print(f"Saved {len(labels_rep)} total features → {base}_features.npy / {base}_labels.npy")

        del features_rep, labels_rep
        gc.collect()
