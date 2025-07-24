import h5py
import numpy as np
import random
import os
import gc
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

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
segment_len = 4 * fs  # 4 seconds
lowcut, highcut = 0.5, 40
three_hours = 3 * 60 * 60 * fs
four_hours = 4 * 60 * 60 * fs
non_seizure_ratio = 3
threshold_max = 1000  # µV

# -----------------------------
# Helpers
# -----------------------------
def bandpass(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

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
        seizure_starts = [int(row[0].item()) for row in tszr]
        print(f"Found {len(seizure_starts)} annotated event(s)")
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

    for rep in range(4):
        print(f"\nPreparing dataset version {rep+1}/4...")
        segments_rep, labels_rep = [], []

        # Seizure segments (label = 1)
        seizure_count = 0
        for start in seizure_starts:
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
                segments_rep.append(seg.astype(np.float32))
                labels_rep.append(1)
                seizure_count += 1

        print(f"Collected {seizure_count} seizure segments")

        # Preictal segments (label = 2)
        preictal_count = 0
        preictal_end_offset = 2 * 60 * fs
        preictal_max_start_offset = 5 * 60 * fs

        for start in seizure_starts:
            pre_end = int(start - preictal_end_offset)
            pre_start = max(0, int(start - preictal_max_start_offset))

            if pre_end <= pre_start or pre_end > total_len:
                continue

            overlaps = False
            for other_start in seizure_starts:
                if abs(other_start - start) < 1e-3:
                    continue  # skip self
                other_end = other_start + 120 * fs
                if pre_end > other_start and pre_start < other_end:
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
                segments_rep.append(seg.astype(np.float32))
                labels_rep.append(2)
                preictal_count += 1

        print(f"Collected {preictal_count} preictal segments (as long as 2–5 min available)")

        # Non-seizure segments (label = 0)
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

            segments_rep.append(raw.astype(np.float32))
            labels_rep.append(0)
            nonseizure_count += 1

        print(f"Final collected segments — Seizure: {seizure_count}, Preictal: {preictal_count}, Non-seizure: {nonseizure_count}")

        base = f"{folder}_LEAR1_REAR1_v{rep+1}_raw_three"
        np.save(f"{base}_segments.npy", np.array(segments_rep, dtype=np.float32))
        np.save(f"{base}_labels.npy", np.array(labels_rep, dtype=np.int64))
        print(f"Saved {len(labels_rep)} total segments → {base}_segments.npy / {base}_labels.npy")

        del segments_rep, labels_rep
        gc.collect()
