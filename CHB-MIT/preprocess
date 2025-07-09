import os
import numpy as np
import pyedflib
import random
from collections import Counter

# Set paths
root_dir = "/home/zhuoying/chbmit_full"
output_dir = "/home/zhuoying/chbmit_preprocessed_raw"
os.makedirs(output_dir, exist_ok=True)

# Constants
segment_duration_sec = 2
fs_default = 256
n_channels = 23

# Helper function
def get_segments(signal, start_idx, end_idx, seg_len):
    segments = []
    for i in range(start_idx, end_idx - seg_len + 1, seg_len):
        seg = signal[:, i:i + seg_len]
        if seg.shape == (n_channels, seg_len):
            segments.append(seg)
    return segments

# Initialize per-version counters
version_counts = [0, 0, 0, 0]
version_targets = [[], [], [], []]

# Process each subject folder
chb_folders = sorted([f for f in os.listdir(root_dir) if f.startswith("chb")])
for chb in chb_folders:
    chb_path = os.path.join(root_dir, chb)
    summary_path = os.path.join(chb_path, f"{chb}-summary.txt")
    if not os.path.exists(summary_path):
        print(f"Missing summary for {chb}")
        continue

    seizure_files = []
    nonseizure_files = []
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()

    current_file = None
    seizure_times = []
    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            if current_file and seizure_times:
                seizure_files.append((os.path.join(chb_path, current_file), seizure_times))
            current_file = line.split(":")[1].strip()
            seizure_times = []
        elif "Number of Seizures in File:" in line and "0" in line:
            if current_file:
                nonseizure_files.append(os.path.join(chb_path, current_file))
        elif "Seizure" in line and "Start Time" in line:
            seizure_start = float(line.split(":")[1].strip().split()[0])
        elif "Seizure" in line and "End Time" in line:
            seizure_end = float(line.split(":")[1].strip().split()[0])
            seizure_times.append((seizure_start, seizure_end))
    if current_file and seizure_times:
        seizure_files.append((os.path.join(chb_path, current_file), seizure_times))

    # Extract seizure segments
    seizure_segments = []
    for edf_path, intervals in seizure_files:
        if not os.path.exists(edf_path):
            continue
        try:
            f = pyedflib.EdfReader(edf_path)
            fs = f.getSampleFrequency(0)
            sigbufs = np.array([f.readSignal(i) for i in range(f.signals_in_file)])[:n_channels]
            f._close()
            seg_len = int(segment_duration_sec * fs)
            for start, end in intervals:
                idx_start = int(start * fs)
                idx_end = int(end * fs)
                seizure_segments.extend(get_segments(sigbufs, idx_start, idx_end, seg_len))
        except Exception as e:
            print(f"Error reading {edf_path}: {e}")
            continue

    # Extract non-seizure segments
    background_segments = []
    for edf_path in nonseizure_files:
        if not os.path.exists(edf_path):
            continue
        if len(background_segments) >= 12 * len(seizure_segments):
            break
        try:
            f = pyedflib.EdfReader(edf_path)
            fs = f.getSampleFrequency(0)
            sigbufs = np.array([f.readSignal(i) for i in range(f.signals_in_file)])[:n_channels]
            duration = sigbufs.shape[1] // fs
            seg_len = int(segment_duration_sec * fs)
            for _ in range(100):
                if duration < 2:
                    continue
                start = random.randint(0, duration - segment_duration_sec)
                idx_start = int(start * fs)
                idx_end = idx_start + seg_len
                seg = sigbufs[:, idx_start:idx_end]
                if seg.shape == (n_channels, seg_len):
                    background_segments.append(seg)
                    if len(background_segments) >= 12 * len(seizure_segments):
                        break
            f._close()
        except Exception as e:
            print(f"Error reading {edf_path}: {e}")
            continue

    if not seizure_segments:
        print(f"Skipped {chb}: no seizure segments found.")
        continue
    if len(background_segments) < 3 * len(seizure_segments):
        print(f"Skipped {chb}: not enough non-seizure data.")
        continue

    # Combine and shuffle
    random.shuffle(seizure_segments)
    random.shuffle(background_segments)
    filtered = [(f, 1) for f in seizure_segments] + [(f, 0) for f in background_segments]
    random.shuffle(filtered)

    # Save each segment incrementally by assigning to folds
    for seg, label in filtered:
        fold = version_counts.index(min(version_counts))
        filename = f"X_v{fold+1}_{version_counts[fold]:06d}.npy"
        np.save(os.path.join(output_dir, filename), seg.astype(np.float32))
        version_targets[fold].append(label)
        version_counts[fold] += 1

    print(f"{chb}: saved {len(filtered)} segments.")

# Save label arrays
for i in range(4):
    y = np.array(version_targets[i])
    np.save(os.path.join(output_dir, f"y_v{i+1}.npy"), y)
    print(f"Saved labels for v{i+1}: {len(y)} entries")

    
