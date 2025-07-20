import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

# -----------------------------
# Configuration
# -----------------------------

bids_root = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data_bids"
mat_root = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data"

sampling_rate = 1024  # Hz
chunk_duration_sec = 1800*2  # 60 minutes
seizure_duration_sec = 120  # As used in your preprocessing

subjects = [
    "HUP262b_phaseII",
    "HUP267_phaseII",
    "HUP269_phaseII",
    "HUP270_phaseII",
    "HUP271_phaseII",
    "HUP272_phaseII",
    "HUP273_phaseII",
    "HUP273c_phaseII"
]

# -----------------------------
# Main Loop per Subject
# -----------------------------

for subject in subjects:
    subj_label = subject.replace('HUP', '').replace('_phaseII', '').replace('b', 'b')
    eeg_dir = os.path.join(bids_root, f"sub-{subj_label}", "ses-phaseII", "eeg")
    mat_file = os.path.join(mat_root, subject, f"{subject}.mat")

    if not os.path.exists(eeg_dir) or not os.path.exists(mat_file):
        print(f"Skipping {subject} (missing data)")
        continue

    summary_lines = []
    summary_lines.append(f"Data Sampling Rate: {sampling_rate} Hz")
    summary_lines.append("*" * 25 + "\n")

    # Channels in EDF (if available)
    channels_file = os.path.join(eeg_dir, f"sub-{subj_label}_ses-phaseII_task-phaseII_channels.tsv")
    if os.path.exists(channels_file):
        ch_df = pd.read_csv(channels_file, sep="\t")
        summary_lines.append("Channels in EDF Files:")
        summary_lines.append("*" * 23)
        for idx, row in ch_df.iterrows():
            summary_lines.append(f"Channel {idx+1}: {row['name']}")
        summary_lines.append("")
    else:
        summary_lines.append("Channels in EDF Files: [Not Found]\n")

    # Load seizure annotations
    try:
        mat = loadmat(mat_file)
        tszr = mat.get("tszr", [])
        seizure_starts = [int(row[0].item()) for row in tszr] if len(tszr) > 0 else []
    except Exception as e:
        print(f"Failed to load seizure info for {subject}: {e}")
        seizure_starts = []

    # Check EDFs
    edf_files = sorted([f for f in os.listdir(eeg_dir) if f.endswith("_eeg.edf")])
    current_time_sec = 0

    for edf_file in edf_files:
        run_idx = int(edf_file.split("run-")[1].split("_")[0])
        file_start_sample = (run_idx - 1) * chunk_duration_sec * sampling_rate
        file_end_sample = run_idx * chunk_duration_sec * sampling_rate

        summary_lines.append(f"File Name: {edf_file}")
        summary_lines.append(f"File Start Time: {int(current_time_sec // 3600):02d}:{int((current_time_sec % 3600) // 60):02d}:{int(current_time_sec % 60):02d}")
        end_time_sec = current_time_sec + chunk_duration_sec
        summary_lines.append(f"File End Time: {int(end_time_sec // 3600):02d}:{int((end_time_sec % 3600) // 60):02d}:{int(end_time_sec % 60):02d}")

        seizures_in_file = []
        for sz_start in seizure_starts:
            sz_end = sz_start + seizure_duration_sec * sampling_rate
            if sz_end > file_start_sample and sz_start < file_end_sample:
                onset_in_file_sec = max(0, (sz_start - file_start_sample) / sampling_rate)
                end_in_file_sec = min(chunk_duration_sec, (sz_end - file_start_sample) / sampling_rate)
                seizures_in_file.append((onset_in_file_sec, end_in_file_sec))

        summary_lines.append(f"Number of Seizures in File: {len(seizures_in_file)}")
        for onset, end in seizures_in_file:
            summary_lines.append(f"Seizure Start Time: {int(onset)} seconds")
            summary_lines.append(f"Seizure End Time: {int(end)} seconds")

        summary_lines.append("")
        current_time_sec += chunk_duration_sec

    # Save summary.txt
    with open(os.path.join(eeg_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print(f"Generated summary.txt for {subject}")

print("\nAll subject summaries created.")
