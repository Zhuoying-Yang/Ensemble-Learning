import os
from scipy.io import loadmat

# -----------------------------
# Configuration
# -----------------------------

bids_root = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data_bids"
mat_root = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data"

sampling_rate = 1024  # Hz
chunk_duration_sec = 3600  # 1 hour per chunk

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
# Process Subjects
# -----------------------------

for subject in subjects:
    subj_label = subject.replace('HUP', '').replace('_phaseII', '').replace('b', 'b')
    eeg_dir = os.path.join(bids_root, f"sub-{subj_label}", "ses-phaseII", "eeg")
    mat_file = os.path.join(mat_root, subject, f"{subject}.mat")

    if not os.path.exists(eeg_dir):
        print(f"Skipping {subject} — EEG folder missing.")
        continue

    summary_lines = []
    summary_lines.append(f"Data Sampling Rate: {sampling_rate} Hz")
    summary_lines.append("*" * 25 + "\n")

    # Load seizure onsets (if any)
    seizure_starts = []
    if os.path.exists(mat_file):
        try:
            mat = loadmat(mat_file)
            tszr = mat.get('tszr', [])
            if len(tszr) > 0:
                seizure_starts = [int(round(entry[0][0][0])) for entry in tszr]
        except Exception as e:
            print(f"Error loading {subject}: {e}")

    # Get sorted list of EDF files with their numeric run indices
    edf_files = []
    for f in os.listdir(eeg_dir):
        if f.endswith("_eeg.edf"):
            try:
                idx = int(f.split("run-")[1].split("_")[0])
                edf_files.append((idx, f))
            except Exception:
                print(f"Skipping invalid EDF: {f}")

    edf_files_sorted = sorted(edf_files, key=lambda x: x[0])

    # For each EDF file, check for seizures and write summary
    for chunk_idx, (run_idx, edf_file) in enumerate(edf_files_sorted):
        file_start_sample = chunk_idx * chunk_duration_sec * sampling_rate
        file_end_sample = (chunk_idx + 1) * chunk_duration_sec * sampling_rate

        summary_lines.append(f"File Name: {edf_file}")
        summary_lines.append(f"File Start Time: {chunk_idx:02d}:00:00")
        summary_lines.append(f"File End Time: {chunk_idx + 1:02d}:00:00")

        seizures_in_file = []
        for sz_start in seizure_starts:
            if file_start_sample <= sz_start < file_end_sample:
                onset_in_file_sec = (sz_start - file_start_sample) / sampling_rate
                seizures_in_file.append(int(onset_in_file_sec))

        summary_lines.append(f"Number of Seizures in File: {len(seizures_in_file)}")
        for onset_sec in sorted(seizures_in_file):
            summary_lines.append(f"Seizure Start Time: {onset_sec} seconds")
        summary_lines.append("")  # Blank line between files

    # Save summary.txt
    with open(os.path.join(eeg_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print(f"Saved summary.txt for {subject}")

print("\nAll summaries generated.")
