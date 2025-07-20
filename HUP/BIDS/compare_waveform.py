import os
import numpy as np
import h5py
import mne
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
subject = "HUP262b_phaseII" 
base_input = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data"
base_output = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data_bids"
fs = 1024  # Hz
compare_seconds = 10  # Compare first 10 seconds
compare_samples = compare_seconds * fs

tags = ['LA', 'LC', 'LEAR', 'REAR']

# -----------------------------
# Helper Functions
# -----------------------------
def load_and_preprocess_channel(file_path):
    with h5py.File(file_path, 'r') as f:
        key = list(f.keys())[0]
        data = np.squeeze(f[key][()]).astype(np.float32)
        if not np.all(np.isfinite(data)):
            idx = np.isfinite(data)
            if np.any(idx):
                data[~idx] = np.interp(np.flatnonzero(~idx), np.flatnonzero(idx), data[idx])
            else:
                data = np.zeros_like(data)
        # Apply your saved scaling and clipping
        data = data * 1e-5
        data = np.clip(data, -500, 500)
        return data

# -----------------------------
# Load and Stack Original Data
# -----------------------------
subject_path = os.path.join(base_input, subject)
raw_data_list = []
channel_names = []

for filename in sorted(os.listdir(subject_path)):
    if filename.endswith('.mat') and any(tag in filename for tag in tags):
        file_path = os.path.join(subject_path, filename)
        data = load_and_preprocess_channel(file_path)
        raw_data_list.append(data[:compare_samples])
        channel_names.append(filename.replace('.mat', ''))

original_data = np.stack(raw_data_list, axis=0)

# -----------------------------
# Load Exported EDF
# -----------------------------
subject_label = subject.replace('HUP', '').replace('_phaseII', '').replace('b', 'b')
edf_path = os.path.join(base_output, f"sub-{subject_label}", "ses-phaseII", "eeg",
                        f"sub-{subject_label}_ses-phaseII_task-phaseII_run-01_eeg.edf")
raw_edf = mne.io.read_raw_edf(edf_path, preload=True)
edf_data = raw_edf.get_data()[:, :compare_samples]

# -----------------------------
# Plot for Comparison
# -----------------------------
fig, axes = plt.subplots(len(channel_names), 1, figsize=(12, 2 * len(channel_names)), sharex=True)

time = np.arange(compare_samples) / fs

for idx, ch in enumerate(channel_names):
    axes[idx].plot(time, original_data[idx], label='Original (MAT)', alpha=0.7)
    axes[idx].plot(time, edf_data[idx], label='Exported (EDF)', alpha=0.7, linestyle='--')
    axes[idx].set_title(ch)
    axes[idx].legend()

plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()
plt.savefig("comparison_plot.png", dpi=300)
