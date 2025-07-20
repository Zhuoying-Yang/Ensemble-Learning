import os
import numpy as np
import h5py
import mne
import pandas as pd
import gc

# -----------------------------
# Configuration
# -----------------------------
base_input = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data"
base_output = "/home/zhuoying/projects/def-xilinliu/data/UPenn_data_bids"
os.makedirs(base_output, exist_ok=True)

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

fs = 1024  # Hz
chunk_size = 1  # Channels to load at once
export_chunk_sec = 1800*2  # 30 minutes = 1800 seconds

# -----------------------------
# Helper Functions
# -----------------------------
def load_channel(file_path):
    with h5py.File(file_path, 'r') as f:
        key = list(f.keys())[0]
        data = np.squeeze(f[key][()]).astype(np.float32)
        return data

def clean_signal(data):
    if not np.all(np.isfinite(data)):
        if np.all(np.isnan(data)):
            print("Channel is all NaN — replaced with zeros")
            return np.zeros_like(data)
        else:
            print("Channel has NaN/Inf — interpolating")
            idx = np.isfinite(data)
            data[~idx] = np.interp(np.flatnonzero(~idx), np.flatnonzero(idx), data[idx])
    return data

def scale_and_clip_signal(data, scale_factor=1e-5, clip_value=500):
    print(f"Signal min={np.min(data)}, max={np.max(data)} before scaling")
    data = data * scale_factor
    print(f"Signal min={np.min(data)}, max={np.max(data)} after scaling")
    data = np.clip(data, -clip_value, clip_value)
    print(f"Final data min={np.min(data)}, max={np.max(data)} after clipping")
    return data

def process_subject(subject_folder):
    subject_path = os.path.join(base_input, subject_folder)
    subject_label = subject_folder.replace('HUP', '').replace('_phaseII', '').replace('b', 'b')
    output_path = os.path.join(base_output, f"sub-{subject_label}", "ses-phaseII", "eeg")
    os.makedirs(output_path, exist_ok=True)

    print(f"\nProcessing subject: {subject_folder} → sub-{subject_label}")

    channel_files = []
    lengths = []

    for filename in sorted(os.listdir(subject_path)):
        if filename.endswith('.mat') and any(tag in filename for tag in ['LA', 'LC', 'LEAR', 'REAR']):
            ch_path = os.path.join(subject_path, filename)
            try:
                data = load_channel(ch_path)
                data = clean_signal(data)
                data = scale_and_clip_signal(data)
                channel_files.append((filename.replace('.mat', ''), data))
                lengths.append(len(data))
                print(f"  Loaded {filename}, length: {len(data)}")
            except Exception as e:
                print(f"Failed loading {filename}: {e}")

    if len(channel_files) == 0:
        print("No valid channels found, skipping.")
        return

    min_len = min(lengths)
    print(f"  Aligning all channels to {min_len} samples.")
    raw = None
    for i in range(0, len(channel_files), chunk_size):
        chunk = channel_files[i:i + chunk_size]
        data_list = []
        names_list = []
        for name, data in chunk:
            data_list.append(data[:min_len])
            names_list.append(name)
        stacked_data = np.stack(data_list, axis=0)
        info = mne.create_info(ch_names=names_list, sfreq=fs, ch_types='eeg')
        raw_chunk = mne.io.RawArray(stacked_data, info)
        raw_chunk._data = raw_chunk._data.astype(np.float32)

        if raw is None:
            raw = raw_chunk
        else:
            raw.add_channels([raw_chunk])

        del data_list, names_list, stacked_data, raw_chunk
        gc.collect()

    scans = []
    total_samples = raw.n_times
    samples_per_chunk = export_chunk_sec * fs
    num_chunks = (total_samples + samples_per_chunk - 1) // samples_per_chunk

    for idx in range(num_chunks):
        start = idx * samples_per_chunk
        stop = min((idx + 1) * samples_per_chunk, total_samples)
        raw_sub_chunk = raw.copy().crop(tmin=start / fs, tmax=(stop - 1) / fs)
        edf_name = f"sub-{subject_label}_ses-phaseII_task-phaseII_run-{idx+1:02d}_eeg.edf"
        edf_path = os.path.join(output_path, edf_name)
        raw_sub_chunk.export(edf_path, fmt='edf')
        print(f"Exported {edf_name} [{start}–{stop}] samples")
        scans.append({"filename": f"eeg/{edf_name}", "acq_time": "n/a"})

        del raw_sub_chunk
        gc.collect()

    scans_df = pd.DataFrame(scans)
    scans_df.to_csv(os.path.join(output_path, f"sub-{subject_label}_ses-phaseII_scans.tsv"), sep='\t', index=False)

    channels_df = pd.DataFrame({
        "name": raw.ch_names,
        "type": ["EEG"] * len(raw.ch_names),
        "unit": ["µV"] * len(raw.ch_names),
        "sampling_frequency": [fs] * len(raw.ch_names),
        "status": ["good"] * len(raw.ch_names)
    })
    channels_df.to_csv(os.path.join(output_path, f"sub-{subject_label}_ses-phaseII_task-phaseII_channels.tsv"), sep='\t>
    electrodes_csv = [f for f in os.listdir(subject_path) if f.endswith('_electrodes.csv')]
    if electrodes_csv:
        try:
            electrodes_df = pd.read_csv(os.path.join(subject_path, electrodes_csv[0]))
            electrodes_df.columns = ['name', 'x', 'y', 'z']
            electrodes_df.to_csv(os.path.join(output_path, f"sub-{subject_label}_electrodes.tsv"), sep='\t', index=Fals>            print("✅ Converted electrodes.csv to electrodes.tsv")
        except Exception as e:
            print(f"Failed electrodes.csv conversion: {e}")
    else:
        print("No electrodes.csv found.")

    del raw
    gc.collect()

# -----------------------------
# Main Loop
# -----------------------------
for subject in subjects:
    process_subject(subject)

print("\nAll subjects processed with chunked EDF export and cleaned signals.")
