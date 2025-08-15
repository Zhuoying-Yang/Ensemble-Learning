import os
import re
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
    "HUP273c_phaseII",
    "HUP297_phaseII"
]

fs = 1024  # Hz
chunk_size = 1            # channels to load at once
export_chunk_sec = 1800*2 # 60 minutes per run (30 min * 2)

# Ear-EEG channel name pattern (case-insensitive): LEAR#, REAR#, etc.
EAR_PATTERN = re.compile(r'^(L|R)EAR\d+$', flags=re.IGNORECASE)

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
            # indices where data is finite
            good = np.flatnonzero(idx)
            bad = np.flatnonzero(~idx)
            # handle edge cases: if leading/trailing NaNs exist, extend ends
            if good.size == 0:
                return np.zeros_like(data)
            data[~idx] = np.interp(bad, good, data[idx])
    return data

def scale_and_clip_signal(data, scale_factor=1e-5, clip_value=500):
    print(f"Signal min={np.min(data)}, max={np.max(data)} before scaling")
    data = data * scale_factor
    print(f"Signal min={np.min(data)}, max={np.max(data)} after scaling")
    data = np.clip(data, -clip_value, clip_value)
    print(f"Final data min={np.min(data)}, max={np.max(data)} after clipping")
    return data

def is_ear_channel(name: str) -> bool:
    """Return True if channel name is LEAR* or REAR* (case-insensitive)."""
    return bool(EAR_PATTERN.match(name.upper()))

def process_subject(subject_folder):
    subject_path = os.path.join(base_input, subject_folder)
    subject_label = subject_folder.replace('HUP', '').replace('_phaseII', '')
    output_path = os.path.join(base_output, f"sub-{subject_label}", "ses-phaseII", "eeg")
    os.makedirs(output_path, exist_ok=True)

    print(f"\nProcessing subject: {subject_folder} → sub-{subject_label}")

    # --- collect only ear-EEG channels into a dict ---
    ear_channels = {}  # dict: {channel_name: np.array}
    lengths = []

    for filename in sorted(os.listdir(subject_path)):
        if not filename.endswith('.mat'):
            continue
        # skip the metadata .mat file
        if filename.startswith(subject_folder.replace('_phaseII', '')) or filename.startswith("HUP") and filename.endswith("_phaseII.mat"):
            continue

        ch_name = filename[:-4]  # strip '.mat'
        if not is_ear_channel(ch_name):
            # Only keep LEAR*/REAR* channels
            continue

        ch_path = os.path.join(subject_path, filename)
        try:
            data = load_channel(ch_path)
            data = clean_signal(data)
            data = scale_and_clip_signal(data)
            ear_channels[ch_name] = data
            lengths.append(len(data))
            print(f"  Loaded EAR ch {filename}, length: {len(data)}")
        except Exception as e:
            print(f"Failed loading {filename}: {e}")

    if len(ear_channels) == 0:
        print("No EAR (LEAR/REAR) channels found, skipping.")
        return

    # --- align to common length ---
    min_len = min(lengths)
    print(f"  Aligning all EAR channels to {min_len} samples.")

    raw = None
    ch_items = list(ear_channels.items())
    for i in range(0, len(ch_items), chunk_size):
        chunk = ch_items[i:i + chunk_size]
        names_list = [name for name, _ in chunk]
        data_list = [data[:min_len] for _, data in chunk]
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

    # --- export in fixed-length runs ---
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

    # --- BIDS-like sidecars ---
    scans_df = pd.DataFrame(scans)
    scans_df.to_csv(os.path.join(output_path, f"sub-{subject_label}_ses-phaseII_scans.tsv"), sep='\t', index=False)

    channels_df = pd.DataFrame({
        "name": raw.ch_names,
        "type": ["EEG"] * len(raw.ch_names),
        "unit": ["uV"] * len(raw.ch_names),
        "sampling_frequency": [fs] * len(raw.ch_names),
        "status": ["good"] * len(raw.ch_names)
    })
    channels_df.to_csv(
        os.path.join(output_path, f"sub-{subject_label}_ses-phaseII_task-phaseII_channels.tsv"),
        sep='\t', index=False
    )

    # electrodes TSV (iEEG coordinates) if present — optional; safe no-op for ear EEG
    electrodes_csv = [f for f in os.listdir(subject_path) if f.endswith('_electrodes.csv')]
    if electrodes_csv:
        try:
            electrodes_df = pd.read_csv(os.path.join(subject_path, electrodes_csv[0]))
            # Ensure expected columns exist
            cols = [c.lower() for c in electrodes_df.columns]
            # try to map to ['name','x','y','z'] if present
            rename_map = {}
            for src, dst in zip(electrodes_df.columns, ['name', 'x', 'y', 'z'][:len(electrodes_df.columns)]):
                rename_map[src] = dst
            electrodes_df = electrodes_df.rename(columns=rename_map)
            electrodes_df.to_csv(
                os.path.join(base_output, f"sub-{subject_label}", "ses-phaseII", f"sub-{subject_label}_electrodes.tsv"),
                sep='\t', index=False
            )
            print("Converted electrodes.csv to electrodes.tsv")
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

print("\nAll subjects processed with EAR (LEAR/REAR) channels only.")
