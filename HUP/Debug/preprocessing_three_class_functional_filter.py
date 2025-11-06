# Updated on 2025-11-04: add bandpass filter for both train and test segments

# Updated on 2025-11-02: deal with different FS for different patients

# Updated on 2025-10-04 update different pre-ictal for 4 version
# patient-dependent preprocessing
# process and save segments for all patients
# 1. three training versions + DT features: same training data + same validation data + different test data
# 2. four training versions: different training data + same validation data + same test data
# inter-ictal (label = 0), pre-ictal (label = 1), ictal (label = 2)
# 0000 ********4 hours**** 111 ********* 1 hours *** 2222  *** 3 hours ***
# non-seizure ************* pre-ictal *************** ictal ***************

# NOTE: HUP273c_phaseII.mat
# ['annotations'] the last annotation is 7.9263e+5 (seconds)
# ['fs'] 512
# LEAR1 etc has the length of 231046573
# 231046573 / 512 = 451,262.837890625 seconds < 7.9263e+5 (seconds)
# I suppose they just recorded part of the eeg signal but kept the annotations for the whole duration ???


import h5py
import numpy as np
import os
import sys
import gc
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, filtfilt
import random
import math
from imblearn.over_sampling import SMOTE
import torch

# helpers
from utils import bandpass

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Constants
SEGMENT_LEN_SEC = 2  # 2 seconds base length
THREE_HOURS_SEC = 3 * 60 * 60
FOUR_HOURS_SEC = 4 * 60 * 60
GAP_THRESHOLD_SEC = 2 * 60
SEIZURE_DURATION_SEC = 120
NON_SEIZURE_DURATION_SEC = SEIZURE_DURATION_SEC
PREICTAL_DURATION_SEC = SEIZURE_DURATION_SEC
PREICTAL_MAX_OFFSET_SEC = 60 * 60



LOWCUT, HIGHCUT = 0.5, 40
CLIP_THRESHOLD = 5.0  # Standard deviation multiplier for clipping outliers
IQR_SCALE = 1.5      # IQR method coefficient for outlier detection


NON_SEIZURE_RATIO = 3   # 3:1 ratio of non-seizure to seizure
PREICTAL_RATIO = 1      # 1:1 ratio of pre-ictal to seizure
THRESHOLD_MAX = 1000  # µV

NUM_CHANNELS = 2  # LEAR1 and REAR1
TEST_RATIO = 0.10344            # 12 seizures out of 116 total seizures
TRAIN_RATIO = 1 - TEST_RATIO    # 104 seizures out of 116 total seizures

NUM_VERSIONS = 4  # four training versions

def check_window_valid(new_start, existing_starts, window_size, seizure_starts, duration_aft_seizure=None, duration_bef_seizure=None):
    """
    Check if a new window is valid:
    1. No overlap with existing windows
    2. Far from seizure events
    3. Index not already used
    """
    # Check if index already exists
    if new_start in existing_starts:
        return False

    # Check if too close to seizures
    if duration_aft_seizure is not None and duration_bef_seizure is not None:
        if not all(new_start < s - duration_bef_seizure or new_start > s + duration_aft_seizure
                for s in seizure_starts):
            return False

    # Check window overlap
    new_end = new_start + window_size
    for start in existing_starts:
        end = start + window_size
        if new_start < end and new_end > start:
            return False

    return True

def check_noNan(eeg):
    # eeg = eeg_all[:, i:i + SEGMENT_LEN].copy()
    for ch in range(eeg.shape[0]):
        if np.isnan(eeg[ch]).any():
            if np.all(np.isnan(eeg[ch])):
                return False
            eeg[ch] = np.interp(np.arange(len(eeg[ch])),
                                np.flatnonzero(~np.isnan(eeg[ch])),
                                eeg[ch][~np.isnan(eeg[ch])])
    return True



def test_index_sampling(grouped_seizure_starts, eeg_all, sampling_ratio, fs, segment_length_sec=SEGMENT_LEN_SEC, test_ratio=TEST_RATIO, duration_aft_seizure_sec=SEIZURE_DURATION_SEC, duration_bef_seizure_sec=SEIZURE_DURATION_SEC):
    n_test_non = math.ceil(len(grouped_seizure_starts) * test_ratio) * sampling_ratio   # non-seizure: seizure = 3:1 ratio; pre-ictal: seizure = 1:1 ratio
    test_idxs = []
    segment_length = segment_length_sec * fs
    attempts = 0
    total_len = eeg_all.shape[1]

    while len(test_idxs) < n_test_non and attempts < 200000:
        i = random.randint(0, total_len - segment_length - 1)
        if check_window_valid(i, test_idxs, segment_length, grouped_seizure_starts, duration_aft_seizure=duration_aft_seizure_sec*fs, duration_bef_seizure=duration_bef_seizure_sec*fs):
            eeg = eeg_all[:, i:i + segment_length].copy()
            if check_noNan(eeg):        # make sure no Nan
                test_idxs.append(i)
            else:
                # TODO
                # print("Invalid test_index_sampling due to NaNs, skipping...")
                continue
        attempts += 1

    test_idxs = sorted(test_idxs)
    print(f"len(test_idxs): {len(test_idxs)}, should = {n_test_non}")
    return test_idxs


def train_index_sampling(grouped_seizure_starts, corr_test_index, eeg_all, sampling_ratio, fs, segment_length_sec=SEGMENT_LEN_SEC, train_ratio=TRAIN_RATIO, duration_aft_seizure_sec=SEIZURE_DURATION_SEC, duration_bef_seizure_sec=SEIZURE_DURATION_SEC):
    n_train_non = math.floor(len(grouped_seizure_starts) * train_ratio) * sampling_ratio * 4
    segment_length = segment_length_sec * fs
    train_idxs = []
    attempts = 0

    while len(train_idxs) < n_train_non and attempts < 200000:
        i = random.randint(0, total_len - segment_length - 1)
        # Check against both test and train windows
        if check_window_valid(i, corr_test_index + train_idxs, segment_length, grouped_seizure_starts, duration_aft_seizure=duration_aft_seizure_sec*fs, duration_bef_seizure=duration_bef_seizure_sec*fs):
            eeg = eeg_all[:, i:i + segment_length].copy()
            if check_noNan(eeg):        # make sure no NaNs
                train_idxs.append(i)
            else:
                # print("Invalid [non/pre]-seizure segment due to NaNs, skipping...")
                continue
        attempts += 1

    train_idxs = sorted(train_idxs)
    print(f"len(train_idxs): {len(train_idxs)}, should = {n_train_non}")
    return train_idxs

#


def build_segments(eeg_all, start, fs, label):
    end = start + SEIZURE_DURATION_SEC * fs
    if end > eeg_all.shape[1]:
        print(f"Segment end {end} exceeds data length {eeg_all.shape[1]}, skipping...")
        return [], []

    raw = eeg_all[:, start:end].copy()
    assert check_noNan(raw), "Error: Test ictal segment contains NaNs"

    # Preprocess segment
    raw = preprocess_segment(raw, fs)

    # Validate preprocessed data
    if not check_noNan(raw):
        print(f"Preprocessing resulted in invalid data")
        return [], []

    n = raw.shape[1] // SEGMENT_LEN_SEC // fs
    raw = raw[:, :n * SEGMENT_LEN_SEC * fs]

    segments, labels = [], []
    for seg_idx in range(n):
        seg = raw[:, seg_idx * SEGMENT_LEN_SEC * fs: (seg_idx + 1) * SEGMENT_LEN_SEC * fs]
        # Validate segment quality
        if validate_segment_quality(seg):
            segments.append(seg.astype(np.float32))
            labels.append(label)
        else:
            print(f"Segment {seg_idx}, Label {label} failed quality validation, skipping...")
            continue

    print(f"Built {len(segments)} valid segments starting at index {start} with label {label}.")
    return segments, labels


# TODO: create test (non, pre) based on (test_non_idxs, test_pre_idxs)
def build_test_non_pre(eeg_all, test_idxs, fs, label):
    segments, labels = [], []
    for i in test_idxs:
        segs, labs = build_segments(eeg_all, i, fs, label)
        if not segs:
            raise ValueError(f"empty segments for index {i}")
        segments.append(segs)
        labels.append(labs)
    return segments, labels

# TODO: create test (ictal) based on grouped_seizure_starts: math.ceil(len(grouped_seizure_starts) * TEST_RATIO)
def build_test_ictal(eeg_all, seizure_starts, fs, label=2):
    test_ictal_num = math.ceil(len(seizure_starts) * TEST_RATIO)
    print(f"test_ictal_num: {test_ictal_num}")
    selected_starts = random.sample(seizure_starts, test_ictal_num)
    segments, labels = [], []
    for start in selected_starts:
        segs, labs = build_segments(eeg_all, start, fs, label)
        if not segs:
            continue
        segments.append(segs)
        labels.append(labs)
    return segments, labels, selected_starts

# TODO: create train (non, pre) based on (train_non_splits, train_pre_splits)
def build_train_non_pre(eeg_all, train_idxs, fs, label):
    train_splits = np.array_split(train_idxs, NUM_VERSIONS)
    train_sets = {i: [] for i in range(NUM_VERSIONS)}
    train_labels = {i: [] for i in range(NUM_VERSIONS)}

    for version in range(NUM_VERSIONS):     # version: 0-4
        for idx in train_splits[version]:
            segments, labels = build_segments(eeg_all, idx, fs, label)
            if not segments:
                continue
            train_sets[version].extend(segments)
            train_labels[version].extend(labels)
    # Debug输出
    print(f"\nDebug - train label {label} Version sizes:")
    for v in range(NUM_VERSIONS):
        print(f"Version {v}: {len(train_sets[v])} segments")

    return train_sets, train_labels

# TODO: create train (ictal) based on grouped_seizure_starts: math.floor(len(grouped_seizure_starts) * TRAIN_RATIO)
# the same for all versions
def build_train_ictal(eeg_all, seizure_starts, test_starts, fs, label=2):
    # exclude test ictal starts
    train_starts = [s for s in seizure_starts if s not in test_starts]
    train_ictal_num = math.floor(len(seizure_starts) * TRAIN_RATIO)
    print(f"train_ictal_num: {train_ictal_num} should = {len(train_starts)}")
    segments, labels = [], []
    for start in train_starts:
        segs, labs = build_segments(eeg_all, start, fs, label)
        if not segs:
            continue
        segments.extend(segs)
        labels.extend(labs)
    return segments, labels


def robust_scale(data):
    """IQR-based robust scaling"""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr != 0:
        scaled = (data - q25) / iqr
        return scaled
    return data

def remove_outliers(data, threshold=CLIP_THRESHOLD):
    """Remove outliers beyond a certain threshold of standard deviations"""
    mean = np.mean(data)
    std = np.std(data)
    mask = np.abs(data - mean) <= threshold * std
    return mask

def enhanced_bandpass(data, fs):
    """Enhanced bandpass filtering"""
    nyquist = fs / 2
    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

def preprocess_segment(raw, fs):
    """Enhanced preprocessing pipeline"""
    processed = raw.copy()
    for ch in range(NUM_CHANNELS):
        # 1. Remove DC component
        processed[ch] -= np.mean(processed[ch])

        # 2. Robust scaling
        processed[ch] = robust_scale(processed[ch])

        # 3. Bandpass filtering
        processed[ch] = enhanced_bandpass(processed[ch], fs)

        # 4. Outlier handling
        mask = remove_outliers(processed[ch])
        if not mask.all():
            # Interpolate over outliers
            x = np.arange(len(processed[ch]))
            valid_data = processed[ch][mask]
            valid_indices = x[mask]
            processed[ch] = np.interp(x, valid_indices, valid_data)

        # 5. Standardization
        std = np.std(processed[ch])
        if std != 0:
            processed[ch] /= std

    return processed

def validate_segment_quality(segment):
    """Validate segment quality"""
    # Check for excessive fluctuations
    if np.any(np.abs(segment) > CLIP_THRESHOLD):
        return False

    # Check if the signal's variance is too low (possibly invalid signal)
    if np.any(np.std(segment, axis=1) < 1e-6):
        return False

    return True


def save_version_dataset(process_path, output_path):
    """
    Merge all patients' data for each version and save in the desired structure
    """
    os.makedirs(output_path, exist_ok=True)

    # Process each version
    for v in range(1, NUM_VERSIONS + 1):
        # Get all training files for this version
        train_files = sorted([f for f in os.listdir(process_path)
                            if f.endswith(f"_v{v}_train.pt")])

        # Merge training data for this version
        all_segments = []
        all_labels = []

        for train_file in train_files:
            # Use weights_only=False to allow loading numpy arrays
            data_dict = torch.load(os.path.join(process_path, train_file), weights_only=False)
            all_segments.append(data_dict['segments'])
            all_labels.append(data_dict['labels'])

        # Concatenate all data
        merged_segments = torch.cat(all_segments, dim=0)
        merged_labels = torch.cat(all_labels, dim=0)

        # Save merged version data
        version_dict = {
            'segments': merged_segments,
            'labels': merged_labels
        }
        torch.save(version_dict, os.path.join(output_path, f"v{v}.pt"))

        # Clean up memory
        del all_segments, all_labels, merged_segments, merged_labels
        gc.collect()

    # Process test data
    test_files = sorted([f for f in os.listdir(process_path)
                        if f.endswith("_test.pt")])

    all_test_segments = []
    all_test_labels = []

    for test_file in test_files:
        # Load test data with weights_only=False
        test_dict = torch.load(os.path.join(process_path, test_file), weights_only=False)
        all_test_segments.append(test_dict['segments'])
        all_test_labels.append(test_dict['labels'])

    # Merge and save test data
    merged_test_segments = torch.cat(all_test_segments, dim=0)
    merged_test_labels = torch.cat(all_test_labels, dim=0)

    test_dict = {
        'segments': merged_test_segments,
        'labels': merged_test_labels
    }
    torch.save(test_dict, os.path.join(output_path, "test.pt"))



if __name__ == "__main__":

    # -----------------------------
    # Configuration
    # -----------------------------
    home_path = os.path.expanduser("~")
    base_path = os.path.join(home_path, "links/projects/rrg-xilinliu/data/UPenn_data/")
    process_save = './pre_save_1104/'
    os.makedirs(process_save, exist_ok=True)

    # define all patients and their corresponding sampling frequencies
    patient_configs = {
        "HUP262b_phaseII": 1024,
        "HUP267_phaseII": 1024,
        "HUP269_phaseII": 1024,
        "HUP270_phaseII": 1024,
        "HUP271_phaseII": 1024,
        "HUP272_phaseII": 1024,
        "HUP273c_phaseII": 512,
    }

    # loop over patients
    for folder, fs_config in patient_configs.items():
        print(f"\nProcessing folder {folder} with FS={fs_config} Hz...")

        # dynamically set sampling frequency related constants
        FS = fs_config

        folder_path = os.path.join(base_path, folder)
        label_path = os.path.join(folder_path, folder + ".mat")

        # loading labels
        print(f"Loading labels from {label_path}...")
        try:
            mat = loadmat(label_path)
            tszr = mat.get("tszr", [])
            fs_from_file = mat.get("fs", [[fs_config]])[0][0]  # read FS from file, default to fs_config if not found
            print(f"Sampling frequency from file: {fs_from_file} Hz, Expected: {fs_config} Hz")

            # validate FS
            if abs(fs_from_file - fs_config) > 1:
                print(f"Warning: FS mismatch for {folder}. File: {fs_from_file}, Config: {fs_config}")
                # 使用文件中的采样频率并重新计算常量
                FS = int(fs_from_file)

            seizure_starts = sorted(round(float(row[0].item() * FS)) for row in tszr)
            print(f"Seizure starts (in samples): {seizure_starts}")
            print(f"Found {len(seizure_starts)} seizures.")
        except Exception as e:
            print(f"Error loading {label_path}: {e}")
            continue

        # loading data
        print("Loading LA01 & LA02 data...")
        try:
            with h5py.File(os.path.join(folder_path, "LA01.mat"), 'r') as f:
                eeg_lear1 = np.squeeze(f[list(f.keys())[0]][()])
            with h5py.File(os.path.join(folder_path, "LA02.mat"), 'r') as f:
                eeg_rear1 = np.squeeze(f[list(f.keys())[0]][()])

            assert eeg_lear1.shape == eeg_rear1.shape, "LEAR1 and REAR1 must have the same length"

            min_len = min(len(eeg_lear1), len(eeg_rear1))
            eeg_all = np.stack([eeg_lear1[:min_len], eeg_rear1[:min_len]], axis=0)
        except Exception as e:
            print(f"Error loading EEG pair: {e}")
            continue
        print(f"eeg_all shape: {eeg_all.shape}")
        total_len = eeg_all.shape[1]

        # Group seizure events that are close
        grouped_seizure_starts = []
        for s in seizure_starts:
            if not grouped_seizure_starts or s - grouped_seizure_starts[-1] > GAP_THRESHOLD_SEC * FS:
                grouped_seizure_starts.append(s)
        print(f"Grouped seizure starts (in samples): {grouped_seizure_starts}")
        print(f"Number of grouped seizures: {len(grouped_seizure_starts)}")

        # Test set creation
        test_non_idxs = test_index_sampling(grouped_seizure_starts, eeg_all=eeg_all, sampling_ratio=NON_SEIZURE_RATIO, fs=FS)
        print(f"test_non_idxs length: {len(test_non_idxs)}")
        test_pre_idxs = test_index_sampling(grouped_seizure_starts, eeg_all=eeg_all, sampling_ratio=PREICTAL_RATIO, fs=FS)
        print(f"test_pre_idxs length: {len(test_pre_idxs)}")

        test_set_non, test_set_non_labels = build_test_non_pre(eeg_all, test_non_idxs, fs=FS, label=0)
        test_set_pre, test_set_pre_labels = build_test_non_pre(eeg_all, test_pre_idxs, fs=FS, label=1)
        test_set_ictal, test_set_ictal_labels, test_set_ictal_starts = build_test_ictal(eeg_all, grouped_seizure_starts, fs=FS, label=2)
        print(f"Test set sizes - Non-seizure: {len(test_set_non)}, Preictal: {len(test_set_pre)}, Ictal: {len(test_set_ictal)}")

        # combine test set
        X_test, y_test = [], []
        for segs in test_set_non:
            X_test.extend(segs)
        for labs in test_set_non_labels:
            y_test.extend(labs)
        for segs in test_set_pre:
            X_test.extend(segs)
        for labs in test_set_pre_labels:
            y_test.extend(labs)
        for segs in test_set_ictal:
            X_test.extend(segs)
        for labs in test_set_ictal_labels:
            y_test.extend(labs)
        print(f"Total test set size: {len(X_test)}")

        # dynamic downsampling
        if FS == 1024:
            print("Downsampling from 1024 Hz to 512 Hz...")
            X_test = [x[:, ::2] for x in X_test]
            final_segment_len = SEGMENT_LEN_SEC * FS // 2
            print(f"Downsampled test segments to shape: {X_test[0].shape if X_test else 'N/A'}")
        else:
            print(f"Keeping original sampling rate {FS} Hz")
            final_segment_len = SEGMENT_LEN_SEC * FS

        assert len(X_test) == len(y_test), "Mismatched X and y lengths"

        # save test set
        base = os.path.join(process_save, f"{folder}")
        test_data_dict = {
            'segments': torch.tensor(np.array(X_test), dtype=torch.float32),
            'labels': torch.tensor(np.array(y_test), dtype=torch.long)
        }
        torch.save(test_data_dict, f"{base}_test.pt")

        del X_test, y_test
        gc.collect()

        # Training set creation
        train_non_idxs = train_index_sampling(grouped_seizure_starts, test_non_idxs, eeg_all=eeg_all, fs=FS, sampling_ratio=NON_SEIZURE_RATIO)    # from three_class
        print(f"train_non_idxs length: {len(train_non_idxs)}")
        train_pre_idxs = train_index_sampling(grouped_seizure_starts, test_pre_idxs, eeg_all=eeg_all, fs=FS, sampling_ratio=PREICTAL_RATIO)
        print(f"train_pre_idxs length: {len(train_pre_idxs)}")

        train_sets_non, train_sets_non_labels = build_train_non_pre(eeg_all, train_non_idxs, fs=FS, label=0)
        train_sets_pre, train_sets_pre_labels = build_train_non_pre(eeg_all, train_pre_idxs, fs=FS, label=1)
        print(f"Training set sizes per version - Non-seizure: {[len(v) for v in train_sets_non.values()]}, Preictal: {[len(v) for v in train_sets_pre.values()]}")

        train_set_ictal, train_set_ictal_labels = build_train_ictal(eeg_all, grouped_seizure_starts, test_set_ictal_starts, fs=FS, label=2)
        print(f"Training ictal set size: {len(train_set_ictal)}")

        # process each version's training set
        for i in range(NUM_VERSIONS):
            X_train, y_train = [], []
            X_train.extend(train_set_ictal)
            y_train.extend(train_set_ictal_labels)
            X_train.extend(train_sets_pre[i])
            y_train.extend(train_sets_pre_labels[i])
            X_train.extend(train_sets_non[i])
            y_train.extend(train_sets_non_labels[i])
            print(f"Version {i} - Training set size: {len(train_set_ictal) + len(train_sets_pre[i]) + len(train_sets_non[i])}")

            # dynamic downsampling
            if FS == 1024:
                X_train = [x[:, ::2] for x in X_train]
                print(f"Downsampled train segments to shape: {X_train[0].shape if X_train else 'N/A'}")

            assert len(X_train) == len(y_train), "Mismatched X and y lengths"

            # SMOTE and tensor conversion
            smote = SMOTE(random_state=SEED)
            X_resampled, y_resampled = smote.fit_resample(np.array(X_train).reshape(len(X_train), -1), y_train)
            X_resampled = X_resampled.reshape(-1, NUM_CHANNELS, final_segment_len)
            print(f"After SMOTE, training set size: {X_resampled.shape}, labels count: {len(y_resampled)}")
            print(f"Class distribution after SMOTE: {np.bincount(y_resampled)}")

            data_dict = {
                'segments': torch.tensor(X_resampled, dtype=torch.float32),
                'labels': torch.tensor(y_resampled, dtype=torch.long)
            }
            base = os.path.join(process_save, f"{folder}_v{i+1}")
            torch.save(data_dict, f"{base}_train.pt")
            del X_train, y_train, X_resampled, y_resampled, data_dict
            gc.collect()



    ############ After processing all patients, merge and save final datasets
    # PROCESSED_DATA_PATH = './data/'
    PROCESSED_DATA_PATH = './iEEG_data/'
    save_version_dataset(process_save, PROCESSED_DATA_PATH)

    print("\nFinal dataset statistics:")
    for v in range(1, NUM_VERSIONS + 1):
        data = torch.load(os.path.join(PROCESSED_DATA_PATH, f"v{v}.pt"))
        print(f"\nVersion {v}:")
        print(f"Segments shape: {data['segments'].shape}")
        print(f"Labels shape: {data['labels'].shape}")
        print(f"Labels distribution: {torch.bincount(data['labels'])}")

    test_data = torch.load(os.path.join(PROCESSED_DATA_PATH, "test.pt"))
    print("\nTest set:")
    print(f"Segments shape: {test_data['segments'].shape}")
    print(f"Labels shape: {test_data['labels'].shape}")
    print(f"Labels distribution: {torch.bincount(test_data['labels'])}")
