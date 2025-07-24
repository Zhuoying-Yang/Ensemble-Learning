import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ========= CONFIGURATION =========
DATA_DIR = "/home/zhuoying"
SAVE_DIR = "/home/zhuoying/preprocessed_data"
os.makedirs(SAVE_DIR, exist_ok=True)

version_suffixes = ["v1", "v2", "v3", "v4"]
merged_val_data = []
merged_val_labels = []
merged_test_data = []
merged_test_labels = []

# ========= PROCESS EACH VERSION =========
for version in version_suffixes:
    X_list, y_list = [], []

    for file in sorted(os.listdir(DATA_DIR)):
        if file.endswith(f"{version}_raw_three_segments.npy") and "LEAR1_REAR1" in file:
            seg_path = os.path.join(DATA_DIR, file)
            label_path = seg_path.replace("_segments.npy", "_labels.npy")
            if not os.path.exists(label_path):
                continue
            seg = np.load(seg_path)
            labels = np.load(label_path)
            if seg.size == 0 or len(labels) != len(seg):
                continue
            X_list.append(seg)
            y_list.append(labels)

    if not X_list:
        print(f"[{version}] Skipped (no valid files).")
        continue

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    mask = ~np.isnan(X).any(axis=(1, 2))
    X, y = X[mask], y[mask]

    # Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.08, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1/0.92, stratify=y_trainval, random_state=42)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train.reshape(len(X_train), -1), y_train)
    X_train_sm = X_train_sm.reshape(-1, 2, X.shape[2])

    # Convert to tensors
    train_X = torch.tensor(X_train_sm, dtype=torch.float32)
    train_y = torch.tensor(y_train_sm, dtype=torch.long)
    val_X = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.long)
    test_X = torch.tensor(X_test, dtype=torch.float32)
    test_y = torch.tensor(y_test, dtype=torch.long)

    # Save individual sets
    torch.save((train_X, train_y), f"{SAVE_DIR}/train_{version}_raw_three.pt")
    torch.save((val_X, val_y), f"{SAVE_DIR}/val_{version}_raw_three.pt")
    torch.save((test_X, test_y), f"{SAVE_DIR}/test_{version}_raw_three.pt")
    print(f"[{version}] Saved train/val/test with _raw_three suffix.")

    # Append for merged sets
    merged_val_data.append(val_X)
    merged_val_labels.append(val_y)
    merged_test_data.append(test_X)
    merged_test_labels.append(test_y)

# ========= SAVE MERGED VALIDATION AND TEST SETS =========
torch.save((torch.cat(merged_val_data, dim=0), torch.cat(merged_val_labels, dim=0)),
           f"{SAVE_DIR}/merged_val_raw_three.pt")
torch.save((torch.cat(merged_test_data, dim=0), torch.cat(merged_test_labels, dim=0)),
           f"{SAVE_DIR}/merged_test_raw_three.pt")
print("Merged validation and test sets saved with _raw_three suffix.")
