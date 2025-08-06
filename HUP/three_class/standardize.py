import os
import torch

def standardize_folder(folder_path):
    print(f"\nProcessing folder: {folder_path}")
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pt") and not f.endswith("_standard.pt")])

    # Step 1: Get training files
    train_files = [f for f in all_files if f.startswith("train_v")]
    train_tensors = []

    for fname in train_files:
        path = os.path.join(folder_path, fname)
        data = torch.load(path)

        # Extract X from (X, y) or list of (X, y)
        if isinstance(data, list):
            data = [d[0] if isinstance(d, (tuple, list)) else d for d in data]
            data = torch.stack(data)
        elif isinstance(data, (tuple, list)):
            data = data[0]
        train_tensors.append(data)

    train_all = torch.cat(train_tensors, dim=0)
    mean, std = train_all.mean(), train_all.std()

    print(f"Mean: {mean:.4f}, Std: {std:.4f}")

    # Step 2: Standardize all files and preserve labels if present
    for fname in all_files:
        path = os.path.join(folder_path, fname)
        original = torch.load(path)

        # Extract X and y
        if isinstance(original, (tuple, list)) and len(original) == 2 and isinstance(original[0], torch.Tensor):
            # Case: (X, y) tuple
            X, y = original
        elif isinstance(original, list) and isinstance(original[0], (tuple, list)):
            # Case: list of (X, y) pairs
            X = [d[0] for d in original]
            y = [d[1] for d in original]
            X = torch.stack(X)
            y = torch.stack(y)
        else:
            # Only X
            X = original
            y = None

        # Standardize
        X_standard = (X - mean) / std

        # Save as (X_standard, y) if y exists
        save_path = os.path.join(folder_path, fname.replace(".pt", "_standard.pt"))
        if y is not None:
            torch.save((X_standard, y), save_path)
        else:
            torch.save(X_standard, save_path)

        print(f"Saved: {os.path.basename(save_path)}")

# Run for both folders
standardize_folder("/project/def-xilinliu/data/HUP_two_class")
standardize_folder("/project/def-xilinliu/data/HUP_three_class")
