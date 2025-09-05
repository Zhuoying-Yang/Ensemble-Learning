# HUP / three_class — Seizure Detection Ensemble

Two-stage pipeline for 3-class EEG classification on the HUP dataset:
1) **Decision Tree filter** to remove obvious non-seizure segments.
2) **Deep ensemble** (CNN, ResNet, RNN) for the hard cases.

> **Dependencies**: add a `requirements.txt` later. Typical stack: Python 3.9+, NumPy, SciPy, scikit-learn, PyTorch, pandas, mne/pyedflib (for EEG I/O), matplotlib.

---

## File-by-file (what each script does)

**Data & features**
- `process.py` — Core EEG preprocessing (reads raw data, segments/windows, labels).
- `save_data.py` — Creates **4 training versions** with consistent `val`/`test` splits and different `train` splits; saves to disk.
- `standardize.py` — Standardizes data using **train** mean/variance; applies to train/val/test.
- `process_feature.py` — Computes tabular features (statistics, etc.) used by the Decision Tree.

**Decision-tree stage**
- `decision_tree.py` — Trains the Decision Tree on features, tunes class weights and decision threshold, and saves the filter/checkpoint.

**Deep models & ensembling**
- `models/` — Model definitions (CNN / ResNet / RNN and utilities).
- `train_each_class.py` — Trains **CNN, ResNet, RNN** individually (per-class handling as implemented inside).
- `grid_each_class.py` — Grid-search for **per-class ensemble weights**.
- `train.py` — Builds the final ensemble from trained CNN/ResNet/RNN using learned weights.
- `train_focal.py` — Alternative ensemble that uses focal loss.
- `train_meta.py` — Meta-ensemble variant (stacking-style).
- `train_thresh.py` — Tunes post-hoc decision thresholds on validation.
- `train_vote.py` — Simple voting-based ensemble.

---
## Typical run order (end-to-end)

1) **Create splits (4 training versions)**
python process.py
python save_data.py
2) **Standardize by train statistics**
python standardize.py
3) **Prepare DT features**
python process_feature.py
5) **Train Decision Tree filter**
python decision_tree.py
7) **Train deep models (CNN/ResNet/RNN)**
python train_each_class.py
