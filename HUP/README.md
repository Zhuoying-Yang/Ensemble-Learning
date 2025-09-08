# HUP / two_class — Seizure Detection Ensemble
Ensemble of CNN, ResNet, and RNN on 2-class EEG classification on the HUP dataset:
> **Dependencies**: add a `requirements.txt` later. 

---
**Data & features**
- `preprocessing.py` — Process EGG signals with features with labels.
- `process_raw.py` — Process EEG signals with raw segments with labels.
- `save_data.py` —  Creates **4 training versions** with consistent `val`/`test` splits and different `train` splits; saves to disk.


**Deep models & ensembling trials**
- `models/` — Trained Models.
- `CNN_calculate.py` — Calculate the size of trained **CNN** model.
- `RF_calculate.py` — Calculate the size of trained **RF** model (not used in the final model).
- `convert_onnx.py` — Convert the model from `.pt` to `.onnx`.
- `train_raw.py` — Train CNN, Resnet, Transformer using raw segments, and ensemble the models using trained ensemble optimization.
- `train_transformer.py` — Train CNN, Resnet, Transformer using extracted features, and ensemble the models using grid optimization.
- `train_with_saved_dataset.py` — Train CNN, EEGNet, ResNet on standardized raw segments, and ensemble the models using trained ensemble optimization.
- `trainable_model_weight.py` — Train CNN, Resnet, Transformer using extracted features, and ensemble the models using trained ensemble optimization.
- `training.py` — Train CNN, Resnet, RNN using extracted features, and ensemble the models using grid optimization.

---
## Typical run order (end-to-end)

```bash
# 1. Process data
python process_raw.py

# 2. Create splits (4 training versions)
python save_data.py

# 3. Standardize by train statistics
python standardize.py

# 4. Train deep models (CNN / ResNet / RNN)
python train_with_saved_dataset.py
```



# HUP / three_class — Seizure Detection Ensemble

Two-stage pipeline for 3-class EEG classification on the HUP dataset:
1) **Decision Tree filter** to remove obvious non-seizure segments.
2) **Deep ensemble** (CNN, ResNet, EEGNet) for the hard cases.

> **Dependencies**: add a `requirements.txt` later. 

---

## File-by-file (what each script does)

**Data & features**
- `process.py` — Core EEG preprocessing (reads raw data, segments/windows, labels).
- `save_data.py` — Creates **4 training versions** with consistent `val`/`test` splits and different `train` splits; saves to disk.
- `standardize.py` — Standardizes data using **train** mean/variance; applies to train/val/test.
- `process_feature.py` — Computes tabular features (statistics, etc.) used by the Decision Tree.

**Decision-tree stage**
- `decision_tree.py` — Trains the Decision Tree on features, tunes class weights and decision threshold, and saves the filter/checkpoint.

**Deep models & ensembling trials**
- `models/` — Trained Models.
- `train_each_class.py` — Trains **CNN, ResNet, EEGNet** individually (trained per-class weight for each modlr).
- `grid_each_class.py` — Grid-search for **per-class ensemble weights**.
- `train.py` — Builds the final ensemble from trained CNN/ResNet/EEGNet using learned weights.
- `train_focal.py` — Alternative ensemble that uses focal loss.
- `train_meta.py` — Meta-ensemble variant (stacking-style).
- `train_thresh.py` — Tunes post-hoc decision thresholds on validation.
- `train_vote.py` — Simple voting-based ensemble.

---
## Typical run order (end-to-end)

```bash
# 1. Process data
python process.py

# 2. Create splits (4 training versions)
python save_data.py

# 3. Standardize by train statistics
python standardize.py

# 4. Prepare Decision Tree features (optional)
python process_feature.py

# 5. Train Decision Tree filter
python decision_tree.py

# 6. Train deep models (CNN / ResNet / RNN)
python train_each_class.py

# Alternatives to train.py:
python train_focal.py    # Focal loss ensemble
python train_meta.py     # Meta-learning ensemble
python train_thresh.py   # Threshold-tuned ensemble
python train_vote.py     # Voting ensemble

