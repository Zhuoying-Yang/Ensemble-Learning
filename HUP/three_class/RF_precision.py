import os
import numpy as np
import torch
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_recall_fscore_support, classification_report

# ========== PATHS ==========
PREPROCESSED_DIR = os.path.expanduser("~/data_link")
SAVE_DIR         = os.path.expanduser("~/models_link")
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== LOAD DATA ==========
train_X, train_y = torch.load(f"{PREPROCESSED_DIR}/train_v4_raw_three.pt")
val_X,   val_y   = torch.load(f"{PREPROCESSED_DIR}/val_v4_raw_three.pt")
test_X,  test_y  = torch.load(f"{PREPROCESSED_DIR}/merged_test_raw_three.pt")

# Flatten tensors for RF
X_train = train_X.reshape(len(train_X), -1).numpy()
y_train = train_y.numpy()
X_val   = val_X.reshape(len(val_X), -1).numpy()
y_val   = val_y.numpy()
X_test  = test_X.reshape(len(test_X), -1).numpy()
y_test  = test_y.numpy()

# ========== PARAM GRID ==========
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 8, 16],
    'class_weight': [{0: w, 1:1, 2:1} for w in [1.0, 2.0, 5.0, 10.0]]
}

# ========== GRID SEARCH (ONLY CLASS 0 PRECISION) ==========
best_score  = -1.0
best_params = None

print("Searching for best RF params (optimizing precision on class 0)...\n")
for params in ParameterGrid(param_grid):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    # Only look at class 0 precision:
    pr, _, _, _ = precision_recall_fscore_support(
        y_val, y_pred,
        labels=[0],    # only compute for class 0
        zero_division=0
    )
    prec0 = pr[0]
    print(f"  params={params} → class 0 precision = {prec0:.4f}")

    if prec0 > best_score:
        best_score, best_params = prec0, params

print(f"\nBest params (precision@0 = {best_score:.4f}):\n   {best_params}\n")

# ========== TRAIN FINAL RF WITH BEST PARAMS ==========
print("Training final RF on v4 train set with those params...")
final_rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
final_rf.fit(X_train, y_train)

# Save it:
out_path = os.path.join(SAVE_DIR, "RF_v4_only_tuned.joblib")
joblib.dump(final_rf, out_path)
print(f"Saved tuned RF to {out_path}\n")

# ========== VALIDATION ON v4 ==========
print("Validation on v4:")
y_val_pred = final_rf.predict(X_val)
print(classification_report(y_val, y_val_pred, digits=4, zero_division=0))

# ========== EVALUATION ON MERGED TEST SET ==========
print("Evaluation on merged test set:")
y_test_pred = final_rf.predict(X_test)
print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))
