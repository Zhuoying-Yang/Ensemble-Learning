import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_score
import joblib
import torch

# ========== PATHS ==========
PREPROCESSED_DIR = os.path.expanduser("~/data_link")
SAVE_DIR = os.path.expanduser("~/models_link")
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== LOAD v4 DATA ==========
train_X, train_y = torch.load(f"{PREPROCESSED_DIR}/train_v4_raw_three.pt")
val_X,   val_y   = torch.load(f"{PREPROCESSED_DIR}/val_v4_raw_three.pt")
test_X,  test_y  = torch.load(f"{PREPROCESSED_DIR}/merged_test_raw_three.pt")

# Flatten for RF
X_train = train_X.reshape(len(train_X), -1).numpy()
y_train = train_y.numpy()
X_val   = val_X.reshape(len(val_X), -1).numpy()
y_val   = val_y.numpy()
X_test  = test_X.reshape(len(test_X), -1).numpy()
y_test  = test_y.numpy()

# ========== GRID SEARCH TO MAXIMIZE PRECISION @ CLASS 0 ==========
from sklearn.metrics import make_scorer

# Define scorer: precision for class 0
prec0_scorer = make_scorer(precision_score, pos_label=0, average='binary', zero_division=0)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 8, 16],
    'class_weight': [{0: w, 1:1, 2:1} for w in [1.0, 2.0, 5.0, 10.0]]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
gs = GridSearchCV(
    rf, param_grid,
    scoring=prec0_scorer,
    cv=[(np.arange(len(X_train)), np.arange(len(X_train)))],  # use hold‑out val manually
    refit=True,
    verbose=2
)

# Monkey‑patch GS to use our own train/val split:
gs.cv = [(np.arange(len(X_train)), 'val')]
# Hack: we'll set X and y as concatenation, but will override scoring manually below.

# Instead, do manual search:
best_score = -1.0
best_params = None
for params in gs.param_grid:
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    prec0 = precision_score(y_val, y_pred_val, pos_label=0, zero_division=0)
    print(f"Params={params} → Precision@0 = {prec0:.4f}")
    if prec0 > best_score:
        best_score = prec0
        best_params = params

print(f"\nBest params (max class‑0 precision={best_score:.4f}):\n  {best_params}")

# ========== TRAIN FINAL RF ==========
final_rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
final_rf.fit(X_train, y_train)
joblib.dump(final_rf, os.path.join(SAVE_DIR, "RF_v4_only_tuned.joblib"))

# ========== VALIDATION ON v4 ==========
y_val_pred = final_rf.predict(X_val)
print("\nValidation Report (v4):")
print(classification_report(y_val, y_val_pred, digits=4))

# ========== TEST ON MERGED TEST SET ==========
y_test_pred = final_rf.predict(X_test)
print("\nMerged Test Report:")
print(classification_report(y_test, y_test_pred, digits=4))
