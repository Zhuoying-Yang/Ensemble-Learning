import os
import numpy as np
import torch
import joblib
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support

# ========== PATHS ==========
PREPROCESSED_DIR = os.path.expanduser("~/data_link")
SAVE_DIR         = os.path.expanduser("~/models_link")
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== LOAD DATA ==========
train_X, train_y = torch.load(f"{PREPROCESSED_DIR}/train_v4_raw_three.pt")
val_X,   val_y   = torch.load(f"{PREPROCESSED_DIR}/val_v4_raw_three.pt")
test_X,  test_y  = torch.load(f"{PREPROCESSED_DIR}/merged_test_raw_three.pt")

# Flatten for decision tree
X_train = train_X.reshape(len(train_X), -1).numpy()
y_train = train_y.numpy()
X_val   = val_X.reshape(len(val_X), -1).numpy()
y_val   = val_y.numpy()
X_test  = test_X.reshape(len(test_X), -1).numpy()
y_test  = test_y.numpy()

# ========== TRAIN DECISION TREE ==========
tree = DecisionTreeClassifier(max_depth=8, class_weight={0:5, 1:1, 2:1}, random_state=42)
tree.fit(X_train, y_train)

# ========== SAVE MODEL ==========
model_path = os.path.join(SAVE_DIR, "DecisionTree_v4_tuned.joblib")
joblib.dump(tree, model_path)

# Report model size
size_kb = os.path.getsize(model_path) / 1024
print(f"Saved model to {model_path} ({size_kb:.2f} KB)")

# Print model parameters
print("\nModel Parameters:")
for k, v in tree.get_params().items():
    print(f"  {k}: {v}")

# ========== VALIDATION ==========
print("\nValidation on v4:")
val_pred = tree.predict(X_val)
print(classification_report(y_val, val_pred, digits=4, zero_division=0))

# ========== TEST ==========
print("Evaluation on merged test set:")
test_pred = tree.predict(X_test)
print(classification_report(y_test, test_pred, digits=4, zero_division=0))
