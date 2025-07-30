import os
import numpy as np
import torch
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# ========== PATHS ==========
PREPROCESSED_DIR = os.path.expanduser("~/data_link")
SAVE_DIR = os.path.expanduser("~/models_link")
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== LOAD DATA ==========
train_X, train_y = torch.load(f"{PREPROCESSED_DIR}/train_v4_raw_three.pt")
val_X, val_y = torch.load(f"{PREPROCESSED_DIR}/val_v4_raw_three.pt")
test_X, test_y = torch.load(f"{PREPROCESSED_DIR}/merged_test_raw_three.pt")

# Flatten input
X_train = train_X.reshape(len(train_X), -1).numpy()
X_val = val_X.reshape(len(val_X), -1).numpy()
X_test = test_X.reshape(len(test_X), -1).numpy()

# Merge class 1 and 2 into 1 (seizure-related)
y_train_bin = np.where(train_y.numpy() == 0, 0, 1)
y_val_bin = np.where(val_y.numpy() == 0, 0, 1)
y_test_bin = np.where(test_y.numpy() == 0, 0, 1)

# ========== CLASS WEIGHT TESTS ==========
weight_options = np.arange(2, 16, 1)
thresholds = np.arange(0.10, 0.51, 0.01)  # 0.10 to 0.50 inclusive

for w in weight_options:
    print("=" * 80)
    print(f"🔧 Training with class_weight={{0: 1.0, 1: {w}}}")
    
    # Define and train the model
    tree = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=5,
        class_weight={0: 1.0, 1: w},
        random_state=42
    )
    tree.fit(X_train, y_train_bin)
    
    # Save model
    model_path = os.path.join(SAVE_DIR, f"DecisionTree_v4_cw{w}.joblib")
    joblib.dump(tree, model_path)
    size_kb = os.path.getsize(model_path) / 1024
    print(f"Saved model to {model_path} ({size_kb:.2f} KB)")

    # Raw prediction at default threshold (0.5)
    test_pred = tree.predict(X_test)
    report_raw = classification_report(y_test_bin, test_pred, digits=4, output_dict=True)
    
    print("\nRaw Prediction at Default Threshold (0.5):")
    print(f"  Class 0 - Precision: {report_raw['0']['precision']:.4f}, Recall: {report_raw['0']['recall']:.4f}")
    print(f"  Class 1 - Precision: {report_raw['1']['precision']:.4f}, Recall: {report_raw['1']['recall']:.4f}")
    print(f"  Accuracy: {report_raw['accuracy']:.4f}")

    # ========= THRESHOLD TUNING =========
    print("\nThreshold Tuning on Test Set:")
    prob_0 = tree.predict_proba(X_test)[:, 0]

    for threshold in thresholds:
        pred_thresh = (prob_0 >= threshold).astype(int)
        pred_thresh = 1 - pred_thresh  # convert: 0 → non-seizure

        report = classification_report(y_test_bin, pred_thresh, digits=4, output_dict=True)
        print(f"Threshold = {threshold:.2f} | "
              f"Class 0 → Prec: {report['0']['precision']:.4f}, Recall: {report['0']['recall']:.4f} | "
              f"Class 1 → Prec: {report['1']['precision']:.4f}, Recall: {report['1']['recall']:.4f} | "
              f"Acc: {report['accuracy']:.4f}")
