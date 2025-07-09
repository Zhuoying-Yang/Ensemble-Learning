import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gc
import joblib
from thop import profile, clever_format

# ========================
# Device Setup
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ========================
# Model Size Reporting
# ========================
def report_model_stats(model, input_shape, name):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    model_path = f"{name}.pt"
    torch.save(model.state_dict(), model_path)
    size_MB = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nModel Summary for {name}:")
    print(f"Saved File Size: {size_MB:.2f} MB")
    print(f"Total Parameters: {params}")
    print(f"FLOPs: {flops}")
    os.remove(model_path)

# ========================
# Model Definitions
# ========================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, 5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 32, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out) + out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

# ========================
# Training and Evaluation
# ========================
def train(model, loader, criterion, optimizer):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

def predict_dl(model, X_tensor, batch_size=64):
    model.eval()
    preds = []
    loader = DataLoader(X_tensor, batch_size=batch_size)
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1)[:, 1]
            preds.append(prob.cpu())
    return torch.cat(preds).numpy()

# ========================
# Training Loop
# ========================
data_dir = "/home/zhuoying"
version_suffixes = ["_v1", "_v2", "_v3", "_v4"]
model_types = ["CNN", "RNN", "ResNet", "RF"]
test_data_all, test_labels_all = [], []
val_labels_all, model_preds, model_val_preds = [], [], []

for version, model_type in zip(version_suffixes, model_types):
    print(f"\n Processing {model_type} on {version}...")
    X_list, y_list = [], []

    for file in sorted(os.listdir(data_dir)):
        if file.endswith(f"{version}_features.csv") and "LEAR1_REAR1" in file:
            feat_path = os.path.join(data_dir, file)
            seg_path = feat_path.replace("_features.csv", "_segments.npy")
            if not os.path.exists(seg_path):
                continue
            df = pd.read_csv(feat_path)
            seg = np.load(seg_path)
            if seg.size == 0 or len(df) != len(seg):
                print(f"Skipping malformed file: {file} with shape {seg.shape}")
                continue
            X_list.append(seg)
            y_list.append(df["label"].values)

    if not X_list:
        continue

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    mask = ~np.isnan(X).any(axis=(1, 2))
    X, y = X[mask], y[mask]

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.08, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1/0.92, stratify=y_trainval, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train.reshape(len(X_train), -1), y_train)
    X_train_sm = X_train_sm.reshape(-1, 2, X.shape[2])

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    val_labels_all.append(y_val_tensor)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_data_all.append(X_test_tensor)
    test_labels_all.append(y_test_tensor)

    if model_type == "RF":
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=1)
        rf.fit(X_train_sm.reshape(len(X_train_sm), -1), y_train_sm)
        val_flat = X_val.reshape(len(X_val), -1)
        test_flat = X_test.reshape(len(X_test), -1)
        val_prob = rf.predict_proba(val_flat)[:, 1]
        test_prob = rf.predict_proba(test_flat)[:, 1]
        model_val_preds.append(val_prob)
        model_preds.append(test_prob)
        joblib.dump(rf, f"RF_{version}.joblib")
    else:
        model = CNN() if model_type == "CNN" else RNN() if model_type == "RNN" else ResNet1D()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_sm, dtype=torch.float32),
                                                torch.tensor(y_train_sm, dtype=torch.long)),
                                  batch_size=64, shuffle=True)

        best_val_loss, wait = float("inf"), 0
        for epoch in range(500):
            train(model, train_loader, criterion, optimizer)
            if epoch % 10 == 0:
                val_loss, val_acc = evaluate(model, DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64), criterion)
                print(f"Epoch {epoch}: Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss, wait = val_loss, 0
            else:
                wait += 1
                if wait >= 30:
                    break

        val_preds = predict_dl(model, X_val_tensor)
        test_preds = predict_dl(model, X_test_tensor)
        model_val_preds.append(val_preds)
        model_preds.append(test_preds)
        torch.save(model.state_dict(), f"{model_type}_{version}.pt")
        report_model_stats(model, (1, 2, X.shape[2]), f"{model_type}_{version}")

    free_memory()

# ========================
# Unified Evaluation
# ========================
print("\nMerging test sets for unified evaluation...")
X_test_merged = torch.cat(test_data_all, dim=0)
y_test_merged = torch.cat(test_labels_all, dim=0).numpy()

print("\nEvaluating individual models on merged test set...")
model_preds_merged = []

for i, model_type in enumerate(model_types):
    print(f"Evaluating {model_type} on unified test set...")
    if model_type == "RF":
        rf = joblib.load(f"RF_{version_suffixes[i]}.joblib")
        test_flat = X_test_merged.reshape(len(X_test_merged), -1)
        test_prob = rf.predict_proba(test_flat)[:, 1]
        model_preds_merged.append(test_prob)
    else:
        model = CNN() if model_type == "CNN" else RNN() if model_type == "RNN" else ResNet1D()
        model.load_state_dict(torch.load(f"{model_type}_{version_suffixes[i]}.pt"))
        model.to(device)
        test_prob = predict_dl(model, X_test_merged)
        model_preds_merged.append(test_prob)

    pred_binary = (test_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test_merged, pred_binary)
    print(f"{model_type} Accuracy on merged test set: {acc:.4f}")

# ========================
# Optimize Ensemble on Validation Set
# ========================
print("\nOptimizing ensemble on validation set...")
min_len_val = min(len(p) for p in model_val_preds)
model_val_preds = [p[:min_len_val] for p in model_val_preds]
y_val_combined = torch.cat(val_labels_all, dim=0).numpy()[:min_len_val]

best_f1, best_weights, best_thresh = -1, None, None
for weights in itertools.product(np.arange(0, 1.1, 0.2), repeat=len(model_val_preds)):
    if sum(weights) == 0:
        continue
    weights = np.array(weights) / sum(weights)
    combined_val = sum(w * p for w, p in zip(weights, model_val_preds))
    for thresh in np.arange(0.35, 0.56, 0.02):
        pred_val = (combined_val >= thresh).astype(int)
        f1 = f1_score(y_val_combined, pred_val)
        if f1 > best_f1:
            best_f1, best_weights, best_thresh = f1, weights, thresh

# ========================
# Apply Ensemble on Unified Test Set
# ========================
print("\nEvaluating ensemble on merged test set...")
model_preds_stack = np.stack(model_preds_merged, axis=1)
final_combined = np.dot(model_preds_stack, best_weights)
final_pred = (final_combined >= best_thresh).astype(int)

conf_matrix = confusion_matrix(y_test_merged, final_pred)
report_text = classification_report(y_test_merged, final_pred)

with open("hybrid_classification_report_merged.txt", "w") as f:
    f.write("Best ensemble weights and threshold (optimized on validation set):\n")
    f.write(str(best_weights))
    f.write(f"\nThreshold: {best_thresh:.2f}\n\n")
    f.write(report_text)
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Ensemble (Unified Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("optimized_confusion_matrix_merged.png")
