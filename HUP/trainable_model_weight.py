'''Training code using extracted features, RNN, and trained ensemble optimization'''
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import joblib
from thop import profile, clever_format

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def report_model_stats(model, input_shape, name):
    model.eval()
    torch.save(model.state_dict(), f"{name}.pt")
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    size_MB = os.path.getsize(f"{name}.pt") / (1024 * 1024)
    print(f"\nModel Summary for {name}:\n  Size: {size_MB:.2f} MB\n  Params: {params}\n  FLOPs: {flops}")

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
    def forward(self, x): return self.model(x)

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

class TrainableEnsemble(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.ones(num_models))
    def forward(self, model_outputs):
        weights = torch.softmax(self.raw_weights, dim=0)
        return torch.sum(weights * model_outputs, dim=1)

def train(model, loader, criterion, optimizer):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, criterion):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += criterion(out, y).item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss / len(loader), correct / total

def predict_dl(model, X_tensor):
    model.eval()
    loader = DataLoader(X_tensor, batch_size=64)
    preds = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            prob = torch.softmax(model(x), dim=1)[:, 1]
            preds.append(prob.cpu())
    return torch.cat(preds).numpy()

# Storage
data_dir = "/home/zhuoying"
version_suffixes = ["v1", "v2", "v3", "v4"]
model_types = ["CNN", "RNN", "ResNet", "RF"]
test_data_all, test_labels_all = [], []
val_labels_all, val_data_all, model_preds, model_val_preds = [], [], [], []

# Training
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
    val_data_all.append(X_val_tensor)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_data_all.append(X_test_tensor)
    test_labels_all.append(y_test_tensor)

    if model_type == "RF":
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=1)
        rf.fit(X_train_sm.reshape(len(X_train_sm), -1), y_train_sm)
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
        torch.save(model.state_dict(), f"{model_type}_{version}.pt")
        report_model_stats(model, (1, 2, X.shape[2]), f"{model_type}_{version}")

    free_memory()

# Save merged val set
X_val_merged = torch.cat(val_data_all, dim=0)
y_val_merged = torch.cat(val_labels_all, dim=0)
torch.save(X_val_merged, "X_val_merged.pt")
torch.save(y_val_merged, "y_val_merged.pt")

# === Load merged validation set ===
X_val_merged = torch.load("X_val_merged.pt").to(device)
y_val_merged = torch.load("y_val_merged.pt").to(device)

# === Predict on validation set using 4 models ===
model_types = ["CNN", "RNN", "ResNet", "RF"]
version_suffixes = ["v1", "v2", "v3", "v4"]
model_val_preds = []

for model_type, version in zip(model_types, version_suffixes):
    if model_type == "RF":
        rf = joblib.load(f"RF_{version}.joblib")
        val_prob = rf.predict_proba(X_val_merged.cpu().numpy().reshape(len(X_val_merged), -1))[:, 1]
        model_val_preds.append(val_prob)
    else:
        cls = CNN if model_type == "CNN" else RNN if model_type == "RNN" else ResNet1D
        model = cls().to(device)
        model.load_state_dict(torch.load(f"{model_type}_{version}.pt", map_location=device))
        model.eval()
        model_val_preds.append(predict_dl(model, X_val_merged))

# === Train ensemble ===
min_len = min(len(p) for p in model_val_preds)
X_stack = torch.tensor(np.stack([p[:min_len] for p in model_val_preds], axis=1), dtype=torch.float32).to(device)
y_stack = y_val_merged[:min_len].to(dtype=torch.float32)

ensemble = TrainableEnsemble(num_models=4).to(device)
optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

for _ in range(300):
    optimizer.zero_grad()
    out = ensemble(X_stack)
    loss = criterion(out, y_stack)
    loss.backward()
    optimizer.step()

torch.save(ensemble.state_dict(), "TrainableEnsemble_HUP.pt")

# === Evaluate on test set ===
X_test = torch.load("X_test_merged.pt").to(device)
y_test = torch.load("y_test_merged.pt").to(device)

test_preds = []
for model_type, version in zip(model_types, version_suffixes):
    if model_type == "RF":
        rf = torch.load(f"RF_{version}.joblib")
        prob = rf.predict_proba(X_test.cpu().numpy().reshape(len(X_test), -1))[:, 1]
        test_preds.append(prob)
    else:
        cls = CNN if model_type == "CNN" else RNN if model_type == "RNN" else ResNet1D
        model = cls().to(device)
        model.load_state_dict(torch.load(f"{model_type}_{version}.pt", map_location=device))
        test_preds.append(predict_dl(model, X_test))

min_len_test = min(len(p) for p in test_preds)
X_test_stack = torch.tensor(np.stack([p[:min_len_test] for p in test_preds], axis=1), dtype=torch.float32).to(device)
y_test = y_test[:min_len_test]

# Predict ensemble
with torch.no_grad():
    pred_logits = ensemble(X_test_stack)
    final_pred = (torch.sigmoid(pred_logits) >= 0.5).cpu().numpy().astype(int)

conf_matrix = confusion_matrix(y_test.cpu().numpy(), final_pred)
report = classification_report(y_test.cpu().numpy(), final_pred)

# Save results
with open("hybrid_classification_report_HUP.txt", "w") as f:
    f.write("Trainable ensemble weights:\n")
    f.write(str(torch.softmax(ensemble.raw_weights, dim=0).detach().cpu().numpy()))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

# Plot
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Trainable Ensemble")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("ensemble_confusion_matrix_HUP.png") 
