'''Training code using raw segments, Transformer, and trained ensemble optimization'''
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

# ========== DEVICE AND PATHS ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "/home/zhuoying/models_hup"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== UTILITY FUNCTIONS ==========
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

# ========== MODEL DEFINITIONS ==========
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

class TransformerModel(nn.Module):
    def __init__(self, input_dim=2, seq_len=256, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)
        self.seq_len = seq_len

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

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
        logits = torch.sum(weights * model_outputs, dim=1)
        return logits

# ========== TRAIN INDIVIDUAL MODELS ==========
data_dir = "/home/zhuoying"
version_suffixes = ["v1", "v2", "v3", "v4"]
model_types = ["CNN", "Transformer", "ResNet", "RF"]
val_data_all, val_labels_all = [], []
test_data_all, test_labels_all = [], []

for version, model_type in zip(version_suffixes, model_types):
    print(f"\nProcessing {model_type} on {version}...")
    X_list, y_list = [], []

    for file in sorted(os.listdir(data_dir)):
        if file.endswith(f"{version}_raw_segments.npy") and "LEAR1_REAR1" in file:
            seg_path = os.path.join(data_dir, file)
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
        joblib.dump(rf, f"RF_{version}_raw.joblib")
    else:
        model = CNN() if model_type == "CNN" else TransformerModel() if model_type == "Transformer" else ResNet1D()
        model.to(device)
        model_path = os.path.join(SAVE_DIR, f"{model_type}_{version}_raw.pt")

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train_sm, dtype=torch.float32),
                              torch.tensor(y_train_sm, dtype=torch.long)),
                batch_size=64, shuffle=True
            )
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
            torch.save(model.state_dict(), model_path)
            report_model_stats(model, (1, 2, X.shape[2]), model_path)

    free_memory()

# ========== ENSEMBLE USING MERGED VALIDATION SET ==========
merged_val_data = torch.cat(val_data_all, dim=0)
merged_val_labels = torch.cat(val_labels_all, dim=0)
merged_preds = []

for model_type, version in zip(model_types, version_suffixes):
    if model_type == "RF":
        rf = joblib.load(f"RF_{version}_raw.joblib")
        prob = rf.predict_proba(merged_val_data.cpu().numpy().reshape(len(merged_val_data), -1))[:, 1]
    else:
        cls = CNN if model_type == "CNN" else TransformerModel if model_type == "Transformer" else ResNet1D
        model = cls().to(device)
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{model_type}_{version}_raw.pt"), map_location=device))
        prob = predict_dl(model, merged_val_data)
    merged_preds.append(prob)

X_val_stack = torch.tensor(np.column_stack(merged_preds), dtype=torch.float32).to(device)
y_val_bin = merged_val_labels.float().to(device)

ensemble = TrainableEnsemble(num_models=len(merged_preds)).to(device)
optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(300):
    optimizer.zero_grad()
    out = ensemble(X_val_stack)
    loss = criterion(out, y_val_bin)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"[Ensemble] Epoch {epoch}: Loss = {loss.item():.4f}")

with torch.no_grad():
    val_logits = ensemble(X_val_stack).cpu().numpy()
    val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()

thresholds = np.linspace(0.1, 0.9, 100)
best_f1, best_thresh = 0, 0.5
for t in thresholds:
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(y_val_bin.cpu().numpy(), preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"[Trainable Ensemble] Best threshold: {best_thresh:.3f} | Best F1: {best_f1:.4f}")

# ========== PREDICT ON TEST SET ==========
X_test = torch.cat(test_data_all, dim=0).to(device)
y_test = torch.cat(test_labels_all, dim=0).to(device)

test_preds = []
for model_type, version in zip(model_types, version_suffixes):
    if model_type == "RF":
        rf = joblib.load(f"RF_{version}_raw.joblib")
        prob = rf.predict_proba(X_test.cpu().numpy().reshape(len(X_test), -1))[:, 1]
        test_preds.append(prob)
    else:
        cls = CNN if model_type == "CNN" else TransformerModel if model_type == "Transformer" else ResNet1D
        model = cls().to(device)
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{model_type}_{version}_raw.pt"), map_location=device))
        prob = predict_dl(model, X_test)
        test_preds.append(prob)

X_test_stack = torch.tensor(np.column_stack(test_preds), dtype=torch.float32).to(device)

with torch.no_grad():
    pred_logits = ensemble(X_test_stack)
    final_probs = torch.sigmoid(pred_logits).cpu().numpy()
    final_pred = (final_probs >= best_thresh).astype(int)

y_test_np = y_test.cpu().numpy()
conf_matrix = confusion_matrix(y_test_np, final_pred)
report = classification_report(y_test_np, final_pred)

# ========== SAVE RESULTS ==========
with open("hybrid_classification_report_HUP_trainable.txt", "w") as f:
    f.write("Trainable ensemble weights:\n")
    f.write(str(torch.softmax(ensemble.raw_weights, dim=0).cpu().detach().numpy()))
    f.write(f"\nThreshold: {best_thresh:.3f}\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Trainable Ensemble")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("ensemble_confusion_matrix_HUP.png")

