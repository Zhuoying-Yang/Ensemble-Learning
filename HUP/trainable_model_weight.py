'''Training code using extracted features, Transformer, and trained ensemble optimization'''
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

MODEL_DIR = "/home/zhuoying/models_hup"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
            # nn.MaxPool1d(2),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.model(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim=2, seq_len=256, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)
        self.seq_len = seq_len

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, S, C)
        x = self.input_proj(x)  # (B, S, D)
        x = self.transformer_encoder(x)  # (B, S, D)
        x = x.mean(dim=1)  # Global average pooling over sequence
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

# =================== Data Preparation and Model Training ===================
data_dir = "/home/zhuoying"
version_suffixes = ["v1", "v2", "v3", "v4"]
model_types = ["CNN", "Transformer", "ResNet", "RF"]

val_data_all = []
val_labels_all = []
model_val_preds = []
test_data_all = []
test_labels_all = []


for version, model_type in zip(version_suffixes, model_types):
    print(f"\nProcessing {model_type} on {version}...")
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

    # Debugging prints for each model
    print(f"[{model_type} {version}] Total loaded segments: {len(X)}")
    print(f"[{model_type} {version}] Train+Val: {len(X_trainval)} | Test: {len(X_test)}")
    
    # Collect test sets for merging later
    test_data_all.append(torch.tensor(X_test, dtype=torch.float32))
    test_labels_all.append(torch.tensor(y_test, dtype=torch.long))


    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train.reshape(len(X_train), -1), y_train)
    X_train_sm = X_train_sm.reshape(-1, 2, X.shape[2])


    print(f"[{model_type} {version}] Train: {len(X_train)} | Val: {len(X_val)} | After SMOTE: {len(X_train_sm)}")
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    if model_type == "RF":
        rf_path = os.path.join(MODEL_DIR, f"RF_{version}.joblib")

        # always retrain on this run’s data
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=1)
        rf.fit(X_train_sm.reshape(len(X_train_sm), -1), y_train_sm)
        joblib.dump(rf, rf_path)


        total_nodes = sum(est.tree_.node_count for est in rf.estimators_)
        total_splits = sum(est.tree_.node_count - est.tree_.n_leaves for est in rf.estimators_)
        print(f"[RF {version}] Total nodes: {total_nodes} | Split nodes: {total_splits}")


        val_prob = rf.predict_proba(X_val.reshape(len(X_val), -1))[:, 1]


    else:
        model_path = os.path.join(MODEL_DIR, f"{model_type}_{version}.pt")
        model = CNN() if model_type == "CNN" else TransformerModel() if model_type == "Transformer" else ResNet1D()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
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
        model.to(device)
        val_prob = predict_dl(model, X_val_tensor)

    model_val_preds.append(val_prob)
    val_data_all.append(X_val_tensor)
    val_labels_all.append(y_val_tensor)



# Merge all validation data
X_val_merged = torch.cat(val_data_all, dim=0).to(device)
y_val_merged = torch.cat(val_labels_all, dim=0).to(device)

# =================== After Merged Validation Data ===================

print("Training unified RF on merged validation set...")

# Train RF on merged validation set
rf_merged = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=1)
rf_merged.fit(X_val_merged.cpu().numpy().reshape(len(X_val_merged), -1), y_val_merged.cpu().numpy())
joblib.dump(rf_merged, os.path.join("/home/zhuoying", "RF_merged.joblib"))

# =================== Ensemble Validation Predictions ===================

print("Generating validation predictions for ensemble...")

model_types = ["CNN", "Transformer", "ResNet"]
version_suffixes = ["v1", "v2", "v3"]

model_val_preds = []

# Predict with CNN, Transformer, ResNet
for model_type, version in zip(model_types, version_suffixes):
    cls = CNN if model_type == "CNN" else TransformerModel if model_type == "Transformer" else ResNet1D
    model = cls().to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{model_type}_{version}.pt"), map_location=device))
    model.eval()
    model_val_preds.append(predict_dl(model, X_val_merged))

# Predict with unified RF
rf = joblib.load(os.path.join("/home/zhuoying", "RF_merged.joblib"))

total_nodes = sum(est.tree_.node_count for est in rf.estimators_)
total_splits = sum(est.tree_.node_count - est.tree_.n_leaves for est in rf.estimators_)
print(f"[RF_merged] Total nodes: {total_nodes} | Split nodes: {total_splits}")

val_prob_rf = rf.predict_proba(X_val_merged.cpu().numpy().reshape(len(X_val_merged), -1))[:, 1]
model_val_preds.append(val_prob_rf)

# =================== Trainable Ensemble Weight Optimization ===================

print("Training ensemble weights...")

X_stack = torch.tensor(np.column_stack(model_val_preds), dtype=torch.float32).to(device)
y_stack = y_val_merged.to(dtype=torch.float32)

ensemble = TrainableEnsemble(num_models=4).to(device)
optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(300):
    optimizer.zero_grad()
    out = ensemble(X_stack)
    loss = criterion(out, y_stack)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

with torch.no_grad():
    val_logits = ensemble(X_stack).cpu().numpy()
    val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()

thresholds = np.linspace(0.1, 0.9, 100)
best_f1, best_thresh = 0, 0.5
for t in thresholds:
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(y_stack.cpu().numpy(), preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"[Trainable] Best threshold after tuning: {best_thresh:.3f} with F1 score: {best_f1:.4f}")

# =================== Merge All Collected Test Sets and Evaluate ===================

print("\nMerging all collected test sets...")

X_test = torch.cat(test_data_all, dim=0).to(device)
y_test = torch.cat(test_labels_all, dim=0).to(device)

print(f"Total merged test samples: {len(X_test)}")

print("\nEvaluating on test set...")

test_preds = []

# Predict with CNN, Transformer, ResNet
model_types_eval = ["CNN", "Transformer", "ResNet"]
version_suffixes_eval = ["v1", "v2", "v3"]

for model_type, version in zip(model_types_eval, version_suffixes_eval):
    cls = CNN if model_type == "CNN" else TransformerModel if model_type == "Transformer" else ResNet1D
    model = cls().to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{model_type}_{version}.pt"), map_location=device))
    model.eval()
    test_preds.append(predict_dl(model, X_test))

# Predict with unified RF
rf = joblib.load(os.path.join("/home/zhuoying", "RF_merged.joblib"))
prob_rf = rf.predict_proba(X_test.cpu().numpy().reshape(len(X_test), -1))[:, 1]
test_preds.append(prob_rf)

# Stack predictions and apply ensemble
X_test_stack = torch.tensor(np.column_stack(test_preds), dtype=torch.float32).to(device)

with torch.no_grad():
    pred_logits = ensemble(X_test_stack)
    final_probs = torch.sigmoid(pred_logits).cpu().numpy()
    final_pred = (final_probs >= best_thresh).astype(int)

y_test_np = y_test.cpu().numpy()

conf_matrix = confusion_matrix(y_test_np, final_pred)
report = classification_report(y_test_np, final_pred)

with open("hybrid_classification_report_HUP.txt", "w") as f:
    f.write("Trainable ensemble weights:\n")
    f.write(str(torch.softmax(ensemble.raw_weights, dim=0).detach().cpu().numpy()))
    f.write(f"\n\nBest threshold: {best_thresh}\n")
    f.write("\n\nClassification Report:\n")
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

