import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gc
import glob

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Paths and config
data_dir = "/home/zhuoying/chbmit_preprocessed_raw"
version_suffixes = ["_v1", "_v2", "_v3", "_v4"]
model_types = ["CNN", "Transformer", "ResNet", "RF"]
test_data_all = []
test_labels_all = []
val_labels_all = []
model_preds = []
model_val_preds = []

# Models
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(23, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
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
        self.conv1 = nn.Conv1d(23, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out) + out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

# Training and evaluation
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
    acc = correct / total
    return total_loss / len(loader), acc

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

# Main loop
for version, model_type in zip(version_suffixes, model_types):
    print(f"\nProcessing {model_type} on {version}...")

    X_files = sorted(glob.glob(os.path.join(data_dir, f"X{version}_*.npy")))
    y_file = os.path.join(data_dir, f"y{version}.npy")

    if not os.path.exists(y_file):
        print(f"Missing label file: {y_file}")
        continue

    if not X_files:
        print(f"No X files for version {version}")
        continue

    X_list = [np.load(xf) for xf in X_files]
    X = np.stack(X_list, axis=0)
    y = np.load(y_file)

    if len(X) != len(y):
        print(f"Mismatch: X has {len(X)} samples, y has {len(y)} samples")
        continue

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.18, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, stratify=y_trainval, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train.reshape(len(X_train), -1), y_train)
    X_train_sm = X_train_sm.reshape(-1, X.shape[1], X.shape[2])

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
        val_prob = rf.predict_proba(X_val.reshape(len(X_val), -1))[:, 1]
        test_prob = rf.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
        model_val_preds.append(val_prob)
        model_preds.append(test_prob)
    else:
        model = CNN() if model_type == "CNN" else TransformerModel() if model_type == "Transformer" else ResNet1D()
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
                if wait >= 60:
                    break

        val_preds = predict_dl(model, X_val_tensor)
        test_preds = predict_dl(model, X_test_tensor)
        model_val_preds.append(val_preds)
        model_preds.append(test_preds)

    free_memory()

# Ensemble optimization
print("\nOptimizing ensemble on validation set...")
best_f1, best_weights, best_thresh = -1, None, None

min_len_val = min(len(p) for p in model_val_preds)
model_val_preds = [p[:min_len_val] for p in model_val_preds]
y_val_combined = torch.cat(val_labels_all, dim=0).numpy()[:min_len_val]

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

# Final test
print("\nEvaluating on test set...")
min_len_test = min(len(p) for p in model_preds)
model_preds = [p[:min_len_test] for p in model_preds]
y_test_combined = torch.cat(test_labels_all, dim=0).numpy()[:min_len_test]

final_combined = sum(w * p for w, p in zip(best_weights, model_preds))
final_pred = (final_combined >= best_thresh).astype(int)
conf_matrix = confusion_matrix(y_test_combined, final_pred)
report_text = classification_report(y_test_combined, final_pred)

with open("hybrid_classification_report_sampled.txt", "w") as f:
    f.write("Best ensemble weights and threshold (optimized on validation set):\n")
    f.write(str(best_weights))
    f.write(f"\nThreshold: {best_thresh:.2f}\n\n")
    f.write(report_text)
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Ensemble (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("optimized_confusion_matrix_sampled.png")
