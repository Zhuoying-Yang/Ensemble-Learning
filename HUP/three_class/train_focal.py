import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from thop import profile, clever_format

# ========== DEVICE AND PATHS ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESSED_DIR = os.path.expanduser("~/data_link")
SAVE_DIR = os.path.expanduser("~/models_link")
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
            prob = torch.softmax(model(x), dim=1)
            preds.append(prob.cpu())
    return torch.cat(preds).numpy()

# ========== MODEL DEFINITIONS ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # tensor of shape (num_classes,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-CE_loss)  # prevents nans when probability is 0
        focal_loss = (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.model(x)


class EEGNet(nn.Module):
    def __init__(self, num_classes=3, input_channels=2, samples=2048):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(input_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25)
        )
        # Compute the flattened feature size
        dummy_input = torch.zeros(1, 1, input_channels, samples)
        out = self._forward_features(dummy_input)
        self.classify = nn.Linear(out.shape[1], num_classes)

    def _forward_features(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self._forward_features(x)
        return self.classify(x)
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResidualBlock(2, 64)
        self.block2 = ResidualBlock(64, 128)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        return self.fc(x)


class TrainableEnsemble(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.ones(num_models))
    def forward(self, model_outputs):  # (N, M, 3)
        weights = torch.softmax(self.raw_weights, dim=0)  # (M,)
        weighted = torch.einsum('m,nmc->nc', weights, model_outputs)
        return weighted

# ========== TRAIN INDIVIDUAL MODELS ==========
version_suffixes = ["v1", "v2", "v3", "v4"]
model_types = ["CNN", "EEGNet", "ResNet", "RF"]

val_data_all, val_labels_all = [], []
test_data_all, test_labels_all = [], []

for version, model_type in zip(version_suffixes, model_types):
    print(f"\nLoading preprocessed data for {model_type} on {version}...")
    train_X, train_y = torch.load(f"{PREPROCESSED_DIR}/train_{version}_raw_three.pt")
    val_X, val_y = torch.load(f"{PREPROCESSED_DIR}/val_{version}_raw_three.pt")
    test_X, test_y = torch.load(f"{PREPROCESSED_DIR}/test_{version}_raw_three.pt")


    val_data_all.append(val_X)
    val_labels_all.append(val_y)
    test_data_all.append(test_X)
    test_labels_all.append(test_y)

    model_name = f"{model_type}_{version}_raw_three_focal"

    if model_type == "RF":
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=1)
        rf.fit(train_X.reshape(len(train_X), -1).numpy(), train_y.numpy())
        joblib.dump(rf, f"{model_name}.joblib")
    else:
        model = CNN() if model_type == "CNN" else EEGNet() if model_type == "EEGNet" else ResNet1D()
        model.to(device)
        model_path = os.path.join(SAVE_DIR, f"{model_name}.pt")

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            alpha = torch.tensor([1.0, 2.0, 2.0], device=device)
            criterion = FocalLoss(alpha=alpha, gamma=2.0)

            train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=64, shuffle=True)

            best_val_loss, wait = float("inf"), 0
            for epoch in range(500):
                train(model, train_loader, criterion, optimizer)
                if epoch % 10 == 0:
                    val_loss, val_acc = evaluate(model, DataLoader(TensorDataset(val_X, val_y), batch_size=64), criterion)
                    print(f"Epoch {epoch}: Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss, wait = val_loss, 0
                else:
                    wait += 1
                    if wait >= 60:
                        break
            torch.save(model.state_dict(), model_path)
            report_model_stats(model, (1, 2, train_X.shape[2]), model_path)

    free_memory()

# ========== ENSEMBLE USING MERGED VALIDATION SET ==========
merged_train_data = []
merged_train_labels = []
merged_val_data, merged_val_labels = torch.load(f"{PREPROCESSED_DIR}/merged_val_raw_three.pt")
for version in version_suffixes:
    train_X, train_y = torch.load(f"{PREPROCESSED_DIR}/train_{version}_raw_three.pt")
    merged_train_data.append(train_X)
    merged_train_labels.append(train_y)

X_train_all = torch.cat(merged_train_data, dim=0)
y_train_all = torch.cat(merged_train_labels, dim=0)

train_preds = []
for model_type, version in zip(model_types, version_suffixes):
    model_name = f"{model_type}_{version}_raw_three_focal"
    if model_type == "RF":
        rf = joblib.load(f"{model_name}.joblib")
        prob = rf.predict_proba(X_train_all.cpu().numpy().reshape(len(X_train_all), -1))
    else:
        cls = CNN if model_type == "CNN" else EEGNet if model_type == "EEGNet" else ResNet1D
        model = cls().to(device)
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{model_name}.pt"), map_location=device))
        prob = predict_dl(model, X_train_all)
    train_preds.append(prob)

X_train_stack = torch.tensor(np.stack(train_preds, axis=1), dtype=torch.float32).to(device)  # (N, M, 3)
y_train_true = y_train_all.long().to(device)
merged_preds = []

for model_type, version in zip(model_types, version_suffixes):
    model_name = f"{model_type}_{version}_raw_three_focal"
    if model_type == "RF":
        rf = joblib.load(f"{model_name}.joblib")
        prob = rf.predict_proba(merged_val_data.cpu().numpy().reshape(len(merged_val_data), -1))
    else:
        cls = CNN if model_type == "CNN" else EEGNet if model_type == "EEGNet" else ResNet1D
        model = cls().to(device)
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{model_name}.pt"), map_location=device))
        prob = predict_dl(model, merged_val_data)
    merged_preds.append(prob)
X_val_stack = torch.tensor(np.stack(merged_preds, axis=1), dtype=torch.float32).to(device)
y_val_true = merged_val_labels.long().to(device)


ensemble = TrainableEnsemble(num_models=len(train_preds)).to(device)
optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.01)
alpha = torch.tensor([1.0, 1.0, 3.0], device=device)
criterion = FocalLoss(alpha=alpha, gamma=2.0)


for epoch in range(300):
    optimizer.zero_grad()
    out = ensemble(X_train_stack)
    loss = criterion(out, y_train_true)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"[Ensemble Train] Epoch {epoch}: Loss = {loss.item():.4f}")
        
with torch.no_grad():
    val_logits = ensemble(X_val_stack).cpu().numpy()
    val_pred = val_logits.argmax(axis=1)

y_true = y_val_true.cpu().numpy()
conf_matrix = confusion_matrix(y_true, val_pred)
report = classification_report(y_true, val_pred, digits=4)

# ========== PREDICT ON TEST SET ==========
X_test, y_test = torch.load(f"{PREPROCESSED_DIR}/merged_test_raw_three.pt")
X_test, y_test = X_test.to(device), y_test.to(device)

test_preds = []
for model_type, version in zip(model_types, version_suffixes):
    model_name = f"{model_type}_{version}_raw_three_focal"
    if model_type == "RF":
        rf = joblib.load(f"{model_name}.joblib")
        prob = rf.predict_proba(X_test.cpu().numpy().reshape(len(X_test), -1))
    else:
        cls = CNN if model_type == "CNN" else EEGNet if model_type == "EEGNet" else ResNet1D
        model = cls().to(device)
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{model_name}.pt"), map_location=device))
        prob = predict_dl(model, X_test)
    test_preds.append(prob)

X_test_stack = torch.tensor(np.stack(test_preds, axis=1), dtype=torch.float32).to(device)

with torch.no_grad():
    pred_logits = ensemble(X_test_stack)
    final_pred = pred_logits.argmax(dim=1).cpu().numpy()

y_test_np = y_test.cpu().numpy()
conf_matrix = confusion_matrix(y_test_np, final_pred)
report = classification_report(y_test_np, final_pred, digits=4)

# ========== SAVE RESULTS ==========
with open("hybrid_classification_report_HUP_trainable_multiclass_focal.txt", "w") as f:
    f.write("Trainable ensemble weights:\n")
    f.write(str(torch.softmax(ensemble.raw_weights, dim=0).cpu().detach().numpy()))
    f.write("\n\n" + report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Trainable Ensemble (3-class)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("ensemble_confusion_matrix_HUP_multiclass_focal.png")
