import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from thop import profile, clever_format
import pandas as pd
from itertools import product

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

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.conv_list = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, out_channels, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes)+1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x_bottleneck = self.bottleneck(x)
        conv_outputs = [conv(x_bottleneck) for conv in self.conv_list]
        pool_output = self.maxpool(x)
        out = torch.cat(conv_outputs + [pool_output], dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    def __init__(self, in_channels=2, num_blocks=3, out_channels=32, bottleneck_channels=32, num_classes=3):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(InceptionBlock1D(in_channels if not blocks else out_channels * 4,
                                           out_channels,
                                           bottleneck_channels=bottleneck_channels))
        self.inception_blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # x: (B, C, T)
        x = self.inception_blocks(x)
        x = self.gap(x)
        return self.fc(x)

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

# ========== TRAIN INDIVIDUAL MODELS ==========
model_types = ["CNN", "EEGNet", "ResNet", "InceptionTime"]
version_suffixes = ["v1", "v2", "v3", "v4"] 


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

    model_name = f"{model_type}_{version}_raw_three_complex"

    model = (
    CNN() if model_type == "CNN" else
    EEGNet() if model_type == "EEGNet" else
    ResNet1D() if model_type == "ResNet" else
    InceptionTime()
)
    model.to(device)
    model_path = os.path.join(SAVE_DIR, f"{model_name}.pt")

    retrain = not os.path.exists(model_path)
    if retrain:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
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
                if wait >= 80:
                    break
        torch.save(model.state_dict(), model_path)
        report_model_stats(model, (1, 2, train_X.shape[2]), model_path)
        
    else:
        print(f"Loading pre-trained {model_type} model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

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
    model_name = f"{model_type}_{version}_raw_three_complex"

    cls = (
        CNN if model_type == "CNN" else
        EEGNet if model_type == "EEGNet" else
        ResNet1D if model_type == "ResNet" else
        InceptionTime
    )
    model = cls().to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{model_name}.pt"), map_location=device))
    prob = predict_dl(model, X_train_all)
    train_preds.append(prob)


X_train_stack = torch.tensor(np.stack(train_preds, axis=1), dtype=torch.float32).to(device)  # (N, M, 3)
y_train_true = y_train_all.long().to(device)

merged_preds = []
for model_type, version in zip(model_types, version_suffixes):
    model_name = f"{model_type}_{version}_raw_three_complex"

    cls = (
        CNN if model_type == "CNN" else
        EEGNet if model_type == "EEGNet" else
        ResNet1D if model_type == "ResNet" else
        InceptionTime
    )
    model = cls().to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{model_name}.pt"), map_location=device))
    prob = predict_dl(model, merged_val_data)
    merged_preds.append(prob)

X_val_stack = torch.tensor(np.stack(merged_preds, axis=1), dtype=torch.float32).to(device)
y_val_true = merged_val_labels.long().to(device)

# Normalize weights later per class (softmax-like but manual grid)
weight_grid = np.linspace(0, 1, 11)
num_models = len(train_preds)
num_classes = 3

val_preds_np = X_val_stack.cpu().numpy()
best_weights = np.zeros((num_models, num_classes))
best_scores = {'f1': [0.0]*num_classes}

print("\n=== Grid Search: Optimizing Per-Class Weights ===")
for class_idx in range(num_classes):
    best_f1 = 0
    best_weight = None

    for combo in product(weight_grid, repeat=num_models):
        if np.isclose(sum(combo), 0):
            continue
        normed = np.array(combo) / sum(combo)

        weighted_output = np.einsum('m,nmc->nc', normed, val_preds_np)
        pred_label = np.argmax(weighted_output, axis=1)

        f1_all = f1_score(y_val_true.cpu().numpy(), pred_label, average=None, zero_division=0)
        f1 = f1_all[class_idx] if class_idx < len(f1_all) else 0

        if f1 > best_f1:
            best_f1 = f1
            best_weight = normed

    best_weights[:, class_idx] = best_weight
    best_scores['f1'][class_idx] = best_f1

best_weights_tensor = torch.tensor(best_weights, dtype=torch.float32).to(device)

# Validation performance using best weights
with torch.no_grad():
    val_logits = torch.einsum('mc,nmc->nc', best_weights_tensor, X_val_stack)
    val_pred = val_logits.argmax(dim=1).cpu().numpy()
    y_val_np = y_val_true.cpu().numpy()
    val_conf_matrix = confusion_matrix(y_val_np, val_pred)
    val_report = classification_report(y_val_np, val_pred, digits=4)

# Predict on Test Set
X_test, y_test = torch.load(f"{PREPROCESSED_DIR}/merged_test_raw_three.pt")
X_test, y_test = X_test.to(device), y_test.to(device)
y_test_np = y_test.cpu().numpy()

test_preds = []
print("\nIndividual Model Test Performance")
trained_models = [
    ("CNN", "v1"),
    ("EEGNet", "v2"),
    ("ResNet", "v3"),
    ("InceptionTime", "v4")
]

for model_type, version in trained_models:
    model_name = f"{model_type}_{version}_raw_three_complex"
    try:
        cls = (
            CNN if model_type == "CNN" else
            EEGNet if model_type == "EEGNet" else
            ResNet1D if model_type == "ResNet" else
            InceptionTime
        )
        model = cls().to(device)
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{model_name}.pt"), map_location=device))
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(X_test), dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)

        acc = accuracy_score(y_test_np, pred)
        prec = precision_score(y_test_np, pred, average='macro', zero_division=0)
        rec = recall_score(y_test_np, pred, average='macro', zero_division=0)
        class_report = classification_report(y_test_np, pred, digits=4)

        print(f"\n{model_type} ({version})")
        print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(class_report)

    except Exception as e:
        print(f"[ERROR] Failed evaluating {model_type} ({version}): {e}")

    test_preds.append(prob)

X_test_stack = torch.tensor(np.stack(test_preds, axis=1), dtype=torch.float32).to(device)

with torch.no_grad():
    test_logits = torch.einsum('mc,nmc->nc', best_weights_tensor, X_test_stack)
    final_pred = test_logits.argmax(dim=1).cpu().numpy()

conf_matrix = confusion_matrix(y_test_np, final_pred)
report = classification_report(y_test_np, final_pred, digits=4)

# Save and Print Weights
weights = best_weights_tensor.cpu().numpy()
model_names = [f"{name}_{version}" for name, version in trained_models]
df = pd.DataFrame(weights, columns=["Class 0", "Class 1", "Class 2"], index=model_names)

print("\n=== Per-Class Ensemble Weights ===")
print(df.round(4))

# Save Reports
with open("hybrid_classification_report_HUP_trainable_multiclass_com_four.txt", "w") as f:
    f.write("Grid-Search Optimized Ensemble Weights:\n")
    f.write(str(weights))
    f.write("\n\nTest Classification Report:\n")
    f.write(report)
    f.write("\n\nTest Confusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nValidation Classification Report:\n")
    f.write(val_report)
    f.write("\n\nValidation Confusion Matrix:\n")
    f.write(str(val_conf_matrix))

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Grid Search Ensemble (3-class)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("ensemble_confusion_matrix_HUP_multiclass.png")
