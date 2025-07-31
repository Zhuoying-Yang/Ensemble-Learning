import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import glob
from thop import profile, clever_format
import pandas as pd

# ========== DEVICE AND PATHS ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "/home/zhuoying/chbmit_preprocessed_raw"
SAVE_DIR = "."
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== UTILITY FUNCTIONS ==========
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def report_model_stats(model, input_shape, name):
    model.eval()
    model_path = f"{name}.pt"
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, model_path))
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    size_MB = os.path.getsize(os.path.join(SAVE_DIR, model_path)) / (1024 * 1024)
    print(f"\nModel Summary for {name}:\n  Size: {size_MB:.2f} MB\n  Params: {params}\n  FLOPs: {flops}")

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
            prob = torch.softmax(out, dim=1)
            preds.append(prob.cpu())
    return torch.cat(preds).numpy()

# ========== UPDATED MODEL ARCHITECTURES ==========
class CNN(nn.Module):
    def __init__(self, in_channels=23, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
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
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)

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
    def __init__(self, in_channels=23, num_classes=2):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, 64)
        self.block2 = ResidualBlock(64, 128)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        return self.fc(x)

class EEGNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=23, samples=128):
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
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        return self.classify(x)

class TrainableEnsemble(nn.Module):
    def __init__(self, num_models, num_classes=2):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.ones(num_models, num_classes))

    def forward(self, model_outputs):
        weights = torch.softmax(self.raw_weights, dim=0)
        return torch.einsum('mc,nmc->nc', weights, model_outputs)

# ========== MAIN TRAINING AND EVALUATION ==========
version_suffixes = ["_v1", "_v2", "_v3"]
model_types = ["CNN", "EEGNet", "ResNet"]
val_data_all, val_labels_all = [], []
test_data_all, test_labels_all = [], []
all_trained_models = []

individual_model_report_path = os.path.join(SAVE_DIR, "individual_model_performance.txt")
with open(individual_model_report_path, "w") as f:
    f.write("Individual Model Performance on Test Sets\n")
    f.write("="*50 + "\n\n")

for version, model_type in zip(version_suffixes, model_types):
    print(f"\nProcessing {model_type} on {version}...")

    X_files = sorted(glob.glob(os.path.join(data_dir, f"X{version}_*.npy")))
    y_file = os.path.join(data_dir, f"y{version}.npy")

    if not os.path.exists(y_file) or not X_files:
        print(f"Missing data for {version}")
        continue

    X_list = [np.load(xf) for xf in X_files]
    X = np.stack(X_list, axis=0)
    y = np.load(y_file)

    if len(X) != len(y):
        print(f"Mismatch in samples for {version}")
        continue
    
    num_channels = X.shape[1]
    num_samples = X.shape[2]

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.08, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2/0.92, stratify=y_trainval, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train.reshape(len(X_train), -1), y_train)
    X_train_sm = X_train_sm.reshape(-1, num_channels, num_samples)

    val_data_all.append(torch.tensor(X_val, dtype=torch.float32))
    val_labels_all.append(torch.tensor(y_val, dtype=torch.long))
    test_data_all.append(torch.tensor(X_test, dtype=torch.float32))
    test_labels_all.append(torch.tensor(y_test, dtype=torch.long))

    model = CNN(in_channels=num_channels) if model_type == "CNN" else ResNet1D(in_channels=num_channels) if model_type == "ResNet" else EEGNet(input_channels=num_channels, samples=num_samples)
    model.to(device)
    model_path = os.path.join(SAVE_DIR, f"{model_type}_{version}_mit.pt")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_sm, dtype=torch.float32),
                                            torch.tensor(y_train_sm, dtype=torch.long)),
                              batch_size=64, shuffle=True)

    best_val_loss, wait = float("inf"), 0
    for epoch in range(500):
        train(model, train_loader, criterion, optimizer)
        if epoch % 10 == 0:
            val_loss, val_acc = evaluate(model, DataLoader(TensorDataset(val_data_all[-1], val_labels_all[-1]), batch_size=64), criterion)
            print(f"Epoch {epoch}: Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss, wait = val_loss, 0
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            if wait >= 60:
                break
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    all_trained_models.append(model)
    
    # Individual model evaluation on its own test set
    test_preds_prob = predict_dl(model, test_data_all[-1])
    test_preds_labels = np.argmax(test_preds_prob, axis=1)
    report_text = classification_report(test_labels_all[-1].cpu().numpy(), test_preds_labels, digits=4)
    print(f"\nIndividual Test Set Performance for {model_type} on {version}:")
    print(report_text)
    
    with open(individual_model_report_path, "a") as f:
        f.write(f"Model: {model_type}_{version}\n")
        f.write(report_text)
        f.write("-" * 40 + "\n\n")
    
    if model_type == "EEGNet":
        dummy_input_shape = (1, num_channels, num_samples)
    else:
        dummy_input_shape = (1, num_channels, num_samples)

    report_model_stats(model, dummy_input_shape, f"{model_type}_{version}")
    free_memory()


# ========== ENSEMBLE TRAINING ON MERGED VALIDATION SET ==========
print("\nPreparing merged validation set for ensemble training...")
X_val_merged = torch.cat(val_data_all, dim=0)
y_val_merged = torch.cat(val_labels_all, dim=0)

# Get predictions from each trained model on the merged validation set
ensemble_val_preds = []
for model in all_trained_models:
    preds_on_merged_val = predict_dl(model, X_val_merged)
    ensemble_val_preds.append(preds_on_merged_val)

X_val_stack = torch.tensor(np.stack(ensemble_val_preds, axis=1), dtype=torch.float32).to(device) # (N_val, M, 2)
y_val_true = y_val_merged.long().to(device)

print(f"\nTraining trainable ensemble on merged validation set (data shape: {X_val_stack.shape})...")
ensemble_model = TrainableEnsemble(num_models=len(all_trained_models), num_classes=2).to(device)
optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
lambda_reg = 1e-2 

best_loss = float('inf')
wait = 0
patience = 20
ensemble_path = os.path.join(SAVE_DIR, "TrainableEnsemble_CHBMIT.pt")

for epoch in range(300):
    ensemble_model.train()
    optimizer.zero_grad()
    out = ensemble_model(X_val_stack)
    ce_loss = criterion(out, y_val_true)
    reg_loss = lambda_reg * torch.norm(ensemble_model.raw_weights, p=2)
    loss = ce_loss + reg_loss
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Ensemble Epoch {epoch}: Loss = {loss.item():.4f}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        wait = 0
        torch.save(ensemble_model.state_dict(), ensemble_path)
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

ensemble_model.load_state_dict(torch.load(ensemble_path, map_location=device))
report_model_stats(ensemble_model, X_val_stack.shape, "TrainableEnsemble_CHBMIT")


# ========== FINAL ENSEMBLE EVALUATION ON MERGED TEST SET ==========
print("\nPreparing final merged test set for ensemble evaluation...")
X_test_merged = torch.cat(test_data_all, dim=0)
y_test_merged = torch.cat(test_labels_all, dim=0)
X_test_merged = X_test_merged.to(device)
y_test_merged = y_test_merged.to(device)

# Now, have each trained model predict on the *merged* test set
ensemble_test_preds = []
for model in all_trained_models:
    preds_on_merged_test = predict_dl(model, X_test_merged)
    ensemble_test_preds.append(preds_on_merged_test)

model_preds_test_tensor = torch.tensor(np.stack(ensemble_test_preds, axis=1), dtype=torch.float32).to(device)

print(f"\nEvaluating trainable ensemble on merged test set (data shape: {model_preds_test_tensor.shape})...")
ensemble_model.eval()
with torch.no_grad():
    final_logits = ensemble_model(model_preds_test_tensor).cpu().numpy()

final_pred = np.argmax(final_logits, axis=1)

conf_matrix = confusion_matrix(y_test_merged.cpu().numpy(), final_pred)
report_text = classification_report(y_test_merged.cpu().numpy(), final_pred, digits=4)
final_weights = torch.softmax(ensemble_model.raw_weights, dim=0).detach().cpu().numpy()

model_names = [f"{name}_{version}" for name, version in zip(model_types, version_suffixes)]
weights_df = pd.DataFrame(final_weights, columns=["Class 0", "Class 1"], index=model_names)

print("\n=== Per-Class Ensemble Weights ===")
print(weights_df.round(4))

with open(os.path.join(SAVE_DIR, "hybrid_classification_report_chbmit_updated.txt"), "w") as f:
    f.write("Trainable ensemble weights:\n")
    f.write(str(weights_df.round(4)))
    f.write("\n\nClassification Report (Trainable Ensemble):\n")
    f.write(report_text)
    f.write("\nConfusion Matrix (Trainable Ensemble):\n")
    f.write(str(conf_matrix))

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Trainable Ensemble (CHB-MIT Merged Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "trainable_ensemble_confusion_matrix_chbmit_updated.png"))
