# convert .pt to .onnx
import os
import torch
import torch.nn as nn


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

def convert_onnx(onnx_path, model, dummy_input):
    torch.onnx.export(model, dummy_input, onnx_path)

### CNN model
model = CNN()
state_dict = torch.load("./models/CNN_v1.pt", map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)
model.eval()
dummy_input = torch.randn(1, 2, 5)  # input shape
convert_onnx("./models/CNN_v1.onnx", model, dummy_input)

### Transformer model
# model = TransformerModel()
# state_dict = torch.load("./models/Transformer_v2.pt", map_location=torch.device('cpu'), weights_only=True)
# model.load_state_dict(state_dict)
# model.eval()
# dummy_input = torch.randn(1, 2, 5)  # input shape
# convert_onnx("./models/Transformer_v2.onnx", model, dummy_input)


### ResNet model
# model = ResNet1D()
# state_dict = torch.load("./models/Transformer_v2.pt", map_location=torch.device('cpu'), weights_only=True)
# model.load_state_dict(state_dict)
# model.eval()
# dummy_input = torch.randn(1, 2, 5)  # input shape
# convert_onnx("./models/Transformer_v2.onnx", model, dummy_input)


### Random Forest
# from skl2onnx import convert_sklearn
# import joblib
# from skl2onnx.common.data_types import FloatTensorType
# initial_type = [('input', FloatTensorType([1, 2, 5]))]
# model = joblib.load("./models/RF_v4.joblib")
# onnx_model = convert_sklearn(model, initial_types=initial_type)
# with open("./models/RF_v4.onnx", "wb") as f:
#     f.write(onnx_model.SerializeToString())


