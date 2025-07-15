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

def convert_onnx(onnx_path, model, dummy_input):
    torch.onnx.export(model, dummy_input, onnx_path)

model = CNN()
state_dict = torch.load("./models/CNN_v1.pt", map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)
model.eval()
dummy_input = torch.randn(1, 2, 5)  # input shape
convert_onnx("./models/CNN_v1.onnx", model, dummy_input)
