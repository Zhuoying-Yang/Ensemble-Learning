import torch
from convert_onnx import CNN, TransformerModel, ResNet1D
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("PWD:", os.getcwd())
    # model_state_dict = torch.load("./CNN_v1.pt", weights_only=True, map_location=torch.device('cpu'))
    # model = CNN()
    # Number of trainable parameters: 20434
    # model size(MB): 0.07794952392578125

    # model_state_dict = torch.load("Transformer_v2.pt", weights_only=True, map_location=torch.device('cpu'))
    # model = TransformerModel()
    # model.load_state_dict(model_state_dict)
    # Number of trainable parameters: 562626
    # model size(MB): 2.1462478637695312

    model_state_dict = torch.load("ResNet_v3.pt", weights_only=True, map_location=torch.device('cpu'))
    model = ResNet1D()
    model.load_state_dict(model_state_dict)
    # Number of trainable parameters: 5570
    # model size(MB): 0.02124786376953125
    num_params = count_parameters(model)
    print("Number of trainable parameters: ", num_params)
    total_size_MB = num_params * 4 / 1024 ** 2
    print("model size (MB):", total_size_MB)
