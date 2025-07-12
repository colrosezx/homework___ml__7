import torch
import torchvision
from model import prepare_model

def export_onnx(path="weights/resnet18.onnx"):
    model = prepare_model()
    model.load_state_dict(torch.load("weights/resnet18.pth"))
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        model, dummy_input, path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"ONNX model exported to {path}")

if __name__ == "__main__":
    export_onnx()