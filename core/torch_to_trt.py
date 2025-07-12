import torch
from model import prepare_model
import torch_tensorrt

def compile_with_torchtrt():
    model = prepare_model()
    model.load_state_dict(torch.load("weights/resnet18.pth"))
    model.eval().cuda()

    trt_model = torch_tensorrt.compile(model,
        inputs=[torch_tensorrt.Input((1, 3, 128, 128))],
        enabled_precisions={torch.float},
        workspace_size=1 << 20
    )
    torch.jit.save(trt_model, "weights/resnet18_trt.ts")
    print("Torch-TensorRT model saved.")

if __name__ == "__main__":
    compile_with_torchtrt()