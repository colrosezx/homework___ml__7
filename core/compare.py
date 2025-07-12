import time
import torch
import onnxruntime
import numpy as np
import csv
import os
from model import prepare_model

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def benchmark(model_fn, input_tensor, device):
    times = []
    for _ in range(10):
        start = time.time()
        _ = model_fn(input_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)
    times = sorted(times)[1:-1]  # exclude min and max (outliers)
    return np.mean(times)

def save_to_csv(data: dict, filename="results/comparison.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def run_comparison(batch_size=1, img_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(batch_size, 3, img_size, img_size).to(device)

    # PyTorch native
    model = prepare_model().to(device)
    model.load_state_dict(torch.load("weights/resnet18.pth", map_location=device))
    model.eval()
    pt_time = benchmark(lambda x: model(x), input_tensor, device)

    # ONNX
    ort_session = onnxruntime.InferenceSession(
        "weights/resnet18.onnx", providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider']
    )
    ort_input = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
    ort_time = benchmark(lambda x: ort_session.run(None, ort_input), input_tensor, device)

    # Запись в CSV
    save_to_csv({
        "backend": "PyTorch",
        "batch_size": batch_size,
        "image_size": img_size,
        "device": device,
        "avg_latency_sec": round(pt_time, 6)
    })
    save_to_csv({
        "backend": "ONNX",
        "batch_size": batch_size,
        "image_size": img_size,
        "device": device,
        "avg_latency_sec": round(ort_time, 6)
    })

    print(f"[PyTorch] avg latency: {pt_time:.6f}s")
    print(f"[ONNX]    avg latency: {ort_time:.6f}s")

if __name__ == "__main__":
    run_comparison(batch_size=1, img_size=128)