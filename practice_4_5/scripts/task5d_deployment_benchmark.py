import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image
from sklearn.metrics import accuracy_score
from ultralytics import YOLO


def preprocess_640(img_path):
    img = Image.open(img_path).convert("RGB").resize((640, 640))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 640, 640)
    return np.ascontiguousarray(arr)


def benchmark_onnx_model(onnx_path, test_dir, num_warmup = 20, num_timed = 100):
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    samples = []
    for cls_name, gt in [("bad", 0), ("good", 1)]:
        for p in sorted((test_dir / cls_name).glob("*.*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((p, gt))

    tensors = [preprocess_640(samples[i % len(samples)][0]) for i in range(num_warmup + num_timed)]
    for i in range(num_warmup):
        _ = session.run(None, {input_name: tensors[i]})

    start = time.perf_counter()
    for i in range(num_warmup, num_warmup + num_timed):
        _ = session.run(None, {input_name: tensors[i]})
    latency_ms = round(((time.perf_counter() - start) / num_timed) * 1000.0, 3)

    y_true, y_pred = [], []
    for p, gt in samples:
        out = session.run(None, {input_name: preprocess_640(p)})[0]
        pred = int(np.argmax(out, axis=1)[0])
        y_true.append(gt)
        y_pred.append(pred)

    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    size_mb = round(onnx_path.stat().st_size / (1024 * 1024), 2)
    return {"size_mb": size_mb, "latency_ms": latency_ms, "test_accuracy": acc}


parser = argparse.ArgumentParser(description="Task 5D: ONNX Export, Quantization & Edge Benchmark")
parser.add_argument("--weights", type=Path, default=Path("results/week7_yolov8n_cls/weights/best.pt"))
parser.add_argument("--test_dir", type=Path, default=Path("data/quality_split/test"))
parser.add_argument("--results_json", type=Path, default=Path("results/week7_yolov8n_cls/results.json"))
args = parser.parse_args()

base_data = json.loads(args.results_json.read_text(encoding="utf-8"))
pt_size = base_data["model_size_mb"]
pt_lat = base_data["latency_ms_per_image"]
pt_acc = base_data["metrics"]["accuracy"]

model = YOLO(str(args.weights))
exported_file = Path(model.export(format="onnx", imgsz=640, opset=12, simplify=True))
fp32_metrics = benchmark_onnx_model(exported_file, args.test_dir)
fp32_metrics["accuracy_drop"] = round(pt_acc - fp32_metrics["test_accuracy"], 4)

int8_path = exported_file.with_name(f"{exported_file.stem}_int8.onnx")
quantize_dynamic(model_input=str(exported_file), model_output=str(int8_path), weight_type=QuantType.QUInt8)
int8_metrics = benchmark_onnx_model(int8_path, args.test_dir)
int8_metrics["accuracy_drop"] = round(pt_acc - int8_metrics["test_accuracy"], 4)

print("\n=== DEPLOYMENT & QUANTIZATION BENCHMARK (CPU) ===")
print(f"{'Format':<18} | {'Size (MB)':>9} | {'Latency (ms)':>12} | {'Test Acc':>8} | {'Acc Drop':>8}")
print("-" * 68)
print(f"{'PyTorch (.pt)':<18} | {pt_size:>9.2f} | {pt_lat:>12.3f} | {pt_acc:>8.4f} | {'0.0000':>8}")
print(f"{'ONNX FP32':<18} | {fp32_metrics['size_mb']:>9.2f} | {fp32_metrics['latency_ms']:>12.3f} | {fp32_metrics['test_accuracy']:>8.4f} | {fp32_metrics['accuracy_drop']:>+8.4f}")
print(f"{'ONNX INT8':<18} | {int8_metrics['size_mb']:>9.2f} | {int8_metrics['latency_ms']:>12.3f} | {int8_metrics['test_accuracy']:>8.4f} | {int8_metrics['accuracy_drop']:>+8.4f}")

base_data["deployment_benchmark"] = {
    "capstone_targets": {"quality_stage_end_to_end_ms": 52.0, "yolov8_gpu_ms": 2.1, "yolov8_size_mb": 2.8},
    "pytorch_fp32": {"file": str(args.weights), "size_mb": pt_size, "latency_ms": pt_lat, "test_accuracy": pt_acc},
    "onnx_fp32": {"file": str(exported_file), **fp32_metrics},
    "onnx_int8": {"file": str(int8_path), **int8_metrics},
}
args.results_json.write_text(json.dumps(base_data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nAttached deployment_benchmark to {args.results_json}")
