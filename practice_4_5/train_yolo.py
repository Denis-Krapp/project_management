import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ultralytics import YOLO


def setup_yolo_dataset(split_dir, yolo_data_dir):
    yolo_data_dir.mkdir(parents=True, exist_ok=True)
    for s_from, s_to in [("train", "train"), ("validation", "val"), ("test", "test")]:
        target = yolo_data_dir / s_to
        source = split_dir / s_from
        if not target.exists():
            try:
                os.symlink(source.resolve(), target, target_is_directory=True)
            except (OSError, NotImplementedError):
                shutil.copytree(source, target)
    return yolo_data_dir


def measure_yolo_latency(model, test_folder, num_warmup=20, num_timed=100):
    image_paths = sorted(list(test_folder.glob("*/*.*")))
    if len(image_paths) < (num_warmup + num_timed):
        image_paths = (image_paths * ((num_warmup + num_timed) // len(image_paths) + 1))

    for i in range(num_warmup):
        _ = model(str(image_paths[i]), imgsz=640, verbose=False)

    start = time.perf_counter()
    for i in range(num_warmup, num_warmup + num_timed):
        _ = model(str(image_paths[i]), imgsz=640, verbose=False)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_timed) * 1000.0
    return round(avg_ms, 3)


def evaluate_yolo_on_test(model, test_dir):
    classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    y_true = []
    y_pred = []

    for cls_name in classes:
        cls_folder = test_dir / cls_name
        for img_path in sorted(cls_folder.glob("*.*")):
            results = model(str(img_path), imgsz=640, verbose=False)
            pred_idx = int(results[0].probs.top1)
            y_true.append(class_to_idx[cls_name])
            y_pred.append(pred_idx)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    per_class = {}
    for cls_name, cls_idx in class_to_idx.items():
        p = float(precision_score(y_true == cls_idx, y_pred == cls_idx, zero_division=0))
        r = float(recall_score(y_true == cls_idx, y_pred == cls_idx, zero_division=0))
        f1 = float(f1_score(y_true == cls_idx, y_pred == cls_idx, zero_division=0))
        sup = int(np.sum(y_true == cls_idx))
        per_class[cls_name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": sup,
        }

    return acc, p_macro, r_macro, f1_m, cm, per_class, len(y_true)


def run_yolo_experiment(variant, data_dir, output_base, seed, epochs=50, patience=5, batch=32):
    print(f"\n=======================================================")
    print(f"  Executing YOLOv8 Variant: {variant}")
    print(f"=======================================================")

    results_dir = output_base / f"week7_{variant.replace('-', '_')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_weight = f"{variant}.pt"
    model = YOLO(model_weight)

    start_t = time.time()
    train_results = model.train(
        data=str(data_dir),
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=640,
        seed=seed,
        project=str(results_dir),
        name="train_run",
        exist_ok=True,
        verbose=True,
    )
    train_time_s = round(time.time() - start_t, 2)

    best_pt_path = results_dir / "train_run" / "weights" / "best.pt"
    if not best_pt_path.exists():
        best_pt_path = Path(train_results.save_dir) / "weights" / "best.pt"

    trained_model = YOLO(str(best_pt_path))

    test_dir = data_dir / "test"
    acc, p_m, r_m, f1_m, cm, per_class, test_count = evaluate_yolo_on_test(trained_model, test_dir)

    latency_ms = measure_yolo_latency(trained_model, test_dir)

    model_size_mb = round(best_pt_path.stat().st_size / (1024 * 1024), 2)
    param_count = sum(p.numel() for p in trained_model.model.parameters())

    epochs_run = len(train_results.csv.get("epoch", [])) if hasattr(train_results, "csv") else epochs

    results = {
        "experiment": f"task3_{variant.replace('-', '_')}",
        "week": 7,
        "model": variant,
        "regime": "fine_tuning",
        "input_size": [640, 640, 3],
        "seed": seed,
        "device": "cuda" if model.device.type == "cuda" else "cpu",
        "epochs_planned": epochs,
        "epochs_run": epochs_run,
        "train_time_s": train_time_s,
        "latency_ms_per_image": latency_ms,
        "model_size_mb": model_size_mb,
        "params": param_count,
        "test_samples": test_count,
        "metrics": {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_m, 4),
            "precision_macro": round(p_m, 4),
            "recall_macro": round(r_m, 4),
            "per_class": per_class,
        },
        "confusion_matrix": cm.tolist(),
    }

    out_json = results_dir / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted {variant}:")
    print(f"  Test Accuracy: {acc:.4f} | F1: {f1_m:.4f} | Latency: {latency_ms} ms | Size: {model_size_mb} MB")
    print(f"  Artifacts saved to {results_dir}")


parser = argparse.ArgumentParser(description="Task 3: YOLOv8 Quality Classification Pipeline")
parser.add_argument("--data_dir", type=Path, default=Path("data/quality_split"))
parser.add_argument("--output_dir", type=Path, default=Path("results"))
parser.add_argument("--seed", type=int, default=20261023)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch", type=int, default=32)
args = parser.parse_args()

yolo_data = setup_yolo_dataset(args.data_dir, Path("data/yolo_quality_data"))

for variant in ["yolov8n-cls", "yolov8s-cls"]:
    run_yolo_experiment(
        variant=variant,
        data_dir=yolo_data,
        output_base=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
    )