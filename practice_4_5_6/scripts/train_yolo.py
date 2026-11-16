import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ultralytics import YOLO


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def count_completed_epochs(results_csv, fallback):
    if not results_csv.exists():
        return fallback
    with results_csv.open("r", encoding="utf-8") as f:
        rows = [row for row in csv.reader(f) if row]
    return max(1, len(rows) - 1)


def measure_yolo_latency(model, test_folder, num_warmup = 20, num_timed = 100):
    image_paths = sorted(
        [p for p in test_folder.glob("*/*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if len(image_paths) < (num_warmup + num_timed):
        image_paths = image_paths * ((num_warmup + num_timed) // len(image_paths) + 1)

    for i in range(num_warmup):
        _ = model(str(image_paths[i]), imgsz=640, verbose=False)

    start = time.perf_counter()
    for i in range(num_warmup, num_warmup + num_timed):
        _ = model(str(image_paths[i]), imgsz=640, verbose=False)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_timed) * 1000.0
    return round(avg_ms, 3)


def evaluate_yolo_on_test(model, test_dir, errors_dir):
    classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    if errors_dir.exists():
        shutil.rmtree(errors_dir)
    errors_dir.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []
    error_count = 0

    for cls_name in classes:
        cls_folder = test_dir / cls_name
        gt = class_to_idx[cls_name]
        for img_path in sorted(cls_folder.glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            results = model(str(img_path), imgsz=640, verbose=False)
            pred_idx = int(results[0].probs.top1)
            conf = float(results[0].probs.top1conf)

            y_true.append(gt)
            y_pred.append(pred_idx)

            if gt != pred_idx:
                error_count += 1
                dst_name = (
                    f"err_{error_count:03d}_true_{idx_to_class[gt]}"
                    f"_pred_{idx_to_class[pred_idx]}_conf_{conf:.2f}_{img_path.name}"
                )
                shutil.copy2(img_path, errors_dir / dst_name)

    print(f"Copied {error_count} misclassified test images to {errors_dir}")
    return np.array(y_true), np.array(y_pred)


def train_and_evaluate_variant(
    weights_name,
    exp_id,
    folder_name,
    yolo_data_dir,
    results_root,
    epochs,
    patience,
    batch_size,
    seed,
):
    run_dir = (results_root / folder_name).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_label = weights_name.replace(".pt", "")
    print(f"\n==========================================")
    print(f"Training {model_label} -> {run_dir}")
    print(f"==========================================")

    model = YOLO(weights_name)
    start_train_t = time.time()

    model.train(
        data=str(yolo_data_dir.resolve()),
        epochs=epochs,
        patience=patience,
        imgsz=640,
        batch=batch_size,
        device="cpu",
        seed=seed,
        workers=0,
        project=str(results_root.resolve()),
        name=folder_name,
        exist_ok=True,
    )

    train_time_s = round(time.time() - start_train_t, 2)
    epochs_run = count_completed_epochs(run_dir / "results.csv", fallback=epochs)

    best_weights_path = run_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_weights_path))

    test_dir = yolo_data_dir / "test"
    y_true, y_pred = evaluate_yolo_on_test(best_model, test_dir, errors_dir=run_dir / "errors")

    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    latency_ms = measure_yolo_latency(best_model, test_dir, num_warmup=20, num_timed=100)

    model_size_mb = round(best_weights_path.stat().st_size / (1024 * 1024), 2)
    param_count = sum(p.numel() for p in best_model.model.parameters())

    results_payload = {
        "experiment": exp_id,
        "week": 7,
        "model": model_label,
        "regime": "fine_tuning",
        "input_size": [640, 640, 3],
        "seed": seed,
        "device": "cpu",
        "epochs_planned": epochs,
        "epochs_run": epochs_run,
        "train_time_s": train_time_s,
        "latency_ms_per_image": latency_ms,
        "model_size_mb": model_size_mb,
        "params": param_count,
        "test_samples": len(y_true),
        "metrics": {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_m, 4),
            "precision_macro": round(p_macro, 4),
            "recall_macro": round(r_macro, 4),
            "per_class": {
                "bad": {
                    "precision": round(float(precision_score(y_true == 0, y_pred == 0, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_true == 0, y_pred == 0, zero_division=0)), 4),
                    "f1": round(float(f1_score(y_true == 0, y_pred == 0, zero_division=0)), 4),
                    "support": int(np.sum(y_true == 0)),
                },
                "good": {
                    "precision": round(float(precision_score(y_true == 1, y_pred == 1, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_true == 1, y_pred == 1, zero_division=0)), 4),
                    "f1": round(float(f1_score(y_true == 1, y_pred == 1, zero_division=0)), 4),
                    "support": int(np.sum(y_true == 1)),
                },
            },
        },
        "confusion_matrix": cm.tolist(),
    }

    out_json = run_dir / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[{model_label}] Complete -> {out_json}")
    print(f"Test Accuracy: {acc:.4f} | Macro F1: {f1_m:.4f} | Latency: {latency_ms} ms | Size: {model_size_mb} MB")


parser = argparse.ArgumentParser(description="Task 3: YOLOv8 Classification Benchmark")
parser.add_argument("--split_dir", type=Path, default=Path("data/quality_split"))
parser.add_argument("--yolo_data_dir", type=Path, default=Path("data/yolo_quality_data"))
parser.add_argument("--results_dir", type=Path, default=Path("results"))
parser.add_argument(
    "--models",
    nargs="+",
    choices=["yolov8n-cls", "yolov8s-cls"],
    default=["yolov8n-cls", "yolov8s-cls"],
)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=245000004)
args = parser.parse_args()

set_seed(args.seed)
yolo_data = setup_yolo_dataset(args.split_dir, args.yolo_data_dir)

all_variants = {
    "yolov8n-cls": ("yolov8n-cls.pt", "task3_yolov8n_cls", "week7_yolov8n_cls"),
    "yolov8s-cls": ("yolov8s-cls.pt", "task3_yolov8s_cls", "week7_yolov8s_cls"),
}

for model_key in args.models:
    weights_name, exp_id, folder_name = all_variants[model_key]
    train_and_evaluate_variant(
        weights_name=weights_name,
        exp_id=exp_id,
        folder_name=folder_name,
        yolo_data_dir=yolo_data,
        results_root=args.results_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )