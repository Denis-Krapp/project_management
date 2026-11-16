import argparse
import csv
import json
import random
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from ultralytics import YOLO


DEFECT_REGEX = re.compile(r"^bad_[a-z0-9а-яё]+_(blur|glare|skew|occlusion|exposure)_")


def parse_category(filename, folder_label):
    if folder_label == "good":
        return "good (suitable)"
    m = DEFECT_REGEX.match(filename)
    return m.group(1) if m else "other_bad"


def apply_targeted_glare_and_blur(img_bgr, rng):
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    overlay = out.copy()
    cx = rng.randint(int(w * 0.2), int(w * 0.8))
    cy = rng.randint(int(h * 0.2), int(h * 0.8))
    ax1 = rng.randint(int(w * 0.08), int(w * 0.22))
    ax2 = rng.randint(int(h * 0.06), int(h * 0.18))
    angle = rng.randint(0, 180)
    alpha = rng.randint(150, 230) / 255.0

    cv2.ellipse(overlay, (cx, cy), (ax1, ax2), angle, 0, 360, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (21, 21), 0)
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

    if rng.random() < 0.5:
        k = rng.choice([7, 11])
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        out = cv2.filter2D(out, -1, kernel)

    return out


def prepare_train_only_augmented_split(src_split, dst_split, seed, aug_fraction = 0.5):
    if dst_split.exists():
        shutil.rmtree(dst_split)

    shutil.copytree(src_split / "validation", dst_split / "val")
    shutil.copytree(src_split / "test", dst_split / "test")
    shutil.copytree(src_split / "train", dst_split / "train")

    rng = random.Random(seed)
    bad_train_imgs = sorted((dst_split / "train" / "bad").glob("*.*"))
    selected = rng.sample(bad_train_imgs, k=int(len(bad_train_imgs) * aug_fraction))

    augmented_count = 0
    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        aug_img = apply_targeted_glare_and_blur(img, rng)
        aug_name = f"{img_path.stem}_glareaug{img_path.suffix}"
        cv2.imwrite(str(img_path.parent / aug_name), aug_img)
        augmented_count += 1

    print(f"Added {augmented_count} glare-augmented images to train/bad ONLY (val & test untouched).")
    return augmented_count


def evaluate_per_defect(model, test_dir):
    y_true, y_pred, categories = [], [], []
    for cls_name, gt in [("bad", 0), ("good", 1)]:
        for img_path in sorted((test_dir / cls_name).glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            res = model(str(img_path), imgsz=640, verbose=False)
            pred = int(res[0].probs.top1)
            y_true.append(gt)
            y_pred.append(pred)
            categories.append(parse_category(img_path.name, cls_name))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    per_cat = {}
    for cat in sorted(set(categories)):
        mask = np.array([c == cat for c in categories])
        gt_sub = y_true[mask]
        pr_sub = y_pred[mask]
        correct = int(np.sum(gt_sub == pr_sub))
        total = int(len(gt_sub))
        target_cls = 1 if cat == "good (suitable)" else 0
        sub_f1 = float(f1_score(y_true == target_cls, y_pred == target_cls, zero_division=0)) if cat == "good (suitable)" else round(correct / total, 4)
        per_cat[cat] = {
            "support": total,
            "correct": correct,
            "recall_or_acc": round(correct / total, 4),
            "f1": round(sub_f1, 4),
        }
    return y_true, y_pred, per_cat


parser = argparse.ArgumentParser(description="Task 5C: Error-Driven Glare Augmentation")
parser.add_argument("--split_dir", type=Path, default=Path("data/quality_split"))
parser.add_argument("--aug_data_dir", type=Path, default=Path("data/yolo_glare_aug_data"))
parser.add_argument("--baseline_weights", type=Path, default=Path("results/week7_yolov8n_cls/weights/best.pt"))
parser.add_argument("--results_dir", type=Path, default=Path("results"))
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=245000004)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

aug_added = prepare_train_only_augmented_split(args.split_dir, args.aug_data_dir, args.seed, aug_fraction=0.5)

folder_name = "week9_yolov8n_glare_aug"
run_dir = (args.results_dir / folder_name).resolve()
if run_dir.exists():
    shutil.rmtree(run_dir)

model = YOLO("yolov8n-cls.pt")
start_t = time.time()
model.train(
    data=str(args.aug_data_dir.resolve()),
    epochs=args.epochs,
    patience=args.patience,
    imgsz=640,
    batch=args.batch_size,
    device="cpu",
    seed=args.seed,
    workers=0,
    project=str(args.results_dir.resolve()),
    name=folder_name,
    exist_ok=True,
)
train_time_s = round(time.time() - start_t, 2)

csv_path = run_dir / "results.csv"
epochs_run = max(1, len([r for r in csv.reader(csv_path.open()) if r]) - 1) if csv_path.exists() else args.epochs

best_aug_model = YOLO(str(run_dir / "weights" / "best.pt"))
test_dir = args.aug_data_dir / "test"
y_true, y_pred, after_per_cat = evaluate_per_defect(best_aug_model, test_dir)

before_per_cat = {}
if args.baseline_weights.exists():
    _, _, before_per_cat = evaluate_per_defect(YOLO(str(args.baseline_weights)), test_dir)

print("\n=== PER-CATEGORY TEST DELTA (BEFORE vs. AFTER GLARE AUGMENTATION) ===")
print(f"{'Category':<18} | {'Support':>7} | {'Before':>8} | {'After':>8} | {'Delta':>8}")
print("-" * 60)
for cat, stats in after_per_cat.items():
    b_val = before_per_cat.get(cat, {}).get("recall_or_acc", 0.0)
    a_val = stats["recall_or_acc"]
    print(f"{cat:<18} | {stats['support']:>7} | {b_val:>8.4f} | {a_val:>8.4f} | {a_val - b_val:>+8.4f}")

results_payload = {
    "experiment": "task5c_yolov8n_glare_aug",
    "week": 9,
    "model": "yolov8n-cls",
    "regime": "fine_tuning_glare_aug",
    "seed": args.seed,
    "epochs_run": epochs_run,
    "train_time_s": train_time_s,
    "augmentation_config": {
        "applied_to": "train_split_only",
        "glare_shape": "white_ellipse_gaussian_bloom",
        "alpha_range": [150, 230],
        "motion_blur_prob": 0.5,
        "augmented_samples_added": aug_added,
    },
    "metrics": {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    },
    "per_category_before": before_per_cat,
    "per_category_after": after_per_cat,
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
}

out_json = run_dir / "results.json"
out_json.write_text(json.dumps(results_payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved Task 5C results to {out_json}")