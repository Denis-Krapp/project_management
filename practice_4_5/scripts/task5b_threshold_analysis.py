import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from ultralytics import YOLO


def collect_validation_probs(weights_path, val_dir):
    model = YOLO(str(weights_path))
    y_true_bad = []
    p_bad_list = []

    for cls_name, is_bad in [("bad", 1), ("good", 0)]:
        cls_folder = val_dir / cls_name
        for img_path in sorted(cls_folder.glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            res = model(str(img_path), imgsz=640, verbose=False)
            p_bad = float(res[0].probs.data[0].item())
            y_true_bad.append(is_bad)
            p_bad_list.append(p_bad)

    return np.array(y_true_bad), np.array(p_bad_list)


parser = argparse.ArgumentParser(description="Task 5B: Validation-Only Threshold & Cost Sweep")
parser.add_argument("--val_dir", type=Path, default=Path("data/quality_split/validation"))
parser.add_argument("--weights", type=Path, default=Path("results/week7_yolov8n_cls/weights/best.pt"))
parser.add_argument("--results_json", type=Path, default=Path("results/week7_yolov8n_cls/results.json"))
parser.add_argument("--out_plot", type=Path, default=Path("results/threshold_pr_curve.png"))
parser.add_argument("--fa_cost", type=float, default=10.0, help="Cost of False Accept (bad photo enters OCR)")
parser.add_argument("--fr_cost", type=float, default=1.0, help="Cost of False Reject (user retakes photo)")
args = parser.parse_args()

y_true_bad, p_bad = collect_validation_probs(args.weights, args.val_dir)
thresholds = np.round(np.arange(0.10, 0.901, 0.05), 2)

sweep_rows = []
print("\n=== VALIDATION THRESHOLD SWEEP (Positive Class = 'bad') ===")
print(f"{'Thresh':>6} | {'Prec(bad)':>9} | {'Rec(bad)':>8} | {'F1(bad)':>7} | {'FalseAccept':>11} | {'FalseReject':>11} | {'TotalCost':>9}")
print("-" * 78)

for tau in thresholds:
    pred_bad = (p_bad >= tau).astype(int)
    prec = float(precision_score(y_true_bad, pred_bad, zero_division=0))
    rec = float(recall_score(y_true_bad, pred_bad, zero_division=0))
    f1 = float(f1_score(y_true_bad, pred_bad, zero_division=0))

    false_accept = int(np.sum((y_true_bad == 1) & (pred_bad == 0)))
    false_reject = int(np.sum((y_true_bad == 0) & (pred_bad == 1)))
    total_cost = round(false_accept * args.fa_cost + false_reject * args.fr_cost, 1)

    row = {
        "threshold": float(tau),
        "precision_bad": round(prec, 4),
        "recall_bad": round(rec, 4),
        "f1_bad": round(f1, 4),
        "false_accept": false_accept,
        "false_reject": false_reject,
        "weighted_cost": total_cost,
    }
    sweep_rows.append(row)
    print(f"{tau:6.2f} | {prec:9.4f} | {rec:8.4f} | {f1:7.4f} | {false_accept:11d} | {false_reject:11d} | {total_cost:9.1f}")

best_row = min(sweep_rows, key=lambda r: (r["weighted_cost"], -r["recall_bad"], r["threshold"]))
default_row = next(r for r in sweep_rows if abs(r["threshold"] - 0.50) < 1e-6)

print(f"\nDefault (0.50): FalseAccept={default_row['false_accept']}, FalseReject={default_row['false_reject']}, Cost={default_row['weighted_cost']}")
print(f"Chosen ({best_row['threshold']:.2f}): FalseAccept={best_row['false_accept']}, FalseReject={best_row['false_reject']}, Cost={best_row['weighted_cost']}")

recalls = [r["recall_bad"] for r in sweep_rows]
precisions = [r["precision_bad"] for r in sweep_rows]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(recalls, precisions, marker="o", color="#2563eb", linewidth=2, label="Validation PR Sweep (0.10–0.90)")

ax.scatter(
    default_row["recall_bad"], default_row["precision_bad"],
    s=140, color="#64748b", marker="s", zorder=5,
    label=f"Default τ = 0.50 (FA={default_row['false_accept']}, FR={default_row['false_reject']})"
)
ax.scatter(
    best_row["recall_bad"], best_row["precision_bad"],
    s=180, color="#dc2626", marker="*", zorder=6,
    label=f"Chosen τ = {best_row['threshold']:.2f} (FA={best_row['false_accept']}, FR={best_row['false_reject']})"
)

ax.set_title("Validation Precision–Recall Curve ('bad' Class Gatekeeper)", fontsize=11.5, fontweight="bold")
ax.set_xlabel("Recall ('bad' Defect Capture Rate)", fontsize=10, fontweight="bold")
ax.set_ylabel("Precision ('bad' Rejection Accuracy)", fontsize=10, fontweight="bold")
ax.set_xlim(min(recalls) - 0.03, 1.01)
ax.set_ylim(min(precisions) - 0.03, 1.01)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(args.out_plot, dpi=300)
plt.close()
print(f"Saved PR curve to {args.out_plot}")

if args.results_json.exists():
    data = json.loads(args.results_json.read_text(encoding="utf-8"))
    data["chosen_threshold"] = best_row["threshold"]
    data["threshold_validation_summary"] = {
        "split": "validation_only",
        "default_0_50": default_row,
        "chosen_operating_point": best_row,
    }
    args.results_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {args.results_json} with chosen_threshold={best_row['threshold']:.2f}")