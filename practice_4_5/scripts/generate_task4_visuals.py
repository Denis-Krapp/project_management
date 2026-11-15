import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def discover_results(results_root):
    json_files = sorted(results_root.rglob("results.json"))
    if not json_files:
        raise FileNotFoundError(f"No results.json files found under {results_root.resolve()}")

    records = []
    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "latency_ms_per_image" in data and "metrics" in data and "accuracy" in data["metrics"]:
            data["_source_path"] = str(path)
            records.append(data)

    if not records:
        raise ValueError(f"Found {len(json_files)} JSON file(s), but none matched the model results schema.")

    return records


def plot_accuracy_vs_latency_line(records, out_path, budget_ms = 100.0):
    sorted_records = sorted(records, key=lambda r: float(r["latency_ms_per_image"]))
    latencies = [float(r["latency_ms_per_image"]) for r in sorted_records]
    accuracies = [float(r["metrics"]["accuracy"]) for r in sorted_records]

    fig, ax = plt.subplots(figsize=(10, 5.4))

    ax.plot(
        latencies,
        accuracies,
        color="#2563eb",
        linewidth=2.5,
        marker="o",
        markersize=8,
        markerfacecolor="#1d4ed8",
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Evaluated Models (from results.json)",
        zorder=4,
    )

    x_min, x_max = min(latencies), max(latencies)
    y_min, y_max = min(accuracies), max(accuracies)
    x_span = max(x_max - x_min, 10.0)
    y_span = max(y_max - y_min, 0.005)

    x_left = max(0.0, x_min - 0.12 * x_span)
    x_right = max(x_max, budget_ms) + 0.18 * x_span
    y_bottom = y_min - 0.25 * y_span
    y_top = min(1.005, y_max + 0.30 * y_span)

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)

    offset_patterns = [(14, 14), (14, -28), (14, 18), (-14, 16)]
    for idx, rec in enumerate(sorted_records):
        lat = float(rec["latency_ms_per_image"])
        acc = float(rec["metrics"]["accuracy"])
        size = float(rec.get("model_size_mb", 0.0))
        model_name = rec.get("model", Path(rec["_source_path"]).parent.name)
        exp_name = rec.get("experiment", "")

        dx, dy = offset_patterns[idx % len(offset_patterns)]
        if lat > (x_left + 0.75 * (x_right - x_left)):
            dx = -abs(dx)
            ha = "right"
        else:
            ha = "left"

        title_str = f"{model_name} ({exp_name})" if exp_name else model_name
        ax.annotate(
            f"{title_str}\nAcc: {acc:.4f} | {lat:.2f} ms | {size:.2f} MB",
            xy=(lat, acc),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.1),
            zorder=5,
        )

    ax.axvline(
        budget_ms,
        color="#dc2626",
        linestyle="--",
        linewidth=1.6,
        label=f"Stage 1 Budget ({budget_ms:.0f} ms)",
        zorder=3,
    )
    ax.axvspan(0, budget_ms, color="#22c55e", alpha=0.07, label=f"Target Zone (≤ {budget_ms:.0f} ms)", zorder=1)

    ax.set_title("Test Accuracy vs. CPU Latency", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("CPU Latency (ms/image)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Accuracy", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_side_by_side_cm(records, out_path):
    cm_records = [r for r in records if "confusion_matrix" in r]
    cm_records.sort(key=lambda r: (r.get("week", 999), r.get("experiment", "")))

    n = len(cm_records)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 5.0))
    if n == 1:
        axes = [axes]

    global_max = max(np.array(r["confusion_matrix"]).max() for r in cm_records)

    for ax, rec in zip(axes, cm_records):
        cm = np.array(rec["confusion_matrix"])
        model_name = rec.get("model", "Model")
        exp_name = rec.get("experiment", "")
        title = f"{model_name}\n({exp_name})" if exp_name else model_name

        ax.imshow(cm, cmap=plt.cm.Blues, interpolation="nearest", vmin=0, vmax=global_max)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["bad", "good"])
        ax.set_yticklabels(["bad", "good"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white" if cm[i, j] > (global_max * 0.5) else "black",
                )

    total_samples = cm_records[0].get("test_samples", int(np.array(cm_records[0]["confusion_matrix"]).sum()))
    plt.suptitle(f"Test Confusion Matrices (N = {total_samples})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


parser = argparse.ArgumentParser(description="Generate Task 4 plots dynamically from results.json files.")
parser.add_argument("--results_dir", type=Path, default=Path("results"))
parser.add_argument("--budget_ms", type=float, default=100.0)
args = parser.parse_args()

records = discover_results(args.results_dir)
print(f"Loaded {len(records)} model result file(s):")
for r in records:
    print(f"  - {r['_source_path']} ({r.get('model')}: acc={r['metrics']['accuracy']}, lat={r['latency_ms_per_image']} ms)")

acc_lat_path = args.results_dir / "accuracy_vs_latency.png"
cm_side_path = args.results_dir / "side_by_side_cm.png"

plot_accuracy_vs_latency_line(records, acc_lat_path, budget_ms=args.budget_ms)
plot_side_by_side_cm(records, cm_side_path)

print(f"\nSaved: {acc_lat_path}")
print(f"Saved: {cm_side_path}")