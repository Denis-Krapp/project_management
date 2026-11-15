import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class QualityFolderDataset(Dataset):
    def __init__(self, folder_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {"bad": 0, "good": 1}

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = folder_dir / cls_name
            if cls_folder.exists():
                for img_p in sorted(cls_folder.glob("*.*")):
                    if img_p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        self.samples.append((img_p, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(path)


def load_and_unfreeze_top30(checkpoint_path, device, top_n_layers = 30):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    for param in model.parameters():
        param.requires_grad = False

    all_named_params = list(model.named_parameters())
    for name, param in all_named_params[-top_n_layers:]:
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Unfroze top {top_n_layers} parameter layers: {trainable_params:,} / {total_params:,} trainable params")
    return model.to(device)


def measure_pytorch_latency(model, test_loader, device, num_warmup=20, num_timed=100):
    model.eval()
    images_list = []
    for imgs, _, _ in test_loader:
        for img in imgs:
            images_list.append(img.unsqueeze(0).to(device))
            if len(images_list) >= (num_warmup + num_timed):
                break
        if len(images_list) >= (num_warmup + num_timed):
            break

    with torch.no_grad():
        for i in range(num_warmup):
            _ = model(images_list[i])

    start = time.perf_counter()
    with torch.no_grad():
        for i in range(num_warmup, num_warmup + num_timed):
            _ = model(images_list[i])
    elapsed = time.perf_counter() - start
    return round((elapsed / num_timed) * 1000.0, 3)


def plot_metrics(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss", color="royalblue")
    ax1.plot(history["val_loss"], label="Val Loss", color="crimson")
    ax1.set_title("ResNet-18 Fine-Tune Top 30: Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc", color="royalblue")
    ax2.plot(history["val_acc"], label="Val Acc", color="crimson")
    ax2.set_title("ResNet-18 Fine-Tune Top 30: Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cm(cm, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["bad", "good"],
        yticklabels=["bad", "good"],
        title="ResNet-18 Fine-Tune Top 30 CM",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontweight="bold"
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def copy_errors(test_loader, y_true, y_pred, y_prob, errors_dir):
    if errors_dir.exists():
        shutil.rmtree(errors_dir)
    errors_dir.mkdir(parents=True, exist_ok=True)

    idx_to_class = {0: "bad", 1: "good"}
    all_paths = [p for _, _, paths in test_loader for p in paths]
    error_count = 0
    for idx, (gt, pred) in enumerate(zip(y_true, y_pred)):
        if gt != pred:
            error_count += 1
            src = Path(all_paths[idx])
            conf = y_prob[idx] if pred == 1 else (1.0 - y_prob[idx])
            dst = f"err_{error_count:03d}_true_{idx_to_class[gt]}_pred_{idx_to_class[pred]}_conf_{conf:.2f}_{src.name}"
            shutil.copy2(src, errors_dir / dst)
    print(f"Copied {error_count} misclassified test images to {errors_dir}")


parser = argparse.ArgumentParser(description="Task 5A: Partial Fine-Tuning (Top 30 Layers)")
parser.add_argument("--data_dir", type=Path, default=Path("data/quality_split"))
parser.add_argument("--frozen_ckpt", type=Path, default=Path("results/resnet18_frozen/best_resnet18.pth"))
parser.add_argument("--frozen_json", type=Path, default=Path("results/resnet18_frozen/results.json"))
parser.add_argument("--results_dir", type=Path, default=Path("results/resnet18_finetune_top30"))
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=245000004)
args = parser.parse_args()

set_seed(args.seed)
args.results_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cpu")

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = QualityFolderDataset(args.data_dir / "train", transform=tfm)
val_ds = QualityFolderDataset(args.data_dir / "validation", transform=tfm)
test_ds = QualityFolderDataset(args.data_dir / "test", transform=tfm)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

model = load_and_unfreeze_top30(args.frozen_ckpt, device, top_n_layers=30)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

best_val_loss = float("inf")
best_wts = copy.deepcopy(model.state_dict())
best_epoch = 0
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

start_t = time.time()
epochs_run = 0
for epoch in range(args.epochs):
    epochs_run += 1
    model.train()
    r_loss, r_corr = 0.0, 0
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        r_loss += loss.item() * inputs.size(0)
        r_corr += torch.sum((torch.sigmoid(outputs) >= 0.5).int() == labels.int()).item()

    tr_loss = r_loss / len(train_ds)
    tr_acc = r_corr / len(train_ds)

    model.eval()
    v_loss, v_corr = 0.0, 0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            v_loss += loss.item() * inputs.size(0)
            v_corr += torch.sum((torch.sigmoid(outputs) >= 0.5).int() == labels.int()).item()

    val_loss = v_loss / len(val_ds)
    val_acc = v_corr / len(val_ds)

    history["train_loss"].append(round(tr_loss, 5))
    history["val_loss"].append(round(val_loss, 5))
    history["train_acc"].append(round(tr_acc, 5))
    history["val_acc"].append(round(val_acc, 5))

    print(f"Epoch {epoch+1:02d}/{args.epochs} - Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_wts = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), args.results_dir / "best_resnet18_finetune_top30.pth")
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch+1} (best epoch: {best_epoch})")
            break

train_time_s = round(time.time() - start_t, 2)
model.load_state_dict(best_wts)
plot_metrics(history, args.results_dir / "training_curves.png")

model.eval()
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for inputs, labels, _ in test_loader:
        outputs = model(inputs.to(device))
        probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_pred.extend((probs >= 0.5).astype(int).tolist())
        y_true.extend(labels.numpy().tolist())

y_true, y_pred = np.array(y_true), np.array(y_pred)
acc = round(float(accuracy_score(y_true, y_pred)), 4)
p_macro = round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4)
r_macro = round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4)
f1_m = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
cm = confusion_matrix(y_true, y_pred)

plot_cm(cm, args.results_dir / "confusion_matrix.png")
copy_errors(test_loader, y_true, y_pred, y_prob, args.results_dir / "errors")

latency_ms = measure_pytorch_latency(model, test_loader, device)
ckpt_file = args.results_dir / "best_resnet18_finetune_top30.pth"
size_mb = round(ckpt_file.stat().st_size / (1024 * 1024), 2)
total_params = sum(p.numel() for p in model.parameters())

results = {
    "experiment": "task5a_resnet18_finetune_top30",
    "week": 9,
    "model": "ResNet18",
    "regime": "finetune_top30",
    "lr": args.lr,
    "unfrozen_layers": 30,
    "input_size": [224, 224, 3],
    "seed": args.seed,
    "device": "cpu",
    "epochs_planned": args.epochs,
    "epochs_run": epochs_run,
    "best_epoch": best_epoch,
    "train_time_s": train_time_s,
    "latency_ms_per_image": latency_ms,
    "model_size_mb": size_mb,
    "params": total_params,
    "test_samples": len(y_true),
    "train_val_gap_at_best": round(history["train_acc"][best_epoch - 1] - history["val_acc"][best_epoch - 1], 4),
    "metrics": {
        "accuracy": acc,
        "f1_macro": f1_m,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
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
    "history": history,
}

out_json = args.results_dir / "results.json"
with out_json.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

if args.frozen_json.exists():
    frozen = json.loads(args.frozen_json.read_text(encoding="utf-8"))
    fm = frozen["metrics"]
    print("\n=== FROZEN vs. FINETUNE_TOP30 DELTA ===")
    print(f"Accuracy:  {fm['accuracy']:.4f} -> {acc:.4f} (Delta: {acc - fm['accuracy']:+.4f})")
    print(f"Macro F1:  {fm['f1_macro']:.4f} -> {f1_m:.4f} (Delta: {f1_m - fm['f1_macro']:+.4f})")
    print(f"Latency:   {frozen['latency_ms_per_image']:.3f} ms -> {latency_ms:.3f} ms")
    print(f"Size:      {frozen['model_size_mb']:.2f} MB -> {size_mb:.2f} MB")
    print(f"Train-Val Gap (best epoch {best_epoch}): {results['train_val_gap_at_best']:+.4f}")
