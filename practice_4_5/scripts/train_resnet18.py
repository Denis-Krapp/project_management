import argparse
import copy
import json
import os
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
        self.folder_dir = folder_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {"bad": 0, "good": 1}

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = folder_dir / cls_name
            if cls_folder.exists():
                for img_p in sorted(cls_folder.glob("*.*")):
                    if img_p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.samples.append((img_p, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(path)


def build_resnet18(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )
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

    avg_ms = (elapsed / num_timed) * 1000.0
    return round(avg_ms, 3)


def plot_metrics(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss", color="royalblue")
    ax1.plot(history["val_loss"], label="Val Loss", color="crimson")
    ax1.set_title("ResNet-18 Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc", color="royalblue")
    ax2.plot(history["val_acc"], label="Val Acc", color="crimson")
    ax2.set_title("ResNet-18 Binary Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cm(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="ResNet-18 Confusion Matrix (Test)",
        ylabel="Ground Truth",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def copy_errors(test_loader, y_true, y_pred, y_prob, errors_dir):
    if errors_dir.exists():
        shutil.rmtree(errors_dir)
    errors_dir.mkdir(parents=True, exist_ok=True)

    idx_to_class = {0: "bad", 1: "good"}
    all_paths = []
    for _, _, paths in test_loader:
        all_paths.extend(paths)

    error_count = 0
    for idx, (gt, pred) in enumerate(zip(y_true, y_pred)):
        if gt != pred:
            error_count += 1
            src_file = Path(all_paths[idx])
            prob = y_prob[idx] if pred == 1 else (1.0 - y_prob[idx])
            dst_name = f"err_{error_count:03d}_true_{idx_to_class[gt]}_pred_{idx_to_class[pred]}_conf_{prob:.2f}_{src_file.name}"
            shutil.copy2(src_file, errors_dir / dst_name)

    print(f"Copied {error_count} misclassified test images to {errors_dir}")


parser = argparse.ArgumentParser(description="Task 4: ResNet-18 Quality Classifier")
parser.add_argument("--data_dir", type=Path, default=Path("data/quality_split"))
parser.add_argument("--results_dir", type=Path, default=Path("results/resnet18_frozen"))
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=245000004)
args = parser.parse_args()

set_seed(args.seed)
args.results_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = QualityFolderDataset(args.data_dir / "train", transform=data_transform)
val_ds = QualityFolderDataset(args.data_dir / "validation", transform=data_transform)
test_ds = QualityFolderDataset(args.data_dir / "test", transform=data_transform)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

model = build_resnet18(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

best_val_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())
patience_counter = 0

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("\n--- Training ResNet-18 (Frozen Backbone) ---")
start_train_t = time.time()
epochs_run = 0

for epoch in range(args.epochs):
    epochs_run += 1
    model.train()
    running_loss = 0.0
    corrects = 0

    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).int()
        corrects += torch.sum(preds == labels.int()).item()

    epoch_train_loss = running_loss / len(train_ds)
    epoch_train_acc = corrects / len(train_ds)

    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).int()
            val_corrects += torch.sum(preds == labels.int()).item()

    epoch_val_loss = val_loss / len(val_ds)
    epoch_val_acc = val_corrects / len(val_ds)

    history["train_loss"].append(round(epoch_train_loss, 5))
    history["val_loss"].append(round(epoch_val_loss, 5))
    history["train_acc"].append(round(epoch_train_acc, 5))
    history["val_acc"].append(round(epoch_val_acc, 5))

    print(f"Epoch {epoch+1:02d}/{args.epochs} - "
            f"Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
        torch.save(model.state_dict(), args.results_dir / "best_resnet18.pth")
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

train_time_s = round(time.time() - start_train_t, 2)
model.load_state_dict(best_model_wts)

plot_metrics(history, args.results_dir / "training_curves.png")

print("\n--- Evaluating on Test Split ---")
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for inputs, labels, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        y_prob.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = float(accuracy_score(y_true, y_pred))
p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
cm = confusion_matrix(y_true, y_pred)

plot_cm(cm, ["bad", "good"], args.results_dir / "confusion_matrix.png")
copy_errors(test_loader, y_true, y_pred, y_prob, args.results_dir / "errors")

latency_ms = measure_pytorch_latency(model, test_loader, device)

best_pth = args.results_dir / "best_resnet18.pth"
model_size_mb = round(best_pth.stat().st_size / (1024 * 1024), 2)
param_count = sum(p.numel() for p in model.parameters())

results = {
    "experiment": "task4_resnet18_frozen",
    "week": 8,
    "model": "ResNet18",
    "regime": "frozen_base",
    "input_size": [224, 224, 3],
    "seed": args.seed,
    "device": "cpu",
    "epochs_planned": args.epochs,
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
    "history": history,
}

out_json = args.results_dir / "results.json"
with out_json.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nExecution Complete.")
print(f"Results saved to: {out_json}")
print(f"Test Accuracy: {acc:.4f} | Macro F1: {f1_m:.4f} | Latency: {latency_ms} ms | Size: {model_size_mb} MB")
