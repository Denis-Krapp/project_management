import argparse
import json
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_dataset(split_dir):
    print("\n--- [Step 1] Verifying Dataset Splits ---")
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory '{split_dir}' not found. Run make_split_folders.py first.")

    counts = {}
    for split in ["train", "validation", "test"]:
        split_path = split_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split folder: {split_path}")
        counts[split] = {}
        for cls in ["good", "bad"]:
            cls_path = split_path / cls
            c = len(list(cls_path.glob("*.*"))) if cls_path.exists() else 0
            counts[split][cls] = c
            print(f"  {split:>10} / {cls:<4}: {c:4d} images")
    return counts


def build_generators(split_dir, batch_size, seed):
    print("\n--- [Step 2] Building Data Generators ---")
    datagen_train = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
    )
    datagen_eval = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
    )

    train_gen = datagen_train.flow_from_directory(
        directory=str(split_dir / "train"),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=seed,
    )

    val_gen = datagen_eval.flow_from_directory(
        directory=str(split_dir / "validation"),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = datagen_eval.flow_from_directory(
        directory=str(split_dir / "test"),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def build_mobilenetv3_model(input_shape=(224, 224, 3)):
    print("\n--- [Step 3] Building Frozen MobileNetV3-Large ---")
    base_model = tf.keras.applications.MobileNetV3Large(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_03")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_binary")(x)

    model = models.Model(inputs, outputs, name="MobileNetV3_Quality_Baseline")
    return model


def plot_and_save_curves(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="Train Loss", color="royalblue")
    ax1.plot(history.history["val_loss"], label="Val Loss", color="crimson")
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax1_twin = ax2
    ax1_twin.plot(history.history["accuracy"], label="Train Acc", color="royalblue")
    ax1_twin.plot(history.history["val_accuracy"], label="Val Acc", color="crimson")
    ax1_twin.set_title("Binary Accuracy")
    ax1_twin.set_xlabel("Epoch")
    ax1_twin.set_ylabel("Accuracy")
    ax1_twin.grid(True, linestyle="--", alpha=0.6)
    ax1_twin.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_and_save_cm(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix (Test Split)",
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


def measure_latency_protocol(model, test_gen, num_warmup=20, num_timed=100):
    print("\n--- [Step 9] Latency Protocol (20 Warm-up + 100 Timed) ---")
    test_images = []
    for i in range(len(test_gen)):
        batch_x, _ = test_gen[i]
        for img in batch_x:
            test_images.append(np.expand_dims(img, axis=0))
            if len(test_images) >= (num_warmup + num_timed):
                break
        if len(test_images) >= (num_warmup + num_timed):
            break

    for i in range(num_warmup):
        _ = model(test_images[i], training=False)

    start_time = time.perf_counter()
    for i in range(num_warmup, num_warmup + num_timed):
        _ = model(test_images[i], training=False)
    end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) / num_timed) * 1000.0
    print(f"Measured Latency: {avg_ms:.2f} ms/image over {num_timed} single inferences")
    return round(avg_ms, 3)


def copy_misclassified_images(test_gen, y_true, y_pred, y_prob, errors_dir):
    if errors_dir.exists():
        shutil.rmtree(errors_dir)
    errors_dir.mkdir(parents=True, exist_ok=True)

    idx_to_label = {v: k for k, v in test_gen.class_indices.items()}
    file_paths = test_gen.filepaths

    error_count = 0
    for idx, (gt, pred) in enumerate(zip(y_true, y_pred)):
        if gt != pred:
            error_count += 1
            src_file = Path(file_paths[idx])
            pred_label = idx_to_label[pred]
            gt_label = idx_to_label[gt]
            conf = y_prob[idx] if pred == 1 else (1.0 - y_prob[idx])
            dst_name = f"err_{error_count:03d}_true_{gt_label}_pred_{pred_label}_conf_{conf:.2f}_{src_file.name}"
            shutil.copy(src_file, errors_dir / dst_name)

    print(f"Copied {error_count} misclassified test images to {errors_dir}")
    return error_count


parser = argparse.ArgumentParser(description="Task 2: MobileNetV3 Baseline Quality Classifier")
parser.add_argument("--data_dir", type=Path, default=Path("data/quality_split"))
parser.add_argument("--results_dir", type=Path, default=Path("results/mobilenetv3_frozen"))
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=20261023)
args = parser.parse_args()

set_seed(args.seed)
args.results_dir.mkdir(parents=True, exist_ok=True)

verify_dataset(args.data_dir)

train_gen, val_gen, test_gen = build_generators(args.data_dir, args.batch_size, args.seed)
labels_map = train_gen.class_indices
class_names = [k for k, v in sorted(labels_map.items(), key=lambda item: item[1])]
print(f"Class indices: {labels_map}")

model = build_mobilenetv3_model(input_shape=(224, 224, 3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

best_weights_path = args.results_dir / "best_mobilenetv3_checkpoint.keras"
callbacks = [
    EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=str(best_weights_path), monitor="val_loss", save_best_only=True, verbose=1),
]

print("\n--- [Step 5] Training Baseline ---")
start_train_t = time.time()
history = model.fit(
    train_gen,
    epochs=args.epochs,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1,
)
train_time_s = round(time.time() - start_train_t, 2)
epochs_run = len(history.history["loss"])

curves_img_path = args.results_dir / "training_curves.png"
plot_and_save_curves(history, curves_img_path)

print("\n--- [Step 6] Test Split Evaluation ---")
test_gen.reset()
probs = model.predict(test_gen, verbose=1).flatten()
y_pred = (probs >= 0.5).astype(int)
y_true = test_gen.classes

acc = float(accuracy_score(y_true, y_pred))
p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

cm = confusion_matrix(y_true, y_pred)
cm_path = args.results_dir / "confusion_matrix.png"
plot_and_save_cm(cm, class_names, cm_path)

per_class = {}
for cls_name, cls_idx in labels_map.items():
    cls_p = float(precision_score(y_true == cls_idx, y_pred == cls_idx, zero_division=0))
    cls_r = float(recall_score(y_true == cls_idx, y_pred == cls_idx, zero_division=0))
    cls_f1 = float(f1_score(y_true == cls_idx, y_pred == cls_idx, zero_division=0))
    cls_sup = int(np.sum(y_true == cls_idx))
    per_class[cls_name] = {
        "precision": round(cls_p, 4),
        "recall": round(cls_r, 4),
        "f1": round(cls_f1, 4),
        "support": cls_sup,
    }

errors_dir = args.results_dir / "errors"
copy_misclassified_images(test_gen, y_true, y_pred, probs, errors_dir)

latency_ms = measure_latency_protocol(model, test_gen)

model_size_mb = round(best_weights_path.stat().st_size / (1024 * 1024), 2)
params_count = int(model.count_params())

device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
results = {
    "experiment": "task2_baseline",
    "week": 6,
    "model": "MobileNetV3Large",
    "regime": "frozen_base",
    "input_size": [224, 224, 3],
    "seed": args.seed,
    "device": device_name,
    "epochs_planned": args.epochs,
    "epochs_run": epochs_run,
    "train_time_s": train_time_s,
    "latency_ms_per_image": latency_ms,
    "model_size_mb": model_size_mb,
    "params": params_count,
    "test_samples": len(y_true),
    "metrics": {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_m, 4),
        "precision_macro": round(p_macro, 4),
        "recall_macro": round(r_macro, 4),
        "per_class": per_class,
    },
    "confusion_matrix": cm.tolist(),
    "history": {k: [round(v, 5) for v in vals] for k, vals in history.history.items()},
}

results_json_path = args.results_dir / "results.json"
with results_json_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nExecution Complete.")
print(f"Results saved to: {results_json_path}")
print(f"Test Accuracy: {acc:.4f} | Macro F1: {f1_m:.4f} | Latency: {latency_ms} ms")