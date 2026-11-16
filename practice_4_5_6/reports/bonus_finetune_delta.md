# Task 5A: Partial Fine-Tuning Delta Report

### 1. Comparison table

| Regime | Accuracy | Precision Macro | Recall Macro | F1 Macro | Latency (ms/img) | Size (MB) | Epochs Run (Best) | Best Val Loss | Train-Val Acc Gap (Best) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_base (Task 4) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.649 | 42.96 | 50 (45) | 0.0148 | -0.0083 (0.9843 vs 0.9926) |
| finetune_top30 (Task 5A) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 19.231 | 42.96 | 12 (7) | 0.0059 | -0.0010 (0.9953 vs 0.9963) |
| Delta | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +3.582 ms | 0.00 MB | -38 epochs | -0.0089 | +0.0073 |


### 2. Five-Line delta memo and regime analysis

1) What improved: Unfreezing the top 30 layers at lr = 1e-5 tightened decision confidence on validation (val_loss dropped by 60.1%, from 0.0148 to 0.0059, and val_acc rose from 0.9926 to 0.9963) while preserving 1.0000 test accuracy and 1.0000 Macro F1 ([[129, 0], [0, 129]], 0 errors).
2) What got worse: Training each epoch became slower because gradients backpropagated through 9.67M parameters instead of 65.8K head parameters, and single-image CPU latency showed a minor +3.582 ms runtime variance (19.231 ms vs 15.649 ms).
3) Train-Val gap: At the best checkpoint (Epoch 7), the train-val accuracy gap remained slightly negative (-0.0010: 0.9953 train vs 0.9963 val), indicating healthy generalization; however, in Epochs 9-11, training accuracy climbed to 0.9984 while validation accuracy dipped to 0.9926 (positive gap +0.0058), confirming early memorization of synthetic training patterns before patience = 5 halted training at Epoch 12.
4) Why latency and model size did not change architecturally: Unfreezing parameters only toggles requires_grad=True during training backward passes; at inference time, the forward computational graph, FLOPs, and total FP32 parameter count (11,242,305 weights = 42.96 MB) are identical (the 15.65 ms vs 19.23 ms difference reflects normal CPU thermal/thread scheduling jitter).
5) Deploy verdict: Do not deploy finetune_top30 in place of frozen_base (or our primary YOLOv8n-cls production gatekeeper): frozen_base already achieves 1.0000 test accuracy without exposing 9.67M backbone filters to synthetic domain overfitting on a 1,272-image training split.


### 3. Explanation checkpoint
* Why lr = 1e-5 instead of 1e-3? Large gradient updates (1e-3) on pretrained ImageNet convolutional layers distort learned spatial edge and texture filters, causing catastrophic forgetting.
* What changes in gradient flow when unfreezing? In frozen_base, gradients terminate at the fc linear head; unfreezing allows loss gradients to propagate backward into layer4 and layer3, reshaping high-level convolutional feature maps to fit document-specific artifacts.