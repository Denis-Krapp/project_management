# Practice 6. Task 1. Quality Classification Model. Model Selection

Name: Krapp Denis Vladimirovich

Group: AIb24-1

## Candidate Architecture Comparison Table

| Architecture | Citation | Input Size | Approximate Params | GFLOPs | Key Architectural Strength | Pipeline Risk |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large (Baseline, Task1) | [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) | 224 x 224 | 5,4 M | 0,28 | Hard-swish replaces sigmoid/exp activations, eliminating costly transcendental functions on CPU while retaining attention benefits. | --- |
| YOLOv8n-cls (Task3) | [A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS](https://arxiv.org/abs/2304.00501)  | 224 x 224 | 2,7 M | 4,30 | Smallest binary size (6 MB); C2f blocks optimize gradient flow across parallel split paths; fastest CPU latency (2-4 ms). | --- |
| ResNet-18 (Task4) | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 224 x 224 | 11,7 M | 1,80 | Dense 3 by 3 convolutions preserve joint spatial-channel correlations; high inductive bias stabilizes small-sample transfer learning. | Computational overhead is higher on edge; low-power CPU microservices. |
| EfficientNet-B0 (Task 4) | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | 224 x 224 | 5,3 M | 0,39 | Channel attention dynamically recalibrates channel weights, suppressing background and emphasizing text boundaries. | Higher Memory Access Cost due to dynamic SE branches and Swish activations yields higher wall-clock CPU latency than FLOPs suggest. |

## Transfer Learning Foundations
In the Frozen Base regime, all pretrained convolutional backbone layers are frozen. A task-specific classification head is appended and trained exclusively via binary cross-entropy loss. The base learns nothing: its convolutional kernels, batch normalization statistics, and spatial weights remain static. The model strictly trains the dense classification head to establish an optimal separating hyperplane across the frozen feature embeddings.

In the Fine-Tuning regime, backbone layers are unfrozen, allowing backpropagation to adjust convolutional weights throughout the network. To prevent catastrophic forgetting (overwriting general-purpose edge and texture filters with noisy gradients from small synthetic batches), fine-tuning must use a significantly reduced learning rate and should only begin after the classification head has converged during an initial frozen warm-up stage.

## Pre-Registered Numeric Acceptance Criteria

| Metric | Threshold |
| --- | --- |
| Overall Test Accuracy | >= 90% |
| Macro F1-Score | >= 90% |
| Defect Recall | >= 95% |
| Single-Image CPU Latency | <= 100 ms, Target: <= 30 ms |
| Model Binary Size | <= 50 MB, Target: <= 30 MB |

## Questions

1) The quality classifier is positioned at the entry point of the Telegram Bot ingestion worker. Every single image submitted by a user must pass this gate. The user waits synchronously in the Telegram chat. The entire roundtrip must complete within <= 2 seconds. Heavy backbones (e.g., ResNet-50) violate this budget and are excluded.
2) Training any deep CNN from random weight initialization on about 1200 samples causes catastrophic overfitting. The model memorizes synthetic font kerning, paper textures, and border geometry rather than learning general optical defect primitives. Transfer learning from ImageNet pretrained weights is mandatory. Early convolutional layers must leverage existing general-purpose Gabor-like edge filters and spatial frequency discriminators.
3) Lightweight weights minimize container cold-start delays, allow high worker concurrency in low-RAM environments, and enable frictionless mobile execution. Also, by setting our pre-registered synthetic test threshold at 90% guarantees that even under the maximum anticipated 5.8% transfer penalty, real-world deployment performance remains above the 85% threshold.


