### 1. Comparison table

| Model | accuracy | precision macro | recall macro | F1 macro | latency ms/image | size MB | parameters | training time | epochs run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large (Task 2) | 0.9961 | 0.9962 | 0.9961 | 0.9961 | 127.952 | 13.51 | 3,119,489 | 274.32 s | 20 |
| YOLOv8n-cls (Task 3) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 18.862 | 2.83 | 1,440,850 | 2,244.45 s | 15 |
| YOLOv8s-cls (Task 3) | 0.9380 | 0.9448 | 0.9380 | 0.9377 | 44.512 | 9.78 | 5,083,298 | 1,994.65 s | 6 |

### 2. Architectural comparison and analysis

Evaluating both YOLOv8 variants against the Task 2 MobileNetV3-Large baseline on the held-out 258-image test split demonstrates decisive advantages in inference speed, storage efficiency, and generalization for the compact nano architecture.

While MobileNetV3-Large achieved a strong test accuracy of 0.9961 under frozen linear probing at 224×224 resolution, its uncompiled Keras CPU latency of 127.952 ms/image breached our pre-registered <100 ms first-gate budget.

In contrast, YOLOv8n-cls trained end-to-end at native 640×640 resolution converged via early stopping at epoch 1 and achieved 1.0000 test accuracy and 1.0000 Macro F1 (0 errors across all 258 test images), while cutting CPU latency by 6.78× down to 18.862 ms/image.

Comparing the two YOLO variants directly reveals the impact of model capacity on a 1,272-image training split: YOLOv8s-cls (5,083,298 parameters) peaked on validation at epoch 1 and triggered early stopping at epoch 6 (`patience=5`), resulting in premature convergence and 16 false negatives on defective documents (`bad` recall = 0.8760, accuracy = 0.9380) at more than double the latency (44.512 ms/image).

Architecturally, YOLOv8n-cls's lightweight C2f backbone (1,440,850 parameters, 2.83 MB) acts as a natural regularizer at 640×640 resolution, capturing fine text-edge degradation without overfitting or incurring memory latency penalties, making it the clear production gatekeeper for the Telegram Bot ingestion pipeline.