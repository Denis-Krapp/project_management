### 1. Decision
YOLOv8n-cls is selected as the production quality gatekeeper for the Telegram bot ingestion pipeline.

### 2. Comparison table

| Model | accuracy | precision macro | recall macro | F1 macro | latency ms/image | size MB | parameters | training time | epochs run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large | 0.9961 | 0.9962 | 0.9961 | 0.9961 | 127.952 | 13.51 | 3,119,489 | 274.32 s | 20 |
| YOLOv8n-cls | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 18.862 | 2.83 | 1,440,850 | 2,244.45 s | 15 |
| YOLOv8s-cls | 0.9380 | 0.9448 | 0.9380 | 0.9377 | 44.512 | 9.78 | 5,083,298 | 1,994.65 s | 6 |
| ResNet-18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.649 | 42.96 | 11,242,305 | 1,742.18 s | 50 |

* YOLOv8n-cls achieved 1.0000 test accuracy and 1.0000 Macro F1 ([[129, 0], [0, 129]] confusion matrix - 0 false positives and 0 false negatives across all 258 test samples), clearing the >90.8% synthetic threshold required to sustain >85% real-world accuracy.
* YOLOv8n-cls executes in 18.862 ms/image on CPU - 5.3× faster than the < 100 ms hard budget of the first pipeline gate and well inside our ≤ 30 ms engineering target, preserving >1.95 seconds of the 2.0-second Telegram roundtrip budget for downstream OCR.
* At 2.83 MB (1,440,850 parameters), YOLOv8n-cls occupies only 5.7% of the 50 MB maximum size constraint and is 15.2× smaller than ResNet-18.


### 3. Rejected alternatives
1) MobileNetV3-Large - rejected due to latency violation:  
   Despite strong classification metrics (0.9961 accuracy, 0.9961 Macro F1, 1 error), uncompiled TensorFlow/Keras CPU execution required 127.952 ms/image, violating the mandatory < 100 ms Stage 1 latency budget by 28%. Its checkpoint (13.51 MB) is also 4.8× heavier than YOLOv8n-cls.
2) YOLOv8s-cls - rejected due to premature convergence and lower accuracy:  
   With 3.5x more parameters (5,083,298 vs. 1,440,850), YOLOv8s-cls hit its validation peak at epoch 1 and stopped early at epoch 6 (patience=5), dropping to 0.9380 accuracy (0.9377 Macro F1) with 16 false negatives on defective images (bad recall = 0.8760) while increasing CPU latency to 44.512 ms/image and size to 9.78 MB.
3) ResNet-18 - rejected as primary gate due to memory footprint and resolution mismatch:  
   ResNet-18 matched YOLOv8n-cls in accuracy (1.0000, 0 errors) and achieved the fastest raw CPU latency (15.649 ms/image) at 224×224 resolution via dense 3x3 BLAS/GEMM convolutions. However, its 42.96 MB checkpoint (11,242,305 parameters) is 15.2× larger than YOLOv8n-cls (approaching the 50 MB ceiling) and forces a separate 224×224 preprocessing branch instead of sharing the native 640×640 tensor used by downstream document detectors.


### 4. Risks and mitigation
* Synthetic-to-Real domain gap (2.8-5.8%):  
  Benchmarks in the thesis document a 2.8-5.8% accuracy drop when moving from synthetic defect augmentation to real smartphone photos in production. Subtracting this gap from our 100.0% synthetic accuracy yields a projected 94.2-97.2% real-world accuracy, remaining comfortably above the 85.0% SLA, though unseen backgrounds and lighting require active safeguards.
* Specular glare weakness:  
  Error analysis across the defect taxonomy (blur, glare, skew, occlusion, exposure) confirms that glare is the hardest defect overall because localized flash reflections over low-text margins resemble clean white paper.
* Mitigation plan:  
  1) Upgrade apply_glare from uniform ellipses to multi-layer radial Gaussian alpha gradients with local contrast washout anchored directly over high-density text regions (passport numbers, dates, stamps).
  2) Lower the rejection cutoff from P(bad) >= 0.50 to P(bad) >= 0.40 in production so borderline glare frames are rejected before consuming expensive OCR compute.
  3) Route uncertain predictions (0.35 <= P(bad) < 0.65) and user-disputed rejections into an asynchronous operator review queue to harvest real-world failure cases for quarterly fine-tuning.


### 5. Deployment note
* Current deployment:  
  Deploy results/week7_yolov8n_cls/weights/best.pt inside the Telegram bot backend container on CPU. At 18.862 ms/image, a single worker processes ~53 images/sec and blocks degraded or non-document uploads before invoking downstream PaddleOCR.
* Future deployment:  
  Because YOLOv8n-cls is only 2.83 MB, it can be exported via Ultralytics (model.export(format="onnx", simplify=True) or format="tflite", int8=True) to run directly on mobile clients or edge gateways (<1.5 MB quantized), providing instant on-device camera quality feedback prior to cellular upload.