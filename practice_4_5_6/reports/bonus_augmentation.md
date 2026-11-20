# Task 5C: Error-Driven Glare Augmentation Report

- Model architecture: YOLOv8n-cls (`results/week9_yolov8n_glare_aug/weights/best.pt`)
- Training protocol: epochs = 50 (early stopped at Epoch 13, best checkpoint at Epoch 8), patience = 5, batch_size = 16, imgsz = 640, seed = 245000004
- Augmentation config (train split only): Added 318 augmented samples (50% of train/bad, expanding train from 1,272 to 1,590 images) using Gaussian-bloomed white specular ellipses (alpha = 150-230) and 50% directional motion blur; validation (N = 270) and test (N = 258) remained completely untouched.


### 1. Per-class and per-defect test delta table (N = 258)

| Category | Support | Baseline (YOLOv8n-cls) | After Glare Aug (Task 5C) | Delta | Errors After |
| --- | --- | --- | --- | --- | --- |
| glare (Target Defect) | 31 | 1.0000 | 0.9677 | -0.0323 | 1 / 31 |
| exposure | 24 | 1.0000 | 0.9583 | -0.0417 | 1 / 24 |
| blur | 22 | 1.0000 | 1.0000 | +0.0000 | 0 / 22 |
| occlusion | 31 | 1.0000 | 1.0000 | +0.0000 | 0 / 31 |
| skew | 21 | 1.0000 | 1.0000 | +0.0000 | 0 / 21 |
| good | 129 | 1.0000 | 1.0000 | +0.0000 | 0 / 129 |
| Overall Test Accuracy | 258 | 1.0000 | 0.9922 | -0.0078 | 2 / 258 |


### 2. Five-line conclusion and analysis

1. Specular glare was targeted because Task 4 error analysis and the capstone baseline (F1 = 0.92) identified localized white flash reflections over low-text paper margins as the hardest defect to separate from clean white documents.
2. Did it help or hurt? Because the baseline YOLOv8n-cls was already saturated at 1.0000 across all test categories, stacking synthetic alpha = 150-230 glare ellipses onto train/bad slightly degraded glare recall (-0.0323, 1 missed image) and spilled over into exposure (-0.0417, 1 missed image), lowering overall test accuracy from 1.0000 to 0.9922 (256/258).
3. Adding 318 heavy white-ellipse overlays shifted the training class balance (954 bad vs 636 good) and biased the filter weights toward sharp synthetic bloom edges, slightly reducing sensitivity to subtle, diffuse exposure washouts while leaving good (suitable), blur, occlusion, and skew untouched at 1.0000.
4. Why validation and test must stay clean? Augmenting validation or test alters the evaluation distribution and injects synthetic artifacts into benchmark scores, hiding real-world generalization; keeping both splits unmodified ensures an honest apples-to-apples comparison.
5. Monitoring & Control verdict: This experiment completes the Monitoring & Control cycle (measure errors $\rightarrow$ analyze dominant defect $\rightarrow$ correct training data $\rightarrow$ re-measure per-class deltas); because the Task 3 baseline already achieves 1.0000 without synthetic class-imbalance drift, we retain `results/week7_yolov8n_cls/weights/best.pt` for production and reserve targeted glare augmentation for future real-world operator failure samples.

### 3. Explanation checkpoint
* What happens if glare F1 rises while good (suitable) precision falls? The model becomes overly conservative, misclassifying bright, glossy, or high-luminance clean documents as specular glare and triggering unnecessary user retake prompts.
* Why is this the Monitoring & Control phase? Rather than blindly tuning hyperparameters, changes to the training pipeline are triggered strictly by empirical per-defect failure diagnostics and verified via per-class delta measurement before deployment.