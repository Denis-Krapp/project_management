# Model Card: Stage 1 Document Quality Gatekeeper

- Model Architecture: YOLOv8n-cls (1,440,850 parameters, 640×640 RGB input)
- Production Format: ONNX FP32 (`results/week7_yolov8n_cls/weights/best.onnx`, 5.51 MB; INT8 variant 1.44 MB benchmarked in Task 5D)
- Operating Threshold: P(bad) >= 0.25 (selected via Task 5B validation sweep to cut False Accepts by 66.7%)
- Deployed Service: `@pm_quality_gate_bot` (`https://project-management-fpx0.onrender.com`)

## Quantitative performance

| Metric | PyTorch | ONNX FP32 (Production) | ONNX INT8 (Edge) | Capstone Target |
| --- | --- | --- | --- | --- |
| Test Accuracy | 1.0000 | 1.0000 | 0.9961 (-0.39%) | > 0.908 synthetic |
| Macro Precision / Recall / F1 | 1.0000 | 1.0000 | 0.9961 | - |
| CPU Latency (ms/image) | 18.862 ms | 11.004 ms | 19.905 ms | < 52.0 ms stage |
| Artifact Size | 2.83 MB | 5.51 MB | 1.44 MB | ~2.8 MB |

## Scope, limitations and synthetic-to-real gap

- Stage 2 document-type classification (SNILS, INN, PASSPORT) is intentionally deferred to Semester 2; accepted responses return `"document_type": "null"` per the Semester 1 schema contract.
- Trained on synthetic augmentations (blur, glare, skew, occlusion, exposure). Real smartphone photos in production are expected to experience a 2.8%-5.8% accuracy drop (projected 94.2%-97.2% real-world accuracy).
- Specular flash glare over blank white paper margins mimics clean document backgrounds; mitigated via the lowered 0.25 rejection threshold.