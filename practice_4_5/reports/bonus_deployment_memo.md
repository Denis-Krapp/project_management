# Task 5D: Deployment Export & Edge Benchmark Memo

- Base model: YOLOv8n-cls (results/week7_yolov8n_cls/weights/best.pt, 1,440,850 parameters, input 1x3x640x640)
- Exported srtifacts: results/week7_yolov8n_cls/weights/best.onnx (FP32) and results/week7_yolov8n_cls/weights/best_int8.onnx (Dynamic QUInt8)
- Benchmark protocol: CPU (AMD Ryzen 5 5600, 4 intra-op threads), 20 warm-up + 100 timed single-image inferences, evaluated on full held-out test split (N = 258)


### 1. Export and quantization benchmark table

| Format | Artifact File | Size (MB) | CPU Latency (ms/img) | Test Accuracy | Accuracy Drop | Capstone Target Comparison (52 ms Stage / 2.8 MB Size) |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP16/32 (.pt) | best.pt | 2.83 MB | 18.862 ms | 1.0000 | - | Matches 2.8 MB size target; 2.76x faster than 52 ms stage budget |
| ONNX FP32 (.onnx) | best.onnx | 5.51 MB | 11.004 ms | 1.0000 | 0.0000 (0.00%) | 4.73x faster than 52 ms stage budget; zero accuracy drop |
| ONNX INT8 (_int8.onnx) | best_int8.onnx | 1.44 MB | 19.905 ms | 0.9961 | 0.0039 (0.39%) | 48.6% smaller than 2.8 MB target; drop well inside <= 1.0% limit |


### 2. Technical analysis of export and quantization trade-offs

* Why ONNX FP32 latency dropped by 41.7% (18.862 ms $\rightarrow$ 11.004 ms):  
  onnxslim fused consecutive Conv + BatchNorm + SiLU blocks and eliminated Python/PyTorch dispatcher overhead, allowing ONNX Runtime's C++ CPUExecutionProvider to execute single-image inference in 11.004 ms (~91 FPS on CPU) with 0.0000 accuracy change.
* Why ONNX FP32 size increased (2.83 MB $\rightarrow$ 5.51 MB):  
  Ultralytics strips optimizer state and stores .pt inference weights in half-precision (FP16, 2 bytes/param $\approx 2.83\text{ MB}$), whereas standard opset=12 ONNX export serializes weights in full 32-bit floating point (FP32, 4 bytes/param $\approx 5.51\text{ MB}$).
* Why ONNX INT8 shrinks binary size to 1.44 MB but runs at 19.905 ms on desktop CPU:  
  Dynamic QUInt8 quantization compresses weights from 32-bit floats to 8-bit integers, cutting disk footprint by 73.9% (5.51 MB $\rightarrow$ 1.44 MB). However, on x86 desktop CPUs without static activation calibration (QDQ operators), dynamic quantization inserts runtime float-to-int8 activation conversions around convolutional layers, adding ~8.9 ms of dequantization overhead compared to ONNX FP32.
* Comparison to capstone targets (52 ms stage / 2.1 ms GPU / 2.8 MB):  
  All three formats execute well inside the 52 ms end-to-end quality stage budget (11.0-19.9 ms on CPU, leaving 32-41 ms for image decoding and network I/O). The 2.1 ms capstone reference reflects dedicated GPU inference, whereas our 11.004 ms benchmark achieves real-time throughput strictly on CPU.


### 3. Deployment recommendation (server vs. edge)

1. Production CPU server (Telegram bot backend) - deploy ONNX FP32 (best.onnx):  
   At 11.004 ms/image and 1.0000 test accuracy (0.0000 drop), ONNX FP32 delivers the lowest CPU latency (~90.9 images/sec per worker) and zero quantization noise, while its 5.51 MB footprint is negligible in server RAM.
2. Mobile / edge client (on-device pre-upload gate) - deploy ONNX INT8 (best_int8.onnx):  
   At 1.44 MB and 0.9961 test accuracy (only 1 borderline misclassification out of 258 images, 0.39% drop), ONNX INT8 minimizes mobile app bundle size and cellular download bandwidth while still clearing the < 52 ms interactive camera frame budget (19.905 ms).


### 4. Quantization risks and explanation checkpoints

* Single deciding metric for edge feasibility:  
  Per-image inference latency on the target edge hardware relative to the stage latency budget (e.g., < 52 ms for real-time camera quality feedback), subject to meeting minimum accuracy and RAM/storage ceilings.
* Why INT8 quantization risks accuracy:  
  Compressing continuous 32-bit floating-point weights and activations into 256 discrete 8-bit integer levels ([-128, 127]) introduces rounding quantization error. For subtle, low-contrast border cases-such as faint specular glare over white paper margins-small activation shifts can push logits across the decision boundary (0.9961 vs 1.0000, 0.39% drop).