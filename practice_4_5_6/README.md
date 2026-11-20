# Document Quality Gate — Telegram Bot Service

Stage 1 Document Quality Gatekeeper built with `YOLOv8n-cls` (ONNX FP32) and `python-telegram-bot`, deployed live on Render.

- Public Bot Username: `@pm_quality_gate_bot`
- Render Health/Wake URL: `https://project-management-fpx0.onrender.com`


## Architecture

```text
[Telegram User Photo]
        │
        ▼
[bot_telegram.py] ── (File attachment) ──► Prompt user to send as Photo
        │ (Photo downloaded to temp file, cleaned in finally block)
        ▼
[pipeline.py: DocumentPipeline] (Loaded once at startup via config.yaml)
        ├──► File & PIL Image Validation ── (Corrupt / .txt) ──► status: "error"
        └──► YOLOv8n-cls ONNX (640x640 CPU, ~11.0 ms)
                ├──► P(bad) >= 0.25 ──► status: "rejected" (reason: "poor_quality")
                └──► P(bad) <  0.25 ──► status: "accepted" (document_type: "null")