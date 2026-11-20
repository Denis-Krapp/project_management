# Final Task Retrospective

## What failed during integration
1. Ultralytics auto-install hook on corrupted files: Feeding a corrupted `.jpg` into `YOLO()` triggered Ultralytics' background `pip install pi-heif` check, stalling tests on network retries and returning `No module named 'pi_heif'` instead of `unsupported_file`. Setting `YOLO_AUTOINSTALL=False` before importing `YOLO` and wrapping `Image.open().verify()` in a broad `try/except` fixed both speed and error schema compliance.
2. Local network TLS blocks to Telegram: Local `python-telegram-bot` polling failed with `httpx.ConnectError` due to local network restrictions on `api.telegram.org`. Deploying directly to Render's cloud environment resolved connectivity immediately.
3. Render web service port binding and token logging: Render's free Web Service tier requires binding to `$PORT` and logs `httpx` request URLs by default. Adding a lightweight background HTTP health server and raising the `httpx` logger level to `WARNING` kept the service alive without exposing the bot token in logs.

## What I would change
- Removing `tensorflow` from the root `requirements.txt` for the deployment branch would cut Render build times and container RAM usage roughly in half since inference runs purely on `onnxruntime`.
- For higher production traffic, I would replace `getUpdates` polling with FastAPI webhooks and route borderline predictions (0.20 <= P(bad) <= 0.45) to an operator review panel to harvest real smartphone failure cases for quarterly retraining.