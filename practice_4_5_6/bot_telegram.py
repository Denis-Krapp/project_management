import os
import logging
import tempfile
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from pipeline import DocumentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipe = DocumentPipeline("config.yaml")

def start_health_server():
    port = os.environ.get("PORT")
    if not port:
        return

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        def log_message(self, format, *args):
            pass

    server = HTTPServer(("0.0.0.0", int(port)), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

async def start(update, context):
    await update.message.reply_text(
        "📄 Document Quality Gate\n"
        "Send a PHOTO of a document (SNILS, INN, passport).\n"
        "I will check the quality before sending it to OCR."
    )

async def on_photo(update, context):
    msg = update.message
    photo = msg.photo[-1]
    tg_file = await photo.get_file()
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        await tg_file.download_to_drive(tmp.name)
        path = tmp.name
        
    try:
        result = pipe.run(path)
    finally:
        pathlib.Path(path).unlink(missing_ok=True)

    if result["status"] == "accepted":
        text = (f"✅ ACCEPTED\n"
                f"Quality: {result['quality']} ({result['quality_confidence']:.1%})\n"
                f"Document: {result['document_type']} (Next Semester)\n"
                f"Latency: {result['latency_ms']} ms")
    elif result["status"] == "rejected":
        text = (f"❌ REJECTED: {result['reason']}\n"
                f"Confidence: {result['quality_confidence']:.1%}\n"
                "Please retake the photo: ensure no blur, glare, or cutoff edges.")
    else:
        text = f"⚠️ Could not process this file. Reason: {result['reason']}"
        
    await msg.reply_text(text)

async def on_document(update, context):
    await update.message.reply_text(
        "You sent a FILE. Please send it as a PHOTO "
        "(Telegram menu → Photo) so I can process it safely."
    )


start_health_server()
token = os.environ.get("TELEGRAM_BOT_TOKEN")
if not token:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is missing!")
    
app = Application.builder().token(token).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.PHOTO, on_photo))
app.add_handler(MessageHandler(filters.Document.ALL, on_document))

print("Bot is polling…")
app.run_polling(allowed_updates=Update.ALL_TYPES)
