import time
import yaml
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


class DocumentPipeline:
    def __init__(self, config_path = "config.yaml"):
        self.base_dir = Path(__file__).parent
        with open(self.base_dir / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.threshold = float(self.config["thresholds"]["quality_reject_p_bad"])
        self.q_version = self.config["versions"]["quality_gate"]
        self.d_version = self.config["versions"]["document_type"]

        q_model_path = self.base_dir / self.config["models"]["quality_gate"]
        if not q_model_path.exists():
            raise FileNotFoundError(f"Model missing at {q_model_path}")
        
        self.quality_model = YOLO(str(q_model_path), task="classify")

    def run(self, image_path):
        start_t = time.perf_counter()
        img_path_obj = Path(image_path)
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        result = {
            "file": img_path_obj.name,
            "status": "error",
            "quality": "null",
            "quality_confidence": 0.0,
            "document_type": "null",
            "document_type_confidence": 0.0,
            "reason": "null",
            "model_versions": {"quality": self.q_version, "doctype": self.d_version},
            "latency_ms": 0,
            "timestamp": timestamp
        }

        try:
            if not img_path_obj.exists():
                raise FileNotFoundError("File does not exist on disk.")
            if img_path_obj.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                result["reason"] = "unsupported_file"
                raise ValueError("Not a JPG/PNG file.")

            try:
                img = Image.open(img_path_obj)
                img.verify()
            except (UnidentifiedImageError, SyntaxError):
                result["reason"] = "unsupported_file"
                raise ValueError("Corrupted image file.")

            inference_start = time.perf_counter()
            pred = self.quality_model(str(img_path_obj), imgsz=640, verbose=False)[0]
            
            p_bad = float(pred.probs.data[0].item())
            p_good = float(pred.probs.data[1].item())
            
            if p_bad >= self.threshold:
                result["status"] = "rejected"
                result["quality"] = "bad"
                result["quality_confidence"] = round(p_bad, 4)
                result["reason"] = "poor_quality"
            else:
                result["status"] = "accepted"
                result["quality"] = "good"
                result["quality_confidence"] = round(p_good, 4)
                result["document_type"] = "null" 
                result["document_type_confidence"] = 0.0

        except Exception as e:
            result["status"] = "error"
            if result["reason"] == "null":
                result["reason"] = str(e)

        finally:
            result["latency_ms"] = int((time.perf_counter() - start_t) * 1000)
            return result