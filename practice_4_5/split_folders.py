import json
import shutil
from pathlib import Path

SOURCE_DIR = Path("data/demo_quality_dataset")
OUTPUT_DIR = Path("data/quality_split")
METADATA_PATH = SOURCE_DIR / "metadata.jsonl"


counts = {"train": {"good": 0, "bad": 0},
            "validation": {"good": 0, "bad": 0},
            "test": {"good": 0, "bad": 0}}

with METADATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line.strip())
        
        split = rec.get("split")
        quality = rec.get("quality_label")
        rel_path = rec.get("image_path")

        if split not in counts or quality not in ["good", "bad"]:
            continue

        src_file = SOURCE_DIR / rel_path
        if not src_file.exists():
            continue

        dest_dir = OUTPUT_DIR / split / quality
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / src_file.name

        if not dest_file.exists():
            shutil.copy2(src_file, dest_file)

        counts[split][quality] += 1

print("--- Quality Split Folder Generation Complete ---")
for s, q_counts in counts.items():
    print(f"Split [{s:>10}]: good ={q_counts['good']:4d} | bad ={q_counts['bad']:4d} | total = {q_counts['good'] + q_counts['bad']:4d}")
