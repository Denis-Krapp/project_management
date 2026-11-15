import argparse
import json
import random
from datetime import date
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES_REGULAR = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
]

FONT_CANDIDATES_BOLD = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
]

BADGE_SIZE = (650, 950)

def load_font(size, bold=False):
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()

def draw_centered_text(draw, y, text, font, fill, canvas_width):
    bbox = draw.textbbox((0, 0), str(text), font=font)
    text_w = bbox[2] - bbox[0]
    x = (canvas_width - text_w) // 2
    draw.text((x, y), str(text), font=font, fill=fill)

def draw_qr_placeholder(draw, box, cell_size=8, seed=42):
    x0, y0, x1, y1 = box
    rng = random.Random(seed)
    draw.rectangle(box, fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    for corner_x, corner_y in [(x0 + 6, y0 + 6), (x1 - 38, y0 + 6), (x0 + 6, y1 - 38)]:
        draw.rectangle([corner_x, corner_y, corner_x + 32, corner_y + 32], fill=(20, 20, 20))
        draw.rectangle([corner_x + 6, corner_y + 6, corner_x + 26, corner_y + 26], fill=(255, 255, 255))
        draw.rectangle([corner_x + 10, corner_y + 10, corner_x + 22, corner_y + 22], fill=(20, 20, 20))
    cols = (x1 - x0 - 16) // cell_size
    rows = (y1 - y0 - 16) // cell_size
    for r in range(rows):
        for c in range(cols):
            cx = x0 + 8 + c * cell_size
            cy = y0 + 8 + r * cell_size
            if (c < 5 and r < 5) or (c > cols - 6 and r < 5) or (c < 5 and r > rows - 6):
                continue
            if rng.random() > 0.5:
                draw.rectangle([cx, cy, cx + cell_size - 1, cy + cell_size - 1], fill=(20, 20, 20))

def create_conference_template(template_path, size=BADGE_SIZE):
    template_path = Path(template_path)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.rounded_rectangle([w // 2 - 45, 18, w // 2 + 45, 32], radius=7, fill=(220, 224, 230))
    draw.rectangle([0, 50, w, 190], fill=(24, 32, 54))
    draw_centered_text(draw, 75, "GLOBAL TECH SUMMIT", load_font(34, bold=True), (255, 255, 255), w)
    draw_centered_text(draw, 125, "ANNUAL DEVELOPER CONFERENCE • 2026", load_font(18), (170, 190, 220), w)
    draw_centered_text(draw, 155, "SAN FRANCISCO, CA", load_font(15, bold=True), (0, 210, 180), w)

    avatar_box = [w // 2 - 90, 225, w // 2 + 90, 405]
    draw.rectangle(avatar_box, fill=(240, 243, 248), outline=(200, 208, 220), width=2)
    draw.ellipse([w // 2 - 35, 255, w // 2 + 35, 325], fill=(180, 190, 205))
    draw.chord([w // 2 - 65, 335, w // 2 + 65, 435], start=180, end=0, fill=(180, 190, 205))

    label_font = load_font(14, bold=True)
    draw.text((65, 625), "COMPANY / AFFILIATION", font=label_font, fill=(130, 138, 150))
    draw.text((65, 705), "PRIMARY TRACK", font=label_font, fill=(130, 138, 150))
    draw.text((65, 785), "BADGE ID", font=label_font, fill=(130, 138, 150))

    draw.line([(65, 680), (460, 680)], fill=(230, 235, 242), width=2)
    draw.line([(65, 760), (460, 760)], fill=(230, 235, 242), width=2)

    draw.rectangle([0, h - 45, w, h], fill=(245, 245, 245))
    draw.text((40, h - 32), "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT", font=load_font(16, bold=True), fill=(200, 30, 30))
    img.save(template_path, quality=95)

def generate_badge_data(fake, rng):
    tier_info = rng.choice([
        {"tier": "ALL-ACCESS", "bg_color": (0, 150, 136), "text_color": (255, 255, 255)},
        {"tier": "KEYNOTE SPEAKER", "bg_color": (220, 53, 69), "text_color": (255, 255, 255)},
        {"tier": "VIP GUEST", "bg_color": (245, 166, 35), "text_color": (20, 20, 20)},
        {"tier": "EXHIBITOR", "bg_color": (52, 73, 94), "text_color": (255, 255, 255)},
        {"tier": "MEDIA / PRESS", "bg_color": (142, 68, 173), "text_color": (255, 255, 255)},
    ])
    return {
        "attendee_name": fake.name().upper(),
        "organization": fake.company(),
        "badge_tier": tier_info["tier"],
        "tier_bg_color": tier_info["bg_color"],
        "tier_text_color": tier_info["text_color"],
        "track": rng.choice(["Machine Learning", "Cloud Systems", "Edge Robotics", "Cybersecurity"]),
        "badge_id": f"GTS26-{rng.randint(10000, 99999)}",
        "issue_date": date.today().strftime("%Y-%m-%d"),
        "qr_seed": rng.randint(1000, 999999),
    }

def render_badge(fields, template_path, output_path, positions=None):
    if positions is None:
        positions = {
            "attendee_name_y": 550,
            "tier_badge_y": 485,
            "organization": [65, 645],
            "track": [65, 725],
            "badge_id": [65, 805],
            "qr_box": [485, 630, 595, 740],
        }
    img = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, _ = BADGE_SIZE

    tier_y = positions.get("tier_badge_y", 485)
    tier_text = fields["badge_tier"]
    tier_font = load_font(20, bold=True)
    t_bbox = draw.textbbox((0, 0), tier_text, font=tier_font)
    pill_w = (t_bbox[2] - t_bbox[0]) + 36
    pill_x0 = (w - pill_w) // 2
    draw.rounded_rectangle([pill_x0, tier_y, pill_x0 + pill_w, tier_y + 36], radius=8, fill=tuple(fields["tier_bg_color"]))
    draw_centered_text(draw, tier_y + 6, tier_text, tier_font, tuple(fields["tier_text_color"]), w)

    name_y = positions.get("attendee_name_y", 550)
    draw_centered_text(draw, name_y, fields["attendee_name"], load_font(30, bold=True), (20, 28, 40), w)

    val_font = load_font(22, bold=False)
    draw.text(tuple(positions["organization"]), fields["organization"][:28], font=val_font, fill=(35, 45, 60))
    draw.text(tuple(positions["track"]), fields["track"], font=val_font, fill=(35, 45, 60))
    draw.text(tuple(positions["badge_id"]), fields["badge_id"], font=load_font(24, bold=True), fill=(0, 120, 160))
    draw_qr_placeholder(draw, positions["qr_box"], seed=fields["qr_seed"])
    img.save(output_path, quality=95)

def generate_dataset(n, seed, template_path, output_dir, positions=None):
    Faker.seed(seed)
    fake = Faker("en_US")
    fake.seed_instance(seed)
    rng = random.Random(seed)

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    metadata_path = output_dir / "metadata.jsonl"

    if not template_path.exists():
        create_conference_template(template_path)

    images_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        for i in range(1, n + 1):
            fields = generate_badge_data(fake, rng)
            img_path = images_dir / f"badge_{i:05d}.jpg"
            render_badge(fields, template_path, img_path, positions=positions)
            clean_fields = {k: v for k, v in fields.items() if not k.endswith("_color")}
            record = {
                "image_id": f"badge_{i:05d}",
                "document_type": "CONFERENCE_BADGE",
                "template_name": template_path.stem,
                "seed": seed,
                "fields": clean_fields,
                "image_path": str(img_path),
            }
            meta_file.write(json.dumps(record) + "\n")
    print(f"Complete: {n} images generated in {images_dir}")


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--seed", type=int, default=20261019)
parser.add_argument("--template", type=Path, default=Path("templates/card_template.jpg"))
parser.add_argument("--output", type=Path, default=Path("data/card"))
parser.add_argument("--positions", type=Path, default=None)
args, unknown = parser.parse_known_args()

pos = None
if args.positions and args.positions.exists():
    with args.positions.open("r") as f:
        pos = json.load(f)

generate_dataset(args.n, args.seed, args.template, args.output, positions=pos)