import argparse
import json
import random
from datetime import date, timedelta
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


def load_font(size, bold = False):
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_inn_data(fake, rng):
    gender = rng.choice(["male", "female"])
    if gender == "male":
        full_name = f"{fake.last_name_male()} {fake.first_name_male()} {fake.middle_name_male()}".upper()
        gender_str = "МУЖ."
    else:
        full_name = f"{fake.last_name_female()} {fake.first_name_female()} {fake.middle_name_female()}".upper()
        gender_str = "ЖЕН."

    b_date = date(1965, 1, 1) + timedelta(days=rng.randint(0, 13000))
    city = fake.city_name().upper()
    district = f"{city[:5]}СКИЙ Р-Н"

    inn_12 = f"{rng.randint(77, 78)}{rng.randint(1000000000, 9999999999)}"

    return {
        "full_name": full_name,
        "gender": gender_str,
        "birth_date": b_date.strftime("%d.%m.%Y"),
        "birth_place": f"Г. {city}, {district}",
        "inn_number": inn_12,
    }


def render_inn(fields, template_path, output_path, positions=None):
    img = Image.open(template_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    if positions is None:
        positions = {
            "full_name":     [int(w * 0.135), int(h * 0.310)],
            "gender":        [int(w * 0.170), int(h * 0.348)],
            "birth_date":    [int(w * 0.625), int(h * 0.348)],
            "birth_place":   [int(w * 0.260), int(h * 0.387)],
            "first_digit_x": int(w * 0.453),
            "digit_step":    int(w * 0.0275),
            "digit_y":       int(h * 0.500),
        }

    val_font = load_font(int(h * 0.016), bold=True)
    fio_font = load_font(int(h * 0.015), bold=True)
    digit_font = load_font(int(h * 0.017), bold=True)
    text_color = (20, 20, 20)

    draw.text(tuple(positions["full_name"]), fields["full_name"], font=fio_font, fill=text_color)
    draw.text(tuple(positions["gender"]), fields["gender"], font=val_font, fill=text_color)
    draw.text(tuple(positions["birth_date"]), fields["birth_date"], font=val_font, fill=text_color)
    draw.text(tuple(positions["birth_place"]), fields["birth_place"], font=val_font, fill=text_color)

    x_start = positions["first_digit_x"]
    step = positions["digit_step"]
    y_pos = positions["digit_y"]

    for i, digit in enumerate(fields["inn_number"]):
        draw.text((x_start + i * step, y_pos), digit, font=digit_font, fill=(0, 45, 120))

    bar_h = int(h * 0.035)
    draw.rectangle([0, h - bar_h, w, h], fill=(245, 245, 245))
    draw.text(
        (int(w * 0.05), h - bar_h + int(bar_h * 0.2)),
        "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT",
        font=load_font(int(h * 0.015), bold=True),
        fill=(205, 30, 30),
    )

    img.save(output_path, quality=95)


def generate_inn_dataset(n, seed, template_path, output_dir, positions=None):
    Faker.seed(seed)
    fake = Faker("ru_RU")
    fake.seed_instance(seed)
    rng = random.Random(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as meta_file:
        for i in range(1, n + 1):
            fields = generate_inn_data(fake, rng)
            img_path = output_dir / f"inn_{i:05d}.jpg"
            render_inn(fields, template_path, img_path, positions=positions)

            record = {
                "image_id": f"inn_{i:05d}",
                "document_type": "INN",
                "template_name": template_path.stem,
                "synthetic": True,
                "seed": seed,
                "fields": fields,
                "image_path": str(img_path),
            }
            meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {n} INN certificates in {output_dir}")


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=5)
parser.add_argument("--seed", type=int, default=20261023)
parser.add_argument("--template", type=Path, default=Path("templates/inn_template.jpg"))
parser.add_argument("--output", type=Path, default=Path("inn"))
parser.add_argument("--positions", type=Path, default=None)
args, _ = parser.parse_known_args()

pos = None
if args.positions and args.positions.exists():
    with args.positions.open("r", encoding="utf-8") as f:
        pos = json.load(f)

generate_inn_dataset(args.n, args.seed, args.template, args.output, positions=pos)