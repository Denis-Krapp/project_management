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


def generate_snils_data(fake, rng):
    gender = rng.choice(["male", "female"])
    if gender == "male":
        surname = fake.last_name_male().upper()
        name = fake.first_name_male().upper()
        patronymic = fake.middle_name_male().upper()
        gender_str = "МУЖСКОЙ"
    else:
        surname = fake.last_name_female().upper()
        name = fake.first_name_female().upper()
        patronymic = fake.middle_name_female().upper()
        gender_str = "ЖЕНСКИЙ"

    p1 = f"{rng.randint(100, 999)}"
    p2 = f"{rng.randint(100, 999)}"
    p3 = f"{rng.randint(100, 999)}"
    chk = f"{rng.randint(10, 99):02d}"
    snils_number = f"{p1}-{p2}-{p3} {chk}"

    b_date = date(1965, 1, 1) + timedelta(days=rng.randint(0, 13000))
    reg_date = b_date + timedelta(days=rng.randint(6000, 12000))

    city = fake.city_name().upper()
    region = rng.choice(["МОСКОВСКАЯ ОБЛ.", "ЛЕНИНГРАДСКАЯ ОБЛ.", "СВЕРДЛОВСКАЯ ОБЛ."])

    return {
        "snils_number": snils_number,
        "surname": surname,
        "name": name,
        "patronymic": patronymic,
        "birth_date": b_date.strftime("%d.%m.%Y"),
        "city": f"Г. {city}",
        "region": region,
        "gender": gender_str,
        "reg_date": reg_date.strftime("%d.%m.%Y"),
    }


def render_snils(fields, template_path, output_path, positions=None):
    img = Image.open(template_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    if positions is None:
        positions = {
            "snils_number": [int(w * 0.28), int(h * 0.330)],
            "surname":      [int(w * 0.28), int(h * 0.405)],
            "name":         [int(w * 0.28), int(h * 0.450)],
            "patronymic":   [int(w * 0.28), int(h * 0.495)],
            "birth_date":   [int(w * 0.435), int(h * 0.568)],
            "birth_place":  [int(w * 0.28), int(h * 0.628)],
            "birth_region": [int(w * 0.28), int(h * 0.672)],
            "gender":       [int(w * 0.25), int(h * 0.818)],
            "reg_date":     [int(w * 0.40), int(h * 0.872)],
        }

    font_snils = load_font(int(h * 0.038), bold=True)
    font_main = load_font(int(h * 0.029), bold=True)
    font_small = load_font(int(h * 0.024), bold=True)

    text_color = (25, 25, 25)

    draw.text(tuple(positions["snils_number"]), fields["snils_number"], font=font_snils, fill=(15, 60, 35))
    draw.text(tuple(positions["surname"]), fields["surname"], font=font_main, fill=text_color)
    draw.text(tuple(positions["name"]), fields["name"], font=font_main, fill=text_color)
    draw.text(tuple(positions["patronymic"]), fields["patronymic"], font=font_main, fill=text_color)
    draw.text(tuple(positions["birth_date"]), fields["birth_date"], font=font_main, fill=text_color)
    draw.text(tuple(positions["birth_place"]), fields["city"], font=font_small, fill=text_color)
    draw.text(tuple(positions["birth_region"]), fields["region"], font=font_small, fill=text_color)
    draw.text(tuple(positions["gender"]), fields["gender"], font=font_main, fill=text_color)
    draw.text(tuple(positions["reg_date"]), fields["reg_date"], font=font_main, fill=text_color)

    bar_height = int(h * 0.055)
    draw.rectangle([0, h - bar_height, w, h], fill=(245, 245, 245))
    draw.text(
        (int(w * 0.04), h - bar_height + int(bar_height * 0.2)),
        "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT",
        font=load_font(int(h * 0.024), bold=True),
        fill=(205, 30, 30),
    )

    img.save(output_path, quality=95)


def generate_snils_dataset(n, seed, template_path, output_dir, positions=None):
    Faker.seed(seed)
    fake = Faker("ru_RU")
    fake.seed_instance(seed)
    rng = random.Random(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as meta_file:
        for i in range(1, n + 1):
            fields = generate_snils_data(fake, rng)
            img_path = output_dir / f"snils_{i:05d}.jpg"
            render_snils(fields, template_path, img_path, positions=positions)

            record = {
                "image_id": f"snils_{i:05d}",
                "document_type": "SNILS",
                "template_name": template_path.stem,
                "synthetic": True,
                "seed": seed,
                "fields": fields,
                "image_path": str(img_path),
            }
            meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {n} SNILS cards in {output_dir}")


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=5)
parser.add_argument("--seed", type=int, default=20261023)
parser.add_argument("--template", type=Path, default=Path("templates/snils_template.jpg"))
parser.add_argument("--output", type=Path, default=Path("snils"))
parser.add_argument("--positions", type=Path, default=None)
args, _ = parser.parse_known_args()

pos = None
if args.positions and args.positions.exists():
    with args.positions.open("r", encoding="utf-8") as f:
        pos = json.load(f)

generate_snils_dataset(args.n, args.seed, args.template, args.output, positions=pos)