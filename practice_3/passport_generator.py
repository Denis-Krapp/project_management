import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

CONFIG = {
    "police_department": (0.210, 0.110),
    "oblast": (0.080, 0.145),
    "given_date": (0.190, 0.225),
    "dep_number": (0.560, 0.225),
    "surname": (0.430, 0.560),
    "name": (0.400, 0.635),
    "patronymic": (0.430, 0.675),
    "gender": (0.390, 0.715),
    "birth_date": (0.580, 0.715),
    "city": (0.450, 0.755),
    "district": (0.400, 0.790),
    "region": (0.400, 0.827),
    "photo_box": (0.045, 0.612, 0.325, 0.846),
    "rotated_num_1": (0.950, 0.240),
    "rotated_num_2": (0.950, 0.650),
    "sign_1": (0.660, 0.345),
    "sign_2": (0.150, 0.310),
}

REGION_PACKS = [
    {
        "region_code": "77",
        "department": "ОТДЕЛОМ УФМС РОССИИ ПО ГОР. МОСКВЕ",
        "sub_department": "ПО РАЙОНУ ХАМОВНИКИ",
        "dep_code": "770-001",
        "region": "ГОР. МОСКВА",
        "district": "ЦЕНТРАЛЬНЫЙ АО",
        "cities": ["МОСКВА"],
    },
    {
        "region_code": "78",
        "department": "ОТДЕЛОМ УФМС РОССИИ ПО САНКТ-ПЕТЕРБУРГУ",
        "sub_department": "И ЛЕНИНГРАДСКОЙ ОБЛАСТИ В ЦЕНТРАЛЬНОМ Р-НЕ",
        "dep_code": "780-002",
        "region": "САНКТ-ПЕТЕРБУРГ",
        "district": "ЦЕНТРАЛЬНЫЙ Р-Н",
        "cities": ["САНКТ-ПЕТЕРБУРГ"],
    },
    {
        "region_code": "54",
        "department": "ГУ МВД РОССИИ ПО НОВОСИБИРСКОЙ ОБЛАСТИ",
        "sub_department": "В ЛЕНИНСКОМ РАЙОНЕ Г. НОВОСИБИРСКА",
        "dep_code": "540-011",
        "region": "НОВОСИБИРСКАЯ ОБЛ.",
        "district": "ЛЕНИНСКИЙ Р-Н",
        "cities": ["НОВОСИБИРСК", "БЕРДСК", "ИСКИТИМ"],
    },
    {
        "region_code": "66",
        "department": "УМВД РОССИИ ПО СВЕРДЛОВСКОЙ ОБЛАСТИ",
        "sub_department": "В КИРОВСКОМ РАЙОНЕ Г. ЕКАТЕРИНБУРГА",
        "dep_code": "660-008",
        "region": "СВЕРДЛОВСКАЯ ОБЛ.",
        "district": "КИРОВСКИЙ Р-Н",
        "cities": ["ЕКАТЕРИНБУРГ", "НИЖНИЙ ТАГИЛ", "ПЕРВОУРАЛЬСК"],
    },
]

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


def load_font(size, bold: bool = False):
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_person_data(fake: Faker, rng: random.Random):
    pack = rng.choice(REGION_PACKS)
    gender = rng.choice(["male", "female"])

    if gender == "male":
        surname = fake.last_name_male().upper()
        name = fake.first_name_male().upper()
        patronymic = fake.middle_name_male().upper()
        gender_str = "МУЖ."
    else:
        surname = fake.last_name_female().upper()
        name = fake.first_name_female().upper()
        patronymic = fake.middle_name_female().upper()
        gender_str = "ЖЕН."

    b_date = date(1960, 1, 1) + timedelta(days=rng.randint(0, 16000))
    issue_year_offset = rng.choice([14, 20, 25, 45])
    issue_date = b_date + timedelta(days=int(issue_year_offset * 365.25) + rng.randint(10, 300))
    if issue_date > date.today():
        issue_date = date.today() - timedelta(days=rng.randint(30, 700))

    series_year = str(issue_date.year)[2:]
    passport_series = f"{pack['region_code']} {series_year}"
    passport_number = f"{rng.randint(100000, 999999)}"
    full_number = f"{passport_series} {passport_number}"

    city = rng.choice(pack["cities"])

    return {
        "police_department": pack["department"],
        "oblast": pack["sub_department"],
        "given_date": issue_date.strftime("%d.%m.%Y"),
        "dep_number": pack["dep_code"],
        "surname": surname,
        "name": name,
        "patronymic": patronymic,
        "gender": gender_str,
        "birth_date": b_date.strftime("%d.%m.%Y"),
        "city": f"Г. {city}",
        "district": pack["district"],
        "region": pack["region"],
        "series": passport_series,
        "number": passport_number,
        "full_number": full_number,
    }


def fit_portrait(photo_path, target_size):
    target_w, target_h = target_size
    target_aspect = target_w / target_h

    if photo_path and photo_path.exists():
        photo = Image.open(photo_path).convert("RGB")
    else:
        photo = Image.new("RGB", (target_w, target_h), (210, 215, 222))
        d = ImageDraw.Draw(photo)
        d.ellipse([target_w * 0.25, target_h * 0.15, target_w * 0.75, target_h * 0.55], fill=(130, 140, 155))
        d.chord([target_w * 0.10, target_h * 0.50, target_w * 0.90, target_h * 1.10], start=180, end=0, fill=(130, 140, 155))
        return photo

    img_w, img_h = photo.size
    img_aspect = img_w / img_h

    if img_aspect > target_aspect:
        new_w = int(img_h * target_aspect)
        left = (img_w - new_w) // 2
        photo = photo.crop((left, 0, left + new_w, img_h))
    else:
        new_h = int(img_w / target_aspect)
        top = (img_h - new_h) // 2
        photo = photo.crop((0, top, img_w, top + new_h))

    return photo.resize((target_w, target_h), Image.Resampling.LANCZOS)


def add_rotated_text(base_img, text, pos_xy, font, fill=(190, 25, 25)):
    temp_draw = ImageDraw.Draw(base_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    txt_img = Image.new("RGBA", (text_w + 10, text_h + 10), (255, 255, 255, 0))
    d = ImageDraw.Draw(txt_img)
    d.text((0, 0), text, font=font, fill=fill)

    rotated = txt_img.rotate(270, expand=True)
    base_img.paste(rotated, pos_xy, rotated)


def generate_dataset(n, seed, template_path, output_dir, portraits_dir, signs_dir):
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    Faker.seed(seed)
    fake = Faker("ru_RU")
    fake.seed_instance(seed)
    rng = random.Random(seed)

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    portrait_files = sorted(list(portraits_dir.glob("*.jpg")) + list(portraits_dir.glob("*.png"))) if portraits_dir.exists() else []
    sign_files = sorted(list(signs_dir.glob("*.png"))) if signs_dir.exists() else []

    with metadata_path.open("w", encoding="utf-8") as meta_file:
        for i in range(1, n + 1):
            fields = generate_person_data(fake, rng)
            img = Image.open(template_path).convert("RGB")
            w, h = img.size
            draw = ImageDraw.Draw(img)

            font_main = load_font(int(h * 0.016), bold=True)
            font_fio = load_font(int(h * 0.018), bold=True)
            font_rotated = load_font(int(h * 0.021), bold=True)
            text_color = (25, 25, 25)

            for key in ["police_department", "oblast", "given_date", "dep_number",
                        "gender", "birth_date", "city", "district", "region"]:
                px = int(w * CONFIG[key][0])
                py = int(h * CONFIG[key][1])
                draw.text((px, py), fields[key], font=font_main, fill=text_color)

            for key in ["surname", "name", "patronymic"]:
                px = int(w * CONFIG[key][0])
                py = int(h * CONFIG[key][1])
                draw.text((px, py), fields[key], font=font_fio, fill=text_color)

            bx1 = int(w * CONFIG["photo_box"][0])
            by1 = int(h * CONFIG["photo_box"][1])
            bx2 = int(w * CONFIG["photo_box"][2])
            by2 = int(h * CONFIG["photo_box"][3])
            target_pw, target_ph = (bx2 - bx1, by2 - by1)

            chosen_photo = portrait_files[(i - 1) % len(portrait_files)] if portrait_files else None
            fitted_portrait = fit_portrait(chosen_photo, (target_pw, target_ph))
            img.paste(fitted_portrait, (bx1, by1))

            if sign_files:
                sign_img_1 = Image.open(rng.choice(sign_files)).convert("RGBA")
                sign_img_1.thumbnail((int(w * 0.18), int(h * 0.05)), Image.Resampling.LANCZOS)
                s1_x = int(w * CONFIG["sign_1"][0])
                s1_y = int(h * CONFIG["sign_1"][1])
                img.paste(sign_img_1, (s1_x, s1_y), sign_img_1)

                sign_img_2 = Image.open(rng.choice(sign_files)).convert("RGBA")
                sign_img_2.thumbnail((int(w * 0.15), int(h * 0.05)), Image.Resampling.LANCZOS)
                s2_x = int(w * CONFIG["sign_2"][0])
                s2_y = int(h * CONFIG["sign_2"][1])
                img.paste(sign_img_2, (s2_x, s2_y), sign_img_2)

            num_rot_1 = (int(w * CONFIG["rotated_num_1"][0]), int(h * CONFIG["rotated_num_1"][1]))
            num_rot_2 = (int(w * CONFIG["rotated_num_2"][0]), int(h * CONFIG["rotated_num_2"][1]))
            add_rotated_text(img, fields["full_number"], num_rot_1, font=font_rotated, fill=(185, 28, 28))
            add_rotated_text(img, fields["full_number"], num_rot_2, font=font_rotated, fill=(185, 28, 28))

            bar_h = int(h * 0.035)
            draw.rectangle([0, h - bar_h, w, h], fill=(245, 245, 245))
            draw.text(
                (int(w * 0.04), h - bar_h + int(bar_h * 0.2)),
                "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT",
                font=load_font(int(h * 0.015), bold=True),
                fill=(205, 30, 30),
            )

            out_img_path = images_dir / f"passport_{i:05d}.jpg"
            img.save(out_img_path, quality=95)

            record = {
                "image_id": f"passport_{i:05d}",
                "document_type": "PASSPORT_RF",
                "template_name": template_path.stem,
                "synthetic": True,
                "seed": seed,
                "fields": fields,
                "portrait_file": chosen_photo.name if chosen_photo else "procedural_fallback",
                "image_path": str(out_img_path),
            }
            meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Dataset generated: {n} images in {images_dir}")
    print(f"Metadata cataloged: {metadata_path}")


parser = argparse.ArgumentParser(description="Synthetic Russian Passport Generator.")
parser.add_argument("--n", type=int, default=5, help="Number of passports to generate")
parser.add_argument("--seed", type=int, default=20261023, help="Deterministic student ID seed")
parser.add_argument("--template", type=Path, default=Path("templates/passport_template.jpg"))
parser.add_argument("--output", type=Path, default=Path("generated_passports"))
parser.add_argument("--portraits", type=Path, default=Path("portraits"))
parser.add_argument("--signs", type=Path, default=Path("signs"))

args, _ = parser.parse_known_args()

generate_dataset(
    n=args.n,
    seed=args.seed,
    template_path=args.template,
    output_dir=args.output,
    portraits_dir=args.portraits,
    signs_dir=args.signs,
)