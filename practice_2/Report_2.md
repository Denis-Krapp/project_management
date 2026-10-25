# Practice 2. Synthetic Document Generator (SNILS & INN)

Name: Krapp Denis Vladimirovich

Group: AIb24-1

In this work I made a Python program that generates synthetic Russian identity documents: the SNILS insurance card and the INN tax registration certificate. The program saves the rendered document images and a JSONL metadata file for each document.

## Coordinate tables
Formula format: proportional ratios based on image width and height.
SNILS:
| Field | X | Y |
| -------- | -------- | -------- |
| SNILS number | 0.280 * W | 0.330 * H |
| Surname | 0.280 * W | 0.405 * H |
| Name | 0.280 * W | 0.450 * H |
| Patronymic | 0.280 * W | 0.495 * H |
| Birth date | 0.435 * W | 0.568 * H |
| City | 0.280 * W | 0.628 * H |
| Region | 0.280 * W | 0.672 * H |
| Gender | 0.250 * W | 0.818 * H |
| Registration date | 0.400 * W | 0.872 * H |

INN:
| Field | X | Y |
| -------- | -------- | -------- |
| Full name | 0.135 * W | 0.310 * H |
| Gender | 0.170 * W | 0.348 * H |
| Birth date | 0.625 * W | 0.348 * H |
| Birth place | 0.260 * W | 0.387 * H  |
| First INN digit (x start) | 0.453 * W | - |
| Digit step (distance between boxes) | 0.0275 * W | - |
| Digit y | - | 0.500 * H |

## 1. Which template was chosen
I implemented generators for two standard Russian government document layouts:
- SNILS layout: The green laminated pension insurance certificate.
- INN layout: The official certificate of individual tax registration.

Both templates include a bottom disclaimer bar: "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT"

## 2. Where the template came from
The blank templates, suggested by you, (snils_template.jpg and inn_template.jpg) were placed in the templates/ folder. They are blank background forms of the official document layouts with all personal data fields left empty for programmatic text rendering.

## 3. What fields were generated
All personal details were generated using the Faker library (ru_RU locale) and Python's random module:
For SNILS:
- snils_number: 11-digit formatted number (XXX-XXX-XXX YY)
- surname, name, patronymic: Russian Cyrillic name in uppercase
- gender: "МУЖСКОЙ" or "ЖЕНСКИЙ"
- birth_date: Formatted date (DD.MM.YYYY)
- city and region: Birth location (e.g., city and region)
- reg_date: Registration date (generated chronologically after the birth date)

For INN:
- full_name: Single-line full name in uppercase (LAST FIRST MIDDLE)
- gender: Short format ("МУЖ." or "ЖЕН.")
- birth_date: Formatted date (DD.MM.YYYY)
- birth_place: City and district (e.g., "Г. МОСКВА, МОСКОВСКИЙ Р-Н")
- inn_number: 12-digit taxpayer number rendered into discrete boxes

## 4. How coordinates were found
Instead of hardcoded absolute pixels, positions were calculated as proportions of the template width and height (int(w * ratio), int(h * ratio)):
- SNILS: Proportional coordinates place the number, 3-line name, dates, location, and gender lines at their respective form lines.
- INN: Coordinates place the personal data lines and iterate over the 12-digit INN using a start coordinate (first_digit_x) and a horizontal step (digit_step = int(w * 0.0275)) to align each number into its printed box.
- Both scripts accept an optional --positions JSON file to adjust coordinates without changing the code.

## 5. What seed was used
I used seed number 6 as my student ID according to the table (maybe I misunderstood it).

## 6. What problems appeared
- Default PIL bitmap fonts do not render Cyrillic characters properly. I added fallback lists targeting system TrueType fonts (arial.ttf, calibri.ttf, DejaVuSans.ttf).
- Fixed pixel spacing for the 12 digits caused drift across boxes on templates of different resolutions. Switching to proportional step sizes (int(w * 0.0275)) resolved the misalignment.
- Long city and district names exceeded the printable width on the forms.

## 8. What should be improved
- Add text wrapping or dynamic font scaling for long place-of-birth and patronymic strings.
- Add synthetic scan distortions (slight rotation, noise, perspective skew, and paper texture) to improve realism for OCR training.
- Bundle an open-source Cyrillic TTF font file directly in the repository so font loading does not depend on the host operating system.
