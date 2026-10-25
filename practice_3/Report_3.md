# Practice 3. Synthetic Russian Passport Dataset Generator

Name: Krapp Denis Vladimirovich

Group: AIb24-1

In this work, I developed a Python script that produces synthetic Russian passport main spread images. The program combines text rendering, photo fitting, signature overlays, rotated red document numbers, and watermarking, outputting rendered images alongside a metadata.jsonl catalog.

## Coordinates table
| Item | X | Y |
| --- | --- | --- |
| police_department | 0.210 | 0.110 |
| oblast | 0.080 | 0.145 |
| given_date | 0.190 | 0.225 |
| dep_number | 0.560 | 0.225 |
| surname | 0.430 | 0.560 |
| name | 0.400 | 0.635 |
| patronymic | 0.430 | 0.675 |
| gender | 0.390 | 0.715 |
| birth_date | 0.580 | 0.715 |
| city | 0.450 | 0.755 |
| district | 0.400 | 0.790 |
| region | 0.400 | 0.827 |
| photo window (top-left) | 0.045 | 0.612 |
| photo window (bottom-right) | 0.325 | 0.846 |
| rotated number position 1 | 0.950 | 0.240 |
| rotated number position 2 | 0.950 | 0.650 |
| sign position 1 | 0.660 | 0.345 |
| sign position 2 | 0.150 | 0.310 |

## 1. Which template was used and where it came from.
A layout is a 2-page spread of a standard Russian internal passport (Pages 2 and 3: Issuing Authority, Personal Info, Photo, and Signatures).

The clean, unpopulated background form was downloaded from Pinterest (passport_template.jpg) and stored in the templates/ directory. It contains the official graphic background, lines, and field labels, with all data-entry zones and photo windows left blank.

## 2. How coordinates were measured
Instead of fixed pixel offsets, all positions in CONFIG are stored as normalized floats (x / width, y / height) between 0.0 and 1.0. When an image is opened, coordinates are multiplied by the template dimensions: px = int(w * CONFIG[key][0]), py = int(h * CONFIG[key][1]). This allows the script to work on templates of any resolution without breaking field alignment.

Regions like photo_box use four relative coordinates (x1, y1, x2, y2) to define placement and size for automated scaling and cropping.

## 3. Which fields are generated and how
Faker generates gender-coherent Cyrillic full names (surname, name, patronymic) and formatted gender strings (МУЖ. / ЖЕН.).

Standard Faker data produces mismatched administrative data, so preconfigured regional packs ensure consistency between:
- Issuing department name (police_department) and sub-unit (oblast)
- Department code (dep_number) matching the region prefix
- Place of birth (city, district, region)
- Series prefix matching the region code
- Birth dates and issue dates respect legal age milestones, and random 6-digit passport numbers are generated.

Red passport numbers (full_number) on the right margins are rendered into temporary RGBA canvases, rotated 270°, and pasted onto the template.

## 4. Which seed was used
I used the number 6 as my student ID according to the table (maybe I misunderstood it).

## 5. What problems appeared
- Long authority names (police_department) and sub-departments collided with pre-printed baseline rules or adjacent fields. Relative coordinates and font sizes had to be calibrated to prevent line cross-over.
- Input portrait photos vary in size and ratio. The fit_portrait() function was implemented to calculate aspect ratios, crop from the center, and resample using Lanczos interpolation so faces are not stretched or distorted. If no photos exist, it renders a vector placeholder.
- Signatures pasted without proper alpha masking caused gray rectangular bounding boxes. Loading them as RGBA and using the image itself as its paste mask (img.paste(sign_img, (x, y), sign_img)) resolved the issue.

## 6. Ethics statement
- All personal details, registration codes, and numbers are synthetically generated or pseudorandomized.
- The pipeline uses AI-generated/synthetic faces or procedural silhouette fallbacks rather than photos of real living individuals.
- very generated image includes a prominent, indelible high-contrast warning bar across the bottom: "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT", preventing any misuse as counterfeit identification.

## 7. What should be improved next
- Add random blur, sensor noise, paper textures, and lighting glare/shadows across the laminated spread.
- Apply random 3D rotations, perspective shifts, and page folds to simulate real-world mobile capture.
- Add generation of the standard two-line OCR-B machine-readable zone at the bottom of Page 3.
- Automatically export bounding box coordinates for each field into .txt label files for end-to-end detection training.
