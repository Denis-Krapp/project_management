# Synthetic Club Membership Card Generator

Name: Krapp Denis Vladimirovich

Group: AIb24-1

In this practice work I made a Python program that creates fake conference badges, saves the rendered badge images and a JSONL file with the metadata of each badge.

## 1. Which template was chosen
I chose a conference badge for a fictional event called "Global Tech Summit 2026". The badge has a lanyard slot cut-out, an event header, an avatar silhouette placeholder, a color-coded badge tier pill, field lines for details, a QR code box, and a red synthetic disclaimer banner at the bottom.

## 2. Where the template came from
I generated the template directly in Python using the Pillow library. No external image or template was downloaded from the internet.

## 3. License or permission
No license is required because all visual elements and text are programmatically generated from scratch. The event is fictional. Each badge contains a visible warning label: "SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT".

## 4. What fields were generated
The program generates the badge data using the Faker library and Python's random module:
- Attendee name
- Organization / Company
- Badge tier (ALL-ACCESS, KEYNOTE SPEAKER, VIP GUEST, EXHIBITOR, or MEDIA / PRESS)
- Primary track (Machine Learning, Cloud Systems, Edge Robotics, or Cybersecurity)
- Badge ID (format GTS26-XXXXX)
- Procedural QR code (rendered using a random seed)

The issue_date is also generated and stored in the metadata JSONL, but it is not printed on the badge surface.

## 5. How coordinates were found
For centered fields like the attendee name and the tier pill, I used draw.textbbox() to calculate the text width and dynamically center them. The remaining fields (organization, track, badge_id, and qr_box) use fixed coordinates aligned to the template lines.

## 6. What seed was used
I used 20260915 as the default seed initialy. Then I used the seed number 6, as my student ID according to the table (maybe I misunderstood it).

## 7. What problems appeared
- Long company names extended beyond the divider lines. I truncated the text to 28 characters to fit the layout.
- Badge tiers have different word lengths, which made fixed-width pills look uneven. I calculated the pill width dynamically using textbbox.
- Different operating systems store fonts in different locations. I created candidate font lists for Windows and Linux with a fallback to load_default().

## 8. What should be improved
- Use the qrcode library to generate real, scannable QR codes instead of pseudo-random blocks.
- Insert synthetic face photos instead of the generic silhouette avatar.
- Implement automatic font scaling or text wrapping instead of hard-cutting long company names.
- Bundle an open-source TTF font (e.g., DejaVuSans or Roboto) directly in the project so the layout renders identically on all machines.
