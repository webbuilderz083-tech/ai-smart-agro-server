"""
plant_check.py
=====================================================
A lightweight, free, local heuristic that checks whether an uploaded
image plausibly shows GREEN plant material before sending it to the
(rate-limited) Hugging Face model.

HONEST LIMITATION (please read, and mention in your project report):
This is a simple color heuristic, not an AI model, and out-of-distribution
detection is a genuinely hard, unsolved problem in machine learning even
for real research systems. An earlier version of this check also tried to
accept brown/yellow "dried leaf" tones, but that made it too easy to
mistake beige/tan fabric, wood, or skin tones for a diseased leaf — those
color ranges genuinely overlap. This version deliberately only accepts
green-dominant images, which is much more reliable at rejecting clearly
wrong uploads (faces, screenshots, random objects), at the cost of also
rejecting some heavily-diseased leaf photos where very little green
remains. That trade-off is intentional: it's better to occasionally ask
a farmer to retake a very brown leaf photo than to confidently return a
fake diagnosis for a non-plant photo.
"""
from PIL import Image
import io

# Minimum fraction of green-dominant pixels required to pass the check.
PLANT_PIXEL_THRESHOLD = 0.10


def is_probably_plant_image(image_bytes):
    """
    Returns (is_plant: bool, plant_pixel_ratio: float)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return True, 1.0  # don't block on our own parsing failure

    img = img.resize((80, 80))
    pixels = list(img.getdata())

    green_count = 0
    for r, g, b in pixels:
        # Skip near-white/near-gray/near-black pixels (walls, paper,
        # overexposed backgrounds, shadows) — real leaf colors have some
        # saturation and a clear green tilt.
        if max(r, g, b) - min(r, g, b) < 15:
            continue
        if g > r + 8 and g > b + 8 and g > 35:
            green_count += 1

    ratio = green_count / len(pixels)
    return ratio >= PLANT_PIXEL_THRESHOLD, round(ratio, 3)
