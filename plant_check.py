"""
plant_check.py
=====================================================
A lightweight, free, local heuristic that checks whether an uploaded
image plausibly shows a plant/leaf BEFORE sending it to the (rate-limited)
Hugging Face model. This is not itself an AI model — it's a simple color
analysis using Pillow — but it is a real, working, and honest check that
catches obviously wrong uploads (faces, random objects, screenshots),
so the site can show "This doesn't look like a plant photo" instead of
a confusing, confidently-wrong disease prediction.

This is intentionally lenient: real diseased leaves are often brown/yellow
rather than green, so the threshold accepts green, yellow-green, and
brown/dried-leaf tones, not just pure green.
"""
from PIL import Image
import io

# Minimum fraction of "plant-like" pixels required to pass the check.
# Tuned to be lenient — real disease photos can be mostly brown/yellow.
PLANT_PIXEL_THRESHOLD = 0.12


def is_probably_plant_image(image_bytes):
    """
    Returns (is_plant: bool, plant_pixel_ratio: float)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        # If we can't even open it as an image, let the caller handle that separately.
        return True, 1.0  # don't block on our own parsing failure

    # Downsample for speed — we only need a rough color distribution.
    img = img.resize((80, 80))
    pixels = list(img.getdata())

    plant_like_count = 0
    for r, g, b in pixels:
        # Skip near-white/near-gray/near-black pixels (walls, paper,
        # overexposed backgrounds) — real leaf colors have some saturation.
        if max(r, g, b) - min(r, g, b) < 20:
            continue

        # Green-dominant (healthy leaf greens, various shades)
        is_green = g > r and g > b and g > 40
        # Yellow/brown-ish (dried, diseased, or autumn leaf tones).
        # Real leaf browns/yellows have red and green channels close to
        # each other (true yellow-brown hue). Human skin tones have a much
        # bigger gap between red and green (more orange/pink), so a tight
        # gap threshold here helps tell them apart.
        is_yellow_brown = (
            r > 60 and g > 40 and
            r >= b and g >= b and
            abs(int(r) - int(g)) < 35
        )
        if is_green or is_yellow_brown:
            plant_like_count += 1

    ratio = plant_like_count / len(pixels)
    return ratio >= PLANT_PIXEL_THRESHOLD, round(ratio, 3)
