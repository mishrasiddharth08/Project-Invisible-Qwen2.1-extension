"""Clean dirty alpha channels from Qwen 2.1 Transparent PNG output.

Qwen 2.1's RGBA decode often leaves semi-transparent noise: pixels that
should be fully opaque come out at e.g. 90% alpha and cut-out edges keep a
ghosted fringe. Thresholding the channel fixes this while keeping genuine
soft edges (hair, glass) untouched.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Alpha values (0-255) below/above these bounds are snapped to 0/255.
LOW = 5
HIGH = 250


def clean(image: Image.Image, low: int = LOW, high: int = HIGH) -> Image.Image:
    """Snap near-opaque alpha to 255 and near-transparent alpha to 0."""
    if image.mode != "RGBA":
        return image
    low = max(0, min(255, int(low)))
    high = max(0, min(255, int(high)))
    if low > high:  # nonsensical input: fall back to the safe defaults
        low, high = LOW, HIGH
    pixels = np.asarray(image).copy()
    alpha = pixels[..., 3]
    pixels[..., 3] = np.where(alpha < low, 0, np.where(alpha > high, 255, alpha))
    return Image.fromarray(pixels, "RGBA")