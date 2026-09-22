"""Small post-process filter for Qwen Image 2.1 Nyquist pattern noise."""

from __future__ import annotations

import numpy as np
from PIL import Image


_KERNEL = np.asarray((-1, 6, -15, 20, -15, 6, -1), dtype=np.float32) / 64.0


def _band_pass(source: np.ndarray, axis: int) -> np.ndarray:
    """Apply B along one axis, extending edge pixels by clamping."""
    length = source.shape[axis]
    base = np.arange(length)
    result = np.zeros_like(source)
    scratch = np.empty_like(source)

    for offset, coefficient in zip(range(-3, 4), _KERNEL):
        indices = np.clip(base + offset, 0, length - 1)
        np.take(source, indices, axis=axis, out=scratch)
        result += scratch * coefficient
    return result


def remove_moire(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Remove horizontal/vertical Nyquist patterning while preserving alpha."""
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("image mode must be RGB or RGBA")
    if strength == 0.0:
        return image.copy()

    pixels = np.asarray(image)
    rgb = pixels[..., :3].astype(np.float32)

    # (I - Bx)(I - By) = I - Bx - By + Bxy.
    filtered = rgb - _band_pass(rgb, axis=1)
    filtered -= _band_pass(filtered, axis=0)
    filtered = rgb + strength * (filtered - rgb)
    output_rgb = np.clip(np.rint(filtered), 0, 255).astype(np.uint8)

    if image.mode == "RGB":
        return Image.fromarray(output_rgb, "RGB")

    output = np.empty_like(pixels)
    output[..., :3] = output_rgb
    output[..., 3] = pixels[..., 3]
    return Image.fromarray(output, "RGBA")
