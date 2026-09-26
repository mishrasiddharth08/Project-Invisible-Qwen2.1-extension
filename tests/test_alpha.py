from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from lib.alpha import clean


def _image(alpha_row):
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[..., 0] = 10
    pixels[..., 1] = 20
    pixels[..., 2] = 30
    pixels[..., 3] = np.asarray(alpha_row, dtype=np.uint8)[None, :]
    return Image.fromarray(pixels, "RGBA")


class TestCleanAlpha(unittest.TestCase):
    def test_non_rgba_passthrough(self):
        source = Image.new("RGB", (4, 4), (1, 2, 3))
        self.assertIs(clean(source), source)

    def test_extremes_snapped(self):
        # 0, 2 snap to 0; 253, 255 snap to 255; mid values untouched.
        result = np.asarray(clean(_image([0, 2, 253, 255])))
        self.assertEqual(list(result[1, :, 3]), [0, 0, 255, 255])

    def test_mid_alpha_untouched(self):
        result = np.asarray(clean(_image([40, 128, 200, 245])))
        self.assertEqual(list(result[1, :, 3]), [40, 128, 200, 245])

    def test_rgb_channels_preserved(self):
        source = _image([0, 2, 253, 255])
        before = np.asarray(source)[..., :3]
        result = np.asarray(clean(source))[..., :3]
        self.assertTrue((before == result).all())

    def test_custom_thresholds(self):
        result = np.asarray(clean(_image([50, 60, 200, 210]), low=55, high=205))
        self.assertEqual(list(result[1, :, 3]), [0, 60, 200, 255])

    def test_swapped_thresholds_clamped(self):
        result = np.asarray(clean(_image([0, 128, 200, 255]), low=250, high=5))
        self.assertEqual(list(result[1, :, 3]), [0, 128, 200, 255])


if __name__ == "__main__":
    unittest.main()
