"""Klein-style edit composite: detection, cleanup and blending."""
import importlib.util
import sys
import unittest
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qwen21_composite', ROOT / 'lib/composite.py')
composite = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = composite
spec.loader.exec_module(composite)


def solid(color, size=(32, 32)):
    return Image.new('RGB', size, color)


def with_patch(image, color, box=(8, 8, 24, 24)):
    out = image.copy()
    out.paste(color, box)
    return out


class CompositeTests(unittest.TestCase):
    def test_identical_images_change_nothing(self):
        base = solid((120, 80, 40))
        blended, share = composite.composite(base, base.copy())
        self.assertEqual(share, 0.0)
        self.assertEqual(blended.getpixel((2, 2)), (120, 80, 40))

    def test_real_edit_is_detected_and_preserved(self):
        base = solid((120, 80, 40))
        edited = with_patch(base, (250, 240, 10))
        blended, share = composite.composite(base, edited)
        self.assertGreater(share, 0.01)
        self.assertGreaterEqual(blended.size, base.size)
        # The patch center must carry the edited color; the corner must not.
        center = blended.getpixel((16, 16))
        corner = blended.getpixel((2, 2))
        self.assertGreater(center[0], 200)
        self.assertLess(corner[0], 160)

    def test_mask_refine_removes_single_pixel_specks(self):
        mask = [[False] * 32 for _ in range(32)]
        mask[10][10] = True  # isolated speck far below the island threshold
        cleaned = composite.refine(composite.numpy.array(mask, dtype=bool))
        self.assertFalse(cleaned.any())

    def test_color_match_returns_an_image(self):
        base = solid((100, 100, 100))
        edited = with_patch(base, (200, 50, 50))
        matched = composite.color_match(base, edited, composite.change_mask(base, edited))
        self.assertEqual(matched.size, base.size)


if __name__ == '__main__':
    unittest.main()