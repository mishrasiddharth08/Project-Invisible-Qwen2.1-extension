import unittest

import numpy as np
from PIL import Image

from lib.denoise import remove_moire


KERNEL = np.asarray((-1, 6, -15, 20, -15, 6, -1), dtype=np.float32) / 64.0


def reference_band_pass(source, axis):
    padding = [(0, 0)] * source.ndim
    padding[axis] = (3, 3)
    padded = np.pad(source, padding, mode="edge")
    result = np.zeros_like(source)
    for index, coefficient in enumerate(KERNEL):
        section = [slice(None)] * source.ndim
        section[axis] = slice(index, index + source.shape[axis])
        result += padded[tuple(section)] * coefficient
    return result


def reference_filter(pixels, strength=1.0):
    source = pixels.astype(np.float32)
    bx = reference_band_pass(source, axis=1)
    by = reference_band_pass(source, axis=0)
    bxy = reference_band_pass(bx, axis=0)
    filtered = source - bx - by + bxy
    return np.clip(np.rint(source + strength * (filtered - source)), 0, 255).astype(np.uint8)


class DenoiseTests(unittest.TestCase):
    def test_checkerboard_removed_interior(self):
        checker = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
        image = Image.fromarray(np.repeat(checker[..., None], 3, axis=2), "RGB")
        result = np.asarray(remove_moire(image))
        self.assertEqual(np.ptp(result[6:-6, 6:-6]), 0)

    def test_flat_image_unchanged(self):
        pixels = np.full((13, 17, 3), 91, dtype=np.uint8)
        image = Image.fromarray(pixels, "RGB")
        np.testing.assert_array_equal(np.asarray(remove_moire(image)), pixels)

    def test_alpha_preserved_exactly(self):
        pixels = np.zeros((12, 15, 4), dtype=np.uint8)
        pixels[..., :3] = np.indices((12, 15)).sum(axis=0)[..., None] % 2 * 255
        pixels[..., 3] = np.arange(180, dtype=np.uint8).reshape(12, 15)
        result = np.asarray(remove_moire(Image.fromarray(pixels, "RGBA")))
        np.testing.assert_array_equal(result[..., 3], pixels[..., 3])

    def test_rgb_mode_and_dimensions_preserved(self):
        image = Image.new("RGB", (19, 11), (20, 40, 60))
        result = remove_moire(image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, image.size)

    def test_zero_strength_is_unchanged(self):
        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, (10, 14, 4), dtype=np.uint8)
        result = remove_moire(Image.fromarray(pixels, "RGBA"), strength=0)
        np.testing.assert_array_equal(np.asarray(result), pixels)

    def test_matches_direct_bx_by_bxy_formula(self):
        rng = np.random.default_rng(11)
        pixels = rng.integers(80, 176, (9, 12, 3), dtype=np.uint8)
        result = np.asarray(remove_moire(Image.fromarray(pixels, "RGB")))
        np.testing.assert_array_equal(result, reference_filter(pixels))

    def test_strength_blends_with_original(self):
        rng = np.random.default_rng(13)
        pixels = rng.integers(80, 176, (10, 13, 3), dtype=np.uint8)
        result = np.asarray(remove_moire(Image.fromarray(pixels, "RGB"), strength=0.35))
        np.testing.assert_array_equal(result, reference_filter(pixels, strength=0.35))


if __name__ == "__main__":
    unittest.main()
