"""Klein-style edit composite: blend an AI edit back over the original image.

Adapted from comfyui-klein-edit-composite (difference detection, mask
refinement, optional color matching, feathered blending). Only the parts that
matter for Forge single-image edits are ported, in pure NumPy/PIL so no new
dependencies are required: alignment (optional OpenCV ECC when available),
hybrid change detection, mask cleanup, Reinhard background color transfer and
edge-aware feathered compositing.
"""
import math
import numpy


def _to_array(image):
    return numpy.asarray(image.convert('RGB'), dtype=numpy.float32)


def align(original, generated, iterations=64):
    """Warp the generated image onto the original when OpenCV is available.

    Klein uses two-pass SIFT + homography + optical flow; for same-size Forge
    edits the dominant residual is small perspective/lighting drift, which an
    ECC intensity alignment handles well. Without OpenCV the images are used
    as-is (they already share the canvas in our pipeline).
    """
    try:
        import cv2
    except ImportError:
        return generated
    orig = _to_array(original)
    gen = _to_array(generated)
    if orig.shape != gen.shape:
        return generated
    try:
        warp = numpy.eye(2, 3, dtype=numpy.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(iterations), 1e-6)
        _, warp = cv2.findTransformECC(
            cv2.cvtColor(orig.astype(numpy.uint8), cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(gen.astype(numpy.uint8), cv2.COLOR_RGB2GRAY),
            warp, cv2.MOTION_AFFINE, criteria, None, 5)
        h, w = orig.shape[:2]
        warped = cv2.warpAffine(gen, warp, (w, h),
                                flags=cv2.INTER_LINEAR +
                                cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
        return PIL_from_array(warped)
    except cv2.error:
        return generated


def PIL_from_array(array):
    from PIL import Image
    return Image.fromarray(numpy.clip(array, 0, 255).astype(numpy.uint8), 'RGB')


def delta_e(original, generated):
    """Approximate perceptual difference in LAB space (Delta E, CIE76)."""
    try:
        import cv2
        o = cv2.cvtColor(original.astype(numpy.uint8), cv2.COLOR_RGB2LAB)
        g = cv2.cvtColor(generated.astype(numpy.uint8), cv2.COLOR_RGB2LAB)
        return numpy.linalg.norm(o.astype(numpy.float32) - g, axis=2)
    except ImportError:
        # Fallback: weighted RGB distance (rough perceptual proxy).
        return numpy.sqrt(numpy.sum((original - generated) ** 2, axis=2)) / 1.5


def structure(original, generated):
    """Sobel gradient-magnitude difference (structural change cue)."""
    def gray(image):
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def sobel(channel):
        gx = numpy.zeros_like(channel)
        gy = numpy.zeros_like(channel)
        gx[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
        gy[1:-1, :] = channel[2:, :] - channel[1:-1, :]
        return numpy.sqrt(gx * gx + gy * gy)

    return numpy.abs(sobel(gray(original)) - sobel(gray(generated)))


def change_mask(original, generated, color_threshold=12.0,
                structure_threshold=0.12):
    """Hybrid diff map: perceptual color OR structural change counts."""
    orig = _to_array(original)
    gen = _to_array(generated)
    if orig.shape != gen.shape:
        gen = _to_array(generated.resize(original.size))
    color = delta_e(orig, gen)
    struct = structure(orig, gen)
    struct_max = float(struct.max()) or 1.0
    mask = (color >= color_threshold) | (struct / struct_max >= structure_threshold)
    return mask


def refine(mask, min_island=64, border_bleed=2):
    """Remove specks, fill holes, prune border slivers, keep float 0..1."""
    from scipy import ndimage
    clean = ndimage.binary_closing(mask, structure=numpy.ones((3, 3)))
    clean = ndimage.binary_opening(clean, structure=numpy.ones((3, 3)))
    filled = ndimage.binary_fill_holes(clean)
    labels, count = ndimage.label(filled)
    if count:
        sizes = numpy.bincount(labels.ravel())
        small = numpy.where(sizes < min_island)[0]
        filled[numpy.isin(labels, small[small > 0])] = False
    if border_bleed:
        edge = numpy.zeros_like(filled)
        b = border_bleed
        edge[:b, :] = True; edge[-b:, :] = True
        edge[:, :b] = True; edge[:, -b:] = True
        filled &= ~ndimage.binary_dilation(edge, iterations=1)
    return filled.astype(numpy.float32)


def color_match(original, generated, mask):
    """Reinhard LAB transfer toward the original, using background stats only."""
    try:
        import cv2
    except ImportError:
        return generated
    orig_lab = cv2.cvtColor(_to_array(original).astype(numpy.uint8), cv2.COLOR_RGB2LAB).astype(numpy.float32)
    gen_lab = cv2.cvtColor(_to_array(generated).astype(numpy.uint8), cv2.COLOR_RGB2LAB).astype(numpy.float32)
    background = mask < 0.5
    if background.sum() < 16:
        return generated
    shifted = gen_lab.copy()
    for channel in range(3):
        o_mean, o_std = orig_lab[..., channel][background].mean(), orig_lab[..., channel][background].std() + 1e-5
        g_mean, g_std = gen_lab[..., channel][background].mean(), gen_lab[..., channel][background].std() + 1e-5
        shifted[..., channel] = (gen_lab[..., channel] - g_mean) * (o_std / g_std) + o_mean
    out = cv2.cvtColor(numpy.clip(shifted, 0, 255).astype(numpy.uint8), cv2.COLOR_LAB2RGB)
    return PIL_from_array(out.astype(numpy.float32))


def blend(original, generated, mask, feather=9):
    """Guided-feather alpha blend of the generated edit over the original."""
    from PIL import Image, ImageFilter
    alpha = PIL_from_array(numpy.dstack([mask * 255] * 3)).convert('L')
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather / 2.0))
        alpha = alpha.point(lambda v: 255 if v >= 128 else int(v * 2))
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather / 2.0))
    return Image.composite(generated, original, alpha)


def composite(original, generated, color_threshold=12.0,
              structure_threshold=0.12, match_colors=True, feather=9):
    """Full Klein-style pass; returns (blended image, changed-pixel share)."""
    mask = change_mask(original, generated, color_threshold, structure_threshold)
    clean = refine(mask)
    if match_colors:
        generated = color_match(original, generated, clean)
    share = float(clean.mean())
    if share <= 0.0:
        return original.copy(), 0.0
    return blend(original, generated, clean, feather), share