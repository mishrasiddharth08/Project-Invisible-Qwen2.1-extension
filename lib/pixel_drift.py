"""Optional CPU alignment inspired by Mozer/ComfyUI-PixelDriftFix.

Independent implementation of its recommended SIFT/homography workflow.
No upstream source is bundled (the inspected repository has no license).
Preserves the generated canvas and alpha; rejects uncertain/large warps.
"""
import numpy as np
from PIL import Image


def align(source, edited):
    """Return (image, status); uncertain matches leave the edit unchanged."""
    try:
        import cv2
    except ImportError:
        return edited, 'skipped: OpenCV unavailable'
    if not hasattr(cv2, 'SIFT_create'):
        return edited, 'skipped: SIFT unavailable'
    width, height = edited.size
    scale = min(1.0, 1536.0 / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if min(size) < 32:
        return edited, 'skipped: image too small'
    try:
        reference = np.asarray(source.convert('RGB').resize(size, Image.Resampling.LANCZOS))
        candidate = np.asarray(edited.convert('RGB').resize(size, Image.Resampling.LANCZOS))
        detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.01, edgeThreshold=20)
        ref_points, ref_desc = detector.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None)
        edit_points, edit_desc = detector.detectAndCompute(cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY), None)
        if ref_desc is None or edit_desc is None or min(len(ref_points), len(edit_points)) < 10:
            return edited, 'skipped: insufficient features'
        pairs = cv2.BFMatcher().knnMatch(edit_desc, ref_desc, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance]
        if len(matches) < 10:
            return edited, 'skipped: insufficient matches'
        src = np.float32([edit_points[m.queryIdx].pt for m in matches])
        dst = np.float32([ref_points[m.trainIdx].pt for m in matches])
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return edited, 'skipped: invalid alignment'
        inliers = mask.ravel().astype(bool)
        if inliers.sum() < 10 or inliers.mean() < 0.5:
            return edited, 'skipped: unreliable alignment'
        # A small repeated texture must not determine the whole-image warp.
        spread = np.ptp(dst[inliers], axis=0) / np.asarray(size)
        if np.any(spread < 0.15):
            return edited, 'skipped: matches too localized'
        small = np.diag([size[0] / width, size[1] / height, 1.0])
        matrix = np.linalg.inv(small) @ matrix @ small
        corners = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        projected = cv2.perspectiveTransform(corners[None], matrix)[0]
        if not np.isfinite(projected).all():
            return edited, 'skipped: invalid geometry'
        ratio = cv2.contourArea(projected, oriented=True) / max(1, cv2.contourArea(corners, oriented=True))
        displacement = np.linalg.norm(projected - corners, axis=1).max()
        if not cv2.isContourConvex(projected) or not 0.75 <= ratio <= 1.33 or displacement > 0.15 * np.hypot(width, height):
            return edited, 'skipped: framing change too large'
        if displacement < 0.1:
            return edited, 'already aligned'
        mode = 'RGBA' if 'A' in edited.getbands() else 'RGB'
        pixels = np.asarray(edited.convert(mode))
        fixed = cv2.warpPerspective(pixels, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(fixed), 'applied'
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return edited, 'skipped: alignment failed'


def apply(source, edited, enabled=False):
    if not enabled:
        return edited, 'off'
    if source is None:
        return edited, 'skipped: requires an edit source'
    if isinstance(source, (str, bytes)) or hasattr(source, '__fspath__'):
        with Image.open(source) as image:
            return align(image, edited)
    return align(source, edited)
