"""Small, dependency-light image helpers for importing workflow ideas."""

import math
import numbers
from pathlib import Path

from PIL import Image

MAX_SIDE = 8192
MAX_PIXELS = MAX_SIDE * MAX_SIDE
MAX_REFERENCE_MP = 64.0


def _integer(name, value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value)) or not float(value).is_integer()):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _safe_size(width, height):
    if width < 1 or height < 1 or width > MAX_SIDE or height > MAX_SIDE or width * height > MAX_PIXELS:
        raise ValueError("Image dimensions exceed the safe 8192-pixel/64-megapixel limit")


def prepare_outpaint(image, left, right, top, bottom, overlap=0):
    """Return an opaque-gray RGBA canvas and an L edit mask."""
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a Pillow image")
    pads = [_integer(name, value) for name, value in zip(("left", "right", "top", "bottom"),
                                                         (left, right, top, bottom))]
    if any(value % 32 for value in pads):
        raise ValueError("Outpaint padding must use 32-pixel multiples")
    overlap = _integer("overlap", overlap)
    if not any(pads):
        raise ValueError("Choose at least one outpaint direction")
    left, right, top, bottom = pads
    width, height = image.size
    out_w, out_h = width + left + right, height + top + bottom
    _safe_size(out_w, out_h)
    canvas = Image.new("RGBA", (out_w, out_h), (128, 128, 128, 255))
    source = image.convert("RGBA")
    canvas.paste(source, (left, top))
    mask = Image.new("L", (out_w, out_h), 255)
    mask.paste(0, (left, top, left + width, top + height))
    if overlap:
        from PIL import ImageDraw, ImageChops
        feather=Image.new('L',source.size,0)
        for direction in ('left','right','top','bottom'):
            if not dict(zip(('left','right','top','bottom'),pads))[direction]:continue
            layer=Image.new('L',source.size,0);draw=ImageDraw.Draw(layer)
            length=min(overlap,width if direction in ('left','right') else height)
            for index in range(length):
                strength=round(255*(length-index)/length)
                if direction=='left':draw.line((index,0,index,height-1),fill=strength)
                elif direction=='right':draw.line((width-1-index,0,width-1-index,height-1),fill=strength)
                elif direction=='top':draw.line((0,index,width-1,index),fill=strength)
                else:draw.line((0,height-1-index,width-1,height-1-index),fill=strength)
            feather=ImageChops.lighter(feather,layer)
        mask.paste(feather,(left,top))
    return canvas, mask


def fit_reference(image, megapixels=1.0, multiple=32):
    """Downscale a reference when needed; never enlarge it."""
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a Pillow image")
    try:
        megapixels = float(megapixels)
    except (TypeError, ValueError):
        raise ValueError("megapixels must be a finite positive number") from None
    if not math.isfinite(megapixels) or megapixels <= 0 or megapixels > MAX_REFERENCE_MP:
        raise ValueError("megapixels must be finite and between 0 and 64")
    multiple = _integer("multiple", multiple, positive=True)
    width, height = image.size
    _safe_size(width, height)
    target = megapixels * 1_000_000
    if width * height <= target:
        return image.convert("RGBA").copy()
    scale = math.sqrt(target / (width * height))
    scaled_w, scaled_h = width * scale, height * scale
    new_w = int(scaled_w // multiple) * multiple
    new_h = int(scaled_h // multiple) * multiple
    if new_w < multiple or new_h < multiple:
        raise ValueError("Reference aspect ratio is too extreme for the requested size")
    new_w, new_h = min(new_w, width), min(new_h, height)
    _safe_size(new_w, new_h)
    return image.convert("RGBA").resize((new_w, new_h), Image.Resampling.LANCZOS)


def pose_reference(image, mode="Uploaded pose map", resolution=512):
    """Use an uploaded pose map or Forge's locally installed pose preprocessor."""
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a Pillow image")
    if mode == "Uploaded pose map":
        return image.convert("RGBA").copy()
    names = {"OpenPose": ("openpose", ("body_pose_model.pth", "hand_pose_model.pth", "facenet.pth")),
             "DWPose": ("dw_openpose_full", ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"))}
    if mode not in names:
        raise ValueError("Pose mode must be Uploaded pose map, OpenPose, or DWPose")
    resolution = _integer("resolution", resolution, positive=True)
    if not 64 <= resolution <= 2048:
        raise ValueError("Pose resolution must be between 64 and 2048")
    try:
        from annotator.openpose import OpenposeDetector
        from lib_controlnet import global_state
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Forge pose preprocessors are unavailable; upload a pose map instead") from exc
    missing = [name for name in names[mode][1] if not (Path(OpenposeDetector.model_dir) / name).is_file()]
    if missing:
        raise RuntimeError("Local pose model files are missing; upload a pose map instead: " + ", ".join(missing))
    preprocessor = None
    try:
        preprocessor = global_state.get_preprocessor(names[mode][0])
        result = preprocessor(np.asarray(image.convert("RGB")), resolution)
        if isinstance(result, dict):
            result = result.get("image")
        elif isinstance(result, tuple):
            result = result[0] if result else None
        if isinstance(result, Image.Image):
            return result.convert("RGBA")
        if result is None:
            raise ValueError("preprocessor returned no image")
        return Image.fromarray(result).convert("RGBA")
    except Exception as exc:
        raise RuntimeError(f"{mode} preprocessing failed without downloading models: {exc}") from exc
    finally:
        unload = getattr(preprocessor, "unload_function", None)
        if callable(unload):
            unload()
