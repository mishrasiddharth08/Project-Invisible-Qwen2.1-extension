"""Request-scoped Spectrum acceleration for Diffusers Qwen-Image-2.1.

The forecasting method is adapted from ComfyUI-Spectrum-QwenImage21
(MIT, Copyright (c) 2026 ComfyUI-Spectrum-QwenImage21 contributors):
https://github.com/awdqwdasdg/Comfyui-Spectrum-Qwen2.1

This is a small Diffusers adapter, not the incompatible Forge/Comfy wrapper.
"""
from __future__ import annotations

import contextlib
import math
import types
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SpectrumStats:
    actual: int = 0
    forecast: int = 0
    reason: str = "disabled"


def supported(pipe, true_cfg_scale=1.0):
    model = getattr(pipe, "transformer", None)
    required = ("norm_out", "proj_out", "time_text_embed", "transformer_blocks")
    return (
        model is not None
        and model.__class__.__name__ == "QwenImage21Transformer2DModel"
        and all(hasattr(model, name) for name in required)
        and float(true_cfg_scale or 1.0) <= 1.0
    )


def _basis(torch, values, degree=2):
    x = torch.tensor(values, dtype=torch.float32)
    columns = [torch.ones_like(x), x]
    for _ in range(2, degree + 1):
        columns.append(2 * x * columns[-1] - columns[-2])
    return torch.stack(columns[: degree + 1], dim=1)


def _forecast(torch, history, coord, blend=0.5):
    """Ridge Chebyshev prediction blended with linear extrapolation."""
    coords = [item[0] for item in history]
    design = _basis(torch, coords)
    gram = design.T @ design + 0.1 * torch.eye(design.shape[1])
    phi = _basis(torch, [coord])
    spectral = (phi @ torch.linalg.solve(gram, design.T)).flatten()
    linear = torch.zeros(len(history), dtype=torch.float32)
    spacing = coords[-1] - coords[-2]
    ratio = 0.0 if abs(spacing) < 1e-12 else (coord - coords[-1]) / spacing
    linear[-2], linear[-1] = -ratio, 1.0 + ratio
    weights = blend * spectral + (1.0 - blend) * linear
    result = torch.zeros_like(history[-1][1], dtype=torch.float32)
    for weight, (_, feature) in zip(weights.tolist(), history):
        result.add_(feature.float(), alpha=weight)
    return result.to(history[-1][1].dtype)


def _argument(args, kwargs, name, position, default=None):
    return kwargs.get(name, args[position] if len(args) > position else default)


def _output(model, args, kwargs, feature):
    import torch

    hidden = _argument(args, kwargs, "hidden_states", 0)
    timestep = _argument(args, kwargs, "timestep", 2)
    img_shapes = _argument(args, kwargs, "img_shapes", 3)
    img_mask = _argument(args, kwargs, "img_mask", 4)
    timestep = timestep.to(hidden.dtype)
    if getattr(model.config, "causal_condition", False):
        timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        repeats = torch.where(img_mask, 4, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        _, modulation_mask = model.build_token_metadata(image_pad_mask, img_shapes[0])
        if _argument(args, kwargs, "kv_cache_mode", 8) == "cached":
            prefix_len = int((~modulation_mask).sum())
            modulation_mask = modulation_mask[prefix_len:]
    else:
        modulation_mask = None
    temb = model.time_text_embed(timestep, hidden)
    value = model.proj_out(
        model.norm_out(feature.to(hidden.device, hidden.dtype), temb, modulation_mask)
    )
    if _argument(args, kwargs, "return_dict", 9, True):
        try:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=value)
        except ImportError:
            return types.SimpleNamespace(sample=value)
    return (value,)


@contextlib.contextmanager
def accelerate(pipe, enabled=False, steps=0, stop=None):
    """Temporarily accelerate one pipeline request and always restore hooks."""
    stats = SpectrumStats(reason="disabled")
    if not enabled:
        yield stats
        return
    model = getattr(pipe, "transformer", None)
    if not supported(pipe):
        stats.reason = "unsupported_model"
        yield stats
        return
    total = max(1, int(steps))
    if total < 8:
        stats.reason = "too_few_steps"
        yield stats
        return

    import torch

    original = model.forward
    history = []
    call = 0
    consecutive = 0
    stats.reason = "active"

    def wrapped(_self, *args, **kwargs):
        nonlocal call, consecutive
        if stop is not None and Path(stop).exists():
            raise InterruptedError("Generation interrupted")
        index = call
        call += 1
        coord = 0.0 if total <= 1 else 2.0 * index / (total - 1) - 1.0
        cache_mode = _argument(args, kwargs, "kv_cache_mode", 8)
        actual = (
            cache_mode != "cached"
            or index < 3
            or index >= total - 2
            or len(history) < 3
            or consecutive >= 1
        )
        if not actual:
            predicted = _forecast(torch, history, coord)
            consecutive += 1
            stats.forecast += 1
            return _output(model, args, kwargs, predicted)

        captured = {}
        def capture(_module, values):
            captured["feature"] = values[0].detach().to("cpu").contiguous()
            return None

        handle = model.norm_out.register_forward_pre_hook(capture)
        try:
            result = original(*args, **kwargs)
        finally:
            handle.remove()
        feature = captured.get("feature")
        if feature is not None:
            if history and tuple(history[-1][1].shape) != tuple(feature.shape):
                history.clear()
            history.append((coord, feature))
            del history[:-3]
        consecutive = 0
        stats.actual += 1
        return result

    model.forward = types.MethodType(wrapped, model)
    try:
        yield stats
    finally:
        model.forward = original
        history.clear()
