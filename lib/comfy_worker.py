"""Isolated ComfyUI Qwen-Image-2.1 worker (JSON-lines protocol).

This worker imports a pinned ComfyUI checkout only inside its subprocess.  It
never starts Comfy's server and never downloads models or changes Forge.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path


WIRE = sys.stdout
_MISSING = object()
_EMIT_LOCK = threading.Lock()


def emit(kind, **data):
    with _EMIT_LOCK:
        WIRE.write(json.dumps({"type": kind, **data}, separators=(",", ":")) + "\n")
        WIRE.flush()


def _path(value, label, required=True):
    if not value:
        if required:
            raise ValueError(f"Missing {label} path")
        return None
    result = Path(value).expanduser().resolve()
    if not result.is_file():
        raise ValueError(f"{label} file does not exist: {result}")
    return result


def _component(bundle, *names):
    files = bundle.get("files") or {}
    for name in names:
        if files.get(name):
            return files[name]
    return None


def _node_values(value):
    result = getattr(value, "result", value)
    if not isinstance(result, (tuple, list)):
        raise RuntimeError("Unexpected Comfy node result")
    return tuple(result)


def _validate_qwen21_model(model):
    diffusion = model.get_model_object("diffusion_model")
    identity = (type(diffusion).__module__ + "." + type(diffusion).__name__).lower()
    blocks = getattr(diffusion, "transformer_blocks", None)
    if "qwen_image21" not in identity or blocks is None or len(blocks) != 32:
        raise ValueError("Selected diffusion model is not a supported Qwen-Image-2.1 32-block model")


def _validate_qwen21_control(model_patch):
    control = getattr(model_patch, "model", None)
    identity = (type(control).__module__ + "." + type(control).__name__).lower()
    blocks = getattr(control, "control_blocks", None)
    if "qwenimage21funcontrol" not in identity or blocks is None or len(blocks) != 16:
        raise ValueError("Control model is not a Qwen-Image-2.1 Fun ControlNet with 16 blocks")


def _validate_qwen21_vae(vae):
    try:
        contract = (int(vae.latent_channels), int(vae.output_channels),
                    int(vae.spacial_compression_decode()))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Selected VAE has no valid Qwen-Image-2.1 contract") from exc
    if contract != (64, 4, 16):
        raise ValueError("Selected VAE is not Qwen-Image-2.1 RGBA (64 latent channels, 4 output channels, 16x)")


def _reject_unavailable(command):
    requested = []
    if command.get("spectrum"):
        requested.append("Spectrum")
    if command.get("lanpaint"):
        requested.append("LanPaint")
    if str(command.get("refiner") or "off").lower() != "off":
        requested.append("refiner")
    if requested:
        raise ValueError("Comfy backend does not support: " + ", ".join(requested))


def _sampling_status(completed, total, started, now=None):
    elapsed = max(0, int((time.monotonic() if now is None else now) - started))
    return f"sampling {int(completed)}/{int(total)} — {elapsed}s"


def _ram_preflight(bundle, prof):
    name='_pi_qwen21_comfy_ram'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('ram.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name].preflight(bundle,prof)


def _cuda_budget(torch, prof):
    requested = float((prof or {}).get("vram_gb") or 0)
    if requested <= 0 or not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    external = max(0.0, (total_bytes - free_bytes) / 2**30)
    reserve = float((prof or {}).get("reserve_gb", 1.25 if requested <= 8 else 0.5))
    usable = min(requested, total) - external - reserve
    if usable <= 0:
        raise RuntimeError(f"VRAM profile {requested:g} GB has no free Comfy worker budget")
    torch.cuda.set_per_process_memory_fraction(min(1.0, usable / total), 0)
    return usable


def _cap_comfy_free_memory(model_management, torch, budget_gb):
    """Make Comfy's loader honor this worker's private CUDA budget."""
    current = model_management.get_free_memory
    original = getattr(current, "_pi_original", current)
    if budget_gb <= 0:
        model_management.get_free_memory = original
        return original
    budget = int(float(budget_gb) * 2**30)

    def capped(dev=None, torch_free_too=False):
        actual_total, actual_cached = original(dev, torch_free_too=True)
        target = model_management.get_torch_device() if dev is None else dev
        if getattr(target, "type", str(target).split(":", 1)[0]) != "cuda":
            return (actual_total, actual_cached) if torch_free_too else actual_total
        remaining = max(0, budget - int(torch.cuda.memory_allocated(target)))
        total = min(int(actual_total), remaining)
        cached = min(int(actual_cached), total)
        return (total, cached) if torch_free_too else total

    capped._pi_original = original
    model_management.get_free_memory = capped
    return capped


def _clear_qwen_prefix_cache(model):
    diffusion = model.get_model_object("diffusion_model")
    reset = getattr(diffusion, "reset_prefix_cache", None)
    if reset is not None:
        reset(False)


def _idle_offload(model_management, torch):
    model_management.unload_all_models()
    soft_empty = getattr(model_management, "soft_empty_cache", None)
    if soft_empty is not None:
        soft_empty()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rewrite_data(text, original):
    """Validate enhancer output as data; never interpret it as instructions."""
    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Prompt rewriter did not return valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Prompt rewriter JSON must be an object")
    rewritten = value.get("rewritten_prompt")
    if not isinstance(rewritten, str) or not rewritten.strip() or len(rewritten) > 16000:
        raise ValueError("Prompt rewriter returned an invalid rewritten_prompt")
    follow = value.get("ratio_follow", "")
    if not isinstance(follow, str) or (follow and not re.fullmatch(r"<image(?:[1-9]|10)>", follow)):
        raise ValueError("Prompt rewriter ratio_follow must be empty or <image1> through <image10>")
    ratio = value.get("wh_ratio")
    if ratio is not None and not isinstance(ratio, (str, int, float)):
        raise ValueError("Prompt rewriter wh_ratio must be a number or W:H string")
    if ratio == "":
        ratio = None
    if ratio is not None and follow:
        raise ValueError("Prompt rewriter must choose wh_ratio or ratio_follow, not both")
    return {"prompt": rewritten.strip(), "wh_ratio": ratio, "ratio_follow": follow,
            "original_prompt": original}


def _ratio_size(width, height, ratio):
    if ratio is None:
        return width, height
    if isinstance(ratio, str) and ":" in ratio:
        left, right = ratio.split(":", 1)
        value = float(left) / float(right)
    else:
        value = float(ratio)
    if not 0.25 <= value <= 4.0:
        raise ValueError("Prompt rewriter wh_ratio must be between 1:4 and 4:1")
    area = max(32 * 32, int(width) * int(height))
    new_width = round((area * value) ** 0.5 / 32) * 32
    new_height = round((area / value) ** 0.5 / 32) * 32
    return max(32, min(4096, new_width)), max(32, min(4096, new_height))


def _reference_size(refs, token):
    index = int(re.fullmatch(r"<image(\d+)>", token).group(1)) - 1
    if index >= len(refs):
        raise ValueError(f"Prompt rewriter requested {token}, but that reference image is missing")
    height, width = refs[index].shape[1:3]
    return max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)


def _encoder_role(keys, shape):
    keys = set(keys)
    pe_layers = {int(key.split(".")[3]) for key in keys
                 if key.startswith("model.language_model.layers.") and key.split(".")[3].isdigit()}
    if (shape("lm_head.weight") == (248320, 4096) and
            shape("model.language_model.embed_tokens.weight") == (248320, 4096) and
            pe_layers == set(range(32))):
        return "prompt_encoder"
    base_prefix = "qwen3vl_8b.transformer.model." if any(
        key.startswith("qwen3vl_8b.transformer.model.") for key in keys) else "model."
    base_layers = {int(key[len(base_prefix + "layers."):].split(".")[0]) for key in keys
                   if key.startswith(base_prefix + "layers.") and
                   key[len(base_prefix + "layers."):].split(".")[0].isdigit()}
    if (shape(base_prefix + "embed_tokens.weight") == (151936, 4096) and
            base_layers == set(range(36)) and
            ("model.visual.patch_embed.proj.weight" in keys or
             any(key.startswith("qwen3vl_8b.transformer.visual.") for key in keys))):
        return "qwen3vl_8b"
    return "unknown"


def _encoder_kind(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        def shape(key):
            return tuple(handle.get_slice(key).get_shape()) if key in keys else None
        return _encoder_role(keys, shape)


def _prepare_text_generator(clip, model_management):
    """Text generation cannot run quantized Qwen weights from CPU."""
    device = model_management.get_torch_device()
    if getattr(device, "type", str(device).split(":", 1)[0]) == "cpu":
        raise RuntimeError("Prompt enhancer requires an accelerator device")
    clip.patcher.load_device = device
    return device


def _maybe_full_load_text_generator(clip, model_management):
    patcher = clip.patcher
    reserve = 2 * 2**30
    required = int(patcher.model_size()) + reserve
    if model_management.get_free_memory(patcher.load_device) < required:
        return False
    model_management.load_models_gpu(
        [patcher], memory_required=reserve,
        minimum_memory_required=reserve, force_full_load=True)
    return True


def _enhancer_options(command):
    length = int(command.get("prompt_enhancer_max_length", 4096) or 4096)
    if not 64 <= length <= 16256:
        raise ValueError("Prompt enhancer max length must be between 64 and 16256")
    thinking = bool(command.get("rewrite_thinking", False))
    sampling = ({
        "sampling_mode": "on", "temperature": 1.0, "top_k": 20,
        "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0,
        "presence_penalty": 1.5, "seed": int(command.get("seed", 0)),
    } if thinking else {"sampling_mode": "off"})
    return length, sampling, thinking


def _check_rewrite_stop(stop, model_management):
    if Path(stop).exists():
        model_management.interrupt_current_processing(True)
        return True
    return False


def _scheduler_snapshot(samplers):
    return {
        "SchedulerHandler": samplers.SchedulerHandler,
        "SCHEDULER_HANDLERS": samplers.SCHEDULER_HANDLERS,
        "SCHEDULER_NAMES": samplers.SCHEDULER_NAMES,
        "SCHEDULERS": getattr(samplers, "SCHEDULERS", _MISSING),
        "KSampler.SCHEDULERS": samplers.KSampler.SCHEDULERS,
    }


def _restore_schedulers(samplers, snapshot):
    samplers.SchedulerHandler = snapshot["SchedulerHandler"]
    samplers.SCHEDULER_HANDLERS = snapshot["SCHEDULER_HANDLERS"]
    samplers.SCHEDULER_NAMES = snapshot["SCHEDULER_NAMES"]
    if snapshot["SCHEDULERS"] is _MISSING:
        with contextlib.suppress(AttributeError):
            del samplers.SCHEDULERS
    else:
        samplers.SCHEDULERS = snapshot["SCHEDULERS"]
    samplers.KSampler.SCHEDULERS = snapshot["KSampler.SCHEDULERS"]


def _sigma_tensor(values, steps, torch):
    if not values:
        return None
    sigmas = [float(value) for value in values]
    if len(sigmas) != int(steps) or any(value <= 0 for value in sigmas):
        raise ValueError("Turbo sigma schedule must contain one positive value per step")
    if any(sigmas[i] <= sigmas[i + 1] for i in range(len(sigmas) - 1)):
        raise ValueError("Turbo sigma schedule must be strictly descending")
    # Comfy samplers include the terminal transition explicitly.
    return torch.tensor(sigmas + [0.0], dtype=torch.float32)


def _image(path, torch):
    if not path:
        return None
    from PIL import Image
    import numpy as np
    with Image.open(path) as source:
        value = np.asarray(source.convert("RGBA"), dtype=np.float32) / 255.0
    return torch.from_numpy(value).unsqueeze(0)


def _mask(path, torch):
    if not path:
        return None
    from PIL import Image
    import numpy as np
    with Image.open(path) as source:
        value = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(value).unsqueeze(0)


def _save(image, path):
    from PIL import Image
    import numpy as np
    value = image[0] if getattr(image, "ndim", 0) == 4 else image
    value = value.detach().float().cpu().clamp(0, 1).numpy()
    channels = value.shape[-1]
    if channels not in (3, 4):
        raise ValueError(f"Qwen VAE returned {channels} channels; expected RGB or RGBA")
    if channels == 3:
        value = np.concatenate([value, np.ones((*value.shape[:2], 1), dtype=value.dtype)], axis=-1)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8), "RGBA").save(path, "PNG")


def _native_previewer(model, preview_module):
    latent_format = model.model.latent_format
    factors = latent_format.latent_rgb_factors
    channels = int(getattr(latent_format, "latent_channels", 0))
    if channels != 64 or factors is None or len(factors) != 64:
        raise ValueError("Qwen-Image-2.1 native preview requires its 64-channel RGB factors")
    return preview_module.Latent2RGBPreviewer(
        factors, latent_format.latent_rgb_factors_bias,
        latent_format.latent_rgb_factors_reshape,
    )


def _write_preview(previewer, x0, path, size):
    from PIL import Image
    if getattr(x0, "ndim", 0) < 4 or int(x0.shape[1]) != 64:
        raise ValueError("Qwen native preview expected a 64-channel latent")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = previewer.decode_latent_to_preview(x0.detach())
    image.thumbnail((int(size), int(size)), Image.Resampling.LANCZOS)
    temporary = path.with_name(path.name + ".tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def _composite(image, source, mask, torch):
    """Keep source pixels outside the native sampling mask, including alpha."""
    if source is None or mask is None:
        return image
    target = image[0] if image.ndim == 4 else image
    h, w = target.shape[:2]
    src = source[:1].movedim(-1, 1)
    src = torch.nn.functional.interpolate(src, size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)[0]
    regen = torch.nn.functional.interpolate(mask[:1].unsqueeze(1), size=(h, w), mode="nearest")[0].movedim(0, -1).clamp(0, 1)
    if target.shape[-1] == 3:
        target = torch.cat([target, torch.ones_like(target[..., :1])], dim=-1)
    if src.shape[-1] == 3:
        src = torch.cat([src, torch.ones_like(src[..., :1])], dim=-1)
    result = target * regen + src * (1.0 - regen)
    return result.unsqueeze(0)


class Backend:
    def __init__(self, comfy_root, res4lyf=None, comfy_deps=None, prof=None):
        root = Path(comfy_root).expanduser().resolve()
        if not (root / "comfy_extras" / "nodes_qwen.py").is_file():
            raise ValueError("Pinned ComfyUI checkout has no Qwen node module: " + str(root))
        if comfy_deps:
            deps = Path(comfy_deps).expanduser().resolve()
            if not (deps / "comfy_aimdo").is_dir():
                raise ValueError("Comfy dependency directory lacks comfy_aimdo: " + str(deps))
            sys.path.insert(0, str(deps))
        sys.path.insert(0, str(root))
        options = importlib.import_module("comfy.options")
        options.args_parsing = False
        cli_args = importlib.import_module("comfy.cli_args")
        prof = prof or {}
        low = 0 < float(prof.get("vram_gb") or 0) <= 8
        offload = bool(prof.get("_offload", True))
        cli_args.args.lowvram = bool(low or offload)
        cli_args.args.disable_smart_memory = offload
        cli_args.args.highvram = cli_args.args.gpu_only = cli_args.args.novram = False
        import torch
        self.cuda_budget_gb = _cuda_budget(torch, prof)
        self.root = root
        self.folder_paths = importlib.import_module("folder_paths")
        self.nodes = importlib.import_module("nodes")
        self.sample = importlib.import_module("comfy.sample")
        self.model_management = importlib.import_module("comfy.model_management")
        _cap_comfy_free_memory(self.model_management, torch, self.cuda_budget_gb)
        qwen = importlib.import_module("comfy_extras.nodes_qwen")
        patches = importlib.import_module("comfy_extras.nodes_model_patch")
        self.encoder = getattr(qwen, "TextEncodeQwenImage21", None)
        self.cache_node = getattr(qwen, "QwenImage21Cache", None)
        if self.encoder is None or self.cache_node is None:
            raise RuntimeError("This ComfyUI checkout lacks native Qwen-Image-2.1 nodes")
        self.patch_loader = patches.ModelPatchLoader
        self.control_apply = patches.ZImageFunControlnet
        textgen = importlib.import_module("comfy_extras.nodes_textgen")
        self.text_generate = textgen.TextGenerate
        lib_root = str(Path(__file__).resolve().parent)
        if lib_root not in sys.path:
            sys.path.insert(0, lib_root)
        enhancer = importlib.import_module("vendor.comfy_qwen_enhancer.qwen_image21_conditioning_phrase_encoder")
        reference = importlib.import_module("vendor.comfy_qwen_enhancer.qwen_image21_reference_strength")
        self.phrase_encoder = enhancer.QwenImage21ConditioningPhraseEncoder
        self.reference_strength = reference.QwenImage21ReferenceStrength
        helper_spec = importlib.util.spec_from_file_location("_pi_qwen21_enhancer_parse", Path(__file__).with_name("enhancer.py"))
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        self.references = helper.references
        self.res4lyf = Path(res4lyf).expanduser().resolve() if res4lyf else None
        self._res4lyf_loaded = False

    def register(self, category, path):
        path = _path(path, category)
        self.folder_paths.add_model_folder_path(category, str(path.parent), is_default=True)
        return path.name

    def enable_res2s(self):
        samplers = importlib.import_module("comfy.samplers")
        if "res_2s" in samplers.KSAMPLER_NAMES:
            return
        if not self.res4lyf or not (self.res4lyf / "__init__.py").is_file():
            raise RuntimeError("res_2s requires the explicitly configured RES4LYF directory")
        parent = str(self.res4lyf.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        native_schedulers = _scheduler_snapshot(samplers)
        try:
            importlib.import_module(self.res4lyf.name)
        finally:
            # The isolated RES4LYF compatibility layer registers res_2s but
            # also replaces Comfy's scheduler registry. Keep the sampler and
            # restore the exact native scheduler objects (including Beta).
            _restore_schedulers(samplers, native_schedulers)
        self._res4lyf_loaded = True
        if "res_2s" not in samplers.KSAMPLER_NAMES:
            raise RuntimeError("Configured RES4LYF did not register res_2s")


class Session:
    def __init__(self, backend, bundle, prof):
        self.backend = backend
        self.prof = prof or {}
        files = bundle.get("files") or {}
        if not files:
            raise ValueError("Comfy backend requires explicit local Qwen component files")
        unet = backend.register("diffusion_models", _component(bundle, "transformer", "dit", "unet"))
        clip = backend.register("text_encoders", _component(bundle, "text_encoder", "te", "clip"))
        vae = backend.register("vae", _component(bundle, "vae"))
        self.model = backend.nodes.UNETLoader().load_unet(unet, self.prof.get("weight_dtype", "default"))[0]
        _validate_qwen21_model(self.model)
        clip_full_path = _path(_component(bundle, "text_encoder", "te", "clip"), "text encoder")
        if _encoder_kind(clip_full_path) != "qwen3vl_8b":
            raise ValueError("Base Qwen model requires Qwen3VL-8B; a prompt-enhancer encoder cannot replace it")
        self.clip = backend.nodes.CLIPLoader().load_clip(clip, "qwen_image", self.prof.get("clip_device", "default"))[0]
        self.vae = backend.nodes.VAELoader().load_vae(vae)[0]
        _validate_qwen21_vae(self.vae)
        self.prompt_encoders = {}
        self.control_patches = {}

    def _lora_model(self, adapters):
        model = self.model
        loader = self.backend.nodes.LoraLoaderModelOnly()
        for item in adapters or []:
            path = self.backend.register("loras", item.get("path"))
            model = loader.load_lora_model_only(model, path, float(item.get("weight", 1.0)))[0]
            _validate_qwen21_model(model)
        return model

    def _cache_model(self, model, enabled):
        device = self.prof.get("cache_device", "auto") if enabled else "off"
        dtype = self.prof.get("cache_dtype", "default")
        return _node_values(self.backend.cache_node.execute(model, device, dtype))[0]

    def _reference_model(self, model, value, ref_count):
        weights = self.backend.references(value)
        if weights and max(weights) > ref_count:
            raise ValueError("A selected reference priority is missing")
        for index, strength in weights.items():
            model = _node_values(self.backend.reference_strength.execute(model, index, strength))[0]
        if weights:
            emit("status", status="Reference weighting: " + ", ".join(f"{i}:{w:g}" for i, w in weights.items()))
        return model

    def _conditioning(self, command, prompt, negative, images):
        values = dict(clip=self.clip, prompt=prompt, negative_prompt=negative,
                      vae=self.vae, resolution=int(command.get("reference_resolution", 0) or 0),
                      images=images)
        if command.get("phrase_weights"):
            result = _node_values(self.backend.phrase_encoder.execute(**values))
            emit("status", status="Native Qwen phrase weighting enabled")
            return result[:3]
        return _node_values(self.backend.encoder.execute(**values))

    def _rewrite(self, command, prompt):
        if not command.get("rewrite_prompt"):
            return {"prompt": prompt, "wh_ratio": None, "ratio_follow": "",
                    "original_prompt": prompt}
        path = _path(command.get("prompt_encoder"), "prompt encoder")
        if _encoder_kind(path) != "prompt_encoder":
            raise ValueError("Selected prompt encoder is not a Qwen 2.1 prompt-enhancer model")
        system = str(command.get("system_prompt") or "")
        if not system.strip() or len(system) > 32000:
            raise ValueError("Prompt rewrite requires a selected system prompt resource")
        manager = self.backend.model_management
        stop = Path(command["stop"])
        manager.interrupt_current_processing(False)
        if _check_rewrite_stop(stop, manager):
            manager.interrupt_current_processing(False)
            raise InterruptedError("Prompt rewrite interrupted")
        key = str(path)
        clip = self.prompt_encoders.get(key)
        if clip is None:
            _ram_preflight({"files":{"prompt_encoder":str(path)}},self.prof)
            name = self.backend.register("text_encoders", path)
            clip = self.backend.nodes.CLIPLoader().load_clip(name, "stable_diffusion", self.prof.get("clip_device", "default"))[0]
            _prepare_text_generator(clip, self.backend.model_management)
            _maybe_full_load_text_generator(clip, self.backend.model_management)
            self.prompt_encoders[key] = clip
        emit("status", status="rewriting prompt")
        max_length, sampling, thinking = _enhancer_options(command)
        done = threading.Event()
        started = time.monotonic()

        def monitor():
            while not done.wait(1.0):
                if _check_rewrite_stop(stop, manager):
                    return
                emit("status", status=f"rewriting prompt — {int(time.monotonic() - started)}s")

        thread = threading.Thread(target=monitor, name="qwen-prompt-rewrite-monitor", daemon=True)
        thread.start()
        try:
            output = _node_values(self.backend.text_generate.execute(
                clip, prompt, max_length, sampling, thinking=thinking,
                use_default_template=True, mtp="auto", system_prompt=system,
            ))[0]
        except BaseException as exc:
            if isinstance(exc, manager.InterruptProcessingException):
                raise InterruptedError("Prompt rewrite interrupted") from exc
            raise
        finally:
            done.set()
            thread.join(timeout=2.0)
            manager.interrupt_current_processing(False)
            if self.prof.get("_offload", True):
                import torch
                _idle_offload(manager, torch)
        return _rewrite_data(output, prompt)

    def _control_model(self, model, command, torch, refs, mask):
        control_path = command.get("control_model")
        if not control_path:
            return model
        path = _path(control_path, "control model")
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns)
        patch = self.control_patches.get(identity)
        if patch is None:
            _ram_preflight({"files":{"control":str(path)}},self.prof)
            name = self.backend.register("model_patches", path)
            loaded = self.backend.patch_loader().load_model_patch(name)[0]
            _validate_qwen21_control(loaded)
            self.control_patches[identity] = loaded
            patch = loaded
        control = _image(command.get("control_image"), torch)
        source = refs[0] if refs else None
        strength = float(command.get("control_strength", 1.0))
        start = float(command.get("control_start", 0.0))
        end = float(command.get("control_end", 1.0))
        if not math.isfinite(strength) or not -10 <= strength <= 10:
            raise ValueError("Control strength must be finite and between -10 and 10")
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Control start/end must be finite")
        if not 0 <= start < end <= 1:
            raise ValueError("Control start/end must satisfy 0 <= start < end <= 1")
        return self.backend.control_apply().diffsynth_controlnet(
            model, patch, self.vae, image=control, strength=strength,
            inpaint_image=source if mask is not None else None, mask=mask,
            start_percent=start, end_percent=end,
        )[0]

    def generate(self, command):
        _reject_unavailable(command)
        import torch
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        stop = Path(command["stop"])
        rewrite = self._rewrite(command, command.get("prompt") or "")
        prompt = rewrite["prompt"]
        refs = [_image(path, torch) for path in command.get("images", [])]
        refs = [image for image in refs if image is not None]
        mask = _mask(command.get("mask"), torch)
        if mask is not None and not refs:
            raise ValueError("Masked editing requires a source image")
        model = self._lora_model(command.get("adapters"))
        model = self._cache_model(model, bool(command.get("use_kv_cache", True)))
        model = self._reference_model(model, command.get("reference_priorities", ""), len(refs))
        model = self._control_model(model, command, torch, refs, mask)
        sampler = str(command.get("comfy_sampler") or command.get("sampler_name") or command.get("sampler") or "euler").lower()
        scheduler = str(command.get("comfy_scheduler") or command.get("scheduler") or "simple").lower()
        if sampler == "res_2s":
            self.backend.enable_res2s()
        import comfy.samplers
        if sampler not in comfy.samplers.KSAMPLER_NAMES:
            raise ValueError("Unsupported Comfy sampler: " + sampler)
        if scheduler not in comfy.samplers.SCHEDULER_NAMES:
            raise ValueError("Unsupported Comfy scheduler: " + scheduler)

        images = {f"image_{index}": value for index, value in enumerate(refs, 1)}
        encoded = self._conditioning(command, prompt, command.get("negative_prompt") or "", images)
        positive, negative, latent = encoded
        # Native Forge width/height are authoritative. Follow-source already
        # puts the source dimensions there; manual overrides remain possible.
        width = max(32, int(command.get("width", 1024)) // 32 * 32)
        height = max(32, int(command.get("height", 1024)) // 32 * 32)
        if rewrite["ratio_follow"]:
            width, height = _reference_size(refs, rewrite["ratio_follow"])
        elif rewrite["wh_ratio"] is not None:
            width, height = _ratio_size(width, height, rewrite["wh_ratio"])
        latent = {"samples": torch.zeros((1, 64, height // 16, width // 16),
                                          device=self.backend.model_management.intermediate_device())}
        latent_image = self.backend.sample.fix_empty_latent_channels(model, latent["samples"])
        noise = self.backend.sample.prepare_noise(latent_image, int(command.get("seed", 0)))
        total = int(command.get("num_inference_steps", 25))
        cfg = float(command.get("true_cfg_scale", 1.0))
        if not 1 <= total <= 1000:
            raise ValueError("Comfy inference steps must be between 1 and 1000")
        if not math.isfinite(cfg) or not 0 <= cfg <= 100:
            raise ValueError("CFG must be finite and between 0 and 100")
        sigmas = _sigma_tensor(command.get("sigmas"), total, torch)
        preview_every = max(0, int(command.get("preview_every", 0) or 0))
        preview_path = command.get("preview")
        preview_size = max(64, min(512, int(command.get("preview_size", 256) or 256)))
        previewer = None
        preview_failed = False
        last_preview = 0.0
        if preview_every > 0 and preview_path:
            try:
                previewer = _native_previewer(model, importlib.import_module("latent_preview"))
            except Exception as exc:
                preview_failed = True
                emit("status", status="Native preview unavailable: " + str(exc))

        completed = [0]
        heartbeat_stop = threading.Event()
        heartbeat_started = time.monotonic()

        def heartbeat():
            while not heartbeat_stop.wait(1.0):
                emit("status", status=_sampling_status(completed[0], total, heartbeat_started))

        heartbeat_thread = threading.Thread(target=heartbeat, name="qwen-comfy-heartbeat", daemon=True)

        def progress(step, _x0, _x, total_steps):
            nonlocal last_preview, preview_failed
            if stop.exists():
                raise InterruptedError("Generation interrupted")
            completed[0] = max(completed[0], int(step) + 1)
            emit("progress", step=int(step), total=int(total_steps), phase="base")
            now = time.monotonic()
            if (previewer is None or preview_failed or preview_every <= 0 or
                    (int(step) + 1) % preview_every or now - last_preview < 1.0):
                return
            try:
                _write_preview(previewer, _x0, preview_path, preview_size)
                last_preview = now
                emit("preview", step=int(step), timestep=None,
                     path=str(preview_path), final=False)
            except InterruptedError:
                raise
            except Exception as exc:
                preview_failed = True
                emit("status", status="Native preview failed; generation continues: " + str(exc))

        heartbeat_thread.start()
        try:
            samples = self.backend.sample.sample(
                model, noise, total, cfg,
                sampler, scheduler, positive, negative, latent_image, denoise=1.0,
                noise_mask=mask, sigmas=sigmas, callback=progress, disable_pbar=True,
                seed=int(command.get("seed", 0)),
            )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2.0)
        decoded = self.vae.decode(samples)
        decoded = _composite(decoded, refs[0] if refs else None, mask, torch)
        _save(decoded, command["output"])
        if preview_every > 0:
            emit("preview", step=max(0, total - 1), timestep=None,
                 path=str(command["output"]), final=True)
        if self.prof.get("_offload", True):
            _clear_qwen_prefix_cache(model)
            del decoded, samples, noise, latent_image, latent, encoded, positive, negative, model
            _idle_offload(self.backend.model_management, torch)
        emit("result", path=str(command["output"]), backend="comfy",
             sampler=sampler, scheduler=scheduler, prompt=prompt,
             rewritten_prompt=prompt if command.get("rewrite_prompt") else None,
             width=width, height=height, rewritten=bool(command.get("rewrite_prompt")),
             cuda_memory={
                 "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0,
                 "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20) if torch.cuda.is_available() else 0,
                 "idle_allocated_mib": round(torch.cuda.memory_allocated() / 2**20) if torch.cuda.is_available() else 0,
                 "idle_reserved_mib": round(torch.cuda.memory_reserved() / 2**20) if torch.cuda.is_available() else 0,
                 "scope": "Qwen Comfy worker PyTorch allocator; excludes driver and other applications",
             })

    def release(self):
        self.model = self.clip = self.vae = None
        self.prompt_encoders = {}
        self.control_patches = {}
        with contextlib.suppress(Exception):
            self.backend.model_management.unload_all_models()
        with contextlib.suppress(Exception):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", required=True)
    parser.add_argument("--forge-root", required=False)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    log = open(args.log, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    session = None
    emit("ready")
    for line in sys.stdin:
        try:
            command = json.loads(line)
            op = command.get("op")
            if op == "close":
                if session:
                    session.release()
                emit("closed")
                return 0
            if op == "release":
                if session:
                    session.release()
                    session = None
                emit("released")
                continue
            if op == "init":
                prof = command.get("prof") or {}
                prof = dict(prof, _offload=bool(command.get("offload", True)))
                comfy_root = prof.get("comfy_root")
                _ram_preflight(command['bundle'],prof)
                backend = Backend(comfy_root, prof.get("res4lyf"), prof.get("comfy_deps"), prof)
                session = Session(backend, command["bundle"], prof)
                emit("initialized", backend="comfy")
                continue
            if op == "generate":
                if session is None:
                    raise RuntimeError("Comfy worker is not initialized")
                session.generate(command)
                continue
            raise ValueError("Unknown Comfy worker operation: " + str(op))
        except InterruptedError as exc:
            emit("interrupted", error=str(exc))
        except Exception as exc:
            emit("error", error=str(exc))
            traceback.print_exc(file=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
