"""Dependency-light, read-only readiness checks for Qwen workflow features.

Inspired by AiWithYou/aikimi-forge-neo readiness reporting at commit
1ebd2454ba424fb08fe666c5e494d8eb189fe935. No source code is copied.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path


AIKIMI_SOURCE = {
    "repository": "https://github.com/AiWithYou/aikimi-forge-neo",
    "commit": "1ebd2454ba424fb08fe666c5e494d8eb189fe935",
    "license": "AGPL-3.0-or-later; bundled components keep their own licenses",
}
COMFY_REVISION = "7a5dad695fe1cae25efcb2550530fb20ef68da3d"
POSE_FILES = {
    "OpenPose": ("body_pose_model.pth", "hand_pose_model.pth", "facenet.pth"),
    "DWPose": ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"),
}


def _header(path):
    try:
        with Path(path).open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if not 2 <= size <= 64 * 1024 * 1024:
                return None
            return json.loads(handle.read(size))
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        return None


def _blocks(keys, prefix):
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.")
    return {int(match.group(1)) for key in keys if (match := pattern.match(key))}


def component_kind(path):
    """Return the verified Qwen component kind without loading tensor data."""
    header = _header(path)
    if not header:
        return "invalid"
    keys = set(header) - {"__metadata__"}
    metadata = header.get("__metadata__", {})
    blocks = _blocks(keys, "transformer_blocks.")
    if blocks == set(range(32)) and any(".attn.to_q." in key for key in keys) and any(".img_mlp." in key for key in keys):
        return "qwen21_dit"
    control = _blocks(keys, "control_blocks.")
    if (control == set(range(16)) and "control_img_in.weight" in keys
            and "control_blocks.0.img_mlp.out.weight" in keys):
        return "qwen21_union"
    layers = _blocks(keys, "model.layers.")
    if layers == set(range(36)) and "model.visual.patch_embed.proj.weight" in keys:
        return "qwen3vl_8b"
    if metadata.get("modelspec.architecture") == "qwen_image_2.1_vae" and "decoder.conv1.weight" in keys:
        return "qwen21_vae"
    if "lm_head.weight" in keys and any(key.startswith("model.language_model.") for key in keys):
        return "prompt_encoder"
    return "unknown"


def _files(root, folders):
    found = []
    for folder in folders:
        base = root / folder
        if base.is_dir():
            found.extend(path.resolve() for path in base.rglob("*.safetensors") if path.is_file())
    return sorted(set(found), key=lambda path: str(path).lower())


def _pose_matches(root, names):
    matches = {}
    for base_name in ("ControlNetPreprocessor", "annotator", "controlnet", "ControlNet"):
        base = root / base_name
        if not base.is_dir():
            continue
        for name in names:
            if name in matches:
                continue
            match = next(base.rglob(name), None)
            if match and match.is_file():
                matches[name] = str(match.resolve())
    return matches


def inspect_readiness(models_root, extension_root, *, backend="diffusers",
                      comfy_root=None, comfy_deps=None, sampler="euler", pose="off",
                      control_enabled=False, merged_turbo=False,
                      rewrite_prompt=False, selected=None):
    """Describe readiness using local files only; never imports models or downloads."""
    models_root = Path(models_root).expanduser().resolve()
    extension_root = Path(extension_root).expanduser().resolve()
    comfy_root = Path(comfy_root).expanduser().resolve() if comfy_root else extension_root / "_comfy"
    comfy_deps = Path(comfy_deps).expanduser().resolve() if comfy_deps else extension_root / "_deps_comfy"
    selected = {key: Path(value).expanduser().resolve() for key, value in (selected or {}).items() if value}
    candidates = _files(models_root, ("Stable-diffusion", "diffusion_models", "Qwen-Image-2.1",
                                      "text_encoder", "VAE", "model_patches", "ControlNet"))
    kinds = {str(path): component_kind(path) for path in candidates}
    compatible = {kind: [path for path, actual in kinds.items() if actual == kind]
                  for kind in ("qwen21_dit", "qwen3vl_8b", "qwen21_vae", "qwen21_union", "prompt_encoder")}
    checks = []

    def add(code, ok, message, severity="error"):
        checks.append({"code": code, "ok": bool(ok), "severity": "ok" if ok else severity, "message": message})

    required = (("transformer", "qwen21_dit", "Qwen 2.1 image model"),
                ("text_encoder", "qwen3vl_8b", "Qwen3-VL 8B text/vision encoder"),
                ("vae", "qwen21_vae", "Qwen 2.1 VAE"))
    for key, kind, label in required:
        path = selected.get(key)
        if path:
            ok = path.is_file() and component_kind(path) == kind
            add(key, ok, f"{label}: {'ready' if ok else 'selected file is incompatible'}")
        else:
            ok = bool(compatible[kind])
            add(key, ok, f"{label}: {'found locally' if ok else 'missing'}")

    if backend not in ("diffusers", "comfy"):
        add("backend", False, "Generation engine must be diffusers or comfy")
    elif backend == "comfy":
        code_ok = all((comfy_root / path).is_file() for path in
                      ("comfy_extras/nodes_qwen.py", "comfy_extras/nodes_model_patch.py"))
        add("comfy_code", code_ok, "Isolated Comfy Qwen code: " + ("ready" if code_ok else "missing"))
        marker = comfy_root / ".pi_revision"
        revision_ok = marker.is_file() and marker.read_text(encoding="utf-8").strip() == COMFY_REVISION
        add("comfy_revision", revision_ok, "Pinned Comfy revision: " + ("verified" if revision_ok else "missing or different"), "warning")
        deps_ok = (comfy_deps / "comfy_aimdo").is_dir()
        add("comfy_dependencies", deps_ok, "Isolated Comfy support packages: " + ("ready" if deps_ok else "missing"))
        if sampler == "res_2s":
            res_ok = (comfy_root / "custom_nodes/RES4LYF/__init__.py").is_file()
            add("res4lyf", res_ok, "RES4LYF sampler code: " + ("ready" if res_ok else "missing"))

    if control_enabled:
        path = selected.get("control")
        ok = path.is_file() and component_kind(path) == "qwen21_union" if path else bool(compatible["qwen21_union"])
        add("union", ok, "Qwen 2.1 Union model: " + ("ready" if ok else "missing or incompatible"))
    if merged_turbo:
        merged = selected.get("merged")
        if merged:
            ok = merged.is_file() and component_kind(merged) == "qwen21_dit" and "lora" not in merged.name.lower()
        else:
            ok = any("viggle" in Path(path).name.lower() and "lora" not in Path(path).name.lower()
                     for path in compatible["qwen21_dit"])
        add("merged_turbo", ok, "Merged Viggle Turbo image model: " + ("ready" if ok else "missing"))
    if rewrite_prompt:
        path = selected.get("prompt_encoder")
        ok = path.is_file() and component_kind(path) == "prompt_encoder" if path else bool(compatible["prompt_encoder"])
        add("prompt_encoder", ok, "Separate prompt encoder: " + ("ready" if ok else "missing or incompatible"))

    if pose in POSE_FILES:
        found = _pose_matches(models_root, POSE_FILES[pose])
        missing = [name for name in POSE_FILES[pose] if name not in found]
        add("pose_assets", not missing,
            f"{pose} local assets: " + ("ready" if not missing else "missing " + ", ".join(missing)))
    elif pose not in ("off", "Uploaded pose map"):
        add("pose_assets", False, "Unknown pose mode")

    vae_path = selected.get("vae") or (Path(compatible["qwen21_vae"][0]) if compatible["qwen21_vae"] else None)
    precision = "unknown"
    if vae_path:
        header = _header(vae_path) or {}
        dtype = (header.get("decoder.conv1.weight") or {}).get("dtype", "unknown")
        precision = str(dtype).lower()
    add("decoder_precision", precision in ("bf16", "f32", "float32"),
        f"VAE decoder weights: {precision}; BF16 is the verified baseline", "warning")

    errors = [item for item in checks if not item["ok"] and item["severity"] == "error"]
    return {
        "ready": not errors,
        "summary": "Ready" if not errors else f"Needs attention: {len(errors)} required item(s)",
        "checks": checks,
        "compatible_paths": compatible,
        "downloads_started": False,
        "source": dict(AIKIMI_SOURCE),
    }
