"""Audited presets derived from user-supplied ComfyUI workflow JSON.

The JSON is treated as data only.  These recipes change visible defaults; the
caller remains responsible for applying them through Gradio updates so users
can edit every value before Generate.
"""
from __future__ import annotations

import json
from copy import deepcopy
from types import MappingProxyType
from typing import Any

__all__ = ["CATALOG", "UNSUPPORTED", "choices", "settings", "audit", "provenance"]


_COMMON = dict(native_cache=True, outpaint=False)


def _recipe(**values: Any) -> dict[str, Any]:
    return {**_COMMON, **values}


_RECIPES = {
    "Custom": {},
    "ITLText25": _recipe(steps=25, cfg=1.0, task="t2i", turbo=False,
                      texture_vae=False, lanpaint=False, pose="off", follow_source=False,
                      width=992, height=544),
    "UltraTextAdapted50": _recipe(steps=50, cfg=1.0, task="t2i", turbo=False,
                      texture_vae=True, lanpaint=False, pose="off", follow_source=False,
                      width=1920, height=1088,
                      adaptation_note="Uses supported Euler/Simple instead of res_2s/Beta"),
    "UltraTurboTextAdapted6": _recipe(steps=6, cfg=1.0, task="t2i", turbo=True,
                      texture_vae=True, lanpaint=False, pose="off", follow_source=False,
                      width=1920, height=1088,
                      adaptation_note="Uses base model plus Viggle LoRA; not the merged turbo checkpoint"),
    "ITLEdit25": _recipe(steps=25, cfg=1.0, task="edit", operation="edit", turbo=False,
                      texture_vae=False, lanpaint=False, pose="off", follow_source=True),
    "ITLPose25": _recipe(steps=25, cfg=1.0, task="edit", operation="pose", turbo=False,
                      texture_vae=False, lanpaint=False, pose="OpenPose", follow_source=True),
    "UltraEdit25": _recipe(steps=25, cfg=1.0, task="edit", operation="edit", turbo=False,
                        texture_vae=True, lanpaint=False, pose="off", follow_source=True),
    "UltraInpaint25": _recipe(steps=25, cfg=1.0, task="edit", operation="inpaint", turbo=False,
                           texture_vae=True, lanpaint=True, pose="off", follow_source=True),
    "UltraOutpaint25": _recipe(steps=25, cfg=1.0, task="edit", operation="outpaint", turbo=False,
                            texture_vae=True, lanpaint=True, pose="off", follow_source=True,
                            outpaint=True,
                            pads=(256, 256, 256, 256)),
    "UltraTurboEdit6": _recipe(steps=6, cfg=1.0, task="edit", operation="edit", turbo=True,
                            texture_vae=True, lanpaint=False, pose="off", follow_source=True),
    "UltraTurboInpaint6": _recipe(steps=6, cfg=1.0, task="edit", operation="inpaint", turbo=True,
                               texture_vae=True, lanpaint=True, pose="off", follow_source=True),
    "UltraTurboOutpaint6": _recipe(steps=6, cfg=1.0, task="edit", operation="outpaint", turbo=True,
                                texture_vae=True, lanpaint=True, pose="off", follow_source=True,
                                outpaint=True,
                                pads=(256, 256, 256, 256)),
}

CATALOG = MappingProxyType(_RECIPES)

# These use the native Qwen 2.1 Comfy nodes in an isolated process.
_RECIPES.update({
    'UltraMergedTurboText6':_recipe(steps=6,cfg=1.,task='t2i',turbo=False,merged_turbo=True,texture_vae=True,
        lanpaint=False,pose='off',backend='comfy',sampler='euler',scheduler='simple',width=1920,height=1088,follow_source=False),
    'UltraMergedTurboEdit6':_recipe(steps=6,cfg=1.,task='edit',turbo=False,merged_turbo=True,texture_vae=True,
        lanpaint=False,pose='off',backend='comfy',sampler='euler',scheduler='simple',follow_source=True),
    'UltraMergedTurboInpaint6':_recipe(steps=6,cfg=1.,task='edit',turbo=False,merged_turbo=True,texture_vae=True,
        lanpaint=False,pose='off',backend='comfy',sampler='euler',scheduler='simple',control=True,follow_source=True),
    'UltraMergedTurboOutpaint6':_recipe(steps=6,cfg=1.,task='edit',turbo=False,merged_turbo=True,texture_vae=True,
        lanpaint=False,pose='off',backend='comfy',sampler='euler',scheduler='simple',control=True,follow_source=True,
        outpaint=True,pads=(256,256,256,256)),
    'UltraTextFull50':_recipe(steps=50,cfg=1.,task='t2i',turbo=False,texture_vae=True,lanpaint=False,
        pose='off',backend='comfy',sampler='res_2s',scheduler='beta',width=1920,height=1088,follow_source=False),
    'UltraEditControl25':_recipe(steps=25,cfg=1.,task='edit',turbo=False,texture_vae=True,lanpaint=False,
        pose='off',backend='comfy',sampler='euler',scheduler='simple',control=True,follow_source=True),
    'UltraInpaintControl25':_recipe(steps=25,cfg=1.,task='edit',operation='inpaint',turbo=False,texture_vae=True,lanpaint=False,
        pose='off',backend='comfy',sampler='euler',scheduler='simple',control=True,follow_source=True),
    'UltraOutpaintControl25':_recipe(steps=25,cfg=1.,task='edit',operation='outpaint',turbo=False,texture_vae=True,lanpaint=False,
        pose='off',backend='comfy',sampler='euler',scheduler='simple',control=True,follow_source=True,outpaint=True,pads=(256,256,256,256)),
    'UltraTurboControl6':_recipe(steps=6,cfg=1.,task='edit',turbo=True,texture_vae=True,lanpaint=False,
        pose='off',backend='comfy',sampler='euler',scheduler='simple',control=True,follow_source=True,
        adaptation_note='Uses the validated base + Viggle adapter, not a merged checkpoint.'),
})

# Deliberately not recipes: the extension cannot honestly reproduce them yet.
UNSUPPORTED = MappingProxyType({
    "UltraText": "choose UltraTextFull50 with the Full workflows backend and RES4LYF installed",
    "MergedTurbo": "choose UltraMergedTurboText6 or UltraMergedTurboEdit6; requires the actual merged checkpoint",
    "ControlNetUnion": "choose an Ultra Control preset with the Full workflows engine and its dedicated Union weights",
})


def choices(is_img2img: bool) -> list[str]:
    """Return recipes valid for the active native Forge tab."""
    wanted = "image" if is_img2img else "text"
    result = ["Custom"]
    for name, recipe in _RECIPES.items():
        if name == "Custom":
            continue
        task = recipe["task"]
        if (wanted == "text" and task == "t2i") or (wanted == "image" and task != "t2i"):
            result.append(name)
    return result


def settings(name: str, is_img2img: bool) -> dict[str, Any]:
    """Return a fresh recipe mapping after checking native-tab compatibility."""
    if name not in _RECIPES:
        if name in UNSUPPORTED:
            raise ValueError(f"{name} is unsupported: {UNSUPPORTED[name]}")
        raise ValueError(f"Unknown Qwen workflow recipe: {name}")
    recipe = _RECIPES[name]
    if not recipe:
        return {}
    wrong_tab = (recipe["task"] == "t2i") == bool(is_img2img)
    if wrong_tab:
        tab = "img2img" if is_img2img else "txt2img"
        raise ValueError(f"{name} is not available on the {tab} tab")
    return deepcopy(recipe)


def audit(data: Any, *, limit: int = 2_000_000) -> dict[str, Any]:
    """Parse workflow JSON as inert data and summarize provenance/risk markers."""
    if isinstance(data, bytes):
        if len(data) > limit:
            raise ValueError("Workflow JSON exceeds the audit limit")
        data = data.decode("utf-8")
    if isinstance(data, str):
        if len(data.encode("utf-8")) > limit:
            raise ValueError("Workflow JSON exceeds the audit limit")
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid workflow JSON: {exc.msg}") from exc
    if not isinstance(data, (dict, list)):
        raise TypeError("Workflow audit expects JSON object, array, or text")

    node_types: set[str] = set()
    modes: set[str] = set()
    references: set[str] = set()
    samplers: set[str] = set()
    text_fragments: list[str] = []
    node_count = 0

    def walk(value: Any, key: str = "", depth: int = 0) -> None:
        nonlocal node_count
        if depth > 80 or node_count > 50_000:
            raise ValueError("Workflow JSON is too deeply nested or complex")
        if isinstance(value, dict):
            node_type = value.get("class_type", value.get("type"))
            if isinstance(node_type, str):
                node_types.add(node_type)
                node_count += 1
            mode = value.get("mode")
            if isinstance(mode, (str, int, float, bool)):
                modes.add(str(mode))
            for child_key, child in value.items():
                walk(child, str(child_key).lower(), depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, key, depth + 1)
        elif isinstance(value, str):
            lowered = value.lower()
            text_fragments.append(lowered)
            if any(word in key for word in ("model", "lora", "vae", "image", "control", "file", "ckpt")) or lowered.endswith(('.safetensors','.gguf','.ckpt','.onnx','.pth')):
                references.add(value)
            if "sampler" in key or lowered in {"euler", "res_2s", "beta", "simple"}:
                samplers.add(value)

    walk(data)
    haystack = " ".join(text_fragments + [x.lower() for x in node_types])
    flags = {
        "controlnet_union": "controlnet" in haystack or "controlnet" in " ".join(x.lower() for x in references),
        "openpose": "openpose" in haystack or "dwpose" in haystack,
        "res_2s_or_beta": "res_2s" in haystack or "beta" in haystack,
        "merged_turbo": "turbo" in haystack and any(
            "turbo" in item.lower() and any(ext in item.lower() for ext in (".safetensors", ".gguf", ".ckpt"))
            for item in references
        ),
    }
    return {
        "node_count": node_count,
        "node_types": sorted(node_types),
        "modes": sorted(modes),
        "references": sorted(references),
        "samplers": sorted(samplers),
        "unsupported": flags,
    }


def provenance() -> dict[str, Any]:
    """Machine-readable statement of what was mapped and intentionally omitted."""
    return {
        "source": "five user-supplied Qwen 2.1 ComfyUI workflow JSON files",
        "mapped": list(_RECIPES),
        "unsupported": dict(UNSUPPORTED),
        "execution": "JSON is parsed as inert data only",
    }
