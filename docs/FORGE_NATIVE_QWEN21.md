# October 4, 2026 — Forge native compatibility

Reviewed [Forge PR #1512](https://github.com/Haoming02/sd-webui-forge-classic/pull/1512), open and unmerged when checked. It introduces native Qwen 2.1 txt2img/RGBA/edit support, a `qwen21` preset, component loading, LoRA mapping and alpha-aware saving. This extension does not install the PR or modify Forge core.

Reviewed head: `c4495ae3e901b5d5f14f71e149a98fba23a56025`, October 4, 2026.

## Choose your engine

- **Project Invisible:** select `qwen-image-2.1` and an extension checkpoint. Existing worker, settings and optional enhancements remain available.
- **Native Forge after installing its support:** select `qwen21` and a regular checkpoint. The extension passes generation to Forge, hides its panel and stops applying its LoRA-card restrictions.
- Extension-owned checkpoint aliases remain extension-owned even under another preset. Select a regular Forge checkpoint to use the native engine.
- API `override_settings.forge_preset` follows the same routing rule. On older Forge versions without the native engine, routing remains unchanged.

The native and extension presets have independent defaults. Extension-only enhancements are not automatically transferred to the native engine. No additional environment or model download is needed for this compatibility patch.

## Validation

255 tests passed, one skipped (256 total). Added native routing, extension-marker ownership, old-Forge fallback, API overrides and VAE strip checks. Both real Gradio panels still build with all 35 saved control positions intact. Native GPU inference and browser interaction on the unmerged PR remain untested; this is a compatibility safeguard, not a claim that every future Forge revision is supported.

The installed extension also passed the full 256-test suite and source-integrity verification. The modified worker passed real GPU generation checks for one-step scheduling, neutral pixel identity, uncached weighting, two references, LanPaint, Spectrum, cancellation and recovery. Those 256-pixel generation checks exercise the modified worker; the separate large-feature-map test below exercises the strip threshold itself.

## Adapted memory improvement

The isolated worker now processes known VAE spatial upsamplers in horizontal strips when the projected intermediate exceeds 16 million elements. A one-row overlap preserves convolution boundaries. Temporal caches, spatial attention, normalization, model weights and module names remain untouched; small inputs and unknown architectures follow the original path. Full-image decoding stays enabled, avoiding VAE tile seams. No new checkbox is needed.

On RTX 5090, one real Diffusers VAE upsampler with a 513 x 257 feature map produced identical outputs in FP32, FP16 and BF16. Peak allocated memory for this isolated operation fell from 273.67 to 209.54 MiB in FP32 and 136.83 to 104.77 MiB in FP16/BF16 (about 23%). This does not imply a 23% reduction in total generation VRAM or faster generation; convolution kernel rounding can differ on other hardware. Full transformer fusion and native loader internals were not transplanted because their contracts differ from the worker.

Thanks to Haoming02 and the Forge community. The adaptation's AGPL license/provenance is retained in `lib/vendor/forge/`; existing credits and model licenses remain unchanged.
