# Update history

Newest updates appear first. Dates use YYYY-MM-DD. These entries describe published changes, not guarantees for every device or adapter.

## 2026-09-24 (2)

### Changed

- **Control panel reorganized** to match the layout used by Project Invisible — Ideogram 4: the top of the accordion now shows only a quick-start card, the Output choice and the Quality choice. Everything rarely touched (VRAM profile, maximum size, GPU-memory saving, Spectrum, community LoRAs, Speed boost, prompt helper, model downloads) moved into labeled **Advanced tabs**: Performance, Speed boost, Prompt helper and Model Setup. Every dropdown, checkbox, radio and slider now has a one-line plain-language explanation under it, buttons are compact, and status messages are styled consistently. No controls were removed and no behavior changed - only placement and wording.

## 2026-09-24

### Fixed

- **Speed LoRA dropdown placeholder no longer crashes the UI:** pressing the download button without picking a LoRA (the dropdown still showing `(none)`) now shows a clear "pick a speed LoRA first" message instead of a raw `ValueError` traceback in the Forge console. The check lives inside `download_featured` itself, so every caller is protected at the source.

- **Silent speed-loss fixed:** when the Speed boost box is ticked but no downloaded speed LoRA is actually selected, the extension previously ran the full 40-60 step schedule without saying anything. It now prints a clear console warning naming the control so the few-step schedule is never silently skipped.

- **Device mismatch during quantized offloading fully resolved (Issue #2 / Issue #4):** the reporter's logs showed two remaining crash paths — Forge's packed-Embedding forward ran its dequantize kernel while the packed weights had been offloaded to the CPU and the token indices lived on the GPU, and diffusers' offload hook framework silently bypassed the previous forward patch by replacing `forward` with its own wrapper. The device guard is now a registered forward pre-hook, which runs inside PyTorch's call machinery before any forward replacement and cannot be bypassed. Quantized loads also no longer use group offload, whose parameter swapping is incompatible with packed subclasses. Verified against both reporter worker.logs (RTX 4070 Ti, CUDA 13 torch build).

- **All packed quantization formats supported for DiT, text encoder and VAE inputs:** files using `int8_tensorwise` (with or without ConvRot), `asym_w4a8_int8` (W4A8), `float8_e4m3fn`, `float8_e5m2`, `mxfp8`, `nvfp4` and `convrot_w4a4` now load on every NVIDIA and AMD card. Capable NVIDIA cards run the fast packed kernels; elsewhere the extension unpacks the weights to plain BF16 in system RAM automatically. CPU round-trip accuracy was verified for every format.

- **Adapter coverage widened:** factorized LoKr (`lokr_w1_a/b`), LoHa (Hadamard composition, plain and factorized) and full-difference (`.diff`) adapters are now applied alongside ordinary LoRA and plain LoKr, with fused `gate_up` splitting kept for all kinds. Malformed adapters still fail atomically without touching the model.

- **Duplicate extension copies detected (Issue #3):** installing via Forge's "Install from URL" and via a manual ZIP at the same time leaves two working copies that fight over the preset. The extension prints a startup warning listing both folders, and the README now documents both install paths with a never-both warning.

- **Renamed community DiT files are now recognized (Issue #1):** the extension previously identified model files only by exact filenames. A Qwen-Image-2.1 DiT downloaded from a community mirror (for example Civitai, saved as `qwenImage21INT8INT4_int8.safetensors`) was not recognized, so Forge tried to load it directly and reported "Failed to recognize diffusion model". DiT files are now identified by their internal tensor structure as well as by filename, in both the model scan and checkpoint selection. Adapter (LoRA/LoKr) files and foreign architectures (Flux/SD layouts) are still rejected.

### Verification

- **110 Qwen automated tests passed** (the previous 109 plus a new regression test that reproduces the Issue #4 failure exactly: a diffusers-style forward replacement installed after hardening can no longer bypass the packed-Embedding device guard).

## 2026-09-23

### Fixed

- **Empty negative prompt:** CFG above 1 now falls back to effective CFG 1 with a terminal notice instead of stopping generation. Saved metadata records the effective value. A real negative prompt preserves the selected native CFG.
- **Quantized LoRA loading:** ordinary supported LoRA adapters use additive hooks instead of incompatible PEFT layer replacement.
- **Duplicate CFG control:** removed the extension slider; Forge's native CFG Scale controls generation.
- **Failed generation cleanup:** release a failed worker pipeline before another attempt.
- **Portable installation:** removed the author's drive path from the release configuration and added a Forge-root fallback.

### Repository and documentation

- Published extension source at the repository root instead of distributing only a RAR archive. The previous archive remains in Git history.
- Added beginner installation instructions, complete-terminal-error issue guidance, and honest testing limitations.
- Added `verify_integrity.py`, source checksums, and [update/backup instructions](UPDATE_SAFETY.md).
- Added thanks to the r/SECourses community alongside u/malcolmrey and r/malcolmrey.

### Verification

- **87 Qwen automated tests passed**, including native CFG behavior and quantized adapter regression checks.
- Source integrity checks passed for the published files.
- Text-to-image was previously confirmed by the author. A fresh real generation with the latest fixes remains unverified; other modes and optional features remain experimental or unconfirmed.

### How to apply

1. Let the current generation finish, then stop Forge.
2. Back up the extension outside the Forge folder.
3. Update this extension's files from the repository.
4. Restart Forge and refresh the browser with **Ctrl+F5**.
5. Try a small text-to-image generation first.

### Separate companion fix

The local Model Autolink extension received a checkpoint-list snapshot fix for a model-switching error; its 9 tests passed. **That companion patch is not included in this Qwen repository.** The Qwen test count is 87, not 96.

### Compatibility limits

No Forge core files were changed. Extension-local dependencies and a separate worker reduce interference, but Forge's Python, PyTorch and UI hooks remain shared dependencies. Future Forge compatibility cannot be guaranteed.
