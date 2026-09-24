# Update history

Newest updates appear first. Dates use YYYY-MM-DD. These entries describe published changes, not guarantees for every device or adapter.

## 2026-09-24

### Fixed

- **Device-mismatch crash during offloading (Issue #2):** quantized models use packed weight formats. The standard PyTorch routine for moving a module between CPU and GPU reassigns `param.data`, which can strip the packed-format wrapper and leave the word-lookup (embedding) table on the wrong device, producing "Expected all tensors to be on the same device" on the first generation. The extension now attaches a subclass-safe move routine to every Embedding of the loaded components before offloading engages, mirroring Forge's own quantized `_apply` behavior. No Forge core files are changed.

- **Renamed community DiT files are now recognized (Issue #1):** the extension previously identified model files only by exact filenames. A Qwen-Image-2.1 DiT downloaded from a community mirror (for example Civitai, saved as `qwenImage21INT8INT4_int8.safetensors`) was not recognized, so Forge tried to load it directly and reported "Failed to recognize diffusion model". DiT files are now identified by their internal tensor structure as well as by filename, in both the model scan and checkpoint selection. Adapter (LoRA/LoKr) files and foreign architectures (Flux/SD layouts) are still rejected.

### Verification

- **95 Qwen automated tests passed** (the previous 93 plus two new regression tests for the packed-Embedding device move).

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
