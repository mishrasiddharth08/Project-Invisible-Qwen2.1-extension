# Validation — September 23, 2026

## Current release checks

- 87 automated tests passed on the release checkout using Forge's existing Python.
- Regression checks cover additive LoRA output/scaling on a custom linear module, fused gate/up mapping, invalid-target rollback, adapter removal and native CFG selection.
- Source manifest verified. No model weights, installed dependency folders, runtime logs or generated images are included in the public source tree.
- The earlier live patch was restarted and Forge's HTTP interface responded. The duplicate extension CFG slider was removed.

## Public testing status

The author confirmed text-to-image working. Other modes and optional features remain experimental or unconfirmed for this public release. Automated tests do not prove all real LoRA/LoKr files, hardware configurations or future Forge versions work. A new real generation with the latest LoRA patch has not been verified.

## Quality and performance limits

VAE tiling is disabled in the current implementation. DeGrid cleanup is optional. Spectrum is experimental and off by default. Preview timing depends on model step and decode time; one-second updates and identical output quality are not guaranteed. No universal speed or memory guarantee is made.

## Repeat the checks

From this extension folder, use Forge's existing Python to run:

```text
python -m unittest discover -s tests -q
python verify_integrity.py
```

Use the Python executable inside your Forge installation if `python` points somewhere else. Integrity verification compares source files only; it is not a compatibility or security certification. Read UPDATE_SAFETY.md before updating Forge.

## Dedicated Qwen 2.1 integration repair

The Project Invisible Qwen 2.1 engine bypasses Forge's standard generation callbacks. Head Swap now has an explicit bridge into that engine, so its references, prompts, BFS adapter, character adapter, finishing controls, protected mask and seed lock are actually used.

- Automatic adapter selection follows preset/checkpoint changes. It prefers `bfs_head_v1_qwen_2.1` for Qwen and `bfs_head_v1_flux-klein_9b_step3500_rank128` for Klein 9B, including adapters discovered in subfolders. Turn off automatic selection to use a custom adapter.
- Up to 20 headshots form the selection pool; each output sends its target and selected headshot to Qwen. Best-match, manual-slot and rotation controls remain available. This avoids encoding all 13 headshots for every output.
- Qwen protected-head mode conditions on the complete scene and composites only the chosen head region. It handles RGBA output and preserves original output dimensions. Native inpaint masks must be cleared; use the extension's protected mask control.
- Qwen BFS, character and the installed Viggle v0.2.1 Turbo adapter were tested together with INT8 ConvRot on an RTX 5090. Turbo keeps CFG 1; its negative prompt is inactive. Klein turbo prompt tags also keep CFG 1. Quantization remains owned by each engine; other quantizations and a real Klein generation were not GPU-validated in this repair.
- Saving now resolves an output-folder fallback, honors batch folder/name overrides, checks the written file and reports its path. Explicitly disabled saving remains disabled and is reported.
- Optional CPU identity checks are now connected to completed images and shown in the existing generation report. They are advisory; a passing score does not guarantee likeness. The real protected-head test passed the selected/median identity thresholds and face-height/center check; width/reference-consistency warnings still require visual review.

Restart Forge completely after the current batch finishes to load both updated extensions. A browser refresh alone does not load Python changes.

## Turbo control and profile-face follow-up

- Fixed a companion UI bug that wrapped `gr.skip()` inside a slider value. Toggling Turbo off could subsequently crash a batch with `float() ... not dict`. Callbacks now emit proper skip updates and match the number of bound controls; the option reader also recovers old nested-skip values and validates numeric strengths.
- Added an optional CPU YuNet fallback when MediaPipe misses a sunglasses/profile face. It uses the already installed identity detector and downloads nothing.
- Quality diagnosis: random hairstyle, age, ethnicity and film-camera choices change the requested identity and scene. A clean local identity preset uses protected-head compositing and clears these style changes. Existing presets are retained.
- Identity and geometry measurements are advisory. Profile views and sunglasses can still produce imperfect likeness; inspect the generated face rather than treating a successful save as a quality guarantee.
