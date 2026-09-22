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
