# Upstream attribution

Source: https://github.com/capitan01R/ComfyUI-qwen_img_2_1_enhancer

Inspected commit: `e7204ab908526381c67cd8276adc7aa2e9581c95`.

Copyright (c) 2026 capitan01R. MIT license preserved verbatim in `LICENSE`.

`lib/enhancer.py` adapts the upstream phrase parser and attention-odds method to Diffusers QwenImage21Pipeline. ComfyUI modules and node registration are not imported or bundled. Token mapping uses this worker's actual processor IDs; existing licenses for Diffusers/model components remain unchanged.
