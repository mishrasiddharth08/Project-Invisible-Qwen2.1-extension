# LanPaint core

Source: https://github.com/scraed/LanPaint

Pinned commit: `2d7912f9a5efe5ece8de334c7ca18317b8288c39`.

`lanpaint.py`, `earlystop.py`, and `types.py` are unmodified copies from `src/LanPaint/`. Their GPL-3.0 license is preserved in `LICENSE`. The empty `__init__.py` is extension packaging; no ComfyUI nodes, monkey-patches or runtime stubs are imported.

The separate `lib/lanpaint.py` bridge is GPL-3.0 and connects this algorithm to the existing QwenImage21Pipeline instance. It is not the older Qwen image-edit adapter from LanPaint-diffusers. Existing model and other third-party licenses remain applicable.

Research credit: Candi Zheng, Yuan Lan and Yang Wang, *LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling*, TMLR 2025. Early-stop logic is credited upstream to godnight10061.
