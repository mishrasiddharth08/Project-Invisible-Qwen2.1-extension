# LanPaint masked editing — September 28, 2026

LanPaint is an optional, experimental masked-edit sampler inside the existing Qwen img2img controls. It adds inner refinement calculations to each denoising step. More thinking steps use more time; this is not a speed feature.

## Use it

1. Restart Forge after installing this update and refresh your browser.
2. Select the Qwen preset and checkpoint. Open **img2img → Inpaint**.
3. Upload a source image and paint the region you want to replace. White means edit; black means preserve. The native mask-invert option is honored.
4. Describe the change in the normal prompt field.
5. Open **Qwen · Image 2.1 → Edit**, enable **LanPaint masked edit**, and start with **2 thinking steps** (1–5 available).
6. Press the normal **Generate** button. Use a small image first. More references may be added using the existing reference slots.

For outpainting, supply an already-expanded canvas and mask the added border using native img2img tools. This extension does not add a separate canvas-expansion editor.

## Behavior and limitations

- Off by default, with no separate generation tab, environment, model download or Forge core change.
- Uses the official Qwen 2.1 pipeline, encoder and VAE. The older Qwen edit pipeline is not used.
- Unmasked output pixels are copied from the source after cleanup, including alpha. If output dimensions differ, preservation is relative to the resized source, not the original-resolution pixels.
- Empty masks and Head Swap combinations are rejected with an explanation.
- Spectrum, prefix-KV caching, the refiner, drift alignment and background blending are disabled for the masked edit. Ordinary adapters still use the existing adapter path; compatibility with every adapter is unverified.
- Native progress/preview reporting continues through the existing worker. Thinking calculations add model passes between ordinary steps, so progress can pause on one step while work continues.
- Interrupt is checked before every transformer call. Instance hooks are restored on completion, failure or cancellation. No ComfyUI-wide or Forge-wide sampler patch is installed.
- This bridge currently supports the Euler flow scheduler only. Unexpected pipeline changes fail explicitly.
- No universal hardware, image-quality or memory guarantee is made. Native crop/resize workflows, multiple-reference quality and outpainting quality need further real-world testing.

## Verification

- Real local RTX 5090 test: existing INT8 ConvRot DiT, BF16 encoder, 256 × 256 RGBA canvas, three sampling steps, one thinking step, seed 42.
- Completed in 85.6 seconds after worker loading. Output dimensions and alpha were valid, masked pixels changed, and unmasked pixels matched exactly.
- This low-step run proves execution and pixel preservation, not finished image quality or performance on other GPUs.
- Regression tests cover mask validation/inversion, deterministic sampling with and without CFG, finite output, hook restoration, cancellation, and appended UI-control routing.

## Credits and license

Core: [scraed/LanPaint](https://github.com/scraed/LanPaint), pinned to `2d7912f9a5efe5ece8de334c7ca18317b8288c39`. The unmodified core and its GPL-3.0 license are included in `lib/vendor/lanpaint/`; `lib/lanpaint.py` is the GPL-3.0 bridge. Model weights and other upstream components retain their separate licenses. See [vendor details](../lib/vendor/lanpaint/README.md).
