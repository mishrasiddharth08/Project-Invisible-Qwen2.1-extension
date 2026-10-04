# Project Invisible — Qwen-Image-2.1 for Forge Neo

Qwen 2.1 inside Forge's existing **txt2img**, **img2img**, **Generate** and gallery workflow.
Independent community extension; not an official Qwen product.

## September 28, 2026 update

- Original Forge theme colors with aligned, responsive tabs and less clutter.
- Memory saving now respects its checkbox in Auto mode. Workers unload after a batch.
- Spectrum keeps CFG prediction histories separate and rejects unstable forecasts.
- Optional PixelDriftFix-style alignment for img2img, with alpha preservation.
- Native Forge controls stay visible; reference upload/remove buttons remain usable.
- Qwen Turbo strength has a unique saved setting; disabled turbo no longer validates stale slider values from another extension.

![Qwen panel guide](docs/img/ui-tour.svg)

## Install or update

1. Finish your current generation and close Forge.
2. Install this repository through **Extensions → Install from URL**, or extract its ZIP into Forge's `extensions` folder. Use one installation method, not both.
3. The extension folder must contain `scripts/engine.py` directly. Avoid an extra nested folder.
4. Start Forge. Initial setup installs extension dependencies; model weights are separate.
5. After updates, restart Forge and refresh the browser. A browser refresh alone does not reload Python code.

Repository: https://github.com/mishrasiddharth08/Project-Invisible-Qwen2.1-extension

Keep a backup before updating. See [update safety](UPDATE_SAFETY.md) and [validation](VALIDATION.md).

## Required model files

Put one compatible DiT, one text encoder and the VAE in `models/Qwen-Image-2.1`.

| Component | Available filenames |
|---|---|
| DiT | `qwen_image_2.1_int8_convrot.safetensors` or `qwen_image_2.1_bf16.safetensors` |
| Text encoder | `qwen3vl_8b_int8_convrot.safetensors`, `qwen3vl_8b_w4a8.safetensors` or `qwen3vl_8b_bf16.safetensors` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |

Quantized weights need a compatible Forge/PyTorch/GPU setup. BF16 uses more memory.
The **Models** tab contains setup instructions and optional downloads. Downloads require your license approval and a button click; Generate never downloads models.

## Start here

1. Choose the **qwen-image-2.1** UI preset and compatible checkpoint/components.
2. Enter your prompt. For an edit, upload the main image in native **img2img**.
3. Open **Qwen · Image 2.1**. Start with **Quality**, **Auto** memory and **Save GPU memory** on.
4. Use Forge's native steps, CFG, seed and size controls. Click **Generate**.

![Generation quickstart](docs/img/usage-steps.svg)

## Panel tabs

| Tab | Controls |
|---|---|
| Finish | Refiner and grid-pattern cleanup |
| Edit — img2img only | PixelDriftFix alignment, background compositing, up to nine extra references |
| Speed | Turbo LoRA, sharp sampler and optional Spectrum acceleration |
| Style | Photography style selection, detail-fix LoRAs, texture-fix VAE and approved downloads |
| Memory | VRAM budget, maximum size, offload and advanced adapter compatibility |
| Models | Collapsed setup guide and approved model downloads |

**Fast** selects the turbo adapter and eight steps. Download that adapter first. Native step changes remain respected. The distilled turbo path uses **CFG 1**, even if a higher value was entered. Use the ordinary model for CFG above 1. Refiner passes take additional time and memory; they reuse the loaded model.

### Memory

Offloading reduces GPU residency by using system RAM during generation. After a complete batch, the worker exits and releases its model RAM/VRAM. The next batch reloads the model. Advanced users can set `"keep_loaded": true` in `config.json` to trade idle memory for faster repeated batches.

The resident-fit check uses available VRAM and reserves headroom. It cannot guarantee that every resolution, batch, adapter or GPU will fit. Reduce size or select a lower VRAM budget if needed.

### Spectrum and CFG

Spectrum is optional and approximate. Conditional and unconditional CFG passes now have separate histories. Warmup and final passes are calculated normally; unstable forecasts fall back to normal calculation. PNG metadata records real and forecast pass counts.

For an exact quality baseline, turn Spectrum off and keep the seed, prompt, model, steps and size unchanged. These fixes have automated coverage; this release does not claim a measured visual-quality improvement on every model/GPU.

### PixelDriftFix-style alignment

Enable **Edit → Align edit to source** for small unintended framing shifts. It runs before background compositing, saving and gallery display. It preserves output size and alpha, and skips insufficient matches, large warps and protected head-swap output. Turn it off for intentional reframing.

![Edit processing order](docs/img/img2img-flow.svg)

This is an independent implementation of the recommended SIFT/global-homography workflow in [Mozer/ComfyUI-PixelDriftFix](https://github.com/Mozer/ComfyUI-PixelDriftFix), inspected at commit `62b79b86016e3914e62e258624646afa6cd8d394`. That checkout has no license file; its source is not bundled. Experimental mesh alignment is not included. OpenCV with SIFT is required; missing support leaves the image unchanged. Analysis is capped at a 1536-pixel long side and 6000 features.

## Checks and limitations

- Automated suite: **193 tests passed** for this update, including real Gradio panel construction and all 27 argument positions.
- Isolated browser checks covered desktop/narrow layouts and Fast-mode callbacks. The live Forge session was not restarted during installation.
- Live worker test: RTX 5090, INT8 ConvRot DiT/text encoder, CFG 2, Spectrum on, 512 x 512, 20 steps, seed 987654. A valid PNG was saved with 36 real and 4 forecast passes; test-worker cleanup returned GPU usage to about 1.6 GB. A matched visual-quality comparison and full live Forge gallery validation remain unverified.
- Memory usage, speed and quality depend on hardware, model precision, resolution and adapters. No universal GPU compatibility is claimed.

Report your Forge version, GPU, precision, steps, CFG, enabled options and relevant error log. See [contributing](CONTRIBUTING.md), [license](LICENSE) and [notices](NOTICE).

## Special Thanks

- [**r/sdforall**](https://www.reddit.com/r/sdforall/) - community discussion and testing
- [**r/SECourses**](https://www.reddit.com/r/SECourses/) - community discussion and testing
- [**r/malcolmrey**](https://www.reddit.com/r/malcolmrey/) - community discussion and testing
- [**Haoming02 / sd-webui-forge-classic (neo branch)**](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) - the Forge Neo tree this extension targets
- [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) - reference for upstream sampler/scheduler coverage
- **The Forge / AUTOMATIC1111 community** - for the extension ecosystem this plugs into

Special thanks to u/malcolmrey for support and inspiration. Thank you also to the Forge Neo, Diffusers, Qwen, DeGrid, Spectrum and wider open-source communities.

## September 28, 2026 — LanPaint masked editing

Optional **LanPaint masked edit** is now inside **img2img → Qwen · Image 2.1 → Edit**. Paint a mask in native Inpaint, choose 1–5 thinking steps, and press Generate. Unmasked source pixels are preserved, including alpha. More thinking takes longer.

[Usage, tests and limitations](docs/LANPAINT.md). Restart Forge after updating. Existing acknowledgments remain above.

## October 4, 2026 — reference and phrase priorities

Optional **Weighted prompt phrases** is inside **Qwen → Finish**; use `(warm lighting:1.3)` in positive or negative prompts. **img2img → Qwen → Edit → Reference priorities** accepts `1:1.2, 2:0.8`. Defaults retain native behavior. Restart Forge after updating.

See [usage, research and validation](docs/QWEN_ENHANCER.md). Thanks to capitan01R for the upstream MIT-licensed enhancer; earlier acknowledgments are preserved.

## October 4, 2026 — native Forge compatibility

Project Invisible now coexists with the native engine proposed in Forge PR #1512. Use `qwen-image-2.1` for this extension or `qwen21` with a regular checkpoint for native Forge after installing its support. [Details and test limits](docs/FORGE_NATIVE_QWEN21.md).
