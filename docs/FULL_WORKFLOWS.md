# Full workflows UI guide — October 7, 2026

The Full workflows engine runs in a separate worker while keeping Forge's normal txt2img/img2img pages, native controls, Generate button, saving and gallery. It does not start a Comfy server, create another environment or patch Forge core.

## Beginner path

1. Select a Qwen 2.1 checkpoint/components.
2. Open **Qwen · Image 2.1 → Models → Full workflow backend**.
3. Choose **Full workflows (isolated Comfy)**.
4. Press **Check workflow setup**. Fix only items marked missing.
5. Start with **Euler / Simple**, Union off, prompt enhancer off and merged Turbo off.
6. For editing, upload the main image in native img2img. For outpaint, choose padding in **Edit → Workflow tools** and inspect the gray canvas preview.

The readiness check only reads local folders and safetensors headers. It never downloads or loads model tensors.

## Controls and requirements

| Feature | Local requirement | Important rule |
|---|---|---|
| Euler / Simple | Qwen 2.1 DiT, Qwen3-VL 8B encoder, Qwen 2.1 VAE, isolated Comfy code | Recommended first test |
| RES 2S / Beta | Same files plus RES4LYF | Slower; RES4LYF keeps its license restrictions |
| Union ControlNet | Dedicated Qwen 2.1 16-block Union model | This is a model patch, not ordinary global ControlNet |
| Masked edit / outpaint | Union enabled; source plus mask | Source pixels outside the mask are restored, including alpha |
| Local prompt enhancer | Separate compatible prompt-encoder file | It never replaces the image model's Qwen3-VL encoder |
| Thorough rewriting | Prompt enhancer enabled | Slower and experimental |
| Merged Viggle Turbo | Exact merged Qwen 2.1 checkpoint | Six steps, CFG 1; separate Turbo LoRA must be off |
| OpenPose / DWPose | All local preprocessor assets | Missing assets fail clearly; Generate does not fetch them |

## What the workflow presets change

Presets set visible starting values such as steps, task, backend, sampler, scheduler, texture VAE, source proportions, padding, Union and merged Turbo. You may change those values afterward. Presets cannot move an edit workflow onto txt2img; wrong-tab use fails with a plain message.

All new controls are appended to the **60-position** saved-settings contract. Older settings keep their existing positions.

## Backend combinations

The Full workflows engine requires Spectrum, the extra refiner and LanPaint off; unsupported combinations fail clearly. Use native Union masked editing there. The default pipeline retains its existing options. Native previews show intermediate predictions when steps finish; one-second heartbeat messages do not promise a newly sampled image every second. The final preview uses the decoded output.

## Quality guidance

- BF16 VAE is the verified decoder baseline. There is no established Qwen 2.1 quality benefit from forcing an FP32 VAE in this release.
- Use overlap only when the outpaint boundary needs correction. Zero overlap preserves every original source pixel.
- Keep cache on for normal use. The tested cache-on/off pair produced identical pixels for one seed, but that is not a universal guarantee.
- The isolated worker preserves request settings at launch. Changing controls during a run applies to the next run.

## Tested and untested

See [VALIDATION.md](../VALIDATION.md) for exact execution evidence. Union, RES 2S/Beta, base plus Viggle LoRA, masked/outpaint preservation and cancellation recovery were GPU-tested. Merged Turbo and automatic pose preprocessing were not GPU-tested in that session. Prompt-enhancer loading, status, cancellation and recovery passed; complete rewrite-to-image output and Thorough mode remained unproven when this guide was written.

## Credits and licenses

Existing project acknowledgments remain in the README and NOTICE. Full workflows additionally depend on ComfyUI, RES4LYF, Qwen Image 2.1, Qwen3-VL, Viggle and VideoX-Fun-derived Qwen control concepts. Their original licenses and model terms remain authoritative. Model and code downloads require explicit approval.

Rebuilding references: [IntoTheLatent](../resources/workflows/) supplied workflow designs; [AiWithYou / AiKimi](https://github.com/AiWithYou/aikimi-forge-neo) inspired readiness reports and edit boundaries. Comfy code is pinned to `7a5dad695fe1cae25efcb2550530fb20ef68da3d`, RES4LYF to `26036f647ca15d3048a193daf99a40cecfc3820d`. Their licenses are preserved under `lib/vendor/`. The short T2I prompt-helper resource is an adapted fallback; the edit prompt resource is extracted from the supplied workflow. These are audited adaptations, not promises of pixel-identical Comfy output.

## Refreshed interface

**Workflow preset** shows readable task names. **Starting steps** fills the native steps control; the native value remains editable. Names beginning **Full** require the Full workflows engine. **Merged Turbo** requires a separate model file. Choose **Custom · keep my settings** when you want to adjust everything yourself. **Memory → Keep Qwen ready** controls whether models stay cached between batches.

The Sharp sampler is an optional Default-pipeline feature. Its history-step sign is corrected and checked against four nonuniform log-sigma steps; zero sharpening still uses the multistep method after the first step. Turbo and Full workflows retain their own schedules.
