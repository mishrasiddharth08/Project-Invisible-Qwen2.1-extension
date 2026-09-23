# Project Invisible — Qwen-Image-2.1 for Forge Neo

An independent, unofficial extension that adds Qwen-Image-2.1 to the normal Forge Neo workflow.

> This is my first public project and I am still learning. Please forgive any mistakes or rough edges. Kind, complete bug reports will help improve the project for everyone.

## Latest update — September 23, 2026

- Fixed quantized LoRA loading and empty-negative-prompt CFG errors.
- Kept one native CFG slider and improved failed-pipeline cleanup.
- Added portable paths, source verification, and update/backup guidance.
- **Verified:** 87 automated Qwen tests passed. Restart Forge after updating.

See the [dated update history](CHANGELOG.md) for details, installation steps and testing limits.

## The Project Invisible idea

The extension should feel like built-in Forge support:

- no separate generation tab;
- no extra virtual environment;
- no modified Forge core files;
- use the normal preset, checkpoint selector and **Generate** button;
- show special controls only when this engine is selected;
- release this extension's RAM/VRAM when another model is selected.

“Invisible” means a familiar workflow. It does not mean hiding downloads, errors, limitations or resource use.

## Testing status

- **Text-to-image:** tested and working on the author's computer.
- **Other modes:** untested or not fully confirmed for this first public release.

The code contains experimental support for image editing, references, transparency, compatible LoRA/LoKr files, Spectrum acceleration, DeGrid cleanup and different memory profiles. Treat these as experimental until more users test them.

No extension can guarantee every GPU, driver, model file, adapter or Forge update will work.

## Features

- Normal Forge Neo preset/checkpoint selection and **Generate** button.
- Dedicated model pipeline; it is not routed through an SD or Flux pipeline.
- Dependencies remain inside the extension's `_deps` folder.
- Existing model folders are scanned before downloads are offered.
- Pressing **Generate** never silently downloads model weights.
- Manual installation and optional user-approved downloads.
- Current-image and overall-batch progress bars.
- Live developing previews when Forge live previews are enabled.
- Worker and memory cleanup when switching models.
- Optional DeGrid artifact cleanup with a true bypass checkbox.
- Optional experimental Spectrum acceleration, disabled by default.

## Requirements

- Windows and a working Forge Neo installation.
- A GPU/PyTorch setup already supported by that Forge installation.
- Enough GPU memory and system RAM for the selected profile.
- Internet access for first dependency installation and optional downloads.
- Model weights obtained under their original license.

Model weights are not included in this repository.

## Beginner installation

1. Download this repository as a ZIP.
2. Extract it.
3. Rename the extracted folder to `project-invisible-qwen-image-21`.
4. Copy it into `sd-webui-forge-classic\extensions\`.
5. Make sure the extension is not accidentally nested twice.
6. Start Forge normally. The first start may take longer while local dependencies install.
7. Restart Forge once if the preset/checkpoint is missing.
8. After an update, refresh the browser with `Ctrl+F5`.

Correct:

```text
extensions\project-invisible-qwen-image-21\README.md
```

Incorrect:

```text
extensions\project-invisible-qwen-image-21\project-invisible-qwen-image-21\README.md
```

No extra virtual environment is needed.

## Required model files

Install one DiT, one Qwen3-VL text encoder and the Qwen-Image-2.1 VAE.

```text
DiT
qwen_image_2.1_bf16.safetensors
qwen_image_2.1_int8_convrot.safetensors

Text encoder
qwen3vl_8b_bf16.safetensors
qwen3vl_8b_int8_convrot.safetensors
qwen3vl_8b_w4a8.safetensors

VAE
qwen_image_2.1_vae_bf16.safetensors
```

Manual download is recommended. Put compatible files in Forge's normal model folders or:

```text
sd-webui-forge-classic\models\Qwen-Image-2.1\
```

Do not substitute an older model generation's text encoder or VAE.

The extension's **Models** section can download only selected files after you explicitly accept the license and approve the download.

## First text-to-image test

1. Start Forge Neo.
2. Select the `qwen-image-2.1` UI preset.
3. Select the matching Project Invisible checkpoint.
4. Open the normal **txt2img** tab.
5. Enter a simple prompt.
6. Start with a modest image size.
7. Use 40 steps for the quality default or 25 for a faster test.
8. Keep CFG scale at `1.0` unless you understand its extra cost.
9. Press **Generate**.

For an out-of-memory error, enable **Save GPU memory**, choose a smaller profile and reduce the image size.

## Important controls

- **DeGrid:** enabled by default. It only corrects a detected repeating grid artifact. Uncheck it for a complete filter bypass.
- **Spectrum speedup:** experimental and off by default. It may change image details even with the same seed. CFG scale above `1.0` disables it.
- **Save GPU memory:** reduces GPU use through offloading but can be slower.
- **VRAM profile:** selects safer defaults; it is guidance, not a guarantee.
- **CFG scale:** values above `1.0` need a real negative prompt and more computation.

## Isolation and Forge updates

The extension keeps its code and added dependencies inside its own folder and does not rewrite Forge core files. Generation uses a separate worker with Forge's existing Python interpreter. Forge's own Python, PyTorch and UI APIs remain shared dependencies, so future compatibility cannot be guaranteed.

Keep a known-working backup outside Forge before updating. See [UPDATE_SAFETY.md](UPDATE_SAFETY.md) for practical backup, verification and recovery steps. The included `verify_integrity.py` detects changes to recorded source files; it does not freeze Forge or automatically restore files.

## Updating

1. Stop Forge.
2. Back up the old extension folder.
3. Replace its files with the new release.
4. Do not place model weights inside the extension folder.
5. Start Forge and refresh the browser with `Ctrl+F5`.

## If something goes wrong

Before opening an issue:

1. Restart Forge.
2. Check for a double-nested extension folder.
3. Confirm all required model files exist.
4. Try a smaller image with **Save GPU memory** enabled.
5. Reproduce the problem once and keep the DOS/terminal window open.

Open a GitHub ticket and paste the **COMPLETE error from the DOS/terminal window**. Include everything from the first error line through the final traceback line. Do not paste only the last sentence.

You may also paste the complete error into ChatGPT, Claude, Gemini or Grok and ask for a simple explanation.

Please include:

- Forge Neo version or commit;
- Windows version;
- GPU, VRAM and system RAM;
- checkpoint, text encoder and VAE filenames;
- image size and steps;
- memory profile and offloading setting;
- whether DeGrid, Spectrum or a LoRA/LoKr was enabled;
- exact steps that reproduce the issue;
- complete terminal error;
- `logs/worker.log`, when available.

Remove private usernames, paths, prompts, tokens and images before posting publicly.

## Known limitations

- Only text-to-image is confirmed for this first public release.
- Editing and other modes are experimental or untested.
- Speed and memory use vary by hardware.
- DeGrid fixes a specific fine grid artifact; it cannot repair anatomy, composition or missing detail.
- Spectrum trades exact output matching for possible speed.
- Incompatible adapters are rejected.
- Forge updates may change extension hooks.

## Developer test command

```powershell
& 'path\to\sd-webui-forge-classic\venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Passing automated tests does not prove every GPU workflow works.

## Privacy and safety

- Generation stays local unless you approve a download or use another network-enabled extension.
- Review logs before posting them publicly.
- Never publish passwords, access tokens, private prompts or personal images by accident.
- Download weights only from sources you trust.

## License

This is an independent Forge adapter, not an official Qwen product.

Read [LICENSE](LICENSE), [NOTICE](NOTICE), and the licenses under `resources/` and `lib/vendor/`. Upstream files and model weights keep their original licenses. The supplied Qwen Research License contains non-commercial and redistribution conditions. This repository does not relicense the model or upstream projects.

## Help and contributions

- Use [GitHub Issues](https://github.com/mishrasiddharth08/Project-Invisible-Qwen2.1-extension/issues) for bugs and questions.
- Follow [ISSUE_TEMPLATE.md](ISSUE_TEMPLATE.md) and include the complete terminal error.
- Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change.

I am a beginner too, and this is my first public attempt. Please forgive mistakes. Patient explanations and clear reports are deeply appreciated.

## Special thanks

Special thanks to [u/malcolmrey and the r/malcolmrey community](https://www.reddit.com/r/malcolmrey/) and the [r/SECourses community](https://www.reddit.com/r/SECourses/) for support and inspiration.

Thank you to the Forge Neo, Diffusers, Qwen, DeGrid, Spectrum and wider open-source communities.


### Empty negative prompt

If native CFG Scale is above 1 and the negative prompt is empty, generation uses CFG 1 and prints a notice instead of stopping. Saved image metadata records the effective value. To use CFG above 1, enter a real negative prompt.
