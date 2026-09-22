# Project Invisible — Qwen-Image-2.1 for Forge Neo

## Short GitHub “About” description

Unofficial Forge Neo extension that integrates Qwen-Image-2.1 into the normal preset, checkpoint and Generate workflow—without a separate tab, extra virtual environment or Forge core fork.

## Project overview

Project Invisible is an independent Forge Neo extension created to make an additional image-generation engine feel like a natural part of the existing WebUI.

The basic idea is simple: select the matching UI preset and checkpoint, enter a prompt in the normal Forge interface, adjust familiar settings and press the usual **Generate** button. The extension performs its model-specific work behind the existing workflow instead of asking users to learn a separate application or generation page.

This is the author's first public project. It was built by a beginner who is still learning. Mistakes may exist, and patience is sincerely appreciated. Clear reports and complete error messages will help the project improve.

## The Project Invisible philosophy

“Invisible” means integration with minimum disruption.

The extension is designed around these principles:

- use Forge Neo's normal preset and checkpoint selectors;
- use the existing txt2img interface and Generate button;
- avoid adding a separate generation tab;
- avoid requiring another virtual environment;
- avoid modifying or forking Forge core files;
- show model-specific controls only when the engine is selected;
- leave unrelated models and extensions unchanged;
- release the extension's worker and memory when switching away;
- make downloads, errors and experimental behavior visible and honest.

The extension is “invisible” in workflow, not in responsibility. It should never hide model downloads, hardware limitations, errors, licensing conditions or quality trade-offs from the user.

## Testing status

Text-to-image generation has been tested and worked on the author's computer.

Other modes and hardware combinations are untested or not fully confirmed for this first public release. The code contains experimental paths for image editing and other advanced behavior, but users should not assume those paths are production-ready merely because controls or code exist.

Performance, memory use and compatibility vary with the GPU, driver, PyTorch build, Forge version, selected model files, resolution, adapter and enabled options. No claim is made that every NVIDIA or AMD configuration will work.

## Main behavior

The extension registers its engine through Forge Neo's normal model and preset workflow. When the matching selection is active, generation is routed to the engine's dedicated pipeline. Other checkpoints continue through their original Forge processing paths.

The implementation does not pretend that the engine is an SD or Flux model. It loads the compatible transformer, text encoder, VAE, processor and scheduler expected by the dedicated pipeline.

Dependencies that would otherwise conflict with Forge are installed into an extension-local `_deps` folder. The same Forge Python interpreter is used; a second virtual environment is not created.

## Model handling

Model weights are not included in the repository.

The extension scans supported Forge model folders and its dedicated model directory before reporting missing files. Existing compatible files are reused. Pressing **Generate** does not silently fetch weights.

Users can install weights manually, which is the recommended method. An optional download interface is also available. Automatic downloads occur only after the user selects specific files, accepts the applicable license and explicitly authorizes the download.

A complete component set requires:

- one compatible DiT file;
- one compatible Qwen3-VL text encoder;
- the matching Qwen-Image-2.1 VAE;
- the bundled processor, tokenizer, scheduler and configuration files.

Components from older or unrelated architectures must not be substituted merely because their filenames look similar.

## Progress and previews

The extension integrates with Forge's progress system and presents two kinds of progress:

- the current image's progress;
- the overall batch's progress.

When Forge live previews are enabled, each image begins with a small neutral gradient placeholder. Real intermediate previews then replace it as sampling proceeds. A brief visual fade makes changes less abrupt. The final decoded image replaces all previews at completion.

Intermediate previews are best-effort. They may be delayed by slow steps, decoding time or low available GPU memory. The extension does not invent artificial generation stages or claim that the final sampling step means decoding has already finished.

## Memory management

Hardware-aware profiles provide practical starting settings for different memory sizes. Depending on the selected profile and file format, the extension may use direct GPU loading, model offloading or finer-grained offloading.

These settings are best-effort safeguards, not guarantees. A large model can still exceed available GPU memory or system RAM, especially when other applications or models are using memory.

When the user selects another model or preset, the extension stops and releases its dedicated worker so its RAM and VRAM can be reclaimed. This cleanup affects only resources owned by this extension.

## Image-quality protection

Decoder tiling is disabled because controlled testing showed that it could create repeating colored spots and damage transparency. Decoder slicing and supported offloading remain available.

DeGrid is included as an optional final-image cleanup. It detects a specific two-pixel repeating grid pattern, estimates the required correction and limits that correction to protect real texture and edges. Clean images are passed through unchanged. Alpha transparency is preserved.

DeGrid is enabled by default. Users can uncheck the **DeGrid** control for a complete bypass. It is a targeted artifact correction, not a general image enhancer. It cannot repair anatomy, composition, lighting, identity, missing details or weaknesses already present in the generated image.

## Experimental Spectrum acceleration

Spectrum acceleration is available as an experimental, opt-in setting. It predicts selected transformer steps while retaining real computation during important warm-up and final-detail stages.

Spectrum is disabled by default because it is an approximation. It may improve speed on suitable workloads, but the same prompt and seed can produce different fine details. Very short runs or unsupported configurations remain on the normal inference path. True CFG values above `1.0` disable Spectrum automatically.

## LoRA and LoKr adapters

The extension recognizes compatible adapters through Forge's normal Extra Networks prompt tags.

Compatibility checks examine metadata and actual transformer targets. Unrelated SD, Flux or older-generation adapters are rejected instead of being applied to the wrong architecture.

Standard compatible LoRA files and the supported plain full-factor LoKr layout have dedicated loading paths. LoKr changes are applied through request-scoped hooks instead of merging into quantized base weights. Adapter state is removed after generation and when switching models.

Adapter support remains sensitive to how a file was trained and saved. A matching filename alone does not prove compatibility.

## Beginner installation

Download the repository ZIP, extract it and rename the folder to:

```text
project-invisible-qwen-image-21
```

Copy it into:

```text
sd-webui-forge-classic\extensions\
```

Start Forge normally. The first start may take longer while extension-local dependencies install. Restart Forge once if the new preset or checkpoint does not appear. After updating the extension, refresh the browser with `Ctrl+F5`.

The README contains the recognized model filenames, first-generation steps and troubleshooting guidance.

## Troubleshooting and support

If a problem occurs, users should open a GitHub issue and paste the **complete error from the DOS/terminal window**. The report must begin at the first error line and continue through the final traceback line. A single final sentence is usually not enough to identify the cause.

Reports should also include:

- Windows and Forge Neo versions;
- GPU, VRAM and system RAM;
- checkpoint, text encoder and VAE filenames;
- resolution and sampling steps;
- memory profile and offloading setting;
- whether DeGrid, Spectrum or a LoRA/LoKr was used;
- exact reproduction steps;
- the relevant worker log when available.

Private usernames, paths, prompts, tokens and images should be removed before posting publicly.

Users may also paste the complete error into ChatGPT, Claude, Gemini or Grok and ask for a beginner-friendly explanation.

## Contributions

Bug reports, documentation corrections and focused code improvements are welcome.

Contributors should state exactly what was tested, avoid describing untested modes as working, preserve upstream licenses and never commit model weights, generated images, dependency folders, logs, access tokens or private information.

## License and independence

This extension is independent and unofficial. It is not an official Qwen or Forge product.

The repository does not relicense model weights or incorporated upstream projects. Users and redistributors must read the root `LICENSE`, `NOTICE`, and license files under `resources/` and `lib/vendor/`. The supplied Qwen Research License contains non-commercial and redistribution conditions.

## A humble note from the author

This is a first public attempt by a non-programmer learning through experimentation and community help. Please forgive mistakes. Constructive feedback, patient explanations and complete error reports are welcomed with gratitude.

## Special thanks

Special thanks to [u/malcolmrey and the r/malcolmrey community](https://www.reddit.com/r/malcolmrey/) and the [r/SECourses community](https://www.reddit.com/r/SECourses/) for support and inspiration.

Thanks also to the Forge Neo, Diffusers, Qwen, DeGrid, Spectrum and wider open-source communities whose work made this project possible.
