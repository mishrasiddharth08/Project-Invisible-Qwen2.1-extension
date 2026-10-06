# October 6, 2026 — Full workflows release evidence

The existing Forge UI contract now contains **60 positions**. New controls remain appended; native prompt, steps, CFG, seed, size, Generate, save and gallery controls remain authoritative.

The complete suite ran in both source and installed copies: **319 tests, 318 passed and one unrelated test skipped**. Real panel construction verified both Forge panels and all 60 positions.

Post-installation worker matrix also passed all five cases: T2I, edit, Union strength 0, Union strength 1 and masked outpainting. Each decoded an RGBA PNG; each returned to 32 MiB allocated in that worker. Total test session: 190 seconds, then clean process close. These are small 256-pixel/two-step execution checks, not a generation-speed or visual-quality benchmark.

RTX 5090 isolated-worker evidence:

- Basic T2I, reference edit, Union strength 0 and 1, masked/outpaint compositing and a separate 25-step generation passed at 256×256.
- RES 2S/Beta at 50 steps passed. Viggle r256 LoRA passed at Euler/6 steps with exact sigmas `1, .9375, .875, .75, .5, .25`.
- CFG 2 with a negative prompt, cache off/on, two references, cancellation followed by same-worker recovery, and Union strength 2 over the 0.25–0.75 interval passed.
- Cache-off and cache-on outputs were byte-identical for the tested seed. The two-reference case peaked near 8.0 GiB allocated; the Union strength-2 case peaked near 10.3 GiB. These are worker allocator figures, not total board usage.
- A separate 8 GB-profile weighted Union run passed at 256×256, peaking at 4058 MiB allocated and returning to 256 MiB idle allocated memory.

Boundaries:

- **Tested live:** local INT8 ConvRot Qwen 2.1 DiT and Qwen3-VL encoder, BF16 VAE, Euler/Simple, RES 2S/Beta, base plus Viggle Turbo LoRA, Union model, masking/outpaint preservation, cancellation/recovery.
- **Prompt enhancer:** real Qwen3.5-9B loading, one-second status updates, a hard 64-token cap, cancellation and recovery passed. The bounded non-thinking run emitted one-second status updates, then failed after 77 seconds because its 64-token output was truncated before valid JSON completed. Complete rewrite-to-image output remains unproven. Thorough mode remains untested live.
- **Merged Turbo:** source, filename, license, hash and architecture path are verified, but the 7.3 GB merged checkpoint was not downloaded or GPU-tested in this validation session.
- **Pose:** uploaded pose maps and local-asset checks are CPU-tested. Automatic OpenPose/DWPose execution was not tested because their local preprocessor weights were absent. No preprocessor download occurs during Generate.
- These small images prove execution and routing. They do not prove universal visual quality, all resolutions, all GPUs, every adapter combination or a fresh end-to-end Forge gallery session.

# October 4, 2026 — optional detail-fix recovery

- Missing or empty optional detail-fix files restored by saved presets no longer block Generate. The run skips that adapter, prints a warning and records the skip in image metadata.
- Existing detail-fix files still apply normally. Generate never downloads files; downloads still require approval.
- 258 tests passed, one skipped (259 total), including missing/empty/available adapter cases.

# October 4, 2026 — Forge PR #1512 compatibility and VAE memory

255 passed, one skipped (256 total) in both source and installed copies. Native/extension ownership, older Forge fallback, API overrides and VAE strip boundaries are covered. GPU worker checks passed with weighted phrases/references, LanPaint, Spectrum, cancellation and recovery. One large actual Diffusers VAE upsampler produced identical FP32/FP16/BF16 outputs using about 23% less peak allocated memory for that isolated operation. Whole-generation speed/VRAM gains and unmerged native PR inference remain unverified. Details: [Forge compatibility](docs/FORGE_NATIVE_QWEN21.md).

# October 4, 2026 — enhancer

247 tests run: 246 passed and one skipped. A real RTX 5090 256-pixel, two-step CFG 2 edit with independently weighted positive/negative phrases, reference priority and native KV caching succeeded. Unweighted txt2img subsequently succeeded in the same worker. See docs/QWEN_ENHANCER.md for scope; this is execution validation, not a visual-quality benchmark.

GPU matrix: one-step endpoint, two-step native/neutral exact pixel identity, uncached zero phrase, two references at strengths 0 and 8, LanPaint, Spectrum, cancellation and subsequent worker reuse passed. A separate four-step Turbo refiner with phrase weighting passed. The single-step scheduler defect found during testing is fixed without changing later native schedules. These small 256-pixel execution checks do not establish image quality, all adapters, all GPU vendors or full native gallery acceptance.

# 28 September 2026 — worker reuse and progress

198 CPU/UI checks passed, including Forge saved-default loading. Coverage includes deferred model release, resident-cache reuse, batch/refiner totals and duplicate preview events. Forge was unavailable for a fresh GPU run, so loading-time, VRAM and output-quality improvements remain unmeasured.

# Validation — September 28, 2026

193 automated tests passed in this update, including real Gradio panel construction,
preservation of all 27 script arguments, CFG branch separation, finite forecast fallback,
batch memory release, drift alignment geometry/alpha and output metadata.

The isolated UI was checked in a browser at desktop and narrow widths. Fast mode
updated native steps to eight and selected the turbo adapter. Generation and downloads
were not invoked by the UI preview.

A live dedicated-worker run on the RTX 5090 completed with INT8 ConvRot DiT/text encoder, CFG 2, Spectrum enabled, 512 x 512, 20 steps and seed 987654. It saved a valid PNG and reported 36 real / 4 forecast passes. Test-worker cleanup returned GPU usage to about 1.6 GB. The selected text encoder was qwen3vl_8b_int8_convrot.safetensors. No matched visual-quality comparison or full live Forge gallery test was performed.

The 193-test run included FORGE_ROOT pointing to the installed Forge checkout. The regression used its real UiLoadsave implementation to reproduce Strength=-0.25 and verify the Qwen Turbo slider stays at value 1.0 with range 0–1.5.
Automated tests do not establish universal speed, memory, image-quality or GPU support.
Restart Forge to activate the installed source and new UI.
