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
