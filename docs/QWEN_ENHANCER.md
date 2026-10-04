# Reference priorities and weighted phrases — October 4, 2026

These optional controls adapt the two ideas in [capitan01R's Qwen 2.1 enhancer](https://github.com/capitan01R/ComfyUI-qwen_img_2_1_enhancer) to the extension's existing Diffusers worker. They do not require ComfyUI or additional packages.

## Weighted prompt phrases

1. Enable **Qwen → Finish → Weighted prompt phrases**.
2. Use explicit weights in the normal prompt, for example `Add (warm lighting:1.3) and (soft shadows:1.1).`
3. Negative prompts accept the same syntax independently. Native CFG must exceed 1 with a real negative prompt for the negative branch to be used.

Only the marked occurrence is weighted. Complete words or phrases work best; nested sections and textual-inversion embeddings are unsupported. A phrase that crosses a tokenizer boundary fails with a clear error rather than weighting unrelated tokens.

Weight **1** is native priority. Larger values increase priority, values between **0 and 1** reduce it, and **0** strongly suppresses attention to those text tokens. There is no promise that weight 2 produces twice the visible effect. Weighting is off by default.

## Reference priorities

In **img2img → Qwen → Edit → Reference priorities**, enter:

```text
1:1.2, 2:0.8
```

This increases priority for the main image and reduces priority for the next connected reference. Empty reference slots are skipped. Numbers follow the actual connected-image order, starting at 1. Leave the field blank for normal behavior. Strengths range from 0 to 8; start close to 1. Repeating an image number replaces its earlier setting.

Strength **0** blocks direct attention to that reference's latent tokens. Reference information can remain in the text conditioning, so this does not erase the image's influence completely. This control is not a pixel-preservation lock; use LanPaint or existing background controls when appropriate.

## Implementation research and safeguards

- Inspected upstream README and both node implementations at commit `e7204ab908526381c67cd8276adc7aa2e9581c95`.
- Both upstream nodes add `log(weight)` to selected attention scores before softmax. Text embeddings and source pixels are not multiplied by the weights. The bridge follows that method.
- Phrase positions are mapped against the actual processor token IDs and the retained Qwen text suffix, including reference-image token expansion and the native encoder trim. Positive and negative branches have independent maps.
- Bias applies only to generated-image query rows. Prefix queries keep their native block-causal masks, so prefix-KV caches remain valid. Padding masks remain effective.
- Per-call bias storage is a small broadcast vector over keys rather than a full square target-attention matrix. The attention backend must accept additive masks, and weighted calls can be slower than ordinary attention.
- Pipeline encoding wrappers, attention processors and transformer hooks are confined to the current worker request and restored on failure, cancellation or completion.
- Unsupported processor/token layouts fail explicitly. Protected Head Swap cannot be combined with these controls in this release.
- LanPaint can use the same weighted attention on its inner model calls. Spectrum retains its existing approximate behavior. The optional refiner re-applies phrase weighting when it re-encodes the prompt. Reference priorities apply to the base edit only; the refiner's latent-start pass does not carry reference-image keys.
- All features are optional and the existing Generate workflow remains authoritative.

## Verification and limits

- 247 automated tests run: **246 passed, one skipped** because its optional environment was unavailable.
- Regression checks cover exact marked occurrences, token-boundary rejection, invalid weights, reference ranges, zero-weight masking, normalized attention odds, padding, cleanup and all 35 saved UI positions.
- Real RTX 5090 test used existing INT8 ConvRot DiT, BF16 text encoder, a 256 × 256 reference, two steps, CFG 2 and seed 42. Positive `blue ceramic:1.3`, negative `blurry:0.8` and reference `1:1.2` completed with native prefix caching.
- A subsequent unweighted txt2img request succeeded in the same worker, confirming recovery to the ordinary path.
- GPU boundary tests also passed: one-step scheduler endpoint; exact native/neutral pixel identity at two steps; uncached zero phrase; two references with priorities 0 and 8; LanPaint; Spectrum; cancellation and recovery. A separate four-step Turbo refiner run passed with phrase weighting.
- Fixed one-step terminal stretching that produced non-finite sigmas. The request-only override leaves later 2/40-step schedules and exact speed-adapter sigma paths independent.
- The two-step execution test is not a visual-quality benchmark. Multiple-reference quality, every adapter and all hardware combinations remain unverified. Compare one setting at a time using the same seed.

## Credits

The adapter is inspired by capitan01R's MIT-licensed work. The original copyright and license are preserved in `lib/vendor/qwen_enhancer/LICENSE`. Existing model and third-party licenses remain applicable; prior acknowledgments are preserved.
