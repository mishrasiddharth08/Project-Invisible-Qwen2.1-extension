# Changelog

## 2026-09-23 — follow-up

- Empty negative prompts now use effective CFG 1 instead of raising an error; native CFG with a real negative prompt remains unchanged.
- Regression suite: 87 tests passed.

## 2026-09-23

- Ordinary LoRA adapters now use additive hooks on quantized linear layers instead of incompatible PEFT replacement.
- One native Forge CFG slider controls generation; the legacy saved argument slot remains hidden for compatibility.
- Release a failed inference pipeline before retrying.
- Portable Forge-root fallback; no author-specific drive path in release configuration.
- Add source integrity verification and update/backup instructions.
- Publish readable extension source at repository root, replacing the old RAR distribution. The old archive remains available through Git history.
- First public release testing claim remains text-to-image only; other modes are experimental or unconfirmed.
