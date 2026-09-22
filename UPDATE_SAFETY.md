# Keeping Project Invisible separate from Forge

## What is isolated

- Extension source lives in its own folder. No Forge core files are rewritten.
- Inference runs in a dedicated worker using the existing Forge Python interpreter.
- Extra dependencies install into the extension's `_deps` folder using `--target --no-deps`; the installer does not upgrade Forge's shared packages.
- The dependency list pins the pipeline revision and bounds other package versions.
- Model downloads require explicit selection and approval. Keep model weights in Forge's model folders.
- Runtime hooks route only matching checkpoints; other checkpoints use the original Forge functions.

## What cannot be guaranteed

Forge still supplies Python, PyTorch, the UI and integration hooks. Updates to those components can break compatibility even if every extension file remains unchanged. Existing hook checks handle known missing interfaces, but cannot predict every future change. No software can protect itself from deliberate folder deletion or every external updater.

## Before updating Forge

1. Stop Forge.
2. Copy this entire extension folder to a backup **outside the Forge folder**. Include `_deps` if you want to preserve installed dependencies.
3. Keep a copy of the working Forge version as well; restoring only the extension cannot undo changes to Forge's Python or PyTorch.
4. Update Forge separately. Do not select this extension for an extension update unless you want a new extension version.
5. Start Forge and test one small text-to-image generation without optional adapters or acceleration.

## Check that source files are unchanged

Run `verify_integrity.py` with Forge's Python interpreter, or ask someone to run it for you. It only reads files and compares them with `MANIFEST.json`. It never repairs, downloads or deletes anything. Dependencies, settings changed after installation, generated files and extra files are not a complete environment audit.

If checks fail, compare with your backup before replacing files. If Forge no longer works, stop it and restore the known-working Forge and extension copies. Report the complete terminal error through GitHub Issues.
