import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_package():
    import pi_qwen21  # Reuse modules already loaded by other test modules.
    return pi_qwen21


load_package()
manager = importlib.import_module("pi_qwen21.download.manager")
runtime = importlib.import_module("pi_qwen21.lib.runtime")


class TurboSigmaScheduleTests(unittest.TestCase):
    """The Viggle card mandates exact sigma nodes; the count must equal steps."""

    def test_exact_6step_card_schedule(self):
        self.assertEqual(manager.turbo_sigmas(6), [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25])

    def test_exact_8step_dense_text_schedule(self):
        self.assertEqual(
            manager.turbo_sigmas(8),
            [1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125])

    def test_count_always_matches_steps_and_descends(self):
        for steps in (1, 2, 3, 4, 5, 6, 7, 9, 12, 20, 40):
            sigmas = manager.turbo_sigmas(steps)
            self.assertEqual(len(sigmas), steps, f"steps={steps}")
            self.assertEqual(sigmas[0], 1.0)
            self.assertTrue(
                all(sigmas[i] > sigmas[i + 1] for i in range(len(sigmas) - 1)),
                f"steps={steps} sigmas={sigmas}")

    def test_last_sigma_ends_at_0_25_for_step_counts_over_two(self):
        for steps in (3, 4, 5, 6, 7, 9, 12, 40):
            self.assertEqual(manager.turbo_sigmas(steps)[-1], 0.25)

    def test_featured_entry_carries_card_steps(self):
        for entry in manager.FEATURED.values():
            self.assertEqual(entry["steps"], 6)
            self.assertEqual(entry["cfg"], 1.0)
            self.assertEqual(entry["strength"], 1.0)


class TurboRuntimeWiringTests(unittest.TestCase):
    """A speed LoRA run must send the card's sigmas to the pipeline."""

    def _run(self, steps, speed_enabled=True):
        calls = []

        class SpeedPipe:
            def unload_lora_weights(self):
                pass

            def __call__(self, **kwargs):
                calls.append(kwargs)
                from PIL import Image
                return types.SimpleNamespace(images=[Image.new("RGB", (8, 8))])

        p = types.SimpleNamespace(
            prompt="x", negative_prompt="blurry", width=1024, height=1024,
            steps=steps, cfg_scale=1.0, seed=1, batch_size=1, n_iter=1,
            init_images=[], do_not_save_samples=True, outpath_samples="",
            override_settings={})
        state = types.SimpleNamespace(
            job_count=0, interrupted=False, skipped=False, nextjob=lambda: None)
        modules_mod = types.ModuleType("modules")
        processing = types.ModuleType("modules.processing")
        processing.Processed = lambda p, images, seed, info, **kw: types.SimpleNamespace(
            info=info, images=images)
        shared = types.ModuleType("modules.shared")
        shared.state = state
        shared.opts = types.SimpleNamespace(samples_save=False)
        images_mod = types.ModuleType("modules.images")
        images_mod.save_image = lambda *a, **kw: None
        modules_mod.processing, modules_mod.shared, modules_mod.images = processing, shared, images_mod
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                OutOfMemoryError=RuntimeError,
                is_available=lambda: True,
                get_device_properties=lambda _i: types.SimpleNamespace(
                    total_memory=24 * 2**30)),
            Generator=lambda device: types.SimpleNamespace(manual_seed=lambda s: None))
        name = list(manager.FEATURED)[0]
        with tempfile_dir() as td:
            lora_file = Path(td) / "featured.safetensors"
            lora_file.write_bytes(b"0123456789")
            with mock.patch.dict(sys.modules, {
                "modules": modules_mod, "modules.processing": processing,
                "modules.shared": shared, "modules.images": images_mod,
                "torch": fake_torch,
             }), mock.patch.object(runtime, "resolve", return_value={"folder": "f", "files": {}}), \
                 mock.patch.object(runtime, "load", return_value=SpeedPipe()), \
                 mock.patch.object(runtime.adapter, "parse", side_effect=lambda t, c: (t, [])), \
                 mock.patch.object(runtime.adapter, "apply"), \
                 mock.patch.object(manager, "featured_path", return_value=lora_file):
                runtime.generate(p, "selected", dict(
                    task="t2i", steps=steps, true_cfg=1, profile="24", side=1024,
                    offload=True, community=False, consent=False, mask=None, refs=[],
                    speed_enabled=speed_enabled, speed_lora=name if speed_enabled else "(none)",
                    speed_strength=1.0))
        return calls

    def test_speed_run_passes_exact_6_sigma_schedule(self):
        calls = self._run(6)
        self.assertEqual(calls[0]["sigmas"], [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25])
        self.assertEqual(calls[0]["num_inference_steps"], 6)

    def test_custom_step_count_gets_matching_sigma_count(self):
        calls = self._run(10)
        self.assertEqual(len(calls[0]["sigmas"]), 10)

    def test_no_speed_lora_means_no_sigmas(self):
        calls = self._run(40, speed_enabled=False)
        self.assertNotIn("sigmas", calls[0])


class tempfile_dir:
    def __enter__(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        return self._tmp.name

    def __exit__(self, *exc):
        self._tmp.cleanup()
        return False


if __name__ == "__main__":
    unittest.main()
