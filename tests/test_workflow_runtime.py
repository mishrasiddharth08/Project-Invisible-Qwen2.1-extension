import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_acceptance_cpu as fixtures
from test_registration import load_forge

runtime = fixtures.runtime


class FullWorkerMock:
    def __init__(self):
        self.calls = []
        self.unloaded = 0

    def unload_lora_weights(self):
        self.unloaded += 1

    def __call__(self, prompt, negative_prompt=None, true_cfg_scale=1, width=1024,
                 height=1024, num_inference_steps=40, generator=None, image=None,
                 mask_image=None, use_kv_cache=True, callback_on_step_end=None,
                 output_resolution=None, preview_every=0, status_callback=None,
                 spectrum=False, sigmas=None, refiner="off", lanpaint=False,
                 lanpaint_steps=2, sampler_sharpness=None, phrase_weights=False,
                 reference_priorities="", comfy_sampler="euler",
                 comfy_scheduler="simple", control_model="", control_image=None,
                 control_strength=1.0, control_start=0.0, control_end=1.0,
                 rewrite_prompt=False, prompt_encoder="", system_prompt="",rewrite_thinking=False):
        values = locals().copy()
        values.pop("self")
        self.calls.append(values)
        return types.SimpleNamespace(
            images=[Image.new("RGBA", (width, height), (20, 40, 60, 80))],
            rewritten_prompt=None, spectrum={})


class WorkflowRuntimeTests(unittest.TestCase):
    p = fixtures.RuntimeAcceptanceTests.p
    options = fixtures.RuntimeAcceptanceTests.options
    run_generate = fixtures.RuntimeAcceptanceTests.run_generate

    def setUp(self):
        fixtures.RuntimeAcceptanceTests.setUp(self)
        self.pipe = FullWorkerMock()

    def tearDown(self):
        fixtures.RuntimeAcceptanceTests.tearDown(self)

    def test_full_backend_forwards_cache_and_union_controls(self):
        guide = Image.new("RGB", (64, 64), "white")
        self.run_generate(
            p=self.p(init_images=[guide]),
            backend="comfy", comfy_root="local-comfy", kv_cache=False,
            control_enabled=True, control_model="union.safetensors",
            control_guide=guide, control_strength=1.5, control_start=.1,
            control_end=.9, comfy_sampler="res_2s", comfy_scheduler="beta",
            moire_cleanup=False)
        call = self.pipe.calls[-1]
        self.assertFalse(call["use_kv_cache"])
        self.assertEqual((call["comfy_sampler"], call["comfy_scheduler"]), ("res_2s", "beta"))
        self.assertEqual((call["control_model"], call["control_image"]), ("union.safetensors", guide))
        self.assertEqual((call["control_strength"], call["control_start"], call["control_end"]), (1.5, .1, .9))

    def test_source_proportions_preserve_img2img_ratio(self):
        source = Image.new("RGBA", (640, 384), (1, 2, 3, 77))
        p = self.p([source])
        self.run_generate(p, task="edit", follow_source=True, side=512, moire_cleanup=False)
        self.assertEqual((p.width, p.height), (512, 320))
        self.assertEqual((self.pipe.calls[-1]["width"], self.pipe.calls[-1]["height"]), (512, 320))

    def test_edit_features_fail_closed_on_txt2img(self):
        for values in (dict(task="edit"), dict(follow_source=True), dict(outpaint=True)):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "img2img"):
                self.run_generate(**values)
        self.assertEqual(self.pipe.calls, [])

    def test_workflow_recipe_cannot_drift_to_wrong_tab(self):
        with self.assertRaisesRegex(ValueError, "txt2img"):
            self.run_generate(workflow="UltraEditControl25")
        self.assertEqual(self.pipe.calls, [])

    def test_all_59_ui_controls_reach_runtime_options(self):
        forge = load_forge()
        values = ["edit", 6, 1, "auto", 0, True, False, False, None] + [None] * 9
        values += [False, True, False, "(none)", 1.0, "off", "(none)", False,
                   False, True, False, 2, "(none)", False, 0, False, ""]
        values += ["UltraMergedTurboEdit6", "off", False, 0, 0, 0, 0, 0, 0,
                   True, True, "comfy", "local", False, "(none)", None, 1.0,
                   0.0, 1.0, "euler", "simple", False, "", True]
        self.assertEqual(len(values), 59)
        options = forge.options(
            types.SimpleNamespace(alwayson_scripts=[types.SimpleNamespace(
                _pi_qwen21=True, args_from=0, args_to=59)]),
            types.SimpleNamespace(script_args=values))
        self.assertEqual(options["workflow"], "UltraMergedTurboEdit6")
        self.assertEqual(options["backend"], "comfy")
        self.assertTrue(options["merged_turbo"])

    def test_merged_turbo_rejects_separate_turbo_lora(self):
        with self.assertRaisesRegex(ValueError, "separate Turbo LoRA"):
            self.run_generate(backend="comfy", merged_turbo=True,
                              speed_enabled=True, speed_lora="Viggle Turbo v0.2.1 (6 steps, r256)")
        self.assertEqual(self.pipe.calls, [])

    def test_masked_comfy_edit_preserves_source_rgba_outside_mask(self):
        source = Image.new("RGBA", (64, 64), (180, 20, 10, 37))
        mask = Image.new("L", source.size, 0)
        for x in range(32, 64):
            for y in range(64):
                mask.putpixel((x, y), 255)
        p = self.p([source])
        p.width = p.height = 64
        _p, result = self.run_generate(
            p, task="edit", backend="comfy", mask=mask,
            control_enabled=True, control_model="union.safetensors",
            moire_cleanup=False)
        image = result.images[0]
        self.assertEqual(image.getpixel((4, 4)), source.getpixel((4, 4)))
        self.assertEqual(image.getpixel((4, 4))[3], 37)
        self.assertNotEqual(image.getpixel((60, 4)), source.getpixel((60, 4)))

    def test_missing_backend_or_models_never_triggers_download(self):
        fetches = ["download_union", "download_merged_turbo", "download_selected"]
        patches = [mock.patch.object(runtime.downloads, name) for name in fetches]
        mocks = [item.start() for item in patches]
        try:
            with mock.patch.object(runtime, "resolve", side_effect=ValueError("missing local models")):
                with self.assertRaisesRegex(ValueError, "missing local models"):
                    runtime.generate(self.p(), "selected", self.options(backend="comfy", comfy_root="missing"))
            for called in mocks:
                called.assert_not_called()
        finally:
            for item in reversed(patches):
                item.stop()

    def test_comfy_unsupported_features_fail_before_model_load(self):
        for values in (dict(spectrum=True), dict(refiner="quality")):
            with self.subTest(values=values), mock.patch.object(runtime, "resolve") as resolve:
                with self.assertRaisesRegex(ValueError, "currently requires"):
                    runtime.generate(self.p(), "selected", self.options(backend="comfy", **values))
                resolve.assert_not_called()
