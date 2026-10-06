import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_qwen_comfy_worker", ROOT / "lib/comfy_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FakeBase:
    __module__ = "comfy.ldm.qwen_image21.model"
    def __init__(self):
        self.transformer_blocks = [object()] * 32


class FakeModel:
    def get_model_object(self, name):
        self.requested = name
        return FakeBase()


class QwenImage21FunControl:
    __module__ = "comfy.ldm.qwen_image21.model"
    def __init__(self):
        self.control_blocks = [object()] * 16


class WorkerContractTests(unittest.TestCase):
    def test_component_aliases_and_path_checks(self):
        bundle = {"files": {"dit": "a", "te": "b", "vae": "c"}}
        self.assertEqual(worker._component(bundle, "transformer", "dit"), "a")
        self.assertEqual(worker._component(bundle, "text_encoder", "te"), "b")
        with self.assertRaisesRegex(ValueError, "Missing model"):
            worker._path(None, "model")

    def test_qwen21_model_and_control_contracts(self):
        worker._validate_qwen21_model(FakeModel())
        worker._validate_qwen21_control(types.SimpleNamespace(model=QwenImage21FunControl()))
        broken = FakeModel()
        broken.get_model_object = lambda _name: types.SimpleNamespace(transformer_blocks=[0] * 31)
        with self.assertRaisesRegex(ValueError, "32-block"):
            worker._validate_qwen21_model(broken)
        with self.assertRaisesRegex(ValueError, "16 blocks"):
            worker._validate_qwen21_control(types.SimpleNamespace(model=types.SimpleNamespace(control_blocks=[])))

    def test_qwen21_vae_contract_is_strict_rgba64(self):
        valid = types.SimpleNamespace(latent_channels=64, output_channels=4,
                                      spacial_compression_decode=lambda: 16)
        worker._validate_qwen21_vae(valid)
        for invalid in (
            types.SimpleNamespace(latent_channels=16, output_channels=3,
                                  spacial_compression_decode=lambda: 8),
            types.SimpleNamespace(latent_channels=64, output_channels=2,
                                  spacial_compression_decode=lambda: 16),
        ):
            with self.assertRaisesRegex(ValueError, "not Qwen-Image-2.1 RGBA"):
                worker._validate_qwen21_vae(invalid)

    def test_node_output_contract(self):
        self.assertEqual(worker._node_values(types.SimpleNamespace(result=(1, 2))), (1, 2))
        self.assertEqual(worker._node_values([1]), (1,))
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            worker._node_values(3)

    def test_unsupported_features_fail_instead_of_silent_fallback(self):
        for command in ({"spectrum": True}, {"lanpaint": True}, {"refiner": "fast"}):
            with self.assertRaisesRegex(ValueError, "does not support"):
                worker._reject_unavailable(command)
        worker._reject_unavailable({})

    def test_sampling_heartbeat_reports_actual_state_only(self):
        self.assertEqual(worker._sampling_status(3, 25, 10.0, now=14.9),
                         "sampling 3/25 — 4s")

    def test_absolute_local_files_only(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors"
            path.write_bytes(b"local")
            self.assertEqual(worker._path(path, "model"), path.resolve())
            with self.assertRaisesRegex(ValueError, "does not exist"):
                worker._path(path.with_name("missing"), "model")

    def test_prompt_generator_uses_accelerator_even_with_offload(self):
        device = types.SimpleNamespace(type="cuda")
        clip = types.SimpleNamespace(patcher=types.SimpleNamespace(load_device="cpu"))
        manager = types.SimpleNamespace(get_torch_device=lambda: device)
        self.assertIs(worker._prepare_text_generator(clip, manager), device)
        self.assertIs(clip.patcher.load_device, device)
        cpu = types.SimpleNamespace(type="cpu")
        with self.assertRaisesRegex(RuntimeError, "requires an accelerator"):
            worker._prepare_text_generator(clip, types.SimpleNamespace(get_torch_device=lambda: cpu))

    def test_prompt_enhancer_uses_official_sampling_contract(self):
        length, sampling, thinking = worker._enhancer_options({"seed": 7})
        self.assertEqual(length, 4096)
        self.assertFalse(thinking)
        self.assertEqual(sampling, {"sampling_mode": "off"})
        length, sampling, thinking = worker._enhancer_options({"seed": 7, "rewrite_thinking": True})
        self.assertTrue(thinking)
        self.assertEqual(sampling, {
            "sampling_mode": "on", "temperature": 1.0, "top_k": 20,
            "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0,
            "presence_penalty": 1.5, "seed": 7,
        })
        self.assertEqual(worker._enhancer_options({"prompt_enhancer_max_length": 512})[0], 512)
        with self.assertRaisesRegex(ValueError, "between 64 and 16256"):
            worker._enhancer_options({"prompt_enhancer_max_length": 32})

    def test_prompt_enhancer_full_loads_only_when_private_budget_fits(self):
        calls = []
        patcher = types.SimpleNamespace(load_device=types.SimpleNamespace(type="cuda"),
                                        model_size=lambda: 10 * 2**30)
        clip = types.SimpleNamespace(patcher=patcher)
        manager = types.SimpleNamespace(
            get_free_memory=lambda _device: 13 * 2**30,
            load_models_gpu=lambda *args, **kwargs: calls.append((args, kwargs)))
        self.assertTrue(worker._maybe_full_load_text_generator(clip, manager))
        self.assertTrue(calls[0][1]["force_full_load"])
        manager.get_free_memory = lambda _device: 8 * 2**30
        self.assertFalse(worker._maybe_full_load_text_generator(clip, manager))
        self.assertEqual(len(calls), 1)

    def test_prompt_rewrite_stop_only_interrupts_private_worker(self):
        calls = []
        manager = types.SimpleNamespace(interrupt_current_processing=lambda value: calls.append(value))
        with tempfile.TemporaryDirectory() as td:
            stop = Path(td) / "stop"
            self.assertFalse(worker._check_rewrite_stop(stop, manager))
            stop.touch()
            self.assertTrue(worker._check_rewrite_stop(stop, manager))
        self.assertEqual(calls, [True])

    def test_comfy_cuda_free_memory_is_capped_to_worker_budget(self):
        cuda = types.SimpleNamespace(type="cuda")
        cpu = types.SimpleNamespace(type="cpu")
        manager = types.SimpleNamespace(
            get_torch_device=lambda: cuda,
            get_free_memory=lambda dev=None, torch_free_too=False:
                (12 * 2**30, 3 * 2**30) if torch_free_too else 12 * 2**30,
        )
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            memory_allocated=lambda _dev: 2 * 2**30))
        worker._cap_comfy_free_memory(manager, torch, 8)
        self.assertEqual(manager.get_free_memory(), 6 * 2**30)
        self.assertEqual(manager.get_free_memory(torch_free_too=True),
                         (6 * 2**30, 3 * 2**30))
        self.assertEqual(manager.get_free_memory(cpu, True),
                         (12 * 2**30, 3 * 2**30))

    def test_idle_release_clears_native_cache_and_gpu_allocator(self):
        calls = []
        diffusion = types.SimpleNamespace(reset_prefix_cache=lambda enabled: calls.append(("cache", enabled)))
        model = types.SimpleNamespace(get_model_object=lambda _name: diffusion)
        worker._clear_qwen_prefix_cache(model)
        manager = types.SimpleNamespace(
            unload_all_models=lambda: calls.append("unload"),
            soft_empty_cache=lambda: calls.append("soft"))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True, empty_cache=lambda: calls.append("cuda")))
        worker._idle_offload(manager, torch)
        self.assertEqual(calls, [("cache", False), "unload", "soft", "cuda"])

    def test_emit_protocol_is_one_json_line(self):
        old = worker.WIRE
        stream = types.SimpleNamespace(data="", write=lambda value: None, flush=lambda: None)
        class Wire:
            def __init__(self): self.data = ""
            def write(self, value): self.data += value
            def flush(self): pass
        wire = Wire()
        try:
            worker.WIRE = wire
            worker.emit("result", path="x.png")
        finally:
            worker.WIRE = old
        self.assertEqual(json.loads(wire.data), {"type": "result", "path": "x.png"})
        self.assertEqual(wire.data.count("\n"), 1)

    def test_mask_composite_preserves_source_outside_regeneration(self):
        import torch
        generated = torch.ones((1, 2, 2, 4))
        source = torch.zeros((1, 2, 2, 4))
        source[..., 3] = 0.25
        mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        result = worker._composite(generated, source, mask, torch)
        self.assertTrue(torch.equal(result[0, 0, 0], generated[0, 0, 0]))
        self.assertTrue(torch.equal(result[0, 0, 1], source[0, 0, 1]))
        self.assertEqual(float(result[0, 0, 1, 3]), 0.25)

    def test_native_preview_uses_exact_qwen64_contract(self):
        factors = [[float(i), 0.0, 0.0] for i in range(64)]
        bias = [0.1, 0.2, 0.3]
        reshape = lambda value: value
        latent_format = types.SimpleNamespace(
            latent_channels=64, latent_rgb_factors=factors,
            latent_rgb_factors_bias=bias, latent_rgb_factors_reshape=reshape)
        model = types.SimpleNamespace(model=types.SimpleNamespace(latent_format=latent_format))
        captured = {}
        class Previewer:
            def __init__(self, got_factors, got_bias, got_reshape):
                captured.update(factors=got_factors, bias=got_bias, reshape=got_reshape)
        result = worker._native_previewer(model, types.SimpleNamespace(Latent2RGBPreviewer=Previewer))
        self.assertIsInstance(result, Previewer)
        self.assertIs(captured["factors"], factors)
        self.assertIs(captured["bias"], bias)
        self.assertIs(captured["reshape"], reshape)
        latent_format.latent_channels = 16
        with self.assertRaisesRegex(ValueError, "64-channel"):
            worker._native_previewer(model, types.SimpleNamespace(Latent2RGBPreviewer=Previewer))

    def test_preview_write_is_png_and_atomic(self):
        import torch
        from PIL import Image
        class Previewer:
            def decode_latent_to_preview(self, value):
                self.detached = not value.requires_grad
                return Image.new("RGB", (128, 64), "red")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "preview.png"
            previewer = Previewer()
            worker._write_preview(previewer, torch.ones((1, 64, 2, 2), requires_grad=True), path, 64)
            self.assertTrue(previewer.detached)
            with Image.open(path) as saved:
                self.assertEqual(saved.format, "PNG")
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            with self.assertRaisesRegex(ValueError, "64-channel"):
                worker._write_preview(previewer, torch.ones((1, 16, 2, 2)), path, 64)

    def test_cuda_budget_skips_cpu_and_rejects_exhausted_profile(self):
        cpu = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        self.assertEqual(worker._cuda_budget(cpu, {"vram_gb": 8}), 0.0)
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _i: types.SimpleNamespace(total_memory=8 * 2**30),
            mem_get_info=lambda _i: (1 * 2**30, 8 * 2**30),
            set_per_process_memory_fraction=lambda *_a: None,
        )
        with self.assertRaisesRegex(RuntimeError, "no free"):
            worker._cuda_budget(types.SimpleNamespace(cuda=cuda), {"vram_gb": 4})

    def test_prompt_rewrite_json_is_strict_data(self):
        value = worker._rewrite_data(
            '{"rewritten_prompt":"Detailed cat","wh_ratio":"16:9","ratio_follow":""}',
            "cat")
        self.assertEqual(value["prompt"], "Detailed cat")
        self.assertEqual(worker._ratio_size(1024, 1024, "16:9"), (1376, 768))
        for invalid in ("not json", "[]", '{"rewritten_prompt":""}',
                        '{"rewritten_prompt":"x","ratio_follow":"yes"}',
                        '{"rewritten_prompt":"x","wh_ratio":"1:1","ratio_follow":"<image1>"}'):
            with self.assertRaises(ValueError):
                worker._rewrite_data(invalid, "x")
        follow = worker._rewrite_data(
            '{"rewritten_prompt":"x","wh_ratio":"","ratio_follow":"<image10>"}', "x")
        self.assertEqual(follow["ratio_follow"], "<image10>")

    def test_ratio_follow_requires_the_named_reference(self):
        import torch
        refs = [torch.zeros((1, 640, 960, 4))]
        self.assertEqual(worker._reference_size(refs, "<image1>"), (960, 640))
        with self.assertRaisesRegex(ValueError, "missing"):
            worker._reference_size(refs, "<image2>")

    def test_base_and_prompt_encoders_cannot_be_swapped(self):
        pe = {"lm_head.weight", "model.language_model.embed_tokens.weight"}
        pe.update(f"model.language_model.layers.{i}.input_layernorm.weight" for i in range(32))
        pe_shapes = {"lm_head.weight": (248320, 4096),
                     "model.language_model.embed_tokens.weight": (248320, 4096)}
        self.assertEqual(worker._encoder_role(pe, pe_shapes.get), "prompt_encoder")
        base = {"model.embed_tokens.weight", "model.visual.patch_embed.proj.weight"}
        base.update(f"model.layers.{i}.input_layernorm.weight" for i in range(36))
        self.assertEqual(worker._encoder_role(
            base, {"model.embed_tokens.weight": (151936, 4096)}.get), "qwen3vl_8b")
        self.assertEqual(worker._encoder_role(
            pe, {**pe_shapes, "lm_head.weight": (151936, 4096)}.get), "unknown")

    def test_control_weights_cache_by_file_identity(self):
        calls = []
        class Loader:
            def load_model_patch(self, name):
                calls.append(name)
                return (types.SimpleNamespace(model=QwenImage21FunControl()),)
        class Apply:
            def diffsynth_controlnet(self, model, *_args, **_kwargs):
                return (model,)
        backend = types.SimpleNamespace(
            register=lambda _category, path: Path(path).name,
            patch_loader=Loader, control_apply=Apply)
        session = object.__new__(worker.Session)
        session.backend = backend
        session.vae = object()
        session.control_patches = {}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "control.safetensors"
            path.write_bytes(b"control")
            command = {"control_model": str(path), "control_strength": 1,
                       "control_start": 0, "control_end": 1}
            model = object()
            self.assertIs(session._control_model(model, command, object(), [], None), model)
            self.assertIs(session._control_model(model, command, object(), [], None), model)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(session.control_patches), 1)

    def test_res4lyf_scheduler_restore_keeps_native_objects(self):
        native_handler = object()
        native_handlers = {"beta": object()}
        native_names = ["simple", "beta"]
        native_alias = native_names
        native_class_names = native_names
        samplers = types.SimpleNamespace(
            SchedulerHandler=native_handler, SCHEDULER_HANDLERS=native_handlers,
            SCHEDULER_NAMES=native_names, SCHEDULERS=native_alias,
            KSampler=types.SimpleNamespace(SCHEDULERS=native_class_names))
        snapshot = worker._scheduler_snapshot(samplers)
        samplers.SchedulerHandler = object()
        samplers.SCHEDULER_HANDLERS = {}
        samplers.SCHEDULER_NAMES = []
        samplers.SCHEDULERS = []
        samplers.KSampler.SCHEDULERS = []
        worker._restore_schedulers(samplers, snapshot)
        self.assertIs(samplers.SchedulerHandler, native_handler)
        self.assertIs(samplers.SCHEDULER_HANDLERS, native_handlers)
        self.assertIs(samplers.SCHEDULER_NAMES, native_names)
        self.assertIs(samplers.SCHEDULERS, native_alias)
        self.assertIs(samplers.KSampler.SCHEDULERS, native_class_names)

    def test_scheduler_restore_handles_missing_legacy_global_alias(self):
        samplers = types.SimpleNamespace(
            SchedulerHandler=object(), SCHEDULER_HANDLERS={"beta": object()},
            SCHEDULER_NAMES=["beta"], KSampler=types.SimpleNamespace(SCHEDULERS=["beta"]))
        snapshot = worker._scheduler_snapshot(samplers)
        samplers.SCHEDULERS = []
        worker._restore_schedulers(samplers, snapshot)
        self.assertFalse(hasattr(samplers, "SCHEDULERS"))
        self.assertIn("beta", samplers.SCHEDULER_HANDLERS)

    def test_comfy_sigma_schedule_adds_only_terminal_zero(self):
        import torch
        sigmas = worker._sigma_tensor([1.0, .9375, .875, .75, .5, .25], 6, torch)
        self.assertEqual(sigmas.tolist(), [1.0, .9375, .875, .75, .5, .25, 0.0])
        with self.assertRaisesRegex(ValueError, "one positive"):
            worker._sigma_tensor([1.0], 2, torch)
        with self.assertRaisesRegex(ValueError, "descending"):
            worker._sigma_tensor([1.0, 1.0], 2, torch)


if __name__ == "__main__":
    unittest.main()
