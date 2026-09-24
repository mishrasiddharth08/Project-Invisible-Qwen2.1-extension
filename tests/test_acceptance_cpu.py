import importlib
import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_package():
    """Load the extension under the same namespace used by Forge."""
    for name in tuple(sys.modules):
        if name == "pi_qwen21" or name.startswith("pi_qwen21."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "pi_qwen21", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["pi_qwen21"] = package
    spec.loader.exec_module(package)
    return package


load_package()
assets = importlib.import_module("pi_qwen21.lib.assets")
prompts = importlib.import_module("pi_qwen21.lib.prompts")
adapter = importlib.import_module("pi_qwen21.lora.adapter")
runtime = importlib.import_module("pi_qwen21.lib.runtime")
manager = importlib.import_module("pi_qwen21.download.manager")
preset = importlib.import_module("pi_qwen21.lib.preset")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_safetensors(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"12345678")


class AssetAcceptanceTests(unittest.TestCase):
    def test_bucket_selects_ratio_and_honors_long_side(self):
        self.assertEqual(assets.bucket(1000, 1000), (2048, 2048))
        self.assertEqual(assets.bucket(1600, 900), (2752, 1536))
        w, h = assets.bucket(1600, 900, 1024)
        self.assertEqual(max(w, h), 1024)
        self.assertEqual((w % 32, h % 32), (0, 0))

    def test_profile_contract(self):
        self.assertEqual(assets.profile(8)["te"], "w4a8")
        self.assertEqual(assets.profile(12)["dit"], "int8_convrot")
        self.assertEqual(assets.profile(16)["side"], 1536)
        self.assertEqual(assets.profile(24)["dit"], "bf16")
        self.assertEqual(assets.profile(32)["side"], 0)
        self.assertEqual(assets.profile(32, "8")["te"], "w4a8")

    def _identity_tree(self, root, te_type="qwen3_vl", vae_class="AutoencoderKLQwenImage21"):
        write_json(root / "model_index.json", {"_class_name": "QwenImage21Pipeline"})
        write_json(root / "transformer/config.json", {"_class_name": "QwenImage21Transformer2DModel"})
        write_json(root / "vae/config.json", {"_class_name": vae_class})
        write_json(root / "text_encoder/config.json", {"model_type": te_type})

    def test_identity_accepts_only_qwen_image_21_components(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._identity_tree(root)
            self.assertTrue(assets.identity(root))

            self._identity_tree(root, te_type="qwen2_5_vl")
            self.assertFalse(assets.identity(root), "old text encoder must be rejected")

            self._identity_tree(root, vae_class="AutoencoderKLQwenImage")
            self.assertFalse(assets.identity(root), "old Qwen VAE must be rejected")

    def test_is_dit_file_recognizes_renamed_community_weights(self):
        qwen_keys = {
            "transformer_blocks.0.attn.to_q.weight": {"dtype": "BF16", "shape": [1, 1]},
            "transformer_blocks.0.img_mlp.gate_up.weight": {"dtype": "BF16", "shape": [1, 1]},
        }
        with tempfile.TemporaryDirectory() as td:
            renamed = Path(td) / "qwenImage21INT8INT4_int8.safetensors"
            write_safetensors(renamed, qwen_keys)
            self.assertTrue(assets.is_dit_file(renamed), "renamed community DiT must be recognized by contents")

            lora_keys = {k: v for k, v in qwen_keys.items()}
            lora_keys["transformer_blocks.0.attn.to_q.lora_down.weight"] = {"dtype": "BF16", "shape": [1, 1]}
            adapter_file = Path(td) / "some_lora.safetensors"
            write_safetensors(adapter_file, lora_keys)
            self.assertFalse(assets.is_dit_file(adapter_file), "adapter files must not be treated as DiT weights")

            flux_keys = {"double_blocks.0.img_attn.qkv.weight": {"dtype": "BF16", "shape": [1, 1]}}
            flux_file = Path(td) / "flux_model.safetensors"
            write_safetensors(flux_file, flux_keys)
            self.assertFalse(assets.is_dit_file(flux_file), "foreign architectures must be rejected")

            junk = Path(td) / "junk.safetensors"
            junk.write_bytes(b"not safetensors")
            self.assertFalse(assets.is_dit_file(junk), "corrupt files must be rejected")

    def test_scan_includes_renamed_dit_in_standard_folders(self):
        keys = {
            "transformer_blocks.0.attn.to_q.weight": {"dtype": "BF16", "shape": [1, 1]},
            "transformer_blocks.0.img_mlp.gate_up.weight": {"dtype": "BF16", "shape": [1, 1]},
        }
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            renamed = base / "Stable-diffusion" / "qwenImage21INT8INT4_int8.safetensors"
            write_safetensors(renamed, keys)
            with mock.patch.object(assets, "models_root", return_value=base), \
                 mock.patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=lambda *a, **k: (_ for _ in ()).throw(OSError()))}):
                inventory = assets.scan()
            self.assertIn(str(renamed.resolve()), inventory["dit"])


class PromptRewriteAcceptanceTests(unittest.TestCase):
    def test_plain_prompt_passes_through(self):
        self.assertEqual(prompts.parse("a red fox", 640, 480), ("a red fox", 640, 480))

    def test_official_json_is_consumed(self):
        raw = json.dumps({"rewritten_prompt": "a detailed red fox", "wh_ratio": "16:9"})
        self.assertEqual(prompts.parse(raw, 640, 480), ("a detailed red fox", 2752, 1536))

    def test_bad_json_shape_and_ratio_are_guarded(self):
        unchanged = '{"something_else": true}'
        self.assertEqual(prompts.parse(unchanged, 640, 480), (unchanged, 640, 480))
        with self.assertRaises(ValueError):
            prompts.parse('{"rewritten_prompt":"x","wh_ratio":"banana"}', 640, 480)
        with self.assertRaises(ValueError):
            prompts.parse('{"rewritten_prompt":"   ","wh_ratio":"1:1"}', 640, 480)


class DownloadAcceptanceTests(unittest.TestCase):
    def test_bundled_support_unblocks_existing_manual_weights_without_network(self):
        hub=self.hub_module()
        inv=self.inventory(dit=['/local/qwen_image_2.1_int8_convrot.safetensors'],te=['/local/qwen3vl_8b_bf16.safetensors'],vae=['/local/qwen_image_2.1_vae_bf16.safetensors'])
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules,{'huggingface_hub':hub}), mock.patch.object(manager,'scan',return_value=inv), mock.patch.object(manager,'models_root',return_value=Path(td)):
            bundle=manager.resolve(None,False,{'dit':'bf16','te':'bf16'})
        self.assertEqual(Path(bundle['folder']).name,'qwen21')
        self.assertEqual(bundle['files']['text_encoder'],inv['te'][0])
        hub.snapshot_download.assert_not_called()
        hub.hf_hub_download.assert_not_called()

    def test_selected_download_requires_permission(self):
        with mock.patch.object(manager,'scan') as scan:
            with self.assertRaisesRegex(ValueError,'authorize'):
                manager.download_selected(['qwen_image_2.1_bf16.safetensors'],False)
            scan.assert_not_called()

    def test_selected_download_fetches_only_requested_file(self):
        hub=self.hub_module()
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules,{'huggingface_hub':hub}), mock.patch.object(manager,'scan',return_value=self.inventory()), mock.patch.object(manager,'models_root',return_value=Path(td)):
            manager.download_selected(['qwen3vl_8b_w4a8.safetensors'],True)
        self.assertEqual(hub.hf_hub_download.call_count,1)
        self.assertTrue(hub.hf_hub_download.call_args.args[1].endswith('qwen3vl_8b_w4a8.safetensors'))
        hub.snapshot_download.assert_not_called()

    def test_selected_existing_file_never_fetches(self):
        hub=self.hub_module()
        with mock.patch.dict(sys.modules,{'huggingface_hub':hub}), mock.patch.object(manager,'scan',return_value=self.inventory(dit=['/local/qwen_image_2.1_bf16.safetensors'])):
            manager.download_selected(['qwen_image_2.1_bf16.safetensors'],True)
        hub.hf_hub_download.assert_not_called()
        hub.snapshot_download.assert_not_called()

    def inventory(self, **changes):
        value = {"dit": [], "te": [], "vae": [], "lora": [], "pipelines": []}
        value.update(changes)
        return value

    def hub_module(self):
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = mock.Mock()
        hub.hf_hub_download = mock.Mock()
        hub.list_repo_files = mock.Mock()
        return hub

    def test_complete_selected_official_snapshot_never_downloads(self):
        hub = self.hub_module()
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
             mock.patch.object(manager, "scan", return_value=self.inventory()), \
             mock.patch.object(manager, "complete", return_value=True):
            selected = str(Path(td) / "model_index.json")
            result = manager.resolve(selected, True, {"dit": "bf16", "te": "bf16"})
        self.assertEqual(result, {"folder": td, "files": {}})
        hub.snapshot_download.assert_not_called()
        hub.hf_hub_download.assert_not_called()
        hub.list_repo_files.assert_not_called()

    def test_all_components_and_metadata_avoid_network(self):
        hub = self.hub_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            official = root / "Qwen-Image-2.1" / "official"
            (official / "processor").mkdir(parents=True)
            (official / "scheduler").mkdir(parents=True)
            (official / "processor/tokenizer.json").write_text("{}")
            (official / "scheduler/scheduler_config.json").write_text("{}")
            paths = {
                "dit": str(root / "qwen_image_2.1_bf16.safetensors"),
                "te": str(root / "qwen3vl_8b_bf16.safetensors"),
                "vae": str(root / "qwen_image_2.1_vae_bf16.safetensors"),
            }
            inv = self.inventory(**{k: [v] for k, v in paths.items()})
            with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
                 mock.patch.object(manager, "scan", return_value=inv), \
                 mock.patch.object(manager, "models_root", return_value=root), \
                 mock.patch.object(manager, "identity", return_value=True):
                result = manager.resolve(None, True, {"dit": "bf16", "te": "bf16"})
        self.assertEqual(result["files"], {
            "transformer": paths["dit"], "text_encoder": paths["te"], "vae": paths["vae"]
        })
        hub.snapshot_download.assert_not_called()
        hub.hf_hub_download.assert_not_called()
        hub.list_repo_files.assert_not_called()

    def test_missing_without_consent_errors_before_hub_calls(self):
        hub = self.hub_module()
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
             mock.patch.object(manager, "scan", return_value=self.inventory()), \
             mock.patch.object(manager, "models_root", return_value=Path(td)), \
             mock.patch.object(manager, "identity", return_value=False):
            with self.assertRaisesRegex(ValueError, "Missing assets"):
                manager.resolve(None, False, {"dit": "int8_convrot", "te": "w4a8"})
        hub.snapshot_download.assert_not_called()
        hub.hf_hub_download.assert_not_called()
        hub.list_repo_files.assert_not_called()

    def test_partial_download_fetches_only_missing_and_second_resolve_fetches_nothing(self):
        hub = self.hub_module()
        remote = [
            "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
            "text_encoders/qwen3vl_8b_w4a8.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ]
        hub.list_repo_files.return_value = remote
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dit = root / "existing" / "qwen_image_2.1_int8_convrot.safetensors"
            dit.parent.mkdir(); dit.write_bytes(b"local")
            state = {"te": None, "vae": None, "metadata": False}

            def inventory():
                return self.inventory(
                    dit=[str(dit)],
                    te=[state["te"]] if state["te"] else [],
                    vae=[state["vae"]] if state["vae"] else [],
                )

            def snapshot(*args, **kwargs):
                state["metadata"] = True
                folder = Path(kwargs["local_dir"])
                (folder / "processor").mkdir(parents=True, exist_ok=True)
                (folder / "scheduler").mkdir(parents=True, exist_ok=True)
                (folder / "processor/tokenizer.json").write_text("{}")
                (folder / "scheduler/scheduler_config.json").write_text("{}")
                return str(folder)

            def fetch(_repo, filename, **kwargs):
                path = Path(kwargs["local_dir"]) / Path(filename).name
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fetched")
                if "qwen3vl" in filename: state["te"] = str(path)
                if "vae" in filename: state["vae"] = str(path)
                return str(path)

            hub.snapshot_download.side_effect = snapshot
            hub.hf_hub_download.side_effect = fetch
            with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
                 mock.patch.object(manager, "scan", side_effect=inventory), \
                 mock.patch.object(manager, "models_root", return_value=root), \
                 mock.patch.object(manager, "identity", side_effect=lambda _p: state["metadata"]):
                first = manager.resolve(None, True, {"dit": "int8_convrot", "te": "w4a8"})
                self.assertEqual(hub.hf_hub_download.call_count, 2)
                fetched_names = {call.args[1] for call in hub.hf_hub_download.call_args_list}
                self.assertNotIn(remote[0], fetched_names)
                self.assertEqual(fetched_names, set(remote[1:]))
                hub.snapshot_download.reset_mock(); hub.hf_hub_download.reset_mock(); hub.list_repo_files.reset_mock()
                second = manager.resolve(None, True, {"dit": "int8_convrot", "te": "w4a8"})

        self.assertEqual(first, second)
        hub.snapshot_download.assert_not_called()
        hub.hf_hub_download.assert_not_called()
        hub.list_repo_files.assert_not_called()


class PresetAcceptanceTests(unittest.TestCase):
    def test_install_is_additive_and_sets_qwen_defaults(self):
        class Info:
            def __init__(self, default=None, section=("ui_sd", "SD")):
                self.default, self.section = default, section

        template = {
            "sd_t2i_step": Info(20),
            "sd_t2i_cfg": Info(7),
            "forge_checkpoint_sd": Info("old"),
        }
        registered = mock.Mock(side_effect=lambda target: target.update(template))

        class PresetArch:
            @staticmethod
            def choices():
                return ["ideogram4", "boogu"]

        presets_module = types.ModuleType("modules_forge.presets")
        presets_module.register = registered
        presets_module.PresetArch = PresetArch
        forge_module = types.ModuleType("modules_forge")
        forge_module.presets = presets_module
        added = {}
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(
            data_labels={}, add_option=lambda key, info: added.setdefault(key, info)
        ))
        modules = types.ModuleType("modules"); modules.shared = shared
        with mock.patch.dict(sys.modules, {
            "modules": modules, "modules_forge": forge_module,
            "modules_forge.presets": presets_module,
        }):
            preset.install()

        choices = presets_module.PresetArch.choices()
        self.assertEqual(choices[:2], ["ideogram4", "boogu"])
        self.assertEqual(choices.count(preset.NAME), 1)
        self.assertEqual(added[preset.NAME + "_t2i_step"].default, 40)
        self.assertEqual(added[preset.NAME + "_t2i_cfg"].default, 1.0)
        self.assertEqual(added["forge_checkpoint_" + preset.NAME].default,
                         importlib.import_module("pi_qwen21.lib.forge").LABEL)


class VisibilityCallbackAcceptanceTests(unittest.TestCase):
    def test_callback_binds_only_to_actual_checkpoint_setting_elem_id(self):
        from gradio.context import Context
        scripts = types.ModuleType("modules.scripts")
        scripts.Script = object; scripts.AlwaysVisible = True
        callback_api = types.SimpleNamespace(
            on_after_component=lambda fn: None, on_script_unloaded=lambda fn: None
        )
        modules = types.ModuleType("modules")
        modules.scripts, modules.script_callbacks = scripts, callback_api
        with mock.patch.dict(sys.modules, {
            "modules": modules, "modules.scripts": scripts,
            "modules.script_callbacks": callback_api,
        }), mock.patch.object(importlib.import_module("pi_qwen21.lib.forge"), "install"), \
             mock.patch.object(preset, "install"):
            sys.modules.pop("pi_qwen21.scripts.engine", None)
            engine = importlib.import_module("pi_qwen21.scripts.engine")

        panel = object()
        engine.PANELS[:] = [panel]

        class Component:
            def __init__(self, elem_id):
                self.elem_id = elem_id
                self.change = mock.Mock()

        wrong = Component("setting_sd_checkpoint")
        engine.after_component(wrong)
        wrong.change.assert_not_called()
        actual = Component("setting_sd_model_checkpoint")
        with mock.patch.object(Context, "root_block", object()):
            engine.after_component(actual)
            engine.after_component(actual)
        actual.change.assert_called_once()
        kwargs = actual.change.call_args.kwargs
        self.assertEqual(kwargs["inputs"], [actual])
        self.assertEqual(kwargs["outputs"], [panel])
        self.assertFalse(kwargs["queue"])


class LoraAcceptanceTests(unittest.TestCase):
    def test_lora_requires_21_family_and_lora_tensor_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good = root / "good.safetensors"
            old = root / "old.safetensors"
            flux = root / "flux.safetensors"
            no_lora = root / "not_lora.safetensors"
            pair = {"diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": {},
                    "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": {}}
            write_safetensors(good, {"__metadata__": {"base_model": "Qwen-Image-2.1"}, **pair})
            write_safetensors(old, {"__metadata__": {"base_model": "Qwen-Image-1.0"}, **pair})
            write_safetensors(flux, {"__metadata__": {"base_model": "Qwen-Image-2.1 Flux"}, **pair})
            write_safetensors(no_lora, {"__metadata__": {"base_model": "Qwen-Image-2.1"}, "transformer.weight": {}})
            self.assertTrue(adapter.compatible(good))
            self.assertFalse(adapter.compatible(old))
            self.assertFalse(adapter.compatible(flux))
            self.assertFalse(adapter.compatible(no_lora))

    def test_distilled_lora_needs_opt_in_and_explicit_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            explicit = root / "lightning.safetensors"
            keys_only = root / "turbo.safetensors"
            pair = {"diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": {},
                    "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": {}}
            write_safetensors(explicit, {"__metadata__": {"base_model": "Qwen-Image-2.1", "type": "distill"}, **pair})
            write_safetensors(keys_only, {"__metadata__": {}, **pair})
            self.assertFalse(adapter.compatible(explicit, community=False))
            self.assertTrue(adapter.compatible(explicit, community=True))
            self.assertFalse(adapter.compatible(keys_only, community=True))

    def test_ai_toolkit_qwen2_plain_lokr_is_accepted_but_foreign_targets_are_not(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good, foreign = root / "good.safetensors", root / "foreign.safetensors"
            meta = {"ss_base_model_version": "qwen_image_2", "software": "ai-toolkit"}
            pair = {"diffusion_model.transformer_blocks.0.attn.to_q.lokr_w1": {},
                    "diffusion_model.transformer_blocks.0.attn.to_q.lokr_w2": {}}
            write_safetensors(good, {"__metadata__": meta, **pair})
            write_safetensors(foreign, {"__metadata__": meta,
                                       "diffusion_model.double_blocks.0.attn.lokr_w1": {},
                                       "diffusion_model.double_blocks.0.attn.lokr_w2": {}})
            self.assertTrue(adapter.compatible(good))
            self.assertFalse(adapter.compatible(foreign))

    def test_plain_lokr_hook_matches_dense_kronecker_and_splits_gate_up(self):
        import torch
        class Mlp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_layer = torch.nn.Linear(4, 4, bias=False)
                self.proj = torch.nn.Linear(4, 4, bias=False)
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.img_mlp = Mlp()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.transformer_blocks = torch.nn.ModuleList([Block()])
        model=Model()
        for p in model.parameters(): p.data.zero_()
        w1=torch.tensor([[1.,2.],[3.,4.]])
        w2=torch.tensor([[1.,0.],[0.,1.],[2.,0.],[0.,2.]])
        state={"diffusion_model.transformer_blocks.0.img_mlp.gate_up.lokr_w1":w1,
               "diffusion_model.transformer_blocks.0.img_mlp.gate_up.lokr_w2":w2,
               # Full w1+w2 LoKr is scale 1 in LyCORIS even with a finite alpha.
               "diffusion_model.transformer_blocks.0.img_mlp.gate_up.alpha":torch.tensor(1.)}
        handles=[]
        self.assertEqual(adapter._install_lokr(model,state,0.5,handles), 2)
        x=torch.tensor([[1.,2.,3.,4.]])
        dense=torch.kron(w1,w2)
        self.assertTrue(torch.allclose(model.transformer_blocks[0].img_mlp.gate_layer(x), x@dense[:4].T*0.5))
        self.assertTrue(torch.allclose(model.transformer_blocks[0].img_mlp.proj(x), x@dense[4:].T*0.5))
        for handle in handles: handle.remove()

        bad=dict(state)
        bad["diffusion_model.transformer_blocks.9.attn.to_q.lokr_w1"]=w1
        bad["diffusion_model.transformer_blocks.9.attn.to_q.lokr_w2"]=w2[:2]
        handles=[]
        with self.assertRaisesRegex(ValueError, "missing target"):
            adapter._install_lokr(model,bad,1.0,handles)
        self.assertEqual(handles, [])

    def test_worker_client_queues_lokr_without_needing_a_transformer(self):
        class Client:
            def __init__(self): self.pending={}; self.selected=[]; self.unloaded=0
            def unload_lora_weights(self): self.unloaded += 1
            def load_lora_weights(self, folder, weight_name=None, adapter_name=None, **_kw):
                self.pending[adapter_name] = str(Path(folder) / weight_name)
            def set_adapters(self, names, adapter_weights=None):
                self.selected = list(zip((self.pending[n] for n in names), adapter_weights))
        pipe=Client()
        adapter.apply(pipe, [(r"C:\models\character.safetensors", 0.7)])
        self.assertEqual(pipe.unloaded, 1)
        self.assertEqual(pipe.selected, [(r"C:\models\character.safetensors", 0.7)])


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_properties(_index):
        return types.SimpleNamespace(total_memory=24 * 2**30)


class FakeGenerator:
    def __init__(self, _device):
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakePipe:
    def __init__(self):
        self.calls = []
        self.unloaded = 0

    def unload_lora_weights(self):
        self.unloaded += 1

    def __call__(self, prompt, image=None, mask_image=None, negative_prompt=None,
                 true_cfg_scale=1, width=None, height=None, num_inference_steps=None,
                 generator=None, callback_on_step_end=None, output_resolution=None):
        call = dict(prompt=prompt, image=image, mask_image=mask_image,
                    negative_prompt=negative_prompt, true_cfg_scale=true_cfg_scale,
                    width=width, height=height, num_inference_steps=num_inference_steps,
                    generator=generator, output_resolution=output_resolution)
        self.calls.append(call)
        return types.SimpleNamespace(images=[Image.new("RGBA", (8, 8), (20, 40, 60, 80))])


class RuntimeAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.pipe = FakePipe()
        self.state = types.SimpleNamespace(
            job_count=0, interrupted=False, skipped=False, sampling_step=0,
            sampling_steps=0, nextjob=lambda: None,
        )
        self.saved = []
        modules = types.ModuleType("modules")
        processing = types.ModuleType("modules.processing")
        processing.Processed = lambda p, images, seed, info, **kw: types.SimpleNamespace(
            images=images, seed=seed, info=info, **kw
        )
        shared = types.ModuleType("modules.shared")
        shared.state = self.state
        shared.opts = types.SimpleNamespace(samples_save=False)
        images = types.ModuleType("modules.images")
        images.save_image = lambda *a, **kw: self.saved.append((a, kw))
        modules.processing, modules.shared, modules.images = processing, shared, images
        self.module_patch = mock.patch.dict(sys.modules, {
            "modules": modules, "modules.processing": processing,
            "modules.shared": shared, "modules.images": images,
        })
        self.module_patch.start()
        self.torch = types.SimpleNamespace(cuda=FakeCuda, Generator=FakeGenerator)
        self.torch_patch = mock.patch.dict(sys.modules, {"torch": self.torch})
        self.torch_patch.start()
        runtime._pipe = None
        runtime._key = None

    def tearDown(self):
        runtime._pipe = None
        runtime._key = None
        self.torch_patch.stop()
        self.module_patch.stop()

    def p(self, init_images=None):
        return types.SimpleNamespace(
            prompt="draw it", negative_prompt="", width=1024, height=1024,
            steps=40, cfg_scale=1, seed=7, batch_size=1, n_iter=1,
            init_images=init_images or [], do_not_save_samples=True,
            outpath_samples="", override_settings={},
        )

    def options(self, **changes):
        base = dict(task="t2i", steps=40, true_cfg=1, profile="24", side=1024,
                    offload=True, community=False, consent=False, mask=None, refs=[])
        base.update(changes)
        return base

    def run_generate(self, p=None, **options):
        p = p or self.p()
        bundle = {"folder": "Qwen-Image-2.1", "files": {}}
        with mock.patch.object(runtime, "resolve", return_value=bundle), \
             mock.patch.object(runtime, "load", return_value=self.pipe), \
             mock.patch.object(runtime.adapter, "parse", side_effect=lambda text, community: (text, [])), \
             mock.patch.object(runtime.adapter, "apply"):
            result = runtime.generate(p, "selected", self.options(**options))
        return p, result

    def test_native_steps_and_small_dimensions_are_not_overridden(self):
        p=self.p();p.width=512;p.height=512;p.steps=3
        self.run_generate(p=p,side=0,steps=40)
        self.assertEqual((p.width,p.height,p.steps),(512,512,3))
        self.assertEqual(self.pipe.calls[-1]['num_inference_steps'],3)

    def test_t2i_does_not_send_image_argument(self):
        self.run_generate(task="t2i")
        self.assertIsNone(self.pipe.calls[0]["image"])

    def test_img2img_input_is_first_then_references_and_capped_at_ten(self):
        primary = object()
        refs = [object(), object()]
        self.run_generate(self.p([primary]), task="edit", refs=refs)
        self.assertEqual(self.pipe.calls[0]["image"], [primary] + refs)

        with self.assertRaisesRegex(ValueError, "10"):
            self.run_generate(self.p([primary]), task="edit", refs=[object() for _ in range(10)])

    def test_txt2img_rejects_edit_controls(self):
        for options in [dict(task="edit"), dict(refs=[object()]), dict(mask=object())]:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "img2img"):
                self.run_generate(**options)
        self.assertEqual(self.pipe.calls, [])

    def test_cfg_above_one_requires_negative_prompt(self):
        p = self.p()
        p.cfg_scale = 2
        self.run_generate(p, true_cfg=9)
        self.assertEqual(self.pipe.calls[0]["true_cfg_scale"], 1)
        self.assertIsNone(self.pipe.calls[0]["negative_prompt"])
        self.assertEqual(p.cfg_scale, 1)
        p = self.p()
        p.cfg_scale = 2
        p.negative_prompt = "blurry"
        self.run_generate(p, true_cfg=1)
        self.assertEqual(self.pipe.calls[-1]["negative_prompt"], "blurry")
        self.assertEqual(self.pipe.calls[-1]["true_cfg_scale"], 2)

    def test_rgba_result_is_preserved(self):
        _p, result = self.run_generate(task="rgba")
        self.assertEqual(result.images[0].mode, "RGBA")

    def test_moire_cleanup_defaults_to_full_strength(self):
        with mock.patch.object(runtime, "remove_moire", wraps=runtime.remove_moire) as cleanup:
            _p, result = self.run_generate()
        cleanup.assert_called_once()
        self.assertEqual(cleanup.call_args.kwargs["strength"], 1.0)
        self.assertIn("Qwen moire cleanup: 1", result.info)

    def test_moire_cleanup_can_be_disabled_or_scaled(self):
        with mock.patch.object(runtime, "remove_moire", wraps=runtime.remove_moire) as cleanup:
            _p, disabled = self.run_generate(moire_cleanup=False, moire_strength=0.4)
        cleanup.assert_not_called()
        self.assertIn("Qwen moire cleanup: 0", disabled.info)

        with mock.patch.object(runtime, "remove_moire", wraps=runtime.remove_moire) as cleanup:
            _p, scaled = self.run_generate(moire_strength=0.4)
        self.assertEqual(cleanup.call_args.kwargs["strength"], 0.4)
        self.assertIn("Qwen moire cleanup: 0.4", scaled.info)


class SpeedLoraAcceptanceTests(unittest.TestCase):
    def setUp(self):
        runtime._pipe = None
        runtime._key = None

    def tearDown(self):
        runtime._pipe = None
        runtime._key = None

    def test_catalog_entries_have_required_fields(self):
        from pi_qwen21.download import manager
        for name, entry in manager.FEATURED.items():
            self.assertTrue(entry["repo"], name)
            self.assertTrue(entry["file"].endswith(".safetensors"), name)
            self.assertGreaterEqual(entry["steps"], 2, name)
            self.assertEqual(entry["cfg"], 1.0, name)
            self.assertTrue(entry["source"].startswith("https://huggingface.co/"), name)

    def test_download_requires_explicit_approval(self):
        from pi_qwen21.download import manager
        name = manager.FEATURED_CHOICES[0]
        with self.assertRaisesRegex(ValueError, "license"):
            manager.download_featured(name, approved=False)

    def test_missing_speed_lora_stops_generation_without_downloading(self):
        from pi_qwen21.download import manager
        name = list(manager.FEATURED)[0]
        p = types.SimpleNamespace(
            prompt="x", negative_prompt="", width=1024, height=1024,
            steps=40, cfg_scale=1, seed=1, batch_size=1, n_iter=1,
            init_images=[], do_not_save_samples=True, outpath_samples="",
            override_settings={},
        )
        modules_mod = types.ModuleType("modules")
        processing = types.ModuleType("modules.processing")
        shared = types.ModuleType("modules.shared")
        shared.state = types.SimpleNamespace(job_count=0, interrupted=False, skipped=False, nextjob=lambda: None)
        shared.opts = types.SimpleNamespace(samples_save=False)
        images_mod = types.ModuleType("modules.images")
        images_mod.save_image = lambda *a, **kw: None
        modules_mod.processing, modules_mod.shared, modules_mod.images = processing, shared, images_mod
        with mock.patch.dict(sys.modules, {
            "modules": modules_mod, "modules.processing": processing,
            "modules.shared": shared, "modules.images": images_mod,
            "torch": types.SimpleNamespace(cuda=FakeCuda, Generator=FakeGenerator),
        }), mock.patch.object(manager, "featured_path", return_value=None):
            with self.assertRaisesRegex(ValueError, "never downloads"):
                runtime.generate(p, "selected", dict(
                    task="t2i", steps=40, true_cfg=1, profile="24", side=1024,
                    offload=True, community=False, consent=False, mask=None,
                    refs=[], speed_enabled=True, speed_lora=name, speed_strength=1.0))

    def test_speed_lora_forces_cfg1_and_few_steps(self):
        from pi_qwen21.download import manager
        name = list(manager.FEATURED)[0]
        entry = manager.FEATURED[name]
        calls = []
        with tempfile.TemporaryDirectory() as td:
            lora_file = Path(td) / "featured.safetensors"
            lora_file.write_bytes(b"0123456789")
            self._run_speed_case(name, entry, lora_file, calls)

    def _run_speed_case(self, name, entry, lora_file, calls):

        class SpeedPipe:
            def unload_lora_weights(self): pass

            def __call__(self, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(images=[Image.new("RGB", (8, 8))])

        saved = runtime._pipe
        try:
            saved_key = runtime._key
            p = types.SimpleNamespace(
                prompt="x", negative_prompt="blurry", width=1024, height=1024,
                steps=40, cfg_scale=2, seed=1, batch_size=1, n_iter=1,
                init_images=[], do_not_save_samples=True, outpath_samples="",
                override_settings={},
            )
            state = types.SimpleNamespace(
                job_count=0, interrupted=False, skipped=False, nextjob=lambda: None)
            modules_mod = types.ModuleType("modules")
            processing = types.ModuleType("modules.processing")
            processing.Processed = lambda p, images, seed, info, **kw: types.SimpleNamespace(info=info, images=images)
            shared = types.ModuleType("modules.shared")
            shared.state = state
            shared.opts = types.SimpleNamespace(samples_save=False)
            images_mod = types.ModuleType("modules.images")
            images_mod.save_image = lambda *a, **kw: None
            modules_mod.processing, modules_mod.shared, modules_mod.images = processing, shared, images_mod
            infos = []
            processing.Processed = lambda p, images, seed, info, **kw: (infos.append(info), types.SimpleNamespace(info=info, images=images))[1]
            fake_torch = types.SimpleNamespace(
                cuda=FakeCuda, Generator=FakeGenerator,
                cuda_module=types.SimpleNamespace(OutOfMemoryError=RuntimeError))
            with mock.patch.dict(sys.modules, {
                "modules": modules_mod, "modules.processing": processing,
                "modules.shared": shared, "modules.images": images_mod,
                "torch": fake_torch,
             }), mock.patch.object(runtime, "resolve", return_value={"folder": "f", "files": {}}), \
                 mock.patch.object(runtime, "load", return_value=SpeedPipe()), \
                 mock.patch.object(runtime.adapter, "parse", side_effect=lambda t, c: (t, [])), \
                 mock.patch.object(runtime.adapter, "apply") as apply_adapter, \
                 mock.patch.object(manager, "featured_path", return_value=lora_file):
                runtime.generate(p, "selected", dict(
                    task="t2i", steps=40, true_cfg=2, profile="24", side=1024,
                    offload=True, community=False, consent=False, mask=None, refs=[],
                    speed_enabled=True, speed_lora=name, speed_strength=0.8))
            self.assertEqual(p.steps, entry["steps"], "few-step schedule must override user steps")
            self.assertEqual(p.cfg_scale, 1.0, "speed LoRA must force CFG 1")
            self.assertEqual(calls[0]["true_cfg_scale"], 1.0)
            self.assertEqual(calls[0]["num_inference_steps"], entry["steps"])
            applied = apply_adapter.call_args.args[1]
            self.assertEqual(applied[-1], (str(lora_file), 0.8))
            self.assertTrue(any("Speed LoRA: " + name in i for i in infos))
        finally:
            runtime._pipe = saved
            runtime._key = saved_key


class ForgeIsolationAcceptanceTests(unittest.TestCase):
    def test_unselected_checkpoint_uses_original_forge_processing(self):
        forge = importlib.import_module("pi_qwen21.lib.forge")
        original_result = object()
        original_process = mock.Mock(return_value=original_result)
        original_run = mock.Mock(return_value=None)
        original_list = mock.Mock(return_value="listed")

        class ScriptRunner:
            run = original_run

        class CheckpointInfo:
            pass

        scripts = types.ModuleType("modules.scripts")
        scripts.ScriptRunner = ScriptRunner
        processing = types.ModuleType("modules.processing")
        processing.process_images = original_process
        sd_models = types.ModuleType("modules.sd_models")
        sd_models.list_models = original_list
        sd_models.CheckpointInfo = CheckpointInfo
        sd_models.checkpoints_list = {}
        sd_models.checkpoint_aliases = {}
        callbacks = types.SimpleNamespace(on_app_started=lambda fn: None)
        modules = types.ModuleType("modules")
        modules.scripts, modules.processing, modules.sd_models = scripts, processing, sd_models
        modules.script_callbacks = callbacks
        p = types.SimpleNamespace(scripts=None, override_settings={})

        with mock.patch.dict(sys.modules, {
            "modules": modules, "modules.scripts": scripts,
            "modules.processing": processing, "modules.sd_models": sd_models,
        }), mock.patch.object(forge, "selected", return_value=None):
            forge.install()
            self.assertIs(processing.process_images(p), original_result)
            self.assertIsNone(ScriptRunner.run(object(), p))
        original_process.assert_called_once_with(p)
        original_run.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
