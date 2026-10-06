import json
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("workflow_diagnostics", ROOT / "lib/workflow_diagnostics.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)
component_kind = diagnostics.component_kind
inspect_readiness = diagnostics.inspect_readiness


def tensor(path, keys, metadata=None):
    header = {key: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 0]} for key in keys}
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw)
    return Path(path)


def fixtures(root):
    dit = [f"transformer_blocks.{i}.attn.to_q.weight" for i in range(32)]
    dit += ["transformer_blocks.0.img_mlp.out.weight"]
    te = [f"model.layers.{i}.input_layernorm.weight" for i in range(36)]
    te += ["model.visual.patch_embed.proj.weight"]
    vae = ["decoder.conv1.weight"]
    union = [f"control_blocks.{i}.after_proj.weight" for i in range(16)]
    union += ["control_img_in.weight", "control_blocks.0.img_mlp.out.weight"]
    return {
        "transformer": tensor(root / "Stable-diffusion/qwen21.safetensors", dit),
        "text_encoder": tensor(root / "text_encoder/qwen3vl.safetensors", te),
        "vae": tensor(root / "VAE/qwen21-vae.safetensors", vae,
                      {"modelspec.architecture": "qwen_image_2.1_vae"}),
        "control": tensor(root / "model_patches/qwen21-union.safetensors", union),
    }


class WorkflowDiagnosticsTests(unittest.TestCase):
    def test_exact_component_contracts_use_headers_only(self):
        with tempfile.TemporaryDirectory() as td:
            paths = fixtures(Path(td))
            self.assertEqual(component_kind(paths["transformer"]), "qwen21_dit")
            self.assertEqual(component_kind(paths["text_encoder"]), "qwen3vl_8b")
            self.assertEqual(component_kind(paths["vae"]), "qwen21_vae")
            self.assertEqual(component_kind(paths["control"]), "qwen21_union")

    def test_ready_diffusers_report_discovers_compatible_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixtures(root / "models")
            report = inspect_readiness(root / "models", root / "extension")
            self.assertTrue(report["ready"])
            self.assertFalse(report["downloads_started"])
            self.assertEqual(report["source"]["commit"], "1ebd2454ba424fb08fe666c5e494d8eb189fe935")
            self.assertTrue(report["compatible_paths"]["qwen21_dit"])

    def test_missing_items_are_simple_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = inspect_readiness(root / "models", root / "extension", backend="comfy",
                                       sampler="res_2s", control_enabled=True)
            self.assertFalse(report["ready"])
            codes = {item["code"] for item in report["checks"] if not item["ok"]}
            self.assertTrue({"transformer", "text_encoder", "vae", "comfy_code",
                             "comfy_dependencies", "res4lyf", "union"}.issubset(codes))

    def test_comfy_pose_and_union_readiness_never_downloads(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); selected = fixtures(root / "models")
            comfy = root / "extension/_comfy"
            for name in ("comfy_extras/nodes_qwen.py", "comfy_extras/nodes_model_patch.py",
                         "custom_nodes/RES4LYF/__init__.py"):
                path = comfy / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("")
            (comfy / ".pi_revision").write_text("7a5dad695fe1cae25efcb2550530fb20ef68da3d")
            (root / "extension/_deps_comfy/comfy_aimdo").mkdir(parents=True)
            pose = root / "models/ControlNetPreprocessor"
            pose.mkdir(parents=True)
            for name in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
                (pose / name).write_bytes(b"local")
            report = inspect_readiness(root / "models", root / "extension", backend="comfy",
                                       comfy_root=comfy, comfy_deps=root / "extension/_deps_comfy",
                                       sampler="res_2s", pose="DWPose",
                                       control_enabled=True, selected=selected)
            self.assertTrue(report["ready"])
            self.assertFalse(report["downloads_started"])

    def test_old_or_partial_qwen_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            partial = tensor(Path(td) / "bad.safetensors", [
                "transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.img_mlp.out.weight"])
            self.assertEqual(component_kind(partial), "unknown")

    def test_decoder_precision_is_reported_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); selected = fixtures(root / "models")
            report = inspect_readiness(root / "models", root / "extension", selected=selected)
            check = next(item for item in report["checks"] if item["code"] == "decoder_precision")
            self.assertIn("bf16", check["message"])


if __name__ == "__main__":
    unittest.main()
