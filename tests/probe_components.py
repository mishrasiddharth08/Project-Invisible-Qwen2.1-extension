"""Offline smoke probes for the Qwen 2.1 component loader."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "_deps"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf8"))
FORGE = Path(CONFIG.get("live_forge") or ROOT.parent.parent)
sys.path[:0] = [str(DEPS), str(FORGE), str(FORGE / "modules_forge" / "packages")]

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("pi_qwen21", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules["pi_qwen21"] = package
spec.loader.exec_module(package)


def component_probe():
    from diffusers import QwenImage21Transformer2DModel
    from pi_qwen21.lib.components import load_component

    config = dict(patch_size=1, in_channels=4, out_channels=4, num_layers=1,
                  attention_head_dim=16, num_attention_heads=2, context_in_dim=16,
                  axes_dims_rope=[4, 6, 6])
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "config.json").write_text(json.dumps(config), encoding="utf8")
        source = QwenImage21Transformer2DModel(**config).to(dtype=torch.bfloat16)
        weights = {"model.diffusion_model." + k: v.detach().cpu() for k, v in source.state_dict().items()}
        path = root / "tiny.safetensors"
        save_file(weights, str(path))
        loaded = load_component(QwenImage21Transformer2DModel, root, path, "transformer")
        for key, value in source.state_dict().items():
            assert torch.equal(value.cpu(), loaded.state_dict()[key].cpu()), key

        bad = root / "bad.safetensors"
        save_file({"linear.comfy_quant": torch.tensor([123], dtype=torch.uint8)}, str(bad))
        try:
            load_component(QwenImage21Transformer2DModel, root, bad, "transformer")
        except ValueError as exc:
            assert "descriptor" in str(exc).lower()
        else:
            raise AssertionError("malformed quant descriptor was accepted")


def int8_convrot_probe():
    from pi_qwen21.lib.components import load_component

    class TinyQuant(torch.nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.linear = torch.nn.Linear(256, 256)

        @classmethod
        def from_config(cls, path):
            return cls(json.loads(Path(path).read_text()))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "config.json").write_text("{}", encoding="utf8")
        descriptor = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
        weights = {
            "model.diffusion_model.linear.weight": torch.ones((256, 256), dtype=torch.int8),
            "model.diffusion_model.linear.weight_scale": torch.tensor(0.01),
            "model.diffusion_model.linear.comfy_quant": torch.tensor(list(json.dumps(descriptor).encode()), dtype=torch.uint8),
            "model.diffusion_model.linear.bias": torch.zeros(256, dtype=torch.bfloat16),
        }
        path = root / "tiny-int8.safetensors"
        save_file(weights, str(path))
        model = load_component(TinyQuant, root, path, "transformer").cuda().eval()
        output = model.linear(torch.ones(1, 256, device="cuda", dtype=torch.bfloat16))
        assert output.shape == (1, 256)


def official_import_probe():
    code = "from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21; from transformers import Qwen3VLForConditionalGeneration; print('official-import-ok')"
    paths = [str(DEPS), str(FORGE), str(FORGE / "modules_forge" / "packages")]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-c", code], cwd=str(FORGE), capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    assert "official-import-ok" in result.stdout


if __name__ == "__main__":
    component_probe()
    int8_convrot_probe()
    official_import_probe()
    print("component probes passed")
