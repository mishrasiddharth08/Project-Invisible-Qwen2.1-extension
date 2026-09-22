import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen21_spectrum", ROOT / "lib/spectrum.py")
spectrum = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = spectrum
spec.loader.exec_module(spectrum)


class Norm(torch.nn.Module):
    def forward(self, hidden, temb, modulation_mask=None):
        return hidden


class QwenImage21Transformer2DModel(torch.nn.Module):
    zero_cond_t = False
    def __init__(self):
        super().__init__(); self.norm_out = Norm(); self.proj_out = torch.nn.Identity()
        self.time_text_embed = lambda timestep, hidden: timestep[:, None, None]
        self.transformer_blocks = [object()]; self.real_calls = 0
        self.config = types.SimpleNamespace(causal_condition=True)
    def build_token_metadata(self, image_pad_mask, img_shapes):
        target = torch.zeros_like(image_pad_mask)
        target[-2:] = True
        return None, target
    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, img_shapes=None,
                img_mask=None, kv_cache_mode=None, return_dict=True, **kwargs):
        self.real_calls += 1
        temb = self.time_text_embed(timestep, hidden_states)
        if kv_cache_mode == "extract":
            feature = torch.ones(1, 5, 3) + timestep[:, None, None]
            mask = torch.tensor([False, False, False, True, True])
        else:
            feature = hidden_states + timestep[:, None, None]
            mask = torch.tensor([True, True])
        value = self.proj_out(self.norm_out(feature, temb, mask))
        return types.SimpleNamespace(sample=value) if return_dict else (value,)


class SpectrumTests(unittest.TestCase):
    def pipe(self): return types.SimpleNamespace(transformer=QwenImage21Transformer2DModel())

    def test_disabled_is_exact_noop(self):
        pipe = self.pipe(); original = pipe.transformer.forward
        with spectrum.accelerate(pipe, enabled=False, steps=10) as stats:
            self.assertEqual(stats.reason, "disabled")
        self.assertEqual(pipe.transformer.forward, original)

    def test_opt_in_forecasts_and_restores_even_after_error(self):
        pipe = self.pipe(); original = pipe.transformer.forward
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with spectrum.accelerate(pipe, enabled=True, steps=10) as stats:
                pipe.transformer(torch.ones(1, 2, 3), timestep=torch.tensor([0.0]),
                                 img_shapes=[[(1, 1, 1), (1, 1, 2)]],
                                 img_mask=torch.tensor([[True, False]]), kv_cache_mode="extract")
                for step in range(1, 7):
                    pipe.transformer(torch.ones(1, 2, 3), timestep=torch.tensor([float(step)]),
                                     img_shapes=[[(1, 1, 1), (1, 1, 2)]],
                                     img_mask=torch.tensor([[True, False]]), kv_cache_mode="cached")
                self.assertGreater(stats.forecast, 0)
                self.assertLess(pipe.transformer.real_calls, 7)
                raise RuntimeError("boom")
        self.assertEqual(pipe.transformer.forward, original)
        self.assertFalse(pipe.transformer.norm_out._forward_pre_hooks)

    def test_short_runs_and_wrong_models_fail_closed(self):
        pipe = self.pipe()
        with spectrum.accelerate(pipe, enabled=True, steps=7) as stats:
            self.assertEqual(stats.reason, "too_few_steps")
        wrong = types.SimpleNamespace(transformer=types.SimpleNamespace())
        with spectrum.accelerate(wrong, enabled=True, steps=20) as stats:
            self.assertEqual(stats.reason, "unsupported_model")

    def test_chebyshev_forecast_is_finite(self):
        history = [(-1.0, torch.tensor([1.0])), (-0.5, torch.tensor([2.0])), (0.0, torch.tensor([3.0]))]
        value = spectrum._forecast(torch, history, 0.5)
        self.assertTrue(torch.isfinite(value).all())


if __name__ == "__main__": unittest.main()
