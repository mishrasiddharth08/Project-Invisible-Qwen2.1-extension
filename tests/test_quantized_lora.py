import unittest
from types import SimpleNamespace
import torch
from test_acceptance_cpu import adapter


class QuantizedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = self.out_features = 4

    def forward(self, x):
        return x * 2


class QuantizedLoraTests(unittest.TestCase):
    def test_scaled_adapter_and_unload(self):
        model = torch.nn.Module()
        model.add_module('linear', QuantizedLinear())
        a, b, x = torch.randn(2, 4), torch.randn(4, 2), torch.randn(3, 4)
        handles = []
        adapter._install_lora(model, {'linear.lora_A.weight': a,
            'linear.lora_B.weight': b, 'linear.alpha': torch.tensor(4.)}, .5, handles)
        torch.testing.assert_close(model.linear(x), x * 2 + x @ a.T @ b.T)
        adapter.clear(SimpleNamespace(_pi_qwen21_lokr_handles=handles, unload_lora_weights=lambda: None))
        torch.testing.assert_close(model.linear(x), x * 2)

    def test_invalid_target_is_atomic(self):
        model = torch.nn.Module()
        model.add_module('linear', QuantizedLinear())
        state = {name + suffix: torch.ones(shape) for name in ('linear', 'missing')
            for suffix, shape in (('.lora_A.weight', (2, 4)), ('.lora_B.weight', (4, 2)))}
        handles = []
        with self.assertRaises(ValueError):
            adapter._install_lora(model, state, 1, handles)
        self.assertEqual(handles, [])
        self.assertFalse(model.linear._forward_hooks)

    def test_fused_gate_split(self):
        model = torch.nn.Module()
        model.block = torch.nn.Module()
        model.block.img_mlp = torch.nn.Module()
        for name in ('gate_layer', 'proj'):
            model.block.img_mlp.add_module(name, QuantizedLinear())
        a, b, x = torch.randn(2, 4), torch.randn(8, 2), torch.randn(1, 4)
        handles = []
        adapter._install_lora(model, {'block.img_mlp.gate_up.lora_down.weight': a,
            'block.img_mlp.gate_up.lora_up.weight': b}, 1, handles)
        for name, up in zip(('gate_layer', 'proj'), b.chunk(2)):
            torch.testing.assert_close(getattr(model.block.img_mlp, name)(x), x * 2 + x @ a.T @ up.T)
