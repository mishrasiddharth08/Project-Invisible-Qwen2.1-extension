import unittest
from types import SimpleNamespace
import torch
from test_acceptance_cpu import adapter


class QuantizedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = self.out_features = 4
        self.weight = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)

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


class ExtendedAdapterTests(unittest.TestCase):
    def build(self, *names):
        model = torch.nn.Module()
        for name in names:
            model.add_module(name, QuantizedLinear())
        return model

    def test_kind_detects_factorized_lokr_loha_and_full(self):
        self.assertEqual(adapter._kind(('m.lokr_w1_a.weight', 'm.lokr_w1_b.weight',
            'm.lokr_w2_a.weight', 'm.lokr_w2_b.weight')), 'lokr')
        self.assertEqual(adapter._kind(('m.lokr_w1', 'm.lokr_w2')), 'lokr')
        self.assertEqual(adapter._kind(('m.loha_w1_a.weight', 'm.loha_w1_b.weight',
            'm.loha_w2_a.weight', 'm.loha_w2_b.weight')), 'loha')
        self.assertEqual(adapter._kind(('m.loha_w1', 'm.loha_w2')), 'loha')
        self.assertEqual(adapter._kind(('m.diff',)), 'full')
        self.assertEqual(adapter._kind(('m.lora_A.weight', 'm.lora_B.weight')), 'lora')
        self.assertIsNone(adapter._kind(('m.weight',)))

    def test_factorized_lokr_reconstruction(self):
        model = self.build('linear')
        # LyCORIS: lokr_w1 = w1_a @ w1_b. Plain LoKr weight is kron(w1, w2);
        # with in=4/out=4 use 2x2 factors. Hook math verified via plain path.
        w1_a, w1_b = torch.randn(2, 1), torch.randn(1, 2)
        w2 = torch.randn(2, 2)
        x = torch.randn(1, 4)
        handles = []
        adapter._install_lokr(model, {'linear.lokr_w1_a': w1_a, 'linear.lokr_w1_b': w1_b,
            'linear.lokr_w2': w2}, 1, handles)
        self.assertEqual(len(handles), 1)
        w1 = w1_a @ w1_b
        flat = x.reshape(-1, 2, 2)
        side = torch.einsum('bij,pj->bip', flat, w2)
        expected_delta = torch.einsum('bip,qi->bqp', side, w1).reshape_as(x)
        torch.testing.assert_close(model.linear(x) - x * 2, expected_delta, atol=1e-4, rtol=1e-4)
        adapter.clear(SimpleNamespace(_pi_qwen21_lokr_handles=handles, unload_lora_weights=lambda: None))
        torch.testing.assert_close(model.linear(x), x * 2)

    def test_loha_hadamard_delta(self):
        model = self.build('linear')
        # Both LoHa factors map in=4 to out=4 so their Hadamard product
        # matches the output shape, per LyCORIS LoHa composition.
        w1, w2, x = torch.randn(4, 4), torch.randn(4, 4), torch.randn(1, 4)
        handles = []
        adapter._install_loha(model, {'linear.loha_w1': w1, 'linear.loha_w2': w2,
            'linear.alpha': torch.tensor(2.)}, 1, handles)
        side1, side2 = x @ w1.T, x @ w2.T
        torch.testing.assert_close(model.linear(x), x * 2 + (side1 * side2) * 0.5)
        adapter.clear(SimpleNamespace(_pi_qwen21_lokr_handles=handles, unload_lora_weights=lambda: None))
        torch.testing.assert_close(model.linear(x), x * 2)

    def test_loha_incomplete_pair_is_atomic(self):
        model = self.build('linear')
        handles = []
        with self.assertRaises(ValueError):
            adapter._install_loha(model, {'linear.loha_w1': torch.randn(2, 4)}, 1, handles)
        self.assertEqual(handles, [])
        self.assertFalse(model.linear._forward_hooks)

    def test_full_diff_applies_delta_with_strength(self):
        model = self.build('linear')
        delta, x = torch.randn(4, 4), torch.randn(1, 4)
        handles = []
        adapter._install_full(model, {'linear.diff': delta}, 0.5, handles)
        torch.testing.assert_close(model.linear(x), x * 2 + delta * 0.5)
        adapter.clear(SimpleNamespace(_pi_qwen21_lokr_handles=handles, unload_lora_weights=lambda: None))
        torch.testing.assert_close(model.linear(x), x * 2)

    def test_full_diff_shape_mismatch_is_rejected(self):
        model = self.build('linear')
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            adapter._install_full(model, {'linear.diff': torch.zeros(2, 2)}, 1, [])


class DequantizeFormatTests(unittest.TestCase):
    def descriptor(self, fmt):
        import json
        return torch.tensor(list(json.dumps({'format': fmt}).encode()), dtype=torch.uint8)

    def test_all_packed_formats_roundtrip_to_bf16(self):
        import sys
        sys.path.insert(0, r'G:\FORGE UI NEO\sd-webui-forge-classic')
        import backend.quant_ops  # noqa: registers layout classes
        from comfy_kitchen.tensor import QuantizedTensor
        from backend.quant_ops import QUANT_ALGOS
        from pi_qwen21.lib.components import _dequantize_component
        weight = torch.randn(256, 512, dtype=torch.bfloat16)
        for fmt, qt in DequantizeFormatTests._packs(weight):
            sd = {'b.weight': qt._qdata, 'b.weight_scale': qt._params.scale,
                  'b.comfy_quant': self.descriptor(fmt)}
            if getattr(qt._params, 'block_scale', None) is not None:
                sd['b.block_scale'] = qt._params.block_scale
            out = _dequantize_component(sd, 'transformer', torch.bfloat16)
            self.assertEqual(set(out), {'b.weight'}, fmt)
            err = (out['b.weight'].to(torch.float32) - weight.to(torch.float32)).abs().max().item()
            self.assertLess(err, 0.9, fmt)

    @staticmethod
    def _packs(weight):
        import backend.quant_ops  # noqa
        from comfy_kitchen.tensor import QuantizedTensor
        from backend.quant_ops import QUANT_ALGOS
        result = []
        for fmt in ('float8_e4m3fn', 'float8_e5m2', 'mxfp8', 'nvfp4', 'convrot_w4a4'):
            qt = QuantizedTensor.from_float(weight, QUANT_ALGOS[fmt]['comfy_tensor_layout'],
                                            scale='recalculate')
            result.append((fmt, qt))
        return result
