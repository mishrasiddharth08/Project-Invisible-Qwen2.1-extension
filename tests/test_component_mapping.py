import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch


ROOT=Path(__file__).resolve().parents[1]
if 'pi_qwen21' not in sys.modules:
    spec=importlib.util.spec_from_file_location('pi_qwen21',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
    package=importlib.util.module_from_spec(spec);sys.modules['pi_qwen21']=package;spec.loader.exec_module(package)
from pi_qwen21.lib.components import _map_vae_key,_normalize_component,_embedding_apply,_harden_embedding_apply,_dequantize_component


def descriptor(fmt='int8_tensorwise'):
    return torch.tensor(list(json.dumps({'format':fmt}).encode()),dtype=torch.uint8)


def convrot_descriptor():
    return torch.tensor(list(json.dumps({'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256}).encode()),dtype=torch.uint8)


class ComponentMappingTests(unittest.TestCase):
    def test_fused_gate_up_splits_gate_then_up_with_quant_data(self):
        weight=torch.arange(24,dtype=torch.int8).reshape(6,4)
        scale=torch.arange(6,dtype=torch.float32).reshape(6,1)
        sd={
            'transformer_blocks.0.img_mlp.gate_up.weight':weight,
            'transformer_blocks.0.img_mlp.gate_up.weight_scale':scale,
            'transformer_blocks.0.img_mlp.gate_up.comfy_quant':descriptor(),
        }
        out=_normalize_component(sd,'transformer')
        self.assertTrue(torch.equal(out['transformer_blocks.0.img_mlp.gate_layer.weight'],weight[:3]))
        self.assertTrue(torch.equal(out['transformer_blocks.0.img_mlp.proj.weight'],weight[3:]))
        self.assertTrue(torch.equal(out['transformer_blocks.0.img_mlp.gate_layer.weight_scale'],scale[:3]))
        self.assertTrue(torch.equal(out['transformer_blocks.0.img_mlp.proj.weight_scale'],scale[3:]))
        self.assertIn('transformer_blocks.0.img_mlp.gate_layer.comfy_quant',out)
        self.assertIn('transformer_blocks.0.img_mlp.proj.comfy_quant',out)
        self.assertFalse(any('.gate_up.' in key for key in out))

    def test_packed_or_odd_fused_gate_up_is_refused(self):
        with self.assertRaisesRegex(ValueError,'packed'):
            _normalize_component({'x.img_mlp.gate_up.weight':torch.zeros(6,4),
                                  'x.img_mlp.gate_up.comfy_quant':descriptor('packed_int4')},'transformer')
        with self.assertRaisesRegex(ValueError,'rows'):
            _normalize_component({'x.img_mlp.gate_up.weight':torch.zeros(5,4),
                                  'x.img_mlp.gate_up.comfy_quant':descriptor()},'transformer')

    def test_bf16_fused_gate_up_splits_without_quant_descriptor(self):
        weight=torch.arange(24,dtype=torch.bfloat16).reshape(6,4)
        out=_normalize_component({'x.img_mlp.gate_up.weight':weight},'transformer')
        self.assertTrue(torch.equal(out['x.img_mlp.gate_layer.weight'],weight[:3]))
        self.assertTrue(torch.equal(out['x.img_mlp.proj.weight'],weight[3:]))

    def test_dequantize_component_unpacks_int8_convrot_to_plain_bf16(self):
        # Build a real packed tensor through comfy-kitchen, then unpack it
        # the way unsupported GPUs (ROCm, CUDA<13, fp16-only) must.
        from comfy_kitchen.tensor import QuantizedTensor
        weight=torch.randn(256,512,dtype=torch.bfloat16)
        qt=QuantizedTensor.from_float(weight,'TensorWiseINT8Layout',is_weight=True,per_channel=True,convrot=True,convrot_groupsize=256)
        sd={'block.attn.to_k.weight':qt._qdata,
            'block.attn.to_k.weight_scale':qt._params.scale,
            'block.attn.to_k.comfy_quant':convrot_descriptor()}
        out=_dequantize_component(sd,'transformer',torch.bfloat16)
        self.assertEqual(set(out),{'block.attn.to_k.weight'})
        plain=out['block.attn.to_k.weight']
        self.assertEqual(plain.dtype,torch.bfloat16)
        self.assertEqual(tuple(plain.shape),(256,512))
        self.assertLess((plain.to(torch.float32)-weight.to(torch.float32)).abs().max().item(),0.15)

    def test_dequantize_component_rejects_unknown_format_and_missing_scales(self):
        with self.assertRaisesRegex(ValueError,'unsupported'):
            _dequantize_component({'x.weight':torch.zeros(4,4,dtype=torch.int8),
                                  'x.comfy_quant':descriptor('packed_int4')},'transformer',torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'INT8 scale'):
            _dequantize_component({'x.weight':torch.zeros(4,4,dtype=torch.int8),
                                  'x.comfy_quant':descriptor()},'transformer',torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'without packed weight'):
            _dequantize_component({'x.comfy_quant':descriptor()},'transformer',torch.bfloat16)

    def test_vae_top_convs_and_singleton_temporal_kernels_map_to_diffusers(self):
        conv1=torch.zeros(128,128,1,1,1)
        body=torch.zeros(96,4,1,3,3)
        out=_normalize_component({'conv1.weight':conv1,'conv1.bias':torch.zeros(128),
                                  'conv2.weight':torch.zeros(64,64,1,1,1),
                                  'encoder.conv1.weight':body},'vae')
        self.assertEqual(out['quant_conv.weight'].shape,(128,128,1,1))
        self.assertIn('quant_conv.bias',out)
        self.assertEqual(out['post_quant_conv.weight'].shape,(64,64,1,1))
        self.assertEqual(out['encoder.conv_in.weight'].shape,(96,4,3,3))
        with self.assertRaisesRegex(ValueError,'temporal'):
            _normalize_component({'encoder.conv1.weight':torch.zeros(4,4,3,3,3)},'vae')

    def test_full_comfy_vae_module_families_map_to_official_names(self):
        cases={
            'encoder.conv1.weight':'encoder.conv_in.weight',
            'encoder.downsamples.2.downsamples.0.residual.0.gamma':'encoder.down_blocks.2.resnets.0.norm1.gamma',
            'encoder.downsamples.2.downsamples.0.residual.2.weight':'encoder.down_blocks.2.resnets.0.conv1.weight',
            'encoder.downsamples.2.downsamples.0.residual.3.gamma':'encoder.down_blocks.2.resnets.0.norm2.gamma',
            'encoder.downsamples.2.downsamples.0.residual.6.weight':'encoder.down_blocks.2.resnets.0.conv2.weight',
            'encoder.downsamples.2.downsamples.0.shortcut.weight':'encoder.down_blocks.2.resnets.0.conv_shortcut.weight',
            'encoder.downsamples.2.downsamples.2.resample.1.weight':'encoder.down_blocks.2.downsampler.resample.1.weight',
            'encoder.downsamples.2.downsamples.2.time_conv.weight':'encoder.down_blocks.2.downsampler.time_conv.weight',
            'encoder.middle.0.residual.2.weight':'encoder.mid_block.resnets.0.conv1.weight',
            'encoder.middle.1.to_qkv.weight':'encoder.mid_block.attentions.0.to_qkv.weight',
            'encoder.middle.2.residual.6.weight':'encoder.mid_block.resnets.1.conv2.weight',
            'encoder.head.0.gamma':'encoder.norm_out.gamma',
            'encoder.head.2.weight':'encoder.conv_out.weight',
            'decoder.conv1.weight':'decoder.conv_in.weight',
            'decoder.upsamples.3.upsamples.2.residual.6.weight':'decoder.up_blocks.3.resnets.2.conv2.weight',
            'decoder.upsamples.3.upsamples.3.resample.1.weight':'decoder.up_blocks.3.upsampler.resample.1.weight',
            'decoder.upsamples.3.upsamples.3.time_conv.weight':'decoder.up_blocks.3.upsampler.time_conv.weight',
            'decoder.middle.1.norm.gamma':'decoder.mid_block.attentions.0.norm.gamma',
            'decoder.head.0.gamma':'decoder.norm_out.gamma',
            'decoder.head.2.weight':'decoder.conv_out.weight',
        }
        for source,target in cases.items():
            with self.subTest(source=source): self.assertEqual(_map_vae_key(source),target)

    def test_qwen3vl_flat_text_keys_move_under_language_model_visual_stays(self):
        token=torch.zeros(2,2)
        visual=torch.zeros(2,2)
        out=_normalize_component({
            'lm_head.weight':torch.zeros(2,2),
            'model.embed_tokens.weight':token,
            'model.layers.0.self_attn.q_proj.weight':torch.zeros(2,2),
            'model.norm.weight':torch.zeros(2),
            'model.visual.patch_embed.proj.weight':visual,
        },'text_encoder')
        self.assertIs(out['model.language_model.embed_tokens.weight'],token)
        self.assertIn('model.language_model.layers.0.self_attn.q_proj.weight',out)
        self.assertIn('model.language_model.norm.weight',out)
        self.assertIs(out['model.visual.patch_embed.proj.weight'],visual)
        self.assertFalse(any(key.startswith('lm_head.') for key in out))


class EmbeddingDeviceMoveTests(unittest.TestCase):
    """Issue #2 regression: packed Embedding weights must survive device moves."""

    def test_hardened_embedding_apply_moves_parameter_intact(self):
        # Stock nn.Module._apply reassigns param.data, which can strip a custom
        # Parameter subclass wrapper; the hardened _apply must keep it intact.
        module = torch.nn.Embedding(4, 3)
        module._apply = _embedding_apply.__get__(module, type(module))
        moved = module._apply(lambda p: torch.nn.Parameter(p.clone() + 0, requires_grad=False))
        self.assertIs(moved, module)
        self.assertIsInstance(module.weight, torch.nn.Parameter)
        self.assertEqual(module.weight.shape, (4, 3))

    def test_harden_embedding_apply_covers_nested_modules(self):
        class Holder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(4, 3)

        model = Holder()
        returned = _harden_embedding_apply(model)
        self.assertIs(returned, model)
        self.assertIsInstance(model.embed.weight, torch.nn.Parameter)
        # The hardened _apply must keep parameters as Parameters after moving.
        module = model.embed
        module._apply(lambda p: torch.nn.Parameter(p.clone(), requires_grad=False))
        self.assertIsInstance(module.weight, torch.nn.Parameter)


if __name__=='__main__': unittest.main(verbosity=2)
