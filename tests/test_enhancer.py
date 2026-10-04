import contextlib
import importlib
import math
import sys
import types
import unittest
from unittest.mock import patch
import torch
from test_acceptance_cpu import load_package
load_package()
enh=importlib.import_module('pi_qwen21.lib.enhancer')

class EnhancerTests(unittest.TestCase):
    def test_parser_preserves_repeated_phrase_occurrence(self):
        clean,spans=enh.phrases('blue and (blue:1.3), (green:0)')
        self.assertEqual(clean,'blue and blue, green')
        self.assertEqual(spans,[(9,13,1.3),(15,20,0)])
        for text in ('(blue:-1)','(blue:1e999)','(blue:1e-999)'):
            with self.assertRaises(ValueError):enh.phrases(text)

    def test_reference_validation_last_value_wins(self):
        self.assertEqual(enh.references('1:1.2, 2:0.8; 1:1'),{2:.8})
        for text in ('0:1','11:1','1:nan','1:9','oops'):
            with self.assertRaises(ValueError):enh.references(text)

    def test_joint_layout_and_zero_reference(self):
        mask=torch.tensor([[False,True,False,True]])
        shapes=[[(1,2,2),(1,2,2)]]
        bias,target=enh.joint_bias(torch,mask,shapes,{2:math.log(1.3)},{1:0},'cpu',torch.float32)
        self.assertEqual(target,4)
        self.assertEqual(bias.shape,(1,1,1,10))
        self.assertTrue(torch.isneginf(bias[...,1:5]).all())
        self.assertAlmostEqual(float(bias[...,5]),math.log(1.3),places=6)
        self.assertTrue((bias[...,6:]==0).all())
        with self.assertRaises(ValueError):enh.joint_bias(torch,mask,shapes,{}, {2:1.2},'cpu',torch.float32)

    def test_bias_scales_attention_odds_and_preserves_padding(self):
        bias=torch.tensor([[[[0.,math.log(2),0.]]]])
        mask=torch.tensor([[[[True,True,False]]]])
        probabilities=enh.add_mask(torch,bias,mask).softmax(-1)
        torch.testing.assert_close(probabilities,torch.tensor([[[[1/3,2/3,0.]]]]))

    def test_token_mapping_only_weights_selected_occurrence(self):
        class Tokenizer:
            is_fast=True
            def __call__(self,*args,**kwargs):return {'input_ids':[1,2,3], 'offset_mapping':[(0,4),(5,8),(9,13)]}
        rows=enh.token_rows(Tokenizer(),'blue and blue','blue and blue',[(9,13,2.)],[1,2,3],0)
        self.assertEqual(rows,{2:math.log(2)})
        with self.assertRaises(ValueError):enh.token_rows(Tokenizer(),'blue and blue','blue and blue',[(0,2,2.)],[1,2,3],0)
        with self.assertRaises(ValueError):enh.token_rows(Tokenizer(),'blue and blue','blue and blue',[(9,13,2.)],[1,2,4],0)

    def test_disabled_context_does_not_access_pipeline(self):
        with enh.apply(object()):pass

    def test_processors_and_encoding_restored_on_error(self):
        class Native:pass
        class QwenImage21Pipeline:
            def _get_qwen_prompt_embeds(self,*a,**k):pass
            def encode_prompt(self,*a,**k):pass
        original=Native()
        attn=types.SimpleNamespace(processor=original)
        attn.set_processor=lambda p:setattr(attn,'processor',p)
        pipe=QwenImage21Pipeline();pipe.processor=object()
        pipe.transformer=torch.nn.Module();pipe.transformer.config=types.SimpleNamespace(causal_condition=True)
        pipe.transformer.transformer_blocks=[types.SimpleNamespace(attn=attn)]
        processor=pipe.processor
        fake=types.ModuleType('diffusers.models.transformers')
        fake.transformer_qwenimage21=types.SimpleNamespace(QwenImage21AttnProcessor=Native)
        with patch.dict(sys.modules,{'diffusers.models.transformers':fake}):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with enh.apply(pipe,True):
                    self.assertIsNot(attn.processor,original)
                    raise RuntimeError('injected')
        self.assertIs(attn.processor,original)
        self.assertIs(pipe.processor,processor)
        self.assertNotIn('encode_prompt',pipe.__dict__)
        self.assertFalse(pipe.transformer._forward_pre_hooks)
