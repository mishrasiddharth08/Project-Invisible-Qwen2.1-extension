import importlib
import math
import sys
import types
import unittest
from unittest.mock import patch

import torch

from test_acceptance_cpu import load_package

load_package()
enh = importlib.import_module("pi_qwen21.lib.enhancer")


def attention(q, k, v, mask=None):
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores + mask
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v)


class NativeProcessor:
    _attention_backend = "native-fast"
    _parallel_config = None


class FakeAttention:
    def __init__(self, processor):
        self.processor = processor
        self.to_out = [torch.nn.Identity(), torch.nn.Identity()]

    def set_processor(self, value):
        self.processor = value


class FakeTransformer(torch.nn.Module):
    def __init__(self, processors):
        super().__init__()
        self.config = types.SimpleNamespace(causal_condition=True)
        self.transformer_blocks = [types.SimpleNamespace(attn=FakeAttention(p)) for p in processors]

    def forward(self, **kwargs):
        return kwargs["encoder_hidden_states"]


class QwenImage21Pipeline:
    def __init__(self, processors):
        self.processor = object()
        self.transformer = FakeTransformer(processors)
        self._drop_idx = 0
        self.embedding = torch.zeros(1, 2, 1)

    def _get_qwen_prompt_embeds(self, prompt=None, image=None, device=None):
        return self.embedding, None, None

    def encode_prompt(self, *args, **kwargs):
        return self.embedding, None, None


class EnhancerAttentionTests(unittest.TestCase):
    def native_module(self, calls):
        def prepare(_attn, hidden, *_args):
            return hidden

        def dispatch(q, k, v, attn_mask=None, dropout_p=0.0, backend=None, parallel_config=None):
            calls.append((attn_mask, backend))
            return attention(q, k, v, attn_mask)

        return types.SimpleNamespace(
            QwenImage21AttnProcessor=NativeProcessor,
            _qwenimage21_prepare_qkv=prepare,
            dispatch_attention_fn=dispatch,
        )

    def module_patch(self, native):
        parent = types.ModuleType("diffusers.models.transformers")
        parent.transformer_qwenimage21 = native
        return patch.dict(sys.modules, {"diffusers.models.transformers": parent})

    def activate(self, pipe, embedding):
        pipe.encode_prompt("x")
        pipe.transformer(
            encoder_hidden_states=embedding,
            img_mask=torch.tensor([[True, True]]),
            img_shapes=[[(1, 2, 2), (1, 2, 2)]],
        )

    def test_cached_target_bias_preserves_padding_and_forwards_backend(self):
        calls = []
        native = self.native_module(calls)
        pipe = QwenImage21Pipeline([NativeProcessor()])
        with self.module_patch(native):
            with enh.apply(pipe, reference_priorities="1:2"):
                self.activate(pipe, pipe.embedding)
                q = torch.ones(1, 4, 1, 1)
                k = torch.ones(1, 8, 1, 1)
                v = torch.arange(8.0).view(1, 8, 1, 1)
                valid = torch.tensor([[True] * 7 + [False]])
                out = pipe.transformer.transformer_blocks[0].attn.processor(
                    pipe.transformer.transformer_blocks[0].attn,
                    (q, k, v, 4),
                    segments=None,
                    attention_mask=valid[:, None, None, :],
                    key_valid=valid,
                )
        mask, backend = calls[-1]
        self.assertEqual(backend, "native-fast")
        self.assertTrue(torch.isneginf(mask[..., -1]).all())
        torch.testing.assert_close(mask[..., :4], torch.full((1, 1, 1, 4), math.log(2)))
        expected = attention(q, k, v, mask).flatten(2, 3)
        torch.testing.assert_close(out, expected)

    def test_uncached_bias_changes_only_target_queries(self):
        calls = []
        native = self.native_module(calls)
        pipe = QwenImage21Pipeline([NativeProcessor()])
        with self.module_patch(native):
            with enh.apply(pipe, reference_priorities="1:2"):
                self.activate(pipe, pipe.embedding)
                q = torch.ones(1, 8, 1, 1)
                k = torch.ones(1, 8, 1, 1)
                v = torch.arange(8.0).view(1, 8, 1, 1)
                out = pipe.transformer.transformer_blocks[0].attn.processor(
                    pipe.transformer.transformer_blocks[0].attn,
                    (q, k, v, 8),
                    segments=[(0, 4, False)],
                    key_valid=torch.ones(1, 8, dtype=torch.bool),
                )
        unbiased_prefix = attention(q[:, :4], k[:, :4], v[:, :4]).flatten(2, 3)
        torch.testing.assert_close(out[:, :4], unbiased_prefix)
        unbiased_target = attention(q[:, 4:], k, v).flatten(2, 3)
        self.assertFalse(torch.allclose(out[:, 4:], unbiased_target))
        self.assertEqual(calls[0][1], None)

    def test_neutral_configuration_is_exact_identity(self):
        marker = object()
        with enh.apply(marker, phrase_weights=False, reference_priorities="1:1"):
            self.assertIs(marker, marker)

    def test_partial_install_failure_restores_earlier_processors(self):
        calls = []
        native = self.native_module(calls)
        first = NativeProcessor()
        incompatible = object()
        pipe = QwenImage21Pipeline([first, incompatible])
        with self.module_patch(native):
            with self.assertRaisesRegex(ValueError, "native Qwen"):
                with enh.apply(pipe, reference_priorities="1:2"):
                    pass
        self.assertIs(pipe.transformer.transformer_blocks[0].attn.processor, first)
        self.assertIs(pipe.transformer.transformer_blocks[1].attn.processor, incompatible)
        self.assertFalse(pipe.transformer._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
