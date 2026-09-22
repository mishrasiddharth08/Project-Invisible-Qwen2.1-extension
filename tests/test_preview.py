import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if 'pi_qwen21' not in sys.modules:
    spec=importlib.util.spec_from_file_location('pi_qwen21',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
    package=importlib.util.module_from_spec(spec);sys.modules['pi_qwen21']=package;spec.loader.exec_module(package)
from pi_qwen21.lib.preview import decode_preview,has_preview_memory,preview_context


class VAE:
    dtype=torch.float32
    config=types.SimpleNamespace(latents_mean=[0.0]*64,latents_std=[1.0]*64)
    def __init__(self): self.seen=None
    def decode(self,value,return_dict=False):
        self.seen=value.shape
        _,_,_,h,w=value.shape
        return (torch.zeros(1,3,1,h*16,w*16),)

class Processor:
    def postprocess(self,value,output_type='pil'):
        return [Image.new('RGB',(value.shape[-1],value.shape[-2]))]

class Pipe:
    vae_scale_factor=16
    def __init__(self): self.vae=VAE();self.image_processor=Processor()
    @staticmethod
    def _unpack_latents(latents,height,width,vae_scale_factor):
        h,w=height//vae_scale_factor,width//vae_scale_factor
        return latents.transpose(1,2).reshape(1,64,1,h,w)


class PreviewTests(unittest.TestCase):
    def test_large_latents_are_downsampled_before_qwen_vae_decode(self):
        pipe=Pipe();latents=torch.zeros(1,128*128,64)
        image=decode_preview(pipe,latents,2048,2048,256)
        self.assertEqual(pipe.vae.seen,(1,64,1,16,16))
        self.assertEqual(image.size,(256,256))

    def test_preview_size_is_bounded_to_256(self):
        pipe=Pipe();latents=torch.zeros(1,64*32,64)
        image=decode_preview(pipe,latents,1024,512,999)
        self.assertLessEqual(max(image.size),256)

    def test_low_free_memory_skips_preview(self):
        cuda=types.SimpleNamespace(mem_get_info=lambda: (2*1024**3-1,24*1024**3))
        self.assertFalse(has_preview_memory(types.SimpleNamespace(cuda=cuda)))
        cuda.mem_get_info=lambda: (3*1024**3,24*1024**3)
        self.assertTrue(has_preview_memory(types.SimpleNamespace(cuda=cuda)))

    def test_model_offload_context_restores_hook_and_never_offloads_transformer(self):
        class Previous:
            def __init__(self): self.calls=0
            def offload(self): self.calls+=1
        previous=Previous()
        hook=types.SimpleNamespace(prev_module_hook=previous,init_hook=lambda vae: setattr(vae,'returned',True))
        vae=types.SimpleNamespace(_hf_hook=hook,returned=False)
        pipe=types.SimpleNamespace(_pi_offload_mode='model',vae=vae)
        with self.assertRaisesRegex(RuntimeError,'preview failure'):
            with preview_context(pipe) as allowed:
                self.assertTrue(allowed);self.assertIsNone(hook.prev_module_hook)
                raise RuntimeError('preview failure')
        self.assertIs(hook.prev_module_hook,previous)
        self.assertTrue(vae.returned)
        self.assertEqual(previous.calls,0)

    def test_unknown_model_offload_hook_skips_safely(self):
        pipe=types.SimpleNamespace(_pi_offload_mode='model',vae=types.SimpleNamespace())
        with preview_context(pipe) as allowed: self.assertFalse(allowed)


if __name__=='__main__': unittest.main(verbosity=2)
