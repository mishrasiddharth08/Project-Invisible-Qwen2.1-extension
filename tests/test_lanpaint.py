import contextlib
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from PIL import Image
import numpy as np
import torch
from test_acceptance_cpu import load_package
import importlib

load_package()
lp=importlib.import_module('pi_qwen21.lib.lanpaint')

class Transformer(torch.nn.Module):
    def cache_context(self, name): return contextlib.nullcontext()
    def forward(self, hidden_states, timestep, **kwargs): return (hidden_states*.1,)

class FlowMatchEulerDiscreteScheduler:
    def __init__(self): self.sigmas=torch.tensor([.9,.4,0]); self.step_index=None
    def _init_step_index(self, t): self.step_index=0
    def step(self, velocity, timestep, sample, return_dict=False):
        i=self.step_index; self.step_index+=1
        return (sample+(self.sigmas[i+1]-self.sigmas[i])*velocity,)

class QwenImage21Pipeline:
    def __init__(self):
        self.scheduler=FlowMatchEulerDiscreteScheduler(); self.transformer=Transformer()
        self.vae=NS(dtype=torch.float32); self.latent_channels=4
        self.image_processor=NS(preprocess=lambda *a,**k: torch.ones(1,4,2,2))
    def _encode_vae_image(self, image, generator): return image
    def _pack_latents(self, x, b,c,h,w): return x.reshape(b,c,h*w).transpose(1,2)
    def prepare_latents(self, images, batch_size, num_channels_latents, height, width,
                        dtype, device, generator, latents=None):
        return torch.zeros(1,4,4),None

class LanPaintTests(unittest.TestCase):
    def test_mask_inversion_and_empty_rejection(self):
        image=Image.new('RGBA',(4,4)); mask=Image.new('L',(4,4),255)
        self.assertEqual(lp.prepare_mask(image,mask).getextrema(),(255,255))
        with self.assertRaises(ValueError): lp.prepare_mask(image,mask,True)
        with self.assertRaises(ValueError): lp.prepare_mask(image,None)

    def test_preserves_unmasked_rgba_pixels(self):
        src=Image.new('RGBA',(4,4),(3,4,5,6)); result=Image.new('RGBA',(4,4),(9,8,7,255))
        mask=Image.new('L',(4,4)); mask.putpixel((2,2),255)
        out=lp.preserve(src,result,mask)
        self.assertEqual(out.getpixel((0,0)),src.getpixel((0,0)))
        self.assertEqual(out.getpixel((2,2)),result.getpixel((2,2)))

    def run_sampler(self, cfg):
        pipe=QwenImage21Pipeline()
        source=Image.new('RGBA',(2,2)); mask=Image.new('L',(2,2)); mask.putpixel((1,1),255)
        with lp.sampling(pipe,source,mask,2,2,1,cfg,123):
            x,_=pipe.prepare_latents(None,1,4,2,2,torch.float32,'cpu',None)
            for t in (.9,.4):
                noise=pipe.transformer(hidden_states=x,timestep=torch.tensor([t]))[0]
                if cfg>1: pipe.transformer(hidden_states=x,timestep=torch.tensor([t]))
                x=pipe.scheduler.step(noise,t,x)[0]
        self.assertNotIn('step',pipe.scheduler.__dict__)
        self.assertNotIn('prepare_latents',pipe.__dict__)
        self.assertFalse(pipe.transformer._forward_pre_hooks)
        self.assertTrue(torch.isfinite(x).all())
        torch.testing.assert_close(x[:,:3],torch.ones_like(x[:,:3]))
        return x

    def test_core_deterministic_cond_and_cfg(self):
        for cfg in (1,2):
            torch.testing.assert_close(self.run_sampler(cfg),self.run_sampler(cfg))

    def test_hooks_removed_on_error(self):
        pipe=QwenImage21Pipeline()
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with lp.sampling(pipe,Image.new('RGBA',(2,2)),Image.new('L',(2,2),255),2,2):
                raise RuntimeError('injected')
        self.assertFalse(pipe.transformer._forward_pre_hooks)
        self.assertNotIn('step',pipe.scheduler.__dict__)

    def test_cancel_before_forward(self):
        pipe=QwenImage21Pipeline()
        with self.assertRaises(InterruptedError):
            with lp.sampling(pipe,Image.new('RGBA',(2,2)),Image.new('L',(2,2),255),2,2,stop=NS(exists=lambda:True)):
                pipe.transformer(hidden_states=torch.zeros(1,4,4),timestep=torch.ones(1))
        self.assertFalse(pipe.transformer._forward_pre_hooks)
