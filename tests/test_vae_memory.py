import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('vae_memory',Path(__file__).resolve().parents[1]/'lib/vae_memory.py')
memory=importlib.util.module_from_spec(spec);spec.loader.exec_module(memory)


class VaeMemoryTests(unittest.TestCase):
    def test_strips_match_full_image_including_odd_sizes_and_edges(self):
        torch.manual_seed(17)
        for shape in [(1,2,13,17),(2,2,1,7),(1,2,14,8)]:
            fn=torch.nn.Sequential(torch.nn.Upsample(scale_factor=2,mode='nearest-exact'),torch.nn.Conv2d(2,3,3,padding=1))
            x=torch.randn(shape)
            with torch.inference_mode():
                expected=fn(x);actual=memory.strip_apply(fn,x,limit=80)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_small_and_training_calls_are_unchanged(self):
        x=torch.ones(1,1,3,3)
        calls=[]
        def fn(value): calls.append(value);return value
        self.assertIs(memory.strip_apply(fn,x,limit=1),x)
        with torch.inference_mode(): self.assertIs(memory.strip_apply(fn,x),x)
        self.assertTrue(all(value is x for value in calls))

    def test_known_vae_installs_once_without_changing_weights(self):
        class QwenImage21Resample(torch.nn.Module):
            def __init__(self):
                super().__init__();self.mode='upsample2d'
                self.resample=torch.nn.Sequential(torch.nn.Upsample(scale_factor=2,mode='nearest-exact'),torch.nn.Conv2d(2,3,3,padding=1))
        class AutoencoderKLQwenImage21(torch.nn.Module):
            def __init__(self): super().__init__();self.up=QwenImage21Resample()
        vae=AutoencoderKLQwenImage21();keys=list(vae.state_dict());weight=vae.up.resample[1].weight
        self.assertEqual(memory.install(vae,80),1)
        self.assertEqual(memory.install(vae,80),0)
        self.assertEqual(list(vae.state_dict()),keys)
        self.assertIs(vae.up.resample[1].weight,weight)
        vae.up.resample[1].dilation=(2,2);vae.up.resample._pi_strips=False
        self.assertEqual(memory.install(vae),0)

    def test_unknown_vae_is_untouched(self):
        self.assertEqual(memory.install(torch.nn.Linear(2,2)),0)
