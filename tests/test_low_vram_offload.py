import importlib.util
import sys
import types
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pi_low_vram',ROOT/'lib/offload.py')
offload=importlib.util.module_from_spec(spec);spec.loader.exec_module(offload)


class Packed(torch.Tensor):
    @staticmethod
    def __new__(cls, data):
        out=torch.Tensor._make_wrapper_subclass(cls,data.shape,device=data.device,dtype=data.dtype,requires_grad=False)
        out.inner=data
        return out
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._to_copy.default:
            value=args[0]
            return Packed(value.inner.to(**(kwargs or {})))
        if func is torch.ops.aten.detach.default:
            return args[0]
        raise NotImplementedError(str(func))


class PackedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(Packed(torch.ones(2,2)),requires_grad=False)
    def _apply(self,fn,recurse=True):
        moved=fn(self.weight)
        self.register_parameter('weight',torch.nn.Parameter(moved,requires_grad=False))
        return self
    def forward(self,x):
        self.seen=(type(self.weight),self.weight.device,x.device)
        return x+1


class FakeCuda:
    def is_available(self): return False
    def empty_cache(self): raise AssertionError


class RMS(torch.nn.Module):
    def __init__(self,channels):
        super().__init__();self.channel_first=True;self.scale=channels**0.5
        self.gamma=torch.nn.Parameter(torch.randn(channels,1,1));self.bias=torch.nn.Parameter(torch.randn(channels,1,1))
    def forward(self,x):
        return torch.nn.functional.normalize(x.float(),dim=1).to(x.dtype)*self.scale*self.gamma+self.bias


class DecodeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__();self.proj=torch.nn.Linear(4,4,bias=False)
    def decode(self,value,return_dict=False):
        self.seen=(value.device,self.proj.weight.device)
        if value.device.type == 'cuda':
            raise torch.OutOfMemoryError('fixture GPU decode OOM')
        return (self.proj(value),)


class LowVramTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_gpu_decode_oom_falls_back_to_exact_cpu_latents_and_restores_method(self):
        vae=DecodeVAE();original=vae.decode
        manager=offload.LayerOffload(torch);manager.install(vae);manager.wrap_vae_decode(vae,force_cpu=False)
        latent=torch.randn(2,4,device='cuda')
        result=vae.decode(latent,return_dict=False)[0]
        self.assertEqual(tuple(device.type for device in vae.seen),('cpu','cpu'))
        self.assertEqual(result.device.type,'cpu')
        manager.remove()
        self.assertEqual(vae.decode,original)

    def test_cancel_stops_before_module_work(self):
        model=torch.nn.Linear(2,2)
        manager=offload.LayerOffload(torch,device='cpu');manager.install(model)
        manager.cancel=lambda: True
        with self.assertRaisesRegex(InterruptedError,'interrupted'):
            model(torch.zeros(1,2))
        manager.cancel=lambda: False
        resumed=model(torch.zeros(1,2))
        self.assertEqual(tuple(resumed.shape),(1,2))
        manager.remove()

    def test_chunked_rms_matches_original_exactly(self):
        module=RMS(32).to(dtype=torch.bfloat16)
        x=torch.randn(2,32,65,71,dtype=torch.bfloat16)
        expected=module(x)
        actual=offload._chunked_rms(module,x,module.forward,torch,max_float_bytes=32*71*4*3)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_manager_patches_and_restores_qwen_rms(self):
        RMS.__name__='QwenImage21RMS_norm'
        module=RMS(8)
        original=module.forward
        manager=offload.LayerOffload(torch,device='cpu');manager.install(module)
        self.assertTrue(manager.patches)
        manager.remove()
        self.assertEqual(module.forward,original)

    def test_owner_move_preserves_packed_subclass(self):
        module=PackedLinear()
        offload._move_own(module,'cpu')
        self.assertIsInstance(module.weight,Packed)

    def test_hooks_keep_output_and_return_weights_to_cpu(self):
        model=torch.nn.Sequential(PackedLinear(),torch.nn.ReLU())
        manager=offload.LayerOffload(types.SimpleNamespace(cuda=FakeCuda(),device=torch.device,is_tensor=torch.is_tensor),device='cpu')
        manager.install(model)
        result=model(torch.zeros(1,2))
        self.assertEqual(result.tolist(),[[1.,1.]])
        self.assertEqual(model[0].seen[0],Packed)
        self.assertEqual(model[0].weight.device.type,'cpu')
        self.assertEqual(model._hf_hook.execution_device.type,'cpu')
        manager.remove()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_root_moves_nested_inputs_to_execution_device(self):
        class Probe(torch.nn.Module):
            def forward(self, value, meta=None):
                self.devices=(value.device,meta['mask'].device)
                return value
        model=Probe()
        manager=offload.LayerOffload(torch,device='cuda')
        manager.install(model)
        result=model(torch.zeros(1),meta={'mask':torch.ones(1)})
        self.assertEqual(tuple(device.type for device in model.devices),('cuda','cuda'))
        self.assertEqual(result.device.type,'cuda')
        manager.remove()


if __name__=='__main__': unittest.main()
