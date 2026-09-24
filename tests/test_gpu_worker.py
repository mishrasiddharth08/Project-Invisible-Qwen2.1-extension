import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pi_gpu_worker',ROOT/'lib/worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
if 'pi_qwen21' not in sys.modules:
    package_spec=importlib.util.spec_from_file_location('pi_qwen21',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
    package=importlib.util.module_from_spec(package_spec);sys.modules['pi_qwen21']=package;package_spec.loader.exec_module(package)
from pi_qwen21.lib.assets import hardware_profile, quant_capable


class FakeCuda:
    def __init__(self,available=True,bf16=True,gb=24): self.available,self.bf16,self.gb=available,bf16,gb
    def is_available(self): return self.available
    def is_bf16_supported(self): return self.bf16
    def get_device_properties(self,_index): return types.SimpleNamespace(total_memory=self.gb*2**30)

class VAE:
    def __init__(self): self.tiled=True;self.sliced=False
    def enable_tiling(self): raise AssertionError('Qwen 2.1 tiled decoding causes artifacts')
    def disable_tiling(self): self.tiled=False
    def enable_slicing(self): self.sliced=True

class Pipe:
    def __init__(self,sequential=True):
        self.vae=VAE();self.mode=None
        if not sequential: self.enable_sequential_cpu_offload=None
    def enable_sequential_cpu_offload(self): self.mode='sequential'
    def enable_model_cpu_offload(self): self.mode='model'
    def enable_group_offload(self,**kwargs): self.mode='group';self.group_kwargs=kwargs
    def to(self,device): self.mode=device


class GPUWorkerTests(unittest.TestCase):
    def torch(self,available=True,bf16=True,hip=None,gb=24,cuda='13.0'):
        return types.SimpleNamespace(cuda=FakeCuda(available,bf16,gb),
                                     version=types.SimpleNamespace(hip=hip,cuda=cuda),
                                     bfloat16='bf16',float16='fp16')

    def test_dtype_prefers_bf16_and_falls_back_to_fp16(self):
        self.assertEqual(worker._compute_dtype(self.torch(bf16=True)),'bf16')
        self.assertEqual(worker._compute_dtype(self.torch(bf16=False)),'fp16')
        with self.assertRaisesRegex(ValueError,'CUDA or ROCm'):
            worker._compute_dtype(self.torch(available=False))

    def test_rocm_quant_detection_uses_actual_component_paths_not_profile_preference(self):
        self.assertFalse(worker._quantized({'files':{}},{'dit':'int8_convrot','te':'bf16'}))
        self.assertTrue(worker._quantized({'files':{'text_encoder':'x_w4a8.safetensors'}},{'te':'bf16'}))
        self.assertFalse(worker._quantized({'files':{'vae':'vae_bf16.safetensors'}},{'dit':'bf16','te':'bf16'}))

    def test_hardware_profile_amd_forces_portable_bf16_storage(self):
        result=hardware_profile(self.torch(hip='6.2',gb=16))
        self.assertEqual((result['dit'],result['te']),('bf16','bf16'))
        self.assertTrue(result['portable'])

    def test_hardware_profile_non_bf16_nvidia_forces_portable_bf16_storage(self):
        result=hardware_profile(self.torch(bf16=False,gb=12))
        self.assertEqual((result['dit'],result['te']),('bf16','bf16'))
        self.assertTrue(result['portable'])

    def test_hardware_profile_4gb_and_6gb_choose_recovery_sides(self):
        self.assertEqual(hardware_profile(self.torch(gb=4))['side'],512)
        self.assertEqual(hardware_profile(self.torch(gb=6))['side'],768)

    def test_quant_capable_requires_cuda13_bf16_nvidia(self):
        self.assertTrue(quant_capable(self.torch(gb=12)))
        self.assertFalse(quant_capable(self.torch(hip='6.2',gb=12)))
        self.assertFalse(quant_capable(self.torch(bf16=False,gb=12)))
        self.assertFalse(quant_capable(self.torch(available=False,gb=12)))

    def test_quant_capable_rejects_old_cuda_torch_builds(self):
        self.assertFalse(quant_capable(self.torch(cuda='12.8',gb=12)))
        self.assertFalse(quant_capable(self.torch(cuda=None,gb=12)))
        self.assertFalse(quant_capable(self.torch(cuda='not-a-version',gb=12)))

    def test_hardware_profile_old_cuda_torch_forces_bf16_fallback(self):
        result=hardware_profile(self.torch(cuda='12.4',gb=12))
        self.assertEqual((result['dit'],result['te']),('bf16','bf16'))
        self.assertTrue(result['portable'])

    def test_hardware_profile_keeps_quant_on_capable_cuda13_card(self):
        result=hardware_profile(self.torch(gb=12))
        self.assertEqual(result['dit'],'int8_convrot')
        self.assertNotIn('portable',result)

    def test_low_memory_uses_leaf_group_offload_and_enables_vae_memory_features(self):
        pipe=Pipe();worker._prepare_pipe(pipe,{'te':'w4a8','side':1024},True)
        self.assertEqual(pipe.mode,'group');self.assertFalse(pipe.vae.tiled);self.assertTrue(pipe.vae.sliced)
        self.assertEqual(pipe.group_kwargs['offload_type'],'leaf_level')
        self.assertFalse(pipe.group_kwargs['use_stream'])

    def test_quant_low_memory_without_group_support_uses_model_not_accelerate_sequential(self):
        pipe=Pipe();pipe.enable_group_offload=None
        worker._prepare_pipe(pipe,{'te':'w4a8','side':512},True,quantized=True)
        self.assertEqual(pipe.mode,'model')
        self.assertFalse(pipe.vae.tiled)

    def test_normal_profile_uses_model_offload_and_disabled_offload_moves_to_gpu(self):
        pipe=Pipe();worker._prepare_pipe(pipe,{'te':'bf16','side':2048},True)
        self.assertEqual(pipe.mode,'model')
        direct=Pipe();worker._prepare_pipe(direct,{'te':'bf16','side':2048},False)
        self.assertEqual(direct.mode,'cuda')
        self.assertFalse(direct.vae.tiled)

    def test_dequantized_load_still_gets_low_memory_group_offload(self):
        # After unpacking to plain bf16 the weights are ordinary tensors, so
        # every offload mode is safe again (the quantized guard only blocked
        # the accelerate-sequential fallback for packed subclass weights).
        pipe=Pipe();worker._prepare_pipe(pipe,{'te':'w4a8','side':1024},True,quantized=False)
        self.assertEqual(pipe.mode,'group')
        plain_no_group=Pipe();plain_no_group.enable_group_offload=None
        worker._prepare_pipe(plain_no_group,{'te':'w4a8','side':1024},True,quantized=False)
        self.assertEqual(plain_no_group.mode,'sequential')


if __name__=='__main__': unittest.main(verbosity=2)
