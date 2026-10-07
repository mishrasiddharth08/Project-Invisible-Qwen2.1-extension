import importlib.util
from pathlib import Path
import types
import unittest
import test_gpu_worker as gpu
worker=gpu.worker

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('vram_assets',ROOT/'lib/assets.py')
assets=importlib.util.module_from_spec(spec);spec.loader.exec_module(assets)
spec=importlib.util.spec_from_file_location('vram_comfy',ROOT/'lib/comfy_worker.py')
comfy=importlib.util.module_from_spec(spec);spec.loader.exec_module(comfy)

class VRAMProfiles(unittest.TestCase):
    def test_six_tiers_have_exact_ceiling_and_expected_defaults(self):
        for gb,side,te in [(8,1024,'w4a8'),(10,1024,'w4a8'),(12,1024,'int8_convrot'),(16,1536,'int8_convrot'),(24,2048,'int8_convrot'),(32,0,'int8_convrot')]:
            with self.subTest(gb=gb):
                p=assets.profile(gb)
                self.assertEqual((p['vram_gb'],p['side'],p['te']),(gb,side,te))
                self.assertGreater(p['reserve_gb'],0)
                self.assertEqual(p['offload'],gb<28)

    def test_lower_override_never_exceeds_physical_card(self):
        self.assertEqual(assets.profile(32,'8')['vram_gb'],8)
        self.assertEqual(assets.profile(7.9,'8')['vram_gb'],7.9)
        for value in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):assets.profile(value)
        fake=gpu.GPUWorkerTests().torch(gb=12)
        for value in ['0','-1','nan','inf']:
            self.assertEqual(assets.hardware_profile(fake,value),assets.hardware_profile(fake))

    def test_both_workers_enforce_every_tier_and_external_usage(self):
        for gb in (8,10,12,16,24,32):
            for fn in (worker._set_cuda_budget,comfy._cuda_budget):
                with self.subTest(gb=gb,worker=fn.__module__):
                    fake=gpu.GPUWorkerTests().torch(gb=32);calls=[]
                    fake.cuda.mem_get_info=lambda _: (30*2**30,32*2**30)
                    fake.cuda.set_per_process_memory_fraction=lambda fraction,index:calls.append(fraction)
                    p=assets.profile(32,str(gb))
                    expected=gb-2-p['reserve_gb']
                    self.assertAlmostEqual(fn(fake,p),expected)
                    self.assertAlmostEqual(calls[0],expected/32)

    def test_automatic_workload_guards_and_explicit_override(self):
        for gb,limit in [(8,512),(10,640),(12,768),(16,1024),(24,1536),(32,2048)]:
            p=assets.profile(gb)
            self.assertEqual(assets.generation_side(p,control=True),limit)
            self.assertEqual(assets.generation_side(p,references=4),limit)
        self.assertEqual(assets.generation_side(assets.profile(16),cfg=2),1280)
        self.assertEqual(assets.generation_side(assets.profile(24),3072,control=True),3072)
        self.assertEqual(assets.generation_side(assets.profile(8),3072),1024)
        self.assertEqual(assets.generation_side(assets.profile(32)),0)

    def test_tiny_positive_budget_does_not_round_up_allocator_ceiling(self):
        for fn in (worker._set_cuda_budget,comfy._cuda_budget):
            fake=gpu.GPUWorkerTests().torch(gb=32);calls=[]
            fake.cuda.mem_get_info=lambda _: (32*2**30,32*2**30)
            fake.cuda.set_per_process_memory_fraction=lambda fraction,index:calls.append(fraction)
            usable=fn(fake,{'vram_gb':.51,'reserve_gb':.5})
            self.assertAlmostEqual(usable,.01)
            self.assertLess(calls[0],.01)
