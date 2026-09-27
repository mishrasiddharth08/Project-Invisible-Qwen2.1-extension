import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

spec=importlib.util.spec_from_file_location('idle_worker',Path(__file__).resolve().parents[1]/'lib/worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class IdleMemoryTests(unittest.TestCase):
    def test_offloaded_worker_releases_allocator_after_offload(self):
        events=[]
        pipe=NS(_pi_offload_mode='model',maybe_free_model_hooks=lambda:events.append('offload'))
        torch=NS(cuda=NS(is_available=lambda:True,empty_cache=lambda:events.append('release')))
        worker._release_idle_memory(pipe,torch)
        self.assertEqual(events,['offload','release'])

    def test_direct_mode_preserves_resident_pipeline(self):
        worker._release_idle_memory(NS(_pi_offload_mode='direct'),None)

if __name__=='__main__':unittest.main()
