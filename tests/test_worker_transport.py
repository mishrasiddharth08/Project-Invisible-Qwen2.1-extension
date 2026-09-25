import importlib.util
import shutil
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if "pi_qwen21" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "pi_qwen21", EXTENSION_ROOT / "__init__.py",
        submodule_search_locations=[str(EXTENSION_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["pi_qwen21"] = package
    spec.loader.exec_module(package)

from pi_qwen21.lib import worker_client


COMPONENTS = r'''
import time
from types import SimpleNamespace
from PIL import Image
import torch

class AutoencoderKLQwenImage21:
    dtype = torch.float32
    config = SimpleNamespace(latents_mean=[0.0]*64, latents_std=[1.0]*64)
    def enable_tiling(self):
        self.tiling = True
    def decode(self, value, return_dict=False):
        _, _, _, h, w = value.shape
        return (torch.zeros(1, 3, 1, h*16, w*16),)

class QwenImage21Transformer2DModel:
    pass

class Qwen3VLForConditionalGeneration:
    pass

class QwenImage21Pipeline:
    def __init__(self):
        self.vae = AutoencoderKLQwenImage21()
        self.transformer = QwenImage21Transformer2DModel()
        self.text_encoder = Qwen3VLForConditionalGeneration()
        self.offloaded = False
        self.vae_scale_factor = 16
        self.image_processor = SimpleNamespace(postprocess=lambda value, output_type='pil': [Image.new('RGB', (value.shape[-1], value.shape[-2]))])

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        h, w = height // vae_scale_factor, width // vae_scale_factor
        return latents.transpose(1, 2).reshape(1, 64, 1, h, w)

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def enable_sequential_cpu_offload(self):
        self.offloaded = True

    def enable_group_offload(self, **kwargs):
        self.offloaded = True

    def unload_lora_weights(self):
        pass

    def load_lora_weights(self, *args, **kwargs):
        pass

    def set_adapters(self, *args, **kwargs):
        pass

    def __call__(self, prompt, negative_prompt=None, true_cfg_scale=1,
                 width=32, height=32, num_inference_steps=3, generator=None,
                 image=None, mask_image=None, use_kv_cache=True,
                 callback_on_step_end=None, callback_on_step_end_tensor_inputs=None,
                 output_resolution=None):
        for step in range(3):
            time.sleep(.03)
            if callback_on_step_end:
                callback_on_step_end(self, step, float(step), {'latents': torch.zeros(1, (height//16)*(width//16), 64)})
        if image:
            assert isinstance(image, list)
            result = image[0].convert('RGBA')
        else:
            result = Image.new('RGBA', (width, height), (11, 22, 33, 44))
        return SimpleNamespace(images=[result])

def pipeline(bundle):
    assert bundle['folder'] == 'fixture'
    return QwenImage21Pipeline()
'''


ADAPTER = r'''
def clear(pipe):
    pipe.unload_lora_weights()

def apply(pipe, adapters):
    pipe.unload_lora_weights()
    for path, weight in adapters:
        assert path and isinstance(weight, float)
'''


class WorkerTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pi-qwen21-transport-test-")
        self.root = Path(self.temp.name) / "fixture"
        (self.root / "lib").mkdir(parents=True)
        (self.root / "lora").mkdir()
        (self.root / "_deps").mkdir()
        (self.root / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "lib/__init__.py").write_text("", encoding="utf-8")
        (self.root / "lora/__init__.py").write_text("", encoding="utf-8")
        (self.root / "lib/components.py").write_text(textwrap.dedent(COMPONENTS), encoding="utf-8")
        (self.root / "lora/adapter.py").write_text(textwrap.dedent(ADAPTER), encoding="utf-8")
        shutil.copy2(Path(worker_client.__file__).with_name("worker.py"), self.root / "lib/worker.py")
        shutil.copy2(Path(worker_client.__file__).with_name("preview.py"), self.root / "lib/preview.py")
        shutil.copy2(Path(worker_client.__file__).with_name("spectrum.py"), self.root / "lib/spectrum.py")
        self.root_patch = mock.patch.object(worker_client, "ROOT", self.root)
        self.root_patch.start()
        self.pipe = worker_client.WorkerPipeline(
            {"folder": "fixture", "files": {}},
            {"dit": "bf16", "te": "bf16", "side": 1024},
            offload=True, forge_root=self.root,
        )

    def tearDown(self):
        if getattr(self, "pipe", None) is not None:
            self.pipe.close()
        self.root_patch.stop()
        self.temp.cleanup()

    def test_persistent_round_trip_t2i_then_reference_preserves_rgba(self):
        process = self.pipe._process
        first = self.pipe(
            prompt="t2i", width=16, height=12, num_inference_steps=3,
            generator=type("G", (), {"initial_seed": lambda self: 123})(),
        ).images[0]
        self.assertEqual(first.mode, "RGBA")
        self.assertEqual(first.size, (16, 12))
        self.assertEqual(first.getpixel((0, 0)), (11, 22, 33, 44))

        reference = Image.new("RGBA", (9, 7), (90, 80, 70, 60))
        second = self.pipe(prompt="edit", image=[reference]).images[0]
        self.assertIs(self.pipe._process, process)
        self.assertIsNone(process.poll())
        self.assertEqual(second.mode, "RGBA")
        self.assertEqual(second.size, reference.size)
        self.assertEqual(second.getpixel((0, 0)), (90, 80, 70, 60))

    def test_progress_interrupt_consumes_child_completion_and_worker_remains_usable(self):
        progress = []

        def stop(_pipe, step, _timestep, _kwargs):
            progress.append(step)
            raise InterruptedError("stop requested")

        with self.assertRaisesRegex(InterruptedError, "stop requested"):
            self.pipe(prompt="interrupt", callback_on_step_end=stop)
        self.assertTrue(progress)
        self.assertIsNone(self.pipe._process.poll())
        result = self.pipe(prompt="after interrupt", width=5, height=4).images[0]
        self.assertEqual(result.size, (5, 4))

    def test_remove_all_hooks_closes_worker(self):
        process = self.pipe._process
        self.pipe.remove_all_hooks()
        self.assertIsNotNone(process.poll())
        self.assertIsNone(self.pipe._process)
        self.pipe = None

    def test_status_events_do_not_increment_progress(self):
        statuses=[];steps=[]
        self.pipe(prompt='status',num_inference_steps=3,status_callback=statuses.append,
                  callback_on_step_end=lambda _p,step,_t,_kw: steps.append(step))
        self.assertEqual(statuses,['encoding','decoding'])
        # The last entry is the final-step callback that carries the finished
        # full-size picture; progress events themselves are still 0,1,2.
        self.assertEqual(steps,[0,1,2,2])

    def test_live_preview_round_trip_uses_same_step_without_replacing_final(self):
        events=[]
        self.pipe(prompt='preview',width=512,height=512,num_inference_steps=3,preview_every=1,preview_size=256,
                  callback_on_step_end=lambda _p,step,_t,kw: events.append((step,kw.get('preview'),kw.get('final'))))
        live=[(step,image) for step,image,final in events if image is not None and not final]
        finals=[(step,image) for step,image,final in events if image is not None and final]
        self.assertEqual(len(live),1)
        self.assertLessEqual(max(live[0][1].size),256)
        self.assertEqual([step for step,image in live],[0])
        # The finished picture arrives the moment the last step ends, full size.
        self.assertEqual([step for step,image in finals],[2])
        self.assertEqual(max(finals[0][1].size),512)


class SamplingHeartbeatTests(unittest.TestCase):
    def test_reports_real_layer_position_checks_stop_and_removes_hooks(self):
        worker_path=Path(worker_client.__file__).with_name('worker.py')
        spec=importlib.util.spec_from_file_location('pi_worker_heartbeat_test',worker_path)
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

        class Handle:
            def __init__(self): self.removed=False
            def remove(self): self.removed=True
        class Block:
            def __init__(self): self.hook=None;self.handle=Handle()
            def register_forward_hook(self,hook): self.hook=hook;return self.handle
        blocks=[Block(),Block()]
        pipe=types.SimpleNamespace(transformer=types.SimpleNamespace(transformer_blocks=blocks))
        stop=types.SimpleNamespace(exists=lambda: False)
        times=iter([0.0,1.1,1.2])
        statuses=[]
        with worker._sampling_heartbeat(pipe,stop,4,[0],statuses.append,clock=lambda: next(times)):
            blocks[0].hook(None,None,None)
            blocks[1].hook(None,None,None)
        self.assertEqual(statuses,['sampling step 1/4, layer 1/2'])
        self.assertTrue(all(block.handle.removed for block in blocks))

        stopped=types.SimpleNamespace(exists=lambda: True)
        with self.assertRaises(InterruptedError):
            with worker._sampling_heartbeat(pipe,stopped,4,[1],statuses.append,clock=lambda: 5.0):
                blocks[0].hook(None,None,None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
