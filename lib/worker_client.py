"""Parent-side proxy for the extension-local persistent inference worker."""

import atexit
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class WorkerPipeline:
    def __init__(self, bundle, prof, offload=True, forge_root=None):
        self._lock = threading.RLock()
        self._adapters = []
        self._pending = {}
        self._temp = tempfile.TemporaryDirectory(prefix="pi-qwen21-worker-")
        base = Path(self._temp.name)
        (ROOT/'logs').mkdir(exist_ok=True)
        self._log = ROOT/'logs'/'worker.log'
        forge_root = forge_root or self._forge_root()
        command = [sys.executable, str(ROOT / "lib/worker.py"), "--root", str(ROOT),
                   "--forge-root", str(forge_root), "--log", str(self._log)]
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
        )
        atexit.register(self.close)
        try:
            self._expect("ready")
            self._send(dict(op="init", bundle=bundle, prof=prof, offload=bool(offload)))
            self._expect("initialized")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _forge_root():
        try:
            from modules.paths_internal import script_path
            return script_path
        except ImportError:
            pass
        try:
            from .assets import config
            return config().get("live_forge") or str(ROOT.parent.parent)
        except Exception:
            return str(ROOT.parent.parent)

    def _send(self, value):
        if self._process.poll() is not None:
            raise RuntimeError("Qwen worker exited; see " + str(self._log))
        self._process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _read(self):
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("Qwen worker exited; see " + str(self._log))
        value = json.loads(line)
        if value.get("type") == "error":
            message = value.get("error", "Qwen worker error") + "; see " + str(self._log)
            if "out of memory" in message.lower() or "cuda oom" in message.lower():
                try:
                    import torch
                    raise torch.cuda.OutOfMemoryError(message)
                except ImportError:
                    pass
            raise RuntimeError(message)
        return value

    def _expect(self, kind):
        value = self._read()
        if value.get("type") != kind:
            raise RuntimeError("Unexpected Qwen worker response: " + repr(value))
        return value

    @staticmethod
    def _seed(generator):
        if generator is None:
            return 0
        initial = getattr(generator, "initial_seed", None)
        return int(initial() if callable(initial) else getattr(generator, "seed", 0) or 0)

    @staticmethod
    def _save(value, path):
        if value is None:
            return None
        value.save(path, format="PNG")
        return str(path)

    def __call__(self, prompt, negative_prompt=None, true_cfg_scale=1, width=1024,
                 height=1024, num_inference_steps=40, generator=None, image=None,
                 mask_image=None, use_kv_cache=True, callback_on_step_end=None,
                 output_resolution=None, preview_every=0, preview_size=256,
                 status_callback=None, spectrum=False):
        from PIL import Image
        with self._lock, tempfile.TemporaryDirectory(prefix="job-", dir=self._temp.name) as td:
            job = Path(td)
            refs = [] if image is None else (list(image) if isinstance(image, (list, tuple)) else [image])
            image_paths = [self._save(value, job / ("input-%02d.png" % i)) for i, value in enumerate(refs)]
            mask_path = self._save(mask_image, job / "mask.png")
            output = job / "output.png"
            preview = job / "preview.png"
            stop = job / "stop"
            self._send(dict(
                op="generate", prompt=prompt, negative_prompt=negative_prompt,
                true_cfg_scale=float(true_cfg_scale), width=int(width), height=int(height),
                num_inference_steps=int(num_inference_steps), seed=self._seed(generator),
                images=image_paths, mask=mask_path, use_kv_cache=bool(use_kv_cache),
                output_resolution=output_resolution, output=str(output), stop=str(stop),
                preview_every=int(preview_every or 0),preview_size=int(preview_size or 256),preview=str(preview),
                adapters=[dict(path=p, weight=w) for p, w in self._adapters],
                spectrum=bool(spectrum),
            ))
            interrupted = None
            while True:
                value = self._read()
                kind = value.get("type")
                if kind == 'status':
                    if status_callback is not None: status_callback(value.get('status',''))
                    continue
                if kind == 'preview':
                    if callback_on_step_end is not None and interrupted is None:
                        try:
                            with Image.open(value['path']) as shown: picture=shown.copy()
                            callback_on_step_end(self,value['step'],value.get('timestep'),{'preview':picture})
                        except (OSError,ValueError):
                            pass
                        except InterruptedError as exc:
                            interrupted=exc;stop.touch()
                    continue
                if kind == "progress":
                    if callback_on_step_end is not None and interrupted is None:
                        try:
                            callback_on_step_end(self, value["step"], value.get("timestep"), {})
                        except InterruptedError as exc:
                            interrupted = exc
                            stop.touch()
                    continue
                if kind == "interrupted":
                    raise interrupted or InterruptedError(value.get("error", "Generation interrupted"))
                if kind != "result":
                    raise RuntimeError("Unexpected Qwen worker response: " + repr(value))
                if interrupted is not None:
                    raise interrupted
                with Image.open(value["path"]) as result:
                    return SimpleNamespace(images=[result.copy()])

    def unload_lora_weights(self):
        self._adapters = []
        self._pending = {}

    def load_lora_weights(self, directory, weight_name=None, adapter_name=None, **_kwargs):
        path = Path(directory) / weight_name if weight_name else Path(directory)
        name = adapter_name or path.stem
        self._pending[name] = str(path)

    def set_adapters(self, names, adapter_weights=None, **_kwargs):
        weights = adapter_weights or [1.0] * len(names)
        self._adapters = [(self._pending[name], float(weight)) for name, weight in zip(names, weights)]

    def remove_all_hooks(self):
        self.close()

    def close(self):
        atexit.unregister(self.close)
        process = getattr(self, "_process", None)
        if process is None:
            return
        self._process = None
        try:
            if process.poll() is None:
                process.stdin.write('{"op":"close"}\n'); process.stdin.flush()
                process.wait(timeout=5)
        except Exception:
            process.kill()
            try:
                process.wait(timeout=5)
            except Exception:
                pass
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except Exception:
                pass
        try:
            self._temp.cleanup()
        except Exception:
            pass

    def __del__(self):
        self.close()
