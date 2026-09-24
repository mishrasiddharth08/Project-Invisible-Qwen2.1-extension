"""Persistent isolated Qwen 2.1 inference worker; JSON-lines protocol only."""

import argparse
import contextlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path


WIRE = sys.stdout

def _compute_dtype(torch):
    if not torch.cuda.is_available():
        raise ValueError('Qwen-Image-2.1 requires a CUDA or ROCm GPU')
    supported=getattr(torch.cuda,'is_bf16_supported',lambda: False)
    return torch.bfloat16 if supported() else torch.float16

def _quantized(bundle,prof=None):
    values=list((bundle.get('files') or {}).values())
    return any(any(tag in str(value).lower() for tag in ('int8','w4a8','convrot','fp8','nvfp4','mxfp8')) for value in values if value)

def _prepare_pipe(pipe,prof,offload,quantized=False):
    # Qwen 2.1 tiled decoding introduces colored seams and damages alpha.
    if hasattr(pipe.vae,'disable_tiling'): pipe.vae.disable_tiling()
    if hasattr(pipe.vae,'enable_slicing'): pipe.vae.enable_slicing()
    if not offload:
        pipe.to('cuda')
        pipe._pi_offload_mode='direct'
        return
    prof=prof or {}
    low=prof.get('te')=='w4a8' or int(prof.get('side',9999) or 9999)<=1024
    grouped=getattr(pipe,'enable_group_offload',None)
    if low and not quantized and callable(grouped):
        # Group offload swaps parameters behind the module's back, which can
        # leave packed QuantizedTensor weights on the CPU while inputs run on
        # the GPU (Issue #2 device mismatch). Quantized loads use model
        # offload, whose whole-model moves preserve packed subclasses.
        grouped(onload_device='cuda',offload_device='cpu',offload_type='leaf_level',use_stream=False)
        pipe._pi_offload_mode='group'
        return
    sequential=getattr(pipe,'enable_sequential_cpu_offload',None)
    if low and not quantized and callable(sequential):
        sequential()
        pipe._pi_offload_mode='sequential'
    else:
        pipe.enable_model_cpu_offload()
        pipe._pi_offload_mode='model'

@contextlib.contextmanager
def _sampling_heartbeat(pipe,stop,total,completed,status,clock=time.monotonic,interval=1.0):
    blocks=getattr(getattr(pipe,'transformer',None),'transformer_blocks',None)
    if not blocks:
        yield
        return
    handles=[]
    last=[clock()-interval]
    count=len(blocks)
    def make_hook(index):
        def heartbeat(_module,_inputs,_output):
            if stop.exists(): raise InterruptedError('Generation interrupted')
            now=clock()
            if now-last[0]>=interval:
                last[0]=now
                step=min(int(completed[0])+1,int(total))
                status(f'sampling step {step}/{total}, layer {index+1}/{count}')
        return heartbeat
    try:
        handles=[block.register_forward_hook(make_hook(index)) for index,block in enumerate(blocks)]
        yield
    finally:
        for handle in handles:
            with contextlib.suppress(Exception): handle.remove()


def emit(kind, **data):
    WIRE.write(json.dumps(dict(type=kind, **data), separators=(",", ":")) + "\n")
    WIRE.flush()


def package(root):
    name = "pi_qwen21_worker_" + str(os.getpid())
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


def image(path):
    if not path:
        return None
    from PIL import Image
    with Image.open(path) as value:
        return value.copy()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", required=True)
    parser.add_argument("--forge-root", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    deps = root / "_deps"
    log = open(args.log, "a", encoding="utf-8", buffering=1)
    sys.stdout = log
    sys.stderr = log
    sys.argv = [str(root / "lib/worker.py")]
    sys.path.insert(0, str(deps))
    if args.forge_root:
        forge = Path(args.forge_root).resolve()
        for entry in (forge, forge / "modules_forge" / "packages"):
            if str(entry) not in sys.path:
                sys.path.append(str(entry))

    if not deps.is_dir():
        emit("error", error="Isolated Qwen dependencies are unavailable in " + str(deps))
        return 2

    try:
        alias = package(root)
        components = __import__(alias + ".lib.components", fromlist=["pipeline"])
        lora = __import__(alias + ".lora.adapter", fromlist=["apply"])
        preview = __import__(alias + ".lib.preview", fromlist=["decode_preview"])
        spectrum = __import__(alias + ".lib.spectrum", fromlist=["accelerate"])
        import torch
    except Exception as exc:
        emit("error", error="Unable to import isolated Qwen dependencies: " + str(exc))
        traceback.print_exc(file=log)
        return 2

    pipe = None
    emit("ready")
    for line in sys.stdin:
        try:
            command = json.loads(line)
            op = command.get("op")
            if op == "close":
                emit("closed")
                break
            if op == "init":
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    prof=command.get('prof') or {}
                    dequantize=False
                    if _quantized(command['bundle'],prof):
                        try:
                            alias_assets=__import__(alias+'.lib.assets',fromlist=['quant_capable'])
                            capable=alias_assets.quant_capable(torch)
                        except Exception:
                            capable=False
                        if not capable:
                            # ROCm / older CUDA torch build / fp16-only card:
                            # materialize packed weights to bf16 in system RAM
                            # instead of refusing the files.
                            dequantize=True
                            print('[PI-Qwen21] Packed quantization kernels are unavailable on this GPU; unpacking quantized files to bf16 in system RAM (first load is slower).')
                    dtype=_compute_dtype(torch)
                    params=inspect.signature(components.pipeline).parameters
                    if 'dequantize' in params:
                        pipe = components.pipeline(command["bundle"],dtype=dtype,dequantize=dequantize)
                    elif 'dtype' in params:
                        if dequantize: raise ValueError('Packed quantized files need a compatible extension and Forge build')
                        pipe = components.pipeline(command["bundle"],dtype=dtype)
                    else:
                        if dequantize: raise ValueError('Packed quantized files need a compatible extension and Forge build')
                        pipe = components.pipeline(command["bundle"])
                    if (pipe.__class__.__name__ != "QwenImage21Pipeline"
                            or pipe.vae.__class__.__name__ != "AutoencoderKLQwenImage21"
                            or pipe.transformer.__class__.__name__ != "QwenImage21Transformer2DModel"
                            or pipe.text_encoder.__class__.__name__ != "Qwen3VLForConditionalGeneration"):
                        raise ValueError("Loaded component architecture is not official Qwen-Image-2.1")
                    _prepare_pipe(pipe,prof,command.get('offload',True),_quantized(command['bundle'],prof) and not dequantize)
                emit("initialized")
                continue
            if op != "generate" or pipe is None:
                raise ValueError("Worker is not initialized")

            stop = Path(command["stop"])
            refs = [image(p) for p in command.get("images", [])]
            mask = image(command.get("mask"))
            preview_every=max(0,int(command.get('preview_every',0) or 0))
            preview_size=max(64,min(256,int(command.get('preview_size',256) or 256)))
            preview_path=Path(command.get('preview') or Path(command['output']).with_name('preview.png'))
            preview_active=preview_every>0
            last_preview=0.0
            completed_steps=[0]

            def progress(_pipe, step, timestep, callback_kwargs):
                nonlocal preview_active,last_preview
                final=int(step)>=int(command['num_inference_steps'])-1
                completed_steps[0]=int(step)+1
                if final: emit('status',status='decoding')
                emit("progress", step=int(step), timestep=float(timestep) if timestep is not None else None)
                if stop.exists():
                    raise InterruptedError("Generation interrupted")
                now=time.monotonic()
                interval=1.0
                if (preview_active and int(step)%preview_every==0
                        and (last_preview==0.0 or now-last_preview>=interval)
                        and callback_kwargs.get('latents') is not None
                        and preview.has_preview_memory(torch)):
                    try:
                        with preview.preview_context(pipe) as allowed:
                            if not allowed:
                                preview_active=False
                                return callback_kwargs
                            picture=preview.decode_preview(pipe,callback_kwargs['latents'],command['height'],command['width'],preview_size)
                        picture.save(preview_path,format='PNG');last_preview=now
                        emit('preview',step=int(step),timestep=float(timestep) if timestep is not None else None,path=str(preview_path))
                    except Exception:
                        preview_active=False
                        print('Qwen live preview disabled after decode failure:',file=log)
                        traceback.print_exc(file=log)
                return callback_kwargs

            kwargs = dict(
                prompt=command["prompt"],
                negative_prompt=command.get("negative_prompt"),
                true_cfg_scale=command.get("true_cfg_scale", 1),
                width=command["width"], height=command["height"],
                num_inference_steps=command["num_inference_steps"],
                generator=torch.Generator("cpu").manual_seed(int(command["seed"])),
                use_kv_cache=command.get("use_kv_cache", True),
                callback_on_step_end=progress,
            )
            params=inspect.signature(pipe.__call__).parameters
            if preview_active:
                if 'callback_on_step_end_tensor_inputs' in params:
                    kwargs['callback_on_step_end_tensor_inputs']=['latents']
                else:
                    preview_active=False
            if refs:
                kwargs["image"] = refs
            if mask is not None:
                if "mask_image" not in inspect.signature(pipe.__call__).parameters:
                    raise ValueError("Installed QwenImage21Pipeline does not support mask_image")
                kwargs["mask_image"] = mask
            if command.get("output_resolution") is not None:
                kwargs["output_resolution"] = command["output_resolution"]
            supported = params
            kwargs = {k: v for k, v in kwargs.items() if k in supported}
            adapters = command.get("adapters", [])
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    lora.apply(pipe, [(a["path"], float(a["weight"])) for a in adapters])
                    speedup=bool(command.get('spectrum',False)) and float(command.get('true_cfg_scale',1))<=1
                    if command.get('spectrum',False) and not speedup:
                        emit('status',status='Spectrum disabled: requires True CFG 1')
                    emit('status',status='encoding')
                    with spectrum.accelerate(pipe,enabled=speedup,steps=command['num_inference_steps'],stop=stop) as cache_stats, _sampling_heartbeat(pipe,stop,command['num_inference_steps'],completed_steps,
                                             lambda text: emit('status',status=text)):
                        result = pipe(**kwargs).images[0]
                    if command.get('spectrum',False):
                        emit('status',status=f'Spectrum {cache_stats.reason}: {cache_stats.actual} real, {cache_stats.forecast} forecast steps')
                    result.save(command["output"], format="PNG")
                emit("result", path=command["output"])
            finally:
                with contextlib.suppress(Exception):
                    lora.clear(pipe)
        except InterruptedError as exc:
            emit("interrupted", error=str(exc))
        except Exception as exc:
            traceback.print_exc(file=log)
            emit("error", error=str(exc) or exc.__class__.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
