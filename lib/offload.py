"""Packed-weight-safe layer offload for very small GPUs."""
from types import SimpleNamespace


def _chunked_rms(module, x, original, torch, max_float_bytes=8 * 2**20):
    """Exact channel RMS normalization with bounded temporary FP32 memory."""
    needs_fp32=x.dtype in (torch.float16,torch.bfloat16) or any(
        tag in str(x.dtype) for tag in ('float4_','float8_'))
    if not (needs_fp32 and module.channel_first and x.ndim >= 4):
        return original(x)
    # Normalization reduces channels only. Splitting height cannot change any
    # pixel's reduction, and avoids full-frame FP32 input/output temporaries.
    height=x.shape[-2]
    per_row=max(1,x.numel() // max(1,height))
    rows=max(1,min(height,max_float_bytes // max(1,per_row * 4)))
    if rows >= height:
        return original(x)
    out=torch.empty_like(x)
    for start in range(0,height,rows):
        end=min(height,start+rows)
        part=x[...,start:end,:]
        normalized=torch.nn.functional.normalize(part.float(),dim=1).to(x.dtype)
        out[...,start:end,:]=normalized * module.scale * module.gamma + module.bias
    return out


def _move_own(module, device):
    """Move only tensors owned by one module, preserving tensor subclasses."""
    module._apply(lambda value: value.to(device=device, non_blocking=False), recurse=False)


class LayerOffload:
    def __init__(self, torch, device='cuda'):
        self.torch = torch
        self.device = torch.device(device)
        self.handles = []
        self.modules = []
        self.patches = []
        self.method_patches = []
        self.cpu_components = set()
        self.cancel = lambda: False

    def send(self, value, device):
        if self.torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(self.send(item,device) for item in value)
        if isinstance(value, list):
            return [self.send(item,device) for item in value]
        if isinstance(value, dict):
            return {key:self.send(item,device) for key,item in value.items()}
        return value

    def install(self, component):
        # The marker lets Diffusers create most intermediates on CUDA. The root
        # pre-hook also covers pipelines that inspect component.device (CPU)
        # while preparing token IDs, masks, or VAE inputs.
        component._hf_hook = SimpleNamespace(execution_device=self.device)

        def root_before(_module, args, kwargs):
            if self.cancel():
                raise InterruptedError('Generation interrupted')
            target='cpu' if component in self.cpu_components else self.device
            return self.send(args,target),self.send(kwargs,target)

        self.handles.append(component.register_forward_pre_hook(root_before, with_kwargs=True))
        for module in component.modules():
            if module.__class__.__name__ == 'QwenImage21RMS_norm':
                original=module.forward
                module.forward=lambda x,_m=module,_o=original: _chunked_rms(_m,x,_o,self.torch)
                self.patches.append((module,original))
            if not (list(module.parameters(recurse=False)) or list(module.buffers(recurse=False))):
                continue
            _move_own(module, 'cpu')
            depth = [0]

            def before(_module, _args, _depth=depth, _component=component):
                if self.cancel():
                    raise InterruptedError('Generation interrupted')
                if _depth[0] == 0:
                    target='cpu' if _component in self.cpu_components else self.device
                    _move_own(_module,target)
                _depth[0] += 1

            def after(_module, _args, output, _depth=depth):
                # always_call hooks also run when cancellation raises in the
                # pre-hook, before depth increments. Never underflow state.
                if _depth[0] > 0:
                    _depth[0] -= 1
                if _depth[0] == 0:
                    _move_own(_module,'cpu')
                return output

            self.handles.append(module.register_forward_pre_hook(before))
            try:
                handle = module.register_forward_hook(after, always_call=True)
            except TypeError:  # old PyTorch fallback
                handle = module.register_forward_hook(after)
            self.handles.append(handle)
            self.modules.append(module)
        return component

    def offload_component(self, component):
        for module in component.modules():
            if list(module.parameters(recurse=False)) or list(module.buffers(recurse=False)):
                _move_own(module,'cpu')

    def wrap_vae_decode(self, vae, force_cpu=False):
        original=vae.decode

        def cpu_decode(args,kwargs):
            self.cpu_components.add(vae)
            try:
                self.offload_component(vae)
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                return original(*self.send(args,'cpu'),**self.send(kwargs,'cpu'))
            finally:
                self.cpu_components.discard(vae)

        def decode(*args,**kwargs):
            if force_cpu:
                return cpu_decode(args,kwargs)
            try:
                return original(*args,**kwargs)
            except self.torch.OutOfMemoryError:
                return cpu_decode(args,kwargs)

        vae.decode=decode
        self.method_patches.append((vae,'decode',original))

    def offload(self):
        for module in self.modules:
            _move_own(module, 'cpu')
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def remove(self):
        self.offload()
        for handle in self.handles:
            handle.remove()
        for module,original in self.patches:
            module.forward=original
        for module,name,original in self.method_patches:
            setattr(module,name,original)
        self.handles.clear()
        self.modules.clear()
        self.patches.clear()
        self.method_patches.clear()
        self.cpu_components.clear()


def enable(pipe, torch, prof=None):
    manager = LayerOffload(torch)
    for name in ('text_encoder','transformer','vae'):
        component=getattr(pipe,name,None)
        if component is not None and hasattr(component,'modules'):
            manager.install(component)
    budget=float((prof or {}).get('vram_gb') or 0)
    if 0 < budget <= 8 and getattr(pipe,'vae',None) is not None:
        # Six GB cannot fit Qwen's final full-frame VAE activations. Decode the
        # same latents on CPU; 8 GB first tries GPU then recovers on OOM.
        manager.wrap_vae_decode(pipe.vae,force_cpu=False)
    pipe._pi_layer_offload = manager
    pipe._pi_offload_mode = 'packed-layer'
    return manager
