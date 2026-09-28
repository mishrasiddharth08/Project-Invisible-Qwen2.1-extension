"""GPL-3.0: request-scoped Qwen 2.1 bridge to the LanPaint core.

Core: scraed/LanPaint, 2d7912f9a5efe5ece8de334c7ca18317b8288c39.
No ComfyUI imports, global sampler patches or alternative model pipeline.
"""
from contextlib import contextmanager
import math


def prepare_mask(image, mask, invert=False):
    from PIL import Image, ImageOps
    if image is None or mask is None:
        raise ValueError('LanPaint needs an img2img source and an inpaint mask.')
    mask = mask.convert('L').resize(image.size, Image.Resampling.NEAREST)
    if invert:
        mask = ImageOps.invert(mask)
    if mask.getextrema()[1] == 0:
        raise ValueError('LanPaint mask is empty. Paint the area to change.')
    return mask


def preserve(image, result, mask):
    from PIL import Image
    source = image.convert('RGBA').resize(result.size, Image.Resampling.LANCZOS)
    mask = mask.convert('L').resize(result.size, Image.Resampling.NEAREST)
    return Image.composite(result.convert('RGBA'), source, mask)


@contextmanager
def sampling(pipe, image, mask, width, height, thinking=2, cfg=1., seed=0,
             stop=None, status=lambda message: None):
    """Wrap only this pipeline instance, restoring every hook after errors/cancel."""
    import torch
    import torch.nn.functional as F
    from .vendor.lanpaint.lanpaint import LanPaint
    thinking = int(thinking)
    mask = prepare_mask(image, mask)
    if not 1 <= thinking <= 5 or not math.isfinite(float(cfg)):
        raise ValueError('LanPaint thinking steps must be 1–5 and CFG finite.')
    if pipe.__class__.__name__ != 'QwenImage21Pipeline':
        raise ValueError('LanPaint bridge requires QwenImage21Pipeline.')
    scheduler = pipe.scheduler
    if scheduler.__class__.__name__ != 'FlowMatchEulerDiscreteScheduler':
        raise ValueError('LanPaint currently requires the Euler flow scheduler.')
    saved_step = scheduler.__dict__.get('step')
    saved_prepare = pipe.__dict__.get('prepare_latents')
    original_step, original_prepare = scheduler.step, pipe.prepare_latents
    calls, state = [], {}
    active = [False]

    def cancelled():
        if stop is not None and stop.exists():
            raise InterruptedError('LanPaint interrupted')

    def capture(module, args, kwargs):
        cancelled()
        if not active[0]:
            if args or kwargs.get('kv_cache_mode') is not None:
                raise ValueError('LanPaint requires uncached keyword-based Qwen 2.1 forwards.')
            calls.append(dict(kwargs))

    def prepare(*args, **kwargs):
        # Binding stays compatible with the pinned pipeline's own signature.
        import inspect
        bound = inspect.signature(original_prepare).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        result = original_prepare(*args, **kwargs)
        noise = result[0]
        canvas = pipe.image_processor.preprocess(image.convert('RGBA'), height=height, width=width)
        canvas = canvas.unsqueeze(2).to(device=values['device'], dtype=pipe.vae.dtype)
        known = pipe._encode_vae_image(canvas, values['generator'])
        h, w = known.shape[-2:]
        known = pipe._pack_latents(known, 1, pipe.latent_channels, h, w).to(noise)
        if known.shape != noise.shape:
            raise ValueError('LanPaint canvas/latent dimensions disagree; use aligned image dimensions.')
        import numpy as np
        pixels = torch.from_numpy(np.array(mask.convert('L'), copy=True)).float()[None, None] / 255.
        edit = F.interpolate(pixels, size=(h,w), mode='area').flatten(2).transpose(1,2)
        state.update(known=known, noise=noise.detach().clone(), keep=(1-edit).to(noise))
        return result

    class Model:
        def __init__(self):
            self.inner_model = self
            self.model_sampling = self

        def noise_scaling(self, sigma, noise, latent_image):
            return sigma * noise + (1-sigma) * latent_image

        def __call__(self, x, sigma, **unused):
            cancelled()
            velocities = []
            for index, template in enumerate(calls):
                k = dict(template)
                original = k['hidden_states']
                k['hidden_states'] = torch.cat((original[:, :-x.shape[1]], x.to(original)), dim=1)
                # Same noise level as the outer call; each branch keeps its own conditioning.
                with pipe.transformer.cache_context('cond' if index == 0 else 'uncond'):
                    velocities.append(pipe.transformer(**k)[0][:, -x.shape[1]:].to(x))
            v = velocities[0]
            if len(velocities) == 2:
                v = velocities[1] + cfg * (v - velocities[1])
            x0 = x - sigma.reshape(-1,1,1) * v
            return x0, x0

    algorithm = LanPaint(Model(), thinking, 15., 8., 1., .2, IS_FLOW=True)

    def step(model_output, timestep, sample, *args, **kwargs):
        cancelled()
        if len(calls) not in (1, 2) or not state:
            raise ValueError('LanPaint conditioning contract changed; restart with LanPaint off.')
        if scheduler.step_index is None:
            scheduler._init_step_index(timestep)
        sigma = scheduler.sigmas[scheduler.step_index].to(device=sample.device, dtype=torch.float32)
        t = sigma.reshape(1)
        safe = t.clamp(.001,.999)
        abt = (1-safe).square()/((1-safe).square()+safe.square()+1e-8)
        ve = t.clamp(.000001,.99)/(1-t.clamp(.000001,.99))
        status('LanPaint: refining masked region')
        active[0] = True
        try:
            # Upstream uses torch.randn_like; fork_rng makes each image repeatable
            # without changing the worker's RNG state for future jobs.
            x = sample.float().clone()
            x0 = algorithm(x, state['known'].float(), state['noise'].float(), t,
                           state['keep'].float(), (ve,abt,t), {}, seed)
            if not torch.isfinite(x0).all() or not torch.isfinite(x).all():
                raise ValueError('LanPaint produced non-finite values; disable it for this configuration.')
            velocity = (x-x0)/sigma.clamp_min(1e-6)
            result = original_step(velocity.to(model_output), timestep, x.to(sample), *args, **kwargs)
            return result
        finally:
            active[0] = False
            calls.clear()

    handle = pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        pipe.prepare_latents, scheduler.step = prepare, step
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield
    finally:
        handle.remove()
        if saved_step is None: scheduler.__dict__.pop('step',None)
        else: scheduler.step = saved_step
        if saved_prepare is None: pipe.__dict__.pop('prepare_latents',None)
        else: pipe.prepare_latents = saved_prepare
        calls.clear()
        state.clear()
