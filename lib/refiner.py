"""Second-pass refiner for Qwen-Image-2.1.

Follows the project-invisible philosophy: no Forge core edits, no extra model
downloads, no extra VRAM.  The refiner reuses the pipeline that is already
loaded inside the worker, so it works with every quantization (bf16,
int8_convrot, w4a8), every VRAM profile/offload mode, every preset, CFG value,
sampler, speed-LoRA and community-LoRA combination.  The rendered image is
encoded with the same VAE, partially re-noised in latent space and denoised for
a few steps:

  Turbo   -> 35% noise, 4 steps   (adds only a couple of seconds)
  Quality -> 50% noise, 10 steps  (stronger detail recovery)
"""
from pathlib import Path

MODES = {
    'off': None,
    'turbo': (0.35, 4, True),   # (strength, steps, force_cfg1)
    'quality': (0.5, 10, False),
}


def run(pipe, torch, image, command, base_kwargs, emit, log):
    """Refine `image` (PIL) in latent space; returns the refined PIL image."""
    spec = MODES.get(str(command.get('refiner') or 'off').lower())
    if spec is None or image is None:
        return image
    import inspect
    from numpy import linspace
    from diffusers.utils.torch_utils import randn_tensor

    strength, steps, force_cfg1 = spec
    params = inspect.signature(pipe.__call__).parameters
    if 'latents' not in params or 'sigmas' not in params:
        print('[PI-Qwen21] Refiner unavailable: pipeline lacks latents/sigmas control.', file=log)
        return image

    height, width = int(command['height']), int(command['width'])
    device = pipe._execution_device
    vae_dtype = pipe.vae.dtype
    channels = int(pipe.transformer.config.in_channels)
    seed = int(command['seed']) + 1  # deterministic, independent of the base pass
    generator = torch.Generator('cpu').manual_seed(seed)

    stop = command.get('stop')
    total = [0]

    def refine_progress(_pipe, step, timestep, callback_kwargs):
        total[0] = int(step) + 1
        emit('progress', step=int(step), timestep=float(timestep) if timestep is not None else None)
        if stop is not None and Path(stop).exists():
            raise InterruptedError('Generation interrupted')
        return callback_kwargs

    emit('status', status='refining')

    # If the speed-LoRA pass swapped in a shift_terminal scheduler, restore the
    # base one so the refiner's sigma schedule is exact.
    base_config = getattr(pipe, '_pi_qwen21_base_scheduler_config', None)
    if base_config is not None:
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(dict(base_config))
        pipe._pi_qwen21_base_scheduler_config = None

    # 1. Encode the rendered image with the already-loaded VAE (no new weights).
    tensor = pipe.image_processor.preprocess(image, height=height, width=width)
    tensor = tensor.to(device=device, dtype=vae_dtype)
    encoded = pipe._encode_vae_image(tensor, generator=generator)
    latent_h, latent_w = encoded.shape[3:]
    latents = pipe._pack_latents(encoded, 1, channels, latent_h, latent_w)

    # 2. Re-noise to `strength` (flow matching: x_t = (1-s) x0 + s noise).
    noise = randn_tensor((1, 1, channels, latent_h, latent_w), generator=generator,
                         device=device, dtype=latents.dtype)
    latents = (1.0 - strength) * latents + strength * noise

    # 3. Denoise from `strength` back to 0 in `steps` steps.
    sigmas = [float(v) for v in linspace(strength, strength / steps, steps)]
    cfg = 1.0 if force_cfg1 else float(command.get('true_cfg_scale', 1))
    kwargs = dict(
        prompt=base_kwargs.get('prompt'),
        negative_prompt=base_kwargs.get('negative_prompt') if cfg > 1 else None,
        true_cfg_scale=cfg,
        width=width, height=height,
        num_inference_steps=steps,
        sigmas=sigmas,
        latents=latents,
        generator=generator,
        use_kv_cache=base_kwargs.get('use_kv_cache', True),
        callback_on_step_end=refine_progress,
    )
    if 'output_resolution' in params and base_kwargs.get('output_resolution') is not None:
        kwargs['output_resolution'] = base_kwargs['output_resolution']
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    try:
        result = pipe(**kwargs).images[0]
    finally:
        del latents, noise, encoded, tensor
    emit('status', status=f'refined in {total[0]} steps')
    return result