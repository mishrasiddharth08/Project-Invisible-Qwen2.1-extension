"""Small Qwen-Image-2.1 latent previews; never uses SD preview decoders."""

from contextlib import contextmanager

def has_preview_memory(torch,reserve_bytes=2*1024**3):
    try:
        free,_total=torch.cuda.mem_get_info()
        return int(free)>=int(reserve_bytes)
    except (AttributeError,RuntimeError,TypeError):
        return False

@contextmanager
def preview_context(pipe):
    """Decode without letting model-offload's VAE hook evict the resident transformer."""
    if getattr(pipe,'_pi_offload_mode',None)!='model':
        yield True
        return
    vae=pipe.vae
    hook=getattr(vae,'_hf_hook',None)
    if hook is None or not hasattr(hook,'prev_module_hook') or not callable(getattr(hook,'init_hook',None)):
        yield False
        return
    previous=hook.prev_module_hook
    hook.prev_module_hook=None
    try:
        yield True
    finally:
        hook.prev_module_hook=previous
        hook.init_hook(vae)

def decode_preview(pipe,latents,height,width,max_side=256):
    import torch
    import torch.nn.functional as F
    max_side=max(64,min(256,int(max_side or 256)))
    with torch.inference_mode():
        value=pipe._unpack_latents(latents,height,width,pipe.vae_scale_factor)
        _,_,_,lh,lw=value.shape
        limit=max(2,max_side//int(pipe.vae_scale_factor))
        scale=min(1.0,limit/max(lh,lw))
        target=(max(2,round(lh*scale)),max(2,round(lw*scale)))
        if target!=(lh,lw):
            value=F.interpolate(value.squeeze(2),size=target,mode='area').unsqueeze(2)
        value=value.to(dtype=pipe.vae.dtype)
        mean=torch.tensor(pipe.vae.config.latents_mean,device=value.device,dtype=value.dtype).view(1,-1,1,1,1)
        std=torch.tensor(pipe.vae.config.latents_std,device=value.device,dtype=value.dtype).view(1,-1,1,1,1)
        value=value*std+mean
        decoded=pipe.vae.decode(value,return_dict=False)[0][:,:,0]
        images=pipe.image_processor.postprocess(decoded,output_type='pil')
        return images[0]
