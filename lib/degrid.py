"""PIL adapter for the pinned DeGrid core; CPU-only, with exact alpha preservation."""
import math
import numpy as np
from PIL import Image

def remove_moire(image, strength=1.0):
    strength=float(strength)
    if not math.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('DeGrid strength must be between 0 and 1')
    if image.mode not in ('RGB','RGBA'):
        raise ValueError('DeGrid requires RGB or RGBA')
    if strength==0 or min(image.size)<=8:
        return image.copy()
    import torch
    from .vendor.degrid.degrid_core import extract_grid, lattice_amp, auto_limit, NEGLIGIBLE_AMP
    pixels=np.asarray(image)
    # Use upstream numerical functions without allocating its unused grid visualization.
    with torch.inference_mode():
        x=torch.from_numpy(pixels[...,:3].copy()).permute(2,0,1).unsqueeze(0).float().div_(255)
        correction=extract_grid(x)
        if float(lattice_amp(correction)[0][0])<NEGLIGIBLE_AMP:
            return image.copy()
        limit=auto_limit(correction).view(-1,1,1,1)
        correction.clamp_(-limit,limit).mul_(strength)
        x.sub_(correction).clamp_(0,1).mul_(255).round_()
        rgb=x.squeeze(0).permute(1,2,0).to(torch.uint8).numpy()
    if image.mode=='RGB': return Image.fromarray(rgb)
    result=np.array(pixels,copy=True);result[...,:3]=rgb
    return Image.fromarray(result)
