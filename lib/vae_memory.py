# SPDX-License-Identifier: AGPL-3.0-or-later
# Inspired by Haoming02/sd-webui-forge-classic PR #1512; see vendor/forge/.
"""Halo strips for local VAE upsampling, inspired by Forge PR #1512.

Keeps spatial attention, normalization and temporal caches on their native path.
No weights or dependency source files are modified.
"""

LIMIT = 2**24


def strip_apply(fn, x, limit=LIMIT):
    """Nearest x2 followed by local 3x3 convolution; one input row halo."""
    import torch
    projected=x.numel()*4
    if torch.is_grad_enabled() or x.ndim!=4 or projected<=limit:
        return fn(x)
    count=(projected+limit-1)//limit
    height=x.shape[-2]
    step=max(1,(height+count-1)//count)
    output=None
    for start in range(0,height,step):
        end=min(height,start+step)
        lo=max(0,start-1)
        hi=min(height,end+1)
        block=fn(x[...,lo:hi,:])
        block=block[...,2*(start-lo):2*(end-lo),:]
        if output is None:
            output=block.new_empty(*block.shape[:-2],height*2,block.shape[-1])
        output[...,2*start:2*end,:].copy_(block)
        del block
    return output


def install(vae,limit=LIMIT):
    """Fail closed on changed/unknown resamplers; preserve module names."""
    import torch
    if type(vae).__name__!='AutoencoderKLQwenImage21' or not callable(getattr(vae,'modules',None)): return 0
    installed=0
    for module in vae.modules():
        if type(module).__name__!='QwenImage21Resample' or module.mode not in ('upsample2d','upsample3d'):
            continue
        seq=module.resample
        if not isinstance(seq,torch.nn.Sequential) or len(seq)!=2 or getattr(seq,'_pi_strips',False):
            continue
        up,conv=seq
        if (getattr(up,'mode',None)!='nearest-exact' or getattr(up,'scale_factor',None) not in (2,2.0,(2.0,2.0))
                or not isinstance(conv,torch.nn.Conv2d) or conv.kernel_size!=(3,3)
                or conv.stride!=(1,1) or conv.padding!=(1,1) or conv.dilation!=(1,1)
                or conv.padding_mode!='zeros'):
            continue
        original=seq.forward
        def forward(x,original=original):
            return strip_apply(original,x,limit)
        seq.forward=forward
        seq._pi_strips=True
        installed+=1
    return installed
