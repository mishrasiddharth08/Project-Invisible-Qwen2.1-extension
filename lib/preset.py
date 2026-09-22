"""Adapted from Ideogram4 legacy_preset.py; additive native preset registration."""
from copy import copy, deepcopy
from .forge import LABEL

NAME='qwen-image-2.1'

def install():
    from modules import shared
    from modules_forge import presets
    template={}
    presets.register(template)
    ours={}
    for key,info in template.items():
        if key.startswith('sd_'): target=NAME+key[2:]
        elif key in ('forge_checkpoint_sd','forge_additional_modules_sd','forge_unet_storage_dtype_sd'):
            target=key[:-2]+NAME
        else: continue
        item=copy(info); item.default=deepcopy(info.default)
        if getattr(info,'section',(None,))[0]=='ui_sd': item.section=('ui_qwen21','QWEN-IMAGE-2.1')
        if target.endswith('_sampler'): item.default='Euler'
        elif target.endswith('_scheduler'): item.default='Simple'
        elif target.endswith('_step'): item.default=40
        elif target.endswith('_cfg'): item.default=1.0
        elif target.endswith(('_width','_height')): item.default=2048
        elif target=='forge_checkpoint_'+NAME: item.default=LABEL
        elif target=='forge_additional_modules_'+NAME: item.default=[]
        ours[target]=item
    required={'forge_checkpoint_'+NAME,NAME+'_t2i_step',NAME+'_t2i_cfg'}
    if not required.issubset(ours): raise RuntimeError('Neo preset schema changed; Qwen preset registration skipped')
    for key,info in ours.items():
        if key not in shared.opts.data_labels: shared.opts.add_option(key,info)
    original=presets.PresetArch.choices
    if not getattr(original,'_pi_qwen21',False):
        def choices(*a,**kw):
            values=list(original(*a,**kw))
            if NAME not in values: values.append(NAME)
            return values
        choices._pi_qwen21=True
        presets.PresetArch.choices=staticmethod(choices)

