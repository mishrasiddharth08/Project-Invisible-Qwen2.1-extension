"""Strict Comfy components in official 2.1 classes; Neo supplies quantized Linear/Embedding ops."""
import json
import re
from pathlib import Path

def _embedding_apply(self, fn, recurse=True):
    """Move a whole Embedding module without stripping packed weight subclasses.

    The default ``nn.Module._apply`` reassigns ``param.data``, which unwraps
    QuantizedTensor parameters and leaves the packed data on the original
    device while the module executes elsewhere (Issue #2 device mismatch).
    Applying ``fn`` to the full parameter keeps the subclass and its device.
    Mirrors backend.operations_mixed_precision._quantized_apply.
    """
    import torch
    if recurse:
        for child in self.children():
            child._apply(fn)
    for key, param in list(self._parameters.items()):
        if param is None:
            continue
        moved = fn(param)
        try:
            self.register_parameter(key, torch.nn.Parameter(moved, requires_grad=False))
        except RuntimeError:
            self.register_parameter(key, torch.nn.Parameter(moved.clone(), requires_grad=False))
    for key, buf in self._buffers.items():
        if buf is not None:
            self._buffers[key] = fn(buf)
    return self

def _harden_embedding_apply(model):
    """Attach the subclass-safe move routine to every Embedding of this model."""
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            module._apply = _embedding_apply.__get__(module, type(module))
    return model

def _descriptor(value):
    try:
        return json.loads(bytes(value.tolist()).decode().rstrip('\0'))
    except (AttributeError,UnicodeDecodeError,TypeError,ValueError,KeyError) as exc:
        raise ValueError('Malformed Qwen quantization descriptor') from exc

def _map_vae_key(key):
    if key.startswith('conv1.'): return 'quant_conv.'+key[len('conv1.'):]
    if key.startswith('conv2.'): return 'post_quant_conv.'+key[len('conv2.'):]
    for side,groups,blocks in (('encoder','downsamples','down_blocks'),('decoder','upsamples','up_blocks')):
        if key.startswith(side+'.conv1.'): return side+'.conv_in.'+key[len(side+'.conv1.'):]
        if key.startswith(side+'.head.0.'): return side+'.norm_out.'+key[len(side+'.head.0.'):]
        if key.startswith(side+'.head.2.'): return side+'.conv_out.'+key[len(side+'.head.2.'):]
        match=re.match(rf'{side}\.middle\.([012])\.(.+)',key)
        if match:
            index,tail=match.groups()
            target='attentions.0' if index=='1' else 'resnets.'+('0' if index=='0' else '1')
            key=f'{side}.mid_block.{target}.{tail}'
            break
        match=re.match(rf'{side}\.{groups}\.(\d+)\.{groups}\.(\d+)\.(.+)',key)
        if match:
            group,index,tail=match.groups()
            count=2 if side=='encoder' else 3
            target=f'resnets.{index}' if int(index)<count else ('downsampler' if side=='encoder' else 'upsampler')
            key=f'{side}.{blocks}.{group}.{target}.{tail}'
            break
    key=key.replace('.residual.0.','.norm1.')
    key=key.replace('.residual.2.','.conv1.')
    key=key.replace('.residual.3.','.norm2.')
    key=key.replace('.residual.6.','.conv2.')
    key=key.replace('.shortcut.','.conv_shortcut.')
    return key

def _normalize_component(sd,kind):
    """Map verified Comfy single-file names to official Diffusers/Transformers names."""
    prefix={'transformer':'model.diffusion_model.','text_encoder':'qwen3vl_8b.transformer.','vae':'vae.'}[kind]
    normalized={}
    for key,value in sd.items():
        out=key[len(prefix):] if key.startswith(prefix) else key
        if kind=='vae':
            out=_map_vae_key(out)
            if out.endswith('.weight') and getattr(value,'ndim',0)==5:
                if value.shape[2] != 1:
                    raise ValueError(f'Unsupported temporal VAE kernel: {out} {tuple(value.shape)}')
                value=value.squeeze(2)
        elif kind=='text_encoder':
            if out.startswith('lm_head.'): continue
            if out.startswith(('model.layers.','model.embed_tokens.','model.norm.','model.rotary_emb.')):
                out='model.language_model.'+out[len('model.'):]
            elif out.startswith(('language_model.','visual.')):
                out='model.'+out
        if out in normalized and normalized[out] is not value:
            raise ValueError(f'Conflicting component keys after normalization: {out}')
        normalized[out]=value
    if kind=='transformer':
        fused={k.rsplit('.',1)[0] for k in normalized if '.img_mlp.gate_up.' in k}
        for base in sorted(fused):
            descriptor=normalized.get(base+'.comfy_quant')
            weight=normalized.get(base+'.weight')
            if descriptor is not None:
                if _descriptor(descriptor).get('format')!='int8_tensorwise':
                    raise ValueError(f'Unsupported packed Qwen MLP layout: {base}')
            elif weight is None or not weight.is_floating_point() or base+'.weight_scale' in normalized:
                raise ValueError(f'Unsupported packed Qwen MLP layout: {base}')
            for suffix in ('weight','weight_scale'):
                key=base+'.'+suffix
                value=normalized.pop(key,None)
                if value is None: continue
                if getattr(value,'ndim',0)<1 or value.shape[0]%2:
                    raise ValueError(f'Invalid fused Qwen MLP rows: {key}')
                gate,up=value.chunk(2,dim=0)
                normalized[base.rsplit('.',1)[0]+'.gate_layer.'+suffix]=gate
                normalized[base.rsplit('.',1)[0]+'.proj.'+suffix]=up
            normalized.pop(base+'.comfy_quant',None)
            parent=base.rsplit('.',1)[0]
            if descriptor is not None:
                normalized[parent+'.gate_layer.comfy_quant']=descriptor
                normalized[parent+'.proj.comfy_quant']=descriptor
    return normalized

def load_component(cls, config_dir, path, kind, dtype=None):
    import torch
    dtype=dtype or torch.bfloat16
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from safetensors.torch import load_file
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        from transformers.modeling_utils import no_init_weights
    if not path:
        raise ValueError(f'Missing {kind} component file for Qwen-Image-2.1')
    sd=load_file(str(path),device='cpu')
    with safe_open(str(path),framework='pt') as f: metadata=f.metadata() or {}
    quant=any(k.endswith('.comfy_quant') for k in sd) or '_quantization_metadata' in metadata
    if quant:
        # Mixed precision conversion expects the original Comfy key layout.
        from backend.state_dict import convert_quantization,detect_quantization
        from backend.quant_ops import QUANT_ALGOS
        import backend.operations  # registers the base Forge operation classes first
        from backend.operations_mixed_precision import mixed_precision_ops
        if not any(k.endswith('.comfy_quant') for k in sd): sd,_=convert_quantization(sd,metadata)
    sd=_normalize_component(sd,kind)
    if quant:
        formats=set()
        for key,value in sd.items():
            if not key.endswith('.comfy_quant'): continue
            try:
                descriptor=_descriptor(value)
                fmt=descriptor.get('format')
            except (AttributeError,UnicodeDecodeError,TypeError,ValueError,KeyError) as exc:
                raise ValueError(f'Malformed Qwen quantization descriptor: {key}') from exc
            if not fmt: raise ValueError(f'Missing quantization format: {key}')
            formats.add(fmt)
        if not formats or formats-{'int8_tensorwise','asym_w4a8_int8'} or formats-set(QUANT_ALGOS):
            raise ValueError('Unsupported 2.1 quantization descriptors: '+str(formats))
        qc=detect_quantization(sd,is_unet=kind=='transformer')
        if not qc: raise ValueError('Neo did not recognize this quantization layout')
        qc=dict(qc);qc.pop('TE',None)
        ops=mixed_precision_ops(quant_config=qc,compute_dtype=dtype,full_precision_mm=kind=='text_encoder',disabled=[])
    # Keep rotary and other nonpersistent buffers on CPU; only parameters are meta.
    with no_init_weights(),init_empty_weights(include_buffers=False):
        if kind=='text_encoder':
            from transformers import Qwen3VLConfig
            model=cls(Qwen3VLConfig.from_pretrained(str(config_dir),local_files_only=True))
            # Only hidden states are used. Omitting language logits saves memory
            # and permits Comfy encoders that intentionally omit the LM head.
            model.lm_head=torch.nn.Identity()
        else: model=cls.from_config(json.loads((Path(config_dir)/'config.json').read_text(encoding='utf8')))
    missing=sorted(name for name,_ in model.named_parameters() if name not in sd)
    if missing:
        sample=', '.join(missing[:8])
        raise ValueError(f'Missing official {kind} weights before quantized replacement: {sample}')
    if quant:
        for name,module in list(model.named_modules()):
            if isinstance(module,torch.nn.Linear):
                parent,_,child=name.rpartition('.')
                replacement=ops.Linear(module.in_features,module.out_features,bias=module.bias is not None,device='cpu',dtype=dtype)
                setattr(model.get_submodule(parent) if parent else model,child,replacement)
            elif isinstance(module,torch.nn.Embedding):
                parent,_,child=name.rpartition('.')
                replacement=ops.Embedding(module.num_embeddings,module.embedding_dim,padding_idx=module.padding_idx,max_norm=module.max_norm,norm_type=module.norm_type,scale_grad_by_freq=module.scale_grad_by_freq,sparse=module.sparse,device='cpu',dtype=dtype)
                setattr(model.get_submodule(parent) if parent else model,child,replacement)
    model.load_state_dict(sd,strict=True,assign=True)
    del sd
    if any(t.is_meta for t in list(model.parameters())+list(model.buffers())):
        raise ValueError('Incomplete 2.1 component: unmaterialized tensors')
    model.to(dtype=dtype)
    # Packed Embedding parameters must survive device moves during offloading.
    _harden_embedding_apply(model)
    model.eval().requires_grad_(False)
    return model

def pipeline(bundle,dtype=None):
    import torch
    dtype=dtype or torch.bfloat16
    from diffusers import QwenImage21Pipeline,QwenImage21Transformer2DModel,AutoencoderKLQwenImage21,FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLForConditionalGeneration,Qwen3VLProcessor
    folder=Path(bundle['folder']);files=bundle.get('files',{})
    if not files:
        return QwenImage21Pipeline.from_pretrained(str(folder),torch_dtype=dtype,local_files_only=True)
    classes={'transformer':QwenImage21Transformer2DModel,'text_encoder':Qwen3VLForConditionalGeneration,'vae':AutoencoderKLQwenImage21}
    parts={k:load_component(cls,folder/k,files[k],k,dtype=dtype) for k,cls in classes.items()}
    parts['processor']=Qwen3VLProcessor.from_pretrained(str(folder/'processor'),local_files_only=True)
    parts['scheduler']=FlowMatchEulerDiscreteScheduler.from_pretrained(str(folder/'scheduler'),local_files_only=True)
    return QwenImage21Pipeline(**parts)
