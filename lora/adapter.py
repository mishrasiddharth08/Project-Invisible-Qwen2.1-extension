"""Native Extra Networks tags; never modify Forge/other engines' loaded adapters."""
import json
import re
from pathlib import Path
from ..lib.assets import header

FAMILY = re.compile(r'qwen[-_ ]image[-_ ]2(?:[._]1)?(?:\b|_)', re.I)
BAD = re.compile(r'qwen[-_ ]image[-_ ](?:1[._]0|2511)|qwen.?2[._]5|flux|stable.diffusion|sdxl|sd1[._]5', re.I)
TAGS = re.compile(r'<lora:([^:>]+):([-+\d.eE]+)>')

def _kind(keys):
    keys = tuple(k[:-len('.weight')] if k.endswith('.weight') else k for k in keys)
    def modules_with(suffix):
        return {k[:-len(suffix)] for k in keys if k.endswith(suffix)}
    for tag in ('lokr','loha'):
        plain = modules_with('.'+tag+'_w1')
        if plain and all(m+'.'+tag+'_w2' in keys for m in plain):
            return tag
        f1 = modules_with('.'+tag+'_w1_a')
        if f1 and all(m+'.'+tag+'_w1_b' in keys and m+'.'+tag+'_w2_a' in keys
                      and m+'.'+tag+'_w2_b' in keys for m in f1):
            return tag
    if modules_with('.diff'):
        return 'full'
    a = modules_with('.lora_A') | modules_with('.lora_down')
    b = modules_with('.lora_B') | modules_with('.lora_up')
    if a and a == b:
        return 'lora'
    return None

def _qwen21_targets(keys):
    """Require real 2.1 DiT targets; a filename or loose `qwen` label is insufficient."""
    modules = [k.lower() for k in keys]
    return bool(modules) and all(
        ('transformer_blocks.' in k and any(x in k for x in (
            '.attn.to_q.', '.attn.to_k.', '.attn.to_v.', '.attn.to_out.0.',
            '.img_mlp.gate_up.', '.img_mlp.gate_layer.', '.img_mlp.proj.', '.img_mlp.out.')))
        for k in modules
    )

def compatible(path, community=False):
    try:
        h = header(path)
        metadata = json.dumps(h.get('__metadata__', {}))
        keys = ' '.join(k for k in h if k != '__metadata__')
        tensor_keys = tuple(k for k in h if k != '__metadata__')
        evidence = metadata + ' ' + keys
        if BAD.search(evidence) or not FAMILY.search(metadata):
            return False
        if re.search('lightning|distill|turbo', metadata+' '+Path(path).name, re.I):
            return community and _kind(tensor_keys) is not None and _qwen21_targets(tensor_keys)
        return _kind(tensor_keys) is not None and _qwen21_targets(tensor_keys)
    except (OSError, ValueError, TypeError):
        return False

def parse(prompt, community=False):
    matches = TAGS.findall(prompt)
    adapters=[]
    if matches:
        import networks
        if not networks.available_networks: networks.list_available_networks()
        for name, strength in matches:
            item = networks.available_networks.get(name) or networks.available_network_aliases.get(name)
            if item is None:
                normalized=name.replace('\\','/').removesuffix('.safetensors').lower()
                matches=[entry for key,entry in networks.available_networks.items()
                         if key.replace('\\','/').lower()==normalized or Path(entry.filename).stem.lower()==normalized.split('/')[-1]]
                if len(matches)==1: item=matches[0]
            if item is None or not compatible(item.filename, community):
                raise ValueError(f'{name}: not a Qwen-Image-2.1 LoRA (matching metadata/keys required).')
            adapters.append((item.filename, float(strength)))
    stripped = TAGS.sub('',prompt).strip()
    if '<lora:' in stripped or '<lyco:' in stripped:
        raise ValueError('Use native <lora:name:strength> tags only.')
    return stripped, adapters

def has_speed_adapter(adapters):
    return any(re.search(r'turbo|lightning|distill',Path(path).name,re.I) for path,_ in adapters)

def apply(pipe, adapters):
    clear(pipe)
    # Parent process: WorkerPipeline only records paths/weights for the real
    # pipeline. Format detection and tensor loading belong in the GPU worker.
    if not hasattr(pipe, 'transformer'):
        names=[]; weights=[]
        for i,(path,weight) in enumerate(adapters):
            name=f'pi_qwen21_{i}'
            pipe.load_lora_weights(str(Path(path).parent), weight_name=Path(path).name,
                                   adapter_name=name, local_files_only=True)
            names.append(name); weights.append(weight)
        if names: pipe.set_adapters(names, adapter_weights=weights)
        return
    names=[]; weights=[]
    from safetensors.torch import load_file
    for i,(path,weight) in enumerate(adapters):
        h = header(path)
        kind=_kind(k for k in h if k != '__metadata__')
        state=load_file(path, device='cpu')
        if kind=='lokr':
            _install_lokr(pipe.transformer, state, weight, pipe._pi_qwen21_lokr_handles)
        elif kind=='loha':
            _install_loha(pipe.transformer, state, weight, pipe._pi_qwen21_lokr_handles)
        elif kind=='full':
            _install_full(pipe.transformer, state, weight, pipe._pi_qwen21_lokr_handles)
        else:
            _install_lora(pipe.transformer, state, weight, pipe._pi_qwen21_lokr_handles)
    if names: pipe.set_adapters(names, adapter_weights=weights)

def _install_lora(model, state, strength, handles):
    """Apply ordinary LoRA to native or quantized linears without PEFT replacement."""
    import math
    import torch
    import torch.nn.functional as F
    groups={}
    suffixes={'.lora_A.weight':'a','.lora_down.weight':'a',
              '.lora_B.weight':'b','.lora_up.weight':'b','.alpha':'alpha'}
    for key,value in state.items():
        for suffix,kind in suffixes.items():
            if key.endswith(suffix):
                groups.setdefault(key[:-len(suffix)],{})[kind]=value
                break
        else:
            raise ValueError('Unsupported LoRA tensor: '+key)
    planned=[]
    for raw,pair in groups.items():
        if 'a' not in pair or 'b' not in pair: raise ValueError('Incomplete LoRA pair: '+raw)
        a,b=pair['a'],pair['b']
        if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[1] or a.shape[0]==0:
            raise ValueError('Invalid LoRA dimensions: '+raw)
        factor=float(strength)*float(pair.get('alpha',a.shape[0]))/a.shape[0]
        if not math.isfinite(factor): raise ValueError('Invalid LoRA strength: '+raw)
        path=raw
        for prefix in ('model.diffusion_model.','diffusion_model.','transformer.'):
            if path.startswith(prefix): path=path[len(prefix):]
        targets=[(path,b)]
        if path.endswith('.img_mlp.gate_up'):
            if b.shape[0]%2: raise ValueError('Invalid fused LoRA dimensions: '+raw)
            parent=path[:-len('gate_up')]
            targets=list(zip((parent+'gate_layer',parent+'proj'),b.chunk(2,dim=0)))
        for path,up in targets:
            try: module=model.get_submodule(path)
            except AttributeError: raise ValueError('Missing LoRA target: '+path)
            if (getattr(module,'in_features',None),getattr(module,'out_features',None))!=(a.shape[1],up.shape[0]):
                raise ValueError('LoRA shape mismatch: '+path)
            planned.append((module,a,up,factor))
    if not planned: raise ValueError('No supported LoRA pairs')
    start=len(handles)
    try:
        for module,a,b,factor in planned:
            def hook(_module,args,output,a=a,b=b,factor=factor):
                x=args[0].to(device=output.device,dtype=output.dtype)
                down=a.to(device=output.device,dtype=output.dtype)
                up=b.to(device=output.device,dtype=output.dtype)
                return output+F.linear(F.linear(x,down),up)*factor
            handles.append(module.register_forward_hook(hook))
    except Exception:
        for handle in handles[start:]: handle.remove()
        del handles[start:]
        raise
    return len(planned)

def clear(pipe):
    for handle in getattr(pipe, '_pi_qwen21_lokr_handles', ()):
        try: handle.remove()
        except Exception: pass
    pipe._pi_qwen21_lokr_handles = []
    pipe.unload_lora_weights()

def _install_lokr(model, state, strength, handles):
    import torch
    modules = dict(model.named_modules())
    groups = {}
    for key, value in state.items():
        for suffix in ('.lokr_w1','.lokr_w2','.lokr_w1_a','.lokr_w1_b','.lokr_w2_a','.lokr_w2_b','.alpha'):
            if key.endswith(suffix):
                base, leaf = key.rsplit('.', 1)
                groups.setdefault(base, {})[leaf] = value
                break
    applied = 0
    errors = []
    first_handle = len(handles)
    for raw, sides in groups.items():
        def factor(name):
            # LyCORIS factorized storage: lokr_w1 = w1_a @ w1_b, likewise w2.
            if name in sides:
                return sides[name]
            a, b = sides.get(name+'_a'), sides.get(name+'_b')
            if a is None or b is None or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                return None
            return a @ b
        w1 = factor('lokr_w1'); w2 = factor('lokr_w2')
        if w1 is None or w2 is None:
            errors.append(raw + ': incomplete lokr_w1/lokr_w2 pair'); continue
        path = raw
        for prefix in ('model.diffusion_model.', 'diffusion_model.', 'transformer.'):
            if path.startswith(prefix): path = path[len(prefix):]
        targets = [(path, w1, w2)]
        if path.endswith('.img_mlp.gate_up'):
            if w1.ndim != 2 or w1.shape[0] % 2:
                errors.append(raw + ': fused gate_up cannot be split evenly'); continue
            parent = path[:-len('gate_up')]
            first, second = w1.chunk(2, dim=0)
            targets = [(parent + 'gate_layer', first, w2),
                       (parent + 'proj', second, w2)]
        # LyCORIS full-matrix LoKr (use_w1 + use_w2) forces alpha=lora_dim,
        # hence scale=1; serialized alpha is not applied to this plain format.
        # KohakuBlueleaf/LyCORIS lycoris/modules/lokr.py, lines 199-207.
        scale = 1.0
        for target, w1, w2 in targets:
            module = modules.get(target)
            if module is None:
                errors.append(raw + ': missing target ' + target); continue
            if w1.ndim != 2 or w2.ndim != 2:
                errors.append(raw + ': only plain 2D LoKr factors are supported'); continue
            m1,n1 = w1.shape; m2,n2 = w2.shape
            if (getattr(module, 'in_features', None) != n1*n2 or
                    getattr(module, 'out_features', None) != m1*m2):
                errors.append(f'{raw}: shape does not match {target}'); continue
            def hook(_module, args, output, a=w1, b=w2, factor=float(strength)*scale,
                     in1=n1, in2=n2):
                if not args or not isinstance(output, torch.Tensor): return output
                x=args[0]
                if not isinstance(x, torch.Tensor) or x.shape[-1] != in1*in2: return output
                aa=a.to(device=output.device,dtype=output.dtype)
                bb=b.to(device=output.device,dtype=output.dtype)
                flat=x.reshape(-1,in1,in2).to(output.dtype)
                side=torch.einsum('bij,pj->bip',flat,bb)
                side=torch.einsum('bip,qi->bqp',side,aa).reshape_as(output)
                return output + side * factor
            handles.append(module.register_forward_hook(hook)); applied += 1
    if errors or not applied:
        for handle in handles[first_handle:]:
            try: handle.remove()
            except Exception: pass
        del handles[first_handle:]
        detail = '; '.join(errors[:3]) or 'no supported LoKr pairs'
        raise ValueError('Qwen-Image-2.1 LoKr was not applied: ' + detail)
    return applied

def _install_loha(model, state, strength, handles):
    """LoKr with Hadamard-product factors (LyCORIS LoHa), plain and factorized."""
    import torch
    import torch.nn.functional as F
    modules = dict(model.named_modules())
    groups = {}
    for key, value in state.items():
        for suffix in ('.loha_w1','.loha_w2','.loha_w1_a','.loha_w1_b','.loha_w2_a','.loha_w2_b','.alpha'):
            if key.endswith(suffix):
                base, leaf = key.rsplit('.', 1)
                groups.setdefault(base, {})[leaf] = value
                break
    applied = 0
    errors = []
    first_handle = len(handles)
    for raw, sides in groups.items():
        def factor_pair(name):
            # LyCORIS factorized storage: loha_w1 = w1_a @ w1_b, likewise w2.
            if name in sides:
                return sides[name]
            a, b = sides.get(name+'_a'), sides.get(name+'_b')
            if a is None or b is None or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                return None
            return a @ b
        w1 = factor_pair('loha_w1'); w2 = factor_pair('loha_w2')
        if w1 is None or w2 is None:
            errors.append(raw + ': incomplete loha_w1/loha_w2 pair'); continue
        path = raw
        for prefix in ('model.diffusion_model.', 'diffusion_model.', 'transformer.'):
            if path.startswith(prefix): path = path[len(prefix):]
        targets = [(path, w1, w2)]
        if path.endswith('.img_mlp.gate_up'):
            if w1.shape[0] % 2:
                errors.append(raw + ': fused gate_up cannot be split evenly'); continue
            parent = path[:-len('gate_up')]
            first, second = w1.chunk(2, dim=0)
            targets = [(parent + 'gate_layer', first, w2),
                       (parent + 'proj', second, w2)]
        alpha = float(sides.get('alpha', w1.shape[0] if w1.ndim==2 else 1))
        scale = alpha / max(1, w1.shape[0] if w1.ndim==2 else 1)
        for target, a, b in targets:
            module = modules.get(target)
            if module is None:
                errors.append(raw + ': missing target ' + target); continue
            if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                errors.append(raw + ': LoHa factors do not chain'); continue
            def hook(_module, args, output, a=a, b=b, factor=float(strength)*scale):
                if not args or not isinstance(output, torch.Tensor): return output
                x = args[0]
                if not isinstance(x, torch.Tensor): return output
                aa = a.to(device=output.device, dtype=output.dtype)
                bb = b.to(device=output.device, dtype=output.dtype)
                # Hadamard composition: (x @ w1) * (x @ w2) per LyCORIS LoHa.
                side1 = F.linear(x, aa)
                side2 = F.linear(x, bb)
                delta = side1 * side2
                if delta.shape != output.shape:
                    return output
                return output + delta * factor
            handles.append(module.register_forward_hook(hook)); applied += 1
    if errors or not applied:
        for handle in handles[first_handle:]:
            try: handle.remove()
            except Exception: pass
        del handles[first_handle:]
        detail = '; '.join(errors[:3]) or 'no supported LoHa pairs'
        raise ValueError('Qwen-Image-2.1 LoHa was not applied: ' + detail)
    return applied

def _install_full(model, state, strength, handles):
    """Full-difference adapters: `.diff` holds the entire weight delta."""
    import torch
    planned = []
    for key, value in state.items():
        if not key.endswith('.diff'): continue
        path = key[:-len('.diff')]
        for prefix in ('model.diffusion_model.', 'diffusion_model.', 'transformer.'):
            if path.startswith(prefix): path = path[len(prefix):]
        try: module = model.get_submodule(path)
        except AttributeError:
            raise ValueError('Missing full-diff target: ' + path)
        weight = getattr(module, 'weight', None)
        if weight is None or tuple(weight.shape) != tuple(value.shape):
            raise ValueError('Full-diff shape mismatch: ' + path)
        planned.append((module, value))
    if not planned: raise ValueError('No supported full-diff tensors')
    start = len(handles)
    try:
        for module, delta in planned:
            def hook(_module, args, output, delta=delta, factor=float(strength)):
                if not isinstance(output, torch.Tensor): return output
                return output + delta.to(device=output.device, dtype=output.dtype) * factor
            handles.append(module.register_forward_hook(hook))
    except Exception:
        for handle in handles[start:]: handle.remove()
        del handles[start:]
        raise
    return len(planned)

def install_cards(selected):
    try:
        import ui_extra_networks_lora as page
        cls=page.ExtraNetworksPageLora
        if getattr(cls.create_item,'_pi_qwen21',False): return
        original=cls.create_item
        def create(self,*a,**kw):
            item=original(self,*a,**kw)
            if item and not compatible(item['filename'], community=True):
                item=dict(item)
                # Hidden marker lets the browser update existing cards when the
                # checkpoint changes without mutating other engines' tag payloads.
                item['search_terms']=[*item.get('search_terms',[]),'pi_qwen21_incompatible']
            return item
        create._pi_qwen21=True
        cls.create_item=create
    except (ImportError,AttributeError) as e:
        print('[PI-Qwen21] Native LoRA card hook unavailable:',e)
