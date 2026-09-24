"""Strict 2.1 identity and local-only inventory. Never guess an older architecture."""
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'dit': ('qwen_image_2.1_bf16.safetensors', 'qwen_image_2.1_int8_convrot.safetensors'),
    'te': ('qwen3vl_8b_bf16.safetensors', 'qwen3vl_8b_int8_convrot.safetensors', 'qwen3vl_8b_w4a8.safetensors'),
    'vae': ('qwen_image_2.1_vae_bf16.safetensors',),
}
BUCKETS = ((2048,2048),(2400,1792),(1792,2400),(2528,1696),(1696,2528),(2752,1536),(1536,2752))

def config():
    return json.loads((ROOT / 'config.json').read_text(encoding='utf8'))

def models_root():
    try:
        from modules.paths import models_path
        return Path(models_path)
    except ImportError:
        return Path(config().get('live_forge') or ROOT.parent.parent) / 'models'

def header(path):
    with Path(path).open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if not 2 <= n <= 64 * 1024 * 1024:
            raise ValueError('Invalid safetensors header')
        return json.loads(f.read(n))

_ADAPTER_SUFFIXES = (
    '.lora_A.weight', '.lora_B.weight', '.lora_down.weight', '.lora_up.weight',
    '.lokr_w1', '.lokr_w2',
)
_FOREIGN_PATTERNS = ('double_blocks', 'single_blocks', 'input_blocks',
                     'output_blocks', 'middle_block', 'model.diffusion_model.')

def is_dit_file(path):
    """Recognize a Qwen-Image-2.1 DiT by tensor structure, not by filename.

    Community mirrors (e.g. Civitai) rename the weight file; the internal
    layout is what identifies the architecture. Requires transformer block
    attention projections plus a Qwen-2.1-specific MLP naming, and rejects
    adapter files and other architectures (Flux/SD UNet layouts).
    """
    try:
        h = header(path)
    except (OSError, ValueError):
        return False
    keys = [k for k in h if k != '__metadata__']
    if not keys:
        return False
    lowered = [k.lower() for k in keys]
    if any(k.endswith(_ADAPTER_SUFFIXES) for k in lowered):
        return False
    if any(pat in k for k in lowered for pat in _FOREIGN_PATTERNS):
        return False
    blocks = any('transformer_blocks.' in k for k in lowered)
    attn = any('.attn.to_q.' in k for k in lowered)
    mlp = any('.img_mlp.' in k or '.txt_mlp.' in k for k in lowered)
    return blocks and attn and mlp

def identity(folder):
    p = Path(folder)
    try:
        te=json.loads((p/'text_encoder/config.json').read_text())
        tc=te.get('text_config',{})
        if tc and (tc.get('hidden_size')!=4096 or tc.get('num_hidden_layers')!=36):
            return False
        return (json.loads((p/'model_index.json').read_text())['_class_name'] == 'QwenImage21Pipeline'
            and json.loads((p/'transformer/config.json').read_text())['_class_name'] == 'QwenImage21Transformer2DModel'
            and json.loads((p/'vae/config.json').read_text())['_class_name'] == 'AutoencoderKLQwenImage21'
            and te.get('model_type') == 'qwen3_vl')
    except (OSError, ValueError, KeyError):
        return False

def complete(folder):
    p = Path(folder)
    if not identity(p):
        return False
    for component in ('transformer','text_encoder','vae'):
        d = p/component
        indexes = list(d.glob('*.safetensors.index.json'))
        if indexes:
            try:
                shards = set(json.loads(indexes[0].read_text())['weight_map'].values())
                if not shards or not all((d/x).is_file() and (d/x).stat().st_size > 8 for x in shards):
                    return False
            except (OSError, ValueError, KeyError):
                return False
        elif not any(f.stat().st_size > 8 for f in d.glob('*.safetensors')):
            return False
    return (p/'scheduler/scheduler_config.json').is_file() and (p/'processor/tokenizer_config.json').is_file()

def scan(base=None):
    base = Path(base or models_root())
    result = {'dit': [], 'te': [], 'vae': [], 'lora': [], 'pipelines': []}
    for sub in ('Qwen-Image-2.1','diffusion_models','Stable-diffusion','text_encoder','VAE','Lora'):
        d = base/sub
        if not d.is_dir():
            continue
        for p in d.rglob('*.safetensors'):
            if sub in ('Stable-diffusion', 'diffusion_models') and p.name.lower() not in NAMES['dit'] and is_dit_file(p):
                # Community mirror under a renamed file (e.g. Civitai).
                result['dit'].append(str(p.resolve()))
            for kind, names in NAMES.items():
                if p.name.lower() in names:
                    result[kind].append(str(p.resolve()))
            if sub == 'Lora':
                result['lora'].append(str(p.resolve()))
        for p in d.rglob('model_index.json'):
            if complete(p.parent):
                result['pipelines'].append(str(p.parent.resolve()))
    # Reuse the standard Hub cache without any network request.
    try:
        from huggingface_hub import snapshot_download
        cached = snapshot_download('Qwen/Qwen-Image-2.1', local_files_only=True)
        if complete(cached):
            result['pipelines'].append(cached)
    except Exception:
        pass
    return {k: sorted(set(v)) for k,v in result.items()}

def profile(gb, override='auto'):
    if override != 'auto':
        gb = float(override)
    if gb <= 4: return dict(dit='int8_convrot',te='w4a8',side=512,offload=True)
    if gb <= 6: return dict(dit='int8_convrot',te='w4a8',side=768,offload=True)
    if gb <= 8: return dict(dit='int8_convrot',te='w4a8',side=1024,offload=True)
    if gb <= 12: return dict(dit='int8_convrot',te='int8_convrot',side=1024,offload=True)
    if gb <= 16: return dict(dit='int8_convrot',te='int8_convrot',side=1536,offload=True)
    if gb < 24: return dict(dit='bf16',te='bf16',side=2048,offload=True)
    return dict(dit='bf16',te='bf16',side=0,offload=True)

def bucket(width, height, side=0):
    w,h = min(BUCKETS, key=lambda x: abs(x[0]/x[1] - max(1,width)/max(1,height)))
    if side and max(w,h) > side:
        scale = side/max(w,h)
        w,h = max(32,round(w*scale/32)*32),max(32,round(h*scale/32)*32)
    return w,h


def quant_capable(torch):
    """True only when Forge's packed ConvRot/W4A8 kernels can run on this setup.

    Mirrors backend/quant_ops.py: the comfy-kitchen 'cuda' backend is disabled
    when PyTorch has no CUDA, or when the torch build targets CUDA < 13. The
    kernels also require bf16 compute. Everything else (ROCm, old torch builds,
    fp16-only cards) must use the universal bf16 files.
    """
    if bool(getattr(getattr(torch, 'version', None), 'hip', None)):
        return False
    cuda = getattr(getattr(torch, 'version', None), 'cuda', None)
    if not cuda:
        return False
    try:
        if tuple(map(int, str(cuda).split('.'))) < (13,):
            return False
    except ValueError:
        return False
    if not torch.cuda.is_available():
        return False
    check = getattr(torch.cuda, 'is_bf16_supported', None)
    return bool(check()) if callable(check) else False

def hardware_profile(torch, override='auto'):
    """Pick component precisions the actual GPU can execute."""
    if not torch.cuda.is_available():
        raise ValueError('Qwen 2.1 needs a supported NVIDIA CUDA or AMD ROCm PyTorch GPU. DirectML and CPU-only execution are not supported.')
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    result = profile(gb, override)
    if not quant_capable(torch):
        result.update(dit='bf16', te='bf16', portable=True)
    return result
