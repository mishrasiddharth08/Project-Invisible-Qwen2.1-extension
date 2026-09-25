"""Scan and reuse first; fetch missing assets only after explicit consent."""
from pathlib import Path
from ..lib.assets import scan, complete, models_root, config, identity, NAMES
import json

SUPPORT='Processor and configuration files (already included)'
CATALOG=json.loads(Path(__file__).with_name('catalog.json').read_text())
CHOICES=[name for names in NAMES.values() for name in names]+[SUPPORT]

# Featured acceleration LoRAs. Every entry lives in its own official Hugging
# Face repository; files are fetched only after the user enables the feature
# and approves the download. Strengths/schedules follow the upstream cards.
FEATURED={
    'Viggle Turbo v0.2.1 (6 steps)': dict(
        repo='Viggle/Qwen-Image-2.1-viggle-turbo',
        file='Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors',
        steps=6, cfg=1.0, strength=1.0,
        source='https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo'),
}
FEATURED_CHOICES=list(FEATURED)

def featured_folder():
    return models_root()/'Qwen-Image-2.1'/'featured-loras'

def find_local_featured():
    """Match featured LoRAs already saved anywhere under the Lora folders (incl. subfolders)."""
    by_name={Path(p).name:p for p in scan()['lora']}
    return {name:by_name[entry['file']] for name,entry in FEATURED.items()
            if entry['file'] in by_name}

def featured_path(name):
    """Existing local copy anywhere (featured folder or Lora folders), else the planned target."""
    entry=FEATURED.get(name)
    if not entry: return None
    local=featured_folder()/entry['file']
    if local.is_file() and local.stat().st_size>8: return local
    found=find_local_featured().get(name)
    if found: return Path(found)
    return local

def download_featured(name, approved=False):
    """Fetch one featured LoRA after explicit approval. Never called by Generate."""
    if name is None or str(name).strip() in ('', '(none)'):
        raise ValueError('Pick a speed LoRA from the dropdown first.')
    entry=FEATURED.get(name)
    if entry is None: raise ValueError('Unknown featured LoRA: '+str(name))
    if not approved: raise ValueError('Accept the LoRA license and authorize the download first.')
    existing=featured_path(name)
    if existing and existing.is_file() and existing.stat().st_size>8:
        return 'Reused existing '+entry['file']+' ('+str(existing)+')'
    target=featured_folder()/entry['file']
    target.parent.mkdir(parents=True,exist_ok=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(entry['repo'],entry['file'],local_dir=str(target.parent))
    if not target.is_file() or target.stat().st_size<=8:
        raise ValueError('Download did not produce '+entry['file'])
    return 'Downloaded '+entry['file']+' from '+entry['source']

def featured_instructions():
    rows=['These optional LoRAs trade full quality for much faster generation (4-6 steps instead of 40).']
    rows.append('They download from the official Hugging Face repositories listed below, only after you approve.')
    for name,entry in FEATURED.items():
        rows.append(f"- [{name}]({entry['source']}): `{entry['file']}` — {entry['steps']} steps, CFG {entry['cfg']:g}")
    rows.append('Files are stored in `'+str(featured_folder())+'`.')
    return '\n'.join(rows)

def support_folder():
    local=models_root()/'Qwen-Image-2.1'/'official'
    bundled=Path(__file__).resolve().parents[1]/'resources'/'qwen21'
    for folder in (local,bundled):
        if identity(folder) and (folder/'processor/tokenizer.json').is_file() and (folder/'scheduler/scheduler_config.json').is_file():
            return folder
    return local

def manual_instructions():
    rows=['Download **one DiT + one text encoder + the VAE**, plus the required processor/configuration files. Existing downloads can stay in the scanned model folders.']
    for kind,names in NAMES.items():
        rows.append('**'+{'dit':'DiT','te':'Text encoder','vae':'VAE'}[kind]+'**')
        rows.extend(f'- [{n}](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/{config()["comfy_revision"]}/{CATALOG[n]})' for n in names)
    rows.append('Place weights in `'+str(models_root()/'Qwen-Image-2.1')+'`.')
    rows.append('Processor, tokenizer and configuration files are included with this extension. Download only the model weights. The `qwen3.5_9b_*_pe_*` files are optional prompt rewriters, not the required Qwen3-VL image encoder.')
    return '\n\n'.join(rows)

def download_selected(selected,confirmed):
    if not confirmed: raise ValueError('Accept the model license and authorize the selected downloads first.')
    if not selected or any(name not in CHOICES for name in selected): raise ValueError('Select files from the model list.')
    inventory=scan()
    existing={Path(p).name for k in NAMES for p in inventory[k]}
    from huggingface_hub import hf_hub_download,snapshot_download
    folder=support_folder()
    messages=[]
    for name in dict.fromkeys(selected):
        if name==SUPPORT:
            ready=identity(folder) and (folder/'processor/tokenizer.json').is_file() and (folder/'scheduler/scheduler_config.json').is_file()
            if ready or inventory['pipelines']: messages.append('Reused support files.'); continue
            snapshot_download('Qwen/Qwen-Image-2.1',revision=config()['weights_revision'],local_dir=str(folder),allow_patterns=['*.json','processor/*','scheduler/*','LICENSE'],ignore_patterns=['*.safetensors','*.bin'])
        elif name in existing or inventory['pipelines'] and '_bf16.' in name:
            messages.append('Reused '+name); continue
        else:
            hf_hub_download('Comfy-Org/Qwen-Image-2.1',CATALOG[name],revision=config()['comfy_revision'],local_dir=str(models_root()/'Qwen-Image-2.1'/'components'))
        messages.append('Downloaded '+name)
    return '\n\n'.join(messages)+'\n\nSelected downloads finished. Refresh the checkpoint list when all required files are available.'

def resolve(selected=None, confirmed=False, prof=None):
    inventory=scan()
    prof=prof or dict(dit='bf16',te='bf16')
    single=selected and Path(selected).suffix=='.safetensors'
    if not single and prof['dit']=='bf16':
        if selected and Path(selected).name=='model_index.json' and complete(Path(selected).parent):
            return dict(folder=str(Path(selected).parent),files={})
        if inventory['pipelines']: return dict(folder=inventory['pipelines'][0],files={})
    expected={'dit':f"qwen_image_2.1_{prof['dit']}.safetensors",'te':f"qwen3vl_8b_{prof['te']}.safetensors",'vae':'qwen_image_2.1_vae_bf16.safetensors'}
    files={k:next((p for p in inventory[k] if Path(p).name.lower()==name),None) for k,name in expected.items()}
    # Reuse available lower-precision 2.1 files instead of demanding BF16 downloads.
    for kind in ('dit','te'):
        alternatives=([f'qwen_image_2.1_int8_convrot.safetensors'] if kind=='dit' else
                      (['qwen3vl_8b_int8_convrot.safetensors','qwen3vl_8b_w4a8.safetensors'] if prof['te']!='w4a8' else []))
        # Quantized files are usable on every card: unsupported GPUs unpack
        # them to bf16 in system RAM at load time (lib/components.py).
        if not files[kind]: files[kind]=next((p for p in inventory[kind] if Path(p).name.lower() in alternatives),None)
    if single:
        files['dit']=str(selected)
        if not Path(selected).is_file(): raise ValueError('Selected checkpoint no longer exists')
    folder=support_folder()
    metadata_ready=identity(folder) and (folder/'processor/tokenizer.json').is_file() and (folder/'scheduler/scheduler_config.json').is_file()
    def bundle():
        return dict(folder=str(folder),files=dict(zip(('transformer','text_encoder','vae'),(files['dit'],files['te'],files['vae']))))
    if all(files.values()) and metadata_ready: return bundle()
    if not confirmed:
        missing=[expected[k] for k,v in files.items() if v is None]
        raise ValueError('Missing assets: '+', '.join(missing or ['processor/config files'])+'. Open Models: download manually (recommended), or choose Automatic and Download selected models. Generate never downloads files.')
    from huggingface_hub import snapshot_download,hf_hub_download,list_repo_files
    folder.mkdir(parents=True,exist_ok=True)
    if not single and prof['dit']=='bf16' and not any(files.values()):
        path=snapshot_download('Qwen/Qwen-Image-2.1',revision=config()['weights_revision'],local_dir=str(models_root()/'Qwen-Image-2.1'/'official'))
        if not complete(path): raise ValueError('Incomplete official snapshot; retry to resume')
        return dict(folder=path,files={})
    if not metadata_ready:
        snapshot_download('Qwen/Qwen-Image-2.1',revision=config()['weights_revision'],local_dir=str(folder),allow_patterns=['*.json','processor/*','scheduler/*','LICENSE'],ignore_patterns=['*.safetensors','*.bin'])
    if not all(files.values()):
        revision=config()['comfy_revision']
        remote_files=list_repo_files('Comfy-Org/Qwen-Image-2.1',revision=revision)
        for kind,value in files.items():
            if value: continue
            remote=next((p for p in remote_files if Path(p).name==expected[kind]),None)
            if remote is None: raise ValueError('Comfy repository lacks '+expected[kind])
            files[kind]=hf_hub_download('Comfy-Org/Qwen-Image-2.1',remote,revision=revision,local_dir=str(models_root()/'Qwen-Image-2.1'/'components'))
    return bundle()
