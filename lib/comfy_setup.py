"""Optional code setup only. Generate never calls this or downloads files."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REVISION='7a5dad695fe1cae25efcb2550530fb20ef68da3d'
RES_REVISION='26036f647ca15d3048a193daf99a40cecfc3820d'

def root_path():
    return str(ROOT/'_comfy')

def model_choices():
    from .assets import models_root
    result=['(none)']
    for directory in ('model_patches','ControlNet','Qwen-Image-2.1'):
        folder=models_root()/directory
        if folder.is_dir():
            result.extend(str(path) for path in folder.rglob('*.safetensors')
                          if 'qwen' in path.name.lower() and '2.1' in path.name and 'control' in path.name.lower())
    return sorted(set(result))

def prompt_encoder_path():
    from .assets import models_root
    folder=models_root()/'text_encoder'
    if folder.is_dir():
        matches=[path for path in folder.rglob('*.safetensors') if 'qwen3.5' in path.name.lower() and '_pe_' in path.name.lower()]
        if matches: return str(sorted(matches)[0])
    return ''

def readiness(backend,comfy_root,sampler,pose,control_enabled,control_model,rewrite_prompt,prompt_encoder,merged_turbo):
    from .assets import models_root
    from .workflow_diagnostics import inspect_readiness
    selected={}
    if control_model and control_model!='(none)':selected['control']=control_model
    if prompt_encoder:selected['prompt_encoder']=prompt_encoder
    report=inspect_readiness(models_root(),ROOT,backend=backend,comfy_root=comfy_root,sampler=sampler,pose=pose,
        control_enabled=control_enabled,rewrite_prompt=rewrite_prompt,merged_turbo=merged_turbo,selected=selected)
    return '**'+report['summary']+'**\n\n'+'\n'.join('- '+str(item['message']) for item in report['checks'])

def prepare(approved=False):
    if not approved: raise ValueError('Approve the isolated backend code download first.')
    target=ROOT/'_comfy'
    marker=target/'.pi_revision'
    if not marker.is_file() or marker.read_text().strip()!=REVISION:
        if target.exists(): raise ValueError('Existing backend differs from the pinned version. Preserve or move it before setup.')
        with tempfile.TemporaryDirectory(prefix='pi-comfy-',dir=ROOT) as temporary:
            temporary=Path(temporary);archive=temporary/'source.zip'
            urllib.request.urlretrieve('https://codeload.github.com/Comfy-Org/ComfyUI/zip/'+REVISION,archive)
            extracted=temporary/'unpacked';extracted.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                for item in bundle.infolist():
                    resolved=(extracted/item.filename).resolve()
                    if not resolved.is_relative_to(extracted.resolve()): raise ValueError('Unsafe backend archive path.')
                bundle.extractall(extracted)
            source=next(extracted.iterdir())
            (source/'.pi_revision').write_text(REVISION)
            os.replace(source,target)
    deps=ROOT/'_deps_comfy';deps.mkdir(exist_ok=True)
    # Only these optional packages are added; never replace Forge's Torch.
    marker=deps/'.pi_ready'
    if not marker.is_file():
        subprocess.run([sys.executable,'-m','pip','install','--target',str(deps),'--no-deps',
                        'comfy-aimdo==0.5.5','simpleeval==1.0.8','blake3==1.0.10'],check=True)
        marker.write_text('0.5.5 / 1.0.8 / 1.0.10')
    addon=target/'custom_nodes'/'RES4LYF'
    if not addon.is_dir():
        with tempfile.TemporaryDirectory(prefix='pi-res-',dir=ROOT) as temporary:
            temporary=Path(temporary);archive=temporary/'res.zip'
            urllib.request.urlretrieve('https://codeload.github.com/ClownsharkBatwing/RES4LYF/zip/'+RES_REVISION,archive)
            extracted=temporary/'unpacked';extracted.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                for item in bundle.infolist():
                    if not (extracted/item.filename).resolve().is_relative_to(extracted.resolve()): raise ValueError('Unsafe sampler archive path.')
                bundle.extractall(extracted)
            addon.parent.mkdir(parents=True,exist_ok=True)
            os.replace(next(extracted.iterdir()),addon)
    return 'Full workflow code ready. No models downloaded and no environment created. Restart Forge after setup.'
