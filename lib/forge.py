"""Chained, namespaced hooks based on the Ideogram reference adapter."""
from pathlib import Path
from .assets import scan, models_root, NAMES, is_dit_file
from . import runtime

LABEL='PROJECT INVISIBLE — Qwen-Image-2.1 (official)'
KEYS=('task','steps','true_cfg','profile','side','offload','community','consent','mask')
def selected(p=None):
    from modules import shared,sd_models
    value=(getattr(p,'override_settings',{}) or {}).get('sd_model_checkpoint',shared.opts.sd_model_checkpoint)
    if value==LABEL: return str(Path(__file__).resolve().parents[1]/'resources'/'qwen21'/'model_index.json')
    ci=sd_models.checkpoints_list.get(value) or sd_models.checkpoint_aliases.get(value)
    if ci is not None and getattr(ci,'_pi_qwen21',False): return ci.filename
    filename=getattr(ci,'filename','')
    if Path(filename).name.lower() in NAMES['dit']: return filename
    # Community mirrors rename DiT files (e.g. Civitai); recognize by contents.
    if filename and Path(filename).suffix=='.safetensors' and is_dit_file(filename): return filename
    return None

def register():
    from modules import sd_models
    inventory=scan()
    candidates=[(LABEL, str(Path(__file__).resolve().parents[1]/'resources'/'qwen21'/'model_index.json'))]
    candidates += [('Qwen-Image-2.1 — '+Path(p).parent.name,str(Path(p)/'model_index.json')) for p in inventory['pipelines']]
    candidates += [('Qwen-Image-2.1 — '+Path(p).name,p) for p in inventory['dit']]
    for label,path in candidates:
        same = [v for v in sd_models.checkpoints_list.values()
                if getattr(v, '_pi_qwen21', False)
                and getattr(v, 'filename', None) == path]
        if any(getattr(v, 'name', None) == label for v in same):
            continue
        # Non-weight marker avoids stock hashing/SD model inspection.
        ci=object.__new__(sd_models.CheckpointInfo)
        ci.filename=path; ci.name=label; ci.title=label; ci.short_title=label
        ci.model_name=label; ci.name_for_extra=label; ci.hash='qwen21'; ci.sha256=None; ci.shorthash=None
        ci.metadata={}; ci.is_safetensors=False; ci._pi_qwen21=True; ci.ids=[label,path]
        ci.calculate_shorthash=lambda: None
        for k,v in list(sd_models.checkpoints_list.items()):
            if getattr(v, '_pi_qwen21', False) and getattr(v, 'filename', None) == path:
                sd_models.checkpoints_list.pop(k,None)
        ci.register()

def options(runner,p):
    result={}
    for script in getattr(runner,'alwayson_scripts',[]) or []:
        if getattr(script,'_pi_qwen21',False):
            values=list(getattr(p,'script_args',[]) or [])[script.args_from:script.args_to]
            result.update(zip(KEYS,values[:len(KEYS)])); result['refs']=values[len(KEYS):len(KEYS)+9]
            if len(values)>len(KEYS)+9: result['spectrum']=bool(values[len(KEYS)+9])
            if len(values)>len(KEYS)+10: result['moire_cleanup']=bool(values[len(KEYS)+10])
            if len(values)>len(KEYS)+11:
                result['speed_enabled']=bool(values[len(KEYS)+11])
                result['speed_lora']=str(values[len(KEYS)+12] or '')
                result['speed_strength']=float(values[len(KEYS)+13])
            if len(values)>len(KEYS)+14: result['refiner']=str(values[len(KEYS)+14] or 'off').lower()
            break
    return result

def install_selection_release():
    from modules import shared
    for key in ('sd_model_checkpoint','forge_preset'):
        item=shared.opts.data_labels.get(key)
        if item is None: continue
        previous=item.onchange
        if getattr(previous,'_pi_qwen21_release',False) is True: continue
        def changed(previous=previous,key=key):
            # Release before another engine's selection callback loads its weights.
            if (key=='forge_preset' and shared.opts.forge_preset!='qwen-image-2.1') or (key=='sd_model_checkpoint' and not selected()):
                runtime.release_on_selection()
            if previous is not None: return previous()
        changed._pi_qwen21_release=True
        shared.opts.onchange(key,changed,call=False)

def install():
    from modules import scripts,processing,sd_models,script_callbacks
    required=(getattr(scripts,'ScriptRunner',None),getattr(processing,'process_images',None),getattr(sd_models,'list_models',None))
    if not all(required) or not hasattr(scripts.ScriptRunner,'run'):
        print('[PI-Qwen21] Neo hook drift: adapter disabled; other engines unchanged.'); return
    if getattr(scripts.ScriptRunner.run,'_pi_qwen21',False): return
    original_run=scripts.ScriptRunner.run
    original_process=processing.process_images
    original_list=sd_models.list_models
    def run(runner,p,*args,**kwargs):
        path=selected(p)
        if path: return runtime.generate(p,path,options(runner,p))
        runtime.release()
        return original_run(runner,p,*args,**kwargs)
    def process(p,*args,**kwargs):
        path=selected(p)
        if path: return runtime.generate(p,path,options(getattr(p,'scripts',None),p))
        runtime.release()
        return original_process(p,*args,**kwargs)
    def listing(*args,**kwargs):
        result=original_list(*args,**kwargs); register(); return result
    run._pi_qwen21=True
    scripts.ScriptRunner.run=run
    processing.process_images=process
    sd_models.list_models=listing
    def ready(*args):
        install_selection_release()
        import sys
        for name in ('modules.api.api','modules.txt2img','modules.img2img'):
            module=sys.modules.get(name)
            if module and getattr(module,'process_images',None) is original_process: module.process_images=process
        from ..lora.adapter import install_cards
        install_cards(selected)
        register()
    script_callbacks.on_app_started(ready)
    register()
