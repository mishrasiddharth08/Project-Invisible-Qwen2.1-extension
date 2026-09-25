"""QwenImage21Pipeline exclusively; no SD/Flux/old-Qwen execution paths."""
import gc
import inspect
import random
import threading
from pathlib import Path
from .assets import config, hardware_profile, bucket, identity
from ..download.manager import resolve
from ..download import manager as downloads
from ..lora import adapter
from .prompts import parse as parse_rewrite
from .progress import ForgeProgress
from .degrid import remove_moire

LOCK=threading.RLock()
_pipe=None
_key=None

def release_on_selection():
    # A selection change cancels this extension's dedicated worker, never Forge.
    pipe=_pipe
    process=getattr(pipe,'_process',None)
    if process is not None and process.poll() is None:
        try: process.terminate()
        except ProcessLookupError: pass
    release()

def release():
    global _pipe, _key
    with LOCK:
        if _pipe is None:
            return
        if _pipe is not None:
            try:
                _pipe.unload_lora_weights()
                if hasattr(_pipe,'remove_all_hooks'): _pipe.remove_all_hooks()
            finally:
                _pipe=None; _key=None
                gc.collect()
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()

def load(bundle, prof, offload=True):
    global _pipe,_key
    import torch
    from .worker_client import WorkerPipeline
    folder=bundle['folder']
    if not identity(folder): raise ValueError('Only Qwen 2.1 DiT, Qwen3-VL and Qwen 2.1 VAE are accepted.')
    key=(folder,tuple(sorted(bundle.get('files',{}).items())),tuple(sorted(prof.items())),offload)
    if _key == key and _pipe is not None: return _pipe
    release()
    from backend import memory_management
    memory_management.unload_all_models()
    pipe=WorkerPipeline(bundle,prof,offload)
    _pipe,_key=pipe,key
    return pipe

def generate(p, selected, options):
    import torch
    from modules import processing, shared, images
    with LOCK:
        defaults=config()
        options={**defaults,**options}
        prof=hardware_profile(torch,options.get('profile','auto'))
        prompt=p.prompt if isinstance(p.prompt,str) else p.prompt[0]
        prompt,p.width,p.height=parse_rewrite(prompt,p.width,p.height)
        prompt,adapters=adapter.parse(prompt, options.get('community',False))
        negative=p.negative_prompt if isinstance(p.negative_prompt,str) else p.negative_prompt[0]
        cfg=float(getattr(p,'cfg_scale',1.0))
        # Featured speed LoRAs are distilled for few-step CFG-1 sampling.
        speed_name=options.get('speed_lora') if options.get('speed_enabled') else None
        speed_entry=downloads.FEATURED.get(speed_name) if speed_name else None
        if options.get('speed_enabled') and speed_entry is None:
            print('[PI-Qwen21] Speed boost is ticked but no speed LoRA is selected or recognized (dropdown may show "(none)"). Running the full step schedule - pick a downloaded LoRA under Qwen Controls > Speed boost LoRA.')
        speed_adapter=None
        speed_sigmas=None
        if speed_entry:
                speed_path=downloads.featured_path(speed_name)
                if not speed_path or not speed_path.is_file() or speed_path.stat().st_size<=8:
                    raise ValueError(speed_name+' is not downloaded yet. Open Qwen Controls > Speed boost LoRA and approve the one-time download, or untick the box. Generate never downloads files.')
                if cfg>1: print('[PI-Qwen21] Speed LoRA selected: using CFG 1 as required by the distillation.')
                cfg=1.0
                speed_adapter=(str(speed_path), float(options.get('speed_strength',speed_entry['strength'])))
                # The Viggle card mandates its own sigma nodes and a scheduler
                # with shift_terminal disabled; the base config's 0.02 wrecks
                # the last step. Sampling must match the distillation exactly.
                # Sigmas are built after p.steps is finalized so the count
                # always matches the schedule.
        if cfg>1 and not negative.strip():
            print('[PI-Qwen21] No negative prompt: using CFG 1 for this generation.')
            cfg=1.0
            p.cfg_scale=cfg  # Keep saved generation metadata consistent with actual inference.
        primary=list(getattr(p,'init_images',None) or [])[:1]
        img2img_type=getattr(processing,'StableDiffusionProcessingImg2Img',None)
        is_edit=isinstance(p,img2img_type) if isinstance(img2img_type,type) else bool(primary)
        mask=options.get('mask')
        if mask is None: mask=getattr(p,'image_mask',None)
        task=options.get('task','t2i')
        if not is_edit and (primary or any(im is not None for im in options.get('refs',[])) or mask is not None or task=='edit'):
            raise ValueError('All image editing must use the img2img tab. Upload the main image there.')
        if is_edit and not primary:
            raise ValueError('Upload the main image in the img2img tab before editing.')
        refs=primary
        refs.extend(im for im in options.get('refs',[]) if im is not None)
        if len(refs)>10: raise ValueError('At most 10 reference images are supported.')
        if task=='rgba': prompt='This is an RGBA image with transparency. '+prompt+'. The image has alpha channel and the background is transparent.'
        p.steps=max(1,int(getattr(p,'steps',None) or options.get('steps',40)))
        total=max(1,int(p.batch_size))*max(1,int(p.n_iter))
        display=ForgeProgress(shared,p.steps,total)
        try:
            folder=resolve(selected, False, prof)
            if folder.get('files'):
                print('[PI-Qwen21] Using dedicated 2.1 components: '+', '.join(f'{kind}={Path(path).name}' for kind,path in folder['files'].items()))
            try:
                pipe=load(folder,prof,options.get('offload',True))
            except torch.cuda.OutOfMemoryError as exc:
                release()
                raise RuntimeError('Qwen 2.1 could not fit on this GPU. Enable Save GPU memory and choose a 4 or 6 GB profile. More system RAM may also be needed.') from exc
            params=inspect.signature(pipe.__call__).parameters
            if mask is not None and 'mask_image' not in params: raise ValueError('This QwenImage21Pipeline does not expose mask_image. Use a painted/circled reference and describe the edit instead.')
            requested_side=max(32,int(p.width),int(p.height))
            side=int(options.get('side',0)) or prof['side']
            side=min(requested_side,side) if side else requested_side
            p.width,p.height=bucket(p.width,p.height,side)
            p.steps=max(1,int(getattr(p,'steps',None) or options.get('steps',40))); p.cfg_scale=cfg
            # User steps always win: choosing Fast auto-selects the turbo LoRA
            # and matches the slider, but a slider moved by hand is never overridden.
            if speed_entry: speed_sigmas=downloads.turbo_sigmas(p.steps)
            p.sampler_name='Euler'
            seed=int(p.seed if p.seed is not None else -1)
            if seed<0: seed=random.randrange(2**32)
            p.seed=seed
            output=[]; seeds=[]; infos=[]
            total=max(1,int(p.batch_size))*max(1,int(p.n_iter))
            shared.state.job_count=total
            try:
                if speed_adapter: adapter.apply(pipe,list(adapters)+[speed_adapter])
                else: adapter.apply(pipe,adapters)
                for i in range(total):
                    if shared.state.interrupted or shared.state.skipped: break
                    current_seed=(seed+i)%2**32
                    display.start_image(i,p.width,p.height)
                    args=dict(prompt=prompt,negative_prompt=negative if cfg>1 else None,true_cfg_scale=cfg,width=p.width,height=p.height,num_inference_steps=p.steps,generator=torch.Generator('cpu').manual_seed(current_seed))
                    if speed_sigmas: args['sigmas']=speed_sigmas
                    if refs: args['image']=refs
                    if mask is not None: args['mask_image']=mask
                    if 'use_kv_cache' in params: args['use_kv_cache']=True
                    if 'callback_on_step_end' in params: args['callback_on_step_end']=display.step
                    if 'preview_every' in params: args['preview_every']=display.preview_every
                    if 'status_callback' in params: args['status_callback']=display.status
                    if 'spectrum' in params: args['spectrum']=bool(options.get('spectrum',False))
                    if 'output_resolution' in params: args['output_resolution']=max(p.width,p.height)
                    try:
                        result=pipe(**args).images[0]
                    except torch.cuda.OutOfMemoryError as exc:
                        release()
                        raise RuntimeError('Qwen 2.1 ran out of GPU memory. Enable Save GPU memory and choose 512 or 768 long-side; reduce reference images or use a smaller VRAM profile.') from exc
                    except RuntimeError:
                        release()  # Do not reuse a pipeline left partially modified by a failed adapter.
                        raise
                    cleanup=float(options.get('moire_strength',1.0)) if options.get('moire_cleanup',True) else 0.0
                    if cleanup:
                        result=remove_moire(result,strength=cleanup)
                    display.publish(result, final=True)
                    info=f'{prompt}\nSteps: {p.steps}, Sampler: Euler, Schedule type: simple, CFG scale: {cfg}, Seed: {current_seed}, Size: {p.width}x{p.height}, Model: Qwen-Image-2.1'
                    info+=f', Qwen moire cleanup: {cleanup:g}'
                    info+=f', DeGrid: {"auto" if cleanup else "off"}'
                    info+=f", Spectrum requested: {bool(options.get('spectrum',False))}"
                    if speed_entry: info+=f", Speed LoRA: {speed_name} ({speed_adapter[1]:g})"
                    if not getattr(p,'do_not_save_samples',False) and shared.opts.samples_save:
                        images.save_image(result,p.outpath_samples,'',current_seed,prompt,extension='png',info=info,p=p)
                    output.append(result); seeds.append(current_seed); infos.append(info)
                    shared.state.nextjob()
            except InterruptedError:
                pass
            finally:
                if _pipe is not None: _pipe.unload_lora_weights()
            return processing.Processed(p,output,seed,infos[0] if infos else 'Interrupted',all_seeds=seeds,infotexts=infos)
        finally:
            display.close()
