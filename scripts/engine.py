"""Always-on controls inside the existing txt2img/img2img pages; no extra tab.

STRICT INVISIBILITY SPEC
------------------------
This file is the only place that builds UI for the extension. It must never
add a tab, never touch stock Forge controls, and never change what the user
already sees on the page. Everything lives in one collapsed accordion that
only becomes visible when a Qwen-Image-2.1 checkpoint is selected in the
native checkpoint dropdown.

Ownership map
-------------
- scripts/engine.py (this file) - accordion layout, help text, wiring.
- lib/forge.py                  - hooks the extension into the generation
                                  call and reads the RETURNED control list.
- lib/runtime.py                - model loading, generation, release.
- lib/preset.py                 - saved preset serialization.
- download/manager.py           - model + speed LoRA download catalog.

Data flow: ui() returns a fixed-position tuple; lib/forge.py reads it by
index inside the generation pipeline. The RETURN ORDER IS A CONTRACT and
must not move even when the on-screen layout changes (see banner inside
ui()).
"""  # noqa: D400
import importlib.util
import sys
import weakref
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def _reload_extension_helpers():
    # Forge reloads this entry file but keeps imported extension modules alive.
    # Retire the dedicated worker first, remove callbacks from body-only reloads,
    # then let normal imports build one fresh dependency graph.
    from modules import script_callbacks
    script_callbacks.remove_current_script_callbacks()
    previous=sys.modules.get('pi_qwen21.lib.runtime')
    if previous is not None:
        previous.release()  # LOCK waits for an active generation to finish.
    # Reload in place. Existing generation wrappers keep this module's
    # globals dictionary, so they immediately see the new runtime instead of
    # retaining stale closures or being stacked a second time.
    import importlib
    order=(
        'pi_qwen21.lib.assets','pi_qwen21.lib.components','pi_qwen21.lib.controls',
        'pi_qwen21.lib.alpha','pi_qwen21.lib.degrid','pi_qwen21.lib.denoise',
        'pi_qwen21.lib.offload','pi_qwen21.lib.preview','pi_qwen21.lib.progress',
        'pi_qwen21.lib.prompts','pi_qwen21.lib.refiner','pi_qwen21.lib.spectrum','pi_qwen21.lib.pixel_drift',
        'pi_qwen21.lib.workflows','pi_qwen21.lib.workflow_images',
        'pi_qwen21.lib.worker_client','pi_qwen21.lora.adapter',
        'pi_qwen21.download.manager','pi_qwen21.lib.runtime',
        'pi_qwen21.lib.forge','pi_qwen21.lib.preset')
    for name in order:
        module=sys.modules.get(name)
        if module is not None:
            importlib.reload(module)

_reload_extension_helpers()
if 'pi_qwen21' not in sys.modules:
    spec=importlib.util.spec_from_file_location('pi_qwen21',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
    package=importlib.util.module_from_spec(spec); sys.modules['pi_qwen21']=package; spec.loader.exec_module(package)

TAG='[PI-Qwen21]'

# --------------------------------------------------------------------------- #
# duplicate-copy guard (two installs fight over the same preset and models)
# --------------------------------------------------------------------------- #
def _warn_duplicate_copies():
    """Detect a second installed copy of this extension in the same folder.

    Forge's own "Install from URL" names the folder after the repository
    (Project-Invisible-Qwen2.1-extension) while the README's manual steps use
    project-invisible-qwen-image-21. Installing both ways leaves two working
    copies; Forge loads both because the folder names differ. Warn instead of
    guessing which copy to keep.
    """
    try:
        marker=__file__.replace('\\','/').lower()
        for sibling in ROOT.parent.iterdir():
            if not sibling.is_dir() or sibling.resolve()==ROOT.resolve():
                continue
            twin=sibling/'scripts'/'engine.py'
            if not twin.is_file():
                continue
            try:
                same='pi_qwen21' in twin.read_text(encoding='utf-8',errors='ignore')[:400]
            except OSError:
                continue
            if same:
                print(TAG,'WARNING: two copies of this extension are installed:\n'
                      '  '+str(ROOT)+'\n  '+str(sibling)+'\n'
                      'Delete one (keep only one folder) and restart Forge. Two copies fight over the same preset and models.')
                return
    except OSError:
        pass

_warn_duplicate_copies()

import gradio as gr
from modules import scripts,script_callbacks
from pi_qwen21.lib import forge
from pi_qwen21.lib import runtime
from pi_qwen21.lib import preset
from pi_qwen21.lib.controls import speed_updates, quality_updates
from pi_qwen21.lib import workflows
from pi_qwen21.lib import comfy_setup
from pi_qwen21.download import manager

# --------------------------------------------------------------------------- #
# module state: bindings shared between ui() and the after-component hook
# --------------------------------------------------------------------------- #
PANELS=[]
_BOUND_CHECKPOINTS=weakref.WeakSet()
_BOUND_QUALITY=weakref.WeakSet()
_NATIVE_STEPS={}
_QUALITY_RADIOS=[]
_NATIVE_WORKFLOW={}
_WORKFLOW_BINDINGS=[]
_BOUND_WORKFLOWS=weakref.WeakSet()
_NATIVE_SOURCE={}
_PAD_PREVIEWS=[]
_BOUND_PAD_PREVIEWS=weakref.WeakSet()

# --------------------------------------------------------------------------- #
# help text builders (kept out of ui() so the layout reads like an outline)
# --------------------------------------------------------------------------- #
def _img2img_help():
    """Explains the img2img reference-image model in plain words."""
    return (
        '**Image-to-image (experimental)**  \n'
        'Upload the main image in the native img2img panel. Add up to 9 extra '
        'reference images here for style or subject guidance.'
    )

def _models_help():
    return ('Models come from the official Qwen release on Hugging Face. '
            '**Generate always runs locally** - nothing is sent anywhere. '
            'Nothing downloads unless you approve it below.')

def _refs_help():
    """Reference-image count, per the official Qwen-Image-2.1 model card."""
    return ('Qwen-Image-2.1 supports **up to 10 reference images** at once '
            '(official model card). Your img2img image is reference 1, so add '
            'up to 9 more here. Fewer is usually better - 1-3 give the cleanest results.')

# --------------------------------------------------------------------------- #
# binding helpers: connect extension radios to native Forge controls
# --------------------------------------------------------------------------- #
def _bind_quality(radio, native):
    if native is None or radio in _BOUND_QUALITY: return
    from gradio.context import Context
    if Context.root_block is None: return
    radio.input(fn=lambda value: gr.update(value=int(value)),inputs=[radio],outputs=[native],queue=False)
    _BOUND_QUALITY.add(radio)

def _workflow_updates(name,edit,turbo):
    values=workflows.settings(name,edit)
    if not values: return tuple(gr.skip() for _ in range(23))
    pads=values.get('pads',(0,0,0,0))
    fields=[values['steps'],values['task'],values['turbo'],turbo if values['turbo'] else '(none)',
            values['texture_vae'],values['lanpaint'],values['pose'],values.get('outpaint',False),
            *pads,values.get('follow_source',edit),0,values.get('backend','diffusers'),
            values.get('sampler','euler'),values.get('scheduler','simple'),values.get('control',False),values.get('merged_turbo',False)]
    result=[gr.update(value=value) for value in fields]
    result.extend([gr.update(value=values['steps']),gr.update(value=values['cfg'])])
    result.extend([gr.update(value=values[key]) if key in values else gr.skip() for key in ('width','height')])
    return tuple(result)

def _bind_workflows():
    from gradio.context import Context
    if Context.root_block is None: return
    for recipe,edit,outputs,turbo in _WORKFLOW_BINDINGS:
        mode='img2img' if edit else 'txt2img'
        keys=[mode+'_'+key for key in ('steps','cfg_scale','width','height')]
        if recipe in _BOUND_WORKFLOWS or not all(key in _NATIVE_WORKFLOW for key in keys): continue
        recipe.change(fn=lambda name,edit=edit,turbo=turbo:_workflow_updates(name,edit,turbo),
                      inputs=[recipe],outputs=outputs+[_NATIVE_WORKFLOW[key] for key in keys],queue=False)
        _BOUND_WORKFLOWS.add(recipe)

def _pad_preview(source,enabled,left,right,top,bottom,overlap):
    if source is None or not enabled:return None
    from pi_qwen21.lib.workflow_images import prepare_outpaint
    from PIL import Image
    try:
        canvas,mask=prepare_outpaint(source,left,right,top,bottom,overlap)
        marked=Image.composite(Image.new('RGBA',canvas.size,(128,128,128,255)),canvas,mask)
        marked.thumbnail((512,512),Image.Resampling.LANCZOS)
        return marked
    except (ValueError,TypeError):return None

def _bind_pad_previews():
    from gradio.context import Context
    source=_NATIVE_SOURCE.get('img2img_image')
    if Context.root_block is None or source is None:return
    for preview,controls in _PAD_PREVIEWS:
        if preview in _BOUND_PAD_PREVIEWS:continue
        for component in [source,*controls]:
            component.change(fn=_pad_preview,inputs=[source,*controls],outputs=[preview],queue=False,show_progress='hidden')
        _BOUND_PAD_PREVIEWS.add(preview)

# --------------------------------------------------------------------------- #
# after-component hook: the native checkpoint dropdown and steps slider do
# not exist at ui() time (Forge builds them after extension scripts), so we
# watch components as they are created and bind late. Panels are hidden or
# shown based on the selected checkpoint so non-Qwen models never see this.
# --------------------------------------------------------------------------- #
def after_component(component,**kwargs):
    elem_id=kwargs.get('elem_id',getattr(component,'elem_id',None))
    if elem_id=='img2img_image':
        _NATIVE_SOURCE[elem_id]=component
        _bind_pad_previews()
    if elem_id in {mode+'_'+key for mode in ('txt2img','img2img') for key in ('steps','cfg_scale','width','height')}:
        _NATIVE_WORKFLOW[elem_id]=component
        _bind_workflows()
    if elem_id in ('txt2img_steps','img2img_steps'):
        _NATIVE_STEPS[elem_id]=component
        mode='img2img' if elem_id=='img2img_steps' else 'txt2img'
        for radio,is_img2img in list(_QUALITY_RADIOS):
            if is_img2img == (mode=='img2img'):
                _bind_quality(radio,component)
    if elem_id not in ('setting_sd_model_checkpoint','setting_forge_preset') or not PANELS: return
    if component in _BOUND_CHECKPOINTS: return
    from gradio.context import Context
    if Context.root_block is None: return
    panels=list(PANELS)
    def changed(value):
        from types import SimpleNamespace
        key='forge_preset' if elem_id=='setting_forge_preset' else 'sd_model_checkpoint'
        active=bool(forge.selected(SimpleNamespace(override_settings={key:value})))
        if not active: runtime.release_on_selection()
        return [gr.update(visible=active) for _ in panels]
    component.change(fn=changed,inputs=[component],outputs=panels,queue=False)
    _BOUND_CHECKPOINTS.add(component)

script_callbacks.on_after_component(after_component)

# --------------------------------------------------------------------------- #
# the Script: AlwaysVisible, ZERO changes to stock pages. ui() is the
# accordion (17 RETURNED controls = the script-args contract read by
# lib/forge.py).
# --------------------------------------------------------------------------- #
class Script(scripts.Script):
    _pi_qwen21=True
    def title(self): return 'PROJECT INVISIBLE — Qwen-Image-2.1'
    def show(self,is_img2img): return scripts.AlwaysVisible
    def ui(self,is_img2img):
        mode='edit' if is_img2img else 't2i'
        native=_NATIVE_STEPS.get('img2img_steps' if is_img2img else 'txt2img_steps')
        with gr.Accordion('Qwen · Image 2.1',open=False,visible=bool(forge.selected()),elem_classes=['pi-q21-panel']) as box:
            labels={'ITLText25':'IntoTheLatent · Text to image (25)', 'ITLEdit25':'IntoTheLatent · Image edit (25)',
                    'ITLPose25':'IntoTheLatent · Pose reference (25)', 'UltraTextAdapted50':'Ultra · Text to image (Euler adaptation)',
                    'UltraTurboTextAdapted6':'Ultra Turbo · Text to image (base + LoRA)'}
            workflow=gr.Dropdown([(labels.get(name,name),name) for name in workflows.choices(is_img2img)],value='Custom',label='Workflow preset',info='Applies starting settings. Native controls remain editable; all editing uses img2img.')
            with gr.Row(elem_classes=['pi-q21-pair','pi-q21-output']):
                task=gr.Dropdown([('Standard',mode),('Transparent PNG','rgba')],value=mode,label='Output',scale=1,min_width=180)
                steps=gr.Radio([('Quality',40),('Workflow',25),('Turbo',6),('Fast',8),('Extended',50)],value=40,label='Mode',scale=1,min_width=180)
            _QUALITY_RADIOS.append((steps,is_img2img))
            _bind_quality(steps,native)
            mask=gr.State(None)
            cfg=gr.State(None)
            consent=gr.State(False)
            with gr.Tabs(elem_classes=['pi-q21-tools']):
                with gr.Tab('Finish'):
                    phrase_weights=gr.Checkbox(value=False,label='Weighted prompt phrases',info='Experimental. Example: (warm lighting:1.3). Works independently for positive and negative prompts.')
                    refiner=gr.Dropdown(['Off','Turbo (fast)','Quality (best)'],value='Off',label='Refine details')
                    degrid=gr.Checkbox(value=bool(runtime.config().get('moire_cleanup',True)),label='Remove grid patterns',info='Turn off if fine details look too smooth.')
                if is_img2img:
                    with gr.Tab('Edit'):
                        with gr.Accordion('Workflow tools',open=False):
                            pose=gr.Dropdown(['off','Uploaded pose map','OpenPose','DWPose'],value='off',label='Pose guide',info='Uses reference conditioning, not ControlNet. Automatic extraction needs installed pose models.')
                            follow_source=gr.Checkbox(value=False,label='Use source image proportions')
                            ref_mp=gr.Dropdown([('Original references',0),('0.59 MP',0.59),('1 MP',1)],value=0,label='Reference memory limit',info='Downscales conditioning images only. Masked edits keep their source.')
                            outpaint=gr.Checkbox(value=False,label='Expand canvas (outpaint)',info='Automatically builds the edit mask. Uses LanPaint to preserve the source.')
                            with gr.Row():
                                left=gr.Number(value=0,precision=0,label='Left pixels');right=gr.Number(value=0,precision=0,label='Right pixels')
                                top=gr.Number(value=0,precision=0,label='Top pixels');bottom=gr.Number(value=0,precision=0,label='Bottom pixels')
                            overlap=gr.Slider(0,128,value=0,step=1,label='Blend inside edge (pixels)',info='Zero preserves every source pixel; overlap permits edge correction.')
                            pad_preview=gr.Image(label='Canvas preview · gray areas will be generated',interactive=False,height=180)
                            _PAD_PREVIEWS.append((pad_preview,[outpaint,left,right,top,bottom,overlap]))
                            _bind_pad_previews()
                        reference_priorities=gr.Textbox(value='',label='Reference priorities',placeholder='1:1.2, 2:0.8',info='Image number 1–10:strength 0–8. Strength 1 is native. Main image is reference 1.')
                        lanpaint=gr.Checkbox(value=False,label='LanPaint masked edit',info='Experimental. Paint a mask in native img2img Inpaint. Slower; preserves unmasked pixels.')
                        lanpaint_steps=gr.Slider(1,5,value=2,step=1,label='LanPaint thinking steps')
                        with gr.Row(elem_classes=['pi-q21-pair']):
                            pixel_drift=gr.Checkbox(value=False,label='Align edit to source',info='PixelDriftFix: correct small framing shifts.')
                            composite=gr.Checkbox(value=False,label='Preserve background',info='Blend unchanged areas with the original.')
                        with gr.Accordion('Extra references · up to 9',open=False,elem_classes=['pi-q21-refs']):
                            gr.Markdown('Your main img2img image is reference 1. Add others only when needed.',elem_classes=['pi-q21-status'])
                            refs=[]
                            for row in range(3):
                                with gr.Row(elem_classes=['pi-q21-ref-row']):
                                    for col in range(3):
                                        refs.append(gr.Image(type='pil',sources=['upload'],label=f'Reference {row*3+col+2}',height=120,show_download_button=False,min_width=80,elem_classes=['pi-q21-ref']))
                else:
                    pose=gr.State('off');follow_source=gr.State(False);ref_mp=gr.State(0)
                    outpaint=gr.State(False);left=gr.State(0);right=gr.State(0);top=gr.State(0);bottom=gr.State(0);overlap=gr.State(0)
                    composite=gr.State(False)
                    reference_priorities=gr.State('')
                    lanpaint=gr.State(False)
                    lanpaint_steps=gr.State(2)
                    pixel_drift=gr.State(False)
                    refs=[gr.State(None) for _ in range(9)]
                with gr.Tab('Speed'):
                    turbo=list(manager.FEATURED_CHOICES)[0]
                    speed_enabled=gr.Checkbox(value=False,label='Use turbo LoRA',info='Fast mode selects this automatically. Uses CFG 1.')
                    with gr.Row(elem_classes=['pi-q21-pair']):
                        speed_name=gr.Dropdown(['(none)',turbo],value='(none)',label='Turbo model',scale=1,min_width=180)
                        speed_strength=gr.Slider(0.0,1.5,value=1.0,step=0.05,label='Qwen Turbo strength',elem_id=f'pi_q21_{mode}_turbo_strength',scale=1,min_width=180)
                    spectrum=gr.Checkbox(value=False,label='Spectrum acceleration',info='Experimental; can change details. Turn off for an exact baseline.')
                    kv_cache=gr.Checkbox(value=True,label='Cache reference attention',info='Native prefix KV cache. LanPaint disables it for masked sampling.')
                    sharp=gr.Dropdown([('Off',0),('DPM++ 2M Sharp 0.15',0.15),('Strong 0.35',0.35)],value=0,label='Sharp sampler',info='DPM++ 2M Sharp (envy-ai): sharpened denoised history. Quality runs only; turbo keeps its exact schedule.')
                    speed_status=gr.Markdown(elem_classes=['pi-q21-status'])
                    with gr.Accordion('Download turbo model',open=False):
                        gr.Markdown(manager.featured_instructions())
                        speed_approved=gr.Checkbox(value=False,label='I accept the license and approve this download')
                        speed_button=gr.Button('Download turbo model',size='sm')
                    speed_button.click(fn=manager.download_featured,inputs=[speed_name,speed_approved],outputs=[speed_status])
                    speed_outputs=[speed_name,speed_strength]+([native] if native is not None else [])
                    speed_enabled.input(fn=lambda enabled:speed_updates(enabled,turbo,native=native is not None),inputs=[speed_enabled],outputs=speed_outputs,queue=False,show_progress='hidden')
                    steps.input(fn=lambda value:quality_updates(value,turbo),inputs=[steps],outputs=[speed_enabled,speed_name],queue=False,show_progress='hidden')
                with gr.Tab('Style'):
                    style_name=gr.Dropdown(manager.STYLE_CHOICES,value='(none)',label='Photography style',info='The style trigger is added automatically.')
                    style_status=gr.Markdown(elem_classes=['pi-q21-status'])
                    with gr.Accordion('Download a style',open=False):
                        gr.Markdown(manager.style_instructions())
                        style_approved=gr.Checkbox(value=False,label='I accept the license and approve this download')
                        style_button=gr.Button('Download selected style',size='sm')
                    style_button.click(fn=manager.download_style,inputs=[style_name,style_approved],outputs=[style_status])
                    with gr.Accordion('Detail fix (removes the plastic look)',open=False):
                        fix_name=gr.Dropdown(manager.FIX_CHOICES,value='(none)',label='Detail-fix LoRA',info='Optional; download once below. Missing files are skipped with a warning.')
                        texture_vae=gr.Checkbox(value=False,label='Use the texture-fix VAE',info='Real micro-texture instead of the stock VAE\'s plastic surface.')
                        gr.Markdown(manager.fix_instructions())
                        gr.Markdown(manager.texture_vae_instructions())
                        fix_approved=gr.Checkbox(value=False,label='I accept the license and approve these downloads')
                        fix_button=gr.Button('Download selected detail fix',size='sm')
                        fix_status=gr.Markdown(elem_classes=['pi-q21-status'])
                    fix_button.click(fn=manager.download_fix,inputs=[fix_name,fix_approved],outputs=[fix_status])
                with gr.Tab('Memory'):
                    with gr.Row(elem_classes=['pi-q21-pair']):
                        profile=gr.Dropdown(['auto','4','6','8','12','16','20','24'],value='auto',label='VRAM budget',info='Auto detects your GPU.',scale=1,min_width=180)
                        side=gr.Dropdown([('Automatic',0),('512 px',512),('768 px',768),('1024 px',1024),('1536 px',1536),('2048 px',2048)],value=0,label='Maximum size',scale=1,min_width=180)
                    offload=gr.Checkbox(value=True,label='Save GPU memory',info='Keeps unused model parts off the GPU.')
                    keep_loaded=gr.Checkbox(value=bool(runtime.config().get('keep_loaded',True)),label='Keep Qwen ready',info='Faster next generation. Idle GPU memory is released; model files stay cached in system RAM.')
                    with gr.Accordion('Advanced compatibility',open=False):
                        community=gr.Checkbox(value=False,label='Allow compatible community LoRAs',info='Enable only for compatible Qwen 2.1 adapters.')
                with gr.Tab('Models'):
                    with gr.Accordion('Full workflow backend · advanced',open=False):
                        backend=gr.Dropdown([('Default pipeline','diffusers'),('Full workflows (isolated Comfy)','comfy')],value='diffusers',label='Generation engine',info='Full workflows run in a separate process, with no server, tab or new environment.')
                        comfy_root=gr.Textbox(value=comfy_setup.root_path(),label='Isolated Comfy code folder')
                        with gr.Row():
                            comfy_sampler=gr.Dropdown(['euler','res_2s'],value='euler',label='Full-workflow sampler')
                            comfy_scheduler=gr.Dropdown(['simple','beta'],value='simple',label='Full-workflow scheduler')
                        control_enabled=gr.Checkbox(value=False,label='Use Qwen 2.1 Union ControlNet')
                        control_files=comfy_setup.model_choices()
                        control_model=gr.Dropdown(control_files,value=control_files[-1] if len(control_files)==2 else '(none)',allow_custom_value=True,label='Union control model file',info='Requires the dedicated Qwen 2.1 16-block control model.')
                        union_approved=gr.Checkbox(value=False,label='I accept the publisher license and approve the 3.5 GB Union model download')
                        union_button=gr.Button('Download Union model',size='sm')
                        union_status=gr.Markdown()
                        union_button.click(fn=manager.download_union,inputs=[union_approved],outputs=[union_status])
                        control_guide=gr.Image(type='pil',sources=['upload'],label='Control guide (pose, depth or edges)',height=120)
                        with gr.Row():
                            control_strength=gr.Slider(0,2,value=1,step=.05,label='Control strength')
                            control_start=gr.Slider(0,1,value=0,step=.05,label='Control starts')
                            control_end=gr.Slider(0,1,value=1,step=.05,label='Control ends')
                        rewrite_prompt=gr.Checkbox(value=False,label='Local prompt enhancer',info='Loads a separate prompt encoder. It never replaces the image model\'s Qwen3-VL encoder.')
                        rewrite_thinking=gr.Checkbox(value=False,label='Thorough prompt rewriting',info='Optional reasoning mode from the Ultra workflow; slower and uses more tokens.')
                        prompt_encoder=gr.Textbox(value=comfy_setup.prompt_encoder_path(),label='Local prompt encoder file (optional)')
                        merged_turbo=gr.Checkbox(value=False,label='Use the merged Viggle Turbo checkpoint',info='Full workflow engine only. Use six steps / CFG 1; separate Turbo LoRA must be off.')
                        merged_approved=gr.Checkbox(value=False,label='I accept the Qwen Research License and approve the 7.3 GB merged model download')
                        merged_button=gr.Button('Download merged Turbo model',size='sm')
                        merged_status=gr.Markdown()
                        merged_button.click(fn=manager.download_merged_turbo,inputs=[merged_approved],outputs=[merged_status])
                        readiness_button=gr.Button('Check workflow setup',size='sm')
                        readiness_status=gr.Markdown()
                        readiness_button.click(fn=comfy_setup.readiness,
                            inputs=[backend,comfy_root,comfy_sampler,pose,control_enabled,control_model,rewrite_prompt,prompt_encoder,merged_turbo],outputs=[readiness_status])
                        setup_approved=gr.Checkbox(value=False,label='I approve downloading isolated backend code and packages')
                        gr.Markdown('Comfy code keeps its GPL license. RES4LYF keeps its original license, including commercial-service restrictions. Model licenses are unchanged. See the workflow guide.')
                        setup_button=gr.Button('Prepare full workflow backend',size='sm')
                        setup_status=gr.Markdown()
                        setup_button.click(fn=comfy_setup.prepare,inputs=[setup_approved],outputs=[setup_status])
                    with gr.Accordion('Setup guide',open=False):
                        gr.Markdown(_models_help())
                        gr.Markdown(manager.manual_instructions())
                    with gr.Accordion('Download models',open=False):
                        downloads=gr.CheckboxGroup(manager.CHOICES,value=['qwen_image_2.1_bf16.safetensors','qwen3vl_8b_bf16.safetensors','qwen_image_2.1_vae_bf16.safetensors',manager.SUPPORT],label='Model files',info='Choose one DiT, one text encoder and the VAE.')
                        approved=gr.Checkbox(value=False,label='I accept the license and approve these downloads')
                        button=gr.Button('Download selected models',size='sm')
                        status=gr.Markdown(elem_classes=['pi-q21-status'])
                        button.click(fn=manager.download_selected,inputs=[downloads,approved],outputs=[status])
        self._box=box
        PANELS.append(box)
        _WORKFLOW_BINDINGS.append((workflow,is_img2img,[steps,task,speed_enabled,speed_name,texture_vae,lanpaint,pose,outpaint,left,right,top,bottom,follow_source,ref_mp,backend,comfy_sampler,comfy_scheduler,control_enabled,merged_turbo],turbo))
        _bind_workflows()
        # Preserve existing positions; new controls append to the contract.
        return [task,steps,cfg,profile,side,offload,community,consent,mask,*refs,spectrum,degrid,speed_enabled,speed_name,speed_strength,refiner,style_name,composite,pixel_drift,keep_loaded,lanpaint,lanpaint_steps,fix_name,texture_vae,sharp,phrase_weights,reference_priorities,workflow,pose,outpaint,left,right,top,bottom,overlap,ref_mp,kv_cache,follow_source,backend,comfy_root,control_enabled,control_model,control_guide,control_strength,control_start,control_end,comfy_sampler,comfy_scheduler,rewrite_prompt,prompt_encoder,merged_turbo,rewrite_thinking]


# --------------------------------------------------------------------------- #
# boot: install the generation hooks and clean up on extension unload
# --------------------------------------------------------------------------- #
try:
    forge.install()
    preset.install()
except (ImportError,AttributeError,TypeError,RuntimeError) as exc:
    print(TAG,'Neo hook drift; adapter disabled:',exc)
script_callbacks.on_script_unloaded(runtime.release)
