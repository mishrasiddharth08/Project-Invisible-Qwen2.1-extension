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
from pi_qwen21.download import manager

# --------------------------------------------------------------------------- #
# module state: bindings shared between ui() and the after-component hook
# --------------------------------------------------------------------------- #
PANELS=[]
_BOUND_CHECKPOINTS=weakref.WeakSet()
_BOUND_QUALITY=weakref.WeakSet()
_NATIVE_STEPS={}
_QUALITY_RADIOS=[]

# --------------------------------------------------------------------------- #
# help text builders (kept out of ui() so the layout reads like an outline)
# --------------------------------------------------------------------------- #
def _quickstart():
    """One-paragraph 'how to use this panel' card shown at the top."""
    return (
        '**Quick start**  \n'
        '1. Pick a Qwen-Image-2.1 checkpoint above (this panel appears once you do).  \n'
        '2. Get the models under **Model Setup** if you have not already.  \n'
        '3. Choose **Quality** or **Fast**, type a prompt, press Generate.'
    )

def _img2img_help():
    """Explains the img2img reference-image model in plain words."""
    return (
        '**Image-to-image (experimental)**  \n'
        'Upload the main image in the native img2img panel. Add up to 9 extra '
        'reference images here for style or subject guidance.'
    )

def _steps_note():
    """Why the native Sampling Steps slider still matters."""
    return ('The native **Sampling Steps** slider controls the generation length. '
            'The Quality choice above keeps it in sync automatically.')

def _speed_note():
    """What the Speed boost LoRA does, and its caveats."""
    return ('Enabling this uses CFG 1 and the LoRA\'s own few-step schedule. '
            'Strength 1.0 is the tested default; Spectrum and native CFG above 1 '
            'are ignored while it is active.')

def _prompt_helper_note():
    """How the copy-paste prompt helper is meant to be used."""
    return 'Paste the resulting JSON into the normal prompt field.'

def _models_help():
    """Licensing and local-only reassurance above the download options."""
    return ('Models download from the official Qwen release. **Generate always runs '
            'locally** - nothing is sent anywhere, and nothing downloads without '
            'your explicit approval below.')

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
    radio.change(fn=lambda value: gr.update(value=int(value)),inputs=[radio],outputs=[native],queue=False)
    _BOUND_QUALITY.add(radio)

# --------------------------------------------------------------------------- #
# after-component hook: the native checkpoint dropdown and steps slider do
# not exist at ui() time (Forge builds them after extension scripts), so we
# watch components as they are created and bind late. Panels are hidden or
# shown based on the selected checkpoint so non-Qwen models never see this.
# --------------------------------------------------------------------------- #
def after_component(component,**kwargs):
    elem_id=kwargs.get('elem_id',getattr(component,'elem_id',None))
    if elem_id in ('txt2img_steps','img2img_steps'):
        _NATIVE_STEPS[elem_id]=component
        mode='img2img' if elem_id=='img2img_steps' else 'txt2img'
        for radio,is_img2img in list(_QUALITY_RADIOS):
            if is_img2img == (mode=='img2img'):
                _bind_quality(radio,component)
    if elem_id!='setting_sd_model_checkpoint' or not PANELS: return
    if component in _BOUND_CHECKPOINTS: return
    from gradio.context import Context
    if Context.root_block is None: return
    panels=list(PANELS)
    def changed(value):
        active='qwen-image-2.1' in str(value).lower() or 'qwen_image_2.1' in str(value).lower()
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
        _visible=bool(forge.selected())
        # ---------------------------------------------------------------- #
        # LAYOUT. Everything used to sit in one flat Advanced accordion, so
        # rarely-needed escape hatches (VRAM overrides, Spectrum) had the
        # same visual weight as the two decisions that matter every run:
        # quality and speed. Now the top level is exactly that one decision
        # plus the output type; everything else lives behind "Advanced"
        # tabs. The RETURNED list below is the script-args contract - its
        # order must not move even though the on-screen position has.
        # ---------------------------------------------------------------- #
        with gr.Accordion('Qwen-Image-2.1',open=False,visible=_visible,elem_classes=['pi-q21-panel']) as box:
            gr.Markdown(_quickstart())
            mode='edit' if is_img2img else 't2i'
            with gr.Row(elem_classes=['pi-q21-output']):
                task=gr.Dropdown([('Standard',mode),('Transparent PNG','rgba')],value=mode,label='Output',info='Transparent PNG adds an alpha channel for cut-outs')
                # Auto upscale: one tick runs the Qwen-Image-2.1 "second pass"
                # (the finished image is re-rendered larger as its own reference).
                upscale_enabled=gr.Checkbox(value=False,label='Auto upscale',info='After generating, repaints the image 1.5x or 2x larger, keeping every detail; takes about as long as the first pass')
                upscale_scale=gr.Dropdown([1.5,2.0],value=1.5,label='Upscale size',info='1.5x is the sweet spot; 2x takes longer and can drift on faces')
            if is_img2img:
                gr.Markdown(_img2img_help(),elem_classes=['pi-q21-status'])
            with gr.Accordion('Technical details',open=False):
                steps=gr.Radio([('Quality · 40 steps',40),('Fast · 25 steps',25)],value=40,label='Quality',info='Fast trades detail for speed; both stay sharp')
                gr.Markdown(_steps_note())
                _QUALITY_RADIOS.append((steps,is_img2img))
                _bind_quality(steps,_NATIVE_STEPS.get('img2img_steps' if is_img2img else 'txt2img_steps'))
            mask=gr.State(None)  # Editing masks come from the native img2img tab.
            # Compact reference slots: 3 small thumbnails per row. Qwen-Image-2.1
            # officially supports up to 10 reference images at once (model card);
            # the main img2img image counts as the first, so 9 slots here = 10 total.
            if is_img2img:
                with gr.Accordion('References (optional)',open=False,elem_classes=['pi-q21-refs']):
                    gr.Markdown(_refs_help())
                    refs=[]
                    for row in range(3):
                        with gr.Row():
                            for col in range(3):
                                refs.append(gr.Image(type='pil',label=f'Ref {row*3+col+2}',height=96,show_download_button=False,container=False,elem_classes=['pi-q21-ref']))
            else:
                refs=[gr.State(None) for _ in range(9)]
            with gr.Tabs(elem_classes=['pi-q21-tools']) as tabs:
                with gr.Tab('Performance'):
                    gr.Markdown('Memory is managed automatically - the settings below are only needed if something goes wrong or you want to squeeze harder.')
                    cfg=gr.State(None)  # Preserve saved argument slots; native CFG is authoritative.
                    with gr.Row():
                        profile=gr.Dropdown(['auto','4','6','8','12','16','20','24'],value='auto',label='VRAM profile (GB)',info='auto detects your GPU; lower only if you hit out-of-memory')
                        side=gr.Dropdown([0,512,768,1024,1536,2048],value=0,label='Maximum size (0 = automatic)',info='Caps the longest image edge to save memory')
                    with gr.Row():
                        offload=gr.Checkbox(value=True,label='Save GPU memory',info='Keeps unused parts off the GPU; leave on for 12 GB cards')
                        community=gr.Checkbox(value=False,label='Allow compatible community distillation LoRAs',info='Off by default: only officially tested files')
                        consent=gr.State(False)  # Generate is always local-only.
                        degrid=gr.Checkbox(value=bool(runtime.config().get('moire_cleanup',True)),label='DeGrid cleanup (removes grid/noise patterns)',info='On by default; uncheck only if outputs look over-smoothed')
                        spectrum=gr.Checkbox(value=False,label='Spectrum speedup (experimental; may change details)',info='Extra acceleration pass; disable if output looks off')
                with gr.Tab('Speed boost'):
                    speed_enabled=gr.Checkbox(value=False,label='Enable Speed boost LoRA (generates in 4-6 steps instead of 40)',info='Pick a downloaded turbo LoRA below; tick this AND select one')
                    speed_choices=['(none)']+list(manager.FEATURED_CHOICES)
                    speed_name=gr.Dropdown(speed_choices,value='(none)',label='Which speed LoRA',info='Shown options are already vetted; download takes one click')
                    speed_strength=gr.Slider(0.0,1.5,value=1.0,step=0.05,label='Speed LoRA strength',info='1.0 is the tested default')
                    gr.Markdown(manager.featured_instructions())
                    speed_approved=gr.Checkbox(value=False,label='I accept the LoRA license and authorize this one-time download')
                    with gr.Row():
                        speed_button=gr.Button('Download selected speed LoRA',size='sm')
                    speed_status=gr.Markdown(elem_classes=['pi-q21-status'])
                    # wiring: one-click download, and grey out the LoRA controls
                    # while the boost is off so the state is obvious.
                    speed_button.click(fn=manager.download_featured,inputs=[speed_name,speed_approved],outputs=[speed_status])
                    speed_enabled.change(fn=lambda on: (gr.update(interactive=on),gr.update(interactive=on)),inputs=[speed_enabled],outputs=[speed_name,speed_strength],queue=False)
                    gr.Markdown(_speed_note())
                with gr.Tab('Prompt helper'):
                    gr.Textbox(value=(ROOT/'prompts/system_prompt_t2i.txt').read_text(encoding='utf8'),label='Official instructions for your language model',lines=4,interactive=False,info='Copy this into any chat model to get a ready-to-paste prompt')
                    gr.Markdown(_prompt_helper_note())
                with gr.Tab('Model Setup'):
                    gr.Markdown(_models_help())
                    method=gr.Radio(['Manual (recommended)','Automatic'],value='Manual (recommended)',label='How to get models',info='Manual links are verifiable; Automatic downloads in one click after approval')
                    manual=gr.Markdown(manager.manual_instructions())
                    with gr.Column(visible=False) as automatic:
                        # NO SILENT DOWNLOADS: nothing fetches until you tick the
                        # license box AND press the button. One DiT + one encoder
                        # + the VAE is all a preset needs.
                        downloads=gr.CheckboxGroup(manager.CHOICES,value=['qwen_image_2.1_bf16.safetensors','qwen3vl_8b_bf16.safetensors','qwen_image_2.1_vae_bf16.safetensors',manager.SUPPORT],label='Select files to download (choose one DiT, one encoder and the VAE)',info='Each file is checksum-verified after download')
                        approved=gr.Checkbox(value=False,label='I accept LICENSE and authorize only these downloads')
                        with gr.Row():
                            button=gr.Button('Download selected models',size='sm')
                        status=gr.Markdown(elem_classes=['pi-q21-status'])
                        button.click(fn=manager.download_selected,inputs=[downloads,approved],outputs=[status])
                    # wiring: Manual shows the link sheet, Automatic shows the downloader.
                    method.change(fn=lambda value:(gr.update(visible=value.startswith('Manual')),gr.update(visible=value=='Automatic')),inputs=[method],outputs=[manual,automatic],queue=False)
        self._box=box
        PANELS.append(box)
        # MUST stay in this exact order - lib/forge.py reads these by index:
        # task, steps, cfg, profile, side, offload, community, consent, mask,
        # refs[0..8], spectrum, degrid, speed_enabled, speed_name, speed_strength,
        # upscale_enabled, upscale_scale. New controls append at the end only.
        return [task,steps,cfg,profile,side,offload,community,consent,mask,*refs,spectrum,degrid,speed_enabled,speed_name,speed_strength,upscale_enabled,upscale_scale]

# --------------------------------------------------------------------------- #
# boot: install the generation hooks and clean up on extension unload
# --------------------------------------------------------------------------- #
try:
    forge.install()
    preset.install()
except (ImportError,AttributeError,TypeError,RuntimeError) as exc:
    print(TAG,'Neo hook drift; adapter disabled:',exc)
script_callbacks.on_script_unloaded(runtime.release)