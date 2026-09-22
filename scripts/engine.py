"""Always-on controls inside existing txt2img/img2img; no extra tab."""
import importlib.util
import sys
import weakref
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if 'pi_qwen21' not in sys.modules:
    spec=importlib.util.spec_from_file_location('pi_qwen21',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
    package=importlib.util.module_from_spec(spec); sys.modules['pi_qwen21']=package; spec.loader.exec_module(package)

import gradio as gr
from modules import scripts,script_callbacks
from pi_qwen21.lib import forge
from pi_qwen21.lib import runtime
from pi_qwen21.lib import preset
from pi_qwen21.download import manager

PANELS=[]
_BOUND_CHECKPOINTS=weakref.WeakSet()
_BOUND_QUALITY=weakref.WeakSet()
_NATIVE_STEPS={}
_QUALITY_RADIOS=[]

def _bind_quality(radio, native):
    if native is None or radio in _BOUND_QUALITY: return
    from gradio.context import Context
    if Context.root_block is None: return
    radio.change(fn=lambda value: gr.update(value=int(value)),inputs=[radio],outputs=[native],queue=False)
    _BOUND_QUALITY.add(radio)

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

class Script(scripts.Script):
    _pi_qwen21=True
    def title(self): return 'PROJECT INVISIBLE — Qwen-Image-2.1'
    def show(self,is_img2img): return scripts.AlwaysVisible
    def ui(self,is_img2img):
        with gr.Accordion('Qwen-Image-2.1 Controls',open=False,visible=bool(forge.selected())) as box:
            mode='edit' if is_img2img else 't2i'
            task=gr.Dropdown([('Standard',mode),('Transparent PNG','rgba')],value=mode,label='Output')
            degrid=gr.Checkbox(value=bool(runtime.config().get('moire_cleanup',True)),label='DeGrid: remove grid/noise automatically (uncheck to bypass)')
            steps=gr.Radio([('Quality · 40 steps',40),('Fast · 25 steps',25)],value=40,label='Quality')
            gr.Markdown('Sampling Steps in the native txt2img/img2img controls the generation length.')
            _QUALITY_RADIOS.append((steps,is_img2img))
            _bind_quality(steps,_NATIVE_STEPS.get('img2img_steps' if is_img2img else 'txt2img_steps'))
            mask=gr.State(None)  # Editing masks come from the native img2img tab.
            if is_img2img:
                with gr.Accordion('Extra reference images (optional)',open=False):
                    gr.Markdown('Upload the main image in img2img. Add up to 9 references here.')
                    refs=[gr.Image(type='pil',label=f'Reference {i+2}') for i in range(9)]
            else:
                refs=[gr.State(None) for _ in range(9)]
            with gr.Accordion('Advanced',open=False):
                cfg=gr.State(None)  # Preserve saved argument slots; native CFG is authoritative.
                profile=gr.Dropdown(['auto','4','6','8','12','16','20','24'],value='auto',label='VRAM profile (GB)')
                side=gr.Dropdown([0,512,768,1024,1536,2048],value=0,label='Maximum size (0 = automatic)')
                offload=gr.Checkbox(value=True,label='Save GPU memory')
                spectrum=gr.Checkbox(value=False,label='Spectrum speedup (experimental; may change details)')
                community=gr.Checkbox(value=False,label='Allow compatible community distillation LoRAs')
                consent=gr.State(False)  # Generate is always local-only.
                with gr.Accordion('Prompt helper',open=False):
                    gr.Textbox(value=(ROOT/'prompts/system_prompt_t2i.txt').read_text(encoding='utf8'),label='Official instructions for your language model',lines=4,interactive=False)
                    gr.Markdown('Paste the resulting JSON into the normal prompt field.')
            with gr.Accordion('Models',open=False):
                method=gr.Radio(['Manual (recommended)','Automatic'],value='Manual (recommended)',label='How to get models')
                manual=gr.Markdown(manager.manual_instructions())
                with gr.Column(visible=False) as automatic:
                    downloads=gr.CheckboxGroup(manager.CHOICES,value=['qwen_image_2.1_bf16.safetensors','qwen3vl_8b_bf16.safetensors','qwen_image_2.1_vae_bf16.safetensors',manager.SUPPORT],label='Select files to download (choose one DiT, one encoder and the VAE)')
                    approved=gr.Checkbox(value=False,label='I accept LICENSE and authorize only these downloads')
                    button=gr.Button('Download selected models')
                    status=gr.Markdown()
                    button.click(fn=manager.download_selected,inputs=[downloads,approved],outputs=[status])
                method.change(fn=lambda value:(gr.update(visible=value.startswith('Manual')),gr.update(visible=value=='Automatic')),inputs=[method],outputs=[manual,automatic],queue=False)
        self._box=box
        PANELS.append(box)
        return [task,steps,cfg,profile,side,offload,community,consent,mask,*refs,spectrum,degrid]

try:
    forge.install()
    preset.install()
except (ImportError,AttributeError,TypeError,RuntimeError) as exc:
    print('[PI-Qwen21] Neo hook drift; adapter disabled:',exc)
script_callbacks.on_script_unloaded(runtime.release)
