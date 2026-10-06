"""Build the real UI class in isolation, without loading Forge or any model."""
import ast
import importlib.util
import sys
import types
import weakref
from pathlib import Path
import gradio as gr

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('qwen_preview',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
from qwen_preview.download import manager
from qwen_preview.lib.controls import speed_updates,quality_updates
from qwen_preview.lib import workflows
from qwen_preview.lib import comfy_setup

def build():
    tree=ast.parse((ROOT/'scripts/engine.py').read_text(encoding='utf-8'))
    names={'Script','_bind_quality','_bind_workflows','_workflow_updates','_bind_pad_previews','_pad_preview','_img2img_help','_models_help','_refs_help'}
    tree.body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    ns=dict(gr=gr,forge=types.SimpleNamespace(selected=lambda:True),scripts=types.SimpleNamespace(Script=object,AlwaysVisible=True),
            runtime=types.SimpleNamespace(config=lambda:{'moire_cleanup':True}),manager=manager,
            speed_updates=speed_updates,quality_updates=quality_updates,
            workflows=workflows,comfy_setup=comfy_setup,PANELS=[],_QUALITY_RADIOS=[],_NATIVE_STEPS={},_BOUND_QUALITY=weakref.WeakSet(),
            _NATIVE_WORKFLOW={},_WORKFLOW_BINDINGS=[],_BOUND_WORKFLOWS=weakref.WeakSet(),
            _NATIVE_SOURCE={},_PAD_PREVIEWS=[],_BOUND_PAD_PREVIEWS=weakref.WeakSet())
    exec(compile(tree,str(ROOT/'scripts/engine.py'),'exec'),ns)
    with gr.Blocks(css=(ROOT/'style.css').read_text(encoding='utf-8'),analytics_enabled=False) as app:
        controls=[]
        with gr.Tabs():
            for edit,label in ((False,'Text to image'),(True,'Image to image')):
                with gr.Tab(label):
                    native=gr.Slider(1,80,value=40,step=1,label='Native sampling steps',interactive=True)
                    ns['_NATIVE_STEPS']['img2img_steps' if edit else 'txt2img_steps']=native
                    mode='img2img' if edit else 'txt2img'
                    ns['_NATIVE_WORKFLOW'][mode+'_steps']=native
                    for key,value in (('cfg_scale',1),('width',1024),('height',1024)):
                        ns['_NATIVE_WORKFLOW'][mode+'_'+key]=gr.Number(value=value,label='Native '+key)
                    script=ns['Script']();values=script.ui(edit);script._box.open=True
                    gr.Button('Validate controls',visible=False).click(lambda *values: None,inputs=values,outputs=[])
                    controls.append(values)
    for edit,values in zip((False,True),controls):
        assert len(values)==60
        assert values[0].value==('edit' if edit else 't2i')
        assert values[3].value=='auto' and values[5].value is True
        assert values[18].label=='Spectrum acceleration'
        assert values[23].label=='Refine details'
        assert isinstance(values[26],gr.Checkbox if edit else gr.State)
        assert isinstance(values[28],gr.Checkbox if edit else gr.State)
        assert all(isinstance(x,gr.Image if edit else gr.State) for x in values[9:18])
        assert values[35].value=='Custom' and values[44].value is True
        assert len(ns['_workflow_updates']('Custom',edit,''))==23
        recipe='UltraOutpaint25' if edit else 'ITLText25'
        updates=ns['_workflow_updates'](recipe,edit,'turbo')
        assert updates[19]['value']==25 and updates[20]['value']==1
        if edit: assert updates[7]['value'] is True and updates[8]['value']==256
    print('Both real Gradio panels built; all 60 control slots and workflow callbacks verified.',flush=True)
    return app

if __name__=='__main__':
    app=build()
    if '--serve' in sys.argv:
        app.launch(server_name='127.0.0.1',server_port=7869,inbrowser=False,show_api=False)
