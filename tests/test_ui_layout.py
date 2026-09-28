import unittest
import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import gradio as gr
from ui_fixture import build

class UILayoutTests(unittest.TestCase):
    def test_real_gradio_panels_preserve_control_contract(self):
        app=build()
        self.assertGreater(len(app.config["components"]),100)

    @unittest.skipUnless(os.environ.get('FORGE_ROOT'),'requires the installed Forge UI loader')
    def test_real_forge_saved_defaults_do_not_poison_turbo_slider(self):
        path=Path(os.environ['FORGE_ROOT'])/'modules/ui_loadsave.py'
        stub=NS(InputAccordionImpl=type('InputAccordionImpl',(),{}),ToolButton=type('ToolButton',(),{}))
        with patch.dict('sys.modules',{'modules':NS(errors=NS(display=lambda *a:None)), 'modules.ui_components':stub}):
            spec=importlib.util.spec_from_file_location('qwen_test_ui_loadsave',path)
            loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
        app=build()
        sliders=[x for x in app.blocks.values() if isinstance(x,gr.Slider) and x.label=='Qwen Turbo strength']
        self.assertEqual(len(sliders),2)
        for mode,slider in zip(('txt2img','img2img'),sliders):
            saved=loader.UiLoadsave.__new__(loader.UiLoadsave)
            saved.finalized_ui=False;saved.component_mapping={}
            prefix=f'customscript/engine.py/{mode}/Strength/'
            saved.ui_settings={prefix+'value':-0.25,prefix+'minimum':-1.0,prefix+'maximum':0.25}
            old=gr.Slider(0,1.5,value=1,label='Strength');old.custom_script_source='engine.py'
            saved.add_component(f'{mode}/Strength',old)
            self.assertEqual(old.value,-0.25)  # Reproduce the reported failure.
            slider.custom_script_source='engine.py'
            saved.add_component(f'{mode}/{slider.label}',slider)
            self.assertEqual((slider.value,slider.minimum,slider.maximum),(1.0,0.0,1.5))
