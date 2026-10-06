import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from test_registration import load_forge

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('workflow_prompts',ROOT/'lib/prompts.py')
# Reuse the package loader because prompts imports the extension assets.
class WorkflowIntegrationTests(unittest.TestCase):
    def test_all_appended_positions_preserve_legacy_controls(self):
        forge=load_forge()
        guide=object()
        values=['edit',40,1,'auto',0,True,False,False,None]+[None]*9
        values += [False,True,False,'(none)',1.,'off','(none)',False,False,True,False,2,'(none)',False,0,False,'']
        values += ['UltraOutpaintControl25','off',True,256.,0,32,0,8,1.,False,True,'comfy','backend',True,'union',guide,1.5,.1,.9,'res_2s','beta',True,'pe']
        self.assertEqual(len(values),58)
        result=forge.options(NS(alwayson_scripts=[NS(_pi_qwen21=True,args_from=0,args_to=58)]),NS(script_args=values))
        for key,value in dict(workflow='UltraOutpaintControl25',backend='comfy',outpaint=True,pad_left=256.,pad_right=0,
            pad_top=32,pad_bottom=0,pad_overlap=8,reference_mp=1.,kv_cache=False,follow_source=True,control_enabled=True,
            control_model='union',control_strength=1.5,control_start=.1,control_end=.9,comfy_sampler='res_2s',
            comfy_scheduler='beta',rewrite_prompt=True,prompt_encoder='pe').items():
            self.assertEqual(result[key],value)
        self.assertIs(result['control_guide'],guide)
        self.assertEqual(result['refs'],[None]*9)
        legacy=forge.options(NS(alwayson_scripts=[NS(_pi_qwen21=True,args_from=0,args_to=35)]),NS(script_args=values[:35]))
        self.assertNotIn('backend',legacy)
        self.assertEqual(legacy['sampler_sharp'],0)

    def test_prompt_ratio_follow_uses_connected_image_dimensions(self):
        load_forge()
        from pi_qwen21.lib.prompts import parse
        from PIL import Image
        images=[Image.new('RGB',(64,96)),Image.new('RGB',(128,64))]
        text,w,h=parse('{"rewritten_prompt":"new scene","wh_ratio":"","ratio_follow":"<image2>"}',512,512,images)
        self.assertEqual((text,w,h),('new scene',128,64))
        for text in ['{"rewritten_prompt":"x","ratio_follow":"<image3>"}',
                     '{"rewritten_prompt":"x","wh_ratio":"1:1","ratio_follow":"<image1>"}',
                     '{"rewritten_prompt":"x","wh_ratio":"NaN:1"}']:
            with self.assertRaises(ValueError): parse(text,512,512,images)

    def test_turbo_six_step_mode_arms_adapter_and_regular_workflow_does_not(self):
        load_forge()
        from pi_qwen21.lib.controls import quality_updates
        self.assertTrue(quality_updates(6,'Turbo')[0]['value'])
        self.assertFalse(quality_updates(25,'Turbo')[0]['value'])
