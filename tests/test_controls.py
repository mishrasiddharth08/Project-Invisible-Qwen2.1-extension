import importlib.util
from pathlib import Path
import unittest
from test_registration import load_forge

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_qwen_controls',ROOT/'lib/controls.py')
controls=importlib.util.module_from_spec(spec); spec.loader.exec_module(controls)

class ControlTests(unittest.TestCase):
    def test_toggle_off_never_puts_skip_dict_into_slider_value(self):
        for native in (False,True):
            updates=controls.speed_updates(False,'Turbo',native)
            self.assertEqual(len(updates),3 if native else 2)
            for update in updates:
                self.assertEqual(update['__type__'],'update')
                self.assertNotIn('value',update)

    def test_toggle_on_uses_scalar_values_and_quality_preserves_choice(self):
        self.assertEqual([x['value'] for x in controls.speed_updates(True,'Turbo',True)],['Turbo',1.0,6])
        slow=controls.quality_updates(40,'Turbo')
        self.assertFalse(slow[0]['value']); self.assertNotIn('value',slow[1])

    def test_previous_nested_skip_request_recovers_default_strength(self):
        self.assertEqual(controls.speed_strength({'__type__':'update'}),1.0)
        self.assertEqual(controls.speed_strength({'__type__':'update','value':{'__type__':'update'}}),1.0)
        self.assertEqual(controls.speed_strength({'__type__':'update','value':0.7}),0.7)
        for invalid in ({'image':'wrong field'},float('nan'),float('inf'),-1,2):
            with self.assertRaises(ValueError): controls.speed_strength(invalid)

    def test_options_decode_toggle_off_then_retry_without_crashing(self):
        from types import SimpleNamespace as NS
        forge=load_forge()
        values=['edit',6,1,'auto',0,True,False,False,None]+[None]*9+[False,True,False,{'__type__':'update'},{'__type__':'update'},'off']
        script=NS(_pi_qwen21=True,args_from=0,args_to=len(values))
        result=forge.options(NS(alwayson_scripts=[script]),NS(script_args=values))
        self.assertFalse(result['speed_enabled']); self.assertEqual(result['speed_strength'],1.0)
        self.assertEqual(result['speed_lora'],'')
        for length in (21,22):
            result=forge.options(NS(alwayson_scripts=[script]),NS(script_args=values[:length]))
            self.assertEqual(result['speed_strength'],1.0)

if __name__=='__main__': unittest.main()
