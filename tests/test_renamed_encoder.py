import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import test_acceptance_cpu as fixtures

assets=fixtures.assets;manager=fixtures.manager

def keys(layers=36,width=4096):
    result={'visual.patch_embed.proj.weight':{'shape':[1152,3,2,14,14]}}
    for i in range(layers):
        result[f'model.layers.{i}.input_layernorm.weight']={'shape':[width]}
        result[f'model.layers.{i}.self_attn.q_proj.weight']={'shape':[width,width]}
    return result

class RenamedEncoderTests(unittest.TestCase):
    def test_architecture_checks_reject_wrong_layer_count_width_and_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'renamed.safetensors'
            for data,expected in [(keys(),True),(keys(32),False),(keys(width=5120),False),
                                  ({**keys(),'adapter.lora_A.weight':{'shape':[4,4]}},False),
                                  ({k:v for k,v in keys().items() if not k.startswith('visual.')},False)]:
                fixtures.write_safetensors(p,data)
                self.assertEqual(assets.is_text_encoder_file(p),expected)

    def test_recursive_scan_finds_compatible_renamed_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);p=base/'text_encoder'/'nested'/'shared-int8_convrot.safetensors'
            p.parent.mkdir(parents=True);fixtures.write_safetensors(p,keys())
            with patch.object(assets,'complete',return_value=False):
                inventory=assets.scan(base,refresh=True)
            self.assertIn(str(p.resolve()),inventory['te'])

    def test_resolve_reuses_renamed_encoder_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for name in ('processor/tokenizer.json','scheduler/scheduler_config.json'):
                p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            inventory={'pipelines':[],'dit':['/local/qwen_image_2.1_int8_convrot.safetensors'],
                       'te':['/local/compatible-int8_convrot.safetensors'],
                       'vae':['/local/qwen_image_2.1_vae_bf16.safetensors'],'lora':[]}
            with patch.object(manager,'scan',return_value=inventory),patch.object(manager,'support_folder',return_value=folder),patch.object(manager,'identity',return_value=True):
                result=manager.resolve(None,False,{'dit':'int8_convrot','te':'w4a8'})
            self.assertEqual(result['files']['text_encoder'],inventory['te'][0])
