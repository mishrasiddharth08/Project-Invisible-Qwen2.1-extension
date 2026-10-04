import sys
import types
import unittest
from unittest.mock import patch
from test_registration import load_forge


class NativeCoexistenceTests(unittest.TestCase):
    def choose(self,preset='qwen21',native=True,marker=False,override=None):
        forge=load_forge()
        path='models/qwen_image_2.1_bf16.safetensors'
        ci=types.SimpleNamespace(filename=path,_pi_qwen21=marker)
        models=types.SimpleNamespace(checkpoints_list={'raw':ci},checkpoint_aliases={})
        shared=types.SimpleNamespace(opts=types.SimpleNamespace(sd_model_checkpoint='raw',forge_preset=preset))
        arch=types.SimpleNamespace(**({'qwen21':13} if native else {}))
        modules={'modules':types.SimpleNamespace(shared=shared,sd_models=models),
                 'modules_forge':types.SimpleNamespace(presets=types.SimpleNamespace(PresetArch=arch))}
        with patch.dict(sys.modules,modules):
            return forge.selected(types.SimpleNamespace(override_settings=override or {}))

    def test_native_preset_leaves_raw_checkpoint_to_forge(self):
        self.assertIsNone(self.choose())

    def test_extension_marker_remains_extension_owned(self):
        self.assertIsNotNone(self.choose(marker=True))

    def test_extension_preset_keeps_existing_routing(self):
        self.assertIsNotNone(self.choose(preset='qwen-image-2.1'))

    def test_unavailable_native_engine_keeps_existing_routing(self):
        self.assertIsNotNone(self.choose(native=False))

    def test_api_preset_override_controls_route(self):
        self.assertIsNone(self.choose(preset='qwen-image-2.1',override={'forge_preset':'qwen21'}))
        self.assertIsNotNone(self.choose(override={'forge_preset':'qwen-image-2.1'}))
