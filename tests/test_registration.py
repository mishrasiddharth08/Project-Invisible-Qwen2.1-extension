import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_forge():
    for name in tuple(sys.modules):
        if name == "pi_qwen21" or name.startswith("pi_qwen21."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "pi_qwen21", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["pi_qwen21"] = package
    spec.loader.exec_module(package)
    return __import__("pi_qwen21.lib.forge", fromlist=["forge"])


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.forge = load_forge()
        self.paths = {"pipelines": [], "dit": [], "te": [], "vae": [], "lora": []}

        class CheckpointInfo:
            def register(info):
                self.models.checkpoints_list[info.name] = info

        self.models = types.SimpleNamespace(
            CheckpointInfo=CheckpointInfo,
            checkpoints_list={},
        )
        self.root = Path(tempfile.mkdtemp())

    def register(self, inventory=None):
        with mock.patch.dict(sys.modules, {"modules": types.SimpleNamespace(sd_models=self.models)}), \
             mock.patch.object(self.forge, "scan", return_value=inventory or self.paths), \
             mock.patch.object(self.forge, "models_root", return_value=self.root):
            self.forge.register()

    def test_identical_registration_is_idempotent_and_preserves_other_entry(self):
        path = str(self.root / "qwen_image_2.1_int8_convrot.safetensors")
        other = types.SimpleNamespace(filename=path, name="other-engine", _pi_qwen21=False)
        self.models.checkpoints_list["other-engine"] = other
        inventory = dict(self.paths, dit=[path])
        self.register(inventory)
        first = self.models.checkpoints_list["Qwen-Image-2.1 — qwen_image_2.1_int8_convrot.safetensors"]
        self.register(inventory)
        second = self.models.checkpoints_list["Qwen-Image-2.1 — qwen_image_2.1_int8_convrot.safetensors"]
        self.assertIs(first, second)
        self.assertIs(self.models.checkpoints_list["other-engine"], other)

    def test_placeholder_uses_existing_support_file_for_gallery(self):
        self.register()
        marker=self.models.checkpoints_list[self.forge.LABEL]
        self.assertTrue(Path(marker.filename).is_file())
        self.assertTrue(Path(marker.filename).parent.is_dir())

    def test_changed_alias_replaces_only_old_qwen_entry(self):
        path = str(self.root / "qwen_image_2.1_int8_convrot.safetensors")
        old = types.SimpleNamespace(filename=path, name="old", _pi_qwen21=True)
        other = types.SimpleNamespace(filename=path, name="other-engine", _pi_qwen21=False)
        self.models.checkpoints_list.update(old=old, **{"other-engine": other})
        self.register(dict(self.paths, dit=[path]))
        self.assertNotIn("old", self.models.checkpoints_list)
        self.assertIn("other-engine", self.models.checkpoints_list)
        self.assertIn("Qwen-Image-2.1 — qwen_image_2.1_int8_convrot.safetensors", self.models.checkpoints_list)


if __name__ == "__main__":
    unittest.main()

class SpectrumOptionsTests(unittest.TestCase):
    def test_keep_loaded_is_read_from_appended_ui_control(self):
        forge=load_forge()
        refs=[None]*9
        values=['t2i',40,1,'auto',0,True,False,False,None]+refs+[False,True,False,'(none)',1.0,'off','(none)',False,False,True]
        script=types.SimpleNamespace(_pi_qwen21=True,args_from=0,args_to=len(values))
        result=forge.options(types.SimpleNamespace(alwayson_scripts=[script]),types.SimpleNamespace(script_args=values))
        self.assertTrue(result['keep_loaded'])

    def test_extra_checkbox_does_not_become_reference_image(self):
        forge=load_forge()
        refs=[object() for _ in range(9)]
        values=['edit',40,1,'auto',0,True,False,False,None]+refs+[True,False]
        script=types.SimpleNamespace(_pi_qwen21=True,args_from=0,args_to=len(values))
        result=forge.options(types.SimpleNamespace(alwayson_scripts=[script]),types.SimpleNamespace(script_args=values))
        self.assertEqual(result['refs'],refs)
        self.assertIs(result['spectrum'],True)
        self.assertIs(result['moire_cleanup'],False)

    def test_builtin_spectrum_accordion_enables_qwen_acceleration(self):
        forge=load_forge()
        refs=[object() for _ in range(9)]
        qwen=['edit',40,1,'auto',0,True,False,False,None]+refs+[False,False]
        # Forge stores every script's values in one shared list; each script
        # slices its own range. The built-in Spectrum occupies the first 8.
        script=types.SimpleNamespace(_pi_qwen21=True,args_from=8,args_to=8+len(qwen))
        builtin=types.SimpleNamespace(title=lambda:'Spectrum Integrated',
            args_from=0,args_to=8)
        runner=types.SimpleNamespace(alwayson_scripts=[script,builtin])
        result=forge.options(runner,types.SimpleNamespace(script_args=[True,0.25,6,0.5,2,0.0,6,0.9]+qwen))
        self.assertIs(result['spectrum'],True)
        result=forge.options(runner,types.SimpleNamespace(script_args=[False,0.25,6,0.5,2,0.0,6,0.9]+qwen))
        self.assertIs(result['spectrum'],False)
