import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_qwen_workflows", ROOT / "lib/workflows.py")
workflows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflows)


class WorkflowRecipeTests(unittest.TestCase):
    def test_catalog_and_tab_choices(self):
        expected = {
            "Custom", "ITLText25", "ITLEdit25", "ITLPose25", "UltraEdit25",
            "UltraInpaint25", "UltraOutpaint25", "UltraTurboEdit6",
            "UltraTurboInpaint6", "UltraTurboOutpaint6", "UltraTextAdapted50",
            "UltraTurboTextAdapted6",
            'UltraTextFull50','UltraEditControl25','UltraInpaintControl25','UltraOutpaintControl25','UltraTurboControl6',
            'UltraMergedTurboText6','UltraMergedTurboEdit6','UltraMergedTurboInpaint6','UltraMergedTurboOutpaint6',
        }
        self.assertEqual(set(workflows.CATALOG), expected)
        self.assertEqual(workflows.choices(False),
                         ["Custom", "ITLText25", "UltraTextAdapted50", "UltraTurboTextAdapted6",'UltraMergedTurboText6','UltraTextFull50'])
        self.assertNotIn("ITLText25", workflows.choices(True))
        self.assertEqual(len(workflows.choices(True)), 16)

    def test_text_recipe_has_dimensions_but_edit_recipes_do_not(self):
        text = workflows.settings("ITLText25", False)
        self.assertEqual((text["steps"], text["cfg"], text["width"], text["height"]),
                         (25, 1.0, 992, 544))
        for name in workflows.choices(True):
            edit = workflows.settings(name, True)
            self.assertNotIn("width", edit)
            self.assertNotIn("height", edit)

    def test_text_adaptations_are_labeled_and_use_supported_defaults(self):
        quality = workflows.settings("UltraTextAdapted50", False)
        self.assertEqual((quality["steps"], quality["width"], quality["height"]),
                         (50, 1920, 1088))
        self.assertIn("instead of res_2s/Beta", quality["adaptation_note"])
        turbo = workflows.settings("UltraTurboTextAdapted6", False)
        self.assertTrue(turbo["turbo"])
        self.assertIn("not the merged", turbo["adaptation_note"])

    def test_pose_lanpaint_turbo_and_outpaint_defaults(self):
        self.assertEqual(workflows.settings("ITLPose25", True)["pose"], "OpenPose")
        inpaint = workflows.settings("UltraTurboInpaint6", True)
        self.assertEqual(inpaint["steps"], 6)
        self.assertTrue(inpaint["turbo"])
        self.assertTrue(inpaint["lanpaint"])
        self.assertEqual(inpaint["task"], "edit")
        self.assertEqual(inpaint["operation"], "inpaint")
        outpaint = workflows.settings("UltraOutpaint25", True)
        self.assertEqual(outpaint["pads"], (256, 256, 256, 256))
        self.assertTrue(outpaint["outpaint"])

    def test_custom_is_noop_and_results_are_copies(self):
        self.assertEqual(workflows.settings("Custom", False), {})
        first = workflows.settings("UltraOutpaint25", True)
        first["pads"] = (9, 9, 9, 9)
        self.assertEqual(workflows.settings("UltraOutpaint25", True)["pads"][0], 256)

    def test_wrong_tab_unknown_and_unsupported_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "txt2img"):
            workflows.settings("ITLEdit25", False)
        with self.assertRaisesRegex(ValueError, "img2img"):
            workflows.settings("ITLText25", True)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            workflows.settings("MadeUp", False)
        for name in workflows.UNSUPPORTED:
            with self.assertRaisesRegex(ValueError, "unsupported"):
                workflows.settings(name, False)
        self.assertNotIn("UltraText", workflows.CATALOG)

    def test_audit_reads_inert_json_and_marks_unsupported_features(self):
        raw = json.dumps({"nodes": [
            {"type": "ControlNetLoader", "mode": 0,
             "widgets_values": ["FunControlNetUnion.safetensors"]},
            {"class_type": "KSampler", "inputs": {
                "sampler_name": "res_2s", "scheduler": "beta",
                "ckpt_name": "Viggle-Turbo-int8.safetensors"}},
            {"type": "OpenPosePreprocessor"},
        ]})
        report = workflows.audit(raw)
        self.assertEqual(report["node_count"], 3)
        self.assertIn("KSampler", report["node_types"])
        self.assertEqual(report["modes"], ["0"])
        self.assertTrue(all(report["unsupported"].values()))
        self.assertIn("res_2s", report["samplers"])

    def test_audit_rejects_invalid_or_excessive_input(self):
        with self.assertRaisesRegex(ValueError, "Invalid"):
            workflows.audit("not json")
        with self.assertRaisesRegex(ValueError, "limit"):
            workflows.audit("[]" * 10, limit=4)
        with self.assertRaises(TypeError):
            workflows.audit(4)

    def test_provenance_names_intentional_gaps(self):
        report = workflows.provenance()
        self.assertEqual(report["execution"], "JSON is parsed as inert data only")
        self.assertEqual(set(report["unsupported"]),
                         {"UltraText", "MergedTurbo", "ControlNetUnion"})


if __name__ == "__main__":
    unittest.main()
