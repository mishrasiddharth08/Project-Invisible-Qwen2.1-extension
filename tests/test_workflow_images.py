import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from test_acceptance_cpu import load_package

load_package()
from pi_qwen21.lib.workflow_images import fit_reference, prepare_outpaint, pose_reference


class WorkflowImageTests(unittest.TestCase):
    def test_outpaint_preserves_source_rgba_and_builds_mask(self):
        source = Image.new("RGBA", (64, 32), (10, 20, 30, 40))
        source.putpixel((7, 9), (1, 2, 3, 4))
        canvas, mask = prepare_outpaint(source, 32, 64, 0, 32)
        self.assertEqual(canvas.size, (160, 64))
        self.assertEqual(canvas.crop((32, 0, 96, 32)).tobytes(), source.tobytes())
        self.assertEqual(canvas.getpixel((0, 0)), (128, 128, 128, 255))
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.getpixel((40, 10)), 0)
        self.assertEqual(mask.getpixel((0, 10)), 255)
        self.assertEqual(mask.getpixel((40, 50)), 255)

    def test_outpaint_overlap_only_enters_adjacent_source_edges(self):
        _, mask = prepare_outpaint(Image.new("RGB", (64, 64)), 32, 0, 32, 0, overlap=8)
        self.assertEqual(mask.getpixel((32, 40)), 255)
        self.assertEqual(mask.getpixel((50, 32)), 255)
        self.assertEqual(mask.getpixel((50, 50)), 0)
        self.assertEqual(mask.getpixel((95, 50)), 0)
        self.assertGreater(mask.getpixel((33,50)),mask.getpixel((38,50)))
        self.assertGreater(mask.getpixel((38,50)),0)

    def test_outpaint_rejects_invalid_and_unsafe_sizes(self):
        image = Image.new("RGB", (32, 32))
        for pads in ((0, 0, 0, 0), (1, 0, 0, 0), (-32, 0, 0, 0)):
            with self.subTest(pads=pads), self.assertRaises(ValueError):
                prepare_outpaint(image, *pads)
        with self.assertRaises(ValueError):
            prepare_outpaint(image, 8192, 0, 0, 0)
        canvas, _ = prepare_outpaint(image, 32.0, 0.0, 0.0, 0.0)
        self.assertEqual(canvas.size, (64, 32))
        for value in (True, 31.5, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_outpaint(image, value, 0, 0, 0)

    def test_fit_reference_downscales_snaps_and_preserves_alpha(self):
        image = Image.new("RGBA", (2000, 1000), (1, 2, 3, 77))
        result = fit_reference(image, 1.0, 32)
        self.assertLess(result.width, image.width)
        self.assertLess(result.height, image.height)
        self.assertEqual(result.width % 32, 0)
        self.assertEqual(result.height % 32, 0)
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getchannel("A").getextrema(), (77, 77))

    def test_fit_reference_never_upscales_and_validates_budget(self):
        image = Image.new("LA", (31, 17), (9, 33))
        result = fit_reference(image, 1.0)
        self.assertEqual(result.size, image.size)
        self.assertEqual(result.mode, "RGBA")
        for value in (0, -1, math.nan, math.inf, 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fit_reference(image, value)

    def test_uploaded_pose_map_is_rgba_copy(self):
        image = Image.new("P", (8, 9))
        result = pose_reference(image)
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, image.size)
        self.assertIsNot(result, image)

    def pose_modules(self, model_dir, result):
        calls = []
        class Processor:
            unloads = 0
            def __call__(self, image, resolution):
                calls.append((image.shape, resolution))
                return result
            def unload_function(self):
                self.unloads += 1
        processor = Processor()
        annotator = types.ModuleType("annotator.openpose")
        annotator.OpenposeDetector = type("OpenposeDetector", (), {"model_dir": str(model_dir)})
        controlnet = types.ModuleType("lib_controlnet")
        controlnet.global_state = types.SimpleNamespace(get_preprocessor=lambda name: processor)
        return {"annotator.openpose": annotator, "lib_controlnet": controlnet}, processor, calls

    def test_pose_preprocessor_normalizes_array_and_dict_and_unloads(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            for name in ("body_pose_model.pth", "hand_pose_model.pth", "facenet.pth"):
                (Path(folder) / name).touch()
            for payload in (np.zeros((5, 6, 3), dtype=np.uint8),
                            {"image": np.zeros((5, 6, 3), dtype=np.uint8)}):
                modules, processor, calls = self.pose_modules(folder, payload)
                with patch.dict(sys.modules, modules):
                    output = pose_reference(Image.new("RGB", (6, 5)), "OpenPose", 512.0)
                self.assertEqual(output.size, (6, 5))
                self.assertEqual(output.mode, "RGBA")
                self.assertEqual(len(calls), 1)
                self.assertEqual(processor.unloads, 1)

    def test_pose_missing_any_asset_never_calls_preprocessor(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "body_pose_model.pth").touch()
            (Path(folder) / "hand_pose_model.pth").touch()
            modules, processor, calls = self.pose_modules(folder, np.zeros((2, 2, 3), dtype=np.uint8))
            with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "facenet.pth"):
                pose_reference(Image.new("RGB", (2, 2)), "OpenPose")
            self.assertEqual(calls, [])
            self.assertEqual(processor.unloads, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
