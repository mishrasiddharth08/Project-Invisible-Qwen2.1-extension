import importlib
import math
import types
import unittest

import torch

from test_acceptance_cpu import load_package
from test_registration import load_forge


load_package()
enh = importlib.import_module("pi_qwen21.lib.enhancer")


class ReferenceParameterTests(unittest.TestCase):
    def test_every_reference_index_and_boundary_weight(self):
        for index in range(1, 11):
            for weight in (0, 0.25, 1, 1.2, 8):
                with self.subTest(index=index, weight=weight):
                    result = enh.references(f"{index}:{weight}")
                    expected = {} if weight == 1 else {index: weight}
                    self.assertEqual(result, expected)

    def test_duplicates_use_last_value_and_one_disables(self):
        self.assertEqual(enh.references("1:.25, 1:1.2; 1:8"), {1: 8})
        self.assertEqual(enh.references("2:8, 2:1"), {})

    def test_reference_range_and_number_validation(self):
        for value in ("0:1.2", "11:1.2", "1:-.1", "1:8.01", "1:nan", "1:inf", "oops"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                enh.references(value)

    def test_valid_index_that_is_missing_from_request_fails(self):
        mask = torch.tensor([[False, True, False, True]])
        shapes = [[(1, 2, 2), (1, 2, 2)]]
        for index in range(2, 11):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "missing"):
                enh.joint_bias(torch, mask, shapes, {}, {index: 1.2}, "cpu", torch.float32)


class PhraseParameterTests(unittest.TestCase):
    def test_phrase_weight_boundaries(self):
        for weight in (0, 0.25, 1, 1.3):
            with self.subTest(weight=weight):
                clean, spans = enh.phrases(f"start (blue sky:{weight}) end")
                self.assertEqual(clean, "start blue sky end")
                self.assertEqual(spans, [(6, 14, float(weight))])

    def test_negative_infinite_and_underflow_are_rejected(self):
        for value in ("-1", "1e999", "1e-999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                enh.phrases(f"(blue:{value})")

    def test_repeat_occurrence_maps_only_selected_phrase(self):
        class Tokenizer:
            is_fast = True

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 4), (5, 8), (9, 13)]}

        clean, spans = enh.phrases("blue and (blue:1.3)")
        self.assertEqual(clean, "blue and blue")
        rows = enh.token_rows(Tokenizer(), clean, clean, spans, [1, 2, 3], 0)
        self.assertEqual(rows, {2: math.log(1.3)})

    def test_cross_token_and_trim_drift_fail_closed(self):
        class Tokenizer:
            is_fast = True

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": [1], "offset_mapping": [(0, 8)]}

        with self.assertRaisesRegex(ValueError, "cuts across"):
            enh.token_rows(Tokenizer(), "blue sky", "blue sky", [(1, 4, 1.3)], [1], 0)
        clean, spans = enh.phrases("x (  blue  :1.3) y")
        self.assertEqual(clean, "x   blue   y")
        self.assertEqual(spans, [(4, 8, 1.3)])


class UiContractTests(unittest.TestCase):
    @staticmethod
    def options(values):
        forge = load_forge()
        script = types.SimpleNamespace(_pi_qwen21=True, args_from=0, args_to=len(values))
        runner = types.SimpleNamespace(alwayson_scripts=[script])
        return forge.options(runner, types.SimpleNamespace(script_args=values))

    def test_legacy_controls_keep_positions_and_enhancer_defaults_off(self):
        legacy = ["edit", 40, 1, "auto", 0, True, False, False, None] + [None] * 9
        legacy += [False, True, False, "(none)", 1.0, "off", "(none)", False, False, True, False, 2, "(none)", False, 0]
        result = self.options(legacy)
        self.assertEqual(result["refs"], [None] * 9)
        self.assertEqual(result["sampler_sharp"], 0)
        self.assertNotIn("phrase_weights", result)
        self.assertNotIn("reference_priorities", result)

    def test_new_controls_are_exactly_positions_33_and_34(self):
        values = ["edit", 40, 1, "auto", 0, True, False, False, None] + [None] * 9
        values += [False, True, False, "(none)", 1.0, "off", "(none)", False, False, True, False, 2, "(none)", False, 0, True, "10:8"]
        self.assertEqual(len(values), 35)
        result = self.options(values)
        self.assertIs(result["phrase_weights"], True)
        self.assertEqual(result["reference_priorities"], "10:8")
        self.assertEqual(result["sampler_sharp"], 0)


if __name__ == "__main__":
    unittest.main()
