"""DPM++ 2M Sharp core math: multistep combination and sharpening contract."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qwen21_sampling', ROOT / 'lib/sampling.py')
sampling = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sampling
spec.loader.exec_module(sampling)


class FakeSample(float):
    """Minimal tensor-like scalar: all arithmetic is native float math."""


class SharpHistoryTests(unittest.TestCase):
    def test_zero_sharpness_first_step_matches_euler(self):
        # First step: no history, x0 = x - sigma*v; x_next = ratio*x + (1-ratio)*x0
        state = sampling.SharpHistory(sharpness=0.0)
        sigmas = [1.0, 0.5, 0.0]
        sample = FakeSample(2.0)   # x = x0 + sigma*v with x0=1, v=1
        result = state.step(sample, 1.0, sigmas, 0, 2)
        self.assertAlmostEqual(float(result), 1.5)  # 0.5*x + 0.5*x0

    def test_sharpening_scales_history_progressively(self):
        state_a = sampling.SharpHistory(sharpness=0.0)
        state_b = sampling.SharpHistory(sharpness=0.5)
        sigmas = [1.0, 0.5, 0.25, 0.0]
        for state in (state_a, state_b):
            state.step(FakeSample(2.0), 1.0, sigmas, 0, 3)
            state.step(FakeSample(2.0), 1.0, sigmas, 1, 3)
        # x0 is identical for both states at each step (history only affects
        # d_d), so the stored difference is exactly the adjustment factor:
        # 1 + 0.5*(1/3)^2 vs 1 + 0.
        ratio = state_b.prev / state_a.prev
        self.assertAlmostEqual(ratio, 1.0 + 0.5 / 9.0, places=5)

    def test_second_step_uses_multistep_combination(self):
        state = sampling.SharpHistory(sharpness=0.0)
        sigmas = [1.0, 0.5, 0.0]
        state.step(FakeSample(2.0), 1.0, sigmas, 0, 2)
        # Second step: sigma=0.5 -> 0: returns x0 directly (final step).
        result = state.step(FakeSample(1.0), 1.0, sigmas, 1, 2)
        self.assertAlmostEqual(float(result), 0.5)


if __name__ == '__main__':
    unittest.main()