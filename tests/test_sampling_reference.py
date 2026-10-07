"""CPU-only four-step DPM++ 2M regression against log-sigma equations."""
import importlib.util
import math
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

LIVE = Path(__file__).resolve().parents[1] / 'lib' / 'sampling.py'
spec = importlib.util.spec_from_file_location('sampling_under_test', LIVE)
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)

class SamplerRegression(unittest.TestCase):
    def test_scheduler_wrapper_four_steps(self):
        class Scalar(float):
            def to(self, dtype):
                return self
            def __add__(self, other): return Scalar(float(self) + float(other))
            __radd__ = __add__
            def __sub__(self, other): return Scalar(float(self) - float(other))
            def __rsub__(self, other): return Scalar(float(other) - float(self))
            def __mul__(self, other): return Scalar(float(self) * float(other))
            __rmul__ = __mul__
        class Base:
            @classmethod
            def from_config(cls, config):
                obj = cls()
                obj.sigmas = [1.0, 0.7, 0.4, 0.2, 0.0]
                obj._step_index = None
                return obj
            @property
            def step_index(self):
                return self._step_index
            def _init_step_index(self, timestep):
                self._step_index = 0
        module = SimpleNamespace(FlowMatchEulerDiscreteSchedulerOutput=lambda **kw: SimpleNamespace(**kw))
        with patch.object(sampling, 'load_scheduler_class', return_value=Base), patch.dict('sys.modules', {'diffusers.schedulers.scheduling_flow_match_euler_discrete': module}):
            scheduler = sampling.build(SimpleNamespace(config={}), sharpness=0.15)
        state = sampling.SharpHistory(0.15)
        actual = expected = Scalar(2.0)
        for i in range(4):
            velocity = Scalar(0.3 * expected + 0.1 * (i + 1))
            velocity.dtype = 'float32'
            expected = state.step(expected, velocity, scheduler.sigmas, i, 4)
            actual = scheduler.step(velocity, i, actual).prev_sample
            actual = Scalar(actual)
            self.assertAlmostEqual(actual, expected, places=12)

    def test_four_steps_varying_denoiser(self):
        for sharpness in (0.0, 0.15, 0.5):
            with self.subTest(sharpness=sharpness):
                sigmas = [1.0, 0.7, 0.4, 0.2, 0.0]
                actual = expected = euler = 2.0
                old_denoised = None
                state = sampling.SharpHistory(sharpness)
                for i, (sigma, next_sigma) in enumerate(zip(sigmas, sigmas[1:])):
                    velocity = 0.3 * expected + 0.1 * (i + 1)
                    denoised = expected - sigma * velocity
                    if next_sigma == 0:
                        expected = denoised
                    else:
                        h = -math.log(next_sigma) + math.log(sigma)
                        if old_denoised is None:
                            combined = denoised
                        else:
                            h_last = -math.log(sigma) + math.log(sigmas[i - 1])
                            r = h_last / h
                            combined = (1 + 1 / (2 * r)) * denoised - old_denoised / (2 * r)
                        expected = math.exp(-h) * expected + (-math.expm1(-h)) * combined
                    old_denoised = denoised * (1 + sharpness * (i / 4) ** 2)
                    actual = state.step(actual, velocity, sigmas, i, 4)
                    euler += (next_sigma - sigma) * (0.3 * euler + 0.1 * (i + 1))
                    self.assertAlmostEqual(actual, expected, places=12, msg=f'nonterminal step {i}')
                self.assertNotAlmostEqual(actual, euler, places=8)

if __name__ == '__main__':
    unittest.main(verbosity=2)
