import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "_deps"


class SchedulerEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEPS.is_dir():
            raise unittest.SkipTest("bundled Diffusers dependencies are unavailable")
        script = textwrap.dedent(rf'''
            import inspect,json,math,sys,textwrap,types
            from pathlib import Path
            root=Path({str(ROOT)!r});sys.path.insert(0,str(root/'_deps'))
            from diffusers import FlowMatchEulerDiscreteScheduler as Scheduler
            base=json.loads((root/'resources/qwen21/scheduler/scheduler_config.json').read_text())
            source=(root/'lib/worker.py').read_text(encoding='utf-8')
            start=source.index('            sigmas = command.get("sigmas") or None')
            marker='            kwargs = {{k: v for k, v in kwargs.items() if k in supported}}'
            selection=compile(textwrap.dedent(source[start:source.index(marker,start)+len(marker)]),'worker.py:scheduler-selection','exec')
            pipe=types.SimpleNamespace(scheduler=Scheduler.from_config(base))
            def select(steps,sigmas=None):
                command={{'num_inference_steps':steps,'sigmas':sigmas,'sampler_sharpness':None}};kwargs={{}}
                scope={{'pipe':pipe,'command':command,'kwargs':kwargs,'params':{{'sigmas':None}},'inspect':inspect,'alias':'unused'}}
                exec(selection,scope,scope);return kwargs
            result={{}}
            for steps in (1,2,40):
                select(steps);pipe.scheduler.set_timesteps(num_inference_steps=steps,mu=.7)
                result[f'finite{{steps}}']=all(math.isfinite(x) for x in pipe.scheduler.sigmas.tolist())
                result[f'shift{{steps}}']=pipe.scheduler.config.shift_terminal
            result['restored_terminal']=float(pipe.scheduler.sigmas[-2])
            result['base_shift']=pipe._pi_qwen21_base_scheduler_config['shift_terminal']
            select(1);requested=[1.,.5,.25];kwargs=select(3,requested)
            result['forwarded']=kwargs['sigmas'];result['custom_shift']=pipe.scheduler.config.shift_terminal
            pipe.scheduler.set_timesteps(sigmas=kwargs['sigmas'],mu=.7)
            result['custom_finite']=all(math.isfinite(x) for x in pipe.scheduler.sigmas.tolist())
            print(json.dumps(result))
        ''')
        run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
        if run.returncode:
            if "No module named" in run.stderr:
                raise unittest.SkipTest(run.stderr.strip())
            raise AssertionError(run.stderr)
        cls.result = json.loads(run.stdout.strip().splitlines()[-1])

    def test_real_worker_block_keeps_one_two_and_forty_steps_finite(self):
        self.assertTrue(all(self.result[f"finite{x}"] for x in (1, 2, 40)))
        self.assertIsNone(self.result["shift1"])
        self.assertEqual(self.result["shift2"], 0.02)
        self.assertEqual(self.result["shift40"], 0.02)

    def test_real_worker_block_restores_base_after_one_step(self):
        self.assertAlmostEqual(self.result["restored_terminal"], 0.02, places=6)
        self.assertEqual(self.result["base_shift"], 0.02)

    def test_real_worker_block_forwards_exact_custom_sigmas_independently(self):
        self.assertEqual(self.result["forwarded"], [1.0, 0.5, 0.25])
        self.assertIsNone(self.result["custom_shift"])
        self.assertTrue(self.result["custom_finite"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
