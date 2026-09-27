import ast
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

class FinalPreviewTests(unittest.TestCase):
    def test_last_step_does_not_decode_when_previews_disabled(self):
        tree=ast.parse((Path(__file__).parents[1]/'lib/worker.py').read_text(encoding='utf-8'))
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='progress')
        fn.body=[ast.Global(names=n.names) if isinstance(n,ast.Nonlocal) else n for n in fn.body]
        module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        events=[]
        def forbidden(*args,**kwargs):
            self.fail('Final sampling callback must not perform another full VAE decode')
        env=dict(preview_active=False,last_preview=0.0,completed_steps=[0],
                 command={'num_inference_steps':20},time=time,stop=SimpleNamespace(exists=lambda:False),
                 emit=lambda *a,**kw:events.append((a,kw)),preview=SimpleNamespace(decode_preview=forbidden))
        exec(compile(module,'worker-progress-test','exec'),env)
        payload={'latents':object()}
        self.assertIs(env['progress'](None,19,0,payload),payload)
        self.assertEqual(env['completed_steps'],[20])
        self.assertTrue(any(k.get('final') for _,k in events))
