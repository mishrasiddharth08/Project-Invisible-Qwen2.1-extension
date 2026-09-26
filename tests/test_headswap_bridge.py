from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import tempfile
import unittest
from PIL import Image
import test_acceptance_cpu as fixtures
runtime=fixtures.runtime


class HeadSwapBridgeTests(unittest.TestCase):
    setUp=fixtures.RuntimeAcceptanceTests.setUp
    tearDown=fixtures.RuntimeAcceptanceTests.tearDown
    p=fixtures.RuntimeAcceptanceTests.p
    options=fixtures.RuntimeAcceptanceTests.options
    run_generate=fixtures.RuntimeAcceptanceTests.run_generate
    def test_external_hook_consumes_headshot_and_returns_finished_image(self):
        primary=Image.new('RGB',(64,64),'red'); ref=Image.new('RGB',(64,64),'blue')
        finished=Image.new('RGB',(40,60),'green')
        p=self.p([primary]); owner=NS(last_report={})
        plan=NS(positive='swap with character',negative='blur',cfg=1)
        session=NS(prepare=Mock(return_value=(plan,[primary,ref])),
                   finish_image=Mock(return_value=finished),metadata=lambda:'verified routing',owner=owner)
        closed=[]
        @contextmanager
        def callback(request):
            self.assertIs(request,p)
            try: yield session
            finally: closed.append(True)
        p.scripts=NS(alwayson_scripts=[NS(headswap_external_context=callback)])
        _,result=self.run_generate(p)
        self.assertEqual(self.pipe.calls[0]['image'],[primary,ref])
        self.assertEqual(self.pipe.calls[0]['prompt'],'swap with character')
        self.assertIs(result.images[0],finished)
        self.assertIn('verified routing',result.info)
        self.assertEqual((result.width,result.height),(40,60))
        self.assertEqual(closed,[True])

    def test_external_hook_cleanup_runs_on_sampling_error(self):
        p=self.p([Image.new('RGB',(64,64))]); closed=[]
        @contextmanager
        def callback(request):
            try: yield NS(prepare=Mock(side_effect=ValueError('bad adapter')))
            finally: closed.append(True)
        p.scripts=NS(alwayson_scripts=[NS(headswap_external_context=callback)])
        with self.assertRaisesRegex(ValueError,'bad adapter'): self.run_generate(p)
        self.assertEqual(closed,[True])

    def test_external_seed_lock_is_honored_across_outputs(self):
        p=self.p([Image.new('RGB',(64,64))]); p.n_iter=2
        session=NS(cfg={'seed_lock':True},owner=NS(last_report={}),
                   prepare=Mock(return_value=(NS(positive='swap',negative='',cfg=1),p.init_images)),
                   finish_image=lambda image,index:image,metadata=lambda:'locked')
        @contextmanager
        def callback(request): yield session
        p.scripts=NS(alwayson_scripts=[NS(headswap_external_context=callback)])
        self.run_generate(p)
        self.assertEqual([call.args[3] for call in session.prepare.call_args_list],[7,7])


class SaveTests(unittest.TestCase):
    def test_save_fallback_and_file_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            shared=NS(opts=NS(samples_save=True,outdir_img2img_samples=directory))
            p=NS(outpath_samples='',override_settings={})
            def save(im,path,*args,**kwargs):
                filename=Path(path)/'output.png'; im.save(filename); return str(filename),None
            images=NS(save_image=Mock(side_effect=save))
            result=runtime.save_output(Image.new('RGB',(8,8)),p,7,'test','metadata',shared,images)
            self.assertTrue(Path(result).is_file())
            self.assertEqual(images.save_image.call_args.kwargs['info'],'metadata')
            images.save_image.return_value=(str(Path(directory)/'missing.png'),None)
            images.save_image.side_effect=None
            with self.assertRaisesRegex(RuntimeError,'not saved'):
                runtime.save_output(Image.new('RGB',(8,8)),p,7,'test','metadata',shared,images)

if __name__=='__main__': unittest.main()
