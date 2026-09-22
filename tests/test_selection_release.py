import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from test_registration import load_forge

class SelectionReleaseTests(unittest.TestCase):
    def test_selection_hooks_chain_and_install_once(self):
        forge=load_forge()
        previous=Mock()
        item=SimpleNamespace(onchange=previous)
        opts=SimpleNamespace(data_labels={'sd_model_checkpoint':item},forge_preset='other')
        opts.onchange=lambda key, fn, call=False: setattr(opts.data_labels[key],'onchange',fn)
        with patch.dict('sys.modules',{'modules':SimpleNamespace(shared=SimpleNamespace(opts=opts))}), patch.object(forge,'selected',return_value=None), patch.object(forge.runtime,'release_on_selection') as release:
            forge.install_selection_release()
            installed=item.onchange
            forge.install_selection_release()
            self.assertIs(installed,item.onchange)
            item.onchange()
            release.assert_called_once()
            previous.assert_called_once()

    def test_qwen_selection_keeps_worker(self):
        forge=load_forge()
        item=SimpleNamespace(onchange=None)
        opts=SimpleNamespace(data_labels={'sd_model_checkpoint':item})
        opts.onchange=lambda key, fn, call=False: setattr(item,'onchange',fn)
        with patch.dict('sys.modules',{'modules':SimpleNamespace(shared=SimpleNamespace(opts=opts))}), patch.object(forge,'selected',return_value='qwen'), patch.object(forge.runtime,'release_on_selection') as release:
            forge.install_selection_release();item.onchange()
            release.assert_not_called()

    def test_switch_terminates_only_dedicated_worker_before_cleanup(self):
        runtime=load_forge().runtime
        process=Mock();process.poll.return_value=None
        with patch.object(runtime,'_pipe',SimpleNamespace(_process=process)), patch.object(runtime,'release') as release:
            runtime.release_on_selection()
            process.terminate.assert_called_once()
            release.assert_called_once()
