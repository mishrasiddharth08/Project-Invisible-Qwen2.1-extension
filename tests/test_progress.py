import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location('qwen_progress', Path(__file__).resolve().parents[1]/'lib/progress.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ProgressTests(unittest.TestCase):
    def make(self, enabled=True, interval=1):
        state = SimpleNamespace(interrupted=False, skipped=False, assign_current_image=Mock())
        shared = SimpleNamespace(state=state, opts=SimpleNamespace(live_previews_enable=enabled, show_progress_every_n_steps=interval), total_tqdm=Mock())
        return module.ForgeProgress(shared, 4, 2), shared

    def test_current_bar_resets_without_resetting_overall(self):
        bridge, shared = self.make()
        bridge.bar = Mock()
        bridge.step(None, 3, None, {})
        bridge.start_image(1)
        bridge.bar.reset.assert_called_once_with(total=4)
        bridge.step(None, 0, None, {})
        self.assertEqual(shared.total_tqdm.update.call_count, 5)
        self.assertIn('image 2/2 |', shared.state.textinfo)

    def test_gradient_starts_each_image_and_final_replaces_it(self):
        bridge, shared = self.make()
        bridge.start_image(0,512,1024)
        canvas=shared.state.assign_current_image.call_args.args[0]
        self.assertEqual(canvas.size,(128,256))
        self.assertNotEqual(canvas.getpixel((0,0)),canvas.getpixel((0,255)))
        final=object()
        bridge.publish(final,final=True)
        shared.state.assign_current_image.assert_called_with(final)
        disabled, state=self.make(False)
        disabled.start_image(0)
        state.state.assign_current_image.assert_not_called()

    def test_final_image_always_published(self):
        bridge, shared = self.make(False)
        image = object()
        bridge.publish(image, final=True)
        shared.state.assign_current_image.assert_called_once_with(image)
        self.assertIn('image complete', shared.state.textinfo)

    def test_initializes_before_loading_and_resets_next_image(self):
        bridge, shared = self.make()
        self.assertEqual((shared.state.job_count, shared.state.sampling_steps, shared.state.sampling_step), (2,4,0))
        self.assertIn('loading', shared.state.textinfo)
        self.assertIsNone(shared.state.current_latent)
        bridge.step(None, 2, None, {})
        bridge.start_image(1)
        self.assertEqual(shared.state.sampling_step, 0)
        self.assertIn('2/2', shared.state.job)
        bridge.close()
        shared.total_tqdm.clear.assert_called_once()

    def test_native_preview_and_duplicate_events_do_not_double_count(self):
        bridge, shared = self.make()
        image = object()
        bridge.step(None, 0, None, {})
        bridge.step(None, 0, None, {'preview': image})
        shared.total_tqdm.update.assert_called_once()
        shared.state.assign_current_image.assert_called_once_with(image)
        self.assertEqual(shared.state.current_image_sampling_step, 1)
        bridge.step(None, 3, None, {})
        self.assertEqual(shared.total_tqdm.update.call_count, 4)
        self.assertIn('decoding', shared.state.textinfo)

    def test_disabled_and_final_only_preview_preferences(self):
        bridge, shared = self.make(False)
        self.assertEqual(bridge.preview_every, 0)
        bridge.publish(object())
        shared.state.assign_current_image.assert_not_called()
        bridge, shared = self.make(True, -1)
        self.assertEqual(bridge.preview_every, 0)
        bridge.publish(object())
        shared.state.assign_current_image.assert_called_once()

    def test_visible_previews_check_each_step_even_with_a_long_native_interval(self):
        bridge, shared = self.make(True, 10)
        self.assertEqual(bridge.preview_every, 1)

    def test_interrupt_propagates_to_worker(self):
        bridge, shared = self.make()
        shared.state.interrupted = True
        with self.assertRaises(InterruptedError): bridge.step(None, 0, None, {})

if __name__ == '__main__': unittest.main()
