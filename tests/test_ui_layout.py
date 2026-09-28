import unittest
from ui_fixture import build

class UILayoutTests(unittest.TestCase):
    def test_real_gradio_panels_preserve_control_contract(self):
        app=build()
        self.assertGreater(len(app.config["components"]),100)
