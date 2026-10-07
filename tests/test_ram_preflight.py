import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('ram_policy',Path(__file__).resolve().parents[1]/'lib/ram.py')
ram=importlib.util.module_from_spec(spec);spec.loader.exec_module(ram)

class RAMPreflight(unittest.TestCase):
    def test_empty_bundle_does_not_block(self):
        self.assertIsNone(ram.preflight({},available=0))
    def test_bf16_checks_physical_ram_before_allocation(self):
        with patch.object(Path,'stat',return_value=types.SimpleNamespace(st_size=8*2**30)):
            b={'files':{'te':'qwen3vl_8b_bf16.safetensors'}}
            self.assertEqual(ram.preflight(b,available=12*2**30)['required_bytes'],10*2**30)
            with self.assertRaisesRegex(RuntimeError,'available system RAM'):
                ram.preflight(b,available=9*2**30)
    def test_portable_expansion_and_transient_packed_copy(self):
        with patch.object(Path,'stat',return_value=types.SimpleNamespace(st_size=2*2**30)):
            b={'files':{'te':'qwen3vl_8b_w4a8.safetensors'}}
            self.assertEqual(ram.preflight(b,{'portable':True},available=16*2**30)['required_bytes'],12*2**30)
            self.assertEqual(ram.preflight(b,available=16*2**30)['required_bytes'],4*2**30)
            with self.assertRaises(RuntimeError):ram.preflight(b,{},dequantize=True,available=8*2**30)
    def test_overcommit_is_explicit_and_failed_query_is_best_effort(self):
        with patch.object(Path,'stat',return_value=types.SimpleNamespace(st_size=2*2**30)):
            b={'files':{'te':'qwen3vl_8b_bf16.safetensors'}}
            with patch('builtins.print'):
                self.assertIsNotNone(ram.preflight(b,{'allow_ram_overcommit':True},available=0))
            with patch.object(ram,'available_bytes',return_value=None):
                self.assertIsNone(ram.preflight(b)['available_bytes'])

    def test_renamed_quantization_is_detected_without_loading_tensors(self):
        import json,struct
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            p=Path(td)/'renamed.safetensors'
            data=json.dumps({'layer.weight':{'dtype':'I8','shape':[4,4],'data_offsets':[0,16]}}).encode()
            p.write_bytes(struct.pack('<Q',len(data))+data+b'0'*16)
            self.assertEqual(ram.packed_factor(p),2)
            data=json.dumps({'layer.comfy_quant':{'dtype':'U8','shape':[16],'data_offsets':[0,16]}}).encode()
            p.write_bytes(struct.pack('<Q',len(data))+data+b'0'*16)
            self.assertEqual(ram.packed_factor(p),4)
