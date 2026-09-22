import unittest
import numpy as np
from PIL import Image
from lib.degrid import remove_moire
from lib.vendor.degrid.degrid_core import degrid
import torch

class DeGridTests(unittest.TestCase):
    def test_matches_upstream_auto_and_preserves_alpha(self):
        y,x=np.indices((64,64));a=np.zeros((64,64,4),dtype=np.uint8)
        a[...,:3]=(128+((x+y)%2*2-1)*4)[...,None];a[...,3]=(x*4).astype(np.uint8)
        result=np.asarray(remove_moire(Image.fromarray(a)))
        expected,_,stats=degrid(torch.from_numpy(a[...,:3].copy()).float().div(255).unsqueeze(0))
        self.assertFalse(stats[0]['skipped'])
        np.testing.assert_array_equal(result[...,:3],expected.squeeze(0).mul(255).round().to(torch.uint8).numpy())
        np.testing.assert_array_equal(result[...,3],a[...,3])
        self.assertLess(np.std(result[8:-8,8:-8,:3]),np.std(a[8:-8,8:-8,:3]))

    def test_clean_and_bypass_unchanged(self):
        for size in ((4,4),(64,48)):
            image=Image.new('RGBA',size,(97,122,178,193))
            np.testing.assert_array_equal(remove_moire(image),image)
        rng=np.random.default_rng(10);image=Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8))
        np.testing.assert_array_equal(remove_moire(image,0),image)

    def test_limits_change_and_validates_strength(self):
        y,x=np.indices((48,48));a=np.repeat((128+((x+y)%2*2-1)*60)[...,None],3,axis=2).astype(np.uint8)
        image=Image.fromarray(a);out=np.asarray(remove_moire(image)).astype(int)
        self.assertLessEqual(np.abs(out-a.astype(int)).max(),13)
        for strength in (-1,2,float('nan')):
            with self.assertRaises(ValueError):remove_moire(image,strength)
