import importlib.util
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import cv2

spec=importlib.util.spec_from_file_location('pixel_drift',Path(__file__).resolve().parents[1]/'lib/pixel_drift.py')
drift=importlib.util.module_from_spec(spec);spec.loader.exec_module(drift)

class PixelDriftTests(unittest.TestCase):
    def scene(self):
        rng=np.random.default_rng(321)
        pixels=np.full((384,512,3),160,np.uint8)
        for _ in range(150):
            center=tuple(int(x) for x in rng.integers([20,20],[490,360]))
            color=tuple(int(x) for x in rng.integers(0,255,3))
            cv2.circle(pixels,center,int(rng.integers(2,12)),color,-1)
        cv2.putText(pixels,'ALIGN 123',(80,180),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        return pixels

    def test_disabled_is_exact_noop(self):
        edited=Image.fromarray(self.scene())
        fixed,status=drift.apply(None,edited,False)
        self.assertIs(fixed,edited);self.assertEqual(status,'off')

    def test_translation_recovers_alignment_and_keeps_alpha(self):
        original=self.scene()
        rgba=np.dstack((original,np.full(original.shape[:2],180,np.uint8)))
        moved=cv2.warpAffine(rgba,np.float32([[1,0,7],[0,1,-5]]),(512,384),borderMode=cv2.BORDER_REPLICATE)
        fixed,status=drift.align(Image.fromarray(original),Image.fromarray(moved))
        self.assertEqual(status,'applied');self.assertEqual(fixed.size,(512,384));self.assertEqual(fixed.mode,'RGBA')
        before=np.abs(moved[15:-15,15:-15,:3].astype(float)-original[15:-15,15:-15]).mean()
        after=np.abs(np.asarray(fixed)[15:-15,15:-15,:3].astype(float)-original[15:-15,15:-15]).mean()
        self.assertLess(after,before*0.2)
        self.assertTrue((np.asarray(fixed)[:,:,3]==180).all())

    def test_blank_and_unrelated_inputs_are_unchanged(self):
        original=Image.fromarray(self.scene());edited=Image.new('RGB',original.size,'white')
        fixed,status=drift.align(original,edited)
        self.assertIs(fixed,edited);self.assertTrue(status.startswith('skipped:'))

    def test_large_reframing_is_rejected(self):
        pixels=self.scene()
        moved=Image.fromarray(cv2.warpAffine(pixels,np.float32([[1,0,140],[0,1,0]]),(512,384)))
        fixed,status=drift.align(Image.fromarray(pixels),moved)
        self.assertIs(fixed,moved);self.assertTrue(status.startswith('skipped:'))

if __name__=='__main__':unittest.main()
