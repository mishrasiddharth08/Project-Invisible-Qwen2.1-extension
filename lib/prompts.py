"""Consume the exact JSON format returned by Qwen's official prompt rewriter."""
import json
import re
from .assets import BUCKETS

RATIOS=dict(zip(('1:1','4:3','3:4','3:2','2:3','16:9','9:16'),BUCKETS))

def parse(prompt,width,height,images=None):
    if not prompt.lstrip().startswith('{'): return prompt,width,height
    try: data=json.loads(prompt)
    except ValueError: return prompt,width,height
    if not isinstance(data,dict) or 'rewritten_prompt' not in data: return prompt,width,height
    text=data['rewritten_prompt']
    if not isinstance(text,str) or not text.strip(): raise ValueError('rewritten_prompt must be a nonempty string')
    ratio=data.get('wh_ratio','')
    follow=data.get('ratio_follow','')
    if ratio and follow: raise ValueError('Set wh_ratio or ratio_follow, not both.')
    if follow:
        match=re.fullmatch(r'<image([1-9]|10)>',str(follow))
        if not match: raise ValueError('ratio_follow must name a connected image, such as <image1>.')
        index=int(match.group(1))-1
        images=list(images or [])
        if index>=len(images): raise ValueError('ratio_follow refers to a missing image.')
        width,height=images[index].size
    if ratio in RATIOS: width,height=RATIOS[ratio]
    elif ratio:
        try:
            w,h=map(float,ratio.split(':'))
            import math
            if not math.isfinite(w) or not math.isfinite(h) or w<=0 or h<=0: raise ValueError()
            width,height=w,h
        except (ValueError,AttributeError): raise ValueError('wh_ratio must be a positive ratio such as 3:2')
    return text,width,height
