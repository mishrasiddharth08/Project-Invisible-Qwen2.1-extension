"""Consume the exact JSON format returned by Qwen's official prompt rewriter."""
import json
from .assets import BUCKETS

RATIOS=dict(zip(('1:1','4:3','3:4','3:2','2:3','16:9','9:16'),BUCKETS))

def parse(prompt,width,height):
    if not prompt.lstrip().startswith('{'): return prompt,width,height
    try: data=json.loads(prompt)
    except ValueError: return prompt,width,height
    if not isinstance(data,dict) or 'rewritten_prompt' not in data: return prompt,width,height
    text=data['rewritten_prompt']
    if not isinstance(text,str) or not text.strip(): raise ValueError('rewritten_prompt must be a nonempty string')
    ratio=data.get('wh_ratio','')
    if ratio in RATIOS: width,height=RATIOS[ratio]
    elif ratio:
        try:
            w,h=map(float,ratio.split(':'))
            if w<=0 or h<=0: raise ValueError()
            width,height=w,h
        except (ValueError,AttributeError): raise ValueError('wh_ratio must be a positive ratio such as 3:2')
    return text,width,height

