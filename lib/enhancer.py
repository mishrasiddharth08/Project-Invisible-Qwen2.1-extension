"""MIT: Diffusers bridge inspired by capitan01R's Qwen 2.1 enhancer.

Source e7204ab908526381c67cd8276adc7aa2e9581c95; see vendor license.
Request-local target-query attention bias; no embedding/pixel scaling.
"""
import contextlib
import inspect
import math
import re
from decimal import Decimal

PATTERN=re.compile(r'\((?P<phrase>[^()]+):(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)')

def phrases(text):
    parts=[]; spans=[]; cursor=length=0
    for m in PATTERN.finditer(text):
        raw=m['phrase']; phrase=raw.strip(); number=Decimal(m['weight']); weight=float(number)
        if not phrase or number<0 or not math.isfinite(weight) or (weight==0 and number!=0):
            raise ValueError('Phrase weights must be finite and nonnegative; use complete words.')
        prefix=text[cursor:m.start()]; parts.extend((prefix,raw)); length+=len(prefix)
        spans.append((length+len(raw)-len(raw.lstrip()),length+len(raw.rstrip()),weight))
        length+=len(raw);cursor=m.end()
    parts.append(text[cursor:]); return ''.join(parts),spans

def references(value):
    result={}
    for item in re.split(r'[,;\n]+',str(value or '')):
        if not item.strip():continue
        try:
            index,weight=item.strip().split(':');index=int(index);weight=float(weight)
        except (ValueError,TypeError):
            raise ValueError('Reference priorities: use 1:1.2, 2:0.8 (image number:strength).') from None
        if not 1<=index<=10 or not math.isfinite(weight) or not 0<=weight<=8:
            raise ValueError('Reference number must be 1–10 and strength 0–8.')
        result[index]=weight
    return {i:w for i,w in result.items() if w!=1}

def token_rows(tokenizer, rendered, clean, spans, actual_ids, drop):
    if not spans:return {}
    if not getattr(tokenizer,'is_fast',False):raise ValueError('Phrase weights require the bundled fast tokenizer.')
    offset=rendered.rfind(clean)
    if offset<0:raise ValueError('Weighted prompt could not be located in the native text template.')
    encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids']; offsets=encoded['offset_mapping']; shift=len(actual_ids)-len(ids)
    rows={}; owners=set()
    for start,end,weight in spans:
        found=[]
        for i,(a,b) in enumerate(offsets):
            if a>=offset+end or b<=offset+start:continue
            row=i+shift
            if row<drop or ids[i:]!=actual_ids[row:]:
                raise ValueError('Phrase tokens do not match the native conditioning suffix; use phrases after all image markers.')
            outside=rendered[a:min(b,offset+start)]+rendered[max(a,offset+end):b]
            if outside.strip() or row in owners:
                raise ValueError('A phrase cuts across a token or overlaps another phrase. Mark a complete word/phrase.')
            owners.add(row);found.append(row-drop)
        if not found:raise ValueError('Weighted phrase has no retained text tokens.')
        if weight!=1:
            for row in found:rows[row]=math.log(weight) if weight else -80.
    return rows

def joint_bias(torch, image_mask, shapes, text_rows, weights, device, dtype):
    mask=image_mask[0].bool(); lengths=[math.prod(s) for s in shapes[0]]
    refs=lengths[:-1];target=lengths[-1]
    if weights and max(weights)>len(refs):raise ValueError('A selected reference is missing. Main img2img image is reference 1.')
    repeats=torch.where(mask,4,1)
    starts=torch.cumsum(repeats,0)-repeats
    key_length=int(repeats.sum());bias=torch.zeros((1,1,1,key_length),device=device,dtype=dtype)
    for row,amount in text_rows.items():
        if row>=mask.numel() or mask[row]:raise ValueError('Phrase row maps to an image slot; token layout changed.')
        bias[...,int(starts[row])]=amount
    slots=starts[mask].tolist();positions=[p+j for p in slots for j in range(4)]
    if len(positions)!=sum(lengths):raise ValueError('Reference latent layout differs from native Qwen 2.1 slots.')
    cursor=0
    for i,n in enumerate(refs,1):
        if i in weights:
            bias[...,positions[cursor:cursor+n]]=math.log(weights[i]) if weights[i] else float('-inf')
        cursor+=n
    return bias,target

def add_mask(torch,bias,mask):
    if mask is None:return bias
    if mask.dtype==torch.bool:return bias.masked_fill(~mask,float('-inf'))
    return mask+bias.to(mask.dtype)

@contextlib.contextmanager
def apply(pipe, phrase_weights=False, reference_priorities='', report=lambda text:None):
    weights=references(reference_priorities)
    if not phrase_weights and not weights:
        yield;return
    import torch
    from diffusers.models.transformers import transformer_qwenimage21 as native
    if pipe.__class__.__name__!='QwenImage21Pipeline' or not pipe.transformer.config.causal_condition:
        raise ValueError('Enhancer requires the causal QwenImage21Pipeline.')
    saved={name:pipe.__dict__.get(name) for name in ('processor','_get_qwen_prompt_embeds','encode_prompt')}
    original_processor,original_get,original_encode=pipe.processor,pipe._get_qwen_prompt_embeds,pipe.encode_prompt
    layouts={};current={};selected={};old_processors=[];handle=None

    class ProcessorProxy:
        def __getattr__(self,name):return getattr(original_processor,name)
        def __call__(self,*args,**kwargs):
            result=original_processor(*args,**kwargs)
            current['rendered']=kwargs['text'][0]
            current['ids']=result.input_ids[0][result.attention_mask[0].bool()].tolist()
            return result

    def get(prompt=None,image=None,device=None):
        texts=[prompt] if isinstance(prompt,str) else prompt
        if len(texts)!=1:raise ValueError('Enhancer expects one prompt per worker call.')
        clean,spans=phrases(texts[0]) if phrase_weights else (texts[0],[])
        current.clear();result=original_get([clean],image,device)
        current['rows']=token_rows(original_processor.tokenizer,current['rendered'],clean,spans,current['ids'],pipe._drop_idx)
        if current['rows']:report(f'Phrase weighting: {len(current["rows"])} text tokens')
        return result

    def encode(*args,**kwargs):
        current.clear();result=original_encode(*args,**kwargs)
        layouts[result[0].data_ptr()]=dict(current.get('rows',{}))
        return result

    def before(module,args,kwargs):
        if args:raise ValueError('Enhancer requires native keyword-based transformer calls.')
        embedding=kwargs['encoder_hidden_states']
        rows=layouts.get(embedding.data_ptr())
        if rows is None:raise ValueError('Conditioning changed after phrase mapping; enhancer cannot apply safely.')
        selected['bias'],selected['target']=joint_bias(torch,kwargs['img_mask'],kwargs['img_shapes'],rows,weights,embedding.device,embedding.dtype)
        selected['active']=bool(rows or weights)

    class Attention:
        def __init__(self,original):self.original=original
        def __call__(self,attn,hidden_states,attention_mask=None,rotary_emb=None,layer_cache=None,kv_cache_mode=None,cache_write_slice=None,segments=None,key_valid=None):
            bias=selected['bias']
            if not selected['active']:
                return self.original(attn,hidden_states,attention_mask=attention_mask,rotary_emb=rotary_emb,layer_cache=layer_cache,kv_cache_mode=kv_cache_mode,cache_write_slice=cache_write_slice,segments=segments,key_valid=key_valid)
            q,k,v,n=native._qwenimage21_prepare_qkv(attn,hidden_states,rotary_emb,layer_cache,kv_cache_mode,cache_write_slice)
            if k.shape[1]!=bias.shape[-1]:raise ValueError('Enhancer key layout changed in attention.')
            def attention(query,key,value,mask=None):
                return native.dispatch_attention_fn(query,key,value,attn_mask=mask,dropout_p=0.,backend=getattr(self.original,'_attention_backend',None) if segments is None else None,parallel_config=getattr(self.original,'_parallel_config',None))
            if segments is None:
                if q.shape[1]!=selected['target']:raise ValueError('Enhancer only modifies target queries.')
                out=attention(q,k,v,add_mask(torch,bias,attention_mask))
            else:
                outputs=[]
                for start,end,is_text in segments:
                    m=None
                    if is_text:
                        m=torch.cat((torch.ones(end-start,start,device=q.device,dtype=torch.bool),torch.tril(torch.ones(end-start,end-start,device=q.device,dtype=torch.bool))),dim=1)[None,None]
                    if key_valid is not None:
                        valid=key_valid[:,None,None,:end];m=valid if m is None else m&valid
                    outputs.append(attention(q[:,start:end],k[:,:end],v[:,:end],m))
                prefix=segments[-1][1] if segments else 0
                if q.shape[1]-prefix!=selected['target']:raise ValueError('Enhancer prefix/target query layout changed.')
                m=None if key_valid is None else key_valid[:,None,None,:]
                outputs.append(attention(q[:,prefix:],k,v,add_mask(torch,bias,m)))
                out=torch.cat(outputs,dim=1)
            out=out[:,:n].flatten(2,3).type_as(q)
            return attn.to_out[1](attn.to_out[0](out))

    try:
        pipe.processor=ProcessorProxy();pipe._get_qwen_prompt_embeds=get;pipe.encode_prompt=encode
        handle=pipe.transformer.register_forward_pre_hook(before,with_kwargs=True)
        for block in pipe.transformer.transformer_blocks:
            original=block.attn.processor
            if not isinstance(original,native.QwenImage21AttnProcessor):
                raise ValueError('Enhancer needs the native Qwen 2.1 attention processor; another patch is active.')
            old_processors.append((block.attn,original));block.attn.set_processor(Attention(original))
        if weights:report('Reference weighting: '+', '.join(f'{i}:{w:g}' for i,w in weights.items()))
        yield
    finally:
        if handle is not None:handle.remove()
        for attn,original in old_processors:attn.set_processor(original)
        for name,value in saved.items():
            if value is None:pipe.__dict__.pop(name,None)
            else:setattr(pipe,name,value)
        layouts.clear();current.clear();selected.clear()
