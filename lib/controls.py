"""Gradio callbacks must return skip updates, never store them as values."""
def speed_updates(enabled, turbo, native=False):
    import gradio as gr
    if not enabled:
        return tuple(gr.skip() for _ in range(3 if native else 2))
    values=[gr.update(value=turbo),gr.update(value=1.0)]
    if native: values.append(gr.update(value=6))
    return tuple(values)

def quality_updates(value, turbo):
    import gradio as gr
    fast=value==6
    return gr.update(value=fast), gr.update(value=turbo) if fast else gr.skip()

def control_value(value, default=None):
    # Also recover requests from a tab containing the previous nested-skip bug.
    for _ in range(4):
        if not isinstance(value,dict) or value.get('__type__')!='update': return value
        value=value.get('value',default)
    raise ValueError('Qwen controls are out of sync. Reload the Forge page.')

def speed_strength(value):
    import math
    value=control_value(value,1.0)
    try: number=float(value)
    except (TypeError,ValueError):
        raise ValueError('Qwen Turbo strength is invalid. Reload the Forge page and select the speed LoRA again.') from None
    if not math.isfinite(number) or not 0<=number<=1.5:
        raise ValueError('Qwen Turbo strength must be between 0 and 1.5.')
    return number
