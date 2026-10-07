"""DPM++ 2M Sharp sampler for the Qwen-Image-2.1 pipeline.

Port of envy-ai/ComfyUI-DPMpp-2M-Sharp (GPL-3.0, adapted from ComfyUI's
k-diffusion DPM++ 2M). The reference operates in k-diffusion log-sigma space;
for Qwen's flow-matching schedule (linear sigma 1 -> 0) the same update
reduces to a multistep combination of x0 predictions:

    x_next = ratio * x + (1 - ratio) * d_d
    d_d    = (1 + 1/(2r)) * d_i - (1/(2r)) * d_prev_adj
    ratio  = sigma_next / sigma
    r      = log(sigma_prev / sigma) / log(sigma / sigma_next)

and the sharpening stores each x0 prediction scaled by
``1 + sharpness * (i / total)**2`` before it is used as history.
Zero sharpness with a plain schedule reproduces plain Euler behaviour on the
first step and the 2M multistep combination afterwards (identical model-call
count to Euler; this is the reference's contract).
"""
import math


class SharpHistory:
    """Multistep state shared between pipeline steps."""

    def __init__(self, sharpness=0.15):
        self.sharpness = float(sharpness)
        self.prev = None        # scaled previous x0 prediction
        self.prev_sigma = None  # sigma at the previous step

    def reset(self):
        self.prev = None
        self.prev_sigma = None

    def step(self, sample, velocity, sigmas, index, total):
        """Return the next sample given x, velocity and the sigma schedule."""
        sigma = float(sigmas[index])
        sigma_next = float(sigmas[index + 1]) if index + 1 < len(sigmas) else 0.0
        # Flow matching: x = x0 + sigma * v, so x0 = x - sigma * v.
        x0 = sample - sigma * velocity
        if sigma_next <= 0.0:
            result = x0
        else:
            ratio = sigma_next / sigma
            if self.prev is None or self.prev_sigma is None or self.prev_sigma <= 0.0:
                d_d = x0
            else:
                h = math.log(sigma / sigma_next)
                h_last = math.log(self.prev_sigma / sigma)
                r = h_last / h if h else 1.0
                if not math.isfinite(r) or r <= 0.0:
                    d_d = x0
                else:
                    d_d = (1.0 + 1.0 / (2.0 * r)) * x0 - (1.0 / (2.0 * r)) * self.prev
            result = ratio * sample - (ratio - 1.0) * d_d
        adjustment = 1.0 + self.sharpness * (index / max(1, total)) ** 2
        self.prev = x0 * adjustment
        self.prev_sigma = sigma
        return result


def load_scheduler_class():
    """Resolve the flow-match scheduler class without importing torch eagerly."""
    import diffusers
    return diffusers.FlowMatchEulerDiscreteScheduler


def build(scheduler, sharpness=0.15):
    """Return a SharpScheduler wrapper around an existing scheduler config.

    The wrapper keeps the underlying scheduler's timestep/sigma schedule and
    config (the pipeline calls set_timesteps on it as usual); only ``step``
    is replaced with the sharpened 2M update. ``sharpness=0`` reproduces the
    reference's zero-sharpness contract: identical model-call count and the
    standard DPM++ 2M multistep combination after the first Euler step.
    """
    base_class = load_scheduler_class()
    output_class = __import__(
        'diffusers.schedulers.scheduling_flow_match_euler_discrete',
        fromlist=['FlowMatchEulerDiscreteSchedulerOutput']
    ).FlowMatchEulerDiscreteSchedulerOutput

    class SharpScheduler(base_class):
        _pi_sharpness = float(sharpness)
        _pi_state = None

        def step(self, model_output, timestep, sample, *args, **kwargs):
            return_dict = kwargs.get('return_dict', True)
            per_token = kwargs.get('per_token_timesteps', None)
            if self._pi_state is None:
                self._pi_state = SharpHistory(self._pi_sharpness)
            if self.step_index is None:
                self._init_step_index(timestep)
            sigma_idx = self.step_index
            sigma = float(self.sigmas[sigma_idx])
            sigma_next = float(self.sigmas[sigma_idx + 1]) if sigma_idx + 1 < len(self.sigmas) else 0.0
            state = self._pi_state
            x0 = sample - sigma * model_output
            if sigma_next <= 0.0:
                prev = x0
            else:
                ratio = sigma_next / sigma
                if state.prev is None or state.prev_sigma is None or state.prev_sigma <= 0.0:
                    d_d = x0
                else:
                    h = math.log(sigma / sigma_next)
                    h_last = math.log(state.prev_sigma / sigma)
                    r = h_last / h if h else 1.0
                    if not math.isfinite(r) or r <= 0.0:
                        d_d = x0
                    else:
                        d_d = (1.0 + 1.0 / (2.0 * r)) * x0 - (1.0 / (2.0 * r)) * state.prev
                prev = ratio * sample - (ratio - 1.0) * d_d
            adjustment = 1.0 + self._pi_sharpness * (sigma_idx / max(1, len(self.sigmas) - 1)) ** 2
            state.prev = x0 * adjustment
            state.prev_sigma = sigma
            self._step_index += 1
            prev = prev.to(model_output.dtype)
            if per_token is not None:
                prev = prev
            if not return_dict:
                return (prev,)
            return output_class(prev_sample=prev)

    return SharpScheduler.from_config(scheduler.config)