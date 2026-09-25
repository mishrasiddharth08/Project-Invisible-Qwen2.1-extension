"""Publish worker progress through Forge's existing progress and preview state."""

class ForgeProgress:
    def __init__(self, shared, steps, total):
        self.shared = shared
        self.steps = steps
        self.total = total
        self.completed = 0
        self.index = 0
        self.bar = None
        state = shared.state
        state.job_count = total
        state.job_no = 0
        state.sampling_steps = steps
        state.sampling_step = 0
        state.preview_step = 0
        state.current_latent = None  # Qwen latents must never reach the SD preview decoder.
        state.current_image = None
        state.current_image_sampling_step = 0
        state.textinfo = 'Qwen: loading model'
        state.job = 'Qwen-Image-2.1'
        opts = shared.opts
        enabled = getattr(opts, 'live_previews_enable', False)
        interval = int(getattr(opts, 'show_progress_every_n_steps', 1))
        self.preview_every = 1 if enabled and interval >= 0 else 0
        console = not getattr(getattr(shared, 'cmd_opts', None), 'disable_console_progressbars', True)
        if console:
            from tqdm.auto import tqdm
            self.bar = tqdm(total=steps, desc='Current image', position=0, leave=False, file=getattr(shared, 'progress_print_out', None))

    def status(self, text):
        self.shared.state.textinfo = f'Qwen: image {self.index+1}/{self.total} | ' + str(text)

    def start_image(self, index, width=256, height=256):
        self.index = index
        self.completed = 0
        if self.bar is not None:
            self.bar.reset(total=self.steps)
            self.bar.set_description(f'Current image {index+1}/{self.total}')
        state = self.shared.state
        state.sampling_step = 0
        state.preview_step = 0
        state.current_image_sampling_step = 0
        state.current_image = None
        state.job = f'Qwen image {index + 1}/{self.total}'
        self.status('encoding prompt and reference images')
        if self.preview_every:
            from PIL import Image, ImageOps
            scale=256/max(1,width,height)
            size=(max(1,round(width*scale)),max(1,round(height*scale)))
            canvas=ImageOps.colorize(Image.linear_gradient('L').resize(size),'#202530','#c5cbd4')
            self.publish(canvas)

    def publish(self, image, final=False):
        state = self.shared.state
        if not final and not getattr(self.shared.opts, 'live_previews_enable', False):
            return
        assign = getattr(state, 'assign_current_image', None)
        if callable(assign):
            assign(image)
        else:
            state.current_image = image
            state.id_live_preview = getattr(state, 'id_live_preview', 0) + 1
        state.current_image_sampling_step = state.sampling_step
        if final:
            self.status('image complete')

    def step(self, pipeline, step, timestep, kwargs):
        state = self.shared.state
        if state.interrupted or state.skipped:
            raise InterruptedError('Generation interrupted')
        current = min(self.steps, int(step) + 1)
        delta = max(0, current - self.completed)
        self.completed = max(self.completed, current)
        state.sampling_step = self.completed
        state.preview_step = self.completed
        state.sampling_steps = self.steps
        self.status(f'step {self.completed}/{self.steps}' if self.completed < self.steps else 'decoding image')
        if delta and self.bar is not None:
            self.bar.update(delta)
        total_bar = getattr(self.shared, 'total_tqdm', None)
        if total_bar is not None:
            for _ in range(delta):
                total_bar.update()
        if kwargs.get('preview') is not None:
            self.publish(kwargs['preview'], final=bool(kwargs.get('final')))
        return kwargs

    def close(self):
        if self.bar is not None:
            self.bar.close()
        total_bar = getattr(self.shared, 'total_tqdm', None)
        if total_bar is not None:
            total_bar.clear()
