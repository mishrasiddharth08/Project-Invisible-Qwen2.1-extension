// Reuse Forge's polling and overall bar; add only the current Qwen image bar.
onUiLoaded(function () {
    if (typeof requestProgress !== 'function' || requestProgress._qwenDual) return;
    const original = requestProgress;
    requestProgress = function (id, container, gallery, atEnd, onProgress, timeout) {
        let current = null, overall = null;
        const previewGallery = gallery && gallery.classList.contains('hidden')
            ? gallery.parentElement.querySelector('.gradio-video') : gallery;
        const clear = () => { if (current) current.remove(); current = null; };
        return original(id, container, gallery, function () {
            if (previewGallery) previewGallery.classList.remove('pi-qwen-developing');
            clear(); if (atEnd) atEnd();
        }, function (res) {
            const match = /^Qwen: image (\d+)\/(\d+) \|/.exec(res.textinfo || '');
            if (previewGallery) previewGallery.classList.toggle('pi-qwen-developing', !!match && !!res.active);
            if (match && res.active && opts.show_progressbar) {
                if (!current) {
                    overall = container.previousElementSibling;
                    current = document.createElement('div');
                    current.className = 'progressDiv pi-qwen-current';
                    current.appendChild(document.createElement('div')).className = 'progress';
                    container.parentNode.insertBefore(current, overall);
                }
                const index = Number(match[1]), total = Number(match[2]);
                const fraction = Math.max(0, Math.min(1, (res.progress || 0) * total - (index - 1)));
                const inner = current.firstElementChild;
                inner.style.width = (fraction * 100) + '%';
                inner.textContent = `Current image ${index}/${total}: ${Math.round(fraction * 100)}%`;
                const native = overall && overall.querySelector('.progress');
                if (native) native.textContent = `Overall: ${Math.round((res.progress || 0) * 100)}% — ${res.textinfo.split(' | ')[1]}`;
            } else clear();
            if (onProgress) onProgress(res);
        }, timeout);
    };
    requestProgress._qwenDual = true;
});
