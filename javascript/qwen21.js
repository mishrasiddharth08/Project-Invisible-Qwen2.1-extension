// Scope UI visibility and disabled native LoRA cards to this checkpoint only.
onUiUpdate(function () {
    const root = gradioApp();
    const picker = root.querySelector('#setting_sd_model_checkpoint input');
    const active = /qwen[-_]image[-_]2[._]1/i.test(picker ? picker.value : '');
    root.querySelectorAll('.card').forEach(card => {
        const disabled = active && card.textContent.includes('pi_qwen21_incompatible');
        card.classList.toggle('pi-qwen21-incompatible', disabled);
        if (disabled) card.setAttribute('data-pi-qwen21-note', 'not a Qwen-Image-2.1 LoRA');
        else card.removeAttribute('data-pi-qwen21-note');
    });
});

// The Project Invisible refiner replaces Forge core's two-stage refiner; its
// "LoRA Replacements" accordion is unused here. Hide it without editing core.
onUiUpdate(function () {
    const root = gradioApp();
    if (!root) return;
    // Label-based: gradio's accordion markup differs between versions, so
    // match any element whose text starts with the label, then hide its
    // accordion container (details/summary wrapper or .gr-accordion ancestor).
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (!node.textContent.trim().startsWith('LoRA Replacements')) continue;
        let el = node.parentElement;
        let target = null;
        while (el && el !== root) {
            if (el.tagName === 'DETAILS' || (el.classList && (
                el.classList.contains('gr-accordion') ||
                el.classList.contains('accordion')))) target = el;
            el = el.parentElement;
        }
        if (target) target.style.display = 'none';
        else if (node.parentElement) node.parentElement.style.display = 'none';
    }
});
