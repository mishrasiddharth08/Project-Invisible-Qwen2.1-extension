// Scope UI visibility and disabled native LoRA cards to this checkpoint only.
onUiUpdate(function () {
    const root = gradioApp();
    if (!root) return;
    // Native Forge's qwen21 preset owns its own cards. Follow the server's
    // extension-panel visibility instead of guessing ownership by filename.
    const active = [...root.querySelectorAll('.pi-q21-panel')].some(
        panel => !panel.hidden && getComputedStyle(panel).display !== 'none'
    );
    root.querySelectorAll('.card').forEach(card => {
        const disabled = active && card.textContent.includes('pi_qwen21_incompatible');
        card.classList.toggle('pi-qwen21-incompatible', disabled);
        if (disabled) card.setAttribute('data-pi-qwen21-note', 'not a Qwen-Image-2.1 LoRA');
        else card.removeAttribute('data-pi-qwen21-note');
    });
});
