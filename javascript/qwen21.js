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
