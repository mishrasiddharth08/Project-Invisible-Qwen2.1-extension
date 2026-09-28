# Validation — September 28, 2026

190 automated tests passed in this update, including real Gradio panel construction,
preservation of all 27 script arguments, CFG branch separation, finite forecast fallback,
batch memory release, drift alignment geometry/alpha and output metadata.

The isolated UI was checked in a browser at desktop and narrow widths. Fast mode
updated native steps to eight and selected the turbo adapter. Generation and downloads
were not invoked by the UI preview.

No live GPU generation or matched visual comparison was performed for these changes.
Automated tests do not establish universal speed, memory, image-quality or GPU support.
Restart Forge to activate the installed source and new UI.
