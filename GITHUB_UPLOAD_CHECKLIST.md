# GitHub upload checklist

## Repository root

These files and folders should be at the top level of the repository:

```text
.github/
.gitattributes
.gitignore
CHANGELOG.md
CONTRIBUTING.md
GITHUB_DESCRIPTION.md
GITHUB_UPLOAD_CHECKLIST.md
ISSUE_TEMPLATE.md
LICENSE
MANIFEST.json
NOTICE
README.md
UPDATE_SAFETY.md
VALIDATION.md
verify_integrity.py
__init__.py
config.json
download/
install.py
javascript/
lib/
lora/
prompts/
requirements.txt
resources/
scripts/
style.css
tests/
```

The `.github/ISSUE_TEMPLATE/bug_report.md` file must remain inside `.github` so GitHub can offer it automatically.

## Do not upload

```text
_deps/
__pycache__/
logs/
model weights
generated images
temporary files
access tokens or private data
```

## D:\GITHUB PROJECT copy

The uploadable folder should be:

```text
D:\GITHUB PROJECT\project-invisible-qwen\
```

It should contain the exact same GitHub-ready tree listed above. Upload the contents of that folder to the repository root.

## Final check before upload

- Open `README.md` and confirm it renders correctly.
- Confirm `_deps`, logs and model files are absent.
- Confirm `LICENSE`, `NOTICE` and vendor licenses are present.
- Confirm the issue template asks for the complete DOS/terminal error.
- Confirm only text-to-image is described as tested.
- Run the included tests if possible.
- Check the folder for private usernames, tokens, prompts and images.
