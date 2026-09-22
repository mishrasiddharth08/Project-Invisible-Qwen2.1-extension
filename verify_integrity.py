"""Read-only source integrity check. No third-party dependencies."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    failures = []
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            failures.append(name + ': invalid path')
        elif not path.is_file():
            failures.append(name + ': missing')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name + ': changed')
    return failures


if __name__ == '__main__':
    try:
        errors = verify(Path(__file__).parent)
    except (OSError, ValueError) as exc:
        raise SystemExit('Could not check integrity: ' + str(exc))
    print('\n'.join(errors) if errors else 'All recorded source files match this release.')
    raise SystemExit(bool(errors))
