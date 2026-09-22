"""Extension-local packages, same Forge interpreter. No venv or model downloads."""
from pathlib import Path
import subprocess
import sys
import os

def main():
    requirements = Path(__file__).with_name("requirements.txt")
    target = requirements.parent / '_deps'
    # Avoid a Git reinstall on every Forge launch when the exact revision is installed.
    import importlib.metadata as md
    import json
    from packaging.requirements import Requirement
    installed = {d.metadata['Name'].lower().replace('_','-'): d for d in md.distributions(path=[str(target)])}
    ready = True
    for line in requirements.read_text().splitlines():
        req = Requirement(line)
        try:
            dist = installed[req.name.lower().replace('_','-')]
            if req.url:
                direct = json.loads(dist.read_text("direct_url.json") or "{}")
                ready &= direct.get("vcs_info", {}).get("commit_id") == req.url.rsplit("@", 1)[-1]
            else:
                ready &= dist.version in req.specifier
        except (KeyError, md.PackageNotFoundError):
            ready = False
    if not ready:
        env = dict(os.environ)
        # Use Git's bundled TLS instead of Windows credential-dependent Schannel.
        if os.name == 'nt':
            count = int(env.get('GIT_CONFIG_COUNT','0'))
            env.update(GIT_CONFIG_COUNT=str(count+1))
            env[f'GIT_CONFIG_KEY_{count}']='http.sslBackend'
            env[f'GIT_CONFIG_VALUE_{count}']='openssl'
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(target),
                        "--no-deps", "--no-cache-dir", "--upgrade", "--no-warn-script-location", "-r", str(requirements)], check=True,env=env)

if __name__ == "__main__":
    main()
