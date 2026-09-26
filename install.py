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
        if _pip_install(requirements, target):
            return
        if _pip_install(requirements, target, insecure_git=True):
            return
        if _pip_install(requirements, target, tarball_fallback=True):
            return
        # Last resort with every fallback exhausted: plain retry so the
        # original error surfaces for diagnosis.
        _pip_install(requirements, target, check=True)


def _pip_install(requirements, target, insecure_git=False, tarball_fallback=False, check=False):
    """Install requirements into _deps. Returns True on success.

    insecure_git: retry git clones with certificate verification disabled,
        which fixes "unable to get local issuer certificate" behind proxies.
    tarball_fallback: replace git+https VCS lines with plain HTTPS tarball
        URLs of the same pinned commit, avoiding git entirely (common fix
        for networks where github.com git access is blocked or MITM'd).
    """
    import tempfile
    import shutil
    import urllib.request

    req_file = requirements
    tmp_dir = None
    if tarball_fallback:
        lines = []
        for line in requirements.read_text().splitlines():
            if line.startswith('diffusers @ git+https://github.com/huggingface/diffusers.git@'):
                commit = line.rsplit('@', 1)[-1]
                lines.append('diffusers @ https://github.com/huggingface/diffusers/archive/%s.zip' % commit)
            else:
                lines.append(line)
        tmp_dir = Path(tempfile.mkdtemp(prefix='piq21_req_'))
        req_file = tmp_dir / 'requirements.txt'
        req_file.write_text('\n'.join(lines) + '\n')

    env = dict(os.environ)
    if os.name == 'nt':
        count = int(env.get('GIT_CONFIG_COUNT', '0'))
        env.update(GIT_CONFIG_COUNT=str(count + 1))
        # Prefer Git's bundled OpenSSL TLS over Windows Schannel, which can
        # lack the issuer chain when behind corporate proxies.
        env[f'GIT_CONFIG_KEY_{count}'] = 'http.sslBackend'
        env[f'GIT_CONFIG_VALUE_{count}'] = 'openssl'
        if insecure_git:
            count += 1
            env.update(GIT_CONFIG_COUNT=str(count + 1))
            env[f'GIT_CONFIG_KEY_{count}'] = 'http.sslVerify'
            env[f'GIT_CONFIG_VALUE_{count}'] = 'false'
    elif insecure_git:
        env['GIT_SSL_NO_VERIFY'] = 'true'

    # pip needs a writable temp dir; some systems point TMP at a path the
    # sandbox cannot use, which shows up as git clone failures.
    tmp = Path(tempfile.gettempdir())
    if not tmp.exists():
        try:
            tmp.mkdir(parents=True, exist_ok=True)
        except OSError:
            env['TMPDIR'] = env['TEMP'] = env['TMP'] = str(target.parent)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", str(target),
             "--no-deps", "--no-cache-dir", "--upgrade", "--no-warn-script-location",
             "-r", str(req_file)],
            check=check, env=env)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
