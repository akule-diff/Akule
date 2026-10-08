"""Fetch versioned archives, verify SHA256, and safely install model assets."""
import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def install(archive, destination):
    destination = destination.resolve()
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not member.isfile() or not target.is_relative_to(destination):
                raise ValueError('Unsafe archive member: ' + member.name)
        # Validate the entire archive before writing any file.
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as dst:
                temporary = Path(dst.name)
                try: shutil.copyfileobj(src, dst)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            os.replace(temporary, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true')
    group.add_argument('--env', choices=['weave', 'highways', 'conveyor', 'basic', 'dense', 'shelf', 'room', 'smd', 'scaledweave'])
    p.add_argument('--cache', type=Path, default=ROOT / '.cache/downloads')
    p.add_argument('--offline', action='store_true', help='Use only verified archives already in --cache')
    p.add_argument('--destination', type=Path, default=ROOT)
    args = p.parse_args()
    manifest = json.loads((ROOT / 'checkpoints/manifest.json').read_text())
    required = set(a['name'] for a in manifest['archives']) if args.all else set(manifest['environments']['basic' if args.env == 'smd' else args.env])
    args.cache.mkdir(parents=True, exist_ok=True)
    for asset in manifest['archives']:
        if asset['name'] not in required: continue
        target = args.cache / asset['filename']
        valid = target.is_file() and target.stat().st_size == asset['size_bytes'] and digest(target) == asset['sha256']
        if not valid:
            if args.offline: raise RuntimeError('Missing or corrupt cached archive: ' + target.name)
            partial = target.with_suffix(target.suffix + '.partial')
            try:
                request = urllib.request.Request(asset['url'], headers={'User-Agent': 'Akule-checkpoint-downloader/1.0'})
                with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
                if partial.stat().st_size != asset['size_bytes'] or digest(partial) != asset['sha256']:
                    raise RuntimeError('Checkpoint archive integrity failure: ' + target.name)
                os.replace(partial, target)
            finally: partial.unlink(missing_ok=True)
        install(target, args.destination)
        print('Verified and installed ' + target.name)


if __name__ == '__main__': main()
