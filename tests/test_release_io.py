import io
import json
import tarfile
from pathlib import Path
import numpy as np
import pytest
from scripts.download_checkpoints import install, digest

ROOT=Path(__file__).resolve().parents[1]


def test_saved_scenes_have_complete_endpoints():
    for file in (ROOT/'benchmarks/scenes').glob('*.json'):
        scene=json.loads(file.read_text())
        starts=np.asarray(scene['starts']);goals=np.asarray(scene['goals'])
        assert starts.shape==goals.shape and starts.shape[1]==2
        assert np.isfinite(starts).all() and np.isfinite(goals).all()
        assert len(scene['input_sha256'])==64


def test_archive_installer_rejects_traversal_before_writing(tmp_path):
    archive=tmp_path/'malicious.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ['valid.txt','../escape.txt']:
            info=tarfile.TarInfo(name);info.size=3
            tar.addfile(info,io.BytesIO(b'bad'))
    destination=tmp_path/'install';destination.mkdir()
    with pytest.raises(ValueError,match='Unsafe'):install(archive,destination)
    assert not list(destination.iterdir())
    assert not (tmp_path/'escape.txt').exists()


def test_archive_installer_and_hash(tmp_path):
    archive=tmp_path/'weights.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        info=tarfile.TarInfo('checkpoints/example.txt');info.size=4
        tar.addfile(info,io.BytesIO(b'test'))
    install(archive,tmp_path/'installed')
    assert (tmp_path/'installed/checkpoints/example.txt').read_bytes()==b'test'
    assert len(digest(archive))==64
