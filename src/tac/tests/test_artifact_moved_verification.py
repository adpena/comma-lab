"""Regression controls for real data forks and certification-time drift."""
import hashlib
import json

import pytest

from tac import artifact_moved as moved


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('name,data', [('._payload', b'x' * 8192),
                                      ('payload', b'\x00\x05\x16\x07' + b'x' * 8192)])
def test_sidecar_source_never_creates_certificate(tmp_path, name, data):
    src, dst = tmp_path / name, tmp_path / 'destination'
    src.write_bytes(data)
    with pytest.raises(moved.MovedArtifactError, match='APPLEDOUBLE'):
        moved.move_with_manifest(src, dst, 'custody regression', 'fixture bytes')
    assert src.read_bytes() == data
    assert not src.with_name(src.name + '.MOVED.json').exists()
    assert not dst.exists()


def test_sidecar_destination_never_published(tmp_path):
    src, dst = tmp_path / 'payload', tmp_path / '._payload'
    src.write_bytes(b'valid payload')
    with pytest.raises(moved.MovedArtifactError, match='APPLEDOUBLE'):
        moved.move_with_manifest(src, dst, 'custody regression', 'fixture bytes')
    assert src.exists()
    assert not dst.exists()


def test_metadata_only_verification_does_not_hash(tmp_path, monkeypatch):
    p = tmp_path / 'payload'
    p.write_bytes(b'payload')
    monkeypatch.setattr(moved, '_sha', lambda p: pytest.fail('metadata gate hashed bulk'))
    result = moved.verify_payload(p, 7, sha(b'payload'), hash_payload=False)
    assert result['hash_verified'] is False
    assert result['sha256'] is None


@pytest.mark.parametrize('kind', ['missing', 'directory', 'symlink', 'size', 'sha'])
def test_verifier_refuses_nonmatching_payload(tmp_path, kind):
    p = tmp_path / 'payload'
    if kind == 'directory':
        p.mkdir()
    elif kind == 'symlink':
        real = tmp_path / 'real'
        real.write_bytes(b'payload')
        p.symlink_to(real)
    elif kind != 'missing':
        p.write_bytes(b'payload')
    with pytest.raises((moved.MovedArtifactError, OSError)):
        moved.verify_payload(p, 8 if kind == 'size' else 7,
                             '0' * 64 if kind == 'sha' else sha(b'payload'))


def test_same_size_copy_corruption_keeps_both_files(tmp_path, monkeypatch):
    src, dst = tmp_path / 'source', tmp_path / 'destination'
    src.write_bytes(b'payload')
    monkeypatch.setattr(moved.shutil, 'copyfileobj', lambda source, target, size: target.write(b'corrupt'))
    with pytest.raises(moved.MovedArtifactError, match='DRIFT'):
        moved.move_with_manifest(src, dst, 'custody regression', 'fixture bytes')
    assert src.read_bytes() == b'payload'
    assert dst.read_bytes() == b'corrupt'
    assert not src.with_name(src.name + '.MOVED.json').exists()


def test_destination_removed_at_certificate_publication_keeps_source(tmp_path, monkeypatch):
    src, dst = tmp_path / 'source', tmp_path / 'destination'
    src.write_bytes(b'payload')
    real = moved.atomic_write_json
    def publish(path, row):
        real(path, row)
        dst.unlink()
    monkeypatch.setattr(moved, 'atomic_write_json', publish)
    with pytest.raises((moved.MovedArtifactError, OSError)):
        moved.move_with_manifest(src, dst, 'custody regression', 'fixture bytes')
    assert src.read_bytes() == b'payload'


def test_verified_copy_keeps_source_and_existing_target(tmp_path):
    src, dst = tmp_path / 'source', tmp_path / 'destination'
    src.write_bytes(b'payload')
    receipt = moved.copy_verified(src, dst, expected_bytes=7, expected_sha256=sha(b'payload'))
    assert receipt['hash_verified'] is True
    assert src.read_bytes() == dst.read_bytes() == b'payload'
    with pytest.raises(moved.MovedArtifactError):
        moved.copy_verified(src, dst)
    assert dst.read_bytes() == b'payload'


def test_small_real_payload_remains_supported(tmp_path):
    src, dst = tmp_path / 'source', tmp_path / 'destination'
    src.write_bytes(b'x')
    moved.move_with_manifest(src, dst, 'small real payload', 'fixture bytes')
    row = json.loads(src.with_name(src.name + '.MOVED.json').read_text())
    assert row['sha256'] == sha(b'x')
    assert moved.resolve(src) == dst
