"""Crash and custody controls for the VR8 overlay restore."""
import hashlib
import json

import pytest

from experiments import ddm_vr8_restore as restore


@pytest.fixture
def bank(tmp_path, monkeypatch):
    go = tmp_path / 'GO'
    go.touch()
    monkeypatch.setattr(restore, 'GO', go)
    root = tmp_path / 'receipts'
    root.mkdir()
    output = tmp_path / 'odd_frames.u8'
    return root, output, go


def record(root, binding, slot, payload):
    restore.persist(root / f'frame_{slot:04d}.json', {
        'binding': binding, 'pair': slot, 'slot': slot, 'bytes': len(payload),
        'sha256': hashlib.sha256(payload).hexdigest()})


def test_checkpointed_prefix_reused(bank):
    root, output, _ = bank
    output.write_bytes(b'abcd')
    record(root, {}, 0, b'abcd')
    assert restore.prefix(output, root, {}, [0, 1], 4) == 1


def test_interrupted_suffix_is_retained(bank):
    root, output, _ = bank
    output.write_bytes(b'abcdX')
    record(root, {}, 0, b'abcd')
    with pytest.raises(ValueError, match='suffix retained'):
        restore.prefix(output, root, {}, [0, 1], 4)
    assert output.read_bytes() == b'abcdX'


def test_corruption_cannot_resume(bank):
    root, output, _ = bank
    output.write_bytes(b'abXd')
    record(root, {}, 0, b'abcd')
    with pytest.raises(ValueError, match='drift'):
        restore.prefix(output, root, {}, [0], 4)


def test_checkpoint_gap_refuses(bank):
    root, output, _ = bank
    record(root, {}, 1, b'abcd')
    with pytest.raises(ValueError, match='gap'):
        restore.prefix(output, root, {}, [0, 1], 4)


def test_go_revocation_refuses_prefix(bank):
    root, output, go = bank
    output.write_bytes(b'abcd')
    record(root, {}, 0, b'abcd')
    go.unlink()
    with pytest.raises(ValueError, match='MAIN_GO'):
        restore.prefix(output, root, {}, [0], 4)


def test_no_receipt_overwrite(bank):
    root, _, _ = bank
    path = root / 'receipt.json'
    restore.persist(path, {'first': True})
    with pytest.raises(FileExistsError):
        restore.persist(path, {'first': False})
    assert json.loads(path.read_text()) == {'first': True}


def test_symlink_ancestor_refuses(bank):
    root, _, _ = bank
    link = root / 'alias'
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        restore.safe_path(link / 'receipt.json')


def test_raw_never_launches(tmp_path):
    path = tmp_path / 'raw.json'
    path.write_text(json.dumps({'schema': 'ddm_vr8.restore.v1', 'kind': 'sj1_parseback'}))
    with pytest.raises(ValueError, match='RAW_PARSEBACK_NOT_FRAME_RESUMABLE'):
        restore.run(path)


def test_frame_loop_resume_and_final_identity(bank, monkeypatch):
    root, output, _ = bank
    payload = b'abcd'
    source = {'expected_bytes': 2400, 'expected_sha256': hashlib.sha256(payload * 600).hexdigest()}
    calls = []
    def render(pair):
        calls.append(pair)
        return payload
    monkeypatch.setattr(restore, 'render_overlay', lambda source: (list(range(600)), 4, render))
    monkeypatch.setattr(restore, 'reserve', lambda *args: None)
    monkeypatch.setattr(restore.vr7, 'verify', lambda source: None)
    binding = {'consumer_store': 'test-recovery'}
    output.write_bytes(payload)
    record(root, binding, 0, payload)
    assert restore.restore_frames(source, output, root, binding) == 0
    assert calls == list(range(1, 600))
    assert output.read_bytes() == payload * 600
    certificate = json.loads((root / 'RESTORE_CERTIFICATE.json').read_text())
    assert certificate['complete'] is True
    calls.clear()
    assert restore.restore_frames(source, output, root, binding) == 0
    assert calls == []


def test_go_revoked_during_render_keeps_inflight_payload(bank, monkeypatch):
    root, output, go = bank
    source = {'expected_bytes': 2400}
    def render(pair):
        go.unlink()
        return b'abcd'
    monkeypatch.setattr(restore, 'render_overlay', lambda source: (list(range(600)), 4, render))
    monkeypatch.setattr(restore, 'reserve', lambda *args: None)
    with pytest.raises(ValueError, match='MAIN_GO'):
        restore.restore_frames(source, output, root, {})
    assert output.read_bytes() == b'abcd'
    assert len(list(root.glob('frame_*.json'))) == 1
