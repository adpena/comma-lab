"""File-custody controls only: no scorer or live SSD mutation."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from tac import artifact_moved as moved
from tac.artifact_moved_gate import violations


def certificate(src, dst, payload=b'payload', **changes):
    dst.write_bytes(payload)
    row = {'moved_to': str(dst), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
               'reason': 'fixture relocation', 'rebuildable_from': 'fixture generator'}
    row.update(changes)
    src.with_name(src.name + '.MOVED.json').write_text(json.dumps(row))
    return row


def test_existing_path_wins(tmp_path):
    src, dst = tmp_path/'old', tmp_path/'new'
    certificate(src, dst)
    src.write_bytes(b'original')
    assert moved.resolve(src).read_bytes() == b'original'


def test_resolution_and_cache_receipt(tmp_path, monkeypatch):
    src, dst = tmp_path/'old', tmp_path/'new'
    certificate(src, dst)
    receipt = []
    assert moved.resolve(src, receipt=receipt) == dst
    assert receipt[0]['cached'] is False
    monkeypatch.setattr(moved, '_sha', lambda _: pytest.fail('unchanged content rehashed'))
    assert moved.resolve(src, receipt=receipt) == dst
    assert receipt[1]['cached'] is True


def test_same_size_drift_invalidates_cache(tmp_path):
    src, dst = tmp_path/'old', tmp_path/'new'
    certificate(src, dst)
    moved.resolve(src)
    stamp = dst.stat()
    dst.write_bytes(b'PAYLOAD')
    os.utime(dst, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(moved.MovedArtifactError, match='DRIFT'):
        moved.resolve(src)


@pytest.mark.parametrize('changes', [{'bytes': 999}, {'bytes': True}, {'bytes': -1},
                                     {'sha256': 'a'*64}, {'sha256': 'g'*64}, {'sha256': ''},
                                     {'reason': ''}, {'rebuildable_from': None}, {'moved_to': 'relative'}])
def test_invalid_or_drifted_manifest_refuses(tmp_path, changes):
    src, dst = tmp_path/'old', tmp_path/'new'
    certificate(src, dst, **changes)
    with pytest.raises(moved.MovedArtifactError):
        moved.resolve(src)


def test_missing_refuses(tmp_path):
    with pytest.raises(moved.MovedArtifactError, match='RESOLUTION_REFUSED'):
        moved.resolve(tmp_path/'absent')


@pytest.mark.parametrize('raw', ['{broken', '[]', 'null'])
def test_malformed_json_refuses(tmp_path, raw):
    src = tmp_path/'old'
    src.with_name('old.MOVED.json').write_text(raw)
    with pytest.raises(moved.MovedArtifactError):
        moved.resolve(src)


def test_chain_validates_every_hop(tmp_path):
    a, b, c = (tmp_path/x for x in 'abc')
    certificate(a, b)
    certificate(b, c)
    b.unlink()
    assert moved.resolve(a) == c
    row = json.loads(a.with_name('a.MOVED.json').read_text())
    row['sha256'] = '0'*64
    a.with_name('a.MOVED.json').write_text(json.dumps(row))
    with pytest.raises(moved.MovedArtifactError, match='DRIFT'):
        moved.resolve(a)


def test_cycle_and_limit_refuse(tmp_path):
    a, b, c = (tmp_path/x for x in 'abc')
    certificate(a, b)
    certificate(b, c)
    b.unlink()
    with pytest.raises(moved.MovedArtifactError, match='CHAIN_LIMIT'):
        moved.resolve(a, max_depth=1)
    certificate(c, a)
    a.unlink()
    c.unlink()
    with pytest.raises(moved.MovedArtifactError, match='CYCLE'):
        moved.resolve(a)


def test_writer_preserves_bytes_and_schema(tmp_path):
    src, dst = tmp_path/'old', tmp_path/'new'
    src.write_bytes(b'payload')
    assert moved.move_with_manifest(src, dst, 'fixture relocation', 'fixture generator') == dst
    assert not src.exists()
    assert moved.resolve(src).read_bytes() == b'payload'
    assert set(json.loads(src.with_name('old.MOVED.json').read_text())) == {
        'moved_to', 'sha256', 'bytes', 'reason', 'rebuildable_from'}


@pytest.mark.parametrize('fault', ['copy', 'hash', 'publish', 'unlink'])
def test_writer_failures_keep_source(tmp_path, monkeypatch, fault):
    src, dst = tmp_path/'old', tmp_path/'new'
    src.write_bytes(b'payload')
    if fault == 'copy':
        monkeypatch.setattr(moved.shutil, 'copyfileobj', lambda *a: (_ for _ in ()).throw(OSError('copy failed')))
    elif fault == 'hash':
        real = moved._sha
        monkeypatch.setattr(moved, '_sha', lambda p: '0'*64 if p == dst else real(p))
    elif fault == 'publish':
        def reject_publish(a, b):
            assert dst.read_bytes() == src.read_bytes()
            assert not Path(b).exists()
            raise OSError('publish failed')
        monkeypatch.setattr(moved.os, 'replace', reject_publish)
    else:
        original = Path.unlink
        def reject_unlink(p, *a, **kw):
            if p == src:
                assert p.with_name(p.name+'.MOVED.json').is_file()
                raise OSError('unlink failed')
            return original(p, *a, **kw)
        monkeypatch.setattr(Path, 'unlink', reject_unlink)
    with pytest.raises((OSError, moved.MovedArtifactError)):
        moved.move_with_manifest(src, dst, 'fixture relocation', 'fixture generator')
    assert src.read_bytes() == b'payload'
    if fault != 'unlink':
        assert not src.with_name('old.MOVED.json').exists()


def test_writer_never_overwrites_destination(tmp_path):
    src, dst = tmp_path/'old', tmp_path/'new'
    src.write_bytes(b'payload')
    dst.write_bytes(b'other owner')
    with pytest.raises(moved.MovedArtifactError):
        moved.move_with_manifest(src, dst, 'fixture relocation', 'fixture generator')
    assert dst.read_bytes() == b'other owner'
    assert src.read_bytes() == b'payload'


@pytest.mark.parametrize('code,count', [
    ('Path("/Volumes/A/pact/ddm_other/raw").stat()', 1),
    ('pretend_resolve(Path("/Volumes/A/pact/ddm_other/raw")).stat()', 1),
    ('from other_module import resolve\nnp.load(resolve("/Volumes/A/pact/ddm_other/raw"))', 1),
    ('open("/Volumes/A/pact/ddm_other/raw", "rb")', 1),
    ('np.load(Path("/Volumes/A/pact/ddm_other/raw"))', 1),
    ('p = Path("/Volumes/A/pact/ddm_other/raw")\np.stat()', 1),
    ('root=Path("/Volumes/A/pact")\np=root/"ddm_other/raw"\nnp.load(p)', 1),
    ('Path("/Volumes/A/pact/ddm_test/raw").stat()', 0),
    ('Path("/ordinary/data").stat()', 0),
    ('open("/Volumes/A/pact/ddm_other/raw", "wb")', 0),
    ('from tac.artifact_moved import resolve as r\nr(Path("/Volumes/A/pact/ddm_other/raw")).stat()', 0),
    ('import tac.artifact_moved as m\nnp.load(m.resolve("/Volumes/A/pact/ddm_other/raw"))', 0),
    ('from tac.artifact_moved import resolve\np=Path("/Volumes/A/pact/ddm_other/raw")\np=resolve(p)\np.stat()', 0),
    ('from tac.artifact_moved import resolve\nresolve("unrelated")\nPath("/Volumes/A/pact/ddm_other/raw").stat()', 1),
    ('Path("/Volumes/A/pact/ddm_other/raw").stat() # MOVED_READ_OK: owner guarantees immutable fixture', 0),
    ('# MOVED_READ_OK: owner guarantees immutable fixture\nPath("/Volumes/A/pact/ddm_other/raw").stat()', 1),
    ('Path("/Volumes/A/pact/ddm_other/raw").stat() # MOVED_READ_OK: <rationale>', 1),
    ('note="MOVED_READ_OK: owner guarantees immutable fixture"\nPath("/Volumes/A/pact/ddm_other/raw").stat()', 1),
    ('p=Path("/Volumes/A/pact/ddm_other/raw")\ndef f(p):\n p.stat()', 0),
    ('p=Path("/Volumes/A/pact/ddm_other/raw")\ndef f():\n p.stat()', 1),
])
def test_gate_source_controls(tmp_path, code, count):
    p = tmp_path/'ddm_test_reader.py'
    p.write_text(code)
    assert len(violations(p)) == count


def test_preflight_warn_and_strict(tmp_path):
    from tac.preflight import PreflightError, check_no_bare_cross_arm_artifact_reads
    folder=tmp_path/'experiments'
    folder.mkdir()
    (folder/'ddm_test_reader.py').write_text('Path("/Volumes/A/pact/ddm_other/raw").stat()')
    assert len(check_no_bare_cross_arm_artifact_reads(repo_root=tmp_path)) == 1
    with pytest.raises(PreflightError, match='417'):
        check_no_bare_cross_arm_artifact_reads(repo_root=tmp_path, strict=True)


def test_file_fact_logical_path_unchanged(tmp_path):
    import importlib.util
    from pathlib import Path as _Path
    _spec = importlib.util.spec_from_file_location(
        "ddm_jg2_tail_reencode", _Path(__file__).resolve().parents[3] / "experiments/ddm_jg2_tail_reencode.py")
    jg2 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(jg2)  # MAIN rebase 2026-09-10: import by path so the test does not need experiments/ on sys.path
    src, dst = tmp_path/'old', tmp_path/'new'
    src.write_bytes(b'payload')
    before = jg2.file_fact(src)
    moved.move_with_manifest(src, dst, 'fixture relocation', 'fixture generator')
    assert jg2.file_fact(src) == before


def test_reclaim_moved_row_is_not_deletion_authority(tmp_path):
    from experiments import ddm_vr3_certified_raw_reclaim as vr3
    src, dst = tmp_path/'old', tmp_path/'new'
    row = certificate(src, dst)
    source = {'path': str(src), 'bytes': row['bytes'], 'sha256': row['sha256']}
    fact = vr3.moved_row(source)
    assert fact['verdict'] == 'MOVED_CERTIFIED'
    assert fact['freed_bytes'] == 0
    assert fact['move_certificate'][0]['sha256'] == row['sha256']
    assert vr3._stat_identity_blockers(src, source) == ['MOVED_CERTIFIED_RETAIN_ELSEWHERE']
    assert dst.read_bytes() == b'payload'
    source['sha256'] = '0'*64
    with pytest.raises(vr3.CertifyError, match='SOURCE_DRIFT'):
        vr3.moved_row(source)


def test_reclaim_verify_pin_returns_resolved_path(tmp_path):
    from experiments import ddm_vr3_certified_raw_reclaim as vr3
    src, dst = tmp_path/'old', tmp_path/'new'
    src.write_bytes(b'payload')
    pin = vr3.pinned_file(src)
    moved.move_with_manifest(src, dst, 'fixture relocation', 'fixture generator')
    assert vr3.pinned_file(src) == pin
    assert vr3.verify_pin(pin) == dst


def test_seal_survives_move_without_digest_redefinition(tmp_path):
    from tac import candidate_seal as seal
    from tac.tests.test_candidate_seal import _seal, _stage_candidate, _write_pointer
    runtime, _ = _stage_candidate(tmp_path)
    path = _seal(tmp_path, runtime)
    document = seal.load_seal(path)
    src, dst = tmp_path/'kept.bin', tmp_path/'cold.bin'
    src.write_bytes(b'payload')
    document['retained_payload_paths'] = [str(src)]
    document['seal_sha256'] = seal.compute_seal_sha256(document)
    seal.write_seal(document, path)
    before = path.read_bytes()
    pointer = _write_pointer(tmp_path)
    assert seal.validate_seal(path, pointer_path=pointer).verdict == seal.SEAL_VALID
    moved.move_with_manifest(src, dst, 'fixture relocation', 'fixture generator')
    assert seal.validate_seal(path, pointer_path=pointer).verdict == seal.SEAL_VALID
    assert path.read_bytes() == before
    dst.write_bytes(b'PAYLOAD')
    assert seal.validate_seal(path, pointer_path=pointer).verdict == seal.SEAL_FILE_MISSING


def test_plan_retained_classifies_move_without_absence(tmp_path, monkeypatch):
    from experiments import ddm_vr3_certified_raw_reclaim as vr3
    from experiments.tests.test_ddm_vr5_certified_raw_reclaim import make_plan
    src, args, apply_args = make_plan(tmp_path, monkeypatch)
    dst = tmp_path/'cold.raw'
    moved.move_with_manifest(src, dst, 'fixture relocation', 'fixture generator')
    vr3.plan_retained(args)
    row = vr3._load_ledger(args.output_ledger)[0]
    assert row['certificate_status'] == row['verdict'] == 'MOVED_CERTIFIED'
    assert row['certificate_complete'] is True
    assert row['freed_bytes'] == 0
    apply_args.expected_ledger_sha256 = vr3.sha256_file(args.output_ledger)
    with pytest.raises(vr3.CertifyError):
        vr3.apply(apply_args)
    assert dst.read_bytes() == b'decoded raw'


def test_move_after_reclaim_plan_preserves_destination(tmp_path, monkeypatch):
    from experiments import ddm_vr3_certified_raw_reclaim as vr3
    from experiments.tests.test_ddm_vr5_certified_raw_reclaim import make_plan
    src, args, apply_args = make_plan(tmp_path, monkeypatch)
    dst = tmp_path/'cold.raw'
    moved.move_with_manifest(src, dst, 'fixture relocation', 'fixture generator')
    monkeypatch.setattr(vr3, 'process_gate', lambda owner: {'visible': True})
    monkeypatch.setattr(vr3, 'lsof_plus_d', lambda path: {'open_descriptor_rows': []})
    assert vr3.apply(apply_args) == 3
    row = vr3._load_ledger(args.output_ledger)[0]
    assert 'MOVED_CERTIFIED_RETAIN_ELSEWHERE' in row['verdict']
    assert 'RAW_MISSING_OR_NOT_REGULAR' not in row['verdict']
    assert dst.read_bytes() == b'decoded raw'
