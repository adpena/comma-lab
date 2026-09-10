from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import ddm_vr3_certified_raw_reclaim as vr3
from experiments import ddm_vr7_rebuild as worker
from experiments import ddm_vr7_reproducer_descriptor as vr7
from experiments.ddm_vr7_rebuild import fact
from tac.candidate_seal import measure_runtime_digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return vr3.pinned_file(path)


def chain(tmp_path, monkeypatch):
    vertigo, ap = tmp_path / 'vertigo', tmp_path / 'ap'
    vertigo.mkdir()
    ap.mkdir()
    monkeypatch.setattr(vr3, 'VERTIGO_ROOT', vertigo)
    monkeypatch.setattr(vr3, 'AP_ROOT', ap)
    original = vertigo / 'ddm_sj1_pass5_price/parseback/0.raw'
    raw = ap / 'cold_store_sj1_bulk_20260910/ddm_sj1_pass5_price/parseback/0.raw'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'real bytes for deletion-safety unit control')
    f = fact(raw)
    moved = write(Path(str(original) + '.MOVED.json'), {
        'moved_to': str(raw), 'bytes': f['bytes'], 'sha256': f['sha256']})
    runtime = vertigo / 'retained_runtime'
    runtime.mkdir()
    archive = runtime / 'archive.zip'
    archive.write_bytes(b'unit archive retained')
    code = tmp_path / 'rebuild_source.py'
    code.write_text('retained rebuild source fixture')
    seal = write(tmp_path / 'SEAL.json', {
        'archive': fact(archive), 'runtime': {'path': str(runtime),
                                           'sha256': measure_runtime_digest(runtime).sha256}})
    receipt = write(original.parent / 'PARSEBACK_RESULT.json', {
        'rendered_raw': {**f, 'path': str(original)}})
    config_path = tmp_path / 'config.json'
    write(config_path, {'pins': [fact(code)], 'argv': ['python', str(code)],
                        'original': str(raw), 'expected_sha256': f['sha256'],
                        'expected_bytes': f['bytes']})
    proof = write(tmp_path / 'REBUILD_CERTIFICATE.json', {
        'kind': 'sj1_parseback', 'complete': True, 'twin_hash_equal': True,
        'original': f, 'rebuilt': {**f, 'path': str(tmp_path / 'own_rebuilt.raw')},
        'argv': ['python', str(code)], 'config': fact(config_path)})
    descriptor = {
        'schema': vr7.SCHEMA, 'kind': 'sj1_parseback', 'target_path': str(raw),
        'original_path': str(original), 'expected_sha256': f['sha256'],
        'expected_bytes': f['bytes'], 'rebuild_argv': [
            'python', '-B', 'experiments/ddm_sj1_joint_admission.py', 'parseback',
            '--runtime', str(runtime), '--out-dir', str(vertigo / 'ddm_vr7_certify/unit'),
            '--threads', '4'],
        'cwd': str(tmp_path), 'moved_manifest': moved, 'seal': seal,
        'seal_archive': fact(archive), 'archive': fact(archive),
        'runtime': {'root': str(runtime), 'files': [fact(archive)]},
        'output_receipt': receipt, 'output_receipt_keys': ['rendered_raw'],
        'receipt_output_path': str(original), 'code_pins': [fact(code)],
        'proof': proof, 'inheritance': {'representative_original': str(raw),
                                     'kind': 'sj1_parseback', 'rationale': 'same instance'},
    }
    pin = write(tmp_path / 'descriptor.json', descriptor)
    stat = raw.stat()
    source = {'path': str(raw), 'owner': 'MAIN / ddm_sj1',
              'certificate_status': vr3.RETAINED_STATUS,
              'reproducer_descriptor': pin, 'bytes': stat.st_size,
              'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino, 'device': stat.st_dev,
              'historical_sha256': f['sha256'],
              'reproducer': vr7.certify(pin, raw, f['sha256'], vr3)}
    rehash = {**f, 'present': True}
    memo = tmp_path / 'closure.md'
    memo.write_text('Finished bulk only; retain every archive and receipt.')
    closure = {'family': 'ddm_sj1', 'disposition': 'REBUILDABLE_FINISHED_BULK',
               'memo': vr3.pinned_file(memo), 'verdict_quote': memo.read_text()}
    return raw, descriptor, source, rehash, closure


def test_real_files_admitted_and_legacy_protection_still_holds(tmp_path, monkeypatch):
    raw, descriptor, source, rehash, closure = chain(tmp_path, monkeypatch)
    assert vr3.certify_retained(source, rehash, closure)['kind'] == 'sj1_parseback'
    assert vr3._forbidden_target_reason(raw) == 'LIVE_POINTER_TREE_PROTECTED'
    assert vr3._forbidden_target_reason(raw, descriptor) is None


@pytest.mark.parametrize('change', ['destination', 'sha', 'seal', 'code', 'proof_kind',
                                    'proof_false', 'inheritance', 'runtime_extra', 'source_reappears'])
def test_drift_blocks_and_preserves_raw(tmp_path, monkeypatch, change):
    raw, descriptor, source, rehash, closure = chain(tmp_path, monkeypatch)
    if change in {'destination', 'sha'}:
        moved = Path(descriptor['moved_manifest']['path'])
        value = json.loads(moved.read_text())
        value['moved_to' if change == 'destination' else 'sha256'] = 'wrong'
        moved.write_text(json.dumps(value))
    elif change == 'seal':
        Path(descriptor['seal']['path']).write_text('{}')
    elif change == 'code':
        Path(descriptor['code_pins'][0]['path']).write_text('changed computation')
    elif change.startswith('proof_'):
        p = Path(descriptor['proof']['path'])
        value = json.loads(p.read_text())
        value['kind' if change == 'proof_kind' else 'twin_hash_equal'] = (
            'sj1_overlay' if change == 'proof_kind' else False)
        descriptor['proof'] = write(p, value)
    elif change == 'inheritance':
        descriptor['inheritance']['kind'] = 'sj1_overlay'
    elif change == 'runtime_extra':
        (Path(descriptor['runtime']['root']) / 'new_payload.bin').write_bytes(b'new')
    else:
        Path(descriptor['original_path']).write_bytes(b'reappeared')
    source['reproducer_descriptor'] = write(Path(source['reproducer_descriptor']['path']), descriptor)
    with pytest.raises(vr3.CertifyError):
        vr3.certify_retained(source, rehash, closure)
    assert raw.read_bytes().startswith(b'real bytes')


@pytest.mark.parametrize('relative', ['rebase_move40/move40/trace/frames/frame_0000.npz',
                                     'move39/candidate_runtime/archive.zip',
                                     'move39/trace/RESULT.json',
                                     'move39/trace/checkpoints/stage_0600.npz'])
def test_tc3_protected_targets_never_admit(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(vr3, 'VERTIGO_ROOT', tmp_path)
    p = tmp_path / 'ddm_tc3_lane_predictor_seal' / relative
    p.parent.mkdir(parents=True)
    p.write_bytes(b'keep')
    descriptor = {'schema': vr7.SCHEMA, 'target_path': str(p), 'kind': 'tc3_trace'}
    assert vr3._forbidden_target_reason(p, descriptor)
    assert p.read_bytes() == b'keep'


@pytest.mark.parametrize('fault', ['none', 'archive_drift', 'observation_exclusion'])
def test_native_plan_and_apply_revalidate_descriptor(tmp_path, monkeypatch, fault):
    raw, descriptor, source, rehash, closure = chain(tmp_path, monkeypatch)
    monkeypatch.setattr(vr3, 'repo_reference_hits', lambda aliases, *a, **k: {p: [] for p in aliases})
    monkeypatch.setattr(vr3, 'process_gate', lambda family: {'visible': True})
    monkeypatch.setattr(vr3, 'lsof_plus_d', lambda path: {'open_descriptor_rows': []})
    src, hashes, closures = [tmp_path / n for n in ('source.jsonl', 'hashes.jsonl', 'closures.json')]
    vr3.atomic_jsonl(src, [source])
    vr3.atomic_jsonl(hashes, [rehash])
    observations = [descriptor['moved_manifest']] if fault == 'observation_exclusion' else []
    write(closures, {'observation_files': observations, 'closures': {'ddm_sj1': closure}})
    ledger = tmp_path / 'plan.jsonl'
    vr3.plan_retained(argparse.Namespace(source_ledger=src, rehash_ledger=hashes, closures=closures,
                                        repo_root=tmp_path, output_ledger=ledger))
    row = json.loads(ledger.read_text())
    assert row['planned_verdict'] == 'DELETABLE'
    # A still-active redirect must refuse even before expensive archive revalidation.
    if fault == 'archive_drift':
        Path(descriptor['archive']['path']).write_bytes(b'drifted archive')
    def expensive_revalidation_must_not_run(row):
        raise AssertionError('active redirect must block before expensive revalidation')
    monkeypatch.setattr(vr3, 'retained_revalidation', expensive_revalidation_must_not_run)
    vr3.apply(argparse.Namespace(ledger=ledger, target_bytes=raw.stat().st_size,
                                expected_ledger_sha256=vr3.sha256_file(ledger),
                                journal=tmp_path / 'journal.jsonl', repo_root=tmp_path))
    assert raw.exists()
    assert 'ACTIVE_MOVED_REDIRECT_TARGET' in json.loads(ledger.read_text())['verdict']
    assert Path(descriptor['archive']['path']).exists()
    assert Path(descriptor['seal']['path']).exists()
    assert Path(descriptor['moved_manifest']['path']).exists()
    predelete = next(json.loads(line) for line in (tmp_path / 'journal.jsonl').read_text().splitlines()
                     if json.loads(line)['phase'] == 'PRE_DELETE')
    assert predelete['raw_sha256_verified_current'] is False
    assert predelete['refreshed_reproducer'] is None


def test_descriptor_status_is_not_a_fake_legacy_certificate(tmp_path, monkeypatch):
    raw, descriptor, source, rehash, closure = chain(tmp_path, monkeypatch)
    source['certificate_status'] = 'DESCRIPTOR_VALIDATION_OWED'
    source['reproducer'] = None
    assert vr3.certify_retained(source, rehash, closure)['archive_sha256'] == descriptor['archive']['sha256']
    assert raw.exists()


def test_external_observation_is_exact_pinned_text_only(tmp_path, monkeypatch):
    repo, ssd = tmp_path / 'repo', tmp_path / 'ssd'
    repo.mkdir()
    ssd.mkdir()
    monkeypatch.setattr(vr3, 'VERTIGO_ROOT', ssd)
    receipt = write(ssd / 'retained/RESULT.json', {'completed': True})
    assert vr3.observation_exclusions([receipt], repo) == [receipt['path']]
    program = ssd / 'retained/consumer.py'
    program.write_text('consume_raw()')
    with pytest.raises(vr3.CertifyError, match='retained text receipt'):
        vr3.observation_exclusions([vr3.pinned_file(program)], repo)
    Path(receipt['path']).write_text('{}')
    with pytest.raises(vr3.CertifyError, match='PIN_SHA256_DRIFT'):
        vr3.observation_exclusions([receipt], repo)


def test_wrong_rebuild_runtime_is_refused(tmp_path, monkeypatch):
    raw, descriptor, source, rehash, closure = chain(tmp_path, monkeypatch)
    descriptor['rebuild_argv'][5] = '/wrong/runtime'
    source['reproducer_descriptor'] = write(Path(source['reproducer_descriptor']['path']), descriptor)
    with pytest.raises(vr3.CertifyError, match='PARSEBACK_ARGV_INPUT_DRIFT'):
        vr3.certify_retained(source, rehash, closure)
    assert raw.exists()


def test_restore_config_dependency_drift_refuses_before_command(tmp_path):
    dependency = tmp_path / 'replay.py'
    dependency.write_text('original numerical implementation')
    config = write(tmp_path / 'restore.json', {'pins': [fact(dependency)]})
    descriptor = {'kind': 'tc3_trace', 'rebuild_config': config,
                  'rebuild_argv': ['python']}
    dependency.write_text('changed implementation')
    with pytest.raises(vr3.CertifyError, match='PIN_SHA256_DRIFT'):
        vr7._command_binding(descriptor, vr3)


def test_bundle_entry_admission_and_post_cache_drift(tmp_path, monkeypatch):
    raw, descriptor, source, rehash, closure = chain(tmp_path, monkeypatch)
    bundle = tmp_path / 'descriptors.json'
    pin = write(bundle, {'schema': 'ddm_vr7.descriptor_bundle.v1',
                         'descriptors': {str(raw): descriptor}})
    source['reproducer_descriptor'] = {**pin, 'entry': str(raw)}
    source['reproducer'] = None
    assert vr3.certify_retained(source, rehash, closure)['kind'] == 'sj1_parseback'
    bundle.write_text('{}')
    with pytest.raises(vr3.CertifyError, match='PIN_SHA256_DRIFT'):
        vr3.certify_retained(source, rehash, closure)
    assert raw.exists()


def test_symlink_cannot_enter_descriptor_exception(tmp_path, monkeypatch):
    raw, descriptor, source, rehash, closure = chain(tmp_path, monkeypatch)
    saved = raw.with_suffix('.saved')
    raw.rename(saved)
    raw.symlink_to(saved)
    with pytest.raises(vr3.CertifyError, match='NOT_REGULAR'):
        vr3.certify_retained(source, rehash, closure)
    assert saved.read_bytes().startswith(b'real bytes')


@pytest.mark.parametrize('script', ['ddm_vr3_certified_raw_reclaim.py',
                                   'ddm_vr7_overlay_rebuild.py', 'ddm_vr7_tc3_replay.py'])
def test_actual_direct_script_entrypoint(script):
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-B', str(repo / 'experiments' / script), '--help'],
                            cwd=repo, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def worker_config(tmp_path, monkeypatch):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(worker, 'SCRATCH', scratch)
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    archive = runtime / 'archive.zip'
    archive.write_bytes(b'retained archive')
    original = tmp_path / 'original.raw'
    original.write_bytes(b'already completed decode')
    root = scratch / 'unit'
    output = root / 'output'
    output.mkdir(parents=True)
    (output / '0.raw').write_bytes(original.read_bytes())
    config_path = tmp_path / 'config.json'
    measured = fact(original)
    config = {'runtime': str(runtime), 'runtime_files': [fact(archive)], 'pins': [],
              'original': str(original), 'expected_bytes': measured['bytes'],
              'expected_sha256': measured['sha256'], 'scratch': str(root), 'cwd': str(tmp_path),
              'argv': [sys.executable, '-B', 'experiments/ddm_sj1_joint_admission.py', 'parseback',
                       '--runtime', str(runtime), '--out-dir', str(output), '--threads', '4']}
    worker.atomic(config_path, config)
    worker.atomic(root / 'BINDING.json', fact(config_path))
    monkeypatch.setattr(worker.subprocess, 'run', lambda *a, **k: pytest.fail('must reuse completed raw'))
    return config_path, original, output


def test_worker_recovers_after_decode_before_certificate(tmp_path, monkeypatch):
    config_path, original, output = worker_config(tmp_path, monkeypatch)
    assert worker.run(config_path) == 0
    proof = json.loads((output.parent / 'REBUILD_CERTIFICATE.json').read_text())
    assert proof['decode_reused_after_interrupt'] and proof['cleanup_complete']
    assert not (output / '0.raw').exists()
    assert original.read_bytes() == b'already completed decode'
    assert worker.run(config_path) == 0


def test_worker_mismatched_completed_output_is_retained(tmp_path, monkeypatch):
    config_path, original, output = worker_config(tmp_path, monkeypatch)
    (output / '0.raw').write_bytes(b'wrong output')
    with pytest.raises(ValueError, match='completed decode output differs'):
        worker.run(config_path)
    assert (output / '0.raw').read_bytes() == b'wrong output'
    assert original.exists()


def test_worker_resumes_partially_completed_certified_cleanup(tmp_path, monkeypatch):
    config_path, original, output = worker_config(tmp_path, monkeypatch)
    created = [fact(output / '0.raw')]
    extra = output / 'tokens.u8'
    extra.write_bytes(b'rebuildable token checkpoint')
    created.append(fact(extra))
    result = {'complete': True, 'twin_hash_equal': True, 'config': fact(config_path),
              'original': fact(original), 'created_files': created, 'cleanup_complete': False}
    worker.atomic(output.parent / 'REBUILD_CERTIFICATE.json', result)
    (output / '0.raw').unlink()  # Simulate the first certified unlink before a crash.
    assert worker.run(config_path) == 0
    assert not extra.exists()
    assert original.exists()


@pytest.mark.parametrize('kind', ['sj1_parseback', 'sj1_overlay', 'tc3_trace', 'future_family'])
def test_active_redirect_guard_is_family_independent(tmp_path, kind):
    raw = tmp_path / 'payload.raw'
    raw.write_bytes(b'keep live destination')
    pin = write(tmp_path / 'MOVED.json', {'moved_to': str(raw),
                'bytes': raw.stat().st_size, 'sha256': vr3.sha256_file(raw),
                'status': 'HISTORICAL_OBSERVATION'})
    assert 'ACTIVE_MOVED_REDIRECT_TARGET' in vr3._active_moved_redirect_blockers(
        raw, {'kind': kind, 'moved_manifest': pin})[0]
    assert raw.exists()


@pytest.mark.parametrize('fault', ['missing', 'drift', 'malformed', 'non_object', 'relative',
                                  'bad_sha', 'duplicate_key', 'bad_pin'])
def test_unverifiable_redirect_guard_fails_closed(tmp_path, fault):
    raw = tmp_path / 'payload.raw'
    raw.write_bytes(b'keep live destination')
    manifest = tmp_path / 'MOVED.json'
    value = {'moved_to': str(raw), 'bytes': raw.stat().st_size, 'sha256': vr3.sha256_file(raw)}
    pin = write(manifest, value)
    if fault == 'missing':
        manifest.unlink()
    elif fault == 'drift':
        manifest.write_text('{}')
    elif fault == 'bad_pin':
        pin = None
    else:
        text = {'malformed': '{', 'non_object': '[]',
                'relative': json.dumps(dict(value, moved_to='payload.raw')),
                'bad_sha': json.dumps(dict(value, sha256='bad')),
                'duplicate_key': '{"moved_to":"/first","moved_to":"/second"}'}[fault]
        manifest.write_text(text)
        pin = vr3.pinned_file(manifest)
    assert 'MOVED_REDIRECT_REVALIDATION_REFUSED' in vr3._active_moved_redirect_blockers(
        raw, {'moved_manifest': pin})[0]
    assert raw.exists()


def test_active_redirect_guard_preserves_normalized_alias(tmp_path):
    raw = tmp_path / 'payload.raw'
    raw.write_bytes(b'keep')
    child = tmp_path / 'child'
    child.mkdir()
    alias = str(child / '..' / raw.name)
    pin = write(tmp_path / 'MOVED.json', {'moved_to': alias, 'bytes': 4,
                                        'sha256': vr3.sha256_file(raw)})
    assert vr3._active_moved_redirect_blockers(raw, {'moved_manifest': pin})
    assert vr3._active_moved_redirect_blockers(raw, None) == []
    assert vr3._active_moved_redirect_blockers(raw, {'kind': 'unrelated'}) == []


def test_rebound_redirect_does_not_alone_block_old_target(tmp_path):
    raw = tmp_path / 'old.raw'
    raw.write_bytes(b'old')
    pin = write(tmp_path / 'MOVED.json', {'moved_to': str(tmp_path / 'new.raw'),
                                        'bytes': 3, 'sha256': vr3.sha256_file(raw)})
    assert vr3._active_moved_redirect_blockers(raw, {'moved_manifest': pin}) == []
    # Passing this guard still requires all existing reproducer/admission checks.
    assert raw.exists()
