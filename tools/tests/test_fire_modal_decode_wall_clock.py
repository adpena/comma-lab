"""Fire-path timing gate controls; synthetic seal fixtures, zero paid dispatch."""
from __future__ import annotations

import json

import pytest

from tac.candidate_seal import compute_seal_sha256, load_seal, validate_seal
from tac.tests.test_candidate_seal import (
    _forbid_subprocess,
    _load_fire_tool,
    _seal,
    _stage_candidate,
    _write_pointer,
)


@pytest.mark.parametrize('dry_run', [True, False])
def test_missing_timing_refuses_before_any_subprocess(tmp_path, monkeypatch, dry_run):
    runtime, _ = _stage_candidate(tmp_path)
    seal = _seal(tmp_path, runtime, tolerance=1.0)
    doc = load_seal(seal)
    doc.pop('decode_wall_clock')
    doc['schema'] = 'candidate_seal.v1'
    doc['seal_sha256'] = compute_seal_sha256(doc)
    seal.write_text(json.dumps(doc))
    pointer = _write_pointer(tmp_path)
    assert validate_seal(seal, pointer_path=pointer).ok
    fire = _load_fire_tool()
    _forbid_subprocess(monkeypatch, fire)
    monkeypatch.setattr(fire, 'validate_seal',
                        lambda path, **kw: validate_seal(path, pointer_path=pointer, **kw))
    output = tmp_path / 'fire_output'
    argv = ['--seal', str(seal), '--output-dir', str(output), '--lane-id', 'lane_test',
            '--instance-job-id', 'job_test', '--single-axis-waiver-reason',
            'synthetic no-dispatch timing contract test']
    if dry_run:
        argv.append('--dry-run')
    assert fire.main(argv) == 7
    refusal = json.loads((output / 'FIRE_REFUSED.json').read_text())
    assert refusal['seal_refusal']['seal_validation']['verdict'] == 'SEAL_DECODE_WALL_CLOCK_MISSING'
    assert seal.with_name(seal.name + '.REFUSED.json').exists()
    assert not (output / 'FIRE_MANIFEST.json').exists()


def test_timing_waiver_is_not_hidden_in_public_smoke_waiver(tmp_path, monkeypatch):
    runtime, _ = _stage_candidate(tmp_path)
    seal = _seal(tmp_path, runtime, tolerance=1.0)
    doc = load_seal(seal)
    doc.pop('decode_wall_clock')
    doc.pop('public_entrypoint_smoke')
    doc['schema'] = 'candidate_seal.v1'
    doc['seal_sha256'] = compute_seal_sha256(doc)
    seal.write_text(json.dumps(doc))
    pointer = _write_pointer(tmp_path)
    fire = _load_fire_tool()
    _forbid_subprocess(monkeypatch, fire)
    monkeypatch.setattr(fire, 'validate_seal',
                        lambda path, **kw: validate_seal(path, pointer_path=pointer, **kw))
    assert fire.main(['--seal', str(seal), '--output-dir', str(tmp_path / 'out'),
        '--lane-id', 'lane_test', '--instance-job-id', 'job_test', '--single-axis-waiver-reason',
        'synthetic no-dispatch timing test', '--allow-seal-without-public-smoke',
        'existing scored custody replay does not authorize missing timing']) == 7

    refusal = json.loads((tmp_path / 'out/FIRE_REFUSED.json').read_text())
    assert refusal['seal_refusal']['seal_validation']['verdict'] == 'SEAL_DECODE_WALL_CLOCK_MISSING'
    assert not (tmp_path / 'out/FIRE_MANIFEST.json').exists()
