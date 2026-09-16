"""Retain source-bound decode coverage; certify raw identity before reclaiming it.

The completed v1 measurement is preserved separately. New runs use v2 bindings;
completion of token frames is observed, never inferred from a planned sample.
This external proof runner is not part of the submission receiver.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

SOURCE = Path('/Volumes/APDataStore/pact/ddm_pd6/candidate2/candidate_runtime')
ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs1')
ARCHIVE_SHA = 'aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957'
RAW_SHA = '8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b'
RAW_BYTES = 3662409600
SEED = 20260916


def fact(path):
    path = path.resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def completed_frame_line():
    """Locate the sealed decoder's post-frame statement without importing it."""
    tree = ast.parse((SOURCE / 'runtime/residual_archive.py').read_text())
    lines = [node.lineno for function in tree.body
             if isinstance(function, ast.FunctionDef)
             and function.name == 'decode_production_tokens'
             for node in ast.walk(function) if isinstance(node, ast.Assign)
             and isinstance(node.value, ast.Name) and node.value.id == 'current'
             and any(isinstance(target, ast.Name) and target.id == 'previous'
                     for target in node.targets)]
    if len(lines) != 1:
        raise ValueError('sealed decoder completion anchor is not unique')
    return lines[0]


def verify_coverage(value, source):
    """Reject coverage for another source or impossible stored line numbers."""
    if value.get('source') != source:
        raise ValueError('coverage source binding mismatch')
    for name, lines in value.get('lines', {}).items():
        if name not in source or not name.endswith('.py'):
            raise ValueError('coverage path is outside the bound Python sources')
        count = len((SOURCE / name).read_text().splitlines())
        if lines != sorted(set(lines)) or any(type(n) is not int or not 1 <= n <= count for n in lines):
            raise ValueError('invalid persisted coverage line numbers')


def cleanup_certified_raw(run, certificate):
    """Also finish a crash after the certificate commit but before raw removal."""
    destination = run / '0.raw'
    expected = dict(path=str(destination), bytes=RAW_BYTES, sha256=RAW_SHA)
    if certificate.get('raw') != expected:
        raise ValueError('certificate raw binding mismatch; retain bytes')
    removed = destination.exists()
    if removed:
        if fact(destination) != expected:
            raise ValueError('raw differs from certificate; retain bytes')
        destination.unlink()
    atomic_json(run / 'RAW_CLEANUP.json', dict(
        schema='mrs1_certified_raw_cleanup.v1', raw=expected,
        certificate=fact(run / 'IDENTITY.json'), removed_by_this_invocation=removed,
        absent_after_cleanup=not destination.exists(), score_claim=False))


def reuse_legacy_completion(run, source, archive):
    """Validate the immutable v1 evidence without relabelling it as v2."""
    inputs = json.loads((run / 'INPUTS.json').read_text())
    coverage = json.loads((run / 'COVERAGE.json').read_text())
    certificate = json.loads((run / 'IDENTITY.json').read_text())
    decode = json.loads((run / 'DECODE_REPORT.json').read_text())['report']
    provenance = json.loads((ROOT / 'trace_runner_v1_provenance.json').read_text())
    if fact(ROOT / 'trace_runner_v1.py') != provenance['preserved_copy']:
        raise ValueError('preserved v1 runner changed')
    for name, expected in provenance['baseline_evidence'].items():
        if fact(run / name) != expected:
            raise ValueError('preserved baseline evidence changed')
    for value in (inputs, certificate):
        if value.get('source') != source or value.get('archive') != archive or value.get('seed') != SEED:
            raise ValueError('legacy completion input binding mismatch')
    verify_coverage(coverage, source)
    if coverage.get('schema') != 'mrs1_dynamic_line_coverage.v1' or coverage.get('seed') != SEED:
        raise ValueError('legacy coverage schema or seed mismatch')
    if (certificate.get('schema') != 'mrs1_rebuildable_raw_certificate.v1'
            or len(certificate.get('pair_sha256', [])) != 600
            or decode.get('pair_count') != 600
            or decode.get('archive_sha256') != ARCHIVE_SHA
            or decode.get('raw_sha256') != RAW_SHA):
        raise ValueError('legacy n600 completion proof mismatch')
    for value in certificate['library_facts'].values():
        if fact(Path(value['path'])) != value:
            raise ValueError('legacy native library binding mismatch')
    cleanup_certified_raw(run, certificate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', type=Path, required=True)
    args = parser.parse_args()
    run = args.resume_from.resolve()
    if not run.is_relative_to(ROOT):
        raise ValueError('run must remain inside the charter store')
    run.mkdir(parents=True, exist_ok=True)
    source_facts = {str(p.relative_to(SOURCE)): fact(p) for p in SOURCE.rglob('*')
                    if p.is_file() and p.suffix in {'.py', '.c', '.sh'}}
    archive = fact(SOURCE / 'archive.zip')
    if archive['sha256'] != ARCHIVE_SHA or archive['bytes'] != 179286:
        raise ValueError('sealed archive identity mismatch')
    inputs_path = run / 'INPUTS.json'
    if inputs_path.exists() and json.loads(inputs_path.read_text()).get('schema') != 'mrs1_trace_inputs.v2':
        if not (run / 'IDENTITY.json').exists():
            raise ValueError('incomplete legacy run: preserve it and use a new v2 directory')
        reuse_legacy_completion(run, source_facts, archive)
        print(json.dumps(dict(status='complete_v1_reused', receipt=str(run / 'IDENTITY.json'))), flush=True)
        return
    build = run / 'build'
    build.mkdir(exist_ok=True)
    commands = {}
    for variable, relative, flags in [
        ('CPR1_RC64_LIBRARY', 'runtime/entropy/rc64_backend.c', []),
        ('F26_CORRECTOR_NATIVE_LIBRARY', 'runtime/f26_corrector_native.c', ['-ffp-contract=off', '-fno-fast-math', '-lm']),
        ('RLC1_GEOMETRY_LIBRARY', 'runtime/rlc1_geometry.c', []),
    ]:
        target = build / (Path(relative).stem + '.so')
        command = ['/usr/bin/cc', '-O3', '-std=c11', '-shared', '-fPIC', str(SOURCE / relative), *flags, '-o', str(target)]
        if not target.exists():
            subprocess.run(command, check=True)
        commands[variable] = command
        os.environ[variable] = str(target)
    os.environ['TC1_RECEIVER_CHECKPOINT_DIR'] = str(run / 'token_stages')
    os.environ['F26_TOKEN_DECODER'] = 'python'
    for name in ['F26_ADVISORY_DECODE_CACHE_ROOT', 'F26_ADVISORY_RENDER_WORKERS',
                 'F26_ADVISORY_PAIR_LIMIT', 'TC1_RECEIVER_STOP_AFTER']:
        os.environ.pop(name, None)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    binding = dict(archive=archive, source=source_facts, seed=SEED, runner=fact(Path(__file__)),
        library_facts={key: fact(Path(os.environ[key])) for key in commands},
        python=sys.version, numpy=np.__version__, torch=torch.__version__, threads=4,
        thread_environment={key: os.environ.get(key) for key in
            ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']})
    if inputs_path.exists():
        if json.loads(inputs_path.read_text()).get('binding') != binding:
            raise ValueError('resume input or runner binding changed')
    else:
        if any((run / name).exists() for name in
               ['COVERAGE.json', 'IDENTITY.json', 'token_stages', 'stage_checkpoints']):
            raise ValueError('refusing unbound pre-existing run artifacts')
        atomic_json(inputs_path, dict(schema='mrs1_trace_inputs.v2', binding=binding,
            argv=sys.argv, build_commands=commands, score_claim=False, axis='[macOS-CPU advisory]',
            expected_raw_sha256=RAW_SHA, retention_cap_bytes=2 * 1024**3,
            cleanup='Only certified matching raw is removed; all checkpoints retained.'))
    coverage_path = run / 'COVERAGE.json'
    prior = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    if prior:
        verify_coverage(prior, source_facts)
        if prior.get('schema') != 'mrs1_dynamic_line_coverage.v2' or prior.get('binding') != binding:
            raise ValueError('coverage run binding mismatch')
    hits = {p: set(lines) for p, lines in prior.get('lines', {}).items()}
    stored_frames = prior.get('completed_traced_token_pairs', [])
    if stored_frames != sorted(set(stored_frames)) or any(type(n) is not int or not 0 <= n < 600 for n in stored_frames):
        raise ValueError('invalid completed token pair coverage')
    completed = set(stored_frames)
    certificate_path = run / 'IDENTITY.json'
    if certificate_path.exists():
        certificate = json.loads(certificate_path.read_text())
        if (certificate.get('schema') != 'mrs1_rebuildable_raw_certificate.v2'
                or certificate.get('binding') != binding or not prior.get('complete')
                or completed != set(range(600)) or len(certificate.get('pair_sha256', [])) != 600
                or certificate.get('coverage') != fact(coverage_path)
                or certificate.get('decode_report') != fact(run / 'DECODE_REPORT.json')):
            raise ValueError('completed certificate binding mismatch')
        cleanup_certified_raw(run, certificate)
        print(json.dumps(dict(status='complete_v2_reused', receipt=str(certificate_path))), flush=True)
        return
    root_prefix = str(SOURCE) + '/'
    completion_line = completed_frame_line()
    selected = sorted(np.random.default_rng(SEED).choice(600, 32, replace=False).tolist())
    last_save = time.monotonic()

    def save_coverage(complete=False):
        atomic_json(coverage_path, dict(schema='mrs1_dynamic_line_coverage.v2',
            binding=binding, source=source_facts, seed=SEED, planned_selected_pairs=selected,
            completed_traced_token_pairs=sorted(completed),
            traced_selected_pairs=sorted(completed.intersection(selected)), complete=complete,
            scope='Python line events only; token-pair completion is observed; native C is not traced.',
            lines={p: sorted(v) for p, v in hits.items()}))

    def trace(frame, event, arg):
        nonlocal last_save
        name = frame.f_code.co_filename
        if not name.startswith(root_prefix):
            return None
        if event == 'line':
            relative = name[len(root_prefix):]
            hits.setdefault(relative, set()).add(frame.f_lineno)
            finished_frame = relative == 'runtime/residual_archive.py' and frame.f_lineno == completion_line
            if finished_frame:
                completed.add(int(frame.f_locals['frame']))
            if time.monotonic() - last_save > 30 or finished_frame and len(completed) % 25 == 0:
                save_coverage()
                last_save = time.monotonic()
        return trace

    if shutil.disk_usage(run).free < RAW_BYTES + 512 * 1024**2:
        raise ValueError('insufficient SSD space for raw plus checkpoint headroom')
    destination = run / '0.raw'
    started = time.monotonic()
    sys.settrace(trace)
    try:
        from runtime.f26_inflate import inflate_archive
        with torch.inference_mode():
            report = inflate_archive(SOURCE / 'archive.zip', destination,
                renderer_dir=SOURCE / 'cpr1', device_name='cpu', num_threads=4,
                checkpoint_dir=run / 'stage_checkpoints')
    finally:
        sys.settrace(None)
        save_coverage()
    report_path = run / 'DECODE_REPORT.json'
    atomic_json(report_path, dict(binding=binding, report=report,
        traced_wall_seconds=time.monotonic() - started))
    raw = fact(destination)
    if raw['sha256'] != RAW_SHA or raw['bytes'] != RAW_BYTES:
        raise ValueError('baseline raw differs; retain bytes and fail closed')
    if completed != set(range(600)):
        raise ValueError('full token coverage is unproven; retain raw and fail closed')
    with destination.open('rb') as stream:
        pair_shas = [hashlib.sha256(stream.read(RAW_BYTES // 600)).hexdigest() for _ in range(600)]
    save_coverage(complete=True)
    certificate = dict(schema='mrs1_rebuildable_raw_certificate.v2', raw=raw,
        pair_sha256=pair_shas, binding=binding, coverage=fact(coverage_path),
        decode_report=fact(report_path),
        command=[sys.executable, str(Path(__file__).resolve()), '--resume-from', str(run)],
        identity='600/600 pairs represented by matching full raw SHA',
        score_claim=False, promotable=False, axis='[macOS-CPU advisory]',
        reason='Rebuildable raw; full identity proven and recorded before removal')
    atomic_json(certificate_path, certificate)
    cleanup_certified_raw(run, certificate)
    print(json.dumps(dict(status='complete', receipt=str(certificate_path),
        coverage=str(coverage_path))), flush=True)


if __name__ == '__main__':
    main()
