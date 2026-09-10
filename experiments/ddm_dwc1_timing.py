"""Time copied public receivers, retaining payloads and resumable stage state.

Scorer-free macOS CPU proxy only. Native shell startup and full rendering are timed.
Concurrency visibility failure is retained and prevents a margin PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.dont_write_bytecode = True
from tac.candidate_seal import measure_runtime_digest

SOURCES = {
    'tc3': Path('/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40/candidate_runtime'),
    'tc4': Path('/Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41/candidate_runtime'),
}
ROOT = Path('/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock')
THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def fact(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def load_snapshot():
    try:
        result = subprocess.run(['ps', '-axo', 'pid,ppid,%cpu,command'], capture_output=True,
                                text=True, timeout=10)
        rows, code, error = result.stdout.splitlines(), result.returncode, result.stderr
    except OSError as exc:
        rows, code, error = [], None, repr(exc)
    return {'load_average': list(os.getloadavg()), 'process_inventory_returncode': code,
            'process_inventory': rows, 'process_inventory_error': error,
            'concurrent_process_count': len(rows) - 1 if code == 0 else None,
            'quiesced': False, 'normalization': None}


def hygiene(root):
    paths = [p for p in root.rglob('*') if p.is_file()]
    seen, size, logical = set(), 0, 0
    for path in paths:
        info = path.stat()
        logical += info.st_size
        key = (info.st_dev, info.st_ino)
        if key not in seen:
            seen.add(key)
            size += info.st_size
    if size > 8 * 1024**3:
        raise RuntimeError('8 GiB retention ceiling reached; retain bytes and block')
    return {'bytes': size, 'unique_inode_bytes': size, 'path_logical_bytes': logical,
            'budget_bytes': 8 * 1024**3,
            'policy': 'KEEP_ALL_PAYLOADS_ON_SSD; no deletion without verified lossless custody',
            'score_claim': False}


def prepare(role):
    work = ROOT / role
    runtime = work / 'runtime'
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < 44 * 1024**3:
        raise RuntimeError('storage reserve plus next 4 GiB decode not available')
    # Reserve the next complete output before a resume can retain another partial raw.
    # Sizes come from this exact receiver's retained public proof, not a generic guess.
    proof = json.loads((SOURCES[role].parent / 'PUBLIC_IDENTITY.json').read_text())
    required = int(proof['candidate_raw']['bytes']) + int(proof['decoded_field']['bytes'])
    required += 64 * 1024**2  # measured lineage checkpoint envelope; actual checkpoints remain retained
    used = hygiene(ROOT)['bytes']
    if not (work / 'TIMING.json').exists() and used + required > 8 * 1024**3:
        save(work / 'STORAGE_BLOCKED.json', {'existing_bytes': used,
             'reserved_next_bytes': required, 'budget_bytes': 8 * 1024**3,
             'source_proof': fact(SOURCES[role].parent / 'PUBLIC_IDENTITY.json'),
             'reason': 'retry plus retained partial output exceeds the artifact budget'})
        raise RuntimeError('next decode would exceed retention budget; existing bytes kept')
    source = measure_runtime_digest(SOURCES[role])
    if not runtime.exists():
        shutil.copytree(SOURCES[role], runtime,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.dylib', '._*'))
    if measure_runtime_digest(runtime).sha256 != source.sha256:
        raise RuntimeError('copied runtime identity differs from source')
    for name in ('data', 'output', 'scratch', 'frame_checkpoints'):
        (work / name).mkdir(exist_ok=True)
    with zipfile.ZipFile(runtime / 'archive.zip') as archive:
        payload = archive.read('p')
    payload_path = work / 'data/p'
    if payload_path.exists() and payload_path.read_bytes() != payload:
        raise RuntimeError('retained input payload drift')
    payload_path.write_bytes(payload)
    (work / 'files.txt').write_text('0.hevc\n')
    binding = {'source_runtime': str(SOURCES[role]), 'runtime_sha256': source.sha256,
               'archive': fact(runtime / 'archive.zip'), 'payload': fact(payload_path),
               'producer': fact(Path(__file__)), 'seed': 20260910,
               'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()}
    prior = work / 'BINDING.json'
    if prior.exists() and json.loads(prior.read_text()) != binding:
        raise RuntimeError('resume binding drift')
    save(prior, binding)
    return work, runtime, binding


def run(role, timeout):
    work, runtime, binding = prepare(role)
    complete = work / 'TIMING.json'
    if complete.exists():
        value = json.loads(complete.read_text())
        if value['binding'] != binding:
            raise RuntimeError('completed timing binding drift')
        return value
    attempt = work / f'attempt_{len(list(work.glob("attempt_*"))):04d}'
    attempt.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('F26_', 'TC1_RECEIVER_', 'TC3_ADVISORY_', 'CPR1_RC64_'))}
    env.update(dict.fromkeys(THREAD_KEYS, '4'))
    env.update(PATH=str(REPO / '.venv/bin') + os.pathsep + env.get('PATH', ''),
               TMPDIR=str(work / 'scratch'), PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='20260910',
               TC1_RECEIVER_CHECKPOINT_DIR=str(work / 'frame_checkpoints'),
               TC1_RECEIVER_STOP_AFTER='600', TC3_ADVISORY_CPU='1', F26_TOKEN_DECODER='python')
    command = ['bash', str(runtime / 'inflate.sh'), str(work / 'data'),
               str(work / 'output'), str(work / 'files.txt')]
    resumed = (work / 'frame_checkpoints/LATEST.json').exists()
    before = load_snapshot()
    save(attempt / 'LAUNCH.json', {'binding': binding, 'command': command,
         'environment': {k: env[k] for k in (*THREAD_KEYS, 'TMPDIR', 'TC3_ADVISORY_CPU',
             'TC1_RECEIVER_CHECKPOINT_DIR', 'TC1_RECEIVER_STOP_AFTER', 'F26_TOKEN_DECODER')},
         'concurrency': before, 'resumed': resumed, 'timeout_seconds': timeout,
         'retention': hygiene(ROOT)})
    started = time.monotonic()
    timed_out = False
    with (attempt / 'stdout.log').open('wb') as output:
        process = subprocess.Popen(command, cwd=work, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    seconds = time.monotonic() - started
    reports = []
    for line in (attempt / 'stdout.log').read_text().splitlines():
        if line.startswith('{"archive_bytes"'):
            reports.append(json.loads(line))
    value = {'schema': 'ddm_dwc1.raw_timing.v1', 'binding': binding, 'command': command,
             'axis': '[macOS-CPU advisory]', 'measurement_kind': 'cuda_path_equivalent_proxy',
             'score_claim': False, 'wall_seconds': seconds, 'cpu_threads': 4,
             'host': platform.node(), 'platform': platform.platform(), 'resumed': resumed,
             'returncode': code, 'timed_out': timed_out, 'concurrency_before': before,
             'concurrency_after': load_snapshot(), 'log': fact(attempt / 'stdout.log'),
             'report': reports[-1] if reports else None, 'retention': hygiene(ROOT)}
    latest = work / 'frame_checkpoints/LATEST.json'
    value['last_token_checkpoint'] = json.loads(latest.read_text()) if latest.exists() else None
    if code == 0:
        if len(reports) != 1 or reports[0]['pair_count'] != 600:
            raise RuntimeError('public shell exited without a full-population report')
        value['raw'] = fact(work / 'output/0.raw')
        value['tokens'] = fact(work / 'output/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8')
        if value['raw']['sha256'] != reports[0]['raw_sha256']:
            raise RuntimeError('raw report identity mismatch')
        save(complete, value)
    save(attempt / 'RESULT.json', value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--role', choices=tuple(SOURCES), required=True)
    parser.add_argument('--timeout-seconds', type=float, default=2400)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT or not 0 < args.timeout_seconds <= 2400:
        raise ValueError('invalid resume root or wall budget')
    run(args.role, args.timeout_seconds)
