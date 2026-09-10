#!/bin/sh
set -eu
exec env PYTHONPATH=src .venv/bin/python -u - "$@" <<'PY'
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
from tac.artifact_moved import move_with_manifest, verify_payload

parser = argparse.ArgumentParser()
parser.add_argument('--resume-from', type=Path, required=True)
args = parser.parse_args()
out = args.resume_from
ap = Path('/Volumes/APDataStore/pact')
vertigo = Path('/Volumes/VertigoDataTier/pact')
plan = json.loads((out / 'AP_MOVE_PLAN.json').read_text())
hashes = {r['path']: r for r in map(json.loads, (out / 'AP_MOVE_SOURCE_HASHES.jsonl').read_text().splitlines())}
assert len(hashes) == len(plan) == 9

def emit(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())

def df():
    return subprocess.check_output(['df', '-k', str(vertigo), str(ap)], text=True)

def free(root):
    s = os.statvfs(root)
    return s.f_bavail * s.f_frsize

# Do not create stale incoming MOVED location claims by moving their targets.
incoming = set()
guard = json.loads((out / 'MOVED_GUARD_FILENAME.json').read_text())
if guard['violations']:
    raise RuntimeError('MOVED custody census has unresolved violations')
for name in guard['certificate_paths']:
    certificate = Path(name)
    raw_certificate = certificate.read_bytes()
    if hashlib.sha256(raw_certificate).hexdigest() != guard['certificate_sha256'][name]:
        raise RuntimeError('MOVED certificate changed after validated census: ' + name)
    text_certificate = raw_certificate.decode('utf-8')
    texts = text_certificate.splitlines() if certificate.suffix == '.jsonl' else [text_certificate]
    for text in texts:
        if not text.strip(): continue
        row = json.loads(text)
        for key in ('moved_to', 'destination'):
            if isinstance(row.get(key), str): incoming.add(row[key])

for i, item in enumerate(plan):
    source = Path(item['path'])
    base_destination = Path(item['destination'])
    source_manifest = source.with_name(source.name + '.MOVED.json')
    certificate = out / f'AP_MOVE_{i:03d}.jsonl'
    expected = hashes[str(source)]
    if not source.exists() and source_manifest.exists():
        record = json.loads(source_manifest.read_text())
        verify_payload(record['moved_to'], expected['bytes'], expected['sha256'])
        resumed = {'status': 'RESUMED_VERIFIED_COMPLETED_MOVE', 'source': str(source), 'manifest': str(source_manifest), 'destination': record['moved_to'], 'sha256': expected['sha256'], 'bytes': expected['bytes']}
        emit(certificate, resumed); emit(out / 'RECLAIM_LEDGER.jsonl', resumed)
        continue
    if free(ap) >= 60 * 2**30: break
    if free(vertigo) - expected['bytes'] < 80 * 2**30:
        emit(out / 'RECLAIM_LEDGER.jsonl', {'status': 'MOVE_BLOCKED_DESTINATION_RESERVE', 'source': str(source), 'df': df()})
        raise SystemExit(19)
    if str(source) in incoming or source_manifest.exists():
        raise RuntimeError('existing incoming custody or source manifest: ' + str(source))
    if not source.is_relative_to(ap) or source.resolve() != source or source.name != '0.raw':
        raise RuntimeError('invalid source path')
    if any(x.startswith(('ddm_gdc2', 'ddm_rlc5_first_measurement_run3', 'scratch_gb1')) for x in source.parts):
        raise RuntimeError('protected source')
    s = source.stat()
    if [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns] != expected['verified_identity']:
        raise RuntimeError('source changed since complete hash')
    if hashlib.sha256(Path(item['producer_memo']).read_bytes()).hexdigest() != item['producer_memo_sha256']:
        raise RuntimeError('producer memo changed')
    opened = subprocess.run(['lsof', '-Fpan', '--', str(source)], capture_output=True, text=True)
    if opened.returncode not in (0, 1) or opened.stdout.strip() or opened.stderr.strip():
        raise RuntimeError('source is open or quiescence inspection failed: ' + opened.stdout + opened.stderr)
    destination = base_destination
    attempt = 0
    while destination.exists():
        # A crash can leave a partial copy. Keep it, record it, and retry to a fresh path.
        retained = {'status': 'RETAINED_PRIOR_COPY_NO_OVERWRITE', 'path': str(destination), 'bytes': destination.stat().st_size}
        emit(certificate, retained); emit(out / 'RECLAIM_LEDGER.jsonl', retained)
        attempt += 1
        destination = base_destination.with_name(base_destination.name + f'.retry{attempt}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = {**item, 'destination': str(destination), 'sha256': expected['sha256'], 'source_sha256': expected['sha256'], 'df_before': df(), 'certificate_path': str(certificate), 'status': 'CERTIFIED_MOVE_INTENT', 'command': ['sh', str(out / 'move_ap.sh'), '--resume-from', str(out)], 'script_sha256': hashlib.sha256((out / 'move_ap.sh').read_bytes()).hexdigest(), 'rebuild_command': ['cp', str(destination), str(source)]}
    emit(certificate, row); emit(out / 'RECLAIM_LEDGER.jsonl', row)
    try:
        result = move_with_manifest(source, destination, 'ddm_sr4: safely externalize quiescent advisory raw; every data byte retained, both hashes verified before source retirement', json.dumps(row['rebuild_command']))
    except PermissionError as exc:
        emit(out / 'RECLAIM_LEDGER.jsonl', {'status': 'STOP_SSD_PERMISSION_ERROR', 'error': str(exc), 'source': str(source)})
        raise
    except (OSError, ValueError) as exc:
        emit(out / 'RECLAIM_LEDGER.jsonl', {'status': 'BLOCKED_COPY_ALL_BYTES_RETAINED', 'error': str(exc), 'source': str(source), 'destination': str(destination)})
        raise
    manifest = json.loads(source_manifest.read_text())
    if manifest['sha256'] != expected['sha256'] or manifest['bytes'] != expected['bytes'] or source.exists():
        raise RuntimeError('post-move custody mismatch')
    # move_with_manifest already hashed the destination before source retirement.
    destination_check = verify_payload(result, expected['bytes'], expected['sha256'], hash_payload=False)
    row.update(status='MOVED_VERIFIED', destination_sha256=expected['sha256'])
    row.update(df_after=df(), source_manifest=str(source_manifest), destination_identity=destination_check['identity'], utc_epoch=time.time())
    emit(certificate, row); emit(out / 'RECLAIM_LEDGER.jsonl', row)
    print('MOVED_VERIFIED', source, result, flush=True)
emit(out / 'RECLAIM_LEDGER.jsonl', {'status': 'AP_MOVE_PASS_COMPLETE', 'df_after': df(), 'score_claim': False})
PY
