#!/bin/sh
set -eu
exec .venv/bin/python -u - "$@" <<'PY'
import argparse, hashlib, json, os, stat, subprocess, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--resume-from', type=Path, required=True)
ap.add_argument('--target-free-gib', type=float, default=115)
args = ap.parse_args()
out = args.resume_from
root = Path('/Volumes/VertigoDataTier/pact')
groups = json.loads((out / 'PRIOR_HASH_DUPLICATE_LEADS.json').read_text())
groups = [g for g in groups if all(Path(r['path']).is_relative_to(root) for r in g)]
groups.sort(key=lambda g: (-len({r['ino'] for r in g}), g[0]['path']))

def emit(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())

def df():
    return subprocess.check_output(['df', '-k', str(root), '/Volumes/APDataStore/pact'], text=True)

def identity(p):
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_size != 3662409600:
        raise RuntimeError('not a regular full raw: ' + str(p))
    if p.resolve() != p or any(x.startswith(('ddm_gdc2_categorical_coolchic_k8_distill', 'ddm_rlc5_first_measurement_run3')) for x in p.parts) or 'upstream' in p.parts:
        raise RuntimeError('protected or redirected path: ' + str(p))
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)

def hash_raw(p):
    before = identity(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    if identity(p) != before: raise RuntimeError('changed during hash: ' + str(p))
    row = {'path': str(p), 'identity': before, 'sha256': h.hexdigest(), 'utc_epoch': time.time()}
    emit(out / 'DEDUP_HASHES.jsonl', row)
    print('HASHED', p, h.hexdigest(), flush=True)
    return before, h.hexdigest()

def writers(paths):
    r = subprocess.run(['lsof', '-Fpan', '--', *map(str, paths)], capture_output=True, text=True)
    if r.returncode not in (0, 1) or r.stderr.strip():
        raise RuntimeError('lsof inspection failed: ' + r.stderr)
    if any(x.startswith('a') and x[1:] != 'r' for x in r.stdout.splitlines()):
        raise RuntimeError('open writer: ' + r.stdout)
    return {'argv': r.args, 'rc': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}

script_sha = hashlib.sha256((out / 'dedup.sh').read_bytes()).hexdigest()
for gi, group in enumerate(groups):
    if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= args.target_free_gib * 2**30: break
    canonical = Path(group[0]['path'])
    expected = group[0]['prior_sha256']
    candidates = [Path(r['path']) for r in group[1:] if identity(Path(r['path']))[:2] != identity(canonical)[:2]]
    if not candidates: continue
    try:
        writers([canonical, *candidates])
        canonical_id, canonical_sha = hash_raw(canonical)
        if canonical_sha != expected: raise RuntimeError('canonical historical SHA drift')
        for si, source in enumerate(candidates):
            if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= args.target_free_gib * 2**30: break
            if identity(source)[:2] == identity(canonical)[:2]: continue
            source_id, source_sha = hash_raw(source)
            if source_sha != canonical_sha: raise RuntimeError('full source/destination hash mismatch')
            open_check = writers([source, canonical])
            if identity(source) != source_id or identity(canonical) != canonical_id: raise RuntimeError('identity drift before link')
            certificate = out / f'HARDLINK_{gi:03d}_{si:03d}.jsonl'
            before_df = df()
            row = {'schema': 'ddm_sr4.twin_raw_dedup_certificate.v1', 'source': str(source), 'destination': str(canonical), 'bytes': source_id[2], 'source_sha256': source_sha, 'destination_sha256': canonical_sha, 'source_identity': source_id, 'destination_identity': canonical_id, 'df_before': before_df, 'certificate_path': str(certificate), 'lsof': open_check, 'script_sha256': script_sha, 'command': ['sh', str(out / 'dedup.sh'), '--resume-from', str(out), '--target-free-gib', str(args.target_free_gib)], 'score_claim': False, 'promotable': False, 'reason': 'Full source and destination SHA equality; retain both original paths with one inode; no payload lost', 'status': 'CERTIFIED_BEFORE_HARDLINK'}
            emit(certificate, row); emit(out / 'RECLAIM_LEDGER.jsonl', row)
            temporary = source.with_name('0.raw.ddm_sr4_verified_link')
            if temporary.exists():
                if identity(temporary)[:2] != canonical_id[:2]: raise RuntimeError('unexpected retained temporary link')
            else: os.link(canonical, temporary)
            if identity(temporary)[:2] != canonical_id[:2] or identity(source) != source_id or identity(canonical) != canonical_id: raise RuntimeError('pre-replace identity drift')
            os.replace(temporary, source)
            directory = os.open(source.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
            if identity(source)[:2] != identity(canonical)[:2]: raise RuntimeError('post-link inode mismatch')
            row = {**row, 'status': 'HARDLINK_VERIFIED', 'df_after': df(), 'source_identity_after': identity(source), 'destination_identity_after': identity(canonical), 'utc_epoch': time.time()}
            emit(certificate, row); emit(out / 'RECLAIM_LEDGER.jsonl', row)
            print('HARDLINK_VERIFIED', source, flush=True)
    except PermissionError as exc:
        emit(out / 'RECLAIM_LEDGER.jsonl', {'status': 'STOP_SSD_PERMISSION_ERROR', 'error': str(exc), 'group': gi})
        raise
    except (OSError, RuntimeError) as exc:
        emit(out / 'RECLAIM_LEDGER.jsonl', {'status': 'BLOCKED_GROUP_BYTES_RETAINED', 'error': str(exc), 'group': gi})
        print('BLOCKED', gi, str(exc), flush=True)
emit(out / 'RECLAIM_LEDGER.jsonl', {'status': 'DEDUP_PASS_COMPLETE', 'df_after': df(), 'score_claim': False})
PY
