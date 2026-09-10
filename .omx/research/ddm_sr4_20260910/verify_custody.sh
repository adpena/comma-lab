#!/bin/sh
set -eu
exec env PYTHONPATH=src .venv/bin/python - "$@" <<'PY'
import argparse, json, os, subprocess, time
from pathlib import Path
from tac.artifact_moved import verify_payload
from tac.artifact_moved_audit import certificate_violations

p = argparse.ArgumentParser()
p.add_argument('--evidence-dir', type=Path, required=True)
args = p.parse_args()
out = args.evidence_dir
rows = [json.loads(s) for s in (out/'RECLAIM_LEDGER.jsonl').read_text().splitlines()]
links = {}
moves = {}
def name(value):
    return value['path'] if isinstance(value, dict) else value
for row in rows:
    if row.get('status') == 'HARDLINK_VERIFIED': links[name(row['source'])] = row
    if row.get('status') == 'MOVED_VERIFIED': moves[row.get('source', row.get('path'))] = row
assert links and len(moves) == 9, (len(links), len(moves))
verified_links = []
for source, row in sorted(links.items()):
    destination = name(row['destination'])
    a, b = Path(source), Path(destination)
    assert not a.is_symlink() and not b.is_symlink()
    sa, sb = a.stat(), b.stat()
    assert (sa.st_dev, sa.st_ino, sa.st_size, sa.st_mtime_ns) == (sb.st_dev, sb.st_ino, sb.st_size, sb.st_mtime_ns)
    assert sa.st_size == 3662409600
    if isinstance(row['destination'], dict):
        old = row['destination']
        expected = (old['dev'], old['ino'], old['bytes'], old['mtime_ns'])
        assert row['source']['sha256'] == old['sha256']
    else:
        expected = tuple(row['destination_identity'])
        assert row['source_sha256'] == row['destination_sha256']
    assert (sb.st_dev, sb.st_ino, sb.st_size, sb.st_mtime_ns) == expected
    verified_links.append({'source': source, 'destination': destination, 'inode': sa.st_ino, 'bytes': sa.st_size})
verified_moves = []
for source, row in sorted(moves.items()):
    assert not os.path.lexists(source)
    manifest = json.loads(Path(row['source_manifest']).read_text())
    assert manifest['moved_to'] == row['destination']
    assert manifest['bytes'] == row['bytes'] and manifest['sha256'] == row['destination_sha256'] == row['source_sha256']
    check = verify_payload(row['destination'], row['bytes'], row['destination_sha256'], hash_payload=False)
    assert list(check['identity']) == row['destination_identity']
    verified_moves.append({'source': source, 'destination': row['destination'], 'manifest': row['source_manifest'], 'sha256': manifest['sha256'], 'bytes': manifest['bytes']})
old = json.loads((out/'MOVED_GUARD_BEFORE.json').read_text())
certificates = [Path(s) for s in old['certificate_paths']] + [Path(r['source_manifest']) for r in moves.values()]
violations = certificate_violations(certificates, hash_payloads=False)
assert not violations, violations
df = subprocess.check_output(['df','-k','/Volumes/VertigoDataTier','/Volumes/APDataStore'],text=True)
(out/'DF_AFTER.txt').write_text(df)
free = {line.split()[-1]: int(line.split()[3])*1024 for line in df.splitlines()[1:]}
assert free['/Volumes/VertigoDataTier'] >= 80*2**30 and free['/Volumes/APDataStore'] >= 60*2**30
result = {'status':'VERIFIED','axis':'macOS filesystem custody, scorer-free','utc_epoch':time.time(),'hardlinks':verified_links,'hardlink_count':len(verified_links),'moves':verified_moves,'move_count':len(verified_moves),'moved_logical_bytes':sum(r['bytes'] for r in verified_moves),'moved_certificates_checked':len(certificates),'moved_certificate_violations':violations,'free_bytes':free,'hash_scope':'Full source and destination SHA at each mutation; final audit revalidates same hash-bound identities, not a redundant full-payload rehash. Old MOVED records checked for live metadata/header custody.','score_claim':False}
(out/'FINAL_CUSTODY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('hardlinks','moves')}))
PY
