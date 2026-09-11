#!/bin/bash
# Research receiver variants only. No seal, no authorization, no dispatch.
set -euo pipefail
cd /Users/adpena/Projects/pact
.venv/bin/python - <<'PY'
import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

root = Path('/Volumes/VertigoDataTier/pact/ddm_rbf1/retained')
source = root / 'receiver_readonly'
treatment = Path('experiments/ddm_rbf1_boundary_treatments.py')
original = (source / 'cpr1/inflate.py').read_text()
needle = 'output[2 * (start + offset) + 1] = master_np[offset]'
if original.count(needle) != 1:
    raise SystemExit('refused: shipped master-write seam no longer unique')
records = []
for mode in ('guided', 'ssaa', 'sdf', 'composition'):
    target = root / 'research_receiver_variants' / mode
    replacement = f'output[2 * (start + offset) + 1] = rbf1_treat(master_np[offset], tokens[start + offset].cpu().numpy(), {mode!r})'
    code = original.replace('import numpy as np\n', 'import numpy as np\nfrom rbf1_treatments import treat as rbf1_treat\n').replace(needle, replacement)
    ast.parse(code)
    for path in sorted(source.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        relative = path.relative_to(source)
        data = code.encode() if str(relative) == 'cpr1/inflate.py' else path.read_bytes()
        if str(relative) == 'MANIFEST.sha256':
            continue
        dest = target / relative
        if dest.exists() and dest.read_bytes() != data:
            raise SystemExit(f'refusing to overwrite changed variant: {dest}')
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    dest = target / 'cpr1/rbf1_treatments.py'
    if dest.exists() and dest.read_bytes() != treatment.read_bytes():
        raise SystemExit('treatment source changed; version the candidate')
    if not dest.exists():
        shutil.copyfile(treatment, dest)
    files = []
    census = []
    for path in sorted(target.rglob('*')):
        if path.is_file() and path.name != 'MANIFEST.sha256':
            files.append({'relative_path':str(path.relative_to(target)), 'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = ''.join(f"{r['sha256']}  {r['relative_path']}\n" for r in files)
    (target / 'MANIFEST.sha256').write_text(manifest)
    for path in (target / 'cpr1/rbf1_treatments.py',):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant):
                census.append({'file':str(path.relative_to(target)), 'line':node.lineno, 'value':repr(node.value)})
    delta = ''.join(difflib.unified_diff(original.splitlines(True), code.splitlines(True), fromfile='source/cpr1/inflate.py', tofile=f'{mode}/cpr1/inflate.py'))
    (root / f'{mode}_receiver.patch').write_text(delta)
    records.append({'mode':mode, 'tree':str(target), 'files':files, 'new_module_literal_census':census,
                    'archive_bytes':(target/'archive.zip').stat().st_size,
                    'archive_sha256':hashlib.sha256((target/'archive.zip').read_bytes()).hexdigest(),
                    'status':'UNMEASURED_RESEARCH_RECEIVER', 'score_claim':False,
                    'rule118':'All new numeric literals define lattice arithmetic, neighborhood geometry, fp32/uint8 handling or generic branch structure. No fitting, class selection or per-video scalar. Mode names select predeclared generic algorithms.',
                    'open_gates':['n600 joint scorer row','best composition selection','cold public parseback','T4 treatment parity','full receiver literal census','timing-risk receipt','MAIN first-measurement intent']})
out = root / 'RESEARCH_RECEIVERS.json'
out.write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps({'manifest':str(out),'variants':len(records),'archive_sha256':records[0]['archive_sha256']}))
PY
