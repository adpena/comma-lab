#!/usr/bin/env bash
set -euo pipefail

BAT00_HOST="${BAT00_HOST:-adpena@bat00.local}"
BAT00_RUN_ROOT="${BAT00_RUN_ROOT:-/home/adpena/bat00-runs}"

cat <<'REMOTE' | ssh "$BAT00_HOST" 'wsl -e bash -s' -- "$BAT00_RUN_ROOT"
set -euo pipefail
run_root="$1"
python3 - <<'PY' "$run_root"
import glob, json, os, sys
run_root = sys.argv[1]
paths = sorted(glob.glob(os.path.join(run_root, '*', 'latest', 'status.json')))
for path in paths:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        print(path, {'status': 'unreadable', 'error': str(exc)})
        continue
    print(json.dumps(data, sort_keys=True))
PY
REMOTE
