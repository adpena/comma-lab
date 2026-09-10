#!/bin/sh
exec .venv/bin/python -B - <<'PY'
import json, resource, runpy, sys, time
from pathlib import Path
root = Path('/Volumes/VertigoDataTier/pact/ddm_tc2_lane_context_map')
sys.argv = [str(Path('experiments/ddm_tc2_lane_context.py').resolve()), 'fit', '--resume-from', str(root)]
started = time.monotonic()
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    record = {'elapsed_s': time.monotonic()-started, 'ru_maxrss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (root/('fit_process_usage_'+str(time.time_ns())+'.json')).write_text(json.dumps(record, indent=2)+'\n')
PY
