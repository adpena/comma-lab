"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``58:49: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``build_hnerv_frontier_scorecard.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/build_hnerv_frontier_scorecard.py'
__recovery_spec__ = 'build_hnerv_frontier_scorecard.recovery_spec.json'
__recovery_ast_error__ = '58:49: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: build_hnerv_frontier_scorecard.cpython-312.pyc (Python 3.12)

'''Build a compact scorecard for public HNeRV frontier intake.

The scorecard joins exact CUDA replay artifacts with forensic payload profiles.
It does not evaluate archives and does not promote prediction-only claims; it is
only a deterministic decision table for the next optimization action.
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

def load_json(path = None):
    return json.loads(path.read_text())


def maybe_load_json(path = None):
    pass
# WARNING: Decompyle incomplete


def profile_by_sha(path = None):
    payload = maybe_load_json(path)
    if not payload:
        return { }
# WARNING: Decompyle incomplete


def contribution(payload = None, key = None):
    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)


def row_from_eval(label = None, path = None, profiles = None):
    payload = load_json(path)
    if not payload.get('provenance'):
        payload.get('provenance')
    prov = { }
    sha = prov.get('archive_sha256')
    profile = profiles.get(sha, { })
    if not profile.get('sections'):
        profile.get('sections')
    sections = []
    largest_section = max(sections, key = (lambda item: item.get('bytes', 0)), default = None)
# WARNING: Decompyle incomplete


def render_markdown(rows = None):
    lines = [
        '# HNeRV Frontier Scorecard',
        '',
        '| label | grade | score | bytes | seg | pose | rate | largest section | archive sha |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for row in sorted(rows, key = (lambda item: pass# WARNING: Decompyle incomplete
)):
        if not row.get('largest_payload_section'):
            row.get('largest_payload_section')
        largest = { }
        largest_text = f'''{largest.get('name')}:{largest.get('bytes')}''' if largest else 'n/a'
        if not row['archive_sha256']:
            row['archive_sha256']
        lines.append('| {label} | {evidence_grade} | {score:.12f} | {archive_bytes} | {seg:.9f} | {pose:.9f} | {rate:.9f} | `{largest}` | `{sha}` |'.format(label = row['label'], evidence_grade = row['evidence_grade'], score = row['score'], archive_bytes = row['archive_bytes'], seg = row['score_seg_contribution'], pose = row['score_pose_contribution'], rate = row['score_rate_contribution'], largest = largest_text, sha = ''[:16]))
    lines.extend([
        '',
        'Interpretation: score truth remains the exact CUDA replay JSON. Payload',
        'sections are forensic signals for the next compression action; they do',
        'not imply score deltas without a new exact archive eval.',
        ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--profile-json', type = Path, help = 'Payload profile JSON emitted by profile_hnerv_frontier_payloads.py.')
    parser.add_argument('--json-out', type = Path, required = True)
    parser.add_argument('--md-out', type = Path, required = True)
    parser.add_argument('evals', nargs = '+', help = 'LABEL=path/to/contest_auth_eval.adjudicated.json')
    args = parser.parse_args()
    profiles = profile_by_sha(args.profile_json)
    rows = []
    for item in args.evals:
        if '=' not in item:
            raise SystemExit(f'''expected LABEL=PATH, got {item!r}''')
        (label, raw_path) = item.split('=', 1)
        rows.append(row_from_eval(label, Path(raw_path), profiles))
    payload = {
        'schema_version': 1,
        'tool': 'build_hnerv_frontier_scorecard',
        'score_truth': 'exact_cuda_auth_eval_json',
        'rows': sorted(rows, key = (lambda item: pass# WARNING: Decompyle incomplete
)) }
    args.json_out.parent.mkdir(parents = True, exist_ok = True)
    args.json_out.write_text(json.dumps(payload, indent = 2, sort_keys = True) + '\n')
    args.md_out.parent.mkdir(parents = True, exist_ok = True)
    args.md_out.write_text(render_markdown(rows))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

"""
