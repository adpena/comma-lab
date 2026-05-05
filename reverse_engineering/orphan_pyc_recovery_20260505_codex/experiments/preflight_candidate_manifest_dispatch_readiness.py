# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``38:66: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``preflight_candidate_manifest_dispatch_readiness.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/preflight_candidate_manifest_dispatch_readiness.py'
__recovery_spec__ = 'preflight_candidate_manifest_dispatch_readiness.recovery_spec.json'
__recovery_ast_error__ = '38:66: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: preflight_candidate_manifest_dispatch_readiness.cpython-312.pyc (Python 3.12)

'''Fail-closed preflight for candidate manifest dispatch readiness.

This is a cheap local guard for the gap between builder-specific readiness
fields and the narrower Lightning submit-time ``exact_eval_dispatch_gate``.
Run it on a candidate ``manifest.json`` before any lane claim or exact-eval
queue command.
'''
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
SCHEMA = 'candidate_manifest_dispatch_readiness_preflight_v1'
BLOCKING_TEXT_MARKERS = ('blocked', 'fail_closed', 'failed', 'missing', 'non_dispatchable', 'not_run', 'planning_only', 'no_remote_dispatch', 'local_only', 'invalid', 'negative')
READY_TEXT_MARKERS = ('eligible_for_exact_eval_after_lane_claim', 'eligible_for_cuda_auth_eval_after_lane_claim', 'ready', 'passed')
FALSE_READY_FIELDS = ('dispatch_unlocked', 'ready_for_exact_eval_after_lane_claim', 'ready_for_exact_eval_dispatch_claim', 'ready_for_exact_eval_dispatch', 'ready_for_fixed_runtime_exact_eval', 'safe_for_exact_eval_dispatch', 'safe_for_remote_dispatch', 'dispatchable', 'dispatch_ready_now')
RUNTIME_CHANGING_MARKERS = ('stbm1br', 'hpm1', 'hpac', 'qfq4', 'qma9', 'qh0', 'qps1', 'qzs3', 'qrgb')
FORMULA_ONLY_KEY_MARKERS = ('formula_only', 'if_components_identical', 'if_archive_is_valid')
RUNTIME_SUPPORT_FALSE_KEYS = ('runtime_can_decode_without_edits', 'runtime_can_decode', 'public_pr85_replay_qfq4_model_loader', 'robust_current_qfq4_renderer_loader')
DECODE_PARITY_FALSE_KEYS = ('decoded_tensor_parity', 'local_decode_byte_parity_proven', 'full_decode_byte_parity_proven', 'byte_parity_achieved', 'pr91_ready_for_exact_eval')
TERMINAL_CLAIM_STATUS_PREFIXES = ('completed_', 'completed_score=', 'completed_no_frontier', 'failed_', 'preempted', 'cancelled', 'refused_dispatch', 'stale_assumed_dead', 'stale_superseded', 'stopped_')

def _load_json_object(path = None):
    
    try:
        payload = json.loads(path.read_text(encoding = 'utf-8'))
        if not isinstance(payload, dict):
            raise SystemExit(f'''manifest must be a JSON object: {path}''')
        return payload
    except json.JSONDecodeError:
        exc = None
        raise SystemExit(f'''manifest is invalid JSON: {path}'''), exc
        exc = None
        del exc



def _stringify_items(items = None):
    pass
# WARNING: Decompyle incomplete


def _has_blocking_text(value = None):
    pass
# WARNING: Decompyle incomplete


def _has_ready_text(value = None):
    pass
# WARNING: Decompyle incomplete


def _append_blocker(blockers = None, code = None, detail = None):
    blockers.append({
        'code': code,
        'severity': 'blocking',
        'detail': detail })


def _contains_runtime_changing_marker(value = None):
    pass
# WARNING: Decompyle incomplete


def _requires_exact_runtime_contract(manifest = None):
    fail_closed = manifest.get('fail_closed_preflight')
    if isinstance(fail_closed, Mapping):
        if fail_closed.get('exact_eval_requires_explicit_pr85_replay_runtime') is True:
            return True
        if fail_closed.get('exact_eval_requires_submitted_runtime_contract') is True:
            return True
    for key in ('candidate_id', 'policy_id', 'tool', 'runtime_support', 'source_policy', 'segments', 'candidate_bundle'):
        if not key in manifest:
            continue
        if not _contains_runtime_changing_marker(manifest.get(key)):
            continue
        ('candidate_id', 'policy_id', 'tool', 'runtime_support', 'source_policy', 'segments', 'candidate_bundle')
        return True
    return False


def _iter_leaf_values(value = None, *, prefix):
    rows = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f'''{prefix}.{key}''' if prefix else str(key)
            rows.extend(_iter_leaf_values(item, prefix = path))
        return rows
    if not None(value, Sequence) and isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            rows.extend(_iter_leaf_values(item, prefix = f'''{prefix}[{index}]'''))
        return rows
    None.append((prefix, value))
    return rows


def _leaf_key(path = None):
    return path.rsplit('.', 1)[-1].split('[', 1)[0]


def _is_nonzero_formula_value(value = None):
    if value in (None, False, '', [], { }):
        return False
    if isinstance(value, (int, float)):
        return value != 0


def _is_explanatory_rate_formula_path(path = None, manifest = None):
    pass
# WARNING: Decompyle incomplete


def _audit_recursive_fail_closed_fields(manifest = None, *, blockers, warnings):
    pass
# WARNING: Decompyle incomplete


def _audit_gate_mapping(gate = None, *, path, blockers, warnings):
    pass
# WARNING: Decompyle incomplete


def _parse_utc(value = None):
    value = value.strip()
    if value.endswith('Z'):
        value = value[:-1] + '+00:00'
# WARNING: Decompyle incomplete


def _is_terminal_claim_status(status = None):
    pass
# WARNING: Decompyle incomplete


def _parse_claim_rows(path = None):
    if not path.is_file():
        return []
    rows = None
# WARNING: Decompyle incomplete


def _manifest_lane_id(manifest = None):
    candidates = [
        manifest.get('lane_id')]
    for key in ('dispatch_readiness', 'exact_eval_dispatch_gate', 'dispatch_gate', 'dispatch'):
        section = manifest.get(key)
        if not isinstance(section, Mapping):
            continue
        candidates.append(section.get('lane_id'))
        claim = section.get('claim')
        if not isinstance(claim, Mapping):
            continue
        candidates.append(claim.get('lane_id'))
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        if not candidate.strip():
            continue
        
        return candidates, candidate.strip()


def _active_lane_claims(claims = None, *, lane_id, now, ttl_hours):
    cutoff = now - dt.timedelta(hours = ttl_hours)
    conflicts = []
    closed_instance_job_ids = set()
# WARNING: Decompyle incomplete


def _lane_claim_report(manifest = None, *, claims_path, now_utc, ttl_hours):
    lane_id = _manifest_lane_id(manifest)
# WARNING: Decompyle incomplete


def build_preflight(manifest_path = None, *, claims_path, now_utc, ttl_hours):
    manifest = _load_json_object(manifest_path)
    blockers = []
    warnings = []
    gate = manifest.get('exact_eval_dispatch_gate')
    if isinstance(gate, Mapping):
        _audit_gate_mapping(gate, path = 'exact_eval_dispatch_gate', blockers = blockers, warnings = warnings)
    dispatch_gate = manifest.get('dispatch_gate')
    if isinstance(dispatch_gate, str):
        if _has_blocking_text(dispatch_gate):
            _append_blocker(blockers, 'dispatch_gate_blocked', f'''dispatch_gate={dispatch_gate!r}''')
        elif _has_ready_text(dispatch_gate):
            warnings.append({
                'code': 'lane_claim_still_required',
                'detail': f'''dispatch_gate={dispatch_gate!r}; verify active Level-2 lane claim before GPU dispatch''' })
        else:
            warnings.append({
                'code': 'unrecognized_dispatch_gate_text',
                'detail': f'''dispatch_gate={dispatch_gate!r}''' })
    elif isinstance(dispatch_gate, Mapping):
        _audit_gate_mapping(dispatch_gate, path = 'dispatch_gate', blockers = blockers, warnings = warnings)
    for key in FALSE_READY_FIELDS:
        if not key in manifest:
            continue
        if not manifest[key] is False:
            continue
        _append_blocker(blockers, f'''{key}_false''', f'''{key} is false''')
    for key in ('exact_eval_readiness_status', 'readiness_status', 'build_status'):
        value = manifest.get(key)
        if not isinstance(value, str):
            continue
        if not _has_blocking_text(value):
            continue
        _append_blocker(blockers, f'''{key}_blocked''', f'''{key}={value!r}''')
    _audit_recursive_fail_closed_fields(manifest, blockers = blockers, warnings = warnings)
    for path, key in (('fail_closed_preflight', 'remaining_exact_eval_blockers'), ('exact_eval_runtime_contract', 'remaining_blockers'), ('runtime_gate', 'remaining_blockers'), ('fixed_runtime_bridge', 'remaining_blockers')):
        section = manifest.get(path)
        if not isinstance(section, Mapping):
            continue
        nested = _stringify_items(section.get(key))
        if not nested:
            continue
        _append_blocker(blockers, f'''{path}:{key}''', f'''{path}.{key}: {', '.join(nested[:8])}''')
    exact_runtime = manifest.get('exact_eval_runtime_contract')
    if not _requires_exact_runtime_contract(manifest) and isinstance(exact_runtime, Mapping):
        _append_blocker(blockers, 'exact_eval_runtime_contract:missing_for_runtime_changing_candidate', 'runtime-changing candidate must record the exact submitted inflate runtime contract')
    if isinstance(exact_runtime, Mapping) and exact_runtime.get('ready_for_exact_eval_runtime') is False:
        _append_blocker(blockers, 'exact_eval_runtime_contract:not_ready', 'exact eval runtime contract is not ready')
    fixed_runtime = manifest.get('fixed_runtime_preflight')
    if isinstance(fixed_runtime, Mapping):
        ready = fixed_runtime.get('ready_for_fixed_runtime_exact_eval')
        ready = fixed_runtime.get('ready_for_fixed_runtime_exact_eval_readiness', ready)
        if ready is False:
            _append_blocker(blockers, 'fixed_runtime_preflight:not_ready', 'fixed runtime preflight is not ready')
    if str(manifest.get('evidence_grade', '')).lower().startswith('a-negative'):
        _append_blocker(blockers, 'evidence_grade_exact_negative', 'manifest evidence_grade is exact-negative')
    if manifest.get('exact_negative') is True:
        _append_blocker(blockers, 'exact_negative_true', 'manifest is marked exact_negative=true')
    lane_claim = _lane_claim_report(manifest, claims_path = claims_path, now_utc = now_utc, ttl_hours = ttl_hours)
    if not lane_claim['checked'] and lane_claim.get('lane_id'):
        _append_blocker(blockers, 'lane_claim_missing_lane_id', 'claims-path was supplied but manifest did not expose a lane_id')
    for conflict in lane_claim.get('active_conflicts', []):
        _append_blocker(blockers, 'active_lane_claim_conflict', f'''active same-lane claim exists: {conflict.get('timestamp_utc')} {conflict.get('agent')} {conflict.get('platform')} {conflict.get('instance_job_id')} status={conflict.get('status')}''')
    return {
        'schema': SCHEMA,
        'manifest_path': str(manifest_path),
        'candidate_id': manifest.get('candidate_id'),
        'ready_for_exact_eval_dispatch': not blockers,
        'blockers': blockers,
        'warnings': warnings,
        'lane_claim': lane_claim,
        'score_claim': bool(manifest.get('score_claim', False)),
        'dispatch_performed': bool(manifest.get('dispatch_performed', False)),
        'remote_jobs_dispatched': bool(manifest.get('remote_jobs_dispatched', False)) }


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--manifest', type = Path, required = True)
    parser.add_argument('--claims-path', type = Path)
    parser.add_argument('--now-utc')
    parser.add_argument('--ttl-hours', type = float, default = 24)
    parser.add_argument('--json-out', type = Path)
    parser.add_argument('--fail-if-not-ready', action = 'store_true')
    args = parser.parse_args(argv)
    payload = build_preflight(args.manifest, claims_path = args.claims_path, now_utc = args.now_utc, ttl_hours = args.ttl_hours)
    text = json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n'
    if args.json_out:
        args.json_out.parent.mkdir(parents = True, exist_ok = True)
        args.json_out.write_text(text, encoding = 'utf-8')
    print(text, end = '')
    if not args.fail_if_not_ready and payload['ready_for_exact_eval_dispatch']:
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

"""
