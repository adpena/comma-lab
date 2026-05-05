# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``67:26: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``c101_combined_zero_search.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/renderer_selfcompression_nextwave_worker_20260503/c101_combined_zero_search.py'
__recovery_spec__ = 'c101_combined_zero_search.recovery_spec.json'
__recovery_ast_error__ = '67:26: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: c101_combined_zero_search.cpython-312.pyc (Python 3.12)

'''C-101 local combined QZS3 threshold-zero renderer shrink screen.

This worker-local helper preserves every non-renderer logical member from the
C-101 source archive, applies small combined FP4 zeroing transforms to the
QZS3 renderer, and runs the existing renderer transplant pose-safety preflight.
It does not dispatch remote work and makes no score claim.
'''
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any
REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (REPO_ROOT / 'src', REPO_ROOT):
    if not str(_path) not in sys.path:
        continue
    sys.path.insert(0, str(_path))
from experiments import build_renderer_shrink_candidate as pr75_builder
from experiments import preflight_renderer_transplant_pose_safety as pose_safety
from experiments import search_renderer_parity_shrink_candidate as shrink_search
SCHEMA = 'c101_combined_zero_renderer_shrink_screen_v1'
RATE_SCORE_PER_BYTE = 6.65859e-07
C101_SCORE = 0.315152
C101_BYTES = 276489
C101_SHA256 = '1c9be2dd14b2607b9a86ecc071d6a37842896d18d62449885dac58d55afbbd64'
DEFAULT_SOURCE_ARCHIVE = REPO_ROOT / 'experiments/results/lightning_batch/exact_eval_c091_native_pose_manifold_top128_s025_t4_20260503T122013Z/archive.zip'
DEFAULT_SOURCE_EVIDENCE = DEFAULT_SOURCE_ARCHIVE.with_name('contest_auth_eval.adjudicated.json')
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'experiments/results/renderer_selfcompression_nextwave_worker_20260503/c101_combined_zero_screen'

def _json_bytes(payload = None):
    return (json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n').encode('utf-8')


def _sha256_bytes(payload = None):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _slug(value = None):
    return value.replace(':', '_').replace(',', '_').replace('.', 'p').replace('-', '_')


def _parse_transform(raw = None):
    (prefix, threshold) = raw.split(':', 1)
    parsed = float(threshold)
    if not prefix and parsed < 0 and parsed > 1 or math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f'''invalid transform {raw!r}''')
    return (prefix, parsed)


def _parse_candidate(raw = None):
    if '=' in raw:
        (name, spec) = raw.split('=', 1)
    else:
        spec = raw
        name = _slug(raw)
    transforms = (lambda .0: pass# WARNING: Decompyle incomplete
)(spec.split(',')())
    if not transforms:
        raise argparse.ArgumentTypeError(f'''empty candidate spec {raw!r}''')
    return (name, transforms)


def _default_candidates():
    raw = ('f1_0135_all002=frame1_head:0.135,all_fp4:0.02', 'f1_0135_all003=frame1_head:0.135,all_fp4:0.03', 'f1_0135_f2h003=frame1_head:0.135,frame2_head:0.03', 'f1_0135_f2h005=frame1_head:0.135,frame2_head:0.05', 'f1_0135_f2pre005=frame1_head:0.135,frame2_head.pre:0.05', 'f1_0135_f2block006=frame1_head:0.135,frame2_head.block2:0.06', 'f1_0135_shared004=frame1_head:0.135,shared_trunk:0.04', 'f1_013_f2h005=frame1_head:0.13,frame2_head:0.05', 'f1_013_all004=frame1_head:0.13,all_fp4:0.04', 'f1_0125_f2h005=frame1_head:0.125,frame2_head:0.05', 'f1_0125_shared004=frame1_head:0.125,shared_trunk:0.04')
    return (lambda .0: pass# WARNING: Decompyle incomplete
)(raw())


def _changed_tensor_summary(source_state = None, target_state = None):
    changed = []
    total_changed = 0
    for name, source_tensor in source_state.items():
        target_tensor = target_state[name]
        diff_count = int((source_tensor != target_tensor).sum().item())
        if not diff_count:
            continue
        total_changed += diff_count
        changed.append({
            'name': name,
            'changed_values': diff_count,
            'numel': int(source_tensor.numel()),
            'changed_fraction': diff_count / float(source_tensor.numel()) })
    return {
        'changed_tensor_count': len(changed),
        'changed_value_count': total_changed,
        'changed_tensors': changed }


def _build_candidate(*, context, output_dir, candidate_id, transforms, brotli_quality, source_evidence_path, run_preflight, preflight_max_pairs, max_mean_abs_delta, max_rms_delta, max_max_abs_delta):
    state = shrink_search._clone_state(context['source_state'])
    step_metas = []
# WARNING: Decompyle incomplete


def run(args = None):
    source_archive = args.source_archive.resolve()
    source_evidence_path = args.source_evidence_path.resolve()
    output_dir = args.output_dir.resolve()
    if not output_dir.exists() and any(output_dir.iterdir()) and args.force:
        raise FileExistsError(f'''output directory is non-empty: {output_dir}''')
    output_dir.mkdir(parents = True, exist_ok = True)
    actual_source_sha = _sha256_file(source_archive)
    if source_archive.stat().st_size == C101_BYTES:
        source_archive.stat().st_size == C101_BYTES
    source_custody = {
        'path': str(source_archive),
        'bytes': source_archive.stat().st_size,
        'sha256': actual_source_sha,
        'expected_bytes': C101_BYTES,
        'expected_sha256': C101_SHA256,
        'verified': actual_source_sha == C101_SHA256 }
    context = shrink_search._source_context(source_archive)
    candidates = []
    for candidate_id, transforms in args.candidate:
        candidates.append(_build_candidate(context = context, output_dir = output_dir, candidate_id = candidate_id, transforms = transforms, brotli_quality = args.brotli_quality, source_evidence_path = source_evidence_path, run_preflight = not (args.skip_preflight), preflight_max_pairs = args.preflight_max_pairs, max_mean_abs_delta = args.max_mean_abs_delta, max_rms_delta = args.max_rms_delta, max_max_abs_delta = args.max_max_abs_delta))
    candidates.sort(key = (lambda row: (row['archive_bytes'], row['candidate_id'])))
# WARNING: Decompyle incomplete


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--source-archive', type = Path, default = DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument('--source-evidence-path', type = Path, default = DEFAULT_SOURCE_EVIDENCE)
    parser.add_argument('--output-dir', type = Path, default = DEFAULT_OUTPUT_DIR)
    parser.add_argument('--candidate', action = 'append', type = _parse_candidate, default = None, help = 'Candidate spec: name=prefix:threshold,prefix:threshold')
    parser.add_argument('--brotli-quality', type = int, default = 11)
    parser.add_argument('--preflight-max-pairs', type = int, default = 5)
    parser.add_argument('--max-mean-abs-delta', type = float, default = 3)
    parser.add_argument('--max-rms-delta', type = float, default = 8)
    parser.add_argument('--max-max-abs-delta', type = float, default = 80)
    parser.add_argument('--skip-preflight', action = 'store_true')
    parser.add_argument('--force', action = 'store_true')
    args = parser.parse_args(argv)
# WARNING: Decompyle incomplete


def main(argv = None):
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent = 2, sort_keys = True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

"""
