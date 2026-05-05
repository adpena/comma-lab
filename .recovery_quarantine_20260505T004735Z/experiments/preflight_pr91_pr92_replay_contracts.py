# Source Generated with Decompyle++
# File: preflight_pr91_pr92_replay_contracts.cpython-312.pyc (Python 3.12)

'''Fail-closed PR91/HPM1 and PR92/RMB1 replay contract preflight.

This tool is local-only. It does not load scorers, dispatch remote jobs, or
make new score claims. It sharpens two endgame replay questions:

* PR91/HPM1: is the downloaded entropy stream locally replayable enough to
  justify a derived exact-eval dispatch?
* PR92/RMB1: was the PR92 randmulti transfer evaluated through the corrected
  replay runtime, and is the exact T4 result internally consistent?
'''
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from tac.pr91_hpm1_codec import DEFAULT_PR91_ARCHIVE, run_pr91_hpm1_probability_variant_matrix
TOOL = 'experiments/preflight_pr91_pr92_replay_contracts.py'
SCHEMA = 'pr91_pr92_replay_contract_preflight_v1'
SCORE_DENOMINATOR = 3.75455e+07
DEFAULT_PR91_PROBABILITY_REPORT = REPO_ROOT / 'experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_probability_variant_matrix_frame0_20260504_current_codex.json'
DEFAULT_PR92_MANIFEST = REPO_ROOT / 'experiments/results/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker/pr85_stbm1br_plus_pr92_rmb1_randmulti_recode/manifest.json'
DEFAULT_PR92_EXACT_JSON = REPO_ROOT / 'experiments/results/lightning_batch/exact_eval_pr85_stbm1br_pr92_rmb1_t4_20260504T082220Z/contest_auth_eval.adjudicated.json'
DEFAULT_OUTPUT_JSON = REPO_ROOT / 'experiments/results/pr91_pr92_replay_contract_preflight_20260504_codex/preflight.json'
DEFAULT_LEDGER = REPO_ROOT / '.omx/research/pr91_pr92_replay_contract_preflight_20260504_codex.md'

def _rel(path = None):
    
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return 



def _sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _json_text(payload = None):
    return json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n'


def _load_json(path = None):
    if not path.is_file():
        raise FileNotFoundError(_rel(path))
    payload = json.loads(path.read_text(encoding = 'utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'''expected JSON object: {_rel(path)}''')
    return payload


def _write_json(path = None, payload = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(_json_text(payload), encoding = 'utf-8')


def _failed_checks(checks = None):
    pass
# WARNING: Decompyle incomplete


def _read_or_run_pr91_probability_report(*, archive, report_path, rerun):
    if rerun:
        report = run_pr91_hpm1_probability_variant_matrix(archive, variants = None, max_frames = 1)
        source = {
            'mode': 'rerun_local_prefix',
            'archive': _rel(archive),
            'report_path': None,
            'score_claim': False,
            'dispatch_performed': False }
        return (report, source)
    report = None(report_path)
    source = {
        'mode': 'read_existing_report',
        'report_path': _rel(report_path),
        'report_sha256': _sha256_file(report_path),
        'archive': _rel(archive),
        'score_claim': False,
        'dispatch_performed': False }
    return (report, source)


def validate_pr91_hpm1_contract(*, archive, probability_report, rerun):
    (report, source) = _read_or_run_pr91_probability_report(archive = archive, report_path = probability_report, rerun = rerun)
    variants = report.get('variant_results', [])
    if not isinstance(variants, list):
        variants = []
# WARNING: Decompyle incomplete


def _score_from_components(payload = None):
    return 100 * float(payload['avg_segnet_dist']) + math.sqrt(10 * float(payload['avg_posenet_dist'])) + 25 * int(payload['archive_size_bytes']) / SCORE_DENOMINATOR


def _artifact_recomputed_score(payload = None):
    """Return the artifact's canonical recomputation field when available.

    Some adjudicated JSONs expose rounded component summaries plus the exact
    score recomputed inside the evaluator. The exact field is the custody value;
    rounded component summaries are retained for human reporting only.
    """
    if 'score_recomputed_from_components' in payload:
        return float(payload['score_recomputed_from_components'])
    return None(payload)


def validate_pr92_rmb1_contract(*, manifest_path, exact_json_path):
    manifest = _load_json(manifest_path)
    exact = _load_json(exact_json_path)
    candidate = manifest.get('candidate_archive', { })
    runtime = manifest.get('exact_eval_runtime_contract', { })
    readiness = manifest.get('dispatch_readiness', { })
    parity = manifest.get('randmulti_decoded_row_parity', { })
    non_noop = manifest.get('non_noop_byte_change', { })
    provenance = exact.get('provenance', { })
    runtime_manifest = provenance.get('inflate_runtime_manifest', { })
    recomputed = _artifact_recomputed_score(exact)
    rounded_component_recompute = _score_from_components(exact)
    runtime_root = str(runtime_manifest.get('runtime_root', ''))
    required_inflate_sh = runtime.get('required_inflate_sh')
# WARNING: Decompyle incomplete


def build_report(args = None):
    pr91 = validate_pr91_hpm1_contract(archive = args.pr91_archive, probability_report = args.pr91_probability_report, rerun = args.rerun_pr91_prefix)
    pr92 = validate_pr92_rmb1_contract(manifest_path = args.pr92_manifest, exact_json_path = args.pr92_exact_json)
    overall_status = 'passed_pr92_a_plus_plus_pr91_fail_closed' if pr91['status'] == 'blocked_hpm1_probability_range_contract_mismatch' and pr92['status'] == 'passed_t4_exact_pr92_rmb1_stack_validated' else 'failed_closed'
    if pr92['score_claim']:
        pr92['score_claim']
    return {
        'schema': SCHEMA,
        'tool': TOOL,
        'recorded_at_utc': dt.datetime.now(dt.timezone.utc).replace(microsecond = 0).isoformat(),
        'status': overall_status,
        'score_claim': overall_status.startswith('passed_'),
        'dispatch_performed': False,
        'remote_jobs_dispatched': False,
        'pr91_hpm1': pr91,
        'pr92_rmb1_stack': pr92,
        'next_actions': [
            pr91['safe_next_action'],
            pr92['next_safe_build_command'],
            pr92['next_safe_exact_eval_command_if_rebuilt']] }


def render_ledger(report = None):
    pr91 = report['pr91_hpm1']
    pr92 = report['pr92_rmb1_stack']
    exact = pr92['exact_eval']
    lines = []['# PR91/PR92 Replay Contract Preflight - 2026-05-04'][''][f'''- tool: `{TOOL}`'''][f'''- status: `{report['status']}`''']['- dispatch_performed: `false`']['- remote_jobs_dispatched: `false`']['']['## PR91 HPM1'][''][f'''- status: `{pr91['status']}`'''][f'''- dispatch_allowed: `{pr91['dispatch_allowed']}`'''][f'''- bug_class: `{pr91['classification']['bug_class']}`'''][f'''- failure_reason: `{pr91['classification']['failure_reason']}`'''][f'''- failed_variants: `{', '.join(pr91['classification']['failed_variants'])}`''']['']['PR91 remains fail-closed. The local source-contract variant still fails at `frame=0 group=10 symbol=191` after `5951` decoded symbols, and no tested probability/range variant decodes frame 0.']['']['## PR92 RMB1 Stack'][''][f'''- status: `{pr92['status']}`'''][f'''- evidence_grade: `{pr92['evidence_grade']}`'''][f'''- score: `{exact['score']}`'''][f'''- archive bytes: `{exact['archive_bytes']}`'''][f'''- archive sha256: `{exact['archive_sha256']}`'''][f'''- avg_segnet_dist: `{exact['avg_segnet_dist']}`'''][f'''- avg_posenet_dist: `{exact['avg_posenet_dist']}`'''][f'''- runtime_tree_sha256: `{exact['runtime_tree_sha256']}`''']['']['PR92/RMB1 is not blocked: the validated opportunity is already realized as a pure-rate randmulti recode stacked onto the PR85 STBM1BR frontier.']['']['## Next Safe Commands']['']['```bash'][pr92['next_safe_build_command']]['```'][''][pr92['next_safe_exact_eval_command_if_rebuilt']]['']
    return '\n'.join(lines)


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--pr91-archive', type = Path, default = DEFAULT_PR91_ARCHIVE)
    parser.add_argument('--pr91-probability-report', type = Path, default = DEFAULT_PR91_PROBABILITY_REPORT)
    parser.add_argument('--rerun-pr91-prefix', action = 'store_true')
    parser.add_argument('--pr92-manifest', type = Path, default = DEFAULT_PR92_MANIFEST)
    parser.add_argument('--pr92-exact-json', type = Path, default = DEFAULT_PR92_EXACT_JSON)
    parser.add_argument('--output-json', type = Path, default = DEFAULT_OUTPUT_JSON)
    parser.add_argument('--ledger-md', type = Path, default = DEFAULT_LEDGER)
    parser.add_argument('--stdout', action = 'store_true')
    return parser.parse_args(argv)


def main(argv = None):
    args = parse_args(argv)
    report = build_report(args)
    _write_json(args.output_json, report)
    args.ledger_md.parent.mkdir(parents = True, exist_ok = True)
    args.ledger_md.write_text(render_ledger(report), encoding = 'utf-8')
    if args.stdout:
        sys.stdout.write(_json_text(report))
        return 0
    print(_json_text({
        'status': report['status'],
        'output_json': _rel(args.output_json) }), end = '')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
