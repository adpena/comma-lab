"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``39:21: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``build_pr95_hnerv_residual_atom_plan.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/build_pr95_hnerv_residual_atom_plan.py'
__recovery_spec__ = 'build_pr95_hnerv_residual_atom_plan.recovery_spec.json'
__recovery_ast_error__ = '39:21: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: build_pr95_hnerv_residual_atom_plan.cpython-312.pyc (Python 3.12)

\"\"\"Plan and build PR95-family latent residual atoms.

This is a local-only bridge from the PR95 HNeRV archive anatomy to concrete
charged atom candidates. The default mode emits a diagnostic opportunity
ledger from the archive's latent stream and optional per-pair component trace.
It does not claim score. Candidate building is intentionally stricter: callers
must provide an explicit atom plan, every atom must change a charged latent
value inside ``0.bin``, and the output archive must remain a single-member
deterministic ZIP that uses the existing PR95 runtime contract.
\"\"\"
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import math
import zipfile
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import brotli
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from experiments.profile_pr95_hnerv_muon_packing import LatentPayload, encode_top_blob, parse_latents_raw, parse_top_blob, read_single_member_zip, sha256_bytes, write_stored_zip
CONTEST_ORIGINAL_BYTES = 37545489
RATE_SCORE_PER_BYTE = 25 / CONTEST_ORIGINAL_BYTES
DEFAULT_PR95_REPACK_EXACT_JSON = Path('experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json')
DEFAULT_PR95_RUNTIME_INFLATE = Path('experiments/results/public_pr95_intake_20260504_codex/pr95_src/submissions/hnerv_muon/inflate.sh')
SIGNED_POLICY_FILENAME = 'pr95_hnerv_residual_atom_plan.signed.json'

class PR95AtomPlanError(ValueError):
    '''Raised when a PR95 atom plan is malformed or unsafe.'''
    pass

PairLatentProfile = <NODE:12>()
ComponentTraceProfile = <NODE:12>()

def sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def load_json(path = None):
    payload = json.loads(Path(path).read_text(encoding = 'utf-8'))
    if not isinstance(payload, dict):
        raise PR95AtomPlanError(f'''{path} must contain a JSON object''')
    return payload


def _finite_float(payload = None, key = None):
    value = float(payload[key])
    if not math.isfinite(value):
        raise PR95AtomPlanError(f'''{key} must be finite, got {payload[key]!r}''')
    return value


def _json_sha256(payload = None):
    encoded = json.dumps(payload, sort_keys = True, separators = (',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_exact_baseline(path = None):
    payload = load_json(path)
    required = ('archive_size_bytes', 'avg_posenet_dist', 'avg_segnet_dist', 'score_recomputed_from_components')
# WARNING: Decompyle incomplete


def _optional_finite_float(payload = None, key = None):
    pass
# WARNING: Decompyle incomplete


def _unwrap_component_trace_payload(payload = None):
    if isinstance(payload.get('component_trace'), dict) and 'samples' not in payload:
        return dict(payload['component_trace'])
    if None(payload.get('trace'), dict) and 'samples' not in payload:
        return dict(payload['trace'])


def _close_enough(actual = None, expected = None, *, tolerance):
    pass
# WARNING: Decompyle incomplete


def _sample_optional_float(sample = None, key = None):
    pass
# WARNING: Decompyle incomplete


def load_component_trace(path = None, *, expected_pairs, expected_archive_sha256, expected_archive_size_bytes, exact_baseline):
    pass
# WARNING: Decompyle incomplete


def latent_rows(payload = None):
    pass
# WARNING: Decompyle incomplete


def latent_payload_from_rows(source = None, rows = None):
    if len(rows) != source.n_pairs:
        raise PR95AtomPlanError(f'''expected {source.n_pairs} latent rows, got {len(rows)}''')
    checked = []
    for pair_index, row in enumerate(rows):
        if len(row) != source.latent_dim:
            raise PR95AtomPlanError(f'''pair {pair_index} expected latent_dim={source.latent_dim}, got {len(row)}''')
        out_row = []
        for dim_index, value in enumerate(row):
            ivalue = int(value)
            raise PR95AtomPlanError(f'''latent value out of uint8 range at pair {pair_index}, dim {dim_index}: {ivalue}''')
            out_row.append(ivalue)
        checked.append(tuple(out_row))
    return LatentPayload(n_pairs = source.n_pairs, latent_dim = source.latent_dim, mins_f16 = source.mins_f16, scales_f16 = source.scales_f16, quantized = tuple(checked))


def estimate_min_patch_bytes(active_dims = None):
    return 4 + max(1, int(active_dims)) * 3


def pose_dist_break_even(rate_score_cost = None, avg_posenet_dist = None):
    if avg_posenet_dist <= 0:
        return None
    derivative = 5 / math.sqrt(10 * avg_posenet_dist)
    return rate_score_cost / derivative


def build_pair_profiles(latents = None, *, baseline, component_trace):
    rows = latent_rows(latents)
    profiles = []
    avg_pose = float(baseline['avg_posenet_dist'])
# WARNING: Decompyle incomplete


def validate_single_member_archive(path = None):
    (member, blob, zip_meta) = read_single_member_zip(path)
    if member != '0.bin':
        raise PR95AtomPlanError(f'''PR95-family archive must contain exactly 0.bin, got {member!r}''')
    zf = zipfile.ZipFile(path, 'r')
    bad = zf.testzip()
# WARNING: Decompyle incomplete


def atom_rows_from_plan_payload(plan = None, *, source_archive_sha256, source_member_sha256, latents, plan_label):
    if plan.get('source_archive_sha256') != source_archive_sha256:
        raise PR95AtomPlanError(f'''atom plan source_archive_sha256 mismatch: {plan.get('source_archive_sha256')!r} != {source_archive_sha256}''')
    if plan.get('source_member_sha256') != source_member_sha256:
        raise PR95AtomPlanError('atom plan source_member_sha256 mismatch')
    if plan.get('forbid_sidecars', True) is not True:
        raise PR95AtomPlanError('atom plan must forbid sidecars')
    atoms = plan.get('atoms')
    if not isinstance(atoms, list) or atoms:
        raise PR95AtomPlanError('atom plan must contain at least one atom')
    original_rows = latent_rows(latents)
# WARNING: Decompyle incomplete


def atom_rows_from_plan(plan_path = None, *, source_archive, source_member_sha256, latents):
    plan = load_json(plan_path)
    return atom_rows_from_plan_payload(plan, source_archive_sha256 = sha256_file(source_archive), source_member_sha256 = source_member_sha256, latents = latents, plan_label = str(plan_path))


def _signed_delta_toward_previous(rows = None, pair_index = None, dim_index = None):
    if pair_index <= 0:
        return None
    current = int(rows[pair_index][dim_index])
    previous = int(rows[pair_index - 1][dim_index])
    if current > previous:
        return -1
    if current < previous:
        return 1


def build_signed_atom_policy(*, source_archive_sha256, source_member_sha256, component_trace, latents, ranked_profiles, signed_policy_pairs, signed_policy_dims_per_pair):
    if not component_trace.samples_by_pair:
        raise PR95AtomPlanError('signed policy requires a component trace')
    if signed_policy_pairs <= 0:
        raise PR95AtomPlanError('signed_policy_pairs must be positive')
    if signed_policy_dims_per_pair <= 0:
        raise PR95AtomPlanError('signed_policy_dims_per_pair must be positive')
    rows = latent_rows(latents)
    atoms = []
    selected_pairs = []
# WARNING: Decompyle incomplete


def build_candidate_archive(*, source_archive, output_archive, atom_plan_json, brotli_quality):
    (member, blob, _zip_meta) = validate_single_member_archive(source_archive)
    parts = parse_top_blob(blob)
    latents = parse_latents_raw(parts['latents_raw'])
    (rows, plan_meta) = atom_rows_from_plan(atom_plan_json, source_archive = source_archive, source_member_sha256 = sha256_bytes(blob), latents = latents)
    new_latents = latent_payload_from_rows(latents, rows).to_bytes()
    if new_latents == parts['latents_raw']:
        raise PR95AtomPlanError('atom plan produced unchanged latent raw stream')
    latents_brotli = brotli.compress(new_latents, quality = brotli_quality)
    candidate_blob = encode_top_blob(parts['meta_brotli'], parts['decoder_brotli'], latents_brotli)
    if candidate_blob == blob:
        raise PR95AtomPlanError('atom plan produced unchanged PR95 member blob')
    write_stored_zip(output_archive, member, candidate_blob)
    if sha256_file(output_archive) == sha256_file(source_archive):
        raise PR95AtomPlanError('candidate archive SHA equals source archive SHA')
# WARNING: Decompyle incomplete


def emit_plan(*, source_archive, exact_json, output_dir, component_trace_json, top_k, build_plan_json, signed_policy_pairs, signed_policy_dims_per_pair, build_generated_signed_policy):
    (member, blob, zip_meta) = validate_single_member_archive(source_archive)
    source_archive_sha = sha256_file(source_archive)
    source_member_sha = sha256_bytes(blob)
    parts = parse_top_blob(blob)
    latents = parse_latents_raw(parts['latents_raw'])
    baseline = load_exact_baseline(exact_json)
    if baseline['archive_size_bytes'] != source_archive.stat().st_size:
        raise PR95AtomPlanError(f'''exact baseline archive_size_bytes mismatch: {baseline['archive_size_bytes']} != {source_archive.stat().st_size}''')
    if baseline.get('archive_sha256') and baseline['archive_sha256'] != source_archive_sha:
        raise PR95AtomPlanError(f'''exact baseline archive SHA mismatch: {baseline['archive_sha256']} != {source_archive_sha}''')
    component_trace = load_component_trace(component_trace_json, expected_pairs = latents.n_pairs, expected_archive_sha256 = source_archive_sha, expected_archive_size_bytes = source_archive.stat().st_size, exact_baseline = baseline)
    profiles = build_pair_profiles(latents, baseline = baseline, component_trace = component_trace.samples_by_pair)
    ranked = sorted(profiles, key = (lambda row: (-(row.proxy_rank_signal), row.pair_index)))
    output_dir.mkdir(parents = True, exist_ok = True)
# WARNING: Decompyle incomplete


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--archive', type = Path, required = True)
    parser.add_argument('--exact-json', type = Path, default = DEFAULT_PR95_REPACK_EXACT_JSON)
    parser.add_argument('--output-dir', type = Path, required = True)
    parser.add_argument('--component-trace-json', type = Path)
    parser.add_argument('--top-k', type = int, default = 64)
    parser.add_argument('--build-plan-json', type = Path)
    parser.add_argument('--signed-policy-pairs', type = int, default = 10)
    parser.add_argument('--signed-policy-dims-per-pair', type = int, default = 2)
    parser.add_argument('--build-generated-signed-policy', action = 'store_true', help = 'When --component-trace-json is present, build the candidate from the generated signed policy.')
    parser.add_argument('--stdout', action = 'store_true')
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error('--top-k must be positive')
    if args.signed_policy_pairs <= 0:
        parser.error('--signed-policy-pairs must be positive')
    if args.signed_policy_dims_per_pair <= 0:
        parser.error('--signed-policy-dims-per-pair must be positive')
# WARNING: Decompyle incomplete


def main(argv = None):
    args = parse_args(argv)
    manifest = emit_plan(source_archive = args.archive, exact_json = args.exact_json, output_dir = args.output_dir, component_trace_json = args.component_trace_json, top_k = args.top_k, build_plan_json = args.build_plan_json, signed_policy_pairs = args.signed_policy_pairs, signed_policy_dims_per_pair = args.signed_policy_dims_per_pair, build_generated_signed_policy = args.build_generated_signed_policy)
    if args.stdout:
        print(json.dumps(manifest, indent = 2, sort_keys = True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

"""
