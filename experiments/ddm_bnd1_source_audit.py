#!/usr/bin/env python3
"""Audit retained BND1 inputs; this is not a segment codec or a description floor.

Scorer-free, read-only on source arms. Retains copied inputs, census arrays and
typed limitations. Resume reuses only byte-identical retained artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STORE = Path('/Volumes/APDataStore/pact/ddm_bnd1_boundary_representation')
RP1 = Path('/Volumes/VertigoDataTier/pact/ddm_rp1_rate_directed_predistortion')
SJ1 = Path('/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion')
BASE = Path('/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder'
            '/rungs/lambda_1p0/retained/decoded_tokens.u8')
ARCHIVE = Path('/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/candidate_runtime/archive.zip')
POINTER_SHA = '670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc'
FIELD_SHA = '813bf1e6770161b604348491433386661110001a5eefd8a4fdcd7d70bc25365c'
BASE_SHA = 'cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb'


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def retain(path: Path, payload: bytes) -> dict:
    """Idempotent retention; refuse changed evidence, never delete existing bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f'retained artifact differs; keep bytes and use a new stage: {path}')
    else:
        # A stranded attempt is retained evidence, not a lock on future resumes.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                         suffix='.partial', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.rename(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': expected}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def read_pointer() -> dict:
    value = json.loads((REPO / '.omx/state/canonical_frontier_pointer.json').read_text())
    if value['effective_frontier']['archive_sha256'] != POINTER_SHA:
        raise ValueError('pointer moved: rebind the population before another stage')
    return value


def audit() -> dict:
    pointer = read_pointer()
    if shutil.disk_usage(STORE).free < 64 * 1024**2:
        raise ValueError('storage blocked: require 64 MiB for bounded retained audit')
    sources = {
        'rank.json': RP1 / 'rank_mixer/RANK.json',
        'source_census.json': RP1 / 'retained/MISPREDICTED_CENSUS.json',
        'candidates.npz': RP1 / 'rank_mixer/candidates.npz',
        'tail_control.bin': RP1 / 'rank_mixer/tail_rp1_mixer_control.bin',
        'pass4_field.npz': SJ1 / 'admission_pass4/field_admitted.npz',
        'pass5_result.json': SJ1 / 'passes/pass5_gt/PASS_RESULT.json',
        'pass5_field.npz': SJ1 / 'passes/pass5_gt/field_after.npz',
        'archive.zip': ARCHIVE,
        'pointer.json': REPO / '.omx/state/canonical_frontier_pointer.json',
        'exact_anchor.json': REPO / 'experiments/results/modal_auth_eval_mirror/contest_auth_eval_cmp2_t4_20260909.json',
    }
    manifest = []
    for name, source in sources.items():
        payload = source.read_bytes()
        record = retain(STORE / 'retained' / name, payload)
        manifest.append({'source_path': str(source), **record})
    # From here onward use the retained snapshots, including the live pointer.
    kept = STORE / 'retained'
    rank = json.loads((kept / 'rank.json').read_text())
    old = json.loads((kept / 'source_census.json').read_text())
    pass5 = json.loads((kept / 'pass5_result.json').read_text())
    anchor = json.loads((kept / 'exact_anchor.json').read_text())
    assert digest(kept / 'archive.zip') == POINTER_SHA == anchor['archive_sha256']
    assert json.loads((kept / 'pointer.json').read_text()) == pointer
    assert old['pointer']['archive_sha256'] == POINTER_SHA
    assert digest(kept / 'pass4_field.npz') == FIELD_SHA
    assert digest(kept / 'pass5_field.npz') == pass5['field_npz_sha256']
    identity = rank['identity_control']
    assert identity['byte_identical'] and identity['frames_encoded'] == 600
    assert digest(kept / 'tail_control.bin') == identity['emitted_sha256'] == identity['cmp1_stream_sha256']
    assert (kept / 'tail_control.bin').stat().st_size == identity['emitted_bytes']
    assert BASE.stat().st_size == 600 * 384 * 512 and digest(BASE) == BASE_SHA
    base = np.memmap(BASE, dtype=np.uint8, mode='r', shape=(600, 384 * 512))
    with np.load(kept / 'candidates.npz', allow_pickle=False) as blob:
        cand = {key: blob[key] for key in blob.files}
    frame = cand['frame'].astype(np.int64)
    pos = cand['pos'].astype(np.int64)
    assert np.all((frame >= 0) & (frame < 600))
    assert np.all((pos >= 0) & (pos < 384 * 512))
    assert np.unique(frame).size == 600
    assert len(np.unique(frame * 384 * 512 + pos)) == len(frame)
    assert np.all(cand['sym'] != cand['best'])
    assert np.all((cand['sym'] < 5) & (cand['best'] < 5))
    with np.load(kept / 'pass4_field.npz', allow_pickle=False) as fields:
        for pair in range(600):
            plane = fields[str(pair)].reshape(-1) if str(pair) in fields else base[pair]
            selection = frame == pair
            assert np.array_equal(plane[pos[selection]], cand['sym'][selection]), pair
    n_total = rank['census']['total_tokens']
    n_miss = n_total - rank['census']['n_symbol_is_coder_argmax']
    n_kept = len(frame)
    assert n_total == 600 * 384 * 512
    assert n_miss == old['denominator']['tokens_mispredicted']
    assert n_kept == rank['candidates_dump']['rows'] == old['denominator']['census_covers_tokens']
    row = pos // 512
    bands = np.searchsorted([128, 256, 320], row, side='right')
    band_counts = np.bincount(bands, minlength=4)
    transitions = np.bincount(cand['sym'].astype(np.int64) * 5 + cand['best'], minlength=25).reshape(5, 5)
    assert int(band_counts.sum()) == int(transitions.sum()) == n_kept
    for index, item in enumerate(old['by_row_band']):
        assert band_counts[index] == item.get('tokens', 0)
    counts = io.BytesIO()
    np.savez_compressed(counts, per_pair=np.bincount(frame, minlength=600),
                        stored_to_coder_best=transitions, row_bands=band_counts)
    payloads = [retain(kept / 'census_counts.npz', counts.getvalue())]
    rows = [{'kind': 'stored_class_to_coder_argmax', 'stored_class': a,
             'coder_argmax_class': b, 'tokens': int(transitions[a, b]),
             'denominator': n_kept, 'population_complete': False}
            for a in range(5) for b in range(5)]
    payloads.append(retain(kept / 'census_counts.jsonl',
                          b''.join(json.dumps(x, sort_keys=True).encode() + b'\n' for x in rows)))
    on_edge = old['gt_geometry']['on_gt_edge']['tokens']
    assert on_edge + old['gt_geometry']['in_region_interior']['tokens'] == n_kept
    score = (100 * anchor['avg_segnet_dist'] + math.sqrt(10 * anchor['avg_posenet_dist'])
             + 25 * anchor['archive_size_bytes'] / 37545489)
    assert math.isclose(score, anchor['score'], abs_tol=1e-14)
    result = {
        'schema': 'ddm_bnd1_source_audit.v1', 'research_only': True, 'score_claim': False,
        'axis': '[macOS-CPU scorer-free source audit]', 'charter_complete': False,
        'verdict': 'BLOCKED_SOURCE_PREMISES', 'verdict_scope': 'CHARTER_PROOF_PREMISES_ONLY',
        'pointer': pointer['effective_frontier'], 'archive_bytes': anchor['archive_size_bytes'],
        'recomputed_existing_score': score, 'new_scorer_pairs': 0, 'drawn_pairs': 0,
        'candidate_builds': 0, 'equation_registered': False,
        'population': {'all_tokens': n_total, 'mispredicted': n_miss, 'retained': n_kept,
                       'missing_locations': n_miss - n_kept, 'fraction_retained': n_kept / n_miss,
                       'retained_symbols_match_pass4': True, 'row_band_counts': band_counts.tolist(),
                       'retained_on_gt_edge_source_count': on_edge,
                       'on_gt_edge_population_fraction_lower_bound': on_edge / n_miss,
                       'on_gt_edge_population_fraction_upper_bound': (on_edge + n_miss - n_kept) / n_miss,
                       'edge_geometry_recomputed': False, 'tangent_runs_measured': False,
                       'joint_residual_census_measured': False},
        'prices': {'whole_mixer_envelope_actual_bytes': identity['emitted_bytes'],
                   'misprediction_attribution_ideal_bytes': rank['census']['bits_on_nonargmax_symbols'] / 8,
                   'misprediction_isolated_actual_bytes': None,
                   'segment_actual_bytes': None, 'ternary_offset_actual_bytes': None,
                   'description_floor_bytes': None},
        'residual': {'shipped_pass4_source_cells': pass5['flips_before'],
                     'unshipped_pass5_source_cells': pass5['flips_after'],
                     'unshipped_pass5_source_d_seg': pass5['d_seg_after']},
        'sources': manifest, 'payloads': payloads,
        'base_source': {'path': str(BASE), 'sha256': BASE_SHA, 'bytes': BASE.stat().st_size},
        'producer': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__)),
                     'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                     'upstream_evaluate_sha256': digest(REPO / 'upstream/evaluate.py'),
                     'argv': sys.argv, 'seed': None, 'randomness': 'none'},
        'retention_policy': 'all audit payloads retained; no move or deletion; source trees read-only',
    }
    assert read_pointer()['effective_frontier'] == pointer['effective_frontier']
    receipt = kept / f"SOURCE_AUDIT.{result['producer']['sha256'][:16]}.json"
    result['audit_receipt_path'] = str(receipt)
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        # Revalidate every input and result above, but preserve the first run's
        # invocation custody after a commit changes HEAD or argv spelling.
        for key in ('git_head', 'argv'):
            result['producer'][key] = prior['producer'][key]
    retain(receipt, json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True,
                        help='fixed arm directory; existing evidence must match byte-for-byte')
    args = parser.parse_args()
    if args.resume_from.resolve() != STORE:
        parser.error(f'only the charter-owned store is permitted: {STORE}')
    result = audit()
    print(json.dumps({key: result[key] for key in ('verdict', 'population', 'prices')}, indent=2))


if __name__ == '__main__':
    main()
