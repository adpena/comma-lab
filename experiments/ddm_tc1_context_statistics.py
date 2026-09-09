#!/usr/bin/env python3
"""All-pair conditional-entropy table on verified HPAC coding rows.

Miller--Madow is a bias correction, NOT a universal code-length bound. The
plug-in oracle minimizes the loss only among distributions constant within its
specified cells. The full-resolution mixer leg must preserve that distinction.
No scorer, field change or candidate entropy codec is implemented here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import ddm_jg2_tail_reencode as jg2
from experiments.ddm_tc1_token_tail_bound import AXIS, ROOT, H, K, N, W, pinned

TOTAL = 1 << 31
LEVELS = dict(spatial2=36, spatial3=216, previous=6, run=64, rowband=12)
Y, X = np.indices((H, W))
GROUP = (X % 64 + 2 * (Y % 64)).reshape(-1)


def frequencies(rows: np.ndarray) -> np.ndarray:
    """The C encoder's cast, minimum-one and winner balance, exactly."""
    values = np.asarray(rows, dtype=np.float32).astype(np.float64)
    freq = np.maximum((values * TOTAL).astype(np.int64), 1)
    winner = values.argmax(axis=1)
    freq[np.arange(len(freq)), winner] += TOTAL - freq.sum(axis=1)
    if np.any(freq <= 0) or np.any(freq >= TOTAL):
        raise RuntimeError('invalid RC64 frequencies')
    return freq


def neighbour(plane: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Read only an in-bounds position in a strictly earlier HPAC group."""
    yy, xx = Y + dy, X + dx
    valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    source = np.clip(yy, 0, H - 1) * W + np.clip(xx, 0, W - 1)
    valid &= GROUP[source] < GROUP.reshape(H, W)
    return np.where(valid, plane.reshape(-1)[source], K).reshape(-1).astype(np.int64)


def contexts(plane: np.ndarray, previous: np.ndarray | None, temporal_run: np.ndarray) -> dict[str, np.ndarray]:
    left = neighbour(plane, 0, -1)
    up = neighbour(plane, -1, 0)
    upright = neighbour(plane, -1, 1)
    # Horizontal run counts only known, contiguous left symbols. Row/patch
    # boundaries reset it. Temporal run is state at t-1, never the unknown t.
    spatial_run = np.zeros(H * W, dtype=np.int64)
    active = left != K
    for distance in range(1, 8):
        value = neighbour(plane, 0, -distance)
        active &= (value != K) & (value == left)
        spatial_run += active
    coloc = np.full(H * W, K, dtype=np.int64) if previous is None else previous.reshape(-1).astype(np.int64)
    return dict(spatial2=left * 6 + up,
                spatial3=(left * 6 + up) * 6 + upright,
                previous=coloc,
                run=spatial_run * 8 + np.minimum(temporal_run.reshape(-1), 7),
                rowband=(Y // 32).reshape(-1))


def entropy(counts: np.ndarray) -> dict:
    counts = np.asarray(counts, dtype=np.float64).reshape(-1, K)
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError('empirical entropy requires finite nonnegative integer counts')
    totals = counts.sum(axis=1)
    occupied = totals > 0
    counts = counts[occupied]
    totals = totals[occupied]
    nonzero = counts > 0
    conditional = np.divide(counts, totals[:, None], out=np.ones_like(counts), where=nonzero)
    plugin = float(-(counts * np.log2(conditional)).sum())
    degrees = int(nonzero.sum() - len(counts))
    correction = degrees / (2 * math.log(2))
    return dict(symbols=int(totals.sum()), occupied_contexts=len(counts),
                occupied_context_symbols=int(nonzero.sum()), degrees=degrees,
                plugin_bits=plugin, miller_madow_bits=plugin + correction,
                correction_bits=correction)


def context_mixing_bound(realized_bits: float, counts: np.ndarray, counted_bytes: int) -> dict:
    """Price a specified constant-cell oracle; refuse a universal-mixer reading."""
    if not math.isfinite(realized_bits) or realized_bits < 0 or counted_bytes < 0:
        raise ValueError('finite nonnegative bit and byte costs required')
    result = entropy(counts)
    return dict(plugin_oracle_net_bytes=(realized_bits - result['plugin_bits']) / 8 - counted_bytes,
                miller_madow_estimated_net_bytes=(realized_bits - result['miller_madow_bits']) / 8 - counted_bytes,
                universal_mixer_ceiling=False,
                scope='distributions constant in the explicitly supplied context cells')


def summarize(root: Path, state: dict, joint: dict[int, int], frame: int) -> dict:
    base = entropy(state['counts_base'])
    base_bits = float(state['class_bits'].sum())
    table = []
    for name in LEVELS:
        row = entropy(state['counts_' + name])
        row.update(context=name,
                   conditional_information_plugin_bits=base['plugin_bits'] - row['plugin_bits'],
                   conditional_information_mm_bits=base['miller_madow_bits'] - row['miller_madow_bits'],
                   saved_vs_realized_plugin_bytes=(base_bits - row['plugin_bits']) / 8,
                   saved_vs_realized_mm_bytes=(base_bits - row['miller_madow_bits']) / 8,
                   counted_shared_weight_bytes=5,
                   rider_header_bytes=5,
                   net_mm_shared_weight_bytes=(base_bits - row['miller_madow_bits']) / 8 - 10,
                   table_fp16_dense_probability_bytes=int(row['occupied_contexts'] * 4 * 2),
                   table_index_bytes='additional; not claimed free')
        table.append(row)
    keys = np.array(sorted(joint), dtype=np.int64)
    values = np.array([joint[int(key)] for key in keys], dtype=np.int64)
    cells, inverse = np.unique(keys // K, return_inverse=True)
    joint_counts = np.zeros((len(cells), K), dtype=np.int64)
    joint_counts[inverse, keys % K] = values
    union = entropy(joint_counts)
    union.update(context='joint_all_five',
                 conditional_information_plugin_bits=base['plugin_bits'] - union['plugin_bits'],
                 conditional_information_mm_bits=base['miller_madow_bits'] - union['miller_madow_bits'],
                 saved_vs_realized_plugin_bytes=(base_bits - union['plugin_bits']) / 8,
                 saved_vs_realized_mm_bytes=(base_bits - union['miller_madow_bits']) / 8,
                 counted_shared_weight_bytes=35,
                 rider_header_bytes=5,
                 net_mm_shared_weight_bytes=(base_bits - union['miller_madow_bits']) / 8 - 40,
                 table_fp16_dense_probability_bytes=int(union['occupied_contexts'] * 4 * 2),
                 table_index_bytes='additional; not claimed free')
    table.append(union)
    return dict(schema='ddm_tc1_conditional_entropy.v1', axis=AXIS,
                score_claim=False, pairs=frame, positions=frame * H * W,
                selection='all positions of all completed pairs; final verdict requires all 600',
                full_n600=frame == N, real_rc64_ideal_bits=base_bits,
                class_counts=state['class_counts'].tolist(), class_bits=state['class_bits'].tolist(),
                class_bits_per_symbol=np.divide(state['class_bits'], state['class_counts'], out=np.zeros(K), where=state['class_counts'] > 0).tolist(),
                binary_decomposition_bits=state['decomposition'].tolist(),
                base=base, table=table,
                bucket_definition='coding-row argmax class x floor(-2 log2(RC64 miss probability)), clipped at 63',
                caveat='Constant-cell oracle only; Miller-Madow is not a universal ceiling and coarse buckets lose within-bin HPAC resolution.',
                mechanism_reductions='No pair subsampling. Joint contexts are measured directly, never summed. No acausal neighbour or current-plane temporal run.',
                verdict='NO_BUILD_DECISION_UNTIL_FULL_RESOLUTION_BOUND')


def context_bit_receipt(store: Path, state: dict, frames: int) -> dict:
    """Persist every occupied context's actual bits and symbol denominator."""
    records = []
    for name, levels in LEVELS.items():
        totals = state['counts_' + name].sum(axis=1)
        for code in np.flatnonzero(totals):
            bits = float(state['bits_' + name][code])
            records.append(dict(context=name, hpac_bucket=int(code // levels),
                                extra_context=int(code % levels),
                                symbols=int(totals[code]), bits=bits,
                                bits_per_symbol=bits / int(totals[code]),
                                symbol_counts=state['counts_' + name][code].tolist()))
    path = store / f'REALIZED_CONTEXT_BITS_{frames:04d}.json'
    jg2.atomic_json(path, dict(axis=AXIS, score_claim=False, pairs=frames,
                             records=records, selection='every occupied cell'))
    return jg2.file_fact(path)


def analyze(args) -> dict:
    root = Path(args.root)
    record = pinned(root)
    full = root / 'TRACE_0600.json'
    if args.frames == N and (not full.exists() or not json.loads(full.read_text())['full_control_byte_identical']):
        raise RuntimeError('n600 bound requires byte-identical trace control')
    store = root / 'statistics'
    store.mkdir(exist_ok=True)
    saved = sorted(p for p in store.glob('state_*.npz') if int(p.stem.split('_')[1]) < args.frames)
    if saved:
        with np.load(saved[-1], allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        start = int(state.pop('frame')[0])
        joint = dict(zip(state.pop('joint_keys').tolist(), state.pop('joint_values').tolist()))
    else:
        start = 0
        state = dict(counts_base=np.zeros((K * 64, K), dtype=np.int64),
                     class_counts=np.zeros(K, dtype=np.int64), class_bits=np.zeros(K),
                     decomposition=np.zeros(3), temporal_run=np.zeros((H, W), dtype=np.uint8))
        for name, levels in LEVELS.items():
            state['counts_' + name] = np.zeros((K * 64 * levels, K), dtype=np.int64)
            state['bits_' + name] = np.zeros(K * 64 * levels)
        joint = {}
    tokens = jg2.load_tokens(Path(record['field']['path']))
    for t in range(start, args.frames):
        path = root / 'rows' / f'frame_{t:04d}.npz'
        if jg2.file_fact(path) != json.loads(path.with_suffix('.json').read_text()):
            raise RuntimeError('retained row checksum failed')
        with np.load(path, allow_pickle=False) as data:
            freq = frequencies(data['rows'])
        truth = tokens[t].reshape(-1).astype(np.int64)
        arg = freq.argmax(axis=1)
        p = freq.astype(np.float64) / TOTAL
        selected = p[np.arange(H * W), truth]
        missp = 1 - p[np.arange(H * W), arg]
        wrong = truth != arg
        bit = -np.log2(selected)
        state['class_counts'] += np.bincount(truth, minlength=K)
        state['class_bits'] += np.bincount(truth, weights=bit, minlength=K)
        yes_bits = float(-np.log2(1 - missp[~wrong]).sum())
        no_bits = float(-np.log2(missp[wrong]).sum())
        other_bits = float(-(np.log2(selected[wrong]) - np.log2(missp[wrong])).sum())
        state['decomposition'] += [yes_bits, no_bits, other_bits]
        bucket = arg * 64 + np.minimum((-2 * np.log2(missp)).astype(np.int64), 63)
        state['counts_base'] += np.bincount(bucket * K + truth, minlength=K * 64 * K).reshape(-1, K)
        ctx = contexts(tokens[t], None if t == 0 else tokens[t - 1], state['temporal_run'])
        joint_code = bucket.copy()
        for name, levels in LEVELS.items():
            code = bucket * levels + ctx[name]
            size = K * 64 * levels
            state['counts_' + name] += np.bincount(code * K + truth, minlength=size * K).reshape(-1, K)
            state['bits_' + name] += np.bincount(code, weights=bit, minlength=size)
            joint_code = joint_code * levels + ctx[name]
        keys, counts = np.unique(joint_code * K + truth, return_counts=True)
        for key, count in zip(keys.tolist(), counts.tolist(), strict=True):
            joint[key] = joint.get(key, 0) + count
        state['temporal_run'] = (np.zeros((H, W), dtype=np.uint8) if t == 0 else
                                 np.where(tokens[t] == tokens[t - 1], np.minimum(state['temporal_run'] + 1, 7), 0).astype(np.uint8))
        if (t + 1) % 50 == 0 or t + 1 == args.frames:
            jkeys = np.array(sorted(joint), dtype=np.int64)
            jg2.atomic_npz(store / f'state_{t + 1:04d}.npz',
                           dict(state, frame=np.array([t + 1]), joint_keys=jkeys,
                                joint_values=np.array([joint[int(k)] for k in jkeys], dtype=np.int64)))
            result = summarize(root, state, joint, t + 1)
            result['checkpoint'] = jg2.file_fact(store / f'state_{t + 1:04d}.npz')
            if t + 1 == args.frames:
                result['realized_context_bits'] = context_bit_receipt(store, state, t + 1)
            jg2.atomic_json(store / f'ENTROPY_{t + 1:04d}.json', result)
            print(json.dumps({'stage': 'entropy', 'pairs': t + 1, 'joint_mm_net_bytes': result['table'][-1]['net_mm_shared_weight_bytes']}), flush=True)
    return result


def register_result(root: Path) -> dict:
    """Register the complete table through the canonical locked append API."""
    from tac.canonical_equations import CanonicalEquation, EmpiricalAnchor, query_equations, register_canonical_equation
    from tac.provenance import build_provenance_for_macos_cpu_advisory

    pinned(root)
    trace = json.loads((root / 'TRACE_0600.json').read_text())
    path = root / 'statistics/ENTROPY_0600.json'
    table = json.loads(path.read_text())
    if not trace['full_control_byte_identical'] or table['positions'] != N * H * W:
        raise RuntimeError('canonical anchor requires exact full-field trace custody')
    equation_id = 'token_tail_context_mixing_bound_v1'
    if any(eq.equation_id == equation_id for eq in query_equations()):
        raise RuntimeError('equation already registered; inspect and append an anchor, never duplicate it')
    stamp = datetime.now(UTC).isoformat()
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=jg2.file_fact(path)['sha256'], source_path=str(path), captured_at_utc=stamp)
    anchor = EmpiricalAnchor(
        anchor_id='ddm_tc1_full_n600_context_table_20260909', measurement_utc=stamp,
        inputs=dict(archive=trace['inputs'], pairs=N, symbols=N * H * W,
                    real_rc64_ideal_bits=table['real_rc64_ideal_bits']),
        predicted_output=dict(kind='descriptive identity; no independent compression forecast',
                              expression='(realized_bits - conditional_entropy_bits)/8 - counted_bytes'),
        empirical_output=dict(base=table['base'], table=table['table'],
                              scope=table['caveat'], universal_mixer_ceiling=False),
        residual=0.0, source_artifact=str(path),
        measurement_method='All encoder-input frequencies; exact n600 stream reproduction; empirical context census with Miller-Madow correction. Residual zero denotes arithmetic identity, not predictive validation.',
        provenance=provenance, empirical_verification_status='VERIFIED_VIA_EMPIRICAL_ANCHOR')
    equation = CanonicalEquation(
        equation_id=equation_id, name='Token-tail context oracle and counted-cost test',
        one_line_summary='Price empirical conditional context structure against actual RC64 bits; distinguish the constant-cell oracle from any full-resolution mixer ceiling.',
        latex_form=r'\Delta B=(L_{RC64}-H_{ML}(Y|B,C)-(K_{BCY}-K_{BC})/(2\ln2))/8-B_w',
        python_callable_module_path='experiments.ddm_tc1_context_statistics:context_mixing_bound',
        domain_of_validity=dict(
            domain_of_validity_included=['fixed full field and coding order', 'explicit causal context cells', 'actual frequency-based RC64 costs'],
            domain_of_validity_excluded=['universal ceiling over full-resolution logistic mixtures', 'summing marginal context gains', 'prefix extrapolation', 'uncounted oracle tables treated as shipped models'],
            universal_mixer_ceiling=False,
            miller_madow='bias correction, not a rigorous finite-sample entropy bound'),
        units_in=dict(realized_bits='bits', counts='symbols', counted_bytes='bytes'),
        units_out=dict(plugin_oracle_net_bytes='bytes', miller_madow_estimated_net_bytes='bytes'),
        empirical_anchors=(anchor,), predicted_vs_empirical_residual={},
        last_calibration_utc=stamp, next_recalibration_trigger='when_operator_invokes_recalibrate_equation',
        canonical_consumers=('experiments.ddm_tc1_shared_predictor_bound.fit',),
        canonical_producers=('experiments.ddm_tc1_context_statistics.analyze',), provenance=provenance)
    register_canonical_equation(equation, agent='codex', subagent_id='ddm_tc1',
                                notes='Full bound-table anchor; explicit refusal of universal-ceiling interpretation.')
    receipt = dict(equation=equation.to_dict(), entropy_receipt=jg2.file_fact(path))
    jg2.atomic_json(REPO / '.omx/research/ddm_tc1_context_mixing_bound_registration_20260909.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--frames', type=int, default=N)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.frames <= N:
        parser.error('--frames must lie in 1..600')
    register_result(Path(args.root)) if args.register else analyze(args)
