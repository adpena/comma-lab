#!/usr/bin/env python3
"""Full-resolution, finite-family logistic bound for the token tail.

This is an analysis instrument, not an entropy encoder. Secondary predictors
update their categorical counts and expected masses after each complete plane.
All five named context families are present. Thirty-five shared coefficients
(seven per HPAC winning class) include temperature and hit/miss calibration.
The convex tangent bound applies to this explicitly specified fixed-weight,
online-predictor family, not to every conceivable adaptive mixer.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import ddm_jg2_tail_reencode as jg2
from experiments.ddm_tc1_context_statistics import LEVELS, TOTAL, contexts, frequencies
from experiments.ddm_tc1_token_tail_bound import AXIS, ROOT, H, K, N, W, pinned

F = 7
Q = 1024
WEIGHT_SCALE = 32
LOW, HIGH = -4.0, 127.0 / WEIGHT_SCALE
KEEP_MISS = 2.0**-14
PREDICTOR_STORE = 'predictor_bound_by_winner'


def log2_fixed(value: np.ndarray) -> np.ndarray:
    """Generic Q10 binary logarithm using only exact exponent split and products.

Eleven squarings supply ten fractional bits and a round-up bit. No libm log
enters a prospective receiver decision; this defines the feature, not merely an
approximation substituted by a different implementation on the other host.
"""
    mantissa, exponent = np.frexp(np.asarray(value, dtype=np.float64))
    if np.any(value <= 0):
        raise ValueError('logarithm requires strictly positive inputs')
    work = mantissa * 2.0
    code = (exponent.astype(np.int64) - 1) * (Q * 2)
    fraction = np.zeros(value.shape, dtype=np.int64)
    for _ in range(11):
        work = work * work
        bit = work >= 2.0
        work = np.where(bit, work * 0.5, work)
        fraction = fraction * 2 + bit
    result = (code + fraction + 1) // 2
    if np.any(result < -32768) or np.any(result > 32767):
        raise ValueError('Q10 log feature exceeds int16')
    return result.astype(np.int16)


def predictor_features(rows: np.ndarray, ctx: dict, counts: dict, expected: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read causal predictor state without observing this frame's outcomes."""
    p = frequencies(rows).astype(np.float64) / TOTAL
    arg = p.argmax(axis=1)
    features = np.zeros((len(rows), K, F), dtype=np.int16)
    for j, name in enumerate(LEVELS):
        # Generic KT smoothing; expected masses are integer RC64 frequencies.
        ratio = (counts[name].astype(np.float64) + 0.5) / (expected[name].astype(np.float64) / TOTAL + 0.5)
        ratio = np.clip(ratio, 1 / 16, 16)
        features[:, :, j] = log2_fixed(ratio)[arg * LEVELS[name] + ctx[name]]
    features[:, :, 5] = log2_fixed(p)
    features[np.arange(len(rows)), arg, 6] = Q
    return features, p


def prepare(args) -> dict:
    root = Path(args.root)
    record = pinned(root)
    folder = root / PREDICTOR_STORE
    folder.mkdir(exist_ok=True)
    if shutil.disk_usage(root).free < 3 * 1024**3:
        raise RuntimeError('predictor-bound storage preflight needs 3 GiB free')
    states = sorted(p for p in folder.glob('state_*.npz') if int(p.stem.split('_')[1]) < args.frames)
    counts = {name: np.zeros((K * levels, K), dtype=np.int64) for name, levels in LEVELS.items()}
    expected = {name: np.zeros((K * levels, K), dtype=np.int64) for name, levels in LEVELS.items()}
    run = np.zeros((H, W), dtype=np.uint8)
    start = 0
    totals = np.zeros(4)  # complete bits, kept bits, omitted bits, retained positions
    if states:
        with np.load(states[-1], allow_pickle=False) as data:
            start = int(data['frame'][0]); run = data['run']; totals = data['totals']
            counts = {name: data['counts_' + name] for name in LEVELS}
            expected = {name: data['expected_' + name] for name in LEVELS}
    tokens = jg2.load_tokens(Path(record['field']['path']))
    for t in range(start, args.frames):
        path = root / 'rows' / f'frame_{t:04d}.npz'
        if jg2.file_fact(path) != json.loads(path.with_suffix('.json').read_text()):
            raise RuntimeError('row checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            rows = data['rows']
        ctx = contexts(tokens[t], None if t == 0 else tokens[t - 1], run)
        phi, p = predictor_features(rows, ctx, counts, expected)
        truth = tokens[t].reshape(-1)
        arg = p.argmax(axis=1)
        picked = p[np.arange(H * W), truth]
        bits = -np.log2(picked)
        keep = ((1 - p.max(axis=1)) >= KEEP_MISS) | (truth != arg)
        out = folder / 'frames' / f'frame_{t:04d}.npz'
        arrays = dict(phi=phi[keep], logp=np.log(p[keep]), truth=truth[keep], group=arg[keep].astype(np.uint8))
        if out.exists():
            with np.load(out, allow_pickle=False) as old:
                if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                    raise RuntimeError('resumed predictor trace differs')
        else:
            jg2.atomic_npz(out, arrays)
        jg2.atomic_json(out.with_suffix('.json'), jg2.file_fact(out))
        totals += [bits.sum(), bits[keep].sum(), bits[~keep].sum(), keep.sum()]
        freq = frequencies(rows)
        for name, levels in LEVELS.items():
            code = arg * levels + ctx[name]
            counts[name] += np.bincount(code * K + truth, minlength=levels * K * K).reshape(levels * K, K)
            for k in range(K):
                # Each per-plane integer sum is <=196608*2^31<2^53;
                # bincount's float64 accumulator is therefore integer-exact.
                expected[name][:, k] += np.bincount(code, weights=freq[:, k], minlength=levels * K).astype(np.int64)
        run = (np.zeros((H, W), dtype=np.uint8) if t == 0 else
               np.where(tokens[t] == tokens[t - 1], np.minimum(run + 1, 7), 0).astype(np.uint8))
        if (t + 1) % 25 == 0 or t + 1 == args.frames:
            state = dict(frame=np.array([t + 1]), run=run, totals=totals)
            state.update({'counts_' + k: v for k, v in counts.items()})
            state.update({'expected_' + k: v for k, v in expected.items()})
            jg2.atomic_npz(folder / f'state_{t + 1:04d}.npz', state)
            result = dict(schema='ddm_tc1_predictor_trace.v2', axis=AXIS, score_claim=False,
                          pairs=t + 1, positions=(t + 1) * H * W,
                          complete_bits=float(totals[0]), kept_bits=float(totals[1]),
                          omitted_bits=float(totals[2]), kept_positions=int(totals[3]),
                          omitted_rule='only correct predictions with RC64 miss mass <2^-14; their loss is bounded below by zero',
                          all_wrong_predictions_retained=True, context_families=list(LEVELS),
                          coefficient_count=K * F, int8_weight_bytes=K * F,
                          weight_box=[LOW, HIGH], weight_scale=WEIGHT_SCALE,
                          predictor_update='all categorical count and expected-mass tables after each complete plane',
                          predictor_conditioning='each expert context crossed with coding-row winning class; retains hit/miss calibration information',
                          checkpoint=jg2.file_fact(folder / f'state_{t + 1:04d}.npz'),
                          frame_stride=1, seed=20260909,
                          scope='fixed shared weights; online causal count predictors; full categorical probabilities')
            jg2.atomic_json(folder / f'PREPARE_{t + 1:04d}.json', result)
            print(json.dumps({'pairs': t + 1, 'kept_positions': int(totals[3]), 'omitted_bytes': totals[2] / 8}), flush=True)
    return result


def objective_function(root: Path, frames: int):
    """Load retained real-frame predictors and bind the independent C objective."""
    folder = root / PREDICTOR_STORE
    source = REPO / 'experiments/ddm_tc1_logistic_bound.c'
    library = folder / 'libtc1_bound.dylib'
    command = ['/usr/bin/cc', '-O3', '-std=c11', '-shared', '-fPIC',
               '-ffp-contract=off', '-fno-fast-math', str(source), '-o', str(library)]
    subprocess.run(command, check=True)
    chunks = {k: [] for k in ('logp', 'phi', 'truth', 'group')}
    for t in range(frames):
        path = folder / 'frames' / f'frame_{t:04d}.npz'
        if jg2.file_fact(path) != json.loads(path.with_suffix('.json').read_text()):
            raise RuntimeError('predictor payload checksum failed')
        with np.load(path, allow_pickle=False) as data:
            for k in chunks:
                chunks[k].append(data[k])
    arrays = {k: np.ascontiguousarray(np.concatenate(v)) for k, v in chunks.items()}
    chunks.clear()
    lib = ctypes.CDLL(str(library))
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    i16 = np.ctypeslib.ndpointer(dtype=np.int16, flags='C_CONTIGUOUS')
    u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    lib.tc1_objective.argtypes = [ctypes.c_size_t, f64, i16, u8, u8, f64, ctypes.c_int, f64, f64]
    lib.tc1_objective.restype = ctypes.c_double

    def objective(weights, derivatives=True):
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        gradient = np.zeros((K, F), dtype=np.float64)
        hessian = np.zeros((K, F, F), dtype=np.float64)
        loss = lib.tc1_objective(len(arrays['truth']), arrays['logp'], arrays['phi'],
                                 arrays['truth'], arrays['group'], weights,
                                 int(derivatives), gradient, hessian)
        if not np.isfinite(loss) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            raise RuntimeError('nonfinite logistic objective')
        return loss, gradient, hessian

    return objective, arrays, dict(source=jg2.file_fact(source), library=jg2.file_fact(library), argv=command)


def fit(args) -> dict:
    """Box-constrained Newton descent with a convex supporting-plane certificate.

Restart state is only the current weights and accepted-iteration index: no
optimizer momentum or unrecorded line-search history affects the next iteration.
Every trial vector and every accepted iterate is retained.
"""
    root = Path(args.root)
    record = pinned(root)
    if args.frames != N:
        raise RuntimeError('the fitted build gate requires all 600 pairs')
    if not json.loads((root / 'TRACE_0600.json').read_text())['full_control_byte_identical']:
        raise RuntimeError('full shipped-stream control required')
    entropy_path = root / 'statistics/ENTROPY_0600.json'
    entropy_table = json.loads(entropy_path.read_text())
    if not entropy_table['full_n600'] or entropy_table['positions'] != N * H * W:
        raise RuntimeError('full context-table denominator required before fitting')
    folder = root / PREDICTOR_STORE
    prepared = json.loads((folder / 'PREPARE_0600.json').read_text())
    objective, arrays, build = objective_function(root, N)
    iterations = folder / 'iterations'
    iterations.mkdir(exist_ok=True)
    previous = sorted(iterations.glob('ITER_*.json'))
    weights = np.zeros((K, F))
    start = 0
    if previous:
        last = json.loads(previous[-1].read_text())
        weights = np.array(last['weights'], dtype=np.float64)
        start = last['iteration']
    baseline_loss = objective(np.zeros_like(weights), False)[0]
    baseline_error_bits = baseline_loss / math.log(2) - prepared['kept_bits']
    if abs(baseline_error_bits) > 0.01:
        raise RuntimeError(f'full-resolution baseline mismatch: {baseline_error_bits} bits')
    result = None
    for iteration in range(start, start + args.iterations):
        loss, grad, hess = objective(weights)
        gap = float(np.sum(grad * np.where(grad >= 0, weights - LOW, weights - HIGH)))
        lower_loss = loss - gap
        upper_saved = (prepared['complete_bits'] - lower_loss / math.log(2)) / 8 - K * F - 5
        result = dict(schema='ddm_tc1_convex_mixer_bound.v1', axis=AXIS,
                      score_claim=False, full_n600=True, pairs=N,
                      iteration=iteration, weights=weights.tolist(),
                      log_loss_nats=loss, dual_gap_nats=gap,
                      lower_loss_nats=lower_loss,
                      upper_net_ideal_bytes=upper_saved,
                      kept_only_gain_bytes=(prepared['kept_bits'] - loss / math.log(2)) / 8,
                      omitted_credit_bytes=prepared['omitted_bits'] / 8,
                      counted_weight_bytes=K * F, rider_header_bytes=5,
                      floating_point_boundary='numerically evaluated convex certificate; actual RC64 quantization and framing are priced only by retained encodes',
                      baseline_error_bits=baseline_error_bits, build=build,
                      archive=record['archive'],
                      context_table=jg2.file_fact(entropy_path),
                      joint_mm_estimated_net_bytes=entropy_table['table'][-1]['net_mm_shared_weight_bytes'],
                      scope='35 fixed shared weights over five causal online predictors, temperature, and hit calibration; ideal normalized probabilities',
                      equation='F(v)>=F(w)+grad(F(w)).(v-w); min_box F>=F(w)-sum_j grad_j*(w_j-endpoint_j)',
                      does_not_bound='arbitrary new predictors, adaptive weight trajectories, or arbitrary 64-parameter architectures',
                      construction_gate='all-context MM table plus full-resolution finite-family certificate; neither is a universal compressor bound')
        jg2.atomic_json(folder / 'BOUND_CURRENT.json', result)
        print(json.dumps({'iteration': iteration, 'upper_net_ideal_bytes': upper_saved,
                          'dual_gap_bytes': gap / math.log(2) / 8}), flush=True)
        if gap / math.log(2) / 8 < 0.25:
            jg2.atomic_json(folder / 'BOUND_FINAL.json', result)
            return result
        direction = np.zeros_like(weights)
        for bank in range(K):
            damping = max(1e-8, float(np.trace(hess[bank])) * 1e-10)
            direction[bank] = np.linalg.solve(hess[bank] + np.eye(F) * damping, -grad[bank])
        trials = []
        accepted = False
        for scale in (1., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625):
            trial = np.clip(weights + scale * direction, LOW, HIGH)
            trial_loss = objective(trial, False)[0]
            trials.append(dict(scale=scale, weights=trial.tolist(), loss_nats=trial_loss))
            if trial_loss <= loss + 1e-4 * float(np.sum(grad * (trial - weights))):
                weights = trial
                accepted = True
                break
        jg2.atomic_json(iterations / f'TRIALS_{iteration + 1:04d}.json', trials)
        if not accepted:
            raise RuntimeError('Newton line search did not descend; bound remains provisional')
        jg2.atomic_json(iterations / f'ITER_{iteration + 1:04d}.json',
                        dict(iteration=iteration + 1, weights=weights.tolist(), loss_nats=trial_loss))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'fit'], default='prepare')
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--frames', type=int, default=N)
    parser.add_argument('--iterations', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.frames <= N:
        parser.error('frames must lie in 1..600')
    prepare(args) if args.stage == 'prepare' else fit(args)
