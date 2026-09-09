#!/usr/bin/env python3
"""Differential and causality checks on retained real-field analysis inputs.

These are numerical implementation tests, never a prefix-based empirical bound.
All materialized test vectors are retained in the arm's SSD verification store.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import ddm_jg2_tail_reencode as jg2
from experiments.ddm_tc1_context_statistics import GROUP, LEVELS, contexts
from experiments.ddm_tc1_shared_predictor_bound import HIGH, LOW, PREDICTOR_STORE, F, K, log2_fixed, objective_function
from experiments.ddm_tc1_token_tail_bound import AXIS, ROOT, H, W, pinned


def verify(root: Path = ROOT) -> dict:
    """Compare the C calculus to NumPy and perturb unavailable real symbols."""
    record = pinned(root)
    store = root / ('verification_' + PREDICTOR_STORE)
    store.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260909)
    objective, arrays, build = objective_function(root, 2)
    weights = rng.uniform(-0.15, 0.15, (K, F))
    trials = rng.uniform(LOW, HIGH, (8, K, F))
    jg2.atomic_npz(store / 'calculus_vectors.npz', dict(weights=weights, tangent_trials=trials))
    loss, gradient, hessian = objective(weights)
    x = arrays['phi'].astype(np.float64) * (math.log(2) / 1024)
    logits = arrays['logp'] + np.einsum('nkf,nf->nk', x, weights[arrays['group']])
    maximum = logits.max(axis=1)
    exponential = np.exp(logits - maximum[:, None])
    probabilities = exponential / exponential.sum(axis=1)[:, None]
    reference_loss = (np.log(exponential.sum(axis=1)) + maximum - logits[np.arange(len(logits)), arrays['truth']]).sum()
    means = np.einsum('nk,nkf->nf', probabilities, x)
    individual = means - x[np.arange(len(x)), arrays['truth']]
    reference_gradient = np.array([individual[arrays['group'] == k].sum(axis=0) for k in range(K)])
    centered = x - means[:, None, :]
    reference_hessian = np.array([
        np.einsum('nk,nka,nkb->ab', probabilities[arrays['group'] == k],
                  centered[arrays['group'] == k], centered[arrays['group'] == k])
        for k in range(K)])
    np.testing.assert_allclose(loss, reference_loss, atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(gradient, reference_gradient, atol=1e-7, rtol=1e-9)
    np.testing.assert_allclose(hessian, reference_hessian, atol=1e-7, rtol=1e-9)
    finite = np.zeros_like(weights)
    for k in range(K):
        for f in range(F):
            a = weights.copy(); b = weights.copy()
            a[k, f] += 1e-5; b[k, f] -= 1e-5
            finite[k, f] = (objective(a, False)[0] - objective(b, False)[0]) / 2e-5
    np.testing.assert_allclose(gradient, finite, atol=2e-4, rtol=1e-5)
    tangent_slack = []
    for trial in trials:
        trial_loss = objective(trial, False)[0]
        slack = trial_loss - loss - float(np.sum(gradient * (trial - weights)))
        if slack < -1e-6:
            raise AssertionError('supporting-plane inequality failed')
        tangent_slack.append(slack)
    tokens = jg2.load_tokens(Path(record['field']['path']))
    run = np.zeros((H, W), dtype=np.uint8)
    original = contexts(tokens[17], tokens[16], run)
    perturbations = {}
    checked_groups = (0, 1, 2, 7, 31, 63, 64, 95, 126, 127, 188, 189)
    for group in checked_groups:
        changed = tokens[17].copy().reshape(-1)
        unavailable = group <= GROUP
        changed[unavailable] = (changed[unavailable] + 1) % K
        changed = changed.reshape(H, W)
        perturbations[f'group_{group:03d}'] = changed
    jg2.atomic_npz(store / 'causal_perturbations.npz', perturbations)
    for group in checked_groups:
        changed = perturbations[f'group_{group:03d}']
        candidate = contexts(changed, tokens[16], run)
        selected = group == GROUP
        for name in LEVELS:
            np.testing.assert_array_equal(candidate[name][selected], original[name][selected])
    powers = np.ldexp(np.ones(46), np.arange(-31, 15))
    np.testing.assert_array_equal(log2_fixed(powers), np.arange(-31, 15) * 1024)
    log_values = np.exp2(rng.uniform(-31, 14, 10000))
    jg2.atomic_npz(store / 'logarithm_vectors.npz', dict(values=log_values, result=log2_fixed(log_values)))
    log_error = float(np.max(np.abs(log2_fixed(log_values).astype(float) / 1024 - np.log2(log_values))))
    if log_error > 0.5 / 1024 + 1e-12:
        raise AssertionError('Q10 logarithm exceeds rounding-error allowance')
    result = dict(schema='ddm_tc1_instrument_verification.v1', axis=AXIS, score_claim=False,
                  verdict='PASS', verification_scope='numerical/causal implementation only; no prefix research verdict',
                  real_calculus_positions=len(arrays['truth']), causal_real_pair=17,
                  causal_groups_checked=list(checked_groups),
                  loss_absolute_error=abs(loss - reference_loss),
                  gradient_max_error=float(np.max(np.abs(gradient - reference_gradient))),
                  hessian_max_error=float(np.max(np.abs(hessian - reference_hessian))),
                  finite_difference_max_error=float(np.max(np.abs(gradient - finite))),
                  supporting_plane_minimum_slack=min(tangent_slack), logarithm_max_error=log_error,
                  build=build, seed=20260909,
                  retained_vectors=[jg2.file_fact(p) for p in sorted(store.glob('*.npz'))])
    jg2.atomic_json(store / 'VERIFICATION.json', result)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    verify(parser.parse_args().root)
