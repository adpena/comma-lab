#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp


def load_csv(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    feature_names = [
        'archive_bytes','scale_w','scale_h','crf','gop','bframes','ref',
        'preset_medium','preset_slow','filter_lanczos_bicubic','filter_lanczos_lanczos','filter_bicubic_bicubic'
    ]
    X = []
    y = []
    ids = []
    for row in rows:
        ids.append(row['run_id'])
        y.append(float(row['score']))
        X.append([float(row[name] or 0.0) for name in feature_names])
    return ids, feature_names, jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)


def standardize(X):
    mean = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)
    std = jnp.where(std < 1e-6, 1.0, std)
    return (X - mean) / std, mean, std


def fit_ridge(X, y, lam=1e-2):
    ones = jnp.ones((X.shape[0], 1), dtype=X.dtype)
    Xb = jnp.concatenate([ones, X], axis=1)
    eye = jnp.eye(Xb.shape[1], dtype=X.dtype)
    eye = eye.at[0,0].set(0.0)
    w = jnp.linalg.solve(Xb.T @ Xb + lam * eye, Xb.T @ y)
    return w


def predict(X, w):
    ones = jnp.ones((X.shape[0], 1), dtype=X.dtype)
    Xb = jnp.concatenate([ones, X], axis=1)
    return Xb @ w


def leave_one_out(X, y, ids, lam=1e-2):
    preds=[]
    abs_err=[]
    for i in range(X.shape[0]):
        mask = jnp.arange(X.shape[0]) != i
        Xt = X[mask]
        yt = y[mask]
        Xs, mean, std = standardize(Xt)
        w = fit_ridge(Xs, yt, lam=lam)
        x = (X[i:i+1] - mean) / std
        pred = float(predict(x, w)[0])
        preds.append(pred)
        abs_err.append(abs(pred - float(y[i])))
    # pairwise ranking accuracy
    total=0
    correct=0
    for i in range(len(preds)):
        for j in range(i+1,len(preds)):
            dy = float(y[i] - y[j])
            dp = preds[i] - preds[j]
            if abs(dy) < 1e-9:
                continue
            total += 1
            if dy == 0:
                continue
            if (dy < 0 and dp < 0) or (dy > 0 and dp > 0):
                correct += 1
    return {
        'loo_predictions': [
            {'run_id': ids[i], 'actual': float(y[i]), 'predicted': preds[i], 'abs_error': abs_err[i]}
            for i in range(len(ids))
        ],
        'mae': sum(abs_err)/len(abs_err),
        'pairwise_accuracy': (correct/total) if total else None,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    ids, feature_names, X, y = load_csv(Path(args.csv))
    loo = leave_one_out(X, y, ids)
    Xs, mean, std = standardize(X)
    w = fit_ridge(Xs, y)
    result = {
        'rows': int(X.shape[0]),
        'features': feature_names,
        'jax_version': jax.__version__,
        'devices': [str(d) for d in jax.devices()],
        'mae': loo['mae'],
        'pairwise_accuracy': loo['pairwise_accuracy'],
        'loo_predictions': loo['loo_predictions'],
        'weights': [float(v) for v in w.tolist()],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({'rows': result['rows'], 'mae': result['mae'], 'pairwise_accuracy': result['pairwise_accuracy']}, indent=2))

if __name__ == '__main__':
    main()
