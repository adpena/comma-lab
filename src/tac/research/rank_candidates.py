#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import jax
import jax.numpy as jnp

FEATURES = [
    'archive_bytes','scale_w','scale_h','crf','gop','bframes','ref',
    'preset_medium','preset_slow','filter_lanczos_bicubic','filter_lanczos_lanczos','filter_bicubic_bicubic'
]

def load_rows(path: Path, target: str | None):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    ids=[]; X=[]; y=[]
    for r in rows:
        ids.append(r['run_id'])
        X.append([float(r[k] or 0.0) for k in FEATURES])
        if target:
            y.append(float(r[target]))
    X = jnp.array(X, dtype=jnp.float32)
    if target:
        return ids, X, jnp.array(y, dtype=jnp.float32)
    return ids, X

def standardize(X):
    mean = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)
    std = jnp.where(std < 1e-6, 1.0, std)
    return (X - mean)/std, mean, std

def fit_ridge(X, y, lam=1e-2):
    Xs, mean, std = standardize(X)
    Xb = jnp.concatenate([jnp.ones((Xs.shape[0],1), dtype=X.dtype), Xs], axis=1)
    eye = jnp.eye(Xb.shape[1], dtype=X.dtype).at[0,0].set(0.0)
    w = jnp.linalg.solve(Xb.T @ Xb + lam * eye, Xb.T @ y)
    return w, mean, std

def predict(X, w, mean, std):
    Xs = (X - mean)/std
    Xb = jnp.concatenate([jnp.ones((Xs.shape[0],1), dtype=X.dtype), Xs], axis=1)
    return Xb @ w

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    train_ids, X, y = load_rows(Path(args.train), 'score')
    cand_ids, C = load_rows(Path(args.candidates), None)
    w, mean, std = fit_ridge(X, y)
    preds = predict(C, w, mean, std)
    ranked = sorted([
        {'run_id': rid, 'predicted_score': float(pred)}
        for rid, pred in zip(cand_ids, preds.tolist())
    ], key=lambda x: x['predicted_score'])
    out = {
        'jax_version': jax.__version__,
        'devices': [str(d) for d in jax.devices()],
        'ranked_candidates': ranked,
        'training_rows': len(train_ids),
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
