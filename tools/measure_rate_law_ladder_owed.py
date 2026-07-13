#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the D36/D37 owed terms of ``rate_law_ladder_v1`` on frozen n600 data.

This is a read-only, deterministic, local-CPU measurement.  It does not render,
train, mutate a run directory, or write a derived cache.  The only outputs are
small atomic JSON receipts under ``--out-dir``.

D36 completion (explicitly ASSUMED in the receipt): the shipped symmetric-int8
per-frame code table is the operational finite label ``q_G(W)`` and frozen
SegNet-label block statistics plus the official cached PoseNet six-vector form
a finite, receiver-oriented proxy for ``U(W)``.  Five-fold cross-fitted ridge
prediction followed by modulo-256 residual coding gives a lossless conditional
codelength upper bound.  The predictor/table charge is reported separately.

D37 completion (explicitly ASSUMED in the receipt): the frozen realized-through-R
argmax map supplies ``F``; the adjacent GT label at a local four-neighbour
boundary supplies the active class-pair proxy ``C`` because runner-up logits are
not stored; cached official PoseNet six-vectors supply ``Qxi``; and the ratio of
the two adjacent margins supplies a local receiver tie coordinate ``Phi``.
Nested pair-blocked cross-fitting compares calibrated beta-binomial context
tables with and without ``C``.  A normalized-margin/velocity proxy is reported
as the registered third diagnostic, not promoted as an exact scorer Jacobian.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import brotli
import numpy as np

N_CLASSES = 5
RATE_DENOM = 37_545_489
SEED = 20260713


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stored_npy_memmap(npz_path: Path, key: str) -> np.memmap:
    """Memory-map one ZIP_STORED ``.npy`` member without inflating the full NPZ."""
    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{npz_path}:{member} is compressed; read-only memmap unavailable")
        local_header = int(info.header_offset)
    with npz_path.open("rb") as f:
        f.seek(local_header)
        header = f.read(30)
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise ValueError(f"bad local ZIP header for {npz_path}:{member}")
        npy_start = local_header + 30 + int(fields[-2]) + int(fields[-1])
        f.seek(npy_start)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        data_offset = f.tell()
    return np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=data_offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def symmetric_int8(a: np.ndarray) -> tuple[np.ndarray, float]:
    scale_max = float(np.abs(a).max()) + 1e-8
    q = np.clip(np.round(a / scale_max * 127.0), -127, 127).astype(np.int8)
    return q, scale_max / 127.0


def bootstrap_interval(values: np.ndarray, *, n_boot: int = 10_000) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(SEED)
    totals = np.empty(n_boot, np.float64)
    # Chunk to avoid constructing a 10k x 600 index matrix.
    for start in range(0, n_boot, 250):
        stop = min(n_boot, start + 250)
        idx = rng.integers(0, values.size, size=(stop - start, values.size))
        totals[start:stop] = values[idx].sum(axis=1)
    lo, hi = np.quantile(totals, [0.025, 0.975])
    return {
        "estimate": float(values.sum()),
        "bootstrap_mean": float(totals.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_bootstrap": int(n_boot),
        "resampling_unit": "pair",
    }


# ---------------------------------------------------------------------------
# D36: conditional codelength of the shipped per-frame code table
# ---------------------------------------------------------------------------
def d36_features(lstars: np.memmap, poses: np.ndarray, n: int) -> np.ndarray:
    """Finite proxy for U: 4x4 class areas, directed adjacencies, official pose."""
    out = np.empty((n, 16 * N_CLASSES + N_CLASSES * N_CLASSES + 6), np.float64)
    for i in range(n):
        lab = np.asarray(lstars[i], np.int8)
        h, w = lab.shape
        block = []
        for c in range(N_CLASSES):
            block.append((lab == c).reshape(4, h // 4, 4, w // 4).mean(axis=(1, 3)).reshape(-1))
        right = lab[:, :-1].astype(np.int16) * N_CLASSES + lab[:, 1:].astype(np.int16)
        down = lab[:-1, :].astype(np.int16) * N_CLASSES + lab[1:, :].astype(np.int16)
        adj = np.bincount(
            np.concatenate([right.reshape(-1), down.reshape(-1)]), minlength=N_CLASSES**2
        ).astype(np.float64)
        adj /= max(1.0, adj.sum())
        out[i] = np.concatenate([np.concatenate(block), adj, poses[i]])
    return out


def ridge_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> np.ndarray:
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-10] = 1.0
    xt = (x_train - mu) / sd
    xv = (x_test - mu) / sd
    ym = y_train.mean(axis=0)
    yc = y_train - ym
    # macOS Accelerate can leave harmless floating-point status flags after DGEMM
    # even when every operand/result is finite.  The finite assertions below are
    # the authority; suppress only those status-flag warnings around BLAS calls.
    with np.errstate(all="ignore"):
        a = xt.T @ xt + float(alpha) * np.eye(xt.shape[1])
        coef = np.linalg.solve(a, xt.T @ yc)
        pred = xv @ coef + ym
    if not (np.isfinite(a).all() and np.isfinite(coef).all() and np.isfinite(pred).all()):
        raise FloatingPointError("non-finite D36 ridge state")
    return pred


def select_ridge_alpha(x: np.ndarray, y: np.ndarray, pair_ids: np.ndarray) -> float:
    alphas = (0.1, 1.0, 10.0, 100.0, 1000.0)
    scores = np.zeros(len(alphas), np.float64)
    for inner in range(3):
        va = pair_ids % 3 == inner
        tr = ~va
        for j, alpha in enumerate(alphas):
            pred = ridge_fit_predict(x[tr], y[tr], x[va], alpha)
            scores[j] += float(np.mean((pred - y[va]) ** 2))
    return float(alphas[int(scores.argmin())])


def fold_codelength_sensitivity(fold_bits: list[int], full_bits: int) -> dict[str, Any]:
    """Fold spread without a fake bootstrap over a dictionary coder.

    Resampling symbol rows creates duplicates that Brotli can exploit and therefore
    gives a severely downward-biased bootstrap.  Instead extrapolate each disjoint
    held-out fold to n600, remove the common per-stream flush shift, and use the
    observed fold spread around the exact full-stream measurement.
    """
    extrapolated = 5.0 * np.asarray(fold_bits, np.float64)
    centered = float(full_bits) + extrapolated - extrapolated.mean()
    se = float(centered.std(ddof=1) / np.sqrt(centered.size))
    half = 2.7764451051977987 * se  # t_(0.975, df=4)
    return {
        "method": "five disjoint held-out pair folds; extrapolated fold lengths centered on exact full-stream Brotli length",
        "extrapolated_fold_bits": extrapolated.tolist(),
        "centered_fold_bits": centered.tolist(),
        "standard_error_bits": se,
        "ci95_low_bits": float(full_bits - half),
        "ci95_high_bits": float(full_bits + half),
        "warning": "finite-coder lengths are order- and dictionary-dependent; this is fold sensitivity, not an iid symbol bootstrap",
    }


def measure_d36(args: argparse.Namespace, lstars: np.memmap, poses: np.ndarray) -> dict[str, Any]:
    t0 = time.time()
    with np.load(args.checkpoint, allow_pickle=False) as z:
        code = np.asarray(z["code"], np.float32)
    if code.shape != (2 * args.n, 32):
        raise ValueError(f"D36 expects code {(2 * args.n, 32)}, got {code.shape}")
    q, scale = symmetric_int8(code)
    raw_rows = q.reshape(args.n, -1)
    raw_bytes = len(brotli.compress(q.tobytes(), quality=11))

    x = d36_features(lstars, poses, args.n)
    y = raw_rows.astype(np.float64)
    pair_ids = np.arange(args.n)
    pred = np.empty_like(y)
    selected: list[float] = []
    fold_cond_bits: list[int] = []
    for outer in range(5):
        va = pair_ids % 5 == outer
        tr = ~va
        alpha = select_ridge_alpha(x[tr], y[tr], pair_ids[tr])
        selected.append(alpha)
        pred[va] = ridge_fit_predict(x[tr], y[tr], x[va], alpha)
    pred_q = np.clip(np.rint(pred), -127, 127).astype(np.int16)
    residual = ((y.astype(np.int16) - pred_q) % 256).astype(np.uint8)
    # Decoder proof for the modulo residual representation.
    decoded = ((pred_q + residual.astype(np.int16)) % 256).astype(np.uint8).view(np.int8)
    if not np.array_equal(decoded, raw_rows):
        raise AssertionError("D36 modulo-256 residual does not decode bit-exactly")
    conditional_bytes = len(brotli.compress(residual.tobytes(), quality=11))
    for outer in range(5):
        fold_cond_bits.append(
            8 * len(brotli.compress(residual[pair_ids % 5 == outer].tobytes(), quality=11))
        )

    chosen_bytes = min(raw_bytes, conditional_bytes)
    # A deterministic int16 matrix is an intentionally conservative receiver charge.
    n_features = int(x.shape[1])
    model_bytes = 32 + 2 * (n_features + 1) * y.shape[1] + 4 * 2 * n_features
    archive_bytes = int(args.archive_bytes)
    result = {
        "debt_id": "D36",
        "estimator": "5-fold pair-blocked cross-fitted ridge predictor; round/clamp; lossless modulo-256 residual; Brotli q11; min(raw, conditional) selector",
        "status": "MEASURED",
        "authority": "[macOS-CPU codelength advisory; n600; non-promotable]",
        "operational_completion": {
            "q_G_W": "ASSUMED: current checkpoint shipped symmetric-int8 per-frame code symbols",
            "U_W": "ASSUMED lossy evaluator-side proxy: 4x4 per-class SegNet-label areas + directed label adjacency + cached official PoseNet six-vector; not public to inflate receiver",
            "scope": "upper bound on H(q_G(W)|U_proxy(W)); not an equality to the unmaterialized full canonical orbit label/statistic",
        },
        "n_pairs": int(args.n),
        "code_shape": list(code.shape),
        "quant_scale": float(scale),
        "feature_count": n_features,
        "selected_alpha_by_outer_fold": selected,
        "unconditional_code_brotli_bytes": int(raw_bytes),
        "conditional_residual_brotli_bytes": int(conditional_bytes),
        "selected_conditional_codelength_bits": int(8 * chosen_bytes),
        "selected_branch": "conditional_residual" if conditional_bytes < raw_bytes else "raw_fallback",
        "conditional_saving_bits_before_model": int(8 * (raw_bytes - conditional_bytes)),
        "model_parameter_charge_bytes_ASSUMED_int16_schema": int(model_bytes),
        "uncharged_side_information_blocker": "U_proxy is sourced from evaluator/GT caches and is not public receiver state; its derivation/signalling cost is not included",
        "net_saving_bits_after_model_charge": int(8 * (raw_bytes - conditional_bytes - model_bytes)),
        "percent_of_total_archive_rate_term": float(100.0 * chosen_bytes / archive_bytes),
        "percent_of_code_section": float(100.0 * chosen_bytes / raw_bytes),
        "rate_term_total": float(25.0 * archive_bytes / RATE_DENOM),
        "rate_term_d36_component": float(25.0 * chosen_bytes / RATE_DENOM),
        "error_accounting": {
            "outer_fold_conditional_bits": fold_cond_bits,
            "outer_fold_sum_bits": int(sum(fold_cond_bits)),
            "fold_codelength_sensitivity": fold_codelength_sensitivity(
                fold_cond_bits, 8 * conditional_bytes
            ),
            "bias": "model codelength is an upper bound; U_proxy is lossy; raw fallback prevents a worse-than-unconditional headline code",
        },
        "decoder_verified_bit_exact": True,
        "consumer_change": "The measured gap is a section-engineering budget ceiling, not a ready codec lever: a successor must make U receiver-public/derivable and beat both its signalling cost and predictor bytes.",
        "elapsed_seconds": float(time.time() - t0),
    }
    return result


# ---------------------------------------------------------------------------
# D37: nested conditional-codelength contrasts on receiver boundary pixels
# ---------------------------------------------------------------------------
def extract_boundary_samples(
    lstars: np.memmap, margins: np.memmap, argmax: np.memmap, poses: np.ndarray, n: int
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        k: [] for k in ("f", "m", "phase", "class_pair", "pair", "mnorm", "velocity")
    }
    eps = np.float32(1e-6)
    for i in range(n):
        lab = np.asarray(lstars[i], np.int8)
        mar = np.asarray(margins[i], np.float32)
        pred = np.asarray(argmax[i], np.int8)
        h, w = lab.shape
        # For each boundary pixel, choose the unlike 4-neighbour with smallest margin.
        best_m = np.full((h, w), np.inf, np.float32)
        best_c = np.full((h, w), -1, np.int8)
        best_phase_m = np.zeros((h, w), np.float32)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            shifted_l = np.roll(lab, (dy, dx), axis=(0, 1))
            shifted_m = np.roll(mar, (dy, dx), axis=(0, 1))
            valid = shifted_l != lab
            if dy == 1:
                valid[0] = False
            elif dy == -1:
                valid[-1] = False
            if dx == 1:
                valid[:, 0] = False
            elif dx == -1:
                valid[:, -1] = False
            take = valid & (shifted_m < best_m)
            best_m[take] = shifted_m[take]
            best_phase_m[take] = shifted_m[take]
            best_c[take] = shifted_l[take]
        mask = best_c >= 0
        yy, xx = np.nonzero(mask)
        m = mar[mask]
        mn = best_phase_m[mask]
        gy, gx = np.gradient(mar.astype(np.float64))
        grad = np.sqrt(gx[mask] ** 2 + gy[mask] ** 2).astype(np.float32)
        nx = gx[mask] / (grad + eps)
        ny = gy[mask] / (grad + eps)
        # ASSUMED image-plane velocity proxy from the first two cached pose coordinates.
        vel = np.abs(float(poses[i, 0]) * nx + float(poses[i, 1]) * ny).astype(np.float32)
        fields["f"].append((pred[mask] != lab[mask]).astype(np.uint8))
        fields["m"].append(m.astype(np.float32))
        fields["phase"].append((m / (m + mn + eps)).astype(np.float32))
        fields["class_pair"].append(
            (lab[mask].astype(np.int16) * N_CLASSES + best_c[mask].astype(np.int16)).astype(np.uint8)
        )
        fields["pair"].append(np.full(yy.size, i, np.uint16))
        fields["mnorm"].append((m / (grad + eps)).astype(np.float32))
        fields["velocity"].append(vel)
    return {k: np.concatenate(v) for k, v in fields.items()}


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1:
        return np.empty(0, np.float64)
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(edges.astype(np.float64))


def pose_contexts(poses: np.ndarray, train_pairs: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:
    if n_bins <= 1:
        return np.zeros(poses.shape[0], np.int32), 1
    xp = poses[train_pairs].astype(np.float64)
    mu = xp.mean(axis=0)
    sd = xp.std(axis=0)
    sd[sd < 1e-12] = 1.0
    zt = (xp - mu) / sd
    _u, _s, vt = np.linalg.svd(zt, full_matrices=False)
    z = (poses - mu) / sd
    with np.errstate(all="ignore"):
        proj = z @ vt[:2].T
    if not np.isfinite(proj).all():
        raise FloatingPointError("non-finite pose PCA projection")
    train_proj = proj[train_pairs]
    e0 = quantile_edges(train_proj[:, 0], n_bins)
    e1 = quantile_edges(train_proj[:, 1], n_bins)
    b0 = np.digitize(proj[:, 0], e0).astype(np.int32)
    b1 = np.digitize(proj[:, 1], e1).astype(np.int32)
    n0, n1 = len(e0) + 1, len(e1) + 1
    return b0 + n0 * b1, int(n0 * n1)


def fit_predict_table(
    samples: dict[str, np.ndarray],
    poses: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    margin_bins: int,
    xi_bins: int,
    phase_bins: int,
    class_aware: bool,
    diagnostic: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    ctx, n_ctx = build_base_context(
        samples, poses, train, margin_bins=margin_bins, xi_bins=xi_bins,
        phase_bins=phase_bins, diagnostic=diagnostic
    )
    if class_aware:
        ctx = ctx + n_ctx * samples["class_pair"].astype(np.int64)
        n_ctx *= N_CLASSES * N_CLASSES
    return fit_context_table(samples["f"], ctx, n_ctx, train, test)


def build_base_context(
    samples: dict[str, np.ndarray],
    poses: np.ndarray,
    train: np.ndarray,
    *,
    margin_bins: int,
    xi_bins: int,
    phase_bins: int,
    diagnostic: bool,
) -> tuple[np.ndarray, int]:
    pair = samples["pair"].astype(np.int32)
    # Pair ids are dense 0..599.  bincount avoids repeatedly sorting millions of
    # sample ids during nested CV.
    train_pairs = np.flatnonzero(np.bincount(pair[train], minlength=poses.shape[0]))
    xi_pair, n_xi = pose_contexts(poses, train_pairs, xi_bins)
    base_name = "mnorm" if diagnostic else "m"
    medges = quantile_edges(samples[base_name][train], margin_bins)
    mbin = np.digitize(samples[base_name], medges).astype(np.int64)
    n_m = len(medges) + 1
    ctx = mbin + n_m * xi_pair[pair].astype(np.int64)
    n_ctx = n_m * n_xi
    if phase_bins > 1:
        pbin = np.minimum((samples["phase"] * phase_bins).astype(np.int64), phase_bins - 1)
        ctx += n_ctx * pbin
        n_ctx *= phase_bins
    if diagnostic:
        vedges = quantile_edges(samples["velocity"][train], 3)
        vbin = np.digitize(samples["velocity"], vedges).astype(np.int64)
        ctx += n_ctx * vbin
        n_ctx *= len(vedges) + 1
    return ctx, int(n_ctx)


def fit_context_table(
    labels: np.ndarray,
    ctx: np.ndarray,
    n_ctx: int,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    y = labels.astype(np.float64)
    tr_ctx = ctx[train]
    totals = np.bincount(tr_ctx, minlength=n_ctx).astype(np.float64)
    ones = np.bincount(tr_ctx, weights=y[train], minlength=n_ctx).astype(np.float64)
    p_global = float((y[train].sum() + 0.5) / (train.sum() + 1.0))
    # Jeffreys-equivalent one pseudo-observation anchored at the training base rate.
    prob = (ones + p_global) / (totals + 1.0)
    pred = np.clip(prob[ctx[test]], 1e-7, 1 - 1e-7)
    return pred, {
        "n_context_capacity": int(n_ctx),
        "n_nonempty_contexts": int(np.count_nonzero(totals)),
    }


def fit_predict_table_pair(
    samples: dict[str, np.ndarray],
    poses: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    margin_bins: int,
    xi_bins: int,
    phase_bins: int,
    diagnostic: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    base_ctx, n_base = build_base_context(
        samples, poses, train, margin_bins=margin_bins, xi_bins=xi_bins,
        phase_bins=phase_bins, diagnostic=diagnostic
    )
    p0, t0 = fit_context_table(samples["f"], base_ctx, n_base, train, test)
    aware_ctx = base_ctx + n_base * samples["class_pair"].astype(np.int64)
    p1, t1 = fit_context_table(
        samples["f"], aware_ctx, n_base * N_CLASSES * N_CLASSES, train, test
    )
    return p0, p1, t0, t1


def nll_bits(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -(y * np.log2(p) + (1.0 - y) * np.log2(1.0 - p))


CONTEXT_GRID = (
    {"margin_bins": 8, "xi_bins": 2, "phase_bins": 4},
    {"margin_bins": 16, "xi_bins": 2, "phase_bins": 8},
    {"margin_bins": 24, "xi_bins": 3, "phase_bins": 8},
)


def select_context_spec(
    samples: dict[str, np.ndarray], poses: np.ndarray, outer_train: np.ndarray, phase: bool
) -> dict[str, int]:
    pair = samples["pair"].astype(np.int32)
    scores = np.zeros(len(CONTEXT_GRID), np.float64)
    for inner in range(3):
        va = outer_train & (pair % 3 == inner)
        tr = outer_train & ~va
        yv = samples["f"][va].astype(np.float64)
        for j, spec in enumerate(CONTEXT_GRID):
            pb = spec["phase_bins"] if phase else 1
            p0, p1, _t0, _t1 = fit_predict_table_pair(
                samples, poses, tr, va, margin_bins=spec["margin_bins"],
                xi_bins=spec["xi_bins"], phase_bins=pb
            )
            scores[j] += float(nll_bits(yv, p0).sum() + nll_bits(yv, p1).sum())
    return dict(CONTEXT_GRID[int(scores.argmin())])


def calibration_rows(y: np.ndarray, p0: np.ndarray, p1: np.ndarray, c: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for cid in sorted(np.unique(c).tolist()):
        m = c == cid
        if not np.any(m):
            continue
        rows.append({
            "class_pair_id": int(cid),
            "class_pair": [int(cid // N_CLASSES), int(cid % N_CLASSES)],
            "n": int(m.sum()),
            "flip_rate": float(y[m].mean()),
            "class_blind_mean_prediction": float(p0[m].mean()),
            "class_aware_mean_prediction": float(p1[m].mean()),
            "class_blind_calibration_residual": float(y[m].mean() - p0[m].mean()),
            "class_aware_calibration_residual": float(y[m].mean() - p1[m].mean()),
            "class_blind_brier": float(np.mean((y[m] - p0[m]) ** 2)),
            "class_aware_brier": float(np.mean((y[m] - p1[m]) ** 2)),
        })
    return rows


def crossfit_contrast(
    samples: dict[str, np.ndarray], poses: np.ndarray, *, phase: bool, diagnostic: bool = False
) -> dict[str, Any]:
    pair = samples["pair"].astype(np.int32)
    y = samples["f"].astype(np.float64)
    p0_all = np.empty(y.size, np.float64)
    p1_all = np.empty(y.size, np.float64)
    specs: list[dict[str, int]] = []
    table_rows: list[dict[str, int]] = []
    for outer in range(5):
        va = pair % 5 == outer
        tr = ~va
        spec = select_context_spec(samples, poses, tr, phase) if not diagnostic else {
            "margin_bins": 16, "xi_bins": 2, "phase_bins": 8
        }
        specs.append(spec)
        pb = spec["phase_bins"] if phase else 1
        p0, p1, t0, t1 = fit_predict_table_pair(
            samples, poses, tr, va, margin_bins=spec["margin_bins"],
            xi_bins=spec["xi_bins"], phase_bins=pb, diagnostic=diagnostic
        )
        p0_all[va], p1_all[va] = p0, p1
        table_rows.append({"outer_fold": outer, "blind_nonempty": t0["n_nonempty_contexts"],
                           "aware_nonempty": t1["n_nonempty_contexts"]})
    delta_sample = nll_bits(y, p0_all) - nll_bits(y, p1_all)
    pair_delta = np.bincount(pair, weights=delta_sample, minlength=poses.shape[0])
    ci = bootstrap_interval(pair_delta)
    selected = Counter(tuple(sorted(s.items())) for s in specs).most_common(1)[0][0]
    modal = dict(selected)
    all_mask = np.ones(y.size, dtype=bool)
    pb = modal["phase_bins"] if phase else 1
    # Full-data table sizes approximate the one receiver model that would actually be signalled.
    _p0, _p1, t0, t1 = fit_predict_table_pair(
        samples, poses, all_mask, all_mask, margin_bins=modal["margin_bins"],
        xi_bins=modal["xi_bins"], phase_bins=pb, diagnostic=diagnostic
    )
    # ASSUMED fixed schema: uint16 context id + two uint32 counts, plus 32-byte model header.
    overhead_bytes = 32 + 10 * max(0, t1["n_nonempty_contexts"] - t0["n_nonempty_contexts"])
    net = float(ci["estimate"] - 8 * overhead_bytes)
    return {
        "gross_gain_bits": float(ci["estimate"]),
        "gross_gain_bits_per_boundary_pixel": float(ci["estimate"] / y.size),
        "gross_gain_bits_per_flip": float(ci["estimate"] / max(1.0, y.sum())),
        "pair_bootstrap": ci,
        "selected_spec_by_outer_fold": specs,
        "full_data_modal_spec": modal,
        "table_rows_by_fold": table_rows,
        "model_table_overhead": {
            "schema": "ASSUMED fixed rows: uint16 context key + uint32 n0 + uint32 n1; 32-byte header",
            "blind_nonempty_contexts": int(t0["n_nonempty_contexts"]),
            "aware_nonempty_contexts": int(t1["n_nonempty_contexts"]),
            "incremental_bytes": int(overhead_bytes),
        },
        "net_gain_bits_after_table_overhead": net,
        "net_gain_ci95_low_bits": float(ci["ci95_low"] - 8 * overhead_bytes),
        "net_gain_ci95_high_bits": float(ci["ci95_high"] - 8 * overhead_bytes),
        "class_pair_calibration": calibration_rows(y, p0_all, p1_all, samples["class_pair"]),
    }


def measure_d37(
    args: argparse.Namespace,
    lstars: np.memmap,
    margins: np.memmap,
    argmax: np.memmap,
    poses: np.ndarray,
) -> dict[str, Any]:
    t0 = time.time()
    samples = extract_boundary_samples(lstars, margins, argmax, poses, args.n)
    print(f"[D37] extracted {samples['f'].size:,} local boundary pixels; "
          f"{int(samples['f'].sum()):,} receiver flips", flush=True)
    base = crossfit_contrast(samples, poses, phase=False)
    print("[D37] completed nested q00/q01 contrast", flush=True)
    phase = crossfit_contrast(samples, poses, phase=True)
    print("[D37] completed nested q10/q11 contrast", flush=True)
    diagnostic = crossfit_contrast(samples, poses, phase=True, diagnostic=True)
    verdict = (
        "CLASS-CONDITIONING-ADMITTED"
        if base["net_gain_ci95_low_bits"] > 0
        else "CLASS-BLIND-SUPPORTED"
        if base["net_gain_ci95_high_bits"] <= 0
        else "INCONCLUSIVE"
    )
    result = {
        "debt_id": "D37",
        "estimator": "nested 5x3 pair-blocked cross-fitted beta-binomial context-table codelength; q00/q01 and phase-aware q10/q11",
        "status": "MEASURED",
        "authority": "[macOS-CPU advisory; n600 real realized-through-R cached witness surface; non-promotable]",
        "surface": {
            "argmax_cache": str(args.argmax_cache),
            "scope": "epoch-50 witness_perclass_baseline_n600 cached receiver surface; not inferred to current mod32cap or contest-CUDA",
        },
        "operational_completion": {
            "F": "MEASURED receiver-visible cached argmax disagreement against frozen GT lstar",
            "M": "MEASURED frozen GT top-1/top-2 margin",
            "C": "ASSUMED local directed four-neighbour GT class edge; runner-up logits were not stored; C must be already decoded or its signalling cost charged in an exact coder",
            "Qxi": "ASSUMED quantization of cached official PoseNet six-vectors by train-fold PCA quantiles",
            "Phi": "ASSUMED local tie coordinate M_self/(M_self+M_adjacent)",
            "jacobian_velocity": "ASSUMED diagnostic proxy M/|grad M| and |pose_xy dot normalized grad M|; not an exact scorer Jacobian",
        },
        "n_pairs": int(args.n),
        "n_boundary_pixels": int(samples["f"].size),
        "n_boundary_flips": int(samples["f"].sum()),
        "boundary_flip_rate": float(samples["f"].mean()),
        "delta_L_MX": base,
        "delta_L_MX_phase_aware": phase,
        "normalized_jacobian_velocity_diagnostic": diagnostic,
        "verdict": verdict,
        "admission_rule": "class-blind only if upper 95% CI on net saved bits <=0 and per-pair calibration residuals are controlled",
        "uncharged_context_blocker": "the flat-table overhead charges model counts but not a class-edge context sequence; exact packed A/B must prove C is already decoded/derivable or charge it jointly",
        "consumer_change": "Positive net class gain admits a class-edge-aware grammar candidate only where C is already decoded/derivable; otherwise joint class signalling and exact packed receiver A/B are mandatory.",
        "verdict_scope": "FORMULATION x EMPIRICAL-SURFACE; no class, margin, or flip-coding family kill",
        "elapsed_seconds": float(time.time() - t0),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-cache", type=Path,
                    default=Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz"))
    ap.add_argument("--argmax-cache", type=Path,
                    default=Path("experiments/results/witness_perclass_baseline_n600/maps_Best.npz"))
    ap.add_argument("--checkpoint", type=Path, default=Path(
        "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz"))
    ap.add_argument("--byteclose-report", type=Path,
                    default=Path("reports/t5_s6_byteclose_mod32cap_ep650_weightsonly_20260707.json"))
    ap.add_argument("--archive-bytes", type=int, default=83_430)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("experiments/results/ladder_owed_measurables_20260713"))
    args = ap.parse_args()
    if args.n != 600:
        raise ValueError("This registered measurement is n600; use a separate probe-labelled tool for subsets")

    started = datetime.now(UTC).isoformat()
    lstars = stored_npy_memmap(args.gt_cache, "lstars")
    margins = stored_npy_memmap(args.gt_cache, "margins")
    argmax = stored_npy_memmap(args.argmax_cache, "argmax")
    poses_mm = stored_npy_memmap(args.gt_cache, "gt_poses")
    poses = np.asarray(poses_mm[: args.n], np.float64).copy()
    if lstars.shape[:1] != (args.n,) or margins.shape != lstars.shape or argmax.shape != lstars.shape:
        raise ValueError(f"custody shape mismatch: lstars={lstars.shape}, margins={margins.shape}, argmax={argmax.shape}")

    d36 = measure_d36(args, lstars, poses)
    atomic_json(args.out_dir / "d36_fiber_completeness_gap_n600.json", d36)
    d37 = measure_d37(args, lstars, margins, argmax, poses)
    atomic_json(args.out_dir / "d37_flip_conditional_mi_n600.json", d37)

    custody = {}
    for name, path in (("gt_cache", args.gt_cache), ("argmax_cache", args.argmax_cache),
                       ("checkpoint", args.checkpoint), ("byteclose_report", args.byteclose_report)):
        custody[name] = {"path": str(path), "bytes": int(path.stat().st_size),
                         "sha256": sha256_file(path)}
    manifest = {
        "schema": "pact.rate_law_ladder_owed_measurables.receipt.v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "measurement_tool": {
            "path": str(Path(__file__).relative_to(Path.cwd())),
            "sha256": sha256_file(Path(__file__)),
        },
        "n_pairs": args.n,
        "execution": "$0 local read-only inputs; atomic small JSON outputs; no render/train/cache mutation",
        "custody": custody,
        "outputs": {
            "d36": "d36_fiber_completeness_gap_n600.json",
            "d37": "d37_flip_conditional_mi_n600.json",
        },
    }
    atomic_json(args.out_dir / "receipt_manifest.json", manifest)
    print(json.dumps({
        "d36_bits": d36["selected_conditional_codelength_bits"],
        "d36_percent_rate": d36["percent_of_total_archive_rate_term"],
        "d37_verdict": d37["verdict"],
        "d37_net_bits": d37["delta_L_MX"]["net_gain_bits_after_table_overhead"],
        "out_dir": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
