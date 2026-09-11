#!/usr/bin/env python3
"""MXO3 oracle upper bound on the move-44 pre-mix collapse.

The MXO2 screen asks whether an online integer stacker over the shipping
corrector's 23 pre-mix family outputs recovers bytes the fixed mixer discards.
A zero-byte online result alone cannot separate "there is no information" from
"this particular learner is weak", so this instrument measures the ceiling
instead.

The corrector controls exactly one number per symbol: q, the probability that
the token equals the argmax class it nominates.  Everything else in the coded
row is rescaled around it.  So the whole pre-mix question is a binary one -- was
the argmax right? -- and the code length attributable to it is the binary cross
entropy of the shipped q against the realised hit.

The oracle partitions the symbols on (hit class, shipped q, statistics of the 23
pre-mix outputs) and scores each cell at its own EMPIRICAL hit frequency.  That
frequency is the maximum-likelihood optimum inside the cell, fitted on the very
data it scores, so the result is an upper bound: no causal online learner
reading the same features can beat an offline fit that already saw the answers.
A label-shuffled control measures how much of the bound is pure overfit.

Advisory, scorer-free, $0: no Modal, no candidate archive, no score claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"

import numpy as np

from experiments.ddm_ls1_shipped_surprise import fact
from experiments.ddm_mxo2_low_rank_stacker import (
    SIZE,
    TOTAL,
    K,
    Mxo2Error,
    N,
    adopt_root,
    checked_npz,
    save_json,
)

AXIS = "[macOS-CPU advisory / scorer-free n600 exact integer-row oracle bound]"
SEED = 20260911
FIRE_SAVING_B = 3_000
TAIL_DEMAND_B = 25_899
Q15 = 32768.0
# The shipped corrector's own Krichevsky-Trofimov prior (f26_corrector_native.c).
KT_ALPHA = 0.5

# log2(1 - q) is the natural axis: it is the miss surprise the corrector prices.
MISS_LOW, MISS_HIGH = -31.0, 0.0
# Fixed decade thresholds in natural-log-odds units; no data-dependent fitting.
STD_EDGES = np.array([1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
MEAN_EDGES = np.array([-3e-1, -1e-1, -1e-2, 0.0, 1e-2, 1e-1, 3e-1])
PEAK_EDGES = np.array([1e-2, 1e-1, 1.0])
AGREE_EDGES = np.array([1, 6, 12, 18, 23])
# Two partitions of the same pre-mix outputs.  "summary" spends few cells so the
# overfit control stays small; "rich" quadruples the baseline resolution and adds
# a crowd-direction axis, so a bound that survives both is not an artefact of a
# thin feature set.
CONTEXT_SETS = {
    "summary": {"miss_bins": 64, "premix_axes": ("std", "mean", "peak")},
    "rich": {"miss_bins": 256, "premix_axes": ("std", "mean", "peak", "agree")},
}
AXIS_LEVELS = {"std": 8, "mean": 8, "peak": 4, "agree": 6}


def binary_bits(probability: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """Code length in bits of the realised hit under the shipped probability."""
    safe = np.clip(probability, 1.0 / TOTAL, 1.0 - 1.0 / TOTAL)
    return -np.where(hit, np.log2(safe), np.log2(1.0 - safe))


def cell_bits(counts: np.ndarray, hits: np.ndarray) -> float:
    """Total bits when every cell is scored at its own empirical frequency."""
    counts = counts.astype(np.float64)
    hits = hits.astype(np.float64)
    misses = counts - hits
    with np.errstate(divide="ignore", invalid="ignore"):
        term_hit = np.where(hits > 0, hits * np.log2(hits / counts), 0.0)
        term_miss = np.where(misses > 0, misses * np.log2(misses / counts), 0.0)
    return float(-(term_hit + term_miss).sum())


def premix_levels(axes: tuple[str, ...]) -> int:
    product = 1
    for name in axes:
        product *= AXIS_LEVELS[name]
    return product


def premix_context(
    family_q15: np.ndarray, mixer_q15: np.ndarray, axes: tuple[str, ...]
) -> np.ndarray:
    """Quantised statistics of the 23 pre-mix opinions, relative to the mix.

    Each family's coded probability shares the corrector's base odds, so the
    difference of log odds isolates that family's own multiplier -- exactly the
    quantity the fixed mixer weights and then discards.
    """
    family = np.clip(family_q15.astype(np.float64), 1.0, Q15 - 1.0) / Q15
    mixed = np.clip(mixer_q15.astype(np.float64), 1.0, Q15 - 1.0) / Q15
    opinion = np.log(family / (1.0 - family)) - np.log(mixed / (1.0 - mixed))[:, None]
    parts = {
        "std": np.digitize(opinion.std(axis=1), STD_EDGES),
        "mean": np.digitize(opinion.mean(axis=1), MEAN_EDGES),
        "peak": np.digitize(np.abs(opinion).max(axis=1), PEAK_EDGES),
        "agree": np.digitize((opinion > 0.0).sum(axis=1), AGREE_EDGES),
    }
    context = np.zeros(len(mixed), dtype=np.int64)
    for name in axes:
        context = context * AXIS_LEVELS[name] + parts[name]
    return context


def miss_bin(probability: np.ndarray, bins: int) -> np.ndarray:
    miss = np.clip(1.0 - probability, 1.0 / TOTAL, 1.0)
    scaled = (np.log2(miss) - MISS_LOW) / (MISS_HIGH - MISS_LOW) * bins
    return np.clip(scaled.astype(np.int64), 0, bins - 1)


def accumulate(
    root: Path,
    frames: int,
    rng: np.random.Generator,
    context_set: str,
    blocks: int = 1,
) -> dict[str, object]:
    """Count hits per cell over the surface.

    `blocks` splits the run into contiguous stretches that get their own cells.
    A global fit would be beatable by an online learner that tracks drift, so
    letting the oracle refit per block keeps it an upper bound under
    non-stationarity too.
    """
    plan = CONTEXT_SETS[context_set]
    miss_bins = int(plan["miss_bins"])
    axes = tuple(plan["premix_axes"])
    levels = premix_levels(axes) * blocks
    cells = K * miss_bins * levels
    shape = (K, miss_bins, levels)
    counts = np.zeros((2, *shape), dtype=np.int64)
    hits = np.zeros((2, *shape), dtype=np.int64)
    shuffled_hits = np.zeros(shape, dtype=np.int64)
    calib_counts = np.zeros((2, K, miss_bins), dtype=np.int64)
    calib_hits = np.zeros((2, K, miss_bins), dtype=np.int64)
    shipped_bits = 0.0
    total_bits = 0.0
    seen = 0
    for frame in range(frames):
        path = root / "surface/frames" / f"frame_{frame:04d}.npz"
        with checked_npz(path) as data:
            family = data["family_q15"]
            mixer = data["mixer_q15"]
            frequencies = data["frequencies"]
            hit_class = data["hit_class"].astype(np.int64)
            symbols = data["symbols"].astype(np.int64)
        rows = np.arange(len(symbols))
        probability = frequencies[rows, hit_class].astype(np.float64) / TOTAL
        hit = symbols == hit_class
        shipped_bits += float(binary_bits(probability, hit).sum())
        total_bits += float(
            (-np.log2(frequencies[rows, symbols].astype(np.float64) / TOTAL)).sum()
        )
        bins = miss_bin(probability, miss_bins)
        block = min(frame * blocks // frames, blocks - 1)
        context = premix_context(family, mixer, axes) * blocks + block
        flat = (hit_class * miss_bins + bins) * levels + context
        fold = frame % 2
        counts[fold] += np.bincount(flat, minlength=cells).reshape(shape)
        hits[fold] += np.bincount(flat, weights=hit, minlength=cells).reshape(shape).astype(np.int64)
        control = rng.permutation(hit)
        shuffled_hits += (
            np.bincount(flat, weights=control, minlength=cells).reshape(shape).astype(np.int64)
        )
        calib_flat = hit_class * miss_bins + bins
        calib_counts[fold] += np.bincount(calib_flat, minlength=K * miss_bins).reshape(K, miss_bins)
        calib_hits[fold] += (
            np.bincount(calib_flat, weights=hit, minlength=K * miss_bins)
            .reshape(K, miss_bins)
            .astype(np.int64)
        )
        seen += len(symbols)
        if (frame + 1) % 50 == 0:
            print(json.dumps({"stage": "oracle", "frames": frame + 1}), flush=True)
    return {
        "counts": counts,
        "hits": hits,
        "shuffled_hits": shuffled_hits,
        "calib_counts": calib_counts,
        "calib_hits": calib_hits,
        "shipped_bits": shipped_bits,
        "total_bits": total_bits,
        "symbols": seen,
        "context_set": context_set,
        "miss_bins": miss_bins,
        "premix_axes": list(axes),
        "blocks": blocks,
        "cells": cells,
    }


def kt_estimate(counts: np.ndarray, hits: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Krichevsky-Trofimov posterior that backs off to a coarser estimate.

    A cell the fit fold never saw returns the prior rather than a coin flip, so
    a finer partition is not punished merely for being finer.  Without the back
    off the held-out comparison would charge the pre-mix table a full bit for
    every unseen context and could hide a real signal.
    """
    return (hits + 2.0 * KT_ALPHA * prior) / (counts + 2.0 * KT_ALPHA)


def cross_bits(counts: np.ndarray, hits: np.ndarray, prior: np.ndarray) -> float:
    """Bits to code each fold under the OTHER fold's table.

    Nothing is scored by a frequency fitted on itself, so this number carries no
    overfit at all and needs no control.  KT_ALPHA is the same Krichevsky-Trofimov
    prior the shipped corrector uses for its own family estimates.
    """
    total = 0.0
    for score, fit in ((0, 1), (1, 0)):
        probability = np.clip(
            kt_estimate(counts[fit], hits[fit], prior[fit]), 1.0 / TOTAL, 1.0 - 1.0 / TOTAL
        )
        misses = counts[score] - hits[score]
        total += float(
            -(hits[score] * np.log2(probability) + misses * np.log2(1.0 - probability)).sum()
        )
    return total


def report(state: dict[str, object], frames: int) -> dict[str, object]:
    shipped_bits = float(state["shipped_bits"])
    counts = state["counts"]
    hits = state["hits"]
    calib_counts = state["calib_counts"]
    calib_hits = state["calib_hits"]
    calibrated = cell_bits(calib_counts.sum(axis=0), calib_hits.sum(axis=0))
    full = cell_bits(counts.sum(axis=0), hits.sum(axis=0))
    control = cell_bits(counts.sum(axis=0), state["shuffled_hits"])
    control_calibrated = cell_bits(
        calib_counts.sum(axis=0), state["shuffled_hits"].sum(axis=2)
    )
    # The calibrated table backs off to each fold's own per-class hit rate; the
    # pre-mix table backs off to that same fold's calibrated estimate, so the two
    # are compared on equal terms and only real pre-mix signal can separate them.
    class_rate = (
        calib_hits.sum(axis=2, keepdims=True) / np.maximum(calib_counts.sum(axis=2, keepdims=True), 1)
    )
    calib_prior = kt_estimate(calib_counts, calib_hits, class_rate)
    held_calibrated = cross_bits(calib_counts, calib_hits, class_rate)
    held_full = cross_bits(counts, hits, calib_prior[..., None])
    occupied = int((counts.sum(axis=0) > 0).sum())
    # The control keeps the partition and destroys the signal, so whatever it
    # "saves" over the same recalibrated baseline is the partition's overfit.
    overfit_bits = max(control_calibrated - control, 0.0)
    premix_bits = calibrated - full
    return {
        "axis": AXIS,
        "score_claim": False,
        "frames": frames,
        "symbols": int(state["symbols"]),
        "token_stream_bits": float(state["total_bits"]),
        "token_stream_bytes": float(state["total_bits"]) / 8,
        "shipped_binary_bits": shipped_bits,
        "shipped_binary_bytes": shipped_bits / 8,
        "recalibration_only_bytes": (shipped_bits - calibrated) / 8,
        "premix_gross_bytes": premix_bits / 8,
        "premix_overfit_bytes": overfit_bits / 8,
        "premix_net_bytes": (premix_bits - overfit_bits) / 8,
        "holdout_recalibration_bytes": (shipped_bits - held_calibrated) / 8,
        "holdout_premix_bytes": (held_calibrated - held_full) / 8,
        "holdout_premix_vs_shipped_bytes": (shipped_bits - held_full) / 8,
        "context_set": state["context_set"],
        "miss_bins": state["miss_bins"],
        "premix_axes": state["premix_axes"],
        "blocks": state["blocks"],
        "cells": int(state["cells"]),
        "cells_occupied": occupied,
        "symbols_per_occupied_cell": float(state["symbols"]) / max(occupied, 1),
        "fire_saving_bytes": FIRE_SAVING_B,
        "tail_demand_bytes": TAIL_DEMAND_B,
        "primary_number": "holdout_premix_bytes",
        "interpretation": (
            "premix_net_bytes bounds an online learner in sample, with the shuffle "
            "control removing the partition's overfit; holdout_premix_bytes is the "
            "same quantity measured with every symbol scored by a table fitted on "
            "frames it is not part of, so it carries no overfit at all"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=N)
    parser.add_argument("--context-set", choices=tuple(CONTEXT_SETS), default="summary")
    parser.add_argument("--blocks", type=int, default=1)
    args = parser.parse_args()
    root = adopt_root(args.resume_from)
    if not 1 <= args.frames <= N:
        raise Mxo2Error("--frames must name a prefix of the n600 surface")
    available = len(list((root / "surface/frames").glob("frame_*.npz")))
    if available < args.frames:
        raise Mxo2Error(f"surface has {available} frames; {args.frames} requested")
    if not 1 <= args.blocks <= args.frames:
        raise Mxo2Error("--blocks must split the requested frames")
    state = accumulate(
        root, args.frames, np.random.default_rng(SEED), args.context_set, args.blocks
    )
    if state["symbols"] != args.frames * SIZE:
        raise Mxo2Error("surface plane geometry changed")
    result = report(state, args.frames)
    result["full_n600"] = args.frames == N
    stem = f"ORACLE_{args.context_set}_b{args.blocks:02d}"
    name = f"{stem}.json" if args.frames == N else f"{stem}_prefix_{args.frames:04d}.json"
    save_json(root / name, result)
    result["receipt"] = fact(root / name)
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
