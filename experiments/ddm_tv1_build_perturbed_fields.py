#!/usr/bin/env python3
"""ddm_tv1 -- build seeded counterfactual token fields and publish them to the receiver.

WHAT THIS MEASURES, AND WHAT IT IS NOT
--------------------------------------
This builds a token field that differs from dx2's shipped field at exactly ``k``
positions, so the frozen scorer can be asked how much of that movement it sees.
It measures **TOLERANCE ALONE**.

It is **NOT a lever and cannot become one.**  Nothing here names WHICH positions
moved, and nothing here codes anything.  An actual representation that wanted to
exploit tolerance would have to transmit the changed set, and the campaign has
measured three times over that naming a subset costs more than the subset holds
(mf1's best perfectly-addressed repair still cost +35,969 B of address payload).
Isolating the numerator from that address tax is the entire point: the two have
never been measured apart, so a small tolerance and a large address tax have been
indistinguishable.

THE ARMS -- the reassignment rule IS the mechanism, not a detail
---------------------------------------------------------------
The dx2 field's bit mass has Gini 0.9951 and the shipped coder is near-certain at
the median position (median cost 1.008e-8 bits).  So "reassign a random position"
is a family, and picking the wrong member measures the wrong thing: a uniform draw
over the alphabet is a MAXIMAL perturbation and reports receiver robustness to
garbage rather than the size of the scorer's equivalence cell.

Two independent binary choices -- WHERE a position is drawn from, and WHAT value
it takes -- give a 2x2 whose corner is the uniform control, so the position effect
and the value effect are separable rather than confounded on a diagonal:

    arm          positions drawn by         value drawn from
    ---------------------------------------------------------------------
    cond_cond    model uncertainty 1-p      coder conditional, value != current
    cond_unif    model uncertainty 1-p      uniform over the 4 alternatives
    unif_cond    uniform over the field     coder conditional, value != current
    unif_unif    uniform over the field     uniform over the 4 alternatives
    unif_marg    uniform over the field     global class marginal, value != current

The arm name is ``<position rule>_<value rule>``, so the CLI accepts exactly the
names in that table.  ``cond_cond`` is the reference form: it is the shipped
coder's OWN geometry in
both axes -- movement within the manifold the field was priced against.
``unif_unif`` is the maximal-perturbation control.  Every arm produces exactly
``k`` CHANGED positions, so the ladder is comparable across arms.

The conditional is the shipped corrector's coding row, captured by
``ddm_tv1_capture_coding_conditionals.py`` under a bit-exact digest control -- it
is the distribution the field's bits are actually priced against, which is also
the distribution the retained per-position cost field is derived from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

NUM_CLASSES = 5
FRAMES, EVAL_H, EVAL_W = 600, 384, 512
PLANE = EVAL_H * EVAL_W
POSITIONS = FRAMES * PLANE
# ddm_tx1_toolbox_crosswalk_20260819.md section 0 -- CITED, never re-derived.
S_PER_BYTE = 6.658590e-07

POSITION_RULES = {"cond", "unif"}
VALUE_RULES = {"cond", "unif", "marg"}


class BuildError(RuntimeError):
    """Fail-closed error for field construction."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_seed(arm: str, k: int, base_seed: int) -> int:
    """Deterministic per-(arm, k) seed: same inputs -> same positions, always."""
    payload = f"ddm_tv1|{arm}|{k}|{base_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def uncertainty_weights(cost_bits: np.ndarray) -> np.ndarray:
    """w = 1 - p(realized), computed from the cost field without cancellation.

    ``p = 2**-cost`` and ``1 - p`` would lose every significant digit in float32
    at the median cost of 1.008e-8 bits.  ``-expm1(-cost * ln2)`` is exact there.
    """
    return -np.expm1(-cost_bits * math.log(2.0))


def draw_positions(rule: str, k: int, rng: np.random.Generator, weights: np.ndarray) -> np.ndarray:
    """k distinct positions, uniformly or by the model's own uncertainty mass."""
    if rule == "unif":
        chosen: set[int] = set()
        while len(chosen) < k:
            draw = rng.integers(0, POSITIONS, size=k - len(chosen), dtype=np.int64)
            chosen.update(int(v) for v in draw)
        return np.sort(np.fromiter(chosen, dtype=np.int64, count=k))
    if rule != "cond":
        raise BuildError(f"unknown position rule {rule!r}")
    # Gumbel top-k == weighted sampling WITHOUT replacement (Plackett-Luce).
    positive = int(np.count_nonzero(weights))
    if positive < k:
        raise BuildError(
            f"only {positive:,} positions carry non-zero model uncertainty; "
            f"cannot draw {k:,} in-manifold positions"
        )
    with np.errstate(divide="ignore"):
        keys = np.log(weights).astype(np.float64)
    keys += rng.gumbel(size=POSITIONS)
    keys[weights <= 0.0] = -np.inf
    top = np.argpartition(keys, POSITIONS - k)[POSITIONS - k:]
    return np.sort(top.astype(np.int64))


def draw_values(
    rule: str,
    positions: np.ndarray,
    current: np.ndarray,
    rng: np.random.Generator,
    conditional: np.memmap | None,
    marginal: np.ndarray,
) -> tuple[np.ndarray, int]:
    """A replacement value != current for each position. Returns (values, fallbacks)."""
    k = positions.size
    if rule == "unif":
        offset = rng.integers(1, NUM_CLASSES, size=k, dtype=np.int64)
        return ((current.astype(np.int64) + offset) % NUM_CLASSES).astype(np.uint8), 0
    if rule == "marg":
        table = np.tile(marginal.astype(np.float64), (k, 1))
    elif rule == "cond":
        if conditional is None:
            raise BuildError("value rule 'cond' requires the captured conditional field")
        table = np.asarray(conditional[positions], dtype=np.float64)
    else:
        raise BuildError(f"unknown value rule {rule!r}")
    table[np.arange(k), current.astype(np.int64)] = 0.0
    totals = table.sum(axis=1)
    # A row can underflow to all-zero when float32 flushed every alternative.
    # Fall back to the global marginal and COUNT it -- a silent fallback would
    # relabel part of the conditional arm as a marginal arm.
    dead = totals <= 0.0
    fallbacks = int(dead.sum())
    if fallbacks:
        replacement = np.tile(marginal.astype(np.float64), (fallbacks, 1))
        replacement[np.arange(fallbacks), current[dead].astype(np.int64)] = 0.0
        table[dead] = replacement
        totals = table.sum(axis=1)
    table /= totals[:, None]
    draws = rng.random(k)
    values = (np.cumsum(table, axis=1) < draws[:, None]).sum(axis=1)
    return np.minimum(values, NUM_CLASSES - 1).astype(np.uint8), fallbacks


def boundary_distance(field: np.ndarray, max_distance: int = 4) -> np.ndarray:
    """Per-position 4-neighbour distance to a class boundary, matching the
    receiver's own ``_boundary_buckets`` definition."""
    out = np.empty(field.shape, dtype=np.uint8)
    for frame in range(field.shape[0]):
        plane = field[frame]
        edge = np.zeros(plane.shape, dtype=bool)
        edge[1:] |= plane[1:] != plane[:-1]
        edge[:-1] |= plane[:-1] != plane[1:]
        edge[:, 1:] |= plane[:, 1:] != plane[:, :-1]
        edge[:, :-1] |= plane[:, :-1] != plane[:, 1:]
        result = np.full(plane.shape, max_distance, dtype=np.uint8)
        active = edge.copy()
        result[active] = 0
        for distance in range(1, max_distance):
            grown = active.copy()
            grown[1:] |= active[:-1]
            grown[:-1] |= active[1:]
            grown[:, 1:] |= active[:, :-1]
            grown[:, :-1] |= active[:, 1:]
            active = grown
            result[(result == max_distance) & active] = distance
        out[frame] = result
    return out


def build(
    *,
    arm: str,
    k: int,
    base_seed: int,
    tokens_path: Path,
    cost_path: Path,
    conditional_path: Path | None,
    out_path: Path,
    distance_path: Path | None,
) -> dict[str, Any]:
    position_rule, value_rule = arm.split("_")
    if position_rule not in POSITION_RULES or value_rule not in VALUE_RULES:
        raise BuildError(f"unknown arm {arm!r}")

    tokens = np.fromfile(tokens_path, dtype=np.uint8)
    if tokens.size != POSITIONS:
        raise BuildError(f"token field is {tokens.size} positions, expected {POSITIONS}")
    cost = np.fromfile(cost_path, dtype="<f8")
    if cost.size != POSITIONS:
        raise BuildError("cost field does not match the token field geometry")
    conditional = (
        np.load(conditional_path, mmap_mode="r") if conditional_path is not None else None
    )
    marginal = np.bincount(tokens, minlength=NUM_CLASSES).astype(np.float64)
    marginal /= marginal.sum()

    seed = arm_seed(arm, k, base_seed)
    rng = np.random.default_rng(seed)
    weights = uncertainty_weights(cost) if position_rule == "cond" else np.empty(0)
    positions = draw_positions(position_rule, k, rng, weights)
    current = tokens[positions]
    values, fallbacks = draw_values(
        value_rule, positions, current, rng, conditional, marginal
    )
    if np.any(values == current):
        raise BuildError("a replacement value equalled the incumbent; k would be wrong")

    perturbed = tokens.copy()
    perturbed[positions] = values
    changed = int(np.count_nonzero(perturbed != tokens))
    if changed != k:
        raise BuildError(f"changed {changed} positions, expected {k}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    perturbed.tofile(out_path)

    transitions = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(transitions, (current.astype(np.int64), values.astype(np.int64)), 1)
    per_frame = np.bincount(positions // PLANE, minlength=FRAMES)
    bits = float(cost[positions].sum())

    distance_hist: list[int] | None = None
    if distance_path is not None and distance_path.is_file():
        distances = np.memmap(distance_path, mode="r", dtype=np.uint8, shape=(POSITIONS,))
        distance_hist = np.bincount(
            np.asarray(distances[positions]), minlength=5
        ).astype(np.int64).tolist()

    return {
        "schema": "ddm_tv1_perturbed_field.v1",
        "arm": arm,
        "position_rule": position_rule,
        "value_rule": value_rule,
        "k_target": k,
        "k_changed_verified": changed,
        "base_seed": base_seed,
        "derived_seed": seed,
        "source_tokens_sha256": sha256_of(tokens_path),
        "field": {
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
            "sha256": sha256_of(out_path),
        },
        "value_marginal_fallbacks": fallbacks,
        "changed_bits_in_shipped_stream": bits,
        "changed_bytes_in_shipped_stream": bits / 8.0,
        "addressing_free_rate_credit_S": S_PER_BYTE * bits / 8.0,
        "break_even_delta_d_seg": S_PER_BYTE * (bits / 8.0) / 100.0,
        "one_to_one_transfer_delta_d_seg": k / POSITIONS,
        "class_transitions_from_to": transitions.tolist(),
        "changed_per_frame": {
            "min": int(per_frame.min()),
            "max": int(per_frame.max()),
            "mean": float(per_frame.mean()),
            "frames_touched": int(np.count_nonzero(per_frame)),
        },
        "boundary_distance_histogram_0_to_4": distance_hist,
    }


def publish(
    *,
    runtime_root: Path,
    archive_path: Path,
    field_path: Path,
    cache_root: Path,
    threads: int,
) -> dict[str, Any]:
    """Publish a field into a token cache the pinned receiver will consume.

    The cache binding is COMPUTED by the receiver's own ``_token_cache_binding``
    from the archive it will decode; no key is ever hand-typed here.
    """
    runtime_root = runtime_root.resolve()
    sys.path.insert(0, str(runtime_root))
    import runtime.ddm_wc1_advisory_runtime as wc1
    import runtime.f26_inflate as f26_inflate
    import runtime.residual_archive as residual_archive

    parts = residual_archive.read_residual_archive(archive_path)
    renderer = f26_inflate._load_renderer(runtime_root / "cpr1")
    fingerprint = f26_inflate._token_decoder_fingerprint(
        renderer=renderer,
        renderer_dir=runtime_root / "cpr1",
        token_decoder="python",
        num_threads=threads,
    )
    binding = f26_inflate._token_cache_binding(
        parts=parts, pair_count=int(renderer.N), fingerprint=fingerprint
    )
    entry, report = wc1.publish_token_cache(
        cache_root,
        binding,
        source=field_path,
        expected_bytes=int(renderer.N) * int(renderer.EVAL_H) * int(renderer.EVAL_W),
        created={
            "token_decoder": {
                "token_codec": "ddm_tv1_counterfactual_field",
                "decoded_token_sha256": sha256_of(field_path),
                "provenance": "ddm_tv1 seeded counterfactual; NOT an arithmetic decode",
            }
        },
    )
    return {
        "schema": "ddm_tv1_token_cache_publish.v1",
        "cache_root": str(cache_root),
        "entry": str(entry),
        "field_sha256": sha256_of(field_path),
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build")
    b.add_argument("--arm", required=True, choices=sorted(
        f"{p}_{v}" for p in POSITION_RULES for v in VALUE_RULES
    ))
    b.add_argument("--k", type=int, required=True)
    b.add_argument("--base-seed", type=int, default=20260824)
    b.add_argument("--tokens", type=Path, required=True)
    b.add_argument("--cost", type=Path, required=True)
    b.add_argument("--conditional", type=Path, default=None)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--distance-field", type=Path, default=None)
    b.add_argument("--manifest", type=Path, required=True)

    d = sub.add_parser("distance")
    d.add_argument("--tokens", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("publish")
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "distance":
        tokens = np.fromfile(args.tokens, dtype=np.uint8).reshape(FRAMES, EVAL_H, EVAL_W)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        boundary_distance(tokens).tofile(args.out)
        print(json.dumps({"distance_field": str(args.out), "sha256": sha256_of(args.out)}))
        return 0
    if args.command == "build":
        report = build(
            arm=args.arm,
            k=args.k,
            base_seed=args.base_seed,
            tokens_path=args.tokens,
            cost_path=args.cost,
            conditional_path=args.conditional,
            out_path=args.out,
            distance_path=args.distance_field,
        )
    else:
        report = publish(
            runtime_root=args.runtime_root,
            archive_path=args.archive,
            field_path=args.field,
            cache_root=args.cache_root,
            threads=args.threads,
        )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
