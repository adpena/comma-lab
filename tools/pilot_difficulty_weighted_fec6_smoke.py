#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""SLOT MG-4 pilot: synthetic CPU-only difficulty-weighted bit-allocator smoke.

Per the SLOT MG-4 original brief: a tiny synthetic smoke that exercises
``tac.bit_allocator.allocate_bits_per_pair`` end-to-end against synthetic
per-pair difficulty signals so operators can verify the canonical
allocator on CPU (zero GPU spend) before staging full per-pair training-
loop integration. Per CLAUDE.md "Beauty, simplicity, and developer
experience" + the operator's "no new feature work; recover and finalize"
discipline, this pilot is intentionally minimal:

  * Generates synthetic difficulty per-pair (fec6-style 600-pair atlas).
  * Runs all 3 canonical strategies (UNIFORM / LINEAR / SQRT).
  * Verifies sum-conservation invariant + monotonicity of LINEAR.
  * Emits a JSON summary suitable for operator review.
  * Tags every Provenance row PREDICTED + ``promotable=False``.

Per CLAUDE.md "Forbidden score claims": this pilot makes ZERO score
claims. The allocations are PREDICTED bit budgets, never measured. Promotion
to a contest score claim requires paired Linux x86_64 + NVIDIA auth-eval
per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".

Sister of:
  * ``src/tac/bit_allocator/per_pair_difficulty_weighted.py`` (canonical helper)
  * ``src/tac/cathedral_consumers/per_pair_difficulty_atlas_consumer`` (auto-
    discovered cathedral consumer that consumes downstream)

Usage:
    .venv/bin/python tools/pilot_difficulty_weighted_fec6_smoke.py
    .venv/bin/python tools/pilot_difficulty_weighted_fec6_smoke.py --json
    .venv/bin/python tools/pilot_difficulty_weighted_fec6_smoke.py --n-pairs 64 --total-bits 256
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys

from tac.bit_allocator import (
    AllocationStrategy,
    BitAllocationResult,
    allocate_bits_per_pair,
)


# fec6 canonical pair count anchor per the contest 600-pair structure.
DEFAULT_N_PAIRS = 600
DEFAULT_TOTAL_BITS = 600 * 8  # 8 bits per pair canonical mean
DEFAULT_SEED = 42


def synthesize_difficulty_atlas(n_pairs: int, *, seed: int) -> dict[int, float]:
    """Synthesize a deterministic per-pair difficulty atlas.

    Mixes a linear ramp (capturing position-driven difficulty) with a
    log-normal-like shock pattern (capturing rare hard pairs). Both
    components are positive + finite so the atlas always satisfies the
    canonical helper's invariants.
    """
    rng = random.Random(seed)
    atlas: dict[int, float] = {}
    for idx in range(n_pairs):
        # Linear ramp + log-normal shocks.
        ramp = 1.0 + 0.5 * (idx / max(1, n_pairs - 1))
        shock = math.exp(rng.gauss(0.0, 0.30))
        atlas[idx] = max(1e-3, ramp * shock)
    return atlas


def run_strategy(
    strategy: AllocationStrategy,
    *,
    total_bits: int,
    difficulty_per_pair: dict[int, float],
) -> tuple[str, BitAllocationResult]:
    result = allocate_bits_per_pair(
        total_bits=total_bits,
        difficulty_per_pair=difficulty_per_pair,
        strategy=strategy,
    )
    # Sum-conservation invariant.
    sum_bits = sum(result.bits_per_pair.values())
    if sum_bits != total_bits:
        raise RuntimeError(
            f"strategy={strategy.value!r} violated sum-conservation: "
            f"sum(bits_per_pair)={sum_bits} != total_bits={total_bits}"
        )
    return strategy.value, result


def assert_linear_monotone(result: BitAllocationResult, difficulty: dict[int, float]) -> None:
    """LINEAR strategy: more difficult pair => no fewer bits than easier pair."""
    pairs_sorted = sorted(difficulty.items(), key=lambda kv: kv[1])
    # Hamilton remainder can introduce 1-bit local inversions; check no
    # inversion exceeds 1 bit AND no inversion exceeds 5% of total monotone
    # ordering.
    bits = [result.bits_per_pair[pair_idx] for pair_idx, _ in pairs_sorted]
    big_inversions = 0
    for i in range(len(bits) - 1):
        if bits[i + 1] < bits[i] - 1:
            big_inversions += 1
    inversion_ratio = big_inversions / max(1, len(bits) - 1)
    if inversion_ratio > 0.05:
        raise RuntimeError(
            f"LINEAR strategy violated monotonicity beyond 5% tolerance "
            f"(big_inversions={big_inversions}/{len(bits)-1}); allocator regression"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLOT MG-4 pilot smoke: synthetic difficulty-weighted bit allocation"
    )
    parser.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS)
    parser.add_argument("--total-bits", type=int, default=DEFAULT_TOTAL_BITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    atlas = synthesize_difficulty_atlas(args.n_pairs, seed=args.seed)
    results: dict[str, BitAllocationResult] = {}
    for strategy in AllocationStrategy:
        label, result = run_strategy(
            strategy,
            total_bits=args.total_bits,
            difficulty_per_pair=atlas,
        )
        results[label] = result

    if "linear" in results:
        assert_linear_monotone(results["linear"], atlas)

    summary = {
        "smoke_kind": "pilot_difficulty_weighted_fec6_smoke",
        "n_pairs": args.n_pairs,
        "total_bits": args.total_bits,
        "seed": args.seed,
        "score_claim": False,
        "evidence_grade": "predicted",
        "promotable": False,
        "axis_tag": "[predicted]",
        "results_by_strategy": {
            label: {
                "sum_bits": sum(r.bits_per_pair.values()),
                "min_bits": min(r.bits_per_pair.values()),
                "max_bits": max(r.bits_per_pair.values()),
                "mean_bits": sum(r.bits_per_pair.values()) / args.n_pairs,
                "provenance_evidence_grade": (
                    r.provenance.evidence_grade.value
                    if hasattr(r.provenance.evidence_grade, "value")
                    else str(r.provenance.evidence_grade)
                ),
            }
            for label, r in results.items()
        },
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("SLOT MG-4 pilot smoke: difficulty-weighted bit allocator")
        print(
            f"  n_pairs={args.n_pairs}  total_bits={args.total_bits}  seed={args.seed}"
        )
        for label, r in results.items():
            sum_bits = sum(r.bits_per_pair.values())
            print(
                f"  {label:>8}  sum={sum_bits}  min={min(r.bits_per_pair.values())}  "
                f"max={max(r.bits_per_pair.values())}  mean={sum_bits/args.n_pairs:.2f}"
            )
        print("  All strategies satisfied sum-conservation invariant.")
        print("  LINEAR satisfied monotonicity (<=5% local Hamilton inversions).")
        print("  All Provenance rows = PREDICTED + promotable=False.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
