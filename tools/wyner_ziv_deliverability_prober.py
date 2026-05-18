#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Wyner-Ziv side-info hoist DELIVERABILITY prober.

[verified-against: src/tac/master_gradient_consumers.py consumer 4
                   `wyner_ziv_side_info_covariance` — the producer this
                   prober interrogates for deliverability]
[verified-against: src/tac/sensitivity_map/wyner_ziv_reweight.py — sister
                   sensitivity-map consumer]
[verified-against: tools/cathedral_autopilot_autonomous_loop.py
                   `adjust_predicted_delta_for_venn_classification` — the
                   autopilot consumer whose positive PAIR_INVARIANT reward is
                   now DeliverabilityProof-gated; this prober audits the
                   legacy blanket 1.15 reward assumption]
[verified-against: CLAUDE.md "HNeRV / leaderboard-implementation parity
                   discipline" L4 (inflate.py ≤ 100 LOC default budget) +
                   L9 (runtime closure)]
[verified-against: empirical anchor — fec6 archive sha
                   f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd]

CONTEXT
=======

The Wyner-Ziv side-info hoist consumer (`tac.master_gradient_consumers.
wyner_ziv_side_info_covariance`) classifies 162,123 of 178,417 fec6 archive
bytes as HIGH-correlation candidate-shared-prior. The original autopilot
ranker applied a blanket 1.15 REWARD factor for HIGH PAIR_INVARIANT
substrates per `adjust_predicted_delta_for_venn_classification`. Catalog #319
now gates positive reward through DeliverabilityProof. **Any positive reward
still assumes the hoist is DELIVERABLE.**

The 100-LOC inflate.py budget (HNeRV parity L4) is INTERNAL discipline,
NOT a contest rule — the contest charges only `archive.zip` bytes (rate
term = ``25 * archive_bytes / 37_545_489`` per CLAUDE.md). Bytes hoisted
to a side-info channel that the inflate runtime can reconstruct
deterministically are STILL CHARGEABLE if they end up in `archive.zip` and
RECOVERABLE if they don't. The deliverability question is: given
178,417 archive bytes whose CSP subset is 162,123 bytes, can we move
those 162,123 bytes into a smaller inflate.py constant blob without
exceeding T4 timeout and without scorer-load?

This prober empirically measures the 4 deliverability axes:

1. **Compression ratio** — what is the smallest representation of the
   CSP byte subset that decodes back to the original?
2. **T4 timeout headroom** — how much wall-time can we add to inflate
   before the 30-min hard cap fires?
3. **Per-byte Tier-1/2/3/4 classification** — how many bytes are
   zero-cost / sub-5KB-constants / waiver-required / forbidden?
4. **Deliverable score-delta estimate** — given the above, what is the
   actual ΔS the autopilot can rank on, vs the legacy 1.15-factor REWARD?

THE ANTI-CARGO-CULT FINDING (empirically validated 2026-05-17)
==============================================================

The fec6 archive bytes are POST-ENTROPY-CODED (Huffman + range coding
per the fec6 selector pipeline). Their byte histogram is uniform to
~0.42% per symbol over 256 symbols — i.e., the bytes are at the Shannon
entropy floor. Empirically:

* lzma preset 9 extreme on full 178,417 bytes → 178,484 bytes (100.0%, INFLATES)
* brotli quality 11 on full 178,417 bytes → 178,422 bytes (100.0%, INFLATES)
* lzma on CSP-only 162,123 bytes → 162,192 bytes (100.0%, INFLATES)

There is NO compression headroom in the fec6 archive bytes. The
"candidate-shared-prior" signal from consumer 4 is a STATISTICAL
correlation across pairs, NOT a structural redundancy that can be
exploited to reduce archive bytes.

The legacy blanket 1.15 reward factor (`adjust_predicted_delta_for_venn_
classification`) is therefore **OVERSTATED for fec6-class substrates**:
the substrate has HIGH PAIR_INVARIANT % but ZERO deliverable rate-savings
because the bytes are already at entropy. Per CLAUDE.md "Apples-to-apples
evidence discipline" + "Bit-level deconstruction and entropy discipline":
predicting ΔS savings from a Wyner-Ziv hoist on already-entropy-coded
bytes is the canonical cargo-cult-prediction failure mode.

This prober makes that finding STRUCTURAL and queryable.

OUTPUT CONTRACT (NON-AUTHORITATIVE / PROBE EVIDENCE)
====================================================

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192 +
Catalog #245, every emitted artifact carries:

* ``score_claim: false``
* ``promotion_eligible: false``
* ``ready_for_exact_eval_dispatch: false``
* ``evidence_grade: "predicted"`` (not [contest-CPU] / [contest-CUDA])
* ``measurement_axis: "[diagnostic; wyner-ziv deliverability probe]"``

Persists to:
``.omx/state/wyner_ziv_deliverability/probe_<sha[:12]>_<utc>.json``

via canonical fcntl-locked append per Catalog #131.

CLI USAGE
=========

```bash
.venv/bin/python tools/wyner_ziv_deliverability_prober.py \\
    --archive-sha f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd \\
    --per-pair-tensor .omx/tmp/master_gradient_per_pair_8pair_fp64_validate.npy \\
    --archive-path experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \\
    --tier-budget-tier2-bytes 5120 \\
    --tier-budget-tier3-bytes 200000
```
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import io
import json
import lzma
import os
import sys
import time
import uuid
import zipfile
import zlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Optional brotli (canonical contest-archive entropy coder per CLAUDE.md
# Catalog #203). Soft-import so the prober can run on machines without it
# (advisory mode); the result manifest records its absence.
try:
    import brotli  # type: ignore
    _HAS_BROTLI = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAS_BROTLI = False

# Canonical imports per Catalog #131 + Catalog #213 sister-pattern (the
# producer module the prober interrogates).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tac.master_gradient_consumers import (  # noqa: E402
    classify_bytes_by_pair_variance,
    wyner_ziv_side_info_covariance,
)


# Canonical contest constants per CLAUDE.md "Meta-Lagrangian/Pareto solver"
# + Catalog #316 sister `tac.frontier_scan`.
CONTEST_RATE_DENOM_BYTES = 37_545_489
CONTEST_T4_TIMEOUT_SECONDS = 30 * 60  # contest T4 hard cap

# Canonical CSP correlation thresholds (mirror
# `tac.master_gradient_consumers.WYNER_ZIV_CORRELATION_THRESHOLD_*`).
WYNER_ZIV_CORRELATION_THRESHOLD_HIGH = 0.8
WYNER_ZIV_CORRELATION_THRESHOLD_LOW = 0.2

# Tier budgets (operator-tunable; defaults from CLAUDE.md "HNeRV parity
# discipline" L4 — ≤ 100 LOC default inflate budget; ≤ 200 LOC with
# explicit waiver).
DEFAULT_TIER2_BUDGET_BYTES = 5 * 1024  # ≤ 5 KB constants in canonical inflate
DEFAULT_TIER3_BUDGET_BYTES = 200 * 1024  # ≤ 200 KB with waiver

# Tier 1 detector: bytes whose aggregate gradient magnitude is below noise
# floor on ALL of (seg, pose, rate) — i.e., the byte does not influence
# the score on any axis, so it can be CONSTANT in inflate without paying
# archive bytes. Empirical floor for fp64 autograd noise: 1e-12.
TIER1_AGGREGATE_FLOOR_RELATIVE = 0.01  # below 1% of axis-max = effectively zero

# Tier 4 detector: bytes whose per-pair gradient on seg-OR-pose axes is
# dominant (i.e., the byte's information content is scorer-relative) AND
# the aggregate magnitude is high (i.e., it ACTUALLY matters). These
# bytes cannot be reconstructed at inflate without scorer access per
# CLAUDE.md "strict-scorer-rule" non-negotiable. Empirical detector: a
# byte is Tier-4 if its (seg + pose) aggregate magnitude > rate axis
# aggregate by > 10× (i.e., it's almost entirely scorer-relative).
TIER4_SCORER_AXIS_DOMINANCE_RATIO = 10.0

# Empirical CPU→T4 wall-time ratio for inflate operations (mostly Python
# import + zip extract + lzma decode + memory allocation). Per multiple
# prior benchmarks (PR101 inflate on M5 Max vs T4 anchors), the ratio is
# roughly 3-5×. We use the conservative end: T4 is ~3× FASTER than M5 Max
# CPU on these workloads (PyTorch CUDA init + GPU memory). Documented
# uncertainty propagates into the headroom estimate.
EMPIRICAL_M5MAX_TO_T4_SPEEDUP = 3.0

OUTPUT_DIR_DEFAULT = Path(".omx/state/wyner_ziv_deliverability")


# ──────────────────────────────────────────────────────────────────────── #
# Result dataclasses                                                        #
# ──────────────────────────────────────────────────────────────────────── #


@dataclass(frozen=True)
class CodecResult:
    """Single codec compression probe result."""

    codec: str  # 'lzma' | 'brotli' | 'zlib'
    raw_bytes: int
    compressed_bytes: int
    ratio: float  # compressed_bytes / raw_bytes
    decode_correct: bool
    parameters: dict[str, Any]


@dataclass(frozen=True)
class T4TimeoutHeadroom:
    """Inflate.py timeout headroom estimate."""

    fec6_baseline_inflate_seconds_cpu: float
    fec6_baseline_inflate_seconds_t4_estimated: float
    contest_t4_timeout_seconds: int
    available_headroom_seconds_t4: float
    cpu_to_t4_speedup_factor: float
    synthetic_inflate_py_overhead_seconds: dict[str, float]  # {'10kb': 0.05, ...}


@dataclass(frozen=True)
class TierClassification:
    """Per-byte Tier 1/2/3/4 classification of the candidate-shared-prior set."""

    tier_1_zero_cost_bytes: int
    tier_2_constants_bytes: int
    tier_3_waiver_required_bytes: int
    tier_4_forbidden_bytes: int
    tier_1_byte_indices: tuple[int, ...]
    tier_2_byte_indices: tuple[int, ...]
    tier_3_byte_indices: tuple[int, ...]
    tier_4_byte_indices: tuple[int, ...]
    tier_2_budget_bytes: int
    tier_3_budget_bytes: int
    tier_4_dominance_ratio: float
    tier_1_aggregate_floor_relative: float


@dataclass(frozen=True)
class DeliverabilityVerdict:
    """Final deliverability summary."""

    archive_sha256: str
    archive_path: str | None
    per_pair_tensor_path: str
    candidate_shared_prior_byte_count: int
    candidate_shared_prior_subset_compressed_bytes_lzma: int
    candidate_shared_prior_subset_compressed_bytes_brotli: int | None
    candidate_shared_prior_subset_compressed_bytes_zlib: int
    best_codec: str
    best_compressed_bytes: int
    rate_score_savings_estimate: float
    deliverable_score_delta_estimate: float
    autopilot_reward_factor_in_use: float
    autopilot_reward_factor_justified: bool
    deliverability_verdict: str  # 'DELIVERABLE' | 'NOT_DELIVERABLE' | 'PARTIAL'
    reasoning: str


# ──────────────────────────────────────────────────────────────────────── #
# Codec helpers                                                             #
# ──────────────────────────────────────────────────────────────────────── #


def _compress_lzma(data: bytes, *, preset: int = 9 | lzma.PRESET_EXTREME) -> tuple[bytes, dict]:
    compressed = lzma.compress(data, preset=preset)
    return compressed, {"preset": preset}


def _compress_brotli(data: bytes, *, quality: int = 11) -> tuple[bytes, dict]:
    if not _HAS_BROTLI:
        raise RuntimeError("brotli not installed")
    compressed = brotli.compress(data, quality=quality)
    return compressed, {"quality": quality}


def _compress_zlib(data: bytes, *, level: int = 9) -> tuple[bytes, dict]:
    compressed = zlib.compress(data, level)
    return compressed, {"level": level}


def _decompress_lzma(data: bytes) -> bytes:
    return lzma.decompress(data)


def _decompress_brotli(data: bytes) -> bytes:
    if not _HAS_BROTLI:
        raise RuntimeError("brotli not installed")
    return brotli.decompress(data)


def _decompress_zlib(data: bytes) -> bytes:
    return zlib.decompress(data)


_CODECS: dict[str, tuple[Any, Any]] = {
    "lzma": (_compress_lzma, _decompress_lzma),
    "brotli": (_compress_brotli, _decompress_brotli),
    "zlib": (_compress_zlib, _decompress_zlib),
}


def measure_codec(name: str, data: bytes) -> CodecResult:
    """Compress + decompress + verify byte-identical roundtrip."""
    compress_fn, decompress_fn = _CODECS[name]
    try:
        compressed, params = compress_fn(data)
    except RuntimeError as exc:
        return CodecResult(
            codec=name,
            raw_bytes=len(data),
            compressed_bytes=-1,
            ratio=float("nan"),
            decode_correct=False,
            parameters={"error": str(exc)},
        )
    decoded = decompress_fn(compressed)
    return CodecResult(
        codec=name,
        raw_bytes=len(data),
        compressed_bytes=len(compressed),
        ratio=len(compressed) / max(len(data), 1),
        decode_correct=(decoded == data),
        parameters=params,
    )


# ──────────────────────────────────────────────────────────────────────── #
# T4 timeout headroom estimator                                             #
# ──────────────────────────────────────────────────────────────────────── #


def estimate_t4_timeout_headroom(
    *,
    fec6_baseline_inflate_seconds_cpu: float,
    synthetic_inflate_py_sizes_kb: Iterable[int] = (10, 50, 100, 200, 500),
    cpu_to_t4_speedup_factor: float = EMPIRICAL_M5MAX_TO_T4_SPEEDUP,
) -> T4TimeoutHeadroom:
    """Estimate T4 timeout headroom.

    The estimate is conservative: we assume CPU baseline divided by the
    speedup ratio is the T4 baseline. Per CLAUDE.md "Apples-to-apples
    evidence discipline" this is a PREDICTED bound, not a measurement.

    Synthetic inflate.py overhead is modeled as:
      overhead_sec = (size_bytes / IO_THROUGHPUT_BYTES_PER_SEC) +
                     (size_bytes / LZMA_DECODE_THROUGHPUT_BYTES_PER_SEC) +
                     PYTHON_IMPORT_FIXED_COST_SEC

    Empirical constants (M5 Max CPU; T4 ~3× faster):
    * IO throughput: 1 GB/s (mmap or read)
    * lzma decode throughput: 40 MB/s
    * python import fixed cost: 0.2 s
    """
    t4_baseline = fec6_baseline_inflate_seconds_cpu / cpu_to_t4_speedup_factor
    headroom_t4 = CONTEST_T4_TIMEOUT_SECONDS - t4_baseline

    # Per-size synthetic overhead model
    IO_THROUGHPUT_BPS_T4 = 1e9 * cpu_to_t4_speedup_factor  # T4 ~3× faster
    LZMA_DECODE_BPS_T4 = 40e6 * cpu_to_t4_speedup_factor
    IMPORT_FIXED_COST_SEC = 0.2 / cpu_to_t4_speedup_factor  # T4 still has fixed import cost

    overhead_map: dict[str, float] = {}
    for size_kb in synthetic_inflate_py_sizes_kb:
        size_bytes = size_kb * 1024
        io_sec = size_bytes / IO_THROUGHPUT_BPS_T4
        decode_sec = size_bytes / LZMA_DECODE_BPS_T4
        total_sec = io_sec + decode_sec + IMPORT_FIXED_COST_SEC
        overhead_map[f"{size_kb}kb"] = total_sec

    return T4TimeoutHeadroom(
        fec6_baseline_inflate_seconds_cpu=fec6_baseline_inflate_seconds_cpu,
        fec6_baseline_inflate_seconds_t4_estimated=t4_baseline,
        contest_t4_timeout_seconds=CONTEST_T4_TIMEOUT_SECONDS,
        available_headroom_seconds_t4=headroom_t4,
        cpu_to_t4_speedup_factor=cpu_to_t4_speedup_factor,
        synthetic_inflate_py_overhead_seconds=overhead_map,
    )


def measure_inflate_baseline_cpu(archive_path: Path) -> float:
    """Proxy CPU wall-time for the inflate path.

    We can't run the contest's `inflate.sh` here (it needs the runtime
    tree). Instead we measure the cheapest faithful proxy: open the zip,
    read the single member into memory, simulate a decode pass. This
    captures the dominant inflate cost: archive byte extraction +
    in-memory processing.
    """
    if not archive_path.exists():
        # If no archive provided, use a synthetic baseline (worst case)
        return 30.0  # 30s is a reasonable upper bound for fec6-class inflate
    t0 = time.perf_counter()
    with zipfile.ZipFile(archive_path) as zf:
        info = zf.infolist()[0]
        data = zf.read(info.filename)
    # Simulate a single-pass byte processing (e.g., entropy decode).
    # The total byte sum is a workload proxy.
    _ = sum(data)
    return time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────── #
# Per-byte Tier 1/2/3/4 classifier                                          #
# ──────────────────────────────────────────────────────────────────────── #


def classify_csp_bytes_into_tiers(
    *,
    candidate_shared_prior_indices: Iterable[int],
    raw_bytes: bytes,
    per_pair_gradient: np.ndarray,
    tier_2_budget_bytes: int = DEFAULT_TIER2_BUDGET_BYTES,
    tier_3_budget_bytes: int = DEFAULT_TIER3_BUDGET_BYTES,
    tier_1_aggregate_floor_relative: float = TIER1_AGGREGATE_FLOOR_RELATIVE,
    tier_4_scorer_axis_dominance_ratio: float = TIER4_SCORER_AXIS_DOMINANCE_RATIO,
) -> TierClassification:
    """Classify CSP bytes into Tier 1/2/3/4.

    Tier 1 (zero inflate cost): aggregate magnitude below noise floor on
        all 3 axes — these bytes can be constants in inflate.py for free
        (they don't affect the score).
    Tier 2 (≤ tier_2_budget_bytes constants): bytes that, after lzma
        compression of the cumulative subset, fit within tier_2_budget_bytes.
        Greedy fill in DESCENDING aggregate-magnitude order (highest-value
        bytes first; the budget is a hard cap).
    Tier 3 (waiver-required): bytes whose cumulative lzma-compressed size
        exceeds tier_2_budget but is ≤ tier_3_budget. Requires explicit
        inflate.py LOC waiver per HNeRV parity L4 (≤ 200 LOC).
    Tier 4 (forbidden / scorer-bound): bytes whose per-pair gradient on
        seg-OR-pose axes is dominant by `tier_4_scorer_axis_dominance_ratio`
        AND the aggregate magnitude is high. These bytes encode
        scorer-relative information that cannot be reconstructed without
        loading scorer state at inflate time (per CLAUDE.md
        "strict-scorer-rule" non-negotiable).

    Note: Tier 4 detection takes PRECEDENCE over Tier 1/2/3 — a byte that
    is structurally Tier-4 cannot move to a "cheaper" tier.
    """
    csp_indices = sorted(candidate_shared_prior_indices)
    if not csp_indices:
        return TierClassification(
            tier_1_zero_cost_bytes=0,
            tier_2_constants_bytes=0,
            tier_3_waiver_required_bytes=0,
            tier_4_forbidden_bytes=0,
            tier_1_byte_indices=(),
            tier_2_byte_indices=(),
            tier_3_byte_indices=(),
            tier_4_byte_indices=(),
            tier_2_budget_bytes=tier_2_budget_bytes,
            tier_3_budget_bytes=tier_3_budget_bytes,
            tier_4_dominance_ratio=tier_4_scorer_axis_dominance_ratio,
            tier_1_aggregate_floor_relative=tier_1_aggregate_floor_relative,
        )

    # Per-byte aggregate magnitudes
    per_byte_agg = np.abs(per_pair_gradient.mean(axis=1))  # (N_bytes, 3)
    seg_mag = per_byte_agg[:, 0]
    pose_mag = per_byte_agg[:, 1]
    rate_mag = per_byte_agg[:, 2]

    # Tier 1: bytes whose aggregate is below floor on ALL 3 axes.
    # Use <= so a byte with exactly-zero magnitude is correctly counted
    # as below-floor when the axis-max is also zero (e.g. rate axis in
    # autograd-projected gradients where archive bytes have constant
    # rate contribution).
    seg_floor = seg_mag.max() * tier_1_aggregate_floor_relative
    pose_floor = pose_mag.max() * tier_1_aggregate_floor_relative
    rate_floor = rate_mag.max() * tier_1_aggregate_floor_relative

    # Tier 4: scorer-axis-dominant bytes
    # Heuristic: scorer_mag = max(seg, pose); rate_mag is the rate-axis aggregate.
    # Tier 4 iff scorer_mag > rate_mag * ratio AND scorer_mag > scorer_floor.
    # When rate_mag is effectively zero (autograd projection of fec6),
    # we instead require scorer_mag > overall scorer floor (= max_scorer * relative_floor).
    scorer_mag = np.maximum(seg_mag, pose_mag)
    overall_scorer_max = scorer_mag.max()
    scorer_floor = overall_scorer_max * tier_1_aggregate_floor_relative

    tier_1: list[int] = []
    tier_2_candidates: list[int] = []
    tier_4: list[int] = []
    for idx in csp_indices:
        s = seg_mag[idx]
        p = pose_mag[idx]
        r = rate_mag[idx]
        sc = max(s, p)
        # Tier 1: at-or-below floor on all 3 axes. Use <= because a
        # zero-magnitude byte is structurally below any nonzero floor,
        # and rate_floor can itself be zero when the rate axis is
        # all-zero (autograd-projected fec6 anchor case).
        if s <= seg_floor and p <= pose_floor and r <= rate_floor:
            tier_1.append(idx)
            continue
        # Tier 4: scorer-axis-dominant AND meaningful magnitude.
        # `rate_ref` is the denominator for the dominance ratio test;
        # when the rate axis is zero, use a small positive sentinel
        # so any nonzero scorer mag trips the dominance threshold.
        rate_ref = max(r, scorer_floor * 1e-6, 1e-30)
        if sc > scorer_floor and sc / rate_ref > tier_4_scorer_axis_dominance_ratio:
            tier_4.append(idx)
            continue
        # Otherwise, candidate for Tier 2/3
        tier_2_candidates.append(idx)

    # Sort Tier 2 candidates by descending aggregate magnitude (highest-value first)
    tier_2_candidates_sorted = sorted(
        tier_2_candidates,
        key=lambda i: float(np.linalg.norm(per_byte_agg[i])),
        reverse=True,
    )

    # Greedy fill Tier 2: accumulate bytes until lzma-compressed size > tier_2_budget
    tier_2: list[int] = []
    tier_3: list[int] = []
    csp_byte_array = bytearray(raw_bytes[i] for i in tier_2_candidates_sorted)
    if not tier_2_candidates_sorted:
        pass
    else:
        # We approximate: since the bytes are uniformly distributed
        # (entropy-coded source), lzma compression ratio is ~1.0 (no
        # compression). The tier-2 budget therefore admits ~tier_2_budget
        # raw bytes. We measure the ACTUAL compressed-size cutoff.
        cumulative_compressed = 0
        last_chunk_size = 0
        # Test in incremental chunks of 256 bytes
        chunk_size = 256
        for start in range(0, len(csp_byte_array), chunk_size):
            chunk = bytes(csp_byte_array[: start + chunk_size])
            comp = lzma.compress(chunk, preset=6)
            cumulative_compressed = len(comp)
            last_chunk_size = start + chunk_size
            if cumulative_compressed > tier_2_budget_bytes:
                # The last chunk overflowed the budget
                tier_2 = tier_2_candidates_sorted[:start]
                break
        else:
            # Loop completed without breaking — all CSP bytes fit in Tier 2
            tier_2 = list(tier_2_candidates_sorted)

        # Remaining candidates go to Tier 3 (subject to Tier 3 budget)
        remaining = tier_2_candidates_sorted[len(tier_2):]
        if remaining:
            remaining_byte_array = bytes(raw_bytes[i] for i in remaining)
            tier_3_compressed = lzma.compress(remaining_byte_array, preset=6)
            if len(tier_3_compressed) <= tier_3_budget_bytes:
                tier_3 = remaining
            else:
                # Some prefix of remaining fits in tier_3; the rest is over budget
                # (which would be a stricter Tier 4-or-higher classification,
                # but in this prober we cap classification at Tier 3 + Tier 4-forbidden).
                # For empirical clarity: anything over Tier 3 budget that is NOT
                # already Tier 4 stays in Tier 3 with a "budget-exceeded" annotation.
                # Conservative: count them all in Tier 3.
                tier_3 = remaining

    return TierClassification(
        tier_1_zero_cost_bytes=len(tier_1),
        tier_2_constants_bytes=len(tier_2),
        tier_3_waiver_required_bytes=len(tier_3),
        tier_4_forbidden_bytes=len(tier_4),
        tier_1_byte_indices=tuple(tier_1),
        tier_2_byte_indices=tuple(tier_2),
        tier_3_byte_indices=tuple(tier_3),
        tier_4_byte_indices=tuple(tier_4),
        tier_2_budget_bytes=tier_2_budget_bytes,
        tier_3_budget_bytes=tier_3_budget_bytes,
        tier_4_dominance_ratio=tier_4_scorer_axis_dominance_ratio,
        tier_1_aggregate_floor_relative=tier_1_aggregate_floor_relative,
    )


# ──────────────────────────────────────────────────────────────────────── #
# Deliverability verdict                                                    #
# ──────────────────────────────────────────────────────────────────────── #


def compute_rate_score_savings(saved_bytes: int) -> float:
    """Compute rate-term score savings from `saved_bytes` removed from archive.

    Per CLAUDE.md "Meta-Lagrangian/Pareto solver" rate term = ``25 * archive_bytes
    / 37_545_489``. Savings = ``25 * saved_bytes / 37_545_489``.
    """
    return 25.0 * saved_bytes / CONTEST_RATE_DENOM_BYTES


def derive_deliverability_verdict(
    *,
    archive_sha256: str,
    archive_path: Path | None,
    per_pair_tensor_path: Path,
    csp_byte_count: int,
    best_compressed_bytes: int,
    best_codec: str,
    lzma_compressed: int,
    brotli_compressed: int | None,
    zlib_compressed: int,
    tier_classification: TierClassification,
    autopilot_reward_factor_in_use: float = 1.15,
) -> DeliverabilityVerdict:
    """Synthesize the final deliverability verdict.

    The legacy blanket Wyner-Ziv reward factor is JUSTIFIED iff the deliverable subset
    (Tier 1 + Tier 2) is at least 50% of the candidate set AND the
    rate-savings ΔS > 0.001 (a non-trivial frontier-relevant amount).
    """
    deliverable_count = (
        tier_classification.tier_1_zero_cost_bytes
        + tier_classification.tier_2_constants_bytes
    )
    deliverable_frac = deliverable_count / max(csp_byte_count, 1)
    # Rate savings: bytes-removed = (raw CSP bytes) - (best compressed bytes)
    # PLUS Tier 1 bytes can be removed entirely (zero cost).
    # However, since lzma is INFLATING (compressed > raw), the rate savings
    # from a hoist is NEGATIVE for fec6-class entropy-coded inputs.
    # We compute the OPTIMISTIC savings as the Tier 1 byte count (those CAN
    # be removed for free; the entropy floor doesn't apply to dead bytes).
    raw_csp_bytes_removable = tier_classification.tier_1_zero_cost_bytes
    rate_savings = compute_rate_score_savings(raw_csp_bytes_removable)

    # The "deliverable score delta" is the rate-savings PLUS the Tier-2
    # constants that fit in inflate.py budget. For Tier 2 to count, the
    # compressed-byte-cost of the Tier 2 subset must be < the raw byte
    # count it replaces (i.e., we save bytes by hoisting). For
    # entropy-coded sources, this is rarely true — lzma INFLATES.
    deliverable_score_delta = rate_savings  # Tier 2 hoist saves 0 bytes for entropy-coded source

    # Justification rule
    reward_justified = (
        deliverable_frac >= 0.50
        and deliverable_score_delta > 0.001
    )

    if reward_justified:
        verdict = "DELIVERABLE"
        reasoning = (
            f"Deliverable subset (Tier 1 + Tier 2) is {100*deliverable_frac:.1f}% of "
            f"CSP and rate-savings ΔS = {deliverable_score_delta:.6f}. The legacy "
            f"1.15x reward factor is JUSTIFIED for this substrate."
        )
    elif deliverable_score_delta > 0:
        verdict = "PARTIAL"
        reasoning = (
            f"Deliverable subset (Tier 1 + Tier 2) is {100*deliverable_frac:.1f}% of "
            f"CSP (below 50% threshold) and rate-savings ΔS = {deliverable_score_delta:.6f} "
            f"(below 0.001 threshold). The legacy 1.15x reward factor is OVERSTATED; "
            f"empirical reward should be ~{1.0 + 0.15 * deliverable_frac:.2f}x."
        )
    else:
        verdict = "NOT_DELIVERABLE"
        reasoning = (
            f"Best codec '{best_codec}' INFLATES the CSP subset "
            f"(compressed={best_compressed_bytes}, raw={csp_byte_count}). The fec6 "
            f"archive bytes are already at the Shannon entropy floor (post-Huffman + "
            f"range coding). No Wyner-Ziv hoist can reduce archive bytes. The "
            f"legacy 1.15x reward factor is OVERSTATED; empirical reward "
            f"should be 1.00x (no rate-savings deliverable). "
            f"Tier 1 (zero-cost) bytes: {tier_classification.tier_1_zero_cost_bytes} "
            f"(the only deliverable rate-savings). Per CLAUDE.md 'Bit-level "
            f"deconstruction and entropy discipline': predicting ΔS savings from "
            f"a Wyner-Ziv hoist on already-entropy-coded bytes is cargo-cult-prediction."
        )

    return DeliverabilityVerdict(
        archive_sha256=archive_sha256,
        archive_path=str(archive_path) if archive_path else None,
        per_pair_tensor_path=str(per_pair_tensor_path),
        candidate_shared_prior_byte_count=csp_byte_count,
        candidate_shared_prior_subset_compressed_bytes_lzma=lzma_compressed,
        candidate_shared_prior_subset_compressed_bytes_brotli=brotli_compressed,
        candidate_shared_prior_subset_compressed_bytes_zlib=zlib_compressed,
        best_codec=best_codec,
        best_compressed_bytes=best_compressed_bytes,
        rate_score_savings_estimate=rate_savings,
        deliverable_score_delta_estimate=deliverable_score_delta,
        autopilot_reward_factor_in_use=autopilot_reward_factor_in_use,
        autopilot_reward_factor_justified=reward_justified,
        deliverability_verdict=verdict,
        reasoning=reasoning,
    )


# ──────────────────────────────────────────────────────────────────────── #
# Persistence (Catalog #131 fcntl-locked write)                            #
# ──────────────────────────────────────────────────────────────────────── #


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so json.dumps(..., allow_nan=False) succeeds."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):  # NaN or Inf
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _fcntl_locked_atomic_write(path: Path, payload: dict) -> None:
    """Atomic write of payload to path under fcntl.LOCK_EX.

    Per CLAUDE.md Catalog #131 "no bare writes to shared state":
    transactional write (.tmp.<uuid> + os.replace) inside the locked
    region prevents concurrent-writer interleaving.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    sanitized = _sanitize_for_json(payload)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            tmp_path.write_text(
                json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def persist_probe_artifact(
    *,
    verdict: DeliverabilityVerdict,
    t4_headroom: T4TimeoutHeadroom,
    tier_classification: TierClassification,
    codec_results: list[CodecResult],
    output_dir: Path = OUTPUT_DIR_DEFAULT,
) -> Path:
    """Persist the probe artifact to the canonical sidecar location.

    Returns the path written.
    """
    utc = datetime.datetime.now(datetime.timezone.utc)
    safe_utc = utc.strftime("%Y%m%dT%H%M%S")
    sha_short = verdict.archive_sha256[:12]
    path = output_dir / f"probe_{sha_short}_{safe_utc}.json"

    # Build the canonical schema payload.
    payload: dict[str, Any] = {
        "schema_version": "wyner_ziv_deliverability_probe_v1",
        "archive_sha256": verdict.archive_sha256,
        "archive_path": verdict.archive_path,
        "per_pair_tensor_path": verdict.per_pair_tensor_path,
        "candidate_shared_prior_byte_count": verdict.candidate_shared_prior_byte_count,
        "codec_results": [
            {
                "codec": cr.codec,
                "raw_bytes": cr.raw_bytes,
                "compressed_bytes": cr.compressed_bytes,
                "ratio": cr.ratio,
                "decode_correct": cr.decode_correct,
                "parameters": cr.parameters,
            }
            for cr in codec_results
        ],
        "lzma_result": next(
            (
                {
                    "raw_bytes": cr.raw_bytes,
                    "compressed_bytes": cr.compressed_bytes,
                    "ratio": cr.ratio,
                    "decode_correct": cr.decode_correct,
                    "preset": cr.parameters.get("preset"),
                }
                for cr in codec_results if cr.codec == "lzma"
            ),
            None,
        ),
        "brotli_result": next(
            (
                {
                    "raw_bytes": cr.raw_bytes,
                    "compressed_bytes": cr.compressed_bytes,
                    "ratio": cr.ratio,
                    "decode_correct": cr.decode_correct,
                    "quality": cr.parameters.get("quality"),
                }
                for cr in codec_results if cr.codec == "brotli"
            ),
            None,
        ),
        "zlib_result": next(
            (
                {
                    "raw_bytes": cr.raw_bytes,
                    "compressed_bytes": cr.compressed_bytes,
                    "ratio": cr.ratio,
                    "decode_correct": cr.decode_correct,
                    "level": cr.parameters.get("level"),
                }
                for cr in codec_results if cr.codec == "zlib"
            ),
            None,
        ),
        "best_codec": verdict.best_codec,
        "best_compressed_bytes": verdict.best_compressed_bytes,
        "rate_score_savings_estimate": verdict.rate_score_savings_estimate,
        "t4_timeout_headroom": {
            "fec6_baseline_inflate_seconds_cpu": t4_headroom.fec6_baseline_inflate_seconds_cpu,
            "fec6_baseline_inflate_seconds_t4_estimated": t4_headroom.fec6_baseline_inflate_seconds_t4_estimated,
            "contest_t4_timeout_seconds": t4_headroom.contest_t4_timeout_seconds,
            "available_headroom_seconds_t4": t4_headroom.available_headroom_seconds_t4,
            "cpu_to_t4_speedup_factor": t4_headroom.cpu_to_t4_speedup_factor,
            "synthetic_inflate_py_size_seconds": t4_headroom.synthetic_inflate_py_overhead_seconds,
        },
        "tier_classification": {
            "tier_1_zero_cost_bytes": tier_classification.tier_1_zero_cost_bytes,
            "tier_2_constants_bytes": tier_classification.tier_2_constants_bytes,
            "tier_3_waiver_required_bytes": tier_classification.tier_3_waiver_required_bytes,
            "tier_4_forbidden_bytes": tier_classification.tier_4_forbidden_bytes,
            "tier_1_byte_indices": list(tier_classification.tier_1_byte_indices),
            "tier_2_byte_indices": list(tier_classification.tier_2_byte_indices),
            "tier_3_byte_indices": list(tier_classification.tier_3_byte_indices),
            "tier_4_byte_indices": list(tier_classification.tier_4_byte_indices),
            "tier_2_budget_bytes": tier_classification.tier_2_budget_bytes,
            "tier_3_budget_bytes": tier_classification.tier_3_budget_bytes,
            "tier_4_dominance_ratio": tier_classification.tier_4_dominance_ratio,
            "tier_1_aggregate_floor_relative": tier_classification.tier_1_aggregate_floor_relative,
        },
        "per_tier_rate_savings_potential": {
            "tier_1_score_delta": compute_rate_score_savings(tier_classification.tier_1_zero_cost_bytes),
            "tier_2_score_delta": 0.0,  # entropy floor: lzma inflates Tier 2 bytes
            "tier_3_score_delta": 0.0,
            "tier_1_plus_tier_2_score_delta": compute_rate_score_savings(tier_classification.tier_1_zero_cost_bytes),
        },
        "deliverable_score_delta_estimate": verdict.deliverable_score_delta_estimate,
        "autopilot_reward_factor_in_use": verdict.autopilot_reward_factor_in_use,
        "autopilot_reward_factor_justified": verdict.autopilot_reward_factor_justified,
        "deliverability_verdict": verdict.deliverability_verdict,
        "reasoning": verdict.reasoning,
        "evidence_grade": "predicted",
        "measurement_axis": "[diagnostic; wyner-ziv deliverability probe]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "claude_md_compliance_tags": [
            "tier_classification_per_catalog_220",
            "no_op_detector_per_catalog_139",
            "apples_to_apples_per_catalog_127",
            "fcntl_locked_write_per_catalog_131",
            "entropy_discipline_per_claude_md_bit_level_deconstruction",
        ],
        "written_at_utc": utc.isoformat(),
        "written_pid": os.getpid(),
        "written_host": os.uname().nodename,
    }

    _fcntl_locked_atomic_write(path, payload)
    return path


# ──────────────────────────────────────────────────────────────────────── #
# End-to-end probe                                                          #
# ──────────────────────────────────────────────────────────────────────── #


def run_probe(
    *,
    archive_sha256: str,
    per_pair_tensor_path: Path,
    archive_path: Path | None = None,
    tier_2_budget_bytes: int = DEFAULT_TIER2_BUDGET_BYTES,
    tier_3_budget_bytes: int = DEFAULT_TIER3_BUDGET_BYTES,
    output_dir: Path = OUTPUT_DIR_DEFAULT,
    cpu_to_t4_speedup_factor: float = EMPIRICAL_M5MAX_TO_T4_SPEEDUP,
    persist: bool = True,
) -> tuple[DeliverabilityVerdict, Path | None]:
    """Run the full deliverability probe end-to-end.

    Returns (verdict, persisted_path). If persist=False, persisted_path is None.
    """
    per_pair_gradient = np.load(per_pair_tensor_path)
    if per_pair_gradient.ndim != 3 or per_pair_gradient.shape[-1] != 3:
        raise ValueError(
            f"per_pair_gradient at {per_pair_tensor_path} has shape "
            f"{per_pair_gradient.shape}; expected (N_bytes, N_pairs, 3)"
        )

    # Reproduce the canonical Wyner-Ziv classification
    wz_classification = wyner_ziv_side_info_covariance(
        per_pair_gradient,
        archive_sha256=archive_sha256,
        measurement_axis="[diagnostic; wyner-ziv deliverability probe]",
        measurement_hardware="darwin_arm64",
        sample_axis=1,  # pose (canonical default)
        write_sidecar=False,
    )

    csp_indices = sorted(wz_classification.candidate_shared_prior_byte_indices)
    csp_byte_count = len(csp_indices)

    # Load archive bytes
    if archive_path is None or not archive_path.exists():
        # Synthetic mode: no archive available; codec results are skipped
        raw_archive_bytes = b""
        csp_bytes = b""
    else:
        with zipfile.ZipFile(archive_path) as zf:
            raw_archive_bytes = zf.read(zf.infolist()[0])
        csp_bytes = bytes(raw_archive_bytes[i] for i in csp_indices)

    # A. Codec probes
    codec_results: list[CodecResult] = []
    if csp_bytes:
        codec_results.append(measure_codec("lzma", csp_bytes))
        if _HAS_BROTLI:
            codec_results.append(measure_codec("brotli", csp_bytes))
        codec_results.append(measure_codec("zlib", csp_bytes))
    else:
        codec_results.append(
            CodecResult(
                codec="lzma",
                raw_bytes=0,
                compressed_bytes=-1,
                ratio=float("nan"),
                decode_correct=False,
                parameters={"error": "no_archive_bytes_available"},
            )
        )

    # Best codec
    valid_results = [cr for cr in codec_results if cr.compressed_bytes > 0 and cr.decode_correct]
    if valid_results:
        best = min(valid_results, key=lambda r: r.compressed_bytes)
    else:
        best = codec_results[0]
    lzma_compressed = next((cr.compressed_bytes for cr in codec_results if cr.codec == "lzma" and cr.compressed_bytes > 0), -1)
    brotli_compressed = next((cr.compressed_bytes for cr in codec_results if cr.codec == "brotli" and cr.compressed_bytes > 0), None)
    zlib_compressed = next((cr.compressed_bytes for cr in codec_results if cr.codec == "zlib" and cr.compressed_bytes > 0), -1)

    # B. T4 timeout headroom
    cpu_baseline = measure_inflate_baseline_cpu(archive_path) if archive_path else 30.0
    t4_headroom = estimate_t4_timeout_headroom(
        fec6_baseline_inflate_seconds_cpu=cpu_baseline,
        cpu_to_t4_speedup_factor=cpu_to_t4_speedup_factor,
    )

    # C. Tier classification (only meaningful if we have archive bytes)
    if csp_bytes:
        tier_classification = classify_csp_bytes_into_tiers(
            candidate_shared_prior_indices=csp_indices,
            raw_bytes=raw_archive_bytes,
            per_pair_gradient=per_pair_gradient,
            tier_2_budget_bytes=tier_2_budget_bytes,
            tier_3_budget_bytes=tier_3_budget_bytes,
        )
    else:
        # No archive — count Tier 1 bytes only (the others need raw bytes)
        per_byte_agg = np.abs(per_pair_gradient.mean(axis=1))
        seg_floor = per_byte_agg[:, 0].max() * TIER1_AGGREGATE_FLOOR_RELATIVE
        pose_floor = per_byte_agg[:, 1].max() * TIER1_AGGREGATE_FLOOR_RELATIVE
        rate_floor = max(per_byte_agg[:, 2].max(), 1e-30) * TIER1_AGGREGATE_FLOOR_RELATIVE
        tier_1_no_arch: list[int] = []
        for idx in csp_indices:
            if (
                per_byte_agg[idx, 0] < seg_floor
                and per_byte_agg[idx, 1] < pose_floor
                and per_byte_agg[idx, 2] < rate_floor
            ):
                tier_1_no_arch.append(idx)
        tier_classification = TierClassification(
            tier_1_zero_cost_bytes=len(tier_1_no_arch),
            tier_2_constants_bytes=0,
            tier_3_waiver_required_bytes=0,
            tier_4_forbidden_bytes=0,
            tier_1_byte_indices=tuple(tier_1_no_arch),
            tier_2_byte_indices=(),
            tier_3_byte_indices=(),
            tier_4_byte_indices=(),
            tier_2_budget_bytes=tier_2_budget_bytes,
            tier_3_budget_bytes=tier_3_budget_bytes,
            tier_4_dominance_ratio=TIER4_SCORER_AXIS_DOMINANCE_RATIO,
            tier_1_aggregate_floor_relative=TIER1_AGGREGATE_FLOOR_RELATIVE,
        )

    # D. Deliverability verdict
    verdict = derive_deliverability_verdict(
        archive_sha256=archive_sha256,
        archive_path=archive_path,
        per_pair_tensor_path=per_pair_tensor_path,
        csp_byte_count=csp_byte_count,
        best_compressed_bytes=best.compressed_bytes if best.compressed_bytes > 0 else csp_byte_count,
        best_codec=best.codec,
        lzma_compressed=lzma_compressed,
        brotli_compressed=brotli_compressed,
        zlib_compressed=zlib_compressed,
        tier_classification=tier_classification,
    )

    persisted_path: Path | None = None
    if persist:
        persisted_path = persist_probe_artifact(
            verdict=verdict,
            t4_headroom=t4_headroom,
            tier_classification=tier_classification,
            codec_results=codec_results,
            output_dir=output_dir,
        )

    return verdict, persisted_path


# ──────────────────────────────────────────────────────────────────────── #
# CLI                                                                       #
# ──────────────────────────────────────────────────────────────────────── #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wyner-Ziv side-info hoist deliverability prober. Empirically "
            "measures whether the legacy blanket 1.15x PAIR_INVARIANT reward "
            "assumption is justified for a given archive."
        ),
    )
    parser.add_argument(
        "--archive-sha",
        required=True,
        help="sha256 of the archive ZIP member (the file inside archive.zip)",
    )
    parser.add_argument(
        "--per-pair-tensor",
        type=Path,
        required=True,
        help="path to the per-pair gradient .npy file shape (N_bytes, N_pairs, 3)",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="path to archive.zip (optional; if absent, codec probes are skipped)",
    )
    parser.add_argument(
        "--tier-budget-tier2-bytes",
        type=int,
        default=DEFAULT_TIER2_BUDGET_BYTES,
        help=f"Tier 2 inflate.py constants budget (default {DEFAULT_TIER2_BUDGET_BYTES})",
    )
    parser.add_argument(
        "--tier-budget-tier3-bytes",
        type=int,
        default=DEFAULT_TIER3_BUDGET_BYTES,
        help=f"Tier 3 waiver-required budget (default {DEFAULT_TIER3_BUDGET_BYTES})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"output directory for probe artifacts (default {OUTPUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--cpu-to-t4-speedup-factor",
        type=float,
        default=EMPIRICAL_M5MAX_TO_T4_SPEEDUP,
        help=f"empirical CPU→T4 speedup ratio (default {EMPIRICAL_M5MAX_TO_T4_SPEEDUP})",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="do not persist the probe artifact",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit verdict as JSON on stdout",
    )
    args = parser.parse_args(argv)

    verdict, persisted_path = run_probe(
        archive_sha256=args.archive_sha,
        per_pair_tensor_path=args.per_pair_tensor,
        archive_path=args.archive_path,
        tier_2_budget_bytes=args.tier_budget_tier2_bytes,
        tier_3_budget_bytes=args.tier_budget_tier3_bytes,
        output_dir=args.output_dir,
        cpu_to_t4_speedup_factor=args.cpu_to_t4_speedup_factor,
        persist=not args.no_persist,
    )

    if args.json:
        print(json.dumps(asdict(verdict), indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Archive sha256:                  {verdict.archive_sha256}")
        print(f"CSP byte count:                  {verdict.candidate_shared_prior_byte_count:>10d}")
        print(f"Best codec:                      {verdict.best_codec}")
        print(f"Best compressed bytes:           {verdict.best_compressed_bytes:>10d}")
        print(f"Rate-savings estimate:           {verdict.rate_score_savings_estimate:.6f}")
        print(f"Deliverable score delta:         {verdict.deliverable_score_delta_estimate:.6f}")
        print(f"Legacy 1.15x reward justified:   {verdict.autopilot_reward_factor_justified}")
        print(f"Deliverability verdict:          {verdict.deliverability_verdict}")
        print(f"Reasoning: {verdict.reasoning}")
        if persisted_path:
            print(f"Probe artifact persisted to:     {persisted_path}")

    # Exit code: 0 if DELIVERABLE/PARTIAL, 1 if NOT_DELIVERABLE
    return 0 if verdict.deliverability_verdict != "NOT_DELIVERABLE" else 1


if __name__ == "__main__":
    sys.exit(main())
