#!/usr/bin/env python3
"""FEC8 Markov 3rd-order rate-attack empirical measurement + sparse 4-tuple counts.

Wave N+24 Option A next-iteration depth-axis extension per CLAUDE.md "Final rate
attack" standing directive + just-saved standing directive
``rate-attack-default-cost-class-is-zero-mlx-or-cpu-not-paid-modal-standing-directive-20260528``.

Sister of:

  * ``tools/measure_fec8_markov_2nd_order_p19_bucket_extension.py`` (the 2nd-order
    sister; emitted ``EMPIRICAL_SECOND_ORDER_TRANSITION_COUNTS`` + measured
    166-byte WIN at -83B vs FEC6 249B baseline + ``-66B vs FEC8 1st-order 232B``)
  * ``submissions/hnerv_fec6_fixed_huffman_k16/encoder/build_pr101_frame_exploit_selector_packet_markov.py``
    (the FEC8 canonical encoder/decoder; this script's predicted output is a
    sister 4-tuple counts table for a NEW ``EMPIRICAL_THIRD_ORDER_TRANSITION_COUNTS``)
  * Cascade A FEC10 hybrid (commit ``1faf05951`` smoke + ``c5a92cce5`` V14V2 +
    canonical equation ``cascade_a_fec10_hybrid_adaptive_blend_savings_v1``)

Measures THREE candidates against live FEC6 (249B) + FEC8 1st-order static (232B)
+ FEC8 2nd-order static (166B) baselines on the PR101 frame-exploit selector
stream (600 pairs from PR101 fp16 archive selector_payload):

  VARIANT-D-TRUE3RD (this script's primary deliverable): TRUE 3rd-order Markov
    ``H(mode_t | mode_{t-1}, mode_{t-2}, mode_{t-3})`` with 16^3 = 4096
    per-context Laplace-smoothed models. Reads 597 4-tuples from the live stream.
    Status: implemented receiver-decode-only static third-order FEC8 variant,
    measured head-to-head against FEC8 2nd-order.

  VARIANT-D-ADAPTIVE: same TRUE 3rd-order Markov but seeded from uniform
    Laplace priors (no transmitted prior table); 4096 per-context adaptive
    Laplace models converged online from 597 4-tuples.

  VARIANT-D-EMPIRICAL-ENTROPY-FLOOR: ``H(mode_t | mode_{t-1}, mode_{t-2}, mode_{t-3})``
    measured directly via conditional Shannon entropy formula. This is the
    information-theoretic floor; the arithmetic coder approaches this asymptotically
    but pays per-context Laplace-smoothing overhead.

Per the data-sparsity analysis: 597 4-tuples observed over 16^3 = 4096 possible
3-prefix contexts means ~0.15 expected observations per context. The 4-tuple
support is HIGHLY SPARSE: many contexts will have zero observations + Laplace
smoothing dominates. Predicted result band per Catalog #296 + sister FEC8 2nd-order
prediction (which carried predicted -82B vs FEC6 + landed -83B EMPIRICAL):

  predicted_wire_bytes: [bytes_saved_min=-15, bytes_saved_max=-25] further vs FEC6
  empirical_data_sparsity_risk: HIGH (4096 contexts vs 597 4-tuples)

[macOS-CPU advisory] axis tag per Catalog #192. NO contest-axis claim. The
measurement is a $0 LOCAL CPU iteration per the just-saved rate-attack-cost-class
standing directive.

# SPDX-License-Identifier: MIT
"""

# PREDICTED_BAND_VIBES_OK:per_Catalog_296_4096_contexts_vs_597_4tuples_data_sparsity_dominates_diminishing_returns_predicted_band_can_go_negative_due_to_per_context_Laplace_smoothing_overhead_acceptable_per_MVP_first_phasing_empirical_proof_methodology

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from collections import Counter
from itertools import pairwise
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FEC8_ENCODER_DIR = REPO_ROOT / "submissions/hnerv_fec6_fixed_huffman_k16/encoder"
if str(FEC8_ENCODER_DIR) not in sys.path:
    sys.path.insert(0, str(FEC8_ENCODER_DIR))

# Reuse canonical FEC8 decoder per Catalog #229 (canonical helper, not re-implementation).
from build_pr101_frame_exploit_selector_packet_markov import (  # type: ignore[import-not-found]  # noqa: E402
    EMPIRICAL_MARGINAL_COUNTS,
    EMPIRICAL_SECOND_ORDER_TRANSITION_COUNTS,
    EMPIRICAL_TRANSITION_COUNTS,
    PALETTE_K,
    _BitReader,
    _BitWriter,
    _ContextModel,
    _build_static_models,
    _build_static_second_order_models,
    _decode_with_models,
    _decode_with_second_order_models,
    _encode_with_models,
    _encode_with_second_order_models,
    decode_fec8_markov_selector,
    encode_fec8_markov_selector_static,
    encode_fec8_markov_selector_static_second_order,
)

from tools.pr101_fec6_wrapper_profile import (  # noqa: E402  (path inserted above)
    FEC6_FIXED_K16_MODE_IDS,
    decode_fec6_fixed_huffman_codes,
)

PRECISION = 32
TOP_VALUE = (1 << PRECISION) - 1
FIRST_QTR = (TOP_VALUE >> 2) + 1
HALF = 2 * FIRST_QTR
THIRD_QTR = 3 * FIRST_QTR

ARCHIVE_PATH = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
    / "archive.zip"
)
FEC6_MAGIC = b"FEC6"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_fec6_selector_payload() -> tuple[bytes, list[int]]:
    """Extract FEC6 selector_payload + decoded per-pair codes from PR101 archive."""
    if not ARCHIVE_PATH.exists():
        raise SystemExit(f"FEC6 archive not found: {ARCHIVE_PATH}")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        member_name = zf.namelist()[0]
        with zf.open(member_name) as fh:
            payload = fh.read()
    if payload[:4] != b"FP11":
        raise SystemExit(f"expected FP11 magic, got {payload[:4]!r}")
    source_len = int.from_bytes(payload[4:8], "little")
    selector_len_start = 8 + source_len
    selector_len = int.from_bytes(
        payload[selector_len_start : selector_len_start + 2], "little"
    )
    selector_start = selector_len_start + 2
    selector_end = selector_start + selector_len
    selector_payload = payload[selector_start:selector_end]
    if selector_payload[:4] != FEC6_MAGIC:
        raise SystemExit(f"FEC6 magic mismatch: {selector_payload[:4]!r}")
    n_pairs = int.from_bytes(selector_payload[4:6], "little")
    if n_pairs != 600:
        raise SystemExit(f"expected n_pairs=600, got {n_pairs}")
    index_payload = selector_payload[6:]
    codes, _used_bits = decode_fec6_fixed_huffman_codes(
        index_payload, n_pairs=n_pairs
    )
    return selector_payload, codes


def _compute_third_order_transition_counts(
    codes: list[int],
) -> dict[tuple[int, int, int, int], int]:
    """Return sparse raw 4-tuple counts (prev3, prev2, prev1, next) over 597 transitions."""
    if len(codes) < 4:
        return {}
    counts: dict[tuple[int, int, int, int], int] = {}
    for prev3, prev2, prev1, nxt in zip(
        codes[:-3], codes[1:-2], codes[2:-1], codes[3:], strict=True
    ):
        key = (prev3, prev2, prev1, nxt)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _shannon_entropy_marginal(codes: list[int]) -> float:
    counter: Counter[int] = Counter(codes)
    return _shannon_entropy_from_counter(counter)


def _shannon_entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def _conditional_entropy_first_order(codes: list[int]) -> float:
    if len(codes) < 2:
        return 0.0
    transitions: Counter[tuple[int, int]] = Counter()
    prev_counts: Counter[int] = Counter()
    for prev, nxt in pairwise(codes):
        transitions[(prev, nxt)] += 1
        prev_counts[prev] += 1
    total = sum(transitions.values())
    h = 0.0
    for (prev, _nxt), count in transitions.items():
        p_joint = count / total
        p_prev = prev_counts[prev] / total
        if p_joint > 0 and p_prev > 0:
            h -= p_joint * math.log2(p_joint / p_prev)
    return h


def _conditional_entropy_second_order(codes: list[int]) -> float:
    if len(codes) < 3:
        return 0.0
    triples: Counter[tuple[int, int, int]] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    for prev2, prev1, nxt in zip(codes[:-2], codes[1:-1], codes[2:], strict=True):
        triples[(prev2, prev1, nxt)] += 1
        pair_counts[(prev2, prev1)] += 1
    total = sum(triples.values())
    h = 0.0
    for (prev2, prev1, _nxt), count in triples.items():
        p_joint = count / total
        p_pair = pair_counts[(prev2, prev1)] / total
        if p_joint > 0 and p_pair > 0:
            h -= p_joint * math.log2(p_joint / p_pair)
    return h


def _conditional_entropy_third_order(codes: list[int]) -> float:
    if len(codes) < 4:
        return 0.0
    quads: Counter[tuple[int, int, int, int]] = Counter()
    triple_counts: Counter[tuple[int, int, int]] = Counter()
    for prev3, prev2, prev1, nxt in zip(
        codes[:-3], codes[1:-2], codes[2:-1], codes[3:], strict=True
    ):
        quads[(prev3, prev2, prev1, nxt)] += 1
        triple_counts[(prev3, prev2, prev1)] += 1
    total = sum(quads.values())
    h = 0.0
    for (prev3, prev2, prev1, _nxt), count in quads.items():
        p_joint = count / total
        p_triple = triple_counts[(prev3, prev2, prev1)] / total
        if p_joint > 0 and p_triple > 0:
            h -= p_joint * math.log2(p_joint / p_triple)
    return h


# -----------------------------------------------------------------------------
# 3rd-order encoder/decoder (prototype; sister to 2nd-order in canonical module)
# -----------------------------------------------------------------------------


def _third_order_context_index(prev3: int, prev2: int, prev1: int) -> int:
    return (prev3 * PALETTE_K + prev2) * PALETTE_K + prev1


def _build_static_third_order_models(
    third_order_counts: dict[tuple[int, int, int, int], int],
) -> tuple[
    _ContextModel,
    list[_ContextModel],
    list[_ContextModel],
    list[_ContextModel],
]:
    """Build prior, 1st-order fallback, 2nd-order fallback, and 4096 3rd-order context models."""
    prior_model, first_order_models = _build_static_models()
    _prior_so, _first_so, second_order_models = _build_static_second_order_models()
    raw_counts = [[0] * PALETTE_K for _ in range(PALETTE_K * PALETTE_K * PALETTE_K)]
    for (prev3, prev2, prev1, nxt), count in third_order_counts.items():
        if not (
            0 <= prev3 < PALETTE_K
            and 0 <= prev2 < PALETTE_K
            and 0 <= prev1 < PALETTE_K
            and 0 <= nxt < PALETTE_K
            and count > 0
        ):
            raise ValueError(
                f"invalid third-order count row: {(prev3, prev2, prev1, nxt, count)}"
            )
        raw_counts[_third_order_context_index(prev3, prev2, prev1)][nxt] += int(count)
    third_order_models = [_ContextModel(counts) for counts in raw_counts]
    return prior_model, first_order_models, second_order_models, third_order_models


def _encode_with_third_order_models(
    codes: list[int],
    n_pairs: int,
    prior_model: _ContextModel,
    first_order_models: list[_ContextModel],
    second_order_models: list[_ContextModel],
    third_order_models: list[_ContextModel],
) -> bytes:
    writer = _BitWriter()
    low = 0
    high = TOP_VALUE
    follow = 0
    prev3: int | None = None
    prev2: int | None = None
    prev1: int | None = None

    for sym in codes:
        if prev1 is None:
            model = prior_model
        elif prev2 is None:
            model = first_order_models[prev1]
        elif prev3 is None:
            from build_pr101_frame_exploit_selector_packet_markov import (
                _second_order_context_index,
            )
            model = second_order_models[_second_order_context_index(prev2, prev1)]
        else:
            model = third_order_models[_third_order_context_index(prev3, prev2, prev1)]
        total = model.total()
        lo = model.cum[sym]
        hi = model.cum[sym + 1]
        rng = high - low + 1
        high = low + (rng * hi) // total - 1
        low = low + (rng * lo) // total

        while True:
            if high < HALF:
                writer.write_bit(0)
                for _ in range(follow):
                    writer.write_bit(1)
                follow = 0
            elif low >= HALF:
                writer.write_bit(1)
                for _ in range(follow):
                    writer.write_bit(0)
                follow = 0
                low -= HALF
                high -= HALF
            elif low >= FIRST_QTR and high < THIRD_QTR:
                follow += 1
                low -= FIRST_QTR
                high -= FIRST_QTR
            else:
                break
            low = (low << 1) & TOP_VALUE
            high = ((high << 1) | 1) & TOP_VALUE

        model.update(sym)
        prev3, prev2, prev1 = prev2, prev1, sym

    follow += 1
    if low < FIRST_QTR:
        writer.write_bit(0)
        for _ in range(follow):
            writer.write_bit(1)
    else:
        writer.write_bit(1)
        for _ in range(follow):
            writer.write_bit(0)

    return writer.finish()


def _decode_with_third_order_models(
    bitstream: bytes,
    n_pairs: int,
    prior_model: _ContextModel,
    first_order_models: list[_ContextModel],
    second_order_models: list[_ContextModel],
    third_order_models: list[_ContextModel],
) -> list[int]:
    reader = _BitReader(bitstream)
    value = 0
    for _ in range(PRECISION):
        value = (value << 1) | reader.read_bit()

    low = 0
    high = TOP_VALUE
    out: list[int] = []
    prev3: int | None = None
    prev2: int | None = None
    prev1: int | None = None

    for _ in range(n_pairs):
        if prev1 is None:
            model = prior_model
        elif prev2 is None:
            model = first_order_models[prev1]
        elif prev3 is None:
            from build_pr101_frame_exploit_selector_packet_markov import (
                _second_order_context_index,
            )
            model = second_order_models[_second_order_context_index(prev2, prev1)]
        else:
            model = third_order_models[_third_order_context_index(prev3, prev2, prev1)]
        total = model.total()
        rng = high - low + 1
        target = ((value - low + 1) * total - 1) // rng
        target = min(target, total - 1)
        sym = model.find_symbol(target)
        lo = model.cum[sym]
        hi = model.cum[sym + 1]
        high = low + (rng * hi) // total - 1
        low = low + (rng * lo) // total

        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                value -= HALF
                low -= HALF
                high -= HALF
            elif low >= FIRST_QTR and high < THIRD_QTR:
                value -= FIRST_QTR
                low -= FIRST_QTR
                high -= FIRST_QTR
            else:
                break
            low = (low << 1) & TOP_VALUE
            high = ((high << 1) | 1) & TOP_VALUE
            value = ((value << 1) | reader.read_bit()) & TOP_VALUE

        out.append(sym)
        model.update(sym)
        prev3, prev2, prev1 = prev2, prev1, sym

    return out


# -----------------------------------------------------------------------------
# Adaptive variant — uniform Laplace priors, no shared prior table
# -----------------------------------------------------------------------------


def _build_adaptive_third_order_models() -> tuple[
    _ContextModel,
    list[_ContextModel],
    list[_ContextModel],
    list[_ContextModel],
]:
    zero = [0] * PALETTE_K
    prior_model = _ContextModel(zero)
    first_order_models = [_ContextModel(zero) for _ in range(PALETTE_K)]
    second_order_models = [_ContextModel(zero) for _ in range(PALETTE_K * PALETTE_K)]
    third_order_models = [
        _ContextModel(zero) for _ in range(PALETTE_K * PALETTE_K * PALETTE_K)
    ]
    return prior_model, first_order_models, second_order_models, third_order_models


# -----------------------------------------------------------------------------
# Measurement harness
# -----------------------------------------------------------------------------


def _huffman_codeword_lengths(counts: Counter) -> dict[int, int]:
    """Compute per-symbol Huffman codeword lengths for a single context."""
    if not counts:
        return {}
    if len(counts) == 1:
        (sym,) = counts.keys()
        return {sym: 1}
    import heapq

    heap: list[tuple[int, int, dict[int, int]]] = []
    tie = 0
    for sym, freq in sorted(counts.items()):
        heap.append((freq, tie, {sym: 0}))
        tie += 1
    heapq.heapify(heap)
    while len(heap) > 1:
        f1, _, d1 = heapq.heappop(heap)
        f2, _, d2 = heapq.heappop(heap)
        merged = {}
        for sym, length in d1.items():
            merged[sym] = length + 1
        for sym, length in d2.items():
            merged[sym] = length + 1
        heapq.heappush(heap, (f1 + f2, tie, merged))
        tie += 1
    return heap[0][2]


def _per_context_huffman_wire_bits(
    codes: list[int], *, context_fn, context_dim_label: str
) -> dict:
    """Per-context Huffman wire-bit measurement.

    Apples-to-apples with sister Catalog measurement at canonical equation
    fec8_2nd_order_true_markov_variant_a_savings_v1 anchor[0] which used the
    same methodology to derive the 166B figure.
    """
    if len(codes) < 2:
        return {"total_wire_bits": 0.0, "bits_per_pair": 0.0, "n_contexts": 0}
    per_context_counts: dict[object, Counter] = {}
    pair_contexts: list[tuple[object, int]] = []
    for t in range(len(codes)):
        ctx = context_fn(codes, t)
        if ctx is None:
            continue
        per_context_counts.setdefault(ctx, Counter())[codes[t]] += 1
        pair_contexts.append((ctx, codes[t]))
    per_context_lengths = {
        ctx: _huffman_codeword_lengths(counts)
        for ctx, counts in per_context_counts.items()
    }
    total_wire_bits = sum(per_context_lengths[ctx][sym] for ctx, sym in pair_contexts)
    return {
        "total_wire_bits": float(total_wire_bits),
        "bits_per_pair": total_wire_bits / max(len(pair_contexts), 1),
        "n_contexts": len(per_context_counts),
        "n_pairs_coded": len(pair_contexts),
        "context_dim_label": context_dim_label,
    }


def _ctx_third_order_true(codes: list[int], t: int):
    return (codes[t - 3], codes[t - 2], codes[t - 1]) if t >= 3 else None


def _ctx_second_order_true(codes: list[int], t: int):
    return (codes[t - 2], codes[t - 1]) if t >= 2 else None


def measure_fec8_third_order(
    codes: list[int], *, n_pairs: int
) -> dict:
    """Measure FEC8 3rd-order static + adaptive against existing baselines."""
    # 1. Live FEC6 baseline.
    fec6_payload, _ = _extract_fec6_selector_payload()
    fec6_bytes_total = len(fec6_payload)

    # 2. FEC8 1st-order static (existing canonical).
    fec8_1st_payload = encode_fec8_markov_selector_static(codes, n_pairs=n_pairs)
    fec8_1st_bytes_total = len(fec8_1st_payload)
    fec8_1st_roundtrip = decode_fec8_markov_selector(fec8_1st_payload)
    assert fec8_1st_roundtrip == codes, "FEC8 1st-order static roundtrip mismatch"

    # 3. FEC8 2nd-order static (existing canonical).
    fec8_2nd_payload = encode_fec8_markov_selector_static_second_order(
        codes, n_pairs=n_pairs
    )
    fec8_2nd_bytes_total = len(fec8_2nd_payload)
    fec8_2nd_roundtrip = decode_fec8_markov_selector(fec8_2nd_payload)
    assert fec8_2nd_roundtrip == codes, "FEC8 2nd-order static roundtrip mismatch"

    # 4. FEC8 3rd-order static (NEW VARIANT-D-TRUE3RD).
    third_order_counts = _compute_third_order_transition_counts(codes)
    (
        prior_so3,
        first_so3,
        second_so3,
        third_so3,
    ) = _build_static_third_order_models(third_order_counts)
    fec8_3rd_static_bitstream = _encode_with_third_order_models(
        codes, n_pairs, prior_so3, first_so3, second_so3, third_so3
    )
    # 6-byte header per FEC8 wire format (magic FEC8 + variant b"\x00\x04" + n_pairs u16)
    fec8_3rd_static_header = b"FEC8" + b"\x00\x04" + n_pairs.to_bytes(2, "little")
    fec8_3rd_static_payload = fec8_3rd_static_header + fec8_3rd_static_bitstream
    fec8_3rd_static_bytes_total = len(fec8_3rd_static_payload)

    # 4a. Roundtrip the 3rd-order static (rebuild fresh models — encode mutates).
    (
        prior_so3_dec,
        first_so3_dec,
        second_so3_dec,
        third_so3_dec,
    ) = _build_static_third_order_models(third_order_counts)
    fec8_3rd_static_roundtrip = _decode_with_third_order_models(
        fec8_3rd_static_bitstream,
        n_pairs,
        prior_so3_dec,
        first_so3_dec,
        second_so3_dec,
        third_so3_dec,
    )
    assert (
        fec8_3rd_static_roundtrip == codes
    ), "FEC8 3rd-order static roundtrip mismatch"

    # 5. FEC8 3rd-order adaptive (NEW VARIANT-D-ADAPTIVE).
    (
        prior_a3,
        first_a3,
        second_a3,
        third_a3,
    ) = _build_adaptive_third_order_models()
    fec8_3rd_adaptive_bitstream = _encode_with_third_order_models(
        codes, n_pairs, prior_a3, first_a3, second_a3, third_a3
    )
    fec8_3rd_adaptive_header = b"FEC8" + b"\x00\x05" + n_pairs.to_bytes(2, "little")
    fec8_3rd_adaptive_payload = fec8_3rd_adaptive_header + fec8_3rd_adaptive_bitstream
    fec8_3rd_adaptive_bytes_total = len(fec8_3rd_adaptive_payload)

    # 5a. Roundtrip the 3rd-order adaptive.
    (
        prior_a3_dec,
        first_a3_dec,
        second_a3_dec,
        third_a3_dec,
    ) = _build_adaptive_third_order_models()
    fec8_3rd_adaptive_roundtrip = _decode_with_third_order_models(
        fec8_3rd_adaptive_bitstream,
        n_pairs,
        prior_a3_dec,
        first_a3_dec,
        second_a3_dec,
        third_a3_dec,
    )
    assert (
        fec8_3rd_adaptive_roundtrip == codes
    ), "FEC8 3rd-order adaptive roundtrip mismatch"

    # 6. Shannon entropy floors.
    h_marg = _shannon_entropy_marginal(codes)
    h_1st = _conditional_entropy_first_order(codes)
    h_2nd = _conditional_entropy_second_order(codes)
    h_3rd = _conditional_entropy_third_order(codes)

    # 6a. Per-context Huffman wire-bit floor measurements (apples-to-apples with
    # the sister 166B canonical anchor for FEC8 2nd-order; same methodology
    # applied to the new 3rd-order context).
    huffman_2nd = _per_context_huffman_wire_bits(
        codes,
        context_fn=_ctx_second_order_true,
        context_dim_label="(prev2, prev1)",
    )
    huffman_3rd = _per_context_huffman_wire_bits(
        codes,
        context_fn=_ctx_third_order_true,
        context_dim_label="(prev3, prev2, prev1)",
    )
    fec8_2nd_huffman_wire_bytes = 8 + math.ceil(huffman_2nd["total_wire_bits"] / 8)
    fec8_3rd_huffman_wire_bytes = 8 + math.ceil(huffman_3rd["total_wire_bits"] / 8)

    # 7. Predicted vs empirical comparison.
    return {
        "measurement_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_pairs": n_pairs,
        "n_third_order_quads_observed": len(third_order_counts),
        "n_third_order_contexts_possible": PALETTE_K * PALETTE_K * PALETTE_K,
        "data_sparsity_ratio_observed_over_possible": (
            sum(third_order_counts.values()) / (PALETTE_K**4)
        ),
        "live_fec6_baseline_bytes": fec6_bytes_total,
        "fec8_1st_order_static_bytes": fec8_1st_bytes_total,
        "fec8_2nd_order_static_bytes": fec8_2nd_bytes_total,
        "fec8_3rd_order_static_bytes_total": fec8_3rd_static_bytes_total,
        "fec8_3rd_order_static_bitstream_bytes": len(fec8_3rd_static_bitstream),
        "fec8_3rd_order_adaptive_bytes_total": fec8_3rd_adaptive_bytes_total,
        "fec8_3rd_order_adaptive_bitstream_bytes": len(fec8_3rd_adaptive_bitstream),
        "delta_bytes_vs_fec6_static_3rd": fec8_3rd_static_bytes_total - fec6_bytes_total,
        "delta_bytes_vs_fec6_adaptive_3rd": fec8_3rd_adaptive_bytes_total - fec6_bytes_total,
        "delta_bytes_vs_fec8_2nd_static_3rd": (
            fec8_3rd_static_bytes_total - fec8_2nd_bytes_total
        ),
        "delta_bytes_vs_fec8_2nd_adaptive_3rd": (
            fec8_3rd_adaptive_bytes_total - fec8_2nd_bytes_total
        ),
        "shannon_h_marginal_bits_per_pair": h_marg,
        "shannon_h_1st_order_bits_per_pair": h_1st,
        "shannon_h_2nd_order_bits_per_pair": h_2nd,
        "shannon_h_3rd_order_bits_per_pair": h_3rd,
        "shannon_h_3rd_order_floor_bytes_for_n_pairs": (h_3rd * n_pairs) / 8,
        "huffman_per_context_2nd_order_wire_bytes_with_header": fec8_2nd_huffman_wire_bytes,
        "huffman_per_context_2nd_order_n_contexts": huffman_2nd["n_contexts"],
        "huffman_per_context_3rd_order_wire_bytes_with_header": fec8_3rd_huffman_wire_bytes,
        "huffman_per_context_3rd_order_n_contexts": huffman_3rd["n_contexts"],
        "huffman_per_context_3rd_order_bits_per_pair": huffman_3rd["bits_per_pair"],
        "delta_huffman_3rd_vs_2nd_bytes": (
            fec8_3rd_huffman_wire_bytes - fec8_2nd_huffman_wire_bytes
        ),
        "delta_huffman_3rd_vs_fec6_bytes": fec8_3rd_huffman_wire_bytes - fec6_bytes_total,
        "fec8_1st_order_static_roundtrip": True,
        "fec8_2nd_order_static_roundtrip": True,
        "fec8_3rd_order_static_roundtrip": True,
        "fec8_3rd_order_adaptive_roundtrip": True,
        "fec8_3rd_order_static_payload_sha256": _sha256_bytes(fec8_3rd_static_payload),
        "fec8_3rd_order_adaptive_payload_sha256": _sha256_bytes(
            fec8_3rd_adaptive_payload
        ),
        "evidence_grade": "macOS-CPU-advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
    }


def emit_empirical_third_order_table(
    third_order_counts: dict[tuple[int, int, int, int], int],
) -> str:
    """Emit a Python source-literal tuple of 4-tuples (prev3, prev2, prev1, next, count)."""
    lines = ["EMPIRICAL_THIRD_ORDER_TRANSITION_COUNTS: tuple[tuple[int, int, int, int, int], ...] = ("]
    sorted_items = sorted(third_order_counts.items())
    for (prev3, prev2, prev1, nxt), count in sorted_items:
        lines.append(f"    ({prev3}, {prev2}, {prev1}, {nxt}, {count}),")
    lines.append(")")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-table-only",
        action="store_true",
        help="emit the EMPIRICAL_THIRD_ORDER_TRANSITION_COUNTS table; suppress measurement",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=REPO_ROOT
        / ".omx/state/fec8_3rd_order_markov_rate_attack_measurement_20260528.json",
        help="JSON report output path",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=REPO_ROOT
        / ".omx/state/fec8_3rd_order_markov_empirical_table_20260528.py",
        help="Python source table output path",
    )
    args = parser.parse_args()

    _, codes = _extract_fec6_selector_payload()
    print(f"[measure-fec8-3rd] decoded {len(codes)} pairs from PR101 FEC6 archive")
    print(f"[measure-fec8-3rd] code distribution: {dict(Counter(codes))}")

    third_order_counts = _compute_third_order_transition_counts(codes)
    print(
        f"[measure-fec8-3rd] observed {len(third_order_counts)} unique 4-tuples over "
        f"{sum(third_order_counts.values())} transitions"
    )
    print(
        f"[measure-fec8-3rd] sparsity: {len(third_order_counts)} unique / "
        f"{PALETTE_K**3} possible contexts = "
        f"{100 * len(third_order_counts) / (PALETTE_K**3):.2f}% coverage"
    )

    table_src = emit_empirical_third_order_table(third_order_counts)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.write_text(table_src + "\n")
    print(f"[measure-fec8-3rd] wrote empirical table to {args.table_out}")

    if args.emit_table_only:
        return

    result = measure_fec8_third_order(codes, n_pairs=len(codes))
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print()
    print("=" * 75)
    print("FEC8 3RD-ORDER MARKOV RATE-ATTACK MEASUREMENT (macOS-CPU advisory)")
    print("=" * 75)
    print(f"  n_pairs:                                 {result['n_pairs']}")
    print(f"  unique 4-tuples observed:                {result['n_third_order_quads_observed']}")
    print(f"  contexts possible (16^3):                {result['n_third_order_contexts_possible']}")
    print(
        f"  data_sparsity_ratio:                     "
        f"{result['data_sparsity_ratio_observed_over_possible']:.4f}"
    )
    print()
    print(f"  live FEC6 baseline (bytes):              {result['live_fec6_baseline_bytes']}")
    print(f"  FEC8 1st-order static (bytes):           {result['fec8_1st_order_static_bytes']}")
    print(f"  FEC8 2nd-order static (bytes):           {result['fec8_2nd_order_static_bytes']}")
    print(f"  FEC8 3rd-order static (bytes):           {result['fec8_3rd_order_static_bytes_total']}")
    print(f"  FEC8 3rd-order adaptive (bytes):         {result['fec8_3rd_order_adaptive_bytes_total']}")
    print()
    print(f"  delta vs FEC6 (3rd static):              {result['delta_bytes_vs_fec6_static_3rd']:+d}")
    print(f"  delta vs FEC6 (3rd adaptive):            {result['delta_bytes_vs_fec6_adaptive_3rd']:+d}")
    print(f"  delta vs FEC8 2nd (3rd static):          {result['delta_bytes_vs_fec8_2nd_static_3rd']:+d}")
    print(f"  delta vs FEC8 2nd (3rd adaptive):        {result['delta_bytes_vs_fec8_2nd_adaptive_3rd']:+d}")
    print()
    print(
        f"  Shannon H(marginal) bits/pair:           {result['shannon_h_marginal_bits_per_pair']:.4f}"
    )
    print(
        f"  Shannon H(X|prev1) bits/pair:            {result['shannon_h_1st_order_bits_per_pair']:.4f}"
    )
    print(
        f"  Shannon H(X|prev1,prev2) bits/pair:      {result['shannon_h_2nd_order_bits_per_pair']:.4f}"
    )
    print(
        f"  Shannon H(X|prev1,prev2,prev3) bits/pair: "
        f"{result['shannon_h_3rd_order_bits_per_pair']:.4f}"
    )
    print(
        f"  Shannon 3rd-order floor (bytes for n_pairs): "
        f"{result['shannon_h_3rd_order_floor_bytes_for_n_pairs']:.1f}"
    )
    print()
    print("APPLES-TO-APPLES PER-CONTEXT HUFFMAN COMPARISON (sister 166B methodology):")
    print(
        f"  per-context Huffman 2nd-order (sister 166B canonical anchor): "
        f"{result['huffman_per_context_2nd_order_wire_bytes_with_header']}B "
        f"({result['huffman_per_context_2nd_order_n_contexts']} contexts)"
    )
    print(
        f"  per-context Huffman 3rd-order (NEW): "
        f"{result['huffman_per_context_3rd_order_wire_bytes_with_header']}B "
        f"({result['huffman_per_context_3rd_order_n_contexts']} contexts; "
        f"{result['huffman_per_context_3rd_order_bits_per_pair']:.4f} bits/pair)"
    )
    print(
        f"  delta huffman 3rd vs 2nd: "
        f"{result['delta_huffman_3rd_vs_2nd_bytes']:+d}B"
    )
    print(
        f"  delta huffman 3rd vs FEC6: "
        f"{result['delta_huffman_3rd_vs_fec6_bytes']:+d}B"
    )
    print()
    print(f"  roundtrip 1st: {result['fec8_1st_order_static_roundtrip']}")
    print(f"  roundtrip 2nd: {result['fec8_2nd_order_static_roundtrip']}")
    print(f"  roundtrip 3rd static: {result['fec8_3rd_order_static_roundtrip']}")
    print(f"  roundtrip 3rd adaptive: {result['fec8_3rd_order_adaptive_roundtrip']}")
    print()
    print(f"  3rd static payload sha256:  {result['fec8_3rd_order_static_payload_sha256']}")
    print(f"  3rd adaptive payload sha256: {result['fec8_3rd_order_adaptive_payload_sha256']}")
    print()
    print(f"  axis_tag:           {result['axis_tag']}")
    print(f"  score_claim:        {result['score_claim']}")
    print(f"  promotable:         {result['promotable']}")
    print()
    print(f"  report saved to: {args.report_out}")


if __name__ == "__main__":
    main()
