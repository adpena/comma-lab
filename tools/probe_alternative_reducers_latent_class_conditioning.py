#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Alternative-reducer H(latent | scorer-class-derived-signal) probes.

Per T2 council Q1 SPLIT-VERDICT reactivation criteria + Catalog #308 META-pattern E
remediation (operator-routed 2026-05-16): the canonical per-pair-dominant SegNet
argmax reducer was empirically falsified on ``upstream/videos/0.mkv``
(600/600 pairs map to class 2 = road; I(latent ; per-pair-dominant-class) = 0.000
bits/symbol per ``experiments/results/wunderkind_g1_v2_real_cuda_section14_reprobe_20260516T185807Z/``).
Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #103
the v2 paradigm class (any SegNet-derived conditioning) cannot be deferred
class-wide until alternative reducers have been independently probed.

This module implements 4 alternative reducers each with their own MI threshold
per T2 council Q1.4 reactivation criteria:

1. ``PER_PIXEL_HISTOGRAM``  (threshold > 0.5 bits/pair)
   For each pair, compute a 5-bin pixel-class histogram over the (384, 512)
   SegNet argmax map. The histogram is a 5-tuple of pixel-counts summing to
   196608. Quantize each bin to one of N_BIN_QUANT levels (default 16 => ~4 bits
   per bin x 5 bins = 20-bit fingerprint) and use the fingerprint as the per-pair
   conditioning symbol. This preserves spatial CLASS DIVERSITY within each pair
   even when a single class dominates the argmax.

2. ``PER_REGION_HISTOGRAM``  (threshold > 1.0 bits/pair)
   Split each 384x512 argmax map into 4 spatial regions (top-left, top-right,
   bottom-left, bottom-right; each 192x256 = 49152 pixels). Compute a 5-bin
   histogram per region. Concatenate 4 region-histograms into a 20-bin
   fingerprint (quantize each bin similarly). Captures spatial-conditional info
   that whole-pair histograms miss (e.g. dashcam sky vs road vs lane-line
   distinguishes top-half from bottom-half).

3. ``PER_PAIR_CLASS_2_FRACTION``  (threshold > 0.2 bits/pair)
   For each pair, compute the fraction of pixels classified as class 2 (road).
   This is a continuous signal in [0.0, 1.0] which we quantize into N_FRACTION_BUCKETS
   buckets (default 32 => ~5-bit per-pair symbol). Even when every pair has
   dominant class 2, the fraction may vary (e.g. 0.60 in an open-road pair vs
   0.95 in a tight road pair) - that variability IS the conditioning signal.

4. ``PER_FRAME_ARGMAX``  (threshold > 0.2 bits/pair)
   The per-pair-dominant reducer aggregates frame_0 + frame_1 argmax maps via
   mode (per ``g1_v2_per_pair_dominant_class_from_segnet_argmax``). The
   per-FRAME variant computes the dominant class of frame_0 and frame_1
   SEPARATELY and concatenates them into a per-pair 2-tuple (range
   [0, num_classes^2) => 25-class fingerprint for num_classes=5). Temporal
   motion (a car appearing/disappearing) can make frame_0_class != frame_1_class
   even when the per-pair-mode is the same.

   For substrates where the renderer outputs ONLY frame_1 (HNeRV-style), this
   reducer degenerates to per-frame argmax of frame_1 only; we still compute it
   as a baseline.

The 4 reducers consume the SAME per-pair (384, 512) argmax maps as input - the
single expensive operation (decode + SegNet forward over 600 pairs) is shared
across all 4 reducers per CLAUDE.md "Tier 1 engineering hygiene".

Each reducer returns a typed verdict per the canonical
``HLatentGivenScorerClassVerdict`` taxonomy (MEANINGFUL_CONDITIONING /
WEAK_CONDITIONING / INDEPENDENT) plus the threshold band the reducer was
evaluated against.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192 + #249:
the emitted JSON is fail-closed (``score_claim=false``, evidence grade
``diagnostic_cpu``, axis explicitly tagged, output directory uses
``alternative_reducer_probes_<timestamp>`` naming convention NOT
``_cuda``/``_cpu``-named directory per Catalog #249).

Per CLAUDE.md "Forbidden /tmp paths" (Catalog #109/#113): outputs land under
``experiments/results/alternative_reducer_probes_<timestamp>/`` with full
provenance.

Per Catalog #110/#113 HISTORICAL_PROVENANCE: result JSONs are append-only;
each probe run gets its own timestamped directory.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Re-use the canonical entropy + verdict primitives so the new reducers are
# Shannon-equivalent on the conditioning side and only differ in the REDUCER
# function applied to the per-pair argmax map.
from tools.probe_latent_conditional_entropy_h_latent_given_scorer_class import (
    INDEPENDENCE_TOLERANCE_BITS,
    _count_joint,
    _count_symbols,
    _plug_in_entropy_bits,
)

# --- Reducer enumeration + per-reducer canonical MI thresholds ---

ReducerName = Literal[
    "per_pixel_histogram",
    "per_region_histogram",
    "per_pair_class_2_fraction",
    "per_frame_argmax",
]

# Per T2 council Q1.4 reactivation criteria (sextet pact 2026-05-16 binding).
# Each reducer's MEANINGFUL threshold reflects the canonical conditioning
# information bandwidth available under that reducer.
REDUCER_MEANINGFUL_THRESHOLD_BITS: dict[ReducerName, float] = {
    "per_pixel_histogram": 0.5,
    "per_region_histogram": 1.0,
    "per_pair_class_2_fraction": 0.2,
    "per_frame_argmax": 0.2,
}

# Quantization defaults - chosen to make the per-pair symbol stream fit in a
# uint8 OR uint16 fingerprint and to keep the symbol-count tractable for the
# entropy estimator (plug-in entropy degrades when ``n_symbols / n_unique`` is
# small - Miller-Madow bias).
PER_PIXEL_HISTOGRAM_BIN_QUANT_DEFAULT: int = 16  # 5 bins x 4 bits = 20-bit fingerprint
PER_REGION_HISTOGRAM_BIN_QUANT_DEFAULT: int = 8  # 20 bins x 3 bits = 60-bit fingerprint
PER_PAIR_CLASS_2_FRACTION_BUCKETS_DEFAULT: int = 32  # ~5-bit per-pair symbol
NUM_SEGNET_CLASSES_DEFAULT: int = 5  # canonical SegNet comma2k19 5-class output
CLASS_2_INDEX_DEFAULT: int = 2  # canonical "road" class index in comma2k19 SegNet


@dataclass(frozen=True)
class AlternativeReducerVerdict:
    """Machine-readable verdict for one (substrate, reducer) pair.

    Mirrors the canonical ``HLatentGivenScorerClassVerdict`` schema but
    annotates the specific reducer name + threshold the verdict was computed
    against so consumers cannot conflate verdicts across reducers.
    """

    substrate_id: str
    reducer_name: ReducerName
    verdict: str  # one of MEANINGFUL_CONDITIONING / WEAK_CONDITIONING / INDEPENDENT
    h_latent_unconditional_bits_per_symbol: float
    h_latent_given_reduced_class_bits_per_symbol: float
    mutual_information_bits: float
    wyner_ziv_gain_ceiling_fraction: float
    num_latent_symbols: int
    num_reduced_class_symbols: int
    num_unique_reduced_classes: int
    meaningful_mi_threshold_bits: float
    independence_tolerance_bits: float
    reducer_specific_parameters: dict[str, int | float]
    evidence_grade: str
    score_claim: bool
    axis_label: str
    observed_at_utc: str
    notes: str


# --- Reducer 1: per-pixel histogram ---


def reduce_per_pixel_histogram(
    *,
    argmax_map: Sequence[int] | object,  # (H * W,) sequence of int classes
    num_classes: int = NUM_SEGNET_CLASSES_DEFAULT,
    bin_quant_levels: int = PER_PIXEL_HISTOGRAM_BIN_QUANT_DEFAULT,
) -> int:
    """Reduce a per-pair argmax map to a single per-pair fingerprint integer.

    Algorithm:
      1. Count the number of pixels per class (``num_classes`` bins).
      2. Quantize each bin's count to ``bin_quant_levels`` levels via uniform
         affine map ``q = floor(count / total * bin_quant_levels)`` clamped to
         ``[0, bin_quant_levels - 1]``.
      3. Pack the quantized bin values into a single integer fingerprint via
         positional encoding ``sum(q[k] * bin_quant_levels^k for k in range(num_classes))``.

    The fingerprint integer is the per-pair conditioning symbol. Two pairs with
    structurally identical pixel-class distributions map to the same
    fingerprint; pairs with distinct distributions map to distinct fingerprints
    (up to quantization granularity).

    Raises ``ValueError`` on contract violations per CLAUDE.md "Comment-only
    contracts are FORBIDDEN".
    """
    if num_classes < 2:
        raise ValueError(f"num_classes={num_classes} must be >= 2")
    if bin_quant_levels < 2:
        raise ValueError(f"bin_quant_levels={bin_quant_levels} must be >= 2")
    flat: list[int] = list(argmax_map)
    if len(flat) == 0:
        raise ValueError("argmax_map is empty")
    counts = [0] * num_classes
    for c in flat:
        ci = int(c)
        if not 0 <= ci < num_classes:
            raise ValueError(
                f"argmax pixel value {ci} out of range [0, {num_classes})"
            )
        counts[ci] += 1
    total = len(flat)
    quantized = [0] * num_classes
    for k in range(num_classes):
        q = int(counts[k] / total * bin_quant_levels)
        if q >= bin_quant_levels:
            q = bin_quant_levels - 1
        quantized[k] = q
    fingerprint = 0
    for k, q in enumerate(quantized):
        fingerprint += q * (bin_quant_levels ** k)
    return fingerprint


# --- Reducer 2: per-region histogram ---


def reduce_per_region_histogram(
    *,
    argmax_map_2d: Sequence[Sequence[int]] | object,  # (H, W) 2D sequence of int classes
    num_classes: int = NUM_SEGNET_CLASSES_DEFAULT,
    bin_quant_levels: int = PER_REGION_HISTOGRAM_BIN_QUANT_DEFAULT,
    num_regions: int = 4,
) -> int:
    """Reduce a per-pair (H, W) argmax map to a per-pair fingerprint integer.

    Algorithm:
      1. Split (H, W) into ``num_regions`` spatial quadrants (default 4:
         top-left, top-right, bottom-left, bottom-right).
      2. For each region, count pixels per class and quantize bins like the
         per-pixel-histogram reducer.
      3. Pack the (num_regions x num_classes) quantized bins into a single
         positional-encoded integer fingerprint.

    For num_regions=4 (the canonical T2 Q1.4 specification), the fingerprint
    captures (5 classes x 4 regions) = 20 bins worth of spatial-conditional
    information - substantially more than the 5-bin whole-pair histogram.

    Raises ``ValueError`` on contract violations.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes={num_classes} must be >= 2")
    if bin_quant_levels < 2:
        raise ValueError(f"bin_quant_levels={bin_quant_levels} must be >= 2")
    if num_regions != 4:
        # We only implement the canonical 2x2 split per T2 Q1.4. Future
        # operator-requested splits should add a generic n-way grid here.
        raise ValueError(
            f"num_regions={num_regions} unsupported; only 4 (2x2 split) implemented"
        )
    rows = list(argmax_map_2d)
    if len(rows) == 0:
        raise ValueError("argmax_map_2d is empty")
    h = len(rows)
    w = len(list(rows[0]))
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(
            f"argmax_map_2d shape ({h}, {w}) must have even H and W for 2x2 split"
        )
    half_h = h // 2
    half_w = w // 2
    # 4 regions x num_classes counts
    region_counts: list[list[int]] = [[0] * num_classes for _ in range(num_regions)]
    region_totals: list[int] = [0] * num_regions
    for y in range(h):
        row = list(rows[y])
        if len(row) != w:
            raise ValueError(
                f"argmax_map_2d row {y} has length {len(row)} != W={w}"
            )
        for x in range(w):
            # Region 0=TL, 1=TR, 2=BL, 3=BR
            ri = (1 if y >= half_h else 0) * 2 + (1 if x >= half_w else 0)
            ci = int(row[x])
            if not 0 <= ci < num_classes:
                raise ValueError(
                    f"argmax pixel ({y}, {x}) value {ci} out of range [0, {num_classes})"
                )
            region_counts[ri][ci] += 1
            region_totals[ri] += 1
    # Quantize each region's bin then positional-encode
    fingerprint = 0
    place = 1
    for r in range(num_regions):
        total_r = region_totals[r]
        if total_r == 0:
            continue
        for k in range(num_classes):
            q = int(region_counts[r][k] / total_r * bin_quant_levels)
            if q >= bin_quant_levels:
                q = bin_quant_levels - 1
            fingerprint += q * place
            place *= bin_quant_levels
    return fingerprint


# --- Reducer 3: per-pair class-2-fraction ---


def reduce_per_pair_class_2_fraction(
    *,
    argmax_map: Sequence[int] | object,
    class_index: int = CLASS_2_INDEX_DEFAULT,
    num_buckets: int = PER_PAIR_CLASS_2_FRACTION_BUCKETS_DEFAULT,
) -> int:
    """Reduce a per-pair argmax map to a quantized class-2-fraction bucket index.

    Algorithm:
      1. Compute fraction of pixels equal to ``class_index`` (default 2 = road).
      2. Quantize fraction to ``num_buckets`` buckets via uniform affine map
         ``b = floor(fraction * num_buckets)`` clamped to ``[0, num_buckets-1]``.

    Even when every pair has dominant class 2 (60-95% road pixels), the
    per-pair class-2-fraction is a continuous signal in roughly [0.55, 0.95]
    on dashcam content. The bucket index per pair is the conditioning symbol.

    Raises ``ValueError`` on contract violations.
    """
    if num_buckets < 2:
        raise ValueError(f"num_buckets={num_buckets} must be >= 2")
    flat: list[int] = list(argmax_map)
    if len(flat) == 0:
        raise ValueError("argmax_map is empty")
    n_class = sum(1 for c in flat if int(c) == class_index)
    fraction = n_class / len(flat)
    b = int(fraction * num_buckets)
    if b >= num_buckets:
        b = num_buckets - 1
    return b


# --- Reducer 4: per-frame argmax ---


def reduce_per_frame_argmax(
    *,
    frame_0_argmax_map: Sequence[int] | object | None,
    frame_1_argmax_map: Sequence[int] | object,
    num_classes: int = NUM_SEGNET_CLASSES_DEFAULT,
) -> int:
    """Reduce a per-pair argmax PAIR to a 2-tuple fingerprint integer.

    Algorithm:
      1. Compute per-frame argmax dominant class for frame_0 and frame_1
         (mode of pixel-class).
      2. Encode as ``f0_class * num_classes + f1_class`` (in [0, num_classes^2)).

    For substrates whose renderer outputs ONLY frame_1 (HNeRV-style A1 / Tishby
    IB-pure), pass ``frame_0_argmax_map=None`` and we'll use ``f1_class * num_classes + f1_class``
    so the fingerprint degenerates to the diagonal of the 2-tuple space - still
    quantitatively comparable across substrates.

    For substrates whose renderer outputs BOTH frames (HNeRVDecoder pair
    output) or substrates probing the source GT video pair-by-pair (Wunderkind
    G1 v2 against ``upstream/videos/0.mkv``), pass both frame argmax maps.

    Raises ``ValueError`` on contract violations.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes={num_classes} must be >= 2")
    flat_f1: list[int] = list(frame_1_argmax_map)
    if len(flat_f1) == 0:
        raise ValueError("frame_1_argmax_map is empty")
    f1_counts = Counter(int(c) for c in flat_f1)
    f1_class = max(f1_counts.items(), key=lambda kv: kv[1])[0]
    if not 0 <= f1_class < num_classes:
        raise ValueError(f"frame_1 dominant class {f1_class} out of range")
    if frame_0_argmax_map is None:
        f0_class = f1_class
    else:
        flat_f0: list[int] = list(frame_0_argmax_map)
        if len(flat_f0) == 0:
            raise ValueError("frame_0_argmax_map is non-None but empty")
        f0_counts = Counter(int(c) for c in flat_f0)
        f0_class = max(f0_counts.items(), key=lambda kv: kv[1])[0]
        if not 0 <= f0_class < num_classes:
            raise ValueError(f"frame_0 dominant class {f0_class} out of range")
    return f0_class * num_classes + f1_class


# --- Unified probe orchestrator ---


def compute_alternative_reducer_verdict(
    *,
    substrate_id: str,
    reducer_name: ReducerName,
    latent_stream: bytes | list[int],
    per_pair_reduced_class: Sequence[int],
    symbols_per_pair: int,
    meaningful_mi_threshold_bits: float | None = None,
    independence_tolerance_bits: float = INDEPENDENCE_TOLERANCE_BITS,
    reducer_specific_parameters: dict[str, int | float] | None = None,
    notes: str = "",
) -> AlternativeReducerVerdict:
    """Compute the conditional-entropy verdict for one alternative reducer.

    The latent_stream is the substrate's latent symbol stream (uint8 or list of
    ints); per_pair_reduced_class is the OUTPUT of the chosen reducer applied
    to each of N pairs (one int per pair). symbols_per_pair is the byte-expansion
    factor required to align the per-pair class with the per-symbol latent
    stream (e.g. 28 for Wunderkind G1 v2's residual_int8 stream).

    Raises ``ValueError`` on contract violations.
    """
    if reducer_name not in REDUCER_MEANINGFUL_THRESHOLD_BITS:
        raise ValueError(
            f"reducer_name={reducer_name!r} not in canonical reducer set "
            f"{set(REDUCER_MEANINGFUL_THRESHOLD_BITS.keys())}"
        )
    if not isinstance(substrate_id, str) or not substrate_id.strip():
        raise ValueError("substrate_id must be a non-empty string")
    if symbols_per_pair < 1:
        raise ValueError(f"symbols_per_pair={symbols_per_pair} must be >= 1")
    if len(per_pair_reduced_class) == 0:
        raise ValueError("per_pair_reduced_class is empty")
    threshold = (
        meaningful_mi_threshold_bits
        if meaningful_mi_threshold_bits is not None
        else REDUCER_MEANINGFUL_THRESHOLD_BITS[reducer_name]
    )
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(
            f"meaningful_mi_threshold_bits={threshold} must be finite >= 0"
        )

    # Byte-expand the per-pair reduced class to align symbol-for-symbol with
    # the latent stream. Each per-pair class value is repeated symbols_per_pair
    # times. The fingerprint can exceed uint8 range - entropy estimator uses a
    # dict-keyed counter so any nonneg int works.
    expanded_class_stream: list[int] = []
    for c in per_pair_reduced_class:
        ci = int(c)
        if ci < 0:
            raise ValueError(f"per_pair_reduced_class value {ci} < 0 unsupported")
        expanded_class_stream.extend([ci] * symbols_per_pair)
    if len(expanded_class_stream) != len(latent_stream):
        raise ValueError(
            f"expanded_class_stream length {len(expanded_class_stream)} != "
            f"latent_stream length {len(latent_stream)}; check symbols_per_pair "
            f"({symbols_per_pair}) x num_pairs ({len(per_pair_reduced_class)})"
        )

    # Compute H(latent), H(latent | class), and MI using the canonical primitives.
    latent_counts = _count_symbols(latent_stream)
    class_counts = _count_symbols(expanded_class_stream)
    joint = _count_joint(latent_stream, expanded_class_stream)

    h_latent = _plug_in_entropy_bits(latent_counts)
    total = sum(class_counts.values())
    h_latent_given_class = 0.0
    for cls, per_class in joint.items():
        p_class = class_counts[cls] / total
        h_per_class = _plug_in_entropy_bits(per_class)
        h_latent_given_class += p_class * h_per_class
    mutual_information = max(h_latent - h_latent_given_class, 0.0)
    wyner_ziv_gain_ceiling_fraction = (
        mutual_information / h_latent if h_latent > 0.0 else 0.0
    )

    # Verdict band per the reducer-specific threshold.
    if mutual_information <= independence_tolerance_bits:
        verdict = "INDEPENDENT"
    elif mutual_information >= threshold:
        verdict = "MEANINGFUL_CONDITIONING"
    else:
        verdict = "WEAK_CONDITIONING"

    return AlternativeReducerVerdict(
        substrate_id=substrate_id,
        reducer_name=reducer_name,
        verdict=verdict,
        h_latent_unconditional_bits_per_symbol=float(h_latent),
        h_latent_given_reduced_class_bits_per_symbol=float(h_latent_given_class),
        mutual_information_bits=float(mutual_information),
        wyner_ziv_gain_ceiling_fraction=float(wyner_ziv_gain_ceiling_fraction),
        num_latent_symbols=len(latent_stream),
        num_reduced_class_symbols=len(expanded_class_stream),
        num_unique_reduced_classes=len(class_counts),
        meaningful_mi_threshold_bits=float(threshold),
        independence_tolerance_bits=float(independence_tolerance_bits),
        reducer_specific_parameters=dict(reducer_specific_parameters or {}),
        evidence_grade="diagnostic_cpu",
        score_claim=False,
        axis_label="[diagnostic-CPU; alternative-reducer H(latent|reduced_class) probe]",
        observed_at_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        notes=notes,
    )


# --- Output dir helper ---


def make_timestamped_output_dir(
    repo_root: Path,
    prefix: str = "alternative_reducer_probes",
) -> Path:
    """Build a per-Catalog #249 device-agnostic timestamped output directory.

    Output directory naming uses a device-AGNOSTIC prefix (NOT ``_cuda``/``_cpu``)
    per Catalog #249 phantom-score-directory anti-pattern. The directory's actual
    contents document the device + axis explicitly.
    """
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = repo_root / "experiments" / "results" / f"{prefix}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_verdict_json(out_dir: Path, verdict: AlternativeReducerVerdict) -> Path:
    """Persist verdict JSON to the output directory.

    Filename includes substrate_id and reducer_name so multiple verdicts per
    output dir do not collide.
    """
    fn = f"alternative_reducer_verdict_{verdict.substrate_id}_{verdict.reducer_name}.json"
    path = out_dir / fn
    path.write_text(json.dumps(asdict(verdict), sort_keys=True, indent=2) + "\n")
    return path


def write_run_manifest(
    out_dir: Path,
    *,
    substrate_id: str,
    verdicts: Iterable[AlternativeReducerVerdict],
    provenance: dict,
) -> Path:
    """Persist a single per-substrate run manifest aggregating 4 reducer verdicts.

    The manifest is the operator-facing summary; per-reducer verdict JSONs are
    the per-result evidence.
    """
    summary = {
        "substrate_id": substrate_id,
        "axis_label": "[diagnostic-CPU; alternative-reducer probe wave per T2 council Q1.4]",
        "evidence_grade": "diagnostic_cpu",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            "alternative_reducer_probe_not_score_claim",
            "requires_separate_auth_eval_result_review_before_score_claim",
        ],
        "reducers": [
            {
                "reducer_name": v.reducer_name,
                "verdict": v.verdict,
                "mutual_information_bits": v.mutual_information_bits,
                "meaningful_mi_threshold_bits": v.meaningful_mi_threshold_bits,
                "wyner_ziv_gain_ceiling_fraction": v.wyner_ziv_gain_ceiling_fraction,
                "num_unique_reduced_classes": v.num_unique_reduced_classes,
                "reducer_specific_parameters": v.reducer_specific_parameters,
            }
            for v in verdicts
        ],
        "any_reducer_meaningful": any(
            v.verdict == "MEANINGFUL_CONDITIONING" for v in verdicts
        ),
        "any_reducer_weak": any(
            v.verdict == "WEAK_CONDITIONING" for v in verdicts
        ),
        "all_reducers_independent": all(
            v.verdict == "INDEPENDENT" for v in verdicts
        ),
        "recommended_phase_2_council_action": _recommend_action_for_verdicts(
            verdicts
        ),
        "provenance": provenance,
        "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
    }
    path = out_dir / f"alternative_reducer_run_manifest_{substrate_id}.json"
    path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return path


def _recommend_action_for_verdicts(verdicts: Iterable[AlternativeReducerVerdict]) -> str:
    """Per T2 council Q1.4: any meaningful -> RECLASSIFY-paradigm-class-REACTIVATED;
    all independent -> CONFIRM-DEFER-paradigm-class-pending-NEW-reducer-methodology.
    Weak conditioning is reported but does not satisfy reactivation.
    """
    verdict_list = list(verdicts)
    meaningful = [v for v in verdict_list if v.verdict == "MEANINGFUL_CONDITIONING"]
    weak = [v for v in verdict_list if v.verdict == "WEAK_CONDITIONING"]
    if meaningful:
        names = ", ".join(v.reducer_name for v in meaningful)
        return (
            f"RECLASSIFY: at least one alternative reducer ({names}) returned "
            "MEANINGFUL_CONDITIONING; v2 paradigm class is REACTIVATED with the "
            "specific reducer(s) that passed. Operator decision: confirm v2 lane "
            "registry update + Phase 2 wire-grammar redesign for the passing reducer."
        )
    if weak:
        names = ", ".join(v.reducer_name for v in weak)
        return (
            f"PARTIAL: alternative reducer(s) ({names}) returned WEAK_CONDITIONING; "
            "MI > tolerance but < meaningful threshold. Paradigm class is "
            "DEFERRED-pending-tighter-reducer-design (Phase 2 council Q1.4 #5)."
        )
    return (
        "CONFIRM-DEFER: ALL alternative reducers returned INDEPENDENT. The "
        "SegNet-derived cooperative-receiver paradigm IS deferred class-wide "
        "(still not KILLED per Catalog #103). Reactivation persists until a NEW "
        "reducer methodology is proposed."
    )


def sha256_file(path: Path) -> str:
    """Canonical sha256 file hasher used by the probe drivers."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Public surface
__all__ = [
    "CLASS_2_INDEX_DEFAULT",
    "NUM_SEGNET_CLASSES_DEFAULT",
    "PER_PAIR_CLASS_2_FRACTION_BUCKETS_DEFAULT",
    "PER_PIXEL_HISTOGRAM_BIN_QUANT_DEFAULT",
    "PER_REGION_HISTOGRAM_BIN_QUANT_DEFAULT",
    "REDUCER_MEANINGFUL_THRESHOLD_BITS",
    "AlternativeReducerVerdict",
    "compute_alternative_reducer_verdict",
    "make_timestamped_output_dir",
    "reduce_per_frame_argmax",
    "reduce_per_pair_class_2_fraction",
    "reduce_per_pixel_histogram",
    "reduce_per_region_histogram",
    "sha256_file",
    "write_run_manifest",
    "write_verdict_json",
]
