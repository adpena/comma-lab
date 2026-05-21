#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""STC residual sidecar over Selfcomp tone-map-delta cover signal — Path 3b probe.

Per OVERNIGHT-Y landing memo `fb58689cb` MEDIUM verdict + Path A recommendation
(`.omx/research/probe_stc_3a_a1_residual_entropy_built_and_run_landed_20260521.md`).

Sister of `tools/probe_stc_3a_a1_residual_entropy.py` (the A1-residual probe).
Y empirically falsified STC paradigm at A1 operating point: A1 residual entropy
7.778 bits/symbol HIGH + sparsity 5.93% LOW → MEDIUM verdict, "wrong substrate"
attribution (A1 residual cover signal does NOT exhibit STC's required sparse-
low-magnitude structure). Y MEDIUM verdict recommendation: build sister probe
testing DIFFERENT cover signal — Selfcomp PR #56 grayscale-LUT paradigm
tone-map-delta over Selfcomp soft-grayscale baseline.

Purpose
=======

The CARGO-CULTED assumption surfaced by Y's MEDIUM verdict is whether the STC
paradigm applies to ANY cover signal in our problem space or only specific
cover signals with structurally compatible distributions. Per CLAUDE.md
"Selfcomp / szabolcs-cs" + "Quantizr intelligence" sections, Selfcomp's PR #56
grayscale-LUT analog mask paradigm computes per-pair tone-map-delta residuals
(delta between LUT-quantized soft-grayscale vs original soft-grayscale) which
MAY exhibit STC's required sparse-low-magnitude structure because:

  1. Soft-grayscale (Y = 0.299R + 0.587G + 0.114B) preserves luminance edge
     structure that quantization-to-LUT-levels naturally rounds toward
  2. LUT quantization at low bit depths (2-4 bits = 4-16 levels) produces
     residuals concentrated near zero for in-distribution video pixels
  3. The tone-map-delta is a 1D scalar-per-pixel signal (vs A1's 3-channel
     RGB residual) which more naturally fits STC's bit-plane decomposition

This probe measures the ACTUAL Selfcomp tone-map-delta distribution on
ground-truth frames from `upstream/videos/0.mkv` by:

  1. Decoding ground-truth frames from upstream video via PyAV
  2. Computing soft-grayscale: Y = 0.299*R + 0.587*G + 0.114*B (BT.601)
  3. Quantizing soft-grayscale to LUT levels (2^lut_bits values)
  4. Computing tone-map-delta = original_grayscale - LUT_quantized_grayscale
  5. Computing Shannon entropy H(R) over tone-map-delta histogram per symbol
  6. Computing 5-tuple sparsity (|delta| <= 2 ratio)
  7. Applying canonical equation #359-sister IN-DOMAIN rate-axis closed-form
  8. Classifying verdict tier HIGH/MEDIUM/LOW per OVERNIGHT-W §9 thresholds

Verdict tiers (mirroring Y's per OVERNIGHT-W §9)
================================================

* HIGH (entropy >= 2.5 bits/symbol AND sparsity >= 0.40) → unlock $5.20 paid
  Modal smoke for STC residual sidecar over Selfcomp base
* MEDIUM (1.5 <= entropy < 2.5 OR 0.20 <= sparsity < 0.40) → sister probe 3c
  (next cover signal candidate; e.g. wavelet coefficients, fec6 frame-exploit
  selector, PR101 grammar bytes, or canonical equation #359-sister IN-DOMAIN
  alternative)
* LOW (entropy < 1.5 AND sparsity < 0.20) → STC paradigm IMPLEMENTATION-LEVEL
  falsified at BOTH A1 + Selfcomp; operator-routable for paradigm-level
  reconsideration per Catalog #307 paradigm-vs-implementation classification +
  Catalog #308 alternative-probe-methodologies (N>=3 enumerated)

Discipline checklist
====================

* Catalog #229 PV: ground-truth video sha verified empirically pre-decode
* Catalog #287 evidence-tag: every claim carries `[macOS-CPU advisory]` /
  `[prediction]` tags; NEVER `[contest-CUDA]` / `[contest-CPU]`
* Catalog #313 probe-outcomes ledger registration via canonical helper
* Catalog #323 canonical Provenance umbrella for verdict + JSON report
* Catalog #344 canonical equation reference: `procedural_predictor_plus_residual_correction_savings_v1`
  IN-DOMAIN via context `stc_predictor_plus_residual_selfcomp_tone_map_delta_per_pair_correction`
* Catalog #110/#113 APPEND-ONLY: probe NEVER mutates Selfcomp substrate; reads only
* HNeRV parity L7: ≤350 LOC tool probe (not a substrate)
* Catalog #192 macOS-CPU advisory: probe verdict drives DISPATCH-DECISION not
  SCORE-DECISION

Usage
=====

    .venv/bin/python tools/probe_stc_3b_selfcomp_tone_map_delta_entropy.py \\
        --ground-truth-video upstream/videos/0.mkv \\
        --sample-pairs 16 \\
        --lut-bits 4 \\
        --output-report-json .omx/state/wyner_ziv_deliverability/stc_3b_selfcomp_tone_map_delta_probe_<utc>.json \\
        --verbose

For CI / test fixture use:

    .venv/bin/python tools/probe_stc_3b_selfcomp_tone_map_delta_entropy.py \\
        --synthetic-test-mode \\
        --verbose
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# OVERNIGHT-W §9 verdict tier thresholds (canonical; shared with probe 3a per Y)
HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL = 2.5
MEDIUM_ENTROPY_THRESHOLD_BITS_PER_SYMBOL = 1.5
HIGH_SPARSITY_THRESHOLD = 0.40
MEDIUM_SPARSITY_THRESHOLD = 0.20

# Canonical STC sidecar byte budget (per OVERNIGHT-W §1.2; shared with 3a)
STC_PREDICTOR_SEED_BYTES = 32
STC_RESIDUAL_STREAM_BYTES = 375

# Ground-truth video constants
N_PAIRS = 600
CAMERA_H = 874
CAMERA_W = 1164

# 5-tuple sparsity threshold (per OVERNIGHT-W §9 spec; tone-map-delta magnitude <= 2)
SPARSITY_RESIDUAL_THRESHOLD = 2

# Default Selfcomp LUT bit depth per PR #56 paradigm
DEFAULT_LUT_BITS = 4

# Cover signal type (canonical token for probe-outcomes ledger + Provenance)
COVER_SIGNAL_TYPE = "selfcomp_tone_map_delta"

# BT.601 soft-grayscale coefficients
GRAYSCALE_R_COEFF = 0.299
GRAYSCALE_G_COEFF = 0.587
GRAYSCALE_B_COEFF = 0.114


@dataclass(frozen=True)
class STCToneMapDeltaProbeVerdict:
    """Canonical verdict per OVERNIGHT-W §9 + Catalog #323 canonical Provenance.

    All score-bearing fields stamped non-promotable (score_claim=False;
    promotable=False) per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192
    macOS-CPU advisory axis.
    """

    tone_map_delta_entropy_bits_per_symbol: float
    five_tuple_sparsity_ratio: float
    predicted_delta_s_rate_only: float
    predicted_delta_s_band: tuple[float, float]
    verdict_tier: str  # HIGH / MEDIUM / LOW
    canonical_equation_id: str
    verdict_rationale: str
    sample_pairs_decoded: int
    sample_pairs_total_residuals: int
    cover_signal_type: str
    lut_bits: int
    ground_truth_video_sha256: str
    score_claim: bool = False
    promotable: bool = False
    axis_tag: str = "[macOS-CPU advisory]"
    evidence_grade: str = "macOS-CPU-advisory"

    def __post_init__(self) -> None:
        if self.verdict_tier not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError(
                f"verdict_tier must be HIGH/MEDIUM/LOW, got {self.verdict_tier!r}"
            )
        if not 0.0 <= self.five_tuple_sparsity_ratio <= 1.0:
            raise ValueError(
                f"five_tuple_sparsity_ratio must be in [0,1], got {self.five_tuple_sparsity_ratio}"
            )
        if self.score_claim is not False:
            raise ValueError("score_claim must be False per Catalog #192 macOS-CPU advisory")
        if self.promotable is not False:
            raise ValueError("promotable must be False per Catalog #192 macOS-CPU advisory")
        if self.cover_signal_type != COVER_SIGNAL_TYPE:
            raise ValueError(
                f"cover_signal_type must be {COVER_SIGNAL_TYPE!r}, got {self.cover_signal_type!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tone_map_delta_entropy_bits_per_symbol": float(self.tone_map_delta_entropy_bits_per_symbol),
            "five_tuple_sparsity_ratio": float(self.five_tuple_sparsity_ratio),
            "predicted_delta_s_rate_only": float(self.predicted_delta_s_rate_only),
            "predicted_delta_s_band": list(self.predicted_delta_s_band),
            "verdict_tier": self.verdict_tier,
            "canonical_equation_id": self.canonical_equation_id,
            "verdict_rationale": self.verdict_rationale,
            "sample_pairs_decoded": int(self.sample_pairs_decoded),
            "sample_pairs_total_residuals": int(self.sample_pairs_total_residuals),
            "cover_signal_type": self.cover_signal_type,
            "lut_bits": int(self.lut_bits),
            "ground_truth_video_sha256": self.ground_truth_video_sha256,
            "score_claim": self.score_claim,
            "promotable": self.promotable,
            "axis_tag": self.axis_tag,
            "evidence_grade": self.evidence_grade,
        }


def soft_grayscale_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Compute BT.601 soft-grayscale Y = 0.299R + 0.587G + 0.114B.

    Input rgb: shape (..., 3) uint8 or float; output float32 in [0, 255].
    """

    rgb_f = rgb.astype(np.float32)
    y = (
        GRAYSCALE_R_COEFF * rgb_f[..., 0]
        + GRAYSCALE_G_COEFF * rgb_f[..., 1]
        + GRAYSCALE_B_COEFF * rgb_f[..., 2]
    )
    return y


def lut_quantize_grayscale(grayscale: np.ndarray, lut_bits: int) -> np.ndarray:
    """Quantize soft-grayscale to LUT levels per PR #56 paradigm.

    LUT has 2^lut_bits levels evenly spaced over [0, 255]. Returns
    quantized grayscale (float32) at the same shape as input.

    Per Selfcomp PR #56: lut_bits = 4 (16 levels) is canonical default;
    lut_bits = 2 (4 levels) is aggressive; lut_bits = 8 (256 levels) is
    no-op (identity for uint8 input).
    """

    if lut_bits < 1 or lut_bits > 8:
        raise ValueError(f"lut_bits must be in [1, 8], got {lut_bits}")
    n_levels = 1 << lut_bits  # 2^lut_bits
    step = 255.0 / (n_levels - 1) if n_levels > 1 else 255.0
    # Round to nearest LUT level
    indices = np.round(grayscale / step)
    indices = np.clip(indices, 0, n_levels - 1)
    return (indices * step).astype(np.float32)


def shannon_entropy_bits_per_symbol(residuals: np.ndarray) -> float:
    """Compute Shannon entropy H(R) = -Σ p(r) log₂(p(r)) over residual symbols.

    Residuals are quantized into int16 buckets covering [-255, +255] (grayscale
    residual range). Buckets with p=0 contribute 0 to the sum per the
    convention 0·log(0) = 0.
    """

    if residuals.size == 0:
        return 0.0
    quantized = np.clip(np.round(residuals).astype(np.int32), -255, 255)
    counts = np.bincount(quantized + 255, minlength=511).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts / total
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def five_tuple_sparsity(residuals: np.ndarray, threshold: int = SPARSITY_RESIDUAL_THRESHOLD) -> float:
    """Fraction of residual values with |r| <= threshold.

    Per OVERNIGHT-W §9: sparsity > 40% indicates a temporal/spatial predictor
    captures most of the signal (i.e., residuals concentrate near zero), which
    is structurally compatible with STC + Dasher AC's exploitable rate-axis.
    """

    if residuals.size == 0:
        return 0.0
    in_band = int(np.sum(np.abs(residuals) <= threshold))
    return float(in_band) / float(residuals.size)


def compute_predicted_delta_s_band(
    *,
    residual_entropy: float,
    sparsity: float,
) -> tuple[float, float]:
    """Compute predicted ΔS band per OVERNIGHT-W §10 tightened band.

    Returns (lower_bound, upper_bound) signed band:
    * HIGH-EV (entropy >= 2.5 + sparsity >= 0.40): [-0.005, +0.001]
    * MEDIUM-EV: [-0.001, +0.001]
    * LOW-EV: [+0.000, +0.001] (rate penalty only; no distortion offset)

    Rate-axis component is always +0.000271 per canonical equation #359-sister.
    """

    rate_penalty = 0.00027100459392072373
    if residual_entropy >= HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL and sparsity >= HIGH_SPARSITY_THRESHOLD:
        return (-0.005, +0.001)
    elif residual_entropy >= MEDIUM_ENTROPY_THRESHOLD_BITS_PER_SYMBOL or sparsity >= MEDIUM_SPARSITY_THRESHOLD:
        return (-0.001, +0.001)
    else:
        return (rate_penalty, rate_penalty + 0.0005)


def classify_verdict_tier(
    *,
    residual_entropy: float,
    sparsity: float,
) -> tuple[str, str]:
    """Classify HIGH/MEDIUM/LOW per OVERNIGHT-W §9 thresholds.

    Returns (tier, rationale).
    """

    if (
        residual_entropy >= HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL
        and sparsity >= HIGH_SPARSITY_THRESHOLD
    ):
        return (
            "HIGH",
            f"entropy={residual_entropy:.3f} >= {HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL} "
            f"AND sparsity={sparsity:.3f} >= {HIGH_SPARSITY_THRESHOLD} "
            f"→ HIGH-ENTROPY-RESIDUAL-PRESENT on Selfcomp tone-map-delta cover signal; "
            f"unlock $5.20 paid Modal smoke per OVERNIGHT-Y MEDIUM verdict cascade",
        )
    elif (
        residual_entropy >= MEDIUM_ENTROPY_THRESHOLD_BITS_PER_SYMBOL
        or sparsity >= MEDIUM_SPARSITY_THRESHOLD
    ):
        return (
            "MEDIUM",
            f"entropy={residual_entropy:.3f}, sparsity={sparsity:.3f}; "
            f"at least one signal meets MEDIUM threshold "
            f"(entropy>={MEDIUM_ENTROPY_THRESHOLD_BITS_PER_SYMBOL} OR "
            f"sparsity>={MEDIUM_SPARSITY_THRESHOLD}) but NOT both at HIGH threshold "
            f"→ MEDIUM-ENTROPY on Selfcomp tone-map-delta; "
            f"defer to sister probe 3c (next cover signal candidate per "
            f"Catalog #308 alternative-probe-methodologies)",
        )
    else:
        return (
            "LOW",
            f"entropy={residual_entropy:.3f} < {MEDIUM_ENTROPY_THRESHOLD_BITS_PER_SYMBOL} "
            f"AND sparsity={sparsity:.3f} < {MEDIUM_SPARSITY_THRESHOLD} "
            f"→ LOW-ENTROPY-RESIDUAL-ABSENT on Selfcomp tone-map-delta cover signal; "
            f"STC paradigm IMPLEMENTATION-LEVEL falsified at BOTH A1 + Selfcomp "
            f"(Y probe 3a reported MEDIUM on A1; 3b on Selfcomp now reports LOW) "
            f"→ operator-routable for paradigm-level reconsideration per Catalog #307 "
            f"paradigm-vs-implementation classification + Catalog #308 "
            f"alternative-probe-methodologies (N>=3 enumerated)",
        )


def verify_ground_truth_video_sha256(video_path: Path) -> str:
    """Compute ground-truth video sha256 for provenance.

    Returns the sha hex string. Does NOT raise on missing/mismatch — that's
    the caller's responsibility (synthetic test mode bypasses verification).
    """

    if not video_path.exists():
        return "0" * 64  # sentinel
    sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
    return sha


def extract_selfcomp_tone_map_delta_residuals(
    *,
    video_path: Path,
    sample_pairs: int,
    lut_bits: int,
    verbose: bool = False,
) -> np.ndarray:
    """Decode ground-truth video frames; return per-pair tone-map-delta residuals.

    Returns a 1D numpy int16 array of tone-map-delta residual values across
    sampled pairs. Each pair contributes 2 frames × CAMERA_H × CAMERA_W scalar
    tone-map-delta values.

    Per CLAUDE.md Selfcomp / szabolcs-cs PR #56 paradigm:
    1. RGB → soft-grayscale (BT.601: Y = 0.299R + 0.587G + 0.114B)
    2. soft-grayscale → LUT-quantized grayscale (2^lut_bits levels)
    3. tone-map-delta = soft-grayscale - LUT-quantized

    The tone-map-delta IS the residual that an STC sidecar would encode.
    """

    try:
        import av
    except ImportError as exc:
        raise ImportError(
            f"PyAV required for actual ground-truth decode; got {exc}. "
            "Use --synthetic-test-mode for fixture/CI runs."
        ) from exc

    # Determine which frame indices we need (pair i = frames 2i and 2i+1)
    sample_indices = list(range(0, min(sample_pairs, N_PAIRS)))
    gt_frames_needed = sorted({2 * pi for pi in sample_indices} | {2 * pi + 1 for pi in sample_indices})
    max_frame_needed = max(gt_frames_needed) + 1

    if verbose:
        print(f"[probe] reading {len(gt_frames_needed)} ground-truth frames "
              f"from {video_path} (max_frame={max_frame_needed})...")

    gt_frames_map: dict[int, np.ndarray] = {}
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx >= max_frame_needed:
                break
            if frame_idx in gt_frames_needed:
                img = frame.to_ndarray(format="rgb24")
                gt_frames_map[frame_idx] = img
    finally:
        container.close()

    if verbose:
        print(f"[probe] decoded {len(gt_frames_map)} frames; computing "
              f"soft-grayscale + LUT-quantize (lut_bits={lut_bits}) + "
              f"tone-map-delta per frame...")

    # Compute tone-map-delta per frame across sampled pairs
    residuals_chunks = []
    for pair_idx in sample_indices:
        for frame_idx in (2 * pair_idx, 2 * pair_idx + 1):
            rgb = gt_frames_map[frame_idx]
            grayscale = soft_grayscale_from_rgb(rgb)
            lut_quantized = lut_quantize_grayscale(grayscale, lut_bits)
            delta = grayscale - lut_quantized
            # Round to int16 for entropy bucket compatibility with 3a
            residuals_chunks.append(np.round(delta).astype(np.int16).ravel())

    residuals = np.concatenate(residuals_chunks)
    if verbose:
        print(f"[probe] computed {residuals.size} tone-map-delta residual values "
              f"across {len(sample_indices)} pairs ({len(sample_indices) * 2} frames)")
    return residuals


def synthetic_tone_map_delta_residuals_for_test(
    *,
    pattern: str = "high",
    n_symbols: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic tone-map-delta residuals for test/fixture mode.

    Patterns mirror probe 3a's synthetic patterns. The tone-map-delta
    distribution should structurally favor low-magnitude clustered residuals
    (residual lives in [-step/2, +step/2] where step = 255 / (2^lut_bits - 1)).

    Patterns:
    * "high": gaussian σ=8 (high-entropy + low sparsity)
    * "medium": gaussian σ=3 (medium entropy + medium sparsity)
    * "low": all zeros + small ε (low entropy + high sparsity)
    * "uniform_random": uniform [-127, 127] (highest entropy)
    * "compressible_lut": uniform [-step/2, +step/2] where step = 255/15 = 17
      (structurally accurate for lut_bits=4)
    """

    rng = np.random.default_rng(seed)
    if pattern == "high":
        return rng.normal(0, 8, n_symbols).astype(np.int16)
    elif pattern == "medium":
        return rng.normal(0, 3, n_symbols).astype(np.int16)
    elif pattern == "low":
        out = np.zeros(n_symbols, dtype=np.int16)
        spike_idx = rng.choice(n_symbols, size=n_symbols // 100, replace=False)
        out[spike_idx] = rng.integers(-1, 2, size=spike_idx.size)
        return out
    elif pattern == "uniform_random":
        return rng.integers(-127, 128, size=n_symbols).astype(np.int16)
    elif pattern == "compressible_lut":
        # Tone-map-delta naturally bounded to [-step/2, +step/2]
        # For lut_bits=4: step=17, half-step=8
        out = rng.integers(-8, 9, size=n_symbols).astype(np.int16)
        return out
    else:
        raise ValueError(f"unknown synthetic pattern: {pattern}")


def compute_stc_tone_map_delta_entropy(
    residuals: np.ndarray,
    *,
    ground_truth_video_sha256: str = "0" * 64,
    lut_bits: int = DEFAULT_LUT_BITS,
    sample_pairs_decoded: int = 0,
) -> STCToneMapDeltaProbeVerdict:
    """Canonical entry point: given residuals array, return verdict.

    Per Catalog #229 PV + Catalog #287 evidence-tag discipline + Catalog #323
    canonical Provenance: every score-bearing field carries non-promotable markers.

    Per OVERNIGHT-W §10 + Catalog #344: canonical equation #359-sister
    `procedural_predictor_plus_residual_correction_savings_v1` IN-DOMAIN via
    context `stc_predictor_plus_residual_selfcomp_tone_map_delta_per_pair_correction`.
    """

    from tac.canonical_equations.procedural_predictor_residual_savings import (
        predict_procedural_predictor_plus_residual_correction_savings,
    )

    entropy = shannon_entropy_bits_per_symbol(residuals)
    sparsity = five_tuple_sparsity(residuals)
    band = compute_predicted_delta_s_band(
        residual_entropy=entropy,
        sparsity=sparsity,
    )
    tier, rationale = classify_verdict_tier(
        residual_entropy=entropy,
        sparsity=sparsity,
    )

    eq_result = predict_procedural_predictor_plus_residual_correction_savings(
        original_payload_bytes=0,
        predictor_seed_or_code_bytes=STC_PREDICTOR_SEED_BYTES,
        residual_stream_bytes=STC_RESIDUAL_STREAM_BYTES,
        container_overhead_bytes=0,
        context="stc_predictor_plus_residual_selfcomp_tone_map_delta_per_pair_correction",
    )

    return STCToneMapDeltaProbeVerdict(
        tone_map_delta_entropy_bits_per_symbol=entropy,
        five_tuple_sparsity_ratio=sparsity,
        predicted_delta_s_rate_only=float(eq_result["predicted_delta_s_rate_only"]),
        predicted_delta_s_band=band,
        verdict_tier=tier,
        canonical_equation_id=eq_result["equation_id"],
        verdict_rationale=rationale,
        sample_pairs_decoded=sample_pairs_decoded,
        sample_pairs_total_residuals=int(residuals.size),
        cover_signal_type=COVER_SIGNAL_TYPE,
        lut_bits=lut_bits,
        ground_truth_video_sha256=ground_truth_video_sha256,
    )


def build_canonical_provenance_for_report(verdict: STCToneMapDeltaProbeVerdict) -> dict[str, Any]:
    """Build canonical Provenance per Catalog #323 for the verdict report."""

    from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256=(
            verdict.ground_truth_video_sha256
            if len(verdict.ground_truth_video_sha256) == 64
            else "0" * 64
        ),
        source_path=".omx/state/wyner_ziv_deliverability/stc_3b_selfcomp_tone_map_delta_probe.json",
    )
    if hasattr(prov, "to_dict"):
        return prov.to_dict()
    from dataclasses import asdict
    return asdict(prov)


def register_verdict_in_probe_outcomes_ledger(
    verdict: STCToneMapDeltaProbeVerdict,
    *,
    report_path: Path | None,
    subagent_id: str | None = None,
) -> dict[str, Any]:
    """Register the verdict in Catalog #313 probe-outcomes ledger.

    Verdict mapping per OVERNIGHT-Y MEDIUM verdict cascade + sister probe 3a:
    * HIGH → VERDICT_PROCEED (unlock $5.20 paid Modal smoke)
    * MEDIUM → VERDICT_DEFER (sister probe 3c next cover signal)
    * LOW → VERDICT_DEFER (paradigm-level reconsideration per Catalog #307+#308)
    """

    from tac.probe_outcomes_ledger import (
        VERDICT_DEFER,
        VERDICT_PROCEED,
        register_probe_outcome,
    )

    if verdict.verdict_tier == "HIGH":
        ledger_verdict = VERDICT_PROCEED
        next_action = (
            "operator-routable: unlock $5.20 paid Modal smoke for STC residual sidecar "
            "over Selfcomp base per OVERNIGHT-Y MEDIUM verdict cascade"
        )
    elif verdict.verdict_tier == "MEDIUM":
        ledger_verdict = VERDICT_DEFER
        next_action = (
            "operator-routable: build sister probe 3c (next cover signal candidate per "
            "Catalog #308 alternative-probe-methodologies)"
        )
    else:
        ledger_verdict = VERDICT_DEFER
        next_action = (
            "operator-routable: STC paradigm IMPLEMENTATION-LEVEL falsified at BOTH A1 + "
            "Selfcomp; paradigm-level reconsideration per Catalog #307 + #308 "
            "(N>=3 alternative-probe-methodologies enumerated)"
        )

    utc = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    probe_id = f"stc_3b_selfcomp_tone_map_delta_entropy_{utc.replace(':', '').replace('-', '')[:15]}"

    return register_probe_outcome(
        probe_id=probe_id,
        substrate="stc_paradigm_reformulation_selfcomp_tone_map_delta_path_3b",
        recipe_path=None,
        probe_kind="stc_3b_selfcomp_tone_map_delta_entropy",
        verdict=ledger_verdict,
        metric_name="tone_map_delta_entropy_bits_per_symbol",
        metric_value=float(verdict.tone_map_delta_entropy_bits_per_symbol),
        threshold=HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL,
        threshold_token=f"high_entropy_>={HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL}",
        evidence_path=str(report_path) if report_path else None,
        next_action=next_action,
        adjudicated_at_utc=utc,
        agent="claude",
        subagent_id=subagent_id,
        notes=verdict.verdict_rationale,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="STC residual sidecar over Selfcomp tone-map-delta — probe 3b",
    )
    parser.add_argument(
        "--ground-truth-video",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="Path to ground-truth video (default: upstream/videos/0.mkv)",
    )
    parser.add_argument(
        "--sample-pairs",
        type=int,
        default=16,
        help="Number of ground-truth pairs to decode for statistical sampling (default: 16)",
    )
    parser.add_argument(
        "--lut-bits",
        type=int,
        default=DEFAULT_LUT_BITS,
        help=f"Selfcomp LUT bit depth per PR #56 paradigm (default: {DEFAULT_LUT_BITS})",
    )
    parser.add_argument(
        "--output-report-json",
        type=Path,
        default=None,
        help="Output JSON report path (default: auto-named under .omx/state/wyner_ziv_deliverability/)",
    )
    parser.add_argument(
        "--synthetic-test-mode",
        action="store_true",
        help="Use synthetic residuals for test/fixture runs (skip video read)",
    )
    parser.add_argument(
        "--synthetic-pattern",
        type=str,
        default="compressible_lut",
        choices=("high", "medium", "low", "uniform_random", "compressible_lut"),
        help="Synthetic residual pattern for test mode (default: compressible_lut)",
    )
    parser.add_argument(
        "--skip-ledger-registration",
        action="store_true",
        help="Skip Catalog #313 probe-outcomes ledger registration (for test runs)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args(argv)

    if args.synthetic_test_mode:
        if args.verbose:
            print(f"[probe] synthetic test mode (pattern={args.synthetic_pattern}; "
                  f"lut_bits={args.lut_bits})")
        residuals = synthetic_tone_map_delta_residuals_for_test(pattern=args.synthetic_pattern)
        gt_sha = "synthetic_test_fixture"
        pairs_decoded = 0
    else:
        gt_sha = verify_ground_truth_video_sha256(args.ground_truth_video)
        if args.verbose:
            print(f"[probe] ground-truth video sha: {gt_sha[:16]}...")
        residuals = extract_selfcomp_tone_map_delta_residuals(
            video_path=args.ground_truth_video,
            sample_pairs=args.sample_pairs,
            lut_bits=args.lut_bits,
            verbose=args.verbose,
        )
        pairs_decoded = min(args.sample_pairs, N_PAIRS)

    verdict = compute_stc_tone_map_delta_entropy(
        residuals,
        ground_truth_video_sha256=gt_sha,
        lut_bits=args.lut_bits,
        sample_pairs_decoded=pairs_decoded,
    )

    utc = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    report = {
        "schema": "stc_3b_selfcomp_tone_map_delta_entropy_probe_v1",
        "generated_at_utc": utc,
        "lane_id": "lane_overnight_aa_stc_3b_selfcomp_tone_map_delta_entropy_probe_build_local_cpu_run_20260521",
        "canonical_equation_id": verdict.canonical_equation_id,
        "canonical_equation_context": "stc_predictor_plus_residual_selfcomp_tone_map_delta_per_pair_correction",
        "cover_signal_type": COVER_SIGNAL_TYPE,
        "verdict": verdict.as_dict(),
        "provenance": build_canonical_provenance_for_report(verdict),
        "next_action_per_verdict": {
            "HIGH": (
                "operator-routable: unlock $5.20 paid Modal smoke for STC residual "
                "sidecar over Selfcomp base per OVERNIGHT-Y MEDIUM cascade"
            ),
            "MEDIUM": (
                "operator-routable: build sister probe 3c (next cover signal candidate)"
            ),
            "LOW": (
                "operator-routable: STC paradigm IMPLEMENTATION-LEVEL falsified at "
                "BOTH A1 + Selfcomp; paradigm-level reconsideration per Catalog "
                "#307 + #308"
            ),
        }[verdict.verdict_tier],
    }

    if args.output_report_json is None and not args.synthetic_test_mode:
        report_dir = REPO_ROOT / ".omx" / "state" / "wyner_ziv_deliverability"
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = utc.replace(":", "").replace("-", "")[:15]
        args.output_report_json = report_dir / f"stc_3b_selfcomp_tone_map_delta_probe_{timestamp}.json"
    if args.output_report_json is not None:
        args.output_report_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_report_json.write_text(json.dumps(report, indent=2, sort_keys=True))
        if args.verbose:
            print(f"[probe] report written to {args.output_report_json}")

    if not args.skip_ledger_registration and not args.synthetic_test_mode:
        ledger_record = register_verdict_in_probe_outcomes_ledger(
            verdict,
            report_path=args.output_report_json,
            subagent_id=f"overnight_aa_stc_3b_probe_{utc.replace(':', '').replace('-', '')[:15]}",
        )
        if args.verbose:
            print(f"[probe] registered in ledger: probe_id={ledger_record.get('probe_id')}")

    print(f"\n=== STC 3b Selfcomp tone-map-delta entropy probe verdict ===")
    print(f"Tier: {verdict.verdict_tier}")
    print(f"Cover signal: {verdict.cover_signal_type} (lut_bits={verdict.lut_bits})")
    print(f"Shannon entropy: {verdict.tone_map_delta_entropy_bits_per_symbol:.4f} bits/symbol")
    print(f"5-tuple sparsity: {verdict.five_tuple_sparsity_ratio:.4f}")
    print(f"Predicted ΔS band: [{verdict.predicted_delta_s_band[0]:+.6f}, "
          f"{verdict.predicted_delta_s_band[1]:+.6f}] [prediction; NON-AUTHORITATIVE]")
    print(f"Rate-only ΔS (canonical eq #359-sister): "
          f"+{verdict.predicted_delta_s_rate_only:.6f}")
    print(f"Sample: {verdict.sample_pairs_decoded} pairs "
          f"({verdict.sample_pairs_total_residuals} residuals)")
    print(f"Axis: {verdict.axis_tag} (NOT contest-axis authoritative per Catalog #192)")
    print(f"\nRationale: {verdict.verdict_rationale}")
    print(f"\nNext action: {report['next_action_per_verdict']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
