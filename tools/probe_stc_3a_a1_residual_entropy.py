#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""STC residual sidecar over A1 — Path 3a per-pair RGB residual entropy probe.

Per OVERNIGHT-W DESIGN landing memo `b45598f2b` Path A op-routable #1
(`.omx/research/stc_residual_sidecar_over_a1_path_a_pivot_design_local_cpu_mvp_landed_20260521.md`)
+ Carmack MVP-first 5-step per CLAUDE.md `be125b878` amendment + 2026-05-17 sister
symposium op-routable #1 (`council_per_substrate_symposium_stc_3a_sidecar_a1_residual_20260517.md`).

Purpose
=======

The CARGO-CULTED assumption surfaced by OVERNIGHT-W Assumption-Adversary verdict 1 +
2026-05-17 + 2026-05-20 sister symposia is whether the A1 per-pair RGB
reconstruction residual (decoder output vs ground-truth pixel) carries the
1D + low-temporal-coherence structural properties STC + Dasher AC require for
exploitable rate-axis savings. Per Shannon's symposium #857 verdict verbatim:
"apply STC where the alternative codec is NOT spatially-correlated (a 1D symbol
stream or a sidecar residual with limited temporal coherence)".

This probe measures the ACTUAL A1 per-pair RGB residual distribution on archive
sha `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` by:

  1. Loading the A1 archive bytes from `submissions/a1/archive.zip` (verified sha)
  2. Decoding via the A1 inflate runtime to produce rendered RGB frames
  3. Decoding ground-truth frames from `upstream/videos/0.mkv` via PyAV
  4. Computing per-pair residual = ground_truth - rendered (signed int16)
  5. Computing Shannon entropy H(R) over residual histogram per byte/symbol
  6. Computing 5-tuple sparsity (|residual| < threshold ratio)
  7. Applying canonical equation #359-sister IN-DOMAIN rate-axis closed-form
  8. Classifying verdict tier HIGH/MEDIUM/LOW per OVERNIGHT-W §9 thresholds

Per Carmack MVP-first phasing: this probe runs on macOS CPU (M5 Max);
per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 the probe verdict
is `[macOS-CPU advisory]` axis NOT contest-axis authoritative. Per Catalog
#313 the verdict registers via `tac.probe_outcomes_ledger.register_probe_outcome`
as the canonical disambiguator for the DEFER-TO-PRE-PROBE verdict.

Verdict tiers (per OVERNIGHT-W §9)
==================================

* HIGH (entropy >= 2.5 bits/symbol AND five_tuple_sparsity >= 0.40 AND
  predicted ΔS band [-0.005, +0.001] non-empty) → unlock $5.20 paid Modal smoke
* MEDIUM (1.5 <= entropy < 2.5 OR 0.20 <= sparsity < 0.40 OR predicted band
  [-0.001, +0.001]) → defer to sister probe 3b
* LOW (entropy < 1.5 AND sparsity < 0.20) → DEFER substrate per Catalog #307
  paradigm-vs-implementation classification (substrate-class falsified for
  THIS A1-residual cover signal; paradigm intact for sister substrates per
  Catalog #308 alternative-probe-methodologies)

Discipline checklist
====================

* Catalog #229 PV: A1 archive sha verified empirically pre-decode
* Catalog #287 evidence-tag: every claim carries `[macOS-CPU advisory]` /
  `[prediction]` tags; NEVER `[contest-CUDA]` / `[contest-CPU]`
* Catalog #313 probe-outcomes ledger registration via canonical helper
* Catalog #323 canonical Provenance umbrella for verdict + JSON report
* Catalog #344 canonical equation reference: `procedural_predictor_plus_residual_correction_savings_v1`
  IN-DOMAIN via context `stc_predictor_plus_residual_a1_per_pair_correction`
* Catalog #110/#113 APPEND-ONLY: probe NEVER mutates A1 substrate; reads only
* HNeRV parity L7: ≤350 LOC tool probe (not a substrate)
* Catalog #192 macOS-CPU advisory: probe verdict drives DISPATCH-DECISION not
  SCORE-DECISION

Usage
=====

    .venv/bin/python tools/probe_stc_3a_a1_residual_entropy.py \\
        --a1-archive-path submissions/a1/archive.zip \\
        --ground-truth-video upstream/videos/0.mkv \\
        --sample-pairs 16 \\
        --output-report-json .omx/state/wyner_ziv_deliverability/stc_3a_a1_residual_probe_<utc>.json \\
        --verbose

For CI / test fixture use:

    .venv/bin/python tools/probe_stc_3a_a1_residual_entropy.py \\
        --synthetic-test-mode \\
        --verbose
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical A1 archive sha (per OVERNIGHT-W §1.2 + submissions/a1/archive_manifest.json)
CANONICAL_A1_ARCHIVE_SHA256 = (
    "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
)
CANONICAL_A1_ARCHIVE_BYTES = 178_162

# OVERNIGHT-W §9 verdict tier thresholds (canonical)
HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL = 2.5
MEDIUM_ENTROPY_THRESHOLD_BITS_PER_SYMBOL = 1.5
HIGH_SPARSITY_THRESHOLD = 0.40
MEDIUM_SPARSITY_THRESHOLD = 0.20

# Canonical STC sidecar byte budget (per OVERNIGHT-W §1.2)
STC_PREDICTOR_SEED_BYTES = 32
STC_RESIDUAL_STREAM_BYTES = 375  # ≤375 per OVERNIGHT-W §1.2 spec

# Ground-truth video constants
N_PAIRS = 600
CAMERA_H = 874
CAMERA_W = 1164

# 5-tuple sparsity threshold (per OVERNIGHT-W §9 spec; residual magnitude <= 2)
SPARSITY_RESIDUAL_THRESHOLD = 2


@dataclass(frozen=True)
class STCResidualEntropyProbeVerdict:
    """Canonical verdict per OVERNIGHT-W §9 + Catalog #323 canonical Provenance.

    All score-bearing fields stamped non-promotable (score_claim=False;
    promotable=False) per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192
    macOS-CPU advisory axis.
    """

    residual_entropy_bits_per_symbol: float
    five_tuple_sparsity_ratio: float
    predicted_delta_s_rate_only: float
    predicted_delta_s_band: tuple[float, float]
    verdict_tier: str  # HIGH / MEDIUM / LOW
    canonical_equation_id: str
    verdict_rationale: str
    sample_pairs_decoded: int
    sample_pairs_total_residuals: int
    a1_archive_sha256: str
    a1_archive_bytes: int
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "residual_entropy_bits_per_symbol": float(self.residual_entropy_bits_per_symbol),
            "five_tuple_sparsity_ratio": float(self.five_tuple_sparsity_ratio),
            "predicted_delta_s_rate_only": float(self.predicted_delta_s_rate_only),
            "predicted_delta_s_band": list(self.predicted_delta_s_band),
            "verdict_tier": self.verdict_tier,
            "canonical_equation_id": self.canonical_equation_id,
            "verdict_rationale": self.verdict_rationale,
            "sample_pairs_decoded": int(self.sample_pairs_decoded),
            "sample_pairs_total_residuals": int(self.sample_pairs_total_residuals),
            "a1_archive_sha256": self.a1_archive_sha256,
            "a1_archive_bytes": int(self.a1_archive_bytes),
            "score_claim": self.score_claim,
            "promotable": self.promotable,
            "axis_tag": self.axis_tag,
            "evidence_grade": self.evidence_grade,
        }


def shannon_entropy_bits_per_symbol(residuals: np.ndarray) -> float:
    """Compute Shannon entropy H(R) = -Σ p(r) log₂(p(r)) over residual symbols.

    Residuals are quantized into int16 buckets covering [-255, +255] (RGB
    residual range). Buckets with p=0 contribute 0 to the sum per the
    convention 0·log(0) = 0.
    """

    if residuals.size == 0:
        return 0.0
    quantized = np.clip(residuals.astype(np.int32), -255, 255)
    # 511 buckets total ([-255..+255] inclusive)
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

    rate_penalty = 0.00027100459392072373  # canonical eq #359-sister at 32+375 bytes
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
            f"→ HIGH-ENTROPY-RESIDUAL-PRESENT; unlock $5.20 paid Modal smoke "
            f"per OVERNIGHT-W §6 cascade gate 1",
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
            f"(entropy>={HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL} AND "
            f"sparsity>={HIGH_SPARSITY_THRESHOLD}) "
            f"→ MEDIUM-ENTROPY; defer to sister probe 3b (Selfcomp tone-map-delta) "
            f"per OVERNIGHT-W §9 Path A op-routable #1 verdict MEDIUM branch",
        )
    else:
        return (
            "LOW",
            f"entropy={residual_entropy:.3f} < {MEDIUM_ENTROPY_THRESHOLD_BITS_PER_SYMBOL} "
            f"AND sparsity={sparsity:.3f} < {MEDIUM_SPARSITY_THRESHOLD} "
            f"→ LOW-ENTROPY-RESIDUAL-ABSENT; DEFER substrate per Catalog #307 "
            f"paradigm-vs-implementation classification (substrate-class falsified "
            f"for THIS A1-residual cover signal; paradigm INTACT for sister substrates "
            f"per Catalog #308 alternative-probe-methodologies)",
        )


def verify_a1_archive_sha256(archive_path: Path) -> str:
    """Verify A1 archive sha256 matches canonical value.

    Raises ValueError if mismatch. Returns the verified sha hex string.
    """

    if not archive_path.exists():
        raise FileNotFoundError(f"A1 archive not found at {archive_path}")
    sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if sha != CANONICAL_A1_ARCHIVE_SHA256:
        raise ValueError(
            f"A1 archive sha mismatch: expected {CANONICAL_A1_ARCHIVE_SHA256} "
            f"got {sha} at {archive_path}"
        )
    return sha


def extract_a1_residuals_from_decode(
    *,
    archive_path: Path,
    video_path: Path,
    sample_pairs: int,
    verbose: bool = False,
) -> np.ndarray:
    """Decode A1 archive + ground-truth frames; return per-pair RGB residual array.

    Returns a 1D numpy int16 array of residual values across the sampled pairs.
    Each pair contributes 2 frames × CAMERA_H × CAMERA_W × 3 channels of residuals.

    Per Catalog #229 PV: this runs the actual A1 inflate (CPU on macOS) so the
    measured entropy IS the empirical A1 residual distribution, not a synthetic
    proxy. Sister `tools/probe_stc_paradigm_reformulation_disambiguator.py` used
    synthetic residuals; this probe is the actual-measurement upgrade.
    """

    # Lazy imports to keep test-mode lightweight
    try:
        import av
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            f"PyAV + torch required for actual A1 decode; got {exc}. "
            "Use --synthetic-test-mode for fixture/CI runs."
        ) from exc

    # Add A1 vendored src to path for codec + model imports
    a1_src = REPO_ROOT / "submissions" / "a1" / "src"
    if str(a1_src) not in sys.path:
        sys.path.insert(0, str(a1_src))

    from codec import (
        LATENT_BLOB_LEN,
        apply_latent_sidecar,
        decode_decoder_compact,
        decode_latents_compact,
    )
    from model import HNeRVDecoder

    LATENT_DIM = 28
    BASE_CHANNELS = 36
    EVAL_H, EVAL_W = 384, 512

    # Read + parse A1 archive
    import zipfile

    with zipfile.ZipFile(archive_path) as zf:
        archive_bytes = zf.read("x")
    section_total = struct.unpack_from("<I", archive_bytes, 0)[0]
    decoder_blob = archive_bytes[4:section_total]
    latent_blob = archive_bytes[section_total:section_total + LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[section_total + LATENT_BLOB_LEN:]
    decoder_sd = decode_decoder_compact(decoder_blob)
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)

    if verbose:
        print(f"[probe] A1 decoder_blob={len(decoder_blob)} B, latent={LATENT_BLOB_LEN} B, "
              f"sidecar={len(sidecar_blob)} B")

    device = torch.device("cpu")
    decoder = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=(EVAL_H, EVAL_W),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    latents = latents.to(device)

    # Decode sampled pairs only (statistical sample for entropy estimation)
    sample_indices = list(range(0, min(sample_pairs, N_PAIRS)))
    if verbose:
        print(f"[probe] decoding {len(sample_indices)} pairs from A1 archive...")

    decoded_frames_list = []
    with torch.inference_mode():
        for pair_idx in sample_indices:
            decoded = decoder(latents[pair_idx:pair_idx + 1])
            flat = decoded.reshape(2, 3, EVAL_H, EVAL_W)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W),
                mode="bicubic", align_corners=False,
            )
            # Per A1 inflate: per-channel offsets
            up = up.reshape(1, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = (
                up.reshape(2, 3, CAMERA_H, CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            decoded_frames_list.append(frames)

    # Read ground-truth frames from video (only the sampled pair frames)
    if verbose:
        print(f"[probe] reading ground-truth frames from {video_path}...")
    gt_frames_needed = sorted({2 * pi for pi in sample_indices} | {2 * pi + 1 for pi in sample_indices})
    max_frame_needed = max(gt_frames_needed) + 1
    gt_frames_map: dict[int, np.ndarray] = {}
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx >= max_frame_needed:
                break
            if frame_idx in gt_frames_needed:
                img = frame.to_ndarray(format="rgb24")
                gt_frames_map[frame_idx] = img.astype(np.int16)
    finally:
        container.close()

    # Compute residuals = ground_truth - rendered per pair
    residuals_chunks = []
    for sample_i, pair_idx in enumerate(sample_indices):
        rendered_pair = decoded_frames_list[sample_i].astype(np.int16)  # (2, H, W, 3)
        gt_frame_0 = gt_frames_map[2 * pair_idx]
        gt_frame_1 = gt_frames_map[2 * pair_idx + 1]
        residual_0 = gt_frame_0 - rendered_pair[0]
        residual_1 = gt_frame_1 - rendered_pair[1]
        residuals_chunks.append(residual_0.ravel())
        residuals_chunks.append(residual_1.ravel())

    residuals = np.concatenate(residuals_chunks)
    if verbose:
        print(f"[probe] computed {residuals.size} residual values across "
              f"{len(sample_indices)} pairs ({len(sample_indices) * 2} frames)")
    return residuals


def synthetic_residuals_for_test(
    *,
    pattern: str = "high",
    n_symbols: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic residuals for test/fixture mode.

    Patterns:
    * "high": gaussian σ=8 (high-entropy + low sparsity)
    * "medium": gaussian σ=3 (medium entropy + medium sparsity)
    * "low": all zeros + small ε (low entropy + high sparsity = LOW tier
      per OVERNIGHT-W §9 since entropy < 1.5 dominates)
    * "uniform_random": uniform [-255, 255] (highest entropy)
    * "compressible": sparse spikes (low entropy + high sparsity)
    """

    rng = np.random.default_rng(seed)
    if pattern == "high":
        return rng.normal(0, 8, n_symbols).astype(np.int16)
    elif pattern == "medium":
        return rng.normal(0, 3, n_symbols).astype(np.int16)
    elif pattern == "low":
        out = np.zeros(n_symbols, dtype=np.int16)
        # add tiny noise to a few positions
        spike_idx = rng.choice(n_symbols, size=n_symbols // 100, replace=False)
        out[spike_idx] = rng.integers(-1, 2, size=spike_idx.size)
        return out
    elif pattern == "uniform_random":
        return rng.integers(-127, 128, size=n_symbols).astype(np.int16)
    elif pattern == "compressible":
        out = np.zeros(n_symbols, dtype=np.int16)
        spike_idx = rng.choice(n_symbols, size=n_symbols // 10, replace=False)
        out[spike_idx] = rng.integers(-20, 21, size=spike_idx.size)
        return out
    else:
        raise ValueError(f"unknown synthetic pattern: {pattern}")


def compute_stc_residual_entropy_on_a1(
    residuals: np.ndarray,
    *,
    a1_archive_sha256: str = CANONICAL_A1_ARCHIVE_SHA256,
    a1_archive_bytes: int = CANONICAL_A1_ARCHIVE_BYTES,
    sample_pairs_decoded: int = 0,
) -> STCResidualEntropyProbeVerdict:
    """Canonical entry point: given residuals array, return verdict.

    Per Catalog #229 PV + Catalog #287 evidence-tag discipline + Catalog #323
    canonical Provenance: every score-bearing field carries non-promotable markers.

    Per OVERNIGHT-W §10 + Catalog #344: canonical equation #359-sister
    `procedural_predictor_plus_residual_correction_savings_v1` IN-DOMAIN via
    context `stc_predictor_plus_residual_a1_per_pair_correction`.
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

    # Canonical equation #359-sister IN-DOMAIN computation
    eq_result = predict_procedural_predictor_plus_residual_correction_savings(
        original_payload_bytes=0,  # ADDITIVE sidecar
        predictor_seed_or_code_bytes=STC_PREDICTOR_SEED_BYTES,
        residual_stream_bytes=STC_RESIDUAL_STREAM_BYTES,
        container_overhead_bytes=0,
        context="stc_predictor_plus_residual_a1_per_pair_correction",
    )

    return STCResidualEntropyProbeVerdict(
        residual_entropy_bits_per_symbol=entropy,
        five_tuple_sparsity_ratio=sparsity,
        predicted_delta_s_rate_only=float(eq_result["predicted_delta_s_rate_only"]),
        predicted_delta_s_band=band,
        verdict_tier=tier,
        canonical_equation_id=eq_result["equation_id"],
        verdict_rationale=rationale,
        sample_pairs_decoded=sample_pairs_decoded,
        sample_pairs_total_residuals=int(residuals.size),
        a1_archive_sha256=a1_archive_sha256,
        a1_archive_bytes=a1_archive_bytes,
    )


def build_canonical_provenance_for_report(verdict: STCResidualEntropyProbeVerdict) -> dict[str, Any]:
    """Build canonical Provenance per Catalog #323 for the verdict report."""

    from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

    # Pass canonical archive sha + canonical source path per builder signature
    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256=(
            verdict.a1_archive_sha256
            if len(verdict.a1_archive_sha256) == 64
            else "0" * 64
        ),
        source_path=".omx/state/wyner_ziv_deliverability/stc_3a_a1_residual_probe.json",
    )
    if hasattr(prov, "to_dict"):
        return prov.to_dict()
    # Fallback: convert dataclass to dict
    from dataclasses import asdict
    return asdict(prov)


def register_verdict_in_probe_outcomes_ledger(
    verdict: STCResidualEntropyProbeVerdict,
    *,
    report_path: Path | None,
    subagent_id: str | None = None,
) -> dict[str, Any]:
    """Register the verdict in Catalog #313 probe-outcomes ledger.

    Verdict mapping per OVERNIGHT-W §9 cascade:
    * HIGH → VERDICT_PROCEED (unlock $5.20 paid Modal smoke)
    * MEDIUM → VERDICT_DEFER (defer to sister probe 3b)
    * LOW → VERDICT_DEFER (DEFER substrate per Catalog #307)
    """

    from tac.probe_outcomes_ledger import (
        VERDICT_DEFER,
        VERDICT_PROCEED,
        register_probe_outcome,
    )

    if verdict.verdict_tier == "HIGH":
        ledger_verdict = VERDICT_PROCEED
        next_action = "operator-routable: unlock $5.20 paid Modal smoke per OVERNIGHT-W §6 cascade gate 1"
    elif verdict.verdict_tier == "MEDIUM":
        ledger_verdict = VERDICT_DEFER
        next_action = "operator-routable: build sister probe 3b (Selfcomp tone-map-delta)"
    else:
        ledger_verdict = VERDICT_DEFER
        next_action = "operator-routable: DEFER substrate per Catalog #307 paradigm-vs-implementation classification"

    utc = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    probe_id = f"stc_3a_a1_residual_entropy_{utc.replace(':', '').replace('-', '')[:15]}"

    return register_probe_outcome(
        probe_id=probe_id,
        substrate="stc_paradigm_reformulation_a1_residual_path_3a",
        recipe_path=None,
        probe_kind="stc_3a_a1_residual_entropy",
        verdict=ledger_verdict,
        metric_name="residual_entropy_bits_per_symbol",
        metric_value=float(verdict.residual_entropy_bits_per_symbol),
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
        description="STC residual sidecar over A1 — per-pair RGB residual entropy probe",
    )
    parser.add_argument(
        "--a1-archive-path",
        type=Path,
        default=REPO_ROOT / "submissions" / "a1" / "archive.zip",
        help="Path to A1 archive.zip (default: submissions/a1/archive.zip)",
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
        help="Number of A1 pairs to decode for statistical sampling (default: 16; "
             "16 pairs * 2 frames * 874 * 1164 * 3 = ~97M residual values which is "
             "sufficient for entropy estimation)",
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
        help="Use synthetic residuals for test/fixture runs (skip A1 decode + video read)",
    )
    parser.add_argument(
        "--synthetic-pattern",
        type=str,
        default="high",
        choices=("high", "medium", "low", "uniform_random", "compressible"),
        help="Synthetic residual pattern for test mode (default: high)",
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

    # Phase 1: extract residuals (actual decode OR synthetic)
    if args.synthetic_test_mode:
        if args.verbose:
            print(f"[probe] synthetic test mode (pattern={args.synthetic_pattern})")
        residuals = synthetic_residuals_for_test(pattern=args.synthetic_pattern)
        a1_sha = "synthetic_test_fixture"
        a1_bytes = 0
        pairs_decoded = 0
    else:
        a1_sha = verify_a1_archive_sha256(args.a1_archive_path)
        if args.verbose:
            print(f"[probe] A1 archive sha verified: {a1_sha}")
        residuals = extract_a1_residuals_from_decode(
            archive_path=args.a1_archive_path,
            video_path=args.ground_truth_video,
            sample_pairs=args.sample_pairs,
            verbose=args.verbose,
        )
        a1_bytes = args.a1_archive_path.stat().st_size
        pairs_decoded = min(args.sample_pairs, N_PAIRS)

    # Phase 2: compute verdict
    verdict = compute_stc_residual_entropy_on_a1(
        residuals,
        a1_archive_sha256=a1_sha,
        a1_archive_bytes=a1_bytes,
        sample_pairs_decoded=pairs_decoded,
    )

    # Phase 3: assemble JSON report
    utc = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    report = {
        "schema": "stc_3a_a1_residual_entropy_probe_v1",
        "generated_at_utc": utc,
        "lane_id": "lane_overnight_y_stc_3a_a1_residual_entropy_probe_build_local_cpu_run_20260521",
        "canonical_equation_id": verdict.canonical_equation_id,
        "canonical_equation_context": "stc_predictor_plus_residual_a1_per_pair_correction",
        "verdict": verdict.as_dict(),
        "provenance": build_canonical_provenance_for_report(verdict),
        "next_action_per_verdict": {
            "HIGH": "operator-routable: unlock $5.20 paid Modal smoke per OVERNIGHT-W §6 cascade gate 1",
            "MEDIUM": "operator-routable: build sister probe 3b (Selfcomp tone-map-delta)",
            "LOW": "operator-routable: DEFER substrate per Catalog #307 paradigm-vs-implementation classification",
        }[verdict.verdict_tier],
    }

    # Phase 4: write report
    if args.output_report_json is None and not args.synthetic_test_mode:
        report_dir = REPO_ROOT / ".omx" / "state" / "wyner_ziv_deliverability"
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = utc.replace(":", "").replace("-", "")[:15]
        args.output_report_json = report_dir / f"stc_3a_a1_residual_probe_{timestamp}.json"
    if args.output_report_json is not None:
        args.output_report_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_report_json.write_text(json.dumps(report, indent=2, sort_keys=True))
        if args.verbose:
            print(f"[probe] report written to {args.output_report_json}")

    # Phase 5: register in Catalog #313 ledger (unless skipped or synthetic)
    if not args.skip_ledger_registration and not args.synthetic_test_mode:
        ledger_record = register_verdict_in_probe_outcomes_ledger(
            verdict,
            report_path=args.output_report_json,
            subagent_id=f"overnight_y_stc_3a_probe_{utc.replace(':', '').replace('-', '')[:15]}",
        )
        if args.verbose:
            print(f"[probe] registered in ledger: probe_id={ledger_record.get('probe_id')}")

    # Phase 6: print verdict
    print(f"\n=== STC 3a A1 residual entropy probe verdict ===")
    print(f"Tier: {verdict.verdict_tier}")
    print(f"Shannon entropy: {verdict.residual_entropy_bits_per_symbol:.4f} bits/symbol")
    print(f"5-tuple sparsity: {verdict.five_tuple_sparsity_ratio:.4f}")
    print(f"Predicted ΔS band: [{verdict.predicted_delta_s_band[0]:+.6f}, "
          f"{verdict.predicted_delta_s_band[1]:+.6f}] [prediction; NON-AUTHORITATIVE]")
    print(f"Rate-only ΔS (canonical eq #359-sister): "
          f"+{verdict.predicted_delta_s_rate_only:.6f}")
    print(f"Sample: {verdict.sample_pairs_decoded} pairs ({verdict.sample_pairs_total_residuals} residuals)")
    print(f"Axis: {verdict.axis_tag} (NOT contest-axis authoritative per Catalog #192)")
    print(f"\nRationale: {verdict.verdict_rationale}")
    print(f"\nNext action: {report['next_action_per_verdict']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
