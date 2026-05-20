#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Class-shift hypothesis empirical disambiguator for the 7 asymptotic-pursuit
candidate substrates from T3 grand-strategy Decision 4.

This is a $0 pre-entropy prober that closes the open question left by BUILD-2
(`feedback_slot_build_2_option_b_archive_member_sweep_top5_contest_landed_20260520.md`):

  BUILD-2 empirically verified 0/5 contest archives have non-zero deliverable
  Wyner-Ziv Tier-1+2 bytes — but ONLY for HNeRV-family archives
  (PR #95 / #101 / #102 / #103 + fec6).

  The class-shift recommendation in T3 Decision 4 ASSUMES non-HNeRV substrates
  will NOT saturate the same way, but this is UNTESTED.

This probe tests the assumption by constructing SYNTHETIC ARCHIVES that mirror
each asymptotic candidate's predicted byte budget per its design memo, then
re-uses the canonical entropy ladder (lzma / brotli / zlib) per
`tools/pre_entropy_substrate_pivot_prober.py::probe_member_compression` to
classify each synthetic candidate as `PRE_ENTROPY` / `AT_FLOOR` / `POST_ENTROPY`.

The result is a probe-disambiguator verdict per CLAUDE.md "Meta-Lagrangian /
Pareto solver" non-negotiable + "Subagent coherence-by-default" Hook #6
probe-disambiguator surface + Catalog #313 probe-outcomes ledger discipline.

Empirical interpretation
========================

* `CLASS_SHIFT_CONFIRMED` — synthetic archive bytes show ratio < 0.99 (lzma /
  brotli / zlib find non-trivial structure). The class-shift hypothesis HOLDS
  for THIS specific synthetic byte mixture (a Wyner-Ziv hoist would deliver
  positive saved-bytes IF the actual substrate archive matched this mixture).
  Verdict: `PROCEED` (substrate is a viable asymptotic-pursuit candidate).
* `CLASS_SHIFT_FALSIFIED` — synthetic archive bytes show ratio >= 1.05 (every
  general-purpose codec INFLATES, the same saturation signature as the
  HNeRV-family). The class-shift hypothesis FAILS for this synthetic mixture.
  Verdict: `DEFER` (substrate joins the HNeRV-family saturation cluster
  pending cargo-cult-unwind redesign per CLAUDE.md "Forbidden premature KILL").
* `INDETERMINATE` — synthetic archive sits in [0.99, 1.05] AT_FLOOR band, OR
  the design-memo data does not support a confident synthetic construction.
  Verdict: `OPERATOR_REVIEW_REQUIRED` (operator routes to next layer of probe).

Synthetic archive construction (per substrate)
=============================================

Each asymptotic candidate's design memo declares a predicted byte budget
(e.g. Z7-Mamba-2 ~110-140 KB total = ~30 KB encoder + ~30 KB decoder + ~30-50
KB predictor + ~5 KB latent + ~10 KB residuals + ~3 KB ego). The synthetic
archive mirrors that mixture via the canonical generator
``synthesize_substrate_archive_bytes`` which composes named byte segments per
the design-memo declaration:

  - ``raw_float_weights`` segment: simulated fp16 encoder/decoder weight tensor
    (Gaussian, std=0.02 per Hinton-Glorot init prior; expected to compress to
    ratio ~0.50-0.70 per Quantizr-class research-sidecar evidence)
  - ``brotli_compressed_weights`` segment: 1-pass brotli-q11 of the above
    (expected to INFLATE on re-compression — AT_FLOOR signature)
  - ``int8_quantized_residuals`` segment: simulated int8 quantization (uniform
    over [-128, 127]; expected ratio ~0.85 per Catalog #344
    per-byte-leverage-uniformly-distributed canonical equation)
  - ``categorical_posterior_stream`` segment: simulated 4-bit categorical via
    arithmetic coding (expected ratio ~0.95-1.00 — POST_ENTROPY)
  - ``header_meta_json`` segment: small overhead (~0.5-1 KB)

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog #287:
every probe outcome row carries `evidence_grade="predicted"` +
`measurement_axis="[diagnostic; synthetic-archive class-shift probe]"` +
`score_claim=false` + `promotion_eligible=false`. Synthetic archives are
NEVER tagged as contest-shipping bytes per Catalog #321.

[verified-against: tools/pre_entropy_substrate_pivot_prober.py
                   classify_compression_ratio + probe_member_compression
                   (canonical entropy classification + lzma/brotli/zlib ladder)]
[verified-against: feedback_slot_build_2_option_b_archive_member_sweep_top5_contest_landed_20260520.md
                   (BUILD-2 empirical 0/5 saturation anchor on HNeRV-family)]
[verified-against: .omx/research/council_t3_grand_strategy_review_20260520T120000Z.md
                   Decision 4 (7 asymptotic-pursuit candidates ranked by EV/$)]
[verified-against: src/tac/probe_outcomes_ledger.py
                   register_probe_outcome (canonical Catalog #313 ledger
                   API) + VALID_VERDICTS + BLOCKING_VERDICTS]
[verified-against: CLAUDE.md "Forbidden premature KILL without research
                   exhaustion" + "Subagent coherence-by-default" Hook #6 +
                   Catalog #287 + Catalog #292 + Catalog #294 + Catalog
                   #303 + Catalog #305 + Catalog #309 + Catalog #313 +
                   Catalog #321 + Catalog #323]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import lzma
import sys
import zlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Soft-import brotli per CLAUDE.md Catalog #203 (canonical contest-archive
# entropy coder). Sister probers gracefully handle absence.
try:
    import brotli  # type: ignore

    _HAS_BROTLI = True
except ImportError:  # pragma: no cover
    brotli = None
    _HAS_BROTLI = False

from tac.probe_outcomes_ledger import (  # noqa: E402
    BLOCKING_VERDICTS,
    register_probe_outcome,
)

# ──────────────────────────────────────────────────────────────────────── #
# Canonical contest constants                                               #
# ──────────────────────────────────────────────────────────────────────── #

# Per `tac.canonical_equations` registry + CLAUDE.md "Meta-Lagrangian/Pareto
# solver" + Catalog #344 canonical equation
# `canonical_frontier_pointer_v1` and `per_byte_leverage_uniformly_distributed_v1`.
CONTEST_RATE_DENOM_BYTES = 37_545_489

# Per sister `tools/pre_entropy_substrate_pivot_prober.py:149-151` canonical
# entropy classification thresholds.
PRE_ENTROPY_RATIO_THRESHOLD = 0.99
AT_FLOOR_RATIO_LOWER = 0.99
AT_FLOOR_RATIO_UPPER = 1.05

# fec6 baseline from BUILD-2 anchor (the saturation signature):
# 178,517 bytes brotli-ratio 1.000028 (INFLATES). Per
# `feedback_slot_build_2_option_b_archive_member_sweep_top5_contest_landed_20260520.md`.
FEC6_BASELINE_BYTES = 178_517
FEC6_BASELINE_BROTLI_RATIO = 1.000028
FEC6_BASELINE_SHA = (
    "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
)

# Probe schema constants.
SCHEMA_VERSION = "class_shift_hypothesis_empirical_disambiguator_v1"

# Output location per CLAUDE.md "Forbidden /tmp paths in any persisted
# artifact" — durable forensic state under .omx/state/.
OUTPUT_DIR_DEFAULT = REPO_ROOT / ".omx" / "state" / "wyner_ziv_deliverability"


# ──────────────────────────────────────────────────────────────────────── #
# Per-substrate predicted byte budgets (derived from design memos)          #
# ──────────────────────────────────────────────────────────────────────── #


@dataclass(frozen=True)
class SyntheticByteSegment:
    """One byte segment of the synthetic archive composition.

    ``kind`` controls how the byte payload is synthesized; ``size_bytes`` is
    the segment's byte budget per the substrate's design memo.
    """

    name: str
    kind: str  # 'raw_float_weights' | 'brotli_compressed_weights' | 'int8_residuals' | 'categorical_arith_4bit' | 'header_meta_json' | 'arith_coded_indices' | 'fp16_latent_init'
    size_bytes: int


@dataclass(frozen=True)
class SubstrateBudgetSpec:
    """Per-candidate synthetic archive composition + design-memo citation.

    The composition mirrors the substrate's predicted archive contents per
    its design memo (verbatim citations live in ``design_memo_citation``).

    ``confidence`` is one of:
      - ``high``: design memo explicitly enumerates a per-section byte budget
      - ``medium``: design memo gives an aggregate budget + we apportion via
        canonical priors
      - ``low``: design memo lacks explicit budget; we use the substrate's
        target_class default
    """

    substrate_id: str
    display_name: str
    composition: tuple[SyntheticByteSegment, ...]
    design_memo_citation: str
    horizon_class: str  # 'asymptotic_pursuit' per Catalog #309
    confidence: str  # 'high' | 'medium' | 'low'
    notes: str = ""


# Per T3 Decision 4 + per-substrate design memos cited inline.
CANONICAL_ASYMPTOTIC_CANDIDATES: tuple[SubstrateBudgetSpec, ...] = (
    # (1) DreamerV3 RSSM categorical posterior C6 paradigm-bridge B2.
    # Design memo: `.omx/research/dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z.md`.
    # Predicted: ~5.6 MB ceiling (T3 symposium target [0.20, 0.40]); frontier-
    # breaking [0.10, 0.18] requires <180 KB. We synthesize the frontier-
    # breaking config (180 KB target):
    SubstrateBudgetSpec(
        substrate_id="dreamerv3_rssm",
        display_name="DreamerV3 RSSM categorical posterior (C6 paradigm-bridge B2)",
        composition=(
            # 192 bits/sample × 1200 samples = 28.8 KB uncompressed categorical posterior;
            # ~5% of frontier archive budget per design memo §EXTREME OPTIMIZATION.
            SyntheticByteSegment("encoder_weights_fp16", "raw_float_weights", 30_000),
            SyntheticByteSegment("decoder_weights_fp16", "raw_float_weights", 30_000),
            SyntheticByteSegment("rssm_recurrent_state_fp16", "raw_float_weights", 40_000),
            SyntheticByteSegment("categorical_posterior_stream", "arith_coded_indices", 28_800),
            SyntheticByteSegment("residuals_int8", "int8_residuals", 30_000),
            SyntheticByteSegment("hyperprior_brotli", "brotli_compressed_weights", 20_000),
            SyntheticByteSegment("header_meta_json", "header_meta_json", 1_200),
        ),
        design_memo_citation=(
            ".omx/research/dreamerv3_rssm_categorical_rd_canonical_equation_derivation_"
            "20260520T131815Z.md (frontier-breaking [0.10, 0.18] target; <180 KB)"
        ),
        horizon_class="asymptotic_pursuit",
        confidence="high",
        notes="Categorical posterior is the distinguishing primitive; expected POST_ENTROPY",
    ),
    # (2) NSCS06 v8 hybrid_class_shift_path_C neural residual decoder
    # (formerly NSCS06 v8 Path B wavelet residual).
    # Design memo: `.omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md`
    # Predicted: ~600 KB after DB4 wavelet decorrelation (vs 4 MB pre-wavelet).
    SubstrateBudgetSpec(
        substrate_id="nscs06_v8_path_b",
        display_name="NSCS06 v8 Path B wavelet residual decoder",
        composition=(
            # Wavelet-decorrelated chroma streams + temporal residual + LUT.
            SyntheticByteSegment("wavelet_db4_low_pass", "raw_float_weights", 150_000),
            SyntheticByteSegment("wavelet_db4_high_pass", "raw_float_weights", 100_000),
            SyntheticByteSegment("chroma_cb_stream", "int8_residuals", 100_000),
            SyntheticByteSegment("chroma_cr_stream", "int8_residuals", 100_000),
            SyntheticByteSegment("temporal_residual", "int8_residuals", 80_000),
            SyntheticByteSegment("grayscale_lut", "raw_float_weights", 50_000),
            SyntheticByteSegment("brotli_wrapper", "brotli_compressed_weights", 20_000),
            SyntheticByteSegment("header_meta_json", "header_meta_json", 800),
        ),
        design_memo_citation=(
            ".omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_"
            "20260516.md (predicted ~600 KB; chroma + temporal + LUT + wavelet)"
        ),
        horizon_class="asymptotic_pursuit",
        confidence="medium",
        notes="Wavelet decorrelation predicted 5-10x; int8 residual streams predicted PRE_ENTROPY",
    ),
    # (3) Z7-Mamba-2 substrate.
    # Design memo: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
    # Predicted: ~110-140 KB total (similar to Z7-GRU).
    SubstrateBudgetSpec(
        substrate_id="z7_mamba2",
        display_name="Z7-Mamba-2 (selective state-space recurrent predictor)",
        composition=(
            # Per design-memo ARCHIVE GRAMMAR section:
            SyntheticByteSegment("encoder_state_dict_fp16_brotli", "brotli_compressed_weights", 30_000),
            SyntheticByteSegment("decoder_state_dict_fp16_brotli", "brotli_compressed_weights", 30_000),
            SyntheticByteSegment("predictor_state_dict_mamba2_brotli", "brotli_compressed_weights", 40_000),
            SyntheticByteSegment("latent_init_int8", "int8_residuals", 5_000),
            SyntheticByteSegment("residuals_int8", "int8_residuals", 10_000),
            SyntheticByteSegment("ego_motion_int8_sidecar", "int8_residuals", 3_000),
            SyntheticByteSegment("header_meta_json", "header_meta_json", 500),
            SyntheticByteSegment("header_overhead", "header_meta_json", 1_000),
        ),
        design_memo_citation=(
            ".omx/research/z7_mamba2_substrate_design_memo_20260518.md "
            "(archive grammar: encoder + decoder + Mamba-2 predictor + int8 streams; "
            "~110-140 KB total)"
        ),
        horizon_class="asymptotic_pursuit",
        confidence="high",
        notes="All weight tensors brotli-pre-compressed per design memo; expected AT_FLOOR",
    ),
    # (4) Z6-v2 Wave 2 dispatch resumption (Multi-layer FiLM candidate).
    # Design memo: `.omx/research/z6_v2_*design*.md` per T3 Decision 4.
    SubstrateBudgetSpec(
        substrate_id="z6_v2",
        display_name="Z6-v2 multi-layer FiLM ego-motion-conditioned predictor",
        composition=(
            # Z6-v1 ~95 KB baseline + ~50 KB FiLM predictor overhead + ~5 KB meta.
            SyntheticByteSegment("z6_v1_baseline_brotli", "brotli_compressed_weights", 95_000),
            SyntheticByteSegment("film_predictor_weights_fp16", "raw_float_weights", 40_000),
            SyntheticByteSegment("ego_motion_conditioning_int8", "int8_residuals", 10_000),
            SyntheticByteSegment("meta_overhead", "header_meta_json", 5_000),
        ),
        design_memo_citation=(
            ".omx/research/z6_v2_*design*.md Candidate 1 Multi-layer FiLM "
            "(~150 KB → rate ~0.10)"
        ),
        horizon_class="asymptotic_pursuit",
        confidence="medium",
        notes="Baseline brotli-pre-compressed (POST_ENTROPY signature); predictor raw fp16 (PRE_ENTROPY signature)",
    ),
    # (5) V1 Faiss V8 learned-compression scaffold.
    # Design memo: `.omx/research/v1_faiss_v8_learned_compression_faiss_design_20260519.md`
    # Predicted band [0.187, 0.193] contest-CPU; encoded representation
    # ~50K-param Ballé 2018 entropy-bottleneck + 4-bit categorical codebook.
    SubstrateBudgetSpec(
        substrate_id="v1_faiss_v8",
        display_name="V1 Faiss V8 (Ballé 2018 entropy-bottleneck + 4-bit codebook)",
        composition=(
            # ~50K params Balle 2018 encoder (fp16 = 100 KB; after brotli ~70 KB)
            SyntheticByteSegment("balle_encoder_fp16", "raw_float_weights", 100_000),
            # 4-bit categorical codebook + posterior stream
            SyntheticByteSegment("codebook_4bit_categorical", "categorical_arith_4bit", 5_000),
            SyntheticByteSegment("posterior_stream_4bit_arith", "arith_coded_indices", 30_000),
            # Faiss IVF-PQ index + per-region histogram side-info
            SyntheticByteSegment("faiss_ivf_pq_index", "raw_float_weights", 20_000),
            SyntheticByteSegment("per_region_histograms_int8", "int8_residuals", 15_000),
            SyntheticByteSegment("header_meta_json", "header_meta_json", 1_000),
        ),
        design_memo_citation=(
            ".omx/research/v1_faiss_v8_learned_compression_faiss_design_20260519.md "
            "(Ballé 2018 entropy-bottleneck + 4-bit categorical; predicted [0.187, 0.193])"
        ),
        horizon_class="asymptotic_pursuit",
        confidence="medium",
        notes="Ballé encoder fp16 raw + arith-coded categorical posterior",
    ),
    # (6) Q4-Q5 Wyner-Ziv deliverability empirical anchor (research_only per
    # 2026-05-17 Q4 BUILD HALT). The Q4 target IS the WZ-deliverable Wyner-Ziv
    # hoist applied to a paired sister-archive's pre-entropy bytes.
    # Since Q4 BUILD HALT designated `research_only`, we synthesize a
    # hypothetical Q4 architecture target archive (sister cooperative-receiver):
    SubstrateBudgetSpec(
        substrate_id="q4_q5_wyner_ziv_hypothetical",
        display_name="Q4-Q5 Wyner-Ziv deliverability (hypothetical hoist target)",
        composition=(
            # Hypothetical WZ hoist target shipping side-info derived from
            # ego-motion priors + scorer-class CDFs from sister archive.
            SyntheticByteSegment("ego_motion_priors_fp16", "raw_float_weights", 80_000),
            SyntheticByteSegment("scorer_class_cdfs_fp16", "raw_float_weights", 60_000),
            SyntheticByteSegment("predictive_receiver_residuals", "int8_residuals", 50_000),
            SyntheticByteSegment("wz_layer_meta_json", "header_meta_json", 1_500),
        ),
        design_memo_citation=(
            "Q4 BUILD HALT memo "
            "feedback_q4_wyner_ziv_pr101_state_dict_first_empirical_anchor_build_HALTED_premise_failure_20260517.md "
            "(research_only after BUILD HALT; reactivation: novel WZ hoist source)"
        ),
        horizon_class="asymptotic_pursuit",
        confidence="low",
        notes="Hypothetical reactivation-criteria target; predicted PRE_ENTROPY for raw fp16 priors",
    ),
    # (7) Rate-attack META-paradigm research wave.
    # Design memo: `.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md`
    # Predicted ΔS [-0.005, -0.001]; target is reducing inflate.py LOC + archive
    # bytes via mutual-information minimization per Wyner-Ziv 1976 framing.
    SubstrateBudgetSpec(
        substrate_id="rate_attack",
        display_name="Rate-attack META-paradigm (mutual-information minimization)",
        composition=(
            # Hypothetical rate-attack archive shipping minimal bytes via
            # compressed dictionary + tight selector.
            SyntheticByteSegment("compressed_dictionary_brotli", "brotli_compressed_weights", 5_000),
            SyntheticByteSegment("tight_selector_arith", "arith_coded_indices", 40_000),
            SyntheticByteSegment("minimal_decoder_fp16", "raw_float_weights", 30_000),
            SyntheticByteSegment("frame_residuals_int8", "int8_residuals", 80_000),
            SyntheticByteSegment("header_meta_json", "header_meta_json", 500),
        ),
        design_memo_citation=(
            ".omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md "
            "(predicted ΔS [-0.005, -0.001]; mutual-information minimization)"
        ),
        horizon_class="asymptotic_pursuit",
        confidence="low",
        notes="Hypothetical archive composition; predicted ratio similar to fec6 if selector well-designed",
    ),
)


# ──────────────────────────────────────────────────────────────────────── #
# Synthetic byte segment generators                                         #
# ──────────────────────────────────────────────────────────────────────── #


def _stable_seed_for_segment(substrate_id: str, segment_name: str) -> int:
    """Deterministic per-(substrate, segment) seed for reproducibility.

    Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog
    #294 dimension 7 (DETERMINISTIC REPRODUCIBILITY): every probe outcome
    must be byte-stable across re-runs. Mixing substrate + segment names
    into the seed avoids cross-segment correlation while preserving
    within-segment determinism.
    """
    h = hashlib.sha256(f"{substrate_id}:{segment_name}:v1".encode()).digest()
    return int.from_bytes(h[:8], "big")


def synthesize_raw_float_weights(size_bytes: int, seed: int) -> bytes:
    """Simulate fp16 weight tensor (Gaussian, std=0.02 per Hinton-Glorot prior).

    Expected to compress to ratio ~0.50-0.70 per Quantizr-class research-
    sidecar evidence (`feedback_pre_entropy_substrate_pivot_prober_landed_20260517.md`
    cites pr106_state_dict.pt 924 KB lzma→209 KB = 0.226 ratio for raw fp16
    weights; we model the conservative end of that range).
    """
    import struct

    rng_state = seed
    n_floats = max(1, size_bytes // 2)  # fp16 = 2 bytes
    out = bytearray()
    for _ in range(n_floats):
        # Linear congruential generator for reproducibility without numpy.
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        # Map to a small float [-0.1, 0.1] approximating Gaussian std=0.02.
        u = (rng_state >> 32) / (1 << 32)  # in [0, 1)
        v = (rng_state & ((1 << 32) - 1)) / (1 << 32)  # in [0, 1)
        # Box-Muller without numpy.
        from math import cos, log, pi, sqrt

        if u <= 0:
            u = 1e-12
        z = sqrt(-2.0 * log(u)) * cos(2.0 * pi * v)
        f = max(-1.0, min(1.0, 0.02 * z))
        # Pack as fp16 (E5M10).
        # Use struct.pack on float then truncate via fp16 cast for simplicity.
        try:
            packed = struct.pack("<e", f)
            out.extend(packed)
        except struct.error:
            out.extend(b"\x00\x00")
    return bytes(out[:size_bytes])


def synthesize_brotli_compressed_weights(size_bytes: int, seed: int) -> bytes:
    """Simulate already-brotli-compressed weights (POST_ENTROPY signature).

    Expected to INFLATE on re-compression (ratio > 1.0) — same signature as
    the HNeRV-family archives per BUILD-2 empirical anchor.
    """
    if not _HAS_BROTLI:
        # Fallback: zlib-compress the raw weights to simulate POST_ENTROPY.
        raw = synthesize_raw_float_weights(size_bytes * 3, seed)
        compressed = zlib.compress(raw, level=9)
        # Pad/truncate to exact size_bytes.
        if len(compressed) >= size_bytes:
            return compressed[:size_bytes]
        return compressed + bytes(size_bytes - len(compressed))
    raw = synthesize_raw_float_weights(size_bytes * 3, seed)
    assert brotli is not None  # type-narrow for mypy
    compressed = brotli.compress(raw, quality=11)
    if len(compressed) >= size_bytes:
        return compressed[:size_bytes]
    return compressed + bytes(size_bytes - len(compressed))


def synthesize_int8_residuals(size_bytes: int, seed: int) -> bytes:
    """Simulate int8 quantized residuals (uniform over [-128, 127]).

    Expected ratio ~0.85 per Catalog #344 per-byte-leverage-uniformly-
    distributed canonical equation. Uniform int8 streams have low structure
    but non-negligible repetition; brotli typically achieves ~0.80-0.90.
    """
    rng_state = seed
    out = bytearray()
    for _ in range(size_bytes):
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append((rng_state >> 24) & 0xFF)
    return bytes(out[:size_bytes])


def synthesize_categorical_arith_4bit(size_bytes: int, seed: int) -> bytes:
    """Simulate 4-bit categorical arithmetic-coded posterior stream.

    Two 4-bit symbols packed per byte. Skewed distribution (mode at 0) per
    Hafner DreamerV3 categorical posterior empirical pattern. Expected ratio
    ~0.92-0.98 (near AT_FLOOR with slight residual entropy).
    """
    rng_state = seed
    out = bytearray()
    for _ in range(size_bytes):
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        # Skewed: high bits dominate near 0, low bits more uniform.
        sym_high = ((rng_state >> 28) & 0xF) % 4  # mode at 0
        sym_low = (rng_state >> 24) & 0xF
        out.append((sym_high << 4) | sym_low)
    return bytes(out[:size_bytes])


def synthesize_arith_coded_indices(size_bytes: int, seed: int) -> bytes:
    """Simulate arithmetic-coded selector indices (highly structured).

    Expected ratio ~0.95-1.02 (NEAR AT_FLOOR; the canonical fec6-style
    signature). This is the closest synthetic surrogate to the BUILD-2
    empirical HNeRV-family saturation pattern.
    """
    rng_state = seed
    out = bytearray()
    for _ in range(size_bytes):
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        # Highly skewed distribution -> near-entropy-floor output.
        u = (rng_state >> 28) & 0xFF
        # Mode-collapse: 80% of bytes ∈ {0-15}; 20% uniform.
        if u < 200:
            out.append((rng_state >> 16) & 0xF)
        else:
            out.append((rng_state >> 8) & 0xFF)
    return bytes(out[:size_bytes])


def synthesize_header_meta_json(size_bytes: int, seed: int) -> bytes:
    """Simulate JSON header / meta block (text; compresses well, but bounded).

    Expected ratio ~0.40-0.60 (text compresses well, BUT small-size overhead
    means brotli dictionary cost dominates near AT_FLOOR for very small
    headers).
    """
    # Repetitive JSON template scaled to size_bytes.
    template = (
        '{"schema_version":"v1","substrate":"synthetic","frame_count":1200,'
        '"archive_grammar":"monolithic","inflate_runtime":"deterministic",'
        '"co_authors":["operator"],"contest_axis":"contest_cpu",'
        '"hardware_substrate":"linux_x86_64_cpu","measurement_axis":"diagnostic",'
        '"evidence_grade":"predicted","promotion_eligible":false}'
    )
    repeats = (size_bytes // len(template)) + 1
    raw = (template * repeats).encode("utf-8")
    return raw[:size_bytes]


def synthesize_segment(substrate_id: str, segment: SyntheticByteSegment) -> bytes:
    """Dispatch to the correct synthetic generator per ``segment.kind``."""
    seed = _stable_seed_for_segment(substrate_id, segment.name)
    generators = {
        "raw_float_weights": synthesize_raw_float_weights,
        "brotli_compressed_weights": synthesize_brotli_compressed_weights,
        "int8_residuals": synthesize_int8_residuals,
        "fp16_latent_init": synthesize_raw_float_weights,
        "categorical_arith_4bit": synthesize_categorical_arith_4bit,
        "arith_coded_indices": synthesize_arith_coded_indices,
        "header_meta_json": synthesize_header_meta_json,
    }
    fn = generators.get(segment.kind)
    if fn is None:
        raise ValueError(
            f"unknown segment kind {segment.kind!r} for substrate "
            f"{substrate_id!r} segment {segment.name!r}"
        )
    return fn(segment.size_bytes, seed)


def synthesize_substrate_archive_bytes(spec: SubstrateBudgetSpec) -> bytes:
    """Compose the synthetic archive bytes by concatenating all segments.

    The resulting byte stream mirrors the substrate's predicted archive
    contents per its design memo. The composition is byte-stable per Catalog
    #294 dimension 7 (DETERMINISTIC REPRODUCIBILITY).
    """
    chunks: list[bytes] = []
    for segment in spec.composition:
        chunks.append(synthesize_segment(spec.substrate_id, segment))
    return b"".join(chunks)


# ──────────────────────────────────────────────────────────────────────── #
# Entropy classification (mirrors sister prober)                            #
# ──────────────────────────────────────────────────────────────────────── #


def _compress_lzma(data: bytes) -> bytes:
    return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)


def _compress_brotli(data: bytes) -> bytes | None:
    if not _HAS_BROTLI or brotli is None:
        return None
    return brotli.compress(data, quality=11)


def _compress_zlib(data: bytes) -> bytes:
    return zlib.compress(data, level=9)


def classify_compression_ratio(ratio: float) -> str:
    """Mirror `tools/pre_entropy_substrate_pivot_prober.py::classify_compression_ratio`.

    < 0.99 = PRE_ENTROPY (compressible)
    [0.99, 1.05] = AT_FLOOR (entropy-saturated)
    > 1.05 = POST_ENTROPY (inflates on re-compression)
    """
    if ratio < PRE_ENTROPY_RATIO_THRESHOLD:
        return "PRE_ENTROPY"
    if ratio > AT_FLOOR_RATIO_UPPER:
        return "POST_ENTROPY"
    return "AT_FLOOR"


@dataclass(frozen=True)
class CompressionProbeResult:
    raw_bytes: int
    lzma_bytes: int
    lzma_ratio: float
    brotli_bytes: int | None
    brotli_ratio: float | None
    zlib_bytes: int
    zlib_ratio: float
    best_codec: str
    best_ratio: float
    classification: str


def probe_archive_compression(data: bytes) -> CompressionProbeResult:
    """Run the canonical lzma/brotli/zlib ladder on the synthetic archive."""
    raw = len(data)
    if raw == 0:
        return CompressionProbeResult(
            raw_bytes=0,
            lzma_bytes=0,
            lzma_ratio=1.0,
            brotli_bytes=0 if _HAS_BROTLI else None,
            brotli_ratio=1.0 if _HAS_BROTLI else None,
            zlib_bytes=0,
            zlib_ratio=1.0,
            best_codec="lzma",
            best_ratio=1.0,
            classification="AT_FLOOR",
        )
    lz = len(_compress_lzma(data))
    lz_ratio = lz / raw
    br_bytes_payload = _compress_brotli(data)
    br = len(br_bytes_payload) if br_bytes_payload is not None else None
    br_ratio = br / raw if br is not None else None
    zl = len(_compress_zlib(data))
    zl_ratio = zl / raw

    candidates: list[tuple[str, float]] = [("lzma", lz_ratio), ("zlib", zl_ratio)]
    if br_ratio is not None:
        candidates.append(("brotli", br_ratio))
    best_codec, best_ratio = min(candidates, key=lambda x: x[1])

    return CompressionProbeResult(
        raw_bytes=raw,
        lzma_bytes=lz,
        lzma_ratio=lz_ratio,
        brotli_bytes=br,
        brotli_ratio=br_ratio,
        zlib_bytes=zl,
        zlib_ratio=zl_ratio,
        best_codec=best_codec,
        best_ratio=best_ratio,
        classification=classify_compression_ratio(best_ratio),
    )


# ──────────────────────────────────────────────────────────────────────── #
# Per-substrate verdict derivation                                          #
# ──────────────────────────────────────────────────────────────────────── #


@dataclass
class SubstrateClassShiftVerdict:
    """One per-candidate verdict + canonical Provenance fields.

    Verdict semantics:
      - ``CLASS_SHIFT_CONFIRMED``: ratio < 0.99; PRE_ENTROPY; PROCEED.
      - ``CLASS_SHIFT_FALSIFIED``: ratio > 1.05; POST_ENTROPY; DEFER.
      - ``INDETERMINATE``: ratio in [0.99, 1.05]; AT_FLOOR; OPERATOR_REVIEW_REQUIRED.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" +
    Catalog #287: every row carries `evidence_grade="predicted"` +
    `score_claim=False` + `promotion_eligible=False`.
    """

    substrate_id: str
    display_name: str
    horizon_class: str
    confidence: str
    design_memo_citation: str
    composition_summary: list[dict[str, Any]]
    synthetic_archive_bytes: int
    synthetic_archive_sha256: str
    compression: CompressionProbeResult
    class_shift_classification: str  # 'CLASS_SHIFT_CONFIRMED' | 'CLASS_SHIFT_FALSIFIED' | 'INDETERMINATE'
    probe_outcome_verdict: str  # one of VALID_VERDICTS
    deliverable_score_savings_estimate: float
    rationale: str
    reactivation_criteria: list[str]
    evidence_grade: str = "predicted"
    measurement_axis: str = "[diagnostic; synthetic-archive class-shift probe]"
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


def _estimate_deliverable_savings(
    raw_bytes: int, best_ratio: float
) -> float:
    """Canonical formula (sister of pre_entropy prober)."""
    if best_ratio >= 1.0 or raw_bytes <= 0:
        return 0.0
    saved = raw_bytes * (1.0 - best_ratio)
    return 25.0 * saved / CONTEST_RATE_DENOM_BYTES


def derive_verdict_for_substrate(
    spec: SubstrateBudgetSpec,
) -> SubstrateClassShiftVerdict:
    """Construct synthetic archive, classify entropy, derive verdict."""
    archive_bytes = synthesize_substrate_archive_bytes(spec)
    sha = hashlib.sha256(archive_bytes).hexdigest()
    compression = probe_archive_compression(archive_bytes)
    classification_bucket = compression.classification

    # Map entropy classification -> class-shift hypothesis verdict.
    if classification_bucket == "PRE_ENTROPY":
        class_shift = "CLASS_SHIFT_CONFIRMED"
        probe_verdict = "PROCEED"
        rationale = (
            f"Synthetic archive ratio {compression.best_ratio:.6f} via "
            f"{compression.best_codec} < 0.99 PRE_ENTROPY threshold. "
            f"Predicted deliverable savings = "
            f"{_estimate_deliverable_savings(compression.raw_bytes, compression.best_ratio):.6f} "
            f"per-archive (NON-AUTHORITATIVE; predicted only). "
            f"Class-shift hypothesis HOLDS for this synthetic mixture: a "
            f"Wyner-Ziv hoist applied to this substrate's actual archive bytes "
            f"could plausibly deliver positive Tier-1+2 savings (vs the "
            f"HNeRV-family 0/5 BUILD-2 baseline)."
        )
        reactivation: list[str] = [
            "Construct actual substrate archive via paid Modal/Lightning smoke",
            "Re-run pre-entropy prober on actual archive.zip member bytes",
            "Verify empirical ratio < 0.99 on the real archive (synthetic only "
            "informs the hypothesis; the real archive may differ)",
        ]
    elif classification_bucket == "POST_ENTROPY":
        class_shift = "CLASS_SHIFT_FALSIFIED"
        probe_verdict = "DEFER"
        rationale = (
            f"Synthetic archive ratio {compression.best_ratio:.6f} via "
            f"{compression.best_codec} > 1.05 POST_ENTROPY threshold. "
            f"Same saturation signature as the HNeRV-family fec6 baseline "
            f"(ratio {FEC6_BASELINE_BROTLI_RATIO:.6f}). "
            f"Class-shift hypothesis FAILS for this synthetic mixture: this "
            f"substrate would join the saturation cluster pending cargo-cult-"
            f"unwind redesign per CLAUDE.md 'Forbidden premature KILL without "
            f"research exhaustion'."
        )
        reactivation = [
            "Apply cargo-cult-unwind methodology per Catalog #303 (NSCS06 v6→v7 "
            "44% improvement precedent) to identify which HARD-EARNED-vs-CARGO-"
            "CULTED assumption produced the saturation",
            "Operator-frontier-override per Catalog #199 to dispatch actual "
            "archive empirically (synthetic may differ from actual)",
            "Alternative substrate redesign per per-substrate symposium "
            "Catalog #325 6-step contract",
        ]
    else:
        class_shift = "INDETERMINATE"
        probe_verdict = "OPERATOR_REVIEW_REQUIRED"
        rationale = (
            f"Synthetic archive ratio {compression.best_ratio:.6f} via "
            f"{compression.best_codec} in [0.99, 1.05] AT_FLOOR band. "
            f"Insufficient empirical signal to discriminate class-shift "
            f"from within-class refinement. Operator routes to next layer "
            f"of probe (e.g. actual paid smoke + post-training Tier-C density "
            f"per Catalog #324)."
        )
        reactivation = [
            "Run actual smoke dispatch + harvest real archive bytes",
            "Apply Catalog #324 post-training Tier-C density validation on "
            "landed archive",
            "Per-substrate symposium per Catalog #325 6-step contract before "
            "paid dispatch",
        ]

    composition_summary = [
        {
            "name": seg.name,
            "kind": seg.kind,
            "size_bytes": seg.size_bytes,
        }
        for seg in spec.composition
    ]

    deliverable_savings = _estimate_deliverable_savings(
        compression.raw_bytes, compression.best_ratio
    )

    return SubstrateClassShiftVerdict(
        substrate_id=spec.substrate_id,
        display_name=spec.display_name,
        horizon_class=spec.horizon_class,
        confidence=spec.confidence,
        design_memo_citation=spec.design_memo_citation,
        composition_summary=composition_summary,
        synthetic_archive_bytes=len(archive_bytes),
        synthetic_archive_sha256=sha,
        compression=compression,
        class_shift_classification=class_shift,
        probe_outcome_verdict=probe_verdict,
        deliverable_score_savings_estimate=deliverable_savings,
        rationale=rationale,
        reactivation_criteria=reactivation,
    )


# ──────────────────────────────────────────────────────────────────────── #
# Output emit + ledger registration                                         #
# ──────────────────────────────────────────────────────────────────────── #


def emit_canonical_manifest(
    verdicts: list[SubstrateClassShiftVerdict],
    output_path: Path,
) -> Path:
    """Write the canonical machine-readable manifest to output_path.

    Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" + Catalog
    #131 (write through fcntl-locked atomic-replace). For this prober we use
    the standard path discipline (output under .omx/state/) but defer to
    direct atomic write (single-writer probe; no concurrent sister
    contention) per Catalog #131 same-line waiver pattern.
    """
    output_str = str(output_path)
    if (
        output_str.startswith("/tmp/")
        or "/private/tmp/" in output_str
        or "/var/tmp/" in output_str
    ):
        raise ValueError(
            f"refusing to write class-shift disambiguator output to tmp: "
            f"{output_str!r} (per CLAUDE.md 'Forbidden /tmp paths in any "
            f"persisted artifact')"
        )

    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "probe_id_prefix": "class_shift_hypothesis_disambiguator",
        "generated_at_utc": now_utc,
        "fec6_baseline": {
            "archive_bytes": FEC6_BASELINE_BYTES,
            "brotli_ratio": FEC6_BASELINE_BROTLI_RATIO,
            "archive_sha256": FEC6_BASELINE_SHA,
            "classification": "AT_FLOOR",
            "source": (
                "feedback_slot_build_2_option_b_archive_member_sweep_top5_contest_"
                "landed_20260520.md"
            ),
        },
        "evidence_grade": "predicted",
        "measurement_axis": "[diagnostic; synthetic-archive class-shift probe]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "brotli_available": _HAS_BROTLI,
        "per_substrate_verdicts": {
            v.substrate_id: {
                "display_name": v.display_name,
                "horizon_class": v.horizon_class,
                "confidence": v.confidence,
                "design_memo_citation": v.design_memo_citation,
                "composition_summary": v.composition_summary,
                "synthetic_archive_bytes": v.synthetic_archive_bytes,
                "synthetic_archive_sha256": v.synthetic_archive_sha256,
                "compression": asdict(v.compression),
                "class_shift_classification": v.class_shift_classification,
                "probe_outcome_verdict": v.probe_outcome_verdict,
                "deliverable_score_savings_estimate": v.deliverable_score_savings_estimate,
                "rationale": v.rationale,
                "reactivation_criteria": v.reactivation_criteria,
                "evidence_grade": v.evidence_grade,
                "measurement_axis": v.measurement_axis,
                "score_claim": v.score_claim,
                "promotion_eligible": v.promotion_eligible,
                "ready_for_exact_eval_dispatch": v.ready_for_exact_eval_dispatch,
            }
            for v in verdicts
        },
        "summary": {
            "candidates_probed": len(verdicts),
            "class_shift_confirmed": sum(
                1 for v in verdicts if v.class_shift_classification == "CLASS_SHIFT_CONFIRMED"
            ),
            "class_shift_falsified": sum(
                1 for v in verdicts if v.class_shift_classification == "CLASS_SHIFT_FALSIFIED"
            ),
            "indeterminate": sum(
                1 for v in verdicts if v.class_shift_classification == "INDETERMINATE"
            ),
        },
        "claude_md_compliance_tags": [
            "synthetic_archive_per_design_memo_byte_budget",
            "entropy_ladder_lzma_brotli_zlib_per_sister_prober",
            "non_authoritative_per_catalog_192",
            "score_claim_false_per_catalog_287_and_323",
            "phantom_score_research_sidecar_rejected_per_catalog_321",
            "reactivation_criteria_pinned_per_forbidden_premature_kill",
            "horizon_class_declared_per_catalog_309",
            "probe_disambiguator_hook_6_active_per_catalog_125",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def register_verdicts_in_ledger(
    verdicts: list[SubstrateClassShiftVerdict],
    manifest_path: Path,
    *,
    subagent_id: str = "wave-3-class-shift-disambiguator-20260520",
) -> list[dict[str, Any]]:
    """Append one Catalog #313 probe-outcomes row per verdict.

    The probe_id schema is::

        probe_class_shift_hypothesis_<substrate_id>_<utc_compact>
    """
    now_compact = (
        datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y%m%dT%H%M%S")
    )
    records: list[dict[str, Any]] = []
    for v in verdicts:
        probe_id = (
            f"probe_class_shift_hypothesis_{v.substrate_id}_{now_compact}"
        )

        if v.probe_outcome_verdict in BLOCKING_VERDICTS:
            next_action = (
                f"do_not_dispatch_{v.substrate_id}_until_reactivation_criteria_met"
            )
        else:
            next_action = (
                f"proceed_to_actual_smoke_dispatch_for_{v.substrate_id}"
            )

        notes = (
            f"{v.class_shift_classification} via {v.compression.best_codec} "
            f"ratio={v.compression.best_ratio:.6f} on synthetic archive "
            f"{v.synthetic_archive_bytes} bytes. {v.rationale}"
        )

        record = register_probe_outcome(
            probe_id=probe_id,
            substrate=v.substrate_id,
            recipe_path=None,
            probe_kind="class_shift_hypothesis_via_synthetic_archive_entropy_ladder",
            verdict=v.probe_outcome_verdict,
            metric_name="best_compression_ratio",
            metric_value=v.compression.best_ratio,
            threshold=PRE_ENTROPY_RATIO_THRESHOLD,
            threshold_token="PRE_ENTROPY_THRESHOLD_0.99",
            evidence_path=str(
                manifest_path.relative_to(REPO_ROOT)
                if manifest_path.is_absolute()
                else manifest_path
            ),
            next_action=next_action,
            agent="claude",
            subagent_id=subagent_id,
            notes=notes,
            class_shift_classification=v.class_shift_classification,
            synthetic_archive_sha256=v.synthetic_archive_sha256,
            synthetic_archive_bytes=v.synthetic_archive_bytes,
            confidence=v.confidence,
            horizon_class=v.horizon_class,
            evidence_grade=v.evidence_grade,
            measurement_axis=v.measurement_axis,
            score_claim=v.score_claim,
            promotion_eligible=v.promotion_eligible,
            ready_for_exact_eval_dispatch=v.ready_for_exact_eval_dispatch,
            reactivation_criteria=v.reactivation_criteria,
        )
        records.append(record)
    return records


# ──────────────────────────────────────────────────────────────────────── #
# CLI                                                                       #
# ──────────────────────────────────────────────────────────────────────── #


def _emit_human_readable_summary(
    verdicts: list[SubstrateClassShiftVerdict],
) -> None:
    """Print operator-facing summary table to stdout."""
    print()
    print("=" * 88)
    print("CLASS-SHIFT HYPOTHESIS EMPIRICAL DISAMBIGUATOR — 7 ASYMPTOTIC-PURSUIT CANDIDATES")
    print("=" * 88)
    print(
        f"{'substrate':<32} {'classification':<22} {'verdict':<24} {'ratio':<10}"
    )
    print("-" * 88)
    for v in verdicts:
        print(
            f"{v.substrate_id:<32} {v.class_shift_classification:<22} "
            f"{v.probe_outcome_verdict:<24} {v.compression.best_ratio:.6f}"
        )
    print("-" * 88)
    n_confirmed = sum(
        1 for v in verdicts if v.class_shift_classification == "CLASS_SHIFT_CONFIRMED"
    )
    n_falsified = sum(
        1 for v in verdicts if v.class_shift_classification == "CLASS_SHIFT_FALSIFIED"
    )
    n_indeterminate = sum(
        1 for v in verdicts if v.class_shift_classification == "INDETERMINATE"
    )
    print(
        f"SUMMARY: {n_confirmed} CONFIRMED  /  {n_falsified} FALSIFIED  /  "
        f"{n_indeterminate} INDETERMINATE  (of {len(verdicts)} total)"
    )
    print(
        f"fec6 baseline (BUILD-2 anchor): brotli_ratio={FEC6_BASELINE_BROTLI_RATIO:.6f} "
        f"(AT_FLOOR; saturation signature)"
    )
    print("=" * 88)
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output manifest path. Defaults to "
            ".omx/state/wyner_ziv_deliverability/"
            "class_shift_hypothesis_empirical_disambiguator_<utc>.json"
        ),
    )
    parser.add_argument(
        "--no-ledger-write",
        action="store_true",
        help=(
            "Skip Catalog #313 ledger writes. Useful for dry-runs or "
            "synthetic test invocations."
        ),
    )
    parser.add_argument(
        "--subagent-id",
        type=str,
        default="wave-3-class-shift-disambiguator-20260520",
        help="Subagent ID for ledger attribution (Catalog #313 schema field).",
    )
    parser.add_argument(
        "--candidate-filter",
        type=str,
        default=None,
        help=(
            "Comma-separated substrate_id list to probe (subset of canonical "
            "7). Default: probe all 7."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip the human-readable summary table.",
    )
    return parser.parse_args(argv)


def run(
    *,
    output: Path | None = None,
    no_ledger_write: bool = False,
    subagent_id: str = "wave-3-class-shift-disambiguator-20260520",
    candidate_filter: Iterable[str] | None = None,
    quiet: bool = False,
) -> tuple[Path, list[SubstrateClassShiftVerdict]]:
    """Programmatic entry point (mirrors CLI)."""
    candidates = CANONICAL_ASYMPTOTIC_CANDIDATES
    if candidate_filter is not None:
        wanted = {c.strip() for c in candidate_filter if c.strip()}
        candidates = tuple(c for c in candidates if c.substrate_id in wanted)
        if not candidates:
            raise ValueError(
                f"candidate_filter {wanted!r} matched zero canonical "
                f"asymptotic substrates"
            )

    verdicts = [derive_verdict_for_substrate(spec) for spec in candidates]

    if output is None:
        now_compact = (
            datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y%m%dT%H%M%S")
        )
        output = (
            OUTPUT_DIR_DEFAULT
            / f"class_shift_hypothesis_empirical_disambiguator_{now_compact}.json"
        )

    manifest_path = emit_canonical_manifest(verdicts, output)

    if not no_ledger_write:
        register_verdicts_in_ledger(
            verdicts, manifest_path, subagent_id=subagent_id
        )

    if not quiet:
        _emit_human_readable_summary(verdicts)
        print(f"Manifest written to: {manifest_path}")
        if not no_ledger_write:
            print(
                f"Catalog #313 probe-outcomes ledger rows appended for "
                f"{len(verdicts)} substrates"
            )

    return manifest_path, verdicts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    filter_list: list[str] | None = None
    if args.candidate_filter is not None:
        filter_list = [c.strip() for c in args.candidate_filter.split(",")]
    try:
        run(
            output=args.output,
            no_ledger_write=args.no_ledger_write,
            subagent_id=args.subagent_id,
            candidate_filter=filter_list,
            quiet=args.quiet,
        )
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
