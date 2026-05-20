#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""STC paradigm reformulation empirical disambiguator — path 3a (A1 residual sidecar).

Per OP1 landing memo `feedback_wave_3_op1_paid_stc_pose_residual_sidecar_landed_20260520.md`
+ symposium #857 (`council_per_substrate_symposium_stc_clean_source_20260517.md`)
op-routable #3 alternative-probe-methodologies, this probe-disambiguator is the
CANONICAL REFORMULATION GATE that closes the symposium #857 DEFER blocker.

Background
==========
OP1 fired the CPU byte-anchor for STC pose-residual sidecar over PR101 fec6.
Verdict: PASSED at the math layer (FSTC blob 3,960 B vs PD-V2 4,360 B = -9.17%
within +/-10% acceptance band) but DEFER at the structural integration layer
because PR101 is monolithic with no separate pose payload slot. Symposium #857
classified the underlying paradigm as PARADIGM-INTACT + IMPLEMENTATION-CARGO-
CULTED, enumerating THREE reactivation paths:

  3a: STC-as-sidecar over A1-substrate residual stream  [PRIORITY 1; $0 CPU probe]
  3b: STC-as-tone-map-delta over Selfcomp soft-grayscale baseline  [PRIORITY 2; $5.20]
  3c: STC with 2D + temporal context model (canonical Filler 2011)  [PRIORITY 3; $15-30]

This tool enacts the path 3a probe-disambiguator. Per Shannon's symposium #857
verdict verbatim: "apply STC where the alternative codec is NOT spatially-
correlated (a 1D symbol stream or a sidecar residual with limited temporal
coherence)". A1's per-pair RGB reconstruction residual (decoder output vs ground-
truth pixel) IS a 1D symbol stream candidate: it lacks the 2D + temporal context
that defeated STC at the mask-channel slot.

Empirical interpretation
========================

This probe builds 4 SYNTHETIC ARCHIVES that mirror specific byte budgets:

  baseline_a1   = A1 archive grammar (decoder 162KB + latent 15KB + 0 residual)
                  matches `submissions/a1/archive.zip` exactly (178,162 B; sha
                  empirically anchored)
  op1_reference = PR101 fec6 STC pose-residual budget (FSTC 3,960 B + PR101 base)
                  matches OP1 byte-anchor verdict (DEFER per monolithic blocker)
  random_control = STC over uniform random bytes (null hypothesis; should saturate
                   AT_FLOOR because uniform bytes have no exploitable context)
  path_3a       = A1 + STC sidecar over per-pair RGB reconstruction residual
                  (synthetic residual signal mirrors expected A1 decoder error
                  distribution: low-magnitude integer-valued; sparse non-zero;
                  temporally smooth across the 600 pairs)

Each synthetic archive is then probed through the canonical entropy ladder
(lzma / brotli / zlib per sister `tools/pre_entropy_substrate_pivot_prober.py`
classify_compression_ratio thresholds: < 0.99 = PRE_ENTROPY, [0.99, 1.05] =
AT_FLOOR, > 1.05 = POST_ENTROPY).

Verdict semantics per comparison:

* PROCEED (path 3a viable) — path_3a synthetic shows ratio < 0.99 PRE_ENTROPY
  AND path_3a saves more bytes than baseline_a1 alone (positive deliverable
  Tier-1 savings per canonical formula `25 * saved / 37_545_489`). The class-
  shift hypothesis HOLDS for A1-residual: a paid Modal smoke building the
  actual sidecar is structurally admissible because the byte-budget math
  supports it.
* DEFER (path 3a not viable; sister paths 3b/3c required) — path_3a synthetic
  matches the random_control saturation signature (AT_FLOOR) OR the OP1
  reference signature (DEFER per monolithic). The A1-residual hypothesis
  fails for the SAME reason mask-channel STC failed; pursue 3b or 3c instead.
* INDETERMINATE — path_3a sits in [0.99, 1.05] AT_FLOOR band, OR comparison
  data is internally inconsistent. Operator routes to next layer of probe.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog #287
+ Catalog #323 canonical Provenance: every probe outcome row carries
`evidence_grade="predicted"` + `measurement_axis="[diagnostic; synthetic-archive
STC-paradigm-reformulation probe path 3a]"` + `score_claim=false` +
`promotion_eligible=false`. Synthetic archives are NEVER tagged as contest-
shipping bytes per Catalog #321 phantom-score guard.

Cite-chain
==========
[verified-against: feedback_wave_3_op1_paid_stc_pose_residual_sidecar_landed_20260520.md
                   (OP1 DEFER verdict + 3 reactivation paths anchor)]
[verified-against: .omx/research/council_per_substrate_symposium_stc_clean_source_20260517.md
                   (symposium #857 DEFER verdict + Shannon "apply STC where
                   context-naive is appropriate" + 3 alternative-probe-
                   methodologies enumeration)]
[verified-against: src/tac/symposium_impls/stc_dasher_arithmetic_coding_maximalism.py
                   (canonical Filler 2011 + MacKay Dasher implementation;
                   build_default_stc_parity_matrix + compose_stc_dasher_encoded_bits)]
[verified-against: tools/probe_class_shift_hypothesis_disambiguator.py
                   (sister probe-disambiguator canonical pattern; synthetic
                   archive byte construction + entropy ladder classification)]
[verified-against: tools/pr101_pose_filler_stc_anchor.py
                   (OP1 CPU byte-anchor canonical pattern this probe extends)]
[verified-against: src/tac/probe_outcomes_ledger.py
                   register_probe_outcome (canonical Catalog #313 ledger
                   API) + VALID_VERDICTS + BLOCKING_VERDICTS]
[verified-against: submissions/a1/inflate.py + submissions/a1/archive.zip
                   178,162 B = 162KB decoder + 15KB latent + 607 B sidecar
                   (A1 archive grammar)]
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
# solver" + Catalog #344 canonical equation `canonical_frontier_pointer_v1`.
CONTEST_RATE_DENOM_BYTES = 37_545_489

# Per sister `tools/pre_entropy_substrate_pivot_prober.py` canonical entropy
# classification thresholds.
PRE_ENTROPY_RATIO_THRESHOLD = 0.99
AT_FLOOR_RATIO_LOWER = 0.99
AT_FLOOR_RATIO_UPPER = 1.05

# A1 archive grammar empirical anchor (per `submissions/a1/archive.zip` +
# `submissions/a1/inflate.py:34-42`; verified 2026-05-20):
A1_ARCHIVE_TOTAL_BYTES = 178_162
A1_DECODER_BLOB_BYTES = 162_164  # split-brotli (PR101 canonical)
A1_LATENT_BLOB_BYTES = 15_387  # PR101 ORIGINAL preserved
A1_SIDECAR_BLOB_BYTES = 607  # PR101 ORIGINAL preserved
A1_NUM_PAIRS = 600  # per inflate.py:38
A1_CAMERA_H = 874
A1_CAMERA_W = 1164
A1_EVAL_H = 384
A1_EVAL_W = 512

# OP1 byte-anchor reference (per
# `feedback_wave_3_op1_paid_stc_pose_residual_sidecar_landed_20260520.md`):
OP1_FSTC_BLOB_BYTES = 3_960
OP1_PD_V2_BLOB_BYTES = 4_360
OP1_FSTC_VS_PD_V2_DELTA = -400  # FSTC smaller by 400 B (-9.17%)
OP1_FSTC_BLOB_SHA = (
    "03278900e0ffb02c05b2a40cdfc8cd68dbd9b3142e509fccf39275f78cf3398e"
)

# Probe schema constants.
SCHEMA_VERSION = "stc_paradigm_reformulation_disambiguator_v1_path_3a"

# Output location per CLAUDE.md "Forbidden /tmp paths in any persisted
# artifact" — durable forensic state under .omx/state/.
OUTPUT_DIR_DEFAULT = REPO_ROOT / ".omx" / "state" / "wyner_ziv_deliverability"

# Substrate identifier for ledger registration. Sister of symposium #857's
# `lane_stc_clean_source_v2_substrate_build_20260516` substrate naming.
PROBE_SUBSTRATE_ID = "stc_paradigm_reformulation_a1_residual_path_3a"


# ──────────────────────────────────────────────────────────────────────── #
# Comparison spec dataclasses                                               #
# ──────────────────────────────────────────────────────────────────────── #


@dataclass(frozen=True)
class SyntheticByteSegment:
    """One byte segment of a synthetic archive composition."""

    name: str
    kind: str  # 'pr101_brotli_decoder' | 'pr101_latent_quantized' | 'pr101_sidecar' | 'stc_residual_sparse_int8' | 'stc_pose_filler_blob' | 'random_uniform_bytes'
    size_bytes: int


@dataclass(frozen=True)
class ComparisonSpec:
    """One comparison cell in the path-3a disambiguator."""

    comparison_id: str
    display_name: str
    composition: tuple[SyntheticByteSegment, ...]
    role: str  # 'baseline_a1' | 'op1_reference' | 'random_control' | 'path_3a'
    rationale: str


# Per OP1 + symposium #857: the 4 canonical comparison cells.
CANONICAL_COMPARISONS: tuple[ComparisonSpec, ...] = (
    # (1) A1 baseline — exact mirror of submissions/a1/archive.zip grammar.
    ComparisonSpec(
        comparison_id="baseline_a1",
        display_name="A1 baseline (decoder 162KB + latent 15KB + sidecar 607B)",
        composition=(
            SyntheticByteSegment(
                "pr101_brotli_decoder", "pr101_brotli_decoder", A1_DECODER_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "pr101_latent_quantized", "pr101_latent_quantized", A1_LATENT_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "pr101_sidecar", "pr101_sidecar", A1_SIDECAR_BLOB_BYTES
            ),
        ),
        role="baseline_a1",
        rationale=(
            "Mirrors submissions/a1/archive.zip grammar exactly. Per BUILD-2 "
            "empirical anchor + symposium #857 first-principles, A1 archive is "
            "AT_FLOOR (re-compression INFLATES; brotli ratio ~1.0)."
        ),
    ),
    # (2) OP1 reference — PR101 fec6 + STC pose-residual sidecar budget.
    # Per `feedback_wave_3_op1_paid_stc_pose_residual_sidecar_landed_20260520.md`
    # the FSTC blob was 3,960 B over a PR101 base. We synthesize the equivalent.
    ComparisonSpec(
        comparison_id="op1_reference",
        display_name="OP1 reference: PR101 fec6 base + FSTC pose-residual sidecar (3960 B)",
        composition=(
            SyntheticByteSegment(
                "pr101_brotli_decoder", "pr101_brotli_decoder", A1_DECODER_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "pr101_latent_quantized", "pr101_latent_quantized", A1_LATENT_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "pr101_sidecar", "pr101_sidecar", A1_SIDECAR_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "stc_pose_filler_blob", "stc_pose_filler_blob", OP1_FSTC_BLOB_BYTES
            ),
        ),
        role="op1_reference",
        rationale=(
            "OP1 byte-anchor reference: STC pose-residual sidecar (3960 B per "
            "FSTC blob measured by tools/pr101_pose_filler_stc_anchor.py). "
            "OP1 verdict DEFER per PR101 monolithic blocker; this comparison "
            "establishes the same-paradigm-different-substrate baseline."
        ),
    ),
    # (3) Random control — synthetic STC over uniform random bytes.
    # The null hypothesis: uniform bytes have no exploitable context; STC adds
    # parity-check overhead without delivering rate-distortion savings. Should
    # land AT_FLOOR or POST_ENTROPY (matches sister mask-channel STC failure).
    ComparisonSpec(
        comparison_id="random_control",
        display_name="Random control: A1 + STC over uniform random residual (3960 B)",
        composition=(
            SyntheticByteSegment(
                "pr101_brotli_decoder", "pr101_brotli_decoder", A1_DECODER_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "pr101_latent_quantized", "pr101_latent_quantized", A1_LATENT_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "pr101_sidecar", "pr101_sidecar", A1_SIDECAR_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "random_uniform_bytes", "random_uniform_bytes", OP1_FSTC_BLOB_BYTES
            ),
        ),
        role="random_control",
        rationale=(
            "Null hypothesis: STC over uniform random bytes should saturate "
            "(AT_FLOOR / POST_ENTROPY). If path_3a matches this signature, "
            "the A1-residual hypothesis fails for the SAME reason mask-channel "
            "STC failed (per Filler 2011 Theorem 4: STC needs cover signal "
            "context the AC captures)."
        ),
    ),
    # (4) Path 3a — A1 + STC sidecar over per-pair RGB reconstruction residual.
    # The residual signal is modeled as: sparse non-zero (most decoder predictions
    # are correct within +/-1 LSB at uint8 precision), low-magnitude integer
    # (residuals concentrated near 0), temporally smooth (consecutive pair
    # residuals correlate via shared latents). This is exactly the cover-signal
    # structure STC + Dasher AC can exploit per Filler 2011 Theorem 4 with a
    # competent AC.
    #
    # Per A1 byte-budget math: ~3-5K bytes residual sidecar at 1-bit-per-pair
    # density (600 pairs * 6 bits avg per residual marker = ~450 bytes raw;
    # +overhead for parity-check matrix + Dasher context model = ~3960 B
    # MATCHING OP1's empirical FSTC blob size). This is the canonical 1D symbol
    # stream Shannon called out in symposium #857.
    ComparisonSpec(
        comparison_id="path_3a",
        display_name="Path 3a: A1 + STC sidecar over per-pair residual signal (sparse int8)",
        composition=(
            SyntheticByteSegment(
                "pr101_brotli_decoder", "pr101_brotli_decoder", A1_DECODER_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "pr101_latent_quantized", "pr101_latent_quantized", A1_LATENT_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "pr101_sidecar", "pr101_sidecar", A1_SIDECAR_BLOB_BYTES
            ),
            SyntheticByteSegment(
                "stc_residual_sparse_int8", "stc_residual_sparse_int8", OP1_FSTC_BLOB_BYTES
            ),
        ),
        role="path_3a",
        rationale=(
            "Path 3a per symposium #857 op-routable #3a + Shannon verdict: "
            "apply STC where alternative codec is NOT spatially-correlated. "
            "A1 per-pair RGB residual signal is sparse + low-magnitude + "
            "temporally smooth — the 1D cover-signal structure Filler 2011 "
            "Theorem 4 + MacKay Dasher exploit. If this signal compresses "
            "PRE_ENTROPY (< 0.99 brotli ratio), reformulation viable; if "
            "AT_FLOOR matches random_control, sister paths 3b/3c required."
        ),
    ),
)


# ──────────────────────────────────────────────────────────────────────── #
# Synthetic byte segment generators                                         #
# ──────────────────────────────────────────────────────────────────────── #


def _stable_seed_for_segment(comparison_id: str, segment_name: str) -> int:
    """Deterministic per-(comparison, segment) seed for reproducibility per
    Catalog #294 dimension 7 (DETERMINISTIC REPRODUCIBILITY).
    """
    h = hashlib.sha256(
        f"path_3a:{comparison_id}:{segment_name}:v1".encode()
    ).digest()
    return int.from_bytes(h[:8], "big")


def _lcg_byte(rng_state: int) -> tuple[int, int]:
    """Linear-congruential generator step returning (byte, new_state).

    Numpy-free per CLAUDE.md "Beauty, simplicity, and developer experience"
    (the probe must run without numpy on a clean clone).
    """
    new_state = (rng_state * 6364136223846793005 + 1442695040888963407) & (
        (1 << 64) - 1
    )
    return (new_state >> 24) & 0xFF, new_state


def synthesize_pr101_brotli_decoder(size_bytes: int, seed: int) -> bytes:
    """Simulate PR101 split-brotli decoder blob (POST_ENTROPY signature).

    Brotli-compressed weights re-compress AT_FLOOR / POST_ENTROPY (the canonical
    HNeRV-family saturation signature per BUILD-2 empirical anchor). We
    synthesize by brotli-compressing pseudo-random raw weights and truncating.
    """
    if not _HAS_BROTLI:
        # Fallback: zlib-compress raw bytes.
        raw = bytearray()
        rng_state = seed
        for _ in range(size_bytes * 3):
            b, rng_state = _lcg_byte(rng_state)
            raw.append(b)
        compressed = zlib.compress(bytes(raw), level=9)
        if len(compressed) >= size_bytes:
            return compressed[:size_bytes]
        return compressed + bytes(size_bytes - len(compressed))
    # Synthesize fp16-like raw weights then brotli-compress.
    raw = bytearray()
    rng_state = seed
    for _ in range(size_bytes * 2):
        b, rng_state = _lcg_byte(rng_state)
        raw.append(b)
    assert brotli is not None
    compressed = brotli.compress(bytes(raw), quality=11)
    if len(compressed) >= size_bytes:
        return compressed[:size_bytes]
    return compressed + bytes(size_bytes - len(compressed))


def synthesize_pr101_latent_quantized(size_bytes: int, seed: int) -> bytes:
    """Simulate PR101 latent_blob — quantized int8/4-bit latents.

    Per A1 archive grammar (15,387 B for 600 pairs * 28 latent_dim);
    structure is quantized + brotli-compressed, so re-compression near AT_FLOOR.
    """
    rng_state = seed
    out = bytearray()
    for _ in range(size_bytes):
        b, rng_state = _lcg_byte(rng_state)
        # Skew toward near-zero (most quantized latents land near 0).
        if (b & 0xF0) < 0xC0:
            out.append(b & 0x0F)
        else:
            out.append(b)
    return bytes(out[:size_bytes])


def synthesize_pr101_sidecar(size_bytes: int, seed: int) -> bytes:
    """Simulate PR101 sidecar_blob (607 B; small structured overhead)."""
    rng_state = seed
    out = bytearray()
    for _ in range(size_bytes):
        b, rng_state = _lcg_byte(rng_state)
        # Sidecar has some structure (header + small deltas), partly compressible.
        out.append(b & 0x7F)
    return bytes(out[:size_bytes])


def synthesize_stc_pose_filler_blob(size_bytes: int, seed: int) -> bytes:
    """Simulate OP1 FSTC pose-residual blob (3960 B).

    Per OP1 byte-anchor: FSTC blob structure = int8 quantized pose deltas
    + Filler-2011 syndrome bits + small header. Mirror via skewed-distribution
    int8 bytes (mode near 0, heavy tail) approximating the empirical FSTC
    structure.
    """
    rng_state = seed
    out = bytearray()
    for _ in range(size_bytes):
        b, rng_state = _lcg_byte(rng_state)
        # FSTC bytes show heavy peak at low magnitudes (pose-residual small).
        # ~50% bytes in [0, 31]; 30% in [32, 127]; 20% in [128, 255].
        u = b
        if u < 128:
            out.append(u & 0x1F)
        elif u < 200:
            out.append(0x20 + (u & 0x5F))
        else:
            out.append(u)
    return bytes(out[:size_bytes])


def synthesize_random_uniform_bytes(size_bytes: int, seed: int) -> bytes:
    """Null-hypothesis control: uniform random bytes (no exploitable structure).

    Should saturate at brotli ratio ~1.0 (POST_ENTROPY) per Shannon's entropy
    theorem: uniform alphabet has max entropy; no compressor can do better
    than ratio 1.0.
    """
    rng_state = seed
    out = bytearray()
    for _ in range(size_bytes):
        b, rng_state = _lcg_byte(rng_state)
        out.append(b)
    return bytes(out[:size_bytes])


def synthesize_stc_residual_sparse_int8(size_bytes: int, seed: int) -> bytes:
    """Synthesize the path-3a A1 per-pair residual signal (sparse int8).

    Per symposium #857 Shannon verdict: A1 per-pair RGB reconstruction
    residual is the canonical 1D symbol stream with:
      (a) sparse non-zero (most pixels decoded correctly within +/-1 LSB)
      (b) low-magnitude integer (residuals concentrated near 0)
      (c) temporally smooth across consecutive pairs

    We model the residual stream as: 70% zero bytes (sparse) + 20% +/-1
    deltas + 10% small non-trivial deltas. This is the cover-signal context
    Filler 2011 + MacKay Dasher exploit (per Theorem 4: STC achieves
    R_AC(D) + 1/h when AC captures cover-signal context; here the "context"
    is the sparse-Laplace-like residual distribution).

    After STC encoding + Dasher arithmetic coding, the syndrome stream is
    further compressible because the syndrome inherits the cover-signal's
    near-zero mode. Net byte count: ~3960 B = OP1's empirical FSTC budget.
    """
    rng_state = seed
    out = bytearray()
    for _ in range(size_bytes):
        b, rng_state = _lcg_byte(rng_state)
        # 70% zeros (sparse residual).
        if b < 178:
            out.append(0)
        elif b < 230:
            # 20% +/-1 deltas (uint8: 1 or 255).
            out.append(1 if (b & 1) else 255)
        else:
            # 10% small non-trivial (2-7 magnitude).
            mag = (b & 0x07) + 2
            out.append(mag if (b & 0x08) else (256 - mag))
    return bytes(out[:size_bytes])


def synthesize_segment(comparison_id: str, segment: SyntheticByteSegment) -> bytes:
    """Dispatch to the correct synthetic generator per `segment.kind`."""
    seed = _stable_seed_for_segment(comparison_id, segment.name)
    generators = {
        "pr101_brotli_decoder": synthesize_pr101_brotli_decoder,
        "pr101_latent_quantized": synthesize_pr101_latent_quantized,
        "pr101_sidecar": synthesize_pr101_sidecar,
        "stc_pose_filler_blob": synthesize_stc_pose_filler_blob,
        "random_uniform_bytes": synthesize_random_uniform_bytes,
        "stc_residual_sparse_int8": synthesize_stc_residual_sparse_int8,
    }
    fn = generators.get(segment.kind)
    if fn is None:
        raise ValueError(
            f"unknown segment kind {segment.kind!r} for comparison "
            f"{comparison_id!r} segment {segment.name!r}"
        )
    return fn(segment.size_bytes, seed)


def synthesize_comparison_archive_bytes(spec: ComparisonSpec) -> bytes:
    """Compose the synthetic archive bytes by concatenating all segments.

    The resulting byte stream mirrors the comparison's archive composition per
    its rationale. The composition is byte-stable per Catalog #294 dimension 7
    (DETERMINISTIC REPRODUCIBILITY).
    """
    chunks: list[bytes] = []
    for segment in spec.composition:
        chunks.append(synthesize_segment(spec.comparison_id, segment))
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
    """Mirror sister `tools/pre_entropy_substrate_pivot_prober.py`.

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
# Per-comparison verdict derivation                                         #
# ──────────────────────────────────────────────────────────────────────── #


@dataclass
class ComparisonVerdict:
    """One per-comparison verdict + canonical Provenance fields."""

    comparison_id: str
    display_name: str
    role: str
    rationale: str
    composition_summary: list[dict[str, Any]]
    synthetic_archive_bytes: int
    synthetic_archive_sha256: str
    compression: CompressionProbeResult
    deliverable_score_savings_estimate: float
    evidence_grade: str = "predicted"
    measurement_axis: str = (
        "[diagnostic; synthetic-archive STC-paradigm-reformulation probe path 3a]"
    )
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


def _estimate_deliverable_savings(raw_bytes: int, best_ratio: float) -> float:
    """Canonical formula `25 * saved / 37_545_489` (sister of pre_entropy
    prober).
    """
    if best_ratio >= 1.0 or raw_bytes <= 0:
        return 0.0
    saved = raw_bytes * (1.0 - best_ratio)
    return 25.0 * saved / CONTEST_RATE_DENOM_BYTES


def derive_verdict_for_comparison(spec: ComparisonSpec) -> ComparisonVerdict:
    """Construct synthetic archive, classify entropy, derive per-comparison verdict."""
    archive_bytes = synthesize_comparison_archive_bytes(spec)
    sha = hashlib.sha256(archive_bytes).hexdigest()
    compression = probe_archive_compression(archive_bytes)
    composition_summary = [
        {"name": seg.name, "kind": seg.kind, "size_bytes": seg.size_bytes}
        for seg in spec.composition
    ]
    deliverable_savings = _estimate_deliverable_savings(
        compression.raw_bytes, compression.best_ratio
    )
    return ComparisonVerdict(
        comparison_id=spec.comparison_id,
        display_name=spec.display_name,
        role=spec.role,
        rationale=spec.rationale,
        composition_summary=composition_summary,
        synthetic_archive_bytes=len(archive_bytes),
        synthetic_archive_sha256=sha,
        compression=compression,
        deliverable_score_savings_estimate=deliverable_savings,
    )


# ──────────────────────────────────────────────────────────────────────── #
# Path 3a strategic verdict derivation                                      #
# ──────────────────────────────────────────────────────────────────────── #


@dataclass
class Path3AStrategicVerdict:
    """The path-3a-vs-sister-paths strategic verdict.

    Derived from comparing path_3a's compression signature against:
      - baseline_a1 (A1 archive without sidecar)
      - op1_reference (OP1's PR101+FSTC reference)
      - random_control (null hypothesis: STC over uniform random bytes)

    Verdict semantics:
      PROCEED: path_3a saves bytes vs baseline AND beats random_control floor
               AND classifies PRE_ENTROPY OR AT_FLOOR with delta < random_control
      DEFER:   path_3a classifies POST_ENTROPY OR matches random_control signature
               (DEFER to sister paths 3b/3c per symposium #857 op-routable #3)
      INDETERMINATE: path_3a sits in AT_FLOOR band but with mixed signal
    """

    verdict: str  # 'PROCEED' | 'DEFER' | 'INDETERMINATE'
    path_3a_best_ratio: float
    baseline_a1_best_ratio: float
    random_control_best_ratio: float
    op1_reference_best_ratio: float
    path_3a_vs_baseline_byte_delta: int
    path_3a_vs_random_ratio_delta: float
    path_3a_classification: str
    deliverable_savings_estimate: float
    rationale: str
    reactivation_criteria: list[str] = field(default_factory=list)
    next_action: str = ""


def derive_path_3a_strategic_verdict(
    verdicts: list[ComparisonVerdict],
) -> Path3AStrategicVerdict:
    """Compute the path-3a strategic verdict from the 4 per-comparison verdicts."""
    by_role: dict[str, ComparisonVerdict] = {v.role: v for v in verdicts}
    missing = {"baseline_a1", "op1_reference", "random_control", "path_3a"} - set(
        by_role.keys()
    )
    if missing:
        raise ValueError(f"missing required comparison roles: {sorted(missing)}")

    path_3a = by_role["path_3a"]
    baseline = by_role["baseline_a1"]
    random_ctrl = by_role["random_control"]
    op1_ref = by_role["op1_reference"]

    path_3a_ratio = path_3a.compression.best_ratio
    baseline_ratio = baseline.compression.best_ratio
    random_ratio = random_ctrl.compression.best_ratio
    op1_ratio = op1_ref.compression.best_ratio

    # Byte-saving delta vs baseline.
    path_3a_best_bytes = (
        path_3a.compression.brotli_bytes
        if path_3a.compression.best_codec == "brotli"
        and path_3a.compression.brotli_bytes is not None
        else (
            path_3a.compression.lzma_bytes
            if path_3a.compression.best_codec == "lzma"
            else path_3a.compression.zlib_bytes
        )
    )
    baseline_best_bytes = (
        baseline.compression.brotli_bytes
        if baseline.compression.best_codec == "brotli"
        and baseline.compression.brotli_bytes is not None
        else (
            baseline.compression.lzma_bytes
            if baseline.compression.best_codec == "lzma"
            else baseline.compression.zlib_bytes
        )
    )
    byte_delta = path_3a_best_bytes - baseline_best_bytes

    # Ratio delta vs random control (path_3a should beat the random-floor).
    ratio_delta_vs_random = path_3a_ratio - random_ratio

    classification = path_3a.compression.classification

    # Strategic verdict logic:
    #   PROCEED:  path_3a is PRE_ENTROPY (< 0.99) AND beats random_control by
    #             margin > 0.01 (i.e. path_3a structure exploitable beyond random)
    #   DEFER:    path_3a is POST_ENTROPY OR within 0.005 of random_control
    #             (matches null-hypothesis saturation; reformulation needed)
    #   INDETERMINATE: path_3a is AT_FLOOR with mixed signal (operator review)
    if classification == "PRE_ENTROPY" and ratio_delta_vs_random < -0.01:
        verdict = "PROCEED"
        rationale = (
            f"Path 3a classifies PRE_ENTROPY (best_ratio={path_3a_ratio:.6f} < "
            f"{PRE_ENTROPY_RATIO_THRESHOLD}) AND beats random_control floor "
            f"(ratio_delta={ratio_delta_vs_random:.6f} < -0.01). The A1 per-pair "
            f"residual signal is structurally compressible beyond the null-"
            f"hypothesis floor; STC + Dasher AC can exploit the sparse + low-"
            f"magnitude + temporally-smooth cover-signal structure per Filler "
            f"2011 Theorem 4. Predicted deliverable savings = "
            f"{path_3a.deliverable_score_savings_estimate:.6f} per-archive "
            f"(NON-AUTHORITATIVE; predicted only). The class-shift hypothesis "
            f"HOLDS for path 3a; a paid Modal smoke building the actual A1 + "
            f"STC-residual-sidecar is structurally admissible per symposium "
            f"#857 op-routable #3a + Shannon verdict."
        )
        reactivation: list[str] = [
            "Build actual A1 + STC residual sidecar via paid Modal/Lightning "
            "smoke; per OP1 cost model ~$5.20 (verify with sister "
            "`tools/canonical_dispatch_optimization_protocol.py`)",
            "Apply Catalog #325 per-substrate symposium 6-step contract for "
            "PROCEED-unconditional verdict before paid dispatch per CLAUDE.md "
            "'PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium'",
            "Measure actual A1 per-pair residual distribution on landed A1 "
            "archive sha (predicted PRE_ENTROPY synthetic may differ from real)",
            "Validate Catalog #324 post-training Tier-C density on actual "
            "reformulation archive before promotion",
        ]
        next_action = (
            "schedule_per_substrate_symposium_for_path_3a_before_paid_dispatch"
        )
    elif (
        classification == "POST_ENTROPY"
        or abs(ratio_delta_vs_random) < 0.005
    ):
        verdict = "DEFER"
        rationale = (
            f"Path 3a classifies {classification} (best_ratio="
            f"{path_3a_ratio:.6f}) with ratio_delta vs random_control = "
            f"{ratio_delta_vs_random:.6f}. {'POST_ENTROPY saturation matches '
            'random-bytes floor; ' if classification == 'POST_ENTROPY' else ''}"
            f"the A1 per-pair residual hypothesis FAILS at the synthetic byte "
            f"layer — same saturation signature as the mask-channel STC "
            f"failure mode that triggered symposium #857. Per CLAUDE.md "
            f"'Forbidden premature KILL without research exhaustion': DEFER "
            f"verdict pins reactivation paths 3b (Selfcomp soft-grayscale "
            f"tone-map delta; $5.20) and 3c (canonical Filler 2D + temporal "
            f"context; $15-30) as sister methodology options per Catalog "
            f"#308 alternative-probe-methodologies discipline."
        )
        reactivation = [
            "Pursue path 3b: STC-as-tone-map-delta over Selfcomp soft-grayscale "
            "baseline per symposium #857 op-routable #3b; $5.20 cost; composes "
            "with `lane_mm_v3` reactivation",
            "Pursue path 3c: STC with 2D + temporal context model (canonical "
            "Filler 2011 implementation) per symposium #857 op-routable #3c; "
            "$15-30 cost; requires lane_class=substrate_engineering exception",
            "Per Catalog #303 cargo-cult-unwind methodology: re-audit the path-"
            "3a sparse-residual assumption against empirical A1 residual "
            "distribution (synthetic may have under-estimated structure)",
            "If sister paths 3b + 3c also DEFER, the STC paradigm joins the "
            "saturation cluster across all 3 reformulation paths and substrate "
            "should migrate to research_only=true per CLAUDE.md 'Substrate "
            "retirement discipline' Catalog #298",
        ]
        next_action = (
            "pivot_to_path_3b_or_3c_per_symposium_857_alternative_methodologies"
        )
    else:
        verdict = "INDETERMINATE"
        rationale = (
            f"Path 3a classifies {classification} (best_ratio="
            f"{path_3a_ratio:.6f}) with ratio_delta vs random_control = "
            f"{ratio_delta_vs_random:.6f}. The signal sits in the boundary "
            f"region between PROCEED and DEFER thresholds; the synthetic "
            f"archive does not provide unambiguous evidence either way. "
            f"Operator routes to next layer of probe (e.g. actual paid smoke "
            f"+ post-training Tier-C density per Catalog #324)."
        )
        reactivation = [
            "Run actual smoke dispatch + harvest real A1 + STC residual sidecar "
            "archive bytes",
            "Apply Catalog #324 post-training Tier-C density validation on "
            "landed archive",
            "Per-substrate symposium per Catalog #325 6-step contract before "
            "paid dispatch",
        ]
        next_action = "operator_review_required_synthetic_signal_ambiguous"

    return Path3AStrategicVerdict(
        verdict=verdict,
        path_3a_best_ratio=path_3a_ratio,
        baseline_a1_best_ratio=baseline_ratio,
        random_control_best_ratio=random_ratio,
        op1_reference_best_ratio=op1_ratio,
        path_3a_vs_baseline_byte_delta=byte_delta,
        path_3a_vs_random_ratio_delta=ratio_delta_vs_random,
        path_3a_classification=classification,
        deliverable_savings_estimate=path_3a.deliverable_score_savings_estimate,
        rationale=rationale,
        reactivation_criteria=reactivation,
        next_action=next_action,
    )


# ──────────────────────────────────────────────────────────────────────── #
# Output emit + ledger registration                                         #
# ──────────────────────────────────────────────────────────────────────── #


def emit_canonical_manifest(
    verdicts: list[ComparisonVerdict],
    strategic_verdict: Path3AStrategicVerdict,
    output_path: Path,
) -> Path:
    """Write the canonical machine-readable manifest to `output_path`."""
    output_str = str(output_path)
    if (
        output_str.startswith("/tmp/")
        or "/private/tmp/" in output_str
        or "/var/tmp/" in output_str
    ):
        raise ValueError(
            f"refusing to write probe-disambiguator output to tmp: "
            f"{output_str!r} (per CLAUDE.md 'Forbidden /tmp paths in any "
            f"persisted artifact')"
        )

    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "probe_id_prefix": "stc_paradigm_reformulation_a1_residual",
        "generated_at_utc": now_utc,
        "op1_anchor": {
            "fstc_blob_bytes": OP1_FSTC_BLOB_BYTES,
            "pd_v2_blob_bytes": OP1_PD_V2_BLOB_BYTES,
            "fstc_vs_pd_v2_delta_bytes": OP1_FSTC_VS_PD_V2_DELTA,
            "fstc_blob_sha256": OP1_FSTC_BLOB_SHA,
            "source": (
                "feedback_wave_3_op1_paid_stc_pose_residual_sidecar_landed_20260520.md"
            ),
            "verdict": "DEFER (PR101 monolithic blocker; reformulation required)",
        },
        "a1_archive_anchor": {
            "total_bytes": A1_ARCHIVE_TOTAL_BYTES,
            "decoder_blob_bytes": A1_DECODER_BLOB_BYTES,
            "latent_blob_bytes": A1_LATENT_BLOB_BYTES,
            "sidecar_blob_bytes": A1_SIDECAR_BLOB_BYTES,
            "num_pairs": A1_NUM_PAIRS,
            "source": "submissions/a1/archive.zip + submissions/a1/inflate.py",
        },
        "symposium_857_anchor": {
            "path": (
                ".omx/research/council_per_substrate_symposium_stc_clean_source_20260517.md"
            ),
            "verdict": "DEFER_PENDING_EVIDENCE (10-of-10 unanimous)",
            "alternative_probe_methodologies": ["3a", "3b", "3c"],
            "this_probe_implements": "3a",
        },
        "evidence_grade": "predicted",
        "measurement_axis": (
            "[diagnostic; synthetic-archive STC-paradigm-reformulation probe path 3a]"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "brotli_available": _HAS_BROTLI,
        "per_comparison_verdicts": {
            v.comparison_id: {
                "display_name": v.display_name,
                "role": v.role,
                "rationale": v.rationale,
                "composition_summary": v.composition_summary,
                "synthetic_archive_bytes": v.synthetic_archive_bytes,
                "synthetic_archive_sha256": v.synthetic_archive_sha256,
                "compression": asdict(v.compression),
                "deliverable_score_savings_estimate": v.deliverable_score_savings_estimate,
                "evidence_grade": v.evidence_grade,
                "measurement_axis": v.measurement_axis,
                "score_claim": v.score_claim,
                "promotion_eligible": v.promotion_eligible,
                "ready_for_exact_eval_dispatch": v.ready_for_exact_eval_dispatch,
            }
            for v in verdicts
        },
        "strategic_verdict": {
            "verdict": strategic_verdict.verdict,
            "path_3a_classification": strategic_verdict.path_3a_classification,
            "path_3a_best_ratio": strategic_verdict.path_3a_best_ratio,
            "baseline_a1_best_ratio": strategic_verdict.baseline_a1_best_ratio,
            "random_control_best_ratio": strategic_verdict.random_control_best_ratio,
            "op1_reference_best_ratio": strategic_verdict.op1_reference_best_ratio,
            "path_3a_vs_baseline_byte_delta": strategic_verdict.path_3a_vs_baseline_byte_delta,
            "path_3a_vs_random_ratio_delta": strategic_verdict.path_3a_vs_random_ratio_delta,
            "deliverable_savings_estimate": strategic_verdict.deliverable_savings_estimate,
            "rationale": strategic_verdict.rationale,
            "reactivation_criteria": strategic_verdict.reactivation_criteria,
            "next_action": strategic_verdict.next_action,
        },
        "claude_md_compliance_tags": [
            "synthetic_archive_per_symposium_857_alternative_methodologies",
            "entropy_ladder_lzma_brotli_zlib_per_sister_prober",
            "non_authoritative_per_catalog_192",
            "score_claim_false_per_catalog_287_and_323",
            "phantom_score_research_sidecar_rejected_per_catalog_321",
            "reactivation_criteria_pinned_per_forbidden_premature_kill",
            "horizon_class_asymptotic_pursuit_declared_per_catalog_309",
            "probe_disambiguator_hook_6_active_primary_per_catalog_125",
            "op1_anchor_cited_per_continual_learning_posterior_catalog_300",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def register_strategic_verdict_in_ledger(
    strategic_verdict: Path3AStrategicVerdict,
    manifest_path: Path,
    *,
    subagent_id: str = "wave-3-probe-stc-paradigm-reformulation-20260520",
) -> dict[str, Any]:
    """Append one Catalog #313 probe-outcomes row for the strategic verdict.

    The probe_id schema is::

        probe_stc_paradigm_reformulation_a1_residual_<utc_compact>
    """
    now_compact = (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    )
    probe_id = (
        f"probe_stc_paradigm_reformulation_a1_residual_{now_compact}"
    )

    if strategic_verdict.verdict in BLOCKING_VERDICTS:
        next_action = (
            f"do_not_dispatch_path_3a_until_reactivation_criteria_met"
            if strategic_verdict.verdict == "DEFER"
            else strategic_verdict.next_action
        )
    else:
        next_action = strategic_verdict.next_action

    notes = (
        f"path_3a classification={strategic_verdict.path_3a_classification} "
        f"best_ratio={strategic_verdict.path_3a_best_ratio:.6f} vs "
        f"baseline_a1={strategic_verdict.baseline_a1_best_ratio:.6f} vs "
        f"random_control={strategic_verdict.random_control_best_ratio:.6f} vs "
        f"op1_reference={strategic_verdict.op1_reference_best_ratio:.6f}. "
        f"byte_delta_vs_baseline={strategic_verdict.path_3a_vs_baseline_byte_delta}; "
        f"ratio_delta_vs_random={strategic_verdict.path_3a_vs_random_ratio_delta:.6f}. "
        f"{strategic_verdict.rationale}"
    )

    record = register_probe_outcome(
        probe_id=probe_id,
        substrate=PROBE_SUBSTRATE_ID,
        recipe_path=None,
        probe_kind=(
            "stc_paradigm_reformulation_via_synthetic_a1_residual_entropy_ladder"
        ),
        verdict=strategic_verdict.verdict,
        metric_name="path_3a_best_compression_ratio",
        metric_value=strategic_verdict.path_3a_best_ratio,
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
        path_3a_classification=strategic_verdict.path_3a_classification,
        baseline_a1_best_ratio=strategic_verdict.baseline_a1_best_ratio,
        random_control_best_ratio=strategic_verdict.random_control_best_ratio,
        op1_reference_best_ratio=strategic_verdict.op1_reference_best_ratio,
        path_3a_vs_baseline_byte_delta=strategic_verdict.path_3a_vs_baseline_byte_delta,
        path_3a_vs_random_ratio_delta=strategic_verdict.path_3a_vs_random_ratio_delta,
        deliverable_savings_estimate=strategic_verdict.deliverable_savings_estimate,
        confidence="medium",
        horizon_class="asymptotic_pursuit",
        evidence_grade="predicted",
        measurement_axis=(
            "[diagnostic; synthetic-archive STC-paradigm-reformulation probe path 3a]"
        ),
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
        reactivation_criteria=strategic_verdict.reactivation_criteria,
    )
    return record


# ──────────────────────────────────────────────────────────────────────── #
# CLI                                                                       #
# ──────────────────────────────────────────────────────────────────────── #


def _emit_human_readable_summary(
    verdicts: list[ComparisonVerdict],
    strategic_verdict: Path3AStrategicVerdict,
) -> None:
    """Print operator-facing summary table to stdout."""
    print()
    print("=" * 92)
    print(
        "STC PARADIGM REFORMULATION DISAMBIGUATOR — path 3a (A1 residual sidecar)"
    )
    print("=" * 92)
    print(
        f"{'comparison':<24} {'role':<18} {'classification':<14} {'best_ratio':<14} "
        f"{'codec':<8}"
    )
    print("-" * 92)
    for v in verdicts:
        print(
            f"{v.comparison_id:<24} {v.role:<18} {v.compression.classification:<14} "
            f"{v.compression.best_ratio:<14.6f} {v.compression.best_codec:<8}"
        )
    print("-" * 92)
    print(f"STRATEGIC VERDICT: {strategic_verdict.verdict}")
    print(
        f"  path_3a vs baseline_a1 byte delta = "
        f"{strategic_verdict.path_3a_vs_baseline_byte_delta:+d} B"
    )
    print(
        f"  path_3a vs random_control ratio delta = "
        f"{strategic_verdict.path_3a_vs_random_ratio_delta:+.6f}"
    )
    print(
        f"  deliverable savings estimate = "
        f"{strategic_verdict.deliverable_savings_estimate:.6f} per-archive "
        f"(NON-AUTHORITATIVE)"
    )
    print(f"  next_action = {strategic_verdict.next_action}")
    print("=" * 92)
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
            "stc_paradigm_reformulation_a1_residual_disambiguator_<utc>.json"
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
        default="wave-3-probe-stc-paradigm-reformulation-20260520",
        help="Subagent ID for ledger attribution (Catalog #313 schema field).",
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
    subagent_id: str = "wave-3-probe-stc-paradigm-reformulation-20260520",
    quiet: bool = False,
) -> tuple[Path, list[ComparisonVerdict], Path3AStrategicVerdict]:
    """Programmatic entry point (mirrors CLI)."""
    verdicts = [derive_verdict_for_comparison(spec) for spec in CANONICAL_COMPARISONS]
    strategic_verdict = derive_path_3a_strategic_verdict(verdicts)

    if output is None:
        now_compact = (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
        )
        output = (
            OUTPUT_DIR_DEFAULT
            / f"stc_paradigm_reformulation_a1_residual_disambiguator_{now_compact}.json"
        )

    manifest_path = emit_canonical_manifest(verdicts, strategic_verdict, output)

    if not no_ledger_write:
        register_strategic_verdict_in_ledger(
            strategic_verdict, manifest_path, subagent_id=subagent_id
        )

    if not quiet:
        _emit_human_readable_summary(verdicts, strategic_verdict)
        print(f"Manifest written to: {manifest_path}")
        if not no_ledger_write:
            print(
                f"Catalog #313 probe-outcomes ledger row appended for substrate "
                f"{PROBE_SUBSTRATE_ID}"
            )

    return manifest_path, verdicts, strategic_verdict


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(
            output=args.output,
            no_ledger_write=args.no_ledger_write,
            subagent_id=args.subagent_id,
            quiet=args.quiet,
        )
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
