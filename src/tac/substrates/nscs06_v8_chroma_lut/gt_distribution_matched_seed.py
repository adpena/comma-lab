# SPDX-License-Identifier: MIT
"""GT-distribution-matched chroma LUT seed derivation (Path 3 C' Phase 3 — UNWIND cargo-cult #3).

Per Path 3 C' Phase 2 substrate-design decision memo
(``.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md``
commit ``bac0ec05d``) Section 3b.

**Cargo-cult #3 (CARGO-CULTED-CRITICAL)**: the existing v8 procedural-seed
path (``src/tac/substrates/nscs06_v8_chroma_lut/procedural_variant.py``)
derives the 4096-byte chroma LUT from a 32-byte PCG64 seed via the canonical
``tac.procedural_codebook_generator.derive_codebook_from_seed`` helper. The
resulting LUT bytes are statistically UNIFORM (PCG64 produces uniform
random bytes regardless of GT input). GT chroma distributions in dashcam
footage are NOT uniform (asphalt-gray + sky-blue + foliage-green clustering).

The structural mismatch between PCG64-uniform-LUT and GT-non-uniform-chroma
is the same cargo-cult META-class as NSCS06 v6 cargo-cult #2 (Y=R=G=B
chroma destruction at seg=64.59), just at lower magnitude. The 6 prior
v8 Modal dispatches (rc=22/rc=1) are receipts that the unmodeled
chroma-distribution-mismatch term may DOMINATE the rate-axis savings.

This module provides the canonical UNWIND alternative arm:
**hash-derived seed from GT LUT bytes**. The 32-byte seed is the SHA-256
hash of the GT-derived chroma LUT bytes, truncated to 32 bytes. At inflate,
the seed is the canonical PCG64 input — but unlike PCG64-uniform-random-seed,
the seed bytes ENCODE the GT-distribution fingerprint via the SHA-256
input. The empirical question is whether 32 bytes of GT-fingerprint is
sufficient for the canonical ``derive_codebook_from_seed`` PCG64 expansion
to produce a chroma LUT that recovers the GT distribution within SegNet's
noise floor.

**Honest disclosure** per CLAUDE.md "Apples-to-apples evidence discipline":
This is an ALTERNATIVE arm, NOT a guaranteed unwind. The empirical question
is whether GT-fingerprint-keyed PCG64 produces a distinguishable-from-PCG64-
uniform inflate output AND whether that distinguishability moves the
SegNet argmax toward correct class boundaries. If the empirical answer is
"no", the unwind path is FALSIFIED-AT-IMPLEMENTATION-LEVEL per Catalog #307
(paradigm-vs-implementation classification) and the next research path is:
replace PCG64 expansion with a chroma-distribution-aware codebook generator
(future Path 4).

**Compliance posture** per
``.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md``
Q4 STRUCTURALLY COMPLIANT verdict:
- Seed bytes live INSIDE archive 0.bin (same CH08 v2 LUT_PAYLOAD slot;
  shape-and-byte-stable with PCG64-uniform-seed path).
- Derivation routine ``derive_codebook_from_seed`` remains the canonical
  procedural-codebook helper.
- Distinguishable from rejected loophole pattern: the seed is NOT a
  literal-stored-outside-archive; it lives in the same in-archive slot
  as the PCG64-uniform-seed variant.

CLAUDE.md compliance:
- Catalog #290 canonical-vs-unique decision per layer: FORK cargo-cult #3
  with explicit hash-derived alternative; PRESERVE PCG64 expansion API
- Catalog #297 signal-axis-destruction reversibility: this module IS the
  alternative-arm reversibility surface
- Catalog #287 + #323 canonical Provenance: no score claim; outputs are
  deterministic bytes
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: NEW module; no
  mutation of sister `procedural_variant.py`
- Catalog #344 canonical equation cross-reference: canonical equation #26
  IN-DOMAIN context `nscs06_v8_chroma_lut` (same as PCG64-uniform-seed
  variant; this module REPLACES the seed-derivation upstream of the
  canonical helper; downstream remains canonical)

6-hook wire-in declaration per Catalog #125:
* hook #1 sensitivity-map = N/A (deterministic derivation; no sensitivity contribution)
* hook #2 Pareto constraint = ACTIVE (same canonical equation #26 IN-DOMAIN ΔS = -0.002706 per byte savings)
* hook #3 bit-allocator = N/A (same 32-byte seed slot as PCG64 variant)
* hook #4 cathedral autopilot dispatch = ACTIVE (auto-discovered alternative arm via canonical equation #26 consumer)
* hook #5 continual-learning posterior = ACTIVE PRIMARY (first paired-smoke empirical anchor for hash-derived-seed arm answers cargo-cult #3 unwind question)
* hook #6 probe-disambiguator = ACTIVE (hash-derived vs PCG64-uniform IS the probe disambiguator for cargo-cult #3)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from tac.procedural_codebook_generator import (
    DEFAULT_GENERATOR_KIND,
    derive_codebook_from_seed,
)

from .architecture import (
    CHROMA_LUT_BYTES_DEFAULT,
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PROCEDURAL_SEED_SIZE_BYTES,
)

__all__ = [
    "GtDistributionMatchedSeedError",
    "GtDistributionMatchedSeedVerdict",
    "SUPPORTED_SEED_DERIVATION_KINDS",
    "derive_chroma_lut_seed_from_gt_lut_bytes",
    "expand_gt_matched_seed_to_lut",
    "verify_seed_encodes_gt_fingerprint",
]


SUPPORTED_SEED_DERIVATION_KINDS: Final[tuple[str, ...]] = (
    "sha256_truncated",
    "blake2b_truncated",
)
"""Canonical seed-derivation kinds for the GT-distribution-matched arm.

* ``sha256_truncated``: SHA-256 of GT LUT bytes, truncated to seed_size
  (default 32). Deterministic, byte-stable, cross-architecture stable.
* ``blake2b_truncated``: BLAKE2b of GT LUT bytes, truncated to seed_size.
  Faster than SHA-256 with comparable distribution; sister alternative for
  benchmark-cost ablation per future research path.

PCG64-uniform-random-seed is the existing canonical PROCEDURAL VARIANT path
(``procedural_variant.py``); this module covers GT-fingerprint alternative
arms ONLY. Adding new kinds requires a sister Phase 4 audit per cargo-cult
discipline.
"""


class GtDistributionMatchedSeedError(RuntimeError):
    """Raised when GT-distribution-matched seed derivation cannot be honored faithfully."""


@dataclass(frozen=True)
class GtDistributionMatchedSeedVerdict:
    """Verdict from the GT-distribution-matched seed derivation arm.

    Carries:
    - the deterministic seed bytes (32 bytes by canonical contract)
    - the LUT bytes derived from the seed via canonical PCG64 expansion
    - per-byte sha256 fingerprints for downstream Provenance
    - non-promotable contract (research-signal axis; no score claim)
    """

    seed_kind: str
    seed_bytes: bytes
    seed_sha256: str
    derived_lut_bytes: bytes
    derived_lut_sha256: str
    gt_lut_input_sha256: str
    """SHA-256 of the GT chroma LUT bytes that produced the seed."""

    # Canonical non-promotable contract per Catalog #287 + #323
    axis_tag: str = "[macOS-MLX research-signal]"
    score_claim: bool = False
    promotion_eligible: bool = False
    evidence_grade: str = "research-signal"

    def __post_init__(self) -> None:
        if self.seed_kind not in SUPPORTED_SEED_DERIVATION_KINDS:
            raise GtDistributionMatchedSeedError(
                f"seed_kind={self.seed_kind!r} not in {SUPPORTED_SEED_DERIVATION_KINDS}"
            )
        if len(self.seed_bytes) != PROCEDURAL_SEED_SIZE_BYTES:
            raise GtDistributionMatchedSeedError(
                f"seed_bytes length {len(self.seed_bytes)} != "
                f"PROCEDURAL_SEED_SIZE_BYTES {PROCEDURAL_SEED_SIZE_BYTES}"
            )
        if self.score_claim is not False:
            raise GtDistributionMatchedSeedError(
                "score_claim MUST be False for GT-matched seed verdict"
            )
        if self.promotion_eligible is not False:
            raise GtDistributionMatchedSeedError(
                "promotion_eligible MUST be False for GT-matched seed verdict"
            )
        if self.evidence_grade != "research-signal":
            raise GtDistributionMatchedSeedError(
                f"evidence_grade MUST be 'research-signal'; got {self.evidence_grade!r}"
            )

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict per Catalog #287 + #323."""
        return {
            "seed_kind": self.seed_kind,
            "seed_sha256": self.seed_sha256,
            "derived_lut_sha256": self.derived_lut_sha256,
            "gt_lut_input_sha256": self.gt_lut_input_sha256,
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "evidence_grade": self.evidence_grade,
        }


def derive_chroma_lut_seed_from_gt_lut_bytes(
    gt_chroma_lut_bytes: bytes,
    *,
    seed_size: int = PROCEDURAL_SEED_SIZE_BYTES,
    kind: Literal["sha256_truncated", "blake2b_truncated"] = "sha256_truncated",
) -> bytes:
    """Derive a deterministic 32-byte seed from GT chroma LUT bytes.

    THIS IS THE CANONICAL CARGO-CULT #3 UNWIND ARM per Path 3 C' Phase 2
    Section 3b. The seed bytes ENCODE a 32-byte fingerprint of the GT LUT
    distribution; downstream ``derive_codebook_from_seed`` PCG64 expansion
    produces uniform-looking LUT bytes BUT keyed by the GT fingerprint.

    The empirical question (answered post-paired-smoke per Catalog #324):
    is 32 bytes of GT-fingerprint sufficient for PCG64 expansion to produce
    a chroma LUT that SegNet's stride-2 stem recovers as class-boundary-
    distinguishing? If yes, the substrate's score-axis decomposition reveals
    a NEW canonical equation IN-DOMAIN context. If no, the cargo-cult-unwind
    moves to Path 4 (replace PCG64 with chroma-distribution-aware generator).

    Args:
        gt_chroma_lut_bytes: GT-derived chroma LUT bytes (typically the
            (16 * 5 * 3 = 240)-byte dense LUT from
            :func:`tac.substrates.nscs06_v8_chroma_lut.architecture.build_chroma_lut_from_ground_truth`).
            MUST be non-empty bytes-like.
        seed_size: target seed size in bytes (default 32 per canonical
            CH08 v2 LUT_PAYLOAD slot).
        kind: derivation kind in :data:`SUPPORTED_SEED_DERIVATION_KINDS`.

    Returns:
        Deterministic ``seed_size``-byte seed bytes encoding the GT LUT
        fingerprint.

    Raises:
        GtDistributionMatchedSeedError: if input is empty, seed_size is
            outside [8, 256], or kind is not supported.
    """
    if not isinstance(gt_chroma_lut_bytes, (bytes, bytearray, memoryview)):
        raise GtDistributionMatchedSeedError(
            f"gt_chroma_lut_bytes must be bytes-like; got "
            f"{type(gt_chroma_lut_bytes).__name__}"
        )
    gt_bytes = bytes(gt_chroma_lut_bytes)
    if not gt_bytes:
        raise GtDistributionMatchedSeedError("gt_chroma_lut_bytes must be non-empty")
    if seed_size < 8 or seed_size > 256:
        raise GtDistributionMatchedSeedError(
            f"seed_size={seed_size} outside [8, 256] (canonical PROCEDURAL_SEED range)"
        )
    if kind not in SUPPORTED_SEED_DERIVATION_KINDS:
        raise GtDistributionMatchedSeedError(
            f"kind={kind!r} not in {SUPPORTED_SEED_DERIVATION_KINDS}"
        )

    if kind == "sha256_truncated":
        full_hash = hashlib.sha256(gt_bytes).digest()
        # SHA-256 produces 32 bytes; if seed_size > 32, extend via re-hash
        # of (prior_seed + gt_bytes) to preserve GT-distribution-encoding.
        while len(full_hash) < seed_size:
            full_hash += hashlib.sha256(full_hash + gt_bytes).digest()
        return full_hash[:seed_size]
    elif kind == "blake2b_truncated":
        # BLAKE2b supports arbitrary digest_size up to 64 bytes natively.
        if seed_size <= 64:
            return hashlib.blake2b(gt_bytes, digest_size=seed_size).digest()
        # For seed_size > 64, extend via re-hash.
        full_hash = hashlib.blake2b(gt_bytes, digest_size=64).digest()
        while len(full_hash) < seed_size:
            full_hash += hashlib.blake2b(
                full_hash + gt_bytes, digest_size=64
            ).digest()
        return full_hash[:seed_size]
    else:
        # Defensive — should be unreachable due to validation above.
        raise GtDistributionMatchedSeedError(f"unreachable: unsupported kind {kind!r}")


def expand_gt_matched_seed_to_lut(
    seed_bytes: bytes,
    *,
    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT,
    num_segnet_classes: int = NUM_SEGNET_CLASSES,
    generator_kind: Literal["xorshift", "lcg", "pcg64"] = DEFAULT_GENERATOR_KIND,
) -> np.ndarray:
    """Expand a GT-matched seed via canonical PCG64 to a (levels, classes, 3) LUT.

    Sister of canonical
    ``tac.procedural_codebook_generator.derive_codebook_from_seed`` — same
    expansion logic, but the seed argument is GT-derived per
    :func:`derive_chroma_lut_seed_from_gt_lut_bytes` rather than uniform-random.

    Args:
        seed_bytes: 32-byte GT-derived seed (typically from
            :func:`derive_chroma_lut_seed_from_gt_lut_bytes`).
        grayscale_levels: LUT level dimension (default 16).
        num_segnet_classes: LUT class dimension (default 5).
        generator_kind: PRNG family (default pcg64 per canonical).

    Returns:
        ``(grayscale_levels, num_segnet_classes, 3)`` uint8 LUT.
    """
    dense_bytes = grayscale_levels * num_segnet_classes * 3
    flat = derive_codebook_from_seed(
        seed_bytes=seed_bytes,
        output_shape=(dense_bytes,),
        dtype=np.uint8,
        generator_kind=generator_kind,
    )
    return flat.reshape(grayscale_levels, num_segnet_classes, 3)


def verify_seed_encodes_gt_fingerprint(
    gt_chroma_lut_bytes_a: bytes,
    gt_chroma_lut_bytes_b: bytes,
    *,
    kind: Literal["sha256_truncated", "blake2b_truncated"] = "sha256_truncated",
) -> bool:
    """Return True iff two DIFFERENT GT LUT inputs produce DIFFERENT seeds.

    Catalog #272 distinguishing-feature integration contract sister: the
    GT-matched seed MUST encode the GT fingerprint such that different GT
    inputs produce different seeds. Returns True for distinct inputs that
    produce distinct seeds; False for identical inputs (which correctly
    produce identical seeds) OR distinct inputs that collide (rare; sha256
    collision in 32-byte truncation has ~2^128 search cost).

    Args:
        gt_chroma_lut_bytes_a: first GT chroma LUT bytes.
        gt_chroma_lut_bytes_b: second GT chroma LUT bytes (typically a
            mutated variant of the first).
        kind: seed-derivation kind.

    Returns:
        True if inputs differ AND derived seeds differ; False otherwise.
    """
    if gt_chroma_lut_bytes_a == gt_chroma_lut_bytes_b:
        return False
    seed_a = derive_chroma_lut_seed_from_gt_lut_bytes(gt_chroma_lut_bytes_a, kind=kind)
    seed_b = derive_chroma_lut_seed_from_gt_lut_bytes(gt_chroma_lut_bytes_b, kind=kind)
    return seed_a != seed_b
