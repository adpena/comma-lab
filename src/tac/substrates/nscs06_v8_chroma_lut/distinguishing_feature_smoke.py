# SPDX-License-Identifier: MIT
"""Per-class chroma byte-mutation distinguishing-feature smoke (Path 3 C' Phase 3 — UNWIND cargo-cult #5).

Per Path 3 C' Phase 2 substrate-design decision memo
(``.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md``
commit ``bac0ec05d``) + Phase 1 adversarial cargo-cult audit
(``.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md``
commit ``a6e2a06e3``).

**Cargo-cult #5 (CARGO-CULTED-CRITICAL)**: L0 SCAFFOLD inflate uses
``cls_full = np.zeros_like(gray_full, dtype=np.uint8)`` (per
``src/tac/substrates/nscs06_v8_chroma_lut/inflate.py:185``). This collapses
the v8 ``(16, 5, 3)`` chroma LUT to per-(level, class=0) only at inflate
time. The 4 other class anchors LUT[:, c, :] for c in {1, 2, 3, 4} are
NEVER consumed at inflate; their bytes are dead. A byte-mutation smoke per
Catalog #272 distinguishing-feature integration contract would PASS for
class=0 mutations but FAIL for class>=1 mutations — STRUCTURAL TEST
INVALIDITY at L0.

**This module IS the canonical reversibility-test surface for v8.** Per
Catalog #297 signal-axis-destruction reversibility audit + Catalog #220
substrate L1+ operational mechanism contract: the per-class chroma anchor
bytes MUST produce frame-level changes when mutated, otherwise the v8
distinguishing feature is structurally vacuous at the empirical anchor.

The helper :func:`verify_per_class_chroma_anchors_consumed_at_inflate`
takes a CH08 v1 archive (inline LUT path), mutates LUT[:, c, :] bytes for
each class c in ``classes_to_mutate``, runs inflate, and compares the
mutated frame-1 RGB bytes vs the unmutated baseline. The verdict reports
per-class status:

- ``PASS_PER_CLASS`` (all mutated classes produce frame changes)
- ``FAIL_AT_CLASS_<c>`` (one or more mutated classes produce identical frames)

The expected verdict at L0 SCAFFOLD (cls=0 uniform inflate) is
``FAIL_AT_CLASS_1`` (mutating class=1 anchor does NOT change frames because
class=0 uniformly is consumed). This verdict EMPIRICALLY confirms cargo-cult
#5; the L1 promotion blocker is to wire cls_stream consumption at inflate
so the verdict becomes ``PASS_PER_CLASS``.

CLAUDE.md compliance:
- Catalog #297 signal-axis-destruction reversibility audit
- Catalog #272 distinguishing-feature integration contract
- Catalog #220 substrate L1+ scaffold operational mechanism
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (new module; no mutation of sister)
- Catalog #287 + #323 canonical Provenance (no score claim; verdict is research-signal only)
- HNeRV parity discipline L11 no-op detector (this IS the canonical no-op detector for v8)

6-hook wire-in declaration per Catalog #125:
* hook #1 sensitivity-map = ACTIVE (per-class chroma anchor consumption IS the per-class sensitivity surface)
* hook #2 Pareto constraint = N/A (per-class smoke is a structural-correctness verifier; no rate/seg/pose decomposition contribution)
* hook #3 bit-allocator = N/A (verifier; no bit allocation)
* hook #4 cathedral autopilot dispatch = ACTIVE (verdict consumable by ranker as canonical structural disambiguator between L0 vs L1 inflate state)
* hook #5 continual-learning posterior = N/A (verdict is per-archive structural; not a posterior anchor)
* hook #6 probe-disambiguator = ACTIVE PRIMARY (this IS the canonical disambiguator between cls=0-uniform-L0 vs cls_stream-consumed-L1 inflate)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .archive import (
    CH08_HEADER_SIZE,
    CH08_SCHEMA_VERSION_INLINE_LUT,
    parse_archive,
)
from .inflate import inflate_one_video

__all__ = [
    "DistinguishingFeatureSmokeError",
    "PerClassChromaDistinguishingFeatureVerdict",
    "PER_CLASS_SMOKE_NON_PROMOTABLE_PROVENANCE",
    "compute_lut_byte_offset_for_class",
    "mutate_class_anchor_bytes_in_archive",
    "verify_per_class_chroma_anchors_consumed_at_inflate",
]


PER_CLASS_SMOKE_NON_PROMOTABLE_PROVENANCE: dict[str, Any] = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "axis_tag": "[structural-verifier]",
    "evidence_grade": "research-signal",
    "blockers": (
        "per_class_chroma_smoke_is_structural_verifier_not_score_claim",
        "verdict_disambiguates_L0_vs_L1_inflate_state_only",
    ),
}
"""Canonical non-promotable provenance per Catalog #287 + #323."""


class DistinguishingFeatureSmokeError(RuntimeError):
    """Raised when the per-class chroma byte-mutation smoke cannot be honored faithfully."""


@dataclass(frozen=True)
class PerClassChromaDistinguishingFeatureVerdict:
    """Per-class verdict from the chroma anchor byte-mutation smoke.

    Carries the canonical non-promotable contract per CLAUDE.md "Apples-to-
    apples evidence discipline" + Catalog #287 + #323. The verdict
    structurally disambiguates between L0 SCAFFOLD (cls=0 uniform inflate;
    expected verdict ``FAIL_AT_CLASS_1``) and L1 INTEGRATION (cls_stream
    consumed at inflate; expected verdict ``PASS_PER_CLASS``).
    """

    verdict_kind: str
    """One of ``PASS_PER_CLASS`` or ``FAIL_AT_CLASS_<c>`` (string)."""

    classes_mutated: tuple[int, ...]
    """Classes for which the LUT[:, c, :] bytes were mutated."""

    classes_with_frame_changes: tuple[int, ...]
    """Classes whose mutation produced frame-1 RGB byte changes."""

    classes_without_frame_changes: tuple[int, ...]
    """Classes whose mutation produced byte-identical frames (NO consumption)."""

    baseline_frame1_sha256: str
    """SHA-256 of unmutated frame-1 RGB bytes (canonical reference)."""

    mutated_frame1_sha256_per_class: dict[int, str]
    """Per-class SHA-256 of mutated frame-1 RGB bytes."""

    expected_l0_verdict: str = "FAIL_AT_CLASS_1"
    """Documented expected verdict at L0 SCAFFOLD per Phase 1 audit cargo-cult #5."""

    expected_l1_verdict: str = "PASS_PER_CLASS"
    """Documented expected verdict at L1 INTEGRATION (cls_stream consumed)."""

    # Canonical non-promotable contract
    axis_tag: str = "[structural-verifier]"
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    rank_or_kill_eligible: bool = False
    evidence_grade: str = "research-signal"
    blockers: tuple[str, ...] = field(
        default_factory=lambda: (
            "per_class_chroma_smoke_is_structural_verifier_not_score_claim",
            "verdict_disambiguates_L0_vs_L1_inflate_state_only",
        )
    )

    def __post_init__(self) -> None:
        # Reject any construction attempt that weakens the non-promotable contract.
        if self.score_claim is not False:
            raise DistinguishingFeatureSmokeError(
                "score_claim MUST be False for per-class chroma smoke verdict"
            )
        if self.promotion_eligible is not False:
            raise DistinguishingFeatureSmokeError(
                "promotion_eligible MUST be False for per-class chroma smoke verdict"
            )
        if self.ready_for_exact_eval_dispatch is not False:
            raise DistinguishingFeatureSmokeError(
                "ready_for_exact_eval_dispatch MUST be False for per-class chroma smoke verdict"
            )
        if self.rank_or_kill_eligible is not False:
            raise DistinguishingFeatureSmokeError(
                "rank_or_kill_eligible MUST be False for per-class chroma smoke verdict"
            )
        if self.axis_tag != "[structural-verifier]":
            raise DistinguishingFeatureSmokeError(
                f"axis_tag MUST be '[structural-verifier]'; got {self.axis_tag!r}"
            )
        if self.evidence_grade != "research-signal":
            raise DistinguishingFeatureSmokeError(
                f"evidence_grade MUST be 'research-signal'; got {self.evidence_grade!r}"
            )
        # Verify verdict_kind structural format.
        if self.verdict_kind != "PASS_PER_CLASS" and not self.verdict_kind.startswith(
            "FAIL_AT_CLASS_"
        ):
            raise DistinguishingFeatureSmokeError(
                f"verdict_kind must be 'PASS_PER_CLASS' or 'FAIL_AT_CLASS_<c>'; "
                f"got {self.verdict_kind!r}"
            )
        # Verify class lists partition cleanly.
        all_mutated = frozenset(self.classes_mutated)
        union = frozenset(self.classes_with_frame_changes) | frozenset(
            self.classes_without_frame_changes
        )
        if union != all_mutated:
            raise DistinguishingFeatureSmokeError(
                f"classes_with_frame_changes ∪ classes_without_frame_changes = "
                f"{sorted(union)} != classes_mutated {sorted(all_mutated)}"
            )
        if frozenset(self.classes_with_frame_changes) & frozenset(
            self.classes_without_frame_changes
        ):
            raise DistinguishingFeatureSmokeError(
                "classes_with_frame_changes and classes_without_frame_changes must be disjoint"
            )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict per Catalog #287 + #323 canonical Provenance."""
        return {
            "verdict_kind": self.verdict_kind,
            "classes_mutated": list(self.classes_mutated),
            "classes_with_frame_changes": list(self.classes_with_frame_changes),
            "classes_without_frame_changes": list(self.classes_without_frame_changes),
            "baseline_frame1_sha256": self.baseline_frame1_sha256,
            "mutated_frame1_sha256_per_class": dict(self.mutated_frame1_sha256_per_class),
            "expected_l0_verdict": self.expected_l0_verdict,
            "expected_l1_verdict": self.expected_l1_verdict,
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "rank_or_kill_eligible": self.rank_or_kill_eligible,
            "evidence_grade": self.evidence_grade,
            "blockers": list(self.blockers),
        }


def compute_lut_byte_offset_for_class(
    class_index: int,
    *,
    grayscale_levels: int,
    num_segnet_classes: int,
) -> tuple[int, int]:
    """Return ``(start_byte, length_bytes)`` for LUT[:, class_index, :] in payload.

    The v8 inline-LUT payload bytes are layout ``LUT[lvl, cls, ch]`` in C-order
    (per ``architecture.py`` and ``np.ndarray.tobytes()`` default). For a
    given class_index c, the byte slice is:

        LUT[lvl, c, ch] = payload_byte[lvl * num_segnet_classes * 3 + c * 3 + ch]

    Per-class slice covers ``grayscale_levels`` (lvl) × 3 (ch) = ``grayscale_levels * 3``
    BYTES SCATTERED across ``grayscale_levels`` slices of size 3. Since the layout
    is strided, this helper returns the (start_byte, total_byte_span_inclusive)
    that ENCOMPASSES every byte of the class anchor; the caller must use the
    canonical numpy-view stride pattern for surgical per-class mutation.

    For mutation simplicity, this helper returns the encompassing span; the
    canonical surgical mutator :func:`mutate_class_anchor_bytes_in_archive` uses
    numpy reshape + slice + tobytes for byte-correct per-class mutation.

    Args:
        class_index: class index in [0, num_segnet_classes).
        grayscale_levels: LUT level dimension (default 16 per CH08).
        num_segnet_classes: LUT class dimension (default 5 per SegNet).

    Returns:
        (start_byte_in_payload, span_bytes_inclusive). start_byte is the
        offset within the LUT_PAYLOAD region (not the full archive bytes;
        caller must add CH08_HEADER_SIZE).
    """
    if class_index < 0 or class_index >= num_segnet_classes:
        raise DistinguishingFeatureSmokeError(
            f"class_index={class_index} outside [0, {num_segnet_classes})"
        )
    if grayscale_levels < 1:
        raise DistinguishingFeatureSmokeError(
            f"grayscale_levels={grayscale_levels} must be >= 1"
        )
    if num_segnet_classes < 1:
        raise DistinguishingFeatureSmokeError(
            f"num_segnet_classes={num_segnet_classes} must be >= 1"
        )
    # Per-class span: from (lvl=0, cls=c, ch=0) to (lvl=grayscale_levels-1, cls=c, ch=2).
    # In C-order flat indexing: offset(lvl, c, ch) = (lvl * num_classes + c) * 3 + ch.
    start = (0 * num_segnet_classes + class_index) * 3 + 0
    end_inclusive = (
        (grayscale_levels - 1) * num_segnet_classes + class_index
    ) * 3 + 2
    span = end_inclusive - start + 1
    return (start, span)


def mutate_class_anchor_bytes_in_archive(
    archive_bytes: bytes,
    class_index: int,
    *,
    xor_byte: int = 0x55,
) -> bytes:
    """Return a copy of ``archive_bytes`` with LUT[:, class_index, :] XOR-mutated.

    Only mutates bytes belonging to ``class_index``'s anchor (strided
    per-class slice within the LUT_PAYLOAD region). Header + pose + grayscale
    + other-class anchors are byte-identical to ``archive_bytes``.

    Args:
        archive_bytes: the CH08 v1 INLINE LUT archive bytes.
        class_index: which class anchor to mutate (in [0, num_segnet_classes)).
        xor_byte: byte to XOR each per-class byte against (default 0x55).
            0x55 is the canonical "alternating-bits" mutation pattern that
            maximally flips bits without being all-zeros (which the inflate
            could plausibly default to).

    Returns:
        Mutated archive bytes (same length as input).

    Raises:
        DistinguishingFeatureSmokeError: if archive is not CH08 v1 inline-LUT
            (only v1 has inline LUT bytes to mutate at fixed offset).
    """
    if not isinstance(archive_bytes, (bytes, bytearray, memoryview)):
        raise DistinguishingFeatureSmokeError(
            f"archive_bytes must be bytes-like; got {type(archive_bytes).__name__}"
        )
    arc = parse_archive(bytes(archive_bytes))
    if arc.schema_version != CH08_SCHEMA_VERSION_INLINE_LUT:
        raise DistinguishingFeatureSmokeError(
            f"per-class mutation requires CH08 v1 inline-LUT archive "
            f"(schema_version={CH08_SCHEMA_VERSION_INLINE_LUT}); got "
            f"schema_version={arc.schema_version}. v2 procedural-seed "
            f"requires sister mutation surface (mutate seed bytes via "
            f"`verify_seed_mutation_changes_lut_bytes` in procedural_variant)."
        )
    if class_index < 0 or class_index >= arc.num_segnet_classes:
        raise DistinguishingFeatureSmokeError(
            f"class_index={class_index} outside [0, {arc.num_segnet_classes})"
        )

    # Reshape the dense LUT bytes (first dense_bytes of LUT_PAYLOAD) to (lvl, cls, 3)
    # for surgical per-class mutation; preserve padding/rest of archive as-is.
    dense_bytes = arc.grayscale_levels * arc.num_segnet_classes * 3
    payload_start = CH08_HEADER_SIZE
    dense_view = np.frombuffer(
        archive_bytes[payload_start : payload_start + dense_bytes], dtype=np.uint8
    ).copy().reshape(arc.grayscale_levels, arc.num_segnet_classes, 3)
    # XOR the per-class slice.
    mutated_view = dense_view.copy()
    mutated_view[:, class_index, :] ^= np.uint8(xor_byte)
    # Reassemble archive bytes.
    mutated_dense_bytes = mutated_view.tobytes()
    # Preserve any padding between dense_bytes and lut_payload_len.
    # (v1 declares chroma_lut_bytes including padding; the dense portion is
    # the first dense_bytes; padding is zero-filled per archive.py:266).
    chroma_payload_len = arc.chroma_lut_bytes
    padding_bytes = archive_bytes[
        payload_start + dense_bytes : payload_start + chroma_payload_len
    ]
    rest_after_payload = archive_bytes[payload_start + chroma_payload_len :]
    mutated_archive = (
        archive_bytes[:payload_start]
        + mutated_dense_bytes
        + padding_bytes
        + rest_after_payload
    )
    if len(mutated_archive) != len(archive_bytes):
        raise DistinguishingFeatureSmokeError(
            f"mutation reassembly produced wrong length: "
            f"{len(mutated_archive)} != {len(archive_bytes)}"
        )
    return bytes(mutated_archive)


def _compute_frame1_sha256_for_archive(
    archive_bytes: bytes,
    output_stem: Path,
) -> str:
    """Run inflate on archive_bytes; return SHA-256 of frame-1 RGB bytes only.

    Frame-1 is the second frame in each (frame_0, frame_1) pair. We isolate
    frame-1 because the LUT chroma anchors directly drive frame_0 (which is
    then warped to frame_1 via the 6-DOF affine warp). Frame-1 byte changes
    are downstream evidence of LUT consumption.

    Returns the SHA-256 hex digest of all frame-1 bytes concatenated.
    """
    raw_path = inflate_one_video(archive_bytes, output_stem)
    raw_bytes = raw_path.read_bytes()
    # raw_bytes layout: num_pairs * 2 * H * W * 3 (frame_0 + frame_1 per pair).
    # We need to extract every other frame block (the frame_1 slots).
    # Parse the archive to compute output_height / output_width / num_pairs.
    arc = parse_archive(archive_bytes)
    frame_bytes = arc.output_height * arc.output_width * 3
    expected_total = arc.num_pairs * 2 * frame_bytes
    if len(raw_bytes) != expected_total:
        raise DistinguishingFeatureSmokeError(
            f"inflate produced {len(raw_bytes)} bytes; expected {expected_total} "
            f"(num_pairs={arc.num_pairs} * 2 * H={arc.output_height} * W="
            f"{arc.output_width} * 3)"
        )
    frame1_chunks: list[bytes] = []
    for p in range(arc.num_pairs):
        # Each pair contributes (frame_0, frame_1); frame_1 starts at offset
        # 2 * p * frame_bytes + frame_bytes (i.e. after frame_0).
        start = 2 * p * frame_bytes + frame_bytes
        frame1_chunks.append(raw_bytes[start : start + frame_bytes])
    frame1_concat = b"".join(frame1_chunks)
    return hashlib.sha256(frame1_concat).hexdigest()


def verify_per_class_chroma_anchors_consumed_at_inflate(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    classes_to_mutate: tuple[int, ...] = (1, 2, 3, 4),
    xor_byte: int = 0x55,
) -> PerClassChromaDistinguishingFeatureVerdict:
    """Verify per-class chroma anchor byte-mutation produces frame-level changes.

    THIS IS THE CANONICAL CATALOG #272 + #297 + #220 EXPECTATION ARTIFACT for v8.

    For each class c in ``classes_to_mutate``, mutate LUT[:, c, :] bytes via
    :func:`mutate_class_anchor_bytes_in_archive`, run inflate on the mutated
    archive, and compare frame-1 RGB bytes vs the unmutated baseline.

    Expected verdicts:

    * **L0 SCAFFOLD** (cls=0 uniform inflate per current ``inflate.py:185``):
      ``FAIL_AT_CLASS_<smallest c in mutated>`` — mutating any class>=1 anchor
      produces IDENTICAL frame-1 bytes because cls_full is uniformly class=0
      at inflate.
    * **L1 INTEGRATION** (cls_stream consumed at inflate, future work):
      ``PASS_PER_CLASS`` — every per-class mutation produces frame changes.

    The L0 expected ``FAIL_AT_CLASS_1`` verdict IS the operational confirmation
    of Path 3 C' Phase 1 cargo-cult #5 (cls=0 uniform inflate structural
    test-invalidity). The L1 promotion blocker is to wire cls_stream
    consumption at inflate so the verdict becomes ``PASS_PER_CLASS``.

    Args:
        archive_bytes: CH08 v1 inline-LUT archive bytes (v2 procedural-seed
            requires sister mutation surface).
        output_dir: writable directory for inflate scratch outputs.
        classes_to_mutate: class indices to mutate (default (1, 2, 3, 4) —
            all non-zero classes per L0 cargo-cult expectation; pass (0,)
            for L0-expected baseline-pass case).
        xor_byte: byte to XOR each per-class byte against (default 0x55).

    Returns:
        :class:`PerClassChromaDistinguishingFeatureVerdict`.

    Raises:
        DistinguishingFeatureSmokeError: if archive is not v1 inline-LUT, or
            if inflate produces unexpected byte counts.
    """
    if not isinstance(archive_bytes, (bytes, bytearray, memoryview)):
        raise DistinguishingFeatureSmokeError(
            f"archive_bytes must be bytes-like; got {type(archive_bytes).__name__}"
        )
    archive_bytes_b = bytes(archive_bytes)
    arc = parse_archive(archive_bytes_b)
    if arc.schema_version != CH08_SCHEMA_VERSION_INLINE_LUT:
        raise DistinguishingFeatureSmokeError(
            f"verify_per_class_chroma_anchors_consumed_at_inflate requires "
            f"CH08 v1 inline-LUT archive (schema_version="
            f"{CH08_SCHEMA_VERSION_INLINE_LUT}); got "
            f"schema_version={arc.schema_version}"
        )
    if not classes_to_mutate:
        raise DistinguishingFeatureSmokeError(
            "classes_to_mutate must be non-empty"
        )
    for c in classes_to_mutate:
        if c < 0 or c >= arc.num_segnet_classes:
            raise DistinguishingFeatureSmokeError(
                f"class {c} in classes_to_mutate outside [0, {arc.num_segnet_classes})"
            )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Baseline: inflate unmutated archive; compute frame-1 SHA-256.
    baseline_stem = output_dir / "_baseline"
    baseline_sha = _compute_frame1_sha256_for_archive(archive_bytes_b, baseline_stem)

    classes_with_changes: list[int] = []
    classes_without_changes: list[int] = []
    mutated_shas: dict[int, str] = {}

    for c in classes_to_mutate:
        mutated_bytes = mutate_class_anchor_bytes_in_archive(
            archive_bytes_b, c, xor_byte=xor_byte
        )
        mutated_stem = output_dir / f"_mutated_class_{c}"
        mutated_sha = _compute_frame1_sha256_for_archive(mutated_bytes, mutated_stem)
        mutated_shas[c] = mutated_sha
        if mutated_sha != baseline_sha:
            classes_with_changes.append(c)
        else:
            classes_without_changes.append(c)

    # Determine verdict.
    if not classes_without_changes:
        verdict_kind = "PASS_PER_CLASS"
    else:
        # Report first-failing class for actionable diagnosis.
        first_failing = min(classes_without_changes)
        verdict_kind = f"FAIL_AT_CLASS_{first_failing}"

    return PerClassChromaDistinguishingFeatureVerdict(
        verdict_kind=verdict_kind,
        classes_mutated=tuple(classes_to_mutate),
        classes_with_frame_changes=tuple(classes_with_changes),
        classes_without_frame_changes=tuple(classes_without_changes),
        baseline_frame1_sha256=baseline_sha,
        mutated_frame1_sha256_per_class=mutated_shas,
    )
