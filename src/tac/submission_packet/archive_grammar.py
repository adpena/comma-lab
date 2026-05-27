# SPDX-License-Identifier: MIT
"""Layer 1 — canonical archive grammar builder per HNeRV parity L3.

Wrap per-substrate ``archive.zip`` building into one canonical entry point per
Phase 1 audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
Layer 1.

The bug class this layer extincts: ad-hoc per-substrate ``archive.zip`` builders
that drift on (i) HNeRV parity discipline L3 monolithic single-file ``0.bin``
declaration, (ii) fixed-offset declaration in source per Catalog #146 inflate
runtime contract, (iii) Catalog #139 packet compiler routing for no-op
detection, (iv) Catalog #105 no-op provenance proof, (v) Catalog #220
operational mechanism declaration, (vi) Catalog #272 distinguishing-feature
integration contract, (vii) Catalog #266 archive-bytes-consumed-by-inflate
empirical verification.

Per the 8th MLX-first numpy-portable individually-fractal standing directive:
the archive grammar is the canonical NUMPY-PORTABLE contract surface. Encoder
training is MLX-first on Apple Silicon; archive bytes are numpy-portable per
HNeRV parity L4 (≤200 LOC inflate.py + ≤2 ext deps).

Per the 11th ORDER-MATTERS standing directive: Layer 1 (this module) is the
SECOND Phase 1 spec consumer; depends on Layer 0
:class:`tac.submission_packet.compression_pipeline.CompressionPipelineResult`
dataclass shape; downstream Phase 4-10 layers depend on this module's
:class:`ArchiveGrammarManifest` shape.

Per the 12th canonicalization × standardization × ease-of-contest-compliance
trinity: ONE canonical helper, ONE return shape, ONE verification protocol.

The helper is OBSERVABILITY-ONLY by construction per Catalog #341 + CLAUDE.md
"Apples-to-apples evidence discipline". Every emitted
:class:`ArchiveGrammarManifest` carries ``score_claim=False`` +
``promotable=False`` + ``axis_tag=[predicted]``. Promotion of an archive-grammar
anchor to a contest score signal REQUIRES paired-CUDA + Linux x86_64 CPU
empirical anchor per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON
1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable (lands at Phase 6 / Phase 10).
"""
from __future__ import annotations

import datetime
import enum
import hashlib
import io
import json
import os
import socket
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac.submission_packet.compression_pipeline import (
    CompressionPipelineResult,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Module-level constants — canonical schemas + canonical fixed offsets
# ---------------------------------------------------------------------------

ARCHIVE_GRAMMAR_SCHEMA_VERSION = "archive_grammar_v1_20260526"
"""Pinned schema for :class:`ArchiveGrammarManifest` persistence rows."""

PHASE_3_LAYER_VERSION = "phase_3_archive_grammar_canonical_landed_20260526"
"""Operator-readable Phase 3 landing marker per Phase 1 audit spec memo."""

CANONICAL_EQUATION_ID = (
    "archive_grammar_canonical_consolidation_savings_v1"
)
"""Canonical equation registered per Phase 1 audit spec memo §13.

FORMALIZATION_PENDING until Phase 10 first-PR-through-canonical-pipeline
regression lands the first paired-CUDA empirical anchor of per-substrate
archive-grammar-divergence collapse (predicted: 14 per-substrate ad-hoc
builders consolidated to ONE canonical helper).
"""

# Per Catalog #341 routing markers (Tier A observability-only).
PREDICTED_AXIS_TAG = "[predicted]"

# Per Catalog #287 placeholder rejection.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset(
    {"<rationale>", "<reason>", "<rationale_here>", "<reason_here>", ""}
)

# Canonical monolithic single-file name per HNeRV parity L3.
CANONICAL_MONOLITHIC_MEMBER_NAME = "0.bin"

# Canonical archive file name (sister to upstream evaluate.py contract).
CANONICAL_ARCHIVE_NAME = "archive.zip"

# Catalog #266: minimum byte addition threshold for byte-mutation smoke
# applicability (matches Catalog #220 threshold for L1+ scaffold byte addition).
BYTE_MUTATION_SMOKE_MIN_BYTES = 1024


# Canonical section kinds per HNeRV parity L3 + L5 vocabulary. Adding a new
# kind requires landing the canonical inflate-runtime consumer in same commit
# batch (per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
# non-negotiable + Catalog #220 operational mechanism declaration).
class SectionKind(enum.StrEnum):
    """Canonical archive-section taxonomy.

    Per HNeRV parity L5 the architecture must be the FULL renderer (RGB out),
    not a single-component slot. Per HNeRV parity L3 the canonical archive is
    monolithic single-file ``0.bin`` (or explicitly justified multi-file).
    """

    DECODER_BLOB = "decoder_blob"
    """Trained renderer weights (FP4 / FP8 / FP16) per HNeRV parity L5."""

    LATENT_BLOB = "latent_blob"
    """HNeRV-class learned latent codes."""

    POSE_BLOB = "pose_blob"
    """Per-frame pose tensors (FastViT-T12 12-channel scorer input)."""

    MASK_BLOB = "mask_blob"
    """Per-frame mask logits / mask MKV bytes (EfficientNet-B2 scorer input)."""

    CDF_TABLE = "cdf_table"
    """Entropy coder CDF table (arithmetic / range / ANS / Huffman variant)."""

    HYPERPRIOR_WEIGHTS = "hyperprior_weights"
    """Ballé-style hyperprior weights (Catalog #266 sister)."""

    HEADER = "header"
    """Archive header bytes (magic + version + section offsets)."""

    SIDE_INFORMATION = "side_information"
    """Wyner-Ziv side-info per Catalog #319 deliverability proof."""

    DISTINGUISHING_FEATURE = "distinguishing_feature"
    """Per-substrate distinguishing feature bytes per Catalog #272."""

    OTHER = "other"
    """Substrate-engineering escape hatch; document via section_name + rationale."""


# Canonical operational mechanism status per Catalog #220.
class OperationalMechanismStatus(enum.StrEnum):
    """Per Catalog #220 substrate L1+ scaffold operational mechanism status."""

    OPERATIONAL = "OPERATIONAL"
    """Inflate runtime consumes the section bytes to modify rendered frames."""

    RESEARCH_ONLY = "RESEARCH_ONLY"
    """Per Catalog #220 opt-out: section is research-substrate-scope only."""

    PRE_BUILD_SCAFFOLD = "PRE_BUILD_SCAFFOLD"
    """Per Catalog #220 opt-out: trainer's _full_main raises NotImplementedError."""


# Canonical byte-mutation smoke verdict per Catalog #105 + #139 + #266 + #272.
class ByteMutationSmokeVerdict(enum.StrEnum):
    """Per Catalog #139 packet compiler no-op detector + Catalog #266."""

    PASSED = "PASSED"
    """Byte mutation produced frame changes (canonical PROOF of consumption)."""

    FAILED_BYTES_NOT_CONSUMED = "FAILED_BYTES_NOT_CONSUMED"
    """Catalog #266 violation: archive bytes structurally consumed but no
    frame changes resulted — the research-substrate trap (8th forbidden
    pattern per HNeRV parity discipline)."""

    NOT_RUN = "NOT_RUN"
    """Byte-mutation smoke not invoked (build-time scaffold; defer until
    Phase 6 paired_auth_eval lands)."""

    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    """Helper-side error (missing inflate.sh / missing archive / Python crash);
    distinct from Catalog #266 violation per Catalog #307 paradigm-vs-
    implementation classification."""


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ArchiveGrammarError(RuntimeError):
    """Archive grammar orchestration error.

    Sister of :class:`tac.submission_packet.compression_pipeline.CompressionPipelineError`
    + :class:`tac.deploy.modal.call_id_ledger.LedgerRegistrationFailedError`.
    Raised by :func:`build_archive_grammar_from_compression_pipeline_result`
    when the trainer + recipe pair cannot satisfy HNeRV parity L3 invariants
    AND no waiver is supplied.
    """


# ---------------------------------------------------------------------------
# Frozen dataclasses — canonical contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveSectionSpec:
    """Per-section descriptor for a canonical archive grammar.

    Per HNeRV parity L3: fixed offsets MUST be declared in source per
    canonical constants. This dataclass is the per-section runtime
    representation; the canonical constants live in the substrate's own
    archive.py per the substrate-engineering split (HNeRV parity L7).
    """

    section_name: str
    """Operator-readable section name (e.g. ``"decoder_blob"`` /
    ``"frame_exploit_selector_bits"`` per Catalog #272 distinguishing feature)."""

    offset_in_archive: int
    """Fixed byte offset in the monolithic single-file ``0.bin`` (HNeRV
    parity L3) OR offset within the named ZIP member when multi-file."""

    length_in_archive: int
    """Section length in bytes."""

    sha256_of_section: str
    """sha256 hex digest of section bytes for canonical reproducibility."""

    section_kind: str
    """One of :class:`SectionKind` values."""

    operational_mechanism_status: str
    """One of :class:`OperationalMechanismStatus` values (Catalog #220)."""

    distinguishing_feature_name: str | None = None
    """Catalog #272 distinguishing-feature name (e.g. ``"score_critical_bits"`` /
    ``"hyperprior_weights"`` / ``"frame_exploit_selector"``); None when the
    section is not a substrate's primary distinguishing feature."""

    member_name: str = CANONICAL_MONOLITHIC_MEMBER_NAME
    """ZIP member name (default canonical monolithic ``0.bin`` per HNeRV
    parity L3)."""

    def __post_init__(self) -> None:
        if not self.section_name or not self.section_name.strip():
            raise ValueError("section_name must be non-empty")
        if self.offset_in_archive < 0:
            raise ValueError(
                f"offset_in_archive {self.offset_in_archive} must be non-negative"
            )
        if self.length_in_archive < 0:
            raise ValueError(
                f"length_in_archive {self.length_in_archive} must be non-negative"
            )
        if len(self.sha256_of_section) != 64:
            raise ValueError(
                f"sha256_of_section must be 64-char hex; "
                f"got len={len(self.sha256_of_section)}"
            )
        if self.section_kind not in {k.value for k in SectionKind}:
            raise ValueError(
                f"section_kind {self.section_kind!r} must be one of "
                f"{[k.value for k in SectionKind]}"
            )
        if self.operational_mechanism_status not in {
            s.value for s in OperationalMechanismStatus
        }:
            raise ValueError(
                f"operational_mechanism_status {self.operational_mechanism_status!r} "
                f"must be one of {[s.value for s in OperationalMechanismStatus]} per Catalog #220"
            )
        if self.distinguishing_feature_name is not None and not self.distinguishing_feature_name.strip():
            raise ValueError("distinguishing_feature_name when set must be non-empty")
        if not self.member_name or not self.member_name.strip():
            raise ValueError("member_name must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "section_name": self.section_name,
            "offset_in_archive": int(self.offset_in_archive),
            "length_in_archive": int(self.length_in_archive),
            "sha256_of_section": self.sha256_of_section,
            "section_kind": self.section_kind,
            "operational_mechanism_status": self.operational_mechanism_status,
            "distinguishing_feature_name": self.distinguishing_feature_name,
            "member_name": self.member_name,
        }


@dataclass(frozen=True)
class ArchiveGrammarManifest:
    """Per-archive descriptor for a canonical archive grammar.

    Sister of :class:`tac.submission_packet.compression_pipeline.CompressionPipelineResult`
    at the archive-grammar sub-surface.

    Per HNeRV parity L3: monolithic single-file ``0.bin`` is the canonical
    default; multi-file archives require explicit ``multi_file_justification``
    rationale per Catalog #287 placeholder rejection.
    """

    schema_version: str
    """Canonical schema version (current: :data:`ARCHIVE_GRAMMAR_SCHEMA_VERSION`)."""

    lane_id: str
    """Lane registry id per CLAUDE.md "Lane maturity registry" lifecycle discipline."""

    substrate_id: str
    """Substrate id from compression pipeline result."""

    archive_path: str
    """Repo-relative or absolute path to the ``archive.zip``."""

    archive_sha256: str
    """sha256 hex digest of the full ``archive.zip`` bytes."""

    archive_bytes: int
    """Total ``archive.zip`` size in bytes."""

    section_specs: tuple[ArchiveSectionSpec, ...]
    """Per-section descriptors; canonical-monolithic-single-file requires all
    sections share the same ``member_name`` (default ``"0.bin"``)."""

    monolithic_single_file: bool
    """True for canonical HNeRV parity L3 single-file ``0.bin`` archives.
    False requires non-empty ``multi_file_justification`` rationale."""

    multi_file_justification: str | None
    """Substantive rationale (≥4 chars, non-placeholder per Catalog #287)
    when ``monolithic_single_file=False``."""

    byte_mutation_smoke_verdict: str
    """One of :class:`ByteMutationSmokeVerdict` values."""

    byte_mutation_smoke_evidence_path: str | None
    """Optional path to the byte-mutation smoke JSON evidence
    (canonical: ``experiments/results/<lane>/distinguishing_feature_byte_mutation_proof.json``)."""

    no_op_detector_passed: bool
    """Per Catalog #105 + #139 packet compiler no-op detector verdict.
    True iff bytes structurally consumed AND frame changes resulted."""

    measurement_utc: str
    """ISO-8601 UTC timestamp of manifest emission."""

    axis_tag: str
    """Always ``"[predicted]"`` per Catalog #341 + canonical Provenance."""

    score_claim: bool
    """Always ``False`` per CLAUDE.md "Apples-to-apples evidence discipline"."""

    promotable: bool
    """Always ``False`` per Catalog #341 + #192."""

    evidence_grade: str
    """Always ``"[predicted; archive-grammar-canonical]"`` per Catalog #287 / #323."""

    canonical_helper_invocation: str
    """``"tac.submission_packet.build_archive_grammar_from_compression_pipeline_result"``."""

    canonical_equation_id: str
    """:data:`CANONICAL_EQUATION_ID` per Catalog #344."""

    canonical_equation_status: str
    """``"FORMALIZATION_PENDING"`` until Phase 10 first empirical anchor."""

    parser_section_manifest_path: str | None
    """Optional sidecar JSON path emitted by
    :func:`emit_parser_section_manifest_sidecar` for inflate-runtime parser
    lookup; canonical naming: ``parser_section_manifest.json``."""

    elapsed_seconds: float
    """Manifest-build elapsed wall-clock (excludes byte-mutation smoke)."""

    canonical_provenance: Mapping[str, Any] = field(default_factory=dict)
    """Per Catalog #323 canonical Provenance umbrella."""

    written_at_utc: str = ""
    """When persisted to a canonical ledger (caller-fills)."""

    written_pid: int = 0
    """Process PID that emitted the manifest."""

    written_host: str = ""
    """Host that emitted the manifest."""

    def __post_init__(self) -> None:
        if self.schema_version != ARCHIVE_GRAMMAR_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {ARCHIVE_GRAMMAR_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}"
            )
        if not self.lane_id:
            raise ValueError("lane_id must be non-empty")
        if not self.substrate_id:
            raise ValueError("substrate_id must be non-empty")
        if not self.archive_path:
            raise ValueError("archive_path must be non-empty")
        if len(self.archive_sha256) != 64:
            raise ValueError(
                f"archive_sha256 must be 64-char hex; got len={len(self.archive_sha256)}"
            )
        if self.archive_bytes < 0:
            raise ValueError("archive_bytes must be non-negative")
        if not isinstance(self.section_specs, tuple):
            raise ValueError("section_specs must be a tuple (frozen)")
        if not isinstance(self.monolithic_single_file, bool):
            raise ValueError("monolithic_single_file must be bool")
        if self.monolithic_single_file:
            # Per HNeRV parity L3: monolithic single-file requires all sections
            # to share the canonical ``0.bin`` member name.
            for spec in self.section_specs:
                if spec.member_name != CANONICAL_MONOLITHIC_MEMBER_NAME:
                    raise ValueError(
                        f"monolithic_single_file=True requires all sections to use "
                        f"member_name={CANONICAL_MONOLITHIC_MEMBER_NAME!r}; section "
                        f"{spec.section_name!r} uses {spec.member_name!r} per HNeRV parity L3"
                    )
        else:
            # Per Catalog #287: multi-file requires substantive justification.
            if self.multi_file_justification is None:
                raise ValueError(
                    "monolithic_single_file=False requires non-None "
                    "multi_file_justification per HNeRV parity L3"
                )
            rationale = self.multi_file_justification.strip()
            if rationale in _PLACEHOLDER_RATIONALES or len(rationale) < 4:
                raise ValueError(
                    f"multi_file_justification {self.multi_file_justification!r} "
                    "must be substantive (>=4 chars, non-placeholder) per Catalog #287"
                )
        if self.byte_mutation_smoke_verdict not in {
            v.value for v in ByteMutationSmokeVerdict
        }:
            raise ValueError(
                f"byte_mutation_smoke_verdict {self.byte_mutation_smoke_verdict!r} "
                f"must be one of {[v.value for v in ByteMutationSmokeVerdict]} per Catalog #139"
            )
        if not isinstance(self.no_op_detector_passed, bool):
            raise ValueError("no_op_detector_passed must be bool")
        # Catalog #266: bytes consumed by inflate. If smoke verdict is PASSED
        # then no_op_detector_passed must be True; if FAILED_BYTES_NOT_CONSUMED
        # then must be False. Other verdicts (NOT_RUN / INFRASTRUCTURE_ERROR)
        # allow either since the smoke is inconclusive.
        if (
            self.byte_mutation_smoke_verdict == ByteMutationSmokeVerdict.PASSED.value
            and not self.no_op_detector_passed
        ):
            raise ValueError(
                "byte_mutation_smoke_verdict=PASSED requires no_op_detector_passed=True "
                "per Catalog #105 + #139 + #266 sister discipline"
            )
        if (
            self.byte_mutation_smoke_verdict
            == ByteMutationSmokeVerdict.FAILED_BYTES_NOT_CONSUMED.value
            and self.no_op_detector_passed
        ):
            raise ValueError(
                "byte_mutation_smoke_verdict=FAILED_BYTES_NOT_CONSUMED requires "
                "no_op_detector_passed=False per Catalog #266"
            )
        if self.axis_tag != PREDICTED_AXIS_TAG:
            raise ValueError(f"axis_tag must equal {PREDICTED_AXIS_TAG!r}; got {self.axis_tag!r}")
        if self.score_claim is not False:
            raise ValueError("score_claim must be False per Catalog #341")
        if self.promotable is not False:
            raise ValueError("promotable must be False per Catalog #341")
        if not self.evidence_grade.startswith("[predicted;"):
            raise ValueError(
                "evidence_grade must start with '[predicted;' per Catalog #287/#323"
            )
        if self.canonical_equation_id != CANONICAL_EQUATION_ID:
            raise ValueError(
                f"canonical_equation_id must equal {CANONICAL_EQUATION_ID!r}; "
                f"got {self.canonical_equation_id!r}"
            )
        if self.canonical_equation_status not in {"FORMALIZATION_PENDING", "REGISTERED"}:
            raise ValueError(
                "canonical_equation_status must be 'FORMALIZATION_PENDING' or 'REGISTERED' per Catalog #344"
            )
        if not self.measurement_utc:
            raise ValueError("measurement_utc must be non-empty")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if not isinstance(self.canonical_provenance, Mapping):
            raise ValueError("canonical_provenance must be a Mapping per Catalog #323")

        # Catalog #146 fixed-offset declared in source: verify no section
        # overlap within the same member. The overlap check is per-member so
        # multi-file archives with distinct members are correctly handled.
        per_member: dict[str, list[tuple[int, int, str]]] = {}
        for spec in self.section_specs:
            per_member.setdefault(spec.member_name, []).append(
                (spec.offset_in_archive, spec.length_in_archive, spec.section_name)
            )
        for member, sections in per_member.items():
            sections.sort(key=lambda s: s[0])
            for i in range(len(sections) - 1):
                offset_i, length_i, name_i = sections[i]
                offset_j, _length_j, name_j = sections[i + 1]
                if offset_i + length_i > offset_j:
                    raise ValueError(
                        f"section overlap in member {member!r}: "
                        f"section {name_i!r} (offset={offset_i}, length={length_i}) "
                        f"overlaps section {name_j!r} (offset={offset_j}) "
                        "per Catalog #146 fixed-offset discipline"
                    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lane_id": self.lane_id,
            "substrate_id": self.substrate_id,
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": int(self.archive_bytes),
            "section_specs": [s.as_dict() for s in self.section_specs],
            "monolithic_single_file": bool(self.monolithic_single_file),
            "multi_file_justification": self.multi_file_justification,
            "byte_mutation_smoke_verdict": self.byte_mutation_smoke_verdict,
            "byte_mutation_smoke_evidence_path": self.byte_mutation_smoke_evidence_path,
            "no_op_detector_passed": bool(self.no_op_detector_passed),
            "measurement_utc": self.measurement_utc,
            "axis_tag": self.axis_tag,
            "score_claim": bool(self.score_claim),
            "promotable": bool(self.promotable),
            "evidence_grade": self.evidence_grade,
            "canonical_helper_invocation": self.canonical_helper_invocation,
            "canonical_equation_id": self.canonical_equation_id,
            "canonical_equation_status": self.canonical_equation_status,
            "parser_section_manifest_path": self.parser_section_manifest_path,
            "elapsed_seconds": float(self.elapsed_seconds),
            "canonical_provenance": dict(self.canonical_provenance),
            "written_at_utc": self.written_at_utc,
            "written_pid": int(self.written_pid),
            "written_host": self.written_host,
        }


# ---------------------------------------------------------------------------
# Core API helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Canonical UTC timestamp (ISO-8601 with tz)."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def _sha256_bytes(data: bytes) -> str:
    """sha256 hex digest of bytes (canonical helper)."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    """Streaming sha256 hex digest of a file (canonical helper)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_archive_zip_members(archive_path: Path) -> tuple[str, ...]:
    """List ZIP members in canonical sort order."""
    with zipfile.ZipFile(archive_path) as zf:
        return tuple(sorted(zf.namelist()))


def _read_archive_zip_member_bytes(archive_path: Path, member_name: str) -> bytes:
    """Read a ZIP member's full bytes."""
    with zipfile.ZipFile(archive_path) as zf:
        with zf.open(member_name) as member:
            return member.read()


def derive_archive_grammar_provenance(
    *,
    lane_id: str,
    substrate_id: str,
    archive_sha256: str,
    measurement_utc: str,
) -> dict[str, Any]:
    """Build the canonical Provenance dict for an archive grammar manifest.

    Per Catalog #323 canonical Provenance umbrella: every persisted row
    carries (axis_tag + evidence_grade + score_claim + promotable +
    canonical_helper_invocation + captured_at_utc). This helper returns the
    canonical shape downstream consumers expect.
    """
    return {
        "axis_tag": PREDICTED_AXIS_TAG,
        "evidence_grade": "[predicted; archive-grammar-canonical]",
        "score_claim": False,
        "promotable": False,
        "canonical_helper_invocation": (
            "tac.submission_packet.build_archive_grammar_from_compression_pipeline_result"
        ),
        "captured_at_utc": measurement_utc,
        "lane_id": lane_id,
        "substrate_id": substrate_id,
        "archive_sha256": archive_sha256,
        "canonical_equation_id": CANONICAL_EQUATION_ID,
        "canonical_equation_status": "FORMALIZATION_PENDING",
        "schema_version": ARCHIVE_GRAMMAR_SCHEMA_VERSION,
    }


def discover_section_specs_from_archive(
    archive_path: Path,
    *,
    monolithic_member_name: str = CANONICAL_MONOLITHIC_MEMBER_NAME,
) -> tuple[tuple[ArchiveSectionSpec, ...], bool]:
    """Auto-discover section specs from an ``archive.zip`` for canonical HNeRV
    parity L3 monolithic-single-file archives.

    Per HNeRV parity L3 canonical: monolithic single-file ``0.bin`` is the
    default. When the archive has only that single member, the helper emits
    one :class:`ArchiveSectionSpec` covering the full member with section_kind
    ``OTHER`` (caller routes substrate-specific grammar refinement). When the
    archive has multiple members, the helper falls back to per-member specs
    with offset 0 in each member; the manifest then carries
    ``monolithic_single_file=False`` and the caller MUST provide
    ``multi_file_justification``.

    Returns:
        ``(section_specs_tuple, is_monolithic_single_file)``.
    """
    members = _list_archive_zip_members(archive_path)
    if not members:
        raise ArchiveGrammarError(
            f"archive {archive_path} has zero ZIP members; "
            "HNeRV parity L3 requires at least the canonical monolithic 0.bin"
        )
    if len(members) == 1 and members[0] == monolithic_member_name:
        # Canonical HNeRV parity L3 monolithic single-file path.
        member_bytes = _read_archive_zip_member_bytes(archive_path, monolithic_member_name)
        spec = ArchiveSectionSpec(
            section_name=monolithic_member_name,
            offset_in_archive=0,
            length_in_archive=len(member_bytes),
            sha256_of_section=_sha256_bytes(member_bytes),
            section_kind=SectionKind.OTHER.value,
            operational_mechanism_status=OperationalMechanismStatus.OPERATIONAL.value,
            distinguishing_feature_name=None,
            member_name=monolithic_member_name,
        )
        return ((spec,), True)
    # Multi-file fallback: emit one spec per member.
    specs: list[ArchiveSectionSpec] = []
    for member_name in members:
        member_bytes = _read_archive_zip_member_bytes(archive_path, member_name)
        specs.append(
            ArchiveSectionSpec(
                section_name=member_name,
                offset_in_archive=0,
                length_in_archive=len(member_bytes),
                sha256_of_section=_sha256_bytes(member_bytes),
                section_kind=SectionKind.OTHER.value,
                operational_mechanism_status=OperationalMechanismStatus.OPERATIONAL.value,
                distinguishing_feature_name=None,
                member_name=member_name,
            )
        )
    return (tuple(specs), False)


def emit_parser_section_manifest_sidecar(
    manifest: ArchiveGrammarManifest,
    output_dir: Path,
    *,
    filename: str = "parser_section_manifest.json",
) -> Path:
    """Emit the canonical parser-section-manifest JSON sidecar.

    Per HNeRV parity L3 + Catalog #146: the inflate runtime needs a stable
    per-section lookup so the canonical 3-arg ``inflate.sh`` signature
    (archive_dir / output_dir / file_list) can locate each section's offset +
    length deterministically. The sidecar JSON is byte-stable (sorted keys)
    so the canonical archive_sha256 is reproducible across machines.

    Args:
        manifest: the canonical ArchiveGrammarManifest to serialize.
        output_dir: directory to write the sidecar into; created if missing.
        filename: sidecar filename (default canonical ``parser_section_manifest.json``).

    Returns:
        Path to the written sidecar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    # Per Catalog #245 / #313 / #344 canonical 4-layer pattern: sort_keys=True
    # + indent=2 for human-diff-ability + canonical byte-stable serialization.
    payload = manifest.as_dict()
    target.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def verify_byte_mutation_smoke_via_canonical_helper(
    *,
    archive_path: Path,
    inflate_sh_path: Path,
    section_specs: tuple[ArchiveSectionSpec, ...],
    output_json_path: Path,
    mutations_per_section: int = 4,
    inflate_timeout_seconds: int = 600,
) -> tuple[str, bool, Path | None]:
    """Run the Catalog #272 distinguishing-feature byte-mutation smoke.

    Routes through the canonical helper
    ``tools/verify_distinguishing_feature_byte_mutation.py::verify_distinguishing_feature_byte_mutation``
    (NOT a subprocess shell-out per Catalog #226 canonical-helper-routing
    discipline). The helper mutates a sample of bytes per declared
    distinguishing-feature section and runs inflate source-vs-mutated to
    compare output frames byte-for-byte.

    Returns:
        ``(verdict, no_op_detector_passed, evidence_path_or_none)``.
        verdict is one of :class:`ByteMutationSmokeVerdict` values.

    Per Catalog #266: when ``archive_bytes > BYTE_MUTATION_SMOKE_MIN_BYTES``
    AND a distinguishing-feature section is declared, the smoke is REQUIRED.
    """
    # Late import (canonical helper lives at tools/ not src/tac/).
    import sys

    canonical_helper_path = REPO_ROOT / "tools" / "verify_distinguishing_feature_byte_mutation.py"
    if not canonical_helper_path.is_file():
        return (
            ByteMutationSmokeVerdict.INFRASTRUCTURE_ERROR.value,
            False,
            None,
        )
    tools_dir = str((REPO_ROOT / "tools").resolve())
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import verify_distinguishing_feature_byte_mutation as _vdfbm
    except ImportError:
        return (
            ByteMutationSmokeVerdict.INFRASTRUCTURE_ERROR.value,
            False,
            None,
        )

    distinguishing_bytes_paths: list[str] = []
    for spec in section_specs:
        if spec.distinguishing_feature_name is not None:
            distinguishing_bytes_paths.append(spec.member_name)
    if not distinguishing_bytes_paths:
        # No distinguishing feature declared; smoke not applicable. Per
        # Catalog #266 this is informational only when archive_bytes is small.
        return (ByteMutationSmokeVerdict.NOT_RUN.value, False, None)

    # Deduplicate (same member may carry multiple sections).
    seen: set[str] = set()
    deduped_paths: list[str] = []
    for path in distinguishing_bytes_paths:
        if path not in seen:
            seen.add(path)
            deduped_paths.append(path)

    try:
        result = _vdfbm.verify_distinguishing_feature_byte_mutation(
            archive=archive_path,
            inflate_sh=inflate_sh_path,
            distinguishing_bytes_paths=deduped_paths,
            output_json=output_json_path,
            mutations_per_section=mutations_per_section,
            inflate_timeout_seconds=inflate_timeout_seconds,
        )
    except Exception:  # noqa: BLE001 — helper-side errors funnel to INFRA verdict
        return (
            ByteMutationSmokeVerdict.INFRASTRUCTURE_ERROR.value,
            False,
            None,
        )

    overall_verdict_raw = result.get("verdict", "INFRASTRUCTURE_ERROR")
    # Helper emits PASSED / FAILED / INFRASTRUCTURE_ERROR; map FAILED to the
    # canonical FAILED_BYTES_NOT_CONSUMED per Catalog #266 semantics.
    if overall_verdict_raw == "PASSED":
        verdict = ByteMutationSmokeVerdict.PASSED.value
        no_op_passed = True
    elif overall_verdict_raw == "FAILED":
        verdict = ByteMutationSmokeVerdict.FAILED_BYTES_NOT_CONSUMED.value
        no_op_passed = False
    else:
        verdict = ByteMutationSmokeVerdict.INFRASTRUCTURE_ERROR.value
        no_op_passed = False

    evidence_path = output_json_path if output_json_path.is_file() else None
    return (verdict, no_op_passed, evidence_path)


def build_archive_grammar_from_compression_pipeline_result(
    *,
    compression_pipeline_result: CompressionPipelineResult,
    archive_path: Path,
    section_specs: tuple[ArchiveSectionSpec, ...] | None = None,
    monolithic_single_file: bool = True,
    multi_file_justification: str | None = None,
    output_dir: Path | None = None,
    inflate_sh_path: Path | None = None,
    verify_byte_mutation_smoke: bool = False,
    byte_mutation_evidence_path: Path | None = None,
    emit_parser_section_manifest: bool = True,
    repo_root: Path | None = None,
) -> ArchiveGrammarManifest:
    """Canonical archive grammar builder (Layer 1) — main entry point.

    Routes the compression pipeline result + archive bytes through HNeRV parity
    L3 monolithic-single-file invariants + canonical section-spec derivation +
    Catalog #146 fixed-offset declaration verification + Catalog #266
    optional byte-mutation smoke + canonical Provenance umbrella + canonical
    equation id stamping. Returns a typed :class:`ArchiveGrammarManifest`
    that downstream Phase 4-10 layers consume.

    The helper does NOT invoke paid Modal/Vast.ai/Lightning dispatch per
    Phase 3 scope. It DERIVES the canonical grammar from existing archive
    bytes (the trainer's emission). Phase 6 ``paired_auth_eval`` is where
    paired-axis empirical anchors land.

    Args:
        compression_pipeline_result: typed Phase 2 Layer 0 result whose
            ``lane_id`` + ``substrate_id`` lineage is canonical.
        archive_path: path to the trainer-emitted ``archive.zip``.
        section_specs: optional pre-built per-section descriptors. When None,
            the helper auto-discovers via :func:`discover_section_specs_from_archive`.
        monolithic_single_file: True (default) per HNeRV parity L3; False
            requires non-empty ``multi_file_justification``.
        multi_file_justification: substantive rationale (≥4 chars, non-
            placeholder per Catalog #287) when ``monolithic_single_file=False``.
        output_dir: optional output directory for sidecars (parser_section_
            manifest.json + byte-mutation smoke JSON). When None, sidecars
            default to ``archive_path.parent``.
        inflate_sh_path: path to ``inflate.sh`` for byte-mutation smoke;
            required when ``verify_byte_mutation_smoke=True``.
        verify_byte_mutation_smoke: when True, runs the Catalog #272 byte-
            mutation smoke via the canonical helper (lives at tools/
            verify_distinguishing_feature_byte_mutation.py).
        byte_mutation_evidence_path: optional explicit path for the byte-
            mutation smoke JSON evidence; when None, defaults to
            ``output_dir / "byte_mutation_smoke_proof.json"``.
        emit_parser_section_manifest: when True (default), emit the canonical
            ``parser_section_manifest.json`` sidecar at ``output_dir``.
        repo_root: override repo root (defaults to module-resolved REPO_ROOT).

    Returns:
        :class:`ArchiveGrammarManifest` with canonical Provenance.

    Raises:
        ArchiveGrammarError: when archive_path is missing OR section overlap
            detected OR HNeRV parity L3 invariants violated.
    """
    started = _utc_now_iso()
    started_perf = datetime.datetime.now(datetime.UTC)
    root = repo_root if repo_root is not None else REPO_ROOT
    if not isinstance(compression_pipeline_result, CompressionPipelineResult):
        raise ArchiveGrammarError(
            "compression_pipeline_result must be a tac.submission_packet.compression_pipeline.CompressionPipelineResult"
        )
    if not isinstance(archive_path, Path):
        raise ArchiveGrammarError("archive_path must be a pathlib.Path")
    archive_abs = archive_path if archive_path.is_absolute() else (root / archive_path).resolve()
    if not archive_abs.is_file():
        raise ArchiveGrammarError(
            f"archive_path {archive_abs} does not exist; "
            "HNeRV parity L3 requires the canonical archive.zip to be present"
        )

    archive_sha = _sha256_file(archive_abs)
    archive_size = archive_abs.stat().st_size

    if section_specs is None:
        derived_specs, derived_is_monolithic = discover_section_specs_from_archive(archive_abs)
        section_specs_tuple = derived_specs
        # If caller did not override monolithic_single_file AND the archive
        # auto-derives as non-monolithic, propagate the discovery so the
        # validator surfaces the multi-file justification requirement
        # immediately rather than silently passing.
        if monolithic_single_file and not derived_is_monolithic:
            monolithic_single_file = False
    else:
        section_specs_tuple = tuple(section_specs)

    out_dir = output_dir if output_dir is not None else archive_abs.parent

    # Catalog #266 byte-mutation smoke (optional; per Phase 3 scope it is
    # operator-routable rather than auto-invoked).
    byte_mutation_verdict = ByteMutationSmokeVerdict.NOT_RUN.value
    no_op_passed = False
    byte_mutation_evidence_actual: Path | None = None
    if verify_byte_mutation_smoke:
        if inflate_sh_path is None:
            raise ArchiveGrammarError(
                "verify_byte_mutation_smoke=True requires inflate_sh_path to be set"
            )
        inflate_abs = (
            inflate_sh_path
            if inflate_sh_path.is_absolute()
            else (root / inflate_sh_path).resolve()
        )
        evidence_path = (
            byte_mutation_evidence_path
            if byte_mutation_evidence_path is not None
            else (out_dir / "byte_mutation_smoke_proof.json")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        verdict, no_op_passed, evidence_actual = (
            verify_byte_mutation_smoke_via_canonical_helper(
                archive_path=archive_abs,
                inflate_sh_path=inflate_abs,
                section_specs=section_specs_tuple,
                output_json_path=evidence_path,
            )
        )
        byte_mutation_verdict = verdict
        byte_mutation_evidence_actual = evidence_actual

    measurement_utc = _utc_now_iso()
    canonical_provenance = derive_archive_grammar_provenance(
        lane_id=compression_pipeline_result.lane_id,
        substrate_id=compression_pipeline_result.substrate_id,
        archive_sha256=archive_sha,
        measurement_utc=measurement_utc,
    )

    elapsed = (datetime.datetime.now(datetime.UTC) - started_perf).total_seconds()

    manifest = ArchiveGrammarManifest(
        schema_version=ARCHIVE_GRAMMAR_SCHEMA_VERSION,
        lane_id=compression_pipeline_result.lane_id,
        substrate_id=compression_pipeline_result.substrate_id,
        archive_path=str(archive_path),
        archive_sha256=archive_sha,
        archive_bytes=int(archive_size),
        section_specs=section_specs_tuple,
        monolithic_single_file=monolithic_single_file,
        multi_file_justification=multi_file_justification,
        byte_mutation_smoke_verdict=byte_mutation_verdict,
        byte_mutation_smoke_evidence_path=(
            str(byte_mutation_evidence_actual)
            if byte_mutation_evidence_actual is not None
            else None
        ),
        no_op_detector_passed=bool(no_op_passed),
        measurement_utc=measurement_utc,
        axis_tag=PREDICTED_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; archive-grammar-canonical]",
        canonical_helper_invocation=(
            "tac.submission_packet.build_archive_grammar_from_compression_pipeline_result"
        ),
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        parser_section_manifest_path=None,
        elapsed_seconds=float(elapsed),
        canonical_provenance=canonical_provenance,
        written_at_utc=measurement_utc,
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )

    if emit_parser_section_manifest:
        sidecar_path = emit_parser_section_manifest_sidecar(manifest, out_dir)
        # Rebuild manifest with the sidecar path filled in (frozen dataclass).
        manifest = ArchiveGrammarManifest(
            schema_version=manifest.schema_version,
            lane_id=manifest.lane_id,
            substrate_id=manifest.substrate_id,
            archive_path=manifest.archive_path,
            archive_sha256=manifest.archive_sha256,
            archive_bytes=manifest.archive_bytes,
            section_specs=manifest.section_specs,
            monolithic_single_file=manifest.monolithic_single_file,
            multi_file_justification=manifest.multi_file_justification,
            byte_mutation_smoke_verdict=manifest.byte_mutation_smoke_verdict,
            byte_mutation_smoke_evidence_path=manifest.byte_mutation_smoke_evidence_path,
            no_op_detector_passed=manifest.no_op_detector_passed,
            measurement_utc=manifest.measurement_utc,
            axis_tag=manifest.axis_tag,
            score_claim=manifest.score_claim,
            promotable=manifest.promotable,
            evidence_grade=manifest.evidence_grade,
            canonical_helper_invocation=manifest.canonical_helper_invocation,
            canonical_equation_id=manifest.canonical_equation_id,
            canonical_equation_status=manifest.canonical_equation_status,
            parser_section_manifest_path=str(sidecar_path),
            elapsed_seconds=manifest.elapsed_seconds,
            canonical_provenance=manifest.canonical_provenance,
            written_at_utc=manifest.written_at_utc,
            written_pid=manifest.written_pid,
            written_host=manifest.written_host,
        )

    return manifest
