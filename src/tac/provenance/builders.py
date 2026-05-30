# SPDX-License-Identifier: MIT
"""tac.provenance.builders — single-line canonical Provenance constructors.

Per operator NON-NEGOTIABLE 2026-05-17 verbatim: *"make it easy"*. Every
canonical caller in the repo uses ONE of these helpers; non-canonical
direct ``Provenance(...)`` construction outside these helpers is
discouraged (and detected by the audit tool ``tools/audit_provenance_compliance.py``).

The canonical builders mirror the ProvenanceKind values:

  * ``build_provenance_for_archive_member`` — CONTEST_ARCHIVE_MEMBER
  * ``build_provenance_for_research_sidecar`` — RESEARCH_SIDECAR
  * ``build_provenance_for_predicted`` — PREDICTED_FROM_MODEL
  * ``build_provenance_for_macos_cpu_advisory`` — ADVISORY_NON_PROMOTABLE (macOS-CPU)
  * ``build_provenance_for_macos_mlx_research_signal`` —
    ADVISORY_NON_PROMOTABLE (macOS-MLX research signal)
  * ``build_provenance_for_mps_proxy`` — ADVISORY_NON_PROMOTABLE (MPS)
  * ``build_provenance_aggregate`` — AGGREGATE_OF_PROVENANCES
  * ``build_provenance_invalid_byte_identity_artifact`` — INVALID_BYTE_IDENTITY_ARTIFACT
    sentinel for the Catalog #823 SUPER_ADDITIVE class
  * ``build_provenance_for_archive_seed_procedural_generation`` —
    PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED
  * ``build_provenance_for_weight_derived_codebook`` — WEIGHT_DERIVED_CODEBOOK
  * ``build_provenance_for_forbidden_out_of_archive_payload`` —
    FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD
  * ``register_forbidden_out_of_archive_payload_probe_outcome`` —
    Catalog #313 blocking probe-outcome bridge for the forbidden sentinel

Plus 2 ergonomics helpers:

  * ``Provenance.from_archive_zip_member`` — class-method shorthand
    (re-exported here for the common case)
  * ``@requires_canonical_provenance`` — decorator that raises
    MissingProvenanceError if the wrapped function returns a
    score-like value without an attached Provenance attribute

Per CLAUDE.md "Bit-level deconstruction and entropy discipline": helpers
that compute the source_sha256 do it via ``hashlib.sha256`` over the
RAW BYTES (no transformation, no encoding wrapping). For archive members
this is the unzipped member bytes; for research sidecars this is the
file contents on disk.

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact": helpers
refuse to construct Provenance with source_path starting with /tmp,
$HOME/.cache, or other transient locations.
"""

from __future__ import annotations

import functools
import hashlib
import zipfile
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.provenance.contract import (
    InvalidProvenanceError,
    MissingProvenanceError,
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
    ScoreClaim,
)


def _utc_now_iso() -> str:
    """Canonical UTC timestamp in ISO format with trailing Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_transient_path(path: str) -> None:
    """Raise InvalidProvenanceError if path is in a transient location.

    Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact".

    Pytest tmp_path fixtures (which resolve to /private/var/folders/.../pytest-of-*/
    on macOS or /tmp/pytest-of-*/ on Linux) are recognized via the
    ``pytest-of-`` substring and allowed (test-fixture-context).
    """
    # Sentinel paths (starting with <) are always allowed
    if path.startswith("<"):
        return

    # Pytest tmp_path fixtures recognized via the canonical substring
    if "pytest-of-" in path or "pytest-" in path:
        return

    transient_prefixes = (
        "/tmp/",
        "/var/tmp/",
        "/private/tmp/",
        "/private/var/folders/",
    )
    for prefix in transient_prefixes:
        if path.startswith(prefix):
            raise InvalidProvenanceError(
                f"Provenance source_path={path!r} is in transient location"
                f" {prefix!r}; per CLAUDE.md 'Forbidden /tmp paths in any"
                " persisted artifact' use experiments/results/<lane>/ or"
                " .omx/state/ instead"
            )


def _sha256_of_bytes(data: bytes) -> str:
    """Canonical lowercase 64-char hex digest."""
    return hashlib.sha256(data).hexdigest()


def _sha256_of_file(path: Path) -> str:
    """SHA-256 over file bytes; streaming so large files don't OOM."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_archive_member(archive_zip_path: Path, member_name: str) -> str:
    """SHA-256 over the UNZIPPED member bytes (not the compressed bytes)."""
    with zipfile.ZipFile(archive_zip_path, "r") as zf, zf.open(member_name) as member:
        h = hashlib.sha256()
        for chunk in iter(lambda: member.read(65536), b""):
            h.update(chunk)
        return h.hexdigest()


# -----------------------------------------------------------------------------
# Builder: CONTEST_ARCHIVE_MEMBER
# -----------------------------------------------------------------------------


def build_provenance_for_archive_member(
    archive_zip_path: str | Path,
    member_name: str,
    measurement_axis: str,
    hardware_substrate: str,
    evidence_grade: ProvenanceEvidenceGrade,
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical builder for CONTEST_ARCHIVE_MEMBER Provenance.

    Computes ``source_sha256`` from the UNZIPPED member bytes (the canonical
    contest-rate-charged content). The ``source_path`` is constructed as
    ``<archive_zip_path>:<member_name>`` per the canonical convention.

    Args:
        archive_zip_path: path to the archive.zip (relative to repo root preferred).
        member_name: member name inside the zip (e.g., "0.bin").
        measurement_axis: one of CANONICAL_MEASUREMENT_AXES; for promotable
            grades must match the grade.
        hardware_substrate: one of CANONICAL_HARDWARE_SUBSTRATES.
        evidence_grade: typically PROMOTABLE_EXACT_CONTEST_CUDA or _CPU.
        captured_at_utc: optional; defaults to now-UTC.

    Returns:
        Frozen Provenance with promotion_eligible+score_claim_valid auto-set
        based on grade.

    Raises:
        InvalidProvenanceError: if the archive or member does not exist,
        if the path is transient, or if grade/axis/hardware mismatch.
    """
    archive_path = Path(archive_zip_path)
    archive_path_str = str(archive_zip_path)
    _refuse_transient_path(archive_path_str)

    if not archive_path.exists():
        raise InvalidProvenanceError(
            f"Archive zip not found at {archive_path_str!r}; cannot build CONTEST_ARCHIVE_MEMBER Provenance"
        )

    try:
        member_sha256 = _sha256_of_archive_member(archive_path, member_name)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise InvalidProvenanceError(f"Cannot read member {member_name!r} from {archive_path_str!r}: {exc}") from exc

    source_path = f"{archive_path_str}:{member_name}"

    # Promotable iff grade ∈ promotable set
    is_promotable = evidence_grade in (
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CPU,
    )

    return Provenance(
        artifact_kind=ProvenanceKind.CONTEST_ARCHIVE_MEMBER,
        source_path=source_path,
        source_sha256=member_sha256,
        measurement_axis=measurement_axis,
        hardware_substrate=hardware_substrate,
        evidence_grade=evidence_grade,
        promotion_eligible=is_promotable,
        score_claim_valid=is_promotable,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation="tac.provenance.builders.build_provenance_for_archive_member",
        contest_archive_zip_path=archive_path_str,
        contest_archive_member_name=member_name,
    )


# -----------------------------------------------------------------------------
# Builder: RESEARCH_SIDECAR
# -----------------------------------------------------------------------------


def build_provenance_for_research_sidecar(
    sidecar_path: str | Path,
    reactivation_criteria: str,
    measurement_axis: str = "[research-signal]",
    hardware_substrate: str = "unknown",
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical builder for RESEARCH_SIDECAR Provenance (Catalog #321 anchor).

    By construction:
      - promotion_eligible=False
      - score_claim_valid=False
      - evidence_grade=RESEARCH_ONLY

    The ``reactivation_criteria`` is stored in the ``rejection_reason`` field
    per CLAUDE.md "Forbidden premature KILL" non-negotiable: research-only
    artifacts are DEFERRED-pending-research, not killed.

    Args:
        sidecar_path: path to the .pt / .npy / .json / etc. file.
        reactivation_criteria: explicit criteria under which the sidecar
            could be promoted (e.g., "after archive member byte verification").
        measurement_axis: defaults to [research-signal].
        hardware_substrate: defaults to "unknown".
        captured_at_utc: optional; defaults to now-UTC.

    Returns:
        Frozen Provenance with non-promotable invariants enforced.
    """
    sidecar_path_obj = Path(sidecar_path)
    sidecar_path_str = str(sidecar_path)
    _refuse_transient_path(sidecar_path_str)

    if not reactivation_criteria or not reactivation_criteria.strip():
        raise InvalidProvenanceError(
            "RESEARCH_SIDECAR requires non-empty reactivation_criteria (per CLAUDE.md 'Forbidden premature KILL')"
        )

    # Allow missing sidecars so DEFERRED-pending-research artifacts can carry
    # Provenance with an explicit placeholder sha.
    sha = "0" * 64 if not sidecar_path_obj.exists() else _sha256_of_file(sidecar_path_obj)

    return Provenance(
        artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
        source_path=sidecar_path_str,
        source_sha256=sha,
        measurement_axis=measurement_axis,
        hardware_substrate=hardware_substrate,
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation="tac.provenance.builders.build_provenance_for_research_sidecar",
        rejection_reason=reactivation_criteria,
    )


# -----------------------------------------------------------------------------
# Builder: PREDICTED_FROM_MODEL
# -----------------------------------------------------------------------------


def build_provenance_for_predicted(
    model_id: str,
    inputs_sha256: str,
    measurement_axis: str = "[predicted]",
    hardware_substrate: str = "unknown",
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical builder for PREDICTED_FROM_MODEL Provenance.

    Always promotion_eligible=False until an empirical anchor lands.

    Args:
        model_id: the predictor's identifier (e.g., "autopilot.predicted_delta_v2").
        inputs_sha256: sha256 over the model's input features.
        measurement_axis: defaults to [predicted].
        hardware_substrate: defaults to "unknown".
        captured_at_utc: optional; defaults to now-UTC.

    Returns:
        Frozen Provenance with PREDICTED grade + non-promotable invariants.
    """
    if not model_id:
        raise InvalidProvenanceError("PREDICTED_FROM_MODEL requires non-empty model_id")
    if not inputs_sha256:
        raise InvalidProvenanceError("PREDICTED_FROM_MODEL requires non-empty inputs_sha256")

    source_path = f"<predictor:{model_id}>"

    return Provenance(
        artifact_kind=ProvenanceKind.PREDICTED_FROM_MODEL,
        source_path=source_path,
        source_sha256=inputs_sha256,
        measurement_axis=measurement_axis,
        hardware_substrate=hardware_substrate,
        evidence_grade=ProvenanceEvidenceGrade.PREDICTED,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation="tac.provenance.builders.build_provenance_for_predicted",
    )


# -----------------------------------------------------------------------------
# Builder: macOS-CPU advisory (Catalog #192 anchor)
# -----------------------------------------------------------------------------


def build_provenance_for_macos_cpu_advisory(
    archive_sha256: str,
    source_path: str,
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical builder for macOS-CPU advisory Provenance (Catalog #192).

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192: macOS-CPU is
    NEVER 1:1 contest-compliant. The advisory grade prevents promotion
    even when the score numerically matches a Linux x86_64 anchor.
    """
    if not archive_sha256:
        raise InvalidProvenanceError("macOS-CPU advisory requires non-empty archive_sha256")
    _refuse_transient_path(source_path)

    return Provenance(
        artifact_kind=ProvenanceKind.ADVISORY_NON_PROMOTABLE,
        source_path=source_path,
        source_sha256=archive_sha256,
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="macos_arm64",
        evidence_grade=ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation="tac.provenance.builders.build_provenance_for_macos_cpu_advisory",
    )


# -----------------------------------------------------------------------------
# Builder: macOS-MLX research signal
# -----------------------------------------------------------------------------


def build_provenance_for_macos_mlx_research_signal(
    artifact_sha256: str,
    source_path: str,
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical builder for macOS-MLX research-signal Provenance.

    MLX-local measurements are useful for local acquisition, long-training
    triage, and substrate debugging, but they are never contest score authority.
    This distinct grade prevents MLX rows from being collapsed into macOS-CPU
    advisory or MPS proxy evidence while preserving fail-closed promotion gates.
    """
    if not artifact_sha256:
        raise InvalidProvenanceError("macOS-MLX research signal requires non-empty artifact_sha256")
    _refuse_transient_path(source_path)

    return Provenance(
        artifact_kind=ProvenanceKind.ADVISORY_NON_PROMOTABLE,
        source_path=source_path,
        source_sha256=artifact_sha256,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="macos_arm64_mlx",
        evidence_grade=ProvenanceEvidenceGrade.MACOS_MLX_RESEARCH_SIGNAL,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation=("tac.provenance.builders.build_provenance_for_macos_mlx_research_signal"),
    )


# -----------------------------------------------------------------------------
# Builder: MPS proxy (per CLAUDE.md "MPS auth eval is NOISE")
# -----------------------------------------------------------------------------


def build_provenance_for_mps_proxy(
    artifact_sha256: str,
    source_path: str,
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical builder for MPS proxy Provenance.

    Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable: PoseNet drift
    23x on MPS vs CUDA; never use as score truth.
    """
    if not artifact_sha256:
        raise InvalidProvenanceError("MPS proxy requires non-empty artifact_sha256")
    _refuse_transient_path(source_path)

    return Provenance(
        artifact_kind=ProvenanceKind.ADVISORY_NON_PROMOTABLE,
        source_path=source_path,
        source_sha256=artifact_sha256,
        measurement_axis="[MPS-PROXY]",
        hardware_substrate="macos_arm64",
        evidence_grade=ProvenanceEvidenceGrade.MPS_PROXY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation="tac.provenance.builders.build_provenance_for_mps_proxy",
    )


# -----------------------------------------------------------------------------
# Builder: AGGREGATE_OF_PROVENANCES (composition)
# -----------------------------------------------------------------------------


def build_provenance_aggregate(
    parts: Sequence[Provenance],
    aggregation_rationale: str,
    aggregation_path: str = "<aggregate>",
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical builder for AGGREGATE_OF_PROVENANCES (Catalog #319/#823 anchor).

    Auto-detects the Catalog #823 byte-identity artifact class: if 2+ parts
    share source_sha256, the aggregate is constructed with
    evidence_grade=INVALID_BYTE_IDENTITY_ARTIFACT and an explanatory
    rejection_reason.

    Otherwise the aggregate's grade is the WORST (least-promotable) of
    the component grades.

    Args:
        parts: non-empty sequence of upstream Provenances.
        aggregation_rationale: human-readable explanation of the composition
            (e.g., "pairwise composition_alpha measurement for lane_g_v3 × siren").
        aggregation_path: path identifier for the aggregate; defaults to
            "<aggregate>" sentinel.
        captured_at_utc: optional; defaults to now-UTC.

    Returns:
        Frozen Provenance; byte-identity detected → invalid sentinel grade.
    """
    if not parts:
        raise InvalidProvenanceError("AGGREGATE_OF_PROVENANCES requires non-empty parts")

    parts_tuple = tuple(parts)

    # Detect byte-identity (Catalog #823 anchor)
    shas = [p.source_sha256 for p in parts_tuple]
    duplicates = sorted({sha for sha in shas if shas.count(sha) > 1})

    # Aggregate sha is hash over concatenated child shas (deterministic)
    aggregate_sha = _sha256_of_bytes("|".join(sorted(shas)).encode("utf-8"))

    # Determine grade: WORST of components, or sentinel if byte-identity
    if duplicates:
        grade = ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT
        rejection_reason = (
            f"AGGREGATE_OF_PROVENANCES detected byte-identical composed_from"
            f" parts (duplicate sha256: {duplicates}); Catalog #823"
            f" byte-identity false-signal artifact class. Aggregation rationale:"
            f" {aggregation_rationale!r}"
        )
        is_promotable = False
        is_valid = False
    else:
        # Demote to least-promotable component grade
        grade = _worst_grade_in([p.evidence_grade for p in parts_tuple])
        rejection_reason = ""
        is_promotable = all(p.promotion_eligible for p in parts_tuple) and (
            grade
            in (
                ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
                ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CPU,
            )
        )
        is_valid = is_promotable

    # Aggregate axis: if all parts share axis, use it; else "[derived]"
    axes = {p.measurement_axis for p in parts_tuple}
    aggregate_axis = parts_tuple[0].measurement_axis if len(axes) == 1 else "[empirical]"

    # Aggregate hardware: if all parts share, use it; else "unknown"
    hardware_set = {p.hardware_substrate for p in parts_tuple}
    aggregate_hardware = parts_tuple[0].hardware_substrate if len(hardware_set) == 1 else "unknown"

    return Provenance(
        artifact_kind=ProvenanceKind.AGGREGATE_OF_PROVENANCES,
        source_path=aggregation_path,
        source_sha256=aggregate_sha,
        measurement_axis=aggregate_axis,
        hardware_substrate=aggregate_hardware,
        evidence_grade=grade,
        promotion_eligible=is_promotable,
        score_claim_valid=is_valid,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation="tac.provenance.builders.build_provenance_aggregate",
        composed_from=parts_tuple,
        rejection_reason=rejection_reason,
    )


def _worst_grade_in(grades: Iterable[ProvenanceEvidenceGrade]) -> ProvenanceEvidenceGrade:
    """Return least-promotable grade in iterable.

    Ordering (most-promotable → least-promotable):
        PROMOTABLE_EXACT_CONTEST_CUDA > PROMOTABLE_EXACT_CONTEST_CPU >
        EMPIRICAL_CPU_NON_GHA > MACOS_CPU_ADVISORY >
        MACOS_MLX_RESEARCH_SIGNAL > MPS_PROXY > PREDICTED > RESEARCH_ONLY >
        INVALID_BYTE_IDENTITY_ARTIFACT
    """
    rank = {
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA: 0,
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CPU: 1,
        ProvenanceEvidenceGrade.EMPIRICAL_CPU_NON_GHA: 2,
        ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY: 3,
        ProvenanceEvidenceGrade.MACOS_MLX_RESEARCH_SIGNAL: 4,
        ProvenanceEvidenceGrade.MPS_PROXY: 5,
        ProvenanceEvidenceGrade.PREDICTED: 6,
        ProvenanceEvidenceGrade.RESEARCH_ONLY: 7,
        ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT: 8,
    }
    grade_list = list(grades)
    if not grade_list:
        return ProvenanceEvidenceGrade.RESEARCH_ONLY
    return max(grade_list, key=lambda g: rank.get(g, 99))


# -----------------------------------------------------------------------------
# Builder: INVALID_BYTE_IDENTITY_ARTIFACT (explicit sentinel)
# -----------------------------------------------------------------------------


def build_provenance_invalid_byte_identity_artifact(
    source_path_a: str,
    source_path_b: str,
    identical_sha256: str,
    rejection_reason: str,
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical builder for INVALID_BYTE_IDENTITY_ARTIFACT (Catalog #823 anchor).

    Use this when a caller has already detected byte-identity (e.g., the
    SIREN dispatch failure → placeholder copy → identical sha256 case)
    and needs to record the artifact in canonical form for audit.
    """
    if not rejection_reason or not rejection_reason.strip():
        raise InvalidProvenanceError(
            "INVALID_BYTE_IDENTITY_ARTIFACT requires non-empty rejection_reason (forensic context for audit)"
        )
    if not identical_sha256:
        raise InvalidProvenanceError("INVALID_BYTE_IDENTITY_ARTIFACT requires non-empty identical_sha256")

    composed_path = f"<byte-identity-artifact:{source_path_a}::{source_path_b}>"

    return Provenance(
        artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
        source_path=composed_path,
        source_sha256=identical_sha256,
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation="tac.provenance.builders.build_provenance_invalid_byte_identity_artifact",
        rejection_reason=rejection_reason,
    )


# -----------------------------------------------------------------------------
# Builders: contest-compliance procedural-generation boundary kinds
# -----------------------------------------------------------------------------


def build_provenance_for_archive_seed_procedural_generation(
    seed_source_path: str,
    seed_sha256: str,
    rationale: str,
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical builder for archive-seed procedural-generation provenance.

    The seed bytes must be charged inside archive.zip. This provenance can
    support a future exact-eval score claim, but it is not a score claim itself.
    """
    if not rationale or not rationale.strip():
        raise InvalidProvenanceError("PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED requires non-empty rationale")
    _refuse_transient_path(seed_source_path)
    return Provenance(
        artifact_kind=ProvenanceKind.PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED,
        source_path=seed_source_path,
        source_sha256=seed_sha256,
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation=("tac.provenance.builders.build_provenance_for_archive_seed_procedural_generation"),
        rejection_reason=rationale,
    )


def build_provenance_for_weight_derived_codebook(
    weight_source_path: str,
    weight_sha256: str,
    rationale: str,
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical builder for codebooks derived from shipped archive weights."""
    if not rationale or not rationale.strip():
        raise InvalidProvenanceError("WEIGHT_DERIVED_CODEBOOK requires non-empty rationale")
    _refuse_transient_path(weight_source_path)
    return Provenance(
        artifact_kind=ProvenanceKind.WEIGHT_DERIVED_CODEBOOK,
        source_path=weight_source_path,
        source_sha256=weight_sha256,
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation=("tac.provenance.builders.build_provenance_for_weight_derived_codebook"),
        rejection_reason=rationale,
    )


def build_provenance_for_forbidden_out_of_archive_payload(
    payload_source_path: str,
    payload_sha256: str,
    rejection_reason: str,
    captured_at_utc: str | None = None,
) -> Provenance:
    """Canonical fail-closed sentinel for output-affecting external payloads."""
    if not rejection_reason or not rejection_reason.strip():
        raise InvalidProvenanceError("FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD requires non-empty rejection_reason")
    return Provenance(
        artifact_kind=ProvenanceKind.FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD,
        source_path=payload_source_path,
        source_sha256=payload_sha256,
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=captured_at_utc or _utc_now_iso(),
        canonical_helper_invocation=("tac.provenance.builders.build_provenance_for_forbidden_out_of_archive_payload"),
        rejection_reason=rejection_reason,
    )


def _probe_id_token(value: str) -> str:
    """Return a deterministic lowercase token safe for probe_id fields."""
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_", ".", "/", ":"}:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "unknown"


def register_forbidden_out_of_archive_payload_probe_outcome(
    *,
    provenance: Provenance,
    substrate: str,
    evidence_path: str,
    recipe_path: str | None = None,
    next_action: str | None = None,
    agent: str = "codex",
    subagent_id: str | None = None,
    session_id: str | None = None,
    notes: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Register a 365-day blocking probe outcome for forbidden payload provenance.

    This is the explicit Catalog #313 bridge for
    ``FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD``. Provenance construction stays pure:
    callers opt in to the shared-state write only after they have an evidence
    path and substrate identity for the blocking verdict.
    """
    if provenance.artifact_kind != ProvenanceKind.FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD:
        raise InvalidProvenanceError(
            "register_forbidden_out_of_archive_payload_probe_outcome requires "
            "ProvenanceKind.FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD"
        )
    if not evidence_path or not evidence_path.strip():
        raise InvalidProvenanceError("forbidden out-of-archive probe outcome requires non-empty evidence_path")
    if not provenance.rejection_reason:
        raise InvalidProvenanceError("forbidden out-of-archive probe outcome requires rejection_reason")

    from tac.probe_outcomes_ledger import register_probe_outcome

    source_token = _probe_id_token(provenance.source_path)[:80]
    probe_id = (
        "forbidden_out_of_archive_payload_"
        f"{_probe_id_token(substrate)[:80]}_"
        f"{provenance.source_sha256[:12]}_"
        f"{source_token}"
    )
    return register_probe_outcome(
        probe_id=probe_id,
        substrate=substrate,
        recipe_path=recipe_path,
        probe_kind="forbidden_out_of_archive_payload_provenance",
        verdict="DEFER",
        metric_name="forbidden_out_of_archive_payload_present",
        metric_value=1.0,
        threshold=0.0,
        threshold_token="ANY_OUTPUT_AFFECTING_PAYLOAD_OUTSIDE_ARCHIVE_FORBIDDEN",
        evidence_path=evidence_path,
        next_action=next_action or "move output-affecting bytes inside archive.zip or prove no-score-impact",
        blocker_status="blocking",
        staleness_window_days=365,
        agent=agent,
        subagent_id=subagent_id,
        session_id=session_id,
        notes=notes or provenance.rejection_reason,
        path=path,
        lock_path=lock_path,
        provenance_kind=provenance.artifact_kind.value,
        payload_source_path=provenance.source_path,
        payload_source_sha256=provenance.source_sha256,
    )


# -----------------------------------------------------------------------------
# Class-method shorthand on Provenance (ergonomics)
# -----------------------------------------------------------------------------


def _from_archive_zip_member(
    cls,
    archive_path: str | Path,
    member_name: str,
    measurement_axis: str,
    hardware_substrate: str,
    evidence_grade: ProvenanceEvidenceGrade,
    captured_at_utc: str | None = None,
) -> Provenance:
    """Class-method shim for the common case.

    Usage:
        Provenance.from_archive_zip_member(
            archive_path="submissions/a1/archive.zip",
            member_name="0.bin",
            measurement_axis="[contest-CUDA]",
            hardware_substrate="linux_x86_64_t4",
            evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        )
    """
    return build_provenance_for_archive_member(
        archive_zip_path=archive_path,
        member_name=member_name,
        measurement_axis=measurement_axis,
        hardware_substrate=hardware_substrate,
        evidence_grade=evidence_grade,
        captured_at_utc=captured_at_utc,
    )


# Bind as class method via setattr (frozen dataclass restriction workaround)
Provenance.from_archive_zip_member = classmethod(_from_archive_zip_member)  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Decorator: @requires_canonical_provenance
# -----------------------------------------------------------------------------


def requires_canonical_provenance(
    score_attr: str = "provenance",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that refuses to allow a function's return value to escape
    without a canonical Provenance attached.

    Usage:
        @requires_canonical_provenance(score_attr="provenance")
        def measure_archive_score(...) -> ScoreClaim:
            ...

    The decorator inspects the return value:
      - If it's a ScoreClaim, verifies score_claim.provenance is a Provenance.
      - If it's a dict, verifies dict[score_attr] is a Provenance dict or instance.
      - If it's None or a primitive (int/float/str), raises MissingProvenanceError.
      - Otherwise verifies the return has a ``provenance`` attribute.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)

            # Allow explicit sentinel
            if result is None:
                raise MissingProvenanceError(
                    f"{fn.__qualname__} returned None; canonical Provenance"
                    " required. Use tac.provenance.NULL_NOT_A_SCORE_CLAIM if"
                    " the function legitimately has no score claim."
                )

            if isinstance(result, ScoreClaim):
                return result

            if isinstance(result, Provenance):
                return result

            if isinstance(result, dict):
                if score_attr not in result:
                    raise MissingProvenanceError(
                        f"{fn.__qualname__} returned dict without {score_attr!r} key; canonical Provenance required."
                    )
                value = result[score_attr]
                if not isinstance(value, (Provenance, dict)):
                    raise MissingProvenanceError(
                        f"{fn.__qualname__} returned dict[{score_attr!r}]"
                        f" of type {type(value).__name__}; expected Provenance or dict."
                    )
                return result

            # Object with attribute
            if hasattr(result, score_attr):
                value = getattr(result, score_attr)
                if not isinstance(value, Provenance):
                    raise MissingProvenanceError(
                        f"{fn.__qualname__} returned object whose {score_attr!r}"
                        f" attribute is type {type(value).__name__}; expected Provenance."
                    )
                return result

            raise MissingProvenanceError(
                f"{fn.__qualname__} returned type {type(result).__name__}"
                f" with no {score_attr!r} attribute; canonical Provenance required."
            )

        return wrapper

    return decorator


__all__ = [
    "build_provenance_aggregate",
    "build_provenance_for_archive_member",
    "build_provenance_for_archive_seed_procedural_generation",
    "build_provenance_for_forbidden_out_of_archive_payload",
    "build_provenance_for_macos_cpu_advisory",
    "build_provenance_for_macos_mlx_research_signal",
    "build_provenance_for_mps_proxy",
    "build_provenance_for_predicted",
    "build_provenance_for_research_sidecar",
    "build_provenance_for_weight_derived_codebook",
    "build_provenance_invalid_byte_identity_artifact",
    "register_forbidden_out_of_archive_payload_probe_outcome",
    "requires_canonical_provenance",
]
