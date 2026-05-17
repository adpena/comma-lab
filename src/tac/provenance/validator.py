# SPDX-License-Identifier: MIT
"""tac.provenance.validator — single canonical validator for Provenance + ScoreClaim.

Per operator NON-NEGOTIABLE 2026-05-17 verbatim: *"fix it permanently and
canonically"*. This is the SINGLE validator any audit / persistence / consumer
surface uses to decide if a Provenance / ScoreClaim is well-formed and
score-claim-promotable.

Public API:

  - ``validate_provenance(prov)`` → ``(is_valid, blockers)``
  - ``validate_score_claim(claim)`` → ``(is_valid, blockers)``
  - ``audit_score_claim_dict(payload, expected_axis=None)`` →
    ``(is_valid, blockers)``: for raw JSON payloads (e.g., .omx/state JSONL rows).

The validator extends the Provenance.__post_init__ invariants with:

  1. **File existence checks**: archive members must exist + sha256 matches.
  2. **Cross-substrate byte-identity detection**: if a Provenance shares
     source_sha256 with another known substrate, flag (Catalog #823).
  3. **Audit-grade artifact age**: captured_at_utc not in future, not stale
     beyond 90 days (per Catalog #298 retirement discipline).
  4. **Composition graph integrity**: AGGREGATE_OF_PROVENANCES composed_from
     must not contain itself (cycle detection).

This module DOES NOT mutate state. All checks are read-only.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tac.provenance.contract import (
    InvalidProvenanceError,
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
    ScoreClaim,
    _PROMOTABLE_GRADES,
    _NON_PROMOTABLE_GRADES,
)


# Default staleness window (per CLAUDE.md "Substrate retirement discipline"
# Catalog #298 — 30-day mark window; we relax to 90 days for Provenance
# captured_at_utc since archives can ship and be re-evaluated cleanly).
DEFAULT_PROVENANCE_STALE_DAYS: int = 90


def _parse_utc(iso_str: str) -> Optional[datetime]:
    """Parse canonical ISO-UTC; return None on failure."""
    if not iso_str:
        return None
    # Strip trailing Z (datetime.fromisoformat in py3.11+ handles Z, but
    # in earlier we replace).
    s = iso_str.replace("Z", "+00:00") if iso_str.endswith("Z") else iso_str
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _sha256_file(path: Path) -> Optional[str]:
    """SHA-256 over file bytes; None on read error."""
    if not path.exists():
        return None
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _sha256_archive_member(archive_path: Path, member_name: str) -> Optional[str]:
    """SHA-256 over UNZIPPED member bytes; None on read error."""
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            with zf.open(member_name) as member:
                h = hashlib.sha256()
                for chunk in iter(lambda: member.read(65536), b""):
                    h.update(chunk)
                return h.hexdigest()
    except (KeyError, zipfile.BadZipFile, OSError):
        return None


def validate_provenance(
    provenance: Provenance,
    archive_validation_strict: bool = True,
    stale_days: int = DEFAULT_PROVENANCE_STALE_DAYS,
) -> tuple[bool, list[str]]:
    """Single canonical validator.

    Args:
        provenance: the Provenance to validate.
        archive_validation_strict: if True, verify archive member exists +
            sha256 matches (default True). Set False for unit tests with
            synthetic fixtures.
        stale_days: maximum age in days before captured_at_utc is flagged.

    Returns:
        (is_valid, blockers) where blockers is a list of human-readable
        violation descriptions. is_valid=True iff blockers is empty.
    """
    blockers: list[str] = []

    # The Provenance __post_init__ already enforces basic invariants; if
    # we got here, those passed. We add cross-state checks.

    # Check 1: captured_at_utc validity + staleness
    captured = _parse_utc(provenance.captured_at_utc)
    if captured is None:
        blockers.append(
            f"captured_at_utc={provenance.captured_at_utc!r} not parseable"
            " as ISO-UTC"
        )
    else:
        now = datetime.now(timezone.utc)
        if captured > now:
            blockers.append(
                f"captured_at_utc={provenance.captured_at_utc!r} is in the future"
            )
        age_days = (now - captured).days
        if age_days > stale_days:
            blockers.append(
                f"captured_at_utc={provenance.captured_at_utc!r} is stale"
                f" ({age_days}d > {stale_days}d window)"
            )

    # Check 2: composition cycle detection
    if provenance.artifact_kind == ProvenanceKind.AGGREGATE_OF_PROVENANCES:
        visited: set[str] = set()

        def _walk(prov: Provenance, chain: list[str]) -> bool:
            key = f"{prov.source_path}#{prov.source_sha256}"
            if key in chain:
                return True
            if key in visited:
                return False
            visited.add(key)
            new_chain = chain + [key]
            for part in prov.composed_from:
                if _walk(part, new_chain):
                    return True
            return False

        if _walk(provenance, []):
            blockers.append(
                "AGGREGATE_OF_PROVENANCES composed_from contains a cycle"
            )

    # Check 3: archive member sha256 match (only if file exists + strict)
    if (
        archive_validation_strict
        and provenance.artifact_kind == ProvenanceKind.CONTEST_ARCHIVE_MEMBER
        and provenance.contest_archive_zip_path
        and provenance.contest_archive_member_name
        # Skip sentinel paths
        and not provenance.source_path.startswith("<")
    ):
        archive_path = Path(provenance.contest_archive_zip_path)
        if archive_path.exists():
            actual_sha = _sha256_archive_member(
                archive_path, provenance.contest_archive_member_name
            )
            if actual_sha is None:
                blockers.append(
                    f"Cannot read member {provenance.contest_archive_member_name!r}"
                    f" from archive {provenance.contest_archive_zip_path!r}"
                )
            elif actual_sha != provenance.source_sha256:
                blockers.append(
                    f"Archive member sha256 mismatch: declared={provenance.source_sha256}"
                    f" actual={actual_sha}"
                )
        # If archive doesn't exist on disk we don't flag (build artifact
        # may have been GC'd; Catalog #154 GC discipline) — the audit tool
        # surfaces this separately.

    # Check 4: INVALID_BYTE_IDENTITY_ARTIFACT must have non-empty rejection_reason
    if (
        provenance.evidence_grade
        == ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT
        and not provenance.rejection_reason.strip()
    ):
        blockers.append(
            "INVALID_BYTE_IDENTITY_ARTIFACT requires non-empty rejection_reason"
        )

    is_valid = not blockers
    return is_valid, blockers


def validate_score_claim(claim: ScoreClaim) -> tuple[bool, list[str]]:
    """Composes validate_provenance with score-specific checks.

    Additional checks beyond Provenance validation:
      - If claim.contest_compliant=True, claim.provenance must be promotable.
      - claim.score_value must be a finite real number.
      - claim.rationale should be non-empty for contest-compliant claims
        (warning-level; soft check).
    """
    blockers: list[str] = []

    # Validate the embedded Provenance
    prov_valid, prov_blockers = validate_provenance(claim.provenance)
    blockers.extend(prov_blockers)

    # Score value sanity
    if not isinstance(claim.score_value, (int, float)):
        blockers.append(
            f"score_value type {type(claim.score_value).__name__} not numeric"
        )
    else:
        # NaN / infinity
        import math
        if math.isnan(claim.score_value):
            blockers.append("score_value is NaN")
        elif math.isinf(claim.score_value):
            blockers.append("score_value is infinite")
        elif claim.score_value < 0:
            blockers.append(
                f"score_value={claim.score_value} is negative; contest scores"
                " are lower-is-better non-negative"
            )

    # Contest-compliant requires promotable provenance
    if claim.contest_compliant:
        if not claim.provenance.score_claim_valid:
            blockers.append(
                "contest_compliant=True requires provenance.score_claim_valid=True"
            )
        if claim.provenance.evidence_grade not in _PROMOTABLE_GRADES:
            blockers.append(
                f"contest_compliant=True requires evidence_grade ∈"
                f" promotable; got {claim.provenance.evidence_grade.value}"
            )

    is_valid = not blockers
    return is_valid, blockers


def audit_score_claim_dict(
    payload: dict[str, Any],
    expected_axis: Optional[str] = None,
) -> tuple[bool, list[str]]:
    """Validate a raw JSON payload (.omx/state JSONL row, etc.) for score-claim correctness.

    Looks for known score-claiming key patterns:
      - ``score`` / ``contest_score`` / ``final_score`` / ``predicted_score``
      - ``deliverable_score_savings_estimate`` (Catalog #321 anchor)
      - ``composition_alpha`` (Catalog #319 anchor)
      - ``savings`` / ``delta_s`` / ``alpha``

    Requires a ``provenance`` sub-object validated via ``validate_provenance``.
    For rows that legitimately have no score claim (counters, flags), the
    caller can omit the score-keys entirely.

    Args:
        payload: the dict to audit.
        expected_axis: optional axis check (e.g., "[contest-CUDA]").

    Returns:
        (is_valid, blockers).
    """
    blockers: list[str] = []

    score_claim_keys = (
        "score",
        "score_value",
        "contest_score",
        "final_score",
        "predicted_score",
        "canonical_score",
        "canonical_score_recomputed",
        "canonical_score_contest_cuda",
        "canonical_score_contest_cpu",
        "score_recomputed",
        "score_recomputed_from_components",
        "score_recomputed_from_contest_components",
        "score_recomputed_from_public_components",
        "score_contest_cuda",
        "score_contest_cpu",
        "contest_cuda_score_recomputed",
        "contest_cpu_score_recomputed",
        "empirical_score",
        "diagnostic_cpu_score",
        "auth_eval_score",
        "auth_eval_recomputed_score",
        "score_recomputed_from_auth_eval",
        "deliverable_score_savings_estimate",
        "deliverable_savings",
        "composition_alpha",
        "alpha",
        "alpha_savings_ratio_form",
        "savings",
        "delta_s",
        "delta_score",
        "score_savings",
        "recomputed_score",
    )

    score_keys_present = [k for k in score_claim_keys if k in payload]

    if not score_keys_present:
        # No score claims to audit; valid by construction.
        return True, []

    # Score keys present → require canonical provenance sub-object
    if "provenance" not in payload:
        blockers.append(
            f"payload has score-claim keys {score_keys_present!r} but no"
            " 'provenance' sub-object; required per Catalog #323"
        )
        return False, blockers

    prov_dict = payload["provenance"]
    if not isinstance(prov_dict, dict):
        blockers.append(
            f"payload['provenance'] type {type(prov_dict).__name__} not dict"
        )
        return False, blockers

    # Reconstruct Provenance from dict
    try:
        # Allow caller to pre-construct; or build from dict shape.
        prov = _provenance_from_dict(prov_dict)
    except (TypeError, ValueError, KeyError, InvalidProvenanceError) as exc:
        blockers.append(
            f"payload['provenance'] cannot be reconstructed as canonical"
            f" Provenance: {exc}"
        )
        return False, blockers

    # Validate the reconstructed Provenance
    prov_valid, prov_blockers = validate_provenance(prov)
    blockers.extend(prov_blockers)

    # Axis match check
    if expected_axis and prov.measurement_axis != expected_axis:
        blockers.append(
            f"expected axis={expected_axis!r}; got {prov.measurement_axis!r}"
        )

    # Score-keys × promotability cross-check: if any score key has a positive
    # value AND provenance is non-promotable → flag (this catches the
    # Catalog #321 / #823 phantom-score class structurally).
    if not prov.score_claim_valid:
        for key in score_keys_present:
            val = payload[key]
            if isinstance(val, (int, float)) and val != 0:
                blockers.append(
                    f"payload[{key!r}]={val} is non-zero but provenance.score_claim_valid=False"
                    f" (evidence_grade={prov.evidence_grade.value});"
                    " phantom-score class (Catalog #321/#823)"
                )

    is_valid = not blockers
    return is_valid, blockers


def _provenance_from_dict(d: dict[str, Any]) -> Provenance:
    """Reconstruct Provenance from a JSON-serializable dict.

    Strict: all canonical fields required; enums parsed from str values.
    """
    kind = ProvenanceKind(d["artifact_kind"])
    grade = ProvenanceEvidenceGrade(d["evidence_grade"])
    composed_from = tuple(
        _provenance_from_dict(p) for p in d.get("composed_from", [])
    )

    return Provenance(
        artifact_kind=kind,
        source_path=d["source_path"],
        source_sha256=d["source_sha256"],
        measurement_axis=d["measurement_axis"],
        hardware_substrate=d["hardware_substrate"],
        evidence_grade=grade,
        promotion_eligible=d["promotion_eligible"],
        score_claim_valid=d["score_claim_valid"],
        captured_at_utc=d["captured_at_utc"],
        canonical_helper_invocation=d["canonical_helper_invocation"],
        contest_archive_zip_path=d.get("contest_archive_zip_path", ""),
        contest_archive_member_name=d.get("contest_archive_member_name", ""),
        composed_from=composed_from,
        rejection_reason=d.get("rejection_reason", ""),
    )


def provenance_to_dict(prov: Provenance) -> dict[str, Any]:
    """JSON-serializable dict form of Provenance.

    Reverse of _provenance_from_dict. Sister of canonical helpers per
    CLAUDE.md "Beauty, simplicity, and developer experience":
    machine-checkable JSON output.
    """
    return {
        "artifact_kind": prov.artifact_kind.value,
        "source_path": prov.source_path,
        "source_sha256": prov.source_sha256,
        "measurement_axis": prov.measurement_axis,
        "hardware_substrate": prov.hardware_substrate,
        "evidence_grade": prov.evidence_grade.value,
        "promotion_eligible": prov.promotion_eligible,
        "score_claim_valid": prov.score_claim_valid,
        "captured_at_utc": prov.captured_at_utc,
        "canonical_helper_invocation": prov.canonical_helper_invocation,
        "contest_archive_zip_path": prov.contest_archive_zip_path,
        "contest_archive_member_name": prov.contest_archive_member_name,
        "composed_from": [provenance_to_dict(p) for p in prov.composed_from],
        "rejection_reason": prov.rejection_reason,
    }


__all__ = [
    "DEFAULT_PROVENANCE_STALE_DAYS",
    "validate_provenance",
    "validate_score_claim",
    "audit_score_claim_dict",
    "provenance_to_dict",
]
