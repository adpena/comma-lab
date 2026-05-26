# SPDX-License-Identifier: MIT
"""tac.provenance.adapters — canonical surface migration ADAPTERS.

Per CLAUDE.md "Subagent coherence-by-default" + "Bugs must be permanently
fixed AND self-protected against" + the briefing's explicit guidance:
*"each migration is BACKWARD-COMPATIBLE via shim (legacy rows without
Provenance get auto-promoted to RESEARCH_SIDECAR + RESEARCH_ONLY grade +
score_claim_valid=False; gives the audit tool grist to surface them as
migration candidates)"*.

The 8 canonical surfaces named in the briefing are HEAVILY contended by
parallel sister subagents. Rather than editing each dataclass in-place
(which would collide with REDO_PIVOT_FIX_ALL etc.), this adapter module
provides:

  * ``contest_result_to_provenance(result)`` — adapt
    ``tac.continual_learning.ContestResult`` → ``Provenance``.
  * ``cost_band_anchor_to_provenance(anchor)`` — adapt cost-band anchor row.
  * ``council_record_to_provenance(record)`` — adapt
    ``tac.council_continual_learning.CouncilDeliberationRecord`` evidence.
  * ``substrate_composition_row_to_provenance(row)`` — adapt
    ``tac.optimization.substrate_composition_matrix.CompositionResult``.
  * ``deliverability_proof_to_provenance(proof)`` — adapt
    ``tac.wyner_ziv_deliverability.DeliverabilityProof``.
  * ``wyner_ziv_layer_result_to_provenance(result)`` — adapt
    ``tac.codec.wyner_ziv_layer.WynerZivLayerResult``.
  * ``master_gradient_plan_to_provenance(plan)`` — adapt
    ``tac.master_gradient_consumers.OptimalPerPairTreatmentPlan``.
  * ``modal_call_id_ledger_event_to_provenance(event)`` — adapt
    ``tac.deploy.modal.call_id_ledger`` events.

Each adapter is RESILIENT: it accepts the legacy shape (dict OR dataclass)
and returns a Provenance with the BEST available evidence_grade given the
legacy field values. Fields that don't exist in the legacy shape default
to RESEARCH_ONLY + non-promotable per the briefing's backward-compat shim.

When a sister subagent later embeds the Provenance directly into the
canonical dataclass, the adapter remains as a fallback for legacy rows
that predate the field addition. The canonical helper hierarchy:

  1. NEW canonical surface: dataclass has ``provenance: Provenance``.
  2. LEGACY surface without provenance: adapter converts on-the-fly.
  3. CORRUPT surface: adapter returns NULL_NOT_A_SCORE_CLAIM sentinel.

This is the "make it easy" directive operationalized — consumers always
get a Provenance, never a None.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from tac.provenance.contract import (
    NULL_NOT_A_SCORE_CLAIM,
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
)
from tac.provenance.builders import (
    build_provenance_for_research_sidecar,
    build_provenance_for_predicted,
    build_provenance_for_macos_cpu_advisory,
    build_provenance_for_mps_proxy,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _placeholder_sha() -> str:
    return "0" * 64


def _normalize_axis_string(axis: str | None) -> str:
    """Normalize legacy axis-tag strings to canonical bracketed form."""
    if not axis:
        return "[research-signal]"
    a = axis.strip()
    if not a.startswith("["):
        # Common legacy forms: "contest-CUDA" / "contest_cuda" / "cuda"
        low = a.lower().replace("_", "-")
        if "cuda" in low and "contest" in low:
            return "[contest-CUDA]"
        if "cpu" in low and "contest" in low and "macos" not in low:
            return "[contest-CPU]"
        if "macos" in low or "mac-os" in low:
            return "[macOS-CPU advisory]"
        if "mps" in low:
            return "[MPS-PROXY]"
        if "predict" in low:
            return "[predicted]"
        if "advisory" in low:
            return "[advisory only]"
        return "[research-signal]"
    return a


def _normalize_hardware_substrate(hw: str | None) -> str:
    if not hw:
        return "unknown"
    h = hw.strip().lower().replace("-", "_")
    # Canonical short→long
    mappings = {
        "t4": "linux_x86_64_t4",
        "a100": "linux_x86_64_a100",
        "a10g": "linux_x86_64_a10g",
        "h100": "linux_x86_64_h100",
        "4090": "linux_x86_64_4090",
        "l40s": "linux_x86_64_l40s",
        "cpu": "linux_x86_64_cpu",
        "modal_cpu": "linux_x86_64_modal_cpu",
        "darwin": "macos_arm64",
        "macos": "macos_arm64",
    }
    return mappings.get(h, h if h.startswith(("linux_", "macos_", "windows_")) else "unknown")


def _grade_for_axis(axis: str) -> ProvenanceEvidenceGrade:
    """Best-effort grade inference from canonical axis."""
    if axis == "[contest-CUDA]":  # CUSTODY_VALIDATOR_OK:this_function_IS_provenance_grade_inference_for_canonical_axis_per_comprehensive_bug_audit_cascade_20260526
        return ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA
    if axis == "[contest-CPU]":  # CUSTODY_VALIDATOR_OK:this_function_IS_provenance_grade_inference_for_canonical_axis_per_comprehensive_bug_audit_cascade_20260526
        return ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CPU
    if axis == "[macOS-CPU advisory]":
        return ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY
    if axis == "[MPS-PROXY]":
        return ProvenanceEvidenceGrade.MPS_PROXY
    if axis == "[predicted]":
        return ProvenanceEvidenceGrade.PREDICTED
    return ProvenanceEvidenceGrade.RESEARCH_ONLY


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Return obj.name OR obj[name] OR default — works for dataclass + dict."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# -----------------------------------------------------------------------------
# ContestResult adapter
# -----------------------------------------------------------------------------

def contest_result_to_provenance(result: Any) -> Provenance:
    """Adapt ``tac.continual_learning.ContestResult`` → ``Provenance``.

    Reads canonical legacy fields:
      * ``archive_sha256`` (or ``archive_path``)
      * ``evidence_grade`` (legacy strings)
      * ``axis`` (or ``score_axis``)
      * ``hardware_substrate`` (or ``hardware``)
      * ``captured_at_utc`` (or ``timestamp``)

    Returns a Provenance with the BEST-available grade. If the result is
    a `contest-CUDA` anchor with full custody, returns PROMOTABLE_EXACT_CONTEST_CUDA.
    Otherwise demotes to RESEARCH_ONLY per backward-compat shim.
    """
    archive_sha = _safe_attr(result, "archive_sha256") or _safe_attr(result, "archive_sha") or _placeholder_sha()
    axis_raw = _safe_attr(result, "axis") or _safe_attr(result, "score_axis")
    hardware_raw = _safe_attr(result, "hardware_substrate") or _safe_attr(result, "hardware")
    captured_raw = _safe_attr(result, "captured_at_utc") or _safe_attr(result, "timestamp")
    archive_path = _safe_attr(result, "archive_path") or _safe_attr(result, "archive_zip_path")
    member_name = _safe_attr(result, "archive_member_name") or "0.bin"

    axis = _normalize_axis_string(axis_raw)
    hardware = _normalize_hardware_substrate(hardware_raw)
    captured = captured_raw if isinstance(captured_raw, str) and captured_raw else _utc_now_iso()

    # Determine grade
    grade = _grade_for_axis(axis)

    # Legacy ContestResult sometimes carries a 'promotion_eligible' field;
    # honor it as a downgrade signal if False
    legacy_promotion = _safe_attr(result, "promotion_eligible", True)
    if not legacy_promotion and grade in (
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CPU,
    ):
        grade = ProvenanceEvidenceGrade.EMPIRICAL_CPU_NON_GHA

    # Validate sha256 shape; replace with placeholder if invalid
    if not isinstance(archive_sha, str) or len(archive_sha) != 64 or not all(c in "0123456789abcdef" for c in archive_sha.lower()):
        archive_sha = _placeholder_sha()
    else:
        archive_sha = archive_sha.lower()

    # Build source_path: prefer archive:member form if we have archive_path
    if archive_path:
        source_path = f"{archive_path}:{member_name}"
    else:
        source_path = f"<contest_result:{archive_sha[:12]}>"

    # For PROMOTABLE grades we need CONTEST_ARCHIVE_MEMBER kind; else demote
    if grade in (
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CPU,
    ):
        if not archive_path:
            # No archive path → cannot claim CONTEST_ARCHIVE_MEMBER kind →
            # demote per backward-compat shim
            grade = ProvenanceEvidenceGrade.EMPIRICAL_CPU_NON_GHA
            kind = ProvenanceKind.DERIVED_AGGREGATE
            promotion_eligible = False
            score_claim_valid = False
        else:
            kind = ProvenanceKind.CONTEST_ARCHIVE_MEMBER
            promotion_eligible = True
            score_claim_valid = True
    else:
        kind = ProvenanceKind.DERIVED_AGGREGATE
        promotion_eligible = False
        score_claim_valid = False

    try:
        return Provenance(
            artifact_kind=kind,
            source_path=source_path,
            source_sha256=archive_sha,
            measurement_axis=axis,
            hardware_substrate=hardware,
            evidence_grade=grade,
            promotion_eligible=promotion_eligible,
            score_claim_valid=score_claim_valid,
            captured_at_utc=captured,
            canonical_helper_invocation="tac.provenance.adapters.contest_result_to_provenance",
            contest_archive_zip_path=str(archive_path) if archive_path else "",
            contest_archive_member_name=member_name if archive_path else "",
        )
    except Exception:
        # Total fallback: NULL sentinel
        return NULL_NOT_A_SCORE_CLAIM


# -----------------------------------------------------------------------------
# Cost-band anchor adapter
# -----------------------------------------------------------------------------

def cost_band_anchor_to_provenance(anchor: Any) -> Provenance:
    """Adapt ``tac.cost_band_calibration`` anchor row → ``Provenance``.

    Cost-band anchors carry GPU dispatch outcomes; the score itself is
    typically a ContestResult under-the-hood, so we delegate when possible.
    """
    # Try to extract a nested ContestResult
    nested = _safe_attr(anchor, "contest_result")
    if nested is not None:
        return contest_result_to_provenance(nested)
    # Else build a research_sidecar-style provenance with cost-band metadata
    return build_provenance_for_research_sidecar(
        sidecar_path=str(_safe_attr(anchor, "anchor_path", "<cost_band_anchor>")),
        reactivation_criteria="cost-band anchor adapter — legacy row pending Provenance migration",
    )


# -----------------------------------------------------------------------------
# CouncilDeliberationRecord adapter
# -----------------------------------------------------------------------------

def council_record_to_provenance(record: Any) -> Provenance:
    """Adapt ``CouncilDeliberationRecord`` → ``Provenance`` for evidence cite-chain."""
    deliberation_id = _safe_attr(record, "deliberation_id", "unknown")
    topic = _safe_attr(record, "topic", "unknown")
    return build_provenance_for_research_sidecar(
        sidecar_path=f"<council_deliberation:{deliberation_id}>",
        reactivation_criteria=f"council deliberation evidence: {topic}",
    )


# -----------------------------------------------------------------------------
# SubstrateCompositionRow adapter (sister-contended)
# -----------------------------------------------------------------------------

def substrate_composition_row_to_provenance(row: Any) -> Provenance:
    """Adapt ``CompositionResult`` (or sister composition row dict) → ``Provenance``.

    Sister-contended by REDO_PIVOT_FIX_ALL; this adapter is the
    backward-compat shim that lets consumers read legacy rows without
    waiting for the dataclass migration to land.

    If the row's pair_key suggests byte-identity (Catalog #823) the
    adapter returns INVALID_BYTE_IDENTITY_ARTIFACT sentinel.
    """
    pair_key = _safe_attr(row, "pair_key") or "unknown_pair"
    sha_a = _safe_attr(row, "candidate_a_sha256", "")
    sha_b = _safe_attr(row, "candidate_b_sha256", "")
    verdict = _safe_attr(row, "verdict", "")

    # Catalog #823 byte-identity detection
    if sha_a and sha_b and sha_a == sha_b:
        from tac.provenance.builders import build_provenance_invalid_byte_identity_artifact
        return build_provenance_invalid_byte_identity_artifact(
            source_path_a=str(_safe_attr(row, "candidate_a_path", "<a>")),
            source_path_b=str(_safe_attr(row, "candidate_b_path", "<b>")),
            identical_sha256=sha_a.lower(),
            rejection_reason=f"composition row pair_key={pair_key} verdict={verdict}",
        )

    # If verdict carries phantom-score markers, demote
    if isinstance(verdict, str) and any(
        m in verdict.lower() for m in ("phantom", "false_signal", "byte_identity")
    ):
        return build_provenance_for_research_sidecar(
            sidecar_path=f"<composition_row:{pair_key}>",
            reactivation_criteria=f"phantom composition row: {verdict}",
        )

    return build_provenance_for_research_sidecar(
        sidecar_path=f"<composition_row:{pair_key}>",
        reactivation_criteria="composition row adapter — legacy row pending Provenance field migration",
    )


# -----------------------------------------------------------------------------
# DeliverabilityProof adapter
# -----------------------------------------------------------------------------

def deliverability_proof_to_provenance(proof: Any) -> Provenance:
    """Adapt ``DeliverabilityProof`` → ``Provenance``.

    The proof's archive_sha256 + tier info maps cleanly; if the proof is
    Tier 4 (FORBIDDEN), the adapter demotes to RESEARCH_ONLY.
    """
    archive_sha = _safe_attr(proof, "archive_sha256", _placeholder_sha())
    tier = _safe_attr(proof, "tier", None)
    if tier is not None:
        tier_str = tier.value if hasattr(tier, "value") else str(tier)
        if "FORBIDDEN" in tier_str.upper():
            return build_provenance_for_research_sidecar(
                sidecar_path=f"<deliverability_proof:{archive_sha[:12]}>",
                reactivation_criteria=f"tier 4 forbidden: {tier_str}",
            )

    return build_provenance_for_research_sidecar(
        sidecar_path=f"<deliverability_proof:{archive_sha[:12]}>",
        reactivation_criteria="deliverability proof adapter — legacy row pending field embedding",
    )


# -----------------------------------------------------------------------------
# WynerZivLayerResult adapter
# -----------------------------------------------------------------------------

def wyner_ziv_layer_result_to_provenance(result: Any) -> Provenance:
    """Adapt ``WynerZivLayerResult`` → ``Provenance``."""
    layer_name = _safe_attr(result, "layer_name", "unknown")
    return build_provenance_for_research_sidecar(
        sidecar_path=f"<wyner_ziv_layer:{layer_name}>",
        reactivation_criteria="wyner-ziv layer result adapter — legacy",
    )


# -----------------------------------------------------------------------------
# OptimalPerPairTreatmentPlan adapter
# -----------------------------------------------------------------------------

def master_gradient_plan_to_provenance(plan: Any) -> Provenance:
    """Adapt ``OptimalPerPairTreatmentPlan`` → ``Provenance``."""
    plan_id = _safe_attr(plan, "plan_id") or _safe_attr(plan, "lane_id", "unknown")
    archive_path = _safe_attr(plan, "archive_path", "")
    archive_sha = _safe_attr(plan, "archive_sha256", _placeholder_sha())

    if archive_path and archive_sha and len(archive_sha) == 64:
        # Promotable-shape — but adapter is conservative; use derived aggregate
        return Provenance(
            artifact_kind=ProvenanceKind.DERIVED_AGGREGATE,
            source_path=f"<plan:{plan_id}>",
            source_sha256=archive_sha.lower(),
            measurement_axis="[research-signal]",
            hardware_substrate="unknown",
            evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc=_utc_now_iso(),
            canonical_helper_invocation="tac.provenance.adapters.master_gradient_plan_to_provenance",
        )

    return build_provenance_for_research_sidecar(
        sidecar_path=f"<plan:{plan_id}>",
        reactivation_criteria="master-gradient plan adapter — legacy",
    )


# -----------------------------------------------------------------------------
# Modal call_id ledger event adapter
# -----------------------------------------------------------------------------

def modal_call_id_ledger_event_to_provenance(event: Any) -> Provenance:
    """Adapt Modal call_id ledger event → ``Provenance``."""
    call_id = _safe_attr(event, "call_id", "unknown")
    score = _safe_attr(event, "score")
    score_axis = _safe_attr(event, "score_axis")
    archive_sha = _safe_attr(event, "archive_sha256", _placeholder_sha())
    hardware = _safe_attr(event, "platform") or _safe_attr(event, "gpu")
    captured = _safe_attr(event, "written_at_utc") or _utc_now_iso()

    if score is not None:
        axis = _normalize_axis_string(score_axis)
        hw = _normalize_hardware_substrate(hardware)
        grade = _grade_for_axis(axis)
        # Conservative: ledger events become DERIVED_AGGREGATE non-promotable
        # until the canonical caller embeds Provenance directly.
        return Provenance(
            artifact_kind=ProvenanceKind.DERIVED_AGGREGATE,
            source_path=f"<modal_call:{call_id}>",
            source_sha256=archive_sha.lower() if len(archive_sha) == 64 else _placeholder_sha(),
            measurement_axis=axis,
            hardware_substrate=hw,
            evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
            promotion_eligible=False,
            score_claim_valid=False,
            captured_at_utc=captured,
            canonical_helper_invocation="tac.provenance.adapters.modal_call_id_ledger_event_to_provenance",
        )

    return build_provenance_for_research_sidecar(
        sidecar_path=f"<modal_call:{call_id}>",
        reactivation_criteria="modal call_id ledger event adapter — no score yet",
    )


__all__ = [
    "contest_result_to_provenance",
    "cost_band_anchor_to_provenance",
    "council_record_to_provenance",
    "substrate_composition_row_to_provenance",
    "deliverability_proof_to_provenance",
    "wyner_ziv_layer_result_to_provenance",
    "master_gradient_plan_to_provenance",
    "modal_call_id_ledger_event_to_provenance",
]
