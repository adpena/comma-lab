"""Fail-closed Omega-OPT nested optimization claim contracts.

The Omega-OPT score numbers are research hypotheses until a matching 1:1
archive/eval artifact exists. This module is intentionally small and
side-effect free so planning tools, scanners, and tests can share the same
promotion discipline.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

OMEGA_OPT_CLAIM_SCHEMA = "tac_omega_opt_nested_claims_v1"

FAIL_CLOSED_FIELDS: tuple[str, ...] = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)

PROMOTION_ALIAS_FIELDS: tuple[str, ...] = (
    "promotion_allowed",
    "dispatchable",
)

EXACT_ARCHIVE_SHA_FIELDS: tuple[str, ...] = (
    "exact_archive_sha256",
    "archive_sha256",
)
EXACT_ARCHIVE_BYTES_FIELDS: tuple[str, ...] = (
    "exact_archive_bytes",
    "archive_bytes",
)
EXACT_AUTH_EVAL_FIELDS: tuple[str, ...] = (
    "contest_auth_eval_json",
    "exact_cuda_auth_eval_json",
    "exact_cuda_auth_eval_artifact",
)
EXACT_1TO1_ANCHOR_FIELDS: tuple[str, ...] = (
    "one_to_one_anchor_artifact",
    "one_to_one_archive_eval_artifact",
    "one_to_one_config_manifest",
)

OMEGA_OPT_TEXT_MARKERS: tuple[str, ...] = (
    "omega_opt",
    "omega-opt",
    "omega opt",
    "\u03a9-opt",
    "\u03a9 opt",
)


@dataclass(frozen=True)
class OmegaOptClaim:
    """One Omega-OPT predicted score level and its promotion blockers."""

    claim_id: str
    label: str
    predicted_score: float
    score_classification: str
    current_anchor_status: str
    missing_anchors: tuple[str, ...]
    next_buildable_1to1_test: str
    reactivation_criteria: tuple[str, ...]
    notes: str = ""

    def to_manifest(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "label": self.label,
            "predicted_score": self.predicted_score,
            "score_classification": self.score_classification,
            "current_anchor_status": self.current_anchor_status,
            "missing_anchors": list(self.missing_anchors),
            "next_buildable_1to1_test": self.next_buildable_1to1_test,
            "reactivation_criteria": list(self.reactivation_criteria),
            "notes": self.notes,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotion_allowed": False,
            "dispatchable": False,
            "requires_exact_1to1_anchor": True,
        }


OMEGA_OPT_CLAIMS: tuple[OmegaOptClaim, ...] = (
    OmegaOptClaim(
        claim_id="omega_opt_linear_stack",
        label="Linear stack",
        predicted_score=0.130,
        score_classification="prediction",
        current_anchor_status="cpu_byte_anchor_only_no_score_anchor",
        missing_anchors=(
            "1:1 composition config manifest for the claimed stack",
            "runtime packet that consumes the composed payload",
            "full retrain or faithful no-retrain disclosure for every layer",
            "full-sample exact CUDA auth eval on exact archive bytes",
        ),
        next_buildable_1to1_test=(
            "Build one PR101 stack archive from the declared arch_shrink, IMP, "
            "lossy coarsening, and brotli sequence; record old/new archive "
            "SHA-256, then run archive.zip -> inflate.sh -> upstream/evaluate.py "
            "on CUDA."
        ),
        reactivation_criteria=(
            "score-affecting payload changed and consumed by inflate",
            "old/new archive bytes and SHA-256 recorded",
            "contest_auth_eval.json records full-sample CUDA components",
        ),
        notes="The existing post-hoc CPU byte measurement is useful, but the 0.130 score remains unanchored.",
    ),
    OmegaOptClaim(
        claim_id="omega_opt_multipass_imp_cycle",
        label="Multi-pass IMP-cycle",
        predicted_score=0.115,
        score_classification="prediction",
        current_anchor_status="no_matching_config_anchor",
        missing_anchors=(
            "declared Q-FAITHFUL five-stage config",
            "IMP-cycle archive materialization",
            "per-pass charged-byte ledger",
            "full-sample exact CUDA auth eval on the final packet",
        ),
        next_buildable_1to1_test=(
            "Freeze a minimal Q-FAITHFUL plus one IMP-cycle config, emit a "
            "single final contest packet, and compare exact CUDA auth eval "
            "against the non-IMP anchor."
        ),
        reactivation_criteria=(
            "matching config committed or manifested",
            "each pass records input/output payload SHA-256",
            "final archive has exact CUDA auth-eval artifact",
        ),
    ),
    OmegaOptClaim(
        claim_id="omega_opt_hstack_of_vstacks",
        label="HStack-of-VStacks",
        predicted_score=0.110,
        score_classification="design",
        current_anchor_status="never_built",
        missing_anchors=(
            "parser-proven internal stream map",
            "serial VStack transform manifests",
            "parallel HStack merge manifest",
            "runtime packet closure and CUDA auth eval",
        ),
        next_buildable_1to1_test=(
            "Use the codec-stack planner to materialize one parser-proven "
            "component map on PR101 or PR106, then build an identity packet "
            "before changing bytes."
        ),
        reactivation_criteria=(
            "identity packet round-trips byte-for-byte or explains metadata deltas",
            "non-identity packet records per-stream byte deltas",
            "exact CUDA eval exists before any score ranking",
        ),
    ),
    OmegaOptClaim(
        claim_id="omega_opt_joint_admm_cross_component",
        label="Joint-ADMM cross-component",
        predicted_score=0.105,
        score_classification="prediction",
        current_anchor_status="engine_exists_no_component_allocation_anchor",
        missing_anchors=(
            "CodecOp to StreamProximalCodec allocation adapter",
            "real component budget ledger",
            "old/new packet SHA-256 for the ADMM-selected allocation",
            "full-sample exact CUDA auth eval",
        ),
        next_buildable_1to1_test=(
            "Run the existing Joint-ADMM engine over one real PR101/PR106 "
            "component budget with fixed proximal codecs, emit the selected "
            "allocation packet, and exact-eval that packet on CUDA."
        ),
        reactivation_criteria=(
            "ADMM convergence or honest divergence recorded",
            "allocation affects charged bytes consumed at inflate",
            "exact CUDA eval validates the selected archive",
        ),
    ),
    OmegaOptClaim(
        claim_id="omega_opt_bilevel_optimization",
        label="Bilevel optimization",
        predicted_score=0.100,
        score_classification="prediction",
        current_anchor_status="scaffold_no_convergence_or_archive_anchor",
        missing_anchors=(
            "outer training convergence proof",
            "middle meta-Lagrangian selection artifact",
            "inner Joint-ADMM allocation artifact",
            "byte-closed archive and exact CUDA auth eval",
        ),
        next_buildable_1to1_test=(
            "Run a one-phase bilevel slice on a fixed substrate with a locked "
            "atom ledger, emit one archive, and require convergence diagnostics "
            "plus exact CUDA evaluation."
        ),
        reactivation_criteria=(
            "training run has deterministic seed/config manifest",
            "atom ledger links every selected transform to bytes",
            "exact CUDA eval exists for the emitted archive",
        ),
    ),
    OmegaOptClaim(
        claim_id="omega_opt_score_feedback_meta_pass",
        label="Score-feedback meta-pass",
        predicted_score=0.095,
        score_classification="prediction",
        current_anchor_status="never_run",
        missing_anchors=(
            "score-feedback loop run log",
            "dispatch-claim custody for every exact-eval iteration",
            "archive identity chain across meta-passes",
            "terminal full-sample exact CUDA auth eval",
        ),
        next_buildable_1to1_test=(
            "Run two score-feedback meta-passes on the same archive family, "
            "record the archive identity chain, and treat only the terminal "
            "CUDA eval as score evidence."
        ),
        reactivation_criteria=(
            "each meta-pass has a dispatch claim and terminal status",
            "all archive SHA-256 transitions are recorded",
            "terminal exact CUDA eval is full-sample",
        ),
    ),
    OmegaOptClaim(
        claim_id="omega_opt_per_tensor_hstack_of_vstacks",
        label="Per-tensor HStack-of-VStacks",
        predicted_score=0.092,
        score_classification="design",
        current_anchor_status="design_only",
        missing_anchors=(
            "per-tensor stream grammar",
            "per-tensor HStack grouping manifest",
            "cross-language conformance vectors",
            "runtime packet and exact CUDA auth eval",
        ),
        next_buildable_1to1_test=(
            "Select two tensors, emit canonical input/output byte vectors for "
            "one serial transform and one parallel grouping, then build a "
            "minimal runtime packet that consumes those sections."
        ),
        reactivation_criteria=(
            "golden vectors include shapes, offsets, lengths, and SHA-256",
            "runtime consumes all per-tensor side info from archive",
            "exact CUDA eval follows packet closure",
        ),
    ),
    OmegaOptClaim(
        claim_id="omega_opt_recursive_residual_gradients",
        label="Recursive residual gradients",
        predicted_score=0.090,
        score_classification="prediction",
        current_anchor_status="aspiration_only",
        missing_anchors=(
            "implemented recursive residual-gradient transform",
            "stable convergence or fail-closed divergence criterion",
            "charged-byte residual packet",
            "full-sample exact CUDA auth eval",
        ),
        next_buildable_1to1_test=(
            "Prototype one residual-gradient correction stage with fixed "
            "iteration count and deterministic payload accounting, then "
            "compare exact CUDA auth eval against the unchanged anchor."
        ),
        reactivation_criteria=(
            "gradient transform is deterministic and scorer-free at inflate",
            "residual payload has old/new SHA-256 and charged bytes",
            "exact CUDA eval validates the changed archive",
        ),
    ),
)

CLAIMS_BY_ID: dict[str, OmegaOptClaim] = {
    claim.claim_id: claim for claim in OMEGA_OPT_CLAIMS
}

LANE_IDS_BY_CLAIM: dict[str, str] = {
    "lane_omega_opt_linear": "omega_opt_linear_stack",
    "lane_omega_opt_multipass_imp": "omega_opt_multipass_imp_cycle",
    "lane_omega_opt_joint_admm": "omega_opt_joint_admm_cross_component",
    "lane_omega_opt_score_feedback": "omega_opt_score_feedback_meta_pass",
}


def omega_opt_claim_rows() -> list[dict[str, Any]]:
    """Return canonical manifest rows for all Omega-OPT claims."""
    return [claim.to_manifest() for claim in OMEGA_OPT_CLAIMS]


def omega_opt_claim_manifest() -> dict[str, Any]:
    """Return the canonical fail-closed Omega-OPT claim table."""
    return {
        "schema": OMEGA_OPT_CLAIM_SCHEMA,
        "claim_count": len(OMEGA_OPT_CLAIMS),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "claims": omega_opt_claim_rows(),
    }


def text_mentions_omega_opt(*parts: object) -> bool:
    """Return true when free text appears to reference Omega-OPT claims."""
    text = " ".join(str(part) for part in parts if part is not None).lower()
    return any(marker in text for marker in OMEGA_OPT_TEXT_MARKERS)


def _field_present(row: Mapping[str, Any], names: Sequence[str]) -> bool:
    for name in names:
        value = row.get(name)
        if value not in (None, "", [], {}):
            return True
    return False


def has_exact_1to1_anchor(row: Mapping[str, Any]) -> bool:
    """Return true only when a row carries exact 1:1 score-anchor custody."""
    text = " ".join(
        str(row.get(key, ""))
        for key in (
            "evidence_grade",
            "evidence_marker",
            "evidence_semantics",
            "source",
            "notes",
        )
    ).lower()
    exact_grade = (
        str(row.get("evidence_grade", "")).strip().lower() in {"a", "a++"}
        or "contest-cuda" in text
        or "contest_cuda" in text
        or "exact_cuda_auth_eval" in text
    )
    return (
        exact_grade
        and _field_present(row, EXACT_ARCHIVE_SHA_FIELDS)
        and _field_present(row, EXACT_ARCHIVE_BYTES_FIELDS)
        and _field_present(row, EXACT_AUTH_EVAL_FIELDS)
        and _field_present(row, EXACT_1TO1_ANCHOR_FIELDS)
    )


def validate_omega_opt_row(row: Mapping[str, Any]) -> list[str]:
    """Validate one evidence/claim row for fail-closed Omega-OPT semantics."""
    text_parts = [row.get("technique"), row.get("claim_id"), row.get("label")]
    text_parts.extend(row.get(key) for key in ("source", "notes", "evidence_semantics"))
    if not text_mentions_omega_opt(*text_parts):
        return []

    exact_anchor = has_exact_1to1_anchor(row)
    findings: list[str] = []

    if exact_anchor:
        return findings

    for field_name in FAIL_CLOSED_FIELDS:
        if field_name not in row:
            findings.append(f"{field_name}_missing_for_unanchored_omega_opt_row")
        elif bool(row.get(field_name)) is not False:
            findings.append(f"{field_name}_must_be_false_without_exact_1to1_anchor")

    for field_name in PROMOTION_ALIAS_FIELDS:
        if field_name in row and bool(row.get(field_name)) is not False:
            findings.append(f"{field_name}_must_be_false_without_exact_1to1_anchor")

    if bool(row.get("family_falsified")) is True:
        findings.append("family_falsified_must_not_be_true_without_exact_1to1_anchor")

    return findings


def validate_omega_opt_claim_table(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Validate a complete Omega-OPT claim table."""
    by_id = {str(row.get("claim_id", "")): row for row in rows}
    findings: list[str] = []
    for claim in OMEGA_OPT_CLAIMS:
        row = by_id.get(claim.claim_id)
        if row is None:
            findings.append(f"{claim.claim_id}: missing_canonical_claim_row")
            continue
        row_findings = validate_omega_opt_row(row)
        findings.extend(f"{claim.claim_id}: {item}" for item in row_findings)
        if row.get("predicted_score") != claim.predicted_score:
            findings.append(f"{claim.claim_id}: predicted_score_drift")
        if row.get("requires_exact_1to1_anchor") is not True:
            findings.append(f"{claim.claim_id}: requires_exact_1to1_anchor_not_true")
        if str(row.get("score_classification", "")) not in {"prediction", "design", "empirical", "A++"}:
            findings.append(f"{claim.claim_id}: invalid_score_classification")
    return findings


def required_ledger_tokens_for_claim(claim: OmegaOptClaim) -> tuple[str, ...]:
    """Return tokens a durable markdown ledger must include for a claim."""
    return (
        claim.claim_id,
        "score_claim=false",
        "promotion_eligible=false",
        "rank_or_kill_eligible=false",
        "ready_for_exact_eval_dispatch=false",
        "Next 1:1 test",
    )


def validate_omega_opt_ledger_text(text: str) -> list[str]:
    """Validate the dated markdown ledger keeps every claim fail-closed."""
    findings: list[str] = []
    for claim in OMEGA_OPT_CLAIMS:
        for token in required_ledger_tokens_for_claim(claim):
            if token not in text:
                findings.append(f"{claim.claim_id}: ledger_missing_token:{token}")
    return findings
