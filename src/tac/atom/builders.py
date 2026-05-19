# SPDX-License-Identifier: MIT
"""Canonical builders for the seven AtomKind variants.

Each builder centralizes construction so the atom-creation surface refuses
malformed kwargs BEFORE persistence per Catalog #131 sister discipline.
Builders auto-attach a canonical ``tac.provenance.Provenance`` via the
existing ``tac.provenance.build_provenance_for_predicted`` helper (the
default predicted-from-model grade for not-yet-empirically-measured atoms).

Citations:
  - ``tac.provenance.build_provenance_for_*`` — Catalog #323 canonical
    provenance builder family.
  - Catalog #131 — fcntl-locked write discipline; builders ensure the
    Mapping shape is valid for the ledger writer.
  - Operator standing directive 2026-05-18: "ensure citations and
    provenance and links" — every builder requires non-empty
    ``literature_citation`` (with PREMISE_VERIFICATION as the documented
    exception per Atom.__post_init__).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from tac.provenance import (
    Provenance,
    build_provenance_for_predicted,
)

from .atom import (
    Atom,
)
from .types import AtomKind, AtomValidationError, ResolutionPath


def _provenance_to_mapping(prov: Provenance | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Provenance dataclass or already-mapping shape to dict.

    Accepts either a canonical ``tac.provenance.Provenance`` dataclass
    instance OR a pre-built Mapping (e.g. legacy JSON row). Returns a
    plain dict suitable for the frozen Atom.

    Enum members (``ProvenanceKind`` / ``ProvenanceEvidenceGrade``) are
    coerced to their ``.value`` string so the resulting dict is JSON-safe
    and downstream consumers (cathedral autopilot ranker / ledger writer)
    see canonical string tokens.
    """
    if isinstance(prov, Mapping):
        d = dict(prov)
    else:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(prov):
            d = asdict(prov)
        else:
            raise AtomValidationError(
                f"provenance must be Provenance dataclass or Mapping "
                f"(got {type(prov).__name__})"
            )
    # Coerce enum members to their .value
    from enum import Enum

    coerced: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Enum):
            coerced[k] = v.value
        else:
            coerced[k] = v
    return coerced


_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _default_predicted_provenance(
    *, atom_id: str, model_id: str | None = None
) -> dict[str, Any]:
    """Default predicted-from-model provenance for not-yet-empirically-measured atoms.

    Per ``tac.provenance.Provenance`` invariants, ``source_sha256`` must be
    a lowercase 64-char hex string. We use the canonical empty-bytes sha256
    sentinel ``e3b0c4...b855`` because the atom-construction surface has no
    input bytes to hash (the atom IS the input); callers that materialize
    a real input artifact should pass an explicit ``provenance=`` kwarg
    via the canonical ``tac.provenance.build_provenance_for_*`` builders.
    """
    return _provenance_to_mapping(
        build_provenance_for_predicted(
            model_id=model_id or f"atom_builder:{atom_id}",
            inputs_sha256=_EMPTY_SHA256,
        )
    )


def build_arbitrary_value_atom(
    *,
    atom_id: str,
    file_path: str,
    current_value: Any,
    predicted_replacement: Any,
    resolution_path: ResolutionPath,
    predicted_ev_delta_s: tuple[float, float] | list[float],
    cost_envelope_usd: float,
    literature_citation: str,
    canonical_helper_repo_link: str = "",
    cheaper_alternative_path: str = "",
    blocking_dependencies: Sequence[str] = (),
    wired_hooks: Sequence[str] = ("cathedral_autopilot_dispatch",),
    observability_surface: Sequence[str] = ("cite_able", "diff_able_across_runs"),
    provenance: Provenance | Mapping[str, Any] | None = None,
    captured_by_subagent: str = "",
    notes: str = "",
    extra_metadata: Mapping[str, Any] | None = None,
) -> Atom:
    """Build an ARBITRARY_VALUE atom subsuming a sister-audit JSONL row.

    Matches the sister arbitrariness-extinction audit canonical 21-field
    schema landed 2026-05-18 (``.omx/state/arbitrariness_extinction_audit_*\
    .jsonl``). The optional ``extra_metadata`` Mapping carries audit-specific
    fields the canonical Atom does not promote to top-level (e.g.
    ``rank_score_per_dollar``).
    """
    if not isinstance(predicted_ev_delta_s, (tuple, list)) or len(predicted_ev_delta_s) != 2:
        raise AtomValidationError(
            f"predicted_ev_delta_s must be a (lo, hi) tuple/list of length 2 "
            f"(got {predicted_ev_delta_s!r})"
        )
    lo, hi = float(predicted_ev_delta_s[0]), float(predicted_ev_delta_s[1])
    if provenance is None:
        provenance = _default_predicted_provenance(atom_id=atom_id)
    else:
        provenance = _provenance_to_mapping(provenance)
    md: dict[str, Any] = {
        "file_path": file_path,
        "current_value": current_value,
        "predicted_replacement": predicted_replacement,
        "cheaper_alternative_path": cheaper_alternative_path,
        "blocking_dependencies": list(blocking_dependencies),
        "captured_by_subagent": captured_by_subagent,
        "notes": notes,
    }
    if extra_metadata:
        md.update(extra_metadata)
    return Atom(
        atom_id=atom_id,
        kind=AtomKind.ARBITRARY_VALUE,
        resolution_path=resolution_path,
        predicted_impact_delta_s_lower=lo,
        predicted_impact_delta_s_upper=hi,
        cost_envelope_usd=float(cost_envelope_usd),
        provenance=provenance,
        wired_hooks=list(wired_hooks),
        observability_surface=list(observability_surface),
        literature_citation=literature_citation,
        canonical_helper_repo_link=canonical_helper_repo_link,
        metadata=md,
    )


def build_meta_lagrangian_atom(
    *,
    atom_id: str,
    family: str,
    family_group: str,
    byte_delta: int,
    expected_seg_dist_delta: float = 0.0,
    expected_pose_dist_delta: float = 0.0,
    expected_score_delta_lower: float = 0.0,
    expected_score_delta_upper: float = 0.0,
    cost_envelope_usd: float = 0.0,
    confidence: float = 0.5,
    source_archive_sha256: str = "",
    conflicts_with_families: Sequence[str] = (),
    conflicts_with_atoms: Sequence[str] = (),
    literature_citation: str = "tac.meta_lagrangian_allocator canonical atom shape",
    canonical_helper_repo_link: str = "src/tac/meta_lagrangian_allocator.py",
    wired_hooks: Sequence[str] = (
        "sensitivity_map",
        "pareto_constraint",
        "bit_allocator",
        "cathedral_autopilot_dispatch",
    ),
    observability_surface: Sequence[str] = (
        "decomposable_per_signal",
        "diff_able_across_runs",
        "queryable_post_hoc",
        "cite_able",
    ),
    provenance: Provenance | Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Atom:
    """Build a META_LAGRANGIAN atom matching the existing allocator atom shape.

    Subsumes the existing
    ``tac.meta_lagrangian_allocator.atoms_from_hnerv_decoder_recode_profile``
    row format. Atoms emitted here round-trip through
    ``Atom.to_meta_lagrangian_atom()`` back into the allocator's expected
    shape for backward compatibility.
    """
    if provenance is None:
        provenance = _default_predicted_provenance(atom_id=atom_id)
    else:
        provenance = _provenance_to_mapping(provenance)
    md: dict[str, Any] = {
        "byte_delta": int(byte_delta),
        "expected_seg_dist_delta": float(expected_seg_dist_delta),
        "expected_pose_dist_delta": float(expected_pose_dist_delta),
        "confidence": float(confidence),
        "family": family,
        "family_group": family_group,
        "conflicts_with_families": list(conflicts_with_families),
        "conflicts_with_atoms": list(conflicts_with_atoms),
    }
    if source_archive_sha256:
        # Push into provenance.source_sha256 if the provenance does not
        # already carry a non-sentinel value. The default predicted
        # provenance uses the empty-bytes sha256 sentinel ``_EMPTY_SHA256``;
        # treating that as "absent" lets callers override with a real
        # archive sha256 while still respecting an explicitly-passed
        # provenance dict.
        current_sha = provenance.get("source_sha256", "")
        if not current_sha or current_sha == _EMPTY_SHA256:
            provenance = {**provenance, "source_sha256": source_archive_sha256}
    if extra_metadata:
        md.update(extra_metadata)
    return Atom(
        atom_id=atom_id,
        kind=AtomKind.META_LAGRANGIAN,
        # Rate-only allocation atoms (the existing canonical case) are
        # ANALYTICAL_SOLVE — the rate term is closed-form per Shannon R(D).
        # Callers building from FORMULA / EXPERIMENTAL paths should override.
        resolution_path=ResolutionPath.ANALYTICAL_SOLVE,
        predicted_impact_delta_s_lower=float(expected_score_delta_lower),
        predicted_impact_delta_s_upper=float(expected_score_delta_upper),
        cost_envelope_usd=float(cost_envelope_usd),
        provenance=provenance,
        wired_hooks=list(wired_hooks),
        observability_surface=list(observability_surface),
        literature_citation=literature_citation,
        canonical_helper_repo_link=canonical_helper_repo_link,
        metadata=md,
    )


def build_cargo_cult_atom(
    *,
    atom_id: str,
    substrate_id: str,
    assumption: str,
    classification: str,
    rationale: str,
    unwind_test_plan: str = "",
    predicted_impact_lower: float = 0.0,
    predicted_impact_upper: float = 0.0,
    cost_envelope_usd: float = 0.0,
    literature_citation: str = (
        "Catalog #303 cargo-cult audit section + hard-earned-vs-cargo-culted "
        "addendum 2026-05-15"
    ),
    canonical_helper_repo_link: str = ".omx/research/*_design_*.md (per-substrate)",
    wired_hooks: Sequence[str] = (
        "continual_learning_posterior",
        "probe_disambiguator",
    ),
    observability_surface: Sequence[str] = ("cite_able", "queryable_post_hoc"),
    provenance: Provenance | Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Atom:
    """Build a CARGO_CULT_ASSUMPTION atom per Catalog #303.

    The Assumption-Adversary council seat (CLAUDE.md "Council conduct"
    sextet-pact amendment 2026-05-15) is the canonical producer; this
    builder is the canonical receiver at the META layer.
    """
    if provenance is None:
        provenance = _default_predicted_provenance(atom_id=atom_id)
    else:
        provenance = _provenance_to_mapping(provenance)
    md: dict[str, Any] = {
        "substrate_id": substrate_id,
        "assumption": assumption,
        "classification": classification,
        "rationale": rationale,
        "unwind_test_plan": unwind_test_plan,
    }
    if extra_metadata:
        md.update(extra_metadata)
    return Atom(
        atom_id=atom_id,
        kind=AtomKind.CARGO_CULT_ASSUMPTION,
        # Cargo-cult classification is itself an EXPERIMENTAL question
        # (HARD-EARNED vs CARGO-CULTED is decided by paired-comparison
        # smoke per the addendum).
        resolution_path=ResolutionPath.EXPERIMENTAL,
        predicted_impact_delta_s_lower=float(predicted_impact_lower),
        predicted_impact_delta_s_upper=float(predicted_impact_upper),
        cost_envelope_usd=float(cost_envelope_usd),
        provenance=provenance,
        wired_hooks=list(wired_hooks),
        observability_surface=list(observability_surface),
        literature_citation=literature_citation,
        canonical_helper_repo_link=canonical_helper_repo_link,
        metadata=md,
    )


def build_premise_verification_atom(
    *,
    atom_id: str,
    premise: str,
    verified: bool,
    verification_method: str,
    callsite_path: str = "",
    callsite_line: int | None = None,
    literature_citation: str = "",
    canonical_helper_repo_link: str = "src/tac/preflight.py::Catalog #229",
    wired_hooks: Sequence[str] = ("probe_disambiguator",),
    observability_surface: Sequence[str] = ("cite_able", "counterfactual_able"),
    provenance: Provenance | Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Atom:
    """Build a PREMISE_VERIFICATION atom per Catalog #229.

    Empty ``literature_citation`` is canonically allowed for this kind
    (premise verification is itself the verification primitive; no
    external citation required). Other kinds REQUIRE non-empty citation
    per Atom.__post_init__.
    """
    if provenance is None:
        provenance = _default_predicted_provenance(atom_id=atom_id)
    else:
        provenance = _provenance_to_mapping(provenance)
    md: dict[str, Any] = {
        "premise": premise,
        "verified": bool(verified),
        "verification_method": verification_method,
        "callsite_path": callsite_path,
        "callsite_line": callsite_line,
    }
    if extra_metadata:
        md.update(extra_metadata)
    return Atom(
        atom_id=atom_id,
        kind=AtomKind.PREMISE_VERIFICATION,
        # Premise verification is itself an experimental check
        # (run importlib.import_module() / grep target / etc.).
        resolution_path=ResolutionPath.EXPERIMENTAL,
        # Premise verification is a binary verifier — no predicted impact.
        predicted_impact_delta_s_lower=0.0,
        predicted_impact_delta_s_upper=0.0,
        cost_envelope_usd=0.0,
        provenance=provenance,
        wired_hooks=list(wired_hooks),
        observability_surface=list(observability_surface),
        literature_citation=literature_citation,
        canonical_helper_repo_link=canonical_helper_repo_link,
        metadata=md,
    )


def build_probe_outcome_atom(
    *,
    atom_id: str,
    probe_id: str,
    substrate: str,
    verdict: str,
    metric_name: str = "",
    metric_value: float | None = None,
    threshold: float | None = None,
    threshold_token: str = "",
    evidence_path: str = "",
    next_action: str = "",
    blocker_status: str = "blocking",
    adjudicated_at_utc: str = "",
    expires_at_utc: str = "",
    literature_citation: str = "Catalog #313 probe-outcomes-ledger canonical schema",
    canonical_helper_repo_link: str = "src/tac/probe_outcomes_ledger.py",
    wired_hooks: Sequence[str] = (
        "cathedral_autopilot_dispatch",
        "probe_disambiguator",
        "continual_learning_posterior",
    ),
    observability_surface: Sequence[str] = (
        "diff_able_across_runs",
        "queryable_post_hoc",
        "cite_able",
    ),
    provenance: Provenance | Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Atom:
    """Build a PROBE_OUTCOME atom mirroring ``tac.probe_outcomes_ledger.ProbeOutcomeView``.

    The 7-verdict taxonomy (INDEPENDENT / KILL / DEFER / PROMOTE / PROCEED /
    PARTIAL / OPERATOR_REVIEW_REQUIRED) is enforced by ``Atom.__post_init__``
    per-kind validation.
    """
    if provenance is None:
        provenance = _default_predicted_provenance(atom_id=atom_id)
    else:
        provenance = _provenance_to_mapping(provenance)
    md: dict[str, Any] = {
        "probe_id": probe_id,
        "substrate": substrate,
        "verdict": verdict,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold": threshold,
        "threshold_token": threshold_token,
        "evidence_path": evidence_path,
        "next_action": next_action,
        "blocker_status": blocker_status,
        "adjudicated_at_utc": adjudicated_at_utc,
        "expires_at_utc": expires_at_utc,
    }
    if extra_metadata:
        md.update(extra_metadata)
    return Atom(
        atom_id=atom_id,
        kind=AtomKind.PROBE_OUTCOME,
        resolution_path=ResolutionPath.EXPERIMENTAL,
        predicted_impact_delta_s_lower=0.0,
        predicted_impact_delta_s_upper=0.0,
        cost_envelope_usd=0.0,
        provenance=provenance,
        wired_hooks=list(wired_hooks),
        observability_surface=list(observability_surface),
        literature_citation=literature_citation,
        canonical_helper_repo_link=canonical_helper_repo_link,
        metadata=md,
    )


def build_council_deliberation_atom(
    *,
    atom_id: str,
    deliberation_id: str,
    topic: str,
    council_tier: str,
    council_verdict: str,
    council_attendees: Sequence[str] = (),
    council_quorum_met: bool = True,
    predicted_impact_lower: float = 0.0,
    predicted_impact_upper: float = 0.0,
    cost_envelope_usd: float = 0.0,
    memory_path: str = "",
    literature_citation: str = "Catalog #300 council-deliberation v2 frontmatter contract",
    canonical_helper_repo_link: str = "src/tac/council_continual_learning.py",
    wired_hooks: Sequence[str] = (
        "cathedral_autopilot_dispatch",
        "continual_learning_posterior",
    ),
    observability_surface: Sequence[str] = (
        "queryable_post_hoc",
        "cite_able",
        "diff_able_across_runs",
    ),
    provenance: Provenance | Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Atom:
    """Build a COUNCIL_DELIBERATION atom mirroring ``CouncilDeliberationRecord``.

    The 4-tier taxonomy (T1/T2/T3/T4) is enforced by ``Atom.__post_init__``
    per-kind validation; sister Catalog #300 STRICT preflight gate enforces
    the same contract at the design-memo frontmatter surface.
    """
    if provenance is None:
        provenance = _default_predicted_provenance(atom_id=atom_id)
    else:
        provenance = _provenance_to_mapping(provenance)
    md: dict[str, Any] = {
        "deliberation_id": deliberation_id,
        "topic": topic,
        "council_tier": council_tier,
        "council_verdict": council_verdict,
        "council_attendees": list(council_attendees),
        "council_quorum_met": bool(council_quorum_met),
        "memory_path": memory_path,
    }
    if extra_metadata:
        md.update(extra_metadata)
    return Atom(
        atom_id=atom_id,
        kind=AtomKind.COUNCIL_DELIBERATION,
        # Council verdicts are analytical solves over the design-decision
        # tradeoff space (per the inner-sextet-pact decision protocol).
        resolution_path=ResolutionPath.ANALYTICAL_SOLVE,
        predicted_impact_delta_s_lower=float(predicted_impact_lower),
        predicted_impact_delta_s_upper=float(predicted_impact_upper),
        cost_envelope_usd=float(cost_envelope_usd),
        provenance=provenance,
        wired_hooks=list(wired_hooks),
        observability_surface=list(observability_surface),
        literature_citation=literature_citation,
        canonical_helper_repo_link=canonical_helper_repo_link,
        metadata=md,
    )


def build_dispatch_claim_atom(
    *,
    atom_id: str,
    lane_id: str,
    provider: str,
    gpu: str,
    instance_or_job_id: str,
    cost_envelope_usd: float,
    predicted_impact_lower: float = 0.0,
    predicted_impact_upper: float = 0.0,
    status: str = "active",
    opened_at_utc: str = "",
    literature_citation: str = (
        "CLAUDE.md 'CROSS-AGENT DISPATCH COORDINATION' non-negotiable + "
        "tools/claim_lane_dispatch.py canonical 24h-TTL helper"
    ),
    canonical_helper_repo_link: str = "tools/claim_lane_dispatch.py",
    wired_hooks: Sequence[str] = (
        "cathedral_autopilot_dispatch",
        "continual_learning_posterior",
    ),
    observability_surface: Sequence[str] = (
        "diff_able_across_runs",
        "queryable_post_hoc",
        "cite_able",
    ),
    provenance: Provenance | Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Atom:
    """Build a DISPATCH_CLAIM atom mirroring ``.omx/state/active_lane_dispatch_claims.md`` rows."""
    if provenance is None:
        provenance = _default_predicted_provenance(atom_id=atom_id)
    else:
        provenance = _provenance_to_mapping(provenance)
    md: dict[str, Any] = {
        "lane_id": lane_id,
        "provider": provider,
        "gpu": gpu,
        "instance_or_job_id": instance_or_job_id,
        "status": status,
        "opened_at_utc": opened_at_utc or datetime.now(UTC).isoformat(),
    }
    if extra_metadata:
        md.update(extra_metadata)
    return Atom(
        atom_id=atom_id,
        kind=AtomKind.DISPATCH_CLAIM,
        # Paid dispatch is an EXPERIMENTAL resolution path (the dispatch
        # itself measures the lane's predicted impact).
        resolution_path=ResolutionPath.EXPERIMENTAL,
        predicted_impact_delta_s_lower=float(predicted_impact_lower),
        predicted_impact_delta_s_upper=float(predicted_impact_upper),
        cost_envelope_usd=float(cost_envelope_usd),
        provenance=provenance,
        wired_hooks=list(wired_hooks),
        observability_surface=list(observability_surface),
        literature_citation=literature_citation,
        canonical_helper_repo_link=canonical_helper_repo_link,
        metadata=md,
    )


__all__ = [
    "build_arbitrary_value_atom",
    "build_cargo_cult_atom",
    "build_council_deliberation_atom",
    "build_dispatch_claim_atom",
    "build_meta_lagrangian_atom",
    "build_premise_verification_atom",
    "build_probe_outcome_atom",
]
