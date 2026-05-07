"""Cross-paradigm adapters for meta-Lagrangian atom planning.

The adapters in this module intentionally stop at planning rows. They do not
build archives, dispatch GPUs, or claim scores. Their only job is to normalize
heterogeneous paradigm outputs into the atom mapping accepted by
``tac.optimization.meta_lagrangian_allocator.build_atom_ledger`` while keeping
source-specific interaction assumptions and blockers auditable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.research_basis import research_basis_ids_for_family

SCHEMA_VERSION = 1

DEFAULT_DISPATCH_BLOCKERS = (
    "cross_paradigm_atom_is_planning_only",
    "requires_stack_interaction_review",
    "requires_byte_closed_archive_manifest_before_dispatch",
    "requires_exact_cuda_auth_eval",
)
ARCHIVE_CUSTODY_SATISFIED_BLOCKERS = {
    "requires_archive_manifest_preflight",
    "requires_byte_closed_archive_manifest_before_dispatch",
}

COMMON_ATOM_FIELDS = (
    "atom_id",
    "family",
    "pareto_scope",
    "byte_delta",
    "expected_seg_dist_delta",
    "expected_pose_dist_delta",
    "confidence",
    "interaction_assumptions",
    "archive_manifest_path",
    "archive_manifest_sha256",
    "research_basis_ids",
    "evidence_grade",
    "dispatch_blockers",
)


class CrossParadigmAtomError(ValueError):
    """Raised when a cross-paradigm adapter cannot normalize its input."""


def build_cross_paradigm_atom_ledger(
    atoms: Iterable[Mapping[str, Any]],
    *,
    base_pose_dist: float,
    source: str,
) -> dict[str, Any]:
    """Build a meta-Lagrangian ledger and preserve adapter-level fields.

    Importing the allocator lazily keeps this module a one-way adapter layer:
    the allocator never imports paradigm code, while callers can still feed its
    canonical ``build_atom_ledger`` implementation.
    """

    normalized = [normalize_cross_paradigm_atom(atom) for atom in atoms]
    _reject_duplicate_atom_ids(normalized)

    from tac.optimization.meta_lagrangian_allocator import (
        _annotate_row_explanations,
        build_atom_ledger,
    )

    ledger = build_atom_ledger(
        normalized,
        base_pose_dist=base_pose_dist,
        source=source,
    )
    by_id = {str(atom["atom_id"]): atom for atom in normalized}
    source_blocker_counts: Counter[str] = Counter()
    paradigm_counts: Counter[str] = Counter()
    adapter_counts: Counter[str] = Counter()
    for row in ledger["rows"]:
        original = by_id[str(row["atom_id"])]
        paradigm = str(original["paradigm"])
        adapter = str(original["adapter"])
        source_blockers = list(original["dispatch_blockers"])
        if row["byte_closed_archive_manifest_attached"]:
            source_blockers = [
                blocker
                for blocker in source_blockers
                if blocker not in ARCHIVE_CUSTODY_SATISFIED_BLOCKERS
            ]
        source_blocker_counts.update(source_blockers)
        paradigm_counts[paradigm] += 1
        adapter_counts[adapter] += 1
        row["cross_paradigm_schema_version"] = SCHEMA_VERSION
        row["adapter"] = adapter
        row["paradigm"] = paradigm
        row["interaction_assumptions"] = list(original["interaction_assumptions"])
        row["adapter_dispatch_blockers"] = source_blockers
        row["source_dispatch_blockers"] = source_blockers
        row["dispatch_blockers"] = _unique_strings(
            [*row["dispatch_blockers"], *source_blockers]
        )
        _annotate_row_explanations([row])
    ledger["allocator_tool"] = ledger["tool"]
    ledger["tool"] = "tac.optimization.cross_paradigm_atoms.build_cross_paradigm_atom_ledger"
    ledger["cross_paradigm_schema_version"] = SCHEMA_VERSION
    ledger["common_atom_fields"] = list(COMMON_ATOM_FIELDS)
    ledger["paradigm_counts"] = dict(sorted(paradigm_counts.items()))
    ledger["adapter_counts"] = dict(sorted(adapter_counts.items()))
    ledger["adapter_dispatch_blocker_counts"] = dict(sorted(source_blocker_counts.items()))
    ledger["dispatch_attempted"] = False
    ledger["ready_for_exact_eval_dispatch"] = False
    ledger["dispatch_blockers"] = _unique_strings(
        [
            *ledger["dispatch_blockers"],
            "cross_paradigm_adapter_rows_require_source_review",
        ]
    )
    return ledger


def normalize_cross_paradigm_atom(atom: Mapping[str, Any]) -> dict[str, Any]:
    """Return one canonical adapter atom mapping for allocator ingestion."""

    atom_id = _required_str(atom, "atom_id")
    byte_delta = _int_value(_first_present(atom, "byte_delta", "estimated_charged_bytes"), "byte_delta")
    confidence = _confidence(atom.get("confidence", 1.0), atom_id=atom_id)
    family = _required_str(atom, "family")
    family_group = str(atom.get("family_group") or family)
    pareto_scope = str(atom.get("pareto_scope") or family_group)
    dispatch_blockers = _unique_strings(
        [
            *DEFAULT_DISPATCH_BLOCKERS,
            *_string_list(atom.get("dispatch_blockers")),
            *_string_list(atom.get("source_dispatch_blockers")),
            *_source_score_claim_blocker(atom),
        ]
    )
    normalized = {
        "cross_paradigm_schema_version": SCHEMA_VERSION,
        "adapter": str(atom.get("adapter") or "manual_cross_paradigm_atom"),
        "paradigm": str(atom.get("paradigm") or family_group),
        "atom_id": atom_id,
        "family": family,
        "family_group": family_group,
        "pareto_scope": pareto_scope,
        "conflicts_with_families": _string_list(atom.get("conflicts_with_families")),
        "conflicts_with_atoms": _string_list(atom.get("conflicts_with_atoms")),
        "byte_delta": byte_delta,
        "expected_seg_dist_delta": _float_value(
            atom.get("expected_seg_dist_delta", 0.0),
            "expected_seg_dist_delta",
        ),
        "expected_pose_dist_delta": _float_value(
            atom.get("expected_pose_dist_delta", 0.0),
            "expected_pose_dist_delta",
        ),
        "confidence": confidence,
        "interaction_assumptions": _string_list(atom.get("interaction_assumptions")),
        "archive_manifest_path": str(atom.get("archive_manifest_path") or ""),
        "archive_manifest_sha256": str(atom.get("archive_manifest_sha256") or ""),
        "evidence_grade": str(atom.get("evidence_grade") or "prediction"),
        "dispatch_blockers": dispatch_blockers,
        "source_dispatch_blockers": dispatch_blockers,
        "score_claim": False,
        "source_score_claim": bool(atom.get("score_claim") is True),
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "pair_support": _int_list(atom.get("pair_support")),
        "hard_pair_support": _int_list(atom.get("hard_pair_support")),
        "class_support": _int_list(atom.get("class_support")),
        "geometry_priors": _string_list(atom.get("geometry_priors")),
        "openpilot_priors": _string_list(atom.get("openpilot_priors")),
        "research_basis_ids": _unique_strings(
            [
                *_string_list(atom.get("research_basis_ids")),
                *research_basis_ids_for_family(
                    family,
                    family_group,
                    pareto_scope,
                    str(atom.get("paradigm") or ""),
                ),
            ]
        ),
        "allocation_inference": bool(atom.get("allocation_inference", False)),
        "evidence_source_path": str(atom.get("evidence_source_path") or ""),
        "evidence_source_sha256": str(atom.get("evidence_source_sha256") or ""),
        "source_archive_sha256": str(atom.get("source_archive_sha256") or ""),
    }
    for passthrough in (
        "raw_equal",
        "rankable",
        "hard_pair_rank",
        "hard_pair_score",
        "latent_norm",
        "source_section_sha256",
        "candidate_section_sha256",
        "candidate_archive_sha256",
        "candidate_archive_path",
    ):
        if passthrough in atom:
            normalized[passthrough] = atom[passthrough]
    return normalized


def atoms_from_hnerv_rate_recode_profile(
    profile: Mapping[str, Any],
    *,
    evidence_source_path: str = "",
    evidence_source_sha256: str = "",
) -> list[dict[str, Any]]:
    """Adapt an HNeRV rate-recode profile into common atom rows."""

    label = str(profile.get("source_label") or profile.get("label") or "hnerv")
    source_archive_sha256 = str(profile.get("source_archive_sha256") or "")
    variants = _mapping_rows(profile.get("variants") or profile.get("rate_recode_variants") or [])
    atoms: list[dict[str, Any]] = []
    for row in sorted(variants, key=lambda item: (_slug(str(item.get("variant") or "variant")), _stable_digest(item))):
        variant = str(row.get("variant") or row.get("name") or "variant")
        raw_equal = bool(row.get("raw_equal") is True or row.get("byte_equivalent") is True)
        evidence_grade = str(
            row.get("evidence_grade")
            or ("empirical_byte_raw_equal" if raw_equal else "invalid")
        )
        atom = {
            "adapter": "hnerv_rate_recode_profile",
            "paradigm": "hnerv_rate_recode",
            "atom_id": f"hnerv_rate_recode:{_slug(label)}:{_slug(variant)}",
            "family": "hnerv_rate_recode",
            "family_group": "hnerv_rate_equivalent_recode",
            "pareto_scope": "hnerv_rate_equivalent_recode",
            "byte_delta": _first_present(
                row,
                "byte_delta",
                "byte_delta_vs_source_section",
                "archive_byte_delta",
                default=0,
            ),
            "expected_seg_dist_delta": 0.0,
            "expected_pose_dist_delta": 0.0,
            "confidence": row.get("confidence", 1.0 if raw_equal else 0.0),
            "interaction_assumptions": [
                "rate_only_raw_equal_required",
                "stack_before_scorer_changing_atoms",
                "conflicts_with_decoder_replacement_for_same_section",
            ],
            "archive_manifest_path": _first_str(
                row,
                profile,
                keys=("archive_manifest_path", "candidate_archive_manifest_path"),
            ),
            "archive_manifest_sha256": _first_str(
                row,
                profile,
                keys=("archive_manifest_sha256", "candidate_archive_manifest_sha256"),
            ),
            "evidence_grade": evidence_grade,
            "dispatch_blockers": _unique_strings(
                [
                    "hnerv_rate_recode_requires_raw_equal_proof",
                    *_string_list(profile.get("dispatch_blockers")),
                    *_string_list(row.get("dispatch_blockers")),
                ]
            ),
            "raw_equal": raw_equal,
            "source_archive_sha256": source_archive_sha256,
            "evidence_source_path": evidence_source_path,
            "evidence_source_sha256": evidence_source_sha256,
        }
        atoms.append(normalize_cross_paradigm_atom(atom))
    return atoms


def atoms_from_wr01_wavelet_plan(
    plan: Mapping[str, Any],
    *,
    evidence_source_path: str = "",
    evidence_source_sha256: str = "",
) -> list[dict[str, Any]]:
    """Adapt WR01 wavelet planning or apply-transform manifests."""

    if "candidate_archive_byte_delta_vs_source_estimate" in plan or "section_byte_delta" in plan:
        return [
            normalize_cross_paradigm_atom(
                _wr01_apply_manifest_atom(
                    plan,
                    evidence_source_path=evidence_source_path,
                    evidence_source_sha256=evidence_source_sha256,
                )
            )
        ]
    # 3rd schema branch (claude:main 2026-05-06): WR01 exact-eval-packets
    # produced by tools/build_wr01_exact_eval_packet.py document an already-
    # built candidate archive (with archive_bytes + archive_sha256 +
    # source_archive_sha256 + changed_section_name). They are NOT planning
    # manifests but ARE eligible atom inputs because the candidate archive
    # is already byte-closed. Pre-fix the cross-paradigm ledger silently
    # produced 0 atoms when given an exact-eval packet — the adapter rejected
    # the file but didn't surface why.
    if (
        "archive_bytes" in plan
        and "archive_sha256" in plan
        and "source_archive_sha256" in plan
        and "changed_section_name" in plan
    ):
        archive_bytes = int(plan.get("archive_bytes") or 0)
        source_bytes = int(plan.get("source_archive_bytes") or 0)
        byte_delta = archive_bytes - source_bytes if source_bytes else 0
        section_name = str(plan.get("changed_section_name") or "section")
        source_label = str(plan.get("lane_id") or plan.get("job_name") or "wr01_exact_eval")
        atom_id = f"wr01_wavelet:{_slug(source_label)}:{_slug(section_name)}:exact_eval_packet"
        archive_manifest_path = _first_str(
            plan,
            keys=("archive_manifest_path", "candidate_archive_manifest_path"),
        ) or _artifact_path_for_suffix(plan.get("artifacts"), "/manifest.json")
        archive_manifest_sha256 = _first_str(
            plan,
            keys=("archive_manifest_sha256", "candidate_archive_manifest_sha256"),
        ) or _sha256_file_if_exists(archive_manifest_path)
        static_packet_ready = (
            plan.get("byte_custody_exact_eval_candidate_ready") is True
            or plan.get("static_packet_ready") is True
            or plan.get("candidate_static_preflight_ready") is True
        )
        ready_for_submit = plan.get("ready_for_submit") is True
        exact_eval_packet_blockers = []
        if not static_packet_ready:
            exact_eval_packet_blockers.append("exact_eval_packet_static_custody_not_ready")
        if static_packet_ready and not ready_for_submit:
            exact_eval_packet_blockers.append("exact_eval_packet_operator_gates_pending")
        return [
            normalize_cross_paradigm_atom(
                {
                    "adapter": "wr01_wavelet_plan",
                    "paradigm": "wr01_wavelet",
                    "atom_id": atom_id,
                    "family": "wr01_wavelet_residual",
                    "family_group": "wr01_wavelet",
                    "pareto_scope": f"wr01_wavelet:{_slug(section_name)}",
                    "byte_delta": byte_delta,
                    # Exact-eval packets carry [predicted] deltas — the
                    # actual contest-CUDA score is operator-gated.
                    "expected_seg_dist_delta": 0.0,
                    "expected_pose_dist_delta": 0.0,
                    "confidence": 0.5 if ready_for_submit else (0.4 if static_packet_ready else 0.25),
                    "interaction_assumptions": [
                        "scorer_changing_wavelet_residual",
                        "stack_after_raw_equal_rate_recodes",
                        "exact_eval_packet_documents_byte_closed_archive",
                    ],
                    "archive_manifest_path": archive_manifest_path,
                    "archive_manifest_sha256": archive_manifest_sha256,
                    "source_archive_sha256": str(plan.get("source_archive_sha256") or ""),
                    "evidence_grade": "exact_eval_packet_byte_closed",
                    "evidence_source_path": evidence_source_path,
                    "evidence_source_sha256": evidence_source_sha256,
                    "dispatch_blockers": [
                        "requires_stack_interaction_review",
                        "requires_exact_cuda_auth_eval",
                        *exact_eval_packet_blockers,
                        *(plan.get("blockers") or []),
                    ],
                }
            )
        ]
    source_label = str(plan.get("source_label") or "wr01")
    source_archive_sha256 = str(plan.get("source_archive_sha256") or "")
    atoms: list[dict[str, Any]] = []
    sections = _mapping_rows(plan.get("sections") or [])
    for section in sorted(sections, key=lambda item: str(item.get("section_name") or "")):
        section_name = str(section.get("section_name") or "section")
        rows = _mapping_rows(section.get("atoms") or [])
        if not rows:
            rows = [section]
        for row in sorted(rows, key=lambda item: _wr01_atom_sort_key(item)):
            atom_id = _wr01_atom_id(source_label, section_name, row)
            byte_delta = _first_present(
                row,
                "byte_delta",
                "estimated_wire_bytes",
                default=section.get("estimated_atom_bytes", 0),
            )
            atoms.append(
                normalize_cross_paradigm_atom(
                    {
                        "adapter": "wr01_wavelet_plan",
                        "paradigm": "wr01_wavelet",
                        "atom_id": atom_id,
                        "family": "wr01_wavelet_residual",
                        "family_group": "wr01_wavelet",
                        "pareto_scope": f"wr01_wavelet:{_slug(section_name)}",
                        "byte_delta": byte_delta,
                        "expected_seg_dist_delta": _first_present(
                            row,
                            "expected_seg_dist_delta",
                            default=section.get("expected_seg_dist_delta", plan.get("expected_seg_dist_delta", 0.0)),
                        ),
                        "expected_pose_dist_delta": _first_present(
                            row,
                            "expected_pose_dist_delta",
                            default=section.get("expected_pose_dist_delta", plan.get("expected_pose_dist_delta", 0.0)),
                        ),
                        "confidence": row.get("confidence", section.get("confidence", plan.get("confidence", 0.25))),
                        "interaction_assumptions": [
                            "scorer_changing_wavelet_residual",
                            "stack_after_raw_equal_rate_recodes",
                            "requires_runtime_consumes_wavelet_atoms",
                        ],
                        "archive_manifest_path": _first_str(
                            row,
                            section,
                            plan,
                            keys=("archive_manifest_path", "candidate_archive_manifest_path"),
                        ),
                        "archive_manifest_sha256": _first_str(
                            row,
                            section,
                            plan,
                            keys=("archive_manifest_sha256", "candidate_archive_manifest_sha256"),
                        ),
                        "evidence_grade": str(row.get("evidence_grade") or section.get("evidence_grade") or plan.get("evidence_grade") or "prediction"),
                        "dispatch_blockers": _unique_strings(
                            [
                                "wr01_wavelet_requires_component_benefit_evidence",
                                "wr01_wavelet_requires_runtime_consumption_proof",
                                *_string_list(plan.get("dispatch_blockers")),
                                *_string_list(section.get("dispatch_blockers")),
                                *_string_list(row.get("dispatch_blockers")),
                            ]
                        ),
                        "source_archive_sha256": source_archive_sha256,
                        "source_section_sha256": section.get("source_section_sha256", ""),
                        "evidence_source_path": evidence_source_path,
                        "evidence_source_sha256": evidence_source_sha256,
                    }
                )
            )
    return _sorted_atoms(atoms)


def atoms_from_categorical_openpilot_mask_plan(
    plan: Mapping[str, Any],
    *,
    evidence_source_path: str = "",
    evidence_source_sha256: str = "",
) -> list[dict[str, Any]]:
    """Adapt categorical/openpilot mask planning manifests into common atoms."""

    plan_grade = str(plan.get("evidence_grade") or "prediction")
    source_archive_sha256 = _first_str(plan, keys=("source_archive_sha256", "input_archive_sha256"))
    archive_manifest_path = _first_str(
        plan,
        keys=("archive_manifest_path", "candidate_archive_manifest_path"),
    )
    archive_manifest_sha256 = _first_str(
        plan,
        keys=("archive_manifest_sha256", "candidate_archive_manifest_sha256"),
    )
    if (
        not archive_manifest_path
        and not archive_manifest_sha256
        and _categorical_byte_closed_archive_parity_proven(plan)
    ):
        archive_manifest_path = evidence_source_path
        archive_manifest_sha256 = evidence_source_sha256
    atoms: list[dict[str, Any]] = []
    for source_name, row in _categorical_rows(plan):
        family = str(row.get("family") or row.get("atom_family") or source_name or "mask_atom")
        class_id = _first_present(row, "class_id", "comma10k_id", default=None)
        row_id = str(
            row.get("atom_id")
            or row.get("policy_name")
            or (
                f"class_{class_id}_{row.get('name')}"
                if class_id is not None and row.get("name")
                else ""
            )
            or _stable_digest(row)[:12]
        )
        score_saved_proxy = _float_value(
            _first_present(
                row,
                "estimated_marginal_score_saved_proxy",
                default=_nested(row, ("lagrangian", "estimated_marginal_score_saved_proxy"), 0.0),
            ),
            "estimated_marginal_score_saved_proxy",
        )
        atom = {
            "adapter": "categorical_openpilot_mask_plan",
            "paradigm": "categorical_openpilot_mask",
            "atom_id": f"categorical_openpilot_mask:{_slug(source_name)}:{_slug(row_id)}",
            "family": f"categorical_openpilot_mask_{_slug(family)}",
            "family_group": "categorical_openpilot_mask_plan",
            "pareto_scope": f"categorical_openpilot_mask:{_slug(family)}",
            "conflicts_with_families": _unique_strings(
                [
                    "whole_mask_replacement",
                    *_string_list(row.get("conflicts_with_families")),
                ]
            ),
            "byte_delta": _first_present(
                row,
                "byte_delta",
                "estimated_charged_bytes",
                "budget_bytes",
                default=_nested(row, ("cost_model", "estimated_charged_bytes"), 0),
            ),
            "expected_seg_dist_delta": _first_present(
                row,
                "expected_seg_dist_delta",
                default=(-score_saved_proxy / 100.0 if score_saved_proxy else 0.0),
            ),
            "expected_pose_dist_delta": _first_present(row, "expected_pose_dist_delta", default=0.0),
            "confidence": row.get("confidence", plan.get("confidence", 0.4)),
            "interaction_assumptions": _unique_strings(
                [
                    "mask_stream_replacement_or_residual",
                    "openpilot_priors_must_be_charged_if_score_affecting",
                    "requires_class_geometry_noop_controls",
                    *(
                        ["categorical_byte_closed_archive_parity_proven"]
                        if _categorical_byte_closed_archive_parity_proven(plan)
                        else []
                    ),
                    *_string_list(row.get("interaction_assumptions")),
                ]
            ),
            "archive_manifest_path": _first_str(
                row,
                plan,
                keys=("archive_manifest_path", "candidate_archive_manifest_path"),
            )
            or archive_manifest_path,
            "archive_manifest_sha256": _first_str(
                row,
                plan,
                keys=("archive_manifest_sha256", "candidate_archive_manifest_sha256"),
            )
            or archive_manifest_sha256,
            "evidence_grade": str(row.get("evidence_grade") or plan_grade),
            "dispatch_blockers": _unique_strings(
                [
                    "categorical_mask_requires_charged_runtime_consumer",
                    "categorical_mask_requires_decode_reencode_parity",
                    *_string_list(plan.get("dispatch_blockers")),
                    *_string_list(row.get("dispatch_blockers")),
                ]
            ),
            "pair_support": _int_list(row.get("pair_indices") or row.get("pair_support")),
            "hard_pair_support": _int_list(row.get("hard_pair_support")),
            "class_support": _int_list(row.get("class_ids") or row.get("class_support") or class_id),
            "geometry_priors": _string_list(row.get("geometry_priors")),
            "openpilot_priors": _unique_strings(
                [
                    *_string_list(plan.get("openpilot_priors")),
                    *_string_list(row.get("openpilot_priors")),
                    *_string_list(row.get("openpilot_prior_hint")),
                    *(["openpilot_prior_required"] if _plan_mentions_openpilot(plan, row) else []),
                ]
            ),
            "source_archive_sha256": source_archive_sha256,
            "evidence_source_path": evidence_source_path,
            "evidence_source_sha256": evidence_source_sha256,
        }
        atoms.append(normalize_cross_paradigm_atom(atom))
    return _dedupe_and_sort_atoms(atoms)


def _categorical_byte_closed_archive_parity_proven(plan: Mapping[str, Any]) -> bool:
    parity = plan.get("byte_closed_archive_parity")
    if not isinstance(parity, Mapping):
        return False
    candidate_archive = parity.get("candidate_archive")
    archive = candidate_archive if isinstance(candidate_archive, Mapping) else {}
    archive_sha256 = str(archive.get("sha256") or "")
    archive_bytes = archive.get("bytes")
    return bool(
        parity.get("proven") is True
        and parity.get("score_claim") is False
        and parity.get("dispatch_attempted") is False
        and not _string_list(parity.get("blockers"))
        and isinstance(archive_bytes, int)
        and not isinstance(archive_bytes, bool)
        and archive_bytes > 0
        and _looks_like_sha256(archive_sha256)
    )


def atoms_from_lapose_plan(
    payload: Mapping[str, Any],
    *,
    evidence_source_path: str = "",
    evidence_source_sha256: str = "",
) -> list[dict[str, Any]]:
    """Adapt LA-pose planning manifests or motion records into common atoms."""

    records = _lapose_rows(payload)
    source_archive_sha256 = _first_str(payload, keys=("source_archive_sha256",))
    atoms: list[dict[str, Any]] = []
    for row in sorted(records, key=lambda item: (int(item.get("pair_index", 0)), str(item.get("atom_id") or ""))):
        pair_index = int(row.get("pair_index", _first_int(row.get("pair_support"), default=0)))
        atom_id = str(row.get("atom_id") or f"lapose_motion_pair:{pair_index}")
        atoms.append(
            normalize_cross_paradigm_atom(
                {
                    "adapter": "lapose_plan",
                    "paradigm": "lapose_planning",
                    "atom_id": atom_id,
                    "family": str(row.get("family") or "lapose_motion_atom"),
                    "family_group": "lapose_motion_atom",
                    "pareto_scope": "lapose_motion",
                    "byte_delta": _first_present(row, "byte_delta", "estimated_charged_bytes", default=0),
                    "expected_seg_dist_delta": _first_present(row, "expected_seg_dist_delta", default=0.0),
                    "expected_pose_dist_delta": _first_present(row, "expected_pose_dist_delta", default=0.0),
                    "confidence": row.get("confidence", 0.5),
                    "interaction_assumptions": _unique_strings(
                        [
                            "lapose_planning_signal_not_payload_until_archive_consumer",
                            "may_coordinate_with_pose_foveation_and_mask_atoms",
                            "requires_charged_motion_or_pose_consumer",
                            *_string_list(row.get("interaction_assumptions")),
                        ]
                    ),
                    "archive_manifest_path": _first_str(
                        row,
                        payload,
                        keys=("archive_manifest_path", "candidate_archive_manifest_path"),
                    ),
                    "archive_manifest_sha256": _first_str(
                        row,
                        payload,
                        keys=("archive_manifest_sha256", "candidate_archive_manifest_sha256"),
                    ),
                    "evidence_grade": str(row.get("evidence_grade") or payload.get("evidence_grade") or "prediction"),
                    "dispatch_blockers": _unique_strings(
                        [
                            "lapose_requires_charged_archive_consumer",
                            "lapose_requires_confidence_calibration",
                            *_string_list(payload.get("dispatch_blockers")),
                            *_string_list(row.get("dispatch_blockers")),
                        ]
                    ),
                    "pair_support": _int_list(row.get("pair_support") or ([pair_index] if pair_index else [])),
                    "hard_pair_support": _int_list(row.get("hard_pair_support")),
                    "class_support": _int_list(row.get("class_support")),
                    "geometry_priors": _string_list(row.get("geometry_priors")),
                    "openpilot_priors": _string_list(row.get("openpilot_priors")),
                    "allocation_inference": bool(row.get("allocation_inference", False)),
                    "evidence_source_path": str(row.get("evidence_source_path") or evidence_source_path),
                    "evidence_source_sha256": str(row.get("evidence_source_sha256") or evidence_source_sha256),
                    "source_archive_sha256": str(row.get("source_archive_sha256") or source_archive_sha256),
                }
            )
        )
    return _dedupe_and_sort_atoms(atoms)


def atoms_from_foveation_plan(
    manifest: Mapping[str, Any],
    *,
    evidence_source_path: str = "",
    evidence_source_sha256: str = "",
) -> list[dict[str, Any]]:
    """Adapt charged foveation readiness manifests into one common atom."""

    payload_sha = str(manifest.get("sha256") or "")
    wire = str(manifest.get("wire_format") or "foveation")
    atom_suffix = payload_sha[:12] if payload_sha else _slug(str(manifest.get("path") or manifest.get("member") or "params"))
    ok = bool(manifest.get("ok", False))
    evidence_grade = str(manifest.get("evidence_grade") or "empirical_payload_custody")
    if not ok and evidence_grade.lower() not in {"invalid", "prediction"}:
        evidence_grade = "invalid"
    atom = {
        "adapter": "foveation_plan",
        "paradigm": "foveation_planning",
        "atom_id": f"foveation:{_slug(wire)}:{_slug(atom_suffix)}",
        "family": "foveation_parameter_payload",
        "family_group": "foveation_planning",
        "pareto_scope": "foveation_geometry",
        "byte_delta": _first_present(manifest, "byte_delta", "bytes", default=0),
        "expected_seg_dist_delta": _first_present(manifest, "expected_seg_dist_delta", default=0.0),
        "expected_pose_dist_delta": _first_present(manifest, "expected_pose_dist_delta", default=0.0),
        "confidence": manifest.get("confidence", 0.7 if ok else 0.0),
        "interaction_assumptions": [
            "requires_charged_foveation_params",
            "geometry_runtime_consumer_required",
            "may_coordinate_with_mask_and_pose_atoms",
        ],
        "archive_manifest_path": _first_str(
            manifest,
            keys=("archive_manifest_path", "candidate_archive_manifest_path"),
        ),
        "archive_manifest_sha256": _first_str(
            manifest,
            keys=("archive_manifest_sha256", "candidate_archive_manifest_sha256"),
        ),
        "evidence_grade": evidence_grade,
        "dispatch_blockers": _unique_strings(
            [
                "foveation_requires_runtime_consumer_and_geometry_preflight",
                *(["foveation_readiness_not_ok"] if not ok else []),
                *_string_list(manifest.get("dispatch_blockers")),
            ]
        ),
        "geometry_priors": ["hyperbolic_foveation_geometry"],
        "source_archive_sha256": str(manifest.get("source_archive_sha256") or ""),
        "evidence_source_path": evidence_source_path,
        "evidence_source_sha256": evidence_source_sha256,
    }
    return [normalize_cross_paradigm_atom(atom)]


def atoms_from_adapter_payload(
    adapter: str,
    payload: Mapping[str, Any],
    *,
    evidence_source_path: str = "",
    evidence_source_sha256: str = "",
) -> list[dict[str, Any]]:
    """Dispatch to one named adapter."""

    normalized = adapter.replace("-", "_").lower()
    if normalized in {"hnerv", "hnerv_rate", "hnerv_rate_recode", "hnerv_rate_recode_profile"}:
        return atoms_from_hnerv_rate_recode_profile(
            payload,
            evidence_source_path=evidence_source_path,
            evidence_source_sha256=evidence_source_sha256,
        )
    if normalized in {"wr01", "wr01_wavelet", "wr01_wavelet_plan"}:
        return atoms_from_wr01_wavelet_plan(
            payload,
            evidence_source_path=evidence_source_path,
            evidence_source_sha256=evidence_source_sha256,
        )
    if normalized in {"categorical", "categorical_mask", "categorical_openpilot_mask"}:
        return atoms_from_categorical_openpilot_mask_plan(
            payload,
            evidence_source_path=evidence_source_path,
            evidence_source_sha256=evidence_source_sha256,
        )
    if normalized in {"lapose", "la_pose", "lapose_plan"}:
        return atoms_from_lapose_plan(
            payload,
            evidence_source_path=evidence_source_path,
            evidence_source_sha256=evidence_source_sha256,
        )
    if normalized in {"foveation", "foveation_plan"}:
        return atoms_from_foveation_plan(
            payload,
            evidence_source_path=evidence_source_path,
            evidence_source_sha256=evidence_source_sha256,
        )
    raise CrossParadigmAtomError(f"unknown cross-paradigm atom adapter: {adapter}")


def _wr01_apply_manifest_atom(
    plan: Mapping[str, Any],
    *,
    evidence_source_path: str,
    evidence_source_sha256: str,
) -> dict[str, Any]:
    source_label = str(plan.get("source_label") or "wr01")
    section_name = str(plan.get("section_name") or "section")
    manifest_path = _first_str(
        plan,
        keys=("archive_manifest_path", "candidate_archive_manifest_path", "manifest_path"),
    )
    manifest_sha = _first_str(
        plan,
        keys=("archive_manifest_sha256", "candidate_archive_manifest_sha256", "manifest_sha256"),
    )
    if manifest_path and not manifest_sha and evidence_source_sha256:
        manifest_sha = (
            _sha256_file_if_exists(manifest_path)
            if manifest_path != evidence_source_path
            else evidence_source_sha256
        )
    return {
        "adapter": "wr01_wavelet_apply_manifest",
        "paradigm": "wr01_wavelet",
        "atom_id": f"wr01_wavelet_apply:{_slug(source_label)}:{_slug(section_name)}",
        "family": "wr01_wavelet_apply_transform",
        "family_group": "wr01_wavelet",
        "pareto_scope": f"wr01_wavelet:{_slug(section_name)}",
        "byte_delta": _first_present(
            plan,
            "candidate_archive_byte_delta_vs_source_estimate",
            "section_byte_delta",
            default=0,
        ),
        "expected_seg_dist_delta": _first_present(plan, "expected_seg_dist_delta", default=0.0),
        "expected_pose_dist_delta": _first_present(plan, "expected_pose_dist_delta", default=0.0),
        "confidence": plan.get("confidence", 0.35 if plan.get("ready_for_archive_preflight") else 0.2),
        "interaction_assumptions": [
            "scorer_changing_wavelet_residual",
            "stack_after_raw_equal_rate_recodes",
            "requires_component_response_or_exact_cuda_eval",
        ],
        "archive_manifest_path": manifest_path,
        "archive_manifest_sha256": manifest_sha,
        "evidence_grade": str(plan.get("evidence_grade") or "empirical_archive_candidate"),
        "dispatch_blockers": _unique_strings(
            [
                "wr01_apply_changes_decoded_output",
                "wr01_apply_requires_component_benefit_evidence",
                *_string_list(plan.get("dispatch_blockers")),
            ]
        ),
        "source_archive_sha256": str(plan.get("source_archive_sha256") or ""),
        "source_section_sha256": str(plan.get("source_section_sha256") or ""),
        "candidate_section_sha256": str(plan.get("candidate_section_sha256") or ""),
        "candidate_archive_sha256": str(plan.get("candidate_archive_sha256") or ""),
        "candidate_archive_path": str(plan.get("candidate_archive_path") or ""),
        "evidence_source_path": evidence_source_path,
        "evidence_source_sha256": evidence_source_sha256,
    }


def _categorical_rows(plan: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    rows.extend(("top_atoms", row) for row in _mapping_rows(plan.get("top_atoms") or []))
    rows.extend(("class_rows", row) for row in _mapping_rows(plan.get("class_rows") or []))
    candidate_construction_plan = plan.get("candidate_construction_plan")
    if isinstance(candidate_construction_plan, Mapping):
        rows.extend(
            ("candidate_construction_plan_class_rows", row)
            for row in _mapping_rows(candidate_construction_plan.get("class_rows") or [])
        )
    atom_tables = plan.get("atom_tables")
    if isinstance(atom_tables, Mapping):
        for table_name in sorted(str(key) for key in atom_tables):
            rows.extend((table_name, row) for row in _mapping_rows(atom_tables.get(table_name) or []))
    allocation = plan.get("trace_weighted_allocation")
    if isinstance(allocation, Mapping):
        rows.extend(
            ("trace_weighted_allocation", row)
            for row in _mapping_rows(allocation.get("allocation_table") or [])
        )
    if not rows:
        rows.extend(
            ("candidate_policies", row)
            for row in _mapping_rows(plan.get("candidate_policies") or [])
        )
    return sorted(rows, key=lambda item: (item[0], str(item[1].get("atom_id") or item[1].get("policy_name") or _stable_digest(item[1]))))


def _artifact_path_for_suffix(value: Any, suffix: str) -> str:
    """Return the first artifact path ending in ``suffix`` from list/dict payloads."""

    paths: list[str] = []
    if isinstance(value, Mapping):
        for item in value.values():
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("path"), str):
                paths.append(str(item["path"]))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("path"), str):
                paths.append(str(item["path"]))
    for path in sorted(paths):
        if path.endswith(suffix):
            return path
    return ""


def _sha256_file_if_exists(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _lapose_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    atoms = _mapping_rows(payload.get("atoms") or [])
    if atoms:
        return atoms
    records = _mapping_rows(payload.get("records") or [])
    if records:
        return records
    ledger = payload.get("atom_ledger")
    if isinstance(ledger, Mapping):
        rows = _mapping_rows(ledger.get("rows") or [])
        if rows:
            return rows
    return _mapping_rows(payload.get("rows") or [])


def _wr01_atom_id(source_label: str, section_name: str, row: Mapping[str, Any]) -> str:
    if row.get("atom_id"):
        return f"wr01_wavelet:{_slug(source_label)}:{_slug(section_name)}:{_slug(str(row['atom_id']))}"
    if "raw_offset" in row and "level" in row and "coefficient_index" in row:
        return (
            f"wr01_wavelet:{_slug(source_label)}:{_slug(section_name)}:"
            f"off{int(row['raw_offset'])}:l{int(row['level'])}:c{int(row['coefficient_index'])}"
        )
    return f"wr01_wavelet:{_slug(source_label)}:{_slug(section_name)}:{_stable_digest(row)[:12]}"


def _wr01_atom_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("raw_offset", 0)),
        int(row.get("level", 0)),
        int(row.get("coefficient_index", 0)),
        str(row.get("atom_id") or ""),
        _stable_digest(row),
    )


def _reject_duplicate_atom_ids(atoms: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(str(atom["atom_id"]) for atom in atoms)
    duplicates = sorted(atom_id for atom_id, count in counts.items() if count > 1)
    if duplicates:
        raise CrossParadigmAtomError(f"duplicate atom_id values: {', '.join(duplicates)}")


def _dedupe_and_sort_atoms(atoms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for atom in sorted((dict(atom) for atom in atoms), key=lambda item: str(item["atom_id"])):
        atom_id = str(atom["atom_id"])
        counts[atom_id] += 1
        if counts[atom_id] > 1:
            atom["atom_id"] = f"{atom_id}:dup{counts[atom_id]}"
        out.append(atom)
    return out


def _sorted_atoms(atoms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(atom) for atom in sorted(atoms, key=lambda item: str(item["atom_id"]))]


def _source_score_claim_blocker(atom: Mapping[str, Any]) -> list[str]:
    return ["source_score_claim_true_ignored_by_adapter"] if atom.get("score_claim") is True else []


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    value = str(mapping.get(key) or "")
    if not value:
        raise CrossParadigmAtomError(f"atom missing {key}")
    return value


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _first_str(*mappings: Mapping[str, Any], keys: Sequence[str]) -> str:
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value is not None and str(value):
                return str(value)
    return ""


def _float_value(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CrossParadigmAtomError(f"{field} must be numeric") from exc


def _int_value(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CrossParadigmAtomError(f"{field} must be an integer") from exc


def _confidence(value: Any, *, atom_id: str) -> float:
    confidence = _float_value(value, "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise CrossParadigmAtomError(f"{atom_id}: confidence must be in [0, 1]")
    return confidence


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        return [json.dumps(value, sort_keys=True, separators=(",", ":"))]
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return [str(value)] if str(value) else []


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(value)] if value else []
    try:
        return [int(item) for item in value]
    except TypeError:
        return [int(value)]


def _first_int(value: Any, *, default: int) -> int:
    values = _int_list(value)
    return values[0] if values else default


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _nested(mapping: Mapping[str, Any], path: Sequence[str], default: Any) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _plan_mentions_openpilot(plan: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    text = json.dumps({"plan": plan.get("inputs", {}), "row": row}, sort_keys=True, default=str).lower()
    return "openpilot" in text or "ego" in text


def _unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _string_list(value):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _slug(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9._:-]+", "_", lowered)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "item"


def _stable_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "COMMON_ATOM_FIELDS",
    "DEFAULT_DISPATCH_BLOCKERS",
    "SCHEMA_VERSION",
    "CrossParadigmAtomError",
    "atoms_from_adapter_payload",
    "atoms_from_categorical_openpilot_mask_plan",
    "atoms_from_foveation_plan",
    "atoms_from_hnerv_rate_recode_profile",
    "atoms_from_lapose_plan",
    "atoms_from_wr01_wavelet_plan",
    "build_cross_paradigm_atom_ledger",
    "normalize_cross_paradigm_atom",
]
