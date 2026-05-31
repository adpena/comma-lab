# SPDX-License-Identifier: MIT
"""Provenance-clean predictive-coding stack-of-stacks contract.

The stack is deliberately a TAC API, not a memo convention: runners can ask for
an executable work-selection plan and get deterministic member rows, bridge
readiness, and fail-closed blockers before any archive materialization happens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import ordered_unique

PREDICTIVE_CODING_STACK_OF_STACKS_SCHEMA = (
    "tac_predictive_coding_stack_of_stacks_plan.v1"
)
PREDICTIVE_CODING_STACK_MEMBER_ROW_SCHEMA = (
    "tac_predictive_coding_stack_member_work_row.v1"
)
CANONICAL_STACK_MEMBER_IDS: tuple[str, ...] = (
    "z8_hierarchical_predictive_coding",
    "z7_mamba2",
    "dreamer_v3_rssm",
    "z6_v2_cargo_cult_unwind",
    "z4_atick_redlich",
)
FALSIFIED_STACK_MEMBER_TOKENS: tuple[str, ...] = (
    "compound_c",
    "pact_nerv_compound_c",
    "compound_c_renderer",
    "falsified_compound_c",
)


class PredictiveCodingStackError(ValueError):
    """Raised when a requested stack contains non-clean provenance."""


@dataclass(frozen=True)
class PredictiveCodingStackMember:
    """One provenance-clean member eligible for stack-of-stacks planning."""

    member_id: str
    package: str
    role: str
    mathematical_position: str
    entropy_position_label: str
    provenance_status: str
    archive_surface: str
    archive_bound_bridge_entrypoint: str | None
    tags: tuple[str, ...]

    @property
    def archive_bound_bridge_ready(self) -> bool:
        return bool(self.archive_bound_bridge_entrypoint)

    def work_row(self, *, order: int) -> dict[str, Any]:
        bridge_blockers = (
            []
            if self.archive_bound_bridge_ready
            else [f"{self.member_id}_archive_bound_bridge_missing"]
        )
        return {
            "schema": PREDICTIVE_CODING_STACK_MEMBER_ROW_SCHEMA,
            "member_id": self.member_id,
            "stack_order": int(order),
            "package": self.package,
            "role": self.role,
            "mathematical_position": self.mathematical_position,
            "entropy_position_label": self.entropy_position_label,
            "provenance_status": self.provenance_status,
            "archive_surface": self.archive_surface,
            "archive_bound_bridge_entrypoint": self.archive_bound_bridge_entrypoint,
            "archive_bound_bridge_ready": self.archive_bound_bridge_ready,
            "tags": list(self.tags),
            "blockers": bridge_blockers,
            "allowed_use": "provenance_clean_stack_work_selection",
            "forbidden_use": "score_claim_or_direct_exact_dispatch",
            **FALSE_AUTHORITY,
        }


CANONICAL_STACK_MEMBERS: dict[str, PredictiveCodingStackMember] = {
    "z8_hierarchical_predictive_coding": PredictiveCodingStackMember(
        member_id="z8_hierarchical_predictive_coding",
        package="tac.substrates.z8_hierarchical_predictive_coding",
        role="canonical_quadruple_terminal",
        mathematical_position="rao_ballard_plus_mallat_plus_dreamerv3_plus_wyner_ziv",
        entropy_position_label="before_entropy_coder",
        provenance_status="provenance_clean_validated_member",
        archive_surface=(
            "tac.substrates.z8_hierarchical_predictive_coding."
            "canonical_quadruple_binding.build_z8hpc1_archive_bytes_from_canonical_quadruple"
        ),
        archive_bound_bridge_entrypoint=None,
        tags=("z8", "predictive_coding", "mallat", "dreamer_v3", "wyner_ziv"),
    ),
    "z7_mamba2": PredictiveCodingStackMember(
        member_id="z7_mamba2",
        package="tac.substrates.time_traveler_l5_z7_mamba2",
        role="state_space_predictive_coding",
        mathematical_position="mamba2_ssd_temporal_state_update",
        entropy_position_label="before_entropy_coder",
        provenance_status="provenance_clean_validated_member",
        archive_surface="tac.substrates.time_traveler_l5_z7_mamba2.archive_candidate.export_z7_mamba2_mlx_archive",
        archive_bound_bridge_entrypoint=(
            "tac.substrates.time_traveler_l5_z7_mamba2.archive_candidate."
            "export_z7_mamba2_mlx_archive_bound_candidate_package"
        ),
        tags=("z7", "mamba2", "predictive_coding", "mlx_substrate"),
    ),
    "dreamer_v3_rssm": PredictiveCodingStackMember(
        member_id="dreamer_v3_rssm",
        package="tac.substrates.dreamer_v3_rssm",
        role="categorical_posterior_world_model",
        mathematical_position="rssm_discrete_categorical_posterior_capacity",
        entropy_position_label="before_entropy_coder",
        provenance_status="provenance_clean_validated_member",
        archive_surface="tac.substrates.dreamer_v3_rssm.archive.pack_archive",
        archive_bound_bridge_entrypoint=None,
        tags=("dreamer_v3", "rssm", "categorical_posterior", "mlx_substrate"),
    ),
    "z6_v2_cargo_cult_unwind": PredictiveCodingStackMember(
        member_id="z6_v2_cargo_cult_unwind",
        package="tac.substrates.z6_v2_cargo_cult_unwind",
        role="ego_motion_conditioned_predictive_coding",
        mathematical_position="rao_ballard_film_ego_motion_conditioning",
        entropy_position_label="before_entropy_coder",
        provenance_status="provenance_clean_validated_member",
        archive_surface="tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate.export_z6_v2_mlx_archive",
        archive_bound_bridge_entrypoint=(
            "tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate."
            "export_z6_v2_mlx_archive_bound_candidate_package"
        ),
        tags=("z6", "predictive_coding", "rao_ballard", "mlx_substrate"),
    ),
    "z4_atick_redlich": PredictiveCodingStackMember(
        member_id="z4_atick_redlich",
        package="tac.substrates.time_traveler_l5_z4",
        role="cooperative_receiver_decorrelation",
        mathematical_position="atick_redlich_spatial_mutual_information_decorrelation",
        entropy_position_label="before_entropy_coder",
        provenance_status="provenance_clean_validated_member",
        archive_surface="tac.substrates.time_traveler_l5_z4.archive_candidate.export_z4_archive",
        archive_bound_bridge_entrypoint=(
            "tac.substrates.time_traveler_l5_z4.archive_candidate."
            "export_z4_archive_bound_candidate_package"
        ),
        tags=("z4", "cooperative_receiver", "atick_redlich"),
    ),
}


def _canonical_member_id(raw: str) -> str:
    value = str(raw or "").strip().lower().replace("-", "_")
    aliases = {
        "z8": "z8_hierarchical_predictive_coding",
        "z7": "z7_mamba2",
        "z7_mamba_2": "z7_mamba2",
        "mamba": "z7_mamba2",
        "mamba2": "z7_mamba2",
        "dreamer": "dreamer_v3_rssm",
        "dreamerv3": "dreamer_v3_rssm",
        "dreamer_v3": "dreamer_v3_rssm",
        "rssm": "dreamer_v3_rssm",
        "z6": "z6_v2_cargo_cult_unwind",
        "z6_v2": "z6_v2_cargo_cult_unwind",
        "z4": "z4_atick_redlich",
        "atick_redlich": "z4_atick_redlich",
    }
    return aliases.get(value, value)


def _falsified_token(value: str) -> str | None:
    normalized = _canonical_member_id(value)
    lower = normalized.lower()
    for token in FALSIFIED_STACK_MEMBER_TOKENS:
        if token in lower:
            return token
    return None


def build_predictive_coding_stack_of_stacks_plan(
    requested_member_ids: Sequence[str] | None = None,
    *,
    require_archive_bound_bridge: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Return a fail-closed executable plan for the validated stack members."""

    raw_ids = tuple(requested_member_ids or CANONICAL_STACK_MEMBER_IDS)
    blockers: list[str] = []
    rejected: list[dict[str, str]] = []
    members: list[PredictiveCodingStackMember] = []
    for raw in raw_ids:
        member_id = _canonical_member_id(raw)
        falsified_token = _falsified_token(member_id)
        if falsified_token:
            rejected.append(
                {
                    "requested_member_id": str(raw),
                    "canonical_token": falsified_token,
                    "reason": "falsified_or_phantom_provenance_member_rejected",
                }
            )
            blockers.append(f"falsified_stack_member_rejected:{falsified_token}")
            continue
        member = CANONICAL_STACK_MEMBERS.get(member_id)
        if member is None:
            rejected.append(
                {
                    "requested_member_id": str(raw),
                    "canonical_token": member_id,
                    "reason": "unknown_stack_member_not_in_validated_set",
                }
            )
            blockers.append(f"unknown_stack_member_rejected:{member_id}")
            continue
        members.append(member)

    member_rows = [member.work_row(order=i) for i, member in enumerate(members)]
    bridge_blockers = [
        blocker
        for row in member_rows
        for blocker in row["blockers"]
        if blocker.endswith("_archive_bound_bridge_missing")
    ]
    if require_archive_bound_bridge:
        blockers.extend(bridge_blockers)
    blockers = ordered_unique(blockers)
    if strict and blockers:
        raise PredictiveCodingStackError("; ".join(blockers))
    archive_bound_bridge_complete = not bridge_blockers
    return {
        "schema": PREDICTIVE_CODING_STACK_OF_STACKS_SCHEMA,
        "requested_member_ids": list(raw_ids),
        "canonical_member_ids": [member.member_id for member in members],
        "canonical_validated_member_set": list(CANONICAL_STACK_MEMBER_IDS),
        "member_rows": member_rows,
        "member_count": len(member_rows),
        "archive_bound_bridge_ready_count": sum(
            1 for row in member_rows if row["archive_bound_bridge_ready"]
        ),
        "archive_bound_bridge_complete": archive_bound_bridge_complete,
        "require_archive_bound_bridge": bool(require_archive_bound_bridge),
        "compound_c_leakage_detected": any(
            item["canonical_token"] == "compound_c"
            or "compound_c" in item["canonical_token"]
            for item in rejected
        ),
        "rejected_members": rejected,
        "blockers": blockers,
        "provenance_clean": not rejected,
        "stack_executable": not blockers,
        "ready_for_exact_eval_dispatch": False,
        "score_authority": False,
        "allowed_use": "bounded_runner_work_selection_and_bridge_gap_detection",
        "forbidden_use": "score_claim_or_exact_dispatch_authority",
        **FALSE_AUTHORITY,
    }


__all__ = [
    "CANONICAL_STACK_MEMBERS",
    "CANONICAL_STACK_MEMBER_IDS",
    "FALSIFIED_STACK_MEMBER_TOKENS",
    "PREDICTIVE_CODING_STACK_MEMBER_ROW_SCHEMA",
    "PREDICTIVE_CODING_STACK_OF_STACKS_SCHEMA",
    "PredictiveCodingStackError",
    "PredictiveCodingStackMember",
    "build_predictive_coding_stack_of_stacks_plan",
]
