# SPDX-License-Identifier: MIT
"""L5 v2 lattice measurement schedule.

This is a planning artifact, not score evidence. It turns the C1/Z5/TT5L
staircase into a first-match measurement lattice so operators can see which
exact probe closes the next blocker without letting additive score-band
language regain rank authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tac.optimization.l5_v2_probe_disambiguator import L5V2_CANDIDATES

L5V2_MEASUREMENT_SCHEDULE_SCHEMA = "l5_v2_lattice_measurement_schedule_v1"
L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH = (
    "tools/build_l5_v2_lattice_measurement_schedule.py"
)
L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH = (
    ".omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.json"
)
L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH = (
    ".omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.md"
)


def _candidate_rows_from_intake(intake: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(intake, Mapping):
        return {}
    verdict = intake.get("verdict")
    if not isinstance(verdict, Mapping):
        return {}
    rows = verdict.get("evaluated_observations")
    if not isinstance(rows, list):
        return {}
    out: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            out[candidate_id] = row
    return out


def _eligible_candidate_ids(intake: Mapping[str, Any] | None) -> set[str]:
    out: set[str] = set()
    for candidate_id, row in _candidate_rows_from_intake(intake).items():
        if row.get("eligible_for_architecture_lock") is True:
            out.add(candidate_id)
    return out


def _candidate_blockers(intake: Mapping[str, Any] | None, candidate_id: str) -> list[str]:
    row = _candidate_rows_from_intake(intake).get(candidate_id)
    blockers = row.get("blockers") if isinstance(row, Mapping) else None
    if not isinstance(blockers, list):
        return ["l5_v2_probe_observation_missing"]
    return [str(item) for item in blockers]


def _measurement(
    *,
    measurement_id: str,
    candidate_id: str,
    purpose: str,
    estimated_cost_usd: float,
    expected_information_gain_nats: float,
    required_axes: tuple[str, ...] = ("contest_cpu", "contest_cuda"),
    output_artifact: str,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "measurement_id": measurement_id,
        "candidate_id": candidate_id,
        "purpose": purpose,
        "estimated_cost_usd": estimated_cost_usd,
        "expected_information_gain_nats": expected_information_gain_nats,
        "required_axes": list(required_axes),
        "output_artifact": output_artifact,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers or [],
    }


def _probe_measurement_ids_for_missing(candidate_ids: list[str]) -> list[str]:
    measurement_by_candidate = {
        "c1_world_model_foveation": "measure_c1_world_model_foveation_paired_exact",
        "z5_predictive_coding_world_model": "measure_z5_predictive_coding_paired_exact",
        "time_traveler_l5_autonomy": "measure_tt5l_autonomy_paired_exact",
    }
    return [
        measurement_by_candidate[candidate_id]
        for candidate_id in candidate_ids
        if candidate_id in measurement_by_candidate
    ]


def build_l5_v2_lattice_measurement_schedule(
    *,
    probe_intake: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the first-match L5 v2 measurement schedule.

    The active rule is computed from probe intake only. Missing or incomplete
    intake routes to paired C1/Z5/TT5L probe filling; fully eligible intake
    routes to the side-info causal effect curve before paired anchor promotion.
    """

    eligible = _eligible_candidate_ids(probe_intake)
    missing_or_blocked = [
        candidate_id for candidate_id in L5V2_CANDIDATES if candidate_id not in eligible
    ]
    measurements = [
        _measurement(
            measurement_id="measure_c1_world_model_foveation_paired_exact",
            candidate_id="c1_world_model_foveation",
            purpose="fill C1 paired CPU/CUDA exact probe observation",
            estimated_cost_usd=2.0,
            expected_information_gain_nats=0.25,
            output_artifact=(
                "experiments/results/l5_v2_probe/"
                "c1_world_model_foveation_paired_exact.json"
            ),
            blockers=_candidate_blockers(probe_intake, "c1_world_model_foveation"),
        ),
        _measurement(
            measurement_id="measure_z5_predictive_coding_paired_exact",
            candidate_id="z5_predictive_coding_world_model",
            purpose="fill Z5 paired CPU/CUDA exact probe observation",
            estimated_cost_usd=5.0,
            expected_information_gain_nats=0.35,
            output_artifact=(
                "experiments/results/l5_v2_probe/"
                "z5_predictive_coding_world_model_paired_exact.json"
            ),
            blockers=_candidate_blockers(
                probe_intake, "z5_predictive_coding_world_model"
            ),
        ),
        _measurement(
            measurement_id="measure_tt5l_autonomy_paired_exact",
            candidate_id="time_traveler_l5_autonomy",
            purpose="fill TT5L paired CPU/CUDA exact probe observation",
            estimated_cost_usd=7.5,
            expected_information_gain_nats=0.55,
            output_artifact=(
                "experiments/results/l5_v2_probe/"
                "time_traveler_l5_autonomy_paired_exact.json"
            ),
            blockers=_candidate_blockers(probe_intake, "time_traveler_l5_autonomy"),
        ),
        _measurement(
            measurement_id="measure_tt5l_sideinfo_effect_curve",
            candidate_id="time_traveler_l5_autonomy",
            purpose=(
                "separate side-info consumption from causal usefulness via "
                "zero, random-LSB, shuffled, trained, and ablated side-info"
            ),
            estimated_cost_usd=1.0,
            expected_information_gain_nats=0.40,
            required_axes=("contest_cuda",),
            output_artifact=(
                "experiments/results/l5_v2_probe/"
                "tt5l_sideinfo_effect_curve.jsonl"
            ),
            blockers=[
                "tt5l_sideinfo_effect_curve_missing",
                "consumption_proof_is_not_yet_usefulness_proof",
            ],
        ),
        _measurement(
            measurement_id="prepare_l5_v2_paired_anchor_packet",
            candidate_id="time_traveler_l5_autonomy",
            purpose=(
                "materialize paired-axis anchor packet only after C1/Z5/TT5L "
                "probe observations and side-info effect curve are present"
            ),
            estimated_cost_usd=0.0,
            expected_information_gain_nats=0.10,
            output_artifact=(
                "experiments/results/l5_v2_probe/"
                "tt5l_paired_anchor_packet_manifest.json"
            ),
            blockers=[
                "requires_probe_disambiguator_architecture_lock",
                "requires_terminal_claim_templates",
            ],
        ),
    ]

    rules = [
        {
            "rule_id": "fill_missing_c1_z5_tt5l_probe_observations",
            "condition": "any required candidate lacks paired exact probe eligibility",
            "matches": bool(missing_or_blocked),
            "measurement_ids": _probe_measurement_ids_for_missing(missing_or_blocked),
            "missing_or_blocked_candidates": missing_or_blocked,
        },
        {
            "rule_id": "measure_tt5l_sideinfo_effect_curve",
            "condition": "all required probes eligible but TT5L causal usefulness curve missing",
            "matches": not missing_or_blocked,
            "measurement_ids": ["measure_tt5l_sideinfo_effect_curve"],
            "missing_or_blocked_candidates": [],
        },
        {
            "rule_id": "prepare_paired_anchor_packet",
            "condition": "probe lock and side-info effect curve are both present",
            "matches": False,
            "measurement_ids": ["prepare_l5_v2_paired_anchor_packet"],
            "missing_or_blocked_candidates": [],
        },
    ]
    active_rule = next((rule for rule in rules if rule["matches"]), rules[-1])
    return {
        "schema": L5V2_MEASUREMENT_SCHEDULE_SCHEMA,
        "tool": L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH,
        "first_match_wins": True,
        "active_rule_id": active_rule["rule_id"],
        "active_measurement_ids": list(active_rule["measurement_ids"]),
        "required_candidates": list(L5V2_CANDIDATES),
        "eligible_candidates": sorted(eligible),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_reward_allowed": False,
        "rules": rules,
        "measurements": measurements,
    }


def render_l5_v2_lattice_measurement_schedule_markdown(
    schedule: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing schedule report."""

    lines = [
        "# L5 v2 lattice measurement schedule",
        "",
        f"- schema: `{schedule.get('schema')}`",
        f"- active_rule_id: `{schedule.get('active_rule_id')}`",
        f"- active_measurement_ids: `{schedule.get('active_measurement_ids')}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "",
        "## Measurements",
    ]
    measurements = schedule.get("measurements")
    if isinstance(measurements, list):
        for row in measurements:
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    "",
                    f"### {row.get('measurement_id')}",
                    "",
                    f"- candidate_id: `{row.get('candidate_id')}`",
                    f"- purpose: {row.get('purpose')}",
                    f"- estimated_cost_usd: `{row.get('estimated_cost_usd')}`",
                    "- evidence authority: planning-only until paired exact artifacts land",
                ]
            )
    return "\n".join(lines) + "\n"


def schedule_json(schedule: Mapping[str, Any]) -> str:
    """Return canonical JSON text for durable artifacts."""

    return json.dumps(schedule, indent=2, sort_keys=True, allow_nan=False) + "\n"


__all__ = [
    "L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH",
    "L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH",
    "L5V2_MEASUREMENT_SCHEDULE_SCHEMA",
    "L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH",
    "build_l5_v2_lattice_measurement_schedule",
    "render_l5_v2_lattice_measurement_schedule_markdown",
    "schedule_json",
]
