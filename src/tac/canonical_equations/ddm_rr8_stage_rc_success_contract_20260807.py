# SPDX-License-Identifier: MIT
"""RR8 stage return-code success contract."""
from __future__ import annotations

from collections.abc import Mapping

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_rr8_stage_rc_success_contract_v1"
SOURCE_ARTIFACT = ".omx/research/ddm_rr8_20260806/ROUND8_FINDINGS.md"


def stage_chain_success(
    *, detached_done_rc: int | None, stage_rcs: Mapping[str, int]
) -> dict[str, bool | tuple[str, ...]]:
    """A detached done receipt is successful only if every named stage rc is zero."""

    failed = tuple(name for name, rc in sorted(stage_rcs.items()) if int(rc) != 0)
    done_ok = detached_done_rc == 0
    return {
        "success": done_ok and not failed,
        "done_receipt_ok": done_ok,
        "failed_stages": failed,
        "false_success_if_done_only": done_ok and bool(failed),
    }


def build_ddm_rr8_stage_rc_success_contract_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        SOURCE_ARTIFACT,
        reactivation_criteria=(
            "append anchors for any detached driver whose outer done rc disagrees with "
            "a named stage rc; convert recurring instances into preflight refusals"
        ),
        measurement_axis="[apparatus / scorer-free]",
        hardware_substrate="source_inspection",
        captured_at_utc="2026-08-07T00:05:41Z",
    )
    predicted = stage_chain_success(
        detached_done_rc=0,
        stage_rcs={"shard_0": 0, "shard_1": 0, "shard_2": 0, "final": 1},
    )
    anchor = EmpiricalAnchor(
        anchor_id="rr8_et4_done_rc0_final_rc1_false_success_20260806",
        measurement_utc="2026-08-07T00:05:41Z",
        inputs={
            "driver": "/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_n600_driver.sh",
            "detached_done_receipt": ".omx/tmp/codex_runs/et4_chain_v3.done.done",
            "detached_done_rc": 0,
            "stage_rcs": {"shard_0": 0, "shard_1": 0, "shard_2": 0, "final": 1},
        },
        predicted_output=predicted,
        empirical_output={
            "false_success_receipt_observed": True,
            "repair_path_later_produced_valid_rc0_final_receipt": True,
            "data_corruption_found_in_checked_receipts": False,
        },
        residual=0.0,
        source_artifact=SOURCE_ARTIFACT,
        measurement_method=(
            "RR8 source/receipt inspection of ET4 detached driver, shards_rc.txt, and "
            "detached .done receipt"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Detached run success requires every named stage return code",
        one_line_summary=(
            "A detached rc=0 receipt is not a success proof when any dependent stage rc is "
            "nonzero; watchers must consume stage receipts or explicit rc propagation."
        ),
        latex_form=r"\mathrm{success} = (r_{done}=0)\wedge\bigwedge_s(r_s=0)",
        python_callable_module_path=(
            "tac.canonical_equations.ddm_rr8_stage_rc_success_contract_20260807:"
            "stage_chain_success"
        ),
        domain_of_validity={
            "included": [
                "detached shell or Python drivers with named dependent stages",
                "watcher receipts that summarize a multi-stage run",
            ],
            "excluded": [
                "single-process jobs with no dependent stage rc surface",
                "claiming final artifact validity without artifact-specific checks",
            ],
            "verdict_scope": "APPARATUS: launch/driver success semantics",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"detached_done_rc": "process return code", "stage_rcs": "mapping stage->rc"},
        units_out={
            "success": "bool",
            "failed_stages": "stage names",
            "false_success_if_done_only": "bool",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"rr8_false_success_contract_residual": 0.0},
        last_calibration_utc="2026-08-07T00:05:41Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/preflight_hook.py rc-fallthrough scanner",
            "detached job watchers",
            "recursive review fire_schedule_composition_surface",
        ),
        canonical_producers=(
            ".omx/research/ddm_rr8_20260806/ROUND8_FINDINGS.md",
            "fw1 rc propagation cure receipts",
        ),
        provenance=provenance,
    )


def populate_ddm_rr8_stage_rc_success_contract_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_ddm_rr8_stage_rc_success_contract_v1()
    register_canonical_equation(
        eq,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="ddm_cq1 registration: RR8 detached stage rc success contract",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "SOURCE_ARTIFACT",
    "build_ddm_rr8_stage_rc_success_contract_v1",
    "populate_ddm_rr8_stage_rc_success_contract_v1",
    "stage_chain_success",
]
