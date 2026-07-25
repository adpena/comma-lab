#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministically reseal the J7 #366 ticket and fail closed on fake starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.ddm_ws1_warm_start import (  # noqa: E402
    parse_ws1_warm_start_archive,
    receive_joint_descent_archive,
)
from tac.optimization.direct_description_entropy_priced_member import (  # noqa: E402
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    J7_PROGRAM_SHA256,
    J7_W_JOINT_PROGRAM_SHA256,
    J7_W_SEG_PROGRAM_SHA256,
    J9_W_JOINT_PROGRAM_SHA256,
    J10_PROGRAM_ID,
    WS3_W_SEG_PROGRAM_SHA256,
    DirectDescriptionJointDescentTypedConfigV1,
    compile_parameterized_archive,
    lift_v15_archive,
    load_stage_checkpoint,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402
from tac.witness_dsl.curriculum_dsl import EmaDecayCalibrated  # noqa: E402
from tac.witness_dsl.lawref import lawref_to_declaration  # noqa: E402

AUTHORITY_SHA256 = "8dac31beda848b94b8bd42f43ffd7008cd024fcf916c0a14149307f68085907e"
AUTHORITY_BYTES = 7497
WS3_AUTHORITY_SHA256 = "b36872d5ce619dc741cf619542878dd87ff244defb94a03eeb16a68fe1bbd6c7"
WS3_AUTHORITY_BYTES = 7441
J10_AUTHORITY_SHA256 = "dc0d7bf469056f66249abe58579c4cd09219051d91096a1384d2062adc4bc9a1"
J10_AUTHORITY_BYTES = 10078
AUTHORITY_CUSTODY = {
    AUTHORITY_SHA256: AUTHORITY_BYTES,
    WS3_AUTHORITY_SHA256: WS3_AUTHORITY_BYTES,
    J10_AUTHORITY_SHA256: J10_AUTHORITY_BYTES,
}
PROGRAM_ID = "ddm_j7_366_pose_gate_history_reseal_n600_seed0"
WS3_W_SEG_PROGRAM_ID = "ddm_ws3_w_seg_window_completion_reformed_n600_seed0"
VERDICT_BATCH = 32
J9_PROGRAM_ID = "ddm_j9_366_geometry_escape_cure_n600_seed0"
J10_ATTEMPT5_RUN = Path("/Volumes/VertigoDataTier/pact/experiments/results/ddm_366_campaign_v5_cured_20260725T062259Z")
J10_ATTEMPT5_CHECKPOINT_SHA256 = "043c2a8b3c89688510cc0ff002f37a375a974205a5f8760d93133c47b7cec7c1"
J10_ATTEMPT5_VERDICT_SHA256 = "3610fb624dc4d807b44d28bab5f867e5e5ad1f05104aa984663ab286977e6951"
J10_ATTEMPT5_RECEIPT_SHA256 = "01bbe8f27d83bf16b0dc3bc5a17d8650f9b4a521f4bb4747023b84846bd0cb97"
J9_ATTEMPT4_RUN = Path("/Volumes/VertigoDataTier/pact/experiments/results/ddm_366_campaign_wjoint_20260725T032557Z")
J9_PROPOSAL_SECONDS = (
    452.71083845803514,
    436.65467699989676,
    410.85283304192126,
)
J9_MAIN_LOOP_SECONDS = (
    272.22402341710404,
    273.1033891250845,
    272.74639812507667,
    272.4739739999641,
    329.62735133292153,
    329.45078045804985,
)
J9_DERIVED_CENTRAL_STEP_SECONDS = 5.2 * 60.0
WARM_START_RECEIPT = REPO / ".omx/research/ddm_ws2_warm_start_custody_producer_receipt_20260724.json"
PROGRAM_SHA_BY_WARM_START = {
    "inherited_v15_control": J7_PROGRAM_SHA256,
    "W_seg": J7_W_SEG_PROGRAM_SHA256,
    "W_joint": J7_W_JOINT_PROGRAM_SHA256,
}


def _derive_j9_replay_decision(run_dir: Path) -> dict[str, Any]:
    """Attempt replay admission and type the forced restart when state custody is absent."""

    root = run_dir.resolve()
    if root != J9_ATTEMPT4_RUN:
        raise DirectDescriptionError("J9 replay custody must use the sealed attempt-4 run directory")
    identity_path = root / "run_identity.json"
    log_path = root / "run.log"
    if not identity_path.is_file() or not log_path.is_file():
        raise DirectDescriptionError("J9 attempt-4 identity or fatal log is unavailable")
    identity = json.loads(identity_path.read_bytes())
    if (
        identity.get("schema") != "ddm_joint_descent_run_identity.v1"
        or int(identity.get("seed", -1)) != 0
        or identity.get("source_archive_sha256") != "5aa45850ab05d47f411583fd7582e27644c5bf289cd6d5bc32c05a52706c433e"
    ):
        raise DirectDescriptionError("J9 attempt-4 run identity differs")
    if "G1 Movable polygon escaped scorer geometry" not in log_path.read_text(encoding="utf-8"):
        raise DirectDescriptionError("J9 attempt-4 fatal geometry line is absent")
    telemetry_paths = sorted((root / "telemetry").glob("step*.json"))
    if len(telemetry_paths) != 7:
        raise DirectDescriptionError("J9 attempt-4 telemetry does not contain exactly steps 1-7")
    telemetry = [json.loads(path.read_bytes()) for path in telemetry_paths]
    if [int(row.get("global_step", -1)) for row in telemetry] != list(range(1, 8)):
        raise DirectDescriptionError("J9 attempt-4 telemetry steps are not contiguous 1-7")
    required_state = {"theta", "ema", "first_moment", "second_moment", "run_cursor"}
    state_fields_present = sorted(required_state & set().union(*(set(row) for row in telemetry)))
    checkpoints = sorted((root / "checkpoints").glob("*.npz")) if (root / "checkpoints").is_dir() else []
    if checkpoints or state_fields_present:
        raise DirectDescriptionError(
            "J9 attempt-4 unexpectedly gained state custody; perform byte-compare replay before reseal"
        )
    return {
        "schema": "ddm_j9_attempt4_replay_decision.v1",
        "attempted": True,
        "attempted_steps": list(range(1, 8)),
        "seed": 0,
        "proposal_sources": [str(row["proposal_source"]) for row in telemetry],
        "telemetry": [
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "current_parseback_archive_sha256": row["current_parseback_archive_sha256"],
            }
            for path, row in zip(telemetry_paths, telemetry, strict=True)
        ],
        "required_state_fields": sorted(required_state),
        "state_fields_present": state_fields_present,
        "preserved_checkpoint_count": len(checkpoints),
        "byte_compare_performed": False,
        "decision": "RESTART_FROM_W_JOINT_INSUFFICIENT_SEED_CUSTODY",
        "reason": (
            "seed and proposal_source rows do not preserve gradients, Adam moments, EMA, "
            "theta, run cursor, or candidate archive bytes"
        ),
        "source_run_identity_sha256": _sha256_file(identity_path),
        "source_run_log_sha256": _sha256_file(log_path),
        "score_claim": False,
    }


def materialize_j10_step50_live_warm_start(
    *,
    run_dir: Path,
    prior_ticket: Path,
    archive_out: Path,
    receipt_out: Path,
) -> dict[str, Any]:
    """Materialize attempt-5 LIVE theta without carrying the poisoned EMA decision."""

    root = run_dir.resolve()
    if root != J10_ATTEMPT5_RUN:
        raise DirectDescriptionError("J10 materialization must use the sealed attempt-5 run")
    allowed_roots = (
        Path("/Volumes/VertigoDataTier/pact"),
        Path("/Volumes/APDataStore/pact"),
    )
    if not any(archive_out.resolve().is_relative_to(prefix) for prefix in allowed_roots):
        raise DirectDescriptionError("J10 materialized warm start must land on the SSD tier")
    checkpoint = root / "checkpoints/01_residual_bucket_realized_acceptance_accepted_global000050.npz"
    verdict_path = root / "verdicts/01_residual_bucket_realized_acceptance_step000050_n600.json"
    run_receipt_path = root / "full_run_receipt.json"
    if (
        _sha256_file(checkpoint) != J10_ATTEMPT5_CHECKPOINT_SHA256
        or _sha256_file(verdict_path) != J10_ATTEMPT5_VERDICT_SHA256
        or _sha256_file(run_receipt_path) != J10_ATTEMPT5_RECEIPT_SHA256
    ):
        raise DirectDescriptionError("J10 attempt-5 checkpoint/verdict/receipt custody differs")
    failed_verdict = json.loads(verdict_path.read_bytes())
    failed_receipt = json.loads(run_receipt_path.read_bytes())
    if (
        failed_verdict.get("parameter_shadow") != "ema"
        or int(failed_verdict.get("realized_parameter_count", -1)) != 3
        or failed_verdict.get("realized_stage_decision") != "BLOCKED_REALIZED_DSEG_REGRESSION"
        or failed_receipt.get("verdict") != "BLOCKED_REALIZED_DSEG_REGRESSION"
        or int(failed_receipt.get("global_step", -1)) != 50
    ):
        raise DirectDescriptionError("J10 attempt-5 failure mechanism differs")
    prior_config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(prior_ticket)
    state, metadata = load_stage_checkpoint(checkpoint, config=prior_config)
    if (
        state.step != 50
        or int(metadata.get("run_cursor", {}).get("global_step", -1)) != 50
        or metadata.get("run_cursor", {}).get("campaign_blocker") != "BLOCKED_REALIZED_DSEG_REGRESSION"
    ):
        raise DirectDescriptionError("J10 attempt-5 checkpoint cursor differs")
    source = Path(prior_config.source_archive_path)
    if not source.is_file() or source.is_symlink():
        raise DirectDescriptionError("J10 attempt-5 source archive is unavailable")
    lift = lift_v15_archive(source.read_bytes())
    materialized, realized = compile_parameterized_archive(
        lift,
        state.theta,
        include_lane_programs=False,
    )
    parsed = parse_ws1_warm_start_archive(materialized)
    if parsed.candidate != "W_joint" or parsed.exact_reemit() != materialized:
        raise DirectDescriptionError("J10 materialized warm start WS1 parse-back differs")
    receiver = receive_joint_descent_archive(materialized)
    if getattr(receiver, "archive", materialized) != materialized:
        raise DirectDescriptionError("J10 materialized receiver output differs")
    rematerialized_lift = lift_v15_archive(materialized)
    zero_reemit, zero_realized = compile_parameterized_archive(
        rematerialized_lift,
        np.zeros(len(rematerialized_lift.parameter_names), dtype=np.float32),
        include_lane_programs=False,
    )
    if zero_reemit != materialized or np.count_nonzero(zero_realized) != 0:
        raise DirectDescriptionError("J10 fresh-anchor zero-state parse-back differs")
    _atomic_bytes(archive_out, materialized)
    if archive_out.read_bytes() != materialized:
        raise DirectDescriptionError("J10 materialized warm-start disk bytes differ")
    state_hashes = {
        field: hashlib.sha256(np.ascontiguousarray(getattr(state, field)).tobytes()).hexdigest()
        for field in ("theta", "ema", "first_moment", "second_moment")
    }
    receipt = {
        "schema": "ddm_j10_step50_live_warm_start_materialization.v1",
        "source_run": str(root),
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": J10_ATTEMPT5_CHECKPOINT_SHA256,
            "global_step": state.step,
            "state_array_sha256": state_hashes,
        },
        "failed_ema_verdict": {
            "path": str(verdict_path),
            "sha256": J10_ATTEMPT5_VERDICT_SHA256,
            "parameter_shadow": "ema",
            "realized_parameter_count": 3,
            "d_seg": failed_verdict["d_seg"],
            "reference_d_seg": failed_verdict["reference_d_seg"],
            "reference_shadow_was_live": True,
        },
        "materialized_archive": {
            "path": str(archive_out),
            "bytes": len(materialized),
            "sha256": hashlib.sha256(materialized).hexdigest(),
            "candidate": parsed.candidate,
            "receiver_parseback_identity": True,
            "fresh_zero_state_reemit_identity": True,
            "live_realized_parameter_count": int(np.count_nonzero(realized)),
        },
        "resume_decision": (
            "FRESH_TICKET_WARM_START_FROM_MATERIALIZED_STEP50_LIVE_THETA;"
            "OLD_CHECKPOINT_TYPED_HASH_AND_POISONED_EMA_HISTORY_NOT_REUSED"
        ),
        "optimizer_arrays_preserved_but_not_loaded": True,
        "fresh_ema_anchored_at_materialized_live_state": True,
        "score_claim": False,
        "pointer_moved": False,
    }
    _atomic_json(receipt_out, receipt)
    return receipt


def _apply_profile(
    semantic: dict[str, Any],
    *,
    profile: str,
    selected_warm_start: str,
    failed_run_dir: Path | None = None,
    materialized_receipt: Path | None = None,
    materialized_baseline_verdict: Path | None = None,
) -> None:
    if profile == "j7_custody_refresh":
        semantic["program_id"] = PROGRAM_ID
        return
    if profile == "j10_ema_verdict_shadow_cure":
        if (
            selected_warm_start != "W_joint"
            or failed_run_dir is None
            or failed_run_dir.resolve() != J10_ATTEMPT5_RUN
            or materialized_receipt is None
            or materialized_baseline_verdict is None
        ):
            raise DirectDescriptionError(
                "J10 EMA-shadow cure requires attempt-5, materialization, and baseline custody"
            )
        materialization = json.loads(materialized_receipt.read_bytes())
        baseline = json.loads(materialized_baseline_verdict.read_bytes())
        archive_row = materialization.get("materialized_archive", {})
        if (
            materialization.get("schema") != "ddm_j10_step50_live_warm_start_materialization.v1"
            or materialization.get("fresh_ema_anchored_at_materialized_live_state") is not True
            or archive_row.get("receiver_parseback_identity") is not True
            or archive_row.get("fresh_zero_state_reemit_identity") is not True
            or int(archive_row.get("live_realized_parameter_count", -1)) != 119
            or baseline.get("schema") != "ddm_materialized_warm_start_n600_verdict.v1"
            or baseline.get("archive_sha256") != archive_row.get("sha256")
            or baseline.get("receiver_parseback_identity") is not True
        ):
            raise DirectDescriptionError("J10 materialized warm-start or baseline custody differs")
        archive_path = Path(str(archive_row["path"]))
        if (
            not archive_path.is_file()
            or archive_path.is_symlink()
            or archive_path.stat().st_size != int(archive_row["bytes"])
            or _sha256_file(archive_path) != archive_row["sha256"]
        ):
            raise DirectDescriptionError("J10 materialized warm-start archive bytes differ")
        receipt = json.loads((failed_run_dir / "full_run_receipt.json").read_bytes())
        step_seconds = [float(value) for value in receipt["step_seconds"]]
        if (
            _sha256_file(failed_run_dir / "full_run_receipt.json") != J10_ATTEMPT5_RECEIPT_SHA256
            or len(step_seconds) != 50
        ):
            raise DirectDescriptionError("J10 attempt-5 cadence custody differs")
        measured = step_seconds[1:]
        median_seconds = float(statistics.median(measured))
        schedule = semantic["full_run_schedule"]
        schedule["stages"][0]["maximum_steps"] = 100
        schedule.update(
            {
                "derived_total_steps": 400,
                "resume_global_step_offset": 50,
                "remaining_stage_steps": [100, 150, 150],
                "measured_seconds_per_step": median_seconds,
                "measured_seconds_per_step_low": min(measured),
                "measured_seconds_per_step_high": max(measured),
                "measured_main_loop_step_seconds_attempt5_excluding_opening": measured,
                "derived_remaining_wall_clock_hours": 400 * median_seconds / 3600.0,
                "derived_remaining_wall_clock_hours_low": 400 * min(measured) / 3600.0,
                "derived_remaining_wall_clock_hours_high": 400 * max(measured) / 3600.0,
                "verdict_shadow_policy": {
                    "schema": "ddm_dual_shadow_verdict_policy.v1",
                    "decision_shadow": "live",
                    "export_shadow": "ema",
                    "emit_dual_rows": True,
                    "same_shadow_reference_required": True,
                    "minimum_realized_parameter_count_delta": 1,
                    "maximum_consecutive_degenerate_verdicts": 1,
                },
            }
        )
        equilibration_budget = 49
        time_constants = 3
        original_updates = 450
        warmup_fraction = 2.0 * equilibration_budget / (time_constants * original_updates)
        ema_lever = EmaDecayCalibrated(
            updates_per_run=original_updates,
            target_seed_fraction=None,
            warmup_fraction=warmup_fraction,
        )
        ema_decay = float(ema_lever.overrides["--ema-decay"])
        if (
            not math.isclose(
                time_constants / (1.0 - ema_decay),
                equilibration_budget,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or 1.0 - ema_decay**50 < 0.5
        ):
            raise DirectDescriptionError("J10 derived EMA geometry does not realize by first verdict")
        manifest = dict(ema_lever.constant_manifest["--ema-decay"])
        manifest.pop("resolved_at", None)
        semantic["joint_objective"]["ema_decay"] = ema_decay
        semantic["joint_objective"]["ema_decay_compile"] = {
            "dsl_lever": "EmaDecayCalibrated",
            "compiled_flag": "--ema-decay",
            "lawref_declaration": lawref_to_declaration(ema_lever.lawrefs["--ema-decay"]),
            "constant_manifest": manifest,
            "run_geometry": {
                "original_total_updates": 450,
                "remaining_updates_after_materialized_step50": 400,
                "stage_updates": [150, 150, 150],
                "remaining_stage_updates": [100, 150, 150],
                "first_scheduled_verdict_updates": 50,
                "strict_equilibration_budget_updates": 49,
                "equilibration_time_constants": 3,
                "learning_rate_quantum_fraction": 0.25,
                "historical_named_parameter_count": 706,
                "receiver_effective_parameter_count": 368,
                "minimum_receiver_quantum_by_first_verdict": 1,
            },
        }
        semantic["program_id"] = J10_PROGRAM_ID
        semantic["warm_start"] = {
            "kind": "receiver_closed_checkpoint_materialization",
            "path": str(archive_path),
            "bytes": int(archive_row["bytes"]),
            "sha256": str(archive_row["sha256"]),
            "receipt_path": str(materialized_receipt),
            "receipt_sha256": _sha256_file(materialized_receipt),
            "baseline_verdict_path": str(materialized_baseline_verdict),
            "baseline_verdict_sha256": _sha256_file(materialized_baseline_verdict),
            "baseline_d_seg": float(baseline["d_seg"]),
            "baseline_d_pose": float(baseline["d_pose"]),
            "optimizer_state_loadable": False,
            "fresh_ema_anchor": True,
        }
        semantic["resume_after_attempt5"] = {
            "decision": ("FRESH_TICKET_WARM_START_FROM_MATERIALIZED_STEP50_LIVE_THETA"),
            "preserved_accepted_steps": 50,
            "preserved_checkpoint_sha256": J10_ATTEMPT5_CHECKPOINT_SHA256,
            "true_resume_legal": False,
            "true_resume_blocker": (
                "J10 changes typed_config_hash and intentionally discards the lagged EMA "
                "history; old optimizer arrays stay preserved in the keep-all checkpoint"
            ),
            "materialized_receipt_sha256": _sha256_file(materialized_receipt),
            "baseline_verdict_sha256": _sha256_file(materialized_baseline_verdict),
        }
        semantic["execution_boundary"].update(
            {
                "this_branch_campaign_launch_authorized": False,
                "standing_go_applies_only_after_main_review_and_READY_TO_FIRE_UNDER_STANDING_GO": True,
            }
        )
        semantic["telemetry"]["every_exact_move"].extend(
            [
                "live decision row with same-shadow reference",
                "EMA export row with receiver informativeness classification",
                "VERDICT_NOT_YET_INFORMATIVE first-degenerate grace and next-verdict escalation",
            ]
        )
        semantic["value_provenance"].update(
            {
                "j10_ema_decay": (
                    "DERIVED_AT_CONFIG by EmaDecayCalibrated LawRef ema_decay_run_geometry_v1; "
                    "3/(1-d)=49<50 and first-verdict blend clears nearest-integer receiver quantum"
                ),
                "j10_verdict_shadow_policy": (
                    "L3 apparatus cure: stage control live-vs-live; EMA is export/promotion-only"
                ),
                "j10_attempt5_preservation": (
                    "MEASURED attempt-5 step50 LIVE theta materialized receiver-closed; "
                    "50 accepted checkpoints and old optimizer arrays preserved"
                ),
                "j10_remaining_schedule": (
                    f"DERIVED 400 remaining updates at MEASURED attempt-5 median {median_seconds} seconds/update"
                ),
            }
        )
        return
    if profile == "j9_geometry_escape_cure":
        if selected_warm_start != "W_joint":
            raise DirectDescriptionError("J9 geometry cure profile is valid only for W_joint")
        if failed_run_dir is None:
            raise DirectDescriptionError("J9 geometry cure profile requires attempt-4 run custody")
        semantic["program_id"] = J9_PROGRAM_ID
        schedule = semantic["full_run_schedule"]
        proposal_median = sorted(J9_PROPOSAL_SECONDS)[len(J9_PROPOSAL_SECONDS) // 2]
        schedule.update(
            {
                "measured_seconds_per_step": J9_DERIVED_CENTRAL_STEP_SECONDS,
                "measured_seconds_per_step_low": min(J9_MAIN_LOOP_SECONDS),
                "measured_seconds_per_step_high": max(J9_MAIN_LOOP_SECONDS),
                "measured_full_n600_proposal_seconds": proposal_median,
                "measured_full_n600_proposal_seconds_low": min(J9_PROPOSAL_SECONDS),
                "measured_full_n600_proposal_seconds_high": max(J9_PROPOSAL_SECONDS),
                "measured_main_loop_step_seconds": list(J9_MAIN_LOOP_SECONDS),
                "derived_wall_clock_hours": (
                    schedule["derived_total_steps"] * J9_DERIVED_CENTRAL_STEP_SECONDS
                    + len(J9_PROPOSAL_SECONDS) * proposal_median
                )
                / 3600.0,
                "derived_wall_clock_hours_low": (
                    schedule["derived_total_steps"] * min(J9_MAIN_LOOP_SECONDS)
                    + len(J9_PROPOSAL_SECONDS) * min(J9_PROPOSAL_SECONDS)
                )
                / 3600.0,
                "derived_wall_clock_hours_high": (
                    schedule["derived_total_steps"] * max(J9_MAIN_LOOP_SECONDS)
                    + len(J9_PROPOSAL_SECONDS) * max(J9_PROPOSAL_SECONDS)
                )
                / 3600.0,
                "accepted_step_checkpoint_policy": (
                    "atomic_keep_all_full_resume_after_every_accepted_step;"
                    "periodic_37_and_stage_boundaries_preserved_additively"
                ),
                "geometry_infeasibility_policy": (
                    "rg1_project_polygon_center_then_cured_or_rejected;"
                    "sha_and_parseback_custody_failures_remain_process_fatal"
                ),
            }
        )
        semantic["resume_after_attempt4"] = _derive_j9_replay_decision(failed_run_dir)
        semantic["telemetry"]["every_exact_move"].extend(
            [
                "proposal_infeasible_geometry status cured or rejected",
                "atomic full-resume checkpoint after every accepted step",
            ]
        )
        semantic["value_provenance"].update(
            {
                "j9_geometry_projection": (
                    "REUSED rg1 #679 project_polygon_center on the exact 512x384 scorer-plane legal set"
                ),
                "j9_checkpoint_every_accepted": (
                    "P0 resumability law; optimizer state is KB-scale; periodic-37 and stage checkpoints remain additive"
                ),
                "j9_attempt4_cadence": (
                    "MEASURED attempt-4 full-n600 proposal seconds "
                    "452.71083845803514,436.65467699989676,410.85283304192126; "
                    "main-loop telemetry seconds "
                    "272.22402341710404,273.1033891250845,272.74639812507667,"
                    "272.4739739999641,329.62735133292153,329.45078045804985; "
                    "DERIVED central 5.2 min/step and 39.36 h including the measured opening window"
                ),
                "j9_verdict_cadence": (
                    "PRESERVED sealed verdict_interval_steps=50; no unsealed cadence semantics invented"
                ),
            }
        )
        return
    if profile != "ws3_w_seg_reformed_opening" or selected_warm_start != "W_seg":
        raise DirectDescriptionError("WS3 reformed opening profile is valid only for W_seg")
    semantic["program_id"] = WS3_W_SEG_PROGRAM_ID
    reform = semantic["full_run_schedule"]["warm_start_reform"]
    reform["realized_acceptance_policy"] = "campaign_component_safe_exact_n600"
    reform["proposal_ordering"] = "seg_lexicographic_proxy_then_exact_component_gate"
    semantic["full_run_schedule"]["stage_transition_rule"] = (
        "quarter-quantum base proposal source; seg-lexicographic local ranking; "
        "each receiver-visible proposal is admitted only when exact n600 priced "
        "joint delta is negative, components are safe, and the cumulative "
        "residual fire gate is green; otherwise continue the finite shrink "
        "ladder and preserve exact rollback"
    )
    semantic["value_provenance"]["ws3_reformed_opening"] = (
        "RECALLED #518 resume-warmup plus v16 validity-radius plus J4 freeze-then-release; W_seg instance cure only"
    )


def _expected_program_sha(profile: str, selected_warm_start: str) -> str | None:
    if profile == "j10_ema_verdict_shadow_cure":
        # J10 is self-sealed after re-deriving the exact external attempt-5
        # materialization/baseline custody. The typed consumer independently
        # re-resolves every policy and LawRef field before accepting the hash.
        return None
    if profile == "j9_geometry_escape_cure":
        return J9_W_JOINT_PROGRAM_SHA256
    if profile == "ws3_w_seg_reformed_opening":
        return WS3_W_SEG_PROGRAM_SHA256
    return PROGRAM_SHA_BY_WARM_START[selected_warm_start]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO,
        text=True,
    ).strip()


def ws1_launchable_archive(candidate_id: str) -> dict[str, Any]:
    """Return exact archive custody or refuse endpoint-only WS1 evidence."""

    receipt = json.loads(WARM_START_RECEIPT.read_bytes())
    row = receipt["archive_custody"][candidate_id]
    required = ("archive_path", "archive_sha256", "archive_bytes")
    missing = [field for field in required if field not in row]
    if missing:
        raise DirectDescriptionError(
            "WS1_START_NOT_LAUNCHABLE_ENDPOINT_ONLY: "
            f"{candidate_id} lacks {','.join(missing)}; "
            "the measured camera transform is neither a receiver-closed archive "
            "nor a live optimizer state"
        )
    path = Path(row["archive_path"])
    if not path.is_file() or path.is_symlink():
        raise DirectDescriptionError(f"WS1_START_ARCHIVE_UNAVAILABLE: {candidate_id}: {path}")
    actual_bytes = path.stat().st_size
    actual_sha = _sha256_file(path)
    if actual_bytes != int(row["archive_bytes"]) or actual_sha != row["archive_sha256"]:
        raise DirectDescriptionError(f"WS1_START_ARCHIVE_CUSTODY_DIFFERS: {candidate_id}")
    return {
        "kind": "receiver_closed_ws1_archive",
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "receipt_path": str(WARM_START_RECEIPT.relative_to(REPO)),
        "optimizer_state_loadable": False,
    }


def reseal(
    *,
    ticket_path: Path,
    authority_path: Path,
    memory_receipt: Path | None,
    selected_warm_start: str,
    profile: str = "j7_custody_refresh",
    failed_run_dir: Path | None = None,
    materialized_receipt: Path | None = None,
    materialized_baseline_verdict: Path | None = None,
) -> dict[str, Any]:
    authority_sha = _sha256_file(authority_path)
    if authority_sha not in AUTHORITY_CUSTODY or authority_path.stat().st_size != AUTHORITY_CUSTODY[authority_sha]:
        raise DirectDescriptionError("J7 delegated authority custody differs")
    ticket = json.loads(ticket_path.read_bytes())
    semantic = ticket["semantic_program"]
    _apply_profile(
        semantic,
        profile=profile,
        selected_warm_start=selected_warm_start,
        failed_run_dir=failed_run_dir,
        materialized_receipt=materialized_receipt,
        materialized_baseline_verdict=materialized_baseline_verdict,
    )
    semantic["telemetry"]["verdict_batch"] = VERDICT_BATCH
    semantic["value_provenance"]["verdict_batch"] = "P0 J7 authority: exact n600 frozen CPU scorer verdicts use batch32"
    if selected_warm_start != "inherited_v15_control" and profile != "j10_ema_verdict_shadow_cure":
        semantic["warm_start"] = ws1_launchable_archive(selected_warm_start)
    semantic_sha = hashlib.sha256(rfc8785_canonicalize(semantic)).hexdigest()
    expected_semantic_sha = _expected_program_sha(profile, selected_warm_start)
    if expected_semantic_sha is not None and semantic_sha != expected_semantic_sha:
        raise DirectDescriptionError(f"J7 semantic hash differs: {semantic_sha} != {expected_semantic_sha}")
    ticket["authority"].update(
        {
            "delegation_prompt_path": str(authority_path),
            "delegation_prompt_sha256": authority_sha,
            "delegation_prompt_bytes": AUTHORITY_CUSTODY[authority_sha],
            "source_commit": _source_commit(),
        }
    )
    ticket["compile_custody"].update(
        {
            "semantic_program_sha256": semantic_sha,
            "claim": (
                "HASH_SEALED_EXECUTABLE_TYPED_J10_DUAL_SHADOW_RESEAL"
                if profile == "j10_ema_verdict_shadow_cure"
                else "HASH_SEALED_EXECUTABLE_TYPED_J7_POSE_HISTORY_RESEAL"
            ),
        }
    )
    sources = ticket["execution_custody"]["source_files"]
    for name, relative in {
        "consumer": "src/tac/optimization/direct_description_joint_descent.py",
        "launcher": "tools/launch_ddm_joint_descent.py",
    }.items():
        path = REPO / relative
        sources[name] = {"path": relative, "sha256": _sha256_file(path)}
    memory = ticket["execution_custody"]["worst_geometry_memory_receipt"]
    if memory_receipt is None:
        memory["sha256"] = None
    else:
        if not memory_receipt.is_file() or memory_receipt.is_symlink():
            raise DirectDescriptionError("J7 memory receipt is unavailable")
        memory.update({"path": str(memory_receipt), "sha256": _sha256_file(memory_receipt)})
    _atomic_json(ticket_path, ticket)
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(ticket_path)
    ticket = json.loads(ticket_path.read_bytes())
    ticket["compile_custody"]["typed_config_hash"] = config.typed_config_hash()
    ticket["compile_custody"]["existing_compiler_accepts_schema"] = True
    ticket["compile_custody"]["existing_governed_launcher_accepts_named_config"] = True
    _atomic_json(ticket_path, ticket)
    result = {
        "schema": (
            "ddm_j10_366_ticket_reseal.v1"
            if profile == "j10_ema_verdict_shadow_cure"
            else "ddm_j7_366_ticket_reseal.v1"
        ),
        "ticket_path": str(ticket_path),
        "ticket_sha256": _sha256_file(ticket_path),
        "semantic_program_sha256": semantic_sha,
        "typed_config_hash": config.typed_config_hash(),
        "selected_warm_start": selected_warm_start,
        "profile": profile,
        "resume_after_attempt4": semantic.get("resume_after_attempt4"),
        "resume_after_attempt5": semantic.get("resume_after_attempt5"),
        "verdict_batch": config.verdict_batch,
        "memory_receipt_sealed": memory_receipt is not None,
        "source_files": sources,
        "score_claim": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument(
        "--base-ticket",
        type=Path,
        help="Initialize an absent candidate ticket from this sealed J7 ticket.",
    )
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--memory-receipt", type=Path)
    parser.add_argument(
        "--selected-warm-start",
        choices=("inherited_v15_control", "W_seg", "W_joint"),
    )
    parser.add_argument(
        "--profile",
        choices=(
            "j7_custody_refresh",
            "ws3_w_seg_reformed_opening",
            "j9_geometry_escape_cure",
            "j10_ema_verdict_shadow_cure",
        ),
        default="j7_custody_refresh",
    )
    parser.add_argument("--failed-run-dir", type=Path)
    parser.add_argument("--materialized-receipt", type=Path)
    parser.add_argument("--materialized-baseline-verdict", type=Path)
    parser.add_argument("--materialize-j10-step50", action="store_true")
    parser.add_argument("--materialized-archive-out", type=Path)
    parser.add_argument("--materialization-receipt-out", type=Path)
    args = parser.parse_args()
    try:
        ticket_path = args.ticket.resolve()
        if args.materialize_j10_step50:
            if (
                args.materialized_archive_out is None
                or args.materialization_receipt_out is None
                or args.failed_run_dir is None
                or args.authority is not None
                or args.memory_receipt is not None
                or args.selected_warm_start is not None
                or args.materialized_receipt is not None
                or args.materialized_baseline_verdict is not None
            ):
                raise DirectDescriptionError(
                    "J10 materialization accepts ticket, failed run, archive out, and receipt out only"
                )
            materialize_j10_step50_live_warm_start(
                run_dir=args.failed_run_dir.resolve(),
                prior_ticket=ticket_path,
                archive_out=args.materialized_archive_out.resolve(),
                receipt_out=args.materialization_receipt_out.resolve(),
            )
            return 0
        if args.authority is None or args.selected_warm_start is None:
            raise DirectDescriptionError("reseal requires --authority and --selected-warm-start")
        if not ticket_path.exists():
            if args.base_ticket is None:
                raise DirectDescriptionError("absent candidate ticket requires --base-ticket")
            base_ticket = args.base_ticket.resolve()
            if not base_ticket.is_file() or base_ticket.is_symlink():
                raise DirectDescriptionError("J7 base ticket is unavailable")
            _atomic_json(ticket_path, json.loads(base_ticket.read_bytes()))
        reseal(
            ticket_path=ticket_path,
            authority_path=args.authority.resolve(),
            memory_receipt=(None if args.memory_receipt is None else args.memory_receipt.resolve()),
            selected_warm_start=args.selected_warm_start,
            profile=args.profile,
            failed_run_dir=(None if args.failed_run_dir is None else args.failed_run_dir.resolve()),
            materialized_receipt=(None if args.materialized_receipt is None else args.materialized_receipt.resolve()),
            materialized_baseline_verdict=(
                None if args.materialized_baseline_verdict is None else args.materialized_baseline_verdict.resolve()
            ),
        )
    except DirectDescriptionError as exc:
        print(json.dumps({"verdict": "REFUSE", "reason": str(exc)}), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
