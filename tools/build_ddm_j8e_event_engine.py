#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the execution-disabled #688 DDM event engine and custody receipts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final

from tac.optimization.ddm_event_continuation import build_j8e_event_continuation
from tac.optimization.ddm_witness_program import (
    DDMWitnessProgramV1,
    MetricSelectorV1,
    SolveHookV1,
)
from tac.optimization.direct_description_joint_descent import (
    DirectDescriptionJointDescentTypedConfigV1,
    rfc8785_canonicalize,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_TICKET: Final = REPO_ROOT / ".omx/research/configs/ddm_ws2_j7_366_w_joint_20260724.json"
OUTPUT_TICKET: Final = REPO_ROOT / ".omx/research/configs/ddm_j8e_688_event_engine_20260724.json"
COMPILE_RECEIPT: Final = REPO_ROOT / ".omx/research/ddm_j8e_688_compile_receipt_20260724.json"
ADAPTER_RECEIPT: Final = REPO_ROOT / ".omx/research/ddm_j8e_688_dm4_adapter_receipt_20260724.json"
RESMOKE_RECEIPT: Final = REPO_ROOT / ".omx/research/ddm_j8e_688_step4_resmoke_readiness_20260724.json"
AUTHORITY_PATH: Final = Path(
    "/Users/adpena/Projects/pact/.omx/tmp/codex_runs/"
    "ddm_j8e_688_engine_build_20260724T154342Z.wrapped.prompt.txt"
)
AUTHORITY_SHA256: Final = "b65bc8e2e38a61618ba85d253f6c1ab44ea7b3ab2ff7ae3d0991960980959cc0"
AUTHORITY_BYTES: Final = 8211
DM4_RECEIPT: Final = (
    REPO_ROOT
    / ".omx/research/ddm_dm4_targeted_realization_cures_20260724T142722Z/"
    "ddm_dm4_targeted_realization_cures_receipt.json"
)
STEP4_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "ddm_ws3_w_joint_exact_history_20260724T132200Z"
)
STEP4_CHECKPOINT: Final = (
    STEP4_ROOT
    / "checkpoints/01_residual_bucket_realized_acceptance_intra_global000004.npz"
)
STEP4_VERDICT: Final = (
    STEP4_ROOT
    / "verdicts/01_residual_bucket_realized_acceptance_step000004_n600.json"
)
STEP4_FULL_RECEIPT: Final = STEP4_ROOT / "full_run_receipt.json"
FINAL_MEMORY_RECEIPT: Final = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "ddm_j8e_688_fresh_memory_preflight_final2_20260724T173000Z/"
    "worst_geometry_memory_preflight.json"
)
BLOCKER: Final = "BLOCKED_DM4_SCORER_RECURSIVE_PROPOSAL_LACKS_J5_COUNTED_APPLICATION_OPERATOR"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_custody(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"custody file unavailable: {path}")
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _repo_custody(relative_path: str) -> dict[str, Any]:
    custody = _file_custody(REPO_ROOT / relative_path)
    custody["path"] = relative_path
    return custody


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _program(
    *,
    source_bindings: dict[str, str],
    ticket_path: str,
) -> DDMWitnessProgramV1:
    graph = build_j8e_event_continuation(
        maximum_receiver_verdicts=450,
        maximum_wall_seconds=49_657.37114489195,
        maximum_counted_bytes=200_000,
        execution_allowed=False,
    )
    hooks = (
        SolveHookV1(
            hook_id="fork_head_solve",
            event="ws3_receiver_closed_start_bound",
            implementation="matched_update_rms_fork_head_solve",
            execution_enabled=False,
            required_receipts=("matched_update_rms", "exact_receiver_parseback"),
            blocker="DDM_MATCHED_UPDATE_RMS_RECEIPT_MISSING",
        ),
        SolveHookV1(
            hook_id="head_offset_solver",
            event="ncde_solve_basin",
            implementation="low_dimensional_head_offset_solver",
            execution_enabled=False,
            required_receipts=("head_offset_receiver_custody", "exact_joint_delta"),
            blocker="DDM_HEAD_OFFSET_RECEIVER_CUSTODY_MISSING",
        ),
        SolveHookV1(
            hook_id="ms2_terminal_solve",
            event="ncde_solve_basin",
            implementation="metric_active_ms2_terminal_solve",
            execution_enabled=False,
            required_receipts=("metric_active_candidate", "exact_joint_delta"),
            blocker="DDM_MS2_NO_ADMISSIBLE_METRIC_ACTIVE_N600_CANDIDATE",
        ),
        SolveHookV1(
            hook_id="mc_finisher",
            event="post_descent",
            implementation="op_gc1_5_preregistered_mc_finisher",
            execution_enabled=False,
            required_receipts=("op_gc1_5_main_review", "exact_joint_delta"),
            blocker="PREREGISTRATION_ONLY",
        ),
    )
    return DDMWitnessProgramV1(
        program_id="ddm_j8e_688_event_continuation_n600_seed0",
        event_continuation=graph,
        metric_selector=MetricSelectorV1(
            selector_id="dm4_corrected_j_rank4_fisher_receiver_exact_acceptance"
        ),
        solve_hooks=hooks,
        ticket_path=ticket_path,
        source_bindings=source_bindings,
        beta2=0.999,
        ema_decay=0.997,
        inference_shadow="ema",
        execution_allowed=False,
        op_gc1_5_execution_enabled=False,
    )


def _build_ticket(program: DDMWitnessProgramV1) -> dict[str, Any]:
    ticket = copy.deepcopy(_read_json(SOURCE_TICKET))
    inherited_execution_custody = copy.deepcopy(ticket["execution_custody"])
    semantic = ticket["semantic_program"]
    old_schedule = semantic["full_run_schedule"]
    semantic["program_id"] = program.program_id
    semantic["full_run_schedule"] = {
        "train_batch": int(old_schedule["train_batch"]),
        "learning_rate_quantum_fraction": float(old_schedule["learning_rate_quantum_fraction"]),
        "measured_seconds_per_step": float(old_schedule["measured_seconds_per_step"]),
        "measured_seconds_per_step_low": float(old_schedule["measured_seconds_per_step_low"]),
        "measured_seconds_per_step_high": float(old_schedule["measured_seconds_per_step_high"]),
        "warm_start_reform": old_schedule["warm_start_reform"],
        "pose_finish_engage": old_schedule["pose_finish_engage"],
        "event_graph": program.event_continuation.to_payload(),
    }
    semantic["warm_start"]["resume_checkpoint"] = {
        **_file_custody(STEP4_CHECKPOINT),
        "global_step": 4,
        "archive_sha256_at_checkpoint": (
            "9601e777010b1dc45ed0841e118fcf34c58452324f8730fe9958a3440502e3a4"
        ),
        "authority": "optimizer_resume_state_plus_exact_realized_archive_identity",
    }
    semantic["execution_boundary"] = {
        "bounded_n600_smoke_only": True,
        "event_engine_execution_allowed": False,
        "main_review_required": True,
        "research_only": True,
        "score_claim": False,
        "this_branch_campaign_launch_authorized": False,
    }
    semantic["proposal_sources"] = {
        "dm4_scorer_recursive": {
            **_file_custody(DM4_RECEIPT),
            "adapter": "tac.optimization.ddm_dm4_j5_adapter.adapt_dm4_proposals",
            "proposal_types": ["seg-only", "pose-only(frame_0)", "joint"],
            "application_authority": "fail_closed_until_counted_J5_application_operator_exists",
        }
    }
    semantic["gc_items_2_6_9"] = {
        "item_2": {
            "ordering": "Pareto(g_S,g_L), nondominated first, exact-admissible next, stable proposal_id tie",
            "scalar_blend_forbidden": True,
        },
        "item_6": {
            "charge_audit": "exact two-part description plus state-transition charge",
            "rate_break_even_threshold": 100.0 * 37_545_489.0 / 25.0,
        },
        "item_9": {
            "arm_id": "ddm_gc1_op5_d_first_v14_falsifier_v1",
            "preregistered": True,
            "execution_enabled": False,
        },
    }
    semantic["ddm_witness_program"] = program.to_payload()
    semantic["value_provenance"].update(
        {
            "event_budgets": (
                "DERIVED safety caps from the resealed J7 450-verdict and measured "
                "13.79371420691443-hour upper geometry; never stages or a descent stop"
            ),
            "ema_decay_j8e": (
                "DERIVED by LawRef ema_decay_run_geometry_v1 and required to resolve "
                "exactly 0.997 without fallback"
            ),
            "box_milestone": (
                "OPERATOR DIRECTIVE 2026-07-24T15:50:30: milestone only; exact joint "
                "Delta-S descent continues until economics/dynamics/resource cap"
            ),
            "dm4_proposal_metric": (
                "MEASURED/DERIVED DM4 corrected-J rank-4 Fisher rows with exact resize "
                "adjoint and scorer-recursive ERF support; application remains blocked"
            ),
        }
    )
    semantic_hash = _sha256_bytes(rfc8785_canonicalize(semantic))
    ticket["authority"] = {
        "delegation_prompt_path": str(AUTHORITY_PATH),
        "delegation_prompt_sha256": AUTHORITY_SHA256,
        "delegation_prompt_bytes": AUTHORITY_BYTES,
        "source_commit": "396202272822ff515366e3759e77aa74be262367",
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
    }
    ticket["compile_custody"] = {
        "claim": "HASH_SEALED_TYPED_J8E_EVENT_CONTINUATION_EXECUTION_DISABLED",
        "hash_algorithm": "sha256",
        "hash_scope": "RFC8785 canonical bytes of semantic_program",
        "semantic_program_sha256": semantic_hash,
        "typed_target": "DDMWitnessProgramV1",
        "typed_schema": "DirectDescriptionJointDescentTypedConfigV1",
    }
    ticket["execution_custody"] = {
        "source_files": {
            name: _repo_custody(path)
            for name, path in {
                "launcher": "tools/launch_ddm_joint_descent.py",
                "consumer": "src/tac/optimization/direct_description_joint_descent.py",
            }.items()
        },
        "j5_producer_artifacts": inherited_execution_custody["j5_producer_artifacts"],
        "worst_geometry_memory_receipt": (
            _file_custody(FINAL_MEMORY_RECEIPT)
            if FINAL_MEMORY_RECEIPT.is_file()
            else inherited_execution_custody["worst_geometry_memory_receipt"]
        ),
        "banked_r1_comparator": inherited_execution_custody["banked_r1_comparator"],
        "event_engine_sources": {
            name: _repo_custody(path)
            for name, path in {
                "event_engine": "src/tac/optimization/ddm_event_continuation.py",
                "witness_program": "src/tac/optimization/ddm_witness_program.py",
                "dm4_adapter": "src/tac/optimization/ddm_dm4_j5_adapter.py",
                "dm4_constructor": "src/tac/optimization/ddm_dm4_targeted_realization_cures.py",
            }.items()
        },
        "step4_checkpoint": _file_custody(STEP4_CHECKPOINT),
        "step4_exact_verdict": _file_custody(STEP4_VERDICT),
        "step4_full_run_receipt": _file_custody(STEP4_FULL_RECEIPT),
        "dm4_receipt": _file_custody(DM4_RECEIPT),
    }
    return ticket


def build() -> dict[str, Any]:
    authority = _file_custody(AUTHORITY_PATH)
    if authority["bytes"] != AUTHORITY_BYTES or authority["sha256"] != AUTHORITY_SHA256:
        raise RuntimeError("delegated authority custody differs")
    source_paths = {
        "launcher": "tools/launch_ddm_joint_descent.py",
        "consumer": "src/tac/optimization/direct_description_joint_descent.py",
        "event_engine": "src/tac/optimization/ddm_event_continuation.py",
        "dm4_adapter": "src/tac/optimization/ddm_dm4_j5_adapter.py",
        "dm4_constructor": "src/tac/optimization/ddm_dm4_targeted_realization_cures.py",
    }
    source_bindings = {
        name: _repo_custody(path)["sha256"] for name, path in source_paths.items()
    }
    ticket_rel = str(OUTPUT_TICKET.relative_to(REPO_ROOT))
    program = _program(source_bindings=source_bindings, ticket_path=ticket_rel)
    ticket = _build_ticket(program)
    _atomic_json(OUTPUT_TICKET, ticket)
    typed = DirectDescriptionJointDescentTypedConfigV1.from_ticket(OUTPUT_TICKET)
    argv, compile_manifest = program.compile_trainer_argv_with_constants(
        repo_root=REPO_ROOT,
        out_dir=(
            "/Volumes/VertigoDataTier/pact/experiments/results/"
            "ddm_j8e_688_event_engine_main_review_required"
        ),
        mode="dry-run",
        resume_from=str(STEP4_CHECKPOINT),
    )
    compile_receipt = {
        **compile_manifest,
        "ticket": _repo_custody(ticket_rel),
        "semantic_program_sha256": typed.dsl_compile_hash,
        "consumer_typed_config_hash": typed.typed_config_hash(),
        "real_argv": list(argv),
        "main_review_required": True,
        "execution_allowed": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    _atomic_json(COMPILE_RECEIPT, compile_receipt)

    base_archive_path = Path(typed.source_archive_path)
    base_archive = base_archive_path.read_bytes()
    dm4_sha = _file_custody(DM4_RECEIPT)["sha256"]
    if dm4_sha != typed.semantic_program["proposal_sources"]["dm4_scorer_recursive"]["sha256"]:
        raise RuntimeError("typed J5 consumer DM4 source SHA differs")
    disabled_output, disabled_proposals, disabled_receipt = typed.dm4_j5_proposal_source(
        base_archive=base_archive,
        enabled=False,
    )
    enabled_output, enabled_proposals, enabled_receipt = typed.dm4_j5_proposal_source(
        base_archive=base_archive,
        enabled=True,
    )
    if disabled_output != base_archive or enabled_output != base_archive or disabled_proposals:
        raise RuntimeError("DM4/J5 adapter violated proposal-source-only archive identity")
    adapter_receipt = {
        "schema": "ddm_j8e_688_dm4_adapter_receipt.v1",
        "disabled_identity": disabled_receipt,
        "enabled_proposal_source": enabled_receipt,
        "proposal_count": len(enabled_proposals),
        "proposals": [proposal.to_payload() for proposal in enabled_proposals],
        "application_contract": {
            "status": BLOCKER,
            "required_input": (
                "counted parse-back-exact RGB/semantic payload or a deterministic map "
                "from DM4 support coordinates into the 368 J5 grammar coordinates"
            ),
            "required_output": (
                "candidate archive bytes re-emitted by the J5 receiver and exact n600 "
                "Seg/Pose plus archive-byte verdict"
            ),
            "silent_pixel_injection_forbidden": True,
        },
        "score_claim": False,
        "pointer_moved": False,
        "main_review_required": True,
    }
    _atomic_json(ADAPTER_RECEIPT, adapter_receipt)

    checkpoint_custody = _file_custody(STEP4_CHECKPOINT)
    verdict_custody = _file_custody(STEP4_VERDICT)
    step4 = _read_json(STEP4_VERDICT)
    resmoke = {
        "schema": "ddm_j8e_688_step4_resmoke_readiness.v1",
        "starting_state": {
            "checkpoint": checkpoint_custody,
            "exact_verdict": verdict_custody,
            "global_step": 4,
            "archive_sha256": step4["archive_sha256"],
            "archive_bytes": int(step4["archive_bytes"]),
            "d_seg": float(step4["d_seg"]),
            "d_pose": float(step4["d_pose"]),
            "advisory_action": float(step4["advisory_action"]),
        },
        "bounded_compatibility_smoke": {
            "attempted": True,
            "dm4_scorer_recursive_proposals_exposed": len(enabled_proposals),
            "proposal_manifest_sha256": enabled_receipt["proposal_manifest_sha256"],
            "exact_candidate_archive_materialized": False,
            "frozen_n600_scorer_replay_executed": False,
            "reason": BLOCKER,
            "delta_S": None,
            "delta_S_per_wall_clock_hour": None,
            "delta_bytes_per_step": None,
        },
        "readiness_verdict": BLOCKER,
        "ready_to_fire_under_standing_go": False,
        "fire_or_dispatch_performed": False,
        "named_cure": (
            "Land one typed counted J5 application operator for a DM4 proposal, preserve "
            "its payload/parse-back bytes, then run exactly one frozen n600 scorer replay "
            "from this Step-4 checkpoint and require measured joint Delta-S < 0 including "
            "the realized coder-byte delta."
        ),
        "box_milestone_policy": (
            "box is a checkpointed milestone for descent and a tolerance stop only for "
            "describe/solve; descent stop remains economic/dynamics/resource-budget based"
        ),
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "main_review_required": True,
    }
    _atomic_json(RESMOKE_RECEIPT, resmoke)
    return {
        "ticket": _repo_custody(ticket_rel),
        "compile_receipt": _repo_custody(str(COMPILE_RECEIPT.relative_to(REPO_ROOT))),
        "adapter_receipt": _repo_custody(str(ADAPTER_RECEIPT.relative_to(REPO_ROOT))),
        "resmoke_receipt": _repo_custody(str(RESMOKE_RECEIPT.relative_to(REPO_ROOT))),
        "proposal_count": len(enabled_proposals),
        "readiness_verdict": BLOCKER,
        "ready_to_fire_under_standing_go": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(build(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
