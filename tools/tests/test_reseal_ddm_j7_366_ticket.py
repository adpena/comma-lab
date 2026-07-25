from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tac.optimization.direct_description_entropy_priced_member import rfc8785_canonicalize
from tac.optimization.direct_description_joint_descent import (
    J9_W_JOINT_PROGRAM_SHA256,
    J10_PROGRAM_ID,
    DirectDescriptionJointDescentTypedConfigV1,
    FullRunScheduleV1,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tools.launch_ddm_joint_descent import _expected_full_run_baseline_dseg
from tools.reseal_ddm_j7_366_ticket import (
    J9_ATTEMPT4_RUN,
    J9_PROGRAM_ID,
    REPO,
    WS3_W_SEG_PROGRAM_ID,
    _apply_profile,
    _source_commit,
    ws1_launchable_archive,
)


def test_source_commit_is_worktree_head() -> None:
    assert (
        _source_commit()
        == subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=REPO,
            text=True,
        ).strip()
    )


def test_ws1_receiver_closed_archives_are_launchable_starts() -> None:
    w_seg = ws1_launchable_archive("W_seg")
    w_joint = ws1_launchable_archive("W_joint")

    assert w_seg["kind"] == w_joint["kind"] == "receiver_closed_ws1_archive"
    assert w_seg["bytes"] == 138_031
    assert w_joint["bytes"] == 138_801
    assert w_seg["sha256"] != w_joint["sha256"]
    assert w_seg["optimizer_state_loadable"] is False
    assert w_joint["optimizer_state_loadable"] is False


def test_ws1_tickets_use_their_sealed_exact_baseline() -> None:
    w_seg = DirectDescriptionJointDescentTypedConfigV1.from_ticket(
        REPO / ".omx/research/configs/ddm_ws2_j7_366_w_seg_20260724.json"
    )
    w_joint = DirectDescriptionJointDescentTypedConfigV1.from_ticket(
        REPO / ".omx/research/configs/ddm_ws2_j7_366_w_joint_20260724.json"
    )

    assert _expected_full_run_baseline_dseg(w_seg) == 0.024124510023328993
    assert _expected_full_run_baseline_dseg(w_joint) == 0.07051923116048177


def test_ws3_profile_is_wseg_only_and_keeps_campaign_acceptance_strict() -> None:
    ticket = json.loads((REPO / ".omx/research/configs/ddm_ws2_j7_366_w_seg_20260724.json").read_bytes())
    semantic = ticket["semantic_program"]
    _apply_profile(
        semantic,
        profile="ws3_w_seg_reformed_opening",
        selected_warm_start="W_seg",
    )
    schedule = FullRunScheduleV1.from_semantic_program(semantic)
    assert semantic["program_id"] == WS3_W_SEG_PROGRAM_ID
    assert schedule is not None
    reform = schedule.warm_start_reform
    assert reform is not None
    assert reform.realized_acceptance_policy == "campaign_component_safe_exact_n600"
    assert reform.proposal_ordering == "seg_lexicographic_proxy_then_exact_component_gate"


def test_j9_profile_types_restart_and_measured_over_24h_schedule() -> None:
    ticket = json.loads((REPO / ".omx/research/configs/ddm_ws3_w_joint_history_fill_20260724.json").read_bytes())
    semantic = ticket["semantic_program"]
    _apply_profile(
        semantic,
        profile="j9_geometry_escape_cure",
        selected_warm_start="W_joint",
        failed_run_dir=J9_ATTEMPT4_RUN,
    )

    schedule = FullRunScheduleV1.from_semantic_program(semantic)
    assert semantic["program_id"] == J9_PROGRAM_ID
    assert hashlib.sha256(rfc8785_canonicalize(semantic)).hexdigest() == J9_W_JOINT_PROGRAM_SHA256
    assert semantic["resume_after_attempt4"]["decision"] == ("RESTART_FROM_W_JOINT_INSUFFICIENT_SEED_CUSTODY")
    assert semantic["resume_after_attempt4"]["byte_compare_performed"] is False
    assert schedule.checkpoint_interval_steps == 37
    assert schedule.measured_seconds_per_step == 312.0
    assert semantic["full_run_schedule"]["derived_wall_clock_hours"] == 39.363878897499916
    assert all(stage.verdict_interval_steps == 50 for stage in schedule.stages)


def test_j10_ticket_binds_materialized_live_state_and_lawref_ema_policy() -> None:
    path = REPO / ".omx/research/configs/ddm_j10_366_ema_verdict_shadow_cure_20260725.json"
    ticket = json.loads(path.read_bytes())
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(path)
    schedule = config.full_run_schedule

    assert ticket["semantic_program"]["program_id"] == J10_PROGRAM_ID
    assert (
        hashlib.sha256(rfc8785_canonicalize(ticket["semantic_program"])).hexdigest()
        == ticket["compile_custody"]["semantic_program_sha256"]
    )
    assert config.typed_config_hash() == ticket["compile_custody"]["typed_config_hash"]
    assert config.source_archive_sha256 == ("2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241")
    assert config.source_baseline_d_seg == 0.06974277072482639
    assert config.source_baseline_d_pose == 35.49982080959101
    assert config.ema_decay == 46.0 / 49.0
    assert schedule is not None
    assert [stage.maximum_steps for stage in schedule.stages] == [100, 150, 150]
    assert all(stage.verdict_interval_steps == 50 for stage in schedule.stages)
    assert schedule.verdict_shadow_policy is not None
    assert schedule.verdict_shadow_policy.decision_shadow == "live"
    assert schedule.verdict_shadow_policy.export_shadow == "ema"
    compile_payload = ticket["semantic_program"]["joint_objective"]["ema_decay_compile"]
    assert compile_payload["lawref_declaration"]["equation_id"] == ("ema_decay_run_geometry_v1")
    assert compile_payload["constant_manifest"]["fallback_used"] is False
    assert 3.0 / (1.0 - config.ema_decay) == pytest.approx(49.0)
    assert 1.0 - config.ema_decay**50 > 0.5
    assert ticket["semantic_program"]["resume_after_attempt5"]["preserved_accepted_steps"] == 50


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("baseline_d_seg", 0.069, "baseline verdict binding differs"),
        ("receipt_sha256", "00" * 32, "receipt SHA-256 differs"),
    ],
)
def test_j10_ticket_refuses_materialized_custody_mutation(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    source = REPO / ".omx/research/configs/ddm_j10_366_ema_verdict_shadow_cure_20260725.json"
    ticket = json.loads(source.read_bytes())
    ticket["semantic_program"]["warm_start"][field] = value
    ticket["compile_custody"]["semantic_program_sha256"] = hashlib.sha256(
        rfc8785_canonicalize(ticket["semantic_program"])
    ).hexdigest()
    mutated = tmp_path / "mutated_j10.json"
    mutated.write_text(json.dumps(ticket), encoding="utf-8")

    with pytest.raises(DirectDescriptionError, match=error):
        DirectDescriptionJointDescentTypedConfigV1.from_ticket(mutated)
