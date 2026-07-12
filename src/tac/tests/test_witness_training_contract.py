from tac.witness_training_contract import (
    CUDA_V9_PORT_BLOCKERS,
    LOSS_TERM_KEYS,
    cuda_v9_port_receipt,
    curriculum_stage,
    loss_terms_row,
)


def test_loss_terms_row_has_stable_complete_schema():
    row = loss_terms_row(epoch=7, accum_batch=3, terms={"seg": 2.0, "pose": 1.0}, total=3.0)
    assert row["stage"] == "loss_terms"
    assert row["ep"] == 7
    assert tuple(row["terms"]) == LOSS_TERM_KEYS
    assert row["terms"]["persistence"] == 0.0
    assert row["sum_minus_total"] == 0.0


def test_curriculum_stage_uses_typed_fail_safe_caps():
    flags = {
        "--seg-chroma-boundary-start-epoch": "4",
        "--muon-start-epoch": "7",
        "--polyak-finisher-start-epoch": "9",
    }
    assert curriculum_stage(3, flags) == "island_birth_boundary_form"
    assert curriculum_stage(4, flags) == "sharpen_repair"
    assert curriculum_stage(7, flags) == "muon_phase_finish"
    assert curriculum_stage(9, flags) == "polyak_finish"


def test_cuda_coverage_receipt_fails_closed_until_active_controllers_are_twins():
    receipt = cuda_v9_port_receipt()
    assert receipt["status"] == "BLOCKED_NOT_1_TO_1"
    assert receipt["blockers"] == list(CUDA_V9_PORT_BLOCKERS)
    assert len(receipt["blockers"]) == 4
    closed = (
        "--pose-carrier",
        "--structured-init",
        "--accum-pairs",
        "--dseg-aware-taper",
        "--per-group-grad-clip",
        "--curriculum-event-triggered",
        "--pose-finish-engage-on",
        "--polyak-finisher-arm",
    )
    assert all(not any(blocker.startswith(flag) for blocker in receipt["blockers"]) for flag in closed)
    assert {
        "generated_table_pose_carrier_frame0_dispatch_and_learnable_dxi",
        "structured_scorer_sdf_prefit_with_resume_suppression",
        "accum_pairs_8_chunk_atomic_updates_and_accepted_fraction",
        "dseg_aware_feature_taper_and_per_group_gradient_clipping",
        "scorer_derived_curriculum_latches_with_atomic_resume_state",
        "sigma_min_plateau_pose_finish_gate_with_degenerate_banked_r1",
        "polyak_finisher_resumable_additional_candidate_export",
    }.issubset(receipt["score_bearing_primitives_ported"])
