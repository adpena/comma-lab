from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = REPO_ROOT / "tools" / "recover_pr95_training_curriculum.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "recover_pr95_training_curriculum",
        HELPER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recover_pr95_curriculum_from_public_intake_source() -> None:
    helper = _load_helper()

    curriculum = helper.recover_curriculum(helper.DEFAULT_SOURCE_DIR)

    assert [stage["name"] for stage in curriculum["stages"]] == [
        "stage1_v328_ce",
        "stage2_v331_softplus",
        "stage3_v332_smooth",
        "stage4_v332_qat",
        "stage5_c1a_l7",
        "stage6_lambda_sweep",
        "stage7_sigma_sweep",
        "stage8_muon_finetune",
    ]
    assert curriculum["total_epochs"] == 29650

    stage4 = curriculum["stages"][3]
    assert stage4["epochs"] == 500
    assert stage4["uses_qat"] is True
    assert stage4["loss_family"] == "smooth_disagreement_seg_loss"

    stage8 = curriculum["stages"][7]
    assert stage8["epochs"] == 5000
    assert stage8["uses_muon"] is True
    assert stage8["adamw_lr"] == 1e-5
    assert stage8["muon_lr"] == 2e-4
    assert stage8["muon_weight_decay"] == 5e-4
    assert stage8["resume"] == "previous_stage_final"


def test_campaign_estimates_are_planning_only_and_cover_required_gpus() -> None:
    helper = _load_helper()

    campaign = helper.estimate_campaign(29650)

    estimates = {row["gpu"]: row for row in campaign["gpu_hour_estimates"]}

    assert set(estimates) >= {"A100_40GB", "H100", "L40S", "T4"}
    assert estimates["T4"]["gpu_hours_low"] == 250
    assert estimates["H100"]["gpu_hours_high"] == 70
    assert campaign["stop_gates"][0]["gate"] == "smoke"
    assert campaign["stop_gates"][-1]["gate"] == "exact_cuda_cpu_eval"


def test_curriculum_mutation_matrix_keeps_pr95_as_control_arm() -> None:
    helper = _load_helper()

    matrix = helper.curriculum_mutation_matrix()

    mutation_ids = {row["id"] for row in matrix}
    assert "control_pr95_exact_replay" in mutation_ids
    assert "score_domain_stage_boundary_controller" in mutation_ids
    assert "pr101_microcodec_export_over_pr95_weights" in mutation_ids
    assert "score_aware_residual_atom_stack" in mutation_ids
    assert all(row["dispatch_eligible"] is False for row in matrix)
    assert all(row["exact_gate"] for row in matrix)


def test_render_markdown_includes_curriculum_mutation_matrix() -> None:
    helper = _load_helper()

    payload = helper.build_payload(helper.DEFAULT_SOURCE_DIR, None)
    markdown = helper.render_markdown(payload)

    assert "## Curriculum Mutation Matrix" in markdown
    assert "`earlier_muon_partition_sweep`" in markdown
    assert "`hard_pair_waterfill_sampler`" in markdown
