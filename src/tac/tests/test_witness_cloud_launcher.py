from tac.deploy.witness_cloud_launcher import CUDA_ENV, LANE_ID, build_plan


def test_modal_plan_is_deterministic_plan_first_and_harvestable():
    kwargs = {
        "provider": "modal",
        "gt_cache": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        "label": "unit_cuda",
        "gpu": "T4",
        "epochs": 3,
        "num_pairs": 2,
    }
    a = build_plan(**kwargs)
    b = build_plan(**kwargs)
    assert a.plan_sha256 == b.plan_sha256
    assert a.lane_id == LANE_ID
    assert a.dispatch_argv[:4] == (
        ".venv/bin/modal", "run", "--detach", "experiments/modal_train_lane.py"
    )
    assert "--lane-id" in a.dispatch_argv
    assert "--from-ledger" in a.harvest_argv and "--execute" in a.harvest_argv
    assert a.environment["WITNESS_TRAINER_MODE"] == "full"
    assert a.environment["DALI_DISABLE_NVML"] == CUDA_ENV["DALI_DISABLE_NVML"]
    assert "GT cache SHA-256 custody value is not supplied" in a.setup_blockers
    assert not a.execution_allowed
    assert a.pointer == {"score": 0.19108282, "axis": "contest-CPU", "moved": False}


def test_modal_plan_carries_exact_asset_custody_into_remote_environment():
    digest = "a" * 64
    plan = build_plan(
        provider="modal",
        gt_cache="cache.npz",
        gt_cache_sha256=digest,
        label="unit",
        gpu="T4",
        epochs=1,
        num_pairs=1,
    )
    assert plan.gt_cache_sha256 == digest
    assert plan.environment["WITNESS_GT_CACHE_SHA256"] == digest
    assert "GT cache SHA-256 custody value is not supplied" not in plan.setup_blockers


def test_scaffold_providers_refuse_execution_without_invented_actuator():
    for provider in ("aws", "gcp"):
        plan = build_plan(
            provider=provider,
            gt_cache="cache.npz",
            label="unit",
            gpu="T4",
            epochs=1,
            num_pairs=1,
        )
        assert plan.status == "scaffold"
        assert not plan.execution_allowed
        assert plan.asset_stage_argv == plan.dispatch_argv == plan.harvest_argv == ()
