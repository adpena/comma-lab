from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.train_levelset_witness_realized_through_R_torch import (
    _generated_pose_pair_dispatch,
    _jacobian_probe_pair_indices,
    _polyak_checkpoint_blob,
    build_parser,
    main,
)
from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom
from tac.cuda_levelset_training import (
    CudaLevelSetConfig,
    TorchLevelSetWitness,
    TorchPoseCarrier,
)
from tac.cuda_v9_controller_runtime import TorchProtectedIslandSeed


def test_dynamic_lane_gate_composes_detected_class_pre_r_only_on_frame1() -> None:
    cfg = CudaLevelSetConfig(
        n_pairs=2,
        in_feat=5,
        hidden_dim=8,
        n_hidden=1,
        mod_dim=4,
        n_classes=5,
        render_h=6,
        render_w=8,
        camera_h=10,
        camera_w=12,
    )
    model = TorchLevelSetWitness.build(cfg, seed=19)
    geom = GroundHomographyGeom.eon(
        native_hw=(cfg.camera_h, cfg.camera_w), pitch=0.0
    )
    carrier = TorchPoseCarrier.build(np.zeros((2, 6), np.float32), geom)
    model.pose_carrier = carrier
    feats = torch.randn(cfg.render_h * cfg.render_w, cfg.in_feat)
    plain, plain_phi = _generated_pose_pair_dispatch(
        model, feats, [0, 1], carrier, cfg
    )
    lane_band = {
        "priors": {
            pair: SimpleNamespace(
                coverage=np.ones((cfg.render_h, cfg.render_w), np.float32)
            )
            for pair in (0, 1)
        },
        "lane_cls": 4,
        "tau": 1e6,
        "eps": 0.35,
        "weight": 1.0,
    }
    banded, banded_phi, probe = _generated_pose_pair_dispatch(
        model,
        feats,
        [0, 1],
        carrier,
        cfg,
        lane_band=lane_band,
        return_probe_inputs=True,
    )
    assert torch.equal(plain[:, 0], banded[:, 0])
    assert not torch.equal(plain[:, 1], banded[:, 1])
    torch.testing.assert_close(plain_phi, banded_phi)
    assert probe["native0"].shape == (2, cfg.camera_h, cfg.camera_w, 3)
    assert probe["scored1"].shape == (2, cfg.render_h, cfg.render_w, 3)


def test_protected_seed_is_scored_but_witness_alone_excludes_it() -> None:
    cfg = CudaLevelSetConfig(
        n_pairs=2,
        in_feat=5,
        hidden_dim=8,
        n_hidden=1,
        mod_dim=4,
        n_classes=5,
        render_h=6,
        render_w=8,
        camera_h=10,
        camera_w=12,
    )
    model = TorchLevelSetWitness.build(cfg, seed=31)
    geom = GroundHomographyGeom.eon(native_hw=(cfg.camera_h, cfg.camera_w), pitch=0.0)
    carrier = TorchPoseCarrier.build(np.zeros((2, 6), np.float32), geom)
    feats = torch.randn(cfg.render_h * cfg.render_w, cfg.in_feat)
    plain, _ = _generated_pose_pair_dispatch(model, feats, [0, 1], carrier, cfg)
    residual = torch.full((2, cfg.render_h, cfg.render_w, 3), 7.0)
    mask = torch.ones((2, cfg.render_h, cfg.render_w), dtype=torch.bool)
    seed = TorchProtectedIslandSeed(residual, mask, mode="shield", damp=0.1)
    composed, _, witness_alone = _generated_pose_pair_dispatch(
        model,
        feats,
        [0, 1],
        carrier,
        cfg,
        protected_seed=seed,
        return_witness_alone=True,
    )
    torch.testing.assert_close(witness_alone, plain)
    assert not torch.equal(composed[:, 1], plain[:, 1])
    torch.testing.assert_close(composed[:, 0], plain[:, 0])


def test_jacobian_probe_uses_product_cadence_and_motion_stratified_tails() -> None:
    flags = {
        "--jacobian-basin-telemetry": True,
        "--eval-every": 5,
        "--jacobian-basin-every": 4,
        "--jacobian-basin-stratify-t": True,
        "--jacobian-basin-k-pairs": 3,
    }
    gt_poses = np.zeros((5, 6), np.float32)
    gt_poses[:, 0] = np.arange(5, dtype=np.float32)
    assert _jacobian_probe_pair_indices(flags, gt_poses, 19) == []
    selected = _jacobian_probe_pair_indices(flags, gt_poses, 20)
    assert selected == [0, 2, 4]
    assert selected == _jacobian_probe_pair_indices(flags, gt_poses, 20)
    bad = dict(flags)
    bad["--jacobian-basin-stratify-t"] = False
    with pytest.raises(ValueError, match="motion stratification"):
        _jacobian_probe_pair_indices(bad, gt_poses, 20)


def test_polyak_candidate_blob_is_additional_cloned_resumable_state() -> None:
    source = np.array([[1.0, 2.0]], np.float32)

    class FakeController:
        polyak = SimpleNamespace(count=7)

        @staticmethod
        def polyak_candidate():
            return {"weight": source}

    blob = _polyak_checkpoint_blob(
        FakeController(), 12, "cfg", ("python", "trainer")
    )
    assert blob["schema"] == "v9_cgauge_torch_polyak_v1"
    assert blob["epoch"] == 12 and blob["count"] == 7
    source[...] = 99.0
    torch.testing.assert_close(
        blob["polyak"]["weight"], torch.tensor([[1.0, 2.0]])
    )


def test_main_loop_structurally_consumes_controller_lifecycle_and_dynamic_gates() -> None:
    tree = ast.parse(inspect.getsource(main))
    methods = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {
        "begin_epoch",
        "observe_scorer_chunk",
        "end_epoch",
        "observe_sigma_min",
        "observe_polyak",
        "state_dict",
        "load_state_dict",
    } <= methods
    source = inspect.getsource(main)
    assert "controller_step.lane_band_on" in source
    assert "controller_step.chroma_on" in source
    assert "controller_step.pose_finish_on" in source
    assert "island_targets.refresh_amplify_" in source
    assert "protected_seed.contain_grad_" in source
    assert "island_birth_perclass_from_signed_torch" in source
    assert "birth_scaled_logit_offsets" in source
    assert "build_torch_muon_adamw" in source
    assert "optimizer.set_epoch(epoch)" in source
    assert "resumed/fired Muon controller" in source
    assert "tail_controller.step" in source
    assert "tail_stop_after_epoch" in source
    assert "tail_controller=tail_controller" in source
    assert "adopt_compiled_training_region" in source
    assert "torch.set_autocast_enabled" in source
    assert "torch.amp.GradScaler" in source
    assert "cudagraph_mark_step_begin" in source
    assert 'telemetry_scope="epoch_final_chunk"' in source
    assert "lstars_device.index_select" in source
    assert "gt_f1_render_device.index_select" in source
    assert '"stage": "training_throughput_epoch"' in source
    assert '"score_claim": False' in source


def test_money_safety_fail_fast_order_and_runtime_receipts_are_structural() -> None:
    source = inspect.getsource(main)
    model_at = source.index("TorchLevelSetWitness.build")
    assert source.index("_resolve_resume_intent") < model_at
    assert source.index("_load_validated_gt_cache") < model_at
    assert source.index("_load_validated_resume") < model_at
    preflight_return_at = source.index("if args.preflight_only:")
    assert source.index("_validate_scorer_custody") < preflight_return_at
    assert source.index("cuda_v9_port_receipt") < preflight_return_at
    assert preflight_return_at < source.index("CurveletBankConfig")
    assert preflight_return_at < model_at
    assert preflight_return_at < source.index("torch.cuda.is_available")
    assert preflight_return_at < source.index("compile_identity_probe")
    assert preflight_return_at < source.index("_run_structured_prefit")
    assert preflight_return_at < source.index("_load_scorers")
    parity_refuse_at = source.index("CUDA/NumPy forward parity gate failed")
    compile_probe_at = source.index("compile_identity_probe")
    assert parity_refuse_at < compile_probe_at
    compile_refuse_at = source.index("compile probe is non-adoptable")
    runtime_scorer_load_at = source.index("_load_scorers", compile_refuse_at)
    assert compile_refuse_at < source.index("_run_structured_prefit")
    assert compile_refuse_at < runtime_scorer_load_at
    assert runtime_scorer_load_at < source.index("_run_structured_prefit")
    assert "run_end_epoch=run_end_epoch" in source
    assert "_canonical_checkpoint_due(" in source
    assert "runtime_stop_after_epochs=args.stop_after_epochs" in source
    throughput_at = source.index('"stage": "training_throughput_epoch"')
    canonical_save_at = source.index(
        "_atomic_torch_save(blob, out / TORCH_RESUME_PT)", throughput_at
    )
    ema_save_at = source.index("out / TORCH_EMA_PT", canonical_save_at)
    flush_at = source.index("_flush_trajectory_rows(out, pending_epoch_rows)", ema_save_at)
    assert throughput_at < canonical_save_at < ema_save_at < flush_at
    assert "_emit_trajectory_row(" not in source[throughput_at:canonical_save_at]
    assert '"runtime_epoch_budget_reached"' in source
    assert '"optimizer_updates_attempted"' in source
    assert '"optimizer_updates_successful"' in source
    assert '"productive_updates_per_second"' in source
    assert '"pointer_delta": "none"' in source
    assert '"pointer":' not in source
    assert "0.19108282" not in source
    assert '"scorer_custody": scorer_custody' in source
    assert '"scorer_sha256": scorer_sha256' in source
    assert "scorer_sha256=scorer_sha256" in source
    assert '"scorer_constructor_load"' in source
    assert '_load_scorers(torch.device("cpu"))' in source


def test_trainer_parser_covers_remote_shell_flags() -> None:
    parser_flags = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    remote_source = (
        Path(__file__).resolve().parents[2] / "scripts" / "remote_v9_cgauge_cuda.sh"
    ).read_text()
    invocation = remote_source[
        remote_source.index("ARGS=(") : remote_source.index("if ! command -v timeout")
    ]
    remote_flags = set(re.findall(r"--[a-z][a-z0-9-]*", invocation))
    assert remote_flags <= parser_flags
    assert "--stop-after-epochs" in parser_flags
    assert "--preflight-only" in parser_flags
    assert "--expected-segnet-sha256" in parser_flags
    assert "--expected-posenet-sha256" in parser_flags
