from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.train_levelset_witness_realized_through_R_torch import (
    _generated_pose_pair_dispatch,
    _jacobian_probe_pair_indices,
    _polyak_checkpoint_blob,
    main,
)
from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom
from tac.cuda_levelset_training import (
    CudaLevelSetConfig,
    TorchLevelSetWitness,
    TorchPoseCarrier,
)


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
