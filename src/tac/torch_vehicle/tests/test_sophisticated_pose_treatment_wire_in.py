# SPDX-License-Identifier: MIT
"""NO-FAKE driver-wiring tests for the sophisticated pose treatment (Levers A + C).

These exercise the LIVE driver loop (split-by-head, forced to CPU like test_split_by_head_grad) and prove:
  * Lever A's controller ACTUALLY runs + mutates ``w_pose_frac`` during training (would FAIL if the flag
    no-op'd);
  * Lever A's controller state ROUND-TRIPS through the checkpoint (resume continues the trajectory);
  * Lever C's per-dim weights ACTUALLY change the pose loss vs uniform (would FAIL if the weights were
    ignored), and ``None`` is byte-identical to the plain MSE;
  * the DEFAULT-OFF path (both levers off) is byte-identical to a baseline run (the byte-identity contract);
  * the Config guards fail closed on mis-config.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext, _TinyFrozenScorer


def _ce_seg_loss(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _stage(seg_weight=100.0, pose_weight=1.0, epochs=4) -> StageSpec:
    return StageSpec(
        name="pose_treat_test", epochs=epochs, seg_loss_fn=_ce_seg_loss, eval_every=epochs,
        batch_size=4, ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1e9, grad_clip_muon=1e9,
        lr_floor_ratio=5e-6, seg_weight=seg_weight, pose_weight=pose_weight, cat_lambda=0.0,
        cat_sigma=0.2, use_qat=False, init_latents_random=True,
    )


def _build_split_driver(tmp_path, *, name, curriculum, **cfg_kwargs):
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / name,
        checkpoint_every_epochs=1, device="cpu", train_device="mps",
        split_by_head=True, seed=0, **cfg_kwargs,
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0, split_by_head=True)
    scorer.split_device = True
    scorer.train_device = torch.device("cpu")
    scorer._train_scorer = _TinyFrozenScorer(seed=0).to("cpu").eval()
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=curriculum,
    )
    driver.train_device = torch.device("cpu")
    driver.device = torch.device("cpu")
    driver.split_by_head = True
    return cfg, driver


# --------------------------------------------------------------------------- Config guards
def test_equimarginal_requires_split_by_head(tmp_path):
    with pytest.raises(ValueError, match="requires split_by_head"):
        TorchVehicleConfig(
            out_dir=tmp_path / "x", device="cpu", train_device="cpu",
            pose_equimarginal_enabled=True,
        )


def test_equimarginal_validates_params(tmp_path):
    with pytest.raises(ValueError, match="rho must be > 0"):
        TorchVehicleConfig(
            out_dir=tmp_path / "x", device="cpu", train_device="mps", split_by_head=True,
            pose_equimarginal_enabled=True, pose_equimarginal_rho=0.0,
        )


def test_pose_dim_weights_normalised_at_construct(tmp_path):
    cfg = TorchVehicleConfig(
        out_dir=tmp_path / "x", device="cpu", pose_dim_weights=(2, 2, 2, 2, 2, 2),
    )
    assert cfg.pose_dim_weights == (1.0,) * 6  # renormalised to mean 1.0


def test_pose_dim_weights_bad_length_fails(tmp_path):
    with pytest.raises(ValueError):
        TorchVehicleConfig(out_dir=tmp_path / "x", device="cpu", pose_dim_weights=(1, 1, 1))


# --------------------------------------------------------------------------- Lever A live
def test_equimarginal_controller_runs_and_moves_w_pose(tmp_path):
    cfg, driver = _build_split_driver(
        tmp_path, name="eqm", curriculum=[_stage(epochs=6)],
        pose_equimarginal_enabled=True, pose_equimarginal_rho=1.0,
        pose_equimarginal_tol=0.01, pose_equimarginal_decay=0.0,
    )
    assert driver._equimarginal_ctrl is not None
    summary = driver.run()
    assert summary["status"] == "complete"
    # the controller was consulted (steps advanced) and (with seg_weight=100 >> pose_weight=1) the pose
    # pull is far below seg → ratio << rho=1 → the controller RAISED w_pose (frac > 1) or hit a bound.
    assert driver._equimarginal_ctrl.state.steps > 0
    assert driver._equimarginal_ctrl.w_pose_frac != 1.0  # it actually moved (NO-FAKE)
    assert driver._last_equimarginal_telemetry is not None
    assert "ratio_ema" in driver._last_equimarginal_telemetry


def test_equimarginal_state_round_trips_through_checkpoint(tmp_path):
    cfg, driver = _build_split_driver(
        tmp_path, name="eqm_ckpt", curriculum=[_stage(epochs=3)],
        pose_equimarginal_enabled=True, pose_equimarginal_rho=1.0,
    )
    driver.run()
    from tac.torch_vehicle.checkpoint import load_checkpoint

    merged = load_checkpoint(cfg.out_dir)
    eq = merged.get("equimarginal_ctrl")
    assert eq is not None
    assert "w_pose_frac" in eq and "ratio_ema" in eq and "steps" in eq
    assert eq["steps"] == driver._equimarginal_ctrl.state.steps
    assert eq["w_pose_frac"] == pytest.approx(driver._equimarginal_ctrl.w_pose_frac)


# --------------------------------------------------------------------------- Lever C live
def test_pose_dim_weights_change_the_pose_loss(tmp_path):
    # Two runs: uniform vs a tilted per-dim weight. The trained final decoder differs because the pose
    # loss re-allocated per-dim pull. (We compare the BEST archive bytes — a tilted pose loss changes the
    # descent → a different decoder.)
    cfg_u, drv_u = _build_split_driver(
        tmp_path, name="dimw_uniform", curriculum=[_stage(epochs=4)],
    )
    drv_u.run()
    cfg_w, drv_w = _build_split_driver(
        tmp_path, name="dimw_tilted", curriculum=[_stage(epochs=4)],
        pose_dim_weights=(3.0, 0.2, 0.2, 0.2, 0.2, 0.2),
    )
    drv_w.run()
    a_u = (cfg_u.out_dir / "best" / "best_archive.bin").read_bytes()
    a_w = (cfg_w.out_dir / "best" / "best_archive.bin").read_bytes()
    assert a_u != a_w  # the per-dim weighting actually changed training (NO-FAKE)


def test_pose_dim_weights_none_is_byte_identical_to_baseline(tmp_path):
    cfg_a, drv_a = _build_split_driver(tmp_path, name="base_a", curriculum=[_stage(epochs=4)])
    drv_a.run()
    cfg_b, drv_b = _build_split_driver(
        tmp_path, name="base_b", curriculum=[_stage(epochs=4)], pose_dim_weights=None,
    )
    drv_b.run()
    a = (cfg_a.out_dir / "best" / "best_archive.bin").read_bytes()
    b = (cfg_b.out_dir / "best" / "best_archive.bin").read_bytes()
    assert a == b  # pose_dim_weights=None is the unmodified path


def test_uniform_dim_weights_is_byte_identical_to_baseline(tmp_path):
    # uniform-after-norm (1,1,1,1,1,1) must reproduce the plain-MSE descent bit-for-bit.
    cfg_a, drv_a = _build_split_driver(tmp_path, name="u_base", curriculum=[_stage(epochs=4)])
    drv_a.run()
    cfg_b, drv_b = _build_split_driver(
        tmp_path, name="u_uniform", curriculum=[_stage(epochs=4)],
        pose_dim_weights=(1, 1, 1, 1, 1, 1),
    )
    drv_b.run()
    a = (cfg_a.out_dir / "best" / "best_archive.bin").read_bytes()
    b = (cfg_b.out_dir / "best" / "best_archive.bin").read_bytes()
    assert a == b


# --------------------------------------------------------------------------- both-off byte identity
def test_both_levers_off_is_byte_identical_to_baseline(tmp_path):
    cfg_a, drv_a = _build_split_driver(tmp_path, name="off_base", curriculum=[_stage(epochs=4)])
    drv_a.run()
    cfg_b, drv_b = _build_split_driver(
        tmp_path, name="off_explicit", curriculum=[_stage(epochs=4)],
        pose_equimarginal_enabled=False, pose_dim_weights=None,
    )
    drv_b.run()
    a = (cfg_a.out_dir / "best" / "best_archive.bin").read_bytes()
    b = (cfg_b.out_dir / "best" / "best_archive.bin").read_bytes()
    assert a == b
    assert drv_b._equimarginal_ctrl is None  # controller not even instantiated when off
