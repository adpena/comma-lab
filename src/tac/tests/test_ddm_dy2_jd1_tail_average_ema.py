"""ddm_dy2 — JD1 plateau-tail EMA mode guards and pure update law."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
WORKTREE = _HERE.parents[3]
for _p in (str(WORKTREE), str(WORKTREE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.train_tr1_partition_renderer_mlx import (  # noqa: E402
    build_argparser,
    jd1_ema_checkpoint_payload,
    jd1_ema_gate_basis_label,
    jd1_ema_initial_state,
    jd1_ema_tail_average_active,
    jd1_ema_tail_average_live_weight,
    no_opt_state,
    refuse_declared_vs_resolved_jd1_ema_decay,
    save_checkpoint,
    TR1Config,
    validate_jd1_pose_finish_args,
)


def _args(*extra: str):
    return build_argparser().parse_args([
        "--variant", "plain",
        "--num-pairs", "4",
        "--epochs", "2",
        "--out-dir", "/tmp/ddm_dy2_test",
        *extra,
    ])


def _armed_args(*extra: str):
    return _args(
        "--jd1-pose-finish-mode", "joint_loss",
        "--jd1-w-pose", "1.0",
        *extra,
    )


def _cfg() -> TR1Config:
    return TR1Config(
        variant="plain",
        num_pairs=1,
        grid_downsample=16,
        code_width=2,
        renderer_width=8,
        token_quant_levels=16,
        seed=1,
        lotto_seed=2,
        lotto_mask_density_init=0.5,
        seg_form_start="ce",
        w_seg=100.0,
        lr=1e-3,
        batch_pairs=1,
        epochs=1,
        gate_every=1,
        ema_decay=0.9,
        ema_decay_provenance="test",
        token_temporal_mode="shared_base",
        token_ste="round",
        class_weight_lane=1.0,
        margin_target=1.0,
    )


class _FakeCheckpointModel:
    def trainable_parameters(self):
        return {"w": np.array([1.0, 2.0], dtype=np.float32)}


def test_default_geometric_mode_keeps_checkpoint_payload_absent():
    args = _args()
    validate_jd1_pose_finish_args(args)
    assert args.jd1_ema_mode == "geometric"
    assert args.jd1_ema_tail_anchor_epoch == -1
    assert jd1_ema_initial_state(args) == {}
    assert jd1_ema_checkpoint_payload(args, {}) == {}


def test_tail_mode_is_refused_when_jd1_is_off():
    args = _args("--jd1-ema-mode", "plateau_tail_average")
    with pytest.raises(SystemExit, match="JD1 value flags"):
        validate_jd1_pose_finish_args(args)


def test_tail_anchor_is_refused_when_jd1_is_off():
    args = _args("--jd1-ema-tail-anchor-epoch", "12")
    with pytest.raises(SystemExit, match="JD1 value flags"):
        validate_jd1_pose_finish_args(args)


def test_geometric_mode_rejects_tail_anchor():
    args = _armed_args("--jd1-ema-tail-anchor-epoch", "12")
    with pytest.raises(SystemExit, match="requires --jd1-ema-mode plateau_tail_average"):
        validate_jd1_pose_finish_args(args)


def test_tail_mode_requires_window_scoped_ema():
    args = _armed_args(
        "--jd1-ema-mode", "plateau_tail_average",
        "--jd1-ema-tail-anchor-epoch", "12",
    )
    with pytest.raises(SystemExit, match="requires --jd1-ema-stage-scope window"):
        validate_jd1_pose_finish_args(args)


def test_tail_mode_requires_explicit_anchor_epoch():
    args = _armed_args(
        "--jd1-ema-stage-scope", "window",
        "--jd1-ema-mode", "plateau_tail_average",
    )
    with pytest.raises(SystemExit, match="requires --jd1-ema-tail-anchor-epoch"):
        validate_jd1_pose_finish_args(args)


def test_tail_average_live_weight_law_includes_anchor_sample():
    assert jd1_ema_tail_average_live_weight(0) == pytest.approx(0.5)
    assert jd1_ema_tail_average_live_weight(1) == pytest.approx(1.0 / 3.0)
    assert jd1_ema_tail_average_live_weight(8) == pytest.approx(0.1)
    with pytest.raises(ValueError, match=">= 0"):
        jd1_ema_tail_average_live_weight(-1)


def test_tail_average_closed_form_matches_running_mean():
    ema = 0.0
    live_values = [2.0, 4.0, 6.0]
    for k, live in enumerate(live_values):
        w = jd1_ema_tail_average_live_weight(k)
        ema = ema + w * (live - ema)
    assert ema == pytest.approx(sum([0.0, *live_values]) / 4.0)


def test_literal_ema_decay_must_match_scope_law_resolution():
    refuse_declared_vs_resolved_jd1_ema_decay(None, 0.996, resolution_hash="abc")
    refuse_declared_vs_resolved_jd1_ema_decay(0.996, 0.996, resolution_hash="abc")
    with pytest.raises(SystemExit, match="conflicts with the JD1 stage scope-law"):
        refuse_declared_vs_resolved_jd1_ema_decay(0.997, 0.996, resolution_hash="abc")


def test_tail_initial_state_and_checkpoint_payload_are_resumable():
    args = _armed_args(
        "--jd1-ema-stage-scope", "window",
        "--jd1-ema-mode", "plateau_tail_average",
        "--jd1-ema-tail-anchor-epoch", "1424",
    )
    validate_jd1_pose_finish_args(args)
    state = jd1_ema_initial_state(args)
    assert state["ema_mode"] == "plateau_tail_average"
    assert state["ema_tail_anchor_epoch"] == 1424
    assert state["ema_tail_average_active"] is False
    assert state["ema_tail_update_count"] == 0
    state.update({
        "ema_tail_average_active": True,
        "ema_tail_update_count": 7,
        "ema_tail_anchor_epoch": 1426,
        "ema_tail_anchor_reason": "explicit_epoch",
    })
    payload = jd1_ema_checkpoint_payload(args, state)
    assert payload["ema_mode"] == "plateau_tail_average"
    assert payload["ema_tail_configured_anchor_epoch"] == 1424
    assert payload["ema_tail_anchor_epoch"] == 1426
    assert payload["ema_tail_average_active"] is True
    assert payload["ema_tail_update_count"] == 7
    assert jd1_ema_tail_average_active(payload) is True


def test_tail_state_persists_through_real_checkpoint_meta_writer(tmp_path):
    args = _armed_args(
        "--jd1-ema-stage-scope", "window",
        "--jd1-ema-mode", "plateau_tail_average",
        "--jd1-ema-tail-anchor-epoch", "1424",
    )
    state = jd1_ema_initial_state(args)
    state.update({
        "engaged": True,
        "ema_tail_average_active": True,
        "ema_tail_update_count": 9,
        "ema_tail_anchor_epoch": 1425,
        "ema_tail_anchor_global_step": 123,
        "ema_tail_anchor_reason": "explicit_epoch",
        "ema_tail_last_live_weight": 0.1,
    })
    payload = jd1_ema_checkpoint_payload(args, state)
    path = tmp_path / "dy2_tail_checkpoint.npz"
    save_checkpoint(
        path,
        model=_FakeCheckpointModel(),
        ema={"w": np.array([1.5, 2.5], dtype=np.float32)},
        opt_state_flat=no_opt_state(
            "test checkpoint exercises JD1 metadata only and is never resumed"
        ),
        epoch=12,
        stage="joint_pose_finish",
        cfg=_cfg(),
        telemetry_tail=[],
        extra_meta={"jd1_pose_finish": payload},
    )
    with np.load(path, allow_pickle=False) as zf:
        meta = json.loads(bytes(zf["meta::json"]).decode())
    assert meta["jd1_pose_finish"] == payload


def test_gate_basis_selector_tracks_live_geometric_and_tail_average():
    assert jd1_ema_gate_basis_label(
        global_step=4,
        ema_warmup_updates=5,
        state={},
    ) == "live_ema_warmup"
    assert jd1_ema_gate_basis_label(
        global_step=5,
        ema_warmup_updates=5,
        state={},
    ) == "ema_shadow"
    assert jd1_ema_gate_basis_label(
        global_step=0,
        ema_warmup_updates=5,
        state={"ema_mode": "plateau_tail_average", "ema_tail_average_active": True},
    ) == "ema_tail_average"
