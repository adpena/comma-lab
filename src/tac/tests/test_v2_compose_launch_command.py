# SPDX-License-Identifier: MIT
"""Tests for tac.v2_compose.launch_command — flag-validated residual-INR launch (never invent flags)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.v2_compose.launch_command import (
    DEFAULT_TRAINER,
    PERF_ENV,
    build_residual_inr_command,
    parse_trainer_flags,
)

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / DEFAULT_TRAINER


@pytest.mark.skipif(not _TRAINER.exists(), reason="trainer file absent")
def test_parse_trainer_flags_finds_real_flags():
    flags, bool_opt = parse_trainer_flags(DEFAULT_TRAINER)
    # spot-check known-real flags from the trainer argparse
    for f in ("--out-dir", "--num-pairs", "--epochs", "--seed", "--gt-cache",
              "--structured-init", "--lane-prior-phi1", "--curriculum", "--ema-decay",
              "--hidden-dim", "--mod-dim", "--stage-checkpoints", "--mlx-device"):
        assert f in flags, f
    # BooleanOptionalAction auto-negations exist
    assert "--no-structured-init" in flags
    assert "--structured-init" in bool_opt


@pytest.mark.skipif(not _TRAINER.exists(), reason="trainer file absent")
def test_build_command_all_flags_valid_and_perf_env():
    cmd = build_residual_inr_command(
        out_dir="experiments/results/residual_run",
        gt_cache="experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        num_pairs=600,
    )
    assert cmd.all_flags_valid is True
    assert cmd.unknown_flags == ()
    # perf-env prefix present (launch-gate discipline)
    assert "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1" in cmd.command
    assert PERF_ENV["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] == "1"
    # the trainer + key flags appear
    assert DEFAULT_TRAINER in cmd.command
    assert "--structured-init" in cmd.command
    assert "--lane-prior-phi1" in cmd.command
    assert "--stage-checkpoints" in cmd.command  # resumability non-negotiable
    # HOLD semantics
    j = cmd.to_json()
    assert j["hold_for_operator_go"] is True
    assert j["auto_launched"] is False
    # honest S3 gap surfaced
    assert "NEEDS-WIRING" in cmd.missing_capability_note


@pytest.mark.skipif(not _TRAINER.exists(), reason="trainer file absent")
def test_invented_flag_raises_strict():
    with pytest.raises(ValueError):
        build_residual_inr_command(
            out_dir="x",
            gt_cache="y",
            extra_flags={"--this-flag-does-not-exist": 3},
            strict=True,
        )


@pytest.mark.skipif(not _TRAINER.exists(), reason="trainer file absent")
def test_invented_flag_recorded_when_not_strict():
    cmd = build_residual_inr_command(
        out_dir="x", gt_cache="y",
        extra_flags={"--this-flag-does-not-exist": 3}, strict=False,
    )
    assert cmd.all_flags_valid is False
    assert "--this-flag-does-not-exist" in cmd.unknown_flags


@pytest.mark.skipif(not _TRAINER.exists(), reason="trainer file absent")
def test_residual_inr_sized_smaller_than_full():
    """The residual INR is sized BELOW the full-partition 96/32 (the rate-win lever)."""
    cmd = build_residual_inr_command(out_dir="x", gt_cache="y", hidden_dim=48, mod_dim=16)
    assert "--hidden-dim 48" in cmd.command
    assert "--mod-dim 16" in cmd.command
