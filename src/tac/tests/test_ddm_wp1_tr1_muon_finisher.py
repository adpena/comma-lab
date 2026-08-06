# SPDX-License-Identifier: MIT
"""ddm_wp1 TR1 JD1 Muon finisher pure wiring tests."""

from __future__ import annotations

import pytest

from experiments.train_tr1_partition_renderer_mlx import (
    RESET_ADAM_BETAS,
    TR1Config,
    build_argparser,
    derive_jd1_muon_momentum,
    opt_state_param_path,
    tr1_muon_adam_split_counts,
    tr1_muon_finisher_param_filter,
    validate_jd1_pose_finish_args,
)


class _Leaf:
    def __init__(self, ndim: int):
        self.ndim = ndim


def test_tr1_muon_filter_routes_renderer_matrices_only():
    for name in ("w_conv0", "w_up0", "w_head", "s_conv0", "s_up2", "s_head"):
        assert tr1_muon_finisher_param_filter(name, _Leaf(4)) is True

    for name, ndim in (
        ("tokens_base", 4),
        ("tokens_delta", 4),
        ("b_conv0", 1),
        ("g_conv0", 1),
        ("pe3_conditioning_gate", 1),
        ("head_relax_gain", 1),
        ("some_other_matrix", 2),
        ("w_conv0", 1),
    ):
        assert tr1_muon_finisher_param_filter(name, _Leaf(ndim)) is False


def test_tr1_muon_split_counts_lotto_tree_without_mlx_import():
    params = {
        "tokens_base": _Leaf(3),
        "tokens_delta": _Leaf(4),
        "s_conv0": _Leaf(4),
        "s_up0": _Leaf(4),
        "s_head": _Leaf(4),
        "g_conv0": _Leaf(1),
        "g_up0": _Leaf(1),
        "g_head": _Leaf(1),
        "b_conv0": _Leaf(1),
        "b_up0": _Leaf(1),
        "b_head": _Leaf(1),
    }
    assert tr1_muon_adam_split_counts(params) == (3, 8)


def test_multioptimizer_state_keys_map_back_to_model_params():
    assert opt_state_param_path("states.0.w_conv0.v") == "w_conv0"
    assert opt_state_param_path("states.1.tokens_delta.m") == "tokens_delta"
    assert opt_state_param_path("states.1.tokens_delta.v") == "tokens_delta"
    assert opt_state_param_path("states.0.step") is None
    assert opt_state_param_path("states.1.learning_rate") is None


def test_muon_momentum_derives_from_tr1_adam_beta1():
    momentum, source = derive_jd1_muon_momentum(RESET_ADAM_BETAS[0])
    assert momentum == pytest.approx(0.9)
    assert "TR1 optimizer beta1" in source


def test_jd1_finisher_default_off_is_args_only_not_config_identity():
    ns = build_argparser().parse_args(["--variant", "plain", "--out-dir", "/unused"])
    validate_jd1_pose_finish_args(ns)
    assert ns.jd1_finisher == "off"
    assert "jd1_finisher" not in TR1Config.__dataclass_fields__


def test_jd1_finisher_refuses_when_pose_finish_is_off():
    ns = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", "/unused",
        "--jd1-finisher", "muon",
    ])
    with pytest.raises(SystemExit, match="JD1 value flags"):
        validate_jd1_pose_finish_args(ns)
