from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.self_compress import (
    SC_PROTECTED_NAME_PATTERNS,
    attribute_score_sensitivity_per_layer,
    get_protected_patterns,
    patterns_from_measured_sensitivity,
)
from tac.experiments.train_renderer import parse_args


REPO = Path(__file__).resolve().parents[3]
QAT_FINETUNE = REPO / "experiments" / "qat_finetune.py"
TRAIN_RENDERER = REPO / "src" / "tac" / "experiments" / "train_renderer.py"


def _argparse_flags(path: Path) -> set[str]:
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", path.read_text()))


def test_segnet_prior_patterns_exist() -> None:
    patterns = get_protected_patterns("segnet_prior")

    assert isinstance(patterns, list)
    assert patterns
    assert "out_conv" in patterns
    assert "decode_head" in patterns


def test_posenet_prior_unchanged() -> None:
    assert get_protected_patterns("posenet_prior") == list(SC_PROTECTED_NAME_PATTERNS)


def test_no_overlap_with_posenet_patterns() -> None:
    segnet = set(get_protected_patterns("segnet_prior"))
    posenet = set(get_protected_patterns("posenet_prior"))

    assert segnet.isdisjoint(posenet)


def test_invalid_pattern_set_raises() -> None:
    with pytest.raises(ValueError, match="Unknown pattern_set"):
        get_protected_patterns("invalid")


def test_patterns_dont_match_scorer_weights() -> None:
    scorer_keys = [
        "backbone.stem.conv.weight",
        "encoder.blocks.0.attn.qkv.weight",
        "posenet.head.weight",
    ]
    patterns = get_protected_patterns("segnet_prior")

    matches = [
        (pattern, key)
        for pattern in patterns
        for key in scorer_keys
        if key == pattern or key.endswith("." + pattern) or pattern in key
    ]
    assert matches == []


def test_train_renderer_argparse_has_protected_pattern_set() -> None:
    flags = _argparse_flags(TRAIN_RENDERER)
    assert "protected-pattern-set" in flags

    args = parse_args(["--tag", "sg_test", "--protected-pattern-set", "segnet_prior"])
    assert args.protected_pattern_set == "segnet_prior"


def test_qat_finetune_argparse_has_protected_pattern_set() -> None:
    flags = _argparse_flags(QAT_FINETUNE)
    assert "protected-pattern-set" in flags
    assert "choices=[\"posenet_prior\", \"segnet_prior\"]" in QAT_FINETUNE.read_text()


# ── Lane SG: measured-sensitivity helper tests ──────────────────────────


class _ToyRenderer(nn.Module):
    """Minimal renderer-shaped module for sensitivity tests.

    Two convs:
      - ``seg_conv``: heavy (16 channels) — drives the "segnet" feature.
      - ``pose_conv``: tiny (2 channels) — drives the "pose" feature.

    Forward returns (B, C_total, H, W); seg_score_fn looks at the seg
    half, pose_score_fn looks at the pose half. This lets the test verify
    the helper correctly attributes sensitivity to the right layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seg_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pose_conv = nn.Conv2d(3, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.seg_conv(x)
        p = self.pose_conv(x)
        return torch.cat([s, p], dim=1)


def _seg_score(out: torch.Tensor) -> torch.Tensor:
    # First 16 channels are "seg" — read their mean magnitude.
    return out[:, :16].abs().mean()


def _pose_score(out: torch.Tensor) -> torch.Tensor:
    # Last 2 channels are "pose".
    return out[:, 16:].abs().mean()


def test_sensitivity_helper_returns_per_layer_dict() -> None:
    torch.manual_seed(0)
    model = _ToyRenderer()
    samples = [(torch.randn(2, 3, 8, 8),)]

    sens = attribute_score_sensitivity_per_layer(
        model, _seg_score, _pose_score, samples,
        delta_frac=0.1, n_repeats=2, device="cpu", seed=42,
    )

    assert "seg_conv" in sens
    assert "pose_conv" in sens
    for name, vals in sens.items():
        assert "seg_dscore" in vals
        assert "pose_dscore" in vals
        assert "n_params" in vals
        assert "rate_per_param" in vals
        assert vals["seg_dscore"] >= 0.0
        assert vals["pose_dscore"] >= 0.0
        assert vals["n_params"] > 0


def test_sensitivity_helper_attributes_to_correct_layer() -> None:
    """Perturbing seg_conv must move seg_score MORE than pose_score."""
    torch.manual_seed(1)
    model = _ToyRenderer()
    samples = [(torch.randn(2, 3, 8, 8),) for _ in range(3)]

    sens = attribute_score_sensitivity_per_layer(
        model, _seg_score, _pose_score, samples,
        delta_frac=0.1, n_repeats=3, device="cpu", seed=7,
    )

    # seg_conv perturbation must move seg score more than pose score
    assert sens["seg_conv"]["seg_dscore"] > sens["seg_conv"]["pose_dscore"]
    # pose_conv perturbation must move pose score more than seg score
    assert sens["pose_conv"]["pose_dscore"] > sens["pose_conv"]["seg_dscore"]


def test_sensitivity_helper_restores_weights() -> None:
    """Helper must leave model state unchanged on return."""
    torch.manual_seed(2)
    model = _ToyRenderer()
    seg_w_before = model.seg_conv.weight.detach().clone()
    pose_w_before = model.pose_conv.weight.detach().clone()

    samples = [(torch.randn(2, 3, 8, 8),)]
    _ = attribute_score_sensitivity_per_layer(
        model, _seg_score, _pose_score, samples,
        delta_frac=0.05, n_repeats=2, device="cpu",
    )

    assert torch.allclose(model.seg_conv.weight, seg_w_before)
    assert torch.allclose(model.pose_conv.weight, pose_w_before)


def test_patterns_from_measured_sensitivity_segnet() -> None:
    sens = {
        "renderer.head":      {"seg_dscore": 0.10, "pose_dscore": 0.005, "n_params": 100, "rate_per_param": 0.01},
        "renderer.fuse_conv": {"seg_dscore": 0.05, "pose_dscore": 0.001, "n_params": 200, "rate_per_param": 0.005},
        "renderer.bottleneck.conv1": {"seg_dscore": 0.001, "pose_dscore": 0.05, "n_params": 1000, "rate_per_param": 0.001},
    }
    out = patterns_from_measured_sensitivity(sens, target="segnet", top_k=2)
    assert out == ["renderer.head", "renderer.fuse_conv"]


def test_patterns_from_measured_sensitivity_posenet() -> None:
    sens = {
        "renderer.head":      {"seg_dscore": 0.10, "pose_dscore": 0.005, "n_params": 100, "rate_per_param": 0.01},
        "motion.head":        {"seg_dscore": 0.001, "pose_dscore": 0.20, "n_params": 50, "rate_per_param": 0.02},
    }
    out = patterns_from_measured_sensitivity(sens, target="posenet", top_k=1)
    assert out == ["motion.head"]


def test_patterns_from_measured_sensitivity_invalid_target() -> None:
    with pytest.raises(ValueError, match="Unknown target"):
        patterns_from_measured_sensitivity({}, target="bogus")


def test_patterns_from_measured_sensitivity_min_dscore_floor() -> None:
    sens = {
        "a": {"seg_dscore": 0.01, "pose_dscore": 0.0, "n_params": 1, "rate_per_param": 1.0},
        "b": {"seg_dscore": 0.001, "pose_dscore": 0.0, "n_params": 1, "rate_per_param": 1.0},
    }
    out = patterns_from_measured_sensitivity(sens, target="segnet", top_k=10, min_dscore=0.005)
    assert out == ["a"]


def test_sensitivity_helper_skip_protected() -> None:
    """skip_protected=True omits anything matching the heuristic patterns."""
    torch.manual_seed(3)

    class _RealNamedRenderer(nn.Module):
        # Use a real protected pattern from SC_PROTECTED_NAME_PATTERNS
        # (renderer.head) so skip_protected has something to filter.
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Conv2d(3, 4, 3, padding=1)
            # build the head as a submodule whose qualified name ends in "renderer.head"
            self.renderer = nn.Module()
            self.renderer.head = nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.body(x) + self.renderer.head(x)

    model = _RealNamedRenderer()
    samples = [(torch.randn(1, 3, 8, 8),)]
    sens = attribute_score_sensitivity_per_layer(
        model, _seg_score, _pose_score, samples,
        n_repeats=1, device="cpu", skip_protected=True,
    )
    # body is unprotected; renderer.head matches SC_PROTECTED_NAME_PATTERNS
    assert "body" in sens
    assert "renderer.head" not in sens
