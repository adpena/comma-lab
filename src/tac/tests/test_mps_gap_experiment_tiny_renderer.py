# SPDX-License-Identifier: MIT
"""Dedicated tests for the MPS-train CUDA-score gap-experiment infrastructure.

Catalog cross-refs: #229 (premise verification before edit),
#192 (macOS-CPU advisory non-promotion), #317 (local research-signal
evidence stamping), #205 (canonical inflate device selector).

NOT a contest substrate suite — purely diagnostic infrastructure tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tac.mps_gap_experiment.harvest_and_verdict import (
    GapManifest,
    classify_verdict,
    compute_gap_components,
)
from tac.mps_gap_experiment.tiny_renderer import (
    TinyRenderer,
    TinyRendererConfig,
    build_tiny_renderer,
    count_params,
)


def test_tiny_renderer_forward_shape() -> None:
    """Forward produces (B, 2, 3, H, W) shape."""
    model = build_tiny_renderer(seed=42)
    pair = torch.zeros(1, 2, 3, 384, 512)
    pose = torch.zeros(1, 12)
    out = model(pair, pose)
    assert out.shape == (1, 2, 3, 384, 512)


def test_tiny_renderer_forward_batch_2() -> None:
    """Forward handles batch=2 shape correctly."""
    model = build_tiny_renderer(seed=42)
    pair = torch.zeros(2, 2, 3, 384, 512)
    pose = torch.zeros(2, 12)
    out = model(pair, pose)
    assert out.shape == (2, 2, 3, 384, 512)


def test_tiny_renderer_backward_produces_grads() -> None:
    """Backward pass produces non-zero grads on every parameter."""
    model = build_tiny_renderer(seed=42)
    pair = torch.randn(1, 2, 3, 384, 512)
    pose = torch.randn(1, 12)
    out = model(pair, pose)
    loss = out.abs().mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing grad on {name}"
        assert p.grad.abs().sum() > 0.0, f"zero grad on {name}"


def test_tiny_renderer_param_count_under_15k() -> None:
    """Default config keeps total params under 15K (Phase 1 budget)."""
    model = build_tiny_renderer()
    total = count_params(model)
    assert total < 15_000, f"params {total} exceeds 15K budget"
    # Sanity floor: we expect roughly 11-13K
    assert total > 5_000, f"params {total} suspiciously low; check architecture"


def test_tiny_renderer_deterministic_with_seed() -> None:
    """Two builds with the same seed produce identical weights."""
    a = build_tiny_renderer(seed=123)
    b = build_tiny_renderer(seed=123)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), f"mismatch on {na}"


def test_tiny_renderer_cpu_forward_ok() -> None:
    """Forward on CPU produces finite outputs (smoke; not numeric check)."""
    model = build_tiny_renderer(seed=42).to("cpu")
    pair = torch.zeros(1, 2, 3, 384, 512)
    pose = torch.zeros(1, 12)
    out = model(pair, pose)
    assert torch.isfinite(out).all()


def test_tiny_renderer_rejects_wrong_input_shape() -> None:
    """Forward refuses a (B, 1, 3, H, W) pair (must be 2-frame pair)."""
    model = build_tiny_renderer(seed=42)
    pair = torch.zeros(1, 1, 3, 384, 512)
    pose = torch.zeros(1, 12)
    with pytest.raises(ValueError, match="frame_pair must be"):
        model(pair, pose)


def test_tiny_renderer_rejects_wrong_pose_shape() -> None:
    """Forward refuses a (B, 6) pose vector (must be 12-dim)."""
    model = build_tiny_renderer(seed=42)
    pair = torch.zeros(1, 2, 3, 384, 512)
    pose = torch.zeros(1, 6)
    with pytest.raises(ValueError, match="pose must be"):
        model(pair, pose)


def test_classify_verdict_thresholds() -> None:
    """Verdict mapping matches the canonical thresholds."""
    assert classify_verdict(0.0) == "LOCAL_MPS_TRAIN_VIABLE"
    assert classify_verdict(0.04) == "LOCAL_MPS_TRAIN_VIABLE"
    assert classify_verdict(0.05) == "LOCAL_MPS_TRAIN_VIABLE_ADVISORY_ONLY"
    assert classify_verdict(0.10) == "LOCAL_MPS_TRAIN_VIABLE_ADVISORY_ONLY"
    assert classify_verdict(0.20) == "LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX"
    assert classify_verdict(0.50) == "LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX"


def test_compute_gap_components_cpu_self_comparison_returns_zero_gap(
    tmp_path: Path,
) -> None:
    """When mps_reference_device == target_device, gap MUST be 0.0.

    This is the meta-validation: the gap-computation primitive itself is
    self-consistent. A real MPS-vs-CUDA comparison happens on the Modal
    A10G dispatch (NOT in this local test).
    """
    model = build_tiny_renderer(seed=42)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "checkpoint_ema.pt"
    torch.save(state, ckpt_path)
    cache = torch.zeros(2, 2, 3, 384, 512)
    cache_path = tmp_path / "frame_cache.pt"
    torch.save(cache, cache_path)
    out_path = tmp_path / "gap_results.json"

    manifest = compute_gap_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_path=out_path,
        target_device="cpu",
        mps_reference_device="cpu",
        include_scorer_components=False,
    )
    assert manifest.verdict == "LOCAL_MPS_TRAIN_VIABLE"
    assert manifest.gap_relative_aggregate == 0.0
    assert manifest.evidence_grade == "MPS-research-signal"
    assert manifest.score_claim is False
    assert manifest.promotion_eligible is False
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["verdict"] == "LOCAL_MPS_TRAIN_VIABLE"
    assert data["num_pairs"] == 2
