# SPDX-License-Identifier: MIT
"""Focused tests for the canonical optimized trainer context (F3)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tac.substrates._shared import trainer_skeleton as ts


class _SegScorer(torch.nn.Module):
    def preprocess_input(self, pair: torch.Tensor) -> torch.Tensor:
        return pair[:, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x.mean(dim=1, keepdim=True)
        return torch.cat([base + float(i) for i in range(5)], dim=1)


class _PoseScorer(torch.nn.Module):
    def preprocess_input(self, pair: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = pair.shape
        pooled = pair.reshape(b, t, c, h * w).mean(dim=3)
        return torch.cat([pooled, pooled, pooled, pooled], dim=2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pose": x}


def _patch_scorer_forward_pair(monkeypatch) -> None:
    def fake_scorer_forward_pair(pair_btchw, posenet, segnet):
        pose = posenet(posenet.preprocess_input(pair_btchw))
        seg = segnet(segnet.preprocess_input(pair_btchw))
        return pose, seg

    monkeypatch.setattr(
        "tac.training_optimization.scorer_cache._resolve_scorer_forward_pair",
        lambda: fake_scorer_forward_pair,
    )


def test_merge_optimization_flags_adds_canonical_gt_cache_flag() -> None:
    merged = ts.merge_optimization_flags({"--epochs": {"default": "10"}})
    assert "--enable-gt-scorer-cache" in merged
    assert merged["--enable-gt-scorer-cache"]["default"] is True
    assert merged["--epochs"]["default"] == "10"


def test_build_optimized_training_context_builds_cache_and_preserves_axis_labels(
    monkeypatch,
) -> None:
    _patch_scorer_forward_pair(monkeypatch)
    args = SimpleNamespace(
        enable_gt_scorer_cache=True,
        enable_torch_compile=False,
        enable_autocast_fp16=False,
        segmentation_temperature=1.0,
        gt_scorer_cache_chunk_size=2,
    )
    target_pairs = torch.rand(3, 2, 3, 4, 5)
    model = torch.nn.Conv2d(3, 3, kernel_size=1)
    ctx = ts.build_optimized_training_context(
        args,
        scorers=(_PoseScorer(), _SegScorer()),
        gt_pairs=target_pairs,
        substrate_model=model,
        device=torch.device("cpu"),
    )

    assert ctx.gt_cache is not None
    assert ctx.gt_cache.n_pairs == 3
    assert ctx.substrate_model is model
    assert ctx.autocast_cfg.enabled is False
    assert ctx.eval_axis_label == "[trainer-proxy; not contest-CPU or contest-CUDA]"
    assert "[contest-CUDA]" in ctx.promotion_requirement


def test_build_optimized_training_context_can_disable_cache(monkeypatch) -> None:
    _patch_scorer_forward_pair(monkeypatch)
    args = SimpleNamespace(
        enable_gt_scorer_cache=False,
        enable_torch_compile=False,
        enable_autocast_fp16=False,
    )
    ctx = ts.build_optimized_training_context(
        args,
        scorers={"posenet": _PoseScorer(), "segnet": _SegScorer()},
        gt_pairs=torch.rand(2, 2, 3, 4, 5),
        substrate_model=torch.nn.Identity(),
        device="cpu",
    )
    assert ctx.gt_cache is None
    assert ctx.eval_axis_label.startswith("[trainer-proxy")
