# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def test_z7_score_aware_loss_loads_scorers_from_canonical_scorer_module(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import experiments.train_substrate_time_traveler_l5_z7_mamba2 as trainer
    import tac.differentiable_eval_roundtrip as roundtrip
    import tac.scorer as scorer

    calls: list[tuple[str, object]] = []

    class TinyScorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean() * self.weight

    def fake_patch() -> dict[str, object]:
        calls.append(("patch_upstream_yuv6_globally", None))
        return {}

    def fake_load_differentiable_scorers(
        upstream_dir: str | Path,
        device: str | torch.device | None = None,
    ) -> tuple[TinyScorer, TinyScorer]:
        calls.append(("load_differentiable_scorers", (str(upstream_dir), str(device))))
        return TinyScorer(), TinyScorer()

    def forbidden_roundtrip_loader(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError("stale tac.differentiable_eval_roundtrip loader used")

    monkeypatch.setattr(roundtrip, "patch_upstream_yuv6_globally", fake_patch)
    monkeypatch.setattr(scorer, "load_differentiable_scorers", fake_load_differentiable_scorers)
    monkeypatch.setattr(
        roundtrip,
        "load_differentiable_scorers",
        forbidden_roundtrip_loader,
        raising=False,
    )

    loss = trainer._build_score_aware_loss(
        args=argparse.Namespace(
            upstream_dir=tmp_path,
            alpha_rate=0.01,
            beta_seg=1.0,
            beta_ib=0.0,
        ),
        device=torch.device("cpu"),
    )

    assert calls == [
        ("patch_upstream_yuv6_globally", None),
        ("load_differentiable_scorers", (str(tmp_path), "cpu")),
    ]
    loss.train()
    assert loss.training is True
    assert loss.seg_scorer.training is False
    assert loss.pose_scorer.training is False
    assert all(not p.requires_grad for p in loss.seg_scorer.parameters())
    assert all(not p.requires_grad for p in loss.pose_scorer.parameters())
