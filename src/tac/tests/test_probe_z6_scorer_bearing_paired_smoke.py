# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "probe_z6_scorer_bearing_paired_smoke.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("probe_z6_scorer_bearing", TOOL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakePoseNet(torch.nn.Module):
    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        return pair_btchw

    def forward(self, pair_btchw: torch.Tensor) -> dict[str, torch.Tensor]:
        if pair_btchw.dim() != 5:
            raise AssertionError(f"expected 5D pair input, got {tuple(pair_btchw.shape)}")
        frame0 = pair_btchw[:, 0]
        frame1 = pair_btchw[:, 1]
        diff = frame1 - frame0
        features = torch.stack(
            [
                frame0.mean(dim=(1, 2, 3)),
                frame1.mean(dim=(1, 2, 3)),
                diff.mean(dim=(1, 2, 3)),
                diff.abs().mean(dim=(1, 2, 3)),
                frame0.std(dim=(1, 2, 3), unbiased=False),
                frame1.std(dim=(1, 2, 3), unbiased=False),
            ],
            dim=1,
        )
        return {"pose": features}


class _FakeSegNet(torch.nn.Module):
    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        if pair_btchw.dim() != 5:
            raise AssertionError(f"expected 5D pair input, got {tuple(pair_btchw.shape)}")
        return pair_btchw[:, -1]

    def forward(self, rgb_bchw: torch.Tensor) -> torch.Tensor:
        if rgb_bchw.dim() != 4:
            raise AssertionError(f"expected 4D RGB input, got {tuple(rgb_bchw.shape)}")
        luma = rgb_bchw.mean(dim=1, keepdim=True) / 255.0
        zeros = torch.zeros_like(luma)
        return torch.cat(
            [
                luma,
                1.0 - luma,
                luma * 0.5,
                zeros + 0.25,
                zeros - 0.25,
            ],
            dim=1,
        )


def _tiny_targets(num_pairs: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    target0 = torch.rand(num_pairs, 3, 48, 64) * 0.8 + 0.1
    target1 = torch.roll(target0, shifts=1, dims=-1).mul(0.95).add(0.02)
    return target0, target1.clamp(0.0, 1.0)


def test_z6_scorer_bearing_probe_is_fail_closed_with_fake_scorers() -> None:
    tool = _load_tool()
    target0, target1 = _tiny_targets()

    payload = tool.run_scorer_bearing_probe_on_targets(
        target0=target0,
        target1=target1,
        posenet=_FakePoseNet(),
        segnet=_FakeSegNet(),
        candidate_ids=("posenet_pose", "random_control"),
        epochs=1,
        seed=3,
        lr=1e-4,
        device="cpu",
    )

    assert payload["schema"] == tool.SCHEMA
    assert payload["probe_id"] == tool.PROBE_ID
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["paradigm_claim_allowed"] is False
    assert payload["hardware_axis"] == "local_cpu_proxy_not_contest_cpu"
    assert payload["candidate_count"] == 2
    assert payload["best_proxy_id"] in {"posenet_pose", "random_control"}
    assert isinstance(payload["semantic_scorer_proxy_supported"], bool)
    assert len(payload["rows"]) == 2
    for row in payload["rows"]:
        assert row["full_film"]["paired_control_initialization"] == (
            "shared_modules_seed_order_matched_v2"
        )
        assert row["identity"]["paired_control_initialization"] == (
            "shared_modules_seed_order_matched_v2"
        )
        assert isinstance(row["identity_minus_full_score_proxy"], float)
        assert row["full_film"]["score_proxy"] > 0.0
        assert row["identity"]["score_proxy"] > 0.0


def test_z6_scorer_bearing_probe_rejects_unknown_candidate() -> None:
    tool = _load_tool()
    target0, target1 = _tiny_targets()

    with pytest.raises(ValueError, match="unknown candidate"):
        tool.run_scorer_bearing_probe_on_targets(
            target0=target0,
            target1=target1,
            posenet=_FakePoseNet(),
            segnet=_FakeSegNet(),
            candidate_ids=("does_not_exist",),
            epochs=1,
            seed=3,
            lr=1e-4,
            device="cpu",
        )
