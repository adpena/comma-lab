# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "probe_z6_real_video_ego_proxy_sweep.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("z6_ego_proxy_sweep_tool", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tiny_targets(num_pairs: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    target0 = torch.zeros(num_pairs, 3, 48, 64)
    target1 = target0.clone()
    for pair_idx in range(num_pairs):
        target1[pair_idx, :, 8:32, 10 + pair_idx : 34 + pair_idx] = (
            float(pair_idx + 1) / 10.0
        )
    return target0, target1


def test_z6_ego_proxy_candidates_are_finite_and_paired() -> None:
    tool = _load_tool()
    target0, target1 = _tiny_targets(num_pairs=5)

    candidates = tool.build_ego_proxy_candidates(
        target0,
        target1,
        ego_motion_dim=4,
        seed=11,
    )

    assert set(candidates) == {
        "zero",
        "ramp",
        "frame_delta",
        "moment_proxy",
        "quadrant_delta",
        "random_control",
    }
    for proxy in candidates.values():
        assert proxy.shape == (5, 4)
        assert torch.isfinite(proxy).all()
    assert float(candidates["zero"].abs().sum().item()) == 0.0
    assert float(candidates["frame_delta"].abs().sum().item()) > 0.0
    assert float(candidates["moment_proxy"].abs().sum().item()) > 0.0
    assert float(candidates["quadrant_delta"].abs().sum().item()) > 0.0


def test_z6_ego_proxy_sweep_is_fail_closed_and_paired() -> None:
    tool = _load_tool()
    target0, target1 = _tiny_targets(num_pairs=2)

    payload = tool.run_sweep_on_targets(
        target0=target0,
        target1=target1,
        epochs=1,
        seed=7,
        lr=1e-4,
    )

    assert payload["schema"] == tool.SCHEMA
    assert payload["probe_id"] == tool.PROBE_ID
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["paradigm_claim_allowed"] is False
    assert payload["candidate_count"] == 6
    assert len(payload["rows"]) == 6
    assert payload["best_proxy_id"] in {row["proxy_id"] for row in payload["rows"]}
    assert "no_contest_cpu_cuda_pair" in payload["blockers"]
    assert "not_paradigm_claim_authority" in payload["blockers"]

    for row in payload["rows"]:
        assert row["full_film"]["identity_predictor"] is False
        assert row["identity"]["identity_predictor"] is True
        assert isinstance(row["identity_minus_full_loss_proxy"], float)
        assert isinstance(row["identity_minus_full_recon"], float)
        assert isinstance(row["identity_minus_full_residual"], float)
        assert isinstance(row["full_minus_identity_archive_bytes"], int)
        assert row["full_film"]["archive_bytes"] > 0
        assert row["identity"]["archive_bytes"] > 0
