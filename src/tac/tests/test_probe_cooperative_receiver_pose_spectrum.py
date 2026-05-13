from __future__ import annotations

import pytest
import torch

from tac.optimization.proxy_candidate_contract import validate_proxy_candidate
from tools.probe_cooperative_receiver_pose_spectrum import analyze_pose_targets_tensor


def test_pose_spectrum_detects_low_frequency_smooth_pose() -> None:
    t = torch.linspace(0.0, 1.0, 128)
    targets = torch.stack(
        [
            torch.sin(2.0 * torch.pi * t),
            torch.cos(2.0 * torch.pi * t),
            0.2 * t,
            0.1 * torch.sin(4.0 * torch.pi * t),
            0.05 * torch.cos(4.0 * torch.pi * t),
            0.01 * t.square(),
        ],
        dim=1,
    )

    report = analyze_pose_targets_tensor(targets, low_frequency_fraction=0.10)

    assert report["schema"] == "tac_cooperative_receiver_pose_spectrum_probe_v1"
    assert report["num_pairs"] == 128
    assert report["pose_dims"] == 6
    assert report["low_frequency_energy_fraction"] > 0.80
    assert report["hypothesis_supported"] is True
    assert report["estimated_sparse_fft_bytes_at_95pct"] > 0
    assert validate_proxy_candidate(report) == []


def test_pose_spectrum_keeps_high_frequency_noise_proxy_only() -> None:
    gen = torch.Generator().manual_seed(123)
    targets = torch.randn((96, 6), generator=gen)

    report = analyze_pose_targets_tensor(targets, low_frequency_fraction=0.10)

    assert report["low_frequency_energy_fraction"] < 0.25
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "requires_exact_eval_readiness_gate" in report["dispatch_blockers"]


def test_pose_spectrum_rejects_invalid_targets() -> None:
    with pytest.raises(ValueError, match="shape"):
        analyze_pose_targets_tensor(torch.zeros(10, 5))
    with pytest.raises(ValueError, match="at least two"):
        analyze_pose_targets_tensor(torch.zeros(1, 6))
    bad = torch.zeros(3, 6)
    bad[1, 2] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        analyze_pose_targets_tensor(bad)
