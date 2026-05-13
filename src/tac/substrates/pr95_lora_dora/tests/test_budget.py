"""Closed-form byte and break-even tests for PR95 LoRA/DoRA."""

from __future__ import annotations

import torch

from tac.substrates.pr95_lora_dora.archive import encode_lora_trailer
from tac.substrates.pr95_lora_dora.budget import (
    DEFAULT_TIER_C_LAYER_DIMS,
    adapter_break_even,
    tier_c_raw_trailer_bytes,
    tier_c_trainable_params,
)


def _record(name: str, out_dim: int, in_dim: int, *, kind: str = "lora", rank: int = 8) -> dict:
    record = {
        "name": name,
        "kind": kind,
        "rank": rank,
        "alpha": float(rank),
        "A": torch.zeros(rank, in_dim),
        "B": torch.zeros(out_dim, rank),
    }
    if kind == "dora":
        record["magnitude"] = torch.ones(out_dim)
    return record


def test_tier_c_lora_r8_budget_matches_archive_encoder() -> None:
    records = [_record(name, out_dim, in_dim) for name, out_dim, in_dim in DEFAULT_TIER_C_LAYER_DIMS]
    assert tier_c_trainable_params(rank=8, kind="lora") == 17_416
    assert tier_c_raw_trailer_bytes(rank=8, kind="lora") == len(encode_lora_trailer(records))


def test_tier_c_dora_r8_budget_includes_raw_conv_output_magnitudes() -> None:
    records = [
        _record(name, out_dim, in_dim, kind="dora")
        for name, out_dim, in_dim in DEFAULT_TIER_C_LAYER_DIMS
    ]
    # DoRA adds one magnitude per raw adapted output channel: 144+144+108+80+72+72 = 620.
    assert tier_c_trainable_params(rank=8, kind="dora") == 17_416 + 620
    assert tier_c_raw_trailer_bytes(rank=8, kind="dora") == len(encode_lora_trailer(records))


def test_break_even_pose_reduction_uses_exact_sqrt_curve_not_bad_linear_guess() -> None:
    budget = adapter_break_even(raw_trailer_bytes=21_000, pose_operating_point=3.4e-5)
    assert 0.0139 < budget.rate_score_penalty < 0.0141
    assert 1.39e-4 < budget.required_seg_reduction < 1.41e-4
    # A +21KB trailer is a large fraction of the current pose term. Exact inverse
    # math requires nearly the whole PR106/PR95 pose distortion, not ~5e-7.
    assert 3.1e-5 < budget.required_pose_reduction_exact < 3.3e-5
    assert budget.required_pose_reduction_exact > 50 * 5e-7
    assert budget.pose_only_feasible is True
    assert budget.score_claim is False
    assert budget.ready_for_exact_eval_dispatch is False


def test_pose_only_break_even_reports_infeasible_when_penalty_exceeds_pose_term() -> None:
    budget = adapter_break_even(raw_trailer_bytes=40_000, pose_operating_point=3.4e-5)
    assert budget.pose_only_feasible is False
    assert budget.required_pose_reduction_exact == 3.4e-5
    assert budget.residual_score_after_zero_pose > 0.0
