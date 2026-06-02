# SPDX-License-Identifier: MIT
"""HiNeRV receiver-visible bitstream preparation tests."""

from __future__ import annotations

import pytest
import torch

from tac.substrates.hi_nerv.bitstream import (
    HI_NERV_BITSTREAM_PREPARATION_SCHEMA,
    HI_NERV_BITSTREAM_RATE_SCORE_PER_BYTE,
    HI_NERV_BITSTREAM_ROUNDTRIP_SCHEMA,
    HI_NERV_BITSTREAM_WATERFILL_SELECTION_SCHEMA,
    HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF,
    HiNervBitstreamError,
    apply_decoder_pruning,
    apply_decoder_quant_noise,
    apply_decoder_waterfill_actions,
    measure_hi_nerv_decoder_bitstream_roundtrip,
    prepare_hi_nerv_decoder_bitstream_state,
    select_hi_nerv_bitstream_codec_by_scorer_waterfill,
)


def _state() -> dict[str, torch.Tensor]:
    torch.manual_seed(41)
    return {
        "stem.weight": torch.linspace(-1.0, 1.0, steps=32).reshape(4, 8),
        "stem.bias": torch.linspace(-0.1, 0.1, steps=4),
        "block.weight": torch.randn(3, 4, 3, 3) * 0.05,
        "norm.weight": torch.ones(4),
    }


def test_hi_nerv_bitstream_preparation_applies_receiver_visible_transforms() -> None:
    base = _state()

    prepared = prepare_hi_nerv_decoder_bitstream_state(
        base,
        pruning_ratio=0.25,
        quant_noise_bits=4,
        quant_noise_scale=0.5,
        quant_noise_seed=7,
    )

    report = prepared.report
    assert report["schema"] == HI_NERV_BITSTREAM_PREPARATION_SCHEMA
    assert report["proof"] == HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF
    assert report["shape_preserved"] is True
    assert report["score_claim"] is False
    assert report["pruning"]["actual_new_zero_values"] > 0
    assert report["quant_noise"]["changed_tensor_count"] > 0
    assert report["quant_noise"]["preserves_existing_zero_symbols"] is True
    assert set(prepared.state_dict) == set(base)
    assert sum(
        int(torch.count_nonzero(tensor == 0).item())
        for name, tensor in prepared.state_dict.items()
        if name.endswith(".weight") and tensor.dim() >= 2
    ) > sum(
        int(torch.count_nonzero(tensor == 0).item())
        for name, tensor in base.items()
        if name.endswith(".weight") and tensor.dim() >= 2
    )


def test_hi_nerv_bitstream_roundtrip_measures_codec_portfolio() -> None:
    report = measure_hi_nerv_decoder_bitstream_roundtrip(
        _state(),
        decoder_codecs=(
            "fp16_enveloped",
            "int8_scale_bundled",
            "int4_scale_bundled",
        ),
        pruning_ratio=0.25,
        quant_noise_bits=4,
        quant_noise_scale=0.25,
        quant_noise_seed=11,
    )

    assert report["schema"] == HI_NERV_BITSTREAM_ROUNDTRIP_SCHEMA
    assert report["proof"] == HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert "score_sensitivity_replay_not_attached" in report["blockers"]
    assert report["portfolio_selection"]["schema"] == (
        HI_NERV_BITSTREAM_WATERFILL_SELECTION_SCHEMA
    )
    assert report["portfolio_selection"]["byte_price_plan"]["schema"] == (
        "compact_nerv_byte_price_controller.v1"
    )
    assert "hi_nerv_bitstream_scorer_value_replay_missing" in report[
        "portfolio_selection"
    ]["blockers"]
    rows = report["rows"]
    assert [row["blob_bytes"] for row in rows] == sorted(
        row["blob_bytes"] for row in rows
    )
    assert report["best_row"] == rows[0]
    assert {row["decoder_codec_requested"] for row in rows} == {
        "fp16_enveloped",
        "int8_scale_bundled",
        "int4_scale_bundled",
    }
    for row in rows:
        assert row["shape_preserved"] is True
        assert row["roundtrip_error"]["missing"] == []
        assert row["roundtrip_error"]["unexpected"] == []


def test_hi_nerv_decoder_waterfill_actions_mutate_real_tensors() -> None:
    base = _state()
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "unit",
        "rows": [
            {
                "group_name": "stem.weight",
                "selected_bits": 0,
                "selected_action": "zero_rle",
            },
            {
                "group_name": "block.weight",
                "selected_bits": 4,
                "selected_action": "int4",
            },
            {
                "group_name": "norm.weight",
                "selected_bits": 32,
                "selected_action": "fp32_protect",
            },
        ],
        "blockers": [],
    }

    changed, report = apply_decoder_waterfill_actions(
        base,
        decoder_weight_waterfill_plan=plan,
    )

    assert report["method"] == "decoder_weight_waterfill_selected_actions"
    assert report["plan_attached"] is True
    assert report["applied_row_count"] == 3
    assert report["blocked_row_count"] == 0
    assert report["changed_tensor_count"] == 2
    assert report["score_claim"] is False
    assert torch.count_nonzero(changed["stem.weight"]).item() == 0
    assert not torch.equal(changed["block.weight"], base["block.weight"])
    assert torch.equal(changed["norm.weight"], base["norm.weight"])
    assert torch.equal(base["stem.weight"], _state()["stem.weight"])
    by_name = {row["group_name"]: row for row in report["applied_rows"]}
    assert by_name["stem.weight"]["sha256_before"] != by_name["stem.weight"][
        "sha256_after"
    ]
    assert by_name["norm.weight"]["changed"] is False


def test_hi_nerv_decoder_waterfill_refuses_blocked_plan() -> None:
    base = _state()
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "unit",
        "rows": [
            {
                "group_name": "stem.weight",
                "selected_bits": 0,
                "selected_action": "zero_rle",
            },
            {
                "group_name": "block.weight",
                "selected_bits": 4,
                "selected_action": "int4",
                "blockers": ["score_loss_proxy_outside_allocator_linearization_basin"],
            },
        ],
        "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
    }

    changed, report = apply_decoder_waterfill_actions(
        base,
        decoder_weight_waterfill_plan=plan,
    )

    assert report["method"] == "decoder_weight_waterfill_blocked"
    assert report["applied_row_count"] == 0
    assert report["blocked_row_count"] == 2
    assert report["changed_tensor_count"] == 0
    assert "contest_cpu_cuda_exact_eval_not_executed" in report["blockers"]
    assert torch.equal(changed["stem.weight"], base["stem.weight"])
    assert torch.equal(changed["block.weight"], base["block.weight"])


def test_hi_nerv_decoder_waterfill_skips_blocked_rows() -> None:
    base = _state()
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "unit",
        "rows": [
            {
                "group_name": "stem.weight",
                "selected_bits": 0,
                "selected_action": "zero_rle",
            },
            {
                "group_name": "block.weight",
                "selected_bits": 4,
                "selected_action": "int4",
                "blockers": ["score_loss_proxy_outside_allocator_linearization_basin"],
            },
        ],
        "blockers": [],
    }

    changed, report = apply_decoder_waterfill_actions(
        base,
        decoder_weight_waterfill_plan=plan,
    )

    assert report["method"] == "decoder_weight_waterfill_selected_actions"
    assert report["applied_row_count"] == 1
    assert report["blocked_row_count"] == 1
    assert report["changed_tensor_count"] == 1
    assert torch.count_nonzero(changed["stem.weight"]).item() == 0
    assert torch.equal(changed["block.weight"], base["block.weight"])
    assert (
        "score_loss_proxy_outside_allocator_linearization_basin"
        in report["blockers"]
    )


def test_hi_nerv_bitstream_waterfill_selector_admits_only_positive_value_per_byte() -> None:
    rows = [
        {
            "decoder_codec_requested": "fp16_enveloped",
            "decoder_codec_emitted": "fp16_enveloped",
            "blob_bytes": 10_000,
        },
        {
            "decoder_codec_requested": "int8_scale_bundled",
            "decoder_codec_emitted": "int8_scale_bundled",
            "blob_bytes": 7_000,
        },
        {
            "decoder_codec_requested": "int4_scale_bundled",
            "decoder_codec_emitted": "int4_scale_bundled",
            "blob_bytes": 4_000,
        },
    ]
    int8_rate_gain = (7_000 - 10_000) * HI_NERV_BITSTREAM_RATE_SCORE_PER_BYTE
    int4_rate_gain = (4_000 - 10_000) * HI_NERV_BITSTREAM_RATE_SCORE_PER_BYTE

    selection = select_hi_nerv_bitstream_codec_by_scorer_waterfill(
        rows,
        scorer_value_rows=[
            {
                "decoder_codec_requested": "int8_scale_bundled",
                "delta_nonrate_score": abs(int8_rate_gain) * 0.5,
            },
            {
                "decoder_codec_requested": "int4_scale_bundled",
                "delta_nonrate_score": abs(int4_rate_gain) * 1.5,
            },
        ],
        baseline_codec="fp16_enveloped",
        candidate_id="receiver_proven_hi_nerv",
        archive_sha256="a" * 64,
        axis_tag="[contest-CUDA]",
        receiver_proof_status="satisfied",
        full_video_coverage=True,
    )

    assert selection["schema"] == HI_NERV_BITSTREAM_WATERFILL_SELECTION_SCHEMA
    assert selection["scorer_value_replay_attached"] is True
    assert selection["byte_price_plan"]["decision_counts"]["cut"] == 1
    assert selection["selected_row"]["decoder_codec_requested"] == "int8_scale_bundled"
    assert selection["selected_economic_row"]["decoder_codec_requested"] == (
        "int8_scale_bundled"
    )
    assert selection["selected_row"]["waterfill_admissible"] is True
    assert selection["selected_row"]["canonical_decision"] == "cut"
    assert all(
        row["decoder_codec_requested"] != "int4_scale_bundled"
        for row in selection["admissible_rows"]
    )
    assert "hi_nerv_bitstream_scorer_value_replay_missing" not in selection[
        "blockers"
    ]
    assert selection["score_claim"] is False


def test_hi_nerv_bitstream_waterfill_selector_keeps_local_economics_false_authority() -> None:
    rows = [
        {
            "decoder_codec_requested": "fp16_enveloped",
            "decoder_codec_emitted": "fp16_enveloped",
            "blob_bytes": 10_000,
        },
        {
            "decoder_codec_requested": "int8_scale_bundled",
            "decoder_codec_emitted": "int8_scale_bundled",
            "blob_bytes": 7_000,
        },
    ]
    int8_rate_gain = (7_000 - 10_000) * HI_NERV_BITSTREAM_RATE_SCORE_PER_BYTE

    selection = select_hi_nerv_bitstream_codec_by_scorer_waterfill(
        rows,
        scorer_value_rows=[
            {
                "decoder_codec_requested": "int8_scale_bundled",
                "delta_nonrate_score": abs(int8_rate_gain) * 0.5,
            },
        ],
        baseline_codec="fp16_enveloped",
    )

    assert selection["selected_row"] is None
    assert selection["selected_economic_row"]["decoder_codec_requested"] == (
        "int8_scale_bundled"
    )
    assert selection["selected_economic_row"]["waterfill_economic_admissible"] is True
    assert selection["selected_economic_row"]["waterfill_admissible"] is False
    assert "full_video_coverage_missing" in selection["byte_price_plan"]["blockers"]
    assert selection["promotion_eligible"] is False


def test_hi_nerv_bitstream_pruning_rejects_invalid_ratio() -> None:
    with pytest.raises(HiNervBitstreamError, match="pruning_ratio"):
        apply_decoder_pruning(_state(), pruning_ratio=1.0)


def test_hi_nerv_bitstream_quant_noise_rejects_invalid_bits() -> None:
    with pytest.raises(HiNervBitstreamError, match="quant_noise_bits"):
        apply_decoder_quant_noise(_state(), quant_bits=3, noise_scale=0.5)


def test_hi_nerv_bitstream_roundtrip_rejects_unknown_codec() -> None:
    with pytest.raises(HiNervBitstreamError, match="unsupported"):
        measure_hi_nerv_decoder_bitstream_roundtrip(
            _state(),
            decoder_codecs=("not_a_codec",),
        )
