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
    HI_NERV_OFFICIAL_ENTROPY_RECEIVER_CONSUMPTION_SCHEMA,
    HI_NERV_OFFICIAL_QUANT_AXIS_RULE,
    HI_NERV_OFFICIAL_QUANTNOISE_METHOD,
    HI_NERV_OFFICIAL_TORCHAC_PARITY_BLOCKER,
    HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF,
    HiNervBitstreamError,
    apply_decoder_pruning,
    apply_decoder_quant_noise,
    apply_decoder_waterfill_actions,
    build_decoder_waterfill_fake_quant_forward_plan,
    decoder_waterfill_fake_quant_bits_by_name,
    measure_hi_nerv_decoder_bitstream_roundtrip,
    measure_hi_nerv_official_entropy_receiver_consumption,
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


def test_hi_nerv_bitstream_quant_noise_accepts_official_six_seven_bits() -> None:
    base = _state()

    for bits in (6, 7):
        changed, report = apply_decoder_quant_noise(
            base,
            quant_bits=bits,
            noise_scale=0.5,
            seed=bits,
        )

        assert report["method"] == HI_NERV_OFFICIAL_QUANTNOISE_METHOD
        assert report["quant_bits"] == bits
        assert report["noise_ratio"] == 0.5
        assert report["official_quant_axis_rule"] == HI_NERV_OFFICIAL_QUANT_AXIS_RULE
        assert report["changed_tensor_count"] > 0
        assert report["selected_value_count"] > 0
        assert report["actual_changed_value_count"] > 0
        assert report["max_abs_delta"] > 0.0
        assert report["preserves_existing_zero_symbols"] is True
        assert report["score_claim"] is False
        assert HI_NERV_OFFICIAL_TORCHAC_PARITY_BLOCKER in report["blockers"]
        assert not torch.equal(changed["block.weight"], base["block.weight"])


def test_hi_nerv_bitstream_quant_noise_is_official_random_replacement_deterministic() -> None:
    base = {
        "wide.weight": torch.linspace(-1.0, 1.0, steps=48).reshape(24, 2),
        "bias": torch.linspace(-0.2, 0.2, steps=24),
    }

    changed_a, report_a = apply_decoder_quant_noise(
        base,
        quant_bits=4,
        noise_scale=1.0,
        seed=99,
    )
    changed_b, report_b = apply_decoder_quant_noise(
        base,
        quant_bits=4,
        noise_scale=1.0,
        seed=99,
    )

    assert torch.equal(changed_a["wide.weight"], changed_b["wide.weight"])
    assert torch.equal(changed_a["bias"], changed_b["bias"])
    assert torch.equal(changed_a["bias"], base["bias"])
    assert report_a["actual_changed_value_count"] == report_b[
        "actual_changed_value_count"
    ]
    wide_row = next(
        row for row in report_a["tensor_rows"] if row["tensor_name"] == "wide.weight"
    )
    assert wide_row["quant_axis"] == 0
    assert wide_row["selected_value_count"] == base["wide.weight"].numel()
    assert torch.count_nonzero(changed_a["wide.weight"] != base["wide.weight"]).item() == (
        wide_row["actual_changed_value_count"]
    )
    assert report_a["score_claim"] is False


def test_hi_nerv_bitstream_roundtrip_measures_codec_portfolio() -> None:
    report = measure_hi_nerv_decoder_bitstream_roundtrip(
        _state(),
        decoder_codecs=(
            "fp16_enveloped",
            "int8_scale_bundled",
            "int7_scale_bundled",
            "int6_mixed",
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
    assert report["official_entropy_receiver_consumption"]["schema"] == (
        HI_NERV_OFFICIAL_ENTROPY_RECEIVER_CONSUMPTION_SCHEMA
    )
    assert report["official_entropy_receiver_consumption"][
        "torchac_encode_decode_bound"
    ] is False
    assert HI_NERV_OFFICIAL_TORCHAC_PARITY_BLOCKER in report["blockers"]
    assert HI_NERV_OFFICIAL_TORCHAC_PARITY_BLOCKER in report[
        "official_entropy_receiver_consumption"
    ]["blockers"]
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
        "int7_scale_bundled",
        "int6_mixed",
        "int4_scale_bundled",
    }
    assert {row["decoder_codec_emitted"] for row in rows} >= {
        "int7_scale_bundled",
        "int6_mixed",
    }
    for row in rows:
        assert row["shape_preserved"] is True
        assert row["roundtrip_error"]["missing"] == []
        assert row["roundtrip_error"]["unexpected"] == []


def test_hi_nerv_official_entropy_receiver_manifest_consumes_pruned_zero_masks() -> None:
    prepared = prepare_hi_nerv_decoder_bitstream_state(
        _state(),
        pruning_ratio=0.30,
        quant_noise_bits=4,
        quant_noise_scale=1.0,
        quant_noise_seed=5,
    )

    manifest = measure_hi_nerv_official_entropy_receiver_consumption(
        prepared.state_dict,
        quant_bits=4,
    )

    assert manifest["schema"] == HI_NERV_OFFICIAL_ENTROPY_RECEIVER_CONSUMPTION_SCHEMA
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["torchac_encode_decode_bound"] is False
    assert manifest["torchac_byte_streams_present"] is False
    assert manifest["removed_zero_value_count"] > 0
    assert manifest["ideal_total_entropy_bytes_lower_bound"] > 0
    assert HI_NERV_OFFICIAL_TORCHAC_PARITY_BLOCKER in manifest["blockers"]
    pruned_rows = [row for row in manifest["rows"] if row["mask_stream_required"]]
    assert pruned_rows, "pruned decoder tensors must require receiver mask streams"
    assert all(row["torchac_byte_stream_present"] is False for row in manifest["rows"])
    assert any(
        row["payload_symbol_count"] < int(torch.numel(prepared.state_dict[row["tensor_name"]]))
        for row in pruned_rows
    )


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


def test_hi_nerv_decoder_waterfill_accepts_official_six_seven_bit_actions() -> None:
    base = _state()
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "unit",
        "rows": [
            {
                "group_name": "stem.weight",
                "selected_bits": 6,
                "selected_action": "int6",
            },
            {
                "group_name": "block.weight",
                "selected_bits": 7,
                "selected_action": "int7",
            },
        ],
        "blockers": [],
    }

    changed, report = apply_decoder_waterfill_actions(
        base,
        decoder_weight_waterfill_plan=plan,
    )

    assert report["method"] == "decoder_weight_waterfill_selected_actions"
    assert report["applied_row_count"] == 2
    assert report["changed_tensor_count"] == 2
    assert not torch.equal(changed["stem.weight"], base["stem.weight"])
    assert not torch.equal(changed["block.weight"], base["block.weight"])
    by_name = {row["group_name"]: row for row in report["applied_rows"]}
    assert by_name["stem.weight"]["selected_bits"] == 6
    assert by_name["block.weight"]["selected_bits"] == 7
    assert report["score_claim"] is False


def test_hi_nerv_decoder_waterfill_allows_authority_only_blocked_plan_locally() -> None:
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

    assert report["method"] == "decoder_weight_waterfill_selected_actions"
    assert report["applied_row_count"] == 1
    assert report["blocked_row_count"] == 1
    assert report["changed_tensor_count"] == 1
    assert "contest_cpu_cuda_exact_eval_not_executed" in report["blockers"]
    assert report["actuation_blockers"] == [
        "score_loss_proxy_outside_allocator_linearization_basin"
    ]
    assert torch.count_nonzero(changed["stem.weight"]).item() == 0
    assert torch.equal(changed["block.weight"], base["block.weight"])


def test_hi_nerv_decoder_waterfill_refuses_actuation_blocked_plan() -> None:
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
        ],
        "blockers": [
            "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin"
        ],
    }

    changed, report = apply_decoder_waterfill_actions(
        base,
        decoder_weight_waterfill_plan=plan,
    )

    assert report["method"] == "decoder_weight_waterfill_blocked"
    assert report["applied_row_count"] == 0
    assert report["blocked_row_count"] == 1
    assert report["changed_tensor_count"] == 0
    assert report["actuation_blockers"] == [
        "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin"
    ]
    assert torch.equal(changed["stem.weight"], base["stem.weight"])


def test_hi_nerv_decoder_waterfill_fake_quant_plan_targets_selected_bits() -> None:
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "unit",
        "rows": [
            {
                "group_name": "stem.weight",
                "selected_bits": 4,
                "selected_action": "int4",
            },
            {
                "group_name": "block.weight",
                "selected_bits": 0,
                "selected_action": "zero_rle",
                "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
            },
            {
                "group_name": "norm.weight",
                "selected_bits": 32,
                "selected_action": "fp32_protect",
            },
        ],
        "blockers": [],
    }

    report = build_decoder_waterfill_fake_quant_forward_plan(plan)

    assert report["method"] == "decoder_weight_waterfill_fake_quant_targets"
    assert report["targeted_tensor_count"] == 2
    assert report["per_tensor_bits"] == {
        "block.weight": 0,
        "stem.weight": 4,
    }
    assert decoder_waterfill_fake_quant_bits_by_name(plan) == {
        "block.weight": 0,
        "stem.weight": 4,
    }
    assert "contest_cpu_cuda_exact_eval_not_executed" in report["blockers"]
    assert report["skipped_rows"][0]["group_name"] == "norm.weight"
    assert report["skipped_rows"][0]["reason"] == (
        "decoder_weight_waterfill_full_precision_no_fake_quant"
    )
    assert report["score_claim"] is False


def test_hi_nerv_decoder_waterfill_fake_quant_plan_blocks_unsafe_actuation() -> None:
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "unit",
        "rows": [
            {
                "group_name": "stem.weight",
                "selected_bits": 4,
                "selected_action": "int4",
            },
        ],
        "blockers": [
            "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin"
        ],
    }

    report = build_decoder_waterfill_fake_quant_forward_plan(plan)

    assert report["method"] == "decoder_weight_waterfill_fake_quant_blocked"
    assert report["per_tensor_bits"] == {}
    assert report["actuation_blockers"] == [
        "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin"
    ]
    assert report["score_claim"] is False


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


def test_hi_nerv_decoder_waterfill_fails_closed_when_no_groups_match() -> None:
    base = _state()
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "unit",
        "rows": [
            {
                "group_name": "missing.weight",
                "selected_bits": 4,
                "selected_action": "int4",
            },
        ],
        "blockers": [],
    }

    changed, report = apply_decoder_waterfill_actions(
        base,
        decoder_weight_waterfill_plan=plan,
    )

    assert report["method"] == "decoder_weight_waterfill_no_matching_groups"
    assert report["applied_row_count"] == 0
    assert report["blocked_row_count"] == 1
    assert report["changed_tensor_count"] == 0
    assert "decoder_weight_waterfill_group_missing:missing.weight" in report[
        "blockers"
    ]
    assert "decoder_weight_waterfill_no_matching_groups_applied" in report[
        "blockers"
    ]
    assert "decoder_weight_waterfill_no_matching_groups_applied" in report[
        "actuation_blockers"
    ]
    assert report["skipped_rows"][0]["group_name"] == "missing.weight"
    assert report["skipped_rows"][0]["reason"] == (
        "decoder_weight_waterfill_group_missing"
    )
    assert report["score_claim"] is False
    for name, tensor in base.items():
        assert torch.equal(changed[name], tensor)


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


def test_hi_nerv_bitstream_quant_noise_rejects_non_probability_noise_ratio() -> None:
    with pytest.raises(HiNervBitstreamError, match="noise_ratio"):
        apply_decoder_quant_noise(_state(), quant_bits=4, noise_scale=1.5)


def test_hi_nerv_bitstream_roundtrip_rejects_unknown_codec() -> None:
    with pytest.raises(HiNervBitstreamError, match="unsupported"):
        measure_hi_nerv_decoder_bitstream_roundtrip(
            _state(),
            decoder_codecs=("not_a_codec",),
        )
