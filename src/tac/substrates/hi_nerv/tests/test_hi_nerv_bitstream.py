# SPDX-License-Identifier: MIT
"""HiNeRV receiver-visible bitstream preparation tests."""

from __future__ import annotations

import pytest
import torch

from tac.substrates.hi_nerv.bitstream import (
    HI_NERV_ARCHIVE_SECTION_QAT_WEIGHT_POLICY_SCHEMA,
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
    build_hinerv_archive_section_qat_weight_policy,
    build_hinerv_train_time_section_byte_control,
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


def test_archive_section_qat_policy_prices_decoder_and_latent_sections() -> None:
    telemetry = {
        "schema": "hinerv_archive_section_telemetry.v1",
        "profile_ready": True,
        "archive_zip_bytes": 200,
        "section_payload_bytes": 196,
        "sections": [
            {"name": "decoder_state", "role": "decoder", "bytes": 100},
            {"name": "latents_coarse", "role": "latent", "bytes": 12},
            {"name": "latents_mid", "role": "latent", "bytes": 18},
            {"name": "latents_fine", "role": "latent", "bytes": 30},
            {"name": "hiv1_header", "role": "header", "bytes": 16},
            {"name": "meta_json", "role": "metadata", "bytes": 20},
        ],
        "sections_with_zip_overhead": [
            {"name": "decoder_state", "role": "decoder", "bytes": 100},
            {"name": "latents_coarse", "role": "latent", "bytes": 12},
            {"name": "latents_mid", "role": "latent", "bytes": 18},
            {"name": "latents_fine", "role": "latent", "bytes": 30},
            {"name": "hiv1_header", "role": "header", "bytes": 16},
            {"name": "meta_json", "role": "metadata", "bytes": 20},
            {
                "name": "archive_zip_overhead",
                "role": "container_overhead",
                "bytes": 4,
            },
        ],
    }
    base_weights = {
        "coder_qat_quant_residual": 0.001,
        "coder_qat_magnitude": 0.0001,
        "coder_qat_delta": 0.0002,
        "coder_qat_c1a_entropy": 0.0003,
    }

    policy = build_hinerv_archive_section_qat_weight_policy(
        telemetry,
        base_weights,
        byte_price_score_per_byte=0.01,
        max_decoder_multiplier=4.0,
    )

    assert policy["schema"] == HI_NERV_ARCHIVE_SECTION_QAT_WEIGHT_POLICY_SCHEMA
    assert policy["active"] is True
    assert policy["score_claim"] is False
    assert policy["decoder_section_bytes"] == 100
    assert policy["latent_section_bytes"] == 60
    assert policy["decoder_pressure_multiplier"] == pytest.approx(2.5)
    assert policy["latent_pressure_multiplier"] == pytest.approx(1.9)
    assert policy["extra_loss_weights"]["coder_qat_quant_residual"] == pytest.approx(
        0.0025
    )
    assert policy["extra_loss_weights"]["latent_qat_quant_residual"] == pytest.approx(
        0.0019
    )
    assert {
        row["section_name"]: row["operator"]
        for row in policy["applied_section_operators"]
    } == {
        "decoder_state": "decoder_coder_qat_loss_weight_scaling",
        "latents_all": "latent_coder_qat_loss_weight_scaling",
    }
    pending_roles = {
        row["section_name"]: row["role"]
        for row in policy["pending_section_operators"]
    }
    assert pending_roles == {
        "archive_zip_overhead": "container_overhead",
        "hiv1_header": "header",
        "meta_json": "metadata",
    }
    control_status = {
        row["name"]: row["control_status"] for row in policy["section_rows"]
    }
    assert control_status["decoder_state"] == "applied_decoder_qat_weight_scaling"
    assert control_status["latents_fine"] == "applied_latent_qat_weight_scaling"


def test_archive_section_qat_policy_fails_closed_when_qat_weights_empty() -> None:
    policy = build_hinerv_archive_section_qat_weight_policy(
        {
            "schema": "hinerv_archive_section_telemetry.v1",
            "profile_ready": True,
            "archive_zip_bytes": 128,
            "sections": [{"name": "decoder_state", "role": "decoder", "bytes": 64}],
        },
        {},
    )

    assert policy["active"] is False
    assert "hinerv_archive_section_qat_base_weights_empty" in policy["blockers"]


def test_archive_section_qat_policy_fails_closed_when_qat_weights_all_zero() -> None:
    policy = build_hinerv_archive_section_qat_weight_policy(
        {
            "schema": "hinerv_archive_section_telemetry.v1",
            "profile_ready": True,
            "archive_zip_bytes": 128,
            "sections": [{"name": "decoder_state", "role": "decoder", "bytes": 64}],
        },
        {
            "coder_qat_quant_residual": 0.0,
            "coder_qat_magnitude": 0.0,
        },
    )

    assert policy["active"] is False
    assert "hinerv_archive_section_qat_base_weights_all_zero" in policy["blockers"]


def test_train_time_section_byte_control_prices_only_actuated_sections() -> None:
    telemetry = {
        "schema": "hinerv_archive_section_telemetry.v1",
        "profile_ready": True,
        "archive_zip_bytes": 400,
        "sections": [
            {"name": "decoder_state", "role": "decoder", "bytes": 200},
            {"name": "latents_coarse", "role": "latent", "bytes": 40},
            {"name": "latents_mid", "role": "latent", "bytes": 40},
            {"name": "meta_json", "role": "metadata", "bytes": 20},
        ],
    }

    control = build_hinerv_train_time_section_byte_control(
        telemetry,
        {
            "coder_qat_quant_residual": 0.001,
            "latent_qat_quant_residual": 0.002,
        },
        hard_byte_ceiling=100,
        byte_price_score_per_byte=0.01,
    )

    assert control["active"] is True
    assert control["section_byte_budgets"] == {
        "decoder_state": 50,
        "latents_coarse": 10,
        "latents_mid": 10,
    }
    assert control["section_byte_loss_weight_key_map"] == {
        "decoder_state": "coder_qat_quant_residual",
        "latents_coarse": "latent_qat_quant_residual",
        "latents_mid": "latent_qat_quant_residual",
    }
    assert control["metrics_payload"]["archive_bytes"] == 400
    assert control["metrics_payload"]["section_bytes"]["meta_json"] == 20
    assert control["metrics_payload"]["rate_score_per_byte"] == pytest.approx(0.01)
    assert control["metrics_payload"]["section_rate_scores"]["decoder_state"] == (
        pytest.approx(2.0)
    )
    assert control["metrics_payload"][
        "train_time_section_rate_score__decoder_state"
    ] == pytest.approx(2.0)
    assert control["metrics_payload"][
        "train_time_section_rate_score__latents_coarse"
    ] == pytest.approx(0.4)
    assert control["metrics_payload"]["train_time_section_rate_score__meta_json"] == (
        pytest.approx(0.2)
    )
    assert control["pending_section_rows"][0]["section_name"] == "meta_json"
    assert control["pending_section_rows"][0]["pending_reason"] == (
        "non_differentiable_archive_section"
    )
    assert control["pending_section_rows"][0]["rate_score"] == pytest.approx(0.2)
    assert control["pending_section_rows"][0][
        "budget_rate_score_if_actuated"
    ] == pytest.approx(0.05)
    assert control["blockers"] == []


def test_train_time_section_byte_control_keeps_decoder_active_when_latent_qat_missing() -> None:
    telemetry = {
        "schema": "hinerv_archive_section_telemetry.v1",
        "profile_ready": True,
        "archive_zip_bytes": 400,
        "sections": [
            {"name": "decoder_state", "role": "decoder", "bytes": 200},
            {"name": "latents_coarse", "role": "latent", "bytes": 80},
            {"name": "meta_json", "role": "metadata", "bytes": 20},
        ],
    }

    control = build_hinerv_train_time_section_byte_control(
        telemetry,
        {"coder_qat_c1a_entropy": 0.001},
        hard_byte_ceiling=100,
        byte_price_score_per_byte=0.01,
    )

    assert control["active"] is True
    assert control["section_byte_budgets"] == {"decoder_state": 50}
    assert control["section_byte_loss_weight_key_map"] == {
        "decoder_state": "coder_qat_c1a_entropy"
    }
    pending = {row["section_name"]: row for row in control["pending_section_rows"]}
    assert pending["latents_coarse"]["pending_reason"] == "active_qat_loss_key_missing"
    assert pending["meta_json"]["pending_reason"] == "non_differentiable_archive_section"
    assert control["blockers"] == []


def test_train_time_section_byte_control_blocks_when_no_sections_are_actuated() -> None:
    telemetry = {
        "schema": "hinerv_archive_section_telemetry.v1",
        "profile_ready": True,
        "archive_zip_bytes": 400,
        "sections": [
            {"name": "decoder_state", "role": "decoder", "bytes": 200},
            {"name": "latents_coarse", "role": "latent", "bytes": 80},
            {"name": "meta_json", "role": "metadata", "bytes": 20},
        ],
    }

    control = build_hinerv_train_time_section_byte_control(
        telemetry,
        {
            "coder_qat_c1a_entropy": 0.0,
            "latent_qat_c1a_entropy": 0.0,
        },
        hard_byte_ceiling=100,
        byte_price_score_per_byte=0.01,
    )

    assert control["active"] is False
    assert control["section_byte_budgets"] == {}
    assert control["section_byte_loss_weight_key_map"] == {}
    assert control["controlled_section_count"] == 0
    assert control["pending_section_count"] == 3
    pending = {row["section_name"]: row for row in control["pending_section_rows"]}
    assert pending["decoder_state"]["pending_reason"] == "active_qat_loss_key_missing"
    assert pending["latents_coarse"]["pending_reason"] == "active_qat_loss_key_missing"
    assert pending["meta_json"]["pending_reason"] == "non_differentiable_archive_section"
    assert control["metrics_payload"]["train_time_section_rate_score__decoder_state"] == (
        pytest.approx(2.0)
    )
    assert control["blockers"] == ["hinerv_train_time_section_byte_no_actuated_sections"]


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
    assert selection["byte_price_plan"]["economic_decision_counts"]["cut"] == 1
    assert selection["byte_price_plan"]["decision_counts"]["demote"] == 3
    assert selection["selected_row"] is None
    assert selection["selected_economic_row"]["decoder_codec_requested"] == (
        "int8_scale_bundled"
    )
    assert selection["selected_economic_row"]["waterfill_economic_admissible"] is True
    assert selection["selected_economic_row"]["canonical_economic_decision"] == "cut"
    assert selection["selected_economic_row"]["canonical_decision"] == "demote"
    assert "receiver_proof_path_missing" in selection["selected_economic_row"][
        "canonical_blockers"
    ]
    assert all(
        row["decoder_codec_requested"] != "int4_scale_bundled"
        for row in selection["economic_admissible_rows"]
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
