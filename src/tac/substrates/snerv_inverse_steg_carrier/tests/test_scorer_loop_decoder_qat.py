# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV scorer-loop decoder/QAT helpers."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod
from tac.analysis.snerv_step_map_coder import encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    HfGenerationDecoder,
    SnervModelSizeConfig,
)
from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (
    COMPONENT_GUARD_MODES,
    CONTEST_BYTE_PRICE,
    SNERV_QAT_RECEIVER_CODEC_PRICING_PROOF,
    SNERV_QAT_SECTION_VALUE_PRESSURE_PROOF,
    QuantizedDecoderStats,
    SnervDecoderEval,
    SnervPairEval,
    SnervScorerLoopDecoderQatError,
    _emit_progress,
    _evaluate_decoder,
    _nes_pair_robust_objective,
    _pack_receiver_archive,
    _PreparedState,
    decoder_eval_pair_deltas,
    decoder_search_direction_labels,
    decoder_trial_passes_pose_guard,
    quantize_decoder_for_qat,
    run_snerv_scorer_loop_decoder_qat,
)


def test_quantize_decoder_for_qat_changes_off_grid_weights_and_preserves_shape() -> None:
    decoder = _decoder()

    quantized, stats = quantize_decoder_for_qat(decoder, bits=4)

    assert quantized.levels == decoder.levels
    assert stats.bits == 4
    assert stats.total_weights == 27
    assert stats.payload_bytes_fp32_receiver > 0
    assert len(stats.payload_sha256_fp32_receiver) == 64
    assert stats.max_abs_error > 0.0
    assert quantized.kernels[0]["LH"].shape == (3, 3)
    assert not np.array_equal(quantized.kernels[0]["LH"], decoder.kernels[0]["LH"])


def test_first_class_qat_runner_preserves_false_authority_wrapper(monkeypatch) -> None:
    calls = {}

    def fake_smoke(**kwargs):
        calls.update(kwargs)
        return "sentinel_result"

    monkeypatch.setattr(
        "tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat."
        "run_snerv_scorer_loop_decoder_qat_smoke",
        fake_smoke,
    )

    result = run_snerv_scorer_loop_decoder_qat(n_pairs=2, wavelet="haar")

    assert result == "sentinel_result"
    assert calls == {"n_pairs": 2, "wavelet": "haar"}


def test_scorer_loop_progress_callback_emits_eval_row() -> None:
    rows = []
    row = _eval(label="progress_row", score=6.5, d_pose=0.2, replay=True)

    _emit_progress(rows.append, row)
    _emit_progress(None, row)

    assert rows == [row]


def test_scorer_loop_cli_progress_jsonl_writer_is_fail_closed(tmp_path) -> None:
    from tools.run_snerv_scorer_loop_decoder_qat_smoke import (
        _build_progress_callback,
    )

    progress_path = tmp_path / "progress.jsonl"
    callback = _build_progress_callback(str(progress_path))
    assert callback is not None

    callback(_eval(label="cli_progress_row", score=6.5, d_pose=0.2, replay=True))

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "snerv_scorer_loop_decoder_qat_progress.v1"
    assert payload["axis_tag"] == "[macOS-CPU advisory]"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["row"]["label"] == "cli_progress_row"


def test_prepare_state_disables_score_exact_saliency_diagnostics_hot_loop() -> None:
    src = inspect.getsource(qat_mod._prepare_state)

    assert "compute_s_seg_flip_risk(segnet, pairs[pair_idx], diagnostics=False)" in src
    assert "compute_s_pose_fisher(posenet, pairs[pair_idx], diagnostics=False)" in src


def test_prepare_state_projects_pair_saliency_once_per_pair(monkeypatch) -> None:
    calls: list[tuple[tuple[int, int], int]] = []
    pairs = torch.zeros((2, 2, 3, 4, 4), dtype=torch.float32)

    monkeypatch.setattr(qat_mod, "decode_real_pairs", lambda *_args, **_kwargs: pairs)
    monkeypatch.setattr(
        qat_mod,
        "encode_frame_lf",
        lambda *_args, **_kwargs: SimpleNamespace(
            lf=np.ones((2, 2), dtype=np.float32)
        ),
    )
    monkeypatch.setattr(
        qat_mod,
        "fit_hf_decoder_least_squares",
        lambda *_args, **_kwargs: _decoder(),
    )
    monkeypatch.setattr(
        qat_mod,
        "compute_s_seg_flip_risk",
        lambda *_args, **_kwargs: SimpleNamespace(
            flip_risk=torch.ones((4, 4), dtype=torch.float32)
        ),
    )
    monkeypatch.setattr(
        qat_mod,
        "compute_s_pose_fisher",
        lambda *_args, **_kwargs: SimpleNamespace(
            s_pose=torch.full((4, 4), 2.0, dtype=torch.float32)
        ),
    )

    def fake_push_pixel_saliency_to_lf(*_args, carrier_hw, levels, **_kwargs):
        calls.append((tuple(carrier_hw), int(levels)))
        return np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(
        qat_mod,
        "push_pixel_saliency_to_lf",
        fake_push_pixel_saliency_to_lf,
    )
    monkeypatch.setattr(
        qat_mod,
        "allocate_lf_linf",
        lambda *_args, **_kwargs: SimpleNamespace(
            steps=np.ones((2, 2), dtype=np.float32)
        ),
    )
    monkeypatch.setattr(
        qat_mod,
        "quantize_lf",
        lambda lf, **_kwargs: (
            np.zeros_like(lf, dtype=np.int64),
            1.0,
            0.0,
        ),
    )

    state = qat_mod._prepare_state(
        posenet=object(),
        segnet=object(),
        n_pairs=2,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=1.0,
        pair_stride=1,
        start_pair=0,
        video_path="unused.mkv",
        device="cpu",
        step_map_bins=4,
        model_size=_decoder().model_size,
    )

    assert calls == [((4, 4), 1), ((4, 4), 1)]
    assert len(state.step_maps) == 12


def test_qat_receiver_codec_pricing_proof_is_backed_by_archive_byte_path(
    monkeypatch,
) -> None:
    from tac.substrates.snerv_inverse_steg_carrier import scorer_loop_decoder_qat

    assert SNERV_QAT_RECEIVER_CODEC_PRICING_PROOF
    assert SNERV_QAT_SECTION_VALUE_PRESSURE_PROOF

    monkeypatch.setattr(
        scorer_loop_decoder_qat,
        "measure_pair_d_seg_d_pose",
        lambda *_args, **_kwargs: (0.125, 0.09),
    )
    step_maps = tuple(np.ones((2, 2), dtype=np.float32) for _ in range(6))
    step_packet = encode_step_maps(list(step_maps), bins=4).packet
    prepared = _PreparedState(
        pairs=torch.zeros((1, 2, 3, 4, 4), dtype=torch.float32),
        codes=(),
        lf_quant_planes=tuple(np.zeros((2, 2), dtype=np.int64) for _ in range(6)),
        lf_zero_points=tuple(0.0 for _ in range(6)),
        step_maps=step_maps,
        step_map_packet=step_packet,
        baseline_decoder=_decoder(),
        model_size=_decoder().model_size,
        levels=1,
        wavelet="haar",
        orig_hw=(4, 4),
    )

    row = _evaluate_decoder(
        _decoder(),
        prepared=prepared,
        posenet=object(),
        segnet=object(),
        qat_bits=8,
        decoder_payload_codec="int4_symmetric",
        label="receiver_priced_eval",
        iteration=1,
        accepted=False,
        byte_pressure_multiplier=3.0,
        section_value_pressure_multiplier=2.0,
    )

    assert row.receiver_archive_replay_verified is True
    assert row.section_value_pressure_ready is True
    assert len(row.section_value_neutralizations) == 2
    assert row.section_value_pressure_linf >= 0.0
    assert row.rate_term == pytest.approx(CONTEST_BYTE_PRICE * row.archive_bytes)
    assert row.score_linf == pytest.approx(12.5 + np.sqrt(0.9) + row.rate_term)
    assert row.rate_aware_objective_linf == pytest.approx(
        row.score_linf + 2.0 * row.rate_term + 2.0 * row.section_value_pressure_linf
    )


def test_evaluate_decoder_scores_section_value_pressure_during_qat(
    monkeypatch,
) -> None:
    from tac.substrates.snerv_inverse_steg_carrier import scorer_loop_decoder_qat

    responses = iter(
        [
            (0.01, 0.01),
            (0.40, 0.40),
            (0.01, 0.01),
        ]
    )
    monkeypatch.setattr(
        scorer_loop_decoder_qat,
        "measure_pair_d_seg_d_pose",
        lambda *_args, **_kwargs: next(responses),
    )
    step_maps = tuple(
        np.linspace(1.0, 2.0 + idx, 4, dtype=np.float32).reshape(2, 2)
        for idx in range(6)
    )
    prepared = _PreparedState(
        pairs=torch.zeros((1, 2, 3, 4, 4), dtype=torch.float32),
        codes=(),
        lf_quant_planes=tuple(np.zeros((2, 2), dtype=np.int64) for _ in range(6)),
        lf_zero_points=tuple(0.0 for _ in range(6)),
        step_maps=step_maps,
        step_map_packet=encode_step_maps(list(step_maps), bins=4).packet,
        baseline_decoder=_decoder(),
        model_size=_decoder().model_size,
        levels=1,
        wavelet="haar",
        orig_hw=(4, 4),
        step_map_bins=4,
    )

    row = _evaluate_decoder(
        _decoder(),
        prepared=prepared,
        posenet=object(),
        segnet=object(),
        qat_bits=8,
        decoder_payload_codec="int4_symmetric",
        label="section_value_bound_eval",
        iteration=1,
        accepted=False,
        byte_pressure_multiplier=1.0,
        section_value_pressure_multiplier=2.0,
    )

    sections = {item.section: item for item in row.section_value_neutralizations}
    assert row.section_value_pressure_ready is True
    assert set(sections) == {"decoder_payload", "step_map_packet"}
    assert sections["decoder_payload"].economic_surplus_linf > 0.0
    assert sections["decoder_payload"].pressure_linf == 0.0
    assert sections["step_map_packet"].delta_nonrate_score == pytest.approx(0.0)
    assert sections["step_map_packet"].pressure_linf == pytest.approx(
        sections["step_map_packet"].section_rate_cost_linf
    )
    assert row.section_value_pressure_linf == pytest.approx(
        sections["step_map_packet"].section_rate_cost_linf
    )
    assert row.rate_aware_objective_linf == pytest.approx(
        row.score_linf + 2.0 * row.section_value_pressure_linf
    )


def test_pack_receiver_archive_records_scorer_loop_adapter_config() -> None:
    cfg = SnervModelSizeConfig(
        fc_dim=12,
        emb_size=2,
        patch_radius=1,
        mfu_scales=(1, 3),
        hfr_gain=0.25,
        temporal_context=1,
        adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
    )
    step_maps = tuple(np.ones((2, 2), dtype=np.float32) for _ in range(6))
    prepared = _PreparedState(
        pairs=torch.zeros((1, 2, 3, 4, 4), dtype=torch.float32),
        codes=(),
        lf_quant_planes=tuple(np.zeros((2, 2), dtype=np.int64) for _ in range(6)),
        lf_zero_points=tuple(0.0 for _ in range(6)),
        step_maps=step_maps,
        step_map_packet=encode_step_maps(list(step_maps), bins=4).packet,
        baseline_decoder=HfGenerationDecoder.zeros(1, model_size=cfg),
        model_size=cfg,
        levels=1,
        wavelet="haar",
        orig_hw=(4, 4),
    )

    archive = _pack_receiver_archive(prepared, HfGenerationDecoder.zeros(1, model_size=cfg))

    assert archive.metadata["snerv_model_size_adapter"] == SNERV_SPECTRA_PRESERVING_ADAPTER
    assert archive.metadata["snerv_spectra_preserving_adapter_enabled"] is True
    assert archive.metadata["snerv_mfu_scales"] == [1, 3]
    assert archive.metadata["snerv_hfr_gain"] == pytest.approx(0.25)
    assert archive.metadata["snerv_temporal_context"] == 1
    assert archive.metadata["decoder_payload_codec"] == "float32_lzma"
    assert archive.metadata["decoder_feature_count"] == cfg.feature_count


def test_component_guard_modes_are_validated() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=2.0, d_seg=0.02, replay=True)
    candidate = _eval(label="candidate", score=6.8, d_pose=1.9, d_seg=0.01, replay=True)

    assert COMPONENT_GUARD_MODES == ("score_primary", "pose_hard", "pose_seg_hard")
    with pytest.raises(SnervScorerLoopDecoderQatError, match="component_guard_mode"):
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            component_guard_mode="not_a_mode",
        )


def test_decoder_trial_score_primary_allows_score_gain_with_pose_tradeoff() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=2.0, d_seg=0.02, replay=True)
    candidate = _eval(
        label="lower_score_pose_damage",
        score=6.8,
        d_pose=2.1,
        d_seg=0.01,
        replay=True,
    )

    assert decoder_trial_passes_pose_guard(candidate, current, pose_slack=0.0) is True
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            pose_slack=0.0,
            component_guard_mode="pose_hard",
        )
        is False
    )
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            pose_slack=0.2,
            component_guard_mode="pose_hard",
        )
        is True
    )


def test_decoder_trial_score_primary_allows_score_gain_with_seg_tradeoff() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=2.0, d_seg=0.01, replay=True)
    candidate = _eval(
        label="lower_score_seg_damage",
        score=6.8,
        d_pose=1.9,
        d_seg=0.011,
        replay=True,
    )

    assert decoder_trial_passes_pose_guard(candidate, current, pose_slack=0.0) is True
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            pose_slack=0.0,
            component_guard_mode="pose_seg_hard",
        )
        is False
    )


def test_decoder_trial_pose_guard_can_allow_explicit_seg_slack() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=2.0, d_seg=0.01, replay=True)
    candidate = _eval(
        label="lower_score_pose_safe_seg_slack",
        score=6.8,
        d_pose=1.9,
        d_seg=0.011,
        replay=True,
    )

    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            component_guard_mode="pose_seg_hard",
        )
        is False
    )
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            seg_slack=0.001,
            component_guard_mode="pose_seg_hard",
        )
        is True
    )


def test_decoder_trial_pose_guard_requires_receiver_replay() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=2.0, d_seg=0.02, replay=True)
    candidate = _eval(label="no_replay", score=6.0, d_pose=1.0, d_seg=0.01, replay=False)

    assert decoder_trial_passes_pose_guard(candidate, current) is False


def test_decoder_trial_pose_guard_can_hard_block_archive_byte_growth() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.2,
        d_seg=0.01,
        replay=True,
        archive_bytes=1000,
        rate_term=0.010,
    )
    candidate = _eval(
        label="lower_score_bigger_archive",
        score=6.9,
        d_pose=0.19,
        d_seg=0.009,
        replay=True,
        archive_bytes=1001,
        rate_term=0.011,
    )

    assert decoder_trial_passes_pose_guard(candidate, current) is True
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            max_archive_byte_growth=0,
        )
        is False
    )


def test_decoder_trial_pose_guard_uses_byte_pressure_objective() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.2,
        d_seg=0.01,
        replay=True,
        archive_bytes=1000,
        rate_term=0.010,
    )
    candidate = _eval(
        label="tiny_score_gain_large_rate_growth",
        score=6.99,
        d_pose=0.19,
        d_seg=0.009,
        replay=True,
        archive_bytes=5000,
        rate_term=1.000,
    )

    assert decoder_trial_passes_pose_guard(candidate, current) is True
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            byte_pressure_multiplier=8.0,
        )
        is False
    )


def test_decoder_trial_pose_guard_uses_section_value_pressure_objective() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.2,
        d_seg=0.01,
        replay=True,
        archive_bytes=1000,
        rate_term=0.010,
        section_value_pressure=0.0,
    )
    candidate = _eval(
        label="tiny_score_gain_low_value_sections",
        score=6.99,
        d_pose=0.19,
        d_seg=0.009,
        replay=True,
        archive_bytes=1000,
        rate_term=0.010,
        section_value_pressure=1.0,
    )

    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            section_value_pressure_multiplier=0.0,
        )
        is True
    )
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            section_value_pressure_multiplier=1.0,
        )
        is False
    )


def test_decoder_trial_pose_guard_fails_closed_without_section_value_binding() -> None:
    current = _eval(label="baseline", score=7.0, d_pose=0.2, replay=True)
    candidate = _eval(
        label="score_gain_missing_section_binding",
        score=6.8,
        d_pose=0.19,
        replay=True,
        section_value_pressure_ready=False,
    )

    assert decoder_trial_passes_pose_guard(candidate, current) is False


def test_decoder_trial_pose_guard_never_accepts_raw_score_regression() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.2,
        d_seg=0.01,
        replay=True,
        archive_bytes=1000,
        rate_term=0.100,
    )
    candidate = _eval(
        label="lower_rate_but_raw_score_regresses",
        score=7.1,
        d_pose=0.19,
        d_seg=0.009,
        replay=True,
        archive_bytes=100,
        rate_term=0.001,
    )

    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            byte_pressure_multiplier=8.0,
        )
        is False
    )


def test_decoder_trial_pose_guard_can_require_pair_robust_score_improvement() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.4,
        d_seg=0.02,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.01,
                d_pose_linf=0.3,
                score_linf_without_rate=2.0,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.03,
                d_pose_linf=0.5,
                score_linf_without_rate=5.0,
            ),
        ),
    )
    candidate = _eval(
        label="aggregate_win_pair_cancellation",
        score=6.9,
        d_pose=0.39,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.011,
                d_pose_linf=0.29,
                score_linf_without_rate=2.1,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.027,
                d_pose_linf=0.49,
                score_linf_without_rate=4.5,
            ),
        ),
    )

    assert decoder_trial_passes_pose_guard(candidate, current) is True
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            pair_guard_min_score_improved_fraction=1.0,
        )
        is False
    )


def test_decoder_trial_pose_guard_can_limit_pair_pose_worsening() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.4,
        d_seg=0.02,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.01,
                d_pose_linf=0.3,
                score_linf_without_rate=2.0,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.03,
                d_pose_linf=0.5,
                score_linf_without_rate=5.0,
            ),
        ),
    )
    candidate = _eval(
        label="aggregate_win_pose_cancellation",
        score=6.9,
        d_pose=0.39,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.009,
                d_pose_linf=0.31,
                score_linf_without_rate=1.9,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.029,
                d_pose_linf=0.47,
                score_linf_without_rate=4.8,
            ),
        ),
    )

    assert decoder_trial_passes_pose_guard(candidate, current) is True
    assert (
        decoder_trial_passes_pose_guard(
            candidate,
            current,
            pair_guard_max_pose_worsened_fraction=0.0,
        )
        is False
    )


def test_decoder_eval_json_preserves_pair_local_detector_response() -> None:
    row = _eval(
        label="candidate",
        score=6.0,
        d_pose=0.5,
        d_seg=0.01,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.01,
                d_pose_linf=0.5,
                score_linf_without_rate=3.23606797749979,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.02,
                d_pose_linf=0.25,
                score_linf_without_rate=3.58113883008419,
            ),
        ),
    )

    payload = row.as_jsonable()

    assert payload["rate_aware_objective_linf"] == pytest.approx(6.0)
    assert payload["per_pair"] == [
        {
            "pair_index": 0,
            "d_seg_linf": 0.01,
            "d_pose_linf": 0.5,
            "score_linf_without_rate": 3.23606797749979,
        },
        {
            "pair_index": 1,
            "d_seg_linf": 0.02,
            "d_pose_linf": 0.25,
            "score_linf_without_rate": 3.58113883008419,
        },
    ]


def test_decoder_eval_pair_deltas_preserve_direction_per_pair() -> None:
    baseline = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.4,
        d_seg=0.02,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.01,
                d_pose_linf=0.3,
                score_linf_without_rate=2.7,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.03,
                d_pose_linf=0.5,
                score_linf_without_rate=5.0,
            ),
        ),
    )
    candidate = _eval(
        label="candidate",
        score=6.9,
        d_pose=0.39,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.009,
                d_pose_linf=0.25,
                score_linf_without_rate=2.1,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.031,
                d_pose_linf=0.53,
                score_linf_without_rate=5.2,
            ),
        ),
    )

    deltas = decoder_eval_pair_deltas(baseline, candidate)

    assert [row.as_jsonable() for row in deltas] == [
        {
            "pair_index": 0,
            "d_seg_linf_delta": pytest.approx(-0.001),
            "d_pose_linf_delta": pytest.approx(-0.05),
            "score_linf_without_rate_delta": pytest.approx(-0.6),
        },
        {
            "pair_index": 1,
            "d_seg_linf_delta": pytest.approx(0.001),
            "d_pose_linf_delta": pytest.approx(0.03),
            "score_linf_without_rate_delta": pytest.approx(0.2),
        },
    ]


def test_decoder_eval_pair_deltas_fail_closed_on_pair_mismatch() -> None:
    baseline = _eval(label="baseline", score=7.0, d_pose=0.4, d_seg=0.02, replay=True)
    candidate = _eval(
        label="candidate",
        score=6.9,
        d_pose=0.39,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=9,
                d_seg_linf=0.009,
                d_pose_linf=0.25,
                score_linf_without_rate=2.1,
            ),
        ),
    )

    with pytest.raises(SnervScorerLoopDecoderQatError, match="missing from baseline"):
        decoder_eval_pair_deltas(baseline, candidate)


def test_quantize_decoder_rejects_invalid_bit_width() -> None:
    with pytest.raises(SnervScorerLoopDecoderQatError, match="bits"):
        quantize_decoder_for_qat(_decoder(), bits=1)


def test_top_weight_coordinate_direction_labels_are_deterministic() -> None:
    labels = decoder_search_direction_labels(
        np.array([0.1, -4.0, 3.0, 0.2]),
        max_trials=2,
        search_mode="top_weight_coordinate",
    )

    assert labels == ("coord_001", "coord_002")


def test_learned_random_subspace_direction_labels_are_deterministic() -> None:
    labels = decoder_search_direction_labels(
        np.array([0.1, -4.0, 3.0, 0.2]),
        max_trials=3,
        search_mode="learned_random_subspace",
        seed=99,
    )

    assert labels == (
        "learned_subspace_001",
        "learned_subspace_002",
        "learned_subspace_003",
    )


def test_nes_pair_robust_direction_labels_are_deterministic() -> None:
    labels = decoder_search_direction_labels(
        np.array([0.1, -4.0, 3.0, 0.2]),
        max_trials=3,
        search_mode="nes_pair_robust",
        seed=99,
    )

    assert labels == ("nes_probe_001", "nes_probe_002", "nes_probe_003")


def test_nes_pair_robust_objective_penalizes_pair_pose_damage() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.40,
        d_seg=0.020,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.010,
                d_pose_linf=0.20,
                score_linf_without_rate=4.0,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.020,
                d_pose_linf=0.40,
                score_linf_without_rate=3.0,
            ),
        ),
    )
    pair_robust = _eval(
        label="pair_robust",
        score=6.8,
        d_pose=0.39,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.009,
                d_pose_linf=0.19,
                score_linf_without_rate=3.8,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.019,
                d_pose_linf=0.39,
                score_linf_without_rate=2.9,
            ),
        ),
    )
    pose_damaged = _eval(
        label="pose_damaged",
        score=6.7,
        d_pose=0.41,
        d_seg=0.019,
        replay=True,
        per_pair=(
            SnervPairEval(
                pair_index=0,
                d_seg_linf=0.009,
                d_pose_linf=0.19,
                score_linf_without_rate=3.8,
            ),
            SnervPairEval(
                pair_index=1,
                d_seg_linf=0.019,
                d_pose_linf=0.42,
                score_linf_without_rate=2.8,
            ),
        ),
    )

    assert _nes_pair_robust_objective(
        pair_robust,
        current,
        pose_slack=0.0,
        seg_slack=0.0,
        pair_guard_min_score_improved_fraction=1.0,
        pair_guard_max_pose_worsened_fraction=0.0,
    ) < _nes_pair_robust_objective(
        pose_damaged,
        current,
        pose_slack=0.0,
        seg_slack=0.0,
        pair_guard_min_score_improved_fraction=1.0,
        pair_guard_max_pose_worsened_fraction=0.0,
    )


def test_nes_pair_robust_objective_penalizes_archive_byte_growth() -> None:
    current = _eval(
        label="baseline",
        score=7.0,
        d_pose=0.20,
        d_seg=0.010,
        replay=True,
        archive_bytes=1000,
        rate_term=0.010,
    )
    rate_safe = _eval(
        label="rate_safe",
        score=6.9,
        d_pose=0.19,
        d_seg=0.009,
        replay=True,
        archive_bytes=1000,
        rate_term=0.010,
    )
    byte_growing = _eval(
        label="byte_growing",
        score=6.8,
        d_pose=0.19,
        d_seg=0.009,
        replay=True,
        archive_bytes=1005,
        rate_term=0.015,
    )

    assert _nes_pair_robust_objective(
        rate_safe,
        current,
        pose_slack=0.0,
        seg_slack=0.0,
        byte_pressure_multiplier=8.0,
        max_archive_byte_growth=0,
        pair_guard_min_score_improved_fraction=0.0,
        pair_guard_max_pose_worsened_fraction=1.0,
    ) < _nes_pair_robust_objective(
        byte_growing,
        current,
        pose_slack=0.0,
        seg_slack=0.0,
        byte_pressure_multiplier=8.0,
        max_archive_byte_growth=0,
        pair_guard_min_score_improved_fraction=0.0,
        pair_guard_max_pose_worsened_fraction=1.0,
    )


def test_decoder_search_direction_labels_rejects_unknown_mode() -> None:
    with pytest.raises(SnervScorerLoopDecoderQatError, match="search_mode"):
        decoder_search_direction_labels(
            np.array([1.0, 2.0]),
            max_trials=1,
            search_mode="not_a_mode",
        )


def _decoder() -> HfGenerationDecoder:
    base = np.array(
        [
            [0.0103, -0.0217, 0.0331],
            [0.0449, -0.0555, 0.0662],
            [0.0777, -0.0888, 0.0999],
        ],
        dtype=np.float64,
    )
    return HfGenerationDecoder(
        kernels={
            0: {
                "LH": base,
                "HL": base * 0.5,
                "HH": -base * 0.25,
            }
        },
        levels=1,
    )


def _eval(
    *,
    label: str,
    score: float,
    d_pose: float,
    replay: bool,
    d_seg: float = 0.01,
    archive_bytes: int = 1234,
    rate_term: float = 0.001,
    section_value_pressure: float = 0.0,
    section_value_pressure_ready: bool = True,
    per_pair: tuple[SnervPairEval, ...] | None = None,
) -> SnervDecoderEval:
    if per_pair is None:
        per_pair = (
            SnervPairEval(
                pair_index=0,
                d_seg_linf=float(d_seg),
                d_pose_linf=float(d_pose),
                score_linf_without_rate=100.0 * float(d_seg)
                + float(np.sqrt(10.0 * max(float(d_pose), 0.0))),
            ),
        )
    return SnervDecoderEval(
        label=label,
        iteration=0,
        archive_bytes=int(archive_bytes),
        archive_sha256="0" * 64,
        d_seg_linf=d_seg,
        d_pose_linf=d_pose,
        score_linf=score,
        rate_aware_objective_linf=float(score),
        rate_term=float(rate_term),
        receiver_archive_replay_verified=replay,
        accepted=False,
        blockers=() if replay else ("receiver_archive_replay_failed",),
        quantized_decoder=QuantizedDecoderStats(
            bits=8,
            scale_count=3,
            zero_scale_count=0,
            total_weights=27,
            zero_weight_fraction=0.0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            payload_bytes_fp32_receiver=99,
            payload_sha256_fp32_receiver="1" * 64,
        ),
        per_pair=per_pair,
        section_value_pressure_linf=float(section_value_pressure),
        section_value_pressure_ready=bool(section_value_pressure_ready),
    )
