# SPDX-License-Identifier: MIT
"""Focused tests for SNeRV advisory compact step-map consumption."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tac.substrates.snerv_inverse_steg_carrier.advisory import (
    CONTEST_BYTE_PRICE,
    _combine_hf_decoder_saliency,
    _encode_decode_linf_steps_packet,
    run_snerv_advisory,
)


def test_advisory_step_packet_is_charged_and_receiver_decoded() -> None:
    maps = _maps()
    packet, decoded = _encode_decode_linf_steps_packet(maps)

    assert packet.total_bytes < packet.fp32_lzma_baseline_bytes
    assert packet.ready_for_exact_eval_dispatch is False
    assert len(decoded) == len(maps)
    for ref, got in zip(maps, decoded, strict=True):
        assert got.shape == ref.shape
        assert np.all(got > 0)
    assert packet.max_relative_error < 0.01


def test_advisory_step_packet_supports_adaptive_constant_groups() -> None:
    maps = _maps()
    packet, decoded = _encode_decode_linf_steps_packet(
        maps,
        mode="adaptive",
        map_importance=np.linspace(0.0, 1.0, len(maps)),
        adaptive_bin_choices=(128, 16, 4),
        constant_importance_quantile=0.25,
    )
    constant_group = next(group for group in packet.groups if group["bins"] == 0)

    assert packet.schema == "snerv_step_map_coder.adaptive.v1"
    assert constant_group["payload_bytes"] == 0
    assert constant_group["map_indices"] == [0, 1]
    assert len(decoded) == len(maps)
    for idx in constant_group["map_indices"]:
        assert np.unique(decoded[idx]).size == 1


def test_advisory_adaptive_step_packet_requires_importance() -> None:
    with pytest.raises(RuntimeError, match="map_importance"):
        _encode_decode_linf_steps_packet(_maps(), mode="adaptive")


def test_advisory_charges_l2_with_separate_receiver_packet(monkeypatch) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod

    yy, xx = torch.meshgrid(
        torch.arange(16, dtype=torch.float32),
        torch.arange(24, dtype=torch.float32),
        indexing="ij",
    )
    pairs = torch.stack(
        [
            torch.stack(
                [
                    yy + xx + frame,
                    yy * 0.5 + frame,
                    xx * 0.25 + frame,
                ]
            )
            for frame in range(2)
        ]
    ).unsqueeze(0)

    saliency_diagnostics: list[bool | None] = []

    def fake_seg(_segnet, _pair, *, diagnostics=None):
        saliency_diagnostics.append(diagnostics)
        field = torch.linspace(0.1, 2.0, 16 * 24, dtype=torch.float32).reshape(16, 24)
        return SimpleNamespace(flip_risk=field)

    def fake_pose(_posenet, _pair, *, diagnostics=None):
        saliency_diagnostics.append(diagnostics)
        field = torch.linspace(2.0, 0.1, 16 * 24, dtype=torch.float32).reshape(16, 24)
        return SimpleNamespace(s_pose=field)

    monkeypatch.setattr(advisory_mod, "load_score_exact_scorers", lambda **_kw: (object(), object()))
    monkeypatch.setattr(advisory_mod, "decode_real_pairs", lambda *_a, **_kw: pairs.clone())
    monkeypatch.setattr(advisory_mod, "compute_s_seg_flip_risk", fake_seg)
    monkeypatch.setattr(advisory_mod, "compute_s_pose_fisher", fake_pose)
    monkeypatch.setattr(
        advisory_mod,
        "measure_pair_d_seg_d_pose",
        lambda *_a, **_kw: (0.25, 0.04),
        raising=False,
    )
    monkeypatch.setattr(
        advisory_mod,
        "measure_pairs_d_seg_d_pose_batched",
        lambda *_a, **_kw: (
            np.asarray([0.25], dtype=np.float64),
            np.asarray([0.04], dtype=np.float64),
        ),
    )

    result = run_snerv_advisory(
        n_pairs=1,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=2.0,
        video_path="unused",
        upstream_dir=None,
        device="cpu",
        step_map_coder_bins=8,
    )

    expected_l2 = 25.0 + np.sqrt(0.4) + result.rate_term_l2
    assert result.receiver_archive_replay_verified is True
    assert result.receiver_archive_l2_replay_verified is True
    assert result.archive_bytes_total_l2 == result.receiver_archive_l2_packet_bytes
    assert result.rate_term_l2 == pytest.approx(
        CONTEST_BYTE_PRICE * result.archive_bytes_total_l2
    )
    assert result.score_l2 == pytest.approx(expected_l2)
    assert saliency_diagnostics
    assert set(saliency_diagnostics) == {False}


def test_hf_decoder_saliency_component_selector_is_explicit() -> None:
    seg = np.array([[1.0, 2.0], [3.0, 4.0]])
    pose = np.array([[10.0]])

    combined = _combine_hf_decoder_saliency(
        seg,
        pose,
        component="combined",
        target_hw=(2, 2),
    )
    seg_only = _combine_hf_decoder_saliency(
        seg,
        pose,
        component="seg",
        target_hw=(2, 2),
    )
    pose_only = _combine_hf_decoder_saliency(
        seg,
        pose,
        component="pose",
        target_hw=(2, 2),
    )

    np.testing.assert_array_equal(seg_only, seg)
    np.testing.assert_array_equal(pose_only, np.full((2, 2), 10.0))
    np.testing.assert_array_equal(combined, seg + 10.0)
    with pytest.raises(RuntimeError, match="hf_decoder_saliency_component"):
        _combine_hf_decoder_saliency(
            seg,
            pose,
            component="bad",
            target_hw=(2, 2),
        )


def _maps() -> list[np.ndarray]:
    yy, xx = np.mgrid[0:48, 0:64].astype(np.float32)
    return [
        np.exp2(0.5 + 0.16 * np.sin(xx / 11.0) + i * 0.015)
        for i in range(8)
    ]
