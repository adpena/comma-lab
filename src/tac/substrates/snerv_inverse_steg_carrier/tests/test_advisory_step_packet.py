# SPDX-License-Identifier: MIT
"""Focused tests for SNeRV advisory compact step-map consumption."""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.advisory import (
    _combine_hf_decoder_saliency,
    _encode_decode_linf_steps_packet,
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
