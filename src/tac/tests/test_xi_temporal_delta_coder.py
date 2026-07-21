# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tac.boundary_math.analytic_lane_render_band import (
    _RD_F_NEAR,
    LaneBandRenderConfig,
    _pack_pairs_to_matrix,
    _quantize_matrix,
    derive_rd_base_steps,
)
from tac.boundary_math.lane_sdf_component import LaneLine
from tac.boundary_math.lane_track_and_smooth import coherent_slot_pack
from tac.optimization.xi_temporal_delta_coder import (
    MAGIC,
    PREFIX,
    XiTemporalDeltaError,
    _predictor_reconstruct,
    decode_lane_xi_temporal,
    decode_lane_xi_temporal_grid,
    encode_lane_xi_temporal,
    encode_quantized_lane_xi_temporal,
    inspect_lane_xi_temporal,
    semantic_lane_sha256,
    semantic_quantized_lane_sha256,
)


def _line(offset: float, dashed: bool) -> LaneLine:
    return LaneLine(
        centerline_coeffs=np.asarray([1e-5, -2e-4, 0.01, offset], dtype=np.float64),
        halfwidth_coeffs=np.asarray([0.001, 1.5], dtype=np.float64),
        dash_period_m=6.0 if dashed else 0.0,
        dash_phase_m=1.25 if dashed else 0.0,
        dash_duty=0.5,
        forward_range=(5.0, 60.0),
    )


def _fixture(pair_count: int = 16) -> tuple[list[list[LaneLine]], np.ndarray]:
    pairs = []
    xi = np.zeros((pair_count, 6), dtype=np.float64)
    for index in range(pair_count):
        pairs.append(
            [
                _line(-1.8 + 0.02 * index, True),
                _line(1.8 - 0.015 * index, False),
            ]
        )
        if index:
            xi[index] = np.asarray(
                [
                    0.01 * np.sin(index),
                    0.002 * np.cos(index),
                    0.6 + 0.01 * index,
                    0.001 * np.sin(index / 2),
                    0.002 * np.cos(index / 3),
                    0.0015 * np.sin(index / 4),
                ]
            )
    return pairs, xi


@pytest.mark.parametrize("predictor", ["identity", "planar3_from_composed_screw"])
def test_roundtrip_is_deterministic_and_semantically_exact(predictor: str) -> None:
    pairs, xi = _fixture()
    steps = derive_rd_base_steps()
    kwargs = {
        "base_steps": steps,
        "f_near": _RD_F_NEAR,
        "predictor": predictor,
        "seed": 1234,
    }
    left = encode_lane_xi_temporal(pairs, LaneBandRenderConfig(), xi, **kwargs)
    right = encode_lane_xi_temporal(pairs, LaneBandRenderConfig(), xi, **kwargs)
    decoded, header = decode_lane_xi_temporal(left.payload)

    assert left.payload == right.payload
    assert left.payload.startswith(MAGIC)
    assert header["predictor"] == predictor
    assert header["xi"]["representation"] == "corrected_composed_full_screw_translation_first"
    assert semantic_lane_sha256(decoded, steps, _RD_F_NEAR) == semantic_lane_sha256(pairs, steps, _RD_F_NEAR)


def test_strict_outer_digest_and_trailing_byte_rejection() -> None:
    pairs, xi = _fixture(8)
    artifact = encode_lane_xi_temporal(
        pairs,
        LaneBandRenderConfig(),
        xi,
        base_steps=derive_rd_base_steps(),
        f_near=_RD_F_NEAR,
    )
    corrupted = bytearray(artifact.payload)
    corrupted[len(corrupted) // 2] ^= 1
    with pytest.raises(XiTemporalDeltaError, match="digest"):
        decode_lane_xi_temporal(bytes(corrupted))
    with pytest.raises(XiTemporalDeltaError, match="digest"):
        decode_lane_xi_temporal(artifact.payload + b"x")


def test_inspection_validates_and_exposes_counted_entropy_segments() -> None:
    pairs, xi = _fixture(12)
    artifact = encode_lane_xi_temporal(
        pairs,
        LaneBandRenderConfig(),
        xi,
        base_steps=derive_rd_base_steps(),
        f_near=_RD_F_NEAR,
    )
    header = inspect_lane_xi_temporal(artifact.payload)
    assert header["entropy"]["backend"].startswith("tac.shared_pmf_model")
    assert header["context"]["selected_bins"] in header["context"]["candidate_bins"]
    assert header["segments"]["xi"]["bytes"] == artifact.xi_payload_bytes
    assert header["segments"]["shared_pmf_model"]["bytes"] == artifact.model_bytes
    assert header["segments"]["range_payload"]["bytes"] == artifact.range_payload_bytes


def test_rejects_mismatched_xi_coverage() -> None:
    pairs, xi = _fixture(10)
    with pytest.raises(XiTemporalDeltaError, match="same positive pair count"):
        encode_lane_xi_temporal(
            pairs,
            LaneBandRenderConfig(),
            xi[:-1],
            base_steps=derive_rd_base_steps(),
            f_near=_RD_F_NEAR,
        )


@pytest.mark.parametrize("predictor", ["identity", "planar3_from_composed_screw"])
def test_predictors_reject_signed_int64_overflow(predictor: str) -> None:
    steps = derive_rd_base_steps()
    q_lane = np.zeros((2, len(steps)), dtype=np.int64)
    q_lane[0, 0] = np.iinfo(np.int64).min
    q_lane[1, 0] = np.iinfo(np.int64).max
    presence = np.ones((2, 1), dtype=bool)
    xi = np.zeros((2, 6), dtype=np.float64)
    with pytest.raises(XiTemporalDeltaError, match="innovation exceeds signed int64"):
        encode_quantized_lane_xi_temporal(
            q_lane,
            presence,
            LaneBandRenderConfig(),
            xi,
            base_steps=steps,
            f_near=_RD_F_NEAR,
            predictor=predictor,  # type: ignore[arg-type]
        )

    innovation = np.zeros_like(q_lane)
    innovation[0, 0] = np.iinfo(np.int64).max
    innovation[1, 0] = 1
    with pytest.raises(XiTemporalDeltaError, match="reconstruction exceeds signed int64"):
        _predictor_reconstruct(
            innovation,
            presence,
            np.tile(steps, 1),
            xi,
            1,
            predictor,  # type: ignore[arg-type]
        )


def _rewrite_header(payload: bytes, mutate: object) -> bytes:
    _magic, _version, header_size = PREFIX.unpack_from(payload)
    start = PREFIX.size
    end = start + header_size
    header = json.loads(payload[start:end].decode("ascii"))
    mutate(header)  # type: ignore[operator]
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    without_digest = PREFIX.pack(MAGIC, 1, len(encoded)) + encoded + payload[end:-32]
    return without_digest + hashlib.sha256(without_digest).digest()


def _corrupt_redigested_segment(
    payload: bytes,
    name: str,
    *,
    relative_offset: int = 0,
    replacement: int | None = None,
) -> bytes:
    _magic, _version, header_size = PREFIX.unpack_from(payload)
    start = PREFIX.size
    end = start + header_size
    header = json.loads(payload[start:end].decode("ascii"))
    body = bytearray(payload[end:-32])
    segment_order = ("xi", "presence", "shared_pmf_model", "range_payload")
    offset = sum(header["segments"][prior]["bytes"] for prior in segment_order[: segment_order.index(name)])
    size = header["segments"][name]["bytes"]
    assert 0 <= relative_offset < size
    absolute_offset = offset + relative_offset
    body[absolute_offset] = body[absolute_offset] ^ 1 if replacement is None else replacement
    header["segments"][name]["sha256"] = hashlib.sha256(body[offset : offset + size]).hexdigest()
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    without_digest = PREFIX.pack(MAGIC, 1, len(encoded)) + encoded + body
    return without_digest + hashlib.sha256(without_digest).digest()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda h: h.pop("pair_count"),
        lambda h: h.__setitem__("pair_count", 16.9),
        lambda h: h.__setitem__("slot_count", 2.9),
        lambda h: h.__setitem__("seed", 1234.9),
        lambda h: h["context"].__setitem__("selected_bins", True),
        lambda h: h["entropy"].__setitem__("n_models", 1.9),
        lambda h: h["entropy"].__setitem__(
            "estimated_range_payload_bytes", h["entropy"]["estimated_range_payload_bytes"] + 1
        ),
        lambda h: h["xi"].__setitem__("q_levels", 4096.0),
        lambda h: h["render"].__setitem__("weight_semantics", "unchecked"),
        lambda h: h["rd"].__setitem__("f_near", 16.0),
        lambda h: h.__setitem__("receiver_status", "UNVERIFIED"),
    ],
)
def test_rejects_canonical_redigested_header_contract_mutations(mutate: object) -> None:
    pairs, xi = _fixture(16)
    artifact = encode_lane_xi_temporal(
        pairs,
        LaneBandRenderConfig(),
        xi,
        base_steps=derive_rd_base_steps(),
        f_near=_RD_F_NEAR,
    )
    with pytest.raises(XiTemporalDeltaError):
        decode_lane_xi_temporal(_rewrite_header(artifact.payload, mutate))


def test_rejects_redigested_malformed_shared_pmf_as_domain_error() -> None:
    pairs, xi = _fixture(16)
    artifact = encode_lane_xi_temporal(
        pairs,
        LaneBandRenderConfig(),
        xi,
        base_steps=derive_rd_base_steps(),
        f_near=_RD_F_NEAR,
    )
    with pytest.raises(XiTemporalDeltaError, match="decode failed"):
        decode_lane_xi_temporal_grid(_corrupt_redigested_segment(artifact.payload, "shared_pmf_model"))


def test_rejects_redigested_malformed_xi_zlib_as_domain_error() -> None:
    pairs, xi = _fixture(16)
    artifact = encode_lane_xi_temporal(
        pairs,
        LaneBandRenderConfig(),
        xi,
        base_steps=derive_rd_base_steps(),
        f_near=_RD_F_NEAR,
    )
    malformed = _corrupt_redigested_segment(
        artifact.payload,
        "xi",
        relative_offset=32,
        replacement=1,
    )
    with pytest.raises(XiTemporalDeltaError, match="decode failed"):
        decode_lane_xi_temporal_grid(malformed)


def test_coherent_slot_lattice_survives_without_lateral_repack() -> None:
    pairs = [
        [_line(-3.0, False), _line(0.0, False), _line(3.0, False)],
        [_line(0.0, False), _line(3.0, False)],
        [_line(0.0, False), _line(3.0, False)],
    ]
    steps = derive_rd_base_steps()
    coherent = coherent_slot_pack(pairs, f_near=_RD_F_NEAR)
    sort_matrix, sort_presence, sort_slots = _pack_pairs_to_matrix(pairs, f_near=_RD_F_NEAR)
    assert sort_slots == coherent.K
    assert not np.array_equal(coherent.presence, sort_presence)
    steps_full = np.tile(steps, coherent.K)
    q_coherent = _quantize_matrix(coherent.M, steps_full)
    q_sort = _quantize_matrix(sort_matrix, steps_full)
    assert not np.array_equal(q_coherent, q_sort)

    config = LaneBandRenderConfig()
    xi = np.zeros((len(pairs), 6), dtype=np.float64)
    artifact = encode_quantized_lane_xi_temporal(
        q_coherent,
        coherent.presence,
        config,
        xi,
        base_steps=steps,
        f_near=_RD_F_NEAR,
        predictor="identity",
        pack_mode="coherent_slot",
    )
    decoded_q, decoded_presence, header = decode_lane_xi_temporal_grid(artifact.payload)
    assert np.array_equal(decoded_q, q_coherent)
    assert np.array_equal(decoded_presence, coherent.presence)
    assert header["rd"]["pack_mode"] == "coherent_slot"
    assert header["semantic"]["grid_sha256"] == semantic_quantized_lane_sha256(
        q_coherent,
        coherent.presence,
        steps,
        _RD_F_NEAR,
        config,
        pack_mode="coherent_slot",
    )
