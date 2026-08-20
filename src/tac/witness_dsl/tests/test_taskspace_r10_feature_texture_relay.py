from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from tac.boundary_math.dash_phase_carrier import (
    DashPhaseConfig,
    encode_dash_phase_carrier,
)
from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom
from tac.boundary_math.xi_pose_coder import quantize_xi, serialize_xi_payload
from tac.witness_dsl.taskspace_r10_feature_texture_relay import (
    R10_CONSTRAINTS,
    R10BaseFeatureRecordV1,
    R10IdentityV1,
    R10PacketV1,
    R10PullbackPolygonV1,
    R10RelayError,
    R10RelayMode,
    R10Section,
    R10ShootingKnotV1,
    R10StratifiedFlowV1,
    R10TextureRecordV1,
    build_r10_decode_receipt,
    build_r10_selected_solution_adapter,
    decode_r10_packet,
    packet_physical_spans,
    pair_population_sha256,
    parse_r10_packet,
    r10_control_channel_energy_shuffle,
    r10_control_delete_shooting_knots,
    r10_control_delete_texture,
    r10_control_merge_shooting_knots,
    r10_control_mode,
    r10_control_phase_time_permutation,
    realization_sha256,
    serialize_r10_packet,
)


def _fixture() -> tuple[R10PacketV1, bytes, np.ndarray]:
    pairs, height, width = 2, 32, 48
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.uint16),
        np.arange(width, dtype=np.uint16),
        indexing="ij",
    )
    base = np.empty((pairs, 2, height, width, 3), dtype=np.uint8)
    for pair in range(pairs):
        for frame in range(2):
            base[pair, frame, ..., 0] = (3 * xx + 2 * yy + 11 * pair + frame) % 256
            base[pair, frame, ..., 1] = (xx + 5 * yy + 7 * pair + 3 * frame) % 256
            base[pair, frame, ..., 2] = (4 * xx + yy + 5 * pair + 9 * frame) % 256
    xi = np.zeros((pairs, 6), dtype=np.float64)
    xi[:, 0] = (0.0005, 0.0008)
    xi[:, 5] = (0.0002, 0.0003)
    q, scales = quantize_xi(xi, q_levels=2048)
    xip2 = serialize_xi_payload(q, scales, coder="none")
    lstars = np.zeros((pairs, height, width), dtype=np.uint8)
    lstars[0, 14:19, 11:15] = 1
    lstars[1, 14:19, 13:17] = 1
    cfg = DashPhaseConfig(
        min_area=3,
        border_px=1,
        match_radius_px=10.0,
        q_px=1.0,
        pitch=0.0,
        include_xi=False,
    )
    geom = GroundHomographyGeom.eon(native_hw=(height, width), pitch=0.0)
    dash, _report, _decoded = encode_dash_phase_carrier(lstars, xi, cfg, geom=geom)
    source_sha = hashlib.sha256(b"r10-test-source").hexdigest()
    population_sha = pair_population_sha256(source_sha, tuple(range(pairs)))
    packet = R10PacketV1(
        mode=R10RelayMode.JOINT,
        identity=R10IdentityV1(source_sha, population_sha, realization_sha256(base)),
        pair_indices=tuple(range(pairs)),
        height=height,
        width=width,
        pitch_q20=0,
        base_features=(
            R10BaseFeatureRecordV1(64, 48, 24, 32, (256, 192, 128)),
            R10BaseFeatureRecordV1(-32, 56, 16, 40, (224, 160, 96)),
        ),
        xip2_payload=xip2,
        textures=(
            R10TextureRecordV1(768, 96, 128, 320),
            R10TextureRecordV1(640, 112, 256, 288),
        ),
        shooting_knots=(
            R10ShootingKnotV1(0, 32, 16, 8, 64, 32, 32),
            R10ShootingKnotV1(1, -16, 24, 12, 32, -16, 48),
        ),
        dash1_payload=dash,
        pullback_polygons=(
            R10PullbackPolygonV1(
                0,
                ((7000, 12000), (26000, 12000), (26000, 27000), (7000, 27000)),
            ),
        ),
        stratified_flows=(R10StratifiedFlowV1(0, (8, 0, 0, 8, 16, -8)),),
    )
    return packet, serialize_r10_packet(packet), base


def _decode(packet_bytes: bytes, base: np.ndarray) -> np.ndarray:
    packet = parse_r10_packet(packet_bytes)
    return decode_r10_packet(
        packet_bytes,
        base,
        expected_source_sha256=packet.identity.source_sha256,
        expected_pair_population_sha256=packet.identity.pair_population_sha256,
    )


def test_r10_packet_executes_uint8_and_strictly_roundtrips() -> None:
    packet, packet_bytes, base = _fixture()
    parsed = parse_r10_packet(packet_bytes)
    assert parsed == packet
    assert serialize_r10_packet(parsed) == packet_bytes
    output_a = _decode(packet_bytes, base)
    output_b = _decode(packet_bytes, base)
    assert output_a.dtype == np.uint8
    assert output_a.shape == base.shape
    assert np.array_equal(output_a, output_b)
    assert not np.array_equal(output_a, base)
    spans = packet_physical_spans(packet_bytes)
    assert sum(row.byte_length for row in spans) < len(packet_bytes)
    assert {row.section for row in spans} == {item.name for item in R10Section}


def test_identity_corruption_and_spatial_nonidentifiability_fail_closed() -> None:
    packet, packet_bytes, base = _fixture()
    with pytest.raises(R10RelayError, match="source identity"):
        decode_r10_packet(
            packet_bytes,
            base,
            expected_source_sha256="0" * 64,
            expected_pair_population_sha256=packet.identity.pair_population_sha256,
        )
    drifted = base.copy()
    drifted[0, 0, 0, 0, 0] ^= np.uint8(1)
    with pytest.raises(R10RelayError, match="base realized-pair bytes drifted"):
        _decode(packet_bytes, drifted)
    with pytest.raises(R10RelayError, match="trailing bytes"):
        parse_r10_packet(packet_bytes + b"x")
    corrupt = bytearray(packet_bytes)
    corrupt[-1] ^= 1
    with pytest.raises(R10RelayError, match="CRC mismatch"):
        parse_r10_packet(bytes(corrupt))
    with pytest.raises(R10RelayError, match="coefficient-only texture"):
        replace(packet, dash1_payload=None)
    with pytest.raises(R10RelayError, match="spatially nonidentifiable"):
        replace(packet, pullback_polygons=())


def test_required_controls_are_physical_and_execute() -> None:
    packet, packet_bytes, base = _fixture()
    main = _decode(packet_bytes, base)
    relay_only = r10_control_mode(packet_bytes, R10RelayMode.RELAY_ONLY)
    assert not np.array_equal(_decode(relay_only, base), main)
    permuted = r10_control_phase_time_permutation(packet_bytes, (1, 0))
    assert not np.array_equal(_decode(permuted, base), main)
    shuffled = r10_control_channel_energy_shuffle(packet_bytes, (1, 2, 0))
    shuffled_packet = parse_r10_packet(shuffled)
    for before, after in zip(packet.base_features, shuffled_packet.base_features, strict=True):
        assert sum(v * v for v in before.channel_weights_q8) == sum(
            v * v for v in after.channel_weights_q8
        )
    assert not np.array_equal(_decode(shuffled, base), main)
    texture_deleted = r10_control_delete_texture(packet_bytes)
    assert len(texture_deleted) < len(packet_bytes)
    assert parse_r10_packet(texture_deleted).textures is None
    assert R10Section.TEXTURE.name not in {
        row.section for row in packet_physical_spans(texture_deleted)
    }
    knot_deleted = r10_control_delete_shooting_knots(packet_bytes)
    assert len(knot_deleted) < len(packet_bytes)
    assert not parse_r10_packet(knot_deleted).shooting_knots
    knot_merged = r10_control_merge_shooting_knots(packet_bytes)
    assert len(knot_merged) == len(packet_bytes) - 14
    assert len(parse_r10_packet(knot_merged).shooting_knots) == 1


def test_receipt_and_g23_adapter_preserve_debt_and_exact_spans() -> None:
    packet, packet_bytes, base = _fixture()
    output, receipt = build_r10_decode_receipt(
        packet_bytes,
        base,
        expected_source_sha256=packet.identity.source_sha256,
        expected_pair_population_sha256=packet.identity.pair_population_sha256,
    )
    assert receipt["score_claim"] is False
    assert receipt["deterministic_double_replay"] is True
    assert receipt["output_sha256"] == realization_sha256(output)
    assert set(receipt["operand_mutation_canaries"]) == set(R10_CONSTRAINTS)
    assert receipt["strict_proxy_debt"]
    assert receipt["costates"]["lambda_bytes_vs_distortion"] is None
    assert receipt["maximum_inverse_solve_hook"]["learned_state_policy"] == (
        "counted_irreducible_residual_only"
    )
    adapter = build_r10_selected_solution_adapter(packet_bytes)
    assert adapter.constraints == R10_CONSTRAINTS
    assert {row.constraint for row in adapter.operand_spans} == set(R10_CONSTRAINTS)
    assert adapter.public_inflate_receiver_abi["inflate_bundle_executed"] is False
    assert adapter.asymmetric_codec_placement[
        "counted_learned_irreducible_residual_sections"
    ] == ()
    assert adapter.public_video_emission_abi["video_mux_or_file_emitter_present"] is False
    assert adapter.blockers
