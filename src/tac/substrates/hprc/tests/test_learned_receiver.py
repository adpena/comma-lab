# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import tac.substrates.hprc.archive_candidate as hprc_candidate
from tac.substrates.hprc.archive import HprcSectionKind, pack_hprc_packet, parse_hprc_packet
from tac.substrates.hprc.inflate import CAMERA_H, CAMERA_W, hprc_preview_digest, inflate_one_video
from tac.substrates.hprc.learned_receiver import (
    COMPACT_NUMPY_DECODER_FAMILY_ID,
    COMPACT_RECEIVER_MODE,
    build_compact_receiver_packet_from_lowres_frames,
    compact_receiver_reconstruction_metrics,
    compact_receiver_section_byte_profile,
    compact_receiver_section_value_profile,
    decode_compact_receiver_packet,
    mutate_compact_receiver_section,
    neutralize_compact_receiver_section,
    render_compact_receiver_frame,
    render_compact_receiver_frame_batch,
    transform_compact_receiver_residual,
)


def _frames() -> np.ndarray:
    y = np.arange(8, dtype=np.float32)[:, None, None]
    x = np.arange(10, dtype=np.float32)[None, :, None]
    c = np.arange(3, dtype=np.float32)[None, None, :]
    frames = []
    for idx in range(4):
        checker = ((x.astype(np.int32) + y.astype(np.int32) + idx) % 2).astype(
            np.float32
        )
        moving_patch = (
            ((y < 4) if idx % 2 == 0 else (x < 5)).astype(np.float32) * 53.0
        )
        frames.append(
            (
                30.0
                + idx * 7.0
                + x * 5.0
                + y * 3.0
                + c * 11.0
                + checker * 47.0
                + moving_patch
            )
            % 255.0
        )
    return np.stack(frames, axis=0).astype(np.float32)


def test_compact_receiver_packet_decodes_semantic_sections() -> None:
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        _frames(),
        basis_count=4,
        residual_grid_h=4,
        residual_grid_w=5,
        source_manifest={"source": "unit"},
    )

    packet = parse_hprc_packet(packet_bytes)
    compact = decode_compact_receiver_packet(packet)
    manifest = json.loads(packet.section_map()[HprcSectionKind.MANIFEST_JSON])

    assert packet.config.decoder_family_id == COMPACT_NUMPY_DECODER_FAMILY_ID
    assert packet.config.frames == 4
    assert packet.config.height == 8
    assert packet.config.width == 10
    assert manifest["hprc_receiver_mode"] == COMPACT_RECEIVER_MODE
    rendered = render_compact_receiver_frame(compact, 0, height=16, width=20)
    assert rendered.shape == (16, 20, 3)
    assert rendered.dtype == np.uint8
    metrics = compact_receiver_reconstruction_metrics(compact, _frames())
    assert metrics["metric_scope"] == "decoder_grid_lowres_advisory_not_contest_score"
    assert metrics["frames"] == 4
    assert metrics["score_claim"] is False
    byte_profile = compact_receiver_section_byte_profile(packet)
    rows = {row["section"]: row for row in byte_profile["section_rows"]}
    assert rows["residual_rc"]["bytes"] > rows["latents_rc"]["bytes"]
    assert byte_profile["score_claim"] is False
    value_profile = compact_receiver_section_value_profile(compact, _frames())
    assert value_profile["metric_scope"] == "decoder_grid_lowres_advisory_not_contest_score"
    assert value_profile["score_claim"] is False
    assert any(row["delta_mse_rgb255"] > 0 for row in value_profile["section_rows"])


def test_compact_receiver_section_proof_uses_valid_semantic_mutations() -> None:
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        _frames(),
        basis_count=4,
        residual_grid_h=4,
        residual_grid_w=5,
    )
    proof = hprc_candidate.build_hprc_section_mutation_proof(packet_bytes)

    assert proof["receiver_mode"] == COMPACT_RECEIVER_MODE
    assert proof["section_mutation_preview_ready"] is True
    assert proof["blockers"] == []
    per_section = {row["section"]: row for row in proof["per_section"]}
    for name in (
        "decoder_qw",
        "latents_rc",
        "selectors_rc",
        "residual_rc",
        "rdo_plan",
        "receiver_state",
    ):
        assert per_section[name]["receiver_preview_changed"] is True
        assert per_section[name]["proof_scope"] == "valid_semantic_packet_mutation_preview"
    assert per_section["manifest_json"]["receiver_preview_changed"] is False


def test_compact_receiver_inflate_writes_contest_resolution_raw(tmp_path: Path) -> None:
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        _frames(),
        basis_count=3,
        residual_grid_h=2,
        residual_grid_w=3,
    )
    out = tmp_path / "0.raw"

    inflate_one_video(packet_bytes, out, device="cpu")

    assert out.stat().st_size == 4 * CAMERA_H * CAMERA_W * 3
    assert hprc_preview_digest(packet_bytes)


def test_compact_receiver_vectorized_raw_writer_matches_frame_renderer(tmp_path: Path) -> None:
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        _frames(),
        basis_count=3,
        residual_grid_h=2,
        residual_grid_w=3,
    )
    compact = decode_compact_receiver_packet(parse_hprc_packet(packet_bytes))
    out = tmp_path / "0.raw"

    from tac.substrates.hprc.learned_receiver import write_compact_receiver_raw

    write_compact_receiver_raw(parse_hprc_packet(packet_bytes), out, height=16, width=20)

    expected = b"".join(
        render_compact_receiver_frame(compact, idx, height=16, width=20).tobytes()
        for idx in range(4)
    )
    assert out.read_bytes() == expected
    batch = render_compact_receiver_frame_batch(compact, 0, 4, height=16, width=20)
    assert batch.tobytes() == expected


def test_compact_receiver_manifest_mutation_is_metadata_only() -> None:
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        _frames(),
        basis_count=3,
        residual_grid_h=2,
        residual_grid_w=3,
    )
    packet = parse_hprc_packet(packet_bytes)
    section_map = packet.section_map()
    mutated_manifest = mutate_compact_receiver_section(
        packet,
        HprcSectionKind.MANIFEST_JSON,
        salt=1,
    )
    assert mutated_manifest is not None
    mutated = dict(section_map)
    mutated[HprcSectionKind.MANIFEST_JSON] = mutated_manifest
    mutated_packet = pack_hprc_packet(mutated, config=packet.config)

    assert hprc_preview_digest(mutated_packet) == hprc_preview_digest(packet_bytes)


def test_compact_receiver_neutralization_returns_valid_packet() -> None:
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        _frames(),
        basis_count=3,
        residual_grid_h=2,
        residual_grid_w=3,
    )
    packet = parse_hprc_packet(packet_bytes)

    neutralized = neutralize_compact_receiver_section(packet, HprcSectionKind.RESIDUAL_RC)
    compact = decode_compact_receiver_packet(parse_hprc_packet(neutralized))

    assert compact.residual.q.shape == (4, 2, 3, 3)
    assert np.count_nonzero(compact.residual.q) == 0
    assert hprc_preview_digest(neutralized) != hprc_preview_digest(packet_bytes)


def test_compact_receiver_residual_transform_is_valid_and_charged() -> None:
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        _frames(),
        basis_count=3,
        residual_grid_h=2,
        residual_grid_w=3,
    )
    packet = parse_hprc_packet(packet_bytes)

    transformed = transform_compact_receiver_residual(
        packet,
        transform="threshold_abs_le=64",
    )
    compact = decode_compact_receiver_packet(parse_hprc_packet(transformed))
    rdo_payload = parse_hprc_packet(transformed).section_map()[HprcSectionKind.RDO_PLAN]
    rdo = json.loads(rdo_payload)

    assert compact.residual.q.shape == (4, 2, 3, 3)
    assert np.count_nonzero(compact.residual.q) < np.count_nonzero(
        decode_compact_receiver_packet(packet).residual.q
    )
    assert rdo["residual_token_transform"]["kind"] == "threshold_abs_le"
    assert hprc_preview_digest(transformed) != hprc_preview_digest(packet_bytes)


def test_compact_receiver_pair_scoped_residual_transform_preserves_protected_pairs() -> None:
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        _frames(),
        basis_count=3,
        residual_grid_h=2,
        residual_grid_w=3,
    )
    packet = parse_hprc_packet(packet_bytes)
    original = decode_compact_receiver_packet(packet).residual.q.copy()

    transformed = transform_compact_receiver_residual(
        packet,
        transform="threshold_abs_le_pairs=127@0",
    )
    compact = decode_compact_receiver_packet(parse_hprc_packet(transformed))
    rdo_payload = parse_hprc_packet(transformed).section_map()[HprcSectionKind.RDO_PLAN]
    rdo = json.loads(rdo_payload)

    assert np.count_nonzero(compact.residual.q[:2]) == 0
    np.testing.assert_array_equal(compact.residual.q[2:], original[2:])
    assert rdo["residual_token_transform"]["kind"] == "threshold_abs_le_pairs"
    assert rdo["residual_token_transform"]["pair_ranges"] == [[0, 0]]
    assert rdo["residual_token_transform"]["realized_frame_count"] == 2


def test_compact_receiver_local_acquisition_frame_cap(tmp_path: Path, monkeypatch) -> None:
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        _frames(),
        basis_count=3,
        residual_grid_h=2,
        residual_grid_w=3,
    )
    out = tmp_path / "0.raw"
    monkeypatch.setenv("PACT_LOCAL_ACQUISITION_MAX_PAIRS", "1")

    inflate_one_video(packet_bytes, out, device="cpu")

    assert out.stat().st_size == 2 * CAMERA_H * CAMERA_W * 3
