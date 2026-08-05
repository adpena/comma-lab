# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.optimization import ddm_od4_weak_stage1_packet as od4


def _blank() -> np.ndarray:
    return np.zeros((2, od4.SEG_H, od4.SEG_W), dtype=np.uint8)


def test_sparse_packet_parseback_and_replay_exact_sites() -> None:
    base = _blank()
    record = od4.SparsePairCorrections(
        pair=1,
        flat_indices=(3, 17, od4.PIXELS_PER_PAIR - 1),
        target_labels=(1, 2, 4),
    )
    packet = od4.serialize_sparse_packet([record])
    parsed = od4.parse_sparse_packet(packet)

    assert parsed.payload_sha256 == od4.sha256_bytes(packet)
    assert od4.serialize_sparse_packet(parsed.pair_records) == packet

    decoded = od4.apply_sparse_packet(base, parsed)[1].reshape(-1)
    assert decoded[3] == 1
    assert decoded[17] == 2
    assert decoded[od4.PIXELS_PER_PAIR - 1] == 4
    assert int(np.count_nonzero(decoded)) == 3


def test_select_sparse_corrections_uses_only_target_correcting_cells() -> None:
    cur = np.zeros((od4.SEG_H, od4.SEG_W), dtype=np.uint8)
    gt = cur.copy()
    target = cur.copy()
    gt.reshape(-1)[:10] = 1
    target.reshape(-1)[:8] = 1
    target.reshape(-1)[8:10] = 2

    record = od4.select_sparse_corrections(
        pair=0,
        current_argmax=cur,
        gt_argmax=gt,
        target_argmax=target,
        desired_fix_count=6,
        fraction=0.5,
    )

    assert record.flat_indices == (0, 1, 2)
    assert record.target_labels == (1, 1, 1)


def test_fidelity_totals_compare_receiver_against_od2_rows() -> None:
    base = _blank()
    gt = _blank()
    gt[0].reshape(-1)[:10] = 1
    record = od4.SparsePairCorrections(pair=0, flat_indices=(0, 1, 2, 3), target_labels=(1, 1, 1, 1))
    parsed = od4.parse_sparse_packet(od4.serialize_sparse_packet([record]))
    od2_rows = {0: {"stage1": {"flips_after": 4}, "n_described": 10}}

    fidelity = od4.fidelity_for_packet(
        current_argmax=base,
        gt_argmax=gt,
        packet=parsed,
        od2_rows_by_pair=od2_rows,
    )

    totals = fidelity["totals"]
    assert totals["flips_before"] == 10
    assert totals["flips_after_receiver"] == 6
    assert totals["retained_fix_count"] == 4
    assert totals["od2_fix_count"] == 6
    assert totals["eta_receiver"] == 0.4
    assert totals["changed_pixels"] == 4
    assert totals["parseback_exact"] is True


def test_coder_race_roundtrips_brotli_and_lzma() -> None:
    packet = od4.serialize_sparse_packet(
        [od4.SparsePairCorrections(pair=0, flat_indices=(0, 100), target_labels=(1, 2))]
    )
    rows = od4.race_packet_coders(packet)

    assert {row.codec for row in rows} == {"brotli-q11", "lzma1-raw"}
    assert all(row.parseback_exact for row in rows)
    assert all(row.bytes > 0 for row in rows)


def test_packet_refuses_body_tamper() -> None:
    packet = bytearray(
        od4.serialize_sparse_packet(
            [od4.SparsePairCorrections(pair=0, flat_indices=(0,), target_labels=(1,))]
        )
    )
    packet[-1] ^= 1
    with pytest.raises(od4.OD4PacketError, match="SHA-256"):
        od4.parse_sparse_packet(bytes(packet))


def test_od5_sectioned_packet_parseback_exact() -> None:
    packet = od4.serialize_od5_packet(
        [
            od4.OD5Section("pe3_generator_coords", b"coords"),
            od4.OD5Section("st2_context_table", b"context"),
        ]
    )
    parsed = od4.parse_od5_packet(packet)

    assert parsed.payload_sha256 == od4.sha256_bytes(packet)
    assert parsed.section_count == 2
    assert [(section.name, section.payload) for section in parsed.sections] == [
        ("pe3_generator_coords", b"coords"),
        ("st2_context_table", b"context"),
    ]
    assert od4.serialize_od5_packet(parsed.sections) == packet


def test_od5_sectioned_packet_refuses_tamper() -> None:
    packet = bytearray(od4.serialize_od5_packet([od4.OD5Section("residual", b"abc")]))
    packet[-1] ^= 1

    with pytest.raises(od4.OD4PacketError, match="SHA-256"):
        od4.parse_od5_packet(bytes(packet))


def test_select_masked_sparse_corrections_retains_only_masked_useful_cells() -> None:
    cur = np.zeros((od4.SEG_H, od4.SEG_W), dtype=np.uint8)
    gt = cur.copy()
    target = cur.copy()
    mask = np.zeros_like(cur, dtype=bool)
    gt.reshape(-1)[:8] = 1
    target.reshape(-1)[:6] = 1
    target.reshape(-1)[6:8] = 2
    mask.reshape(-1)[2:7] = True

    record = od4.select_masked_sparse_corrections(
        pair=0,
        current_argmax=cur,
        gt_argmax=gt,
        target_argmax=target,
        constraint_mask=mask,
        max_count=3,
    )

    assert record.flat_indices == (2, 3, 4)
    assert record.target_labels == (1, 1, 1)


def test_projection_rows_keep_n32_exact_and_n600_projected_bytes_distinct() -> None:
    row = od4.projection_rows(
        n32_packet_bytes=320,
        n_pairs=32,
        retained_fix_count=7000,
        include_od2_pose_credit=True,
    )

    assert row["packet_bytes_n32_exact"] == 320
    assert row["packet_bytes_n600_linear_projection"] == 6000
    assert row["stage2_pose_delta_s"] == od4.OD2_STAGE2_POSE_DELTA_S
    assert row["projected_s"] < od4.CURRENT_OWN_S


def test_projection_rows_accept_measured_n600_packet_bytes() -> None:
    row = od4.projection_rows_with_projected_packet_bytes(
        n32_packet_bytes=4000,
        n600_packet_bytes_projected=74_408,
        n_pairs=32,
        retained_fix_count=6000,
        include_od2_pose_credit=True,
        projection_scope="measured PE3 n600 section bytes plus exact n32 mask replay",
    )

    assert row["packet_bytes_n32_exact"] == 4000
    assert row["packet_bytes_n600_projected"] == 74_408
    assert row["packet_rate_s_projected_n600"] == 74_408 * od4.RATE_PER_BYTE
    assert row["rate_cost_over_seg_win"] > 0
