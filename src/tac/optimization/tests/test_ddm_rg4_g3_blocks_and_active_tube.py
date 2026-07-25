# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from tac.optimization.ddm_pc1_pose_stream import PC1PosePacketV1
from tac.optimization.ddm_rg4_g3_blocks_and_active_tube import (
    DDMRG4Error,
    active_tube_report,
    apply_source_preserving_delta,
    build_source_local_composition_archive,
    parse_source_local_composition_archive,
    rg3_typed_exclusions,
)
from tools import run_ddm_rg4_g3_blocks_and_active_tube as rg4_runner


def _packet() -> PC1PosePacketV1:
    return PC1PosePacketV1(
        active=True,
        pair_count=2,
        xi_scales=(0.01,) * 6,
        residual_scale=0.25,
        q_xi=np.asarray([[0, 1, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0]], dtype=np.int16),
        q_luma_phase=np.zeros((2, 4), dtype=np.int8),
    )


def test_source_preserving_delta_has_exact_zero_identity_and_uint8_clip() -> None:
    parent = np.asarray([[[[[10, 250, 30]]]]], dtype=np.uint8)
    zero = np.asarray([[[[[20, 20, 20]]]]], dtype=np.uint8)
    same = apply_source_preserving_delta(
        parent_camera=parent,
        absolute_candidate=zero,
        absolute_zero_home=zero,
    )
    assert np.array_equal(same, parent)
    candidate = np.asarray([[[[[0, 100, 255]]]]], dtype=np.uint8)
    changed = apply_source_preserving_delta(
        parent_camera=parent,
        absolute_candidate=candidate,
        absolute_zero_home=zero,
    )
    assert changed.tolist() == [[[[[0, 255, 255]]]]]


def test_source_local_composition_roundtrips_exactly() -> None:
    parent = b"sha-pinned-parent"
    archive = build_source_local_composition_archive(
        parent_archive=parent,
        parent_sha256=hashlib.sha256(parent).hexdigest(),
        packet=_packet(),
    )
    parsed_parent, parsed_packet, manifest = parse_source_local_composition_archive(archive)
    assert parsed_parent == parent
    assert np.array_equal(parsed_packet.q_xi, _packet().q_xi)
    assert manifest["source_local_zero_identity"] is True
    assert (
        build_source_local_composition_archive(
            parent_archive=parsed_parent,
            parent_sha256=manifest["parent_sha256"],
            packet=parsed_packet,
        )
        == archive
    )


def test_active_tube_reports_full_membership_and_bounded_dimension_count() -> None:
    center = np.zeros((2, 6), dtype=np.float64)
    pose = np.zeros_like(center)
    pose[:, 0] = 0.12
    factors = np.broadcast_to(np.eye(6) / np.sqrt(6.0), (2, 6, 6)).copy()
    report = active_tube_report(
        pose6=pose,
        centers=center,
        low_rank_factors=factors,
        tube_radius=0.05,
    )
    assert report["active_dimension_count"] == 1
    assert report["all_pairs_inside"] is True
    assert report["mean_pair_quadratic"] == pytest.approx(0.12**2 / 6.0)
    outside = active_tube_report(
        pose6=np.full((2, 6), 1.0),
        centers=center,
        low_rank_factors=factors,
        tube_radius=0.05,
    )
    assert outside["active_dimension_count"] == 6
    assert outside["all_pairs_inside"] is False
    with pytest.raises(DDMRG4Error, match="n-by-6"):
        active_tube_report(
            pose6=np.empty((0, 6)),
            centers=np.empty((0, 6)),
            low_rank_factors=np.empty((0, 6, 6)),
            tube_radius=0.05,
        )


def test_rg3_exclusions_require_all_25_exact_obstruction_rows() -> None:
    missing = [{"pair_id": index, "bucket_id": f"bucket-{index}"} for index in range(25)]
    residual = []
    for index in range(25):
        actuator_ids = [
            f"rg3.pair{index}.mag1",
            f"rg3.pair{index}.mag2",
        ]
        residual.append(
            {
                "pair_id": index,
                "bucket_id": f"bucket-{index}",
                "rg3_family": "TEST_FAMILY",
                "rg3_receiver_actuator_ids": actuator_ids,
                "rg3_probe_blocker": {
                    "classification": ("NO_TARGET_BUCKET_EVENT_CHANGED_BY_ANY_COUNTED_RG3_MAGNITUDE_OR_SIGN"),
                    "derived_next_coordinate_family": "NEW_COORDINATE",
                    "probe_count": 4,
                    "probes": [
                        {
                            "checkpoint_sha256": hashlib.sha256(f"{actuator_id}:{direction}".encode()).hexdigest(),
                            "receiver_actuator_id": actuator_id,
                            "direction_id": direction,
                            "target_bucket_event_count": 0,
                            "target_bucket_hit": False,
                            "target_pair_joined": False,
                        }
                        for actuator_id in actuator_ids
                        for direction in (
                            "NEGATIVE_ONE_QUANTUM",
                            "POSITIVE_ONE_QUANTUM",
                        )
                    ],
                },
            }
        )
    summary = {
        "schema": "ddm_ms6_receiver_support_resume_summary.v1",
        "g3_top24_coverage": {
            "coverage_proven": False,
            "missing_block_count": 25,
            "missing_blocks": missing,
        },
        "receiver_coordinate_derivation": {"residual": residual},
    }
    assert len(rg3_typed_exclusions(summary)) == 25
    drifted = copy.deepcopy(summary)
    drifted["receiver_coordinate_derivation"]["residual"][0]["rg3_probe_blocker"]["classification"] = "DRIFT"
    with pytest.raises(DDMRG4Error):
        rg3_typed_exclusions(drifted)

    duplicate_key = copy.deepcopy(summary)
    duplicate_key["g3_top24_coverage"]["missing_blocks"][1] = copy.deepcopy(
        duplicate_key["g3_top24_coverage"]["missing_blocks"][0]
    )
    with pytest.raises(DDMRG4Error, match="duplicate keys"):
        rg3_typed_exclusions(duplicate_key)

    incomplete_probe_grid = copy.deepcopy(summary)
    incomplete_probe_grid["receiver_coordinate_derivation"]["residual"][0]["rg3_probe_blocker"]["probes"].pop()
    with pytest.raises(DDMRG4Error, match="custody differs"):
        rg3_typed_exclusions(incomplete_probe_grid)


def test_runner_refuses_nonfinite_pose_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rg4_runner, "_memory_receipt", lambda _stage: {"admission": True})
    monkeypatch.setattr(
        rg4_runner,
        "_forward",
        lambda *_args: (
            np.zeros((1, 384, 512), dtype=np.uint8),
            np.full((1, 6), np.nan),
        ),
    )
    with pytest.raises(rg4_runner.RunnerError, match="finiteness"):
        rg4_runner._score_batch(
            camera=np.zeros((1, 2, 1, 1, 3), dtype=np.uint8),
            pair_ids=(0,),
            labels=np.zeros((1, 384, 512), dtype=np.uint8),
            poses=np.zeros((1, 6), dtype=np.float64),
            scorers=(object(), object()),
            archive_bytes=1,
            stage="UNIT",
        )
