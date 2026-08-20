# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import json
import lzma
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.optimization import direct_description_carrier_compose as compose
from tac.optimization.direct_description_carrier_compose import (
    ARCHIVE_SCHEMA_V2,
    ARCHIVE_SCHEMA_V3,
    ARCHIVE_SCHEMA_V4,
    ARCHIVE_SCHEMA_V5,
    ARCHIVE_SCHEMA_V6,
    BOUNDARY_CORRECTION_MEMBER,
    BOUNDARY_SHEARLET_MEMBER,
    EVENT_CORRECTION_MEMBER,
    ISLAND_SHAPE_MEMBER,
    LANE_KNOT_MEMBER,
    LANE_PROGRAM_MEMBER,
    REALIZATION_PROFILE_MEMBER,
    REALIZATION_STATIC_RULE_MEMBER,
    SCORER_SOLVED_TEMPLATE_MEMBER,
    WORLDSHEET_G1_MEMBER,
    WORLDSHEET_KNOT_MEMBER,
    WORLDSHEET_TRACK_MEMBER,
    BoundaryCoefficientDelta,
    BoundaryShearletAtomV1,
    DirectDescriptionV9CarrierComposeConfigV1,
    DirectDescriptionV10FisherEventSearchConfigV1,
    DirectDescriptionV11ObligationSearchConfigV1,
    DirectDescriptionV12ObligationDrainConfigV1,
    DirectDescriptionV13WorldsheetPredictorConfigV1,
    IslandShapeAtomV1,
    LaneDriftKnotV1,
    LanePeriodicProgramV1,
    MovableWorldsheetKnotV1,
    MovableWorldsheetTrackV1,
    ReceiverRealizationProfileV1,
    RowBandScorerTemplateV1,
    ScorerSolvedTemplateBankV1,
    TopologyEventV1,
    compile_carrier_compose_archive,
    decode_scorer_solved_template_bank,
    encode_scorer_solved_template_bank,
    parse_carrier_compose_archive,
    prove_carrier_archive_fail_closed,
    receive_carrier_compose_archive,
    recursive_carrier_byte_rows,
)
from tac.optimization.direct_description_entropy_priced_member import (
    StructuredS4SourcesV1,
    compile_composed_structured_member_archive,
)
from tac.optimization.direct_description_entropy_streams import STREAM_ORDER, compile_entropy_chart_archive
from tac.optimization.direct_description_g1_worldsheet import (
    decode_g1_movable_worldsheet,
    encode_g1_movable_worldsheet,
)
from tac.optimization.direct_description_measurement_ladder import (
    _ANCHOR_RECORD,
    _GRADIENT_RECORD,
    _POSE_RECORD,
    _RESIDUAL_RECORD,
    CountedChartStreamV1,
    DirectDescriptionChartZV1,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.predictor_upgrade_xi_chart import LaneCoefficientDelta
from tools.measure_ddm_v14_realization_fidelity import DDMV14RealizationFidelityConfigV1
from tools.run_ddm_v9_carrier_compose import (
    _batch_score_cache_bytes,
    _BatchScore,
    _bundle_obligation_candidates,
    _bundle_obligation_candidates_full_drain,
    _consecutive_flat_budget_tail_rungs,
    _joint_objective_delta,
    _load_batch_score_cache,
    _rank_values,
    _SearchCandidate,
    _select_diverse_candidates,
)


def _z() -> DirectDescriptionChartZV1:
    bodies = {name: bytearray() for name in STREAM_ORDER}
    for pair_id in range(64):
        for plane_id in range(2):
            bodies["global_chart_anchors"].extend(_ANCHOR_RECORD.pack(pair_id, plane_id, 96, 112, 128))
            bodies["axial_chart_gradients"].extend(_GRADIENT_RECORD.pack(pair_id, plane_id, 0, 0, 0, 0, 0, 0))
            for stratum, stream_name in enumerate(STREAM_ORDER[2:5]):
                for chart_id in range(stratum * 64, (stratum + 1) * 64):
                    bodies[stream_name].extend(_RESIDUAL_RECORD.pack(pair_id, plane_id, chart_id, 0, 0, 0))
        bodies["pose6_pair_codes"].extend(_POSE_RECORD.pack(pair_id, 20, 21, 22, 23, 24, 25))
    return DirectDescriptionChartZV1(
        n_pairs=64,
        **{name: CountedChartStreamV1(payload=bytes(bodies[name])) for name in STREAM_ORDER},
    )


def _lane_payload() -> tuple[bytes, tuple[tuple[np.ndarray, ...], ...], dict[str, object]]:
    vector = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 50.0])
    lines = tuple((vector.copy(),) for _ in range(600))
    header: dict[str, object] = {
        "cx": 256.0,
        "dash_forward_max_m": 50.0,
        "dash_gate": True,
        "rd": {"K": 1, "base_steps": [0.25] * 11, "d_slot": 11, "n_pairs": 600},
        "softness": 1.0,
        "v_h": 150.0,
    }
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    presence = np.packbits(np.ones((600, 1), dtype=np.uint8)).tobytes()
    quantized = np.rint(np.stack([vector] * 600) / 0.25).astype(np.int64)
    delta = np.diff(quantized, axis=0, prepend=np.zeros((1, 11), dtype=np.int64))
    zigzag = ((delta << 1) ^ (delta >> 63)).astype("<u4")
    raw = (
        b"LBND2\0"
        + struct.pack("<I", len(raw_header))
        + raw_header
        + struct.pack("<I", len(presence))
        + presence
        + zigzag.tobytes(order="C")
    )
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}]
    return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=filters), lines, header


def _predictor() -> bytes:
    empty = tuple(() for _ in range(64))
    classes: list[tuple[tuple[np.ndarray, ...], ...]] = [empty for _ in range(5)]
    for class_id in range(5):
        rows = list(empty)
        rows[0] = (np.asarray([class_id * 8, class_id * 8 + 1], dtype=np.int64),)
        classes[class_id] = tuple(rows)
    road = np.zeros((384, 512), dtype=bool)
    road[200:220, 100:120] = True
    hood = np.zeros((384, 512), dtype=bool)
    hood[360:, 220:292] = True
    encoded, lines, header = _lane_payload()
    sources = StructuredS4SourcesV1(
        pair_count=64,
        palette=np.asarray(
            [[153, 255, 51], [51, 255, 204], [0, 153, 0], [102, 204, 51], [0, 255, 153]],
            dtype=np.uint8,
        ),
        camera={"height_m": 1.2, "fx_scorer": 400.3, "fy_scorer": 399.5},
        static_masks={
            "Road": road,
            "Undrivable": np.zeros((384, 512), dtype=bool),
            "MyCar": hood,
        },
        lane_encoded=encoded,
        lane_lines=lines,
        lane_header=header,
        events=tuple(classes),
        components=tuple(empty for _ in range(5)),
        custody={"fixture": True},
        role_class_ids={"Road": 0, "Lane": 1, "UndrivableBoundary": 2, "Movable": 3, "MyCar": 4},
        role_rgb_u8={
            "Road": (153, 255, 51),
            "Lane": (51, 255, 204),
            "UndrivableBoundary": (0, 153, 0),
            "Movable": (102, 204, 51),
            "MyCar": (0, 255, 153),
        },
        routing_custody={"fixture": True},
    )
    baseline = compile_entropy_chart_archive(_z()).archive
    return compile_composed_structured_member_archive(baseline, sources, pair_start=0)[0]


def test_v9_config_is_strict_local_only_and_window_bounded() -> None:
    config = DirectDescriptionV9CarrierComposeConfigV1(
        pair_start=448,
        pair_count=64,
        v6_receipt_path="v6.json",
        v6_receipt_sha256="1" * 64,
        predictor_archive_path="predictor.zip",
        predictor_archive_sha256="2" * 64,
        upstream_root="/absolute/upstream",
        scorer_threads=1,
    )
    assert config.execution_allowed is False
    assert config.score_claim is False
    with pytest.raises(ValueError, match="bridge windows"):
        DirectDescriptionV9CarrierComposeConfigV1(
            pair_start=344,
            pair_count=64,
            v6_receipt_path="v6.json",
            v6_receipt_sha256="1" * 64,
            predictor_archive_path="predictor.zip",
            predictor_archive_sha256="2" * 64,
            upstream_root="/absolute/upstream",
            scorer_threads=1,
        )


def test_v10_config_is_typed_candidate_search_and_budget_bounded() -> None:
    config = DirectDescriptionV10FisherEventSearchConfigV1(
        run_id="fixture_v10",
        pair_start=448,
        pair_count=64,
        v6_receipt_path="v6.json",
        v6_receipt_sha256="1" * 64,
        predictor_archive_path="predictor.zip",
        predictor_archive_sha256="2" * 64,
        upstream_root="/absolute/upstream",
        scorer_threads=1,
    )
    assert config.correction_policy.startswith("greedy_measured_fisher_margin_candidate_search")
    assert config.added_budget_bytes == (0, 5120, 15360, 40960, 102400)
    assert config.execution_allowed is False
    with pytest.raises(ValueError, match="100 KiB"):
        DirectDescriptionV10FisherEventSearchConfigV1(
            run_id="fixture_v10",
            pair_start=448,
            pair_count=64,
            v6_receipt_path="v6.json",
            v6_receipt_sha256="1" * 64,
            predictor_archive_path="predictor.zip",
            predictor_archive_sha256="2" * 64,
            upstream_root="/absolute/upstream",
            scorer_threads=1,
            added_budget_bytes=(0, 102401),
        )


def test_v11_config_binds_joint_objective_and_full_n600_window() -> None:
    common = {
        "run_id": "fixture_v11",
        "v6_receipt_path": "v6.json",
        "v6_receipt_sha256": "1" * 64,
        "predictor_archive_path": "predictor.zip",
        "predictor_archive_sha256": "2" * 64,
        "upstream_root": "/absolute/upstream",
        "scorer_threads": 1,
    }
    config = DirectDescriptionV11ObligationSearchConfigV1(pair_start=0, pair_count=600, **common)
    assert config.admission_policy == "greedy_measured_joint_contest_objective_pose_tube_safety_only"
    assert config.boundary_basis.endswith("fourier_free")
    assert config.total_archive_ceiling_bytes == 200_000
    assert config.score_claim is False
    with pytest.raises(ValueError, match="v11 windows"):
        DirectDescriptionV11ObligationSearchConfigV1(pair_start=0, pair_count=256, **common)


def test_v12_config_binds_full_n600_resumable_drain() -> None:
    common = {
        "run_id": "fixture_v12",
        "v6_receipt_path": "v6.json",
        "v6_receipt_sha256": "1" * 64,
        "predictor_archive_path": "predictor.zip",
        "predictor_archive_sha256": "2" * 64,
        "upstream_root": "/absolute/upstream",
        "scorer_threads": 1,
    }
    config = DirectDescriptionV12ObligationDrainConfigV1(pair_start=0, pair_count=600, **common)
    assert config.max_measured_candidates == 512
    assert config.max_bundles_per_invocation == 64
    assert config.drain_policy.startswith("exhaustive_conflict_free")
    assert config.base_cache_policy.startswith("immutable_zlib")
    with pytest.raises(ValueError, match="exact full n600"):
        DirectDescriptionV12ObligationDrainConfigV1(pair_start=448, pair_count=64, **common)


def test_v13_config_seals_natural_production_ladder_and_false_authority() -> None:
    config = DirectDescriptionV13WorldsheetPredictorConfigV1(
        run_id="fixture_v13",
        pair_start=448,
        pair_count=64,
        v6_receipt_path="v6.json",
        v6_receipt_sha256="1" * 64,
        predictor_archive_path="predictor.zip",
        predictor_archive_sha256="2" * 64,
        upstream_root="/absolute/upstream",
        scorer_threads=1,
    )
    assert config.composition_ladder == ("base", "islands", "lane", "both")
    assert "persist_unless_event" in config.worldsheet_policy
    assert "periodic_dash_phase_xi" in config.lane_policy
    assert config.lane_policy_status == "measured_pre_20260722T1916Z_operator_addenda_baseline_only"
    assert config.lane_successor_required.startswith("bev_curvature_dash_comb_range_gate")
    assert config.movable_successor_required.startswith("projective_flow_depth_magnification")
    assert config.execution_allowed is config.score_claim is config.d_seg_claim is False


def test_v11_joint_objective_can_admit_priced_pose_worsening() -> None:
    seg_delta, pose_delta, rate_delta, joint_delta = _joint_objective_delta(
        current_errors=10_000,
        proposed_errors=9_000,
        sites=64 * 384 * 512,
        current_dpose=159.0,
        proposed_dpose=159.001,
        marginal_bytes=700,
    )
    assert seg_delta < 0.0 < pose_delta
    assert rate_delta > 0.0
    assert joint_delta < 0.0


def test_v11_bundles_absolute_pair_ids_against_window_relative_batch_geometry() -> None:
    candidates = [
        _SearchCandidate(
            "first",
            "Lane/center_c0_rank4_obligation",
            3.0,
            (448,),
            lane_symbols=(LaneCoefficientDelta(448, 0, 0, 1.0),),
        ),
        _SearchCandidate(
            "last_in_batch",
            "Lane/center_c0_rank4_obligation",
            2.0,
            (455,),
            lane_symbols=(LaneCoefficientDelta(455, 0, 0, 1.0),),
        ),
        _SearchCandidate(
            "next_batch",
            "Lane/center_c0_rank4_obligation",
            1.0,
            (456,),
            lane_symbols=(LaneCoefficientDelta(456, 0, 0, 1.0),),
        ),
        _SearchCandidate(
            "crosses_batch",
            "Lane/center_c0_rank4_obligation",
            99.0,
            (455, 456),
        ),
    ]
    bundles = _bundle_obligation_candidates(
        candidates,
        pair_start=448,
        batch_size=8,
        maximum_bundles=2,
        maximum_atoms=8,
        minimum_per_family=1,
    )
    assert {row.candidate_id for row in bundles} == {
        "obligation_bundle_batch_0000_lane_center",
        "obligation_bundle_batch_0008_lane_center",
    }
    assert all(455 not in row.source_pair_ids or 456 not in row.source_pair_ids for row in bundles)


def test_v12_full_drain_partitions_conflicts_and_covers_every_atom() -> None:
    rows = [
        _SearchCandidate(
            f"lane_{index}",
            "Lane/center_c0_rank4_obligation",
            10.0 - index,
            (index // 2,),
            lane_symbols=(LaneCoefficientDelta(index // 2, 0, 0, float(index + 1)),),
        )
        for index in range(4)
    ]
    rows.append(
        _SearchCandidate(
            "movable",
            "Movable/birth_moments_curvelet_xi_transport",
            1.0,
            (0,),
            island_shapes=(IslandShapeAtomV1(0, "birth", 1, 100, 200, 10, 18, 32, 8, -4, 12),),
        )
    )
    bundles = _bundle_obligation_candidates_full_drain(
        rows,
        pair_start=0,
        batch_size=16,
        maximum_bundles=16,
        maximum_atoms=2,
        family_stratum_mass={
            "lane_center": 0.4,
            "lane_width": 0.4,
            "lane_phase": 0.4,
            "road_boundary": 0.1,
            "undrivable_boundary": 0.1,
            "movable_shape": 0.9,
        },
    )
    assert sum(len(row.lane_symbols) + len(row.island_shapes) for row in bundles) == len(rows)
    assert len(bundles) == 3
    assert all(len(row.conflict_keys()) == len(row.lane_symbols) + len(row.island_shapes) for row in bundles)
    assert all(row.mechanism.startswith(("Lane/", "Movable/")) for row in bundles)


def test_v12_batch_score_cache_is_immutable_compact_roundtrip(tmp_path: Path) -> None:
    score = _BatchScore(
        cells=np.arange(32, dtype=np.uint8).reshape(2, 4, 4) % 5,
        poses=np.arange(12, dtype=np.float32).reshape(2, 6),
        errors=7,
        pose_squared_error=1.25,
    )
    path = tmp_path / "0000.score.zlib"
    payload = _batch_score_cache_bytes(score, identity="a" * 64, start=0)
    path.write_bytes(payload)
    loaded = _load_batch_score_cache(path, identity="a" * 64, start=0)
    assert np.array_equal(loaded.cells, score.cells)
    assert np.array_equal(loaded.poses, score.poses)
    assert loaded.errors == score.errors
    assert loaded.pose_squared_error == score.pose_squared_error
    assert len(payload) < score.cells.nbytes + score.poses.nbytes + 512


def test_v12_spearman_ranks_average_ties() -> None:
    values = np.asarray([3.0, 1.0, 1.0, 2.0, 3.0], dtype=np.float64)
    assert np.array_equal(_rank_values(values), np.asarray([3.5, 0.5, 0.5, 2.0, 3.5]))


def test_v12_flattening_uses_exact_budget_rungs_not_candidate_count() -> None:
    def rung(archive_sha: str, d_seg: str, d_pose: str) -> dict[str, object]:
        return {
            "archive": {"sha256": archive_sha},
            "bridge": {
                "segmentation": {"d_seg": d_seg},
                "pose": {"d_pose": d_pose},
            },
        }

    ladder = [
        rung("base", "0.04", "164"),
        rung("final", "0.034", "163"),
        rung("final", "0.034", "163"),
        rung("final", "0.034", "163"),
    ]
    assert _consecutive_flat_budget_tail_rungs(ladder) == 3
    ladder[-2] = rung("different", "0.035", "163")
    assert _consecutive_flat_budget_tail_rungs(ladder) == 1


def test_v10_candidate_cutoff_preserves_every_mechanism_family() -> None:
    rows = [
        _SearchCandidate(f"road_{index}", "Road/cubic_boundary_coefficients", 100.0 - index, (index,))
        for index in range(20)
    ]
    rows.extend(
        (
            _SearchCandidate("lane", "Lane/G2CS1_centerline_c3", 1.0, (100,)),
            _SearchCandidate("lane_event", "Lane/birth_bbox_ellipse_xi_transport", 0.9, (101,)),
            _SearchCandidate("movable_event", "Movable/birth_bbox_ellipse_xi_transport", 0.8, (102,)),
        )
    )
    selected = _select_diverse_candidates(rows, maximum=8, minimum_per_family=1)
    assert {row.mechanism for row in selected} >= {
        "Road/cubic_boundary_coefficients",
        "Lane/G2CS1_centerline_c3",
        "Lane/birth_bbox_ellipse_xi_transport",
        "Movable/birth_bbox_ellipse_xi_transport",
    }


def test_v9_archive_roundtrip_unique_homes_and_six_nonduplicated_strata() -> None:
    archive, homes = compile_carrier_compose_archive(_predictor())
    members, replay_homes = parse_carrier_compose_archive(archive)
    receiver = receive_carrier_compose_archive(archive)
    rows = recursive_carrier_byte_rows(archive)
    assert set(members) == {"manifest.json", "predictor.zip"}
    assert homes == replay_homes
    assert sum(row["zip_home_bytes"] for row in homes) == len(archive)
    assert [row["stratum"] for row in rows[:6]] == [
        "Road",
        "Lane",
        "Undrivable",
        "Movable",
        "MyCar",
        "xi/Pose6",
    ]
    assert all(row["nested_unique_home_bytes"] > 0 for row in rows[:6])
    assert receiver.custody["nested_pose6_owner_reused"] is True
    assert receiver.custody["pixel_coordinate_or_rgb_patch_present"] is False
    assert np.array_equal(
        receiver.render_pairs((0, 63)), receive_carrier_compose_archive(archive).render_pairs((0, 63))
    )


def test_g2cs1_refinement_is_consumed_before_region_coherent_rasterization() -> None:
    predictor = _predictor()
    base = receive_carrier_compose_archive(compile_carrier_compose_archive(predictor)[0])
    symbol = LaneCoefficientDelta(pair_index=0, line_index=0, coefficient_index=3, coefficient_delta=0.5)
    archive, _homes = compile_carrier_compose_archive(predictor, (symbol,))
    corrected = receive_carrier_compose_archive(archive)
    assert corrected.symbols == (symbol,)
    assert not np.array_equal(base.render_pairs((0,)), corrected.render_pairs((0,)))
    assert corrected.custody["region_coherent_chart_rerasterization"] is True
    rows = recursive_carrier_byte_rows(archive)
    assert rows[-1]["nested_unique_home_bytes"] > 0


def test_v10_boundary_and_xi_event_vocab_are_counted_and_receiver_consumed() -> None:
    predictor = _predictor()
    boundary = (
        BoundaryCoefficientDelta(0, "Road", 0, 3.0),
        BoundaryCoefficientDelta(0, "Road", 1, 1.0),
    )
    events = (TopologyEventV1(0, "Movable", "birth", "ellipse", 2, 100, 200, 122, 232, 1, -1),)
    archive, homes = compile_carrier_compose_archive(
        predictor,
        boundary_symbols=boundary,
        topology_events=events,
    )
    members, replay_homes = parse_carrier_compose_archive(archive)
    receiver = receive_carrier_compose_archive(archive)
    manifest = json.loads(members["manifest.json"])
    assert manifest["schema"] == ARCHIVE_SCHEMA_V2
    assert BOUNDARY_CORRECTION_MEMBER in members
    assert EVENT_CORRECTION_MEMBER in members
    assert homes == replay_homes
    assert receiver.boundary_symbols == boundary
    assert receiver.topology_events == events
    assert receiver.custody["boundary_symbol_parse_reencode_identical"] is True
    assert receiver.custody["topology_event_parse_reencode_identical"] is True
    assert receiver.custody["topology_events_consume_counted_pose6_transport"] is True
    assert not np.array_equal(
        receive_carrier_compose_archive(compile_carrier_compose_archive(predictor)[0]).render_pairs((0, 1)),
        receiver.render_pairs((0, 1)),
    )
    rows = recursive_carrier_byte_rows(archive)
    assert rows[-2]["stratum"] == "road_boundary_coefficients"
    assert rows[-1]["stratum"] == "xi_topology_events"
    assert all(row["nested_unique_home_bytes"] > 0 for row in rows[-2:])


def test_v11_obligation_atoms_are_counted_fourier_free_and_receiver_consumed() -> None:
    predictor = _predictor()
    lane = (LaneCoefficientDelta(0, 0, 5, 2.0),)
    shearlets = (BoundaryShearletAtomV1(0, "Road", 200, 110, 8, 32, 0, 128),)
    islands = (IslandShapeAtomV1(0, "birth", 1, 100, 200, 10, 18, 32, 8, -4, 12),)
    archive, homes = compile_carrier_compose_archive(
        predictor,
        lane,
        boundary_shearlets=shearlets,
        island_shapes=islands,
        obligation_vocabulary=True,
    )
    members, replay_homes = parse_carrier_compose_archive(archive)
    receiver = receive_carrier_compose_archive(archive)
    manifest = json.loads(members["manifest.json"])
    assert manifest["schema"] == ARCHIVE_SCHEMA_V3
    assert BOUNDARY_SHEARLET_MEMBER in members and ISLAND_SHAPE_MEMBER in members
    assert "Fourier-free" in manifest["corrections"]["boundary_shearlet_atoms"]["basis"]
    assert homes == replay_homes
    assert receiver.boundary_shearlets == shearlets
    assert receiver.island_shapes == islands
    assert receiver.custody["boundary_shearlet_parse_reencode_identical"] is True
    assert receiver.custody["island_shape_parse_reencode_identical"] is True
    assert receiver.custody["pixel_coordinate_or_rgb_patch_present"] is False
    assert not np.array_equal(
        receive_carrier_compose_archive(compile_carrier_compose_archive(predictor)[0]).render_pairs((0,)),
        receiver.render_pairs((0,)),
    )
    rows = recursive_carrier_byte_rows(archive)
    assert [row["stratum"] for row in rows[-2:]] == [
        "boundary_shearlet_obligations",
        "movable_shape_obligations",
    ]


def test_v13_worldsheet_and_lane_natural_productions_are_counted_and_consumed() -> None:
    predictor = _predictor()
    tracks = (MovableWorldsheetTrackV1(7, 0, 3, 100, 200, 10, 18, 32, 8, -4, 12, 1, -1),)
    worldsheet_knots = (MovableWorldsheetKnotV1(7, 1, 16, -16, 8, -8, 4, 0, 1, -1, 2),)
    lane_programs = (LanePeriodicProgramV1(0, 0, 3, 32, 4, 512, 0),)
    lane_knots = (LaneDriftKnotV1(0, 1, 0, 0, 0, 64, 64, 32),)
    archive, homes = compile_carrier_compose_archive(
        predictor,
        worldsheet_tracks=tracks,
        worldsheet_knots=worldsheet_knots,
        lane_programs=lane_programs,
        lane_knots=lane_knots,
    )
    members, replay_homes = parse_carrier_compose_archive(archive)
    receiver = receive_carrier_compose_archive(archive)
    manifest = json.loads(members["manifest.json"])
    assert manifest["schema"] == ARCHIVE_SCHEMA_V4
    assert set(members) == {
        "manifest.json",
        "predictor.zip",
        WORLDSHEET_TRACK_MEMBER,
        WORLDSHEET_KNOT_MEMBER,
        LANE_PROGRAM_MEMBER,
        LANE_KNOT_MEMBER,
    }
    assert manifest["grammar"]["temporal_default"] == "persist-unless-birth-or-death-event"
    assert "one dash phase" in manifest["grammar"]["lane"]["periodic_programs"]["alphabet"]
    assert homes == replay_homes
    assert receiver.worldsheet_tracks == tracks
    assert receiver.worldsheet_knots == worldsheet_knots
    assert receiver.lane_programs == lane_programs
    assert receiver.lane_knots == lane_knots
    assert receiver.custody["worldsheet_stores_only_xi_deviations"] is True
    assert receiver.custody["lane_one_dash_phase_per_object_not_per_dash"] is True
    assert receiver.custody["pixel_coordinate_or_rgb_patch_present"] is False
    base = receive_carrier_compose_archive(compile_carrier_compose_archive(predictor)[0])
    assert not np.array_equal(base.render_pairs((0, 1, 2)), receiver.render_pairs((0, 1, 2)))
    assert [row["stratum"] for row in recursive_carrier_byte_rows(archive)[-4:]] == [
        "movable_worldsheet_lifecycle_shape",
        "movable_worldsheet_xi_deviation_morph_knots",
        "lane_periodic_phase_width_visibility",
        "lane_polynomial_drift_knots",
    ]


def test_v13_adopted_g1_polygon_worldsheet_is_semantic_parseback_and_receiver_visible() -> None:
    labels = np.zeros((64, 384, 512), dtype=np.int64)
    for pair in range(64):
        labels[pair, 24:40, 32 + pair : 48 + pair] = 3
    payload, metadata = encode_g1_movable_worldsheet(labels)
    decoded, replay_metadata = decode_g1_movable_worldsheet(payload, expected_pairs=64)
    assert np.array_equal(decoded, labels == 3)
    assert metadata.payload_sha256 == replay_metadata.payload_sha256
    assert metadata.production_counted_bytes.keys() == {"EVENT", "CENTROID", "SHAPE", "envelope_header"}
    archive, _homes = compile_carrier_compose_archive(_predictor(), worldsheet_g1_payload=payload)
    members, _parse_homes = parse_carrier_compose_archive(archive)
    receiver = receive_carrier_compose_archive(archive)
    assert WORLDSHEET_G1_MEMBER in members
    assert receiver.custody["worldsheet_g1_semantic_parseback"] is True
    assert receiver.custody["worldsheet_g1_pair_count"] == 64
    assert receiver.custody["pixel_coordinate_or_rgb_patch_present"] is False
    assert recursive_carrier_byte_rows(archive)[-1]["stratum"] == "movable_g1_polygon_worldsheet_derivation"
    base = receive_carrier_compose_archive(compile_carrier_compose_archive(_predictor())[0])
    assert not np.array_equal(base.render_pairs((0, 63)), receiver.render_pairs((0, 63)))


def test_v14_counted_realization_profile_places_hard_semantics_at_camera_resolution() -> None:
    labels = np.zeros((64, 384, 512), dtype=np.int64)
    labels[:, 24:40, 32:48] = 3
    payload, _metadata = encode_g1_movable_worldsheet(labels)
    profile = ReceiverRealizationProfileV1(
        role_rgb_u8=((0, 153, 0), (11, 3, 9), (51, 255, 204), (107, 0, 114), (63, 72, 63))
    )
    legacy_archive, _ = compile_carrier_compose_archive(_predictor(), worldsheet_g1_payload=payload)
    archive, _homes = compile_carrier_compose_archive(
        _predictor(),
        worldsheet_g1_payload=payload,
        realization_profile=profile,
    )
    members, _parse_homes = parse_carrier_compose_archive(archive)
    receiver = receive_carrier_compose_archive(archive)
    assert json.loads(members["manifest.json"])["schema"] == ARCHIVE_SCHEMA_V5
    assert len(members[REALIZATION_PROFILE_MEMBER]) == 23
    assert receiver.custody["camera_resolution_placement"] is True
    assert receiver.custody["movable_g1_replaces_inherited_mask"] is True
    assert receiver.custody["pixel_coordinate_or_rgb_patch_present"] is False
    assert np.array_equal(
        receive_carrier_compose_archive(legacy_archive).render_pairs((0, 63)),
        receiver.render_pairs((0, 63)),
    )
    camera = receiver.render_camera_pairs((0, 63))
    assert camera.shape == (2, 2, 874, 1164, 3)
    assert camera.dtype == np.uint8
    assert any(row["stratum"] == "receiver_realization_profile" for row in recursive_carrier_byte_rows(archive))

    static_rule = b"\x00" + struct.pack(">4sBHHBB", b"G4MB", 1, 174, 215, 1, 0)
    ruled_archive, _ = compile_carrier_compose_archive(
        _predictor(),
        worldsheet_g1_payload=payload,
        realization_profile=profile,
        realization_static_rule_payload=static_rule,
        realization_static_rule_id="movable_midband_parametric",
    )
    ruled_members, _ = parse_carrier_compose_archive(ruled_archive)
    ruled = receive_carrier_compose_archive(ruled_archive)
    assert ruled_members[REALIZATION_STATIC_RULE_MEMBER] == static_rule
    assert ruled.custody["realization_static_rule_bytes"] == 12
    assert ruled.custody["realization_static_rule_active_sites"] == 42 * 512
    assert ruled.custody["realization_static_rule_id"] == "movable_midband_parametric"
    assert any(row["stratum"] == "receiver_static_cell_rule" for row in recursive_carrier_byte_rows(ruled_archive))
    assert not np.array_equal(camera, ruled.render_camera_pairs((0, 63)))


def test_v15_counted_row_band_templates_extend_v14_receiver_without_decode_scorer() -> None:
    labels = np.zeros((64, 384, 512), dtype=np.int64)
    labels[:, 24:40, 32:48] = 3
    payload, _metadata = encode_g1_movable_worldsheet(labels)
    profile = ReceiverRealizationProfileV1(
        role_rgb_u8=((0, 153, 0), (11, 3, 9), (51, 255, 204), (107, 0, 114), (63, 72, 63))
    )
    bank = ScorerSolvedTemplateBankV1(
        (
            RowBandScorerTemplateV1("Lane", "inner_boundary", 0, 384, 1, 2, bytes((5, 6, 7, 8, 9, 10))),
            RowBandScorerTemplateV1("Movable", "fill", 0, 192, 2, 2, bytes(range(12))),
            RowBandScorerTemplateV1("Movable", "fill", 192, 384, 1, 1, bytes((200, 20, 80))),
        )
    )
    encoded = encode_scorer_solved_template_bank(bank)
    assert decode_scorer_solved_template_bank(encoded) == bank
    v14_archive, _ = compile_carrier_compose_archive(
        _predictor(), worldsheet_g1_payload=payload, realization_profile=profile
    )
    noop_bank = ScorerSolvedTemplateBankV1(
        (
            RowBandScorerTemplateV1("Lane", "inner_boundary", 0, 384, 1, 1, bytes((51, 255, 204))),
            RowBandScorerTemplateV1("Movable", "fill", 0, 384, 1, 1, bytes((107, 0, 114))),
        )
    )
    noop_archive, _ = compile_carrier_compose_archive(
        _predictor(),
        worldsheet_g1_payload=payload,
        realization_profile=profile,
        scorer_solved_templates=noop_bank,
    )
    v14_camera = receive_carrier_compose_archive(v14_archive).render_camera_pairs((0, 63))
    noop_receiver = receive_carrier_compose_archive(noop_archive)
    assert np.array_equal(v14_camera, noop_receiver.render_camera_pairs((0, 63)))
    lane_mask = noop_receiver.template_camera_masks((0, 63), noop_bank.templates[0])
    # Without this the identity assertion below is vacuous: an all-False mask
    # selects nothing, `patched` stays a plain copy, and an implementation
    # replaced by `np.zeros((n, 874, 1164), dtype=bool)` passes.
    assert lane_mask.any()
    patched = v14_camera.copy()
    for index, mask in enumerate(lane_mask):
        patched[index, 0, mask] = (51, 255, 204)
        patched[index, 1, mask] = (51, 255, 204)
    assert np.array_equal(v14_camera, patched)
    # Positive control for the same mask: painting a colour that DIFFERS at every
    # selected pixel must change the camera.  255 - x != x for every uint8, so this
    # fails outright on an all-False mask.
    probe = v14_camera.copy()
    for index, mask in enumerate(lane_mask):
        probe[index, 0, mask] = 255 - probe[index, 0, mask]
        probe[index, 1, mask] = 255 - probe[index, 1, mask]
    assert not np.array_equal(v14_camera, probe)
    archive, _homes = compile_carrier_compose_archive(
        _predictor(),
        worldsheet_g1_payload=payload,
        realization_profile=profile,
        scorer_solved_templates=bank,
    )
    members, _parse_homes = parse_carrier_compose_archive(archive)
    receiver = receive_carrier_compose_archive(archive)
    manifest = json.loads(members["manifest.json"])
    assert manifest["schema"] == ARCHIVE_SCHEMA_V6
    assert members[SCORER_SOLVED_TEMPLATE_MEMBER] == encoded
    assert manifest["scorer_solved_templates"]["decode_boundary"] == "deterministic_template_tiling_no_scorer"
    assert receiver.custody["scorer_solved_template_count"] == 3
    assert receiver.custody["decode_scorer_dependency"] is False
    assert receiver.custody["pixel_coordinate_or_rgb_patch_present"] is True
    assert any(
        row["stratum"] == "encode_side_scorer_solved_shared_templates" for row in recursive_carrier_byte_rows(archive)
    )
    assert not np.array_equal(
        receive_carrier_compose_archive(v14_archive).render_camera_pairs((0, 63)),
        receiver.render_camera_pairs((0, 63)),
    )
    band_mask = receiver.template_camera_masks((0,), bank.templates[1])
    assert band_mask.shape == (1, 874, 1164)
    assert band_mask.any()

    with pytest.raises(DirectDescriptionError, match="overlap"):
        ScorerSolvedTemplateBankV1(
            (
                RowBandScorerTemplateV1("Movable", "fill", 0, 200, 1, 1, b"\x00\x00\x00"),
                RowBandScorerTemplateV1("Movable", "fill", 100, 384, 1, 1, b"\x00\x00\x00"),
            )
        )


def test_v14_measurement_config_is_full_window_local_only_and_stage_bounded() -> None:
    config = DDMV14RealizationFidelityConfigV1(
        run_id="fixture_v14",
        pair_start=0,
        pair_count=600,
        v13_receipt_path="v13.json",
        v13_receipt_sha256="1" * 64,
        lane_phase_receipt_path="phase.json",
        lane_phase_receipt_sha256="2" * 64,
        target_cache_path="gt_n600.npz",
        target_cache_bytes=5_078_017_610,
        target_cache_sha256="3" * 64,
        upstream_root="/absolute/upstream",
        scorer_threads=1,
    )
    assert config.candidate_ladder == ("islands", "both")
    assert config.max_candidate_stages_per_invocation == 1
    assert config.movable_prototype_rgb_u8 == (107, 0, 114)
    assert config.execution_allowed is False
    assert config.score_claim is False
    with pytest.raises(ValueError, match="v14 windows"):
        DDMV14RealizationFidelityConfigV1(
            **{
                **config.model_dump(by_alias=True),
                "pair_start": 448,
                "pair_count": 600,
            }
        )


def test_v13_refuses_orphan_knots_and_postsolve_correction_mixing() -> None:
    predictor = _predictor()
    with pytest.raises(DirectDescriptionError, match="outside its declared object lifecycle"):
        compile_carrier_compose_archive(
            predictor,
            worldsheet_knots=(MovableWorldsheetKnotV1(9, 1, 16),),
        )
    with pytest.raises(DirectDescriptionError, match="cannot be mixed"):
        compile_carrier_compose_archive(
            predictor,
            symbols=(LaneCoefficientDelta(0, 0, 3, 0.5),),
            worldsheet_tracks=(MovableWorldsheetTrackV1(1, 0, 2, 100, 200, 10, 18, 0, 0, 0, 0),),
        )
    labels = np.zeros((64, 384, 512), dtype=np.int64)
    labels[:, 20:30, 20:30] = 3
    payload, _metadata = encode_g1_movable_worldsheet(labels)
    with pytest.raises(DirectDescriptionError, match="cannot be mixed with the superseded"):
        compile_carrier_compose_archive(
            predictor,
            worldsheet_tracks=(MovableWorldsheetTrackV1(1, 0, 2, 100, 200, 10, 18, 0, 0, 0, 0),),
            worldsheet_g1_payload=payload,
        )


def test_v10_refuses_inert_or_out_of_window_semantic_events() -> None:
    predictor = _predictor()
    with pytest.raises(DirectDescriptionError, match="outside"):
        compile_carrier_compose_archive(
            predictor,
            topology_events=(TopologyEventV1(64, "Movable", "birth", "box", 1, 1, 1, 3, 3),),
        )
    with pytest.raises(DirectDescriptionError, match="no-op"):
        receive_carrier_compose_archive(
            compile_carrier_compose_archive(
                predictor,
                topology_events=(TopologyEventV1(0, "Movable", "death", "box", 1, 1, 1, 3, 3),),
            )[0]
        )


def test_v10_refuses_noncanonical_correction_member_order() -> None:
    archive = compile_carrier_compose_archive(
        _predictor(),
        boundary_symbols=(BoundaryCoefficientDelta(0, "Road", 0, 3.0),),
        topology_events=(TopologyEventV1(0, "Movable", "birth", "box", 1, 100, 200, 120, 220),),
    )[0]
    with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
        members = {name: reader.read(name) for name in reader.namelist()}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as writer:
        for name in ("manifest.json", "predictor.zip", EVENT_CORRECTION_MEMBER, BOUNDARY_CORRECTION_MEMBER):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            writer.writestr(info, members[name])
    with pytest.raises(DirectDescriptionError, match="member order"):
        parse_carrier_compose_archive(output.getvalue())


def test_v9_refuses_pixel_or_absent_chart_addresses_and_sampled_mutations() -> None:
    predictor = _predictor()
    with pytest.raises(DirectDescriptionError, match="source window"):
        compile_carrier_compose_archive(
            predictor,
            (LaneCoefficientDelta(64, 0, 3, 0.5),),
        )
    with pytest.raises(DirectDescriptionError, match="centerline"):
        receive_carrier_compose_archive(
            compile_carrier_compose_archive(
                predictor,
                (LaneCoefficientDelta(0, 0, 4, 0.5),),
            )[0]
        )
    archive = compile_carrier_compose_archive(predictor)[0]
    proof = prove_carrier_archive_fail_closed(archive)
    # `all_samples_refused_or_changed_decode` is `refused + changed == len(positions)`,
    # which is `0 == 0` -> True on an empty member walk.  Pin the DENOMINATOR so a
    # regression that mutates nothing cannot pass as a fail-closed proof.
    with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
        non_empty_members = sum(1 for info in reader.infolist() if info.file_size)
    assert non_empty_members >= 2
    assert proof["sampled_member_payload_homes"] == non_empty_members
    assert proof["non_empty_member_payload_count"] == non_empty_members
    assert proof["refused"] + proof["changed_decode"] == proof["sampled_member_payload_homes"]
    assert proof["unique_home_coverage_bytes"] == len(archive)
    assert proof["all_samples_refused_or_changed_decode"] is True


def test_v9_fail_closed_proof_refuses_a_vacuous_zero_sample_walk() -> None:
    """Positive control: zero sampled homes must RAISE, not report True.

    Without this the proof is NO-FAKE forbidden class 2 — a returned constant
    rather than behaviour — and 6 measurement tools publish it as
    `fail_closed_mutation_proof`.
    """

    archive = compile_carrier_compose_archive(_predictor())[0]
    positions, payload_members = compose._sampled_member_payload_positions(archive)
    assert positions and payload_members >= 2

    original = compose._sampled_member_payload_positions
    compose._sampled_member_payload_positions = lambda _archive: ([], 0)
    try:
        with pytest.raises(DirectDescriptionError, match="sampled no member payload homes"):
            prove_carrier_archive_fail_closed(archive)
    finally:
        compose._sampled_member_payload_positions = original
    # The guard must not have been a one-way trip.
    assert prove_carrier_archive_fail_closed(archive)["all_samples_refused_or_changed_decode"] is True
