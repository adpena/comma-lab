# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tac.optimization.ddm_g3_score_atlas import (
    PRIMARY_RANK_KEY,
    DdmG3CostatePairSignalV1,
    DdmG3ScoreAtlasConfigV1,
    ScoreAtlasError,
    ScoreMassV1,
    _accepted_bytes_by_pair,
    _boundary_mask,
    _load_batch_cache,
    _rank4_flip_distances,
    _scene_event_proxies,
    _seg_cube,
    audit_evaluator_response_cone_maps,
    build_admission_efficiency_rows,
    select_stratified_control,
)


def _config(**updates) -> dict:
    row = {
        "schema": "DdmG3ScoreAtlasConfigV1",
        "run_id": "fixture",
        "evaluator_atlas_path": "/ssd/atlas.jsonl",
        "evaluator_atlas_sha256": "a" * 64,
        "v12_receipt_path": "receipt.json",
        "v12_receipt_sha256": "b" * 64,
        "admission_receipts": [
            {"version": "v10", "pair_count": 256, "path": "v10.json", "sha256": "c" * 64},
            {"version": "v11", "pair_count": 600, "path": "v11.json", "sha256": "d" * 64},
            {"version": "v12", "pair_count": 600, "path": "receipt.json", "sha256": "b" * 64},
        ],
        "v13_pointer_ledger_path": ".omx/state/operator_p0_ledger.jsonl",
        "v13_pointer_ledger_sha256": "e" * 64,
        "v13_pointer_p0_id": "p0_fixture",
        "output_directory": "/ssd/out",
        "compact_receipt_directory": ".omx/research/out",
        "n_pairs": 600,
        "cache_chunk_size": 16,
        "row_bins": 12,
        "seed": 1234,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
    }
    row.update(updates)
    return row


def _rows() -> list[dict]:
    return [
        {
            "pair_index": index,
            "score_rank": 600 - index,
            "score_mass": {"distortion_score_mass": float(index + 1)},
            "segmentation": {
                "boundary_flip_count": index % 17,
                "class_flip_counts": {"Road": index % 11, "Lane": index % 13, "Movable": index % 19},
            },
            "scene_covariates": {
                "movable_track_count": index % 5,
                "stored_turn_code_l2": float(index % 23),
                "dash_phase_delta": float((index % 7) - 3),
            },
        }
        for index in range(600)
    ]


def test_typed_config_hash_is_deterministic_and_authority_is_literal():
    left = DdmG3ScoreAtlasConfigV1.model_validate(_config())
    right = DdmG3ScoreAtlasConfigV1.model_validate(_config())
    assert left.typed_hash() == right.typed_hash()
    with pytest.raises(ValidationError):
        DdmG3ScoreAtlasConfigV1.model_validate(_config(score_claim=True))
    with pytest.raises(ValidationError):
        DdmG3ScoreAtlasConfigV1.model_validate(_config(evaluator_atlas_sha256="bad"))


def test_score_mass_rejects_nonadditive_or_noncanonical_rank_key():
    valid = ScoreMassV1(
        seg_score_mass=1.0,
        pose_score_mass=2.0,
        distortion_score_mass=3.0,
        rate_score_mass_diagnostic=0.1,
    )
    assert valid.rank_key == PRIMARY_RANK_KEY
    with pytest.raises(ValidationError):
        ScoreMassV1(
            seg_score_mass=1.0,
            pose_score_mass=2.0,
            distortion_score_mass=4.0,
            rate_score_mass_diagnostic=0.1,
        )
    with pytest.raises(ValidationError):
        ScoreMassV1(
            seg_score_mass=1.0,
            pose_score_mass=2.0,
            distortion_score_mass=3.0,
            rate_score_mass_diagnostic=0.1,
            rank_key="l2_energy",
        )


def test_costate_consumer_schema_is_strict_and_score_currency_bound():
    row = {
        "schema": "ddm_g3_costate_pair_signal.v1",
        "pair_index": 4,
        "lambda_proxy_score_debt": 1.0,
        "seg_flip_count": 3,
        "median_rank4_flip_distance": 0.2,
        "pose_squared_error_sum": 2.0,
        "pose_binds_fraction": 0.5,
        "allocated_bytes": 4.0,
        "ranking_currency": "exact_flip_plus_pose_objective_mass",
        "energy_rank_forbidden": True,
        "score_claim": False,
    }
    assert DdmG3CostatePairSignalV1.model_validate(row).pair_index == 4
    with pytest.raises(ValidationError):
        DdmG3CostatePairSignalV1.model_validate({**row, "energy_rank_forbidden": False})
    with pytest.raises(ValidationError):
        DdmG3CostatePairSignalV1.model_validate({**row, "unknown": 1})


def test_rank4_flip_distance_uses_target_predicted_head_pair_norm():
    target = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
    predicted = np.asarray([[1, 1], [4, 0]], dtype=np.uint8)
    margins = np.asarray([[3.953, 8.0], [2.869, 2.942]], dtype=np.float32)
    distances = _rank4_flip_distances(target, predicted, margins)
    assert distances.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_boundary_mask_marks_both_sides_and_not_flat_interior():
    labels = np.zeros((1, 4, 4), dtype=np.uint8)
    labels[:, :, 2:] = 1
    boundary = _boundary_mask(labels)[0]
    assert boundary[:, 1:3].all()
    assert not boundary[:, 0].any()
    assert not boundary[:, 3].any()


def test_seg_cube_exposes_conditional_and_global_score_mass():
    target = np.zeros((2, 2), dtype=np.uint8)
    errors = np.asarray([[True, False], [False, False]])
    margins = np.full((2, 2), 0.05, dtype=np.float32)
    boundary = np.asarray([[True, True], [False, False]])
    cube = _seg_cube(target, errors, margins, boundary, total_video_sites=100)
    cell = cube["Road"]["[0,0.1)"]["boundary_codim1"]
    assert cell["conditional_d_seg"] == 0.5
    assert cell["global_d_seg_mass"] == 0.01
    assert cell["seg_score_mass"] == 1.0


def test_accepted_byte_attribution_is_exact_and_causal():
    rows = [
        {
            "admitted": True,
            "candidate": {"source_pair_ids": [1, 3]},
            "measurement": {"marginal_archive_bytes": 10},
        },
        {
            "admitted": False,
            "candidate": {"source_pair_ids": [1]},
            "measurement": {"marginal_archive_bytes": 99},
        },
    ]
    result = _accepted_bytes_by_pair(rows, 5)
    assert result.sum() == 10
    assert result.tolist() == [0, 5, 0, 5, 0]


def test_batch_cache_loader_validates_sha_schema_and_tensor_shapes(tmp_path: Path):
    cells = np.zeros((2, 384, 512), dtype=np.uint8)
    poses = np.zeros((2, 6), dtype=np.float64)
    header = {
        "schema": "ddm_canonical_batch_score_cache.v1",
        "identity": "fixture",
        "start": 0,
        "cells_dtype": cells.dtype.str,
        "cells_shape": list(cells.shape),
        "cells_bytes": cells.nbytes,
        "poses_dtype": poses.dtype.str,
        "poses_shape": list(poses.shape),
        "poses_bytes": poses.nbytes,
        "errors": 0,
        "pose_squared_error": "0",
    }
    payload = (
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        + zlib.compress(cells.tobytes() + poses.tobytes())
    )
    path = tmp_path / "0000.score.zlib"
    path.write_bytes(payload)
    manifest = {"path": str(path), "start": 0, "sha256": hashlib.sha256(payload).hexdigest()}
    loaded = _load_batch_cache(tmp_path, manifest)
    assert loaded.cells.shape == cells.shape
    with pytest.raises(ScoreAtlasError):
        _load_batch_cache(tmp_path, {**manifest, "sha256": "0" * 64})


def test_evaluator_response_audit_rehashes_every_cone_map(tmp_path: Path):
    rows = {}
    for pair_index, payload in enumerate((b"cone-a", b"cone-b")):
        path = tmp_path / f"cone_{pair_index}.npz"
        path.write_bytes(payload)
        rows[pair_index] = {
            "sensitivity_refs": {
                "cone_map_path": str(path),
                "cone_map_sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
    result = audit_evaluator_response_cone_maps(rows)
    assert result["map_count"] == 2
    assert result["map_bytes"] == 12
    assert result["consumability_verdict"] == "INTACT_FOR_GEOMETRY_ONLY"
    rows[0]["sensitivity_refs"]["cone_map_sha256"] = "0" * 64
    with pytest.raises(ScoreAtlasError):
        audit_evaluator_response_cone_maps(rows)


def test_stratified_control_is_temporal_score_stratified_and_disjoint():
    rows = _rows()
    excluded = set(range(64))
    result = select_stratified_control(rows, excluded=excluded, k=24)
    assert len(result) == len(set(result)) == 24
    assert not set(result) & excluded
    assert [sum(1 for value in result if bucket * 100 <= value < (bucket + 1) * 100) for bucket in range(6)] == [4] * 6


def test_named_scene_proxies_are_deterministic_and_explicitly_proxies():
    result = _scene_event_proxies(_rows())
    labels = [label for values in result.values() for label in values]
    assert sorted(labels) == ["intersection_proxy", "lane_change_proxy", "lead_car_pass_proxy"]
    assert _scene_event_proxies(_rows()) == result


def test_admission_efficiency_uses_exact_flip_delta_and_bytes():
    receipt = {
        "candidate_search": {
            "admission_rows": [
                {
                    "candidate": {"candidate_id": "c0", "source_pair_ids": [9]},
                    "admitted": True,
                    "measurement": {
                        "marginal_archive_bytes": 2,
                        "errors_before": 10,
                        "errors_after": 6,
                        "measured_objective_gain": "0.5",
                    },
                }
            ]
        }
    }
    result = build_admission_efficiency_rows(receipt, source_version="v12", n_pairs=600)[0]
    assert result["seg_flip_reduction"] == 4
    assert result["delta_d_seg_per_byte"] == pytest.approx(2 / (600 * 384 * 512))
    assert result["joint_objective_gain_per_byte"] == 0.25
    assert math.isfinite(result["delta_d_seg_per_byte"])


@pytest.mark.parametrize(
    ("version", "measurement", "expected_gain"),
    [
        (
            "v10",
            {
                "marginal_archive_bytes": 2,
                "errors_before": 10,
                "errors_after": 6,
                "distortion_gain_score_units": "0.5",
                "rate_cost_score_units": "0.1",
            },
            0.4,
        ),
        (
            "v11",
            {
                "marginal_archive_bytes": 2,
                "errors_before": 10,
                "errors_after": 6,
                "joint_objective_delta": "-0.4",
            },
            0.4,
        ),
    ],
)
def test_admission_efficiency_normalizes_version_sign_conventions(version, measurement, expected_gain):
    receipt = {
        "candidate_search": {
            "admission_rows": [
                {
                    "candidate": {"candidate_id": "c0", "source_pair_ids": [9]},
                    "admitted": True,
                    "measurement": measurement,
                }
            ]
        }
    }
    result = build_admission_efficiency_rows(receipt, source_version=version, n_pairs=600)[0]
    assert result["joint_objective_gain"] == pytest.approx(expected_gain)
