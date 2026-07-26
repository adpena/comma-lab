from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_control.taskspace_fresh_scorer_plane_materializer_v1 import (
    CONFIG_SCHEMA,
    MODE_DIRECT_TASK_LAYERED,
    SEMANTIC_STATUS_OWED,
    FreshScorerPlaneMaterializationError,
    FreshScorerPlaneOperandLoaderV1,
    exact_resize_round_u8,
    file_identity,
    materialize,
    open_stored_npy_memmap,
    payload_sha256,
    run_preflight,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _member_binding(path: Path, name: str) -> dict[str, object]:
    array = open_stored_npy_memmap(path, name)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": int(array.nbytes),
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


def make_fixture(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    pair_count = 4
    rng = np.random.default_rng(17)
    arrays = {
        "n_pairs": np.asarray(pair_count, dtype=np.int64),
        "gt_f0": rng.integers(0, 256, (pair_count, 4, 4, 3), dtype=np.uint8),
        "gt_f1": rng.integers(0, 256, (pair_count, 4, 4, 3), dtype=np.uint8),
        "gt_poses": rng.normal(size=(pair_count, 6)).astype(np.float64),
    }
    source = tmp_path / "source.npz"
    np.savez(source, **arrays, ignored=np.asarray([91], dtype=np.int64))
    labels = rng.integers(0, 5, (pair_count, 2, 2), dtype=np.uint8)
    label_path = tmp_path / "labels.u8"
    label_path.write_bytes(labels.tobytes())
    label_binding = {
        **file_identity(label_path),
        "shape": [pair_count, 2, 2],
        "dtype": "uint8",
    }
    teacher_body = {
        "schema": "tac.test_fresh_teacher.v1",
        "pair_count": pair_count,
        "batch_size": 16,
        "encoder_only": True,
        "candidate_payload_allowed": False,
        "score_claim": False,
        "target_labels": label_binding,
    }
    teacher = {**teacher_body, "receipt_sha256": payload_sha256(teacher_body)}
    teacher_path = tmp_path / "teacher.json"
    _write_json(teacher_path, teacher)
    producer_path = tmp_path / "producer.py"
    producer_path.write_text("# fixture producer\n", encoding="utf-8")
    source_identity = file_identity(source)
    config = {
        "schema": CONFIG_SCHEMA,
        "run_id": "fixture_fresh_planes",
        "mode": MODE_DIRECT_TASK_LAYERED,
        "semantic_status": SEMANTIC_STATUS_OWED,
        "source_npz": {
            **source_identity,
            "members": {
                name: _member_binding(source, name)
                for name in ("n_pairs", "gt_f0", "gt_f1", "gt_poses")
            },
        },
        "fresh_teacher_receipt": {
            **file_identity(teacher_path),
            "sealed_receipt_sha256": teacher["receipt_sha256"],
            "scorer_pair_batch_size": 16,
        },
        "target_labels": label_binding,
        "producer_sources": [{"role": "fixture", **file_identity(producer_path)}],
        "output_root": str((tmp_path / "out").resolve()),
        "pair_count": pair_count,
        "stage_pairs": 2,
        "camera_hw": [4, 4],
        "scorer_hw": [2, 2],
        "resume": True,
        "test_only_small_fixture": True,
        "required_free_bytes": 1,
        "truth": {
            "research_only": True,
            "score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "source_cache_pose_advisory_only": True,
            "fresh_pose_target_custody": False,
            "program_residual_layered_available": False,
        },
    }
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    return config_path, {**arrays, "labels": labels}


def test_materialize_resume_and_bounded_loader_are_exact(tmp_path: Path) -> None:
    config_path, arrays = make_fixture(tmp_path)
    preflight_path, preflight = run_preflight(config_path)
    assert preflight_path.is_file()
    assert preflight["materialization_started"] is False
    aggregate_path, _aggregate = materialize(config_path)
    receipt_sha = sha256_file(aggregate_path)
    loader = FreshScorerPlaneOperandLoaderV1.open(
        aggregate_path, expected_sha256=receipt_sha
    )
    stages = list(loader.iter_stages(max_pairs=2))
    assert [stage.pair_range for stage in stages] == [(0, 2), (2, 4)]
    operator = DisjointResizeOperator.build(
        camera_h=4, camera_w=4, scorer_h=2, scorer_w=2
    )
    expected = exact_resize_round_u8(operator, arrays["gt_f0"][0])
    assert np.array_equal(stages[0].y0_u8[0], expected)
    assert np.array_equal(stages[0].target_labels_u8, arrays["labels"][:2])
    assert np.array_equal(
        stages[0].gt_poses_f32, arrays["gt_poses"][:2].astype(np.float32)
    )
    assert stages[0].pose_authority == "SEALED_SOURCE_CACHE_ADVISORY_ONLY"
    assert [chunk.pair_range for chunk in loader.iter_chunks(1)] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    aggregate_path_2, aggregate_2 = materialize(config_path)
    assert aggregate_path_2 == aggregate_path
    assert aggregate_2["coverage"]["all_stage_rederive_equal"] is True


def test_loader_refuses_stage_tamper(tmp_path: Path) -> None:
    config_path, _arrays = make_fixture(tmp_path)
    aggregate_path, aggregate = materialize(config_path)
    stage_file = Path(aggregate["stages"][0]["files"]["y0_u8"]["path"])
    with stage_file.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(FreshScorerPlaneMaterializationError, match="identity differs"):
        FreshScorerPlaneOperandLoaderV1.open(aggregate_path)


def test_config_refuses_unexpected_keys_and_authority_minting(tmp_path: Path) -> None:
    config_path, _arrays = make_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["historical_payload_path"] = "/forbidden"
    _write_json(config_path, config)
    with pytest.raises(FreshScorerPlaneMaterializationError, match="config keys differ"):
        run_preflight(config_path)
    config.pop("historical_payload_path")
    config["truth"]["fresh_pose_target_custody"] = True
    _write_json(config_path, config)
    with pytest.raises(FreshScorerPlaneMaterializationError, match="truth boundary"):
        run_preflight(config_path)


def test_preflight_refuses_target_label_drift(tmp_path: Path) -> None:
    config_path, _arrays = make_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    label_path = Path(config["target_labels"]["path"])
    with label_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(FreshScorerPlaneMaterializationError, match="identity differs"):
        run_preflight(config_path)
