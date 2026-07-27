from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.taskspace_batch16_margin_base_scorer_cache_v1 import (
    PREFLIGHT_SCHEMA,
    Batch16MarginBaseScorerCacheError,
    BatchProductsV1,
    MarginBaseScorerCacheLoaderV1,
    PreparedBatchV1,
    canonical_json_bytes,
    file_identity,
    materialize_margin_base_scorer_cache,
    payload_sha256,
    seal_preflight,
    sha256_bytes,
    sha256_file,
    storage_preflight,
)


def _fixture(tmp_path: Path, *, run: str = "a"):
    root = tmp_path / f"out_{run}"
    labels = np.arange(10 * 2 * 3, dtype=np.uint8).reshape(10, 2, 3) % 5
    label_path = tmp_path / f"labels_{run}.u8"
    label_path.write_bytes(labels.tobytes())
    label_identity = {
        **file_identity(label_path),
        "shape": [10, 2, 3],
        "dtype": "uint8",
        "encoder_only": True,
        "candidate_payload_allowed": False,
    }
    sealed_inputs = [
        {
            "role": "target_labels",
            **{key: label_identity[key] for key in ("path", "bytes", "sha256")},
        }
    ]
    g51_stages = []
    for index in range(5):
        stage_root = tmp_path / f"g51_{run}_{index}"
        stage_root.mkdir()
        files = {}
        for name, payload in (
            ("manifest", canonical_json_bytes({"stage_index": index}) + b"\n"),
            ("y0_u8", bytes([index, 0])),
            ("y1_u8", bytes([index, 1])),
            ("gt_poses_f32", np.asarray([index], dtype="<f4").tobytes()),
        ):
            path = stage_root / name
            path.write_bytes(payload)
            files[name] = file_identity(path)
            sealed_inputs.append({"role": f"g51_{index}_{name}", **files[name]})
        g51_stages.append(
            {
                "stage_index": index,
                "pair_range": [2 * index, 2 * (index + 1)],
                "stage_receipt_sha256": f"{index + 1:064x}",
                "y0_y1_rederive_performed_by_g78": False,
                **files,
            }
        )
    typed_config_sha256 = "a" * 64
    camera_checkpoints = []
    camera_chain_material = []
    for index, start in enumerate(range(0, 10, 4)):
        stop = min(start + 4, 10)
        ids = np.arange(start, stop, dtype=np.uint16)
        camera_sha256 = sha256_bytes(ids.tobytes() + b"camera")
        identity_payload = {
            "schema": "ddm_v15_full_p_camera_identity_batch.v1",
            "typed_config_sha256": typed_config_sha256,
            "local_pair_range": [start, stop],
            "base_camera_sha256": camera_sha256,
            "final_camera_sha256": camera_sha256,
            "byte_identical": True,
            "camera_bytes_released_after_compare": True,
            "score_claim": False,
        }
        identity_path = tmp_path / f"v15_identity_{run}_{start:04d}_{stop:04d}.json"
        identity_path.write_bytes(canonical_json_bytes(identity_payload) + b"\n")
        identity = {
            **file_identity(identity_path),
            "pair_range": [start, stop],
            "camera_sha256": camera_sha256,
            "byte_identical": True,
        }
        camera_checkpoints.append(identity)
        camera_chain_material.append(camera_sha256 * 2)
        sealed_inputs.append({"role": f"v15_identity_{index}", **file_identity(identity_path)})
    runtime_path = tmp_path / f"runtime_{run}.py"
    runtime_path.write_bytes(b"# synthetic runtime custody\n")
    runtime_identity = {"role": "synthetic_runtime", **file_identity(runtime_path)}
    sealed_inputs.append(runtime_identity)
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "run_id": f"test_margin_base_{run}",
        "evidence_axis": "[synthetic test-only]",
        "research_only": True,
        "score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "pointer_mutation_allowed": False,
        "encoder_only": True,
        "dense_fields_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "output_root": str(root.resolve()),
        "pair_count": 10,
        "stage_pairs": 2,
        "stage_count": 5,
        "scorer_batch_pairs": 4,
        "scorer_hw": [2, 3],
        "class_count": 5,
        "seed": 1234,
        "num_threads": 1,
        "test_only_small_fixture": True,
        "storage_preflight": storage_preflight(
            root,
            required_free_bytes=1,
            test_only_small_fixture=True,
        ),
        "config": {"path": "/synthetic/config.json", "sha256": "1" * 64},
        "source_custody": {"fixture": True},
        "scorer_custody": {"fixture": True},
        "target_custody": {
            "target_labels": label_identity,
            "labels_reused_not_rederived": True,
        },
        "g51_y0_y1_custody": {
            "stages": g51_stages,
            "y0_y1_reused_not_rederived": True,
        },
        "semantic_custody": {
            "fixture": True,
            "full_p_camera_identity": {
                "pair_count": 10,
                "batch_size": 4,
                "batch_count": 3,
                "typed_config_sha256": typed_config_sha256,
                "digest_chain_sha256": hashlib.sha256("".join(camera_chain_material).encode("ascii")).hexdigest(),
                "checkpoints": camera_checkpoints,
                "all_camera_bytes_identical": True,
            },
        },
        "runtime_custody": {"fixture": True, "files": [runtime_identity]},
        "sealed_input_files": sealed_inputs,
        "run_argv": ["synthetic", "--materialize"],
        "resume_contract": {
            "global_batch_atomic": True,
            "stage_atomic": True,
        },
        "blockers_closed_by_successful_aggregate": [
            "G72_FRESH_BATCH16_TARGET_MARGIN_CUSTODY_OWED",
            "G72_FRESH_V15_CAMERA_R_BATCH16_BASE_SCORER_STAGE_CACHE_OWED",
        ],
    }
    return seal_preflight(body), labels, label_path


def _preparer(
    labels: np.ndarray,
    *,
    wrong_target: bool = False,
    camera_drift: bool = False,
    v15_input_drift: bool = False,
):
    counts = {"prepare": 0, "infer": 0}

    def prepare(pair_ids: tuple[int, ...]) -> PreparedBatchV1:
        counts["prepare"] += 1
        ids = np.asarray(pair_ids, dtype=np.uint16)

        def infer() -> BatchProductsV1:
            counts["infer"] += 1
            target = labels[list(pair_ids)].copy()
            if wrong_target:
                target[0, 0, 0] = (target[0, 0, 0] + 1) % 5
            margin = np.broadcast_to(
                np.asarray(pair_ids, dtype=np.float32)[:, None, None] + 0.25,
                target.shape,
            ).copy()
            described = ((labels[list(pair_ids)] + 1) % 5).astype(np.uint8)
            described_margin = margin + np.float32(0.5)
            return BatchProductsV1(
                target_cells_u8=target,
                target_margins_f32=margin,
                described_cells_u8=described,
                described_margins_f32=described_margin,
            )

        return PreparedBatchV1(
            source_pair_batch_sha256=sha256_bytes(ids.tobytes() + b"source"),
            target_scorer_input_sha256=sha256_bytes(ids.tobytes() + b"target-input"),
            v15_camera_sha256=sha256_bytes(ids.tobytes() + (b"camera-drift" if camera_drift else b"camera")),
            v15_scorer_input_sha256=sha256_bytes(
                ids.tobytes() + (b"v15-input-drift" if v15_input_drift else b"v15-input")
            ),
            infer=infer,
        )

    return prepare, counts


def _write_resealed_json(path: Path, value: dict, *, seal_field: str) -> dict:
    body = {key: item for key, item in value.items() if key != seal_field}
    sealed = {**body, seal_field: payload_sha256(body)}
    path.write_bytes(canonical_json_bytes(sealed) + b"\n")
    return sealed


def _rebind_stage_and_aggregate(
    aggregate_path: Path,
    aggregate: dict,
    *,
    stage_index: int,
    stage: dict,
) -> dict:
    stage_path = Path(aggregate["stages"][stage_index]["path"])
    stage = _write_resealed_json(stage_path, stage, seal_field="stage_receipt_sha256")
    aggregate["stages"][stage_index].update(file_identity(stage_path))
    aggregate["stages"][stage_index]["stage_receipt_sha256"] = stage["stage_receipt_sha256"]
    chain = hashlib.sha256()
    for binding in aggregate["stages"]:
        current = json.loads(Path(binding["path"]).read_bytes())
        chain.update(bytes.fromhex(current["stage_receipt_sha256"]))
        binding["digest_chain_sha256"] = chain.hexdigest()
    aggregate["stage_digest_chain_sha256"] = chain.hexdigest()
    return _write_resealed_json(
        aggregate_path,
        aggregate,
        seal_field="aggregate_receipt_sha256",
    )


def test_five_stages_preserve_global_batch_fragments_and_dense_fields(tmp_path: Path) -> None:
    preflight, labels, _label_path = _fixture(tmp_path)
    prepare, counts = _preparer(labels)
    aggregate_path, aggregate = materialize_margin_base_scorer_cache(
        preflight=preflight,
        prepare_batch=prepare,
        allowed_roots=(),
    )
    assert counts == {"prepare": 3, "infer": 3}
    assert [row["pair_range"] for row in aggregate["batches"]] == [
        [0, 4],
        [4, 8],
        [8, 10],
    ]
    assert [row["pair_range"] for row in aggregate["stages"]] == [
        [0, 2],
        [2, 4],
        [4, 6],
        [6, 8],
        [8, 10],
    ]
    loader = MarginBaseScorerCacheLoaderV1.open(
        aggregate_path,
        expected_sha256=file_identity(aggregate_path)["sha256"],
        allowed_roots=(),
    )
    stages = list(loader.iter_stages())
    assert len(stages) == 5
    for stage in stages:
        start, stop = stage.pair_range
        np.testing.assert_array_equal(stage.target_cells_u8, labels[start:stop])
        np.testing.assert_array_equal(
            stage.described_cells_u8,
            (labels[start:stop] + 1) % 5,
        )
        expected = np.broadcast_to(
            np.arange(start, stop, dtype=np.float32)[:, None, None] + 0.25,
            (stop - start, 2, 3),
        )
        np.testing.assert_array_equal(stage.target_margins_f32, expected)
        np.testing.assert_array_equal(stage.described_margins_f32, expected + 0.5)
    stage_one = aggregate["stages"][1]
    manifest = Path(stage_one["path"]).read_text(encoding="utf-8")
    assert '"batch_pair_range":[0,4]' in manifest
    assert '"local_slice":[2,4]' in manifest


def test_resume_rehashes_prepared_inputs_but_skips_completed_forwards(tmp_path: Path) -> None:
    preflight, labels, _label_path = _fixture(tmp_path)
    first, first_counts = _preparer(labels)
    path, _aggregate = materialize_margin_base_scorer_cache(
        preflight=preflight,
        prepare_batch=first,
        allowed_roots=(),
    )
    assert first_counts["infer"] == 3
    resumed, resumed_counts = _preparer(labels)
    resumed_path, resumed_aggregate = materialize_margin_base_scorer_cache(
        preflight=preflight,
        prepare_batch=resumed,
        allowed_roots=(),
    )
    assert resumed_path == path
    assert resumed_counts == {"prepare": 3, "infer": 0}
    assert resumed_aggregate["aggregate_receipt_sha256"] == _aggregate["aggregate_receipt_sha256"]


def test_resume_refuses_fresh_v15_camera_drift_before_completed_forward(tmp_path: Path) -> None:
    preflight, labels, _label_path = _fixture(tmp_path)
    first, _first_counts = _preparer(labels)
    materialize_margin_base_scorer_cache(
        preflight=preflight,
        prepare_batch=first,
        allowed_roots=(),
    )
    drifted, counts = _preparer(labels, camera_drift=True)
    with pytest.raises(
        Batch16MarginBaseScorerCacheError,
        match="fresh V15 camera bytes differ from owned identity",
    ):
        materialize_margin_base_scorer_cache(
            preflight=preflight,
            prepare_batch=drifted,
            allowed_roots=(),
        )
    assert counts == {"prepare": 1, "infer": 0}


def test_resume_refuses_v15_live_r_input_drift_without_forward(tmp_path: Path) -> None:
    preflight, labels, _label_path = _fixture(tmp_path)
    first, _first_counts = _preparer(labels)
    materialize_margin_base_scorer_cache(
        preflight=preflight,
        prepare_batch=first,
        allowed_roots=(),
    )
    drifted, counts = _preparer(labels, v15_input_drift=True)
    with pytest.raises(
        Batch16MarginBaseScorerCacheError,
        match="resumed V15 scorer input differs",
    ):
        materialize_margin_base_scorer_cache(
            preflight=preflight,
            prepare_batch=drifted,
            allowed_roots=(),
        )
    assert counts == {"prepare": 1, "infer": 0}


def test_owned_g46_target_mismatch_refuses_before_checkpoint(tmp_path: Path) -> None:
    preflight, labels, _label_path = _fixture(tmp_path)
    prepare, _counts = _preparer(labels, wrong_target=True)
    with pytest.raises(
        Batch16MarginBaseScorerCacheError,
        match="owned G46 labels",
    ):
        materialize_margin_base_scorer_cache(
            preflight=preflight,
            prepare_batch=prepare,
            allowed_roots=(),
        )
    assert not (Path(preflight["output_root"]) / "10_batch_checkpoints").exists()


def test_strict_loader_refuses_dense_stage_tamper(tmp_path: Path) -> None:
    preflight, labels, _label_path = _fixture(tmp_path)
    prepare, _counts = _preparer(labels)
    path, aggregate = materialize_margin_base_scorer_cache(
        preflight=preflight,
        prepare_batch=prepare,
        allowed_roots=(),
    )
    stage = Path(aggregate["stages"][0]["path"]).parent / "described_cells.u8"
    payload = bytearray(stage.read_bytes())
    payload[0] ^= 1
    stage.write_bytes(payload)
    with pytest.raises(Batch16MarginBaseScorerCacheError, match="described_cells_u8"):
        MarginBaseScorerCacheLoaderV1.open(
            path,
            expected_sha256=sha256_file(path),
            allowed_roots=(),
        )


def test_strict_loader_refuses_bound_target_bank_tamper(tmp_path: Path) -> None:
    preflight, labels, label_path = _fixture(tmp_path)
    prepare, _counts = _preparer(labels)
    path, _aggregate = materialize_margin_base_scorer_cache(
        preflight=preflight,
        prepare_batch=prepare,
        allowed_roots=(),
    )
    payload = bytearray(label_path.read_bytes())
    payload[-1] ^= 1
    label_path.write_bytes(payload)
    with pytest.raises(Batch16MarginBaseScorerCacheError, match="sealed input changed"):
        MarginBaseScorerCacheLoaderV1.open(
            path,
            expected_sha256=sha256_file(path),
            allowed_roots=(),
        )


def test_strict_loader_refuses_resealed_wrong_g51_stage_binding(tmp_path: Path) -> None:
    preflight, labels, _label_path = _fixture(tmp_path)
    prepare, _counts = _preparer(labels)
    path, aggregate = materialize_margin_base_scorer_cache(
        preflight=preflight,
        prepare_batch=prepare,
        allowed_roots=(),
    )
    stage_path = Path(aggregate["stages"][0]["path"])
    stage = json.loads(stage_path.read_bytes())
    stage["g51_y0_y1_stage"] = {"adversarial": "not the preflight G51 stage"}
    _rebind_stage_and_aggregate(
        path,
        aggregate,
        stage_index=0,
        stage=stage,
    )
    with pytest.raises(Batch16MarginBaseScorerCacheError, match="stage 0 G51 custody differs"):
        MarginBaseScorerCacheLoaderV1.open(
            path,
            expected_sha256=sha256_file(path),
            allowed_roots=(),
        )


def test_strict_loader_refuses_resealed_stage_bytes_divergent_from_batches(tmp_path: Path) -> None:
    preflight, labels, _label_path = _fixture(tmp_path)
    prepare, _counts = _preparer(labels)
    path, aggregate = materialize_margin_base_scorer_cache(
        preflight=preflight,
        prepare_batch=prepare,
        allowed_roots=(),
    )
    stage_path = Path(aggregate["stages"][0]["path"])
    stage = json.loads(stage_path.read_bytes())
    dense_path = Path(stage["files"]["described_cells_u8"]["path"])
    payload = bytearray(dense_path.read_bytes())
    payload[0] = (payload[0] + 1) % 5
    dense_path.write_bytes(payload)
    stage["files"]["described_cells_u8"].update(file_identity(dense_path))
    _rebind_stage_and_aggregate(
        path,
        aggregate,
        stage_index=0,
        stage=stage,
    )
    with pytest.raises(
        Batch16MarginBaseScorerCacheError,
        match="stage 0 described_cells_u8 differs from validated batch fragments",
    ):
        MarginBaseScorerCacheLoaderV1.open(
            path,
            expected_sha256=sha256_file(path),
            allowed_roots=(),
        )
