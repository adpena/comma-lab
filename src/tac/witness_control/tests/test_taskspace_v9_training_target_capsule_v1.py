from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tac.witness_control.taskspace_v9_training_target_capsule_v1 as capsule
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    PREFLIGHT_SCHEMA,
    ScoredSourceBatchV1,
    V9TrainingTargetCapsuleError,
    V9TrainingTargetCapsuleLoaderV1,
    file_identity,
    materialize_v9_training_target_capsule,
    seal_preflight,
    sha256_array_bytes,
    sha256_file,
    storage_preflight,
)


def _fixture(tmp_path: Path, *, name: str = "a"):
    root = tmp_path / f"capsule_{name}"
    pair_count = 5
    batch_pairs = 2
    camera_hw = (2, 3)
    seg_hw = (2, 2)
    labels = np.arange(pair_count * seg_hw[0] * seg_hw[1], dtype=np.uint8).reshape(pair_count, *seg_hw) % 5
    label_path = tmp_path / f"g46_labels_{name}.u8"
    label_path.write_bytes(labels.tobytes())
    source_path = tmp_path / f"source_{name}.mkv"
    source_path.write_bytes(b"exact source fixture")
    segnet_path = tmp_path / f"segnet_{name}.safetensors"
    segnet_path.write_bytes(b"exact segnet fixture")
    posenet_path = tmp_path / f"posenet_{name}.safetensors"
    posenet_path.write_bytes(b"exact posenet fixture")
    runtime_path = tmp_path / f"runtime_{name}.py"
    runtime_path.write_bytes(b"# exact runtime fixture\n")
    receipt_path = tmp_path / f"g46_receipt_{name}.json"
    receipt_path.write_bytes(b'{"fixture":true}\n')
    preflight_path = tmp_path / f"g46_preflight_{name}.json"
    preflight_path.write_bytes(b'{"fixture":true}\n')
    sealed = [
        {"role": "g46_labels", **file_identity(label_path)},
        {"role": "source", **file_identity(source_path)},
        {"role": "segnet", **file_identity(segnet_path)},
        {"role": "posenet", **file_identity(posenet_path)},
        {"role": "runtime", **file_identity(runtime_path)},
        {"role": "g46_receipt", **file_identity(receipt_path)},
        {"role": "g46_preflight", **file_identity(preflight_path)},
    ]
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "run_id": f"test_v9_capsule_{name}",
        "evidence_axis": "[synthetic test-only]",
        "research_only": True,
        "encoder_only": True,
        "score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "pointer_mutation_allowed": False,
        "dense_targets_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "output_root": str(root.resolve()),
        "pair_count": pair_count,
        "batch_pairs": batch_pairs,
        "camera_hw": list(camera_hw),
        "seg_hw": list(seg_hw),
        "class_count": 5,
        "pose_dim": 6,
        "seed": 1234,
        "num_threads": 1,
        "test_only_small_fixture": True,
        "storage_preflight": storage_preflight(
            root,
            required_free_bytes=1,
            test_only_small_fixture=True,
        ),
        "config": {"fixture": True},
        "g46_custody": {
            "receipt": file_identity(receipt_path),
            "receipt_sha256": "1" * 64,
            "preflight": file_identity(preflight_path),
            "preflight_sha256": "2" * 64,
            "target_labels": {
                **file_identity(label_path),
                "shape": [pair_count, *seg_hw],
                "dtype": "uint8",
                "encoder_only": True,
                "candidate_payload_allowed": False,
            },
            "pair_checkpoint_root_sha256": "3" * 64,
            "compile_ready_reopened": True,
            "labels_reused_as_validation_authority": True,
        },
        "source_custody": {
            "source_video": file_identity(source_path),
            "fixture": True,
        },
        "scorer_custody": {
            "segnet_weights": file_identity(segnet_path),
            "posenet_weights": file_identity(posenet_path),
            "fixture": True,
        },
        "runtime_custody": {
            "files": [{"role": "runtime", **file_identity(runtime_path)}],
            "fixture": True,
        },
        "sealed_input_files": sealed,
        "run_argv": ["synthetic", "--materialize"],
        "resume_contract": {
            "batch_atomic": True,
            "completed_batches_skip_scorer_forward": True,
        },
        "cleanup_contract": {
            "scratch_root": str((root / ".scratch").resolve()),
            "certify_or_block": True,
        },
    }
    sources: list[np.ndarray] = []
    for start in range(0, pair_count, batch_pairs):
        stop = min(start + batch_pairs, pair_count)
        batch = np.zeros((stop - start, 2, *camera_hw, 3), dtype=np.uint8)
        for local, pair_id in enumerate(range(start, stop)):
            batch[local, 0, 0, 0, 0] = pair_id
        sources.append(batch)
    return (
        seal_preflight(body),
        labels,
        sources,
        {
            "root": root,
            "posenet": posenet_path,
        },
    )


def _scorer(
    labels: np.ndarray,
    *,
    wrong_first_label: bool = False,
    fail_on_call: int | None = None,
):
    counts = {"calls": 0}

    def score(source: np.ndarray) -> ScoredSourceBatchV1:
        counts["calls"] += 1
        if fail_on_call == counts["calls"]:
            raise RuntimeError("synthetic scorer interruption")
        pair_ids = source[:, 0, 0, 0, 0].astype(np.int64)
        target = labels[pair_ids]
        logits = np.full(
            (len(pair_ids), 5, *target.shape[1:]),
            -10.0,
            dtype=np.float32,
        )
        expected_margins = pair_ids.astype(np.float32) + np.float32(0.25)
        for local, _pair_id in enumerate(pair_ids):
            for y in range(target.shape[1]):
                for x in range(target.shape[2]):
                    winner = int(target[local, y, x])
                    runner = (winner + 1) % 5
                    logits[local, runner, y, x] = np.float32(1.0)
                    logits[local, winner, y, x] = np.float32(1.0 + expected_margins[local])
        if wrong_first_label and int(pair_ids[0]) == 0:
            winner = int(target[0, 0, 0])
            logits[0, (winner + 2) % 5, 0, 0] = np.float32(100.0)
        poses = pair_ids.astype(np.float32)[:, None] + np.arange(6, dtype=np.float32)[None, :] / np.float32(10.0)
        return ScoredSourceBatchV1(
            seg_logits_f32=logits,
            source_pose6_f32=np.ascontiguousarray(poses, dtype=np.float32),
            segnet_input_sha256=sha256_array_bytes(source.astype(np.float32)),
            posenet_input_sha256=sha256_array_bytes(source[:, ::-1].astype(np.float32)),
        )

    return score, counts


def test_materializes_margin_pose_shapes_and_strict_loader_projection(
    tmp_path: Path,
) -> None:
    preflight, labels, sources, _paths = _fixture(tmp_path)
    score, counts = _scorer(labels)
    receipt_path, receipt = materialize_v9_training_target_capsule(
        preflight=preflight,
        source_batches=sources,
        score_source_batch=score,
        allowed_roots=(),
    )
    assert counts == {"calls": 3}
    assert [row["pair_range"] for row in receipt["batches"]] == [
        [0, 2],
        [2, 4],
        [4, 5],
    ]
    loader = V9TrainingTargetCapsuleLoaderV1.open(
        receipt_path,
        expected_sha256=sha256_file(receipt_path),
        allowed_roots=(),
    )
    assert loader.targets.seg_labels_u8.shape == (5, 2, 2)
    assert loader.targets.seg_top1_minus_top2_margin_f32.shape == (5, 2, 2)
    assert loader.targets.source_pose6_f32.shape == (5, 6)
    assert np.array_equal(loader.targets.seg_labels_u8, labels)
    expected_margin = np.broadcast_to(
        np.arange(5, dtype=np.float32)[:, None, None] + np.float32(0.25),
        (5, 2, 2),
    )
    assert np.array_equal(
        loader.targets.seg_top1_minus_top2_margin_f32,
        expected_margin,
    )
    expected_pose = np.arange(5, dtype=np.float32)[:, None] + np.arange(6, dtype=np.float32)[None, :] / np.float32(10.0)
    assert np.array_equal(loader.targets.source_pose6_f32, expected_pose)
    with np.load(receipt["npz"]["path"], allow_pickle=False) as archive:
        assert archive.files == [
            "seg_labels_u8",
            "seg_top1_minus_top2_margin_f32",
            "source_pose6_f32",
        ]
        assert np.array_equal(archive["seg_labels_u8"], labels)


def test_batch_checkpoint_is_last_and_partial_batch_resumes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, labels, sources, paths = _fixture(tmp_path, name="atomic")
    score, _counts = _scorer(labels)
    original = capsule._write_immutable_bytes
    failed = {"done": False}

    def interrupt_pose(path: Path, payload: bytes) -> None:
        if path.name.endswith(".source_pose6.f32") and not failed["done"]:
            failed["done"] = True
            raise RuntimeError("synthetic write interruption")
        original(path, payload)

    monkeypatch.setattr(capsule, "_write_immutable_bytes", interrupt_pose)
    with pytest.raises(RuntimeError, match="synthetic write interruption"):
        materialize_v9_training_target_capsule(
            preflight=preflight,
            source_batches=sources,
            score_source_batch=score,
            allowed_roots=(),
        )
    first = capsule._batch_paths(paths["root"], 0, 2)
    assert first["labels"].is_file()
    assert first["margins"].is_file()
    assert not first["poses"].exists()
    assert not first["checkpoint"].exists()
    monkeypatch.setattr(capsule, "_write_immutable_bytes", original)
    resumed_score, resumed_counts = _scorer(labels)
    receipt_path, _receipt = materialize_v9_training_target_capsule(
        preflight=preflight,
        source_batches=sources,
        score_source_batch=resumed_score,
        allowed_roots=(),
    )
    assert resumed_counts == {"calls": 3}
    assert receipt_path.is_file()
    assert first["checkpoint"].is_file()


def test_full_resume_rehashes_source_and_skips_all_scorer_forwards(
    tmp_path: Path,
) -> None:
    preflight, labels, sources, _paths = _fixture(tmp_path, name="resume")
    score, _counts = _scorer(labels)
    first_path, first_receipt = materialize_v9_training_target_capsule(
        preflight=preflight,
        source_batches=sources,
        score_source_batch=score,
        allowed_roots=(),
    )

    def forbidden(_source: np.ndarray) -> ScoredSourceBatchV1:
        raise AssertionError("completed batches must skip scorer forward")

    second_path, second_receipt = materialize_v9_training_target_capsule(
        preflight=preflight,
        source_batches=[batch.copy() for batch in sources],
        score_source_batch=forbidden,
        allowed_roots=(),
    )
    assert second_path == first_path
    assert second_receipt == first_receipt
    drifted = [batch.copy() for batch in sources]
    drifted[0][0, 1, 0, 0, 0] = 1
    with pytest.raises(V9TrainingTargetCapsuleError, match="source bytes differ"):
        materialize_v9_training_target_capsule(
            preflight=preflight,
            source_batches=drifted,
            score_source_batch=forbidden,
            allowed_roots=(),
        )


def test_argmax_mismatch_refuses_before_batch_commit(tmp_path: Path) -> None:
    preflight, labels, sources, paths = _fixture(tmp_path, name="labels")
    score, _counts = _scorer(labels, wrong_first_label=True)
    with pytest.raises(V9TrainingTargetCapsuleError, match="argmax differs from G46"):
        materialize_v9_training_target_capsule(
            preflight=preflight,
            source_batches=sources,
            score_source_batch=score,
            allowed_roots=(),
        )
    first = capsule._batch_paths(paths["root"], 0, 2)
    assert not first["labels"].exists()
    assert not first["margins"].exists()
    assert not first["poses"].exists()
    assert not first["checkpoint"].exists()


def test_loader_refuses_raw_tamper_and_model_custody_drift(tmp_path: Path) -> None:
    preflight, labels, sources, paths = _fixture(tmp_path, name="tamper")
    score, _counts = _scorer(labels)
    receipt_path, receipt = materialize_v9_training_target_capsule(
        preflight=preflight,
        source_batches=sources,
        score_source_batch=score,
        allowed_roots=(),
    )
    margin_path = Path(receipt["raw_arrays"]["margins"]["path"])
    payload = bytearray(margin_path.read_bytes())
    payload[0] ^= 1
    margin_path.write_bytes(payload)
    with pytest.raises(V9TrainingTargetCapsuleError, match="file identity differs"):
        V9TrainingTargetCapsuleLoaderV1.open(
            receipt_path,
            expected_sha256=sha256_file(receipt_path),
            allowed_roots=(),
        )

    preflight2, labels2, sources2, paths2 = _fixture(tmp_path, name="custody")
    score2, _counts2 = _scorer(labels2)
    receipt_path2, _receipt2 = materialize_v9_training_target_capsule(
        preflight=preflight2,
        source_batches=sources2,
        score_source_batch=score2,
        allowed_roots=(),
    )
    paths2["posenet"].write_bytes(b"tampered posenet fixture")
    with pytest.raises(V9TrainingTargetCapsuleError, match="sealed input changed"):
        V9TrainingTargetCapsuleLoaderV1.open(
            receipt_path2,
            expected_sha256=sha256_file(receipt_path2),
            allowed_roots=(),
        )


def test_loader_requires_external_receipt_sha_and_cleanup_is_certified(
    tmp_path: Path,
) -> None:
    preflight, labels, sources, paths = _fixture(tmp_path, name="loader")
    scratch = paths["root"] / ".scratch"
    scratch.mkdir(parents=True)
    (scratch / "orphan.tmp.crash").write_bytes(b"rebuildable scratch")
    score, _counts = _scorer(labels)
    receipt_path, _receipt = materialize_v9_training_target_capsule(
        preflight=preflight,
        source_batches=sources,
        score_source_batch=score,
        allowed_roots=(),
    )
    assert not (scratch / "orphan.tmp.crash").exists()
    cleanup_receipts = list((paths["root"] / "01_cleanup_receipts").glob("cleanup_*.json"))
    assert len(cleanup_receipts) == 1
    with pytest.raises(
        V9TrainingTargetCapsuleError,
        match="aggregate receipt file SHA-256 differs",
    ):
        V9TrainingTargetCapsuleLoaderV1.open(
            receipt_path,
            expected_sha256="0" * 64,
            allowed_roots=(),
        )
