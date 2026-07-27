from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.taskspace_v9_training_target_binding_v1 import (
    CHECKPOINT_PROJECTION_KEY,
    CHECKPOINT_PROJECTION_SHA_KEY,
    BoundV9TrainingTargetsV1,
    V9TrainingTargetBindingError,
    bind_v9_training_targets,
    reopen_v9_training_target_projection,
)
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    PREFLIGHT_SCHEMA,
    ScoredSourceBatchV1,
    file_identity,
    materialize_v9_training_target_capsule,
    seal_preflight,
    sha256_array_bytes,
    sha256_file,
    storage_preflight,
)


def _fixture(tmp_path: Path):
    root = tmp_path / "capsule"
    pair_count, batch_pairs = 5, 2
    camera_hw, seg_hw = (2, 3), (2, 2)
    gt_f0: list[np.ndarray] = []
    gt_f1: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    pair_source_rows: list[dict[str, object]] = []
    for pair_id in range(pair_count):
        pair = np.zeros((2, *camera_hw, 3), dtype=np.uint8)
        pair[0, 0, 0, 0] = pair_id
        pair[1, 0, 0, 1] = pair_id + 10
        gt_f0.append(pair[0].copy())
        gt_f1.append(pair[1].copy())
        pair_source_rows.append(
            {
                "pair_index": pair_id,
                "source_pair_rgb_sha256": sha256_array_bytes(pair),
            }
        )
    for start in range(0, pair_count, batch_pairs):
        sources.append(
            np.stack(
                [
                    np.stack((gt_f0[pair_id], gt_f1[pair_id]), axis=0)
                    for pair_id in range(start, min(start + batch_pairs, pair_count))
                ],
                axis=0,
            )
        )
    labels = (
        np.arange(pair_count * seg_hw[0] * seg_hw[1], dtype=np.uint8)
        .reshape(pair_count, *seg_hw)
        % 5
    )
    label_path = tmp_path / "g46_labels.u8"
    label_path.write_bytes(labels.tobytes())
    source_path = tmp_path / "source.mkv"
    source_path.write_bytes(b"source fixture")
    segnet_path = tmp_path / "segnet.safetensors"
    segnet_path.write_bytes(b"segnet fixture")
    posenet_path = tmp_path / "posenet.safetensors"
    posenet_path.write_bytes(b"posenet fixture")
    runtime_path = tmp_path / "runtime.py"
    runtime_path.write_bytes(b"# runtime fixture\n")
    g46_receipt_path = tmp_path / "g46_receipt.json"
    g46_receipt_path.write_text(
        json.dumps({"pair_checkpoints": pair_source_rows}),
        encoding="utf-8",
    )
    g46_preflight_path = tmp_path / "g46_preflight.json"
    g46_preflight_path.write_text('{"fixture":true}\n', encoding="utf-8")
    sealed = [
        {"role": "g46_labels", **file_identity(label_path)},
        {"role": "source", **file_identity(source_path)},
        {"role": "segnet", **file_identity(segnet_path)},
        {"role": "posenet", **file_identity(posenet_path)},
        {"role": "runtime", **file_identity(runtime_path)},
        {"role": "g46_receipt", **file_identity(g46_receipt_path)},
        {"role": "g46_preflight", **file_identity(g46_preflight_path)},
    ]
    preflight = seal_preflight(
        {
            "schema": PREFLIGHT_SCHEMA,
            "run_id": "test_g111_v9_binding",
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
            "seed": 0,
            "num_threads": 1,
            "test_only_small_fixture": True,
            "storage_preflight": storage_preflight(
                root,
                required_free_bytes=1,
                test_only_small_fixture=True,
            ),
            "config": {"fixture": True},
            "g46_custody": {
                "receipt": file_identity(g46_receipt_path),
                "receipt_sha256": "1" * 64,
                "preflight": file_identity(g46_preflight_path),
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
    )

    def score(source: np.ndarray) -> ScoredSourceBatchV1:
        pair_ids = source[:, 0, 0, 0, 0].astype(np.int64)
        target = labels[pair_ids]
        logits = np.full((len(pair_ids), 5, *seg_hw), -10.0, dtype=np.float32)
        for local in range(len(pair_ids)):
            for y in range(seg_hw[0]):
                for x in range(seg_hw[1]):
                    winner = int(target[local, y, x])
                    logits[local, (winner + 1) % 5, y, x] = 1.0
                    logits[local, winner, y, x] = float(pair_ids[local]) + 1.25
        poses = pair_ids.astype(np.float32)[:, None] + np.arange(6, dtype=np.float32)
        return ScoredSourceBatchV1(
            seg_logits_f32=logits,
            source_pose6_f32=poses,
            segnet_input_sha256=sha256_array_bytes(source.astype(np.float32)),
            posenet_input_sha256=sha256_array_bytes(source[:, ::-1].astype(np.float32)),
        )

    receipt_path, _receipt = materialize_v9_training_target_capsule(
        preflight=preflight,
        source_batches=sources,
        score_source_batch=score,
        allowed_roots=(),
    )
    return receipt_path, gt_f0, gt_f1, labels


def _scalar_cfg(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {key: np.asarray(value).item() for key, value in arrays.items()}


def test_real_capsule_source_binding_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    receipt_path, gt_f0, gt_f1, labels = _fixture(tmp_path)
    bound = bind_v9_training_targets(
        aggregate_receipt_path=receipt_path,
        expected_receipt_sha256=sha256_file(receipt_path),
        gt_f0=gt_f0,
        gt_f1=gt_f1,
        allowed_roots=(),
    )
    assert isinstance(bound, BoundV9TrainingTargetsV1)
    assert np.array_equal(bound.targets.seg_labels_u8, labels)
    arrays = bound.checkpoint_arrays(
        active_target_authority_sha256="a" * 64,
        verdict_batch=16,
    )
    projection = json.loads(str(np.asarray(arrays[CHECKPOINT_PROJECTION_KEY]).item()))
    assert projection == bound.projection
    assert projection["aggregate_schema"].endswith("_capsule_aggregate.v1")
    assert projection["same_forward_seg_margin_pose"] is True
    bound.validate_checkpoint_cfg(
        _scalar_cfg(arrays),
        active_target_authority_sha256="a" * 64,
        verdict_batch=16,
    )
    reopened = reopen_v9_training_target_projection(
        projection_json=str(np.asarray(arrays[CHECKPOINT_PROJECTION_KEY]).item()),
        expected_projection_sha256=str(
            np.asarray(arrays[CHECKPOINT_PROJECTION_SHA_KEY]).item()
        ),
    )
    assert reopened == bound.projection


def test_physical_projection_reopen_rejects_self_attested_json(
    tmp_path: Path,
) -> None:
    receipt_path, gt_f0, gt_f1, _labels = _fixture(tmp_path)
    bound = bind_v9_training_targets(
        aggregate_receipt_path=receipt_path,
        expected_receipt_sha256=sha256_file(receipt_path),
        gt_f0=gt_f0,
        gt_f1=gt_f1,
        allowed_roots=(),
    )
    arrays = bound.checkpoint_arrays(
        active_target_authority_sha256="9" * 64,
        verdict_batch=16,
    )
    projection_text = str(np.asarray(arrays[CHECKPOINT_PROJECTION_KEY]).item())
    with pytest.raises(V9TrainingTargetBindingError, match="SHA-256 differs"):
        reopen_v9_training_target_projection(
            projection_json=projection_text,
            expected_projection_sha256="8" * 64,
        )
    forged = json.loads(projection_text)
    forged["batch_digest_chain_sha256"] = "7" * 64
    forged_text = json.dumps(
        forged,
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    with pytest.raises(V9TrainingTargetBindingError, match="differs from"):
        reopen_v9_training_target_projection(
            projection_json=forged_text,
            expected_projection_sha256=hashlib.sha256(
                forged_text.encode("utf-8")
            ).hexdigest(),
        )


def test_source_cache_drift_and_resume_projection_drift_fail_closed(
    tmp_path: Path,
) -> None:
    receipt_path, gt_f0, gt_f1, _labels = _fixture(tmp_path)
    drifted = [frame.copy() for frame in gt_f1]
    drifted[3][0, 0, 2] ^= 1
    with pytest.raises(V9TrainingTargetBindingError, match="pair range"):
        bind_v9_training_targets(
            aggregate_receipt_path=receipt_path,
            expected_receipt_sha256=sha256_file(receipt_path),
            gt_f0=gt_f0,
            gt_f1=drifted,
            allowed_roots=(),
        )
    bound = bind_v9_training_targets(
        aggregate_receipt_path=receipt_path,
        expected_receipt_sha256=sha256_file(receipt_path),
        gt_f0=gt_f0,
        gt_f1=gt_f1,
        allowed_roots=(),
    )
    arrays = bound.checkpoint_arrays(
        active_target_authority_sha256="b" * 64,
        verdict_batch=16,
    )
    cfg = _scalar_cfg(arrays)
    cfg[CHECKPOINT_PROJECTION_KEY] = "{}"
    with pytest.raises(V9TrainingTargetBindingError, match=CHECKPOINT_PROJECTION_KEY):
        bound.validate_checkpoint_cfg(
            cfg,
            active_target_authority_sha256="b" * 64,
            verdict_batch=16,
        )


def test_verdict_batch_must_match_upstream() -> None:
    bound = BoundV9TrainingTargetsV1(
        targets=object(),  # type: ignore[arg-type]
        projection={},
        projection_sha256="c" * 64,
        target_evidence_sha256="d" * 64,
    )
    with pytest.raises(V9TrainingTargetBindingError, match="batch"):
        bound.checkpoint_arrays(
            active_target_authority_sha256="e" * 64,
            verdict_batch=32,
        )
