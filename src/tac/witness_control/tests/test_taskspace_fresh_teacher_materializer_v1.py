# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (
    UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
    FreshTeacherMaterializationError,
    PreparedTeacherBatchV1,
    build_fresh_teacher_preflight,
    load_and_reverify_materialization_receipt,
    load_compile_ready_materialization_receipt,
    materialize_fresh_teacher_from_batches,
    reverify_preflight,
)


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _inputs(
    tmp_path: Path,
    *,
    pair_count: int = 3,
    batch_size: int = 2,
) -> tuple[dict, tuple[np.ndarray, ...]]:
    source = tmp_path / "inputs" / "0.mkv"
    weights = tmp_path / "inputs" / "segnet.safetensors"
    upstream = tmp_path / "upstream"
    source.parent.mkdir(parents=True)
    upstream.mkdir()
    source.write_bytes(b"exact-source-video")
    weights.write_bytes(b"exact-segnet-weights")
    for relative in ("evaluate.py", "frame_utils.py", "modules.py", "public_test_video_names.txt"):
        (upstream / relative).write_bytes((relative + "\n").encode())
    output = tmp_path / "ssd" / "teacher"
    preflight = build_fresh_teacher_preflight(
        source_video=source,
        segnet_weights=weights,
        upstream_root=upstream,
        output_root=output,
        pair_count=pair_count,
        batch_size=batch_size,
        num_threads=1,
        seed=7,
        package_versions={"numpy": "test", "torch": "test"},
        run_argv=("python", "tool.py", "--mode", "run"),
        camera_hw=(2, 3),
        seg_hw=(2, 2),
        safety_reserve_bytes=0,
        allowed_ssd_roots=(tmp_path / "ssd",),
    )
    pairs = np.arange(pair_count * 2 * 2 * 3 * 3, dtype=np.uint8).reshape(pair_count, 2, 2, 3, 3)
    return preflight, (pairs[:2].copy(), pairs[2:].copy())


def test_preflight_exposes_batch_geometry_and_never_claims_contest_authority(tmp_path: Path) -> None:
    diagnostic, _ = _inputs(tmp_path / "diagnostic", batch_size=2)
    matched, _ = _inputs(
        tmp_path / "matched",
        batch_size=UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
    )

    assert diagnostic["batch_geometry_matches_upstream_default"] is False
    assert diagnostic["batch_geometry_authority"] == "NONAUTHORITATIVE_DIAGNOSTIC_GEOMETRY"
    assert matched["scorer_pair_batch_size"] == 16
    assert matched["upstream_evaluate_default_pair_batch_size"] == 16
    assert matched["batch_geometry_matches_upstream_default"] is True
    assert matched["batch_geometry_authority"] == "UPSTREAM_DEFAULT_MATCH_MACOS_CPU_ADVISORY"
    assert matched["contest_axis_authority"] is False
    assert matched["segnet_frame_selector"] == "last_frame_index_1_of_nonoverlapping_pair"


class _Preparer:
    def __init__(self, *, hash_salt: bytes = b"", invalid_labels: bool = False) -> None:
        self.hash_salt = hash_salt
        self.invalid_labels = invalid_labels
        self.inferred: list[tuple[int, ...]] = []

    def __call__(self, batch: np.ndarray) -> PreparedTeacherBatchV1:
        hashes = tuple(hashlib.sha256(self.hash_salt + memoryview(pair).cast("B")).hexdigest() for pair in batch)

        def infer_missing(indices: tuple[int, ...]):
            self.inferred.append(indices)
            result = {}
            for local_index in indices:
                value = 5 if self.invalid_labels else int(batch[local_index, 0, 0, 0, 0]) % 5
                result[local_index] = np.full((2, 2), value, dtype=np.uint8)
            return result

        return PreparedTeacherBatchV1(hashes, infer_missing)


def _materialize(tmp_path: Path):
    preflight, batches = _inputs(tmp_path)
    preparer = _Preparer()
    receipt = materialize_fresh_teacher_from_batches(
        preflight=preflight,
        source_batches=batches,
        prepare_batch=preparer,
        allowed_ssd_roots=(tmp_path / "ssd",),
    )
    return preflight, batches, preparer, receipt


def test_materializes_gap_free_path_backed_encoder_only_population(tmp_path: Path) -> None:
    preflight, _batches, preparer, receipt = _materialize(tmp_path)

    assert preparer.inferred == [(0, 1), (0,)]
    assert receipt["pair_count"] == 3
    assert receipt["chronological_pair_order"] == [0, 1, 2]
    assert receipt["full_public_population_proven"] is False
    assert receipt["encoder_only"] is True
    assert receipt["candidate_payload_allowed"] is False
    assert receipt["target_labels_serialized_in_candidate"] is False
    assert receipt["scorer_weights_serialized_in_candidate"] is False
    assert receipt["batch_geometry_matches_upstream_default"] is False
    assert receipt["contest_axis_authority"] is False
    assert receipt["next_consumer_contract"]["semantic_compile_geometry_ready"] is False
    aggregate = Path(receipt["target_labels"]["path"])
    assert aggregate.stat().st_size == 3 * 2 * 2
    values = np.fromfile(aggregate, dtype=np.uint8).reshape(3, 2, 2)
    assert np.all(values[0] == 0)
    assert np.all(values[1] == 1)
    assert np.all(values[2] == 2)
    assert receipt["preflight_sha256"] == preflight["preflight_sha256"]


def test_resume_reopens_every_shard_and_runs_no_redundant_forward(tmp_path: Path) -> None:
    preflight, batches, _first, first_receipt = _materialize(tmp_path)
    resumed = _Preparer()

    second_receipt = materialize_fresh_teacher_from_batches(
        preflight=preflight,
        source_batches=batches,
        prepare_batch=resumed,
        allowed_ssd_roots=(tmp_path / "ssd",),
    )

    assert resumed.inferred == []
    assert second_receipt == first_receipt


def test_resume_refuses_source_pair_or_scorer_input_drift(tmp_path: Path) -> None:
    preflight, batches, _preparer, _receipt = _materialize(tmp_path)
    changed_batches = [batch.copy() for batch in batches]
    changed_batches[0][0, 0, 0, 0, 0] ^= np.uint8(1)
    with pytest.raises(FreshTeacherMaterializationError, match="source_pair_rgb_sha256 drift"):
        materialize_fresh_teacher_from_batches(
            preflight=preflight,
            source_batches=changed_batches,
            prepare_batch=_Preparer(),
            allowed_ssd_roots=(tmp_path / "ssd",),
        )

    with pytest.raises(FreshTeacherMaterializationError, match="scorer_input_sha256 drift"):
        materialize_fresh_teacher_from_batches(
            preflight=preflight,
            source_batches=batches,
            prepare_batch=_Preparer(hash_salt=b"drift"),
            allowed_ssd_roots=(tmp_path / "ssd",),
        )


def test_resume_refuses_tampered_or_orphaned_pair_shard(tmp_path: Path) -> None:
    preflight, batches, _preparer, receipt = _materialize(tmp_path)
    shard = Path(receipt["pair_checkpoints"][1]["target_shard_path"])
    shard.write_bytes(b"\x04" * 4)
    with pytest.raises(FreshTeacherMaterializationError, match="identity drifted"):
        materialize_fresh_teacher_from_batches(
            preflight=preflight,
            source_batches=batches,
            prepare_batch=_Preparer(),
            allowed_ssd_roots=(tmp_path / "ssd",),
        )

    shard.unlink()
    with pytest.raises(FreshTeacherMaterializationError, match="orphaned shard/checkpoint"):
        materialize_fresh_teacher_from_batches(
            preflight=preflight,
            source_batches=batches,
            prepare_batch=_Preparer(),
            allowed_ssd_roots=(tmp_path / "ssd",),
        )


def test_refuses_short_extra_or_invalid_label_populations(tmp_path: Path) -> None:
    short_preflight, short_batches = _inputs(tmp_path / "short")
    with pytest.raises(FreshTeacherMaterializationError, match="observed 2, expected exactly 3"):
        materialize_fresh_teacher_from_batches(
            preflight=short_preflight,
            source_batches=(short_batches[0],),
            prepare_batch=_Preparer(),
            allowed_ssd_roots=(tmp_path / "short/ssd",),
        )


def test_refuses_source_batches_that_change_declared_scorer_geometry(tmp_path: Path) -> None:
    preflight, batches = _inputs(tmp_path, pair_count=3, batch_size=3)
    with pytest.raises(FreshTeacherMaterializationError, match="batch cardinality changed scorer geometry"):
        materialize_fresh_teacher_from_batches(
            preflight=preflight,
            source_batches=batches,
            prepare_batch=_Preparer(),
            allowed_ssd_roots=(tmp_path / "ssd",),
        )

    extra_preflight, extra_batches = _inputs(tmp_path / "extra", pair_count=2)
    extra_population = (*extra_batches, np.zeros((1, 2, 2, 3, 3), dtype=np.uint8))
    with pytest.raises(FreshTeacherMaterializationError, match="more than the exact requested"):
        materialize_fresh_teacher_from_batches(
            preflight=extra_preflight,
            source_batches=extra_population,
            prepare_batch=_Preparer(),
            allowed_ssd_roots=(tmp_path / "extra/ssd",),
        )

    invalid_preflight, invalid_batches = _inputs(tmp_path / "invalid")
    with pytest.raises(FreshTeacherMaterializationError, match=r"class outside 0\.\.4"):
        materialize_fresh_teacher_from_batches(
            preflight=invalid_preflight,
            source_batches=invalid_batches,
            prepare_batch=_Preparer(invalid_labels=True),
            allowed_ssd_roots=(tmp_path / "invalid/ssd",),
        )


def test_preflight_and_completed_receipt_reopen_exact_input_and_output_bytes(tmp_path: Path) -> None:
    preflight, _batches, _preparer, receipt = _materialize(tmp_path)
    reverify_preflight(preflight, allowed_ssd_roots=(tmp_path / "ssd",))
    receipt_path = Path(preflight["output_root"]) / "12_encoder_only_receipt.json"
    assert (
        load_and_reverify_materialization_receipt(
            receipt_path,
            allowed_ssd_roots=(tmp_path / "ssd",),
        )
        == receipt
    )

    Path(preflight["source_video"]["path"]).write_bytes(b"source-drift")
    with pytest.raises(FreshTeacherMaterializationError, match="source_video identity drifted"):
        reverify_preflight(preflight, allowed_ssd_roots=(tmp_path / "ssd",))


def test_compile_ready_loader_refuses_diagnostic_batch_geometry(tmp_path: Path) -> None:
    preflight, _batches, _preparer, _receipt = _materialize(tmp_path)
    with pytest.raises(FreshTeacherMaterializationError, match="not compile-ready"):
        load_compile_ready_materialization_receipt(
            Path(preflight["output_root"]) / "12_encoder_only_receipt.json",
            allowed_ssd_roots=(tmp_path / "ssd",),
        )


def test_completed_receipt_refuses_aggregate_drift(tmp_path: Path) -> None:
    preflight, _batches, _preparer, receipt = _materialize(tmp_path)
    aggregate = Path(receipt["target_labels"]["path"])
    data = bytearray(aggregate.read_bytes())
    data[0] ^= 1
    aggregate.write_bytes(data)
    with pytest.raises(FreshTeacherMaterializationError, match="aggregate target labels drifted"):
        load_and_reverify_materialization_receipt(
            Path(preflight["output_root"]) / "12_encoder_only_receipt.json",
            allowed_ssd_roots=(tmp_path / "ssd",),
        )


def test_completed_receipt_refuses_stage_zero_custody_drift(tmp_path: Path) -> None:
    preflight, _batches, _preparer, _receipt = _materialize(tmp_path)
    stage_zero = Path(preflight["output_root"]) / "00_custody_storage_preflight.json"
    payload = bytearray(stage_zero.read_bytes())
    payload[-2] ^= 1
    stage_zero.write_bytes(payload)
    with pytest.raises(FreshTeacherMaterializationError, match=r"cannot read JSON mapping|does not match"):
        load_and_reverify_materialization_receipt(
            Path(preflight["output_root"]) / "12_encoder_only_receipt.json",
            allowed_ssd_roots=(tmp_path / "ssd",),
        )


def test_pair_scorer_hash_is_content_bound_not_attested(tmp_path: Path) -> None:
    _preflight, batches, _preparer, receipt = _materialize(tmp_path)
    expected = hashlib.sha256(memoryview(batches[0][0]).cast("B")).hexdigest()
    assert receipt["pair_checkpoints"][0]["scorer_input_sha256"] == expected
    assert _sha(np.fromfile(receipt["target_labels"]["path"], dtype=np.uint8)) == receipt["target_labels"]["sha256"]
