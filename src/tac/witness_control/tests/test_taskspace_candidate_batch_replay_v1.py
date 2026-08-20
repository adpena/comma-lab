# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.taskspace_candidate_batch_replay_v1 import (
    BatchDistancesV1,
    CandidateBatchReplayError,
    build_candidate_replay_preflight,
    load_and_reverify_candidate_replay_receipt,
    replay_candidate_batches,
)


def _files(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    source = root / "source.mkv"
    archive = root / "candidate" / "archive.zip"
    raw = root / "candidate" / "inflated" / "0.raw"
    upstream = root / "upstream"
    implementation = root / "runner.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    raw.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source-bytes")
    archive.write_bytes(b"archive-bytes")
    raw.write_bytes(bytes(range(108)))  # 3 pairs * 2 frames * 2 * 3 * RGB
    implementation.write_bytes(b"# replay implementation\n")
    for relative in (
        "evaluate.py",
        "frame_utils.py",
        "modules.py",
        "public_test_video_names.txt",
        "models/segnet.safetensors",
        "models/posenet.safetensors",
    ):
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    return source, archive, raw, upstream, implementation


def _preflight(tmp_path: Path) -> dict:
    source, archive, raw, upstream, implementation = _files(tmp_path)
    return build_candidate_replay_preflight(
        source_video=source,
        candidate_archive=archive,
        candidate_raw=raw,
        upstream_root=upstream,
        output_root=tmp_path / "ssd" / "run",
        pair_count=3,
        batch_size=2,
        num_threads=1,
        seed=1234,
        package_versions={"numpy": np.__version__},
        implementation_files=(implementation,),
        run_argv=["python", "replay.py"],
        camera_hw=(2, 3),
        safety_reserve_bytes=1,
        allowed_ssd_roots=(tmp_path / "ssd",),
    )


def _batches() -> list[tuple[np.ndarray, np.ndarray]]:
    source0 = np.zeros((2, 2, 2, 3, 3), dtype=np.uint8)
    source0[1] = 1
    candidate0 = source0.copy()
    candidate0[1] = 3
    source1 = np.full((1, 2, 2, 3, 3), 2, dtype=np.uint8)
    candidate1 = np.full_like(source1, 4)
    return [(source0, candidate0), (source1, candidate1)]


def test_full_replay_is_stage_resumable_and_float32_aggregated(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    calls: list[int] = []

    def score(source: np.ndarray, candidate: np.ndarray) -> BatchDistancesV1:
        calls.append(int(source.shape[0]))
        delta = (candidate.astype(np.float32) - source.astype(np.float32)).reshape(source.shape[0], -1)
        pose = np.mean(delta * delta, axis=1, dtype=np.float32)
        seg = np.mean(delta != 0, axis=1, dtype=np.float32)
        return BatchDistancesV1(
            d_pose=pose,
            d_seg=seg,
            d_pose_batch_sum_f32=float(np.sum(pose, dtype=np.float32)),
            d_seg_batch_sum_f32=float(np.sum(seg, dtype=np.float32)),
        )

    receipt = replay_candidate_batches(
        preflight=preflight,
        paired_batches=_batches(),
        score_batch=score,
        allowed_ssd_roots=(tmp_path / "ssd",),
    )
    assert calls == [2, 1]
    assert receipt["pair_count"] == 3
    assert receipt["batch_stage_count"] == 2
    assert receipt["distortion"]["d_pose"] == pytest.approx(8 / 3)
    assert receipt["distortion"]["d_seg"] == pytest.approx(2 / 3)
    assert receipt["score_claim"] is False
    assert receipt["evidence_axis"] == "[macOS-CPU deterministic batch16 scorer advisory]"
    assert receipt["authority"]["exact_numeric_upstream_host_mirror"] is False
    assert receipt["authority"]["implementation_closure_hash_bound"] is True

    calls.clear()
    resumed = replay_candidate_batches(
        preflight=preflight,
        paired_batches=_batches(),
        score_batch=score,
        allowed_ssd_roots=(tmp_path / "ssd",),
    )
    assert calls == []
    assert resumed == receipt
    reopened = load_and_reverify_candidate_replay_receipt(
        tmp_path / "ssd" / "run" / "11_batch_replay_receipt.json",
        allowed_ssd_roots=(tmp_path / "ssd",),
    )
    assert reopened == receipt


def test_replay_refuses_changed_batch_geometry(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    bad = _batches()
    bad[0] = (bad[0][0][:1], bad[0][1][:1])
    with pytest.raises(CandidateBatchReplayError, match="exact scorer geometry"):
        replay_candidate_batches(
            preflight=preflight,
            paired_batches=bad,
            score_batch=lambda source, candidate: BatchDistancesV1(
                d_pose=np.zeros(source.shape[0]),
                d_seg=np.zeros(source.shape[0]),
                d_pose_batch_sum_f32=0.0,
                d_seg_batch_sum_f32=0.0,
            ),
            allowed_ssd_roots=(tmp_path / "ssd",),
        )


def test_preflight_refuses_raw_size_not_matching_pair_geometry(tmp_path: Path) -> None:
    source, archive, raw, upstream, implementation = _files(tmp_path)
    raw.write_bytes(b"short")
    with pytest.raises(CandidateBatchReplayError, match="candidate raw has"):
        build_candidate_replay_preflight(
            source_video=source,
            candidate_archive=archive,
            candidate_raw=raw,
            upstream_root=upstream,
            output_root=tmp_path / "ssd" / "run",
            pair_count=3,
            batch_size=2,
            num_threads=1,
            seed=1234,
            package_versions={"numpy": np.__version__},
            implementation_files=(implementation,),
            run_argv=["python", "replay.py"],
            camera_hw=(2, 3),
            safety_reserve_bytes=1,
            allowed_ssd_roots=(tmp_path / "ssd",),
        )


def test_preflight_refuses_changed_implementation(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    implementation = Path(preflight["implementation_closure"]["members"][0]["path"])
    implementation.write_bytes(b"# drifted replay implementation\n")
    with pytest.raises(CandidateBatchReplayError, match="implementation closure drifted"):
        replay_candidate_batches(
            preflight=preflight,
            paired_batches=_batches(),
            score_batch=lambda source, candidate: BatchDistancesV1(
                d_pose=np.zeros(source.shape[0]),
                d_seg=np.zeros(source.shape[0]),
                d_pose_batch_sum_f32=0.0,
                d_seg_batch_sum_f32=0.0,
            ),
            allowed_ssd_roots=(tmp_path / "ssd",),
        )
