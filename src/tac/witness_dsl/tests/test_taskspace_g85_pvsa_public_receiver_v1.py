# SPDX-License-Identifier: MIT
"""Behavioral tests for the bounded G85 raw writer and public ABI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl import taskspace_g85_pvsa_public_receiver_v1 as g85


class _DeterministicReceiver:
    def __init__(self, geometry: g85._RawGeometry) -> None:
        self.geometry = geometry
        self.calls: list[tuple[int, ...]] = []

    def render_camera_pair_batch(self, local_pair_ids: tuple[int, ...]) -> np.ndarray:
        self.calls.append(local_pair_ids)
        result = np.empty(
            (
                len(local_pair_ids),
                self.geometry.frames_per_pair,
                self.geometry.height,
                self.geometry.width,
                self.geometry.channels,
            ),
            dtype=np.uint8,
        )
        for local_index, pair_id in enumerate(local_pair_ids):
            for frame_id in range(self.geometry.frames_per_pair):
                result[local_index, frame_id].fill((pair_id * 17 + frame_id * 3) % 256)
        return result


def test_exact_public_geometry_is_derived() -> None:
    assert (
        g85.PAIR_COUNT * g85.FRAMES_PER_PAIR * g85.CAMERA_HEIGHT * g85.CAMERA_WIDTH * g85.CHANNELS
        == g85.EXPECTED_RAW_BYTES
        == 3_662_409_600
    )
    assert g85.DEFAULT_BATCH_PAIRS == g85.MAX_STREAM_BATCH_PAIRS == 16
    assert "PUBLIC_ENTRYPOINT_DOUBLE_DECODE_OWED" in g85.PUBLIC_AUTHORITY_BLOCKERS
    assert "CONTEST_CPU_OR_CUDA_AUTHORITY_EVAL_OWED" in g85.PUBLIC_AUTHORITY_BLOCKERS
    assert "PVSA_FULL_N600_FRESH_DOUBLE_DECODE_OWED" not in g85.PUBLIC_AUTHORITY_BLOCKERS


def test_internal_writer_is_chronological_bounded_and_exact(tmp_path: Path) -> None:
    geometry = g85._RawGeometry(
        pair_count=5,
        frames_per_pair=2,
        height=3,
        width=4,
        channels=3,
    )
    receiver = _DeterministicReceiver(geometry)
    member = b"exact-counted-member"
    output = tmp_path / "0.raw"
    output_sha, resumed, batch_count, _elapsed = g85._stream_receiver_to_raw(
        receiver=receiver,
        member=member,
        output_path=output,
        batch_pairs=2,
        geometry=geometry,
    )
    expected = np.concatenate(
        [
            np.full((3, 4, 3), (pair_id * 17 + frame_id * 3) % 256, dtype=np.uint8)
            for pair_id in range(5)
            for frame_id in range(2)
        ],
        axis=0,
    )
    assert output.read_bytes() == expected.tobytes(order="C")
    assert output_sha == hashlib.sha256(output.read_bytes()).hexdigest()
    assert resumed == 0
    assert batch_count == 3
    assert receiver.calls == [(0, 1), (2, 3), (4,)]
    partial, checkpoint = g85._checkpoint_paths(output)
    assert not partial.exists()
    assert not checkpoint.exists()


def test_resume_rehashes_prefix_and_continues_without_rewriting(tmp_path: Path) -> None:
    geometry = g85._RawGeometry(
        pair_count=4,
        frames_per_pair=2,
        height=2,
        width=3,
        channels=1,
    )
    receiver = _DeterministicReceiver(geometry)
    member = b"member"
    output = tmp_path / "resume.raw"
    partial, checkpoint_path = g85._checkpoint_paths(output)
    first = receiver.render_camera_pair_batch((0, 1)).tobytes(order="C")
    partial.write_bytes(first)
    checkpoint = g85.PVSAWriteCheckpointV1(
        member_bytes=len(member),
        member_sha256=hashlib.sha256(member).hexdigest(),
        output_name=output.name,
        batch_pairs=2,
        pair_count=geometry.pair_count,
        pair_bytes=geometry.pair_bytes,
        completed_pairs=2,
        partial_bytes=len(first),
        partial_sha256=hashlib.sha256(first).hexdigest(),
    )
    checkpoint_path.write_bytes(checkpoint.to_bytes())
    receiver.calls.clear()

    _sha, resumed, batches, _elapsed = g85._stream_receiver_to_raw(
        receiver=receiver,
        member=member,
        output_path=output,
        batch_pairs=2,
        geometry=geometry,
    )
    assert resumed == 2
    assert batches == 2
    assert receiver.calls == [(2, 3)]
    assert output.read_bytes().startswith(first)


def test_resume_rolls_back_one_uncertified_kill_window_batch(tmp_path: Path) -> None:
    geometry = g85._RawGeometry(
        pair_count=4,
        frames_per_pair=2,
        height=2,
        width=3,
        channels=1,
    )
    receiver = _DeterministicReceiver(geometry)
    member = b"member"
    output = tmp_path / "kill-window.raw"
    partial, checkpoint_path = g85._checkpoint_paths(output)
    certified = receiver.render_camera_pair_batch((0, 1)).tobytes(order="C")
    uncertified = b"\xff" * (2 * geometry.pair_bytes)
    partial.write_bytes(certified + uncertified)
    checkpoint_path.write_bytes(
        g85.PVSAWriteCheckpointV1(
            member_bytes=len(member),
            member_sha256=hashlib.sha256(member).hexdigest(),
            output_name=output.name,
            batch_pairs=2,
            pair_count=geometry.pair_count,
            pair_bytes=geometry.pair_bytes,
            completed_pairs=2,
            partial_bytes=len(certified),
            partial_sha256=hashlib.sha256(certified).hexdigest(),
        ).to_bytes()
    )
    receiver.calls.clear()

    _sha, resumed, _batches, _elapsed = g85._stream_receiver_to_raw(
        receiver=receiver,
        member=member,
        output_path=output,
        batch_pairs=2,
        geometry=geometry,
    )
    assert resumed == 2
    assert receiver.calls == [(2, 3)]
    assert output.read_bytes().startswith(certified)
    assert b"\xff" * geometry.pair_bytes not in output.read_bytes()


def test_resume_rejects_corrupt_prefix_and_member_drift(tmp_path: Path) -> None:
    geometry = g85._RawGeometry(
        pair_count=2,
        frames_per_pair=2,
        height=2,
        width=2,
        channels=1,
    )
    receiver = _DeterministicReceiver(geometry)
    output = tmp_path / "bad.raw"
    partial, checkpoint_path = g85._checkpoint_paths(output)
    member = b"member"
    partial.write_bytes(b"\x00" * geometry.pair_bytes)
    checkpoint = g85.PVSAWriteCheckpointV1(
        member_bytes=len(member),
        member_sha256=hashlib.sha256(member).hexdigest(),
        output_name=output.name,
        batch_pairs=1,
        pair_count=geometry.pair_count,
        pair_bytes=geometry.pair_bytes,
        completed_pairs=1,
        partial_bytes=geometry.pair_bytes,
        partial_sha256=hashlib.sha256(b"\x01" * geometry.pair_bytes).hexdigest(),
    )
    checkpoint_path.write_bytes(checkpoint.to_bytes())
    with pytest.raises(g85.G85PublicReceiverError, match="prefix hash"):
        g85._stream_receiver_to_raw(
            receiver=receiver,
            member=member,
            output_path=output,
            batch_pairs=1,
            geometry=geometry,
        )
    checkpoint_path.write_bytes(
        g85.PVSAWriteCheckpointV1(
            member_bytes=len(member),
            member_sha256=hashlib.sha256(member).hexdigest(),
            output_name=output.name,
            batch_pairs=1,
            pair_count=geometry.pair_count,
            pair_bytes=geometry.pair_bytes,
            completed_pairs=1,
            partial_bytes=geometry.pair_bytes,
            partial_sha256=hashlib.sha256(partial.read_bytes()).hexdigest(),
        ).to_bytes()
    )
    with pytest.raises(g85.G85PublicReceiverError, match="different member"):
        g85._stream_receiver_to_raw(
            receiver=receiver,
            member=b"changed",
            output_path=output,
            batch_pairs=1,
            geometry=geometry,
        )


def test_resume_rejects_more_than_one_uncertified_batch(tmp_path: Path) -> None:
    geometry = g85._RawGeometry(
        pair_count=4,
        frames_per_pair=2,
        height=2,
        width=2,
        channels=1,
    )
    member = b"member"
    output = tmp_path / "too-long.raw"
    partial, checkpoint_path = g85._checkpoint_paths(output)
    partial.write_bytes(b"\x00" * (geometry.pair_bytes * 3 + 1))
    checkpoint_path.write_bytes(
        g85.PVSAWriteCheckpointV1(
            member_bytes=len(member),
            member_sha256=hashlib.sha256(member).hexdigest(),
            output_name=output.name,
            batch_pairs=1,
            pair_count=geometry.pair_count,
            pair_bytes=geometry.pair_bytes,
            completed_pairs=0,
            partial_bytes=0,
            partial_sha256=hashlib.sha256(b"").hexdigest(),
        ).to_bytes()
    )
    with pytest.raises(g85.G85PublicReceiverError, match="more than one"):
        g85._stream_receiver_to_raw(
            receiver=_DeterministicReceiver(geometry),
            member=member,
            output_path=output,
            batch_pairs=1,
            geometry=geometry,
        )


def test_checkpoint_and_receipt_truth_fail_closed() -> None:
    checkpoint = g85.PVSAWriteCheckpointV1(
        member_bytes=1,
        member_sha256="a" * 64,
        output_name="0.raw",
        batch_pairs=16,
        pair_count=600,
        pair_bytes=g85._CONTEST_GEOMETRY.pair_bytes,
        completed_pairs=0,
        partial_bytes=0,
        partial_sha256=hashlib.sha256(b"").hexdigest(),
    )
    assert g85.PVSAWriteCheckpointV1.from_bytes(checkpoint.to_bytes()) == checkpoint
    changed = json.loads(checkpoint.to_bytes())
    changed["completed_pairs"] = 1
    with pytest.raises(g85.G85PublicReceiverError):
        g85.PVSAWriteCheckpointV1.from_bytes(g85._canonical_json(changed))

    receipt = g85.PVSAWriteReceiptV1(
        member_bytes=133_363,
        member_sha256="b" * 64,
        output_name="0.raw",
        output_bytes=g85.EXPECTED_RAW_BYTES,
        output_sha256="c" * 64,
        pair_count=600,
        frame_count=1200,
        camera_height=874,
        camera_width=1164,
        channels=3,
        batch_pairs=16,
        batch_count=38,
        resumed_from_pairs=0,
        elapsed_seconds=1.0,
        addressed_operand_pair_count=1,
    )
    assert receipt.repository_runtime_dependency is True
    assert receipt.self_contained_public_runtime is False
    assert receipt.addressed_operand_pair_count == 1
    assert receipt.double_decode_proven is False
    assert receipt.upstream_evaluator_invoked is False
    assert receipt.score_claim is False
    assert receipt.research_only is True


def test_video_name_and_staging_runtime_contract() -> None:
    root = Path(__file__).resolve().parents[4]
    staging = root / "submissions/robust_current/taskspace_pvsa_staging"
    shell = (staging / "inflate.sh").read_text(encoding="utf-8")
    launcher = (staging / "inflate.py").read_text(encoding="utf-8")
    readme = (staging / "README.md").read_text(encoding="utf-8")
    assert '"$1" "$2" "$3"' in shell
    assert "PYTHONDONTWRITEBYTECODE=1" in shell
    assert "PYTHONPATH" in shell
    assert "taskspace_g85_pvsa_public_receiver_v1" in launcher
    assert "repository-bound and research-only" in readme
    assert "pydantic" in readme


def test_video_names_requires_one_safe_n600_name(tmp_path: Path) -> None:
    names = tmp_path / "names.txt"
    names.write_text("0.mkv\n", encoding="utf-8")
    assert g85._video_output_name(names) == "0.raw"
    names.write_text("0.mkv\n1.mkv\n", encoding="utf-8")
    with pytest.raises(g85.G85PublicReceiverError, match="exactly one"):
        g85._video_output_name(names)
    names.write_text("../0.mkv\n", encoding="utf-8")
    with pytest.raises(g85.G85PublicReceiverError, match="safe"):
        g85._video_output_name(names)
