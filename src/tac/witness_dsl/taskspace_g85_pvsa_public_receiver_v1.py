# SPDX-License-Identifier: MIT
"""Bounded, resumable raw-output bridge for the compact PVSA1 receiver.

This module closes one deliberately narrow part of the public receiver: given
an already extracted, counted ``PVSA1`` member, it writes the exact contest raw
ABI in chronological ``Y0,Y1`` order without retaining the full video in
memory.  The actual pixels are produced by the committed G80 receiver; this
module does not reinterpret or approximate them.

The current implementation is repository-bound.  Importing the G80 receiver
recursively imports the research direct-description implementation, including
``pydantic``, which is not present in the frozen upstream evaluator lock.  The
staged ``inflate.sh`` therefore fails closed unless the repository runtime is
available and must not be called a self-contained public submission runtime.
The missing next landing is a tree-shaken decoder-only implementation with no
non-upstream dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    V15RoleAwareOverlayError,
    parse_role_aware_boundary_shearlet_operand,
)
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    MAX_STREAM_BATCH_PAIRS,
    CompactActuatorTypeV1,
    CompactPVSAError,
    CompactPVSAReceiverV1,
    parse_compact_pvsa_member,
)

PAIR_COUNT: Final = 600
FRAMES_PER_PAIR: Final = 2
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
CHANNELS: Final = 3
EXPECTED_RAW_BYTES: Final = PAIR_COUNT * FRAMES_PER_PAIR * CAMERA_HEIGHT * CAMERA_WIDTH * CHANNELS
DEFAULT_BATCH_PAIRS: Final = 16
MEMBER_NAME: Final = "0.bin"
CHECKPOINT_SCHEMA: Final = "tac.g85_pvsa_raw_write_checkpoint.v1"
RECEIPT_SCHEMA: Final = "tac.g85_pvsa_raw_write_receipt.v1"
RUNTIME_CONTRACT_ID: Final = "tac.pvsa1.bounded_chronological_raw_receiver.v1"
SELF_CONTAINED_RUNTIME_BLOCKER: Final = "PVSA_DECODER_ONLY_TREE_SHAKE_WITHOUT_PYDANTIC_OR_REPOSITORY_IMPORTS_OWED"
PUBLIC_AUTHORITY_BLOCKERS: Final = (
    SELF_CONTAINED_RUNTIME_BLOCKER,
    "PVSA_PUBLIC_RUNTIME_RECURSIVE_DEPENDENCY_CLOSURE_OWED",
    "PUBLIC_ENTRYPOINT_DOUBLE_DECODE_OWED",
    "CONTEST_CPU_OR_CUDA_AUTHORITY_EVAL_OWED",
)

_SAFE_VIDEO_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class G85PublicReceiverError(RuntimeError):
    """The counted member, receiver transition, or raw custody failed."""


class _BatchReceiver(Protocol):
    def render_camera_pair_batch(self, local_pair_ids: tuple[int, ...]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class _SparseG74BatchReceiver:
    """Render semantic P once outside the exact counted-operand support.

    G74's generic proof receiver renders both immutable P and ephemeral P+A for
    every requested pair, even when no counted atom addresses that pair.  The
    output transition is identity outside the operand's addressed pair set.
    This adapter preserves the generic path on addressed pairs and dispatches
    directly to immutable P everywhere else.  It changes scheduling only, not
    the wire, receiver state, or output bytes.
    """

    receiver: CompactPVSAReceiverV1
    addressed_local_pairs: frozenset[int]

    @classmethod
    def open(cls, receiver: CompactPVSAReceiverV1) -> _SparseG74BatchReceiver:
        if type(receiver) is not CompactPVSAReceiverV1:
            raise G85PublicReceiverError("sparse dispatch requires the exact compact PVSA receiver")
        receiver._validate_custody()
        parsed = receiver.parsed
        if not parsed.actuators:
            addressed = frozenset()
        else:
            if (
                len(parsed.actuators) != 1
                or parsed.actuators[0].actuator_type is not CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT
            ):
                raise G85PublicReceiverError("sparse dispatch only admits the closed G74 transition")
            actuator = parsed.actuators[0]
            try:
                operand = parse_role_aware_boundary_shearlet_operand(
                    actuator.payload,
                    expected_sha256=actuator.sha256,
                    maximum_operand_bytes=len(actuator.payload),
                )
            except V15RoleAwareOverlayError as exc:
                raise G85PublicReceiverError("sparse dispatch refused the counted G74 operand") from exc
            source_start = receiver.overlay_decoder.receiver.predictor.source_pair_start
            pair_count = receiver.overlay_decoder.receiver.z.n_pairs
            addressed = frozenset(atom.pair_index - source_start for atom in operand.atoms)
            if not addressed or any(value < 0 or value >= pair_count for value in addressed):
                raise G85PublicReceiverError("sparse dispatch operand addresses escaped semantic P")
        return cls(receiver=receiver, addressed_local_pairs=addressed)

    def render_camera_pair_batch(self, local_pair_ids: tuple[int, ...]) -> np.ndarray:
        self.receiver._validate_custody()
        if (
            type(local_pair_ids) is not tuple
            or not 1 <= len(local_pair_ids) <= MAX_STREAM_BATCH_PAIRS
            or any(type(value) is not int or not 0 <= value < PAIR_COUNT for value in local_pair_ids)
            or local_pair_ids != tuple(range(local_pair_ids[0], local_pair_ids[0] + len(local_pair_ids)))
        ):
            raise G85PublicReceiverError("sparse stream batch must be 1..16 contiguous exact n600 pair IDs")
        try:
            output = self.receiver.overlay_decoder.receiver.render_camera_pairs(local_pair_ids)
            for local_index, pair_id in enumerate(local_pair_ids):
                if pair_id in self.addressed_local_pairs:
                    output[local_index] = self.receiver.render_camera_pair_batch((pair_id,))[0]
        except (CompactPVSAError, ValueError) as exc:
            raise G85PublicReceiverError("sparse exact-support dispatch failed") from exc
        expected_shape = (
            len(local_pair_ids),
            FRAMES_PER_PAIR,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        )
        if output.dtype != np.uint8 or output.shape != expected_shape:
            raise G85PublicReceiverError("sparse dispatch changed the exact camera ABI")
        result = np.ascontiguousarray(output)
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class _RawGeometry:
    pair_count: int
    frames_per_pair: int
    height: int
    width: int
    channels: int

    def __post_init__(self) -> None:
        if (
            type(self.pair_count) is not int
            or type(self.frames_per_pair) is not int
            or type(self.height) is not int
            or type(self.width) is not int
            or type(self.channels) is not int
            or min(
                self.pair_count,
                self.frames_per_pair,
                self.height,
                self.width,
                self.channels,
            )
            < 1
        ):
            raise G85PublicReceiverError("raw geometry must contain exact positive integers")

    @property
    def pair_bytes(self) -> int:
        return self.frames_per_pair * self.height * self.width * self.channels

    @property
    def raw_bytes(self) -> int:
        return self.pair_count * self.pair_bytes


_CONTEST_GEOMETRY = _RawGeometry(
    pair_count=PAIR_COUNT,
    frames_per_pair=FRAMES_PER_PAIR,
    height=CAMERA_HEIGHT,
    width=CAMERA_WIDTH,
    channels=CHANNELS,
)
if _CONTEST_GEOMETRY.raw_bytes != EXPECTED_RAW_BYTES or EXPECTED_RAW_BYTES != 3_662_409_600:
    raise RuntimeError("G85 contest raw geometry drifted")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_bytes(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class PVSAWriteCheckpointV1:
    member_bytes: int
    member_sha256: str
    output_name: str
    batch_pairs: int
    pair_count: int
    pair_bytes: int
    completed_pairs: int
    partial_bytes: int
    partial_sha256: str
    schema: str = CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != CHECKPOINT_SCHEMA
            or type(self.member_bytes) is not int
            or self.member_bytes < 1
            or not re.fullmatch(r"[0-9a-f]{64}", self.member_sha256)
            or type(self.output_name) is not str
            or not self.output_name
            or type(self.batch_pairs) is not int
            or not 1 <= self.batch_pairs <= MAX_STREAM_BATCH_PAIRS
            or type(self.pair_count) is not int
            or self.pair_count < 1
            or type(self.pair_bytes) is not int
            or self.pair_bytes < 1
            or type(self.completed_pairs) is not int
            or not 0 <= self.completed_pairs <= self.pair_count
            or type(self.partial_bytes) is not int
            or self.partial_bytes != self.completed_pairs * self.pair_bytes
            or not re.fullmatch(r"[0-9a-f]{64}", self.partial_sha256)
        ):
            raise G85PublicReceiverError("G85 checkpoint fields or geometry drifted")

    def to_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @classmethod
    def from_bytes(cls, payload: bytes) -> PVSAWriteCheckpointV1:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise G85PublicReceiverError("G85 checkpoint is not JSON") from exc
        if not isinstance(value, dict) or set(value) != {
            "schema",
            "member_bytes",
            "member_sha256",
            "output_name",
            "batch_pairs",
            "pair_count",
            "pair_bytes",
            "completed_pairs",
            "partial_bytes",
            "partial_sha256",
        }:
            raise G85PublicReceiverError("G85 checkpoint fields differ")
        result = cls(**value)
        if result.to_bytes() != payload:
            raise G85PublicReceiverError("G85 checkpoint is not canonical JSON")
        return result


@dataclass(frozen=True, slots=True)
class PVSAWriteReceiptV1:
    member_bytes: int
    member_sha256: str
    output_name: str
    output_bytes: int
    output_sha256: str
    pair_count: int
    frame_count: int
    camera_height: int
    camera_width: int
    channels: int
    batch_pairs: int
    batch_count: int
    resumed_from_pairs: int
    elapsed_seconds: float
    addressed_operand_pair_count: int
    runtime_contract_id: str = RUNTIME_CONTRACT_ID
    chronological_y0_then_y1: bool = True
    output_dtype: str = "uint8"
    maximum_live_batch_pairs: int = MAX_STREAM_BATCH_PAIRS
    repository_runtime_dependency: bool = True
    self_contained_public_runtime: bool = False
    full_n600_raw_written: bool = True
    double_decode_proven: bool = False
    upstream_evaluator_invoked: bool = False
    score_claim: bool = False
    candidate_claim: bool = False
    research_only: bool = True
    schema: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != RECEIPT_SCHEMA
            or self.runtime_contract_id != RUNTIME_CONTRACT_ID
            or type(self.member_bytes) is not int
            or self.member_bytes < 1
            or not re.fullmatch(r"[0-9a-f]{64}", self.member_sha256)
            or self.output_bytes != EXPECTED_RAW_BYTES
            or not re.fullmatch(r"[0-9a-f]{64}", self.output_sha256)
            or self.pair_count != PAIR_COUNT
            or self.frame_count != PAIR_COUNT * FRAMES_PER_PAIR
            or (self.camera_height, self.camera_width, self.channels) != (CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
            or not 1 <= self.batch_pairs <= MAX_STREAM_BATCH_PAIRS
            or self.batch_count != (PAIR_COUNT + self.batch_pairs - 1) // self.batch_pairs
            or not 0 <= self.resumed_from_pairs <= PAIR_COUNT
            or type(self.elapsed_seconds) is not float
            or self.elapsed_seconds < 0.0
            or type(self.addressed_operand_pair_count) is not int
            or not 0 <= self.addressed_operand_pair_count <= PAIR_COUNT
            or self.chronological_y0_then_y1 is not True
            or self.output_dtype != "uint8"
            or self.maximum_live_batch_pairs != MAX_STREAM_BATCH_PAIRS
            or self.repository_runtime_dependency is not True
            or self.self_contained_public_runtime is not False
            or self.full_n600_raw_written is not True
            or self.double_decode_proven is not False
            or self.upstream_evaluator_invoked is not False
            or self.score_claim is not False
            or self.candidate_claim is not False
            or self.research_only is not True
        ):
            raise G85PublicReceiverError("G85 raw receipt truth or exact ABI drifted")

    def to_bytes(self) -> bytes:
        return _canonical_json(asdict(self))


def _checkpoint_paths(output_path: Path) -> tuple[Path, Path]:
    return (
        output_path.with_name(f".{output_path.name}.partial"),
        output_path.with_name(f".{output_path.name}.checkpoint.json"),
    )


def _resume_state(
    *,
    member: bytes,
    output_path: Path,
    batch_pairs: int,
    geometry: _RawGeometry,
) -> tuple[Path, Path, int, hashlib._Hash]:
    partial_path, checkpoint_path = _checkpoint_paths(output_path)
    digest = hashlib.sha256()
    if not partial_path.exists() and not checkpoint_path.exists():
        return partial_path, checkpoint_path, 0, digest
    if (
        partial_path.is_symlink()
        or checkpoint_path.is_symlink()
        or not partial_path.is_file()
        or not checkpoint_path.is_file()
    ):
        raise G85PublicReceiverError("G85 resume requires both ordinary partial and checkpoint files")
    checkpoint = PVSAWriteCheckpointV1.from_bytes(checkpoint_path.read_bytes())
    expected = {
        "member_bytes": len(member),
        "member_sha256": _sha256_bytes(member),
        "output_name": output_path.name,
        "batch_pairs": batch_pairs,
        "pair_count": geometry.pair_count,
        "pair_bytes": geometry.pair_bytes,
    }
    if any(getattr(checkpoint, key) != value for key, value in expected.items()):
        raise G85PublicReceiverError("G85 checkpoint belongs to a different member/output/config")
    if checkpoint.completed_pairs > geometry.pair_count:
        raise G85PublicReceiverError("G85 checkpoint exceeds requested geometry")
    observed_partial_bytes = partial_path.stat().st_size
    if observed_partial_bytes < checkpoint.partial_bytes:
        raise G85PublicReceiverError("G85 partial is shorter than its certified checkpoint")
    uncheckpointed_bytes = observed_partial_bytes - checkpoint.partial_bytes
    if uncheckpointed_bytes > batch_pairs * geometry.pair_bytes:
        raise G85PublicReceiverError("G85 partial has more than one uncheckpointed batch")
    with partial_path.open("rb") as handle:
        remaining = checkpoint.partial_bytes
        while remaining:
            chunk = handle.read(min(8 << 20, remaining))
            if not chunk:
                raise G85PublicReceiverError("G85 certified partial prefix is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    if digest.hexdigest() != checkpoint.partial_sha256:
        raise G85PublicReceiverError("G85 partial prefix hash differs from checkpoint")
    if uncheckpointed_bytes:
        # The checkpoint is the authority. A kill after the batch fsync but
        # before atomic checkpoint publication can leave at most one
        # uncertified batch tail. Roll that scratch tail back deterministically
        # so the certified batch can be regenerated byte-for-byte.
        with partial_path.open("r+b") as handle:
            handle.truncate(checkpoint.partial_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return partial_path, checkpoint_path, checkpoint.completed_pairs, digest


def _stream_receiver_to_raw(
    *,
    receiver: _BatchReceiver,
    member: bytes,
    output_path: Path,
    batch_pairs: int,
    geometry: _RawGeometry,
) -> tuple[str, int, int, float]:
    """Internal resumable writer; the public wrapper always uses n600 geometry."""

    if type(batch_pairs) is not int or not 1 <= batch_pairs <= MAX_STREAM_BATCH_PAIRS:
        raise G85PublicReceiverError("batch_pairs must be an exact integer in [1,16]")
    if output_path.is_symlink() or output_path.exists():
        raise G85PublicReceiverError("G85 refuses to overwrite an existing final raw")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path, checkpoint_path, completed, digest = _resume_state(
        member=member,
        output_path=output_path,
        batch_pairs=batch_pairs,
        geometry=geometry,
    )
    resumed_from = completed
    started = time.monotonic()
    mode = "ab" if completed else "xb"
    with partial_path.open(mode) as handle:
        while completed < geometry.pair_count:
            stop = min(completed + batch_pairs, geometry.pair_count)
            ids = tuple(range(completed, stop))
            batch = receiver.render_camera_pair_batch(ids)
            expected_shape = (
                len(ids),
                geometry.frames_per_pair,
                geometry.height,
                geometry.width,
                geometry.channels,
            )
            if batch.dtype != np.uint8 or batch.shape != expected_shape or not batch.flags.c_contiguous:
                raise G85PublicReceiverError("PVSA receiver batch changed uint8 chronological camera ABI")
            payload = memoryview(batch).cast("B")
            written = handle.write(payload)
            if written != len(payload):
                raise G85PublicReceiverError("short write while materializing PVSA raw")
            digest.update(payload)
            completed = stop
            handle.flush()
            os.fsync(handle.fileno())
            checkpoint = PVSAWriteCheckpointV1(
                member_bytes=len(member),
                member_sha256=_sha256_bytes(member),
                output_name=output_path.name,
                batch_pairs=batch_pairs,
                pair_count=geometry.pair_count,
                pair_bytes=geometry.pair_bytes,
                completed_pairs=completed,
                partial_bytes=completed * geometry.pair_bytes,
                partial_sha256=digest.hexdigest(),
            )
            _atomic_write(checkpoint_path, checkpoint.to_bytes())
    if partial_path.stat().st_size != geometry.raw_bytes:
        raise G85PublicReceiverError("completed PVSA raw has the wrong exact byte count")
    os.replace(partial_path, output_path)
    checkpoint_path.unlink()
    return (
        digest.hexdigest(),
        resumed_from,
        (geometry.pair_count + batch_pairs - 1) // batch_pairs,
        (time.monotonic() - started),
    )


def write_pvsa_member_to_raw(
    *,
    member_path: Path,
    output_path: Path,
    batch_pairs: int = DEFAULT_BATCH_PAIRS,
) -> PVSAWriteReceiptV1:
    """Strictly parse one counted PVSA1 member and write all 1,200 raw frames."""

    if member_path.is_symlink() or not member_path.is_file():
        raise G85PublicReceiverError("PVSA member must be an ordinary file")
    member = member_path.read_bytes()
    try:
        parsed = parse_compact_pvsa_member(
            member,
            maximum_member_bytes=len(member),
            maximum_section_bytes=len(member),
        )
        receiver = _SparseG74BatchReceiver.open(
            parsed.open_receiver(verify_member_effects=True),
        )
    except (CompactPVSAError, G85PublicReceiverError) as exc:
        raise G85PublicReceiverError("strict PVSA member/receiver open failed") from exc
    output_sha, resumed_from, batch_count, elapsed = _stream_receiver_to_raw(
        receiver=receiver,
        member=member,
        output_path=output_path,
        batch_pairs=batch_pairs,
        geometry=_CONTEST_GEOMETRY,
    )
    if output_path.stat().st_size != EXPECTED_RAW_BYTES or _sha256_file(output_path) != output_sha:
        raise G85PublicReceiverError("final PVSA raw parse-back hash/size differs")
    return PVSAWriteReceiptV1(
        member_bytes=len(member),
        member_sha256=_sha256_bytes(member),
        output_name=output_path.name,
        output_bytes=output_path.stat().st_size,
        output_sha256=output_sha,
        pair_count=PAIR_COUNT,
        frame_count=PAIR_COUNT * FRAMES_PER_PAIR,
        camera_height=CAMERA_HEIGHT,
        camera_width=CAMERA_WIDTH,
        channels=CHANNELS,
        batch_pairs=batch_pairs,
        batch_count=batch_count,
        resumed_from_pairs=resumed_from,
        elapsed_seconds=float(elapsed),
        addressed_operand_pair_count=len(receiver.addressed_local_pairs),
    )


def _video_output_name(video_names_path: Path) -> str:
    if video_names_path.is_symlink() or not video_names_path.is_file():
        raise G85PublicReceiverError("video names input must be an ordinary file")
    names = [line.strip() for line in video_names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 1 or not _SAFE_VIDEO_NAME.fullmatch(names[0]) or "/" in names[0] or "\\" in names[0]:
        raise G85PublicReceiverError("PVSA V1 requires exactly one safe n600 video name")
    stem = Path(names[0]).stem
    if not stem:
        raise G85PublicReceiverError("video name has no output stem")
    return f"{stem}.raw"


def inflate_extracted_pvsa(
    archive_dir: Path,
    output_dir: Path,
    video_names_path: Path,
) -> PVSAWriteReceiptV1:
    """Implement the contest three-positional-argument inflate contract."""

    if archive_dir.is_symlink() or not archive_dir.is_dir():
        raise G85PublicReceiverError("archive directory must be an ordinary directory")
    member_path = archive_dir / MEMBER_NAME
    output_name = _video_output_name(video_names_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return write_pvsa_member_to_raw(
        member_path=member_path,
        output_path=output_dir / output_name,
        batch_pairs=DEFAULT_BATCH_PAIRS,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Repository-bound G85 PVSA1 staging receiver")
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("video_names_file", type=Path)
    args = parser.parse_args(argv)
    receipt = inflate_extracted_pvsa(
        args.archive_dir,
        args.output_dir,
        args.video_names_file,
    )
    print(receipt.to_bytes().decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
