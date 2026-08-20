# SPDX-License-Identifier: MIT
"""Bounded, source-custodied adapters for the original ep725 LVLS1 predictor.

The legacy adapter reopens the exact source object for historical custody.  The
causal adapter instead accepts immutable counted ``0.bin`` bytes and immutable
runtime bytes explicitly: both the repository NumPy reference and the shipped
runtime execute those arguments, and no source archive path is read.  Both
derive semantics from the member's own ``phi.argmax`` and emit no score,
target, candidate, or promotion claim.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol

import numpy as np

from tac.witness_dsl.taskspace_predictor_state_v2 import (
    NONE_TRANSPORT_CONTENT_SHA256,
    NoTransportV2,
    TaskspacePredictorStateV2,
)

EP725_SOURCE_DIRECTORY = Path("/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet")
EP725_ARCHIVE_SHA256 = "149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3"
EP725_MEMBER_SHA256 = "f0c3e648f00f52e48c7be98997fb7dd57c2e5a607ed385846931af68f88cc78c"
EP725_RUNTIME_SHA256 = "4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224"
EP725_MANIFEST_SHA256 = "1b8ff55fd7c21395fe7d596558406e2608421522bb0eb3eb11d1e3b0cf047088"
EP725_ARCHIVE_BYTES = 83_838
EP725_MEMBER_BYTES = 84_536
EP725_MANIFEST_BYTES = 1_906
EP725_RUNTIME_BYTES = 56_814
EP725_N_PAIRS = 600
EP725_RENDER_HEIGHT = 384
EP725_RENDER_WIDTH = 512
EP725_CAMERA_HEIGHT = 874
EP725_CAMERA_WIDTH = 1164
EP725_N_CLASSES = 5
MAX_BOUNDED_DECODE_PAIRS = 2
EP725_BOUNDED_RECEIPT_SCHEMA = "tac.ep725_bounded_predictor_decode_receipt.v2"
EP725_CAUSAL_RECEIPT_SCHEMA = "tac.ep725_counted_member_causal_decode_receipt.v3"
EP725_CANONICAL_RENDERER_SHA256 = "1cecaa3ee9873d1378eaeb66b04cee56b621bccabe3fcecc01f60d3a0e692ddc"
_DECODER_LABELS_ORIGIN = "same_counted_LVLS1_member_decoder_phi_argmax"
_FROZEN_BOUNDED_OUTPUTS: dict[tuple[int, ...], dict[str, object]] = {
    (0,): {
        "chronological_camera_frame_sha256": (
            "fdd3d4d7a7d8c26f48c6dc89f248eda381971b08ea7cb4e9f0c71289a07e2e01",
            "38dfcaa5d3ef49467eca94d788f480feb6583f36ac58f6f7b47a06f2adddf758",
        ),
        "chronological_raw_prefix_sha256": "22b994567d3db018df29a95c597606053a115c631d98e89b72ec7eeba93666b3",
        "labels_sha256": "635b1bce0b16f3c2d612c965855938141a836e80a6db734c2429642bd41bfd28",
        "predictor_semantic_binding_sha256": "f9d63a1990601e4d2eb7235047dca0b64a5058967618e173a170ffc82c0a73bc",
        "predictor_state_binding_sha256": "b095f5e47c37a5bcf3ff1f574413cd29d66269e8508b8f9f8314e23e7a2ff672",
        "receipt_sha256": "07b2060ccc1d4ca4ffdb3a3d2e38bf789cdd49eab0fdcba4e0288c6f8e692493",
    },
    (0, 1): {
        "chronological_camera_frame_sha256": (
            "fdd3d4d7a7d8c26f48c6dc89f248eda381971b08ea7cb4e9f0c71289a07e2e01",
            "38dfcaa5d3ef49467eca94d788f480feb6583f36ac58f6f7b47a06f2adddf758",
            "b4d9a6ec279780f10c7a05d76188c882d0b6004ef21cb544f0d757cc2290e1c5",
            "06cb6d8ef75fb230018fdbed20f55202e44dd2a0adeeacfc815e0d283814504e",
        ),
        "chronological_raw_prefix_sha256": "23f12caeb9f43149d94732289a7cbac34ef08c3171c6613750074e3c8e76b9e9",
        "labels_sha256": "9165ef7fc86fe997e01b26ecf751e877b8ae30352e7b3d3db659d920a908068e",
        "predictor_semantic_binding_sha256": "4ec471a707a37d38f704586584f84fd5d0adb4ec2a70a8069481dc6e6113197d",
        "predictor_state_binding_sha256": "a25ee50104fc887ba5ee5e92110caac736ee431338584a3ad6ac8b54ab7cfae9",
        "receipt_sha256": "eb765a3e3252155857861dc11564a98d46b4147190ef634d710af69c1ebe7445",
    },
}
_LVLS1_MAGIC = b"LVLS1\x00"
_ZIP_EOCD = struct.Struct("<4s4H2LH")
_ZIP_EOCD_MAGIC = b"PK\x05\x06"
_ZIP_LOCAL_MAGIC = b"PK\x03\x04"
_ZIP_CENTRAL_MAGIC = b"PK\x01\x02"


class Ep725LevelsetPredictorAdapterError(ValueError):
    """Source custody, grammar, or state construction failed closed."""


class Ep725PredictorDecodeBlocker(Ep725LevelsetPredictorAdapterError):
    """The structural source is valid but executable parity cannot be proved."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Ep725LevelsetPredictorAdapterError(f"{label} must be lowercase SHA-256 hex")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Ep725LevelsetPredictorAdapterError("receipt is not canonical JSON") from exc


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_descriptor_payload(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_source_leaf(directory_descriptor: int, name: str, *, label: str) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None or no_follow == 0:
        raise Ep725LevelsetPredictorAdapterError("O_NOFOLLOW is required for source custody")
    try:
        named_before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise Ep725LevelsetPredictorAdapterError(f"{label} cannot be statted") from exc
    if stat.S_ISLNK(named_before.st_mode) or not stat.S_ISREG(named_before.st_mode):
        raise Ep725LevelsetPredictorAdapterError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise Ep725LevelsetPredictorAdapterError(f"{label} cannot be descriptor-bound") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (named_before.st_dev, named_before.st_ino):
            raise Ep725LevelsetPredictorAdapterError(f"{label} changed before descriptor binding")
        payload = _read_descriptor_payload(descriptor)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise Ep725LevelsetPredictorAdapterError(f"{label} disappeared after read") from exc
    if (
        not stat.S_ISREG(named_after.st_mode)
        or _stable_identity(opened) != _stable_identity(after_read)
        or _stable_identity(after_read) != _stable_identity(named_after)
        or len(payload) != after_read.st_size
    ):
        raise Ep725LevelsetPredictorAdapterError(f"{label} mutated during descriptor-bound read")
    return payload


def _read_source_directory(source_directory: Path) -> tuple[bytes, bytes]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None or no_follow == 0:
        raise Ep725LevelsetPredictorAdapterError("O_NOFOLLOW is required for source custody")
    try:
        named_before = os.lstat(source_directory)
    except OSError as exc:
        raise Ep725LevelsetPredictorAdapterError("source directory does not exist") from exc
    if stat.S_ISLNK(named_before.st_mode) or not stat.S_ISDIR(named_before.st_mode):
        raise Ep725LevelsetPredictorAdapterError("source directory must be a regular non-symlink directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | no_follow
    try:
        directory_descriptor = os.open(source_directory, flags)
    except OSError as exc:
        raise Ep725LevelsetPredictorAdapterError("source directory cannot be descriptor-bound") from exc
    try:
        opened = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (named_before.st_dev, named_before.st_ino):
            raise Ep725LevelsetPredictorAdapterError("source directory changed before descriptor binding")
        archive = _read_source_leaf(directory_descriptor, "archive.zip", label="source archive")
        runtime = _read_source_leaf(directory_descriptor, "inflate.py", label="source runtime")
        after_read = os.fstat(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    try:
        named_after = os.lstat(source_directory)
    except OSError as exc:
        raise Ep725LevelsetPredictorAdapterError("source directory disappeared after read") from exc
    if (
        not stat.S_ISDIR(named_after.st_mode)
        or _stable_identity(opened) != _stable_identity(after_read)
        or _stable_identity(after_read) != _stable_identity(named_after)
    ):
        raise Ep725LevelsetPredictorAdapterError("source directory mutated during descriptor-bound read")
    return archive, runtime


def _read_stable_regular_path(path: Path, *, label: str) -> bytes:
    parent = path.parent
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None or no_follow == 0:
        raise Ep725LevelsetPredictorAdapterError("O_NOFOLLOW is required for source custody")
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | no_follow
    try:
        directory_descriptor = os.open(parent, directory_flags)
    except OSError as exc:
        raise Ep725LevelsetPredictorAdapterError(f"{label} parent cannot be descriptor-bound") from exc
    try:
        return _read_source_leaf(directory_descriptor, path.name, label=label)
    finally:
        os.close(directory_descriptor)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Ep725LevelsetPredictorAdapterError(f"LVLS1 manifest repeats key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class _ParsedLvls1V2:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    sections: tuple[bytes, ...]


def _take_block(payload: bytes, offset: int, *, label: str) -> tuple[bytes, int]:
    if offset + 4 > len(payload):
        raise Ep725LevelsetPredictorAdapterError(f"LVLS1 {label} length is truncated")
    (length,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    stop = offset + length
    if stop > len(payload):
        raise Ep725LevelsetPredictorAdapterError(f"LVLS1 {label} bytes are truncated")
    return payload[offset:stop], stop


def _parse_lvls1_exact(member: bytes) -> _ParsedLvls1V2:
    if type(member) is not bytes or not member.startswith(_LVLS1_MAGIC):
        raise Ep725LevelsetPredictorAdapterError("source member is not LVLS1")
    offset = len(_LVLS1_MAGIC)
    sections: list[bytes] = []
    for label in ("manifest", "base", "code", "pose"):
        block, offset = _take_block(member, offset, label=label)
        sections.append(block)
    try:
        manifest = json.loads(
            sections[0].decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Ep725LevelsetPredictorAdapterError(f"non-finite JSON constant {value}")
            ),
        )
    except Ep725LevelsetPredictorAdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Ep725LevelsetPredictorAdapterError("LVLS1 manifest is not strict UTF-8 JSON") from exc
    if type(manifest) is not dict:
        raise Ep725LevelsetPredictorAdapterError("LVLS1 manifest must be an object")
    for key in ("lane_render_band", "pose_carrier", "chart_payload", "palette_residual"):
        if manifest.get(key) is not None:
            block, offset = _take_block(member, offset, label=key)
            sections.append(block)
    if offset != len(member):
        raise Ep725LevelsetPredictorAdapterError("LVLS1 member has unconsumed trailing/unknown blocks")
    return _ParsedLvls1V2(
        manifest=manifest,
        manifest_bytes=sections[0],
        sections=tuple(sections),
    )


def _require_exact_zip_eof(archive: bytes) -> None:
    search_start = max(0, len(archive) - (65_535 + _ZIP_EOCD.size))
    eocd_offset = archive.rfind(_ZIP_EOCD_MAGIC, search_start)
    if eocd_offset < 0 or eocd_offset + _ZIP_EOCD.size > len(archive):
        raise Ep725LevelsetPredictorAdapterError("source ZIP has no exact EOCD")
    (
        magic,
        disk_number,
        central_disk,
        entries_on_disk,
        entries_total,
        central_size,
        central_offset,
        comment_length,
    ) = _ZIP_EOCD.unpack_from(archive, eocd_offset)
    if magic != _ZIP_EOCD_MAGIC or any((disk_number, central_disk)):
        raise Ep725LevelsetPredictorAdapterError("source ZIP is split or malformed")
    if entries_on_disk != 1 or entries_total != 1:
        raise Ep725LevelsetPredictorAdapterError("source ZIP must contain exactly one member")
    if eocd_offset + _ZIP_EOCD.size + comment_length != len(archive):
        raise Ep725LevelsetPredictorAdapterError("source ZIP has a comment or trailing bytes")
    if central_offset + central_size != eocd_offset:
        raise Ep725LevelsetPredictorAdapterError("source ZIP central directory is not exactly consumed")
    if archive[:4] != _ZIP_LOCAL_MAGIC or archive[central_offset : central_offset + 4] != _ZIP_CENTRAL_MAGIC:
        raise Ep725LevelsetPredictorAdapterError("source ZIP has prepended data or invalid directory offsets")


def _strict_single_member(archive: bytes) -> bytes:
    _require_exact_zip_eof(archive)
    try:
        with zipfile.ZipFile(io.BytesIO(archive), mode="r") as handle:
            if handle.comment:
                raise Ep725LevelsetPredictorAdapterError("source ZIP comment is forbidden")
            infos = handle.infolist()
            if len(infos) != 1:
                raise Ep725LevelsetPredictorAdapterError("source ZIP must contain exactly one member")
            info = infos[0]
            if (
                info.filename != "0.bin"
                or info.is_dir()
                or info.header_offset != 0
                or info.flag_bits != 0
                or info.compress_type != zipfile.ZIP_DEFLATED
                or info.extra
                or info.comment
                or info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.create_system != 3
                or (info.external_attr >> 16) != 0o644
            ):
                raise Ep725LevelsetPredictorAdapterError(
                    "source ZIP member metadata differs from the canonical one-member form"
                )
            member = handle.read(info)
            if len(member) != info.file_size:
                raise Ep725LevelsetPredictorAdapterError("source ZIP member length differs after CRC decode")
    except Ep725LevelsetPredictorAdapterError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise Ep725LevelsetPredictorAdapterError("source ZIP cannot be decoded exactly") from exc
    if zipfile.is_zipfile(io.BytesIO(member)):
        raise Ep725LevelsetPredictorAdapterError("nested ZIP predictor members are forbidden")
    return member


@dataclass(frozen=True, slots=True)
class Ep725SourceExpectationsV2:
    archive_sha256: str
    archive_bytes: int
    member_sha256: str
    member_bytes: int
    runtime_sha256: str
    manifest_sha256: str
    manifest_bytes: int
    n_pairs: int
    render_height: int
    render_width: int
    camera_height: int
    camera_width: int
    n_classes: int


EP725_FROZEN_EXPECTATIONS = Ep725SourceExpectationsV2(
    archive_sha256=EP725_ARCHIVE_SHA256,
    archive_bytes=EP725_ARCHIVE_BYTES,
    member_sha256=EP725_MEMBER_SHA256,
    member_bytes=EP725_MEMBER_BYTES,
    runtime_sha256=EP725_RUNTIME_SHA256,
    manifest_sha256=EP725_MANIFEST_SHA256,
    manifest_bytes=EP725_MANIFEST_BYTES,
    n_pairs=EP725_N_PAIRS,
    render_height=EP725_RENDER_HEIGHT,
    render_width=EP725_RENDER_WIDTH,
    camera_height=EP725_CAMERA_HEIGHT,
    camera_width=EP725_CAMERA_WIDTH,
    n_classes=EP725_N_CLASSES,
)


@dataclass(frozen=True, slots=True)
class Ep725SourceObjectV2:
    source_directory: Path
    archive: bytes
    member: bytes
    runtime: bytes
    manifest_bytes: bytes
    manifest: dict[str, Any]

    @property
    def archive_sha256(self) -> str:
        return _sha256(self.archive)

    @property
    def member_sha256(self) -> str:
        return _sha256(self.member)

    @property
    def runtime_sha256(self) -> str:
        return _sha256(self.runtime)

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.manifest_bytes)


class _Ep725ExecutableSource(Protocol):
    """The only bytes either decoder implementation is allowed to consume."""

    member: bytes
    runtime: bytes


def _validate_member_against_expectations(
    member: bytes,
    *,
    expectations: Ep725SourceExpectationsV2,
) -> _ParsedLvls1V2:
    if type(member) is not bytes:
        raise Ep725LevelsetPredictorAdapterError("counted predictor member must be exact immutable bytes")
    if zipfile.is_zipfile(io.BytesIO(member)):
        raise Ep725LevelsetPredictorAdapterError("nested ZIP predictor members are forbidden")
    parsed = _parse_lvls1_exact(member)
    exact_member = {
        "member_sha256": _sha256(member),
        "member_bytes": len(member),
        "manifest_sha256": _sha256(parsed.manifest_bytes),
        "manifest_bytes": len(parsed.manifest_bytes),
    }
    for field_name, observed in exact_member.items():
        if observed != getattr(expectations, field_name):
            raise Ep725LevelsetPredictorAdapterError(f"source {field_name} differs from frozen custody")
    manifest_expected = {
        "n_pairs": expectations.n_pairs,
        "render_h": expectations.render_height,
        "render_w": expectations.render_width,
        "camera_h": expectations.camera_height,
        "camera_w": expectations.camera_width,
        "n_classes": expectations.n_classes,
        "has_pose_sidecar": False,
    }
    for key, expected in manifest_expected.items():
        if parsed.manifest.get(key) != expected:
            raise Ep725LevelsetPredictorAdapterError(f"LVLS1 manifest field {key!r} differs from custody")
    if any(
        parsed.manifest.get(key) is not None
        for key in ("lane_render_band", "pose_carrier", "chart_payload", "palette_residual")
    ):
        raise Ep725LevelsetPredictorAdapterError("ep725 source unexpectedly carries an optional receiver block")
    return parsed


@dataclass(frozen=True, slots=True)
class Ep725ExplicitMemberRuntimeSourceV3:
    """Exact causal decode inputs, with no archive or source-directory handle."""

    member: bytes
    runtime: bytes

    def __post_init__(self) -> None:
        _validate_member_against_expectations(
            self.member,
            expectations=EP725_FROZEN_EXPECTATIONS,
        )
        if type(self.runtime) is not bytes:
            raise Ep725LevelsetPredictorAdapterError("predictor runtime must be exact immutable bytes")
        if len(self.runtime) != EP725_RUNTIME_BYTES or _sha256(self.runtime) != EP725_RUNTIME_SHA256:
            raise Ep725LevelsetPredictorAdapterError("explicit predictor runtime differs from frozen custody")

    @property
    def member_sha256(self) -> str:
        return _sha256(self.member)

    @property
    def runtime_sha256(self) -> str:
        return _sha256(self.runtime)


def inspect_ep725_source(
    source_directory: str | Path = EP725_SOURCE_DIRECTORY,
    *,
    expectations: Ep725SourceExpectationsV2 = EP725_FROZEN_EXPECTATIONS,
) -> Ep725SourceObjectV2:
    """Reopen and strictly parse a source object without executing a decoder."""

    source_path = Path(source_directory)
    archive, runtime = _read_source_directory(source_path)
    exact_observed = {
        "archive_sha256": _sha256(archive),
        "archive_bytes": len(archive),
        "runtime_sha256": _sha256(runtime),
    }
    for field_name, observed in exact_observed.items():
        if observed != getattr(expectations, field_name):
            raise Ep725LevelsetPredictorAdapterError(f"source {field_name} differs from frozen custody")
    member = _strict_single_member(archive)
    parsed = _validate_member_against_expectations(member, expectations=expectations)
    return Ep725SourceObjectV2(
        source_directory=source_path,
        archive=archive,
        member=member,
        runtime=runtime,
        manifest_bytes=parsed.manifest_bytes,
        manifest=parsed.manifest,
    )


def _load_canonical_decoder_module() -> tuple[ModuleType, str]:
    repository = Path(__file__).resolve().parents[3]
    source_path = repository / "tools" / "levelset_byte_close_and_eval.py"
    try:
        before = _read_stable_regular_path(source_path, label="canonical NumPy renderer source")
        code = compile(before, str(source_path), "exec", dont_inherit=True)
        module = ModuleType("_tac_ep725_descriptor_bound_levelset_decoder")
        module.__file__ = str(source_path)
        module.__package__ = "tools"
        exec(code, module.__dict__)
        after = _read_stable_regular_path(source_path, label="canonical NumPy renderer source")
    except Ep725PredictorDecodeBlocker:
        raise
    except Exception as exc:
        raise Ep725PredictorDecodeBlocker(
            "CANONICAL_RENDERER_UNAVAILABLE", f"cannot import bound NumPy renderer: {exc}"
        ) from exc
    if before != after:
        raise Ep725PredictorDecodeBlocker("CANONICAL_RENDERER_MUTATED", "renderer source changed while importing")
    required = (
        "_read_blob_bytes_full",
        "_dequant_blob",
        "numpy_oracle_reference_frames",
    )
    if any(not callable(getattr(module, name, None)) for name in required):
        raise Ep725PredictorDecodeBlocker(
            "CANONICAL_RENDERER_API_MISSING", f"required callables are absent: {required}"
        )
    return module, _sha256(before)


def _canonical_decode_once(
    module: ModuleType,
    source: _Ep725ExecutableSource,
    pair_count: int,
) -> tuple[bytes, np.ndarray]:
    try:
        tool_sections = module._read_blob_bytes_full(source.member)
        structural = _parse_lvls1_exact(source.member)
        tool_manifest = tool_sections[0]
        tool_payloads = tuple(block for block in tool_sections[1:] if block is not None)
        if tool_manifest != structural.manifest or tool_payloads != structural.sections[1:]:
            raise Ep725PredictorDecodeBlocker(
                "LVLS1_PARSER_PARITY_MISMATCH",
                "canonical parser disagrees with strict source parser",
            )
        manifest, params, code, lane_pairs, pose_carrier, chart_payload = module._dequant_blob(source.member)
        if lane_pairs is not None or pose_carrier is not None or chart_payload is not None:
            raise Ep725PredictorDecodeBlocker(
                "EP725_OPTIONAL_STATE_UNEXPECTED",
                "frozen ep725 unexpectedly decoded an optional receiver block",
            )
        frames, argmaxes = module.numpy_oracle_reference_frames(
            params,
            code,
            manifest,
            pair_count,
        )
    except Ep725PredictorDecodeBlocker:
        raise
    except Exception as exc:
        raise Ep725PredictorDecodeBlocker(
            "CANONICAL_NUMPY_DECODE_FAILED", f"reference decode refused exact member: {exc}"
        ) from exc
    expected_frames = pair_count * 2
    if len(frames) != expected_frames or len(argmaxes) != pair_count:
        raise Ep725PredictorDecodeBlocker(
            "CANONICAL_NUMPY_CARDINALITY_MISMATCH",
            f"got {len(frames)} frames and {len(argmaxes)} labels for {pair_count} pairs",
        )
    frame_bytes: list[bytes] = []
    for frame in frames:
        value = np.asarray(frame)
        if value.dtype != np.uint8 or value.shape != (EP725_CAMERA_HEIGHT, EP725_CAMERA_WIDTH, 3):
            raise Ep725PredictorDecodeBlocker(
                "CANONICAL_CAMERA_ABI_MISMATCH", f"unexpected frame ABI {value.dtype} {value.shape}"
            )
        frame_bytes.append(np.ascontiguousarray(value).tobytes())
    labels = np.stack(argmaxes, axis=0)
    if labels.shape != (pair_count, EP725_RENDER_HEIGHT, EP725_RENDER_WIDTH):
        raise Ep725PredictorDecodeBlocker("CANONICAL_LABEL_ABI_MISMATCH", f"unexpected phi.argmax shape {labels.shape}")
    if labels.dtype.kind not in "iu" or int(labels.min()) < 0 or int(labels.max()) >= EP725_N_CLASSES:
        raise Ep725PredictorDecodeBlocker(
            "CANONICAL_LABEL_UNIVERSE_MISMATCH", "phi.argmax escaped the five-class integer universe"
        )
    return b"".join(frame_bytes), np.ascontiguousarray(labels, dtype=np.uint8)


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            offset += os.write(descriptor, view[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _shipped_decode_once(
    source: _Ep725ExecutableSource,
    pair_count: int,
    *,
    timeout_seconds: float,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="pact_ep725_n2_") as temporary:
        root = Path(temporary)
        runtime_path = root / "inflate.py"
        member_path = root / "0.bin"
        raw_path = root / "prefix.raw"
        _write_exclusive(runtime_path, source.runtime)
        _write_exclusive(member_path, source.member)
        if _read_stable_regular_path(runtime_path, label="copied shipped runtime") != source.runtime:
            raise Ep725PredictorDecodeBlocker(
                "SHIPPED_RUNTIME_COPY_MISMATCH", "temporary runtime differs before launch"
            )
        if _read_stable_regular_path(member_path, label="copied counted member") != source.member:
            raise Ep725PredictorDecodeBlocker("SHIPPED_MEMBER_COPY_MISMATCH", "temporary member differs before launch")
        environment = os.environ.copy()
        environment.update(
            {
                "INFLATE_MAX_PAIRS": str(pair_count),
                "INFLATE_WORKERS": "1",
                "INFLATE_FP32": "0",
                "VECLIB_MAXIMUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        try:
            process = subprocess.run(
                [sys.executable, str(runtime_path), str(member_path), str(raw_path)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise Ep725PredictorDecodeBlocker(
                "SHIPPED_RUNTIME_TIMEOUT",
                f"exact shipped runtime exceeded {timeout_seconds:g}s on {pair_count} pairs",
            ) from exc
        if process.returncode != 0:
            raise Ep725PredictorDecodeBlocker(
                "SHIPPED_RUNTIME_FAILED",
                f"rc={process.returncode}; stderr={process.stderr[-1000:]}",
            )
        if _read_stable_regular_path(runtime_path, label="copied shipped runtime") != source.runtime:
            raise Ep725PredictorDecodeBlocker("SHIPPED_RUNTIME_COPY_MUTATED", "temporary runtime changed during launch")
        if _read_stable_regular_path(member_path, label="copied counted member") != source.member:
            raise Ep725PredictorDecodeBlocker("SHIPPED_MEMBER_COPY_MUTATED", "temporary member changed during launch")
        try:
            raw = _read_stable_regular_path(raw_path, label="shipped raw output")
        except Ep725LevelsetPredictorAdapterError as exc:
            raise Ep725PredictorDecodeBlocker(
                "SHIPPED_RUNTIME_OUTPUT_MISSING", "runtime did not publish a stable regular prefix.raw"
            ) from exc
        expected_bytes = pair_count * 2 * EP725_CAMERA_HEIGHT * EP725_CAMERA_WIDTH * 3
        if len(raw) != expected_bytes:
            raise Ep725PredictorDecodeBlocker(
                "SHIPPED_RUNTIME_OUTPUT_LENGTH",
                f"raw bytes {len(raw)} != expected {expected_bytes}",
            )
        return raw


def _expected_semantic_binding(receipt: Ep725BoundedDecodeReceiptV2) -> str:
    return _sha256(
        _canonical_json(
            {
                "schema": "tac.taskspace_predictor_semantic_binding.v2",
                "predictor_program_sha256": receipt.source_member_sha256,
                "predictor_renderer_sha256": receipt.canonical_renderer_sha256,
                "source_archive_sha256": receipt.source_archive_sha256,
                "source_runtime_sha256": receipt.source_runtime_sha256,
                "source_member_name": "0.bin",
                "source_pair_ids": list(receipt.source_pair_ids),
                "labels_sha256": receipt.labels_sha256,
            }
        )
    )


def _expected_state_binding(semantic_binding_sha256: str) -> str:
    return _sha256(
        _canonical_json(
            {
                "schema": "tac.taskspace_predictor_binding.v2",
                "semantic_binding_sha256": semantic_binding_sha256,
                "transport": {
                    "kind": "NONE",
                    "counted_bytes": 0,
                    "counted_content_sha256": NONE_TRANSPORT_CONTENT_SHA256,
                    "decoded_values_sha256": NONE_TRANSPORT_CONTENT_SHA256,
                },
            }
        )
    )


@dataclass(frozen=True, slots=True)
class Ep725BoundedDecodeReceiptV2:
    """Canonical machine-verifiable hash receipt for a bounded exact decode."""

    schema: str
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    source_archive_sha256: str
    source_archive_bytes: int
    source_member_sha256: str
    counted_predictor_bytes: int
    source_runtime_sha256: str
    canonical_renderer_sha256: str
    source_pair_ids: tuple[int, ...]
    chronological_camera_frame_sha256: tuple[str, ...]
    chronological_raw_prefix_sha256: str
    chronological_raw_prefix_bytes: int
    labels_sha256: str
    decoder_labels_origin: str
    transport_kind: str
    deterministic_double_decode: bool
    shipped_numpy_raw_bit_exact: bool
    source_zip_wrapper_is_counted_p: bool
    member_is_counted_p: bool
    dense_frames_persisted: bool
    scorer_invoked: bool
    score_claim: bool
    candidate_claim: bool
    promotion_eligible: bool
    research_only: bool

    def __post_init__(self) -> None:
        if self.schema != EP725_BOUNDED_RECEIPT_SCHEMA:
            raise Ep725LevelsetPredictorAdapterError("bounded receipt schema is not the closed V2 schema")
        for field_name in (
            "predictor_state_binding_sha256",
            "predictor_semantic_binding_sha256",
            "source_archive_sha256",
            "source_member_sha256",
            "source_runtime_sha256",
            "canonical_renderer_sha256",
            "chronological_raw_prefix_sha256",
            "labels_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        if (
            type(self.source_pair_ids) is not tuple
            or any(type(value) is not int for value in self.source_pair_ids)
            or self.source_pair_ids not in {(0,), (0, 1)}
        ):
            raise Ep725LevelsetPredictorAdapterError("bounded receipt pair IDs must be exact prefix n1 or n2")
        if type(self.chronological_camera_frame_sha256) is not tuple or len(
            self.chronological_camera_frame_sha256
        ) != 2 * len(self.source_pair_ids):
            raise Ep725LevelsetPredictorAdapterError("bounded receipt must bind two chronological frames per pair")
        for index, digest in enumerate(self.chronological_camera_frame_sha256):
            _require_sha256(digest, label=f"chronological_camera_frame_sha256[{index}]")
        integer_expectations = {
            "source_archive_bytes": EP725_ARCHIVE_BYTES,
            "counted_predictor_bytes": EP725_MEMBER_BYTES,
            "chronological_raw_prefix_bytes": (
                len(self.source_pair_ids) * 2 * EP725_CAMERA_HEIGHT * EP725_CAMERA_WIDTH * 3
            ),
        }
        for field_name, expected in integer_expectations.items():
            value = getattr(self, field_name)
            if type(value) is not int or value != expected:
                raise Ep725LevelsetPredictorAdapterError(
                    f"bounded receipt {field_name} differs from exact cardinality {expected}"
                )
        frozen_hashes = {
            "source_archive_sha256": EP725_ARCHIVE_SHA256,
            "source_member_sha256": EP725_MEMBER_SHA256,
            "source_runtime_sha256": EP725_RUNTIME_SHA256,
        }
        for field_name, expected in frozen_hashes.items():
            if getattr(self, field_name) != expected:
                raise Ep725LevelsetPredictorAdapterError(f"bounded receipt {field_name} differs from frozen custody")
        if self.decoder_labels_origin != _DECODER_LABELS_ORIGIN or self.transport_kind != "NONE":
            raise Ep725LevelsetPredictorAdapterError("bounded receipt changed decoder-label or absent-transport truth")
        truth_contract = {
            "deterministic_double_decode": True,
            "shipped_numpy_raw_bit_exact": True,
            "source_zip_wrapper_is_counted_p": False,
            "member_is_counted_p": True,
            "dense_frames_persisted": False,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "research_only": True,
        }
        for field_name, expected in truth_contract.items():
            if getattr(self, field_name) is not expected:
                raise Ep725LevelsetPredictorAdapterError(f"bounded receipt truth flag {field_name} changed")
        expected_semantic = _expected_semantic_binding(self)
        if self.predictor_semantic_binding_sha256 != expected_semantic:
            raise Ep725LevelsetPredictorAdapterError("bounded receipt semantic-state binding is not recomposable")
        if self.predictor_state_binding_sha256 != _expected_state_binding(expected_semantic):
            raise Ep725LevelsetPredictorAdapterError("bounded receipt full state binding is not recomposable")
        if self.canonical_renderer_sha256 != EP725_CANONICAL_RENDERER_SHA256:
            raise Ep725LevelsetPredictorAdapterError("bounded receipt renderer differs from frozen NumPy oracle")
        expected_output = _FROZEN_BOUNDED_OUTPUTS[self.source_pair_ids]
        output_fields = (
            "chronological_camera_frame_sha256",
            "chronological_raw_prefix_sha256",
            "labels_sha256",
            "predictor_semantic_binding_sha256",
            "predictor_state_binding_sha256",
        )
        mismatches = [
            field_name for field_name in output_fields if getattr(self, field_name) != expected_output[field_name]
        ]
        if mismatches:
            raise Ep725LevelsetPredictorAdapterError(
                f"bounded receipt output differs from frozen exact decode: {sorted(mismatches)}"
            )
        if self.receipt_sha256 != expected_output["receipt_sha256"]:
            raise Ep725LevelsetPredictorAdapterError("bounded receipt bytes differ from frozen exact receipt")

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_claim": self.candidate_claim,
            "canonical_renderer_sha256": self.canonical_renderer_sha256,
            "chronological_camera_frame_sha256": list(self.chronological_camera_frame_sha256),
            "chronological_raw_prefix_bytes": self.chronological_raw_prefix_bytes,
            "chronological_raw_prefix_sha256": self.chronological_raw_prefix_sha256,
            "counted_predictor_bytes": self.counted_predictor_bytes,
            "decoder_labels_origin": self.decoder_labels_origin,
            "dense_frames_persisted": self.dense_frames_persisted,
            "deterministic_double_decode": self.deterministic_double_decode,
            "labels_sha256": self.labels_sha256,
            "member_is_counted_p": self.member_is_counted_p,
            "predictor_semantic_binding_sha256": self.predictor_semantic_binding_sha256,
            "predictor_state_binding_sha256": self.predictor_state_binding_sha256,
            "promotion_eligible": self.promotion_eligible,
            "research_only": self.research_only,
            "schema": self.schema,
            "score_claim": self.score_claim,
            "scorer_invoked": self.scorer_invoked,
            "shipped_numpy_raw_bit_exact": self.shipped_numpy_raw_bit_exact,
            "source_archive_bytes": self.source_archive_bytes,
            "source_archive_sha256": self.source_archive_sha256,
            "source_member_sha256": self.source_member_sha256,
            "source_pair_ids": list(self.source_pair_ids),
            "source_runtime_sha256": self.source_runtime_sha256,
            "source_zip_wrapper_is_counted_p": self.source_zip_wrapper_is_counted_p,
            "transport_kind": self.transport_kind,
        }

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


_BOUNDED_RECEIPT_KEYS = frozenset(Ep725BoundedDecodeReceiptV2.__dataclass_fields__)


def _unique_receipt_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Ep725LevelsetPredictorAdapterError(f"bounded receipt repeats key {key!r}")
        result[key] = value
    return result


def parse_ep725_bounded_decode_receipt(payload: bytes) -> Ep725BoundedDecodeReceiptV2:
    """Strictly parse and re-emit the canonical bounded-decode receipt."""

    if type(payload) is not bytes:
        raise Ep725LevelsetPredictorAdapterError("bounded receipt must be exact bytes")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_unique_receipt_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                Ep725LevelsetPredictorAdapterError(f"bounded receipt has non-finite constant {item}")
            ),
        )
    except Ep725LevelsetPredictorAdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Ep725LevelsetPredictorAdapterError("bounded receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != _BOUNDED_RECEIPT_KEYS:
        raise Ep725LevelsetPredictorAdapterError("bounded receipt keys differ from the closed schema")
    if _canonical_json(value) != payload:
        raise Ep725LevelsetPredictorAdapterError("bounded receipt is not byte-canonical JSON")
    if type(value["source_pair_ids"]) is not list or type(value["chronological_camera_frame_sha256"]) is not list:
        raise Ep725LevelsetPredictorAdapterError("bounded receipt array fields must be exact JSON arrays")
    parsed = Ep725BoundedDecodeReceiptV2(
        **{
            **value,
            "source_pair_ids": tuple(value["source_pair_ids"]),
            "chronological_camera_frame_sha256": tuple(value["chronological_camera_frame_sha256"]),
        }
    )
    if parsed.to_receipt_bytes() != payload:
        raise Ep725LevelsetPredictorAdapterError("bounded receipt changed on typed parse-back")
    return parsed


@dataclass(frozen=True, slots=True)
class Ep725CountedMemberCausalDecodeReceiptV3:
    """Proof that explicit P/runtime bytes, not the historical ZIP, were run."""

    schema: Literal["tac.ep725_counted_member_causal_decode_receipt.v3"]
    decode_input_kind: Literal["explicit_immutable_counted_lvls1_member_bytes"]
    runtime_input_kind: Literal["explicit_immutable_shipped_runtime_bytes"]
    canonical_renderer_input_kind: Literal["repository_hash_bound_numpy_renderer_source"]
    counted_member_sha256: str
    counted_member_bytes: int
    explicit_runtime_sha256: str
    explicit_runtime_bytes: int
    canonical_renderer_sha256: str
    historical_source_archive_lineage_sha256: str
    historical_source_archive_lineage_bytes: int
    legacy_v2_bounded_receipt_sha256: str
    legacy_v2_predictor_state_binding_sha256: str
    legacy_v2_predictor_semantic_binding_sha256: str
    source_pair_ids: tuple[int, ...]
    chronological_camera_frame_sha256: tuple[str, ...]
    chronological_raw_prefix_sha256: str
    labels_sha256: str
    source_directory_or_archive_read: Literal[False] = False
    source_archive_bytes_read: Literal[0] = 0
    historical_archive_lineage_is_decode_input: Literal[False] = False
    counted_member_consumed_by_canonical_numpy: Literal[True] = True
    counted_member_consumed_by_shipped_runtime: Literal[True] = True
    explicit_runtime_executed: Literal[True] = True
    canonical_renderer_source_executed: Literal[True] = True
    canonical_numpy_double_decode_exact: Literal[True] = True
    shipped_runtime_double_decode_exact: Literal[True] = True
    shipped_numpy_raw_bit_exact: Literal[True] = True
    legacy_v2_state_binding_includes_historical_archive_lineage: Literal[True] = True
    standalone_runtime_packaged: Literal[False] = False
    dense_frames_persisted: Literal[False] = False
    scorer_invoked: Literal[False] = False
    score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != EP725_CAUSAL_RECEIPT_SCHEMA:
            raise Ep725LevelsetPredictorAdapterError("causal receipt schema is not closed V3")
        if self.decode_input_kind != "explicit_immutable_counted_lvls1_member_bytes":
            raise Ep725LevelsetPredictorAdapterError("causal receipt changed counted-member input kind")
        if self.runtime_input_kind != "explicit_immutable_shipped_runtime_bytes":
            raise Ep725LevelsetPredictorAdapterError("causal receipt changed runtime input kind")
        if self.canonical_renderer_input_kind != "repository_hash_bound_numpy_renderer_source":
            raise Ep725LevelsetPredictorAdapterError("causal receipt changed renderer input kind")
        for field_name in (
            "counted_member_sha256",
            "explicit_runtime_sha256",
            "canonical_renderer_sha256",
            "historical_source_archive_lineage_sha256",
            "legacy_v2_bounded_receipt_sha256",
            "legacy_v2_predictor_state_binding_sha256",
            "legacy_v2_predictor_semantic_binding_sha256",
            "chronological_raw_prefix_sha256",
            "labels_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        if (
            type(self.counted_member_bytes) is not int
            or self.counted_member_bytes != EP725_MEMBER_BYTES
            or type(self.explicit_runtime_bytes) is not int
            or self.explicit_runtime_bytes != EP725_RUNTIME_BYTES
            or type(self.historical_source_archive_lineage_bytes) is not int
            or self.historical_source_archive_lineage_bytes != EP725_ARCHIVE_BYTES
            or type(self.source_archive_bytes_read) is not int
            or self.source_archive_bytes_read != 0
        ):
            raise Ep725LevelsetPredictorAdapterError("causal receipt byte counts differ from exact custody")
        exact_hashes = {
            "counted_member_sha256": EP725_MEMBER_SHA256,
            "explicit_runtime_sha256": EP725_RUNTIME_SHA256,
            "canonical_renderer_sha256": EP725_CANONICAL_RENDERER_SHA256,
            "historical_source_archive_lineage_sha256": EP725_ARCHIVE_SHA256,
        }
        for field_name, expected in exact_hashes.items():
            if getattr(self, field_name) != expected:
                raise Ep725LevelsetPredictorAdapterError(f"causal receipt {field_name} differs from custody")
        if (
            type(self.source_pair_ids) is not tuple
            or any(type(value) is not int for value in self.source_pair_ids)
            or self.source_pair_ids not in {(0,), (0, 1)}
        ):
            raise Ep725LevelsetPredictorAdapterError("causal receipt pair IDs must be exact prefix n1 or n2")
        if type(self.chronological_camera_frame_sha256) is not tuple or len(
            self.chronological_camera_frame_sha256
        ) != 2 * len(self.source_pair_ids):
            raise Ep725LevelsetPredictorAdapterError("causal receipt frame hashes changed factor-2 cardinality")
        for index, digest in enumerate(self.chronological_camera_frame_sha256):
            _require_sha256(digest, label=f"causal chronological frame[{index}]")
        expected = _FROZEN_BOUNDED_OUTPUTS[self.source_pair_ids]
        expected_fields = {
            "legacy_v2_bounded_receipt_sha256": expected["receipt_sha256"],
            "legacy_v2_predictor_state_binding_sha256": expected["predictor_state_binding_sha256"],
            "legacy_v2_predictor_semantic_binding_sha256": expected["predictor_semantic_binding_sha256"],
            "chronological_camera_frame_sha256": expected["chronological_camera_frame_sha256"],
            "chronological_raw_prefix_sha256": expected["chronological_raw_prefix_sha256"],
            "labels_sha256": expected["labels_sha256"],
        }
        mismatches = [
            field_name
            for field_name, expected_value in expected_fields.items()
            if getattr(self, field_name) != expected_value
        ]
        if mismatches:
            raise Ep725LevelsetPredictorAdapterError(
                f"causal receipt differs from frozen exact output: {sorted(mismatches)}"
            )
        fixed_truth = (
            self.source_directory_or_archive_read is False
            and self.source_archive_bytes_read == 0
            and self.historical_archive_lineage_is_decode_input is False
            and self.counted_member_consumed_by_canonical_numpy is True
            and self.counted_member_consumed_by_shipped_runtime is True
            and self.explicit_runtime_executed is True
            and self.canonical_renderer_source_executed is True
            and self.canonical_numpy_double_decode_exact is True
            and self.shipped_runtime_double_decode_exact is True
            and self.shipped_numpy_raw_bit_exact is True
            and self.legacy_v2_state_binding_includes_historical_archive_lineage is True
            and self.standalone_runtime_packaged is False
            and self.dense_frames_persisted is False
            and self.scorer_invoked is False
            and self.score_claim is False
            and self.candidate_claim is False
            and self.promotion_eligible is False
            and self.research_only is True
        )
        if not fixed_truth:
            raise Ep725LevelsetPredictorAdapterError("causal receipt truth labels became permissive")

    def as_dict(self) -> dict[str, object]:
        value = {
            field_name: getattr(self, field_name)
            for field_name in Ep725CountedMemberCausalDecodeReceiptV3.__dataclass_fields__
        }
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["chronological_camera_frame_sha256"] = list(self.chronological_camera_frame_sha256)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


_CAUSAL_RECEIPT_KEYS = frozenset(Ep725CountedMemberCausalDecodeReceiptV3.__dataclass_fields__)


def parse_ep725_counted_member_causal_decode_receipt(
    payload: bytes,
) -> Ep725CountedMemberCausalDecodeReceiptV3:
    """Strictly parse and re-emit the explicit-input V3 receipt."""

    if type(payload) is not bytes:
        raise Ep725LevelsetPredictorAdapterError("causal receipt must be exact bytes")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_unique_receipt_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                Ep725LevelsetPredictorAdapterError(f"causal receipt has non-finite constant {item}")
            ),
        )
    except Ep725LevelsetPredictorAdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Ep725LevelsetPredictorAdapterError("causal receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != _CAUSAL_RECEIPT_KEYS:
        raise Ep725LevelsetPredictorAdapterError("causal receipt keys differ from the closed schema")
    if _canonical_json(value) != payload:
        raise Ep725LevelsetPredictorAdapterError("causal receipt is not byte-canonical JSON")
    if type(value["source_pair_ids"]) is not list or type(value["chronological_camera_frame_sha256"]) is not list:
        raise Ep725LevelsetPredictorAdapterError("causal receipt array fields must be exact JSON arrays")
    parsed = Ep725CountedMemberCausalDecodeReceiptV3(
        schema=value["schema"],
        decode_input_kind=value["decode_input_kind"],
        runtime_input_kind=value["runtime_input_kind"],
        canonical_renderer_input_kind=value["canonical_renderer_input_kind"],
        counted_member_sha256=value["counted_member_sha256"],
        counted_member_bytes=value["counted_member_bytes"],
        explicit_runtime_sha256=value["explicit_runtime_sha256"],
        explicit_runtime_bytes=value["explicit_runtime_bytes"],
        canonical_renderer_sha256=value["canonical_renderer_sha256"],
        historical_source_archive_lineage_sha256=value["historical_source_archive_lineage_sha256"],
        historical_source_archive_lineage_bytes=value["historical_source_archive_lineage_bytes"],
        legacy_v2_bounded_receipt_sha256=value["legacy_v2_bounded_receipt_sha256"],
        legacy_v2_predictor_state_binding_sha256=value["legacy_v2_predictor_state_binding_sha256"],
        legacy_v2_predictor_semantic_binding_sha256=value["legacy_v2_predictor_semantic_binding_sha256"],
        source_pair_ids=tuple(value["source_pair_ids"]),
        chronological_camera_frame_sha256=tuple(value["chronological_camera_frame_sha256"]),
        chronological_raw_prefix_sha256=value["chronological_raw_prefix_sha256"],
        labels_sha256=value["labels_sha256"],
        source_directory_or_archive_read=value["source_directory_or_archive_read"],
        source_archive_bytes_read=value["source_archive_bytes_read"],
        historical_archive_lineage_is_decode_input=value["historical_archive_lineage_is_decode_input"],
        counted_member_consumed_by_canonical_numpy=value["counted_member_consumed_by_canonical_numpy"],
        counted_member_consumed_by_shipped_runtime=value["counted_member_consumed_by_shipped_runtime"],
        explicit_runtime_executed=value["explicit_runtime_executed"],
        canonical_renderer_source_executed=value["canonical_renderer_source_executed"],
        canonical_numpy_double_decode_exact=value["canonical_numpy_double_decode_exact"],
        shipped_runtime_double_decode_exact=value["shipped_runtime_double_decode_exact"],
        shipped_numpy_raw_bit_exact=value["shipped_numpy_raw_bit_exact"],
        legacy_v2_state_binding_includes_historical_archive_lineage=value[
            "legacy_v2_state_binding_includes_historical_archive_lineage"
        ],
        standalone_runtime_packaged=value["standalone_runtime_packaged"],
        dense_frames_persisted=value["dense_frames_persisted"],
        scorer_invoked=value["scorer_invoked"],
        score_claim=value["score_claim"],
        candidate_claim=value["candidate_claim"],
        promotion_eligible=value["promotion_eligible"],
        research_only=value["research_only"],
    )
    if parsed.to_receipt_bytes() != payload:
        raise Ep725LevelsetPredictorAdapterError("causal receipt changed on typed parse-back")
    return parsed


@dataclass(frozen=True, slots=True)
class Ep725BoundedDecodeV2:
    """Hash-only bounded proof; no dense decoded frame is retained."""

    predictor_state: TaskspacePredictorStateV2
    source_archive_sha256: str
    source_archive_bytes: int
    source_member_sha256: str
    counted_predictor_bytes: int
    source_runtime_sha256: str
    canonical_renderer_sha256: str
    source_pair_ids: tuple[int, ...]
    chronological_camera_frame_sha256: tuple[str, ...]
    chronological_raw_prefix_sha256: str
    chronological_raw_prefix_bytes: int
    labels_sha256: str
    decoder_labels_origin: str = _DECODER_LABELS_ORIGIN
    transport_kind: str = "NONE"
    deterministic_double_decode: bool = True
    shipped_numpy_raw_bit_exact: bool = True
    source_zip_wrapper_is_counted_p: bool = False
    member_is_counted_p: bool = True
    dense_frames_persisted: bool = False
    scorer_invoked: bool = False
    score_claim: bool = False
    candidate_claim: bool = False
    promotion_eligible: bool = False
    research_only: bool = True

    def __post_init__(self) -> None:
        if type(self.predictor_state) is not TaskspacePredictorStateV2:
            raise Ep725LevelsetPredictorAdapterError("bounded result requires exact TaskspacePredictorStateV2")
        state = self.predictor_state
        if type(state.transport) is not NoTransportV2:
            raise Ep725LevelsetPredictorAdapterError("bounded ep725 state must carry canonical NONE transport")
        receipt = self.as_receipt()
        state_equalities = {
            "source_archive_sha256": state.source_archive_sha256,
            "source_member_sha256": state.source_member_sha256,
            "counted_predictor_bytes": state.predictor_program_bytes,
            "source_runtime_sha256": state.source_runtime_sha256,
            "canonical_renderer_sha256": state.predictor_renderer_sha256,
            "source_pair_ids": state.source_pair_ids,
            "labels_sha256": state.labels_sha256,
        }
        mismatches = [
            field_name for field_name, expected in state_equalities.items() if getattr(self, field_name) != expected
        ]
        if mismatches:
            raise Ep725LevelsetPredictorAdapterError(
                f"bounded result fields differ from live predictor state: {sorted(mismatches)}"
            )
        if receipt.predictor_semantic_binding_sha256 != state.semantic_binding_sha256:
            raise Ep725LevelsetPredictorAdapterError("bounded receipt semantic binding differs from live state")
        if receipt.predictor_state_binding_sha256 != state.binding_sha256:
            raise Ep725LevelsetPredictorAdapterError("bounded receipt full binding differs from live state")

    def as_receipt(self) -> Ep725BoundedDecodeReceiptV2:
        state = self.predictor_state
        return Ep725BoundedDecodeReceiptV2(
            schema=EP725_BOUNDED_RECEIPT_SCHEMA,
            predictor_state_binding_sha256=state.binding_sha256,
            predictor_semantic_binding_sha256=state.semantic_binding_sha256,
            source_archive_sha256=self.source_archive_sha256,
            source_archive_bytes=self.source_archive_bytes,
            source_member_sha256=self.source_member_sha256,
            counted_predictor_bytes=self.counted_predictor_bytes,
            source_runtime_sha256=self.source_runtime_sha256,
            canonical_renderer_sha256=self.canonical_renderer_sha256,
            source_pair_ids=self.source_pair_ids,
            chronological_camera_frame_sha256=self.chronological_camera_frame_sha256,
            chronological_raw_prefix_sha256=self.chronological_raw_prefix_sha256,
            chronological_raw_prefix_bytes=self.chronological_raw_prefix_bytes,
            labels_sha256=self.labels_sha256,
            decoder_labels_origin=self.decoder_labels_origin,
            transport_kind=self.transport_kind,
            deterministic_double_decode=self.deterministic_double_decode,
            shipped_numpy_raw_bit_exact=self.shipped_numpy_raw_bit_exact,
            source_zip_wrapper_is_counted_p=self.source_zip_wrapper_is_counted_p,
            member_is_counted_p=self.member_is_counted_p,
            dense_frames_persisted=self.dense_frames_persisted,
            scorer_invoked=self.scorer_invoked,
            score_claim=self.score_claim,
            candidate_claim=self.candidate_claim,
            promotion_eligible=self.promotion_eligible,
            research_only=self.research_only,
        )

    def as_dict(self) -> dict[str, object]:
        return self.as_receipt().as_dict()

    def to_receipt_bytes(self) -> bytes:
        return self.as_receipt().to_receipt_bytes()

    @property
    def receipt_sha256(self) -> str:
        return self.as_receipt().receipt_sha256


def _validate_bounded_decode_request(pair_count: int, timeout_seconds: float) -> None:
    if type(pair_count) is not int or not 1 <= pair_count <= MAX_BOUNDED_DECODE_PAIRS:
        raise Ep725LevelsetPredictorAdapterError(f"bounded proof pair_count must be in [1,{MAX_BOUNDED_DECODE_PAIRS}]")
    if type(timeout_seconds) not in {int, float} or not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise Ep725LevelsetPredictorAdapterError("timeout_seconds must be a finite positive scalar")


def _decode_executable_source_materialized(
    source: _Ep725ExecutableSource,
    *,
    source_archive_lineage_sha256: str,
    source_archive_lineage_bytes: int,
    pair_count: int,
    timeout_seconds: float,
) -> tuple[Ep725BoundedDecodeV2, bytes]:
    """Run both decoders from the source object's member/runtime attributes."""

    _validate_bounded_decode_request(pair_count, timeout_seconds)
    if type(source.member) is not bytes or type(source.runtime) is not bytes:
        raise Ep725LevelsetPredictorAdapterError("decoder source must expose exact immutable member/runtime bytes")
    _require_sha256(source_archive_lineage_sha256, label="source_archive_lineage_sha256")
    if type(source_archive_lineage_bytes) is not int or source_archive_lineage_bytes < 1:
        raise Ep725LevelsetPredictorAdapterError("source archive lineage bytes must be a positive exact integer")
    module, canonical_renderer_sha256 = _load_canonical_decoder_module()
    numpy_raw_1, labels_1 = _canonical_decode_once(module, source, pair_count)
    numpy_raw_2, labels_2 = _canonical_decode_once(module, source, pair_count)
    if numpy_raw_1 != numpy_raw_2 or not np.array_equal(labels_1, labels_2):
        raise Ep725PredictorDecodeBlocker(
            "CANONICAL_NUMPY_NONDETERMINISTIC",
            "two fresh reference decodes differ in raw bytes or phi.argmax",
        )
    shipped_raw_1 = _shipped_decode_once(source, pair_count, timeout_seconds=float(timeout_seconds))
    shipped_raw_2 = _shipped_decode_once(source, pair_count, timeout_seconds=float(timeout_seconds))
    if shipped_raw_1 != shipped_raw_2:
        raise Ep725PredictorDecodeBlocker(
            "SHIPPED_RUNTIME_NONDETERMINISTIC", "two fresh shipped-runtime decodes differ"
        )
    if shipped_raw_1 != numpy_raw_1:
        raise Ep725PredictorDecodeBlocker(
            "SHIPPED_NUMPY_PARITY_MISMATCH",
            "exact shipped runtime raw bytes differ from canonical NumPy reference",
        )
    source_pair_ids = tuple(range(pair_count))
    state = TaskspacePredictorStateV2(
        predictor_program=source.member,
        predictor_renderer_sha256=canonical_renderer_sha256,
        source_archive_sha256=source_archive_lineage_sha256,
        source_runtime_sha256=_sha256(source.runtime),
        source_member_name="0.bin",
        source_pair_ids=source_pair_ids,
        labels=labels_1,
        transport=NoTransportV2(),
    )
    frame_bytes = EP725_CAMERA_HEIGHT * EP725_CAMERA_WIDTH * 3
    frame_hashes = tuple(
        _sha256(shipped_raw_1[offset : offset + frame_bytes]) for offset in range(0, len(shipped_raw_1), frame_bytes)
    )
    result = Ep725BoundedDecodeV2(
        predictor_state=state,
        source_archive_sha256=source_archive_lineage_sha256,
        source_archive_bytes=source_archive_lineage_bytes,
        source_member_sha256=_sha256(source.member),
        counted_predictor_bytes=len(source.member),
        source_runtime_sha256=_sha256(source.runtime),
        canonical_renderer_sha256=canonical_renderer_sha256,
        source_pair_ids=source_pair_ids,
        chronological_camera_frame_sha256=frame_hashes,
        chronological_raw_prefix_sha256=_sha256(shipped_raw_1),
        chronological_raw_prefix_bytes=len(shipped_raw_1),
        labels_sha256=state.labels_sha256,
    )
    return result, shipped_raw_1


def _decode_ep725_prefix_materialized(
    source_directory: str | Path = EP725_SOURCE_DIRECTORY,
    *,
    pair_count: int = MAX_BOUNDED_DECODE_PAIRS,
    timeout_seconds: float = 60.0,
) -> tuple[Ep725BoundedDecodeV2, bytes]:
    """Legacy archive-backed decode retained byte-identically for compatibility."""

    _validate_bounded_decode_request(pair_count, timeout_seconds)
    source = inspect_ep725_source(source_directory)
    return _decode_executable_source_materialized(
        source,
        source_archive_lineage_sha256=source.archive_sha256,
        source_archive_lineage_bytes=len(source.archive),
        pair_count=pair_count,
        timeout_seconds=timeout_seconds,
    )


@dataclass(frozen=True, slots=True)
class Ep725EphemeralRuntimeSurfaceV2:
    """Exact in-memory camera frames; never serialized or counted a second time."""

    bounded_decode: Ep725BoundedDecodeV2
    chronological_camera_frames: np.ndarray
    surface_origin: str = "exact_shipped_runtime_raw_equal_canonical_numpy"
    persisted: bool = False
    additional_counted_bytes: int = 0
    scorer_invoked: bool = False
    candidate_claim: bool = False

    def __post_init__(self) -> None:
        if type(self.bounded_decode) is not Ep725BoundedDecodeV2:
            raise Ep725LevelsetPredictorAdapterError("ephemeral surface requires exact bounded decode custody")
        frames = np.asarray(self.chronological_camera_frames)
        expected_shape = (
            len(self.bounded_decode.source_pair_ids),
            2,
            EP725_CAMERA_HEIGHT,
            EP725_CAMERA_WIDTH,
            3,
        )
        if frames.dtype != np.uint8 or frames.shape != expected_shape:
            raise Ep725LevelsetPredictorAdapterError("ephemeral surface changed chronological camera uint8 ABI")
        immutable = np.ascontiguousarray(frames, dtype=np.uint8).copy()
        immutable.setflags(write=False)
        flat_frames = immutable.reshape(-1, EP725_CAMERA_HEIGHT, EP725_CAMERA_WIDTH, 3)
        frame_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in flat_frames)
        raw_hash = _sha256(memoryview(immutable).cast("B"))
        if frame_hashes != self.bounded_decode.chronological_camera_frame_sha256:
            raise Ep725LevelsetPredictorAdapterError("ephemeral camera frames differ from bounded frame hashes")
        if raw_hash != self.bounded_decode.chronological_raw_prefix_sha256:
            raise Ep725LevelsetPredictorAdapterError("ephemeral camera surface differs from bounded raw prefix hash")
        if (
            self.surface_origin != "exact_shipped_runtime_raw_equal_canonical_numpy"
            or self.persisted is not False
            or type(self.additional_counted_bytes) is not int
            or self.additional_counted_bytes != 0
            or self.scorer_invoked is not False
            or self.candidate_claim is not False
        ):
            raise Ep725LevelsetPredictorAdapterError("ephemeral surface custody truth changed")
        object.__setattr__(self, "chronological_camera_frames", immutable)

    @property
    def predictor_state(self) -> TaskspacePredictorStateV2:
        return self.bounded_decode.predictor_state

    @property
    def frame1_camera(self) -> np.ndarray:
        return self.chronological_camera_frames[:, 1]


@dataclass(frozen=True, slots=True)
class Ep725CountedMemberCausalSurfaceV3:
    """Explicit-input causality proof plus the unchanged V2 camera/state ABI."""

    ephemeral_surface: Ep725EphemeralRuntimeSurfaceV2
    causal_receipt: Ep725CountedMemberCausalDecodeReceiptV3

    def __post_init__(self) -> None:
        if type(self.ephemeral_surface) is not Ep725EphemeralRuntimeSurfaceV2:
            raise Ep725LevelsetPredictorAdapterError("causal surface requires exact V2 ephemeral output")
        if type(self.causal_receipt) is not Ep725CountedMemberCausalDecodeReceiptV3:
            raise Ep725LevelsetPredictorAdapterError("causal surface requires exact V3 receipt")
        bounded = self.ephemeral_surface.bounded_decode
        receipt_equalities = {
            "legacy_v2_bounded_receipt_sha256": bounded.receipt_sha256,
            "legacy_v2_predictor_state_binding_sha256": bounded.predictor_state.binding_sha256,
            "legacy_v2_predictor_semantic_binding_sha256": bounded.predictor_state.semantic_binding_sha256,
            "source_pair_ids": bounded.source_pair_ids,
            "chronological_camera_frame_sha256": bounded.chronological_camera_frame_sha256,
            "chronological_raw_prefix_sha256": bounded.chronological_raw_prefix_sha256,
            "labels_sha256": bounded.labels_sha256,
        }
        mismatches = [
            field_name
            for field_name, expected in receipt_equalities.items()
            if getattr(self.causal_receipt, field_name) != expected
        ]
        if mismatches:
            raise Ep725LevelsetPredictorAdapterError(
                f"causal surface differs from bounded V2 output: {sorted(mismatches)}"
            )

    @property
    def bounded_decode(self) -> Ep725BoundedDecodeV2:
        return self.ephemeral_surface.bounded_decode

    @property
    def predictor_state(self) -> TaskspacePredictorStateV2:
        return self.ephemeral_surface.predictor_state

    @property
    def chronological_camera_frames(self) -> np.ndarray:
        return self.ephemeral_surface.chronological_camera_frames

    @property
    def frame1_camera(self) -> np.ndarray:
        return self.ephemeral_surface.frame1_camera


def decode_ep725_counted_member_ephemeral_surface(
    counted_member: bytes,
    *,
    shipped_runtime: bytes,
    pair_count: int = MAX_BOUNDED_DECODE_PAIRS,
    timeout_seconds: float = 60.0,
) -> Ep725CountedMemberCausalSurfaceV3:
    """Decode actual explicit P/runtime bytes without reading a source archive.

    The archive SHA retained by the unchanged V2 state is historical lineage,
    not an input to this function.  V3 records that distinction explicitly.
    """

    source = Ep725ExplicitMemberRuntimeSourceV3(
        member=counted_member,
        runtime=shipped_runtime,
    )
    result, raw = _decode_executable_source_materialized(
        source,
        source_archive_lineage_sha256=EP725_ARCHIVE_SHA256,
        source_archive_lineage_bytes=EP725_ARCHIVE_BYTES,
        pair_count=pair_count,
        timeout_seconds=timeout_seconds,
    )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(
        len(result.source_pair_ids),
        2,
        EP725_CAMERA_HEIGHT,
        EP725_CAMERA_WIDTH,
        3,
    )
    ephemeral = Ep725EphemeralRuntimeSurfaceV2(
        bounded_decode=result,
        chronological_camera_frames=frames,
    )
    causal_receipt = Ep725CountedMemberCausalDecodeReceiptV3(
        schema=EP725_CAUSAL_RECEIPT_SCHEMA,
        decode_input_kind="explicit_immutable_counted_lvls1_member_bytes",
        runtime_input_kind="explicit_immutable_shipped_runtime_bytes",
        canonical_renderer_input_kind="repository_hash_bound_numpy_renderer_source",
        counted_member_sha256=source.member_sha256,
        counted_member_bytes=len(source.member),
        explicit_runtime_sha256=source.runtime_sha256,
        explicit_runtime_bytes=len(source.runtime),
        canonical_renderer_sha256=result.canonical_renderer_sha256,
        historical_source_archive_lineage_sha256=result.source_archive_sha256,
        historical_source_archive_lineage_bytes=result.source_archive_bytes,
        legacy_v2_bounded_receipt_sha256=result.receipt_sha256,
        legacy_v2_predictor_state_binding_sha256=result.predictor_state.binding_sha256,
        legacy_v2_predictor_semantic_binding_sha256=result.predictor_state.semantic_binding_sha256,
        source_pair_ids=result.source_pair_ids,
        chronological_camera_frame_sha256=result.chronological_camera_frame_sha256,
        chronological_raw_prefix_sha256=result.chronological_raw_prefix_sha256,
        labels_sha256=result.labels_sha256,
    )
    if parse_ep725_counted_member_causal_decode_receipt(causal_receipt.to_receipt_bytes()) != causal_receipt:
        raise Ep725LevelsetPredictorAdapterError("causal receipt failed strict parse/re-emit closure")
    return Ep725CountedMemberCausalSurfaceV3(
        ephemeral_surface=ephemeral,
        causal_receipt=causal_receipt,
    )


def decode_ep725_prefix(
    source_directory: str | Path = EP725_SOURCE_DIRECTORY,
    *,
    pair_count: int = MAX_BOUNDED_DECODE_PAIRS,
    timeout_seconds: float = 60.0,
) -> Ep725BoundedDecodeV2:
    """Decode twice through both bound implementations and discard dense bytes."""

    result, _raw = _decode_ep725_prefix_materialized(
        source_directory,
        pair_count=pair_count,
        timeout_seconds=timeout_seconds,
    )
    return result


def decode_ep725_prefix_ephemeral_surface(
    source_directory: str | Path = EP725_SOURCE_DIRECTORY,
    *,
    pair_count: int = MAX_BOUNDED_DECODE_PAIRS,
    timeout_seconds: float = 60.0,
) -> Ep725EphemeralRuntimeSurfaceV2:
    """Expose exact camera frames in memory without persisting or recounting them."""

    result, raw = _decode_ep725_prefix_materialized(
        source_directory,
        pair_count=pair_count,
        timeout_seconds=timeout_seconds,
    )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(
        len(result.source_pair_ids),
        2,
        EP725_CAMERA_HEIGHT,
        EP725_CAMERA_WIDTH,
        3,
    )
    return Ep725EphemeralRuntimeSurfaceV2(
        bounded_decode=result,
        chronological_camera_frames=frames,
    )


__all__ = [
    "EP725_ARCHIVE_SHA256",
    "EP725_BOUNDED_RECEIPT_SCHEMA",
    "EP725_CANONICAL_RENDERER_SHA256",
    "EP725_CAUSAL_RECEIPT_SCHEMA",
    "EP725_FROZEN_EXPECTATIONS",
    "EP725_MANIFEST_SHA256",
    "EP725_MEMBER_SHA256",
    "EP725_RUNTIME_BYTES",
    "EP725_RUNTIME_SHA256",
    "EP725_SOURCE_DIRECTORY",
    "Ep725BoundedDecodeReceiptV2",
    "Ep725BoundedDecodeV2",
    "Ep725CountedMemberCausalDecodeReceiptV3",
    "Ep725CountedMemberCausalSurfaceV3",
    "Ep725EphemeralRuntimeSurfaceV2",
    "Ep725ExplicitMemberRuntimeSourceV3",
    "Ep725LevelsetPredictorAdapterError",
    "Ep725PredictorDecodeBlocker",
    "Ep725SourceExpectationsV2",
    "Ep725SourceObjectV2",
    "decode_ep725_counted_member_ephemeral_surface",
    "decode_ep725_prefix",
    "decode_ep725_prefix_ephemeral_surface",
    "inspect_ep725_source",
    "parse_ep725_bounded_decode_receipt",
    "parse_ep725_counted_member_causal_decode_receipt",
]
