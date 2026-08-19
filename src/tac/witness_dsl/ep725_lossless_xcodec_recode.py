# SPDX-License-Identifier: MIT
"""Exact lossless cross-tensor recoding of the frozen ep725 LVLS1 payload.

The search changes only reversible storage coordinates already understood by
the frozen ep725 receiver.  Candidate selection is made on complete ZIP bytes,
and every selected point is decoded back to the full signed-int8 state before
it can leave this module.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import io
import json
import struct
import zipfile
from dataclasses import dataclass
from typing import Any, Final

import brotli
import numpy as np

from tac.boundary_math.witness_crosstensor_codec import (
    CODE_TRANSFORM_FRAME_DELTA_MOD256,
    CODE_TRANSFORM_RAW,
    decode_base_quantized,
    decode_code_quantized,
    encode_code_quantized,
)

MAGIC: Final = b"LVLS1\x00"
MEMBER_NAME: Final = "0.bin"
SCORE_RATE_DENOMINATOR: Final = 37_545_489
DEFAULT_DEFLATE_PROFILES: Final = (None, *range(1, 10))
CODE_MODE_TO_WIRE: Final = {
    CODE_TRANSFORM_RAW: 0,
    CODE_TRANSFORM_FRAME_DELTA_MOD256: 1,
}
WIRE_TO_CODE_MODE: Final = {value: key for key, value in CODE_MODE_TO_WIRE.items()}

_U32 = struct.Struct("<I")
_STATE_DOMAIN = b"PACT-EP725-XCODEC-QUANTIZED-STATE-V1\x00"


class Ep725LosslessXCodecError(ValueError):
    """Raised when exact source, grammar, search, or parse-back custody fails."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _take(data: bytes, offset: int, size: int, *, label: str) -> tuple[bytes, int]:
    end = offset + size
    if size < 0 or end > len(data):
        raise Ep725LosslessXCodecError(f"truncated {label}: need {size} bytes at {offset}, member has {len(data)}")
    return data[offset:end], end


@dataclass(frozen=True, slots=True)
class ParsedEp725LVLS1:
    member_bytes: bytes
    manifest_bytes: bytes
    manifest: dict[str, Any]
    base_brotli: bytes
    code_brotli: bytes
    pose_bytes: bytes
    base_order: tuple[str, ...]
    base_quantized: dict[str, np.ndarray]
    code_quantized: np.ndarray


@dataclass(frozen=True, slots=True)
class SourceZipProfile:
    archive_bytes: bytes
    member_bytes: bytes
    info: zipfile.ZipInfo
    archive_comment: bytes
    reproducing_deflate_profiles: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class Ep725XCodecSelection:
    transformed: bool
    transpose_mask: int
    transpose_indices: tuple[int, ...]
    transpose_names: tuple[str, ...]
    code_mode: str
    code_wire: int
    deflate_level: int | None
    archive_bytes: bytes
    member_bytes: bytes
    manifest_bytes: bytes
    base_brotli_bytes: int
    code_brotli_bytes: int


@dataclass(frozen=True, slots=True)
class Ep725XCodecRecodeResult:
    source: SourceZipProfile
    source_lvls1: ParsedEp725LVLS1
    selected: Ep725XCodecSelection
    base_candidate_names: tuple[str, ...]
    transformed_points_measured: int
    decoded_state_sha256: str
    selected_decoded_state_sha256: str

    @property
    def archive_delta_bytes(self) -> int:
        return len(self.selected.archive_bytes) - len(self.source.archive_bytes)

    @property
    def member_delta_bytes(self) -> int:
        return len(self.selected.member_bytes) - len(self.source.member_bytes)

    @property
    def rate_score_delta(self) -> float:
        return 25.0 * self.archive_delta_bytes / SCORE_RATE_DENOMINATOR

    def structural_receipt(self) -> dict[str, Any]:
        source = self.source_lvls1
        selected = parse_ep725_lvls1(self.selected.member_bytes, require_source_form=False)
        return {
            "schema": "tac.ep725_lossless_xcodec_recode.v1",
            "authority": "structural exact archive and full quantized-state equality",
            "lane_id": "ep725_lossless_xcodec_recode_20260726",
            "source": {
                "archive_bytes": len(self.source.archive_bytes),
                "archive_sha256": _sha256(self.source.archive_bytes),
                "member_bytes": len(self.source.member_bytes),
                "member_sha256": _sha256(self.source.member_bytes),
                "manifest_bytes": len(source.manifest_bytes),
                "base_brotli_bytes": len(source.base_brotli),
                "code_brotli_bytes": len(source.code_brotli),
                "pose_bytes": len(source.pose_bytes),
                "reproducing_deflate_profiles": list(self.source.reproducing_deflate_profiles),
            },
            "search": {
                "base_candidate_names": list(self.base_candidate_names),
                "base_masks": 1 << len(self.base_candidate_names),
                "code_modes": [CODE_TRANSFORM_RAW, CODE_TRANSFORM_FRAME_DELTA_MOD256],
                "deflate_profiles": list(DEFAULT_DEFLATE_PROFILES),
                "transformed_points_measured": self.transformed_points_measured,
                "selection_surface": "exact complete archive.zip bytes",
            },
            "selected": {
                "transformed": self.selected.transformed,
                "transpose_mask": self.selected.transpose_mask,
                "transpose_indices": list(self.selected.transpose_indices),
                "transpose_names": list(self.selected.transpose_names),
                "code_mode": self.selected.code_mode,
                "code_wire": self.selected.code_wire,
                "deflate_level": self.selected.deflate_level,
                "archive_bytes": len(self.selected.archive_bytes),
                "archive_sha256": _sha256(self.selected.archive_bytes),
                "member_bytes": len(self.selected.member_bytes),
                "member_sha256": _sha256(self.selected.member_bytes),
                "manifest_bytes": len(self.selected.manifest_bytes),
                "base_brotli_bytes": self.selected.base_brotli_bytes,
                "code_brotli_bytes": self.selected.code_brotli_bytes,
            },
            "exact_delta": {
                "archive_bytes": self.archive_delta_bytes,
                "member_bytes": self.member_delta_bytes,
                "rate_score_units": self.rate_score_delta,
                "formula": "25*archive_delta_bytes/37545489",
            },
            "proof": {
                "source_state_sha256": self.decoded_state_sha256,
                "selected_state_sha256": self.selected_decoded_state_sha256,
                "full_quantized_state_equal": hmac.compare_digest(
                    self.decoded_state_sha256, self.selected_decoded_state_sha256
                ),
                "base_arrays_equal": all(
                    np.array_equal(source.base_quantized[name], selected.base_quantized[name])
                    for name in source.base_order
                ),
                "code_array_equal": np.array_equal(source.code_quantized, selected.code_quantized),
                "pose_bytes_equal": source.pose_bytes == selected.pose_bytes,
                "manifest_equal_after_removing_xcodec": _manifest_without_xcodec(source.manifest)
                == _manifest_without_xcodec(selected.manifest),
                "deterministic_rebuild_equal": True,
            },
            "system_wire_in": {
                "sensitivity_map": "zero decoded-state delta; exact rate-only action",
                "pareto_constraint": "same decoded quantized state and lower exact archive bytes",
                "bit_allocator_hook": "REQUANTIZE_STORAGE proposal with exact whole-object price",
                "cathedral_autopilot_hook": "full n600 replay then same-byte contest CPU/CUDA eval owed",
                "continual_learning_update": "record section-local versus whole-archive interaction gain",
                "probe_disambiguator": "finite exhaustive final-ZIP search",
            },
            "truth": {
                "research_only": True,
                "candidate_claim": False,
                "score_claim": False,
                "exact_eval_invoked": False,
                "promotion_eligible": False,
                "pointer_moved": False,
                "public_payload_reused": False,
                "full_n600_receiver_replay_owed": True,
                "contest_cpu_cuda_same_bytes_owed": True,
            },
        }


def _manifest_without_xcodec(manifest: dict[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result.pop("xcodec", None)
    return result


def _validate_manifest(manifest: dict[str, Any], *, require_source_form: bool) -> None:
    if manifest.get("format_version") != 1:
        raise Ep725LosslessXCodecError("LVLS1 format_version must be exactly 1")
    if manifest.get("n_pairs") != 600 or manifest.get("code_shape") != [1200, 32]:
        raise Ep725LosslessXCodecError("recode is scoped to the frozen ep725 n600 code population")
    for optional in ("lane_render_band", "pose_carrier", "chart_payload", "palette_residual"):
        if manifest.get(optional) is not None:
            raise Ep725LosslessXCodecError(f"optional LVLS1 section {optional!r} is outside G20 scope")
    if require_source_form and "xcodec" in manifest:
        raise Ep725LosslessXCodecError("frozen source must not already contain xcodec")
    order = manifest.get("base_param_order")
    shapes = manifest.get("base_shapes")
    if not isinstance(order, list) or not order or not all(isinstance(name, str) for name in order):
        raise Ep725LosslessXCodecError("base_param_order must be a non-empty string list")
    if len(set(order)) != len(order) or not isinstance(shapes, dict) or set(shapes) != set(order):
        raise Ep725LosslessXCodecError("base tensor order/shapes are duplicate, missing, or foreign")


def parse_ep725_lvls1(member: bytes, *, require_source_form: bool) -> ParsedEp725LVLS1:
    """Strictly parse and decode the exact four-section ep725 LVLS1 grammar."""
    if not isinstance(member, bytes) or not member.startswith(MAGIC):
        raise Ep725LosslessXCodecError("member is not immutable LVLS1 bytes")
    offset = len(MAGIC)
    blocks: list[bytes] = []
    for label in ("manifest", "base", "code", "pose"):
        if offset + _U32.size > len(member):
            raise Ep725LosslessXCodecError(f"truncated {label} length")
        (size,) = _U32.unpack_from(member, offset)
        offset += _U32.size
        block, offset = _take(member, offset, size, label=label)
        blocks.append(block)
    if offset != len(member):
        raise Ep725LosslessXCodecError(f"LVLS1 has {len(member) - offset} unconsumed trailing bytes")
    try:
        manifest = json.loads(blocks[0].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Ep725LosslessXCodecError("manifest is not canonical ASCII JSON") from exc
    if not isinstance(manifest, dict) or _canonical_json(manifest) != blocks[0]:
        raise Ep725LosslessXCodecError("manifest bytes are not canonical compact JSON")
    _validate_manifest(manifest, require_source_form=require_source_form)
    if blocks[3]:
        raise Ep725LosslessXCodecError("frozen ep725 pose section must be empty")
    try:
        base_raw = brotli.decompress(blocks[1])
        code_raw = brotli.decompress(blocks[2])
    except brotli.error as exc:
        raise Ep725LosslessXCodecError("base/code Brotli stream failed exact decode") from exc

    order = tuple(manifest["base_param_order"])
    xcodec = manifest.get("xcodec") or {}
    if not isinstance(xcodec, dict) or set(xcodec).difference({"p", "c"}):
        raise Ep725LosslessXCodecError("xcodec must contain only p and c fields")
    indices = tuple(int(value) for value in xcodec.get("p", ()))
    if len(set(indices)) != len(indices) or tuple(sorted(indices)) != indices:
        raise Ep725LosslessXCodecError("xcodec transpose indices must be unique and increasing")
    if any(index < 0 or index >= len(order) for index in indices):
        raise Ep725LosslessXCodecError("xcodec transpose index is out of range")
    names = tuple(order[index] for index in indices)
    try:
        base_quantized = decode_base_quantized(base_raw, order, manifest["base_shapes"], names)
        wire = int(xcodec.get("c", 0))
        mode = WIRE_TO_CODE_MODE[wire]
        code_quantized = decode_code_quantized(code_raw, manifest["code_shape"], mode)
    except (KeyError, TypeError, ValueError) as exc:
        raise Ep725LosslessXCodecError("xcodec/base/code parse-back failed") from exc
    return ParsedEp725LVLS1(
        member_bytes=member,
        manifest_bytes=blocks[0],
        manifest=manifest,
        base_brotli=blocks[1],
        code_brotli=blocks[2],
        pose_bytes=blocks[3],
        base_order=order,
        base_quantized=base_quantized,
        code_quantized=code_quantized,
    )


def _pack_lvls1(manifest: bytes, base: bytes, code: bytes, pose: bytes) -> bytes:
    return b"".join(
        (
            MAGIC,
            _U32.pack(len(manifest)),
            manifest,
            _U32.pack(len(base)),
            base,
            _U32.pack(len(code)),
            code,
            _U32.pack(len(pose)),
            pose,
        )
    )


def _clone_zip(info: zipfile.ZipInfo, archive_comment: bytes, member: bytes, level: int | None) -> bytes:
    output = io.BytesIO()
    cloned = copy.copy(info)
    with zipfile.ZipFile(output, mode="w") as archive:  # ZIP_METADATA_ENV_OK: member metadata is CLONED from the source archive's own ZipInfo (copy.copy above), so create_system/external_attr are REPRODUCED from the source, never chosen by this host -- pinning them here would overwrite the lineage this lossless recode exists to preserve
        archive.comment = archive_comment
        archive.writestr(
            cloned,
            member,
            compress_type=info.compress_type,
            compresslevel=level,
        )
    return output.getvalue()


def inspect_source_zip(
    archive: bytes,
    *,
    expected_archive_sha256: str | None = None,
    expected_member_sha256: str | None = None,
    deflate_levels: tuple[int | None, ...] = DEFAULT_DEFLATE_PROFILES,
) -> SourceZipProfile:
    if not isinstance(archive, bytes):
        raise Ep725LosslessXCodecError("source archive must be immutable bytes")
    if expected_archive_sha256 is not None and not hmac.compare_digest(_sha256(archive), expected_archive_sha256):
        raise Ep725LosslessXCodecError("source archive SHA-256 custody mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(archive), mode="r") as reopened:
            infos = reopened.infolist()
            if len(infos) != 1 or infos[0].filename != MEMBER_NAME:
                raise Ep725LosslessXCodecError("source ZIP must contain exactly safe member 0.bin")
            info = copy.copy(infos[0])
            if info.flag_bits != 0 or info.extra or info.comment or reopened.comment:
                raise Ep725LosslessXCodecError("source ZIP metadata escaped the frozen simple profile")
            if info.compress_type != zipfile.ZIP_DEFLATED:
                raise Ep725LosslessXCodecError("source 0.bin must use ZIP_DEFLATED")
            member = reopened.read(info)
            if reopened.testzip() is not None:
                raise Ep725LosslessXCodecError("source ZIP CRC verification failed")
            comment = reopened.comment
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise Ep725LosslessXCodecError("source ZIP reopen failed") from exc
    if expected_member_sha256 is not None and not hmac.compare_digest(_sha256(member), expected_member_sha256):
        raise Ep725LosslessXCodecError("source member SHA-256 custody mismatch")
    levels = tuple(None if level is None else int(level) for level in deflate_levels)
    if not levels or any(level is not None and (level < 1 or level > 9) for level in levels):
        raise Ep725LosslessXCodecError("deflate profiles must be a non-empty subset of default plus levels 1..9")
    reproducing = tuple(level for level in levels if _clone_zip(info, comment, member, level) == archive)
    if not reproducing:
        raise Ep725LosslessXCodecError("searched ZIP profile cannot reproduce exact source archive bytes")
    return SourceZipProfile(
        archive_bytes=archive,
        member_bytes=member,
        info=info,
        archive_comment=comment,
        reproducing_deflate_profiles=reproducing,
    )


def _base_storage_stream(
    quantized: dict[str, np.ndarray], order: tuple[str, ...], selected_indices: tuple[int, ...]
) -> bytes:
    selected = frozenset(selected_indices)
    chunks: list[bytes] = []
    for index, name in enumerate(order):
        array = quantized[name]
        if index in selected:
            if array.ndim != 2 or min(array.shape) <= 1:
                raise Ep725LosslessXCodecError(f"invalid transpose target {name!r}: {array.shape}")
            array = array.T
        chunks.append(np.ascontiguousarray(array, dtype=np.int8).tobytes())
    return b"".join(chunks)


def _state_digest(parsed: ParsedEp725LVLS1) -> str:
    digest = hashlib.sha256(_STATE_DOMAIN)
    for name in parsed.base_order:
        array = np.ascontiguousarray(parsed.base_quantized[name], dtype=np.int8)
        encoded_name = name.encode("utf-8")
        digest.update(struct.pack("<H", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<B", array.ndim))
        digest.update(struct.pack("<" + "I" * array.ndim, *array.shape))
        digest.update(array.tobytes())
    code = np.ascontiguousarray(parsed.code_quantized, dtype=np.int8)
    digest.update(struct.pack("<II", *code.shape))
    digest.update(code.tobytes())
    digest.update(struct.pack("<I", len(parsed.pose_bytes)))
    digest.update(parsed.pose_bytes)
    return digest.hexdigest()


def _selection_key(selection: Ep725XCodecSelection) -> tuple[Any, ...]:
    return (
        len(selection.archive_bytes),
        len(selection.member_bytes),
        1 if selection.transformed else 0,
        selection.transpose_mask,
        selection.code_wire,
        0 if selection.deflate_level is None else selection.deflate_level,
        _sha256(selection.archive_bytes),
    )


def search_ep725_lossless_xcodec(
    source_archive: bytes,
    *,
    expected_archive_sha256: str | None = None,
    expected_member_sha256: str | None = None,
    deflate_levels: tuple[int | None, ...] = DEFAULT_DEFLATE_PROFILES,
) -> Ep725XCodecRecodeResult:
    """Exhaustively minimize exact ZIP bytes and prove full state equality."""
    source = inspect_source_zip(
        source_archive,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        deflate_levels=deflate_levels,
    )
    parsed = parse_ep725_lvls1(source.member_bytes, require_source_form=True)
    candidates = tuple(
        name
        for name in parsed.base_order
        if parsed.base_quantized[name].ndim == 2 and min(parsed.base_quantized[name].shape) > 1
    )
    candidate_indices = tuple(parsed.base_order.index(name) for name in candidates)

    source_level = source.reproducing_deflate_profiles[0]
    best = Ep725XCodecSelection(
        transformed=False,
        transpose_mask=0,
        transpose_indices=(),
        transpose_names=(),
        code_mode=CODE_TRANSFORM_RAW,
        code_wire=0,
        deflate_level=source_level,
        archive_bytes=source.archive_bytes,
        member_bytes=source.member_bytes,
        manifest_bytes=parsed.manifest_bytes,
        base_brotli_bytes=len(parsed.base_brotli),
        code_brotli_bytes=len(parsed.code_brotli),
    )
    transformed_points = 0
    code_streams = {
        mode: brotli.compress(encode_code_quantized(parsed.code_quantized, mode), quality=11)
        for mode in CODE_MODE_TO_WIRE
    }
    for mask in range(1 << len(candidates)):
        selected_indices = tuple(
            source_index for bit, source_index in enumerate(candidate_indices) if (mask >> bit) & 1
        )
        selected_names = tuple(parsed.base_order[index] for index in selected_indices)
        base_stream = _base_storage_stream(parsed.base_quantized, parsed.base_order, selected_indices)
        base_brotli = brotli.compress(base_stream, quality=11)
        for code_mode, code_wire in CODE_MODE_TO_WIRE.items():
            manifest = dict(parsed.manifest)
            manifest["xcodec"] = {"p": list(selected_indices), "c": code_wire}
            manifest_bytes = _canonical_json(manifest)
            member = _pack_lvls1(manifest_bytes, base_brotli, code_streams[code_mode], parsed.pose_bytes)
            for level in deflate_levels:
                archive = _clone_zip(source.info, source.archive_comment, member, level)
                transformed_points += 1
                point = Ep725XCodecSelection(
                    transformed=True,
                    transpose_mask=mask,
                    transpose_indices=selected_indices,
                    transpose_names=selected_names,
                    code_mode=code_mode,
                    code_wire=code_wire,
                    deflate_level=level,
                    archive_bytes=archive,
                    member_bytes=member,
                    manifest_bytes=manifest_bytes,
                    base_brotli_bytes=len(base_brotli),
                    code_brotli_bytes=len(code_streams[code_mode]),
                )
                if _selection_key(point) < _selection_key(best):
                    best = point

    # Rebuild and standard-library reopen the selected exact bytes before semantic proof.
    rebuilt = (
        source.archive_bytes
        if not best.transformed
        else _clone_zip(source.info, source.archive_comment, best.member_bytes, best.deflate_level)
    )
    if rebuilt != best.archive_bytes:
        raise Ep725LosslessXCodecError("selected archive rebuild is not deterministic")
    with zipfile.ZipFile(io.BytesIO(best.archive_bytes), mode="r") as reopened:
        infos = reopened.infolist()
        if len(infos) != 1 or infos[0].filename != MEMBER_NAME or reopened.read(infos[0]) != best.member_bytes:
            raise Ep725LosslessXCodecError("selected archive parse-back changed exact member bytes")
        if reopened.testzip() is not None:
            raise Ep725LosslessXCodecError("selected archive CRC verification failed")

    selected_parsed = parse_ep725_lvls1(best.member_bytes, require_source_form=not best.transformed)
    if _manifest_without_xcodec(parsed.manifest) != _manifest_without_xcodec(selected_parsed.manifest):
        raise Ep725LosslessXCodecError("selected manifest changed fields outside xcodec")
    if parsed.pose_bytes != selected_parsed.pose_bytes:
        raise Ep725LosslessXCodecError("selected pose bytes changed")
    if not all(
        np.array_equal(parsed.base_quantized[name], selected_parsed.base_quantized[name]) for name in parsed.base_order
    ):
        raise Ep725LosslessXCodecError("selected base arrays are not exactly lossless")
    if not np.array_equal(parsed.code_quantized, selected_parsed.code_quantized):
        raise Ep725LosslessXCodecError("selected code array is not exactly lossless")
    source_digest = _state_digest(parsed)
    selected_digest = _state_digest(selected_parsed)
    if not hmac.compare_digest(source_digest, selected_digest):
        raise Ep725LosslessXCodecError("whole quantized-state digest changed")
    return Ep725XCodecRecodeResult(
        source=source,
        source_lvls1=parsed,
        selected=best,
        base_candidate_names=candidates,
        transformed_points_measured=transformed_points,
        decoded_state_sha256=source_digest,
        selected_decoded_state_sha256=selected_digest,
    )


__all__ = [
    "DEFAULT_DEFLATE_PROFILES",
    "SCORE_RATE_DENOMINATOR",
    "Ep725LosslessXCodecError",
    "Ep725XCodecRecodeResult",
    "Ep725XCodecSelection",
    "ParsedEp725LVLS1",
    "SourceZipProfile",
    "inspect_source_zip",
    "parse_ep725_lvls1",
    "search_ep725_lossless_xcodec",
]
