# SPDX-License-Identifier: MIT
"""Full-resolution, chunked measurement ladder for Task #603 DDM.

This local-only apparatus fits a counted per-chart/per-stratum description to
the exact C1 target planes.  It never calls a scorer and its RGB/Pose integer
diagnostics are not ``d_seg``, ``d_pose``, or a contest score.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import struct
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.optimization.direct_description_minimizer import (
    POINTER_SCORE_TEXT,
    SEED,
    DirectDescriptionError,
    _publish_new_bytes,
    _read_regular_file_once,
    _require_sha256,
    _sha256,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_real_target_rung0 import (
    DirectDescriptionTargetPlaneReceiptV1,
    _committed_source_custody,
    _load_pose_source,
    _pose6_ordinal_codes,
)

LADDER_SCHEMA: Final = "direct_description_measurement_ladder_rungs123.v1"
CHECKPOINT_SCHEMA: Final = "DirectDescriptionMeasurementLadderCheckpointV1"
CONFIG_SCHEMA: Final = "DirectDescriptionMeasurementLadderConfigV1"
PAIR_HW: Final = (384, 512)
CHART_HW: Final = (32, 32)
CHART_GRID: Final = (12, 16)
CHARTS_PER_PLANE: Final = CHART_GRID[0] * CHART_GRID[1]
RUNG1_PAIRS: Final = 64
EVIDENCE_AXIS: Final = "[macOS-CPU full-resolution real-plane apparatus]"

STREAM_ORDER: Final = (
    "global_chart_anchors",
    "axial_chart_gradients",
    "low_variation_chart_residuals",
    "mid_variation_chart_residuals",
    "high_variation_chart_residuals",
    "pose6_pair_codes",
)
MEMBER_BY_STREAM: Final = {name: f"ddm_chart_v3/{index:02d}_{name}.bin" for index, name in enumerate(STREAM_ORDER)}
STREAM_BY_MEMBER: Final = {member: name for name, member in MEMBER_BY_STREAM.items()}
STREAM_MAGIC: Final = {
    "global_chart_anchors": b"D3ANCHR\0",
    "axial_chart_gradients": b"D3GRAD\0\0",
    "low_variation_chart_residuals": b"D3LOW\0\0\0",
    "mid_variation_chart_residuals": b"D3MID\0\0\0",
    "high_variation_chart_residuals": b"D3HIGH\0\0",
    "pose6_pair_codes": b"D3POSE\0\0",
}

_STREAM_FRAME = struct.Struct("<8sHHII")
_ANCHOR_RECORD = struct.Struct("<HB3B")
_GRADIENT_RECORD = struct.Struct("<HB6h")
_RESIDUAL_RECORD = struct.Struct("<HBH3h")
_POSE_RECORD = struct.Struct("<H6B")
_ZIP_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_ZIP_CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
_ZIP_EOCD = struct.Struct("<4s4H2LH")
_RECORD_STRUCT = {
    "global_chart_anchors": _ANCHOR_RECORD,
    "axial_chart_gradients": _GRADIENT_RECORD,
    "low_variation_chart_residuals": _RESIDUAL_RECORD,
    "mid_variation_chart_residuals": _RESIDUAL_RECORD,
    "high_variation_chart_residuals": _RESIDUAL_RECORD,
    "pose6_pair_codes": _POSE_RECORD,
}


class CountedChartStreamV1(BaseModel):
    """One independently framed semantic byte stream in ``A(z)``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    payload: bytes
    codec: Literal["zip_stored_independent_chart_frame"] = "zip_stored_independent_chart_frame"
    ownership: Literal["per_chart_or_per_stratum_not_per_pixel"] = "per_chart_or_per_stratum_not_per_pixel"

    @model_validator(mode="after")
    def _nonempty(self) -> CountedChartStreamV1:
        if not self.payload:
            raise ValueError("every counted chart stream must be nonempty")
        return self


class DirectDescriptionChartZV1(BaseModel):
    """Complete six-stream description for one contiguous pair prefix."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    n_pairs: StrictInt = Field(ge=1, le=600)
    global_chart_anchors: CountedChartStreamV1
    axial_chart_gradients: CountedChartStreamV1
    low_variation_chart_residuals: CountedChartStreamV1
    mid_variation_chart_residuals: CountedChartStreamV1
    high_variation_chart_residuals: CountedChartStreamV1
    pose6_pair_codes: CountedChartStreamV1

    def payload_ledger(self) -> list[dict[str, Any]]:
        return [
            {
                "stream": name,
                "member": MEMBER_BY_STREAM[name],
                "semantic_payload_bytes": len(getattr(self, name).payload),
                "semantic_payload_sha256": _sha256(getattr(self, name).payload),
                "ownership": getattr(self, name).ownership,
            }
            for name in STREAM_ORDER
        ]

    def replace_stream_payload(self, stream_name: str, payload: bytes) -> DirectDescriptionChartZV1:
        if stream_name not in STREAM_ORDER:
            raise DirectDescriptionError(f"unknown chart stream {stream_name!r}")
        stream = getattr(self, stream_name)
        return self.model_copy(update={stream_name: stream.model_copy(update={"payload": payload})})


@dataclass(frozen=True, slots=True)
class ChartArchiveBuildResultV1:
    archive: bytes
    framed_members: Mapping[str, bytes]
    z: DirectDescriptionChartZV1

    def custody(self) -> dict[str, Any]:
        homes = _zip_unique_home_ledger(self.archive)
        return {
            "schema": "direct_description_chart_archive_build.v1",
            "compiler": "ddm_chart_stratum_zip_stored.v1",
            "archive_bytes": len(self.archive),
            "archive_sha256": _sha256(self.archive),
            "member_count": len(self.framed_members),
            "member_order": [MEMBER_BY_STREAM[name] for name in STREAM_ORDER],
            "stream_ledger": self.z.payload_ledger(),
            "unique_final_zip_homes": homes,
            "unique_home_coverage_bytes": sum(row["home_bytes"] for row in homes),
            "all_archive_bytes_have_one_home": sum(row["home_bytes"] for row in homes) == len(self.archive),
            "receiver_consumption_verified": False,
        }


def _expected_record_count(stream_name: str, n_pairs: int) -> int:
    if stream_name in {"global_chart_anchors", "axial_chart_gradients"}:
        return n_pairs * 2
    if stream_name in {
        "low_variation_chart_residuals",
        "mid_variation_chart_residuals",
        "high_variation_chart_residuals",
    }:
        return n_pairs * 2 * (CHARTS_PER_PLANE // 3)
    if stream_name == "pose6_pair_codes":
        return n_pairs
    raise DirectDescriptionError(f"unknown chart stream {stream_name!r}")


def _frame_stream(stream_name: str, payload: bytes, n_pairs: int) -> bytes:
    if stream_name not in STREAM_ORDER:
        raise DirectDescriptionError(f"unknown chart stream {stream_name!r}")
    expected = _expected_record_count(stream_name, n_pairs) * _RECORD_STRUCT[stream_name].size
    if len(payload) != expected:
        raise DirectDescriptionError(f"{stream_name} payload bytes {len(payload)} != canonical {expected}")
    return (
        _STREAM_FRAME.pack(
            STREAM_MAGIC[stream_name], 1, n_pairs, _expected_record_count(stream_name, n_pairs), len(payload)
        )
        + payload
    )


def _parse_stream(stream_name: str, framed: bytes, n_pairs: int | None) -> tuple[int, bytes]:
    if len(framed) < _STREAM_FRAME.size:
        raise DirectDescriptionError(f"{stream_name} frame is truncated")
    magic, version, observed_pairs, record_count, body_bytes = _STREAM_FRAME.unpack_from(framed)
    if magic != STREAM_MAGIC[stream_name] or version != 1 or not 1 <= observed_pairs <= 600:
        raise DirectDescriptionError(f"{stream_name} frame identity mismatch")
    if n_pairs is not None and observed_pairs != n_pairs:
        raise DirectDescriptionError("chart streams disagree on pair count")
    body = framed[_STREAM_FRAME.size :]
    if (
        record_count != _expected_record_count(stream_name, observed_pairs)
        or body_bytes != len(body)
        or len(body) != record_count * _RECORD_STRUCT[stream_name].size
    ):
        raise DirectDescriptionError(f"{stream_name} frame length/count mismatch")
    if _frame_stream(stream_name, body, observed_pairs) != framed:
        raise DirectDescriptionError(f"{stream_name} frame is not canonical")
    return observed_pairs, body


def _deterministic_zip(framed_members: Mapping[str, bytes]) -> bytes:
    expected = tuple(MEMBER_BY_STREAM[name] for name in STREAM_ORDER)
    if tuple(framed_members) != expected:
        raise DirectDescriptionError("chart archive members are incomplete or reordered")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.comment = b""
        for member_name in expected:
            info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            info.extra = b""
            info.comment = b""
            archive.writestr(info, framed_members[member_name], compress_type=zipfile.ZIP_STORED)
    return buffer.getvalue()


def compile_chart_archive(z: DirectDescriptionChartZV1) -> ChartArchiveBuildResultV1:
    framed = {MEMBER_BY_STREAM[name]: _frame_stream(name, getattr(z, name).payload, z.n_pairs) for name in STREAM_ORDER}
    return ChartArchiveBuildResultV1(archive=_deterministic_zip(framed), framed_members=framed, z=z)


def parse_chart_archive(archive: bytes | Path) -> ChartArchiveBuildResultV1:
    archive_bytes = _read_regular_file_once(archive) if isinstance(archive, Path) else archive
    if not isinstance(archive_bytes, bytes) or not archive_bytes:
        raise DirectDescriptionError("chart archive must be nonempty exact bytes")
    expected = tuple(MEMBER_BY_STREAM[name] for name in STREAM_ORDER)
    framed: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as handle:
            infos = handle.infolist()
            if tuple(row.filename for row in infos) != expected or handle.comment:
                raise DirectDescriptionError("chart archive order/comment is noncanonical")
            for row in infos:
                if (
                    row.compress_type != zipfile.ZIP_STORED
                    or row.flag_bits != 0
                    or row.date_time != (1980, 1, 1, 0, 0, 0)
                    or row.extra
                    or row.comment
                    or row.compress_size != row.file_size
                ):
                    raise DirectDescriptionError("chart ZIP member framing is noncanonical")
                framed[row.filename] = handle.read(row.filename)
    except DirectDescriptionError:
        raise
    except (
        zipfile.BadZipFile,
        KeyError,
        NotImplementedError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise DirectDescriptionError("chart archive ZIP is malformed") from exc
    values: dict[str, CountedChartStreamV1] = {}
    n_pairs: int | None = None
    for member_name in expected:
        stream_name = STREAM_BY_MEMBER[member_name]
        n_pairs, payload = _parse_stream(stream_name, framed[member_name], n_pairs)
        values[stream_name] = CountedChartStreamV1(payload=payload)
    assert n_pairs is not None
    z = DirectDescriptionChartZV1(n_pairs=n_pairs, **values)
    rebuilt = compile_chart_archive(z)
    if rebuilt.framed_members != framed or rebuilt.archive != archive_bytes:
        raise DirectDescriptionError("chart archive parse/re-encode identity failed")
    return rebuilt


def _zip_unique_home_ledger(archive: bytes) -> list[dict[str, Any]]:
    """Partition every final ZIP byte into one semantic or container home."""

    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as handle:
            infos = handle.infolist()
    except zipfile.BadZipFile as exc:
        raise DirectDescriptionError("cannot derive homes from malformed chart ZIP") from exc
    eocd_offset = archive.rfind(b"PK\x05\x06")
    if eocd_offset < 0 or eocd_offset + _ZIP_EOCD.size != len(archive):
        raise DirectDescriptionError("chart ZIP must end in one comment-free EOCD")
    eocd = _ZIP_EOCD.unpack_from(archive, eocd_offset)
    central_bytes, central_offset = eocd[5], eocd[6]
    if (
        eocd[0] != b"PK\x05\x06"
        or eocd[1:3] != (0, 0)
        or eocd[3] != len(infos)
        or eocd[4] != len(infos)
        or eocd[7] != 0
        or central_offset + central_bytes != eocd_offset
    ):
        raise DirectDescriptionError("chart ZIP central directory custody mismatch")
    rows: list[dict[str, Any]] = []
    coverage: list[tuple[int, int, str]] = []
    central_cursor = central_offset
    for info in infos:
        local = _ZIP_LOCAL_HEADER.unpack_from(archive, info.header_offset)
        if local[0] != b"PK\x03\x04":
            raise DirectDescriptionError("chart ZIP local header signature mismatch")
        payload_start = info.header_offset + _ZIP_LOCAL_HEADER.size + local[-2] + local[-1]
        local_end = payload_start + info.compress_size
        central = _ZIP_CENTRAL_HEADER.unpack_from(archive, central_cursor)
        if central[0] != b"PK\x01\x02":
            raise DirectDescriptionError("chart ZIP central header signature mismatch")
        name_bytes, extra_bytes, comment_bytes = central[10:13]
        central_end = central_cursor + _ZIP_CENTRAL_HEADER.size + name_bytes + extra_bytes + comment_bytes
        local_name = archive[
            info.header_offset + _ZIP_LOCAL_HEADER.size : info.header_offset + _ZIP_LOCAL_HEADER.size + local[-2]
        ].decode("ascii")
        central_name = archive[
            central_cursor + _ZIP_CENTRAL_HEADER.size : central_cursor + _ZIP_CENTRAL_HEADER.size + name_bytes
        ].decode("ascii")
        if local_name != info.filename or central_name != info.filename or info.filename not in STREAM_BY_MEMBER:
            raise DirectDescriptionError("chart ZIP member ownership mismatch")
        owner = STREAM_BY_MEMBER[info.filename]
        ranges = (
            (info.header_offset, local_end, "local_header_frame_and_payload"),
            (central_cursor, central_end, "central_directory_record"),
        )
        coverage.extend((start, end, f"{owner}:{kind}") for start, end, kind in ranges)
        rows.append(
            {
                "owner": owner,
                "member": info.filename,
                "home_ranges": [
                    {"start": start, "end": end, "bytes": end - start, "kind": kind} for start, end, kind in ranges
                ],
                "member_payload_range": {
                    "start": payload_start,
                    "end": local_end,
                    "bytes": local_end - payload_start,
                },
                "home_bytes": sum(end - start for start, end, _kind in ranges),
            }
        )
        central_cursor = central_end
    if central_cursor != eocd_offset:
        raise DirectDescriptionError("chart ZIP central directory has trailing bytes")
    coverage.append((eocd_offset, len(archive), "container_framing:EOCD"))
    rows.append(
        {
            "owner": "container_framing",
            "member": None,
            "home_ranges": [
                {
                    "start": eocd_offset,
                    "end": len(archive),
                    "bytes": len(archive) - eocd_offset,
                    "kind": "end_of_central_directory",
                }
            ],
            "member_payload_range": None,
            "home_bytes": len(archive) - eocd_offset,
        }
    )
    cursor = 0
    for start, end, _label in sorted(coverage):
        if start != cursor or end <= start:
            raise DirectDescriptionError("chart final-ZIP byte homes overlap or leave a gap")
        cursor = end
    if cursor != len(archive):
        raise DirectDescriptionError("chart final-ZIP byte homes do not cover the archive")
    return rows


def load_target_receipt(path: Path, expected_sha256: str) -> DirectDescriptionTargetPlaneReceiptV1:
    payload = _read_regular_file_once(Path(path))
    if _sha256(payload) != _require_sha256(expected_sha256, "target receipt sha256"):
        raise DirectDescriptionError("measurement-ladder target receipt SHA-256 mismatch")
    try:
        receipt = DirectDescriptionTargetPlaneReceiptV1.model_validate_json(payload)
    except ValueError as exc:
        raise DirectDescriptionError("measurement-ladder target receipt schema is invalid") from exc
    canonical = rfc8785_canonicalize(receipt.model_dump(mode="json", by_alias=True)) + b"\n"
    if canonical != payload:
        raise DirectDescriptionError("measurement-ladder target receipt is not canonical JCS plus LF")
    return receipt


def _verified_source_bytes(path: Path, expected_bytes: int, expected_sha256: str) -> bytes:
    payload = _read_regular_file_once(path)
    if len(payload) != expected_bytes or _sha256(payload) != expected_sha256:
        raise DirectDescriptionError(f"target source bytes/hash mismatch: {path}")
    return payload


def iter_target_plane_chunks(
    receipt: DirectDescriptionTargetPlaneReceiptV1,
    n_pairs: int,
) -> Iterator[tuple[tuple[int, ...], np.ndarray]]:
    """Yield exact target planes in bounded SSD-backed chunks."""

    yield from iter_target_plane_window_chunks(receipt, pair_start=0, n_pairs=n_pairs)


def iter_target_plane_window_chunks(
    receipt: DirectDescriptionTargetPlaneReceiptV1,
    *,
    pair_start: int,
    n_pairs: int,
) -> Iterator[tuple[tuple[int, ...], np.ndarray]]:
    """Yield a contiguous target window with local receiver pair IDs.

    The target receipt is globally indexed over 600 pairs, while every counted
    chart archive remains locally indexed from zero.  Keeping that distinction
    explicit prevents a non-prefix event window from being silently paired with
    prefix RGB/Pose targets.
    """

    if (
        isinstance(pair_start, bool)
        or not isinstance(pair_start, int)
        or pair_start < 0
        or isinstance(n_pairs, bool)
        or not isinstance(n_pairs, int)
        or n_pairs < 1
        or pair_start + n_pairs > receipt.pairs
    ):
        raise DirectDescriptionError("target window must be an exact contiguous subset of [0,600)")
    window_stop = pair_start + n_pairs
    observed = 0
    for row in receipt.chunks:
        row_start = row.pair_ids[0]
        row_stop = row.pair_ids[-1] + 1
        overlap_start = max(pair_start, row_start)
        overlap_stop = min(window_stop, row_stop)
        if overlap_start >= overlap_stop:
            if row_start >= window_stop:
                break
            continue
        if observed >= n_pairs:
            break
        manifest_payload = _verified_source_bytes(Path(row.manifest.path), row.manifest.bytes, row.manifest.sha256)
        try:
            manifest = json.loads(manifest_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectDescriptionError("target chunk manifest is malformed") from exc
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("complete") is not True
            or tuple(manifest.get("pair_ids", ())) != row.pair_ids
            or manifest.get("y0_sha256") != row.y0.sha256
            or manifest.get("y1_sha256") != row.y1.sha256
        ):
            raise DirectDescriptionError("target chunk manifest lineage mismatch")
        y0_payload = _verified_source_bytes(Path(row.y0.path), row.y0.bytes, row.y0.sha256)
        y1_payload = _verified_source_bytes(Path(row.y1.path), row.y1.bytes, row.y1.sha256)
        source_lo = overlap_start - row_start
        source_hi = overlap_stop - row_start
        y0 = np.frombuffer(y0_payload, dtype=np.uint8).reshape(12, *PAIR_HW, 3)[source_lo:source_hi]
        y1 = np.frombuffer(y1_payload, dtype=np.uint8).reshape(12, *PAIR_HW, 3)[source_lo:source_hi]
        planes = np.ascontiguousarray(np.stack((y0, y1), axis=1))
        local_ids = tuple(range(observed, observed + len(planes)))
        yield local_ids, planes
        observed += len(planes)
    if observed != n_pairs:
        raise DirectDescriptionError("target receipt did not cover requested contiguous pair window")


def load_pose_target_codes(receipt: DirectDescriptionTargetPlaneReceiptV1) -> np.ndarray:
    source_path = Path(receipt.source_cache.path)
    if not source_path.is_file() or source_path.stat().st_size != receipt.source_cache.bytes:
        raise DirectDescriptionError("Pose6 source cache size/path custody mismatch")
    poses = _load_pose_source(source_path)
    if _sha256(poses.tobytes(order="C")) != receipt.pose6_source_sha256:
        raise DirectDescriptionError("Pose6 source array SHA-256 mismatch")
    return np.ascontiguousarray(_pose6_ordinal_codes(poses))


def _round_div_signed(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise DirectDescriptionError("signed rounding denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def _chart_means_and_ranges(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(plane)
    if value.dtype != np.uint8 or value.shape != (*PAIR_HW, 3):
        raise DirectDescriptionError("chart fitter requires one uint8 [384,512,3] plane")
    cells = value.reshape(CHART_GRID[0], CHART_HW[0], CHART_GRID[1], CHART_HW[1], 3)
    sums = cells.astype(np.uint64).sum(axis=(1, 3), dtype=np.uint64)
    means = ((sums + (CHART_HW[0] * CHART_HW[1]) // 2) // (CHART_HW[0] * CHART_HW[1])).astype(np.int16)
    ranges = np.ptp(cells, axis=(1, 3)).astype(np.uint16).sum(axis=2, dtype=np.uint32)
    return np.ascontiguousarray(means), np.ascontiguousarray(ranges)


def _predict_chart(anchor: np.ndarray, gradients: np.ndarray, chart_y: int, chart_x: int) -> np.ndarray:
    predicted = np.empty(3, dtype=np.int16)
    for channel in range(3):
        row_term = _round_div_signed(int(gradients[0, channel]) * (2 * chart_y - 11), 22)
        column_term = _round_div_signed(int(gradients[1, channel]) * (2 * chart_x - 15), 30)
        predicted[channel] = int(anchor[channel]) + row_term + column_term
    return predicted


def fit_chart_description(
    receipt: DirectDescriptionTargetPlaneReceiptV1,
    pose_codes: np.ndarray,
    n_pairs: int,
    *,
    pair_start: int = 0,
) -> DirectDescriptionChartZV1:
    """Fit deterministic chart means while never retaining a full pair set."""

    if pose_codes.dtype != np.uint8 or pose_codes.shape != (600, 6):
        raise DirectDescriptionError("Pose6 target codes must be uint8 [600,6]")
    bodies = {name: bytearray() for name in STREAM_ORDER}
    observed_pair = 0
    for pair_ids, planes in iter_target_plane_window_chunks(receipt, pair_start=pair_start, n_pairs=n_pairs):
        if pair_ids != tuple(range(observed_pair, observed_pair + len(pair_ids))):
            raise DirectDescriptionError("target fitter requires canonical contiguous pair order")
        for local_pair, pair_id in enumerate(pair_ids):
            source_pair_id = pair_start + pair_id
            for plane_id in range(2):
                means, ranges = _chart_means_and_ranges(planes[local_pair, plane_id])
                anchor = ((means.astype(np.int64).sum(axis=(0, 1)) + CHARTS_PER_PLANE // 2) // CHARTS_PER_PLANE).astype(
                    np.int16
                )
                row_gradient = np.asarray(
                    [
                        _round_div_signed(
                            int(means[-1, :, channel].sum(dtype=np.int64))
                            - int(means[0, :, channel].sum(dtype=np.int64)),
                            CHART_GRID[1],
                        )
                        for channel in range(3)
                    ],
                    dtype=np.int16,
                )
                column_gradient = np.asarray(
                    [
                        _round_div_signed(
                            int(means[:, -1, channel].sum(dtype=np.int64))
                            - int(means[:, 0, channel].sum(dtype=np.int64)),
                            CHART_GRID[0],
                        )
                        for channel in range(3)
                    ],
                    dtype=np.int16,
                )
                gradients = np.stack((row_gradient, column_gradient), axis=0)
                bodies["global_chart_anchors"].extend(
                    _ANCHOR_RECORD.pack(pair_id, plane_id, *(int(value) for value in anchor))
                )
                bodies["axial_chart_gradients"].extend(
                    _GRADIENT_RECORD.pack(pair_id, plane_id, *(int(value) for value in gradients.reshape(-1)))
                )
                chart_ids = np.arange(CHARTS_PER_PLANE, dtype=np.int64)
                variance_order = np.lexsort((chart_ids, ranges.reshape(-1)))
                strata = (
                    ("low_variation_chart_residuals", variance_order[:64]),
                    ("mid_variation_chart_residuals", variance_order[64:128]),
                    ("high_variation_chart_residuals", variance_order[128:]),
                )
                for stream_name, stratum_chart_ids in strata:
                    for chart_id in sorted(int(value) for value in stratum_chart_ids):
                        chart_y, chart_x = divmod(chart_id, CHART_GRID[1])
                        residual = means[chart_y, chart_x] - _predict_chart(anchor, gradients, chart_y, chart_x)
                        bodies[stream_name].extend(
                            _RESIDUAL_RECORD.pack(
                                pair_id,
                                plane_id,
                                chart_id,
                                *(int(value) for value in residual),
                            )
                        )
            bodies["pose6_pair_codes"].extend(
                _POSE_RECORD.pack(pair_id, *(int(value) for value in pose_codes[source_pair_id]))
            )
            observed_pair += 1
    if observed_pair != n_pairs:
        raise DirectDescriptionError("chart fitter pair coverage mismatch")
    return DirectDescriptionChartZV1(
        n_pairs=n_pairs,
        **{name: CountedChartStreamV1(payload=bytes(bodies[name])) for name in STREAM_ORDER},
    )


@dataclass(frozen=True, slots=True)
class ChartReceiverResultV1:
    archive: bytes
    z: DirectDescriptionChartZV1
    anchors: np.ndarray
    gradients: np.ndarray
    residuals: np.ndarray
    pose6_codes: np.ndarray
    custody: Mapping[str, Any]

    def render_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        indexes = np.asarray(tuple(pair_ids), dtype=np.int64)
        if indexes.ndim != 1 or indexes.size == 0 or indexes.min() < 0 or indexes.max() >= self.z.n_pairs:
            raise DirectDescriptionError("receiver pair ids are outside the described prefix")
        charts = np.empty((len(indexes), 2, *CHART_GRID, 3), dtype=np.int16)
        for local_index, pair_id in enumerate(indexes):
            for plane_id in range(2):
                for chart_y in range(CHART_GRID[0]):
                    for chart_x in range(CHART_GRID[1]):
                        charts[local_index, plane_id, chart_y, chart_x] = (
                            _predict_chart(
                                self.anchors[pair_id, plane_id],
                                self.gradients[pair_id, plane_id],
                                chart_y,
                                chart_x,
                            )
                            + self.residuals[pair_id, plane_id, chart_y, chart_x]
                        )
        if np.any((charts < 0) | (charts > 255)):
            raise DirectDescriptionError("chart receiver left the uint8 range")
        output = np.repeat(np.repeat(charts.astype(np.uint8), CHART_HW[0], axis=2), CHART_HW[1], axis=3)
        if output.shape != (len(indexes), 2, *PAIR_HW, 3):
            raise DirectDescriptionError("chart receiver full-resolution geometry mismatch")
        return np.ascontiguousarray(output)


def receive_chart_archive(archive: bytes | Path) -> ChartReceiverResultV1:
    parsed = parse_chart_archive(archive)
    n_pairs = parsed.z.n_pairs
    anchors = np.empty((n_pairs, 2, 3), dtype=np.int16)
    gradients = np.empty((n_pairs, 2, 2, 3), dtype=np.int16)
    residuals = np.empty((n_pairs, 2, *CHART_GRID, 3), dtype=np.int16)
    pose = np.empty((n_pairs, 6), dtype=np.uint8)
    anchor_seen = np.zeros((n_pairs, 2), dtype=np.bool_)
    gradient_seen = np.zeros((n_pairs, 2), dtype=np.bool_)
    residual_seen = np.zeros((n_pairs, 2, *CHART_GRID), dtype=np.bool_)
    pose_seen = np.zeros(n_pairs, dtype=np.bool_)
    for offset in range(0, len(parsed.z.global_chart_anchors.payload), _ANCHOR_RECORD.size):
        pair_id, plane_id, *rgb = _ANCHOR_RECORD.unpack_from(parsed.z.global_chart_anchors.payload, offset)
        if pair_id >= n_pairs or plane_id >= 2 or anchor_seen[pair_id, plane_id]:
            raise DirectDescriptionError("anchor stream coverage is noncanonical")
        anchors[pair_id, plane_id] = rgb
        anchor_seen[pair_id, plane_id] = True
    for offset in range(0, len(parsed.z.axial_chart_gradients.payload), _GRADIENT_RECORD.size):
        pair_id, plane_id, *values = _GRADIENT_RECORD.unpack_from(parsed.z.axial_chart_gradients.payload, offset)
        if pair_id >= n_pairs or plane_id >= 2 or gradient_seen[pair_id, plane_id]:
            raise DirectDescriptionError("gradient stream coverage is noncanonical")
        gradients[pair_id, plane_id] = np.asarray(values, dtype=np.int16).reshape(2, 3)
        gradient_seen[pair_id, plane_id] = True
    for stream_name in STREAM_ORDER[2:5]:
        payload = getattr(parsed.z, stream_name).payload
        per_pair_plane = np.zeros((n_pairs, 2), dtype=np.uint16)
        for offset in range(0, len(payload), _RESIDUAL_RECORD.size):
            pair_id, plane_id, chart_id, *values = _RESIDUAL_RECORD.unpack_from(payload, offset)
            chart_y, chart_x = divmod(chart_id, CHART_GRID[1])
            if (
                pair_id >= n_pairs
                or plane_id >= 2
                or chart_id >= CHARTS_PER_PLANE
                or residual_seen[pair_id, plane_id, chart_y, chart_x]
            ):
                raise DirectDescriptionError(f"{stream_name} coverage is noncanonical")
            residuals[pair_id, plane_id, chart_y, chart_x] = values
            residual_seen[pair_id, plane_id, chart_y, chart_x] = True
            per_pair_plane[pair_id, plane_id] += 1
        if not np.all(per_pair_plane == CHARTS_PER_PLANE // 3):
            raise DirectDescriptionError(f"{stream_name} must own exactly one variance tertile")
    for offset in range(0, len(parsed.z.pose6_pair_codes.payload), _POSE_RECORD.size):
        pair_id, *values = _POSE_RECORD.unpack_from(parsed.z.pose6_pair_codes.payload, offset)
        if pair_id >= n_pairs or pose_seen[pair_id]:
            raise DirectDescriptionError("Pose6 stream coverage is noncanonical")
        pose[pair_id] = values
        pose_seen[pair_id] = True
    if not (anchor_seen.all() and gradient_seen.all() and residual_seen.all() and pose_seen.all()):
        raise DirectDescriptionError("receiver left a semantic record unconsumed")
    custody = {
        **parsed.custody(),
        "schema": "direct_description_full_resolution_chart_receiver.v1",
        "receiver": "numpy_integer_uint8_chart_reference.v1",
        "receiver_domain": "integer_uint8_full_384x512",
        "n_pairs": n_pairs,
        "output_shape_per_pair": [2, *PAIR_HW, 3],
        "output_dtype": "uint8",
        "chart_hw": list(CHART_HW),
        "chart_grid": list(CHART_GRID),
        "semantic_unit": "per_chart_or_per_stratum_not_per_pixel",
        "all_members_consumed_once": True,
        "receiver_consumption_verified": True,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    return ChartReceiverResultV1(parsed.archive, parsed.z, anchors, gradients, residuals, pose, custody)


def _fraction_text(numerator: int, denominator: int) -> str:
    with localcontext() as context:
        context.prec = 40
        return format(Decimal(numerator) / Decimal(denominator), ".12f")


def measure_quantity_bridge(
    receiver: ChartReceiverResultV1,
    receipt: DirectDescriptionTargetPlaneReceiptV1,
    pose_target_codes: np.ndarray,
) -> dict[str, Any]:
    """Measure exact plane agreement without calling or impersonating a scorer."""

    n_pairs = receiver.z.n_pairs
    if pose_target_codes.dtype != np.uint8 or pose_target_codes.shape != (600, 6):
        raise DirectDescriptionError("quantity bridge Pose6 target geometry mismatch")
    total_pixels = n_pairs * 2 * PAIR_HW[0] * PAIR_HW[1]
    total_channels = total_pixels * 3
    exact_pixels = 0
    exact_channels = 0
    rgb_argmax_agree = 0
    plane_l1_debt = 0
    plane_max_abs = 0
    target_digest = hashlib.sha256()
    receiver_digest = hashlib.sha256()
    per_pair: list[dict[str, Any]] = []
    observed_pair = 0
    for pair_ids, target in iter_target_plane_chunks(receipt, n_pairs):
        described = receiver.render_pairs(pair_ids)
        target_digest.update(target.tobytes(order="C"))
        receiver_digest.update(described.tobytes(order="C"))
        delta = np.abs(described.astype(np.int16) - target.astype(np.int16))
        pixel_exact = np.all(delta == 0, axis=-1)
        channel_exact = delta == 0
        described_argmax = np.argmax(described, axis=-1)
        target_argmax = np.argmax(target, axis=-1)
        for local_index, pair_id in enumerate(pair_ids):
            pair_pixels = 2 * PAIR_HW[0] * PAIR_HW[1]
            pair_exact_pixels = int(pixel_exact[local_index].sum(dtype=np.int64))
            pair_argmax_disagreements = int(
                np.count_nonzero(described_argmax[local_index] != target_argmax[local_index])
            )
            pair_pose_debt = int(
                np.abs(
                    receiver.pose6_codes[pair_id].astype(np.int16) - pose_target_codes[pair_id].astype(np.int16)
                ).sum(dtype=np.int64)
            )
            per_pair.append(
                {
                    "pair_id": pair_id,
                    "target_planes_sha256": _sha256(target[local_index].tobytes(order="C")),
                    "described_planes_sha256": _sha256(described[local_index].tobytes(order="C")),
                    "both_planes_byte_exact": bool(np.all(pixel_exact[local_index])),
                    "rgb_pixels_exact": pair_exact_pixels,
                    "rgb_pixels_total": pair_pixels,
                    "rgb_pixel_exact_fraction": _fraction_text(pair_exact_pixels, pair_pixels),
                    "rgb_channel_argmax_disagreements": pair_argmax_disagreements,
                    "plane_integer_l1_debt": int(delta[local_index].sum(dtype=np.int64)),
                    "pose6_integer_l1_debt": pair_pose_debt,
                }
            )
        exact_pixels += int(pixel_exact.sum(dtype=np.int64))
        exact_channels += int(channel_exact.sum(dtype=np.int64))
        rgb_argmax_agree += int(np.count_nonzero(described_argmax == target_argmax))
        plane_l1_debt += int(delta.sum(dtype=np.int64))
        plane_max_abs = max(plane_max_abs, int(delta.max(initial=0)))
        observed_pair += len(pair_ids)
    if observed_pair != n_pairs or [row["pair_id"] for row in per_pair] != list(range(n_pairs)):
        raise DirectDescriptionError("quantity bridge per-pair coverage mismatch")
    pose_delta = np.abs(receiver.pose6_codes.astype(np.int16) - pose_target_codes[:n_pairs].astype(np.int16))
    pose_exact = int(np.count_nonzero(pose_delta == 0))
    pose_total = n_pairs * 6
    argmax_disagreements = total_pixels - rgb_argmax_agree
    return {
        "schema": "direct_description_plane_quantity_bridge.v1",
        "n_pairs": n_pairs,
        "target_prefix_planes_sha256": target_digest.hexdigest(),
        "described_prefix_planes_sha256": receiver_digest.hexdigest(),
        "archive_bytes": len(receiver.archive),
        "archive_sha256": _sha256(receiver.archive),
        "plane_exactness": {
            "rgb_pixels_exact": exact_pixels,
            "rgb_pixels_total": total_pixels,
            "rgb_pixel_exact_fraction": _fraction_text(exact_pixels, total_pixels),
            "channel_values_exact": exact_channels,
            "channel_values_total": total_channels,
            "channel_value_exact_fraction": _fraction_text(exact_channels, total_channels),
            "fully_exact_pairs": sum(1 for row in per_pair if row["both_planes_byte_exact"]),
            "pair_count": n_pairs,
            "plane_integer_l1_debt": plane_l1_debt,
            "plane_max_absolute_delta": plane_max_abs,
        },
        "argmax_relevant_input_delta": {
            "definition": "RGB channel argmax tie-first apparatus on scorer-input planes; not SegNet argmax",
            "rgb_channel_argmax_disagreements": argmax_disagreements,
            "rgb_channel_argmax_total": total_pixels,
            "rgb_channel_argmax_disagreement_fraction": _fraction_text(argmax_disagreements, total_pixels),
            "d_seg_claim": False,
        },
        "pose_debt": {
            "pose6_integer_l1_debt": int(pose_delta.sum(dtype=np.int64)),
            "pose6_coordinates_exact": pose_exact,
            "pose6_coordinates_total": pose_total,
            "pose6_coordinate_exact_fraction": _fraction_text(pose_exact, pose_total),
            "d_pose_claim": False,
        },
        "measured_tuple": {
            "archive_bytes": len(receiver.archive),
            "plane_exactness_rgb_pixel_fraction": _fraction_text(exact_pixels, total_pixels),
            "pose6_integer_l1_debt": int(pose_delta.sum(dtype=np.int64)),
        },
        "per_pair_exact_agreement": per_pair,
        "receiver_full_resolution": True,
        "receiver_output_shape_per_pair": [2, *PAIR_HW, 3],
        "target_projection_used": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }


def prove_sampled_noop_honesty(z: DirectDescriptionChartZV1) -> dict[str, Any]:
    """Sample every stream and ZIP-home class for fail-closed output relevance."""

    baseline = receive_chart_archive(compile_chart_archive(z).archive)
    stream_rows: list[dict[str, Any]] = []
    semantic_samples = 0
    for stream_name in STREAM_ORDER:
        payload = getattr(z, stream_name).payload
        record = _RECORD_STRUCT[stream_name]
        record_count = len(payload) // record.size
        data_start = {
            "global_chart_anchors": 3,
            "axial_chart_gradients": 3,
            "low_variation_chart_residuals": 5,
            "mid_variation_chart_residuals": 5,
            "high_variation_chart_residuals": 5,
            "pose6_pair_codes": 2,
        }[stream_name]
        samples: list[dict[str, Any]] = []
        for record_index in sorted({0, record_count // 2, record_count - 1}):
            pair_id = struct.unpack_from("<H", payload, record_index * record.size)[0]
            baseline_pair = (
                baseline.pose6_codes[pair_id].tobytes()
                if stream_name == "pose6_pair_codes"
                else baseline.render_pairs((pair_id,)).tobytes(order="C")
            )
            changed = False
            chosen: tuple[int, int] | None = None
            for byte_within_record in range(data_start, record.size):
                index = record_index * record.size + byte_within_record
                for mask in (1, 2, 4, 8, 16, 32, 64, 128):
                    mutated_payload = bytearray(payload)
                    mutated_payload[index] ^= mask
                    try:
                        mutated = receive_chart_archive(
                            compile_chart_archive(z.replace_stream_payload(stream_name, bytes(mutated_payload))).archive
                        )
                    except DirectDescriptionError:
                        continue
                    mutated_pair = (
                        mutated.pose6_codes[pair_id].tobytes()
                        if stream_name == "pose6_pair_codes"
                        else mutated.render_pairs((pair_id,)).tobytes(order="C")
                    )
                    if mutated_pair != baseline_pair:
                        changed = True
                        chosen = (index, mask)
                        break
                if changed:
                    break
            if not changed or chosen is None:
                raise DirectDescriptionError(f"NOOP_DETECTOR: no effective sample in {stream_name}")
            samples.append(
                {
                    "record_index": record_index,
                    "pair_id": pair_id,
                    "payload_byte_index": chosen[0],
                    "xor_mask": chosen[1],
                    "changed_receiver_output": True,
                }
            )
            semantic_samples += 1
        stream_rows.append({"stream": stream_name, "sample_count": len(samples), "samples": samples})
    homes = _zip_unique_home_ledger(baseline.archive)
    archive_positions: set[int] = set()
    for row in homes:
        for span in row["home_ranges"]:
            archive_positions.update({span["start"], (span["start"] + span["end"] - 1) // 2, span["end"] - 1})
    refused = 0
    for position in sorted(archive_positions):
        mutated = bytearray(baseline.archive)
        mutated[position] ^= 1
        try:
            receive_chart_archive(bytes(mutated))
        except DirectDescriptionError:
            refused += 1
        else:
            raise DirectDescriptionError(f"NOOP_DETECTOR: sampled archive byte {position} remained canonical")
    return {
        "schema": "direct_description_sampled_noop_honesty.v1",
        "archive_sha256": _sha256(baseline.archive),
        "semantic_samples": semantic_samples,
        "all_six_streams_sampled": len(stream_rows) == len(STREAM_ORDER),
        "all_semantic_samples_changed_receiver_output": True,
        "per_stream": stream_rows,
        "archive_home_samples": len(archive_positions),
        "archive_home_samples_refused": refused,
        "all_archive_home_samples_fail_closed": refused == len(archive_positions),
        "score_claim": False,
    }


class DirectDescriptionMeasurementLadderConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionMeasurementLadderConfigV1"] = Field(
        default=CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_measurement_ladder_rungs123_n256_seed1234"] = "ddm_measurement_ladder_rungs123_n256_seed1234"
    seed: Literal[1234] = SEED
    rung1_pairs: Literal[64] = RUNG1_PAIRS
    rung2_pairs: StrictInt = Field(ge=256, le=600)
    chart_hw: tuple[Literal[32], Literal[32]] = CHART_HW
    target_receipt_path: StrictStr
    target_receipt_sha256: StrictStr
    receiver: Literal["numpy_integer_uint8_chart_reference.v1"] = "numpy_integer_uint8_chart_reference.v1"
    checkpoint_policy: Literal["atomic_preserve_every_stage"] = "atomic_preserve_every_stage"
    evidence_axis: Literal["[macOS-CPU full-resolution real-plane apparatus]"] = EVIDENCE_AXIS
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionMeasurementLadderConfigV1:
        _require_sha256(self.target_receipt_sha256, "target_receipt_sha256")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def dsl_compile_hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(
                {
                    "compile_target": "direct_description_measurement_ladder_rungs123.v1",
                    "typed_config": self.model_dump(mode="json", by_alias=True),
                }
            )
        )


class DirectDescriptionMeasurementLadderProgramV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_path: StrictStr
    output_directory: StrictStr

    def compile_consumer_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/run_direct_description_measurement_ladder.py",
            "--config",
            self.config_path,
            "--output-dir",
            self.output_directory,
            "--execution-allowed",
            "false",
        )


class DirectDescriptionMeasurementLadderCheckpointV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionMeasurementLadderCheckpointV1"] = Field(
        default=CHECKPOINT_SCHEMA, alias="schema", serialization_alias="schema"
    )
    config: dict[str, Any]
    config_sha256: StrictStr
    dsl_compile_hash: StrictStr
    semantic_argv: tuple[StrictStr, ...]
    semantic_argv_sha256: StrictStr
    target_receipt_sha256: StrictStr
    completed_stage_index: StrictInt = Field(ge=0, le=2)
    completed_stage_name: StrictStr
    next_stage_index: StrictInt = Field(ge=1, le=3)
    described_pairs: StrictInt = Field(ge=64, le=600)
    current_archive_b64: StrictStr
    current_archive_sha256: StrictStr
    current_archive_bytes: StrictInt = Field(ge=1)
    quantity_bridge: dict[str, Any]
    stage_history: tuple[dict[str, Any], ...]
    evidence_axis: Literal["[macOS-CPU full-resolution real-plane apparatus]"] = EVIDENCE_AXIS
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionMeasurementLadderCheckpointV1:
        for field in (
            "config_sha256",
            "dsl_compile_hash",
            "semantic_argv_sha256",
            "target_receipt_sha256",
            "current_archive_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        config = DirectDescriptionMeasurementLadderConfigV1.model_validate_json(rfc8785_canonicalize(self.config))
        if config.typed_config_hash() != self.config_sha256 or config.dsl_compile_hash() != self.dsl_compile_hash:
            raise ValueError("measurement-ladder checkpoint config identity mismatch")
        if self.next_stage_index != self.completed_stage_index + 1:
            raise ValueError("measurement-ladder continuation cursor mismatch")
        expected_names = ("rung1_n64_full_resolution", "rung2_pair_scaling", "rung3_quantity_bridge")
        if expected_names[self.completed_stage_index] != self.completed_stage_name:
            raise ValueError("measurement-ladder checkpoint stage name mismatch")
        if _sha256("\0".join(self.semantic_argv).encode()) != self.semantic_argv_sha256:
            raise ValueError("measurement-ladder checkpoint argv hash mismatch")
        try:
            archive = base64.b64decode(self.current_archive_b64, validate=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("measurement-ladder checkpoint archive base64 malformed") from exc
        if (
            base64.b64encode(archive).decode() != self.current_archive_b64
            or len(archive) != self.current_archive_bytes
            or _sha256(archive) != self.current_archive_sha256
        ):
            raise ValueError("measurement-ladder checkpoint archive custody mismatch")
        receiver = receive_chart_archive(archive)
        if receiver.z.n_pairs != self.described_pairs:
            raise ValueError("measurement-ladder checkpoint pair count mismatch")
        if self.quantity_bridge.get("archive_sha256") != self.current_archive_sha256:
            raise ValueError("measurement-ladder checkpoint bridge/archive mismatch")
        if len(self.stage_history) != self.next_stage_index:
            raise ValueError("measurement-ladder checkpoint history cursor mismatch")
        return self

    def to_bytes(self) -> bytes:
        body = self.model_dump(mode="json", by_alias=True)
        canonical = rfc8785_canonicalize(body)
        return rfc8785_canonicalize({"body": body, "body_sha256": _sha256(canonical)})

    @classmethod
    def from_bytes(cls, payload: bytes) -> DirectDescriptionMeasurementLadderCheckpointV1:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectDescriptionError("measurement-ladder checkpoint JSON is malformed") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"body", "body_sha256"}
            or rfc8785_canonicalize(value) != payload
        ):
            raise DirectDescriptionError("measurement-ladder checkpoint envelope is noncanonical")
        if _sha256(rfc8785_canonicalize(value["body"])) != _require_sha256(value["body_sha256"], "body_sha256"):
            raise DirectDescriptionError("measurement-ladder checkpoint body hash mismatch")
        return cls.model_validate_json(rfc8785_canonicalize(value["body"]))

    def filename(self) -> str:
        return f"ddm_measurement_ladder__stage{self.completed_stage_index:03d}_{self.completed_stage_name}.json"

    def write_new(self, directory: Path) -> Path:
        return _publish_new_bytes(Path(directory) / self.filename(), self.to_bytes())


@dataclass(frozen=True, slots=True)
class MeasurementLadderRunResultV1:
    final_receiver: ChartReceiverResultV1
    final_bridge: Mapping[str, Any]
    stage_history: tuple[Mapping[str, Any], ...]
    checkpoint_paths: tuple[Path, ...]
    complete: bool


def load_measurement_ladder_checkpoint(
    path: Path,
    *,
    config: DirectDescriptionMeasurementLadderConfigV1,
    semantic_argv: Sequence[str],
    receipt: DirectDescriptionTargetPlaneReceiptV1,
    pose_codes: np.ndarray,
) -> DirectDescriptionMeasurementLadderCheckpointV1:
    checkpoint = DirectDescriptionMeasurementLadderCheckpointV1.from_bytes(_read_regular_file_once(path))
    if (
        checkpoint.config_sha256 != config.typed_config_hash()
        or checkpoint.dsl_compile_hash != config.dsl_compile_hash()
        or checkpoint.semantic_argv != tuple(semantic_argv)
        or checkpoint.target_receipt_sha256 != config.target_receipt_sha256
    ):
        raise DirectDescriptionError("measurement-ladder resume identity differs from governed run")
    receiver = receive_chart_archive(base64.b64decode(checkpoint.current_archive_b64, validate=True))
    rederived = measure_quantity_bridge(receiver, receipt, pose_codes)
    if rederived != checkpoint.quantity_bridge:
        raise DirectDescriptionError("measurement-ladder checkpoint bridge does not rederive")
    return checkpoint


def run_measurement_ladder_stages(
    config: DirectDescriptionMeasurementLadderConfigV1,
    *,
    checkpoint_directory: Path,
    semantic_argv: Sequence[str],
    receipt: DirectDescriptionTargetPlaneReceiptV1,
    pose_codes: np.ndarray,
    resume_from: Path | None = None,
    stop_after_stage_index: int | None = None,
) -> MeasurementLadderRunResultV1:
    argv = tuple(semantic_argv)
    if not argv:
        raise DirectDescriptionError("measurement ladder requires typed semantic argv")
    history: list[dict[str, Any]] = []
    checkpoint_paths: list[Path] = []
    start_stage = 0
    receiver: ChartReceiverResultV1 | None = None
    bridge: Mapping[str, Any] | None = None
    if resume_from is not None:
        checkpoint = load_measurement_ladder_checkpoint(
            resume_from,
            config=config,
            semantic_argv=argv,
            receipt=receipt,
            pose_codes=pose_codes,
        )
        receiver = receive_chart_archive(base64.b64decode(checkpoint.current_archive_b64, validate=True))
        bridge = checkpoint.quantity_bridge
        history = [dict(row) for row in checkpoint.stage_history]
        start_stage = checkpoint.next_stage_index
    stage_names = ("rung1_n64_full_resolution", "rung2_pair_scaling", "rung3_quantity_bridge")
    stage_roles = (
        "real_full_384x512_targets_no_8x8_projection",
        "chunked_resumable_pair_count_at_least_256",
        "same_artifact_exact_per_pair_plane_quantity_bridge",
    )
    for stage_index in range(start_stage, 3):
        if stage_index == 0:
            z = fit_chart_description(receipt, pose_codes, config.rung1_pairs)
            receiver = receive_chart_archive(compile_chart_archive(z).archive)
        elif stage_index == 1:
            z = fit_chart_description(receipt, pose_codes, config.rung2_pairs)
            receiver = receive_chart_archive(compile_chart_archive(z).archive)
        elif receiver is None or receiver.z.n_pairs != config.rung2_pairs:
            raise DirectDescriptionError("rung 3 requires the exact rung-2 described artifact")
        assert receiver is not None
        bridge = measure_quantity_bridge(receiver, receipt, pose_codes)
        row = {
            "stage_index": stage_index,
            "stage_name": stage_names[stage_index],
            "stage_role": stage_roles[stage_index],
            "described_pairs": receiver.z.n_pairs,
            "archive_bytes": len(receiver.archive),
            "archive_sha256": _sha256(receiver.archive),
            "target_projection_used": False,
            "receiver_full_resolution": True,
            "measured_tuple": bridge["measured_tuple"],
        }
        if stage_index == 2:
            previous = history[-1]
            row["same_artifact_as_rung2"] = (
                previous["archive_sha256"] == row["archive_sha256"]
                and previous["archive_bytes"] == row["archive_bytes"]
            )
            if not row["same_artifact_as_rung2"]:
                raise DirectDescriptionError("rung-3 quantity bridge changed the rung-2 artifact")
        history.append(row)
        checkpoint = DirectDescriptionMeasurementLadderCheckpointV1(
            config=config.model_dump(mode="json", by_alias=True),
            config_sha256=config.typed_config_hash(),
            dsl_compile_hash=config.dsl_compile_hash(),
            semantic_argv=argv,
            semantic_argv_sha256=_sha256("\0".join(argv).encode()),
            target_receipt_sha256=config.target_receipt_sha256,
            completed_stage_index=stage_index,
            completed_stage_name=stage_names[stage_index],
            next_stage_index=stage_index + 1,
            described_pairs=receiver.z.n_pairs,
            current_archive_b64=base64.b64encode(receiver.archive).decode(),
            current_archive_sha256=_sha256(receiver.archive),
            current_archive_bytes=len(receiver.archive),
            quantity_bridge=dict(bridge),
            stage_history=tuple(history),
        )
        checkpoint_paths.append(checkpoint.write_new(checkpoint_directory))
        if stop_after_stage_index is not None and stage_index >= stop_after_stage_index:
            break
    if receiver is None or bridge is None:
        raise DirectDescriptionError("measurement ladder executed no stage")
    return MeasurementLadderRunResultV1(
        final_receiver=receiver,
        final_bridge=bridge,
        stage_history=tuple(history),
        checkpoint_paths=tuple(checkpoint_paths),
        complete=len(history) == 3,
    )


def _checkpoint_hashes(paths: Sequence[Path]) -> list[str]:
    return [_sha256(_read_regular_file_once(path)) for path in paths]


def _storage_preflight(output_directory: Path) -> dict[str, Any]:
    probe = Path(output_directory)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    required = 32 * 1024 * 1024
    if usage.free < required:
        raise DirectDescriptionError("measurement ladder refuses: insufficient local receipt space")
    return {
        "output_tier": str(probe.resolve()),
        "required_free_bytes": required,
        "free_space_gate_satisfied": True,
        "bulk_target_tier": "/Volumes/VertigoDataTier/pact",
        "bulk_target_read_only": True,
        "materializes_full_plane_cache": False,
        "status": "PASS",
    }


def run_measurement_ladder(
    config: DirectDescriptionMeasurementLadderConfigV1,
    *,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    root = Path(output_directory)
    storage = _storage_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    producer = {
        "module": _committed_source_custody("src/tac/optimization/direct_description_measurement_ladder.py"),
        "cli": _committed_source_custody("tools/run_direct_description_measurement_ladder.py"),
    }
    receipt = load_target_receipt(Path(config.target_receipt_path), config.target_receipt_sha256)
    pose_codes = load_pose_target_codes(receipt)
    primary = run_measurement_ladder_stages(
        config,
        checkpoint_directory=root / "primary_stage_receipts",
        semantic_argv=semantic_argv,
        receipt=receipt,
        pose_codes=pose_codes,
    )
    partial = run_measurement_ladder_stages(
        config,
        checkpoint_directory=root / "resume_stage_receipts",
        semantic_argv=semantic_argv,
        receipt=receipt,
        pose_codes=pose_codes,
        stop_after_stage_index=0,
    )
    resumed = run_measurement_ladder_stages(
        config,
        checkpoint_directory=root / "resume_stage_receipts",
        semantic_argv=semantic_argv,
        receipt=receipt,
        pose_codes=pose_codes,
        resume_from=partial.checkpoint_paths[-1],
    )
    if partial.complete or not primary.complete or not resumed.complete:
        raise DirectDescriptionError("measurement ladder did not preserve all stage boundaries")
    if (
        primary.final_receiver.archive != resumed.final_receiver.archive
        or primary.final_bridge != resumed.final_bridge
        or primary.stage_history != resumed.stage_history
    ):
        raise DirectDescriptionError("measurement ladder resume is not bit-identical")
    rebuilt_once = compile_chart_archive(primary.final_receiver.z).archive
    rebuilt_twice = compile_chart_archive(primary.final_receiver.z).archive
    if rebuilt_once != rebuilt_twice or rebuilt_once != primary.final_receiver.archive:
        raise DirectDescriptionError("measurement ladder compiler determinism x2 failed")
    parsed = parse_chart_archive(primary.final_receiver.archive)
    if parsed.archive != primary.final_receiver.archive:
        raise DirectDescriptionError("measurement ladder parse/re-encode identity failed")
    noop = prove_sampled_noop_honesty(primary.final_receiver.z)
    final_archive = _publish_new_bytes(
        root / f"ddm_measurement_ladder_n{config.rung2_pairs}_final.not_a_candidate.zip.receipt-bytes",
        primary.final_receiver.archive,
    )
    rung1 = dict(primary.stage_history[0])
    rung2 = dict(primary.stage_history[1])
    rung3 = dict(primary.stage_history[2])
    result = {
        "schema": LADDER_SCHEMA,
        "task": 603,
        "run_id": config.run_id,
        "seed": config.seed,
        "evidence_axis": EVIDENCE_AXIS,
        "verdict_scope": (
            "local full-resolution RGB/Pose integer apparatus on exact C1 target planes; "
            "not SegNet, PoseNet, d_seg, d_pose, contest score, or promotion evidence"
        ),
        "research_only": True,
        "execution_allowed": False,
        "candidate_archive": False,
        "score_claim": False,
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash(),
        "semantic_argv": list(semantic_argv),
        "producer": producer,
        "target_receipt_path": config.target_receipt_path,
        "target_receipt_sha256": config.target_receipt_sha256,
        "target_source": {
            "source_role": receipt.source_role,
            "full_pairs_available": receipt.pairs,
            "full_resolution_hw": list(receipt.scorer_hw),
            "plane_dtype": receipt.plane_dtype,
            "chunk_count_available": len(receipt.chunks),
            "target_projection_used": False,
            "bulk_read_only_on_ssd": True,
        },
        "rungs": {
            "rung0_predecessor_receipt": (
                ".omx/research/ddm_target_receipt_pose_rung0_603_20260722T010130Z_artifacts/"
                "ddm_real_target_pose_rung0_receipt.json"
            ),
            "rung1": rung1,
            "rung2": rung2,
            "rung3": rung3,
        },
        "quantity_bridge": dict(primary.final_bridge),
        "archive": {
            "path": str(final_archive),
            "bytes": len(primary.final_receiver.archive),
            "sha256": _sha256(primary.final_receiver.archive),
            "candidate_role": "not_a_candidate",
            "semantic_unit": "per_chart_or_per_stratum_not_per_pixel",
            "parse_reencode_identical": True,
            "compiler_determinism_x2": True,
            "custody": dict(primary.final_receiver.custody),
        },
        "sampled_noop_honesty": noop,
        "resume": {
            "resumed_from_stage": 0,
            "terminal_archive_bit_identical": True,
            "terminal_bridge_bit_identical": True,
            "terminal_history_bit_identical": True,
            "all_stage_checkpoints_preserved": True,
            "primary_checkpoint_sha256": _checkpoint_hashes(primary.checkpoint_paths),
            "resume_checkpoint_sha256": _checkpoint_hashes((*partial.checkpoint_paths, *resumed.checkpoint_paths)),
        },
        "blocker_delta": {
            "FOUR_RUNG_CELLS_THEN_POSE_MEASUREMENT_LADDER": "RED_TO_GREEN_MEASURED_APPARATUS_SCOPE",
            "N600_SAME_ARTIFACT_ARCHIVE_CLOSURE": (
                "RED_TO_GREEN_MEASURED_APPARATUS_SCOPE" if config.rung2_pairs == 600 else "REMAINS_RED_N256_ONLY"
            ),
        },
        "storage_preflight": storage,
        "cleanup": {
            "bulk_artifacts_created": False,
            "target_bulk_remains_read_only_on_ssd": True,
            "scratch_policy": "bounded in-memory chunks plus small immutable checkpoints",
        },
    }
    payload = rfc8785_canonicalize(result) + b"\n"
    receipt_path = _publish_new_bytes(root / "ddm_measurement_ladder_rungs123_receipt.json", payload)
    return result, receipt_path


__all__ = [
    "DirectDescriptionChartZV1",
    "DirectDescriptionMeasurementLadderConfigV1",
    "DirectDescriptionMeasurementLadderProgramV1",
    "compile_chart_archive",
    "fit_chart_description",
    "load_measurement_ladder_checkpoint",
    "load_target_receipt",
    "measure_quantity_bridge",
    "parse_chart_archive",
    "prove_sampled_noop_honesty",
    "receive_chart_archive",
    "run_measurement_ladder",
    "run_measurement_ladder_stages",
]
