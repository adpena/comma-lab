# SPDX-License-Identifier: MIT
"""Honest C0B-ABI0 source-identity archive and frozen-decoder receiver.

This module is the mechanical contest-ABI control at the start of the original
task-space codec ladder, not composed C0B scientific state.  Its ``archive.zip``
contains exactly two counted members: the original HEVC-in-Matroska source bytes
and a small canonical state header.  The five SegNet class names and seven
scientific role names are declared only as unpopulated identity-control mappings
to that one source member.  They are not represented as independent streams,
do not populate V9/V10 roles, and make no C0B-gate, factorization, score, or
promotion claim.

The receiver is intentionally standalone (stdlib plus the contest runtime's
PyAV/Torch dependencies).  It imports the exact frozen ``upstream/frame_utils.py``
located beside the contest video-name list, verifies its SHA-256, and uses its
``yuv420_to_rgb`` entrypoint.  Decode is crash-resumable through immutable,
write-once stage files and states.  Temporary files are always cleaned; stage
checkpoints are preserved as certified rebuildable materialization evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import struct
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_SCHEMA = "tac.c0b_abi0_identity_archive.v1"
BUILD_MANIFEST_SCHEMA = "tac.c0b_abi0_identity_archive_manifest.v1"
STAGE_STATE_SCHEMA = "tac.c0b_abi0_identity_stage_state.v1"
INFLATE_MANIFEST_SCHEMA = "tac.c0b_abi0_identity_inflate_manifest.v1"
STORAGE_PREFLIGHT_SCHEMA = "tac.c0b_abi0_storage_preflight.v1"
SOURCE_MEMBER = "source/0.mkv"
STATE_MEMBER = "c0b-state.json"
MEMBER_ORDER = (STATE_MEMBER, SOURCE_MEMBER)
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_REGULAR_MODE = 0o100644
MAX_STATE_BYTES = 1 << 20
MAX_SOURCE_BYTES = (1 << 32) - 1
MAX_FRAME_DIMENSION = 4096
MAX_FRAME_COUNT = 100_000
MAX_STAGE_PAIRS = 10_000
CANONICAL_SOURCE_SHA256 = "2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9"
CANONICAL_SOURCE_BYTES = 37_545_489
CANONICAL_SOURCE_FRAME_COUNT = 1_200
CANONICAL_SOURCE_WIDTH = 1_164
CANONICAL_SOURCE_HEIGHT = 874
CANONICAL_FRAME_UTILS_SHA256 = "d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90"
CANONICAL_FRAME_UTILS_BYTES = 9_345
PREFERRED_ARTIFACT_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

CLASS_NAMES = ("road", "lane_markings", "undrivable", "movable", "my_car")
SCIENTIFIC_ROLE_IDS = (
    "topology_worldsheet",
    "bulk_boundary",
    "lane_chart",
    "movable_mycar",
    "cell_value_preimage",
    "pose_transport_frame0",
    "irreducible_quotient",
)

INFLATE_SH_BYTES = b"""#!/bin/sh
set -eu
if [ "$#" -ne 3 ]; then
  echo "usage: inflate.sh <archive_dir> <output_dir> <video_names_file>" >&2
  exit 2
fi
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
exec "$PYTHON_BIN" "$HERE/inflate.py" "$1" "$2" "$3"
"""


class IdentityReceiverError(ValueError):
    """Fail-closed malformed source, archive, state, runtime, or output."""


@dataclass(frozen=True)
class SourceVideoInfo:
    path: Path
    sha256: str
    byte_length: int
    frame_count: int
    pair_count: int
    width: int
    height: int
    pixel_format: str
    codec_name: str
    container_name: str


@dataclass(frozen=True)
class ParsedIdentityArchive:
    archive_path: Path
    archive_sha256: str
    archive_bytes: int
    state_bytes: bytes
    state_sha256: str
    header: Mapping[str, Any]
    source_sha256: str
    source_bytes: int


@dataclass(frozen=True)
class IdentityArchiveBuildResult:
    archive_path: Path
    manifest_path: Path
    archive_sha256: str
    archive_bytes: int
    state_sha256: str
    state_bytes: int
    source_sha256: str
    source_bytes: int
    pair_count: int
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class RuntimeBundleResult:
    inflate_python_path: Path
    inflate_shell_path: Path
    inflate_python_sha256: str
    inflate_shell_sha256: str


@dataclass(frozen=True)
class IdentityInflateResult:
    completed: bool
    raw_path: Path | None
    raw_sha256: str | None
    raw_bytes: int
    stages_preserved: int
    stage_count: int
    source_sha256: str
    state_sha256: str
    tree_sha256: str | None
    storage_preflight: Mapping[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the only accepted serialized spelling for identity metadata."""

    def require_string_keys(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise IdentityReceiverError(f"canonical JSON key at {path} must be a string")
                require_string_keys(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                require_string_keys(child, f"{path}[{index}]")

    require_string_keys(value, "root")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise IdentityReceiverError("value is not canonical-JSON encodable") from exc


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise IdentityReceiverError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def decode_canonical_json(payload: bytes) -> Any:
    if not isinstance(payload, bytes):
        raise IdentityReceiverError("serialized state must be exact bytes")
    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityReceiverError("state is not strict ASCII JSON") from exc
    if canonical_json_bytes(value) != payload:
        raise IdentityReceiverError("state JSON is not canonical")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
    except OSError as exc:
        raise IdentityReceiverError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise IdentityReceiverError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise IdentityReceiverError(
            f"{label} fields differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise IdentityReceiverError(f"{label} must be a non-empty trimmed string")
    return value


def _exact_int(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IdentityReceiverError(f"{label} must be an exact integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise IdentityReceiverError(f"{label} is outside its admitted bounds")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or text != text.lower() or any(char not in "0123456789abcdef" for char in text):
        raise IdentityReceiverError(f"{label} must be a lowercase SHA-256")
    return text


def _checked_product(values: Sequence[int], label: str, *, maximum: int = (1 << 63) - 1) -> int:
    result = 1
    for value in values:
        result *= _exact_int(value, label, minimum=1)
        if result > maximum:
            raise IdentityReceiverError(f"{label} exceeds its admitted bound")
    return result


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise IdentityReceiverError("cannot resolve a storage-preflight filesystem")
        candidate = candidate.parent
    return candidate


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    root_resolved = root.resolve()
    return resolved == root_resolved or root_resolved in resolved.parents


def _storage_tier(path: Path) -> str:
    if _is_within(path, PREFERRED_ARTIFACT_ROOTS[0]):
        return "vertigo-data-tier"
    if _is_within(path, PREFERRED_ARTIFACT_ROOTS[1]):
        return "ap-data-store"
    return "local"


def storage_preflight(
    output_root: Path | str,
    required_bytes: int,
    *,
    allow_local_spill: bool = False,
    contest_output: bool = False,
) -> Mapping[str, Any]:
    """Check the SSD waterfall and capacity before creating materialized bytes."""

    required = _exact_int(required_bytes, "required_bytes", minimum=0)
    root = Path(output_root)
    tier = "contest-output" if contest_output else _storage_tier(root)
    if tier == "local" and not allow_local_spill:
        raise IdentityReceiverError(
            "local artifact output refused; use an SSD tier or explicitly allow local spill"
        )
    filesystem_root = _nearest_existing_parent(root)
    try:
        free = int(shutil.disk_usage(filesystem_root).free)
    except OSError as exc:
        raise IdentityReceiverError("storage preflight could not read filesystem capacity") from exc
    if free < required:
        raise IdentityReceiverError(
            f"storage preflight refused: need {required} bytes, only {free} free"
        )
    return {
        "schema": STORAGE_PREFLIGHT_SCHEMA,
        "tier": tier,
        "required_bytes": required,
        "free_bytes_at_check": free,
        "passed": True,
    }


def select_artifact_root(
    *,
    required_bytes: int,
    explicit_root: Path | str | None = None,
    allow_local_spill: bool = False,
) -> Path:
    """Choose the first available SSD tier, or an explicitly admitted root."""

    if explicit_root is not None:
        selected = Path(explicit_root)
        storage_preflight(selected, required_bytes, allow_local_spill=allow_local_spill)
        return selected
    for candidate in PREFERRED_ARTIFACT_ROOTS:
        if not candidate.is_dir():
            continue
        try:
            storage_preflight(candidate, required_bytes)
        except IdentityReceiverError:
            continue
        return candidate
    raise IdentityReceiverError(
        "no preferred SSD artifact tier has sufficient space; explicit local opt-in is required"
    )


def inspect_hevc_matroska(source_path: Path | str) -> SourceVideoInfo:
    """Decode-count a real HEVC/Matroska source without retaining frame pixels."""

    path = Path(source_path)
    if not path.is_file() or path.is_symlink():
        raise IdentityReceiverError("source video must be one regular, non-symlink file")
    byte_length = path.stat().st_size
    if byte_length <= 0 or byte_length > MAX_SOURCE_BYTES:
        raise IdentityReceiverError("source video byte length is outside the admitted bound")
    try:
        import av  # type: ignore[import-not-found]
    except ImportError as exc:
        raise IdentityReceiverError("PyAV is required to inspect HEVC/Matroska source bytes") from exc
    try:
        with av.open(str(path), mode="r") as container:
            container_names = tuple(str(container.format.name).split(","))
            if "matroska" not in container_names:
                raise IdentityReceiverError("source container must be Matroska")
            video_streams = list(container.streams.video)
            if len(video_streams) != 1:
                raise IdentityReceiverError("source must contain exactly one video stream")
            stream = video_streams[0]
            codec_name = str(stream.codec_context.name)
            if codec_name != "hevc":
                raise IdentityReceiverError("source video stream must use HEVC")
            width = _exact_int(int(stream.codec_context.width), "source width", minimum=2, maximum=MAX_FRAME_DIMENSION)
            height = _exact_int(
                int(stream.codec_context.height), "source height", minimum=2, maximum=MAX_FRAME_DIMENSION
            )
            if width % 2 or height % 2:
                raise IdentityReceiverError("source geometry must be even for YUV420")
            frame_count = 0
            pixel_format: str | None = None
            for frame in container.decode(stream):
                if frame.width != width or frame.height != height:
                    raise IdentityReceiverError("source frame geometry changes within the stream")
                frame_format = str(frame.format.name)
                if frame_format != "yuv420p":
                    raise IdentityReceiverError("source decode must produce yuv420p frames")
                pixel_format = frame_format
                frame_count += 1
                if frame_count > MAX_FRAME_COUNT:
                    raise IdentityReceiverError("source frame count exceeds the admitted bound")
    except IdentityReceiverError:
        raise
    except Exception as exc:
        raise IdentityReceiverError("source HEVC/Matroska decode inspection failed") from exc
    if frame_count == 0 or frame_count % 2:
        raise IdentityReceiverError("source frame count must be positive and even")
    if pixel_format is None:
        raise IdentityReceiverError("source contains no decoded video frames")
    return SourceVideoInfo(
        path=path,
        sha256=sha256_file(path),
        byte_length=byte_length,
        frame_count=frame_count,
        pair_count=frame_count // 2,
        width=width,
        height=height,
        pixel_format=pixel_format,
        codec_name=codec_name,
        container_name="matroska",
    )


def _alias_metadata() -> dict[str, Any]:
    alias = "abi0_identity_control_alias_same_counted_source"
    return {
        "independent_streams": False,
        "role": "mechanical_identity_control",
        "identity_control_only": True,
        "scientific_evidence": False,
        "scientific_state_composed": False,
        "c0b_gate_complete": False,
        "populated_scientific_role_count": 0,
        "alias_contract": (
            "class and role names are unpopulated ABI0 mappings to the same counted source member; "
            "they do not constitute V9/V10 scientific streams or C0B gate completion"
        ),
        "classes": [
            {
                "class_index": index,
                "class_name": name,
                "storage_mode": alias,
                "source_member": SOURCE_MEMBER,
                "incremental_payload_bytes": 0,
                "scientific_stream_claim": False,
            }
            for index, name in enumerate(CLASS_NAMES)
        ],
        "scientific_roles": [
            {
                "role": role,
                "storage_mode": alias,
                "source_member": SOURCE_MEMBER,
                "incremental_payload_bytes": 0,
                "scientific_role_populated": False,
                "scientific_stream_claim": False,
            }
            for role in SCIENTIFIC_ROLE_IDS
        ],
    }


def _authority_metadata(*, fixture_only: bool) -> dict[str, Any]:
    return {
        "fixture_only": fixture_only,
        "role": "mechanical_identity_control",
        "identity_control_only": True,
        "research_only": True,
        "scientific_evidence": False,
        "scientific_state_composed": False,
        "c0b_gate_complete": False,
        "launch_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_state_header(
    source: SourceVideoInfo,
    *,
    source_origin: str,
    frame_utils_sha256: str,
    frame_utils_bytes: int,
    runtime_source_sha256: str,
    stage_pairs: int,
    fixture_only: bool,
) -> bytes:
    """Build the charged canonical state header for one exact source file."""

    origin = _text(source_origin, "source_origin")
    if not fixture_only and origin != "upstream/videos/0.mkv":
        raise IdentityReceiverError("non-fixture identity archive must bind upstream/videos/0.mkv")
    if not fixture_only and (
        source.sha256,
        source.byte_length,
        source.frame_count,
        source.width,
        source.height,
    ) != (
        CANONICAL_SOURCE_SHA256,
        CANONICAL_SOURCE_BYTES,
        CANONICAL_SOURCE_FRAME_COUNT,
        CANONICAL_SOURCE_WIDTH,
        CANONICAL_SOURCE_HEIGHT,
    ):
        raise IdentityReceiverError("non-fixture archive source differs from the frozen upstream video")
    stage_size = _exact_int(stage_pairs, "stage_pairs", minimum=1, maximum=MAX_STAGE_PAIRS)
    if stage_size > source.pair_count:
        raise IdentityReceiverError("stage_pairs cannot exceed the source pair count")
    frame_utils_hash = _sha256(frame_utils_sha256, "frame_utils_sha256")
    if not fixture_only and (
        frame_utils_hash != CANONICAL_FRAME_UTILS_SHA256
        or frame_utils_bytes != CANONICAL_FRAME_UTILS_BYTES
    ):
        raise IdentityReceiverError("non-fixture archive frame_utils differs from the frozen upstream file")
    runtime_hash = _sha256(runtime_source_sha256, "runtime_source_sha256")
    header = {
        "schema": ARCHIVE_SCHEMA,
        "version": 1,
        "representation": "c0b_abi0_mechanical_source_identity_control",
        "source": {
            "member": SOURCE_MEMBER,
            "origin": origin,
            "sha256": source.sha256,
            "byte_length": source.byte_length,
            "container": source.container_name,
            "video_codec": source.codec_name,
            "pixel_format": source.pixel_format,
            "frame_count": source.frame_count,
            "pair_count": source.pair_count,
            "width": source.width,
            "height": source.height,
        },
        "pair_policy": {
            "sequence_length": 2,
            "pair_order": "canonical_contiguous_source_order",
            "frame0": "decoded_source_frame[2*pair_index]",
            "frame1": "decoded_source_frame[2*pair_index+1]",
            "segnet_obligation": "frame1_only",
            "posenet_obligation": "ordered_frame0_and_frame1",
            "remainder_policy": "refuse_non_pair_tail",
        },
        "identity_aliases": _alias_metadata(),
        "decoder": {
            "frame_utils_origin": "upstream/frame_utils.py",
            "frame_utils_sha256": frame_utils_hash,
            "frame_utils_byte_length": _exact_int(
                frame_utils_bytes, "frame_utils_bytes", minimum=1, maximum=MAX_SOURCE_BYTES
            ),
            "entrypoint": "yuv420_to_rgb",
            "output_layout": "uint8_hwc_c_order",
            "runtime_source_sha256": runtime_hash,
            "inflate_sh_sha256": sha256_bytes(INFLATE_SH_BYTES),
        },
        "receiver": {
            "stage_pairs": stage_size,
            "checkpoint_policy": "immutable_write_once_stage_range",
            "resume_policy": "verify_all_existing_stage_hashes_before_progress",
            "final_assembly": "deterministic_stage_order_no_replace",
            "temporary_cleanup": "automatic_on_success_and_failure",
        },
        "outer_codec": {
            "container": "zip",
            "compression": "stored",
            "member_order": list(MEMBER_ORDER),
            "nested_source_codec": "hevc_in_matroska",
        },
        "authority": _authority_metadata(fixture_only=fixture_only),
    }
    payload = canonical_json_bytes(header)
    if len(payload) > MAX_STATE_BYTES:
        raise IdentityReceiverError("charged state header exceeds its admitted bound")
    validate_state_header(header)
    return payload


def validate_state_header(value: Mapping[str, Any]) -> None:
    """Validate every trust-bearing field of an untrusted archive header."""

    _exact_keys(
        value,
        {
            "schema",
            "version",
            "representation",
            "source",
            "pair_policy",
            "identity_aliases",
            "decoder",
            "receiver",
            "outer_codec",
            "authority",
        },
        "C0B-ABI0 state header",
    )
    if value["schema"] != ARCHIVE_SCHEMA or value["version"] != 1:
        raise IdentityReceiverError("C0B-ABI0 state schema/version differs")
    if value["representation"] != "c0b_abi0_mechanical_source_identity_control":
        raise IdentityReceiverError("C0B-ABI0 representation identity differs")

    source = value["source"]
    _exact_keys(
        source,
        {
            "member",
            "origin",
            "sha256",
            "byte_length",
            "container",
            "video_codec",
            "pixel_format",
            "frame_count",
            "pair_count",
            "width",
            "height",
        },
        "source header",
    )
    if source["member"] != SOURCE_MEMBER:
        raise IdentityReceiverError("source member identity differs")
    _text(source["origin"], "source origin")
    _sha256(source["sha256"], "source sha256")
    _exact_int(source["byte_length"], "source byte length", minimum=1, maximum=MAX_SOURCE_BYTES)
    frame_count = _exact_int(source["frame_count"], "source frame count", minimum=2, maximum=MAX_FRAME_COUNT)
    pair_count = _exact_int(source["pair_count"], "source pair count", minimum=1, maximum=MAX_FRAME_COUNT // 2)
    if frame_count != pair_count * 2:
        raise IdentityReceiverError("pair count does not bind exactly two source frames per pair")
    width = _exact_int(source["width"], "source width", minimum=2, maximum=MAX_FRAME_DIMENSION)
    height = _exact_int(source["height"], "source height", minimum=2, maximum=MAX_FRAME_DIMENSION)
    if width % 2 or height % 2:
        raise IdentityReceiverError("source geometry must be even")
    if (source["container"], source["video_codec"], source["pixel_format"]) != (
        "matroska",
        "hevc",
        "yuv420p",
    ):
        raise IdentityReceiverError("nested HEVC/Matroska source contract differs")

    expected_pair_policy = {
        "sequence_length": 2,
        "pair_order": "canonical_contiguous_source_order",
        "frame0": "decoded_source_frame[2*pair_index]",
        "frame1": "decoded_source_frame[2*pair_index+1]",
        "segnet_obligation": "frame1_only",
        "posenet_obligation": "ordered_frame0_and_frame1",
        "remainder_policy": "refuse_non_pair_tail",
    }
    if value["pair_policy"] != expected_pair_policy:
        raise IdentityReceiverError("frame0/frame1 pair policy differs")
    if value["identity_aliases"] != _alias_metadata():
        raise IdentityReceiverError("five-class/seven-role identity alias metadata differs")

    decoder = value["decoder"]
    _exact_keys(
        decoder,
        {
            "frame_utils_origin",
            "frame_utils_sha256",
            "frame_utils_byte_length",
            "entrypoint",
            "output_layout",
            "runtime_source_sha256",
            "inflate_sh_sha256",
        },
        "decoder header",
    )
    if decoder["frame_utils_origin"] != "upstream/frame_utils.py":
        raise IdentityReceiverError("frozen frame-utils origin differs")
    _sha256(decoder["frame_utils_sha256"], "frame-utils sha256")
    _exact_int(
        decoder["frame_utils_byte_length"],
        "frame-utils byte length",
        minimum=1,
        maximum=MAX_SOURCE_BYTES,
    )
    _sha256(decoder["runtime_source_sha256"], "runtime source sha256")
    if decoder["inflate_sh_sha256"] != sha256_bytes(INFLATE_SH_BYTES):
        raise IdentityReceiverError("inflate.sh identity differs")
    if (decoder["entrypoint"], decoder["output_layout"]) != (
        "yuv420_to_rgb",
        "uint8_hwc_c_order",
    ):
        raise IdentityReceiverError("frozen decoder output contract differs")

    receiver = value["receiver"]
    _exact_keys(
        receiver,
        {
            "stage_pairs",
            "checkpoint_policy",
            "resume_policy",
            "final_assembly",
            "temporary_cleanup",
        },
        "receiver header",
    )
    stage_pairs = _exact_int(receiver["stage_pairs"], "stage_pairs", minimum=1, maximum=MAX_STAGE_PAIRS)
    if stage_pairs > pair_count:
        raise IdentityReceiverError("stage_pairs exceeds pair count")
    if receiver != {
        "stage_pairs": stage_pairs,
        "checkpoint_policy": "immutable_write_once_stage_range",
        "resume_policy": "verify_all_existing_stage_hashes_before_progress",
        "final_assembly": "deterministic_stage_order_no_replace",
        "temporary_cleanup": "automatic_on_success_and_failure",
    }:
        raise IdentityReceiverError("receiver checkpoint contract differs")
    if value["outer_codec"] != {
        "container": "zip",
        "compression": "stored",
        "member_order": list(MEMBER_ORDER),
        "nested_source_codec": "hevc_in_matroska",
    }:
        raise IdentityReceiverError("outer ZIP codec contract differs")

    authority = value["authority"]
    _exact_keys(
        authority,
        {
            "fixture_only",
            "role",
            "identity_control_only",
            "research_only",
            "scientific_evidence",
            "scientific_state_composed",
            "c0b_gate_complete",
            "launch_ready",
            "score_claim",
            "promotion_eligible",
            "rank_or_kill_eligible",
            "ready_for_exact_eval_dispatch",
        },
        "authority header",
    )
    if not isinstance(authority["fixture_only"], bool):
        raise IdentityReceiverError("fixture_only must be a boolean")
    if authority != _authority_metadata(fixture_only=authority["fixture_only"]):
        raise IdentityReceiverError("C0B-ABI0 identity control cannot carry authority claims")
    if not authority["fixture_only"] and source["origin"] != "upstream/videos/0.mkv":
        raise IdentityReceiverError("non-fixture source origin differs")
    if not authority["fixture_only"] and (
        source["sha256"],
        source["byte_length"],
        source["frame_count"],
        source["width"],
        source["height"],
        decoder["frame_utils_sha256"],
        decoder["frame_utils_byte_length"],
    ) != (
        CANONICAL_SOURCE_SHA256,
        CANONICAL_SOURCE_BYTES,
        CANONICAL_SOURCE_FRAME_COUNT,
        CANONICAL_SOURCE_WIDTH,
        CANONICAL_SOURCE_HEIGHT,
        CANONICAL_FRAME_UTILS_SHA256,
        CANONICAL_FRAME_UTILS_BYTES,
    ):
        raise IdentityReceiverError("non-fixture frozen source/decoder identity differs")


def _zip_info(name: str, *, mode: int = ZIP_REGULAR_MODE) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = mode << 16
    return info


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temp_no_replace(temporary: Path, destination: Path) -> None:
    """Atomically publish by hard link; an existing path is only accepted if exact."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = temporary.stat().st_size
    expected_sha = sha256_file(temporary)
    try:
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    except FileExistsError:
        if (
            not destination.is_file()
            or destination.is_symlink()
            or destination.stat().st_size != expected_size
            or sha256_file(destination) != expected_sha
        ):
            raise IdentityReceiverError(f"write-once destination differs: {destination}") from None


def _write_once_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        _publish_temp_no_replace(temporary, path)
        observed_mode = stat.S_IMODE(path.stat().st_mode)
        if observed_mode != mode:
            raise IdentityReceiverError(f"write-once destination mode differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _validate_zip_envelope(archive_path: Path) -> None:
    size = archive_path.stat().st_size
    if size < 22:
        raise IdentityReceiverError("archive.zip is truncated")
    with archive_path.open("rb") as handle:
        if handle.read(4) != b"PK\x03\x04":
            raise IdentityReceiverError("archive.zip has a prepended or missing local header")
        tail_size = min(size, 65_557)
        handle.seek(size - tail_size)
        tail = handle.read(tail_size)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or offset + 22 > len(tail):
        raise IdentityReceiverError("archive.zip has no bounded EOCD")
    comment_length = struct.unpack_from("<H", tail, offset + 20)[0]
    if comment_length != 0 or offset + 22 != len(tail):
        raise IdentityReceiverError("archive.zip comments or trailing bytes are forbidden")


def parse_identity_archive(archive_path: Path | str) -> ParsedIdentityArchive:
    """Reopen an archive and verify exact member, header, and source custody."""

    path = Path(archive_path)
    if not path.is_file() or path.is_symlink():
        raise IdentityReceiverError("archive path must be one regular, non-symlink file")
    archive_bytes = path.stat().st_size
    if archive_bytes <= 0 or archive_bytes > MAX_SOURCE_BYTES + MAX_STATE_BYTES + (1 << 20):
        raise IdentityReceiverError("archive byte length is outside the admitted bound")
    _validate_zip_envelope(path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            if archive.comment:
                raise IdentityReceiverError("archive ZIP comment is forbidden")
            infos = archive.infolist()
            if [info.filename for info in infos] != list(MEMBER_ORDER):
                raise IdentityReceiverError("archive must contain exactly the canonical two members in order")
            for info in infos:
                pure = PurePosixPath(info.filename)
                if info.is_dir() or pure.is_absolute() or ".." in pure.parts:
                    raise IdentityReceiverError("archive contains an unsafe member path")
                if (
                    info.compress_type != zipfile.ZIP_STORED
                    or info.compress_size != info.file_size
                    or info.flag_bits != 0
                    or info.extra
                    or info.comment
                ):
                    raise IdentityReceiverError(
                        "archive members must be bare, unencrypted ZIP_STORED bytes"
                    )
                if info.date_time != ZIP_TIMESTAMP or info.create_system != 3:
                    raise IdentityReceiverError("archive member metadata is not deterministic")
                if (info.external_attr >> 16) != ZIP_REGULAR_MODE:
                    raise IdentityReceiverError("archive member mode differs")
            state_info, source_info = infos
            if state_info.file_size > MAX_STATE_BYTES or source_info.file_size > MAX_SOURCE_BYTES:
                raise IdentityReceiverError("archive member size exceeds its bound")
            state_bytes = archive.read(state_info)
            header = decode_canonical_json(state_bytes)
            validate_state_header(header)
            digest = hashlib.sha256()
            source_bytes = 0
            with archive.open(source_info, "r") as handle:
                while chunk := handle.read(1 << 20):
                    digest.update(chunk)
                    source_bytes += len(chunk)
    except IdentityReceiverError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise IdentityReceiverError("archive.zip cannot be reopened exactly") from exc
    source_sha = digest.hexdigest()
    if source_bytes != header["source"]["byte_length"] or source_sha != header["source"]["sha256"]:
        raise IdentityReceiverError("archived source bytes differ from the charged state header")
    return ParsedIdentityArchive(
        archive_path=path,
        archive_sha256=sha256_file(path),
        archive_bytes=archive_bytes,
        state_bytes=state_bytes,
        state_sha256=sha256_bytes(state_bytes),
        header=header,
        source_sha256=source_sha,
        source_bytes=source_bytes,
    )


def _write_archive_temp(target_parent: Path, state_bytes: bytes, source: SourceVideoInfo) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".c0b-abi0-archive.", suffix=".zip.tmp", dir=target_parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=False) as archive:
            archive.writestr(_zip_info(STATE_MEMBER), state_bytes)
            source_info = _zip_info(SOURCE_MEMBER)
            source_info.file_size = source.byte_length
            with source.path.open("rb") as source_handle, archive.open(source_info, "w") as member_handle:
                shutil.copyfileobj(source_handle, member_handle, length=1 << 20)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_identity_archive(
    source_path: Path | str,
    *,
    archive_path: Path | str,
    frame_utils_path: Path | str,
    source_origin: str = "upstream/videos/0.mkv",
    stage_pairs: int = 12,
    fixture_only: bool = False,
    manifest_path: Path | str | None = None,
    runtime_source_path: Path | str | None = None,
    allow_local_spill: bool = False,
) -> IdentityArchiveBuildResult:
    """Build a deterministic identity archive with write-once parseback custody."""

    source = inspect_hevc_matroska(source_path)
    frame_utils = Path(frame_utils_path)
    if not frame_utils.is_file() or frame_utils.is_symlink():
        raise IdentityReceiverError("frame_utils_path must be one regular, non-symlink file")
    runtime_source = Path(__file__) if runtime_source_path is None else Path(runtime_source_path)
    if not runtime_source.is_file() or runtime_source.is_symlink():
        raise IdentityReceiverError("runtime source must be one regular, non-symlink file")
    state_bytes = build_state_header(
        source,
        source_origin=source_origin,
        frame_utils_sha256=sha256_file(frame_utils),
        frame_utils_bytes=frame_utils.stat().st_size,
        runtime_source_sha256=sha256_file(runtime_source),
        stage_pairs=stage_pairs,
        fixture_only=fixture_only,
    )
    target = Path(archive_path)
    manifest_target = (
        Path(manifest_path)
        if manifest_path is not None
        else target.with_name(f"{target.name}.manifest.json")
    )
    required_bytes = source.byte_length + len(state_bytes) + (2 << 20)
    preflight = storage_preflight(
        target.parent,
        required_bytes,
        allow_local_spill=allow_local_spill,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_archive_temp(target.parent, state_bytes, source)
    try:
        parsed_temp = parse_identity_archive(temporary)
        if parsed_temp.source_sha256 != source.sha256 or parsed_temp.source_bytes != source.byte_length:
            raise IdentityReceiverError("temporary archive source parseback differs from the input")
        _publish_temp_no_replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    parsed = parse_identity_archive(target)
    zip_overhead = parsed.archive_bytes - parsed.source_bytes - len(parsed.state_bytes)
    if zip_overhead < 0:
        raise IdentityReceiverError("ZIP byte accounting underflow")
    manifest = {
        "schema": BUILD_MANIFEST_SCHEMA,
        "archive_sha256": parsed.archive_sha256,
        "archive_bytes": parsed.archive_bytes,
        "member_order": list(MEMBER_ORDER),
        "member_count": 2,
        "source_sha256": parsed.source_sha256,
        "source_bytes": parsed.source_bytes,
        "charged_state_sha256": parsed.state_sha256,
        "charged_state_bytes": len(parsed.state_bytes),
        "zip_framing_bytes": zip_overhead,
        "counted_bytes_reconcile": parsed.source_bytes + len(parsed.state_bytes) + zip_overhead,
        "nested_codec": "hevc-in-matroska-inside-zip-stored",
        "source_origin": source_origin,
        "pair_count": source.pair_count,
        "runtime_source_sha256": parsed.header["decoder"]["runtime_source_sha256"],
        "inflate_sh_sha256": sha256_bytes(INFLATE_SH_BYTES),
        "storage_preflight": {
            "schema": STORAGE_PREFLIGHT_SCHEMA,
            "tier": preflight["tier"],
            "required_bytes": required_bytes,
            "passed": True,
        },
        "cleanup": {
            "temporary_files_auto_removed": True,
            "source_bytes_preserved": True,
            "archive_is_write_once": True,
        },
        **_authority_metadata(fixture_only=fixture_only),
    }
    if manifest["counted_bytes_reconcile"] != manifest["archive_bytes"]:
        raise IdentityReceiverError("archive byte accounting does not reconcile")
    _write_once_bytes(manifest_target, canonical_json_bytes(manifest))
    return IdentityArchiveBuildResult(
        archive_path=target,
        manifest_path=manifest_target,
        archive_sha256=parsed.archive_sha256,
        archive_bytes=parsed.archive_bytes,
        state_sha256=parsed.state_sha256,
        state_bytes=len(parsed.state_bytes),
        source_sha256=parsed.source_sha256,
        source_bytes=parsed.source_bytes,
        pair_count=source.pair_count,
        manifest=manifest,
    )


def emit_standalone_runtime(
    submission_dir: Path | str,
    *,
    runtime_source_path: Path | str | None = None,
) -> RuntimeBundleResult:
    """Publish the standalone receiver and contest-compatible shell entrypoint."""

    root = Path(submission_dir)
    source = Path(__file__) if runtime_source_path is None else Path(runtime_source_path)
    try:
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise IdentityReceiverError("cannot read standalone runtime source") from exc
    python_path = root / "inflate.py"
    shell_path = root / "inflate.sh"
    _write_once_bytes(python_path, source_bytes, mode=0o644)
    _write_once_bytes(shell_path, INFLATE_SH_BYTES, mode=0o755)
    return RuntimeBundleResult(
        inflate_python_path=python_path,
        inflate_shell_path=shell_path,
        inflate_python_sha256=sha256_file(python_path),
        inflate_shell_sha256=sha256_file(shell_path),
    )


def _read_single_video_name(video_names_file: Path) -> str:
    try:
        names = [line for line in video_names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise IdentityReceiverError("cannot read video names file") from exc
    if len(names) != 1:
        raise IdentityReceiverError("C0B-ABI0 archive requires exactly one video name")
    if names[0] != names[0].strip():
        raise IdentityReceiverError("video name must be trimmed exactly")
    return names[0]


def _safe_output_path(output_root: Path, video_name: str) -> Path:
    name = _text(video_name, "video name")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or pure.suffix != ".mkv":
        raise IdentityReceiverError("video name must be a safe relative .mkv path")
    candidate = output_root / Path(*pure.parts).with_suffix(".raw")
    root_resolved = output_root.resolve()
    parent_resolved = candidate.parent.resolve()
    if parent_resolved != root_resolved and root_resolved not in parent_resolved.parents:
        raise IdentityReceiverError("resolved output path escapes output root")
    return candidate


def _refuse_symlink_directory_chain(root: Path, target_directory: Path) -> None:
    try:
        relative = target_directory.relative_to(root)
    except ValueError as exc:
        raise IdentityReceiverError("output directory chain escapes output root") from exc
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise IdentityReceiverError("output directory chain contains a symlink")
        if candidate.exists() and not candidate.is_dir():
            raise IdentityReceiverError("output directory chain contains a non-directory")


def _load_extracted_payload(archive_root: Path) -> tuple[bytes, Mapping[str, Any], Path, str]:
    if not archive_root.is_dir() or archive_root.is_symlink():
        raise IdentityReceiverError("archive_dir must be a regular directory")
    observed_files: list[str] = []
    for path in archive_root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise IdentityReceiverError("extracted archive contains a symlink or special file")
        if path.is_file():
            observed_files.append(path.relative_to(archive_root).as_posix())
    if sorted(observed_files) != sorted(MEMBER_ORDER):
        raise IdentityReceiverError("extracted archive must contain exactly the two charged members")
    state_path = archive_root / STATE_MEMBER
    source_path = archive_root / SOURCE_MEMBER
    try:
        state_bytes = state_path.read_bytes()
    except OSError as exc:
        raise IdentityReceiverError("cannot read extracted C0B-ABI0 state") from exc
    if len(state_bytes) > MAX_STATE_BYTES:
        raise IdentityReceiverError("extracted state exceeds its admitted bound")
    header = decode_canonical_json(state_bytes)
    validate_state_header(header)
    if source_path.stat().st_size != header["source"]["byte_length"]:
        raise IdentityReceiverError("extracted source byte length differs")
    source_sha = sha256_file(source_path)
    if source_sha != header["source"]["sha256"]:
        raise IdentityReceiverError("extracted source SHA-256 differs")
    return state_bytes, header, source_path, source_sha


def _load_frozen_frame_utils(video_names_file: Path, header: Mapping[str, Any]) -> Any:
    path = video_names_file.parent / "frame_utils.py"
    if not path.is_file() or path.is_symlink():
        raise IdentityReceiverError("frozen upstream/frame_utils.py is unavailable beside video names")
    decoder = header["decoder"]
    if path.stat().st_size != decoder["frame_utils_byte_length"] or sha256_file(path) != decoder["frame_utils_sha256"]:
        raise IdentityReceiverError("frozen upstream/frame_utils.py identity differs")
    spec = importlib.util.spec_from_file_location("_c0b_abi0_frozen_frame_utils", path)
    if spec is None or spec.loader is None:
        raise IdentityReceiverError("cannot construct frozen frame-utils import")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise IdentityReceiverError("frozen frame-utils import failed") from exc
    converter = getattr(module, "yuv420_to_rgb", None)
    if not callable(converter):
        raise IdentityReceiverError("frozen frame-utils entrypoint is unavailable")
    return module


def _verify_runtime_identity(header: Mapping[str, Any]) -> None:
    runtime_path = Path(__file__)
    if (
        not runtime_path.is_file()
        or runtime_path.is_symlink()
        or sha256_file(runtime_path) != header["decoder"]["runtime_source_sha256"]
    ):
        raise IdentityReceiverError("executing C0B-ABI0 runtime source identity differs")
    shell_path = runtime_path.with_name("inflate.sh")
    if shell_path.exists() and (
        not shell_path.is_file()
        or shell_path.is_symlink()
        or sha256_file(shell_path) != header["decoder"]["inflate_sh_sha256"]
    ):
        raise IdentityReceiverError("executing C0B-ABI0 inflate.sh identity differs")


def _payload_identity(state_sha256: str, source_sha256: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema": "tac.c0b_abi0_identity_payload_key.v1",
                "state_sha256": state_sha256,
                "source_sha256": source_sha256,
            }
        )
    )


def _stage_bounds(
    stage_index: int,
    *,
    pair_count: int,
    stage_pairs: int,
) -> tuple[int, int, int, int]:
    pair_start = stage_index * stage_pairs
    pair_end = min(pair_count, pair_start + stage_pairs)
    return pair_start, pair_end, pair_start * 2, pair_end * 2


def _stage_state(
    *,
    stage_index: int,
    stage_count: int,
    pair_count: int,
    stage_pairs: int,
    state_sha256: str,
    source_sha256: str,
    stage_sha256: str,
    stage_bytes: int,
    frame_utils_sha256: str,
) -> dict[str, Any]:
    pair_start, pair_end, frame_start, frame_end = _stage_bounds(
        stage_index, pair_count=pair_count, stage_pairs=stage_pairs
    )
    return {
        "schema": STAGE_STATE_SCHEMA,
        "payload_identity_sha256": _payload_identity(state_sha256, source_sha256),
        "state_sha256": state_sha256,
        "source_sha256": source_sha256,
        "frame_utils_sha256": frame_utils_sha256,
        "stage_index": stage_index,
        "stage_count": stage_count,
        "stage_pairs": stage_pairs,
        "pair_start": pair_start,
        "pair_end_exclusive": pair_end,
        "frame_start": frame_start,
        "frame_end_exclusive": frame_end,
        "stage_sha256": stage_sha256,
        "stage_bytes": stage_bytes,
        "rebuildable_from": [STATE_MEMBER, SOURCE_MEMBER, "upstream/frame_utils.py"],
        "checkpoint_policy": "preserved_write_once_no_replace",
        "role": "mechanical_identity_control",
        "identity_control_only": True,
        "research_only": True,
        "scientific_evidence": False,
        "scientific_state_composed": False,
        "c0b_gate_complete": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _stage_paths(stage_root: Path, stage_index: int) -> tuple[Path, Path]:
    stem = f"stage-{stage_index:06d}"
    return stage_root / f"{stem}.raw", stage_root / f"{stem}.json"


def _verify_stage(
    stage_root: Path,
    stage_index: int,
    *,
    header: Mapping[str, Any],
    state_sha256: str,
    source_sha256: str,
) -> bool:
    stage_path, state_path = _stage_paths(stage_root, stage_index)
    if state_path.exists() and not stage_path.exists():
        raise IdentityReceiverError("stage state exists without its preserved stage bytes")
    if not stage_path.exists():
        return False
    if stage_path.is_symlink() or not stage_path.is_file():
        raise IdentityReceiverError("preserved stage is not a regular file")
    if not state_path.exists():
        return False  # Recoverable crash point: bytes published before their state.
    if state_path.is_symlink() or not state_path.is_file():
        raise IdentityReceiverError("preserved stage state is not a regular file")
    try:
        state_bytes = state_path.read_bytes()
    except OSError as exc:
        raise IdentityReceiverError("cannot reopen preserved stage state") from exc
    state = decode_canonical_json(state_bytes)
    source = header["source"]
    receiver = header["receiver"]
    pair_count = int(source["pair_count"])
    stage_pairs = int(receiver["stage_pairs"])
    stage_count = (pair_count + stage_pairs - 1) // stage_pairs
    pair_start, pair_end, _, _ = _stage_bounds(stage_index, pair_count=pair_count, stage_pairs=stage_pairs)
    expected_bytes = _checked_product(
        (pair_end - pair_start, 2, int(source["height"]), int(source["width"]), 3),
        "stage byte count",
    )
    if stage_path.stat().st_size != expected_bytes:
        raise IdentityReceiverError("preserved stage byte length drifted")
    stage_sha = sha256_file(stage_path)
    expected_state = _stage_state(
        stage_index=stage_index,
        stage_count=stage_count,
        pair_count=pair_count,
        stage_pairs=stage_pairs,
        state_sha256=state_sha256,
        source_sha256=source_sha256,
        stage_sha256=stage_sha,
        stage_bytes=expected_bytes,
        frame_utils_sha256=header["decoder"]["frame_utils_sha256"],
    )
    if state != expected_state:
        raise IdentityReceiverError("preserved stage state/source/decoder custody drifted")
    return True


def _publish_stage_from_temp(
    temporary: Path,
    *,
    stage_root: Path,
    stage_index: int,
    stage_count: int,
    header: Mapping[str, Any],
    state_sha256: str,
    source_sha256: str,
    stage_sha256: str,
    stage_bytes: int,
) -> None:
    stage_path, state_path = _stage_paths(stage_root, stage_index)
    if temporary.stat().st_size != stage_bytes or sha256_file(temporary) != stage_sha256:
        raise IdentityReceiverError("stage temporary file failed pre-publication hash check")
    _publish_temp_no_replace(temporary, stage_path)
    source = header["source"]
    receiver = header["receiver"]
    state = _stage_state(
        stage_index=stage_index,
        stage_count=stage_count,
        pair_count=int(source["pair_count"]),
        stage_pairs=int(receiver["stage_pairs"]),
        state_sha256=state_sha256,
        source_sha256=source_sha256,
        stage_sha256=stage_sha256,
        stage_bytes=stage_bytes,
        frame_utils_sha256=header["decoder"]["frame_utils_sha256"],
    )
    _write_once_bytes(state_path, canonical_json_bytes(state))
    if not _verify_stage(
        stage_root,
        stage_index,
        header=header,
        state_sha256=state_sha256,
        source_sha256=source_sha256,
    ):
        raise IdentityReceiverError("stage failed immediate parseback")


def _decode_missing_stages(
    source_path: Path,
    *,
    header: Mapping[str, Any],
    frame_utils: Any,
    stage_root: Path,
    missing_stages: set[int],
    state_sha256: str,
    source_sha256: str,
) -> None:
    if not missing_stages:
        return
    try:
        import av  # type: ignore[import-not-found]
    except ImportError as exc:
        raise IdentityReceiverError("PyAV is required by the C0B-ABI0 receiver") from exc
    source = header["source"]
    pair_count = int(source["pair_count"])
    frame_count = int(source["frame_count"])
    width = int(source["width"])
    height = int(source["height"])
    stage_pairs = int(header["receiver"]["stage_pairs"])
    stage_count = (pair_count + stage_pairs - 1) // stage_pairs
    last_required_frame = max(
        _stage_bounds(stage_index, pair_count=pair_count, stage_pairs=stage_pairs)[3]
        for stage_index in missing_stages
    )
    current_stage: int | None = None
    current_handle: Any = None
    current_temp: Path | None = None
    current_digest: Any = None
    current_bytes = 0

    def finish_current() -> None:
        nonlocal current_stage, current_handle, current_temp, current_digest, current_bytes
        if current_stage is None or current_handle is None or current_temp is None or current_digest is None:
            return
        current_handle.flush()
        os.fsync(current_handle.fileno())
        current_handle.close()
        pair_start, pair_end, _, _ = _stage_bounds(
            current_stage, pair_count=pair_count, stage_pairs=stage_pairs
        )
        expected_bytes = _checked_product(
            (pair_end - pair_start, 2, height, width, 3),
            "decoded stage bytes",
        )
        if current_bytes != expected_bytes:
            raise IdentityReceiverError("decoded stage byte length differs")
        _publish_stage_from_temp(
            current_temp,
            stage_root=stage_root,
            stage_index=current_stage,
            stage_count=stage_count,
            header=header,
            state_sha256=state_sha256,
            source_sha256=source_sha256,
            stage_sha256=current_digest.hexdigest(),
            stage_bytes=current_bytes,
        )
        current_temp.unlink(missing_ok=True)
        current_stage = None
        current_handle = None
        current_temp = None
        current_digest = None
        current_bytes = 0

    try:
        decoded_frames = 0
        with av.open(str(source_path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1 or str(streams[0].codec_context.name) != "hevc":
                raise IdentityReceiverError("runtime source stream contract differs")
            for frame_index, frame in enumerate(container.decode(streams[0])):
                if frame_index >= frame_count:
                    raise IdentityReceiverError("runtime source decoded more frames than charged")
                decoded_frames += 1
                stage_index = frame_index // (2 * stage_pairs)
                if stage_index not in missing_stages:
                    continue
                if current_stage != stage_index:
                    finish_current()
                    descriptor, name = tempfile.mkstemp(
                        prefix=f".stage-{stage_index:06d}.", suffix=".tmp", dir=stage_root
                    )
                    current_stage = stage_index
                    current_handle = os.fdopen(descriptor, "wb")
                    current_temp = Path(name)
                    current_digest = hashlib.sha256()
                if frame.width != width or frame.height != height or str(frame.format.name) != "yuv420p":
                    raise IdentityReceiverError("runtime frame geometry/pixel format differs")
                try:
                    rgb = frame_utils.yuv420_to_rgb(frame)
                    if tuple(rgb.shape) != (height, width, 3) or str(rgb.dtype) != "torch.uint8":
                        raise IdentityReceiverError("frozen frame-utils output contract differs")
                    payload = rgb.contiguous().numpy().tobytes(order="C")
                except IdentityReceiverError:
                    raise
                except Exception as exc:
                    raise IdentityReceiverError("frozen yuv420_to_rgb conversion failed") from exc
                current_handle.write(payload)
                current_digest.update(payload)
                current_bytes += len(payload)
                _, _, _, frame_end = _stage_bounds(stage_index, pair_count=pair_count, stage_pairs=stage_pairs)
                if frame_index + 1 == frame_end:
                    finish_current()
                if frame_index + 1 == last_required_frame:
                    break
        finish_current()
        if decoded_frames != last_required_frame:
            raise IdentityReceiverError("runtime source did not reach the required charged frame boundary")
    finally:
        if current_handle is not None and not current_handle.closed:
            current_handle.close()
        if current_temp is not None:
            current_temp.unlink(missing_ok=True)


def _assemble_raw(
    raw_path: Path,
    *,
    stage_root: Path,
    stage_count: int,
    expected_bytes: int,
) -> tuple[str, int]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    observed_bytes = 0
    if raw_path.exists() and (raw_path.is_symlink() or not raw_path.is_file()):
        raise IdentityReceiverError("final raw destination is not a regular file")
    descriptor: int | None = None
    temporary: Path | None = None
    handle: Any = None
    try:
        if not raw_path.exists():
            descriptor, name = tempfile.mkstemp(prefix=f".{raw_path.name}.", suffix=".tmp", dir=raw_path.parent)
            temporary = Path(name)
            handle = os.fdopen(descriptor, "wb")
            descriptor = None
        for stage_index in range(stage_count):
            stage_path, _ = _stage_paths(stage_root, stage_index)
            with stage_path.open("rb") as stage_handle:
                while chunk := stage_handle.read(1 << 20):
                    digest.update(chunk)
                    observed_bytes += len(chunk)
                    if handle is not None:
                        handle.write(chunk)
        if observed_bytes != expected_bytes:
            raise IdentityReceiverError("final raw stage byte accounting differs")
        expected_sha = digest.hexdigest()
        if handle is not None and temporary is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None
            _publish_temp_no_replace(temporary, raw_path)
        if raw_path.stat().st_size != expected_bytes or sha256_file(raw_path) != expected_sha:
            raise IdentityReceiverError("final raw write-once parseback differs")
        return expected_sha, observed_bytes
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if handle is not None and not handle.closed:
            handle.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def tree_sha256(root: Path | str) -> str:
    base = Path(root)
    if not base.is_dir() or base.is_symlink():
        raise IdentityReceiverError("tree hash root must be a regular directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix()
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise IdentityReceiverError("tree hash refuses symlinks and special files")
        if path.is_dir():
            rows.append({"path": relative, "type": "dir"})
        else:
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sha256_bytes(canonical_json_bytes(rows))


def inflate_archive(
    archive_dir: Path | str,
    output_dir: Path | str,
    video_names_file: Path | str,
    *,
    stop_after_stage: int | None = None,
) -> IdentityInflateResult:
    """Inflate extracted C0B-ABI0 bytes through frozen frame_utils with stage resume."""

    archive_root = Path(archive_dir)
    output_root = Path(output_dir)
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        raise IdentityReceiverError("output_dir must be a regular directory")
    output_root.mkdir(parents=True, exist_ok=True)
    video_names = Path(video_names_file)
    video_name = _read_single_video_name(video_names)
    raw_path = _safe_output_path(output_root, video_name)
    if video_name != "0.mkv":
        raise IdentityReceiverError("C0B-ABI0 archive is bound to the frozen 0.mkv video name")
    state_bytes, header, source_path, source_sha = _load_extracted_payload(archive_root)
    state_sha = sha256_bytes(state_bytes)
    _verify_runtime_identity(header)
    frame_utils = _load_frozen_frame_utils(video_names, header)
    source = header["source"]
    pair_count = int(source["pair_count"])
    stage_pairs = int(header["receiver"]["stage_pairs"])
    stage_count = (pair_count + stage_pairs - 1) // stage_pairs
    if stop_after_stage is None:
        target_stage_count = stage_count
    else:
        stop = _exact_int(stop_after_stage, "stop_after_stage", minimum=0, maximum=stage_count - 1)
        target_stage_count = stop + 1
    stage_root = output_root / ".c0b-abi0-identity-receiver" / raw_path.relative_to(output_root).with_suffix("")
    _refuse_symlink_directory_chain(output_root, raw_path.parent)
    _refuse_symlink_directory_chain(output_root, stage_root)

    existing: set[int] = set()
    for stage_index in range(stage_count):
        if _verify_stage(
            stage_root,
            stage_index,
            header=header,
            state_sha256=state_sha,
            source_sha256=source_sha,
        ):
            existing.add(stage_index)
    wanted = set(range(target_stage_count))
    missing = wanted - existing
    frame_bytes = _checked_product(
        (int(source["height"]), int(source["width"]), 3),
        "frame byte count",
    )
    raw_bytes = _checked_product((int(source["frame_count"]), frame_bytes), "raw byte count")
    missing_stage_bytes = 0
    for stage_index in missing:
        pair_start, pair_end, _, _ = _stage_bounds(stage_index, pair_count=pair_count, stage_pairs=stage_pairs)
        missing_stage_bytes += (pair_end - pair_start) * 2 * frame_bytes
    final_bytes = 0 if raw_path.is_file() or target_stage_count < stage_count else raw_bytes
    largest_stage_bytes = min(stage_pairs, pair_count) * 2 * frame_bytes
    required_bytes = missing_stage_bytes + final_bytes + largest_stage_bytes + (1 << 20)
    deterministic_capacity_requirement = raw_bytes * 2 + largest_stage_bytes + (1 << 20)
    preflight = storage_preflight(output_root, required_bytes, contest_output=True)
    stage_root.mkdir(parents=True, exist_ok=True)
    _refuse_symlink_directory_chain(output_root, stage_root)
    _decode_missing_stages(
        source_path,
        header=header,
        frame_utils=frame_utils,
        stage_root=stage_root,
        missing_stages=missing,
        state_sha256=state_sha,
        source_sha256=source_sha,
    )
    existing_after = {
        stage_index
        for stage_index in range(stage_count)
        if _verify_stage(
            stage_root,
            stage_index,
            header=header,
            state_sha256=state_sha,
            source_sha256=source_sha,
        )
    }
    completed = len(existing_after) == stage_count
    if not completed:
        return IdentityInflateResult(
            completed=False,
            raw_path=None,
            raw_sha256=None,
            raw_bytes=raw_bytes,
            stages_preserved=len(existing_after),
            stage_count=stage_count,
            source_sha256=source_sha,
            state_sha256=state_sha,
            tree_sha256=None,
            storage_preflight=preflight,
        )

    raw_sha, observed_raw_bytes = _assemble_raw(
        raw_path,
        stage_root=stage_root,
        stage_count=stage_count,
        expected_bytes=raw_bytes,
    )
    manifest = {
        "schema": INFLATE_MANIFEST_SCHEMA,
        "video_name": video_name,
        "raw_relative_path": raw_path.relative_to(output_root).as_posix(),
        "raw_sha256": raw_sha,
        "raw_bytes": observed_raw_bytes,
        "source_sha256": source_sha,
        "charged_state_sha256": state_sha,
        "payload_identity_sha256": _payload_identity(state_sha, source_sha),
        "pair_count": pair_count,
        "frame_count": int(source["frame_count"]),
        "frame0_policy": header["pair_policy"]["frame0"],
        "frame1_policy": header["pair_policy"]["frame1"],
        "stage_count": stage_count,
        "stages_preserved": stage_count,
        "decoder": {
            "origin": header["decoder"]["frame_utils_origin"],
            "sha256": header["decoder"]["frame_utils_sha256"],
            "entrypoint": header["decoder"]["entrypoint"],
        },
        "storage_preflight": {
            "schema": STORAGE_PREFLIGHT_SCHEMA,
            "tier": "contest-output",
            "required_bytes": deterministic_capacity_requirement,
            "passed": True,
        },
        "cleanup": {
            "temporary_files_auto_removed": True,
            "stage_checkpoints_preserved": True,
            "stage_rebuild_inputs": [STATE_MEMBER, SOURCE_MEMBER, "upstream/frame_utils.py"],
            "stage_deletion_requires_external_certified_move": True,
        },
        **_authority_metadata(fixture_only=bool(header["authority"]["fixture_only"])),
    }
    manifest_path = stage_root / "inflate-manifest.json"
    _write_once_bytes(manifest_path, canonical_json_bytes(manifest))
    return IdentityInflateResult(
        completed=True,
        raw_path=raw_path,
        raw_sha256=raw_sha,
        raw_bytes=observed_raw_bytes,
        stages_preserved=stage_count,
        stage_count=stage_count,
        source_sha256=source_sha,
        state_sha256=state_sha,
        tree_sha256=tree_sha256(output_root),
        storage_preflight=preflight,
    )


def _runtime_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inflate a C0B-ABI0 mechanical identity archive")
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("video_names_file", type=Path)
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        help="Testing/recovery hook: stop after this zero-based preserved stage index.",
    )
    return parser


def runtime_main(argv: Sequence[str] | None = None) -> int:
    args = _runtime_parser().parse_args(argv)
    try:
        result = inflate_archive(
            args.archive_dir,
            args.output_dir,
            args.video_names_file,
            stop_after_stage=args.stop_after_stage,
        )
    except (OSError, IdentityReceiverError) as exc:
        raise SystemExit(f"C0B-ABI0 identity inflate refused: {exc}") from exc
    print(
        json.dumps(
            {
                "completed": result.completed,
                "raw_path": None if result.raw_path is None else str(result.raw_path),
                "raw_sha256": result.raw_sha256,
                "raw_bytes": result.raw_bytes,
                "stages_preserved": result.stages_preserved,
                "stage_count": result.stage_count,
                "source_sha256": result.source_sha256,
                "state_sha256": result.state_sha256,
                "role": "mechanical_identity_control",
                "identity_control_only": True,
                "research_only": True,
                "scientific_evidence": False,
                "scientific_state_composed": False,
                "c0b_gate_complete": False,
                "launch_ready": False,
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ARCHIVE_SCHEMA",
    "BUILD_MANIFEST_SCHEMA",
    "CANONICAL_FRAME_UTILS_BYTES",
    "CANONICAL_FRAME_UTILS_SHA256",
    "CANONICAL_SOURCE_BYTES",
    "CANONICAL_SOURCE_FRAME_COUNT",
    "CANONICAL_SOURCE_HEIGHT",
    "CANONICAL_SOURCE_SHA256",
    "CANONICAL_SOURCE_WIDTH",
    "CLASS_NAMES",
    "INFLATE_MANIFEST_SCHEMA",
    "INFLATE_SH_BYTES",
    "MEMBER_ORDER",
    "PREFERRED_ARTIFACT_ROOTS",
    "SCIENTIFIC_ROLE_IDS",
    "SOURCE_MEMBER",
    "STAGE_STATE_SCHEMA",
    "STATE_MEMBER",
    "IdentityArchiveBuildResult",
    "IdentityInflateResult",
    "IdentityReceiverError",
    "ParsedIdentityArchive",
    "RuntimeBundleResult",
    "SourceVideoInfo",
    "build_identity_archive",
    "build_state_header",
    "canonical_json_bytes",
    "decode_canonical_json",
    "emit_standalone_runtime",
    "inflate_archive",
    "inspect_hevc_matroska",
    "parse_identity_archive",
    "runtime_main",
    "select_artifact_root",
    "sha256_bytes",
    "sha256_file",
    "storage_preflight",
    "tree_sha256",
    "validate_state_header",
]


if __name__ == "__main__":
    raise SystemExit(runtime_main())
