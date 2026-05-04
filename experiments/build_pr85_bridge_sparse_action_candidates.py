#!/usr/bin/env python3
"""Build PR85 fixed-runtime bridge sparse-action candidates.

This is a build-only local candidate generator. It starts from the expanded
PR85 fixed-runtime bridge archive, preserves ``masks.qma9``, ``renderer.bin``,
and ``optimized_poses.bin``, then emits deterministic variants of
``qpost.bin`` by preserving selected post/motion groups and selected QRM1
randmulti groups.

The emitted archives are byte-closed candidates for later exact eval. This
tool does not load scorers, run inflate, touch dispatch state, claim a score,
or dispatch remote jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import PR85_HEADERLESS_RANDMULTI_SPECS  # noqa: E402


POST_MOTION_BUILDER_PATH = REPO_ROOT / "experiments" / "build_pr85_post_motion_group_policy_candidates.py"
DEFAULT_SOURCE_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_BRIDGE_ARCHIVE = REPO_ROOT / (
    "experiments/results/pr85_fixed_runtime_bridge_candidates_20260504_codex/"
    "expanded_qpost_qrm1_posefp16/archive.zip"
)
DEFAULT_RANDMULTI_POLICY_JSON = REPO_ROOT / (
    "experiments/results/pr85_randmulti_group_waterfill_20260504_codex/candidate_policies.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / (
    "experiments/results/pr85_bridge_sparse_action_candidates_20260504_codex"
)

TOOL = "experiments/build_pr85_bridge_sparse_action_candidates.py"
SCHEMA = "pr85_fixed_runtime_bridge_sparse_action_candidates_v1"
MANIFEST_SCHEMA = "pr85_fixed_runtime_bridge_sparse_action_candidate_v1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_ORDER = ("masks.qma9", "renderer.bin", "optimized_poses.bin", "qpost.bin")
QPOST_MAGIC = b"QPS1"
QPOST_STREAM_ORDER = (
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
    "bias",
    "region",
    "randmulti",
)
POST_GROUPS = (
    "post_stage1",
    "post_stage2",
    "post_stage3",
    "post_stage4",
)
MOTION_GROUPS = (
    "motion_shift",
    "motion_frac",
    "motion_frac2",
    "motion_frac3",
)
POST_MOTION_GROUPS = POST_GROUPS + MOTION_GROUPS
PROTECTED_QPOST_GROUPS = POST_MOTION_GROUPS
EXACT_EVIDENCE_OVERRIDE_REQUIRED_FIELDS = (
    "override_id",
    "evidence_grade",
    "auth_json",
    "archive_sha256",
    "rationale",
    "allowed_preflight_blockers",
)
KNOWN_PREFLIGHT_BLOCKERS = (
    "whole_randmulti_deletion",
    "whole_post_deletion",
    "whole_motion_deletion",
    "whole_post_motion_deletion",
    "protected_qpost_group_deletion",
)
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
KNOWN_PUBLIC_PR85 = {
    "archive_bytes": 236_328,
    "archive_sha256": "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e",
    "score": 0.25806611029397786,
}
KNOWN_PR85_BRIDGE = {
    "archive_bytes": 239_966,
    "archive_sha256": "3c4e1b2d6a2743be17495b533f35d78d15723482e9fcebd6c49c263e19492d97",
}
WHOLE_STREAM_NEGATIVE_CONTEXT = {
    "whole_randmulti_deletion": {
        "removed_stream": "randmulti",
        "bytes_saved_vs_pr85": 16_089,
        "net_score_delta_vs_pr85": 0.024607965186,
        "score_direction": "worse_when_removed",
        "evidence_grade": "A++ exact T4 negative",
        "auth_json": (
            "experiments/results/lightning_batch/"
            "exact_eval_pr85_minus_randmulti_t4_20260504T0002Z/"
            "contest_auth_eval.adjudicated.json"
        ),
    },
    "whole_post_deletion": {
        "removed_stream": "post",
        "bytes_saved_vs_pr85": 1_387,
        "net_score_delta_vs_pr85": 0.051289973556,
        "score_direction": "worse_when_removed",
        "evidence_grade": "A++ exact T4 negative",
        "auth_json": (
            "experiments/results/lightning_batch/"
            "exact_eval_pr85_minus_post_t4_20260504T0002Z/"
            "contest_auth_eval.adjudicated.json"
        ),
    },
    "whole_motion_deletion": {
        "removed_stream": "motion_stack",
        "bytes_saved_vs_pr85": 581,
        "net_score_delta_vs_pr85": 0.105544804133,
        "score_direction": "worse_when_removed",
        "evidence_grade": "A++ exact T4 negative",
        "auth_json": (
            "experiments/results/lightning_batch/"
            "exact_eval_pr85_minus_motion_stack_t4_20260504T0002Z/"
            "contest_auth_eval.adjudicated.json"
        ),
    },
    "whole_post_motion_deletion": {
        "removed_stream": "post_motion",
        "bytes_saved_vs_pr85": 1_968,
        "net_score_delta_vs_pr85": 0.126261457746,
        "score_direction": "worse_when_removed",
        "evidence_grade": "A++ exact T4 negative",
        "auth_json": (
            "experiments/results/lightning_batch/"
            "exact_eval_pr85_minus_post_motion_t4_20260504T0002Z/"
            "contest_auth_eval.adjudicated.json"
        ),
    },
}

ACTION_POLICIES: dict[str, dict[str, Any]] = {
    "bridge_rm_top001_post123_motion": {
        "qpost_policy_id": "preserve_post123_motion",
        "selected_qpost_groups": (
            "post_stage1",
            "post_stage2",
            "post_stage3",
            "motion_shift",
            "motion_frac",
            "motion_frac2",
            "motion_frac3",
        ),
        "randmulti_policy_id": "waterfill_top001",
        "basis": "PR79-style metric-pruned sparse action subset; PR85 randmulti group top-1 and post stage-4 ablation.",
    },
    "bridge_rm_top004_post23_motion": {
        "qpost_policy_id": "preserve_post23_motion",
        "selected_qpost_groups": (
            "post_stage2",
            "post_stage3",
            "motion_shift",
            "motion_frac",
            "motion_frac2",
            "motion_frac3",
        ),
        "randmulti_policy_id": "waterfill_top004",
        "basis": "PR79-style metric-pruned sparse action subset; PR85 randmulti group top-4 and post stage-1/4 ablation.",
    },
    "bridge_rm_top008_motion_only": {
        "qpost_policy_id": "preserve_motion_only",
        "selected_qpost_groups": (
            "motion_shift",
            "motion_frac",
            "motion_frac2",
            "motion_frac3",
        ),
        "randmulti_policy_id": "waterfill_top008",
        "basis": "PR79-style sparse action stress test preserving motion and top-8 randmulti groups only.",
    },
    "bridge_no_randmulti_post123_motion": {
        "qpost_policy_id": "preserve_post123_motion",
        "selected_qpost_groups": (
            "post_stage1",
            "post_stage2",
            "post_stage3",
            "motion_shift",
            "motion_frac",
            "motion_frac2",
            "motion_frac3",
        ),
        "selected_randmulti_group_ids": (),
        "basis": "Bridge-level no-randmulti control with post/motion retained; negative-control candidate only.",
    },
}


class BridgeSparseActionError(ValueError):
    """Raised when a bridge candidate violates the build-only contract."""


@dataclass(frozen=True)
class Qrm1Group:
    group_id: int
    height: int
    width: int
    amplitude: int
    scount: int
    payload: bytes
    raw_payload_bytes: int
    nonzero_choice_count: int


def _load_post_motion_builder() -> Any:
    spec = importlib.util.spec_from_file_location("pr85_bridge_post_motion_builder", POST_MOTION_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load PR85 post/motion builder from {POST_MOTION_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


post_motion = _load_post_motion_builder()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _repo_rel(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _safe_member_name(name: str) -> str:
    path = Path(name)
    if (
        not name
        or name.startswith("/")
        or ".." in path.parts
        or len(path.parts) != 1
        or name.startswith(".")
        or name.startswith("__MACOSX/")
        or name.startswith("._")
    ):
        raise BridgeSparseActionError(f"unsafe ZIP member path: {name!r}")
    return name


def _source_archive_info(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [_safe_member_name(info.filename) for info in infos]
        if names != ["x"]:
            raise BridgeSparseActionError(f"PR85 source archive must contain exactly member 'x'; got {names!r}")
        raw = zf.read(infos[0])
        member = {
            "member_name": infos[0].filename,
            "member_bytes": int(infos[0].file_size),
            "member_compress_size": int(infos[0].compress_size),
            "member_crc32_hex": f"{infos[0].CRC:08x}",
            "member_sha256": _sha256_bytes(raw),
            "zip_compress_type": int(infos[0].compress_type),
        }
    archive_sha = _sha256_file(path)
    return {
        "path": _repo_rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": archive_sha,
        "known_public_pr85_match": {
            "matches": (
                int(path.stat().st_size) == KNOWN_PUBLIC_PR85["archive_bytes"]
                and archive_sha == KNOWN_PUBLIC_PR85["archive_sha256"]
            ),
            "expected_archive_bytes": KNOWN_PUBLIC_PR85["archive_bytes"],
            "expected_archive_sha256": KNOWN_PUBLIC_PR85["archive_sha256"],
            "exact_t4_score_context": KNOWN_PUBLIC_PR85["score"],
        },
        **member,
    }


def _read_bridge_archive(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        seen: set[str] = set()
        names: list[str] = []
        members: dict[str, bytes] = {}
        member_rows: list[dict[str, Any]] = []
        for info in infos:
            name = _safe_member_name(info.filename)
            if name in seen:
                raise BridgeSparseActionError(f"duplicate ZIP member: {name}")
            seen.add(name)
            names.append(name)
            raw = zf.read(info)
            members[name] = raw
            member_rows.append(
                {
                    "name": name,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "crc32_hex": f"{info.CRC:08x}",
                    "sha256": _sha256_bytes(raw),
                    "zip_compress_type": int(info.compress_type),
                }
            )
    if names != list(MEMBER_ORDER):
        raise BridgeSparseActionError(
            f"bridge archive members must be exactly {list(MEMBER_ORDER)!r}; got {names!r}"
        )
    if not members["masks.qma9"].startswith(b"QMA9"):
        raise BridgeSparseActionError("bridge masks.qma9 does not start with QMA9")
    if not members["renderer.bin"].startswith(b"QH0"):
        raise BridgeSparseActionError("bridge renderer.bin does not start with QH0")
    if len(members["optimized_poses.bin"]) != 600 * 6 * 2:
        raise BridgeSparseActionError("bridge optimized_poses.bin is not 600x6 raw fp16")
    _parse_qpost(members["qpost.bin"])
    archive_sha = _sha256_file(path)
    return (
        {
            "path": _repo_rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": archive_sha,
            "known_pr85_bridge_match": {
                "matches": (
                    int(path.stat().st_size) == KNOWN_PR85_BRIDGE["archive_bytes"]
                    and archive_sha == KNOWN_PR85_BRIDGE["archive_sha256"]
                ),
                "expected_archive_bytes": KNOWN_PR85_BRIDGE["archive_bytes"],
                "expected_archive_sha256": KNOWN_PR85_BRIDGE["archive_sha256"],
            },
            "member_count": len(member_rows),
            "members": member_rows,
        },
        members,
    )


def _parse_qpost(raw: bytes) -> tuple[dict[str, bytes], dict[str, Any]]:
    header_size = len(QPOST_MAGIC) + 4 * len(QPOST_STREAM_ORDER)
    if len(raw) < header_size or raw[:4] != QPOST_MAGIC:
        raise BridgeSparseActionError("qpost.bin must start with QPS1 and contain eight stream lengths")
    lengths = struct.unpack_from("<" + "I" * len(QPOST_STREAM_ORDER), raw, len(QPOST_MAGIC))
    pos = header_size
    streams: dict[str, bytes] = {}
    rows: dict[str, dict[str, Any]] = {}
    for name, length in zip(QPOST_STREAM_ORDER, lengths, strict=True):
        end = pos + int(length)
        if end > len(raw):
            raise BridgeSparseActionError(f"qpost stream {name!r} overruns payload")
        stream = raw[pos:end]
        pos = end
        streams[name] = stream
        rows[name] = {
            "bytes": int(len(stream)),
            "sha256": _sha256_bytes(stream),
        }
    if pos != len(raw):
        raise BridgeSparseActionError(f"qpost has {len(raw) - pos} trailing bytes")
    return streams, {
        "magic": QPOST_MAGIC.decode("ascii"),
        "stream_order": list(QPOST_STREAM_ORDER),
        "streams": rows,
        "qpost_bytes": int(len(raw)),
        "qpost_sha256": _sha256_bytes(raw),
    }


def _pack_qpost(streams: Mapping[str, bytes]) -> bytes:
    missing = [name for name in QPOST_STREAM_ORDER if name not in streams]
    extra = sorted(set(streams) - set(QPOST_STREAM_ORDER))
    if missing or extra:
        raise BridgeSparseActionError(f"qpost streams mismatch: missing={missing} extra={extra}")
    lengths = [len(streams[name]) for name in QPOST_STREAM_ORDER]
    out = bytearray(QPOST_MAGIC)
    out += struct.pack("<" + "I" * len(QPOST_STREAM_ORDER), *lengths)
    for name in QPOST_STREAM_ORDER:
        out += bytes(streams[name])
    packed = bytes(out)
    _parse_qpost(packed)
    return packed


def _zip_info(name: str) -> zipfile.ZipInfo:
    if name not in MEMBER_ORDER:
        raise BridgeSparseActionError(f"unexpected bridge member: {name!r}")
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_bridge_archive(path: Path, members: Mapping[str, bytes]) -> list[dict[str, Any]]:
    missing = [name for name in MEMBER_ORDER if name not in members]
    extra = sorted(set(members) - set(MEMBER_ORDER))
    if missing or extra:
        raise BridgeSparseActionError(f"candidate members mismatch: missing={missing} extra={extra}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in MEMBER_ORDER:
            zf.writestr(
                _zip_info(name),
                bytes(members[name]),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    with zipfile.ZipFile(path, "r") as zf:
        rows = []
        names = []
        seen = set()
        for info in zf.infolist():
            name = _safe_member_name(info.filename)
            if name in seen:
                raise BridgeSparseActionError(f"duplicate candidate ZIP member: {name}")
            seen.add(name)
            names.append(name)
            raw = zf.read(info)
            rows.append(
                {
                    "name": name,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "crc32_hex": f"{info.CRC:08x}",
                    "sha256": _sha256_bytes(raw),
                    "zip_compress_type": int(info.compress_type),
                }
            )
        if names != list(MEMBER_ORDER):
            raise BridgeSparseActionError(f"candidate member order changed: {names!r}")
    return rows


def _read_vlq(data: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 128:
            return value, cursor
        shift += 7
        if shift > 63:
            break
    raise BridgeSparseActionError("truncated or overlong QRM1 VLQ stream")


def _consume_sparse_row(raw: bytes, cursor: int) -> tuple[int, int]:
    if cursor >= len(raw):
        raise BridgeSparseActionError("QRM1 row ended before count byte")
    count = int(raw[cursor])
    cursor += 1
    if count == 255:
        if cursor + 2 > len(raw):
            raise BridgeSparseActionError("QRM1 extended row count is truncated")
        count = int.from_bytes(raw[cursor : cursor + 2], "little")
        cursor += 2
    previous = -1
    for _ in range(count):
        gap, cursor = _read_vlq(raw, cursor)
        previous += gap + 1
        if previous < 0 or previous >= 600:
            raise BridgeSparseActionError(f"QRM1 sparse index out of range: {previous}")
    end = cursor + count
    if end > len(raw):
        raise BridgeSparseActionError("QRM1 value stream is truncated")
    return end, count


def _encode_sparse_row(row: bytes) -> bytes:
    if len(row) != 600:
        raise BridgeSparseActionError(f"sparse row must contain 600 choices; got {len(row)}")
    indices = [idx for idx, value in enumerate(row) if value]
    out = bytearray()
    if len(indices) >= 255:
        out.append(255)
        out += len(indices).to_bytes(2, "little")
    else:
        out.append(len(indices))
    last = -1
    for idx in indices:
        gap = idx - last - 1
        while True:
            byte = gap & 0x7F
            gap >>= 7
            if gap:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                break
        last = idx
    out += bytes(row[idx] for idx in indices)
    return bytes(out)


def _parse_qrm1_groups(encoded: bytes) -> tuple[dict[int, Qrm1Group], dict[str, Any]]:
    try:
        raw = brotli.decompress(encoded)
    except brotli.error as exc:
        raise BridgeSparseActionError("randmulti stream is not Brotli-decodable") from exc
    if not raw.startswith(b"QRM1"):
        raise BridgeSparseActionError(f"bridge randmulti stream must be QRM1; got {raw[:4]!r}")
    if len(raw) < 6:
        raise BridgeSparseActionError("QRM1 stream is truncated")
    pos = 4
    group_count = int.from_bytes(raw[pos : pos + 2], "little")
    pos += 2
    groups: dict[int, Qrm1Group] = {}
    for _ in range(group_count):
        if pos + 2 > len(raw):
            raise BridgeSparseActionError("QRM1 group id is truncated")
        group_id = int.from_bytes(raw[pos : pos + 2], "little")
        pos += 2
        if group_id in groups:
            raise BridgeSparseActionError(f"duplicate QRM1 group id: {group_id}")
        if group_id < 0 or group_id >= len(PR85_HEADERLESS_RANDMULTI_SPECS):
            raise BridgeSparseActionError(f"QRM1 group id outside PR85 schedule: {group_id}")
        height, width, amplitude, scount = PR85_HEADERLESS_RANDMULTI_SPECS[group_id]
        payload_start = pos
        nonzero = 0
        for _row_index in range(int(scount)):
            pos, count = _consume_sparse_row(raw, pos)
            nonzero += int(count)
        payload = raw[payload_start:pos]
        groups[group_id] = Qrm1Group(
            group_id=group_id,
            height=int(height),
            width=int(width),
            amplitude=int(amplitude),
            scount=int(scount),
            payload=payload,
            raw_payload_bytes=len(payload),
            nonzero_choice_count=nonzero,
        )
    if pos != len(raw):
        raise BridgeSparseActionError("QRM1 stream has trailing bytes")
    return groups, {
        "raw_bytes": len(raw),
        "raw_sha256": _sha256_bytes(raw),
        "encoded_bytes": len(encoded),
        "encoded_sha256": _sha256_bytes(encoded),
        "group_count": len(groups),
        "group_ids": sorted(groups),
        "nonzero_choice_total": sum(group.nonzero_choice_count for group in groups.values()),
    }


def _brotli_best(raw: bytes) -> tuple[bytes, dict[str, int | str]]:
    return post_motion.recode._brotli_best(raw)


def _encode_qrm1_groups(groups: Mapping[int, Qrm1Group], selected_ids: Sequence[int]) -> tuple[bytes, dict[str, Any]]:
    selected = tuple(sorted(int(group_id) for group_id in selected_ids))
    raw = bytearray(b"QRM1")
    raw += len(selected).to_bytes(2, "little")
    for group_id in selected:
        if group_id not in groups:
            raise BridgeSparseActionError(f"selected QRM1 group {group_id} is missing from bridge source")
        raw += int(group_id).to_bytes(2, "little")
        raw += groups[group_id].payload
    encoded, brotli_params = _brotli_best(bytes(raw))
    check_groups, check_report = _parse_qrm1_groups(encoded)
    if sorted(check_groups) != list(selected):
        raise BridgeSparseActionError("QRM1 selected group roundtrip changed group ids")
    for group_id in selected:
        if check_groups[group_id].payload != groups[group_id].payload:
            raise BridgeSparseActionError(f"QRM1 group {group_id} payload changed during roundtrip")
    return encoded, {
        "brotli_params": brotli_params,
        "qrm1_raw_bytes": len(raw),
        "qrm1_raw_sha256": _sha256_bytes(bytes(raw)),
        "roundtrip": check_report,
    }


def _validate_qpost_groups(selected_groups: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(group) for group in selected_groups)
    duplicates = sorted({group for group in selected if selected.count(group) > 1})
    if duplicates:
        raise BridgeSparseActionError(f"duplicate selected qpost group id(s): {duplicates}")
    unknown = sorted(group for group in selected if group not in POST_MOTION_GROUPS)
    if unknown:
        raise BridgeSparseActionError(f"unknown selected qpost group id(s): {unknown}")
    return tuple(sorted(selected, key=POST_MOTION_GROUPS.index))


def _validate_randmulti_group_ids(selected_group_ids: Sequence[Any]) -> tuple[int, ...]:
    selected: list[int] = []
    for value in selected_group_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise BridgeSparseActionError(f"randmulti group id must be an integer: {value!r}")
        selected.append(int(value))
    duplicates = sorted({group_id for group_id in selected if selected.count(group_id) > 1})
    if duplicates:
        raise BridgeSparseActionError(f"duplicate selected randmulti group id(s): {duplicates}")
    invalid = sorted(
        group_id for group_id in selected if group_id < 0 or group_id >= len(PR85_HEADERLESS_RANDMULTI_SPECS)
    )
    if invalid:
        raise BridgeSparseActionError(f"randmulti group id(s) outside PR85 schedule: {invalid}")
    return tuple(sorted(selected))


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _string_list_field(raw: Any, field_name: str, errors: list[str]) -> list[str]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        errors.append(f"{field_name} must be a list of strings")
        return []
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            errors.append(f"{field_name} must contain only nonempty strings")
            return []
        values.append(item)
    return values


def _exact_evidence_override_report(raw: Any, blocker_ids: Sequence[str]) -> dict[str, Any]:
    blocker_set = set(blocker_ids)
    report: dict[str, Any] = {
        "required": bool(blocker_set),
        "present": raw is not None,
        "valid": False,
        "covers_blockers": False,
        "allowed_preflight_blockers": [],
        "missing_required_fields": [],
        "missing_blocker_coverage": sorted(blocker_set),
        "errors": [],
    }
    if not blocker_set:
        return report
    if raw is None:
        report["errors"].append("missing exact_evidence_override")
        return report
    if not isinstance(raw, Mapping):
        report["errors"].append("exact_evidence_override must be an object")
        return report

    missing_fields = [
        field
        for field in EXACT_EVIDENCE_OVERRIDE_REQUIRED_FIELDS
        if field not in raw or raw.get(field) in (None, "")
    ]
    report["missing_required_fields"] = missing_fields

    errors: list[str] = []
    allowed = _string_list_field(raw.get("allowed_preflight_blockers", []), "allowed_preflight_blockers", errors)
    allowed_set = set(allowed)
    unknown = sorted(allowed_set - set(KNOWN_PREFLIGHT_BLOCKERS))
    if unknown:
        errors.append(f"allowed_preflight_blockers contains unknown blocker(s): {unknown}")
    missing_coverage = sorted(blocker_set - allowed_set)
    evidence_grade = str(raw.get("evidence_grade", ""))
    if evidence_grade not in {"A", "A++"} and "exact" not in evidence_grade.lower():
        errors.append("evidence_grade must be A/A++ or explicitly exact")
    archive_sha = str(raw.get("archive_sha256", ""))
    if not _is_sha256_hex(archive_sha):
        errors.append("archive_sha256 must be a 64-character hex digest")
    auth_json = str(raw.get("auth_json", ""))
    if not auth_json.endswith(".json"):
        errors.append("auth_json must reference a JSON exact-eval artifact")
    for field in ("override_id", "rationale"):
        if not isinstance(raw.get(field), str) or not str(raw.get(field)).strip():
            errors.append(f"{field} must be a nonempty string")

    report.update(
        {
            "override_id": str(raw.get("override_id", "")),
            "evidence_grade": evidence_grade,
            "auth_json": auth_json,
            "archive_sha256": archive_sha,
            "rationale": str(raw.get("rationale", "")),
            "allowed_preflight_blockers": sorted(allowed_set),
            "covers_blockers": not missing_coverage,
            "missing_blocker_coverage": missing_coverage,
            "errors": errors,
            "valid": not missing_fields and not missing_coverage and not errors,
        }
    )
    if "score_delta_vs_pr85" in raw:
        report["score_delta_vs_pr85"] = raw["score_delta_vs_pr85"]
    return report


def _dispatch_preflight_report(
    *,
    policy: Mapping[str, Any],
    transforms: Sequence[Mapping[str, Any]],
    archive_shape_ready: bool,
) -> dict[str, Any]:
    selected_qpost = set(_validate_qpost_groups(policy["selected_qpost_groups"]))
    selected_randmulti = set(_validate_randmulti_group_ids(policy["selected_randmulti_group_ids"]))
    randmulti_transform = next(row for row in transforms if row["segment"] == "randmulti")
    source_randmulti_ids = set(randmulti_transform["source_qrm1"]["group_ids"])
    blockers: list[dict[str, Any]] = []

    if source_randmulti_ids and not selected_randmulti:
        blockers.append(
            {
                "blocker_id": "whole_randmulti_deletion",
                "reason": "selected policy removes every PR85 QRM1 randmulti group",
                "source_group_count": len(source_randmulti_ids),
                "selected_group_count": 0,
                "exact_negative_context": WHOLE_STREAM_NEGATIVE_CONTEXT["whole_randmulti_deletion"],
            }
        )
    keeps_post = bool(selected_qpost.intersection(POST_GROUPS))
    keeps_motion = bool(selected_qpost.intersection(MOTION_GROUPS))
    if not keeps_post:
        blockers.append(
            {
                "blocker_id": "whole_post_deletion",
                "reason": "selected policy neutralizes every PR85 post group",
                "protected_group_family": list(POST_GROUPS),
                "exact_negative_context": WHOLE_STREAM_NEGATIVE_CONTEXT["whole_post_deletion"],
            }
        )
    if not keeps_motion:
        blockers.append(
            {
                "blocker_id": "whole_motion_deletion",
                "reason": "selected policy neutralizes every PR85 motion group",
                "protected_group_family": list(MOTION_GROUPS),
                "exact_negative_context": WHOLE_STREAM_NEGATIVE_CONTEXT["whole_motion_deletion"],
            }
        )
    if not keeps_post and not keeps_motion:
        blockers.append(
            {
                "blocker_id": "whole_post_motion_deletion",
                "reason": "selected policy neutralizes every PR85 post and motion group",
                "protected_group_family": list(POST_MOTION_GROUPS),
                "exact_negative_context": WHOLE_STREAM_NEGATIVE_CONTEXT["whole_post_motion_deletion"],
            }
        )
    deleted_protected_qpost = [
        group for group in PROTECTED_QPOST_GROUPS if group not in selected_qpost
    ]
    if deleted_protected_qpost:
        blockers.append(
            {
                "blocker_id": "protected_qpost_group_deletion",
                "reason": (
                    "PR85 post/motion qpost groups are protected until decoded-parity "
                    "or exact group-level evidence supports deleting the group"
                ),
                "deleted_protected_qpost_groups": deleted_protected_qpost,
                "protected_qpost_groups": list(PROTECTED_QPOST_GROUPS),
            }
        )

    blocker_ids = [row["blocker_id"] for row in blockers]
    override = _exact_evidence_override_report(policy.get("exact_evidence_override"), blocker_ids)
    fail_closed = bool(blockers) and not override["valid"]
    status = "passed"
    if fail_closed:
        status = "blocked"
    elif blockers:
        status = "passed_with_exact_evidence_override"
    return {
        "schema": "pr85_bridge_sparse_action_dispatch_preflight_v1",
        "status": status,
        "dispatch_ready": bool(archive_shape_ready and not fail_closed),
        "archive_shape_ready": bool(archive_shape_ready),
        "fail_closed": bool(fail_closed),
        "blocker_count": len(blockers),
        "blocker_ids": blocker_ids,
        "blockers": blockers,
        "exact_evidence_override": override,
        "score_claim": False,
        "dispatch_performed": False,
    }


def _read_randmulti_policy_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("policies"), list):
        raise BridgeSparseActionError(f"{_repo_rel(path)} is not a PR85 randmulti policy JSON")
    return payload


def _randmulti_policy_ids(policy_payload: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    policies = policy_payload.get("policies")
    if not isinstance(policies, list):
        raise BridgeSparseActionError("randmulti policy JSON field 'policies' must be a list")
    for row in policies:
        if not isinstance(row, Mapping) or not isinstance(row.get("candidate_policy_id"), str):
            raise BridgeSparseActionError("every randmulti policy row must contain candidate_policy_id")
        out[str(row["candidate_policy_id"])] = _validate_randmulti_group_ids(
            row.get("selected_group_ids", [])
        )
    return out


def _resolve_action_policy(
    policy_id: str,
    *,
    randmulti_policies: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    if policy_id not in ACTION_POLICIES:
        raise BridgeSparseActionError(f"unknown sparse action policy id: {policy_id}")
    row = dict(ACTION_POLICIES[policy_id])
    selected_qpost = _validate_qpost_groups(row.get("selected_qpost_groups", ()))
    if "selected_randmulti_group_ids" in row:
        selected_randmulti = _validate_randmulti_group_ids(row["selected_randmulti_group_ids"])
        randmulti_policy_id = "inline"
    else:
        randmulti_policy_id = str(row.get("randmulti_policy_id", ""))
        if randmulti_policy_id not in randmulti_policies:
            raise BridgeSparseActionError(
                f"sparse action policy {policy_id!r} references missing randmulti policy {randmulti_policy_id!r}"
            )
        selected_randmulti = randmulti_policies[randmulti_policy_id]
    return {
        "policy_id": policy_id,
        "qpost_policy_id": str(row.get("qpost_policy_id", "inline")),
        "selected_qpost_groups": selected_qpost,
        "neutralized_qpost_groups": tuple(
            group for group in POST_MOTION_GROUPS if group not in set(selected_qpost)
        ),
        "randmulti_policy_id": randmulti_policy_id,
        "selected_randmulti_group_ids": selected_randmulti,
        "exact_evidence_override": row.get("exact_evidence_override"),
        "basis": str(row.get("basis", "")),
    }


def _transform_qpost_streams(
    streams: Mapping[str, bytes],
    *,
    selected_qpost_groups: Sequence[str],
    selected_randmulti_group_ids: Sequence[int],
) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, Any]]:
    selected_set = set(_validate_qpost_groups(selected_qpost_groups))
    transforms = [
        post_motion._post_transform(streams["post"], selected_set),
        post_motion._motion_transform("shift", streams["shift"], selected_set),
        post_motion._motion_transform("frac", streams["frac"], selected_set),
        post_motion._motion_transform("frac2", streams["frac2"], selected_set),
        post_motion._motion_transform("frac3", streams["frac3"], selected_set),
    ]
    for transform in transforms:
        if transform.semantic_report["status"] != "passed":
            raise BridgeSparseActionError(f"qpost transform failed for {transform.segment}")
    out = dict(streams)
    for transform in transforms:
        out[transform.segment] = transform.candidate_segment
    source_groups, source_qrm1 = _parse_qrm1_groups(streams["randmulti"])
    selected_randmulti = _validate_randmulti_group_ids(selected_randmulti_group_ids)
    if set(selected_randmulti) == set(source_groups):
        out["randmulti"] = streams["randmulti"]
        qrm1_encode = {
            "brotli_params": None,
            "qrm1_raw_bytes": source_qrm1["raw_bytes"],
            "qrm1_raw_sha256": source_qrm1["raw_sha256"],
            "roundtrip": source_qrm1,
        }
    else:
        out["randmulti"], qrm1_encode = _encode_qrm1_groups(source_groups, selected_randmulti)
    candidate_groups, candidate_qrm1 = _parse_qrm1_groups(out["randmulti"])
    omitted = sorted(group_id for group_id in source_groups if group_id not in set(selected_randmulti))
    mismatched = [
        group_id
        for group_id in selected_randmulti
        if candidate_groups[group_id].payload != source_groups[group_id].payload
    ]
    if mismatched:
        raise BridgeSparseActionError(f"selected QRM1 group payload mismatch: {mismatched}")
    if sorted(candidate_groups) != list(selected_randmulti):
        raise BridgeSparseActionError("candidate QRM1 group ids do not match selected policy")
    transform_records = [post_motion._transform_record(transform) for transform in transforms]
    transform_records.append(
        {
            "segment": "randmulti",
            "source_segment_bytes": int(len(streams["randmulti"])),
            "source_segment_sha256": _sha256_bytes(streams["randmulti"]),
            "candidate_segment_bytes": int(len(out["randmulti"])),
            "candidate_segment_sha256": _sha256_bytes(out["randmulti"]),
            "segment_byte_delta": int(len(out["randmulti"]) - len(streams["randmulti"])),
            "selected_group_ids": list(selected_randmulti),
            "omitted_group_ids": omitted,
            "selected_group_count": len(selected_randmulti),
            "omitted_group_count": len(omitted),
            "source_qrm1": source_qrm1,
            "candidate_qrm1": candidate_qrm1,
            "qrm1_encode": qrm1_encode,
            "group_profiles": [
                {
                    "group_id": int(group.group_id),
                    "height": int(group.height),
                    "width": int(group.width),
                    "amplitude": int(group.amplitude),
                    "scount": int(group.scount),
                    "raw_payload_bytes": int(group.raw_payload_bytes),
                    "nonzero_choice_count": int(group.nonzero_choice_count),
                    "selected": int(group.group_id) in set(selected_randmulti),
                }
                for group in sorted(source_groups.values(), key=lambda item: item.group_id)
            ],
            "noop_segment": streams["randmulti"] == out["randmulti"],
        }
    )
    return out, transform_records, {
        "source_qpost": _parse_qpost(_pack_qpost(streams))[1],
        "candidate_qpost": _parse_qpost(_pack_qpost(out))[1],
    }


def _candidate_archive_info(path: Path, members: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": _repo_rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_count": len(members),
        "members": members,
    }


def _build_one(
    *,
    policy: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    bridge_archive: Mapping[str, Any],
    bridge_members: Mapping[str, bytes],
    out_dir: Path,
    randmulti_policy_json: Path,
) -> dict[str, Any]:
    source_streams, qpost_meta = _parse_qpost(bridge_members["qpost.bin"])
    candidate_streams, transforms, qpost_delta = _transform_qpost_streams(
        source_streams,
        selected_qpost_groups=policy["selected_qpost_groups"],
        selected_randmulti_group_ids=policy["selected_randmulti_group_ids"],
    )
    candidate_qpost = _pack_qpost(candidate_streams)
    candidate_members = dict(bridge_members)
    candidate_members["qpost.bin"] = candidate_qpost
    candidate_id = str(policy["policy_id"])
    candidate_dir = out_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    zip_members = _write_bridge_archive(archive_path, candidate_members)
    candidate = _candidate_archive_info(archive_path, zip_members)
    archive_delta = int(candidate["archive_bytes"] - int(bridge_archive["archive_bytes"]))
    qpost_delta_bytes = int(len(candidate_qpost) - len(bridge_members["qpost.bin"]))
    changed_members = [
        name for name in MEMBER_ORDER if candidate_members[name] != bridge_members[name]
    ]
    archive_shape_ready = (
        changed_members == ["qpost.bin"]
        and candidate["member_count"] == len(MEMBER_ORDER)
        and all(row["name"] in MEMBER_ORDER for row in candidate["members"])
    )
    dispatch_preflight = _dispatch_preflight_report(
        policy=policy,
        transforms=transforms,
        archive_shape_ready=archive_shape_ready,
    )
    ready_for_claim = bool(dispatch_preflight["dispatch_ready"])
    if ready_for_claim:
        dispatch_gate = "eligible_for_cuda_auth_eval_after_lane_claim"
        next_gate = (
            "Before any exact eval dispatch, claim the lane with tools/claim_lane_dispatch.py "
            "and run CUDA auth eval on this exact archive/runtime pair."
        )
    elif archive_shape_ready and dispatch_preflight["status"] == "blocked":
        dispatch_gate = "planning_only/preflight_blocked"
        next_gate = (
            "Do not dispatch: sparse-action preflight blocked this policy. "
            "A reviewed exact_evidence_override covering every blocker is required "
            "before this candidate may be considered for exact eval."
        )
    else:
        dispatch_gate = "planning_only/archive_shape_blocked"
        next_gate = "Do not dispatch: candidate archive shape is not exact-eval ready."
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "promotion_status": "non_promotable_pending_exact_cuda_eval",
        "evidence_grade": "empirical_bridge_archive_build_only",
        "source_archive": source_archive,
        "pr85_bridge_archive": bridge_archive,
        "selected_action_policy": {
            "policy_id": candidate_id,
            "basis": policy["basis"],
            "qpost_policy_id": policy["qpost_policy_id"],
            "selected_qpost_groups": list(policy["selected_qpost_groups"]),
            "neutralized_qpost_groups": list(policy["neutralized_qpost_groups"]),
            "randmulti_policy_id": policy["randmulti_policy_id"],
            "selected_randmulti_group_ids": list(policy["selected_randmulti_group_ids"]),
            "randmulti_policy_json": _repo_rel(randmulti_policy_json),
            "pr79_action_tooling_reference": [
                "experiments/build_pr79_action_subset_candidates.py",
                "experiments/build_pr79_action_dictionary_repack_candidates_v2.py",
            ],
        },
        "candidate_archive": candidate,
        "bridge_qpost": qpost_meta,
        "qpost_delta": qpost_delta,
        "transforms": transforms,
        "changed_members": changed_members,
        "charged_byte_deltas": {
            "archive_delta_bytes_vs_pr85_bridge": archive_delta,
            "qpost_member_delta_bytes_vs_pr85_bridge": qpost_delta_bytes,
            "formula_only_rate_score_delta_vs_pr85_bridge": archive_delta * RATE_SCORE_PER_BYTE,
            "member_deltas": {
                name: {
                    "source_bytes": int(len(bridge_members[name])),
                    "candidate_bytes": int(len(candidate_members[name])),
                    "delta_bytes": int(len(candidate_members[name]) - len(bridge_members[name])),
                    "source_sha256": _sha256_bytes(bridge_members[name]),
                    "candidate_sha256": _sha256_bytes(candidate_members[name]),
                }
                for name in MEMBER_ORDER
            },
        },
        "safe_archive_members": {
            "status": "passed",
            "expected_order": list(MEMBER_ORDER),
            "observed_order": [row["name"] for row in candidate["members"]],
            "zip_slip_safe": True,
            "duplicate_members": False,
        },
        "dispatch_preflight": dispatch_preflight,
        "ready_for_exact_eval_dispatch_claim": ready_for_claim,
        "dispatch_gate": dispatch_gate,
        "next_gate": next_gate,
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def build_candidates(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    bridge_archive: Path = DEFAULT_BRIDGE_ARCHIVE,
    randmulti_policy_json: Path = DEFAULT_RANDMULTI_POLICY_JSON,
    out_dir: Path = DEFAULT_OUT_DIR,
    policy_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = _source_archive_info(source_archive)
    bridge, bridge_members = _read_bridge_archive(bridge_archive)
    randmulti_payload = _read_randmulti_policy_payload(randmulti_policy_json)
    randmulti_policies = _randmulti_policy_ids(randmulti_payload)
    selected_policy_ids = list(policy_ids or ACTION_POLICIES)
    policies = [
        _resolve_action_policy(policy_id, randmulti_policies=randmulti_policies)
        for policy_id in selected_policy_ids
    ]
    candidates = [
        _build_one(
            policy=policy,
            source_archive=source,
            bridge_archive=bridge,
            bridge_members=bridge_members,
            out_dir=out_dir,
            randmulti_policy_json=randmulti_policy_json,
        )
        for policy in policies
    ]
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "promotion_status": "non_promotable_pending_exact_cuda_eval",
        "evidence_grade": "empirical_bridge_archive_build_only",
        "source_archive": source,
        "pr85_bridge_archive": bridge,
        "randmulti_policy_json": _repo_rel(randmulti_policy_json),
        "candidate_count": len(candidates),
        "ready_for_exact_eval_dispatch_claim_count": sum(
            1 for row in candidates if row["ready_for_exact_eval_dispatch_claim"]
        ),
        "dispatchable_candidate_count": sum(
            1
            for row in candidates
            if row["dispatch_gate"] == "eligible_for_cuda_auth_eval_after_lane_claim"
        ),
        "dispatch_preflight_blocked_candidate_count": sum(
            1 for row in candidates if row["dispatch_preflight"]["status"] == "blocked"
        ),
        "dispatch_preflight_blocker_ids": sorted(
            {
                blocker_id
                for row in candidates
                for blocker_id in row["dispatch_preflight"]["blocker_ids"]
            }
        ),
        "candidates": candidates,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--bridge-archive", type=Path, default=DEFAULT_BRIDGE_ARCHIVE)
    parser.add_argument("--randmulti-policy-json", type=Path, default=DEFAULT_RANDMULTI_POLICY_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy", action="append", dest="policies", choices=sorted(ACTION_POLICIES))
    args = parser.parse_args(argv)

    payload = build_candidates(
        source_archive=args.source_archive,
        bridge_archive=args.bridge_archive,
        randmulti_policy_json=args.randmulti_policy_json,
        out_dir=args.out_dir,
        policy_ids=args.policies,
    )
    print(_json_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
