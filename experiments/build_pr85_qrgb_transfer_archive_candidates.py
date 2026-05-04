#!/usr/bin/env python3
"""Build byte-closed PR85 archives from grounded PR90 QRGB transfer actions.

This is a local archive builder only. It consumes the pair-action evidence
emitted by ``experiments/plan_pr85_qrgb_transfer_actions.py``, cross-checks the
actions against the richer transfer planning JSON, mutates PR85 bundle bytes via
``tac.pr85_bundle``, and runs the fixed-runtime readiness preflight.

It does not train, import contest scorers, dispatch remote jobs, or claim score
evidence. Any archive marked ready still requires a lane claim before exact CUDA
auth eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import brotli
except ImportError:  # pragma: no cover - environment guard
    brotli = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import preflight_pr85_fixed_runtime_readiness as fixed_preflight  # noqa: E402
from tac.pr85_bundle import (  # noqa: E402
    FIXED_V5_LENGTHS,
    PR85_HEADERLESS_RANDMULTI_SPECS,
    SEGMENT_ORDER,
    Pr85BundleError,
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)


TOOL = "experiments/build_pr85_qrgb_transfer_archive_candidates.py"
SCHEMA = "pr85_qrgb_transfer_archive_candidate_planning_v1"
MANIFEST_SCHEMA = "pr85_qrgb_transfer_archive_candidate_v1"
PAIR_ACTION_EVIDENCE_SCHEMA = "pr85_pair_action_lowering_evidence_v1"
TRANSFER_PLAN_SCHEMA = "pr85_qrgb_transfer_action_plan_v1"
DEFAULT_SOURCE_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_PAIR_ACTION_EVIDENCE = (
    REPO_ROOT / "experiments/results/pr85_qrgb_transfer_actions_20260504_worker/pair_action_evidence.json"
)
DEFAULT_TRANSFER_PLAN = (
    REPO_ROOT / "experiments/results/pr85_qrgb_transfer_actions_20260504_worker/planning.json"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_worker"
)
DEFAULT_LEDGER = (
    REPO_ROOT / ".omx/research/pr85_qrgb_transfer_archive_candidates_20260504_worker.md"
)
DEFAULT_ROBUST_CURRENT = REPO_ROOT / "submissions/robust_current"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PAIR_COUNT = 600
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
KNOWN_PR85 = {
    "archive_bytes": 236_328,
    "archive_sha256": "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e",
    "score": 0.25806611029397786,
}

CHOICE_STREAM_DEFAULTS = {
    "shift": 40,
    "frac": 4,
    "frac2": 4,
    "frac3": 4,
    "bias": 13,
    "region": 0,
}
CHOICE_STREAMS = tuple(CHOICE_STREAM_DEFAULTS)
SUPPORTED_STREAMS = tuple(sorted(set(CHOICE_STREAMS) | {"randmulti"}))


class QrgbTransferArchiveBuilderError(ValueError):
    """Raised when the QRGB transfer archive builder must abort fail-closed."""


@dataclass(frozen=True)
class ActionSpec:
    pair_index: int
    stream: str
    value: int
    source_value: int
    source_artifact_sha256: str
    rationale: str | None
    raw: Mapping[str, Any]
    basis_action_schema: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    header_mode: str
    rank: int | None
    actions: tuple[ActionSpec, ...]
    planning_row: Mapping[str, Any]
    evidence_row: Mapping[str, Any]


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QrgbTransferArchiveBuilderError(f"{_rel(path)} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise QrgbTransferArchiveBuilderError(f"{_rel(path)} must contain a JSON object")
    return payload


def _safe_candidate_id(value: str) -> str:
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in value):
        raise QrgbTransferArchiveBuilderError(f"unsafe candidate_id: {value!r}")
    return value


def _zip_info(name: str = "x") -> zipfile.ZipInfo:
    validate_pr85_member_name(name)
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_single_member_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("x"), payload)


def _read_source_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise QrgbTransferArchiveBuilderError(f"source archive is missing: {_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise QrgbTransferArchiveBuilderError(
                f"PR85 source archive must contain exactly one safe member 'x'; got {names!r}"
            )
        info = infos[0]
        validate_pr85_member_name(info.filename)
        raw = zf.read(info)
    sha = _sha256_file(path)
    return (
        {
            "path": _rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": sha,
            "member_name": info.filename,
            "member_bytes": int(info.file_size),
            "member_compress_size": int(info.compress_size),
            "member_sha256": _sha256_bytes(raw),
            "zip_stored": info.compress_type == zipfile.ZIP_STORED,
            "known_pr85_anchor_match": {
                "matches": int(path.stat().st_size) == KNOWN_PR85["archive_bytes"]
                and sha == KNOWN_PR85["archive_sha256"],
                "expected_archive_bytes": KNOWN_PR85["archive_bytes"],
                "expected_archive_sha256": KNOWN_PR85["archive_sha256"],
                "exact_t4_score_context": KNOWN_PR85["score"],
            },
        },
        raw,
    )


def _archive_info(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise QrgbTransferArchiveBuilderError(
                f"candidate archive must contain one member; found {len(infos)}"
            )
        info = infos[0]
        validate_pr85_member_name(info.filename)
        raw = zf.read(info)
    return {
        "archive_path": _rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_name": info.filename,
        "member_bytes": int(len(raw)),
        "member_sha256": _sha256_bytes(raw),
        "zip_stored": info.compress_type == zipfile.ZIP_STORED,
    }


def _read_varints(raw: bytes, pos: int, count: int) -> tuple[list[int], int]:
    values: list[int] = []
    for _ in range(count):
        value = 0
        shift = 0
        while True:
            if pos >= len(raw):
                raise QrgbTransferArchiveBuilderError("truncated varint stream")
            byte = int(raw[pos])
            pos += 1
            value |= (byte & 0x7F) << shift
            if byte < 128:
                values.append(value)
                break
            shift += 7
            if shift > 63:
                raise QrgbTransferArchiveBuilderError("overlong varint stream")
    return values, pos


def _write_varint(value: int) -> bytes:
    if value < 0:
        raise QrgbTransferArchiveBuilderError(f"varint cannot encode negative value {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_sparse_choice(raw: bytes, *, magic: bytes, default_choice: int) -> bytes:
    if not raw.startswith(magic) or len(raw) < 5:
        raise QrgbTransferArchiveBuilderError(f"bad sparse choice magic for {magic!r}")
    count = int.from_bytes(raw[3:5], "little")
    pos = 5
    gaps, pos = _read_varints(raw, pos, count)
    vals = raw[pos : pos + count]
    if len(vals) != count or pos + count != len(raw):
        raise QrgbTransferArchiveBuilderError("sparse choice payload length mismatch")
    out = bytearray([default_choice] * PAIR_COUNT)
    index = -1
    for gap, value in zip(gaps, vals, strict=True):
        index += int(gap) + 1
        if not 0 <= index < PAIR_COUNT:
            raise QrgbTransferArchiveBuilderError(f"sparse choice index out of range: {index}")
        out[index] = int(value) - 1
    return bytes(out)


def _encode_sparse_choice(magic: bytes, values: bytes, *, default_choice: int) -> bytes:
    if len(values) != PAIR_COUNT:
        raise QrgbTransferArchiveBuilderError(f"choice stream must contain {PAIR_COUNT} pairs")
    indices = [index for index, value in enumerate(values) if value != default_choice]
    out = bytearray(magic + len(indices).to_bytes(2, "little"))
    previous = -1
    for index in indices:
        out += _write_varint(index - previous - 1)
        previous = index
    out += bytes(values[index] + 1 for index in indices)
    return bytes(out)


def _decode_direct_or_delta(
    raw: bytes,
    *,
    direct_magic: bytes,
    delta_magic: bytes | None,
    default_choice: int,
) -> tuple[bytes, str]:
    if raw.startswith(direct_magic):
        values = raw[len(direct_magic) :]
        if len(values) != PAIR_COUNT:
            raise QrgbTransferArchiveBuilderError(f"{direct_magic!r} stream has {len(values)} pairs")
        return bytes(values), direct_magic.decode("ascii")
    if delta_magic is not None and raw.startswith(delta_magic):
        encoded = raw[len(delta_magic) :]
        if len(encoded) != PAIR_COUNT:
            raise QrgbTransferArchiveBuilderError(f"{delta_magic!r} stream has {len(encoded)} pairs")
        return bytes(default_choice if value == 0 else value - 1 for value in encoded), delta_magic.decode("ascii")
    raise QrgbTransferArchiveBuilderError(f"unsupported choice stream magic {raw[:4]!r}")


def _decode_choice_stream(name: str, segment: bytes) -> tuple[bytearray, dict[str, Any]]:
    if brotli is None:
        raise QrgbTransferArchiveBuilderError("brotli is required for PR85 QRGB transfer builds")
    try:
        raw = brotli.decompress(segment)
    except brotli.error as exc:
        raise QrgbTransferArchiveBuilderError(f"PR85 segment {name!r} is not Brotli-decodable") from exc
    default = CHOICE_STREAM_DEFAULTS[name]
    if name == "shift":
        values, codec = _decode_direct_or_delta(raw, direct_magic=b"SH4", delta_magic=b"SD4", default_choice=default)
    elif name == "frac":
        if raw.startswith(b"FV1"):
            values, codec = _decode_sparse_choice(raw, magic=b"FV1", default_choice=default), "FV1"
        else:
            values, codec = _decode_direct_or_delta(raw, direct_magic=b"FH1", delta_magic=None, default_choice=default)
    elif name == "frac2":
        values, codec = _decode_direct_or_delta(raw, direct_magic=b"FH2", delta_magic=None, default_choice=default)
    elif name == "frac3":
        values, codec = _decode_direct_or_delta(raw, direct_magic=b"FH3", delta_magic=b"FD3", default_choice=default)
    elif name == "bias":
        if raw.startswith(b"BV1"):
            values, codec = _decode_sparse_choice(raw, magic=b"BV1", default_choice=default), "BV1"
        else:
            values, codec = _decode_direct_or_delta(raw, direct_magic=b"BH1", delta_magic=b"BD1", default_choice=default)
    elif name == "region":
        if raw.startswith(b"RV1"):
            values, codec = _decode_sparse_choice(raw, magic=b"RV1", default_choice=default), "RV1"
        else:
            values, codec = _decode_direct_or_delta(raw, direct_magic=b"RH1", delta_magic=b"RD1", default_choice=default)
    else:  # pragma: no cover - guarded by action parsing
        raise QrgbTransferArchiveBuilderError(f"unsupported choice stream {name!r}")
    return bytearray(values), {
        "stream": name,
        "codec": codec,
        "source_raw_bytes": len(raw),
        "source_raw_sha256": _sha256_bytes(raw),
        "source_segment_bytes": len(segment),
        "source_segment_sha256": _sha256_bytes(segment),
        "default_symbol": default,
    }


def _encode_choice_stream(name: str, values: bytes, *, codec: str) -> tuple[bytes, dict[str, Any]]:
    if brotli is None:
        raise QrgbTransferArchiveBuilderError("brotli is required for PR85 QRGB transfer builds")
    default = CHOICE_STREAM_DEFAULTS[name]
    if codec in {"SH4", "FH1", "FH2", "FH3", "BH1", "RH1"}:
        raw = codec.encode("ascii") + bytes(values)
    elif codec in {"SD4", "FD3", "BD1", "RD1"}:
        raw = codec.encode("ascii") + bytes(0 if value == default else value + 1 for value in values)
    elif codec in {"FV1", "BV1", "RV1"}:
        raw = _encode_sparse_choice(codec.encode("ascii"), bytes(values), default_choice=default)
    else:
        raise QrgbTransferArchiveBuilderError(f"unsupported re-encode codec {codec!r}")
    encoded = brotli.compress(raw, quality=11, lgwin=24)
    if brotli.decompress(encoded) != raw:
        raise QrgbTransferArchiveBuilderError(f"{name} Brotli roundtrip changed bytes")
    return encoded, {
        "candidate_raw_bytes": len(raw),
        "candidate_raw_sha256": _sha256_bytes(raw),
        "candidate_segment_bytes": len(encoded),
        "candidate_segment_sha256": _sha256_bytes(encoded),
        "brotli_quality": 11,
        "brotli_lgwin": 24,
    }


def _decode_randmulti_row(raw: bytes, pos: int) -> tuple[bytearray, dict[str, Any], int]:
    start = pos
    if pos >= len(raw):
        raise QrgbTransferArchiveBuilderError("truncated PR85 randmulti stream before count")
    count = int(raw[pos])
    pos += 1
    extended_count = False
    if count == 255:
        extended_count = True
        if pos + 2 > len(raw):
            raise QrgbTransferArchiveBuilderError("truncated PR85 randmulti extended count")
        count = int.from_bytes(raw[pos : pos + 2], "little")
        pos += 2
    gaps, pos = _read_varints(raw, pos, count)
    values = raw[pos : pos + count]
    pos += count
    if len(values) != count:
        raise QrgbTransferArchiveBuilderError("truncated PR85 randmulti value stream")
    out = bytearray(PAIR_COUNT)
    index = -1
    for gap, value in zip(gaps, values, strict=True):
        index += int(gap) + 1
        if not 0 <= index < PAIR_COUNT:
            raise QrgbTransferArchiveBuilderError(f"PR85 randmulti sparse index out of range: {index}")
        out[index] = int(value)
    return out, {
        "count": count,
        "extended_count": extended_count,
        "row_bytes": pos - start,
        "nonzero_entries": count,
    }, pos


def _encode_randmulti_row(values: bytes) -> bytes:
    if len(values) != PAIR_COUNT:
        raise QrgbTransferArchiveBuilderError(f"randmulti row must contain {PAIR_COUNT} pairs")
    indices = [index for index, value in enumerate(values) if value != 0]
    out = bytearray()
    if len(indices) >= 255:
        if len(indices) > 0xFFFF:
            raise QrgbTransferArchiveBuilderError("randmulti row has too many nonzero entries")
        out.append(255)
        out.extend(len(indices).to_bytes(2, "little"))
    else:
        out.append(len(indices))
    previous = -1
    for index in indices:
        out += _write_varint(index - previous - 1)
        previous = index
    out += bytes(values[index] for index in indices)
    return bytes(out)


def _decode_randmulti_stream(segment: bytes) -> tuple[list[list[bytearray]], dict[str, Any]]:
    if brotli is None:
        raise QrgbTransferArchiveBuilderError("brotli is required for PR85 QRGB transfer builds")
    try:
        raw = brotli.decompress(segment)
    except brotli.error as exc:
        raise QrgbTransferArchiveBuilderError("PR85 randmulti segment is not Brotli-decodable") from exc
    groups: list[list[bytearray]] = []
    row_meta: list[dict[str, Any]] = []
    pos = 0
    for group_index, (_lh, _lw, _amp, row_count) in enumerate(PR85_HEADERLESS_RANDMULTI_SPECS):
        rows: list[bytearray] = []
        for row_index in range(int(row_count)):
            values, meta, pos = _decode_randmulti_row(raw, pos)
            rows.append(values)
            row_meta.append({"group_index": group_index, "row_index": row_index, **meta})
        groups.append(rows)
    if pos != len(raw):
        raise QrgbTransferArchiveBuilderError("PR85 randmulti stream has trailing bytes")
    return groups, {
        "stream": "randmulti",
        "codec": "pr85_headerless_sparse_randmulti",
        "source_raw_bytes": len(raw),
        "source_raw_sha256": _sha256_bytes(raw),
        "source_segment_bytes": len(segment),
        "source_segment_sha256": _sha256_bytes(segment),
        "group_count": len(groups),
        "row_count": len(row_meta),
        "nonzero_entries": sum(int(row["nonzero_entries"]) for row in row_meta),
    }


def _encode_randmulti_stream(groups: Sequence[Sequence[bytes | bytearray]]) -> tuple[bytes, dict[str, Any]]:
    if brotli is None:
        raise QrgbTransferArchiveBuilderError("brotli is required for PR85 QRGB transfer builds")
    if len(groups) != len(PR85_HEADERLESS_RANDMULTI_SPECS):
        raise QrgbTransferArchiveBuilderError("randmulti group count does not match PR85 schedule")
    raw = bytearray()
    nonzero_entries = 0
    for group_index, ((_lh, _lw, _amp, row_count), rows) in enumerate(
        zip(PR85_HEADERLESS_RANDMULTI_SPECS, groups, strict=True)
    ):
        if len(rows) != int(row_count):
            raise QrgbTransferArchiveBuilderError(
                f"randmulti group {group_index} has {len(rows)} row(s), expected {row_count}"
            )
        for row in rows:
            row_bytes = bytes(row)
            nonzero_entries += sum(1 for value in row_bytes if value != 0)
            raw += _encode_randmulti_row(row_bytes)
    encoded = brotli.compress(bytes(raw), quality=11, lgwin=24)
    if brotli.decompress(encoded) != bytes(raw):
        raise QrgbTransferArchiveBuilderError("randmulti Brotli roundtrip changed bytes")
    return encoded, {
        "candidate_raw_bytes": len(raw),
        "candidate_raw_sha256": _sha256_bytes(bytes(raw)),
        "candidate_segment_bytes": len(encoded),
        "candidate_segment_sha256": _sha256_bytes(encoded),
        "candidate_nonzero_entries": nonzero_entries,
        "brotli_quality": 11,
        "brotli_lgwin": 24,
    }


def _randmulti_target(action: ActionSpec) -> tuple[int, int, dict[str, Any]]:
    target = action.basis_action_schema.get("target_randmulti_group")
    if not isinstance(target, Mapping):
        raise QrgbTransferArchiveBuilderError("randmulti action lacks target_randmulti_group grounding")
    group_index = target.get("group_index")
    selection_row = target.get("selection_row", 0)
    if not isinstance(group_index, int) or isinstance(group_index, bool):
        raise QrgbTransferArchiveBuilderError("randmulti group_index must be an integer")
    if not 0 <= group_index < len(PR85_HEADERLESS_RANDMULTI_SPECS):
        raise QrgbTransferArchiveBuilderError(f"randmulti group_index out of range: {group_index}")
    if not isinstance(selection_row, int) or isinstance(selection_row, bool):
        raise QrgbTransferArchiveBuilderError("randmulti selection_row must be an integer")
    lh, lw, amp, row_count = PR85_HEADERLESS_RANDMULTI_SPECS[group_index]
    expected = {"lh": lh, "lw": lw, "amp": amp}
    mismatches = {
        key: {"expected": value, "actual": target.get(key)}
        for key, value in expected.items()
        if target.get(key) != value
    }
    if mismatches:
        raise QrgbTransferArchiveBuilderError(
            f"randmulti target group spec does not match PR85 schedule: {mismatches}"
        )
    if not 0 <= selection_row < int(row_count):
        raise QrgbTransferArchiveBuilderError(
            f"randmulti selection_row {selection_row} out of range for group {group_index}"
        )
    return group_index, selection_row, dict(target)


def _blocker(blocker_class: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"blocker_class": blocker_class, "reason": reason, **extra}


def _candidate_blocked(
    candidate_id: str,
    blocker_class: str,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "build_status": "blocked",
        "blocker_class": blocker_class,
        "blockers": [_blocker(blocker_class, reason, **extra)],
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "dispatch_unlocked": False,
        "candidate_archive": None,
    }


def _source_artifact_sha(plan: Mapping[str, Any]) -> str | None:
    pair_source = plan.get("pair_planning_source")
    if isinstance(pair_source, Mapping):
        value = pair_source.get("source_sha_for_pair_action")
        if isinstance(value, str) and value:
            return value
    return None


def _candidate_value(raw: Mapping[str, Any]) -> int | None:
    value = raw.get("candidate_value", raw.get("value"))
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _candidate_actions_match_plan(
    evidence_actions: Sequence[Mapping[str, Any]],
    plan_actions: Sequence[Mapping[str, Any]],
) -> bool:
    if len(evidence_actions) != len(plan_actions):
        return False
    for left, right in zip(evidence_actions, plan_actions, strict=True):
        if left.get("stream") != right.get("stream"):
            return False
        if left.get("pair_index") != right.get("pair_index"):
            return False
        if _candidate_value(left) != _candidate_value(right):
            return False
        if left.get("source_value") != right.get("source_value"):
            return False
    return True


def _parse_action_spec(
    *,
    raw: Mapping[str, Any],
    planning_row: Mapping[str, Any],
    expected_source_artifact_sha256: str | None,
) -> tuple[ActionSpec | None, list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    pair_index = raw.get("pair_index")
    stream = raw.get("stream")
    value = _candidate_value(raw)
    source_value = raw.get("source_value")
    source_sha = raw.get("source_artifact_sha256")
    if raw.get("op") != "set":
        blockers.append(_blocker("ungrounded_stream_value_evidence", "action op must be 'set'"))
    if not isinstance(pair_index, int) or isinstance(pair_index, bool) or not 0 <= pair_index < PAIR_COUNT:
        blockers.append(_blocker("ungrounded_stream_value_evidence", "pair_index must be an integer in [0,600)"))
    if stream not in SUPPORTED_STREAMS:
        blockers.append(
            _blocker(
                "ungrounded_stream_value_evidence",
                "stream is not supported by this PR85 transfer builder",
                stream=stream,
                supported_streams=list(SUPPORTED_STREAMS),
            )
        )
    if value is None or not 0 <= value <= 255:
        blockers.append(_blocker("ungrounded_stream_value_evidence", "candidate_value/value must be an integer byte"))
    if not isinstance(source_value, int) or isinstance(source_value, bool) or not 0 <= source_value <= 255:
        blockers.append(_blocker("ungrounded_stream_value_evidence", "source_value must be an integer byte"))
    if not isinstance(source_sha, str) or not source_sha:
        blockers.append(_blocker("source_sha_mismatch", "action source_artifact_sha256 is missing"))
    elif expected_source_artifact_sha256 is not None and source_sha != expected_source_artifact_sha256:
        blockers.append(
            _blocker(
                "source_sha_mismatch",
                "action source_artifact_sha256 does not match transfer plan source",
                expected=expected_source_artifact_sha256,
                actual=source_sha,
            )
        )
    if value is not None and isinstance(source_value, int) and value == source_value:
        blockers.append(_blocker("source_preserving_edit", "candidate_value equals source_value"))
    basis = planning_row.get("basis_action_schema")
    if not isinstance(basis, Mapping):
        blockers.append(_blocker("ungrounded_stream_value_evidence", "planning row lacks basis_action_schema"))
        basis = {}
    elif basis.get("target_stream") != stream:
        blockers.append(
            _blocker(
                "ungrounded_stream_value_evidence",
                "basis target_stream does not match action stream",
                target_stream=basis.get("target_stream"),
                action_stream=stream,
            )
        )
    if stream == "randmulti" and isinstance(basis, Mapping):
        try:
            _randmulti_target(
                ActionSpec(
                    pair_index=int(pair_index or 0),
                    stream="randmulti",
                    value=int(value or 0),
                    source_value=int(source_value or 0),
                    source_artifact_sha256=str(source_sha or ""),
                    rationale=raw.get("rationale") if isinstance(raw.get("rationale"), str) else None,
                    raw=raw,
                    basis_action_schema=basis,
                )
            )
        except QrgbTransferArchiveBuilderError as exc:
            blockers.append(_blocker("ungrounded_stream_value_evidence", str(exc)))
    if blockers:
        return None, blockers
    assert isinstance(pair_index, int)
    assert isinstance(stream, str)
    assert value is not None
    assert isinstance(source_value, int)
    assert isinstance(source_sha, str)
    return (
        ActionSpec(
            pair_index=int(pair_index),
            stream=stream,
            value=int(value),
            source_value=int(source_value),
            source_artifact_sha256=source_sha,
            rationale=raw.get("rationale") if isinstance(raw.get("rationale"), str) else None,
            raw=raw,
            basis_action_schema=basis,
        ),
        [],
    )


def _load_candidate_specs(
    *,
    evidence_json: Path,
    transfer_plan_json: Path,
    source_archive: Mapping[str, Any],
    max_candidates: int,
) -> tuple[dict[str, Any], list[CandidateSpec], list[dict[str, Any]]]:
    evidence = _load_json(evidence_json)
    plan = _load_json(transfer_plan_json)
    blockers: list[dict[str, Any]] = []
    if evidence.get("schema") != PAIR_ACTION_EVIDENCE_SCHEMA:
        blockers.append(_blocker("ungrounded_stream_value_evidence", "unexpected pair-action evidence schema"))
    if evidence.get("score_claim") is not False:
        blockers.append(_blocker("ungrounded_stream_value_evidence", "pair-action evidence score_claim must be false"))
    if evidence.get("dispatch_performed") is not False or evidence.get("remote_jobs_dispatched") is not False:
        blockers.append(_blocker("ungrounded_stream_value_evidence", "pair-action evidence dispatch flags must be false"))
    if plan.get("schema") not in (TRANSFER_PLAN_SCHEMA, None):
        blockers.append(_blocker("ungrounded_stream_value_evidence", "unexpected transfer planning schema"))
    raw_candidates = evidence.get("candidates")
    if not isinstance(raw_candidates, list):
        blockers.append(_blocker("ungrounded_stream_value_evidence", "pair-action evidence candidates must be a list"))
        raw_candidates = []
    ranked_rows = plan.get("ranked_candidates")
    if not isinstance(ranked_rows, list):
        blockers.append(_blocker("ungrounded_stream_value_evidence", "transfer plan ranked_candidates must be a list"))
        ranked_rows = []
    plan_by_id = {
        row.get("candidate_id"): row
        for row in ranked_rows
        if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str)
    }
    expected_source_artifact_sha = _source_artifact_sha(plan)
    seen_candidate_ids: set[str] = set()
    seen_actions: set[tuple[int, str, int, int]] = set()
    seen_pairs: set[int] = set()
    specs: list[CandidateSpec] = []
    skipped_duplicate_pairs: list[dict[str, Any]] = []

    for row_index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, Mapping):
            blockers.append(_blocker("ungrounded_stream_value_evidence", "candidate row must be an object", row_index=row_index))
            continue
        candidate_id_raw = raw_candidate.get("candidate_id")
        if not isinstance(candidate_id_raw, str):
            blockers.append(_blocker("ungrounded_stream_value_evidence", "candidate_id must be a string", row_index=row_index))
            continue
        candidate_id = _safe_candidate_id(candidate_id_raw)
        if candidate_id in seen_candidate_ids:
            blockers.append(_blocker("duplicate_actions", "duplicate candidate_id in action evidence", candidate_id=candidate_id))
            continue
        seen_candidate_ids.add(candidate_id)
        planning_row = plan_by_id.get(candidate_id)
        if not isinstance(planning_row, Mapping):
            blockers.append(
                _blocker(
                    "ungrounded_stream_value_evidence",
                    "candidate is absent from transfer planning ranked_candidates",
                    candidate_id=candidate_id,
                )
            )
            continue
        if planning_row.get("can_feed_pair_action_evidence") is not True:
            blockers.append(
                _blocker(
                    "ungrounded_stream_value_evidence",
                    "planning row is not marked feedable pair-action evidence",
                    candidate_id=candidate_id,
                )
            )
        if planning_row.get("source_archive_sha256") != source_archive["archive_sha256"]:
            blockers.append(
                _blocker(
                    "source_sha_mismatch",
                    "planning row source archive SHA does not match selected source archive",
                    candidate_id=candidate_id,
                    expected=source_archive["archive_sha256"],
                    actual=planning_row.get("source_archive_sha256"),
                )
            )
        if planning_row.get("source_archive_bytes") != source_archive["archive_bytes"]:
            blockers.append(
                _blocker(
                    "source_sha_mismatch",
                    "planning row source archive bytes do not match selected source archive",
                    candidate_id=candidate_id,
                    expected=source_archive["archive_bytes"],
                    actual=planning_row.get("source_archive_bytes"),
                )
            )
        raw_actions = raw_candidate.get("actions")
        plan_actions = planning_row.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            blockers.append(_blocker("ungrounded_stream_value_evidence", "candidate has no actions", candidate_id=candidate_id))
            continue
        if not all(isinstance(item, Mapping) for item in raw_actions):
            blockers.append(_blocker("ungrounded_stream_value_evidence", "candidate actions must be objects", candidate_id=candidate_id))
            continue
        if not isinstance(plan_actions, list) or not all(isinstance(item, Mapping) for item in plan_actions):
            blockers.append(_blocker("ungrounded_stream_value_evidence", "planning row actions are missing", candidate_id=candidate_id))
            continue
        if not _candidate_actions_match_plan(raw_actions, plan_actions):
            blockers.append(
                _blocker(
                    "ungrounded_stream_value_evidence",
                    "pair-action evidence does not match transfer planning row",
                    candidate_id=candidate_id,
                )
            )
        parsed_actions: list[ActionSpec] = []
        for raw_action in raw_actions:
            action, action_blockers = _parse_action_spec(
                raw=raw_action,
                planning_row=planning_row,
                expected_source_artifact_sha256=expected_source_artifact_sha,
            )
            blockers.extend({**blocker, "candidate_id": candidate_id} for blocker in action_blockers)
            if action is not None:
                key = (action.pair_index, action.stream, action.source_value, action.value)
                if key in seen_actions:
                    blockers.append(
                        _blocker(
                            "duplicate_actions",
                            "duplicate stream/pair/source/value action in evidence",
                            candidate_id=candidate_id,
                            action_key=list(key),
                        )
                    )
                seen_actions.add(key)
                parsed_actions.append(action)
        if not parsed_actions:
            continue
        candidate_pairs = sorted({action.pair_index for action in parsed_actions})
        if len(candidate_pairs) != 1:
            blockers.append(
                _blocker(
                    "duplicate_actions",
                    "candidate must target exactly one pair for unique-pair top-3 selection",
                    candidate_id=candidate_id,
                    pair_indices=candidate_pairs,
                )
            )
            continue
        pair_index = candidate_pairs[0]
        if pair_index in seen_pairs:
            skipped_duplicate_pairs.append(
                {
                    "candidate_id": candidate_id,
                    "pair_index": pair_index,
                    "reason": "later candidate for an already selected pair",
                }
            )
            continue
        seen_pairs.add(pair_index)
        if len(specs) >= int(max_candidates):
            continue
        header_mode = raw_candidate.get("header_mode", "explicit_30")
        if header_mode not in {"v5", "explicit_30"}:
            blockers.append(
                _blocker(
                    "ungrounded_stream_value_evidence",
                    "unsupported header_mode",
                    candidate_id=candidate_id,
                    header_mode=header_mode,
                )
            )
            continue
        rank = planning_row.get("rank")
        specs.append(
            CandidateSpec(
                candidate_id=candidate_id,
                header_mode=str(header_mode),
                rank=int(rank) if isinstance(rank, int) and not isinstance(rank, bool) else None,
                actions=tuple(parsed_actions),
                planning_row=planning_row,
                evidence_row=raw_candidate,
            )
        )

    report = {
        "path": _rel(evidence_json),
        "sha256": _sha256_file(evidence_json) if evidence_json.is_file() else None,
        "schema": evidence.get("schema"),
        "transfer_plan": {
            "path": _rel(transfer_plan_json),
            "sha256": _sha256_file(transfer_plan_json) if transfer_plan_json.is_file() else None,
            "schema": plan.get("schema"),
            "source_artifact_sha256": expected_source_artifact_sha,
        },
        "status": "passed" if not blockers else "blocked",
        "blocker_class": "none" if not blockers else blockers[0]["blocker_class"],
        "blockers": blockers,
        "candidate_count": len(raw_candidates),
        "selected_unique_pair_count": len(specs),
        "max_candidates": int(max_candidates),
        "selected_candidate_ids": [spec.candidate_id for spec in specs],
        "skipped_duplicate_pairs": skipped_duplicate_pairs,
    }
    return report, specs, blockers


def _choose_header_mode(
    *,
    source_format: str,
    requested: str,
) -> str:
    if requested == "v5":
        return "v5"
    if requested == "explicit_30":
        return "explicit_30"
    if source_format == "pr85_explicit_30byte_lengths":
        return "explicit_30"
    return "v5"


def _build_one_candidate(
    *,
    spec: CandidateSpec,
    source_archive: Mapping[str, Any],
    source_raw: bytes,
    out_dir: Path,
) -> dict[str, Any]:
    try:
        bundle = parse_pr85_bundle(source_raw)
    except Pr85BundleError as exc:
        return _candidate_blocked(spec.candidate_id, "source_bundle_parse_failed", str(exc))
    source_segments = {name: bytes(bundle.segments[name]) for name in SEGMENT_ORDER}
    candidate_segments = dict(source_segments)
    action_proofs: list[dict[str, Any]] = []
    transform_reports: list[dict[str, Any]] = []
    changed = False

    choice_streams = sorted({action.stream for action in spec.actions if action.stream in CHOICE_STREAMS})
    choice_values: dict[str, bytearray] = {}
    choice_reports: dict[str, dict[str, Any]] = {}
    for stream in choice_streams:
        try:
            values, report = _decode_choice_stream(stream, source_segments[stream])
        except QrgbTransferArchiveBuilderError as exc:
            return _candidate_blocked(spec.candidate_id, "source_stream_decode_failed", str(exc), stream=stream)
        choice_values[stream] = values
        choice_reports[stream] = report

    randmulti_groups: list[list[bytearray]] | None = None
    randmulti_report: dict[str, Any] | None = None
    if any(action.stream == "randmulti" for action in spec.actions):
        try:
            randmulti_groups, randmulti_report = _decode_randmulti_stream(source_segments["randmulti"])
        except QrgbTransferArchiveBuilderError as exc:
            return _candidate_blocked(spec.candidate_id, "source_stream_decode_failed", str(exc), stream="randmulti")

    for action in spec.actions:
        if action.stream in CHOICE_STREAMS:
            values = choice_values[action.stream]
            before = int(values[action.pair_index])
            if before != action.source_value:
                return _candidate_blocked(
                    spec.candidate_id,
                    "source_value_mismatch",
                    "decoded PR85 source value does not match action evidence",
                    stream=action.stream,
                    pair_index=action.pair_index,
                    expected=action.source_value,
                    actual=before,
                )
            if before == action.value:
                return _candidate_blocked(
                    spec.candidate_id,
                    "source_preserving_edit",
                    "action preserves decoded PR85 source value",
                    stream=action.stream,
                    pair_index=action.pair_index,
                    value=action.value,
                )
            values[action.pair_index] = action.value
            changed = True
            action_proofs.append(
                {
                    "pair_index": action.pair_index,
                    "stream": action.stream,
                    "source_value": before,
                    "candidate_value": action.value,
                    "changed": True,
                    "rationale": action.rationale,
                    "source_artifact_sha256": action.source_artifact_sha256,
                    "basis_action_schema": dict(action.basis_action_schema),
                }
            )
        elif action.stream == "randmulti":
            assert randmulti_groups is not None
            try:
                group_index, row_index, target = _randmulti_target(action)
            except QrgbTransferArchiveBuilderError as exc:
                return _candidate_blocked(
                    spec.candidate_id,
                    "ungrounded_stream_value_evidence",
                    str(exc),
                    stream="randmulti",
                    pair_index=action.pair_index,
                )
            values = randmulti_groups[group_index][row_index]
            before = int(values[action.pair_index])
            if before != action.source_value:
                return _candidate_blocked(
                    spec.candidate_id,
                    "source_value_mismatch",
                    "decoded PR85 randmulti source value does not match action evidence",
                    stream="randmulti",
                    group_index=group_index,
                    row_index=row_index,
                    pair_index=action.pair_index,
                    expected=action.source_value,
                    actual=before,
                )
            if before == action.value:
                return _candidate_blocked(
                    spec.candidate_id,
                    "source_preserving_edit",
                    "randmulti action preserves decoded PR85 source value",
                    group_index=group_index,
                    row_index=row_index,
                    pair_index=action.pair_index,
                    value=action.value,
                )
            values[action.pair_index] = action.value
            changed = True
            action_proofs.append(
                {
                    "pair_index": action.pair_index,
                    "stream": "randmulti",
                    "source_value": before,
                    "candidate_value": action.value,
                    "changed": True,
                    "rationale": action.rationale,
                    "source_artifact_sha256": action.source_artifact_sha256,
                    "basis_action_schema": dict(action.basis_action_schema),
                    "randmulti_target": target,
                }
            )
        else:  # pragma: no cover - guarded by parser
            return _candidate_blocked(spec.candidate_id, "unsupported_stream", "unsupported action stream", stream=action.stream)

    if not changed:
        return _candidate_blocked(
            spec.candidate_id,
            "non_noop_proof_failed",
            "every explicit action preserved the source semantic value",
        )

    for stream in choice_streams:
        source_segment = source_segments[stream]
        try:
            encoded, encode_meta = _encode_choice_stream(
                stream,
                bytes(choice_values[stream]),
                codec=choice_reports[stream]["codec"],
            )
        except QrgbTransferArchiveBuilderError as exc:
            return _candidate_blocked(spec.candidate_id, "candidate_stream_encode_failed", str(exc), stream=stream)
        candidate_segments[stream] = encoded
        transform_reports.append(
            {
                **choice_reports[stream],
                **encode_meta,
                "segment_byte_delta": len(encoded) - len(source_segment),
                "semantic_sha256_before": _sha256_bytes(bytes(_decode_choice_stream(stream, source_segment)[0])),
                "semantic_sha256_after": _sha256_bytes(bytes(choice_values[stream])),
                "changed_pair_indices": sorted(
                    {action.pair_index for action in spec.actions if action.stream == stream}
                ),
            }
        )

    if randmulti_groups is not None:
        source_segment = source_segments["randmulti"]
        assert randmulti_report is not None
        try:
            encoded, encode_meta = _encode_randmulti_stream(randmulti_groups)
        except QrgbTransferArchiveBuilderError as exc:
            return _candidate_blocked(spec.candidate_id, "candidate_stream_encode_failed", str(exc), stream="randmulti")
        candidate_segments["randmulti"] = encoded
        transform_reports.append(
            {
                **randmulti_report,
                **encode_meta,
                "segment_byte_delta": len(encoded) - len(source_segment),
                "changed_pair_indices": sorted(
                    {action.pair_index for action in spec.actions if action.stream == "randmulti"}
                ),
                "changed_groups": [
                    action_proof.get("randmulti_target", {}).get("group_index")
                    for action_proof in action_proofs
                    if action_proof.get("stream") == "randmulti"
                ],
            }
        )

    header_mode = _choose_header_mode(source_format=bundle.format, requested=spec.header_mode)
    if header_mode == "v5":
        changed_fixed = [
            name for name in FIXED_V5_LENGTHS if len(candidate_segments[name]) != FIXED_V5_LENGTHS[name]
        ]
        if changed_fixed:
            return _candidate_blocked(
                spec.candidate_id,
                "fixed_v5_segment_length_changed",
                "v5 PR85 header cannot encode changed fixed-length bias/region segment sizes",
                changed_fixed_segments=changed_fixed,
            )
    try:
        payload = pack_pr85_bundle(candidate_segments, header_mode=header_mode)
        parsed = parse_pr85_bundle(payload)
    except Pr85BundleError as exc:
        return _candidate_blocked(spec.candidate_id, "candidate_bundle_validation_failed", str(exc))
    if {name: bytes(parsed.segments[name]) for name in SEGMENT_ORDER} != candidate_segments:
        return _candidate_blocked(
            spec.candidate_id,
            "candidate_bundle_validation_failed",
            "reparsed segments do not match candidate segments",
        )

    payload_changed = _sha256_bytes(source_raw) != _sha256_bytes(payload)
    changed_segments = [name for name in SEGMENT_ORDER if candidate_segments[name] != source_segments[name]]
    if not payload_changed or not changed_segments:
        return _candidate_blocked(
            spec.candidate_id,
            "non_noop_proof_failed",
            "payload or decoded side-channel semantics did not change",
        )

    candidate_dir = out_dir / spec.candidate_id
    archive_path = candidate_dir / "archive.zip"
    _write_single_member_archive(archive_path, payload)
    candidate_archive = _archive_info(archive_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "candidate_id": spec.candidate_id,
        "rank": spec.rank,
        "build_status": "built",
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "evidence_grade": "empirical_local_archive_build_only",
        "source_archive": source_archive,
        "candidate_archive": candidate_archive,
        "source_bundle": {
            "format": bundle.format,
            "header_bytes": bundle.header_bytes,
            "segment_lengths": bundle.segment_lengths,
            "fixed_length_segments": dict(bundle.fixed_length_segments),
            "member_sha256": _sha256_bytes(source_raw),
        },
        "candidate_bundle": {
            "header_mode": header_mode,
            "member_bytes": len(payload),
            "member_sha256": _sha256_bytes(payload),
            "segment_lengths": {name: len(candidate_segments[name]) for name in SEGMENT_ORDER},
        },
        "selected_pair_indices": sorted({action.pair_index for action in spec.actions}),
        "selected_streams": sorted({action.stream for action in spec.actions}),
        "action_proofs": action_proofs,
        "transforms": transform_reports,
        "changed_segments": changed_segments,
        "charged_bytes": {
            "candidate_archive_bytes": candidate_archive["archive_bytes"],
            "source_archive_bytes": source_archive["archive_bytes"],
            "byte_delta_vs_source_archive": int(candidate_archive["archive_bytes"] - source_archive["archive_bytes"]),
            "formula_only_rate_score_delta_vs_source": (
                int(candidate_archive["archive_bytes"] - source_archive["archive_bytes"]) * RATE_SCORE_PER_BYTE
            ),
        },
        "grounding": {
            "planning_candidate_id": spec.planning_row.get("candidate_id"),
            "planning_rank": spec.planning_row.get("rank"),
            "pr90_source_archive_sha256": spec.planning_row.get("pr90_source_archive_sha256"),
            "pr90_source_evidence_sha256": spec.planning_row.get("pr90_source_evidence_sha256"),
            "source_atom_id": spec.planning_row.get("source_atom_id"),
            "expected_break_even": spec.planning_row.get("expected_break_even"),
            "qrgb_pair_summary": spec.planning_row.get("qrgb_pair_summary"),
        },
        "non_noop_proof": {
            "status": "passed",
            "payload_changed": payload_changed,
            "decoded_sidechannel_semantics_changed": True,
            "changed_segments": changed_segments,
            "source_member_sha256": _sha256_bytes(source_raw),
            "candidate_member_sha256": _sha256_bytes(payload),
        },
        "fixed_runtime_preflight": None,
        "dispatch_unlocked": False,
        "dispatch_gate": "blocked_fixed_runtime_preflight_not_run",
        "lane_claim_required_before_exact_eval": True,
        "next_gate": "Run/pass fixed-runtime preflight, then claim the lane before exact CUDA auth eval.",
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def _run_fixed_preflight(
    *,
    manifest: Mapping[str, Any],
    source_archive_path: Path,
    robust_current_dir: Path,
    write_outputs: bool,
) -> dict[str, Any]:
    archive = manifest.get("candidate_archive", {}) if isinstance(manifest.get("candidate_archive"), Mapping) else {}
    archive_path = REPO_ROOT / str(archive.get("archive_path"))
    payload = fixed_preflight.build_preflight(
        archive_path,
        robust_current_dir,
        atom_source_archive=source_archive_path,
        expected_archive_sha256=archive.get("archive_sha256"),
        expected_member_sha256=archive.get("member_sha256"),
    )
    out_path = archive_path.parent / "fixed_runtime_preflight.json"
    if write_outputs:
        _write_json(out_path, payload)
        manifest_path = archive_path.parent / "manifest.json"
        updated = dict(manifest)
        ready = payload.get("ready_for_fixed_runtime_exact_eval") is True
        updated["fixed_runtime_preflight"] = {
            "path": _rel(out_path),
            "ready_for_fixed_runtime_exact_eval": ready,
            "readiness_status": payload.get("readiness_status"),
            "blocker_count": len(payload.get("blockers", [])),
        }
        updated["dispatch_unlocked"] = bool(ready)
        updated["dispatch_gate"] = (
            "eligible_for_exact_eval_after_lane_claim"
            if ready
            else "blocked_fixed_runtime_preflight"
        )
        updated["next_gate"] = (
            "Claim the lane with tools/claim_lane_dispatch.py before any exact CUDA auth eval dispatch."
            if ready
            else "Resolve fixed-runtime preflight blockers before any lane claim or exact eval."
        )
        if not ready:
            updated["blockers"] = list(updated.get("blockers", [])) + [
                _blocker(
                    "fixed_runtime_preflight_failed",
                    "fixed-runtime readiness preflight is not ready",
                    preflight_path=_rel(out_path),
                )
            ]
            updated["blocker_class"] = "fixed_runtime_preflight_failed"
        _write_json(manifest_path, updated)
    return {
        "path": _rel(out_path),
        "ready_for_fixed_runtime_exact_eval": payload.get("ready_for_fixed_runtime_exact_eval"),
        "readiness_status": payload.get("readiness_status"),
        "blocker_count": len(payload.get("blockers", [])),
        "archive_sha256": archive.get("archive_sha256"),
    }


def build_qrgb_transfer_archive_candidates(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    pair_action_evidence_json: Path = DEFAULT_PAIR_ACTION_EVIDENCE,
    transfer_plan_json: Path = DEFAULT_TRANSFER_PLAN,
    out_dir: Path = DEFAULT_OUT_DIR,
    robust_current_dir: Path = DEFAULT_ROBUST_CURRENT,
    max_candidates: int = 3,
    run_fixed_runtime_preflight: bool = True,
    require_known_pr85_anchor: bool = True,
    write_outputs: bool = True,
) -> dict[str, Any]:
    source, source_raw = _read_source_archive(source_archive)
    global_blockers: list[dict[str, Any]] = []
    if require_known_pr85_anchor and not source["known_pr85_anchor_match"]["matches"]:
        global_blockers.append(
            _blocker(
                "source_sha_mismatch",
                "selected source archive is not the known PR85 anchor",
                expected=KNOWN_PR85,
                actual={
                    "archive_bytes": source["archive_bytes"],
                    "archive_sha256": source["archive_sha256"],
                },
            )
        )
    evidence_report, specs, evidence_blockers = _load_candidate_specs(
        evidence_json=pair_action_evidence_json,
        transfer_plan_json=transfer_plan_json,
        source_archive=source,
        max_candidates=max_candidates,
    )
    global_blockers.extend(evidence_blockers)

    candidates: list[dict[str, Any]] = []
    if global_blockers:
        for spec in specs:
            candidates.append(
                _candidate_blocked(
                    spec.candidate_id,
                    str(global_blockers[0]["blocker_class"]),
                    str(global_blockers[0]["reason"]),
                )
            )
    else:
        for spec in specs:
            candidates.append(
                _build_one_candidate(
                    spec=spec,
                    source_archive=source,
                    source_raw=source_raw,
                    out_dir=out_dir,
                )
            )

    preflight_reports: dict[str, dict[str, Any]] = {}
    if run_fixed_runtime_preflight:
        for manifest in candidates:
            if manifest.get("build_status") != "built":
                continue
            report = _run_fixed_preflight(
                manifest=manifest,
                source_archive_path=source_archive,
                robust_current_dir=robust_current_dir,
                write_outputs=write_outputs,
            )
            preflight_reports[str(manifest["candidate_id"])] = report
            manifest["fixed_runtime_preflight"] = report
            manifest["dispatch_unlocked"] = report.get("ready_for_fixed_runtime_exact_eval") is True
            manifest["dispatch_gate"] = (
                "eligible_for_exact_eval_after_lane_claim"
                if manifest["dispatch_unlocked"]
                else "blocked_fixed_runtime_preflight"
            )
    else:
        for manifest in candidates:
            if manifest.get("build_status") == "built":
                manifest["dispatch_unlocked"] = False
                manifest["dispatch_gate"] = "blocked_fixed_runtime_preflight_not_run"
                manifest["blockers"] = list(manifest.get("blockers", [])) + [
                    _blocker("missing_preflight", "fixed-runtime preflight was not run")
                ]

    built = [row for row in candidates if row.get("build_status") == "built"]
    ready = [row for row in built if row.get("dispatch_unlocked") is True]
    candidate_blockers = [
        blocker
        for row in candidates
        for blocker in row.get("blockers", [])
        if isinstance(blocker, Mapping)
    ]
    all_blockers = global_blockers + candidate_blockers
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "scorer_load_performed": False,
        "sidecars_required": False,
        "source_archive": source,
        "pair_action_evidence": evidence_report,
        "max_candidates": int(max_candidates),
        "selected_unique_pair_count": len(specs),
        "candidate_attempt_count": len(candidates),
        "candidate_archive_count": len(built),
        "ready_candidate_count": len(ready),
        "fixed_runtime_preflight_count": len(preflight_reports),
        "dispatch_unlocked": bool(ready),
        "dispatch_gate": (
            "eligible_for_exact_eval_after_lane_claim"
            if ready
            else "blocked_until_non_noop_and_fixed_runtime_preflight_pass"
        ),
        "lane_claim_required_before_exact_eval": True,
        "blocker_class": "none" if not all_blockers else str(all_blockers[0]["blocker_class"]),
        "blockers": all_blockers,
        "candidates": candidates,
        "planning_json_path": _rel(out_dir / "planning.json"),
        "reactivation_criteria": [
            "Exact CUDA auth eval requires a fresh lane claim in .omx/state/active_lane_dispatch_claims.md.",
            "Candidate promotion requires archive.zip -> inflate.sh -> upstream/evaluate.py CUDA evidence for the exact archive SHA.",
            "This builder's local readiness does not claim score or component improvement.",
        ],
    }
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "planning.json", summary)
    return summary


def render_ledger(summary: Mapping[str, Any]) -> str:
    source = summary.get("source_archive", {}) if isinstance(summary.get("source_archive"), Mapping) else {}
    lines = [
        "# PR85 QRGB Transfer Archive Candidates - 2026-05-04",
        "",
        "- tool: `experiments/build_pr85_qrgb_transfer_archive_candidates.py`",
        "- score_claim: false",
        "- dispatch_performed: false",
        "- remote_jobs_dispatched: false",
        f"- dispatch_unlocked: {str(summary.get('dispatch_unlocked')).lower()}",
        f"- blocker_class: `{summary.get('blocker_class')}`",
        "",
        "## Source Anchor",
        "",
        f"- PR85 source archive bytes: `{source.get('archive_bytes')}`",
        f"- PR85 source archive sha256: `{source.get('archive_sha256')}`",
        f"- known PR85 anchor match: {source.get('known_pr85_anchor_match', {}).get('matches')}",
        "",
        "## Build Scope",
        "",
        f"- max candidates: `{summary.get('max_candidates')}`",
        f"- selected unique pairs: `{summary.get('selected_unique_pair_count')}`",
        f"- archive candidates built: `{summary.get('candidate_archive_count')}`",
        f"- fixed-runtime preflights: `{summary.get('fixed_runtime_preflight_count')}`",
        f"- ready after lane claim: `{summary.get('ready_candidate_count')}`",
        "",
        "## Candidate Artifacts",
        "",
    ]
    candidates = summary.get("candidates", [])
    if not candidates:
        lines.append("- none")
    for row in candidates:
        if not isinstance(row, Mapping):
            continue
        archive = row.get("candidate_archive", {}) if isinstance(row.get("candidate_archive"), Mapping) else {}
        preflight = row.get("fixed_runtime_preflight", {}) if isinstance(row.get("fixed_runtime_preflight"), Mapping) else {}
        lines.extend(
            [
                f"- `{row.get('candidate_id')}`",
                f"  - build_status: `{row.get('build_status')}`",
                f"  - dispatch_unlocked: {str(row.get('dispatch_unlocked')).lower()}",
                f"  - archive: `{archive.get('archive_path')}`",
                f"  - bytes: `{archive.get('archive_bytes')}`",
                f"  - sha256: `{archive.get('archive_sha256')}`",
                f"  - changed_segments: `{row.get('changed_segments')}`",
                f"  - fixed_runtime_ready: `{preflight.get('ready_for_fixed_runtime_exact_eval')}`",
            ]
        )
    lines.extend(["", "## Blockers", ""])
    blockers = summary.get("blockers", [])
    if not blockers:
        lines.append("- none")
    for blocker in blockers:
        if isinstance(blocker, Mapping):
            lines.append(f"- `{blocker.get('blocker_class')}`: {blocker.get('reason')}")
    lines.extend(
        [
            "",
            "## Compliance Notes",
            "",
            "- Local archive builds only; no training, scorer import, GPU dispatch, or score claim.",
            "- Archives remain single-member `x` PR85 bundles and are byte-closed.",
            "- Exact eval is allowed only after a lane claim and only through canonical CUDA auth eval.",
            "",
        ]
    )
    return "\n".join(lines)


def write_ledger(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ledger(summary), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--pair-action-evidence-json", type=Path, default=DEFAULT_PAIR_ACTION_EVIDENCE)
    parser.add_argument("--transfer-plan-json", type=Path, default=DEFAULT_TRANSFER_PLAN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--robust-current-dir", type=Path, default=DEFAULT_ROBUST_CURRENT)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--skip-fixed-runtime-preflight", action="store_true")
    parser.add_argument("--allow-synthetic-source", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_qrgb_transfer_archive_candidates(
        source_archive=args.source_archive,
        pair_action_evidence_json=args.pair_action_evidence_json,
        transfer_plan_json=args.transfer_plan_json,
        out_dir=args.out_dir,
        robust_current_dir=args.robust_current_dir,
        max_candidates=args.max_candidates,
        run_fixed_runtime_preflight=not args.skip_fixed_runtime_preflight,
        require_known_pr85_anchor=not args.allow_synthetic_source,
    )
    write_ledger(args.ledger_md, summary)
    if args.stdout:
        sys.stdout.write(_json_text(summary))
    else:
        print(
            _json_text(
                {
                    "planning_json": summary["planning_json_path"],
                    "candidate_archive_count": summary["candidate_archive_count"],
                    "ready_candidate_count": summary["ready_candidate_count"],
                    "dispatch_unlocked": summary["dispatch_unlocked"],
                    "blocker_class": summary["blocker_class"],
                }
            ),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
