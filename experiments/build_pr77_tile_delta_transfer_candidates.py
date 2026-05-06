#!/usr/bin/env python3
"""Build PR77 tile-delta action-transfer candidates on PR79 S2 archives.

This is a deterministic local profiler/builder only. It converts the public
PR77 ``qzs3_tile_delta_r147`` decoded tile-action anatomy into S2 action-wire
hypotheses on top of PR79/S2 mask, renderer, and pose streams. It writes a
candidate only after robust-current payload parsing proves archive closure and
after the decoded action record set is proven not to be a PR79 no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.archive_byte_profile import profile_archive
from tac.submission_archive import validate_seg_tile_actions_payload


S2_BUILDER_PATH = REPO_ROOT / "experiments/build_pr79_action_dictionary_repack_candidates_v2.py"
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_PR77_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip"
)
DEFAULT_PR79_S2_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/"
    "pr79_s2_fixed_adaptive_actions/archive.zip"
)
DEFAULT_PR79_S2_EVAL = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr79_s2_fixed_adaptive_actions_t4_20260503T173023Z/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_RAW_OUTPUT_PARITY = (
    REPO_ROOT
    / "experiments/results/top_submission_reverse_engineering_20260503_pr79/"
    "raw_output_parity_all600_cpu_codex_20260503T1728Z/pr75_raw_output_parity.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr77_tile_delta_transfer_20260503_worker"
)

TOOL = "experiments/build_pr77_tile_delta_transfer_candidates.py"
SCHEMA = "pr77_tile_delta_transfer_candidate_matrix_v1"
MANIFEST_SCHEMA = "pr77_tile_delta_transfer_candidate_manifest_v1"
PAYLOAD_MEMBER = "p"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SEGMENTS = ("masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.qp1")
NON_ACTION_SEGMENTS = ("masks.mkv", "renderer.bin", "optimized_poses.qp1")
RATE_DENOMINATOR = 37_545_489
RATE_NUMERATOR = 25
PR77_EXPECTED_ACTION_WIRE_BYTES = 325
PR77_EXPECTED_RECORD_COUNT = 147
PR79_S2_SCORE = 0.31453355357318635
PR79_S2_BYTES = 277_321
CUDA_AUTH_EVAL_REQUIRED = (
    "No GPU dispatch or score claim from this worker. Before any exact eval, "
    "claim a non-conflicting lane with tools/claim_lane_dispatch.py claim and "
    "run archive.zip -> inflate.sh -> upstream/evaluate.py through "
    "experiments/contest_auth_eval.py --device cuda on the identical archive "
    "SHA-256 and byte count."
)


class CandidateBuildError(ValueError):
    """Raised when a candidate is not local-closed or is a no-op."""


@dataclass(frozen=True)
class ActionRecord:
    pair_index: int
    tile_id: int
    action_id: int
    source_index: int
    source_label: str

    @property
    def key(self) -> tuple[int, int, int]:
        return (int(self.pair_index), int(self.tile_id), int(self.action_id))

    def encode4(self) -> bytes:
        return (
            int(self.pair_index).to_bytes(2, "little")
            + bytes([int(self.tile_id), int(self.action_id)])
        )


@dataclass(frozen=True)
class LoadedArchive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str
    raw_segments: dict[str, bytes]
    decoded: dict[str, bytes]
    runtime_members: dict[str, dict[str, Any]]


def _load_s2_builder() -> Any:
    spec = importlib.util.spec_from_file_location("pr77_transfer_s2_builder", S2_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load S2 builder from {S2_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S2 = _load_s2_builder()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
        raise CandidateBuildError(f"unsafe ZIP member path: {name!r}")
    return name


def _read_single_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        names: list[str] = []
        seen: set[str] = set()
        for info in zf.infolist():
            name = _safe_member_name(info.filename)
            if name in seen:
                raise CandidateBuildError(f"duplicate ZIP member: {name}")
            seen.add(name)
            if not info.is_dir():
                names.append(name)
        if names != [PAYLOAD_MEMBER]:
            raise CandidateBuildError(
                f"{path} must contain exactly member {PAYLOAD_MEMBER!r}; got {names!r}"
            )
        return zf.read(PAYLOAD_MEMBER)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(PAYLOAD_MEMBER), payload)


def _member_summary(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    members = header.get("members")
    if not isinstance(members, list):
        raise CandidateBuildError("runtime parser returned no member table")
    out: dict[str, dict[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping):
            raise CandidateBuildError("runtime parser returned malformed member metadata")
        name = str(item["name"])
        out[name] = {
            "bytes": int(item["bytes"]),
            "codec": str(item["codec"]),
            "decoded_bytes": int(item["decoded_bytes"]),
            "decoded_sha256": str(item["decoded_sha256"]),
            "sha256": str(item["sha256"]),
        }
    return out


def _parse_self_describing_slices(payload: bytes) -> dict[str, bytes] | None:
    if payload.startswith(b"P3"):
        header_size = 2 + struct.calcsize("<IHH")
        if len(payload) <= header_size:
            raise CandidateBuildError("P3 payload is too short")
        mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        cursor = header_size
    elif payload.startswith(b"P6"):
        header_size = 2 + struct.calcsize("<IHHH")
        if len(payload) <= header_size:
            raise CandidateBuildError("P6 payload is too short")
        mask_len, renderer_len, actions_len, _record_count = struct.unpack_from("<IHHH", payload, 2)
        cursor = header_size
    else:
        return None
    if min(mask_len, renderer_len, actions_len) <= 0:
        raise CandidateBuildError("self-describing payload has an empty stream")
    mask_end = cursor + mask_len
    renderer_end = mask_end + renderer_len
    actions_end = renderer_end + actions_len
    if actions_end >= len(payload):
        raise CandidateBuildError("self-describing lengths leave no pose stream")
    return {
        "masks.mkv": payload[cursor:mask_end],
        "renderer.bin": payload[mask_end:renderer_end],
        "seg_tile_actions.bin": payload[renderer_end:actions_end],
        "optimized_poses.qp1": payload[actions_end:],
    }


def _slice_fixed_payload(
    *,
    label: str,
    payload: bytes,
    runtime_members: Mapping[str, Mapping[str, Any]],
) -> dict[str, bytes]:
    missing = sorted(set(SEGMENTS) - set(runtime_members))
    if missing:
        raise CandidateBuildError(f"{label}: missing runtime members {missing}")
    offset = 0
    raw_segments: dict[str, bytes] = {}
    for name in SEGMENTS:
        size = int(runtime_members[name]["bytes"])
        if size <= 0:
            raise CandidateBuildError(f"{label}: non-positive raw size for {name}: {size}")
        raw = payload[offset : offset + size]
        offset += size
        if len(raw) != size:
            raise CandidateBuildError(f"{label}: truncated fixed segment {name}")
        expected_sha = str(runtime_members[name].get("sha256", ""))
        if expected_sha and _sha256_bytes(raw) != expected_sha:
            raise CandidateBuildError(f"{label}: raw SHA mismatch for {name}")
        raw_segments[name] = raw
    if offset != len(payload):
        raise CandidateBuildError(
            f"{label}: fixed slices consume {offset}, payload has {len(payload)}"
        )
    return raw_segments


def _load_archive(label: str, path: Path, *, unpacker: Any) -> LoadedArchive:
    path = path.resolve()
    payload = _read_single_payload(path)
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    runtime_members = _member_summary(header)
    raw_segments = _parse_self_describing_slices(payload)
    if raw_segments is None:
        raw_segments = _slice_fixed_payload(
            label=label,
            payload=payload,
            runtime_members=runtime_members,
        )
    missing = sorted(set(SEGMENTS) - set(decoded))
    if missing:
        raise CandidateBuildError(f"{label}: runtime parser missed members {missing}")
    validate_seg_tile_actions_payload(
        decoded["seg_tile_actions.bin"],
        source_name=f"{label} decoded seg_tile_actions.bin",
    )
    return LoadedArchive(
        label=label,
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=str(header.get("payload_format")),
        raw_segments=raw_segments,
        decoded=decoded,
        runtime_members=runtime_members,
    )


def _parse_records(raw4: bytes, *, source_label: str) -> list[ActionRecord]:
    validation = validate_seg_tile_actions_payload(
        raw4,
        source_name=f"{source_label} decoded seg_tile_actions.bin",
    )
    if validation["record_size"] != 4:
        raise CandidateBuildError(f"{source_label}: expected raw4 runtime action records")
    records: list[ActionRecord] = []
    for offset in range(0, len(raw4), 4):
        records.append(
            ActionRecord(
                pair_index=int.from_bytes(raw4[offset : offset + 2], "little"),
                tile_id=int(raw4[offset + 2]),
                action_id=int(raw4[offset + 3]),
                source_index=offset // 4,
                source_label=source_label,
            )
        )
    if not records:
        raise CandidateBuildError(f"{source_label}: decoded no action records")
    return records


def _records_raw4(records: Sequence[ActionRecord]) -> bytes:
    return b"".join(record.encode4() for record in records)


def _canonical_record_bytes(records: Iterable[ActionRecord]) -> bytes:
    return b"".join(
        int(pair).to_bytes(2, "little") + bytes([tile, action])
        for pair, tile, action in sorted(record.key for record in records)
    )


def _record_counter(records: Iterable[ActionRecord]) -> Counter[tuple[int, int, int]]:
    return Counter(record.key for record in records)


def _dedupe_records(records: Iterable[ActionRecord]) -> list[ActionRecord]:
    out: list[ActionRecord] = []
    seen: set[tuple[int, int, int]] = set()
    for record in records:
        if record.key in seen:
            continue
        seen.add(record.key)
        out.append(record)
    return out


def _record_stats(records: Sequence[ActionRecord]) -> dict[str, Any]:
    pair_tile_counts = Counter((record.pair_index, record.tile_id) for record in records)
    exact_counts = _record_counter(records)
    return {
        "action_max": max(record.action_id for record in records),
        "action_min": min(record.action_id for record in records),
        "duplicate_exact_record_count": sum(count - 1 for count in exact_counts.values() if count > 1),
        "duplicate_pair_tile_group_count": sum(1 for count in pair_tile_counts.values() if count > 1),
        "nondecreasing_pair_order": all(
            records[index].pair_index <= records[index + 1].pair_index
            for index in range(len(records) - 1)
        ),
        "pair_max": max(record.pair_index for record in records),
        "pair_min": min(record.pair_index for record in records),
        "record_count": len(records),
        "records_canonical_sha256": _sha256_bytes(_canonical_record_bytes(records)),
        "records_source_order_sha256": _sha256_bytes(_records_raw4(records)),
        "tile_max": max(record.tile_id for record in records),
        "tile_min": min(record.tile_id for record in records),
        "unique_action_count": len({record.action_id for record in records}),
        "unique_pair_count": len({record.pair_index for record in records}),
        "unique_pair_tile_count": len(pair_tile_counts),
        "unique_tile_count": len({record.tile_id for record in records}),
    }


def _select_policy(
    policy: str,
    *,
    pr77_records: Sequence[ActionRecord],
    pr79_records: Sequence[ActionRecord],
) -> tuple[list[ActionRecord], dict[str, Any]]:
    pr79_pairs = {record.pair_index for record in pr79_records}
    pr79_tiles = {record.tile_id for record in pr79_records}
    pr79_keys = {record.key for record in pr79_records}
    if policy == "replace_pr79_with_pr77_all":
        selected = list(pr77_records)
        rationale = "Replace PR79 action semantics with all 147 PR77 tile-delta records."
    elif policy == "replace_pr79_with_pr77_pair_overlap":
        selected = [record for record in pr77_records if record.pair_index in pr79_pairs]
        rationale = "Transfer only PR77 records on pairs already touched by PR79 actions."
    elif policy == "replace_pr79_with_pr77_tile_overlap":
        selected = [record for record in pr77_records if record.tile_id in pr79_tiles]
        rationale = "Transfer only PR77 records on tiles already touched by PR79 actions."
    elif policy == "augment_pr79_with_pr77_unique":
        selected = _dedupe_records([*pr79_records, *(r for r in pr77_records if r.key not in pr79_keys)])
        rationale = "Keep PR79 actions and add exact-key-unique PR77 tile-delta records."
    else:
        raise CandidateBuildError(f"unsupported transfer policy: {policy}")
    if not selected:
        raise CandidateBuildError(f"{policy}: selected no records")
    return selected, {
        "policy": policy,
        "rationale": rationale,
        "selected_record_count": len(selected),
        "source_pr77_record_count": len(pr77_records),
        "source_pr79_record_count": len(pr79_records),
    }


def _build_p3_payload(base: LoadedArchive, action_wire: bytes) -> bytes:
    return (
        b"P3"
        + struct.pack(
            "<IHH",
            len(base.raw_segments["masks.mkv"]),
            len(base.raw_segments["renderer.bin"]),
            len(action_wire),
        )
        + base.raw_segments["masks.mkv"]
        + base.raw_segments["renderer.bin"]
        + action_wire
        + base.raw_segments["optimized_poses.qp1"]
    )


def _source_summary(source: LoadedArchive) -> dict[str, Any]:
    return {
        "archive_bytes": source.archive_bytes,
        "archive_path": _repo_rel(source.path),
        "archive_sha256": source.archive_sha256,
        "decoded_segments": {
            name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
            for name, data in sorted(source.decoded.items())
            if name in SEGMENTS
        },
        "payload_bytes": len(source.payload),
        "payload_format": source.payload_format,
        "payload_sha256": source.payload_sha256,
        "raw_segments": {
            name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
            for name, data in sorted(source.raw_segments.items())
            if name in SEGMENTS
        },
    }


def _load_s2_eval(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "archive_bytes": PR79_S2_BYTES,
            "path": _repo_rel(path) if path else None,
            "score": PR79_S2_SCORE,
            "status": "fallback_constants",
        }
    payload = json.loads(path.read_text())
    return {
        "archive_bytes": int(payload.get("archive_size_bytes", PR79_S2_BYTES)),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "n_samples": payload.get("n_samples"),
        "path": _repo_rel(path),
        "score": float(payload.get("score_recomputed_from_components", PR79_S2_SCORE)),
        "score_pose_contribution": payload.get("score_pose_contribution"),
        "score_rate_contribution": payload.get("score_rate_contribution"),
        "score_seg_contribution": payload.get("score_seg_contribution"),
        "status": "loaded",
    }


def _break_even_vs_s2(*, archive_bytes: int, s2_eval: Mapping[str, Any]) -> dict[str, Any]:
    base_bytes = int(s2_eval.get("archive_bytes", PR79_S2_BYTES))
    base_score = float(s2_eval.get("score", PR79_S2_SCORE))
    delta_bytes = int(archive_bytes) - base_bytes
    rate_fraction = Fraction(RATE_NUMERATOR * delta_bytes, RATE_DENOMINATOR)
    rate_delta = float(rate_fraction)
    score_if_components_unchanged = base_score + rate_delta
    break_even_component_delta = -rate_delta
    return {
        "archive_delta_bytes_vs_pr79_s2": delta_bytes,
        "baseline_archive_bytes": base_bytes,
        "baseline_score": base_score,
        "break_even_component_delta_vs_pr79_s2": break_even_component_delta,
        "break_even_component_delta_vs_pr79_s2_fraction": (
            f"{-rate_fraction.numerator}/{rate_fraction.denominator}"
        ),
        "max_component_regression_to_tie_s2": max(0.0, break_even_component_delta),
        "required_component_improvement_to_tie_s2": max(0.0, -break_even_component_delta),
        "rate_score_delta_vs_pr79_s2": rate_delta,
        "rate_score_delta_vs_pr79_s2_fraction": (
            f"{rate_fraction.numerator}/{rate_fraction.denominator}"
        ),
        "score_if_components_unchanged": score_if_components_unchanged,
        "would_beat_pr79_s2_if_components_unchanged": score_if_components_unchanged < base_score,
    }


def _raw_output_proof(path: Path | None, *, candidate_id: str) -> dict[str, Any]:
    if path is None:
        return {
            "candidate_raw_output_proof_available": False,
            "status": "not_requested",
        }
    if not path.exists():
        return {
            "candidate_raw_output_proof_available": False,
            "path": _repo_rel(path),
            "status": "missing",
        }
    payload = json.loads(path.read_text())
    all_pairs = payload.get("all_pairs_parity", {})
    render = all_pairs.get("render_public_vs_robust_current", {})
    raw_after = render.get("raw_after_actions", {}) if isinstance(render, Mapping) else {}
    return {
        "candidate_id": candidate_id,
        "candidate_raw_output_proof_available": False,
        "candidate_raw_output_proof_reason": (
            "candidate changes decoded seg_tile_actions.bin; no local raw-output render was run"
        ),
        "source_pr79_parity_artifact": {
            "all_pairs_completed": all_pairs.get("completed"),
            "device": payload.get("device"),
            "evidence_grade": payload.get("evidence_grade"),
            "path": _repo_rel(path),
            "raw_after_actions_exact_equal": raw_after.get("exact_equal"),
            "raw_after_actions_sha256": raw_after.get("lhs_sha256"),
            "score_claim": payload.get("score_claim"),
            "schema": payload.get("schema"),
        },
        "status": "source_baseline_only",
    }


def _changed_member_summary(*, base: LoadedArchive, candidate: LoadedArchive) -> dict[str, Any]:
    logical: dict[str, dict[str, Any]] = {}
    changed = []
    for name in SEGMENTS:
        raw_equal = base.raw_segments[name] == candidate.raw_segments[name]
        decoded_equal = base.decoded[name] == candidate.decoded[name]
        if not (raw_equal and decoded_equal):
            changed.append(name)
        logical[name] = {
            "decoded_equal_to_pr79_s2": decoded_equal,
            "decoded_sha256": _sha256_bytes(candidate.decoded[name]),
            "pr79_s2_decoded_sha256": _sha256_bytes(base.decoded[name]),
            "pr79_s2_raw_sha256": _sha256_bytes(base.raw_segments[name]),
            "raw_equal_to_pr79_s2": raw_equal,
            "raw_sha256": _sha256_bytes(candidate.raw_segments[name]),
        }
    return {
        "archive_members_changed": [PAYLOAD_MEMBER],
        "logical_runtime_members_changed": changed,
        "logical_runtime_members": logical,
        "payload_container_changed": base.payload_format != candidate.payload_format,
    }


def _validate_candidate(
    *,
    candidate_id: str,
    base: LoadedArchive,
    pr77_records: Sequence[ActionRecord],
    pr79_records: Sequence[ActionRecord],
    selected: Sequence[ActionRecord],
    payload: bytes,
    action_wire: bytes,
    unpacker: Any,
) -> tuple[LoadedArchive, dict[str, Any]]:
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    for name in NON_ACTION_SEGMENTS:
        if decoded.get(name) != base.decoded[name]:
            raise CandidateBuildError(f"{candidate_id}: non-action stream changed: {name}")
    actions = decoded.get("seg_tile_actions.bin")
    if actions is None:
        raise CandidateBuildError(f"{candidate_id}: runtime parser did not decode actions")
    validate_seg_tile_actions_payload(actions, source_name=f"{candidate_id} decoded actions")
    decoded_records = _parse_records(actions, source_label=candidate_id)
    selected_counter = _record_counter(selected)
    decoded_counter = _record_counter(decoded_records)
    pr79_counter = _record_counter(pr79_records)
    if decoded_counter != selected_counter:
        raise CandidateBuildError(f"{candidate_id}: S2 decoded record multiset mismatch")
    if decoded_counter == pr79_counter:
        raise CandidateBuildError(f"{candidate_id}: decoded action record set is a PR79 no-op")
    runtime_members = _member_summary(header)
    raw_segments = _parse_self_describing_slices(payload)
    if raw_segments is None:
        raise CandidateBuildError(f"{candidate_id}: expected self-describing P3 payload")
    candidate = LoadedArchive(
        label=candidate_id,
        path=Path("<unwritten>"),
        archive_bytes=0,
        archive_sha256="",
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=str(header.get("payload_format")),
        raw_segments=raw_segments,
        decoded=decoded,
        runtime_members=runtime_members,
    )
    pr77_counter = _record_counter(pr77_records)
    return candidate, {
        "action_record_multiset_equal_to_pr77": decoded_counter == pr77_counter,
        "action_record_multiset_equal_to_pr79_s2": False,
        "action_wire_bytes": len(action_wire),
        "action_wire_sha256": _sha256_bytes(action_wire),
        "decoded_action_bytes": len(actions),
        "decoded_action_sha256": _sha256_bytes(actions),
        "decoded_record_stats": _record_stats(decoded_records),
        "non_action_streams_preserved": True,
        "no_op_status": "changes_pr79_s2_action_record_set",
        "payload_format": candidate.payload_format,
        "runtime_members": runtime_members,
        "runtime_parser": _repo_rel(UNPACKER_PATH),
        "s2_decoded_order_changes_selected_source_order": actions != _records_raw4(selected),
        "selected_records_canonical_sha256": _sha256_bytes(_canonical_record_bytes(selected)),
    }


def _dispatch_recommendation(*, break_even: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    local_plausible = (
        bool(validation.get("non_action_streams_preserved"))
        and validation.get("no_op_status") == "changes_pr79_s2_action_record_set"
        and float(break_even["score_if_components_unchanged"]) < float(break_even["baseline_score"])
    )
    reason = (
        "local byte screen beats PR79 S2 if components are unchanged, but exact CUDA auth eval is still required"
        if local_plausible
        else "local byte/no-op screen alone is insufficient to justify dispatch"
    )
    return {
        "dispatch_ready_now": False,
        "exact_eval_justified_after_lane_claim": local_plausible,
        "lane_claim_required": True,
        "recommended": local_plausible,
        "remote_dispatch_performed": False,
        "reason": reason,
        "required_before_any_dispatch": [
            "verify archive bytes and SHA-256 match this manifest",
            "claim lane with tools/claim_lane_dispatch.py claim",
            "run exact CUDA auth eval through experiments/contest_auth_eval.py --device cuda",
            "record contest_auth_eval.json, runtime tree hash, n_samples, and component gates",
        ],
    }


def _default_policies() -> list[str]:
    return [
        "replace_pr79_with_pr77_all",
        "replace_pr79_with_pr77_pair_overlap",
        "replace_pr79_with_pr77_tile_overlap",
        "augment_pr79_with_pr77_unique",
    ]


def build_candidates(
    *,
    pr77_archive: Path = DEFAULT_PR77_ARCHIVE,
    pr79_s2_archive: Path = DEFAULT_PR79_S2_ARCHIVE,
    pr79_s2_eval: Path | None = DEFAULT_PR79_S2_EVAL,
    raw_output_parity: Path | None = DEFAULT_RAW_OUTPUT_PARITY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    policies: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    unpacker = S2.BASE._load_unpacker()  # noqa: SLF001
    pr77 = _load_archive("pr77_qzs3_tile_delta_r147", pr77_archive, unpacker=unpacker)
    pr79_s2 = _load_archive("pr79_s2_fixed_adaptive_actions", pr79_s2_archive, unpacker=unpacker)
    pr77_records = _parse_records(pr77.decoded["seg_tile_actions.bin"], source_label="pr77")
    pr79_records = _parse_records(pr79_s2.decoded["seg_tile_actions.bin"], source_label="pr79_s2")
    if len(pr77.raw_segments["seg_tile_actions.bin"]) != PR77_EXPECTED_ACTION_WIRE_BYTES:
        raise CandidateBuildError(
            f"expected PR77 action wire to be {PR77_EXPECTED_ACTION_WIRE_BYTES} bytes, "
            f"got {len(pr77.raw_segments['seg_tile_actions.bin'])}"
        )
    if len(pr77_records) != PR77_EXPECTED_RECORD_COUNT:
        raise CandidateBuildError(
            f"expected PR77 to decode {PR77_EXPECTED_RECORD_COUNT} action records, got {len(pr77_records)}"
        )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    s2_eval = _load_s2_eval(pr79_s2_eval)
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_payload_shas: dict[str, str] = {}
    for policy in list(policies or _default_policies()):
        try:
            selected, policy_info = _select_policy(
                policy,
                pr77_records=pr77_records,
                pr79_records=pr79_records,
            )
            selected_raw = _records_raw4(selected)
            s2 = S2.encode_s2_adaptive_actions(selected_raw)
            payload = _build_p3_payload(pr79_s2, s2["wire"])
            payload_sha = _sha256_bytes(payload)
            if payload_sha in seen_payload_shas:
                raise CandidateBuildError(
                    f"duplicates payload emitted by {seen_payload_shas[payload_sha]}"
                )
            candidate_id = f"{policy}_s2_p3_on_pr79"
            candidate_unwritten, validation = _validate_candidate(
                candidate_id=candidate_id,
                base=pr79_s2,
                pr77_records=pr77_records,
                pr79_records=pr79_records,
                selected=selected,
                payload=payload,
                action_wire=s2["wire"],
                unpacker=unpacker,
            )
            seen_payload_shas[payload_sha] = candidate_id
            candidate_dir = output_dir / candidate_id
            archive_path = candidate_dir / "archive.zip"
            if archive_path.exists() and not force:
                raise FileExistsError(f"{archive_path} exists; pass --force")
            _write_archive(archive_path, payload)
            if _read_single_payload(archive_path) != payload:
                raise CandidateBuildError(f"{candidate_id}: archive readback mismatch")
            archive_sha = _sha256_file(archive_path)
            archive_bytes = archive_path.stat().st_size
            candidate = LoadedArchive(
                **{
                    **candidate_unwritten.__dict__,
                    "path": archive_path,
                    "archive_bytes": archive_bytes,
                    "archive_sha256": archive_sha,
                }
            )
            break_even = _break_even_vs_s2(archive_bytes=archive_bytes, s2_eval=s2_eval)
            raw_proof = _raw_output_proof(raw_output_parity, candidate_id=candidate_id)
            changed_members = _changed_member_summary(base=pr79_s2, candidate=candidate)
            dispatch = _dispatch_recommendation(break_even=break_even, validation=validation)
            manifest = {
                "archive_byte_profile": profile_archive(archive_path),
                "break_even_vs_pr79_s2": break_even,
                "candidate_id": candidate_id,
                "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
                "changed_members": changed_members,
                "dispatch_recommendation": dispatch,
                "evidence_grade": "empirical_archive_closed_byte_screen",
                "no_score_claim": True,
                "output_archive": {
                    "bytes": archive_bytes,
                    "path": str(archive_path),
                    "repo_relative_path": _repo_rel(archive_path),
                    "sha256": archive_sha,
                },
                "payload": {
                    "bytes": len(payload),
                    "format": candidate.payload_format,
                    "member": PAYLOAD_MEMBER,
                    "sha256": payload_sha,
                },
                "policy": policy_info,
                "raw_output_proof": raw_proof,
                "remote_dispatch_performed": False,
                "runtime_parse_validation": validation,
                "s2_action_stream": {
                    "action_arithmetic_bytes": s2["action_arithmetic_bytes"],
                    "action_arithmetic_nbits": s2["action_arithmetic_nbits"],
                    "actions_raw_bytes": len(s2["actions_raw"]),
                    "actions_raw_sha256": _sha256_bytes(s2["actions_raw"]),
                    "meta_and_deltas_brotli_bytes": len(s2["meta_and_deltas_brotli"].data),
                    "meta_and_deltas_brotli_params": s2["meta_and_deltas_brotli"].params,
                    "meta_and_deltas_raw_bytes": len(s2["meta_and_deltas_raw"]),
                    "meta_and_deltas_raw_sha256": _sha256_bytes(s2["meta_and_deltas_raw"]),
                    "wire_bytes": len(s2["wire"]),
                    "wire_sha256": _sha256_bytes(s2["wire"]),
                },
                "schema": MANIFEST_SCHEMA,
                "score_claim": False,
                "source_archives": {
                    "pr77": _source_summary(pr77),
                    "pr79_s2": _source_summary(pr79_s2),
                },
                "tool": TOOL,
            }
            manifest_path = candidate_dir / "manifest.json"
            _write_json(manifest_path, manifest)
            candidates.append(
                {
                    "archive_bytes": archive_bytes,
                    "archive_path": _repo_rel(archive_path),
                    "archive_sha256": archive_sha,
                    "break_even_vs_pr79_s2": break_even,
                    "candidate_id": candidate_id,
                    "changed_members": changed_members["logical_runtime_members_changed"],
                    "dispatch_recommendation": dispatch,
                    "manifest_path": _repo_rel(manifest_path),
                    "payload_bytes": len(payload),
                    "payload_sha256": payload_sha,
                    "raw_output_proof_status": raw_proof["status"],
                    "score_claim": False,
                    "selected_record_count": len(selected),
                    "s2_action_wire_bytes": len(s2["wire"]),
                    "s2_action_wire_sha256": _sha256_bytes(s2["wire"]),
                }
            )
        except Exception as exc:
            skipped.append({"policy": policy, "reason": str(exc), "status": "skipped_not_closed_or_noop"})

    matrix = {
        "baseline_pr79_s2": s2_eval,
        "candidate_count": len(candidates),
        "candidates": sorted(candidates, key=lambda row: (row["archive_bytes"], row["candidate_id"])),
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "evidence_grade": "empirical_archive_closed_byte_screen",
        "pr77_action_anatomy": _record_stats(pr77_records),
        "pr79_s2_action_anatomy": _record_stats(pr79_records),
        "remote_dispatch_performed": False,
        "schema": SCHEMA,
        "score_claim": False,
        "skipped_policies": skipped,
        "source_archives": {
            "pr77": _source_summary(pr77),
            "pr79_s2": _source_summary(pr79_s2),
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", matrix)
    return matrix


def _parse_policies(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr77-archive", type=Path, default=DEFAULT_PR77_ARCHIVE)
    parser.add_argument("--pr79-s2-archive", type=Path, default=DEFAULT_PR79_S2_ARCHIVE)
    parser.add_argument("--pr79-s2-eval", type=Path, default=DEFAULT_PR79_S2_EVAL)
    parser.add_argument("--raw-output-parity", type=Path, default=DEFAULT_RAW_OUTPUT_PARITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policies", type=_parse_policies, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    matrix = build_candidates(
        pr77_archive=args.pr77_archive,
        pr79_s2_archive=args.pr79_s2_archive,
        pr79_s2_eval=args.pr79_s2_eval,
        raw_output_parity=args.raw_output_parity,
        output_dir=args.output_dir,
        policies=args.policies,
        force=bool(args.force),
    )
    print(
        json.dumps(
            {
                "best_by_bytes": matrix["candidates"][0] if matrix["candidates"] else None,
                "candidate_count": matrix["candidate_count"],
                "output_dir": _repo_rel(Path(args.output_dir).resolve()),
                "score_claim": False,
                "skipped_count": len(matrix["skipped_policies"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
