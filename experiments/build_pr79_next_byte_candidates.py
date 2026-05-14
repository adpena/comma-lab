#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local PR79/PR77 next-byte archive candidates.

This is a deterministic byte-screening generator only.  It preserves the PR79
decoded renderer and QP1 pose members, composes local mask-body and action-wire
opportunities, and records dispatchability for later exact CUDA screening.
No scorer is loaded and no remote job is dispatched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
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

from tac.submission_archive import validate_seg_tile_actions_payload


UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_PR79_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip"
)
DEFAULT_PR77_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip"
)
DEFAULT_MASK_MATRIX = (
    REPO_ROOT / "experiments/results/pr79_mask_body_reduction_20260503_worker/candidate_matrix.json"
)
DEFAULT_S2_MATRIX = (
    REPO_ROOT / "experiments/results/pr79_pr77_lossless_s3_profile_20260503_codex/candidate_matrix.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr79_next_byte_candidates_crf52_20260503_worker"
)

TOOL = "experiments/build_pr79_next_byte_candidates.py"
SCHEMA = "pr79_next_byte_candidate_matrix_v1"
MANIFEST_SCHEMA = "pr79_next_byte_candidate_manifest_v1"
MEMBER_NAME = "p"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
PR79_SHA256 = "01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446"
PR77_SHA256 = "f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af"
CURRENT_S2_FRONTIER = {
    "archive_bytes": 277_321,
    "archive_sha256": "5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68",
    "score": 0.31453355357318635,
    "score_source": "exact Tesla T4 auth eval, harvested canonical JSON; not rerun by this tool",
}
CUDA_AUTH_EVAL_REQUIRED = (
    "No remote dispatch from this worker. Before any exact CUDA eval, claim a "
    "non-conflicting lane with tools/claim_lane_dispatch.py claim, then run the "
    "identical archive bytes through archive.zip -> inflate.sh -> upstream/evaluate.py "
    "via experiments/contest_auth_eval.py --device cuda."
)
FIXED_SLICE_ORDER = ("masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.qp1")
PRESERVED_MEMBERS = ("renderer.bin", "optimized_poses.qp1")
ACTION_SOURCES = ("pr79", "pr77", "s2")
CANDIDATE_FAMILIES = (
    "stored_rpk1",
    "brotli_rpk1_flatpack",
    "stored_rpk1_raw_action_control",
)


@dataclass(frozen=True)
class LoadedArchive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str | None
    raw_segments: dict[str, bytes]
    decoded: dict[str, bytes]
    runtime_members: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class MaskCandidate:
    candidate_id: str
    archive_path: Path
    archive_bytes: int
    archive_sha256: str
    exact_eval_ready_after_lane_claim: bool
    manifest_path: str | None
    mask_bytes: bytes
    mask_crf: int | None
    plausibility_gate: dict[str, Any]
    runtime_preflight: dict[str, Any]
    source_row: dict[str, Any]


@dataclass(frozen=True)
class ActionWire:
    action_id: str
    source_label: str
    member_name: str
    wire_bytes: bytes
    runtime_raw: bytes | None
    decoded_semantics: bytes | None
    direct_inflate_loader_compatible: bool
    compatibility_reason: str
    source_archive_sha256: str | None
    source_payload_format: str | None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _repo_rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("pr79_next_byte_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_zip_member_name(name: str) -> str:
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
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return name


def _read_single_p(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [_safe_zip_member_name(info.filename) for info in infos]
        if names != [MEMBER_NAME]:
            raise ValueError(f"{path} must contain exactly {MEMBER_NAME!r}; got {names!r}")
        return zf.read(MEMBER_NAME)


def _zip_bytes(payload: bytes) -> bytes:
    bio = io.BytesIO()
    info = zipfile.ZipInfo(MEMBER_NAME, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr(info, payload)
    return bio.getvalue()


def _write_archive(path: Path, payload: bytes) -> dict[str, Any]:
    data_once = _zip_bytes(payload)
    data_twice = _zip_bytes(payload)
    deterministic = data_once == data_twice
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data_once)
    readback = path.read_bytes()
    if readback != data_once:
        raise ValueError(f"archive readback mismatch for {path}")
    return {
        "archive_bytes": len(data_once),
        "archive_sha256": _sha256_bytes(data_once),
        "deterministic_rebuild_sha256": _sha256_bytes(data_twice),
        "deterministic_rebuild_equal": deterministic,
        "single_stored_p": True,
        "zip_member": MEMBER_NAME,
        "zip_timestamp": list(FIXED_ZIP_TIMESTAMP),
    }


def _decode_p_member_for_parse(payload_member: bytes) -> bytes:
    if payload_member.startswith((b"RPK1", b"P3", b"P6")):
        return payload_member
    try:
        return brotli.decompress(payload_member)
    except brotli.error:
        return payload_member


def _member_summary(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in header.get("members", []):
        if not isinstance(item, Mapping):
            continue
        name = str(item["name"])
        out[name] = {
            "bytes": int(item["bytes"]),
            "codec": str(item.get("codec", "raw")),
            "decoded_bytes": (
                int(item["decoded_bytes"]) if item.get("decoded_bytes") is not None else None
            ),
            "decoded_sha256": item.get("decoded_sha256"),
            "sha256": str(item["sha256"]),
        }
    return out


def _normalise_decoded(decoded: Mapping[str, bytes]) -> dict[str, bytes]:
    out = {str(name): bytes(data) for name, data in decoded.items()}
    if "optimized_poses.bin" in out and "optimized_poses.qp1" not in out:
        out["optimized_poses.qp1"] = out["optimized_poses.bin"]
    if "optimized_poses.qp1" in out and "optimized_poses.bin" not in out:
        out["optimized_poses.bin"] = out["optimized_poses.qp1"]
    return out


def _slice_public_payload(payload: bytes, runtime_members: Mapping[str, Mapping[str, Any]]) -> dict[str, bytes]:
    if payload.startswith(b"P3"):
        import struct

        mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        cursor = 10
        lengths = {
            "masks.mkv": mask_len,
            "renderer.bin": renderer_len,
            "seg_tile_actions.bin": actions_len,
            "optimized_poses.qp1": len(payload) - cursor - mask_len - renderer_len - actions_len,
        }
    elif payload.startswith(b"P6"):
        import struct

        mask_len, renderer_len, actions_len, _record_count = struct.unpack_from("<IHHH", payload, 2)
        cursor = 12
        lengths = {
            "masks.mkv": mask_len,
            "renderer.bin": renderer_len,
            "seg_tile_actions.bin": actions_len,
            "optimized_poses.qp1": len(payload) - cursor - mask_len - renderer_len - actions_len,
        }
    else:
        cursor = 0
        lengths = {name: int(runtime_members[name]["bytes"]) for name in FIXED_SLICE_ORDER}

    raw: dict[str, bytes] = {}
    for name in FIXED_SLICE_ORDER:
        size = lengths[name]
        if size < 0:
            raise ValueError(f"negative fixed-slice size for {name}")
        piece = payload[cursor:cursor + size]
        if len(piece) != size:
            raise ValueError(f"truncated fixed-slice member {name}")
        expected_sha = runtime_members.get(name, {}).get("sha256")
        if expected_sha and _sha256_bytes(piece) != expected_sha:
            raise ValueError(f"raw fixed-slice SHA mismatch for {name}")
        raw[name] = piece
        cursor += size
    if cursor != len(payload):
        raise ValueError(f"fixed-slice parser consumed {cursor} bytes, payload has {len(payload)}")
    return raw


def load_archive(label: str, path: Path, unpacker: Any) -> LoadedArchive:
    path = path.resolve()
    payload = _read_single_p(path)
    header, decoded_raw = unpacker._parse_payload(_decode_p_member_for_parse(payload))  # noqa: SLF001
    decoded = _normalise_decoded(decoded_raw)
    runtime_members = _member_summary(header)
    raw_segments = _slice_public_payload(_decode_p_member_for_parse(payload), runtime_members)
    return LoadedArchive(
        label=label,
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=header.get("payload_format"),
        raw_segments=raw_segments,
        decoded=decoded,
        runtime_members=runtime_members,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _candidate_crf(candidate_id: str) -> int | None:
    match = re.search(r"(?:^|_)crf(\d{2})(?:_|$)", candidate_id)
    return int(match.group(1)) if match else None


def _mask_row_clears_trust_region(row: Mapping[str, Any]) -> bool:
    gate = row.get("plausibility_gate") or {}
    preflight = row.get("runtime_preflight") or {}
    if gate and gate.get("passed") is not True:
        return False
    if preflight and preflight.get("runtime_parser_ok") is False:
        return False
    if preflight and preflight.get("non_mask_runtime_members_preserved") is False:
        return False
    if preflight and preflight.get("candidate_mask_sha_matches_expected") is False:
        return False
    if preflight.get("archive_validation_errors"):
        return False
    return True


def _load_mask_candidate_from_row(row: Mapping[str, Any], unpacker: Any) -> MaskCandidate:
    archive_path = _resolve_repo_path(row["archive_path"])
    payload = _read_single_p(archive_path)
    header, members = unpacker._parse_payload(_decode_p_member_for_parse(payload))  # noqa: SLF001
    if "masks.mkv" not in members:
        raise ValueError(f"mask candidate {archive_path} has no masks.mkv member")
    candidate_id = str(row["candidate_id"])
    mask_sha = _sha256_bytes(bytes(members["masks.mkv"]))
    expected_sha = (((row.get("mask_stream") or {}).get("wire_sha256")) or None)
    if expected_sha is not None and mask_sha != expected_sha:
        raise ValueError(f"mask SHA mismatch for {candidate_id}: {mask_sha} != {expected_sha}")
    return MaskCandidate(
        candidate_id=candidate_id,
        archive_path=archive_path,
        archive_bytes=int(row["archive_bytes"]),
        archive_sha256=str(row["archive_sha256"]),
        exact_eval_ready_after_lane_claim=bool(row.get("exact_eval_ready_after_lane_claim", False)),
        manifest_path=row.get("manifest_path"),
        mask_bytes=bytes(members["masks.mkv"]),
        mask_crf=_candidate_crf(candidate_id),
        plausibility_gate=dict(row.get("plausibility_gate") or {}),
        runtime_preflight=dict(row.get("runtime_preflight") or {}),
        source_row={k: v for k, v in row.items() if not str(k).startswith("_")},
    )


def _select_mask_candidates(
    mask_matrix: Path,
    unpacker: Any,
    *,
    mask_crfs: Sequence[int] = (53, 52, 51, 50),
) -> list[MaskCandidate]:
    matrix = _load_json(mask_matrix)
    requested_crfs = set(mask_crfs)
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    recommendation = (matrix.get("recommendation") or {}).get("candidate")
    if recommendation is not None:
        recommendation_crf = _candidate_crf(str(recommendation["candidate_id"]))
        if recommendation_crf is None or recommendation_crf in requested_crfs:
            rows_by_id[str(recommendation["candidate_id"])] = recommendation
    for row in matrix.get("candidates", []):
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            continue
        crf = _candidate_crf(candidate_id)
        if crf in requested_crfs and _mask_row_clears_trust_region(row):
            rows_by_id[candidate_id] = row
    if not rows_by_id:
        ready = [
            c
            for c in matrix.get("candidates", [])
            if isinstance(c, Mapping) and c.get("exact_eval_ready_after_lane_claim")
        ]
        if not ready:
            raise ValueError(f"{mask_matrix} has no ready or trust-region mask-body candidate")
        row = min(ready, key=lambda item: (int(item["archive_bytes"]), str(item["candidate_id"])))
        rows_by_id[str(row["candidate_id"])] = row
    rows = sorted(
        rows_by_id.values(),
        key=lambda item: (
            _candidate_crf(str(item["candidate_id"])) is None,
            -1 if _candidate_crf(str(item["candidate_id"])) is None else -int(_candidate_crf(str(item["candidate_id"]))),
            int(item["archive_bytes"]),
            str(item["candidate_id"]),
        ),
    )
    return [_load_mask_candidate_from_row(row, unpacker) for row in rows]


def _runtime_raw_from_action_br(wire: bytes) -> tuple[bytes | None, str | None]:
    try:
        return brotli.decompress(wire), None
    except brotli.error as exc:
        return None, str(exc)


def _action_wire_from_public_archive(
    source: LoadedArchive,
    *,
    action_id: str,
    unpacker: Any,
) -> ActionWire:
    wire = source.raw_segments["seg_tile_actions.bin"]
    runtime_raw, br_error = _runtime_raw_from_action_br(wire)
    decoded_semantics: bytes | None = None
    compatible = False
    reason = ""
    if runtime_raw is None:
        reason = f"action wire is not Brotli-decodable for seg_tile_actions.br: {br_error}"
    elif runtime_raw.startswith((b"S1", b"S2")):
        reason = (
            "Brotli action wire decodes to S1/S2; packed seg_tile_actions.br would bypass "
            "the public-slice unpacker and is not accepted by inflate_renderer's direct action loader"
        )
    else:
        try:
            validation = validate_seg_tile_actions_payload(runtime_raw)
            decoded_semantics = source.decoded["seg_tile_actions.bin"]
            compatible = validation.get("encoding") in {"raw", "raw4", "TA4", "raw5", "TA5", "SG2"}
            reason = "Brotli action wire decodes to direct inflate-loader-compatible records"
        except Exception as exc:
            reason = f"direct action payload validation failed: {exc}"
    return ActionWire(
        action_id=action_id,
        source_label=source.label,
        member_name="seg_tile_actions.br",
        wire_bytes=wire,
        runtime_raw=runtime_raw,
        decoded_semantics=decoded_semantics,
        direct_inflate_loader_compatible=compatible,
        compatibility_reason=reason,
        source_archive_sha256=source.archive_sha256,
        source_payload_format=source.payload_format,
    )


def _s2_action_wire(default_s2_matrix: Path, unpacker: Any) -> ActionWire | None:
    if not default_s2_matrix.exists():
        return None
    matrix = _load_json(default_s2_matrix)
    row = (matrix.get("recommendation") or {}).get("candidate")
    if not row:
        return None
    archive_path = _resolve_repo_path(row["archive_path"])
    source = load_archive("pr79_s2_frontier", archive_path, unpacker)
    wire = source.raw_segments["seg_tile_actions.bin"]
    runtime_raw, br_error = _runtime_raw_from_action_br(wire)
    decoded_semantics = source.decoded.get("seg_tile_actions.bin")
    compatible = bool(runtime_raw is not None and not runtime_raw.startswith((b"S1", b"S2")))
    reason = (
        "S2 action wire is parser-valid in public fixed slices but not direct-loader-compatible "
        "when stored as packed seg_tile_actions.br"
        if runtime_raw is not None and runtime_raw.startswith(b"S2")
        else (br_error or "S2 action wire checked")
    )
    return ActionWire(
        action_id="pr79_s2_frontier_wire_br_parser_only",
        source_label="pr79_s2_frontier",
        member_name="seg_tile_actions.br",
        wire_bytes=wire,
        runtime_raw=runtime_raw,
        decoded_semantics=decoded_semantics,
        direct_inflate_loader_compatible=compatible,
        compatibility_reason=reason,
        source_archive_sha256=source.archive_sha256,
        source_payload_format=source.payload_format,
    )


def _build_rpk1_payload(
    *,
    source_archive_sha256: str,
    ordered_members: Sequence[tuple[str, bytes]],
) -> bytes:
    header = {
        "members": [
            {
                "bytes": len(data),
                "codec": "raw",
                "name": name,
                "sha256": _sha256_bytes(data),
            }
            for name, data in ordered_members
        ],
        "schema": "renderer_payload_v1",
        "source_archive_sha256": source_archive_sha256,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import struct

    return b"RPK1" + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(
        data for _name, data in ordered_members
    )


def _decode_candidate_payload(payload_member: bytes, unpacker: Any) -> tuple[dict[str, Any], dict[str, bytes]]:
    return unpacker._parse_payload(_decode_p_member_for_parse(payload_member))  # noqa: SLF001


def _direct_action_semantics(
    members: Mapping[str, bytes],
    *,
    unpacker: Any,
) -> tuple[bytes | None, dict[str, Any]]:
    if "seg_tile_actions.bin" in members:
        raw = members["seg_tile_actions.bin"]
        source_name = "seg_tile_actions.bin"
    elif "seg_tile_actions.br" in members:
        source_name = "seg_tile_actions.br"
        try:
            raw = brotli.decompress(members["seg_tile_actions.br"])
        except brotli.error as exc:
            return None, {"compatible": False, "reason": f"brotli decode failed: {exc}"}
    else:
        return None, {"compatible": False, "reason": "no tile-action member present"}
    if raw.startswith((b"S1", b"S2")):
        try:
            decoded = unpacker._decode_seg_tile_actions(raw)  # noqa: SLF001
        except Exception:
            decoded = raw
        return decoded, {
            "compatible": False,
            "reason": "S1/S2 bytes require public-slice unpacker decode before inflate action loading",
            "source_name": source_name,
        }
    try:
        validation = validate_seg_tile_actions_payload(raw)
    except Exception as exc:
        return raw, {"compatible": False, "reason": str(exc), "source_name": source_name}
    return raw, {
        "compatible": True,
        "reason": "direct action payload validates for inflate action loading",
        "source_name": source_name,
        "validation": validation,
    }


def _candidate_score_estimate(archive_bytes: int, s2_frontier: Mapping[str, Any]) -> dict[str, Any]:
    delta_bytes = archive_bytes - int(s2_frontier["archive_bytes"])
    rate_delta = delta_bytes * RATE_SCORE_PER_BYTE
    return {
        "anchor": dict(s2_frontier),
        "candidate_delta_bytes_vs_s2_frontier": delta_bytes,
        "rate_score_delta_vs_s2_frontier": rate_delta,
        "score_if_components_unchanged_vs_current_s2_frontier": float(s2_frontier["score"]) + rate_delta,
        "unchanged_component_estimate_only": True,
        "score_claim": False,
    }


def _changed_members(
    *,
    pr79: LoadedArchive,
    members: Mapping[str, bytes],
    action_semantics: bytes | None,
) -> dict[str, Any]:
    checks = {
        "renderer.bin": members.get("renderer.bin") == pr79.decoded["renderer.bin"],
        "optimized_poses.qp1": members.get("optimized_poses.qp1") == pr79.decoded["optimized_poses.qp1"],
        "masks.mkv": members.get("masks.mkv") == pr79.decoded["masks.mkv"],
        "seg_tile_actions": action_semantics == pr79.decoded["seg_tile_actions.bin"],
    }
    return {
        "changed_vs_pr79_decoded": sorted(name for name, equal in checks.items() if not equal),
        "preserved_renderer_and_pose": checks["renderer.bin"] and checks["optimized_poses.qp1"],
        "semantic_equalities_vs_pr79": checks,
    }


def _dispatchability(
    *,
    changed: Mapping[str, Any],
    action_loader: Mapping[str, Any],
    zip_profile: Mapping[str, Any],
    noop: Mapping[str, Any],
    mask_candidate: MaskCandidate | None,
    require_action_parity: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    if noop["archive_sha_equal_to_pr79"] or (
        noop["payload_sha_equal_to_pr79"] and noop["decoded_members_equal_to_pr79"]
    ):
        reasons.append("no-op source control")
    if not changed["preserved_renderer_and_pose"]:
        reasons.append("renderer or pose member changed")
    if not action_loader.get("compatible"):
        reasons.append(str(action_loader.get("reason", "action loader incompatible")))
    if require_action_parity and not changed["semantic_equalities_vs_pr79"].get("seg_tile_actions"):
        reasons.append("action semantics differ from PR79 decoded action baseline")
    if not zip_profile.get("deterministic_rebuild_equal"):
        reasons.append("archive rebuild is not deterministic")
    if mask_candidate is None and "masks.mkv" in changed["changed_vs_pr79_decoded"]:
        reasons.append("mask changed without mask candidate provenance")
    if mask_candidate is not None and not _mask_row_clears_trust_region(mask_candidate.source_row):
        reasons.append("source mask-body row does not clear local trust-region/runtime guards")
    if mask_candidate is not None and not mask_candidate.exact_eval_ready_after_lane_claim:
        reasons.append("source mask-body row is not exact-eval-ready after lane claim")
    ready = not reasons
    return {
        "exact_screen_dispatchable_after_lane_claim": ready,
        "lane_claim_required_before_dispatch": True,
        "source_mask_body_exact_eval_ready_after_lane_claim": (
            mask_candidate.exact_eval_ready_after_lane_claim if mask_candidate is not None else None
        ),
        "remote_dispatch_performed": False,
        "reason": (
            "local byte-closed candidate preserves renderer/pose and passes direct action-loader checks"
            if ready
            else "; ".join(reasons)
        ),
        "required_exact_gate": CUDA_AUTH_EVAL_REQUIRED,
    }


def _emit_candidate(
    *,
    candidate_id: str,
    payload_member: bytes,
    output_dir: Path,
    pr79: LoadedArchive,
    action: ActionWire | None,
    mask_candidate: MaskCandidate | None,
    candidate_family: str,
    unpacker: Any,
    s2_frontier: Mapping[str, Any],
    force: bool,
    require_action_parity_for_dispatch: bool = False,
    build_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    manifest_path = candidate_dir / "manifest.json"
    if archive_path.exists() and not force:
        raise FileExistsError(f"{archive_path} exists; pass --force")
    zip_profile = _write_archive(archive_path, payload_member)
    header, members = _decode_candidate_payload(payload_member, unpacker)
    action_semantics, action_loader = _direct_action_semantics(members, unpacker=unpacker)
    action_semantics_for_change = (
        action.decoded_semantics
        if action is not None and action.decoded_semantics is not None and action_loader.get("compatible")
        else action_semantics
    )
    changed = _changed_members(pr79=pr79, members=members, action_semantics=action_semantics_for_change)
    noop = {
        "archive_sha_equal_to_pr79": zip_profile["archive_sha256"] == pr79.archive_sha256,
        "payload_sha_equal_to_pr79": _sha256_bytes(payload_member) == pr79.payload_sha256,
        "decoded_members_equal_to_pr79": not changed["changed_vs_pr79_decoded"],
        "status": (
            "byte_identical_noop"
            if zip_profile["archive_sha256"] == pr79.archive_sha256
            else "payload_identical_container_reemit_noop"
            if _sha256_bytes(payload_member) == pr79.payload_sha256
            and not changed["changed_vs_pr79_decoded"]
            else "non_noop_changed_" + "_".join(changed["changed_vs_pr79_decoded"] or ["container"])
        ),
    }
    dispatchability = _dispatchability(
        changed=changed,
        action_loader=action_loader,
        zip_profile=zip_profile,
        noop=noop,
        mask_candidate=mask_candidate,
        require_action_parity=require_action_parity_for_dispatch,
    )
    score_estimate = _candidate_score_estimate(zip_profile["archive_bytes"], s2_frontier)
    parsed_member_bytes = sum(len(data) for data in members.values())
    decoded_payload = _decode_p_member_for_parse(payload_member)
    member_profile = {
        name: {
            "bytes": len(data),
            "delta_bytes_vs_pr79_decoded": len(data) - len(pr79.decoded[name])
            if name in pr79.decoded
            else None,
            "sha256": _sha256_bytes(data),
        }
        for name, data in sorted(members.items())
    }
    body_byte_profile = {
        "archive_bytes": zip_profile["archive_bytes"],
        "archive_delta_bytes_vs_current_s2_frontier": score_estimate[
            "candidate_delta_bytes_vs_s2_frontier"
        ],
        "archive_delta_bytes_vs_pr79_public": zip_profile["archive_bytes"] - pr79.archive_bytes,
        "payload_member_bytes": len(payload_member),
        "payload_member_sha256": _sha256_bytes(payload_member),
        "payload_decode_for_parse_bytes": len(decoded_payload),
        "payload_decode_for_parse_sha256": _sha256_bytes(decoded_payload),
        "parsed_member_total_bytes": parsed_member_bytes,
        "parsed_members": member_profile,
        "top_level_brotli_flatpack": decoded_payload != payload_member,
        "zip_overhead_bytes": zip_profile["archive_bytes"] - len(payload_member),
    }
    provenance = {
        "build_options": dict(build_options or {}),
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "no_remote_dispatch_performed": True,
        "source_action": (
            None
            if action is None
            else {
                "action_id": action.action_id,
                "source_archive_sha256": action.source_archive_sha256,
                "source_label": action.source_label,
                "source_payload_format": action.source_payload_format,
                "wire_sha256": _sha256_bytes(action.wire_bytes),
            }
        ),
        "source_mask_candidate": (
            None
            if mask_candidate is None
            else {
                "archive_path": _repo_rel(mask_candidate.archive_path),
                "archive_sha256": mask_candidate.archive_sha256,
                "candidate_id": mask_candidate.candidate_id,
                "manifest_path": mask_candidate.manifest_path,
                "manifest_sha256": (
                    _sha256_file(_resolve_repo_path(mask_candidate.manifest_path))
                    if mask_candidate.manifest_path is not None
                    and _resolve_repo_path(mask_candidate.manifest_path).exists()
                    else None
                ),
                "mask_crf": mask_candidate.mask_crf,
                "mask_sha256": _sha256_bytes(mask_candidate.mask_bytes),
            }
        ),
        "source_pr79": {
            "archive_path": _repo_rel(pr79.path),
            "archive_sha256": pr79.archive_sha256,
            "payload_format": pr79.payload_format,
        },
        "tool": TOOL,
    }
    row = {
        "archive_bytes": zip_profile["archive_bytes"],
        "archive_path": _repo_rel(archive_path),
        "archive_sha256": zip_profile["archive_sha256"],
        "body_byte_profile": body_byte_profile,
        "candidate_family": candidate_family,
        "candidate_id": candidate_id,
        "changed_members": changed,
        "delta_bytes_vs_current_s2_frontier": score_estimate["candidate_delta_bytes_vs_s2_frontier"],
        "dispatchability": dispatchability,
        "manifest_path": _repo_rel(manifest_path),
        "no_op_detection": noop,
        "provenance": provenance,
        "score_claim": False,
        "score_estimate_vs_current_s2_frontier": score_estimate,
    }
    manifest = {
        "action_wire": (
            None
            if action is None
            else {
                "action_id": action.action_id,
                "compatibility_reason": action.compatibility_reason,
                "decoded_semantics_equal_pr79": action.decoded_semantics == pr79.decoded["seg_tile_actions.bin"],
                "direct_inflate_loader_compatible": action.direct_inflate_loader_compatible,
                "member_name": action.member_name,
                "runtime_raw_bytes": len(action.runtime_raw) if action.runtime_raw is not None else None,
                "runtime_raw_sha256": (
                    _sha256_bytes(action.runtime_raw) if action.runtime_raw is not None else None
                ),
                "source_archive_sha256": action.source_archive_sha256,
                "source_label": action.source_label,
                "source_payload_format": action.source_payload_format,
                "wire_bytes": len(action.wire_bytes),
                "wire_sha256": _sha256_bytes(action.wire_bytes),
            }
        ),
        "candidate": row,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "evidence_grade": "empirical_local_byte_screen_only",
        "manifest_schema": MANIFEST_SCHEMA,
        "mask_candidate": (
            None
            if mask_candidate is None
            else {
                "archive_bytes": mask_candidate.archive_bytes,
                "archive_path": _repo_rel(mask_candidate.archive_path),
                "archive_sha256": mask_candidate.archive_sha256,
                "candidate_id": mask_candidate.candidate_id,
                "exact_eval_ready_after_lane_claim": mask_candidate.exact_eval_ready_after_lane_claim,
                "manifest_path": mask_candidate.manifest_path,
                "mask_crf": mask_candidate.mask_crf,
                "mask_bytes": len(mask_candidate.mask_bytes),
                "mask_sha256": _sha256_bytes(mask_candidate.mask_bytes),
                "plausibility_gate": mask_candidate.plausibility_gate,
                "runtime_preflight": mask_candidate.runtime_preflight,
                "source_manifest_sha256": (
                    _sha256_file(_resolve_repo_path(mask_candidate.manifest_path))
                    if mask_candidate.manifest_path is not None
                    and _resolve_repo_path(mask_candidate.manifest_path).exists()
                    else None
                ),
            }
        ),
        "no_remote_dispatch_performed": True,
        "payload_member": {
            "bytes": len(payload_member),
            "decode_for_parse_sha256": _sha256_bytes(_decode_p_member_for_parse(payload_member)),
            "sha256": _sha256_bytes(payload_member),
            "top_level_brotli": _decode_p_member_for_parse(payload_member) != payload_member,
        },
        "runtime_parse_validation": {
            "action_loader": action_loader,
            "action_semantics_parity": {
                "equal_pr79_decoded_actions": action_semantics_for_change
                == pr79.decoded["seg_tile_actions.bin"],
                "semantics_sha256": (
                    _sha256_bytes(action_semantics_for_change)
                    if action_semantics_for_change is not None
                    else None
                ),
            },
            "members": {
                name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
                for name, data in sorted(members.items())
            },
            "payload_format": header.get("payload_format"),
            "runtime_parser": _repo_rel(UNPACKER_PATH),
        },
        "score_claim": False,
        "source_pr79": {
            "archive_bytes": pr79.archive_bytes,
            "archive_path": _repo_rel(pr79.path),
            "archive_sha256": pr79.archive_sha256,
            "payload_format": pr79.payload_format,
        },
        "tool": TOOL,
        "provenance": provenance,
        "body_byte_profile": body_byte_profile,
        "zip_profile": zip_profile,
    }
    _write_json(manifest_path, manifest)
    return row


def _candidate_payloads(
    *,
    pr79: LoadedArchive,
    pr77: LoadedArchive,
    mask: MaskCandidate,
    s2_action: ActionWire | None,
    unpacker: Any,
    include_source_control: bool = True,
    action_sources: Sequence[str] = ACTION_SOURCES,
    candidate_families: Sequence[str] = CANDIDATE_FAMILIES,
) -> list[tuple[str, bytes, ActionWire | None, MaskCandidate | None, str]]:
    action_source_set = set(action_sources)
    family_set = set(candidate_families)
    source_control = (
        "source_pr79_noop_control",
        pr79.payload,
        None,
        None,
        "source_control",
    )
    actions: list[ActionWire] = []
    if "pr79" in action_source_set:
        actions.append(
            _action_wire_from_public_archive(
                pr79,
                action_id="pr79_public_raw4_action_wire_br",
                unpacker=unpacker,
            )
        )
    if "pr77" in action_source_set:
        actions.append(
            _action_wire_from_public_archive(
                pr77,
                action_id="pr77_public_raw4_action_wire_br",
                unpacker=unpacker,
            )
        )
    if "s2" in action_source_set and s2_action is not None:
        actions.append(s2_action)

    out: list[tuple[str, bytes, ActionWire | None, MaskCandidate | None, str]] = (
        [source_control] if include_source_control else []
    )
    for action in actions:
        members = [
            ("renderer.bin", pr79.decoded["renderer.bin"]),
            ("masks.mkv", mask.mask_bytes),
            (action.member_name, action.wire_bytes),
            ("optimized_poses.qp1", pr79.decoded["optimized_poses.qp1"]),
        ]
        raw_rpk1 = _build_rpk1_payload(
            source_archive_sha256=pr79.archive_sha256,
            ordered_members=members,
        )
        base_id = f"{mask.candidate_id}__{action.action_id}"
        if "stored_rpk1" in family_set:
            out.append((f"{base_id}__stored_rpk1", raw_rpk1, action, mask, "stored_rpk1"))
        if "brotli_rpk1_flatpack" in family_set:
            flat = brotli.compress(raw_rpk1, quality=11, mode=0, lgwin=22)
            out.append(
                (f"{base_id}__brotli_rpk1_flatpack", flat, action, mask, "brotli_rpk1_flatpack")
            )

    if "stored_rpk1_raw_action_control" not in family_set:
        return out
    raw_members = [
        ("renderer.bin", pr79.decoded["renderer.bin"]),
        ("masks.mkv", mask.mask_bytes),
        ("seg_tile_actions.bin", pr79.decoded["seg_tile_actions.bin"]),
        ("optimized_poses.qp1", pr79.decoded["optimized_poses.qp1"]),
    ]
    raw_payload = _build_rpk1_payload(
        source_archive_sha256=pr79.archive_sha256,
        ordered_members=raw_members,
    )
    out.append(
        (
            f"{mask.candidate_id}__pr79_decoded_actions_bin__stored_rpk1_control",
            raw_payload,
            None,
            mask,
            "stored_rpk1_raw_action_control",
        )
    )
    return out


def _rank_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    dispatchable = int(not row["dispatchability"]["exact_screen_dispatchable_after_lane_claim"])
    pr79_action_semantics = bool(
        row["changed_members"]["semantic_equalities_vs_pr79"].get("seg_tile_actions")
    )
    return (dispatchable, int(not pr79_action_semantics), int(row["archive_bytes"]), str(row["candidate_id"]))


def build_candidates(
    *,
    pr79_archive: Path = DEFAULT_PR79_ARCHIVE,
    pr77_archive: Path = DEFAULT_PR77_ARCHIVE,
    mask_matrix: Path = DEFAULT_MASK_MATRIX,
    s2_matrix: Path = DEFAULT_S2_MATRIX,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
    s2_frontier: Mapping[str, Any] = CURRENT_S2_FRONTIER,
    mask_crfs: Sequence[int] = (53, 52, 51, 50),
    action_sources: Sequence[str] = ACTION_SOURCES,
    candidate_families: Sequence[str] = CANDIDATE_FAMILIES,
    include_source_control: bool = True,
    require_action_parity_for_dispatch: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unpacker = _load_unpacker()
    pr79 = load_archive("pr79_public", pr79_archive, unpacker)
    pr77 = load_archive("pr77_public", pr77_archive, unpacker)
    if pr79_archive.resolve() == DEFAULT_PR79_ARCHIVE.resolve() and pr79.archive_sha256 != PR79_SHA256:
        raise ValueError(f"default PR79 SHA mismatch: {pr79.archive_sha256}")
    if pr77_archive.resolve() == DEFAULT_PR77_ARCHIVE.resolve() and pr77.archive_sha256 != PR77_SHA256:
        raise ValueError(f"default PR77 SHA mismatch: {pr77.archive_sha256}")
    for member in PRESERVED_MEMBERS:
        if pr79.decoded[member] != pr77.decoded[member]:
            raise ValueError(f"PR77 does not preserve PR79 {member}")

    masks = _select_mask_candidates(mask_matrix, unpacker, mask_crfs=mask_crfs)
    s2_action = _s2_action_wire(s2_matrix, unpacker)
    build_options = {
        "action_sources": list(action_sources),
        "candidate_families": list(candidate_families),
        "include_source_control": include_source_control,
        "mask_crfs": list(mask_crfs),
        "require_action_parity_for_dispatch": require_action_parity_for_dispatch,
    }
    payload_specs: list[tuple[str, bytes, ActionWire | None, MaskCandidate | None, str]] = []
    for index, mask in enumerate(masks):
        payload_specs.extend(
            _candidate_payloads(
                pr79=pr79,
                pr77=pr77,
                mask=mask,
                s2_action=s2_action,
                unpacker=unpacker,
                include_source_control=include_source_control and index == 0,
                action_sources=action_sources,
                candidate_families=candidate_families,
            )
        )
    rows = []
    for candidate_id, payload_member, action, mask_candidate, family in payload_specs:
        rows.append(
            _emit_candidate(
                candidate_id=candidate_id,
                payload_member=payload_member,
                output_dir=output_dir,
                pr79=pr79,
                action=action,
                mask_candidate=mask_candidate,
                candidate_family=family,
                unpacker=unpacker,
                s2_frontier=s2_frontier,
                force=force,
                require_action_parity_for_dispatch=require_action_parity_for_dispatch,
                build_options=build_options,
            )
        )
    rows.sort(key=_rank_key)
    top = rows[0] if rows else None
    summary = {
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "candidate_count": len(rows),
        "candidates": rows,
        "current_s2_frontier": dict(s2_frontier),
        "evidence_grade": "empirical_local_byte_screen_only",
        "build_options": build_options,
        "mask_candidate_sources": [
            {
                "archive_bytes": mask.archive_bytes,
                "archive_path": _repo_rel(mask.archive_path),
                "archive_sha256": mask.archive_sha256,
                "candidate_id": mask.candidate_id,
                "exact_eval_ready_after_lane_claim": mask.exact_eval_ready_after_lane_claim,
                "mask_crf": mask.mask_crf,
                "matrix_path": _repo_rel(mask_matrix),
                "plausibility_gate_passed": mask.plausibility_gate.get("passed"),
            }
            for mask in masks
        ],
        "no_remote_dispatch_performed": True,
        "recommendation": {
            "candidate": top,
            "decision": (
                "recommend_exact_cuda_screen_after_lane_claim"
                if top and top["dispatchability"]["exact_screen_dispatchable_after_lane_claim"]
                else "no_dispatchable_candidate"
            ),
            "reason": (
                "highest-ranked local byte candidate preserves PR79 renderer/pose and is byte-closed for future exact CUDA screening"
                if top and top["dispatchability"]["exact_screen_dispatchable_after_lane_claim"]
                else "no candidate cleared local byte-closure and direct action-loader checks"
            ),
        },
        "schema": SCHEMA,
        "score_claim": False,
        "source_archives": {
            "pr77": {
                "archive_bytes": pr77.archive_bytes,
                "archive_path": _repo_rel(pr77.path),
                "archive_sha256": pr77.archive_sha256,
                "action_wire_bytes": len(pr77.raw_segments["seg_tile_actions.bin"]),
            },
            "pr79": {
                "archive_bytes": pr79.archive_bytes,
                "archive_path": _repo_rel(pr79.path),
                "archive_sha256": pr79.archive_sha256,
                "action_wire_bytes": len(pr79.raw_segments["seg_tile_actions.bin"]),
            },
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr79-archive", type=Path, default=DEFAULT_PR79_ARCHIVE)
    parser.add_argument("--pr77-archive", type=Path, default=DEFAULT_PR77_ARCHIVE)
    parser.add_argument("--mask-matrix", type=Path, default=DEFAULT_MASK_MATRIX)
    parser.add_argument("--s2-matrix", type=Path, default=DEFAULT_S2_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mask-crfs",
        default="53,52,51,50",
        help="Comma-separated CRF suffixes to select from the mask matrix.",
    )
    parser.add_argument(
        "--action-sources",
        default="pr79,pr77,s2",
        help="Comma-separated action wire sources: pr79, pr77, s2.",
    )
    parser.add_argument(
        "--candidate-families",
        default="stored_rpk1,brotli_rpk1_flatpack,stored_rpk1_raw_action_control",
        help=(
            "Comma-separated families: stored_rpk1, brotli_rpk1_flatpack, "
            "stored_rpk1_raw_action_control, source_control."
        ),
    )
    parser.add_argument("--no-source-control", action="store_true")
    parser.add_argument("--require-action-parity-for-dispatch", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(part) for part in _parse_csv_values(raw)]


def _validate_subset(values: Sequence[str], allowed: Sequence[str], label: str) -> None:
    bad = sorted(set(values) - set(allowed))
    if bad:
        raise ValueError(f"unsupported {label}: {bad}; allowed={list(allowed)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    action_sources = _parse_csv_values(args.action_sources)
    candidate_families = _parse_csv_values(args.candidate_families)
    _validate_subset(action_sources, ACTION_SOURCES, "action sources")
    _validate_subset(candidate_families, CANDIDATE_FAMILIES + ("source_control",), "families")
    summary = build_candidates(
        pr79_archive=args.pr79_archive,
        pr77_archive=args.pr77_archive,
        mask_matrix=args.mask_matrix,
        s2_matrix=args.s2_matrix,
        output_dir=args.output_dir,
        force=bool(args.force),
        mask_crfs=_parse_csv_ints(args.mask_crfs),
        action_sources=action_sources,
        candidate_families=candidate_families,
        include_source_control=not bool(args.no_source_control),
        require_action_parity_for_dispatch=bool(args.require_action_parity_for_dispatch),
    )
    print(json.dumps(summary["recommendation"], indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
