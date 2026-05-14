#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile local PR79/S2 mask-body byte reduction candidates.

This is a local byte-screening and preflight tool only.  It does not dispatch
remote jobs, does not run scorers, and does not make score claims.  Candidate
archives preserve PR79's renderer, S2 action stream, and QP1 pose bytes while
replacing only the charged mask wire slice inside a self-describing public P3
payload.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import struct
import sys
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.submission_archive import validate_archive_seg_tile_actions_payloads


PR79_SOURCE_LOADER = REPO_ROOT / "experiments/build_pr79_action_lossless_repack_candidates.py"
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_PR79_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr79_mask_body_reduction_20260503_worker"
)
DEFAULT_CANDIDATE_MANIFEST_GLOBS = (
    "experiments/results/c067_micro_mask_reencode*/**/protected_mask_reencode_manifest.json",
)

TOOL = "experiments/profile_pr79_mask_body_reduction_candidates.py"
SCHEMA = "pr79_mask_body_reduction_profile_v1"
MANIFEST_SCHEMA = "pr79_mask_body_reduction_candidate_manifest_v1"
RECOMMENDATION_SCHEMA = "pr79_mask_body_reduction_recommendation_v1"
MEMBER_NAME = "p"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
TARGET_SAVINGS_BYTES = 3_000
CUDA_AUTH_EVAL_REQUIRED = (
    "No remote dispatch from this worker. Before any exact CUDA eval, claim a "
    "non-conflicting lane with tools/claim_lane_dispatch.py claim, then run the "
    "identical archive bytes through archive.zip -> inflate.sh -> "
    "upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda."
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _safe_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return cleaned[:160] or "candidate"


def _load_source_archive(pr79_archive: Path) -> Any:
    loader = _load_module(PR79_SOURCE_LOADER, "pr79_mask_body_source_loader")
    return loader.load_source_archive(pr79_archive)


def _load_unpacker() -> Any:
    return _load_module(UNPACKER_PATH, "pr79_mask_body_unpacker")


def _write_single_stored_p_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(MEMBER_NAME, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _read_single_stored_p(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1 or infos[0].filename != MEMBER_NAME:
            raise ValueError(f"{path} must contain exactly one {MEMBER_NAME!r} member")
        if infos[0].compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{MEMBER_NAME} must be ZIP_STORED")
        return zf.read(MEMBER_NAME)


def _zip_profile(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        duplicate_names = sorted(
            name for name in {info.filename for info in infos} if sum(i.filename == name for i in infos) > 1
        )
        members = [
            {
                "compress_size": int(info.compress_size),
                "compress_type": int(info.compress_type),
                "date_time": list(info.date_time),
                "external_attr": int(info.external_attr),
                "file_size": int(info.file_size),
                "filename": info.filename,
            }
            for info in infos
        ]
    single_stored_p = (
        len(members) == 1
        and members[0]["filename"] == MEMBER_NAME
        and members[0]["compress_type"] == zipfile.ZIP_STORED
    )
    return {
        "archive_bytes": path.stat().st_size,
        "archive_sha256": _sha256_file(path),
        "duplicate_member_names": duplicate_names,
        "member_count": len(members),
        "members": members,
        "single_stored_p": single_stored_p,
        "strict_zip_overhead_bytes": (
            path.stat().st_size - int(members[0]["file_size"])
            if single_stored_p
            else None
        ),
    }


def _brotli_param_grid() -> list[dict[str, int]]:
    params: list[dict[str, int]] = []
    for quality in (11, 10, 9, 8, 6, 4):
        for mode in (0, 1, 2):
            for lgwin in (16, 18, 20, 22, 24):
                params.append({"quality": quality, "mode": mode, "lgwin": lgwin})
    return params


def _best_brotli(data: bytes, *, params: Iterable[dict[str, int]] | None = None) -> tuple[bytes, dict[str, int]]:
    best_data: bytes | None = None
    best_params: dict[str, int] | None = None
    for param in params or _brotli_param_grid():
        candidate = brotli.compress(data, **param)
        if best_data is None or len(candidate) < len(best_data):
            best_data = candidate
            best_params = dict(param)
    if best_data is None or best_params is None:
        raise ValueError("brotli parameter grid was empty")
    return best_data, best_params


def _build_p3_payload(
    *,
    mask_wire: bytes,
    renderer_wire: bytes,
    actions_wire: bytes,
    pose_wire: bytes,
) -> bytes:
    if len(renderer_wire) > 0xFFFF:
        raise ValueError("P3 renderer slice does not fit u16")
    if len(actions_wire) > 0xFFFF:
        raise ValueError("P3 action slice does not fit u16")
    return (
        b"P3"
        + struct.pack("<IHH", len(mask_wire), len(renderer_wire), len(actions_wire))
        + mask_wire
        + renderer_wire
        + actions_wire
        + pose_wire
    )


def _build_rpk1_payload(
    *,
    source_archive_sha256: str,
    renderer: bytes,
    mask: bytes,
    actions_wire: bytes,
    pose_qp1: bytes,
) -> bytes:
    members = [
        ("renderer.bin", renderer),
        ("masks.mkv", mask),
        ("seg_tile_actions.bin", actions_wire),
        ("optimized_poses.qp1", pose_qp1),
    ]
    header = {
        "members": [
            {
                "bytes": len(data),
                "codec": "raw",
                "name": name,
                "sha256": _sha256_bytes(data),
            }
            for name, data in members
        ],
        "schema": "renderer_payload_v1",
        "source_archive_sha256": source_archive_sha256,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return b"RPK1" + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(data for _name, data in members)


def _looks_like_mask_obu(data: bytes) -> bool:
    return data.startswith(b"\x12\x00\x0a\x0a") or data.startswith(b"\x12\x00")


def _decode_action_semantics(actions: bytes, *, unpacker: Any) -> bytes:
    if actions == b"":
        return actions
    if actions.startswith((b"S1", b"S2", b"SG2")) or len(actions) % 4 != 0:
        return unpacker._decode_seg_tile_actions(actions)  # noqa: SLF001
    return actions


def _member_action_semantics(members: dict[str, bytes], *, unpacker: Any) -> bytes:
    if "seg_tile_actions.bin" in members:
        return _decode_action_semantics(members["seg_tile_actions.bin"], unpacker=unpacker)
    if "seg_tile_actions.br" in members:
        return _decode_action_semantics(brotli.decompress(members["seg_tile_actions.br"]), unpacker=unpacker)
    raise ValueError("candidate payload lacks seg_tile_actions.bin or seg_tile_actions.br")


def _header_members(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): dict(item) for item in header.get("members", [])}


def _validate_archive(
    *,
    archive_path: Path,
    source: Any,
    candidate_mask_sha256: str,
    unpacker: Any,
) -> dict[str, Any]:
    payload = _read_single_stored_p(archive_path)
    header, members = unpacker._parse_payload(payload)  # noqa: SLF001
    header_by_name = _header_members(header)
    archive_validation_errors = validate_archive_seg_tile_actions_payloads(archive_path)
    decoded_actions = _member_action_semantics(members, unpacker=unpacker)
    non_mask_checks = {
        "renderer_decoded_preserved": members.get("renderer.bin") == source.decoded["renderer.bin"],
        "actions_decoded_preserved": decoded_actions == source.decoded["seg_tile_actions.bin"],
        "pose_decoded_preserved": members.get("optimized_poses.qp1") == source.decoded["optimized_poses.qp1"],
    }
    return {
        "archive_validation_errors": archive_validation_errors,
        "candidate_mask_decoded_sha256": _sha256_bytes(members["masks.mkv"]),
        "candidate_mask_sha_matches_expected": _sha256_bytes(members["masks.mkv"]) == candidate_mask_sha256,
        "non_mask_runtime_members_preserved": all(non_mask_checks.values()),
        "non_mask_runtime_member_checks": non_mask_checks,
        "payload_format": header.get("payload_format"),
        "runtime_parser": _repo_rel(UNPACKER_PATH),
        "runtime_parser_ok": True,
        "runtime_member_headers": {
            name: {
                "bytes": int(meta.get("bytes", 0)),
                "codec": meta.get("codec"),
                "decoded_bytes": meta.get("decoded_bytes"),
                "decoded_sha256": meta.get("decoded_sha256"),
                "sha256": meta.get("sha256"),
            }
            for name, meta in sorted(header_by_name.items())
        },
    }


def _plausibility_gate(source_note: dict[str, Any]) -> dict[str, Any]:
    if source_note.get("kind") == "lossless_rebrotli_control":
        return {
            "passed": True,
            "reason": "lossless source-mask control",
        }
    agreement = source_note.get("mask_agreement") or {}
    overall = (agreement.get("overall") or {}).get("argmax_agreement")
    protected = (agreement.get("protected") or {}).get("argmax_agreement")
    passed = (
        isinstance(overall, (int, float))
        and isinstance(protected, (int, float))
        and float(overall) >= 0.997
        and float(protected) >= 0.99
    )
    return {
        "minimum_overall_argmax_agreement": 0.997,
        "minimum_protected_argmax_agreement": 0.99,
        "overall_argmax_agreement": overall,
        "passed": bool(passed),
        "protected_argmax_agreement": protected,
        "reason": (
            "protected-mask candidate clears local plausibility agreement thresholds"
            if passed
            else "protected-mask candidate is byte-valid but below local plausibility agreement thresholds"
        ),
    }


def _manifest_candidate_id(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(REPO_ROOT / "experiments/results")
    except ValueError:
        rel = path
    stem = str(rel.parent)
    return "protected_" + _safe_id(stem)


def _candidate_manifest_paths(globs: Sequence[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in globs:
        root = REPO_ROOT
        for path in root.glob(pattern):
            if path.is_file():
                found.add(path.resolve())
    return sorted(found)


def _read_candidate_mask_from_manifest(manifest: dict[str, Any]) -> bytes:
    archive_path = Path(manifest["archive"]["path"])
    if not archive_path.is_absolute():
        archive_path = REPO_ROOT / archive_path
    member = str((manifest.get("candidate_mask_stream") or {}).get("member", "masks.mkv"))
    with zipfile.ZipFile(archive_path, "r") as zf:
        return zf.read(member)


def _candidate_stream_rows(
    *,
    source: Any,
    manifest_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    source_mask_sha = _sha256_bytes(source.decoded["masks.mkv"])
    rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        try:
            manifest = _read_json(manifest_path)
        except Exception as exc:
            rows.append(
                {
                    "candidate_id": _manifest_candidate_id(manifest_path),
                    "manifest_path": _repo_rel(manifest_path),
                    "status": "invalid_manifest_json",
                    "reason": str(exc),
                }
            )
            continue
        source_mask = manifest.get("source_mask_stream") or {}
        if source_mask.get("sha256") != source_mask_sha:
            continue
        candidate_mask = manifest.get("candidate_mask_stream") or {}
        try:
            mask_bytes = _read_candidate_mask_from_manifest(manifest)
        except Exception as exc:
            rows.append(
                {
                    "candidate_id": _manifest_candidate_id(manifest_path),
                    "manifest_path": _repo_rel(manifest_path),
                    "status": "missing_candidate_mask",
                    "reason": str(exc),
                }
            )
            continue
        mask_wire, mask_wire_params = _best_brotli(mask_bytes)
        rows.append(
            {
                "_candidate_mask_bytes": mask_bytes,
                "candidate_id": _manifest_candidate_id(manifest_path),
                "candidate_mask_bytes": len(mask_bytes),
                "candidate_mask_sha256": _sha256_bytes(mask_bytes),
                "candidate_mask_wire_bytes": len(mask_wire),
                "candidate_mask_wire_sha256": _sha256_bytes(mask_wire),
                "candidate_mask_wire": mask_wire,
                "candidate_mask_wire_brotli_params": mask_wire_params,
                "existing_manifest": {
                    "archive_path": _repo_rel(manifest.get("archive", {}).get("path")),
                    "manifest_path": _repo_rel(manifest_path),
                    "manifest_sha256": _sha256_file(manifest_path),
                    "source_mask_sha256": source_mask_sha,
                },
                "mask_agreement": {
                    "overall": (candidate_mask.get("overall_agreement_vs_source") or {}),
                    "protected": (candidate_mask.get("protected_agreement_vs_source") or {}),
                    "unprotected": (candidate_mask.get("unprotected_agreement_vs_source") or {}),
                },
                "source": "existing_protected_mask_reencode_manifest",
                "status": "stream_loaded",
            }
        )
    return rows


def _write_candidate(
    *,
    candidate_id: str,
    source: Any,
    output_dir: Path,
    mask_bytes: bytes,
    mask_wire: bytes,
    mask_wire_meta: dict[str, Any],
    source_note: dict[str, Any],
    unpacker: Any,
    force: bool,
) -> dict[str, Any]:
    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    manifest_path = candidate_dir / "manifest.json"
    if archive_path.exists() and not force:
        raise FileExistsError(f"{archive_path} exists; pass --force")
    if _looks_like_mask_obu(mask_bytes):
        payload_format = "public_pr75_qzs3_qp1_segactions_p3"
        payload = _build_p3_payload(
            mask_wire=mask_wire,
            renderer_wire=source.raw_segments["renderer.bin"],
            actions_wire=source.raw_segments["seg_tile_actions.bin"],
            pose_wire=source.raw_segments["optimized_poses.qp1"],
        )
        charged_mask_wire = mask_wire
        charged_mask_wire_meta = mask_wire_meta
    else:
        payload_format = "rpk1_json_raw_mkv_with_pr79_decoded_actions"
        payload = _build_rpk1_payload(
            source_archive_sha256=source.archive_sha256,
            renderer=source.decoded["renderer.bin"],
            mask=mask_bytes,
            actions_wire=source.decoded["seg_tile_actions.bin"],
            pose_qp1=source.decoded["optimized_poses.qp1"],
        )
        charged_mask_wire = mask_bytes
        charged_mask_wire_meta = {
            **mask_wire_meta,
            "rpk1_raw_mask_member": True,
            "unused_p3_brotli_wire_bytes": len(mask_wire),
        }
    _write_single_stored_p_archive(archive_path, payload)
    zip_profile = _zip_profile(archive_path)
    try:
        validation = _validate_archive(
            archive_path=archive_path,
            source=source,
            candidate_mask_sha256=_sha256_bytes(mask_bytes),
            unpacker=unpacker,
        )
    except Exception as exc:
        validation = {
            "archive_validation_errors": [str(exc)],
            "candidate_mask_decoded_sha256": None,
            "candidate_mask_sha_matches_expected": False,
            "non_mask_runtime_members_preserved": False,
            "non_mask_runtime_member_checks": {},
            "payload_format": payload_format,
            "runtime_parser": _repo_rel(UNPACKER_PATH),
            "runtime_parser_ok": False,
            "runtime_parser_error": str(exc),
            "runtime_member_headers": {},
        }
    delta = int(zip_profile["archive_bytes"]) - int(source.archive_bytes)
    plausibility = _plausibility_gate(source_note)
    exact_eval_ready = (
        delta <= -TARGET_SAVINGS_BYTES
        and zip_profile["single_stored_p"]
        and not zip_profile["duplicate_member_names"]
        and not validation["archive_validation_errors"]
        and validation["runtime_parser_ok"]
        and validation["candidate_mask_sha_matches_expected"]
        and validation["non_mask_runtime_members_preserved"]
        and plausibility["passed"]
    )
    manifest = {
        "archive": {
            "bytes": int(zip_profile["archive_bytes"]),
            "delta_bytes_vs_pr79": delta,
            "path": _repo_rel(archive_path),
            "sha256": zip_profile["archive_sha256"],
        },
        "candidate_id": candidate_id,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "dispatch_recommendation": {
            "exact_cuda_eval_justified_after_lane_claim": exact_eval_ready,
            "lane_claim_required_before_dispatch": True,
            "remote_dispatch_performed": False,
            "reason": (
                "local PR79/S2 mask-body candidate passes byte, plausibility, and runtime preflight gates"
                if exact_eval_ready
                else "candidate does not clear the >=3KB byte/preflight recommendation gate"
            ),
        },
        "evidence_grade": "empirical_local_byte_screen",
        "mask_stream": {
            "decoded_bytes": len(mask_bytes),
            "decoded_sha256": _sha256_bytes(mask_bytes),
            "delta_wire_bytes_vs_pr79_mask_slice": len(charged_mask_wire) - len(source.raw_segments["masks.mkv"]),
            "member": "masks.mkv",
            "wire_brotli_params": charged_mask_wire_meta,
            "wire_bytes": len(charged_mask_wire),
            "wire_sha256": _sha256_bytes(charged_mask_wire),
        },
        "no_remote_dispatch_performed": True,
        "payload": {
            "bytes": len(payload),
            "format": payload_format,
            "member": MEMBER_NAME,
            "sha256": _sha256_bytes(payload),
        },
        "plausibility_gate": plausibility,
        "runtime_preflight": validation,
        "schema": MANIFEST_SCHEMA,
        "score_claim": False,
        "source": source_note,
        "source_pr79": {
            "archive_bytes": int(source.archive_bytes),
            "archive_path": _repo_rel(source.path),
            "archive_sha256": source.archive_sha256,
            "mask_decoded_bytes": len(source.decoded["masks.mkv"]),
            "mask_decoded_sha256": _sha256_bytes(source.decoded["masks.mkv"]),
            "mask_wire_bytes": len(source.raw_segments["masks.mkv"]),
            "mask_wire_sha256": _sha256_bytes(source.raw_segments["masks.mkv"]),
            "payload_bytes": len(source.payload),
            "payload_format": source.payload_format,
            "payload_sha256": source.payload_sha256,
        },
        "strict_compliance": {
            "deterministic_single_stored_p_zip": bool(zip_profile["single_stored_p"]),
            "external_sidecars_required": False,
            "fixed_runtime_streams_preserved_except_mask": validation["non_mask_runtime_members_preserved"],
            "hidden_or_duplicate_members": bool(zip_profile["duplicate_member_names"]),
            "scorer_loaded_by_tool": False,
        },
        "tool": TOOL,
        "zip_profile": zip_profile,
    }
    _write_json(manifest_path, manifest)
    return {
        "archive_bytes": manifest["archive"]["bytes"],
        "archive_path": manifest["archive"]["path"],
        "archive_sha256": manifest["archive"]["sha256"],
        "candidate_id": candidate_id,
        "delta_bytes_vs_pr79": delta,
        "exact_eval_ready_after_lane_claim": exact_eval_ready,
        "manifest_path": _repo_rel(manifest_path),
        "mask_stream": manifest["mask_stream"],
        "plausibility_gate": plausibility,
        "payload_format": payload_format,
        "runtime_preflight": validation,
        "score_claim": False,
        "source": source_note,
    }


def _select_recommendation(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ready = [
        candidate
        for candidate in candidates
        if bool(candidate.get("exact_eval_ready_after_lane_claim"))
    ]
    if not ready:
        tested = [
            {
                "candidate_id": candidate.get("candidate_id"),
                "delta_bytes_vs_pr79": candidate.get("delta_bytes_vs_pr79"),
            }
            for candidate in candidates
        ]
        return {
            "candidate": None,
            "decision": "finite_neighborhood_negative",
            "reason": "no candidate passed the >=3KB byte, ZIP, parser, and action-preflight gates",
            "tested_candidates": tested,
        }
    best = min(ready, key=lambda row: (int(row["archive_bytes"]), str(row["candidate_id"])))
    return {
        "candidate": best,
        "decision": "recommend_exact_cuda_eval_after_lane_claim",
        "reason": (
            "best local PR79/S2 mask-body candidate clears >=3KB archive savings "
            "and strict local runtime/preflight checks; no remote dispatch was performed"
        ),
    }


def build_profile(
    *,
    pr79_archive: Path = DEFAULT_PR79_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    candidate_manifest_globs: Sequence[str] = DEFAULT_CANDIDATE_MANIFEST_GLOBS,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _load_source_archive(pr79_archive.resolve())
    unpacker = _load_unpacker()

    candidates: list[dict[str, Any]] = []
    source_mask_wire = source.raw_segments["masks.mkv"]
    source_mask_decoded = source.decoded["masks.mkv"]
    lossless_wire, lossless_params = _best_brotli(source_mask_decoded)
    candidates.append(
        _write_candidate(
            candidate_id="p3_lossless_rebrotli_mask_control",
            source=source,
            output_dir=output_dir,
            mask_bytes=source_mask_decoded,
            mask_wire=lossless_wire,
            mask_wire_meta={
                "brotli_params": lossless_params,
                "lossless_vs_source_mask_decoded": True,
                "source_mask_wire_bytes": len(source_mask_wire),
            },
            source_note={"kind": "lossless_rebrotli_control"},
            unpacker=unpacker,
            force=force,
        )
    )

    manifest_paths = _candidate_manifest_paths(candidate_manifest_globs)
    stream_rows = _candidate_stream_rows(source=source, manifest_paths=manifest_paths)
    stream_errors = [row for row in stream_rows if row.get("status") != "stream_loaded"]
    for row in stream_rows:
        if row.get("status") != "stream_loaded":
            continue
        candidates.append(
            _write_candidate(
                candidate_id=row["candidate_id"],
                source=source,
                output_dir=output_dir,
                mask_bytes=row["_candidate_mask_bytes"],
                mask_wire=row["candidate_mask_wire"],
                mask_wire_meta={
                    "brotli_params": row["candidate_mask_wire_brotli_params"],
                    "lossless_vs_candidate_mask_decoded": True,
                },
                source_note={
                    "kind": row["source"],
                    "existing_manifest": row["existing_manifest"],
                    "mask_agreement": row["mask_agreement"],
                },
                unpacker=unpacker,
                force=force,
            )
        )

    candidates = sorted(candidates, key=lambda row: (int(row["archive_bytes"]), str(row["candidate_id"])))
    recommendation = _select_recommendation(candidates)
    profile = {
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "candidate_manifest_globs": list(candidate_manifest_globs),
        "candidate_manifest_paths_scanned": [_repo_rel(path) for path in manifest_paths],
        "candidates": candidates,
        "evidence_grade": "empirical_local_byte_screen",
        "finite_neighborhood": {
            "controls": ["p3_lossless_rebrotli_mask_control"],
            "matching_protected_mask_streams": sum(1 for row in stream_rows if row.get("status") == "stream_loaded"),
            "minimum_required_archive_savings_bytes": TARGET_SAVINGS_BYTES,
            "stream_errors": stream_errors,
        },
        "no_remote_dispatch_performed": True,
        "recommendation": recommendation,
        "schema": SCHEMA,
        "score_claim": False,
        "source_pr79": {
            "archive_bytes": int(source.archive_bytes),
            "archive_path": _repo_rel(source.path),
            "archive_sha256": source.archive_sha256,
            "decoded_mask_bytes": len(source_mask_decoded),
            "decoded_mask_sha256": _sha256_bytes(source_mask_decoded),
            "payload_bytes": len(source.payload),
            "payload_format": source.payload_format,
            "payload_sha256": source.payload_sha256,
            "raw_segment_bytes": {
                name: len(data) for name, data in sorted(source.raw_segments.items())
            },
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", profile)
    _write_json(
        output_dir / "recommendation.json",
        {
            "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
            "evidence_grade": profile["evidence_grade"],
            "no_remote_dispatch_performed": True,
            "recommendation": recommendation,
            "schema": RECOMMENDATION_SCHEMA,
            "score_claim": False,
            "tool": TOOL,
        },
    )
    return profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr79-archive", type=Path, default=DEFAULT_PR79_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-manifest-glob",
        action="append",
        dest="candidate_manifest_globs",
        default=None,
        help="Repo-relative glob for protected_mask_reencode_manifest.json inputs.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = build_profile(
        pr79_archive=args.pr79_archive,
        output_dir=args.output_dir,
        candidate_manifest_globs=tuple(args.candidate_manifest_globs or DEFAULT_CANDIDATE_MANIFEST_GLOBS),
        force=bool(args.force),
    )
    print(json.dumps(profile["recommendation"], indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
