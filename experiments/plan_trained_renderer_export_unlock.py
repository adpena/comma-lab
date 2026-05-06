#!/usr/bin/env python3
"""Plan the trained-renderer export unlock for C067 Block-FP self-compression.

This is a local-only readiness scanner.  It inspects known trained renderer,
Block-FP, and QBF result directories for candidate exports, checkpoints,
manifests, and preflight summaries.  It never dispatches remote work, never
loads scorers, and never promotes a score claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SCHEMA = "trained_renderer_export_unlock_plan_v1"
TOOL = "experiments/plan_trained_renderer_export_unlock.py"
LANE_ID = "c067_trained_renderer_self_compression_blockfp"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
C067_FRONTIER_SCORE = 0.31561703078448233
C067_FRONTIER_ARCHIVE_BYTES = 276_214
C067_FRONTIER_ARCHIVE_SHA256 = (
    "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"
)
C067_SOURCE_RENDERER_SHA256 = (
    "5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb"
)
TARGET_SCORES = (0.30, 0.24)
ACCEPTED_EXPORT_MAGICS = {
    b"QZS3": "QZS3",
    b"MQZ1": "MQZ1",
    b"QBF1": "QBF1",
}
KNOWN_RENDERER_MAGICS = {
    b"ASYM": "ASYM",
    b"FP4A": "FP4A",
    b"QFAI": "QFAI",
}
DEFAULT_SCAN_DIRS = (
    REPO_ROOT / "experiments/results/trained_renderer_blockfp_preflight_20260502_codex",
    REPO_ROOT / "experiments/results/c067_renderer_self_compression_20260502_blockfp_local_screen",
    REPO_ROOT / "experiments/results/blockfp_c067_candidate_20260502T_qbf1_sweep",
    REPO_ROOT / "experiments/results/blockfp_c067_candidate_20260502T_local_qbf1_b1024",
    REPO_ROOT / "experiments/results/lane_a_blockfp_c067_20260502_local_screen",
    REPO_ROOT / "experiments/results/c067_renderer_self_compression_v2_20260502",
    REPO_ROOT / "experiments/results/c067_self_compression_profile_20260502",
    REPO_ROOT / "experiments/results/fridrich_renderer",
    REPO_ROOT / "experiments/results/small_renderer_trained",
    REPO_ROOT / "experiments/results/lane_e_self_compression",
    REPO_ROOT / "experiments/results/lane_q_faithful_retrain_20260501",
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/trained_renderer_export_unlock_20260502_codex/"
    "trained_renderer_export_unlock_plan.json"
)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _first_bytes(path: Path, n: int = 4) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(n)
    except OSError:
        return b""


def _score_targets() -> dict[str, Any]:
    component_score = C067_FRONTIER_SCORE - (
        C067_FRONTIER_ARCHIVE_BYTES * RATE_SCORE_PER_BYTE
    )
    targets: list[dict[str, Any]] = []
    for target in TARGET_SCORES:
        unchanged_component_max_bytes = math.floor(
            (target - component_score) / RATE_SCORE_PER_BYTE
        )
        required_savings = C067_FRONTIER_ARCHIVE_BYTES - unchanged_component_max_bytes
        targets.append(
            {
                "target_score": target,
                "frontier_score_gap": round(C067_FRONTIER_SCORE - target, 15),
                "unchanged_component_max_archive_bytes": unchanged_component_max_bytes,
                "required_archive_byte_savings_vs_c067_if_components_unchanged": required_savings,
                "required_component_score_improvement_if_archive_bytes_unchanged": round(
                    C067_FRONTIER_SCORE - target, 15
                ),
            }
        )
    return {
        "score_formula": (
            "score = 100*seg_dist + sqrt(10*pose_dist) + "
            "25*archive_bytes/37545489"
        ),
        "rate_score_per_archive_byte": RATE_SCORE_PER_BYTE,
        "frontier": {
            "score": C067_FRONTIER_SCORE,
            "archive_bytes": C067_FRONTIER_ARCHIVE_BYTES,
            "archive_sha256": C067_FRONTIER_ARCHIVE_SHA256,
            "rate_score": C067_FRONTIER_ARCHIVE_BYTES * RATE_SCORE_PER_BYTE,
            "component_score_at_frontier": component_score,
        },
        "targets": targets,
    }


def _stacking_requirements(archive_bytes: int | None) -> dict[str, Any] | None:
    if archive_bytes is None:
        return None
    component_score = C067_FRONTIER_SCORE - (
        C067_FRONTIER_ARCHIVE_BYTES * RATE_SCORE_PER_BYTE
    )
    rate_score = archive_bytes * RATE_SCORE_PER_BYTE
    records: list[dict[str, Any]] = []
    for target in TARGET_SCORES:
        unchanged_component_score = component_score + rate_score
        component_improvement_needed = max(0.0, unchanged_component_score - target)
        max_bytes_unchanged_components = math.floor(
            (target - component_score) / RATE_SCORE_PER_BYTE
        )
        records.append(
            {
                "target_score": target,
                "candidate_archive_bytes": archive_bytes,
                "candidate_rate_score_if_components_unchanged": rate_score,
                "score_if_components_unchanged": unchanged_component_score,
                "additional_archive_byte_savings_needed_if_components_unchanged": max(
                    0, archive_bytes - max_bytes_unchanged_components
                ),
                "component_score_improvement_needed_at_candidate_bytes": round(
                    component_improvement_needed, 15
                ),
            }
        )
    return {
        "archive_bytes": archive_bytes,
        "targets": records,
    }


def _candidate_from_preflight_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    renderer_export = payload.get("renderer_export") or {}
    readiness = payload.get("h100_lightning_readiness") or {}
    best = payload.get("best_by_archive_bytes") or {}
    dispatchable = payload.get("best_dispatchable_after_pose_safety") or {}
    pose_safety_gate = (
        readiness.get("selected_pose_safety_gate")
        or dispatchable.get("pose_safety_gate")
        or best.get("pose_safety_gate")
        or {}
    )
    ready = bool(
        readiness.get("ready") is True
        and renderer_export.get("mode") == "trained_renderer_export"
        and renderer_export.get("dispatchable_trained_export") is True
        and renderer_export.get("same_as_source_renderer") is False
        and readiness.get("pose_safety_required") is True
        and pose_safety_gate.get("safe_for_exact_eval_dispatch") is True
        and readiness.get("next_commands_if_ready")
    )
    blockers: list[str] = []
    if renderer_export.get("mode") != "trained_renderer_export":
        blockers.append("preflight renderer_export.mode is not trained_renderer_export")
    if renderer_export.get("same_as_source_renderer") is True:
        blockers.append("preflight renderer export is identical to the C067 source renderer")
    if renderer_export.get("dispatchable_trained_export") is not True:
        blockers.append("preflight renderer export is not marked dispatchable")
    if readiness.get("ready") is not True:
        blockers.append(str(readiness.get("reason") or "preflight h100 readiness is false"))
    if readiness.get("pose_safety_required") is not True:
        blockers.append("preflight summary predates mandatory renderer pose-safety gate")
    if pose_safety_gate.get("safe_for_exact_eval_dispatch") is not True:
        blockers.append("renderer transplant pose-safety gate missing or failed")
    if readiness.get("next_commands_if_ready") is None:
        blockers.append("preflight did not emit H100/Lightning commands")
    selected = dispatchable if ready else best
    archive_bytes = _int_or_none(selected.get("archive_bytes"))
    return {
        "kind": "preflight_summary",
        "path": _repo_rel(path),
        "schema": payload.get("schema"),
        "candidate_id": selected.get("candidate_id"),
        "archive_path": selected.get("archive_path"),
        "archive_bytes": archive_bytes,
        "archive_sha256": selected.get("archive_sha256"),
        "renderer_export": renderer_export,
        "pose_safety_gate": pose_safety_gate,
        "h100_ready": ready,
        "blockers": sorted(set(item for item in blockers if item)),
        "stacking_requirements": _stacking_requirements(archive_bytes),
        "commands_if_ready": readiness.get("next_commands_if_ready") if ready else None,
        "score_claim": bool(payload.get("score_claim", False)),
        "promotion_eligible": bool(payload.get("promotion_eligible", False)),
    }


def _candidate_from_preflight_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    renderer_export = payload.get("renderer_export") or {}
    output = payload.get("output_archive") or {}
    archive_bytes = _int_or_none(output.get("bytes"))
    non_surrogate = (
        renderer_export.get("mode") == "trained_renderer_export"
        and renderer_export.get("dispatchable_trained_export") is True
        and renderer_export.get("same_as_source_renderer") is False
    )
    blockers: list[str] = []
    if not non_surrogate:
        blockers.append("candidate manifest is surrogate/no-op or missing trained export metadata")
    blockers.append("candidate manifest alone is not the preflight summary H100 gate")
    return {
        "kind": "preflight_candidate_manifest",
        "path": _repo_rel(path),
        "schema": payload.get("schema"),
        "candidate_id": payload.get("candidate_id"),
        "archive_path": output.get("path"),
        "archive_bytes": archive_bytes,
        "archive_sha256": output.get("sha256"),
        "renderer_export": renderer_export,
        "non_surrogate_export": non_surrogate,
        "h100_ready": False,
        "blockers": sorted(set(blockers)),
        "stacking_requirements": _stacking_requirements(archive_bytes),
        "commands_if_ready": None,
        "score_claim": bool(payload.get("score_claim", False)),
        "promotion_eligible": bool(payload.get("promotion_eligible", False)),
    }


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _score_from_eval(payload: dict[str, Any]) -> float | None:
    for key in ("score_recomputed_from_components", "final_score", "score"):
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _eval_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": _repo_rel(path),
        "archive_size_bytes": _int_or_none(payload.get("archive_size_bytes")),
        "archive_sha256": payload.get("archive_sha256") or payload.get("archive_sha"),
        "score": _score_from_eval(payload),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "n_samples": payload.get("n_samples"),
        "score_claim": False,
    }


def _nearby_exact_eval_records(path: Path, *, archive_bytes: int | None) -> list[dict[str, Any]]:
    """Find local exact-eval JSONs associated with a trained-renderer artifact.

    Q-FAITHFUL artifacts were harvested into both lane-local and global
    Lightning/Vast directories. The archive SHA is not always recorded by older
    eval JSONs, so matching is conservative: only q-faithful paths with matching
    archive bytes are attached as known negatives.
    """

    if archive_bytes is None:
        return []
    roots = (
        REPO_ROOT / "experiments/results/lightning_batch",
        REPO_ROOT / "experiments/results/vast_harvest",
        path.parents[0],
    )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("contest_auth_eval*.json"):
            rel = _repo_rel(candidate).lower()
            if "qfaithful" not in rel and "q_faithful" not in rel:
                continue
            payload = _read_json(candidate)
            if payload is None:
                continue
            if _int_or_none(payload.get("archive_size_bytes")) != archive_bytes:
                continue
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            records.append(_eval_record(candidate, payload))
    return sorted(records, key=lambda item: str(item["path"]))


def _candidate_from_qfaithful_export_provenance(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    renderer_bytes = _int_or_none(payload.get("renderer_bin_bytes"))
    renderer_sha = payload.get("renderer_bin_sha256")
    pose_shape = payload.get("pose_shape")
    return {
        "kind": "qfaithful_raw_export_requires_packed_preflight",
        "path": _repo_rel(path),
        "renderer_bin_path": payload.get("renderer_bin"),
        "renderer_bytes": renderer_bytes,
        "renderer_sha256": renderer_sha,
        "wire_format": "QFAI",
        "pose_shape": pose_shape,
        "non_surrogate_export": bool(
            isinstance(renderer_sha, str)
            and renderer_sha != C067_SOURCE_RENDERER_SHA256
            and renderer_bytes
        ),
        "h100_ready": False,
        "blockers": [
            "raw QFAI trained export must be packed/transcoded to QZS3/MQZ1/QBF1 and pass transplant preflight",
            "existing Q-FAITHFUL snapshots are exact-negative unless a successor fixes pose/geometry",
        ],
        "commands_if_ready": None,
        "score_claim": bool(payload.get("score_claim", False)),
        "promotion_eligible": bool(payload.get("promotion_eligible", False)),
    }


def _candidate_from_packed_renderer_payload_provenance(
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    archive_bytes = _int_or_none(payload.get("output_archive_bytes"))
    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    members = header.get("members") if isinstance(header, dict) else []
    renderer_row = None
    if isinstance(members, list):
        for row in members:
            if isinstance(row, dict) and row.get("name") == "renderer.bin":
                renderer_row = row
                break
    renderer_sha = renderer_row.get("sha256") if isinstance(renderer_row, dict) else None
    exact_records = _nearby_exact_eval_records(path, archive_bytes=archive_bytes)
    exact_negative = any(
        isinstance(record.get("score"), (int, float))
        and float(record["score"]) > C067_FRONTIER_SCORE
        for record in exact_records
    )
    blockers = [
        "packed trained renderer has not passed the non-surrogate transplant preflight"
    ]
    if exact_negative:
        blockers.append("matching Q-FAITHFUL archive has exact CUDA negative/collapse evidence")
    elif not exact_records:
        blockers.append("no matching exact CUDA eval JSON found for this packed trained renderer archive")
    return {
        "kind": "qfaithful_packed_renderer_export",
        "path": _repo_rel(path),
        "archive_path": payload.get("output_archive"),
        "archive_bytes": archive_bytes,
        "archive_sha256": payload.get("output_archive_sha256"),
        "renderer_sha256": renderer_sha,
        "payload_format": payload.get("payload_format"),
        "payload_member": payload.get("payload_member"),
        "pose_codec": payload.get("pose_codec"),
        "wire_format": "QZS3" if isinstance(renderer_sha, str) else None,
        "non_surrogate_export": bool(
            isinstance(renderer_sha, str) and renderer_sha != C067_SOURCE_RENDERER_SHA256
        ),
        "known_exact_negative": exact_negative,
        "exact_eval_records": exact_records,
        "h100_ready": False,
        "blockers": blockers,
        "stacking_requirements": _stacking_requirements(archive_bytes),
        "commands_if_ready": None,
        "score_claim": bool(payload.get("score_claim", False)),
        "promotion_eligible": bool(payload.get("promotion_eligible", False)),
    }


def _candidate_from_blockfp_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("best_by_output_archive_bytes") or {}
    archive_bytes = _int_or_none(best.get("bytes"))
    return {
        "kind": "blockfp_source_renderer_summary",
        "path": _repo_rel(path),
        "schema": payload.get("schema"),
        "candidate_id": best.get("candidate_id"),
        "archive_bytes": archive_bytes,
        "archive_sha256": best.get("sha256"),
        "non_surrogate_export": False,
        "h100_ready": False,
        "blockers": [
            "Block-FP summary repacks the source renderer; it is not a trained export preflight",
            "best archive bytes are above the C067 frontier"
            if archive_bytes is not None and archive_bytes > C067_FRONTIER_ARCHIVE_BYTES
            else "missing archive-byte evidence against the C067 frontier",
        ],
        "stacking_requirements": _stacking_requirements(archive_bytes),
        "commands_if_ready": None,
        "score_claim": bool(payload.get("score_claim", False)),
        "promotion_eligible": bool(payload.get("promotion_eligible", False)),
    }


def _candidate_from_file(path: Path) -> dict[str, Any] | None:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".json":
        return None
    if suffix == ".zip":
        return {
            "kind": "archive_without_trained_preflight_gate",
            "path": _repo_rel(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "h100_ready": False,
            "blockers": [
                "archive file exists but no non-surrogate trained-renderer preflight summary gates it"
            ],
        }
    if suffix in {".pt", ".pth", ".ckpt", ".safetensors"}:
        return {
            "kind": "checkpoint_requires_export",
            "path": _repo_rel(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "h100_ready": False,
            "blockers": [
                "checkpoint is not a pickle-free QZS3/MQZ1/QBF1 renderer export",
                "run an export step, then experiments/preflight_trained_renderer_transplant.py locally",
            ],
        }
    if suffix != ".bin" and "renderer" not in name:
        return None
    magic = _first_bytes(path, 4)
    sha = _sha256_file(path)
    if magic in ACCEPTED_EXPORT_MAGICS:
        same_as_source = sha == C067_SOURCE_RENDERER_SHA256
        blockers = [
            "accepted renderer export magic found, but it has not passed the trained transplant preflight"
        ]
        if same_as_source:
            blockers.append("renderer bytes match the C067 source renderer surrogate")
        return {
            "kind": "renderer_export_candidate",
            "path": _repo_rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha,
            "wire_format": ACCEPTED_EXPORT_MAGICS[magic],
            "same_as_c067_source_renderer": same_as_source,
            "non_surrogate_export": not same_as_source,
            "h100_ready": False,
            "blockers": blockers,
        }
    if magic in KNOWN_RENDERER_MAGICS:
        non_surrogate_qfai = magic == b"QFAI" and sha != C067_SOURCE_RENDERER_SHA256
        return {
            "kind": "qfaithful_raw_renderer_export" if non_surrogate_qfai else "unsupported_renderer_wire_format",
            "path": _repo_rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha,
            "wire_format": KNOWN_RENDERER_MAGICS[magic],
            "non_surrogate_export": non_surrogate_qfai,
            "h100_ready": False,
            "blockers": [
                "raw QFAI must be packed/transcoded before transplant preflight"
                if non_surrogate_qfai
                else "renderer uses a legacy/non-preflight wire format",
                "convert/export to QZS3, MQZ1, or QBF1 before transplant preflight",
            ],
        }
    return {
        "kind": "unknown_renderer_artifact",
        "path": _repo_rel(path),
        "bytes": path.stat().st_size,
        "sha256": sha,
        "magic_hex": magic.hex(),
        "h100_ready": False,
        "blockers": [
            "artifact is not an accepted pickle-free trained-renderer export"
        ],
    }


def _interesting_files(scan_dirs: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for raw_dir in scan_dirs:
        root = raw_dir.resolve()
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and _is_interesting(path):
                files.append(path)
    return sorted(set(files), key=lambda item: _repo_rel(item))


def _is_interesting(path: Path) -> bool:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".json", ".zip", ".pt", ".pth", ".ckpt", ".safetensors"}:
        return True
    if suffix == ".bin" and ("renderer" in name or path.parent.name.lower().startswith("unpacked")):
        return True
    return False


def scan_candidates(scan_dirs: tuple[Path, ...]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in _interesting_files(scan_dirs):
        if path.suffix.lower() == ".json":
            payload = _read_json(path)
            if payload is None:
                continue
            schema = payload.get("schema")
            if schema == "trained_renderer_blockfp_transplant_preflight_v1":
                candidates.append(_candidate_from_preflight_summary(path, payload))
            elif schema == "trained_renderer_blockfp_candidate_manifest_v1":
                candidates.append(_candidate_from_preflight_manifest(path, payload))
            elif schema == "blockfp_c067_archive_summary_v1":
                candidates.append(_candidate_from_blockfp_summary(path, payload))
            elif "renderer_bin_sha256" in payload and "renderer_bin_bytes" in payload:
                candidates.append(_candidate_from_qfaithful_export_provenance(path, payload))
            elif (
                "packed_renderer_payload_provenance" in path.name
                or {"output_archive_bytes", "payload_format", "header"}.issubset(payload)
            ):
                candidates.append(_candidate_from_packed_renderer_payload_provenance(path, payload))
            continue
        candidate = _candidate_from_file(path)
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: (str(item.get("kind")), str(item.get("path"))))


def _readiness(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [
        item
        for item in candidates
        if item.get("kind") == "preflight_summary" and item.get("h100_ready") is True
    ]
    ready = sorted(
        ready,
        key=lambda item: (
            int(item.get("archive_bytes") or 10**18),
            str(item.get("candidate_id") or ""),
            str(item.get("path") or ""),
        ),
    )
    blockers: list[str] = []
    if not ready:
        blockers.append("no non-surrogate trained-renderer archive passed preflight")
    if not any(item.get("kind") == "preflight_summary" for item in candidates):
        blockers.append("no trained-renderer Block-FP preflight summary was found")
    if not any(item.get("non_surrogate_export") is True for item in candidates):
        blockers.append("no non-surrogate QZS3/MQZ1/QBF1 export candidate was found")
    if any(item.get("score_claim") is True for item in candidates):
        blockers.append("one or more planning artifacts attempted to carry a score claim")
    best = ready[0] if ready else None
    return {
        "verdict": "h100_ready_after_claim" if best else "blocked_no_h100_dispatch",
        "remote_gpu_dispatch_performed": False,
        "h100_lightning_commands": best.get("commands_if_ready") if best else None,
        "selected_preflight_candidate": best,
        "blockers": sorted(set(blockers)),
    }


def build_plan(scan_dirs: tuple[Path, ...] = DEFAULT_SCAN_DIRS) -> dict[str, Any]:
    """Build a deterministic local readiness plan."""

    resolved_scan_dirs = tuple(Path(item).resolve() for item in scan_dirs)
    candidates = scan_candidates(resolved_scan_dirs)
    non_surrogate_candidates = [
        item for item in candidates if item.get("non_surrogate_export") is True
    ]
    h100_ready = [
        item
        for item in candidates
        if item.get("kind") == "preflight_summary" and item.get("h100_ready") is True
    ]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "lane_id": LANE_ID,
        "scan_dirs": [_repo_rel(item) for item in resolved_scan_dirs],
        "planning_constraints": {
            "local_only": True,
            "remote_gpu_dispatch": False,
            "scorers_loaded": False,
            "preflight_required_for_h100_commands": True,
            "renderer_pose_safety_preflight_required_for_h100_commands": True,
            "deterministic_json": True,
        },
        "byte_targets": _score_targets(),
        "candidate_count": len(candidates),
        "non_surrogate_candidate_count": len(non_surrogate_candidates),
        "h100_ready_preflight_count": len(h100_ready),
        "readiness": _readiness(candidates),
        "candidates": candidates,
    }


def write_plan(output_path: Path, *, scan_dirs: tuple[Path, ...] = DEFAULT_SCAN_DIRS) -> dict[str, Any]:
    plan = build_plan(scan_dirs=scan_dirs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_json_bytes(plan))
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        action="append",
        type=Path,
        default=None,
        help="Directory or file to scan. May be supplied multiple times.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scan_dirs = tuple(args.scan_dir) if args.scan_dir else DEFAULT_SCAN_DIRS
    plan = write_plan(args.output, scan_dirs=scan_dirs)
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
