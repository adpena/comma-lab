#!/usr/bin/env python3
"""Build a contest-faithful frontier stack reconstruction plan.

This tool is local-only. It inventories current public/frontier archives,
matches exact CUDA eval artifacts when available, and ranks stack opportunities
without dispatching, training, or loading scorer models.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import struct
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_frontier_stack_reconstruction.py"
SCHEMA = "frontier_stack_reconstruction_plan_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/frontier_stack_reconstruction_20260504_codex"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_POINTS_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES

DEFAULT_ARCHIVE_GLOBS: tuple[str, ...] = (
    "experiments/results/public_pr85_intake_20260503_codex/archive.zip",
    "experiments/results/public_pr90_intake_20260504_worker/archive.zip",
    "experiments/results/public_pr91_intake_20260504_codex/archive.zip",
    "experiments/results/public_pr91_intake_20260504_worker/archive.zip",
    "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/*/archive.zip",
    "experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_codex/*/archive.zip",
    "experiments/results/pr85_qrgb_pair_atom_combo_candidates_20260504_worker/*/archive.zip",
    "experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/*/archive.zip",
    "experiments/results/pr91_qrgb_pair_atom_candidates_20260504_codex/*/archive.zip",
    "experiments/results/qp1_pose_active_subspace_worker_20260503/*/archive.zip",
    "experiments/results/renderer_pose_stack_worker_20260503/*/archive.zip",
    "experiments/results/vast_harvest/archive_eval_line_search_*_20260502*/archive.zip",
)

DEFAULT_EXACT_EVAL_GLOBS: tuple[str, ...] = (
    "experiments/results/lightning_batch/**/contest_auth_eval.adjudicated.json",
    "experiments/results/lightning_batch/**/contest_auth_eval.json",
    "experiments/results/vast_harvest/**/contest_auth_eval.json",
)

DEFAULT_JSON_GLOBS: tuple[str, ...] = (
    "experiments/results/public_pr85_intake_20260503_codex/profile_pr85_bundle.json",
    "experiments/results/pr85_archive_bit_budget_20260504_codex/profile_pr85_archive_bit_budget.json",
    "experiments/results/public_pr90_intake_20260504_worker/payload_probe.json",
    "experiments/results/public_pr90_intake_20260504_worker/pr90_pull.json",
    "experiments/results/pr90_qma9_mask_prior_transfer_20260504_worker/analysis.json",
    "experiments/results/pr90_qma9_mask_prior_transfer_20260504_worker/ranked_candidate_policy.json",
    "experiments/results/public_pr91_intake_20260504_worker/pr91_archive_anatomy.json",
    "experiments/results/public_pr91_intake_20260504_worker/pr91_transfer_decisions.json",
    "experiments/results/public_pr91_intake_20260504_codex/pr91_hpm1_local_preflight_20260504_codex.json",
    "experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_static_contract_20260504_codex.json",
    "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/candidate_summary.json",
    "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/*/manifest.json",
    "experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/candidate_summary.json",
    "experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/*/manifest.json",
    "experiments/results/pr91_qrgb_pair_atom_candidates_20260504_codex/planning.json",
    "experiments/results/pr91_qrgb_pair_atom_candidates_20260504_codex/*/manifest.json",
    "experiments/results/pr85_full_stack_opportunity_matrix_20260504*/pr85_full_stack_opportunity_matrix.json",
)


class FrontierPlanError(ValueError):
    """Raised when explicit planner input is malformed."""


def _rel(path: Path | str, *, root: Path = REPO_ROOT) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrontierPlanError(f"invalid JSON input: {_rel(path)}") from exc
    if not isinstance(payload, dict):
        raise FrontierPlanError(f"JSON input must be an object: {_rel(path)}")
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: dict[str, Any]) -> str:
    stable = {k: v for k, v in payload.items() if k != "stable_plan_digest_sha256"}
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return _sha256_bytes(encoded)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _rate_score(bytes_value: int | None) -> float | None:
    if bytes_value is None:
        return None
    return round(bytes_value * RATE_POINTS_PER_BYTE, 12)


def _score_from_components(*, archive_bytes: int, seg_dist: float, pose_dist: float) -> float:
    return 100.0 * seg_dist + math.sqrt(10.0 * pose_dist) + 25.0 * archive_bytes / ORIGINAL_VIDEO_BYTES


def _expand_globs(root: Path, patterns: tuple[str, ...] | list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            if Path(pattern).is_absolute():
                paths.extend(Path(path) for path in glob.glob(pattern, recursive=True))
            else:
                paths.extend(root.glob(pattern))
        else:
            path = Path(pattern)
            paths.append(path if path.is_absolute() else root / path)
    return sorted({path for path in paths if path.is_file()}, key=lambda p: str(p))


def _dedupe_exact_eval_paths(paths: list[Path]) -> list[Path]:
    by_parent: dict[Path, Path] = {}
    for path in paths:
        current = by_parent.get(path.parent)
        if current is None or path.name.endswith(".adjudicated.json"):
            by_parent[path.parent] = path
    return sorted(by_parent.values(), key=lambda p: str(p))


def discover_inputs(
    root: Path = REPO_ROOT,
    *,
    archive_globs: list[str] | None = None,
    exact_eval_globs: list[str] | None = None,
    json_globs: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Discover local artifact inputs with optional test/operator overrides."""

    archives = _expand_globs(root, archive_globs or list(DEFAULT_ARCHIVE_GLOBS))
    exact_evals = _dedupe_exact_eval_paths(
        _expand_globs(root, exact_eval_globs or list(DEFAULT_EXACT_EVAL_GLOBS))
    )
    json_inputs = _expand_globs(root, json_globs or list(DEFAULT_JSON_GLOBS))
    return {"archives": archives, "exact_evals": exact_evals, "json_inputs": json_inputs}


def _infer_archive_id(path: Path) -> str:
    rel = _rel(path)
    parent = path.parent.name
    text = rel.lower()
    if "public_pr85" in text:
        return "public_pr85_exact_anchor"
    if "public_pr90" in text:
        return "public_pr90_qrepro_external"
    if "public_pr91" in text:
        return "public_pr91_hpm1_external"
    if "stbm1br_qrgb" in text:
        return parent
    if "stbm1br_mask_recode" in text:
        return "pr85_stbm1br_mask_recode"
    if "pr91_qrgb" in text:
        return parent
    if "qrgb" in text:
        return parent
    if "line_search" in text:
        return parent
    if "renderer_pose_stack" in text:
        return parent
    if "qp1_pose_active_subspace" in text:
        return parent
    return parent


def _infer_family(path: Path) -> str:
    text = _rel(path).lower()
    if "public_pr85" in text:
        return "public_pr85"
    if "public_pr90" in text:
        return "public_pr90"
    if "public_pr91" in text:
        return "public_pr91"
    if "stbm1br_qrgb" in text:
        return "pr85_stbm1br_plus_qrgb"
    if "stbm1br_mask_recode" in text:
        return "pr85_stbm1br"
    if "pr91_qrgb" in text:
        return "pr91_qrgb"
    if "qrgb" in text:
        return "pr85_qrgb"
    if "line_search" in text:
        return "line_search"
    if "renderer_pose_stack" in text:
        return "renderer_pose_stack"
    if "qp1_pose_active_subspace" in text:
        return "pose_active_subspace"
    return "other"


def _local_header_name(zip_handle, info: zipfile.ZipInfo) -> str | None:
    zip_handle.fp.seek(info.header_offset)
    header = zip_handle.fp.read(30)
    if len(header) != 30:
        return None
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50:
        return None
    name_len = fields[9]
    extra_len = fields[10]
    raw_name = zip_handle.fp.read(name_len)
    zip_handle.fp.read(extra_len)
    try:
        return raw_name.decode("utf-8")
    except UnicodeDecodeError:
        return raw_name.decode("cp437", "replace")


def _zip_path_violation(name: str) -> str | None:
    pure = Path(name)
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return "absolute_member_path"
    if any(part == ".." for part in pure.parts):
        return "parent_traversal_member_path"
    if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
        return "macos_resource_fork_member"
    if any(part.startswith(".") for part in pure.parts if part not in {"."}):
        return "hidden_member_path"
    return None


def profile_archive(path: Path, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return strict ZIP/member byte accounting for one archive."""

    archive_bytes = path.stat().st_size
    archive_sha = _sha256_file(path)
    violations: list[str] = []
    members: list[dict[str, Any]] = []
    duplicate_names: list[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        seen: set[str] = set()
        for info in infos:
            if info.filename in seen:
                duplicate_names.append(info.filename)
            seen.add(info.filename)
            local_name = _local_header_name(zf, info)
            if local_name != info.filename:
                violations.append(f"local_central_name_mismatch:{info.filename}")
            path_violation = _zip_path_violation(info.filename)
            if path_violation:
                violations.append(f"{path_violation}:{info.filename}")
            data = zf.read(info)
            members.append(
                {
                    "name": info.filename,
                    "local_header_name": local_name,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                    "zip_stored": info.compress_type == zipfile.ZIP_STORED,
                    "crc32_hex": f"{info.CRC:08x}",
                    "sha256": _sha256_bytes(data),
                }
            )
    if duplicate_names:
        violations.append("duplicate_member_names:" + ",".join(sorted(duplicate_names)))
    member_total = sum(int(row["file_size"]) for row in members)
    profile = {
        "artifact_id": _infer_archive_id(path),
        "family": _infer_family(path),
        "path": _rel(path, root=root),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "member_count": len(members),
        "members": members,
        "member_total_uncompressed_bytes": member_total,
        "zip_overhead_bytes": archive_bytes - member_total,
        "strict_zip_ok": not violations,
        "violations": sorted(set(violations)),
    }
    if members:
        profile["primary_member"] = members[0]["name"]
        profile["primary_member_bytes"] = members[0]["file_size"]
        profile["primary_member_sha256"] = members[0]["sha256"]
    return profile


def _json_by_suffix(json_rows: list[tuple[Path, dict[str, Any]]], suffix: str) -> dict[str, Any] | None:
    matches = [(path, payload) for path, payload in json_rows if str(path).endswith(suffix)]
    return matches[-1][1] if matches else None


def _segments_from_pr85_profile(json_rows: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    payload = _json_by_suffix(json_rows, "profile_pr85_bundle.json")
    if not payload:
        payload = _json_by_suffix(json_rows, "profile_pr85_archive_bit_budget.json")
    segments = payload.get("segments") if isinstance(payload, dict) else None
    return [row for row in segments if isinstance(row, dict)] if isinstance(segments, list) else []


def _segments_from_pr90_probe(json_rows: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    payload = _json_by_suffix(json_rows, "payload_probe.json")
    slices = payload.get("slices") if isinstance(payload, dict) else None
    if not isinstance(slices, dict):
        return []
    rows = []
    for name, row in sorted(slices.items(), key=lambda item: int(item[1].get("offset", 0)) if isinstance(item[1], dict) else 0):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "name": str(name),
                "bytes": _as_int(row.get("len")),
                "offset": _as_int(row.get("offset")),
                "runtime_magic": row.get("runtime_magic"),
            }
        )
    return rows


def _segments_from_pr91_anatomy(json_rows: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    payload = _json_by_suffix(json_rows, "pr91_archive_anatomy.json")
    segments = payload.get("segments") if isinstance(payload, dict) else None
    return [row for row in segments if isinstance(row, dict)] if isinstance(segments, list) else []


def _runtime_contract(profile: dict[str, Any], json_rows: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    family = profile["family"]
    if family == "public_pr85":
        return {
            "contract_id": "pr85_v5_micro_single_member_x_qma9_qh0_qp1_sidechannels",
            "inflate_script": "experiments/results/public_pr85_intake_20260503_codex/replay_submission/inflate.sh",
            "members": "single stored x",
            "segments": _segments_from_pr85_profile(json_rows),
            "score_truth_gate": "exact CUDA auth eval through archive.zip -> inflate.sh -> upstream/evaluate.py",
        }
    if family == "public_pr90":
        return {
            "contract_id": "pr90_qrepro_single_member_p_fixed_offsets_stbm_qfq4_pose_qrgb",
            "inflate_script": "experiments/results/public_pr90_intake_20260504_worker/pr90_src/submissions/qrepro/inflate.sh",
            "members": "single stored p",
            "segments": _segments_from_pr90_probe(json_rows),
            "score_truth_gate": "external PR report only until local exact CUDA replay exists",
        }
    if family in {"public_pr91", "pr91_qrgb"}:
        return {
            "contract_id": "pr91_hpm1_pr85_v5_single_member_x_hpac_mask",
            "inflate_script": "experiments/results/public_pr91_intake_20260504_codex/replay_submission/hpac_coder_hybrid/inflate.sh",
            "members": "single stored x",
            "segments": _segments_from_pr91_anatomy(json_rows),
            "score_truth_gate": "blocked by local HPM1 decode failure until parity is fixed",
        }
    if family == "pr85_stbm1br":
        return {
            "contract_id": "pr85_v5_with_stbm1br_mask_recode",
            "inflate_script": "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/replay_submission_stbm/inflate.sh",
            "members": "single stored x",
            "segments": [],
            "score_truth_gate": "standalone exact CUDA auth eval required before stack promotion",
        }
    return {
        "contract_id": f"{family}_archive_candidate",
        "inflate_script": None,
        "members": "derived from archive member accounting",
        "segments": [],
        "score_truth_gate": "exact CUDA auth eval required before score promotion",
    }


def _load_exact_eval(path: Path, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    payload = _read_json(path)
    archive_bytes = _as_int(payload.get("archive_size_bytes"))
    seg = _as_float(payload.get("avg_segnet_dist"))
    pose = _as_float(payload.get("avg_posenet_dist"))
    score = _as_float(payload.get("score_recomputed_from_components"))
    recomputed = None
    if archive_bytes is not None and seg is not None and pose is not None:
        recomputed = round(_score_from_components(archive_bytes=archive_bytes, seg_dist=seg, pose_dist=pose), 15)
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    samples = _as_int(payload.get("n_samples"))
    device = provenance.get("device")
    archive_sha = provenance.get("archive_sha256")
    gpu_model = provenance.get("gpu_model")
    is_full_cuda = samples == 600 and score is not None and device == "cuda"
    return {
        "path": _rel(path, root=root),
        "parent": _rel(path.parent, root=root),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "n_samples": samples,
        "score": score,
        "score_recomputed_from_components": recomputed,
        "score_recompute_abs_delta": (
            round(abs(score - recomputed), 15) if score is not None and recomputed is not None else None
        ),
        "seg_dist": seg,
        "pose_dist": pose,
        "device": device,
        "gpu_model": gpu_model,
        "gpu_t4_match": provenance.get("gpu_t4_match"),
        "evidence_status": "exact_cuda_full_600" if is_full_cuda else "non_promotable_or_incomplete",
    }


def _match_exact_evals(
    profiles: list[dict[str, Any]], exact_evals: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    by_sha: dict[str, list[dict[str, Any]]] = {}
    for row in exact_evals:
        sha = row.get("archive_sha256")
        if isinstance(sha, str):
            by_sha.setdefault(sha, []).append(row)
    matches: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        sha = profile.get("archive_sha256")
        rows = by_sha.get(sha, []) if isinstance(sha, str) else []
        matches[profile["artifact_id"]] = sorted(
            rows,
            key=lambda row: (
                row.get("evidence_status") != "exact_cuda_full_600",
                row.get("score") if row.get("score") is not None else float("inf"),
                row.get("path"),
            ),
        )
    return matches


def _best_pr85_anchor(exact_evals: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [
        row
        for row in exact_evals
        if "public_pr85" in row["path"]
        and row.get("evidence_status") == "exact_cuda_full_600"
        and row.get("score") is not None
    ]
    if not rows:
        return None
    return sorted(rows, key=lambda row: (float(row["score"]), row["path"]))[0]


def _reported_pr90_from_pull(json_rows: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any] | None:
    payload = _json_by_suffix(json_rows, "pr90_pull.json")
    if not payload:
        return None
    body = str(payload.get("body") or "")
    def number(pattern: str) -> float | None:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if not match:
            return None
        return float(match.group(1).replace(",", ""))

    archive_size = number(r"Submission file size:\s*([0-9,]+)")
    pose = number(r"Average PoseNet Distortion:\s*([0-9.]+)")
    seg = number(r"Average SegNet Distortion:\s*([0-9.]+)")
    score = number(r"Final score:\s*([0-9.]+)")
    recomputed = None
    if archive_size is not None and pose is not None and seg is not None:
        recomputed = _score_from_components(archive_bytes=int(archive_size), seg_dist=seg, pose_dist=pose)
    return {
        "source": _rel(next(path for path, payload2 in json_rows if payload2 is payload)),
        "external_reported_score": score,
        "external_recomputed_score": round(recomputed, 15) if recomputed is not None else None,
        "archive_bytes": int(archive_size) if archive_size is not None else None,
        "pose_dist": pose,
        "seg_dist": seg,
        "score_claim": False,
        "evidence_status": "external_public_pr_text_not_local_exact",
    }


def _external_pr91_report(json_rows: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any] | None:
    payload = _json_by_suffix(json_rows, "pr91_transfer_decisions.json")
    source_pr = payload.get("source_pr") if isinstance(payload, dict) else None
    claimed = source_pr.get("claimed_report") if isinstance(source_pr, dict) else None
    if not isinstance(claimed, dict):
        return None
    archive_bytes = _as_int(claimed.get("archive_bytes"))
    pose = _as_float(claimed.get("pose_dist"))
    seg = _as_float(claimed.get("seg_dist"))
    recomputed = None
    if archive_bytes is not None and pose is not None and seg is not None:
        recomputed = _score_from_components(archive_bytes=archive_bytes, seg_dist=seg, pose_dist=pose)
    return {
        "source": "experiments/results/public_pr91_intake_20260504_worker/pr91_transfer_decisions.json",
        "external_reported_score": _as_float(claimed.get("exact_score_external_text")),
        "external_recomputed_score": round(recomputed, 15) if recomputed is not None else None,
        "archive_bytes": archive_bytes,
        "pose_dist": pose,
        "seg_dist": seg,
        "score_claim": False,
        "evidence_status": "external_public_pr_text_not_local_exact",
    }


def _best_exact_delta_for_markers(
    exact_evals: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    markers: tuple[str, ...],
) -> dict[str, Any] | None:
    if baseline is None or baseline.get("score") is None:
        return None
    rows = []
    baseline_score = float(baseline["score"])
    baseline_bytes = _as_int(baseline.get("archive_bytes"))
    for row in exact_evals:
        path = row["path"].lower()
        if not any(marker in path for marker in markers):
            continue
        if row.get("evidence_status") != "exact_cuda_full_600" or row.get("score") is None:
            continue
        score = float(row["score"])
        archive_bytes = _as_int(row.get("archive_bytes"))
        rows.append(
            {
                **row,
                "score_delta_vs_pr85": round(score - baseline_score, 12),
                "archive_byte_delta_vs_pr85": (
                    archive_bytes - baseline_bytes
                    if archive_bytes is not None and baseline_bytes is not None
                    else None
                ),
            }
        )
    if not rows:
        return None
    return sorted(rows, key=lambda row: (row["score_delta_vs_pr85"], row["path"]))[0]


def _opportunity(
    *,
    rank_seed: int,
    opportunity_id: str,
    surface: str,
    expected_bytes_saved_vs_pr85: int | None,
    expected_component_delta: dict[str, Any],
    evidence_status: str,
    dispatch_readiness: str,
    failure_modes: list[str],
    gates: list[str],
    adversarial_review: list[str],
    source_artifacts: list[str],
    recommendation: str,
    blocked: bool = False,
    refuted: bool = False,
    exact_eval_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "rank_seed": rank_seed,
        "opportunity_id": opportunity_id,
        "surface": surface,
        "expected_bytes_saved_vs_pr85": expected_bytes_saved_vs_pr85,
        "expected_rate_score_delta_if_components_unchanged": (
            round(-expected_bytes_saved_vs_pr85 * RATE_POINTS_PER_BYTE, 12)
            if expected_bytes_saved_vs_pr85 is not None
            else None
        ),
        "expected_component_delta": expected_component_delta,
        "exact_eval_delta_vs_pr85": exact_eval_delta,
        "evidence_status": evidence_status,
        "dispatch_readiness": dispatch_readiness,
        "blocked": blocked,
        "refuted": refuted,
        "failure_modes": failure_modes,
        "required_gates": gates,
        "recursive_adversarial_review": adversarial_review,
        "recommendation": recommendation,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "source_artifacts": source_artifacts,
    }


def _build_opportunities(
    *,
    profiles: list[dict[str, Any]],
    exact_evals: list[dict[str, Any]],
    json_rows: list[tuple[Path, dict[str, Any]]],
    baseline: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    by_family = {profile["family"]: profile for profile in profiles}
    pr85_bytes = _as_int((baseline or {}).get("archive_bytes")) or _as_int(
        (by_family.get("public_pr85") or {}).get("archive_bytes")
    )
    pr91_report = _external_pr91_report(json_rows)
    pr91_diff = (_json_by_suffix(json_rows, "pr91_transfer_decisions.json") or {}).get("pr85_diff", {})
    pr90_report = _reported_pr90_from_pull(json_rows)
    pr90_segments = _segments_from_pr90_probe(json_rows)
    pr91_segments = _segments_from_pr91_anatomy(json_rows)
    pr85_segments = _segments_from_pr85_profile(json_rows)
    pr85_mask = next((_as_int(row.get("bytes")) for row in pr85_segments if row.get("name") == "mask"), None)
    pr85_model = next((_as_int(row.get("bytes")) for row in pr85_segments if row.get("name") == "model"), None)
    pr91_mask = next((_as_int(row.get("bytes")) for row in pr91_segments if row.get("name") == "mask"), None)
    pr90_mask = next((_as_int(row.get("bytes")) for row in pr90_segments if row.get("name") == "mask_body"), None)
    pr90_model = next((_as_int(row.get("bytes")) for row in pr90_segments if row.get("name") == "model_body"), None)

    exact_qrgb = _best_exact_delta_for_markers(exact_evals, baseline, ("qrgb",))
    exact_randmulti = _best_exact_delta_for_markers(exact_evals, baseline, ("randmulti_top", "minus_randmulti"))
    exact_line = _best_exact_delta_for_markers(exact_evals, baseline, ("line_search",))
    exact_stbm = _best_exact_delta_for_markers(exact_evals, baseline, ("stbm1br", "stbm"))

    stack_summary = _json_by_suffix(json_rows, "pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/candidate_summary.json")
    stbm_summary = _json_by_suffix(json_rows, "pr85_stbm1br_mask_recode_20260504_worker/candidate_summary.json")
    pr91_qrgb_plan = _json_by_suffix(json_rows, "pr91_qrgb_pair_atom_candidates_20260504_codex/planning.json")

    opportunities: list[dict[str, Any]] = []
    if pr91_report and pr91_mask is not None and pr85_mask is not None:
        bytes_saved = pr85_mask - pr91_mask
        opportunities.append(
            _opportunity(
                rank_seed=10,
                opportunity_id="recover_pr91_hpm1_mask_contract_on_pr85_runtime",
                surface="PR91 HPM1 HPAC mask stream as PR85-compatible mask replacement",
                expected_bytes_saved_vs_pr85=bytes_saved,
                expected_component_delta={
                    "claim_basis": "external public PR text reports PR85-identical PoseNet and SegNet components",
                    "external_reported_score": pr91_report.get("external_reported_score"),
                    "external_recomputed_score": pr91_report.get("external_recomputed_score"),
                    "local_status": "invalid until HPM1 decode/reencode parity passes",
                },
                evidence_status="external_frontier_signal_plus_local_decode_failure",
                dispatch_readiness="blocked",
                blocked=True,
                failure_modes=[
                    "local exact replays failed before score with HPM1 entropy-model decode assertion",
                    "external public PR score is not local exact CUDA evidence",
                    "HPAC probability/runtime dependency drift can silently change token decode",
                ],
                gates=[
                    "full local HPM1 decode of 600x384x512 tokens",
                    "byte-exact HPM1 reencode or reviewed source-contract explanation",
                    "runtime output parity smoke without scorer loads",
                    "dispatch claim before exact CUDA auth eval",
                    "T4/equivalent exact CUDA auth eval and adjudication",
                ],
                adversarial_review=[
                    "Best route to beat PR91 must first reproduce PR91; otherwise every derivative is external-only.",
                    "Because non-mask PR91 segments are byte-identical to PR85, any component movement is a mask decoder/runtime bug until proven otherwise.",
                    "Do not stack QRGB or model changes on HPM1 while the base mask stream cannot inflate under our canonical replay.",
                ],
                source_artifacts=[
                    "experiments/results/public_pr91_intake_20260504_worker/pr91_transfer_decisions.json",
                    "experiments/results/public_pr91_intake_20260504_worker/pr91_archive_anatomy.json",
                    "experiments/results/lightning_batch/exact_eval_public_pr91_hpac_hybrid_t4_g4dn2x_20260504T0509Z",
                ],
                recommendation="Highest EV, but only as a local parity/replay repair lane. No dispatch of derivatives until the decode contract is fixed.",
            )
        )

    stbm_profile = by_family.get("pr85_stbm1br")
    if stbm_profile and pr85_bytes is not None:
        bytes_saved = pr85_bytes - int(stbm_profile["archive_bytes"])
        opportunities.append(
            _opportunity(
                rank_seed=20,
                opportunity_id="exact_eval_pr85_stbm1br_lossless_mask_recode",
                surface="PR90 STBM1BR lossless mask recode lowered onto PR85 bundle",
                expected_bytes_saved_vs_pr85=bytes_saved,
                expected_component_delta={
                    "claim_basis": "local builder reports decoded render-order parity and diff_pixels=0",
                    "component_expectation": "zero if runtime parity is real; exact CUDA required",
                },
                exact_eval_delta=exact_stbm,
                evidence_status=(
                    "exact_cuda_available" if exact_stbm is not None else "local_candidate_ready_unscored"
                ),
                dispatch_readiness=(
                    "review_exact_result" if exact_stbm is not None else "claim_then_exact_eval_candidate"
                ),
                blocked=False,
                failure_modes=[
                    "mask metadata/order parity may pass locally while scorer-visible raw output differs",
                    "runtime code changes make this a runtime-custody comparison unless manifest hashes are preserved",
                    "mask-only rate win cannot promote if component gates move",
                ],
                gates=[
                    "review stbm1br_preflight.json and manifest SHA against exact archive",
                    "dispatch claim for standalone STBM exact eval if no current exact result exists",
                    "exact CUDA auth eval on STBM archive with runtime manifest hash",
                    "adjudicate component gates against PR85",
                ],
                adversarial_review=[
                    "This is the cleanest byte-only PR85 candidate because it targets the largest segment and claims lossless decoded-mask parity.",
                    "It does not beat external PR91 by itself; it can only become the exact local floor unless paired with independent component/byte wins.",
                    "If exact eval regresses, classify as runtime/parity bug rather than mask-codec family failure.",
                ],
                source_artifacts=[
                    stbm_profile["path"],
                    "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/candidate_summary.json",
                    "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/manifest.json",
                ],
                recommendation="Promotable next only as standalone exact CUDA eval with lane claim; do not stack until standalone result is positive.",
            )
        )

    if pr90_mask is not None and pr85_mask is not None:
        bytes_saved = pr85_mask - pr90_mask
        opportunities.append(
            _opportunity(
                rank_seed=30,
                opportunity_id="pr90_topband_geometry_mask_prior_for_pr85",
                surface="PR90 topband/road/residual semantic geometry as a PR85 QMA9 prior",
                expected_bytes_saved_vs_pr85=bytes_saved,
                expected_component_delta={
                    "claim_basis": "PR90 public report is worse than PR85; use anatomy as a prior, not a score source",
                    "external_pr90_report": pr90_report,
                },
                evidence_status="local_static_anatomy_plus_external_pr_text",
                dispatch_readiness="needs_local_builder",
                blocked=True,
                failure_modes=[
                    "PR90 masks, renderer, pose, and QRGB controls are co-trained and not drop-in PR85 streams",
                    "hard-coded fixed offsets are brittle unless rewritten into a typed PR85 bundle contract",
                    "generic QMA9 alternate/run screens were already byte-negative in existing PR85 planner artifacts",
                ],
                gates=[
                    "derive PR85 decoded-token geometry policy from PR90, not PR90 byte transplant",
                    "full PR85 decoded-token parity or explicit charged residual semantics",
                    "runtime support with fail-closed magic",
                    "local archive byte win before any exact eval claim",
                ],
                adversarial_review=[
                    "Do not promote PR90 as a score candidate: reported score is external and worse than PR85 exact.",
                    "The useful signal is its 152431-byte mask body and decomposed topology, but only if it can be rebuilt against PR85 tokens.",
                    "A sampled body-budget or prefix screen cannot unlock dispatch without full-stream decode and byte closure.",
                ],
                source_artifacts=[
                    "experiments/results/public_pr90_intake_20260504_worker/payload_probe.json",
                    "experiments/results/pr90_qma9_mask_prior_transfer_20260504_worker/analysis.json",
                    "experiments/results/pr90_qma9_mask_prior_transfer_20260504_worker/ranked_candidate_policy.json",
                ],
                recommendation="Implement only a PR85-token geometry profiler/builder; do not transplant PR90 runtime wholesale.",
            )
        )

    if stack_summary and pr85_bytes is not None:
        candidate = stack_summary.get("candidate_archive", {})
        stack_bytes = _as_int(candidate.get("archive_bytes")) if isinstance(candidate, dict) else None
        bytes_saved = pr85_bytes - stack_bytes if stack_bytes is not None else None
        qrgb_exact_bad = exact_qrgb is not None and _as_float(exact_qrgb.get("score_delta_vs_pr85")) is not None and float(exact_qrgb["score_delta_vs_pr85"]) > 0
        opportunities.append(
            _opportunity(
                rank_seed=40,
                opportunity_id="pr85_stbm1br_plus_qrgb_randmulti_pair_0192_stack",
                surface="STBM1BR mask recode plus QRGB randmulti pair 0192",
                expected_bytes_saved_vs_pr85=bytes_saved,
                expected_component_delta={
                    "claim_basis": "local stack builder only; standalone positives required",
                    "qrgb_standalone_exact_delta": exact_qrgb,
                    "stack_summary_readiness": stack_summary.get("fixed_runtime_preflight", {}),
                },
                exact_eval_delta=None,
                evidence_status="local_stack_candidate_blocked_by_standalone_positive_gates",
                dispatch_readiness="blocked",
                blocked=True,
                refuted=qrgb_exact_bad,
                failure_modes=[
                    "QRGB standalone candidates have exact CUDA negative or neutral evidence",
                    "STBM byte win and QRGB component movement may antagonize rather than compose",
                    "fixed-runtime bridge can confound archive-vs-runtime comparisons",
                ],
                gates=[
                    "STBM standalone exact CUDA positive for exact source SHA",
                    "QRGB standalone exact CUDA positive for exact source SHA",
                    "fresh dispatch claim for stacked exact eval",
                    "stacked exact CUDA auth eval and component adjudication",
                ],
                adversarial_review=[
                    "The local byte accounting is attractive, but the gate contract correctly refuses stacked dispatch without standalone positives.",
                    "If QRGB remains exact-negative, use these archives as gradient-sign negatives, not as score candidates.",
                    "A later HPM1 base would need separate PR91 QRGB proof; PR85 QRGB deltas do not transfer automatically.",
                ],
                source_artifacts=[
                    "experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/candidate_summary.json",
                    "experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/pr85_stbm1br_plus_qrgb_f2_randglobal_pair_0192/manifest.json",
                ],
                recommendation="Hold as a blocked reconstruction candidate. Do not dispatch while QRGB standalone exact evidence is negative.",
            )
        )

    if exact_qrgb is not None:
        archive_delta = _as_int(exact_qrgb.get("archive_byte_delta_vs_pr85"))
        opportunities.append(
            _opportunity(
                rank_seed=80,
                opportunity_id="pr85_qrgb_pair_atoms_negative_guardrail",
                surface="PR85 QRGB pair-local bias/region/randmulti atoms",
                expected_bytes_saved_vs_pr85=(-archive_delta if archive_delta is not None and archive_delta < 0 else None),
                expected_component_delta={
                    "claim_basis": "exact CUDA T4 artifacts for tested QRGB atoms",
                    "best_exact_delta": exact_qrgb,
                },
                exact_eval_delta=exact_qrgb,
                evidence_status="exact_cuda_negative_or_non_improving",
                dispatch_readiness="do_not_dispatch_measured_configs",
                blocked=True,
                refuted=True,
                failure_modes=[
                    "pair-local edits moved PoseNet/SegNet enough to erase rate or add bytes",
                    "PR90 QRGB sign transfer is not a reliable component oracle for PR85",
                ],
                gates=["new action family outside measured configs", "exact CUDA improvement before stacking"],
                adversarial_review=[
                    "These exact negatives are valuable training signal for future atom ranking.",
                    "Do not average them away: each changed pair/stream should become a guardrail row.",
                ],
                source_artifacts=[exact_qrgb["path"], "experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_codex"],
                recommendation="Retire measured QRGB configs narrowly; keep the action compiler as a negative-signal profiler.",
            )
        )

    if pr91_qrgb_plan:
        candidates = pr91_qrgb_plan.get("candidates", [])
        count = len(candidates) if isinstance(candidates, list) else None
        opportunities.append(
            _opportunity(
                rank_seed=50,
                opportunity_id="pr91_hpm1_qrgb_atoms_after_hpm1_replay",
                surface="PR85 QRGB action specs ported onto PR91 HPM1 base sidechannels",
                expected_bytes_saved_vs_pr85=None,
                expected_component_delta={
                    "claim_basis": "empirical local archives only; PR91 base is invalid under local exact replay",
                    "candidate_archive_count": count,
                },
                evidence_status="empirical_local_build_blocked_by_pr91_base_decode",
                dispatch_readiness="blocked",
                blocked=True,
                failure_modes=[
                    "base PR91 HPM1 cannot currently inflate under canonical replay",
                    "PR85 QRGB exact negatives may transfer as negatives to PR91",
                    "small sidechannel byte additions require component wins exceeding rate cost",
                ],
                gates=[
                    "PR91 HPM1 base exact replay fixed",
                    "PR91-specific component response or exact eval for each atom",
                    "lane claim before exact eval",
                ],
                adversarial_review=[
                    "This is only useful after PR91 is local-exact; otherwise it multiplies an invalid base.",
                    "Because candidates add 7-8 bytes, they need component benefit rather than rate savings.",
                ],
                source_artifacts=["experiments/results/pr91_qrgb_pair_atom_candidates_20260504_codex/planning.json"],
                recommendation="Keep built archives as post-HPM1 repair probes, not dispatch candidates today.",
            )
        )

    if pr90_model is not None and pr85_model is not None:
        bytes_saved = pr85_model - pr90_model
        opportunities.append(
            _opportunity(
                rank_seed=60,
                opportunity_id="qfq4_model_payload_serializer_probe",
                surface="PR90 QFQ4 grouped FP4 model packing versus PR85 QH0 model payload",
                expected_bytes_saved_vs_pr85=bytes_saved if bytes_saved > 0 else None,
                expected_component_delta={
                    "claim_basis": "static byte comparison only; model tensors are not known equivalent",
                    "pr85_model_bytes": pr85_model,
                    "pr90_model_body_bytes": pr90_model,
                },
                evidence_status="static_anatomy_low_ceiling",
                dispatch_readiness="profiler_only",
                blocked=True,
                failure_modes=[
                    "PR90 model is not proven tensor-equivalent to PR85 QH0",
                    "prior PR85 QH0 serializer screens found no real byte win",
                    "model changes can move both components catastrophically",
                ],
                gates=[
                    "tensor-equivalent PR85 model serialization proof",
                    "local byte-positive deterministic archive",
                    "runtime output parity",
                    "exact CUDA auth eval",
                ],
                adversarial_review=[
                    "The byte ceiling is hundreds of bytes, not a frontier jump by itself.",
                    "Only revisit after HPM1/STBM mask surfaces are resolved or if a true tensor-equivalent QH0 recode appears.",
                ],
                source_artifacts=["experiments/results/public_pr90_intake_20260504_worker/payload_probe.json"],
                recommendation="Low priority: keep as a serializer probe, not a remote eval lane.",
            )
        )

    if exact_randmulti is not None:
        opportunities.append(
            _opportunity(
                rank_seed=90,
                opportunity_id="randmulti_deletion_waterfill_negative_guardrail",
                surface="PR85 randmulti deletion/group waterfill routes",
                expected_bytes_saved_vs_pr85=None,
                expected_component_delta={
                    "claim_basis": "exact CUDA negative artifacts",
                    "best_exact_delta": exact_randmulti,
                },
                evidence_status="exact_cuda_negative",
                dispatch_readiness="do_not_dispatch_measured_configs",
                blocked=True,
                refuted=True,
                failure_modes=["large byte savings collapse PoseNet/SegNet basin"],
                gates=["new protected atom outside measured deletion basin"],
                adversarial_review=[
                    "The rate term looked attractive but exact CUDA showed component cliffs.",
                    "Use as a protected-stream guard, not as evidence against all sidechannel atoms.",
                ],
                source_artifacts=[exact_randmulti["path"]],
                recommendation="Do not dispatch measured randmulti deletion/waterfill candidates.",
            )
        )

    if exact_line is not None:
        opportunities.append(
            _opportunity(
                rank_seed=95,
                opportunity_id="line_search_pose_stack_negative_guardrail",
                surface="older QZS3/QP1 line-search pose stack",
                expected_bytes_saved_vs_pr85=None,
                expected_component_delta={
                    "claim_basis": "exact CUDA line-search artifact exists but is worse than PR85",
                    "best_exact_delta": exact_line,
                },
                exact_eval_delta=exact_line,
                evidence_status="exact_cuda_negative_vs_pr85_anchor",
                dispatch_readiness="do_not_dispatch_as_frontier",
                blocked=True,
                refuted=True,
                failure_modes=["older pose stack sits outside current PR85 scorer basin"],
                gates=["only reuse as pair/hard-case profile signal"],
                adversarial_review=[
                    "Line-search results are evidence for sensitivity directions, not current frontier archives.",
                    "Do not compare H100 diagnostics as promotion-grade against T4 unless exact workflow permits it.",
                ],
                source_artifacts=[exact_line["path"]],
                recommendation="Use only as profile feedback for new pair atoms.",
            )
        )

    return sorted(
        opportunities,
        key=lambda row: (
            row["refuted"],
            row["blocked"] and row["rank_seed"] > 30,
            row["rank_seed"],
            -(row["expected_bytes_saved_vs_pr85"] or 0),
            row["opportunity_id"],
        ),
    )


def _archive_summary_markdown(profiles: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Archive Byte Accounting",
        "",
        "| Artifact | Family | Bytes | Member | Member bytes | ZIP overhead | Strict ZIP |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for row in sorted(profiles, key=lambda p: (p["family"], p["artifact_id"], p["path"])):
        lines.append(
            f"| `{row['artifact_id']}` | {row['family']} | {row['archive_bytes']} | "
            f"`{row.get('primary_member')}` | {row.get('primary_member_bytes')} | "
            f"{row['zip_overhead_bytes']} | {str(row['strict_zip_ok']).lower()} |"
        )
    return lines


def _markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Frontier Stack Reconstruction Plan",
        "",
        "- planning_only: true",
        "- score_claim: false",
        "- dispatch_performed: false",
        "",
        "## Exact Anchor",
    ]
    anchor = plan.get("baseline_pr85_exact_anchor")
    if anchor:
        lines.append(
            f"- PR85 exact anchor: score {anchor.get('score')} at {anchor.get('archive_bytes')} bytes "
            f"from `{anchor.get('path')}`."
        )
    else:
        lines.append("- No PR85 exact CUDA anchor found in discovered inputs.")
    lines.extend(["", "## Ranked Opportunities", ""])
    lines.append("| Rank | Opportunity | Bytes saved vs PR85 | Rate delta if neutral | Evidence | Gate |")
    lines.append("|---:|---|---:|---:|---|---|")
    for index, row in enumerate(plan["ranked_opportunities"], start=1):
        first_gate = row["required_gates"][0] if row["required_gates"] else ""
        lines.append(
            f"| {index} | `{row['opportunity_id']}` | {row['expected_bytes_saved_vs_pr85']} | "
            f"{row['expected_rate_score_delta_if_components_unchanged']} | {row['evidence_status']} | {first_gate} |"
        )
    lines.extend(["", *_archive_summary_markdown(plan["archive_profiles"]), "", "## Failure Mode Review", ""])
    for row in plan["ranked_opportunities"]:
        lines.append(f"- `{row['opportunity_id']}`: {row['recommendation']}")
    return "\n".join(lines) + "\n"


def build_plan(
    *,
    repo_root: Path = REPO_ROOT,
    archive_globs: list[str] | None = None,
    exact_eval_globs: list[str] | None = None,
    json_globs: list[str] | None = None,
) -> dict[str, Any]:
    inputs = discover_inputs(
        repo_root,
        archive_globs=archive_globs,
        exact_eval_globs=exact_eval_globs,
        json_globs=json_globs,
    )
    json_rows = [(path, _read_json(path)) for path in inputs["json_inputs"]]
    archive_profiles = [profile_archive(path, root=repo_root) for path in inputs["archives"]]
    archive_profiles = sorted(
        archive_profiles,
        key=lambda row: (row["family"], row["artifact_id"], row["path"]),
    )
    for profile in archive_profiles:
        profile["runtime_contract"] = _runtime_contract(profile, json_rows)
        profile["rate_score_contribution"] = _rate_score(profile["archive_bytes"])
    exact_evals = [_load_exact_eval(path, root=repo_root) for path in inputs["exact_evals"]]
    exact_matches = _match_exact_evals(archive_profiles, exact_evals)
    baseline = _best_pr85_anchor(exact_evals)
    plan = {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "deterministic": True,
        "score_formula": {
            "original_video_bytes": ORIGINAL_VIDEO_BYTES,
            "rate_points_per_byte": RATE_POINTS_PER_BYTE,
            "formula": "100*seg_dist + sqrt(10*pose_dist) + 25*archive_bytes/original_video_bytes",
        },
        "input_inventory": {
            "archive_count": len(inputs["archives"]),
            "exact_eval_count": len(inputs["exact_evals"]),
            "json_input_count": len(inputs["json_inputs"]),
            "archives": [_rel(path, root=repo_root) for path in inputs["archives"]],
            "exact_evals": [_rel(path, root=repo_root) for path in inputs["exact_evals"]],
            "json_inputs": [_rel(path, root=repo_root) for path in inputs["json_inputs"]],
        },
        "baseline_pr85_exact_anchor": baseline,
        "archive_profiles": archive_profiles,
        "exact_eval_matches_by_artifact": exact_matches,
        "external_public_reports": {
            "pr90": _reported_pr90_from_pull(json_rows),
            "pr91": _external_pr91_report(json_rows),
        },
        "ranked_opportunities": _build_opportunities(
            profiles=archive_profiles,
            exact_evals=exact_evals,
            json_rows=json_rows,
            baseline=baseline,
        ),
    }
    plan["stable_plan_digest_sha256"] = _stable_digest(plan)
    return plan


def write_outputs(plan: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "frontier_stack_reconstruction_plan.json"
    md_path = out_dir / "frontier_stack_reconstruction_plan.md"
    json_path.write_text(_json_text(plan), encoding="utf-8")
    md_path.write_text(_markdown(plan), encoding="utf-8")
    return {"json": _rel(json_path), "markdown": _rel(md_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--archive-glob", action="append", default=None)
    parser.add_argument("--exact-eval-glob", action="append", default=None)
    parser.add_argument("--json-glob", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(
        repo_root=args.repo_root,
        archive_globs=args.archive_glob,
        exact_eval_globs=args.exact_eval_glob,
        json_globs=args.json_glob,
    )
    outputs = write_outputs(plan, args.out_dir)
    print(
        _json_text(
            {
                "outputs": outputs,
                "stable_plan_digest_sha256": plan["stable_plan_digest_sha256"],
                "top_opportunities": [
                    row["opportunity_id"] for row in plan["ranked_opportunities"][:5]
                ],
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
