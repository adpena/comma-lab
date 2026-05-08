#!/usr/bin/env python3
"""Build a fail-closed PR107 CPU-axis lossy-coarsening candidate matrix.

This is a read-only planner/manifest tool. It consumes existing PR107 CPU
anchors, lossy-coarsening build manifests, and GHA CPU adjudication artifacts,
then writes one durable matrix for the next validation decision. It does not
dispatch CPU, CUDA, or remote GPU work.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "pr107_cpu_lossy_candidate_matrix.v1"
TOOL_NAME = "tools/build_pr107_cpu_lossy_candidate_matrix.py"
ORIGINAL_UNCOMPRESSED_BYTES = 37_545_489

DEFAULT_OUTPUT = Path(".omx/research/pr107_cpu_lossy_coarsening_candidate_matrix_20260508_codex.json")
DEFAULT_BASELINE_CPU_EVAL = Path(
    "experiments/results/pr107_apogee_cpu_auth_eval_gha_20260508T124452Z/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_LOCAL_BASELINE_REPORT = Path("experiments/results/pr107_cpu_eval_20260508/logs/report.txt")
DEFAULT_CATHEDRAL_EVIDENCE = Path("reports/cathedral_autopilot_evidence.jsonl")
DEFAULT_CATHEDRAL_CATALOG = Path("reports/cathedral_autopilot_catalog_updated_20260508.json")
DEFAULT_RUNTIME_DIR = Path(
    "experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/"
    "source/submissions/apogee"
)
DEFAULT_INFLATE_SH = DEFAULT_RUNTIME_DIR / "inflate.sh"

REPORT_PATTERNS = {
    "n_samples": re.compile(r"Evaluation results over\s+(\d+)\s+samples"),
    "avg_posenet_dist": re.compile(r"Average PoseNet Distortion:\s*([0-9.eE+-]+)"),
    "avg_segnet_dist": re.compile(r"Average SegNet Distortion:\s*([0-9.eE+-]+)"),
    "archive_size_bytes": re.compile(r"Submission file size:\s*([0-9,]+)\s+bytes"),
    "original_uncompressed_bytes": re.compile(r"Original uncompressed size:\s*([0-9,]+)\s+bytes"),
    "compression_rate": re.compile(r"Compression Rate:\s*([0-9.eE+-]+)"),
    "canonical_score_display": re.compile(r"Final score:.*=\s*([0-9.eE+-]+)"),
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    variant: str
    build_manifest: Path
    cpu_eval: Path | None = None


DEFAULT_CANDIDATES = (
    CandidateSpec(
        "pr107_lossy_b025",
        "uniform_per_tensor_k_coarsening",
        Path("experiments/results/pr107_apogee_lossy_coarsening_b025_20260508T130700Z/build_manifest.json"),
        Path(
            "experiments/results/pr107_apogee_lossy_coarsening_b025_20260508T130700Z/"
            "contest_auth_eval.adjudicated.json"
        ),
    ),
    CandidateSpec(
        "pr107_lossy_b035",
        "uniform_per_tensor_k_coarsening",
        Path("experiments/results/pr107_apogee_lossy_coarsening_b035_20260508T130700Z/build_manifest.json"),
        Path(
            "experiments/results/pr107_apogee_lossy_coarsening_b035_20260508T130700Z/"
            "contest_auth_eval.adjudicated.json"
        ),
    ),
    CandidateSpec(
        "pr107_lossy_b050",
        "uniform_per_tensor_k_coarsening",
        Path("experiments/results/pr107_apogee_lossy_coarsening_b050_20260508T130700Z/build_manifest.json"),
        Path(
            "experiments/results/pr107_apogee_lossy_coarsening_b050_20260508T130700Z/"
            "contest_auth_eval.adjudicated.json"
        ),
    ),
    CandidateSpec(
        "pr107_stack_b050",
        "uniform_per_tensor_k_coarsening_plus_brotli_optuna",
        Path("experiments/results/pr107_apogee_stack_lossy_brotli_20260508T131530Z/b050/build_manifest.json"),
    ),
    CandidateSpec(
        "pr107_stack_b060",
        "uniform_per_tensor_k_coarsening_plus_brotli_optuna",
        Path("experiments/results/pr107_apogee_stack_lossy_brotli_20260508T131530Z/b060/build_manifest.json"),
    ),
    CandidateSpec(
        "pr107_stack_b070",
        "uniform_per_tensor_k_coarsening_plus_brotli_optuna",
        Path("experiments/results/pr107_apogee_stack_lossy_brotli_20260508T131530Z/b070/build_manifest.json"),
        Path(
            "experiments/results/pr107_apogee_stack_b070_cpu_eval_gha_20260508/"
            "contest_auth_eval.adjudicated.json"
        ),
    ),
    CandidateSpec(
        "pr107_stack_b080",
        "uniform_per_tensor_k_coarsening_plus_brotli_optuna",
        Path("experiments/results/pr107_apogee_stack_lossy_brotli_20260508T131530Z/b080/build_manifest.json"),
        Path(
            "experiments/results/pr107_apogee_stack_b080_cpu_eval_gha_20260508/"
            "contest_auth_eval.adjudicated.json"
        ),
    ),
    CandidateSpec(
        "pr107_stack_b100",
        "uniform_per_tensor_k_coarsening_plus_brotli_optuna",
        Path("experiments/results/pr107_apogee_stack_aggressive_20260508/b100/build_manifest.json"),
        Path(
            "experiments/results/pr107_apogee_stack_b100_cpu_eval_gha_20260508/"
            "contest_auth_eval.adjudicated.json"
        ),
    ),
    CandidateSpec(
        "pr107_stack_b120",
        "uniform_per_tensor_k_coarsening_plus_brotli_optuna",
        Path("experiments/results/pr107_apogee_stack_aggressive_20260508/b120/build_manifest.json"),
    ),
)


def _repo_path(repo_root: Path, path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _repo_rel(repo_root: Path, path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(p)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _as_float(value: Any) -> float | None:
    if value in (None, "", False):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value in (None, "", False):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_report_text(text: str) -> dict[str, Any]:
    """Parse upstream evaluate.py report text and recompute the precise score."""

    parsed: dict[str, Any] = {}
    for key, pattern in REPORT_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        raw = match.group(1).replace(",", "")
        parsed[key] = int(raw) if key in {"n_samples", "archive_size_bytes", "original_uncompressed_bytes"} else float(raw)

    required = {"avg_segnet_dist", "avg_posenet_dist", "compression_rate"}
    if required.issubset(parsed):
        parsed["score_recomputed_from_components"] = contest_score(
            avg_segnet_dist=parsed["avg_segnet_dist"],
            avg_posenet_dist=parsed["avg_posenet_dist"],
            compression_rate=parsed["compression_rate"],
        )
    return parsed


def contest_score(*, avg_segnet_dist: float, avg_posenet_dist: float, compression_rate: float) -> float:
    return 100.0 * avg_segnet_dist + math.sqrt(10.0 * avg_posenet_dist) + 25.0 * compression_rate


def rate_term(archive_size_bytes: int) -> float:
    return 25.0 * archive_size_bytes / ORIGINAL_UNCOMPRESSED_BYTES


def _eval_record(repo_root: Path, eval_path: Path) -> dict[str, Any]:
    payload = _load_json(eval_path)
    report_values = parse_report_text(str(payload.get("report_text", "")))

    archive_size = (
        _as_int(payload.get("archive_size_bytes"))
        or _as_int(report_values.get("archive_size_bytes"))
    )
    compression_rate = _as_float(payload.get("compression_rate"))
    if compression_rate is None:
        compression_rate = _as_float(report_values.get("compression_rate"))
    if compression_rate is None and archive_size is not None:
        compression_rate = archive_size / ORIGINAL_UNCOMPRESSED_BYTES

    avg_seg = _as_float(payload.get("avg_segnet_dist"))
    if avg_seg is None:
        avg_seg = _as_float(report_values.get("avg_segnet_dist"))
    avg_pose = _as_float(payload.get("avg_posenet_dist"))
    if avg_pose is None:
        avg_pose = _as_float(report_values.get("avg_posenet_dist"))

    score_recomputed = _as_float(payload.get("score_recomputed_from_components"))
    if score_recomputed is None or (score_recomputed == 0.0 and avg_seg is not None and avg_pose is not None):
        if avg_seg is not None and avg_pose is not None and compression_rate is not None:
            score_recomputed = contest_score(
                avg_segnet_dist=avg_seg,
                avg_posenet_dist=avg_pose,
                compression_rate=compression_rate,
            )
        else:
            score_recomputed = _as_float(report_values.get("score_recomputed_from_components"))

    return {
        "path": _repo_rel(repo_root, eval_path),
        "exists": eval_path.is_file(),
        "archive_relpath": payload.get("archive_relpath"),
        "archive_size_bytes": archive_size,
        "archive_sha256": payload.get("archive_sha256"),
        "archive_member": payload.get("archive_member"),
        "archive_member_sha256": payload.get("archive_member_sha256"),
        "canonical_score_display": _as_float(payload.get("canonical_score"))
        or _as_float(report_values.get("canonical_score_display")),
        "score_recomputed_from_components": score_recomputed,
        "avg_segnet_dist": avg_seg,
        "avg_posenet_dist": avg_pose,
        "compression_rate": compression_rate,
        "n_samples": _as_int(payload.get("n_samples")) or _as_int(report_values.get("n_samples")),
        "device": payload.get("device"),
        "hardware": payload.get("hardware"),
        "runner_image": payload.get("runner_image"),
        "runner_arch": payload.get("runner_arch"),
        "evaluate_py_sha256": payload.get("evaluate_py_sha256"),
        "workflow_run_id": payload.get("workflow_run_id"),
        "workflow_run_url": payload.get("workflow_run_url"),
        "evidence_grade": payload.get("evidence_grade"),
        "lane_tag": payload.get("lane_tag"),
        "is_contest_compliant": payload.get("is_contest_compliant"),
        "report_text_fallback_used": bool(report_values)
        and (
            payload.get("avg_segnet_dist") in (None, "")
            or payload.get("avg_posenet_dist") in (None, "")
            or payload.get("score_recomputed_from_components") in (None, 0, 0.0)
        ),
    }


def _cpu_eval_blockers(
    *,
    cpu_eval: dict[str, Any] | None,
    build: dict[str, Any],
) -> list[str]:
    if not cpu_eval:
        return ["exact_cpu_auth_eval_missing"]
    blockers: list[str] = []
    build_sha = build.get("archive_sha256")
    build_bytes = _as_int(build.get("archive_size_bytes"))
    eval_sha = cpu_eval.get("archive_sha256")
    eval_bytes = _as_int(cpu_eval.get("archive_size_bytes"))
    if not eval_sha:
        blockers.append("exact_cpu_archive_sha256_missing")
    elif build_sha and eval_sha != build_sha:
        blockers.append("exact_cpu_archive_sha256_mismatch_with_build")
    if eval_bytes is None:
        blockers.append("exact_cpu_archive_bytes_missing")
    elif build_bytes is not None and eval_bytes != build_bytes:
        blockers.append("exact_cpu_archive_bytes_mismatch_with_build")
    if str(cpu_eval.get("device") or "").lower() != "cpu":
        blockers.append("exact_cpu_device_mismatch")
    if _as_int(cpu_eval.get("n_samples")) != 600:
        blockers.append("exact_cpu_not_full_sample_600")
    hardware = str(cpu_eval.get("hardware") or "").lower()
    runner_arch = str(cpu_eval.get("runner_arch") or "").lower()
    grade = " ".join(
        str(cpu_eval.get(key) or "").lower()
        for key in ("evidence_grade", "lane_tag")
    )
    if "contest-cpu" not in grade and "contest_cpu" not in grade:
        blockers.append("contest_cpu_evidence_grade_missing")
    if not (
        "github-actions" in hardware
        or "ubuntu" in hardware
        or runner_arch in {"x86_64", "amd64"}
        or "x86_64" in hardware
        or "amd64" in hardware
    ):
        blockers.append("contest_cpu_linux_x86_hardware_missing")
    if cpu_eval.get("is_contest_compliant") is not True:
        blockers.append("contest_cpu_compliance_true_required")
    score = _as_float(cpu_eval.get("score_recomputed_from_components"))
    if score is None or not math.isfinite(score):
        blockers.append("exact_cpu_recomputed_score_missing")
    for key in ("avg_segnet_dist", "avg_posenet_dist", "compression_rate"):
        value = _as_float(cpu_eval.get(key))
        if value is None or not math.isfinite(value):
            blockers.append(f"exact_cpu_{key}_missing")
    return blockers


def _local_report_record(repo_root: Path, report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        return {"path": _repo_rel(repo_root, report_path), "exists": False}
    parsed = parse_report_text(report_path.read_text(encoding="utf-8"))
    return {
        "path": _repo_rel(repo_root, report_path),
        "exists": True,
        "evidence_grade": "[local-macOS-CPU-diagnostic]",
        "score_claim": False,
        **parsed,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_probe(repo_root: Path, archive_relpath: str | None, expected_sha: str | None, expected_bytes: int | None) -> dict[str, Any]:
    if not archive_relpath:
        return {"status": "missing_archive_path", "blockers": ["archive_path_missing"]}
    archive_path = _repo_path(repo_root, archive_relpath)
    if not archive_path.is_file():
        return {
            "path": archive_relpath,
            "status": "missing_archive_file",
            "blockers": ["archive_file_missing"],
        }
    actual_bytes = archive_path.stat().st_size
    actual_sha = _sha256_file(archive_path)
    blockers: list[str] = []
    if expected_bytes is not None and actual_bytes != expected_bytes:
        blockers.append("archive_size_mismatch")
    if expected_sha and actual_sha != expected_sha:
        blockers.append("archive_sha256_mismatch")
    member_records: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                member_records.append(
                    {
                        "name": info.filename,
                        "bytes": info.file_size,
                        "compress_size": info.compress_size,
                        "sha256": hashlib.sha256(zf.read(info)).hexdigest(),
                    }
                )
    except zipfile.BadZipFile:
        blockers.append("archive_not_readable_zip")
    return {
        "path": _repo_rel(repo_root, archive_path),
        "status": "ok" if not blockers else "blocked",
        "actual_bytes": actual_bytes,
        "actual_sha256": actual_sha,
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha,
        "members": member_records,
        "blockers": blockers,
    }


def _build_manifest_record(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    payload = _load_json(manifest_path)
    archive_relpath = (
        payload.get("build_archive_relpath")
        or payload.get("archive_relpath")
        or payload.get("archive_path")
    )
    archive_sha = payload.get("build_archive_sha256") or payload.get("archive_sha256")
    archive_bytes = _as_int(payload.get("build_archive_size_bytes")) or _as_int(payload.get("archive_size_bytes"))
    return {
        "path": _repo_rel(repo_root, manifest_path),
        "schema_version": payload.get("schema_version"),
        "tool": payload.get("tool"),
        "evidence_grade": payload.get("evidence_grade"),
        "evidence_semantics": payload.get("evidence_semantics"),
        "score_claim": payload.get("score_claim", False),
        "promotion_eligible": payload.get("promotion_eligible", False),
        "rank_or_kill_eligible": payload.get("rank_or_kill_eligible", False),
        "ready_for_exact_eval_dispatch": payload.get("ready_for_exact_eval_dispatch", False),
        "dispatch_blockers": payload.get("dispatch_blockers", []),
        "dispatch_attempted": payload.get("dispatch_attempted", False),
        "budget": _as_float(payload.get("build_target_budget")),
        "rel_err": _as_float(payload.get("build_rel_err")),
        "archive_relpath": archive_relpath,
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "member_name": payload.get("build_member_name"),
        "member_sha256": payload.get("build_member_sha256"),
        "member_size_bytes": _as_int(payload.get("build_member_size_bytes")),
        "delta_zip_vs_baseline": _as_int(payload.get("build_delta_zip_vs_baseline")),
        "score_affecting_payload_changed": payload.get("score_affecting_payload_changed"),
        "charged_bits_changed": payload.get("charged_bits_changed"),
        "target_modes": payload.get("target_modes", []),
        "deployment_target": payload.get("deployment_target"),
        "archive_probe": _archive_probe(repo_root, str(archive_relpath) if archive_relpath else None, archive_sha, archive_bytes),
    }


def _candidate_commands(
    *,
    candidate_id: str,
    manifest_path: str,
    archive_relpath: str | None,
    archive_sha: str | None,
    archive_bytes: int | None,
    baseline_score: float | None,
    baseline_archive_bytes: int | None,
    pr_number: Any,
) -> dict[str, str]:
    lane_id = f"pr107_lossy_cpu_axis_{candidate_id}"
    job_name = f"{candidate_id}-cuda-exact"
    archive = archive_relpath or "<candidate_archive.zip>"
    sha = archive_sha or "<archive_sha256>"
    bytes_text = str(archive_bytes) if archive_bytes is not None else "<archive_size_bytes>"
    baseline_score_text = f"{baseline_score:.12f}" if baseline_score is not None else "<baseline_score>"
    baseline_bytes_text = str(baseline_archive_bytes) if baseline_archive_bytes is not None else "<baseline_archive_bytes>"
    pr_number_text = str(pr_number) if pr_number else "1"
    return {
        "candidate_preflight": (
            ".venv/bin/python experiments/preflight_candidate_manifest_dispatch_readiness.py "
            f"--manifest {manifest_path} "
            "--claims-path .omx/state/active_lane_dispatch_claims.md "
            "--now-utc <utc> --fail-if-not-ready"
        ),
        "exact_cpu_gha": (
            ".venv/bin/python tools/dispatch_cpu_eval_via_github_actions.py "
            f"--archive-path {archive} --archive-sha {sha} --submission-name apogee "
            f"--output-dir experiments/results/{candidate_id}_cpu_eval_gha "
            f"--pr-number {pr_number_text}"
        ),
        "claim_before_cuda": (
            ".venv/bin/python tools/claim_lane_dispatch.py claim "
            f"--lane-id {lane_id} --platform lightning --instance-job-id {job_name} "
            "--agent codex --predicted-eta-utc <utc> --status eval "
            "--notes \"PR107 lossy CPU-axis matrix CUDA validation\""
        ),
        "exact_cuda_lightning_after_claim": (
            ".venv/bin/python scripts/launch_lightning_batch_job.py exact-eval "
            f"--job-name {job_name} --archive {archive} "
            "--repo-dir <remote_repo_dir> --upstream-dir <remote_repo_dir>/upstream "
            f"--inflate-sh <remote_repo_dir>/{DEFAULT_INFLATE_SH.as_posix()} "
            f"--dispatch-lane-id {lane_id} --dispatch-claims-path .omx/state/active_lane_dispatch_claims.md "
            f"--expected-archive-sha256 {sha} --expected-archive-size-bytes {bytes_text} "
            "--adjudicate "
            f"--baseline-score {baseline_score_text} --baseline-archive-bytes {baseline_bytes_text} "
            "--component-reference-label pr107_cpu_anchor --max-sane-score 1.0"
        ),
    }


def _cathedral_summary(repo_root: Path, evidence_path: Path, catalog_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "evidence_jsonl_path": _repo_rel(repo_root, evidence_path),
        "catalog_path": _repo_rel(repo_root, catalog_path),
        "evidence_exists": evidence_path.is_file(),
        "catalog_exists": catalog_path.is_file(),
        "rows_read": 0,
        "lossy_rows_count": 0,
        "promotable_lossy_rows_count": 0,
        "exact_negative_lossy_rows_count": 0,
        "latest_lossy_rows": [],
    }
    if not evidence_path.is_file():
        return summary

    latest: list[dict[str, Any]] = []
    for line in evidence_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        summary["rows_read"] += 1
        technique = str(row.get("technique", ""))
        if "lossy_coarsening" not in technique:
            continue
        summary["lossy_rows_count"] += 1
        promotable = all(
            row.get(key) is True
            for key in ("score_claim", "promotion_eligible", "rank_or_kill_eligible", "ready_for_exact_eval_dispatch")
        )
        if promotable:
            summary["promotable_lossy_rows_count"] += 1
        evidence_grade = str(row.get("evidence_grade", ""))
        verdict = str(row.get("contest_dispatch_verdict", ""))
        if "A-negative" in evidence_grade or "negative" in verdict.lower() or "retired" in verdict.lower():
            summary["exact_negative_lossy_rows_count"] += 1
        latest.append(
            {
                "technique": technique,
                "timestamp": row.get("timestamp"),
                "evidence_grade": row.get("evidence_grade"),
                "empirical_archive_bytes": row.get("empirical_archive_bytes"),
                "empirical_score": row.get("empirical_score") or row.get("score_contest_cuda"),
                "contest_dispatch_verdict": row.get("contest_dispatch_verdict"),
                "source": row.get("source"),
            }
        )
    summary["latest_lossy_rows"] = latest[-5:]
    return summary


def _builder_surfaces(repo_root: Path) -> dict[str, Any]:
    paths = {
        "pr107_lossy_builder": Path("tools/pr107_lossy_coarsening_apogee.py"),
        "pr107_lossy_brotli_stack_builder": Path("tools/pr107_lossy_coarsening_brotli_optuna_stack.py"),
        "runtime_packet_builder_pattern": Path("tools/build_pr101_runtime_packet.py"),
        "candidate_preflight": Path("experiments/preflight_candidate_manifest_dispatch_readiness.py"),
        "cpu_eval_dispatcher": Path("tools/dispatch_cpu_eval_via_github_actions.py"),
        "gha_harvester": Path("tools/harvest_gha_runs.py"),
        "cuda_exact_eval_launcher": Path("scripts/launch_lightning_batch_job.py"),
        "lane_claim_helper": Path("tools/claim_lane_dispatch.py"),
        "pre_submission_compliance_gate": Path("scripts/pre_submission_compliance_check.py"),
        "pr107_apogee_runtime_dir": DEFAULT_RUNTIME_DIR,
        "pr107_apogee_inflate_sh": DEFAULT_INFLATE_SH,
    }
    return {
        key: {
            "path": str(path),
            "exists": _repo_path(repo_root, path).exists(),
        }
        for key, path in paths.items()
    }


def _candidate_status(
    *,
    cpu_score: float | None,
    baseline_score: float | None,
    cpu_eval_present: bool,
    cpu_eval_blockers: list[str],
) -> str:
    if not cpu_eval_present:
        return "exact_cpu_missing_fail_closed"
    if cpu_eval_blockers:
        return "exact_cpu_untrusted_fail_closed"
    if cpu_score is None or baseline_score is None:
        return "exact_cpu_unusable_fail_closed"
    if cpu_score < baseline_score:
        return "cpu_axis_improves_pr107_anchor_requires_cuda_confirmation"
    return "retired_on_pr107_cpu_axis_uniform_lossy_config"


def build_matrix(
    *,
    repo_root: Path,
    candidates: tuple[CandidateSpec, ...] = DEFAULT_CANDIDATES,
    baseline_cpu_eval: Path = DEFAULT_BASELINE_CPU_EVAL,
    local_baseline_report: Path = DEFAULT_LOCAL_BASELINE_REPORT,
    cathedral_evidence: Path = DEFAULT_CATHEDRAL_EVIDENCE,
    cathedral_catalog: Path = DEFAULT_CATHEDRAL_CATALOG,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    recorded = recorded_at_utc or dt.datetime.now(tz=dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    baseline_path = _repo_path(repo_root, baseline_cpu_eval)
    baseline = _eval_record(repo_root, baseline_path)
    baseline_score = _as_float(baseline.get("score_recomputed_from_components"))
    baseline_bytes = _as_int(baseline.get("archive_size_bytes"))
    fork_pr_number = None
    try:
        fork_pr_number = _load_json(baseline_path).get("fork_pr_number")
    except (OSError, json.JSONDecodeError):
        pass

    candidate_rows: list[dict[str, Any]] = []
    for spec in candidates:
        manifest_path = _repo_path(repo_root, spec.build_manifest)
        if not manifest_path.is_file():
            candidate_rows.append(
                {
                    "candidate_id": spec.candidate_id,
                    "variant": spec.variant,
                    "build_manifest": {
                        "path": _repo_rel(repo_root, manifest_path),
                        "exists": False,
                    },
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "fail_closed_blockers": ["build_manifest_missing"],
                    "status": "build_manifest_missing_fail_closed",
                }
            )
            continue

        build = _build_manifest_record(repo_root, manifest_path)
        cpu_eval_present = spec.cpu_eval is not None and _repo_path(repo_root, spec.cpu_eval).is_file()
        cpu_eval = None
        if cpu_eval_present and spec.cpu_eval is not None:
            cpu_eval = _eval_record(repo_root, _repo_path(repo_root, spec.cpu_eval))
        cpu_eval_blockers = _cpu_eval_blockers(cpu_eval=cpu_eval, build=build)
        cpu_eval_trusted = cpu_eval is not None and not cpu_eval_blockers
        cpu_score = (
            _as_float(cpu_eval.get("score_recomputed_from_components"))
            if cpu_eval_trusted and cpu_eval
            else None
        )
        cpu_score_delta = (
            cpu_score - baseline_score
            if cpu_score is not None and baseline_score is not None
            else None
        )
        archive_bytes = _as_int(build.get("archive_size_bytes"))
        delta_bytes = archive_bytes - baseline_bytes if archive_bytes is not None and baseline_bytes is not None else None
        rate_saving = -rate_term(delta_bytes) if delta_bytes is not None else None

        blockers: list[str] = []
        blockers.extend(build.get("archive_probe", {}).get("blockers", []))
        blockers.extend(cpu_eval_blockers)
        if not cpu_eval_present:
            pass
        elif not cpu_eval_trusted:
            blockers.append("exact_cpu_auth_eval_untrusted")
        elif cpu_score is None:
            blockers.append("exact_cpu_auth_eval_unusable")
        elif baseline_score is not None and cpu_score >= baseline_score:
            blockers.append("exact_cpu_score_not_lower_than_pr107_anchor")
        blockers.append("exact_cuda_auth_eval_missing")
        blockers.append("do_not_make_score_claim_without_paired_cpu_cuda_review")

        commands = _candidate_commands(
            candidate_id=spec.candidate_id,
            manifest_path=build["path"],
            archive_relpath=build.get("archive_relpath"),
            archive_sha=build.get("archive_sha256"),
            archive_bytes=archive_bytes,
            baseline_score=baseline_score,
            baseline_archive_bytes=baseline_bytes,
            pr_number=fork_pr_number,
        )

        candidate_rows.append(
            {
                "candidate_id": spec.candidate_id,
                "variant": spec.variant,
                "budget": build.get("budget"),
                "rel_err": build.get("rel_err"),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted_by_this_tool": False,
                "status": _candidate_status(
                    cpu_score=cpu_score,
                    baseline_score=baseline_score,
                    cpu_eval_present=cpu_eval_present,
                    cpu_eval_blockers=cpu_eval_blockers,
                ),
                "build_manifest": build,
                "cpu_eval": cpu_eval
                or {
                    "path": _repo_rel(repo_root, _repo_path(repo_root, spec.cpu_eval))
                    if spec.cpu_eval is not None
                    else None,
                    "exists": False,
                    "status": "missing",
                },
                "cpu_eval_trusted_for_score_delta": cpu_eval_trusted,
                "cpu_eval_validation_blockers": cpu_eval_blockers,
                "cuda_eval": {
                    "exists": False,
                    "status": "missing",
                    "evidence_tag": "[contest-CUDA missing]",
                },
                "delta_vs_pr107_cpu_anchor": {
                    "candidate_minus_baseline_bytes": delta_bytes,
                    "rate_term_saving_if_components_unchanged": rate_saving,
                    "candidate_minus_baseline_score_cpu": cpu_score_delta,
                    "seg_term_delta_cpu": (
                        100.0 * (cpu_eval["avg_segnet_dist"] - baseline["avg_segnet_dist"])
                        if cpu_eval
                        and cpu_eval.get("avg_segnet_dist") is not None
                        and baseline.get("avg_segnet_dist") is not None
                        else None
                    ),
                    "pose_term_delta_cpu": (
                        math.sqrt(10.0 * cpu_eval["avg_posenet_dist"])
                        - math.sqrt(10.0 * baseline["avg_posenet_dist"])
                        if cpu_eval
                        and cpu_eval.get("avg_posenet_dist") is not None
                        and baseline.get("avg_posenet_dist") is not None
                        else None
                    ),
                },
                "evidence_tags": [
                    "[CPU-prep]",
                    "[contest-CPU]" if cpu_eval_present else "[contest-CPU missing]",
                    "[contest-CUDA missing]",
                    "[no-score-claim]",
                ],
                "fail_closed_blockers": sorted(set(blockers)),
                "required_validation_commands": commands,
                "medal_path_note": (
                    "Existing uniform PR107 lossy-coarsening config is not a medal-path score claim; "
                    "reactivation requires scorer-aware tensor allocation or recovery plus paired CPU/CUDA exact eval."
                ),
            }
        )

    exact_cpu_present = sum(1 for row in candidate_rows if row.get("cpu_eval", {}).get("exists"))
    cpu_improvers = [
        row["candidate_id"]
        for row in candidate_rows
        if (
            row.get("delta_vs_pr107_cpu_anchor", {}).get("candidate_minus_baseline_score_cpu")
            is not None
            and row["delta_vs_pr107_cpu_anchor"]["candidate_minus_baseline_score_cpu"] < 0
        )
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "created_at_utc": recorded,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "remote_dispatch_attempted": False,
        "summary": {
            "candidate_count": len(candidate_rows),
            "exact_cpu_candidate_count": exact_cpu_present,
            "cpu_axis_improver_count": len(cpu_improvers),
            "cpu_axis_improvers": cpu_improvers,
            "fail_closed_verdict": (
                "No existing PR107 uniform lossy-coarsening candidate is promoted. "
                "CPU-present configs are worse than the PR107 CPU anchor; CPU-missing configs "
                "remain fail-closed; all rows are missing exact CUDA validation."
            ),
            "next_medal_path_artifact": (
                "Build scorer-aware per-tensor allocation or recovery/QAT candidate before any new dispatch; "
                "then rerun this matrix and require paired contest-CPU plus contest-CUDA review."
            ),
        },
        "baseline_cpu_anchor": baseline,
        "local_mac_cpu_diagnostic_anchor": _local_report_record(repo_root, _repo_path(repo_root, local_baseline_report)),
        "cathedral_autopilot": _cathedral_summary(
            repo_root,
            _repo_path(repo_root, cathedral_evidence),
            _repo_path(repo_root, cathedral_catalog),
        ),
        "builder_and_validation_surfaces": _builder_surfaces(repo_root),
        "candidates": candidate_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-cpu-eval", type=Path, default=DEFAULT_BASELINE_CPU_EVAL)
    parser.add_argument("--local-baseline-report", type=Path, default=DEFAULT_LOCAL_BASELINE_REPORT)
    parser.add_argument("--cathedral-evidence", type=Path, default=DEFAULT_CATHEDRAL_EVIDENCE)
    parser.add_argument("--cathedral-catalog", type=Path, default=DEFAULT_CATHEDRAL_CATALOG)
    parser.add_argument("--recorded-at-utc", default=None)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    matrix = build_matrix(
        repo_root=repo_root,
        baseline_cpu_eval=args.baseline_cpu_eval,
        local_baseline_report=args.local_baseline_report,
        cathedral_evidence=args.cathedral_evidence,
        cathedral_catalog=args.cathedral_catalog,
        recorded_at_utc=args.recorded_at_utc,
    )
    output_path = _repo_path(repo_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[pr107-cpu-lossy-matrix] wrote {output_path}")
    print(
        "[pr107-cpu-lossy-matrix] "
        f"candidates={matrix['summary']['candidate_count']} "
        f"exact_cpu={matrix['summary']['exact_cpu_candidate_count']} "
        f"cpu_improvers={matrix['summary']['cpu_axis_improver_count']} "
        "score_claim=false remote_dispatch_attempted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
