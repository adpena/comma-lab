# SPDX-License-Identifier: MIT
"""Incremental-first execution for HPRC pair-scoped runner rows."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive import parse_hprc_packet
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY, export_hprc_archive_bytes
from tac.substrates.hprc.learned_receiver import transform_compact_receiver_residual

HPRC_INCREMENTAL_RUNNER_EXECUTION_PREP_SCHEMA = (
    "hprc_incremental_pair_scoped_runner_execution_prep.v1"
)
HPRC_INCREMENTAL_RUNNER_EXECUTION_SCHEMA = (
    "hprc_incremental_pair_scoped_runner_execution.v1"
)
HPRC_SYNTHETIC_INCREMENTAL_PROFILE_SCHEMA = (
    "hprc_incremental_pair_scoped_synthetic_profile.v1"
)
HPRC_COMPONENT_PROFILE_SCHEMA = "hprc_mlx_component_neutralization_profile.v1"


def prepare_hprc_incremental_runner_execution(
    *,
    runner_plan_path: str | Path,
    candidate_id: str,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    scorer_batch_pairs: int = 1,
    cache_render_batch_pairs: int = 8,
    device: str = "cpu",
    allow_large_tensor_cache: bool = True,
    incremental_tool_path: str | Path = "tools/profile_hprc_incremental_pair_response.py",
) -> dict[str, Any]:
    """Materialize a pair-scoped archive and return an incremental execution row."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    plan_path = _resolve(runner_plan_path, base=root)
    plan = _load_json_object(plan_path)
    row = _select_runner_row(plan, candidate_id)
    candidate_dir = _resolve(plan["candidate_dir"], base=root)
    baseline_profile_path = _resolve(plan["reuse_baseline_profile_path"], base=root)
    baseline_profile = _load_json_object(baseline_profile_path)
    baseline_variant = _variant_row(baseline_profile, "baseline")
    transform = _required_string(row, "residual_transform")
    variant_id = _profile_variant_id(transform)
    selected_output = _resolve(output_dir, base=root)
    variant_dir = selected_output / "variants" / variant_id
    packet_path = candidate_dir / "0.bin"
    if not packet_path.is_file():
        raise FileNotFoundError(f"candidate 0.bin missing: {packet_path}")
    packet = parse_hprc_packet(packet_path.read_bytes())
    transformed_packet = transform_compact_receiver_residual(packet, transform=transform)
    archive_zip, archive_sha, archive_bytes = export_hprc_archive_bytes(
        transformed_packet,
        variant_dir,
        repo_root=root,
        emit_archive_bound_candidate_package=False,
        retain_receiver_proof_output=False,
    )
    hprc_0bin = variant_dir / "0.bin"
    synthetic_profile_path = selected_output / "synthetic_incremental_profile.json"
    synthetic_profile = _build_synthetic_profile(
        plan=plan,
        runner_row=row,
        baseline_profile_path=baseline_profile_path,
        baseline_profile=baseline_profile,
        baseline_variant=baseline_variant,
        variant_id=variant_id,
        archive_zip=archive_zip,
        archive_sha=archive_sha,
        archive_bytes=archive_bytes,
        hprc_0bin=hprc_0bin,
        transform=transform,
        root=root,
    )
    synthetic_profile_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_profile_path.write_text(
        json.dumps(synthetic_profile, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    pair_ranges_arg = _pair_ranges_arg(row.get("pair_ranges", []), transform=transform)
    incremental_output_dir = selected_output / "incremental_pair_response"
    incremental_tool = _resolve(incremental_tool_path, base=root)
    incremental_argv = [
        str(_python_bin(root)),
        incremental_tool.as_posix(),
        "--profile",
        synthetic_profile_path.as_posix(),
        "--candidate-variant-id",
        variant_id,
        "--pair-ranges",
        pair_ranges_arg,
        "--output-dir",
        incremental_output_dir.as_posix(),
        "--repo-root",
        root.as_posix(),
        "--candidate-archive",
        archive_zip.as_posix(),
        "--submission-dir",
        (variant_dir / "submission").as_posix(),
        "--device",
        device,
        "--scorer-batch-pairs",
        str(int(scorer_batch_pairs)),
        "--cache-render-batch-pairs",
        str(int(cache_render_batch_pairs)),
        "--force",
    ]
    if allow_large_tensor_cache:
        incremental_argv.append("--allow-large-tensor-cache")
    if int(scorer_batch_pairs) != 1:
        incremental_argv.append("--allow-batch-shape-research-signal")
    return {
        "schema": HPRC_INCREMENTAL_RUNNER_EXECUTION_PREP_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "repo_root": root.as_posix(),
        "runner_plan_path": plan_path.as_posix(),
        "runner_plan_sha256": _sha256_file(plan_path),
        "candidate_id": candidate_id,
        "runner_row": row,
        "baseline_profile_path": baseline_profile_path.as_posix(),
        "baseline_profile_sha256": _sha256_file(baseline_profile_path),
        "synthetic_profile_path": synthetic_profile_path.as_posix(),
        "synthetic_profile_sha256": _sha256_file(synthetic_profile_path),
        "candidate_variant_id": variant_id,
        "residual_transform": transform,
        "pair_ranges_arg": pair_ranges_arg,
        "archive": {
            "path": archive_zip.as_posix(),
            "bytes": int(archive_bytes),
            "sha256": archive_sha,
            "hprc_0bin_path": hprc_0bin.as_posix(),
            "hprc_0bin_sha256": _sha256_file(hprc_0bin),
        },
        "incremental_output_dir": incremental_output_dir.as_posix(),
        "incremental_command_argv": incremental_argv,
        "expected_incremental_report": (
            incremental_output_dir / "hprc_incremental_pair_response_report.json"
        ).as_posix(),
        "expected_cache_retention_plan": (
            incremental_output_dir / "artifact_retention_plan.json"
        ).as_posix(),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def build_hprc_incremental_runner_execution_report(
    *,
    prep: dict[str, Any],
    incremental_report_path: str | Path,
    incremental_stdout: str = "",
    incremental_stderr_tail: str = "",
    incremental_elapsed_seconds: float | None = None,
    retention_plan_path: str | Path | None = None,
    retention_stdout: str = "",
    retention_stderr_tail: str = "",
    proof_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Bind incremental execution evidence into a false-authority runner report."""

    incremental_path = Path(incremental_report_path).expanduser()
    incremental_report = _load_json_object(incremental_path)
    cleanup = _cleanup_status(None if retention_plan_path is None else Path(retention_plan_path))
    proof_binding = _find_receiver_proof_binding(
        archive_sha256=str(prep.get("archive", {}).get("sha256") or ""),
        proof_roots=[Path(root).expanduser() for root in proof_roots or []],
    )
    blockers = [
        "incremental_mlx_response_is_advisory_not_score_authority",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if proof_binding["receiver_contract_satisfied"] is not True:
        blockers.append("receiver_proof_missing_for_incremental_runner_candidate_sha")
    if cleanup.get("status") == "blocked":
        blockers.append("uncertified_mlx_cache_retained_cleanup_blocker")
    return {
        "schema": HPRC_INCREMENTAL_RUNNER_EXECUTION_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "prep_schema": prep.get("schema"),
        "candidate_id": prep.get("candidate_id"),
        "candidate_variant_id": prep.get("candidate_variant_id"),
        "residual_transform": prep.get("residual_transform"),
        "archive": prep.get("archive"),
        "synthetic_profile_path": prep.get("synthetic_profile_path"),
        "incremental_report_path": incremental_path.as_posix(),
        "incremental_report_sha256": _sha256_file(incremental_path),
        "incremental_command": {
            "argv": prep.get("incremental_command_argv"),
            "elapsed_seconds": incremental_elapsed_seconds,
            "stdout": incremental_stdout.strip(),
            "stderr_tail": incremental_stderr_tail[-4000:],
        },
        "incremental_summary": {
            "changed_pair_count": len(incremental_report.get("changed_pair_rows", [])),
            "full_video_pair_count": incremental_report.get("full_video_pair_count"),
            "archive_bytes_removed_vs_baseline": incremental_report.get(
                "archive_bytes_removed_vs_baseline"
            ),
            "delta_total_mlx_score_advisory": incremental_report.get(
                "delta_total_mlx_score_advisory"
            ),
            "delta_avg_posenet_dist": incremental_report.get("delta_avg_posenet_dist"),
            "delta_avg_segnet_dist": incremental_report.get("delta_avg_segnet_dist"),
        },
        "receiver_proof_binding": proof_binding,
        "cleanup": cleanup,
        "retention_command": {
            "plan_path": None if retention_plan_path is None else str(retention_plan_path),
            "stdout": retention_stdout.strip(),
            "stderr_tail": retention_stderr_tail[-4000:],
        },
        "exact_axis_gate": {
            "ready_for_exact_eval_dispatch": False,
            "blockers": blockers,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def write_hprc_incremental_runner_execution_report(
    *,
    output_path: str | Path,
    report: dict[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    """Write a deterministic incremental runner execution report."""

    path = Path(output_path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"output exists; pass allow_overwrite=True: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _build_synthetic_profile(
    *,
    plan: dict[str, Any],
    runner_row: dict[str, Any],
    baseline_profile_path: Path,
    baseline_profile: dict[str, Any],
    baseline_variant: dict[str, Any],
    variant_id: str,
    archive_zip: Path,
    archive_sha: str,
    archive_bytes: int,
    hprc_0bin: Path,
    transform: str,
    root: Path,
) -> dict[str, Any]:
    baseline_response = _resolve(
        str(baseline_variant["mlx_response"]),
        base=baseline_profile_path.parent,
    )
    reference_cache_dir = _resolve(
        str(baseline_profile["reference_cache_dir"]),
        base=baseline_profile_path.parent,
    )
    baseline_variant_row = dict(baseline_variant)
    baseline_variant_row["mlx_response"] = baseline_response.as_posix()
    return {
        "schema": HPRC_COMPONENT_PROFILE_SCHEMA,
        "profile_kind": HPRC_SYNTHETIC_INCREMENTAL_PROFILE_SCHEMA,
        "created_at_unix": time.time(),
        "repo_root": root.as_posix(),
        "candidate_dir": plan.get("candidate_dir"),
        "reference_cache_dir": reference_cache_dir.as_posix(),
        "baseline_reuse": {
            "schema": "hprc_baseline_mlx_response_reuse_report.v1",
            "enabled": True,
            "source_profile": baseline_profile_path.as_posix(),
            "source_response": baseline_response.as_posix(),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "max_pairs": int(baseline_profile.get("max_pairs") or 600),
        "window_pairs": int(baseline_profile.get("window_pairs") or 50),
        "scorer_batch_pairs": 1,
        "batch_shape_research_signal": False,
        "runner_row_candidate_id": runner_row.get("candidate_id"),
        "residual_transform": transform,
        "variant_rows": [
            baseline_variant_row,
            {
                "variant_id": variant_id,
                "neutralized_section": "residual_rc",
                "archive_zip_path": archive_zip.as_posix(),
                "archive_zip_bytes": int(archive_bytes),
                "archive_zip_sha256": archive_sha,
                "hprc_0bin_path": hprc_0bin.as_posix(),
                "hprc_0bin_sha256": _sha256_file(hprc_0bin),
                "mlx_response": "",
            },
        ],
        "blockers": [
            "synthetic_profile_for_incremental_pair_response_only",
            "mlx_local_response_is_advisory_not_score_authority",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        **FALSE_AUTHORITY,
    }


def _select_runner_row(plan: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    rows = plan.get("runner_rows")
    if not isinstance(rows, list):
        raise ValueError("runner plan missing runner_rows")
    matches = [row for row in rows if isinstance(row, dict) and row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one runner row for {candidate_id!r}")
    return matches[0]


def _variant_row(profile: dict[str, Any], variant_id: str) -> dict[str, Any]:
    rows = profile.get("variant_rows")
    if not isinstance(rows, list):
        raise ValueError("profile missing variant_rows")
    matches = [
        row for row in rows if isinstance(row, dict) and row.get("variant_id") == variant_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one variant row for {variant_id!r}")
    return matches[0]


def _profile_variant_id(transform: str) -> str:
    return f"residual_transform_{_profile_variant_slug(transform)}"


def _profile_variant_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    if len(slug) <= 80:
        return slug
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{slug[:63].rstrip('_')}_{digest}"


def _pair_ranges_arg(pair_ranges: Any, *, transform: str) -> str:
    if isinstance(pair_ranges, list) and pair_ranges:
        parts = []
        for row in pair_ranges:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(f"invalid pair range row: {row!r}")
            start, end = int(row[0]), int(row[1])
            parts.append(str(start) if start == end else f"{start}-{end}")
        return ",".join(parts)
    _, sep, raw = transform.partition("@")
    if not sep or not raw.strip():
        raise ValueError("pair-scoped transform missing pair range suffix")
    return raw.strip()


def _cleanup_status(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "schema": "hprc_incremental_runner_cleanup_status.v1",
            "status": "not_planned",
            "blockers": ["artifact_retention_plan_missing"],
        }
    payload = _load_json_object(path)
    plan = payload.get("plan", {})
    blocked = plan.get("blocked_candidates", []) if isinstance(plan, dict) else []
    candidates = plan.get("candidates", []) if isinstance(plan, dict) else []
    return {
        "schema": "hprc_incremental_runner_cleanup_status.v1",
        "status": "blocked" if blocked else "planned",
        "plan_path": path.as_posix(),
        "candidate_count": len(candidates),
        "blocked_candidate_count": len(blocked),
        "reclaimable_bytes": int(plan.get("total_reclaimable_bytes") or 0)
        if isinstance(plan, dict)
        else 0,
        "blocked_bytes_retained": sum(
            int(row.get("bytes") or 0) for row in blocked if isinstance(row, dict)
        ),
        "blockers": [
            blocker
            for row in blocked
            if isinstance(row, dict)
            for blocker in row.get("blockers", [])
        ],
    }


def _find_receiver_proof_binding(
    *,
    archive_sha256: str,
    proof_roots: list[Path],
) -> dict[str, Any]:
    for root in proof_roots:
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob("hprc_receiver_proof.json"))
        else:
            candidates = []
        for proof_path in candidates:
            proof = _load_json_object(proof_path)
            if proof.get("archive_sha256") != archive_sha256:
                continue
            return {
                "schema": "hprc_incremental_runner_receiver_proof_binding.v1",
                "status": "linked_by_archive_sha256",
                "proof_path": proof_path.as_posix(),
                "archive_sha256": archive_sha256,
                "receiver_contract_satisfied": proof.get("receiver_contract_satisfied") is True,
                "runtime_consumption_proof_ready": proof.get("runtime_consumption_proof_ready") is True,
                "receiver_output_sha256": proof.get("receiver_output_sha256"),
                "receiver_output_bytes": proof.get("receiver_output_bytes"),
                "ready_for_exact_eval_dispatch": False,
                "blockers": proof.get("blockers", []),
            }
    return {
        "schema": "hprc_incremental_runner_receiver_proof_binding.v1",
        "status": "missing",
        "archive_sha256": archive_sha256,
        "receiver_contract_satisfied": False,
        "runtime_consumption_proof_ready": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": ["matching_hprc_receiver_proof_not_found"],
    }


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing string field: {key}")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _resolve(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _python_bin(repo_root: Path) -> Path:
    candidate = repo_root / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else Path("python3")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


__all__ = [
    "HPRC_INCREMENTAL_RUNNER_EXECUTION_PREP_SCHEMA",
    "HPRC_INCREMENTAL_RUNNER_EXECUTION_SCHEMA",
    "build_hprc_incremental_runner_execution_report",
    "prepare_hprc_incremental_runner_execution",
    "write_hprc_incremental_runner_execution_report",
]
