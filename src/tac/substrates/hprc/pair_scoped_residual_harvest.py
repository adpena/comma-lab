# SPDX-License-Identifier: MIT
"""Harvest executed HPRC pair-scoped residual runner rows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_HARVEST_SCHEMA = (
    "hprc_pair_scoped_residual_runner_harvest.v1"
)


def build_pair_scoped_residual_runner_harvest(
    *,
    runner_plan_path: str | Path,
    candidate_id: str,
    proof_roots: list[str | Path] | None = None,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return a typed harvest row for one executed pair-scoped HPRC candidate."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    plan_path = _resolve(runner_plan_path, base=root)
    plan = _load_json_object(plan_path)
    row = _select_runner_row(plan, candidate_id)
    profile_path = _resolve(row["expected_profile_report"], base=root)
    profile = _load_json_object(profile_path)
    variant_row = _select_candidate_variant(profile)
    section_row = _select_section_row(profile, variant_row["variant_id"])
    archive_sha = str(variant_row["archive_zip_sha256"])
    proof_binding = _find_receiver_proof_binding(
        archive_sha256=archive_sha,
        proof_roots=[_resolve(path, base=root) for path in proof_roots or []],
        repo_root=root,
    )
    cleanup = _cleanup_status(profile_path.parent)
    exact_blockers = [
        "contest_cpu_cuda_exact_eval_not_executed",
        "mlx_local_response_is_advisory_not_score_authority",
    ]
    if proof_binding["receiver_contract_satisfied"] is not True:
        exact_blockers.append("receiver_proof_missing_for_harvested_candidate_sha")
    if cleanup.get("status") == "blocked":
        exact_blockers.append("uncertified_mlx_cache_retained_cleanup_blocker")
    return {
        "schema": HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_HARVEST_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "repo_root": root.as_posix(),
        "runner_plan_path": plan_path.as_posix(),
        "candidate_id": candidate_id,
        "profile_path": profile_path.as_posix(),
        "profile_elapsed_seconds": profile.get("elapsed_seconds"),
        "baseline_reuse": profile.get("baseline_reuse"),
        "scorer_batch_pairs": profile.get("scorer_batch_pairs"),
        "batch_shape_research_signal": profile.get("batch_shape_research_signal"),
        "archive": {
            "path": variant_row.get("archive_zip_path"),
            "bytes": int(variant_row["archive_zip_bytes"]),
            "sha256": archive_sha,
            "hprc_0bin_path": variant_row.get("hprc_0bin_path"),
            "hprc_0bin_sha256": variant_row.get("hprc_0bin_sha256"),
        },
        "mlx_advisory_delta": {
            "delta_nonrate_score": section_row.get("delta_nonrate_score"),
            "delta_rate_score": section_row.get("delta_rate_score"),
            "delta_total_mlx_score_advisory": section_row.get(
                "delta_total_mlx_score_advisory"
            ),
            "archive_bytes_removed_vs_baseline": section_row.get(
                "archive_bytes_removed_vs_baseline"
            ),
            "marginal_status": section_row.get("marginal_status"),
        },
        "receiver_proof_binding": proof_binding,
        "cleanup": cleanup,
        "exact_axis_gate": {
            "ready_for_exact_eval_dispatch": False,
            "blockers": exact_blockers,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def write_pair_scoped_residual_runner_harvest(
    *,
    output_path: str | Path,
    harvest: dict[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    """Write a deterministic HPRC pair-scoped runner harvest."""

    path = Path(output_path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"output exists; pass allow_overwrite=True: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(harvest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _select_runner_row(plan: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    rows = plan.get("runner_rows")
    if not isinstance(rows, list):
        raise ValueError("runner plan missing runner_rows")
    matches = [row for row in rows if isinstance(row, dict) and row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one runner row for {candidate_id!r}")
    return matches[0]


def _select_candidate_variant(profile: dict[str, Any]) -> dict[str, Any]:
    rows = profile.get("variant_rows")
    if not isinstance(rows, list):
        raise ValueError("profile missing variant_rows")
    candidates = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("variant_id")) != "baseline"
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one non-baseline variant in runner profile")
    return candidates[0]


def _select_section_row(profile: dict[str, Any], variant_id: str) -> dict[str, Any]:
    rows = profile.get("section_value_rows")
    if not isinstance(rows, list):
        raise ValueError("profile missing section_value_rows")
    matches = [row for row in rows if isinstance(row, dict) and row.get("variant_id") == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one section row for {variant_id!r}")
    return matches[0]


def _find_receiver_proof_binding(
    *,
    archive_sha256: str,
    proof_roots: list[Path],
    repo_root: Path,
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
                "schema": "hprc_pair_scoped_receiver_proof_binding.v1",
                "status": "linked_by_archive_sha256",
                "proof_path": _repo_relative(proof_path, repo_root),
                "archive_sha256": archive_sha256,
                "receiver_contract_satisfied": proof.get("receiver_contract_satisfied") is True,
                "runtime_consumption_proof_ready": proof.get("runtime_consumption_proof_ready") is True,
                "receiver_output_sha256": proof.get("receiver_output_sha256"),
                "receiver_output_bytes": proof.get("receiver_output_bytes"),
                "ready_for_exact_eval_dispatch": False,
                "blockers": proof.get("blockers", []),
            }
    return {
        "schema": "hprc_pair_scoped_receiver_proof_binding.v1",
        "status": "missing",
        "archive_sha256": archive_sha256,
        "receiver_contract_satisfied": False,
        "runtime_consumption_proof_ready": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": ["matching_hprc_receiver_proof_not_found"],
    }


def _cleanup_status(profile_dir: Path) -> dict[str, Any]:
    plan_path = profile_dir / "artifact_retention_plan.json"
    if not plan_path.is_file():
        return {
            "schema": "hprc_pair_scoped_cleanup_status.v1",
            "status": "not_planned",
            "blockers": ["artifact_retention_plan_missing"],
        }
    plan = _load_json_object(plan_path).get("plan", {})
    blocked = plan.get("blocked_candidates", []) if isinstance(plan, dict) else []
    candidates = plan.get("candidates", []) if isinstance(plan, dict) else []
    blocked_bytes = sum(int(row.get("bytes") or 0) for row in blocked if isinstance(row, dict))
    reclaimable_bytes = int(plan.get("total_reclaimable_bytes") or 0) if isinstance(plan, dict) else 0
    return {
        "schema": "hprc_pair_scoped_cleanup_status.v1",
        "status": "blocked" if blocked else "planned",
        "plan_path": plan_path.as_posix(),
        "candidate_count": len(candidates),
        "blocked_candidate_count": len(blocked),
        "reclaimable_bytes": reclaimable_bytes,
        "blocked_bytes_retained": blocked_bytes,
        "blockers": [
            blocker
            for row in blocked
            if isinstance(row, dict)
            for blocker in row.get("blockers", [])
        ],
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _resolve(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


__all__ = [
    "HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_HARVEST_SCHEMA",
    "build_pair_scoped_residual_runner_harvest",
    "write_pair_scoped_residual_runner_harvest",
]
