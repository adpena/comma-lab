#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ASYMPTOTIC PURSUIT ordered dispatch queue + operator-attention budget rollup.

Per CLAUDE.md "Council hierarchy: 4-tier protocol" operator-attention budget
section. Reads the canonical readiness assessment from
``tools/asymptotic_pursuit_candidate_readiness_assessment.py`` + emits an
ordered queue with:

  * Per-candidate dispatch sequence (smoke → 100ep → full eval → paired axis
    per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA")
  * Cost-band rollup ($-low to $-high) per Catalog #270
  * Operator-attention budget per tier (T2 ≤3/day; T3 ≤3/week)

Sister of ``tools/asymptotic_pursuit_candidate_readiness_assessment.py``.
Lane: lane_asymptotic_pursuit_substrate_class_shift_q4_pivot_top_priority_20260517.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
# Add tools/ to sys.path so we can import sibling tool by name.
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from asymptotic_pursuit_candidate_readiness_assessment import (  # noqa: E402
    CANONICAL_CANDIDATES,
    ReadinessAssessment,
    assess_candidates,
    build_operator_authorize_command,
    _parse_recipe,
    _recipe_session_budget_floor_usd,
)
from audit_catalog202_sentinel_cleanliness import (  # noqa: E402
    effective_sentinel_files as _catalog202_effective_sentinel_files,
    git_status_paths as _catalog202_git_status_paths,
    sentinel_set_sha256 as _catalog202_sentinel_set_sha256,
    sha256_file as _catalog202_sha256_file,
)

SESSION_DIRECTIVE_ENV_VAR = "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"
SESSION_BUDGET_ENV_VAR = "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD"
CATALOG202_BYPASS_INTENT_ENV_VAR = "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"  # OPERATOR_AUTHORIZE_CLEAN_BYPASS_OK:constant-only-no-env-set
CATALOG202_BYPASS_ATTESTATION_ENV_VAR = (
    "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED"
)
CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR = "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON"


def count_dirty_paths(repo_root: Path | None = None) -> int:
    """Return current dirty git path count for launch-precondition reporting."""

    repo = repo_root or REPO_ROOT
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.strip())


def _env_truthy(env: Mapping[str, str], name: str) -> bool:
    raw = env.get(name, "")
    return bool(raw and raw.strip().lower() not in {"", "0", "false", "no"})


def _catalog202_dirty_tree_attestation(
    *,
    dirty_path_count: int,
    env: Mapping[str, str],
    latest_sentinel_audit: dict[str, Any] | None = None,
    env_sentinel_audit: dict[str, Any] | None = None,
    current_sentinel_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the paired-env Catalog #202 precondition for dirty trees."""

    intent_truthy = _env_truthy(env, CATALOG202_BYPASS_INTENT_ENV_VAR)
    attestation_truthy = _env_truthy(env, CATALOG202_BYPASS_ATTESTATION_ENV_VAR)
    audit_truthy = _env_truthy(env, CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR)
    required = dirty_path_count > 0
    snapshot_blockers = tuple(
        (current_sentinel_snapshot or {}).get("snapshot_blockers") or ()
    )
    current_snapshot_valid = bool(current_sentinel_snapshot) and not snapshot_blockers
    dirty_sentinel_path_count = int(
        (current_sentinel_snapshot or {}).get("dirty_sentinel_path_count") or 0
    )
    dirty_sentinel_audit_required = bool(
        required and dirty_sentinel_path_count > 0
    )
    latest_audit_matches_current = _catalog202_audit_matches_current_snapshot(
        latest_sentinel_audit,
        current_sentinel_snapshot,
    )
    env_audit_matches_current = _catalog202_audit_matches_current_snapshot(
        env_sentinel_audit,
        current_sentinel_snapshot,
    )
    env_audit_current_if_provided = (not audit_truthy) or env_audit_matches_current
    satisfied = (not required) or (
        intent_truthy
        and attestation_truthy
        and current_snapshot_valid
        and env_audit_current_if_provided
        and (
            not dirty_sentinel_audit_required
            or (audit_truthy and env_audit_matches_current)
        )
    )
    missing: list[str] = []
    if required and not intent_truthy:
        missing.append(f"{CATALOG202_BYPASS_INTENT_ENV_VAR}=1")
    if required and not attestation_truthy:
        missing.append(
            f"{CATALOG202_BYPASS_ATTESTATION_ENV_VAR}=<operator_verified_sentinel_clean_attestation>"
        )
    if dirty_sentinel_audit_required and not audit_truthy:
        missing.append(
            f"{CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR}="
            f"{(latest_sentinel_audit or {}).get('path', '<current_catalog202_audit.json>')}"
        )
    elif dirty_sentinel_audit_required and not env_audit_matches_current:
        missing.append(
            f"{CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR}=<fresh_current_catalog202_audit_json>"
        )
    elif required and audit_truthy and not env_audit_matches_current:
        missing.append(
            f"{CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR}=<fresh_current_catalog202_audit_json>"
        )
    return {
        "required_for_paid_dispatch": required,
        "dirty_worktree_path_count": dirty_path_count,
        "satisfied_in_current_environment": satisfied,
        "intent_env_var": CATALOG202_BYPASS_INTENT_ENV_VAR,
        "attestation_env_var": CATALOG202_BYPASS_ATTESTATION_ENV_VAR,
        "audit_json_env_var": CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR,
        "intent_env_var_currently_truthy": intent_truthy,
        "attestation_env_var_currently_truthy": attestation_truthy,
        "audit_json_env_var_currently_truthy": audit_truthy,
        "current_sentinel_snapshot_valid": current_snapshot_valid,
        "dirty_sentinel_audit_required": dirty_sentinel_audit_required,
        "latest_sentinel_audit_matches_current": latest_audit_matches_current,
        "env_sentinel_audit_matches_current": env_audit_matches_current,
        "current_sentinel_snapshot": current_sentinel_snapshot,
        "latest_sentinel_audit": latest_sentinel_audit,
        "env_sentinel_audit": env_sentinel_audit,
        "missing_env_assignments": missing,
        "operator_action": (
            "Verify the Modal sentinel file set is clean, then export both "
            "Catalog #202 env vars, plus the sentinel audit JSON env var when "
            "any effective sentinel is dirty, before running the paid command; "
            "otherwise --require-clean-head will fail closed on the dirty tree."
            if required and not satisfied
            else None
        ),
    }


def _paid_launch_missing_preconditions(
    *, ready_for_paid_dispatch: bool, catalog202: dict[str, Any]
) -> list[str]:
    if not ready_for_paid_dispatch:
        return []
    missing: list[str] = []
    if (
        catalog202["required_for_paid_dispatch"]
        and not catalog202["satisfied_in_current_environment"]
    ):
        missing.append(
            "CATALOG_202_dirty_worktree_requires_paired_env_attestation_before_paid_dispatch"
        )
    if (
        catalog202.get("dirty_sentinel_audit_required")
        and not catalog202.get("env_sentinel_audit_matches_current")
    ):
        missing.append(
            "CATALOG_202_dirty_sentinel_requires_current_audit_json_before_paid_dispatch"
        )
    if (
        catalog202.get("required_for_paid_dispatch")
        and not catalog202.get("current_sentinel_snapshot_valid")
    ):
        missing.append(
            "CATALOG_202_current_sentinel_snapshot_required_before_paid_dispatch"
        )
    return missing


def _catalog202_audit_matches_current_snapshot(
    audit: dict[str, Any] | None,
    current_sentinel_snapshot: dict[str, Any] | None,
) -> bool:
    """Return True iff an audit artifact describes the current sentinel bytes."""

    return bool(
        audit
        and current_sentinel_snapshot
        and not audit.get("missing")
        and not audit.get("parse_error")
        and not audit.get("audit_backed_attestation_blockers")
        and not current_sentinel_snapshot.get("snapshot_blockers")
        and audit.get("sentinel_set_sha256")
        == current_sentinel_snapshot.get("sentinel_set_sha256")
        and list(audit.get("effective_sentinel_files") or [])
        == list(current_sentinel_snapshot.get("effective_sentinel_files") or [])
        and list(audit.get("dirty_sentinel_paths") or [])
        == list(current_sentinel_snapshot.get("dirty_sentinel_paths") or [])
    )


def _current_catalog202_sentinel_snapshot(
    recipe_path: Path | None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    """Return the current effective Modal sentinel hash/dirty snapshot.

    The queue must not infer dirty-sentinel status from the newest persisted
    audit because the operator may have edited a sentinel after that audit. This
    mirrors the audit helper's effective sentinel set and records only small
    custody fields needed to decide whether an audit-backed launch command is
    current.
    """

    if recipe_path is None or not recipe_path.exists():
        return None
    repo = repo_root or REPO_ROOT
    recipe = _parse_recipe(recipe_path)
    try:
        effective, missing, outside_mount = _catalog202_effective_sentinel_files(
            recipe,
            repo_root=repo,
        )
        git_status = _catalog202_git_status_paths(repo)
    except Exception as exc:
        return {
            "snapshot_blockers": [
                f"catalog202_current_sentinel_snapshot_failed:{type(exc).__name__}"
            ],
        }

    records: list[dict[str, Any]] = []
    dirty_paths = set(git_status)
    for rel in effective:
        path = repo / rel
        try:
            st = path.stat()
            sha = _catalog202_sha256_file(path)
        except OSError:
            missing.append(rel)
            continue
        records.append(
            {
                "path": rel,
                "exists": True,
                "size_bytes": st.st_size,
                "sha256": sha,
                "git_status": git_status.get(rel),
                "dirty_in_git": rel in dirty_paths,
            }
        )

    dirty_sentinels = sorted(
        row["path"] for row in records if row.get("dirty_in_git")
    )
    blockers: list[str] = []
    if missing:
        blockers.append("catalog202_sentinel_file_missing")
    if outside_mount:
        blockers.append("catalog202_sentinel_outside_modal_mount_set")
    return {
        "effective_sentinel_file_count": len(records),
        "effective_sentinel_files": [row["path"] for row in records],
        "sentinel_set_sha256": _catalog202_sentinel_set_sha256(records),
        "dirty_sentinel_path_count": len(dirty_sentinels),
        "dirty_sentinel_paths": dirty_sentinels,
        "missing_sentinel_files": sorted(set(missing)),
        "outside_modal_mount_sentinel_files": sorted(set(outside_mount)),
        "snapshot_blockers": blockers,
    }


def _latest_catalog202_sentinel_audit(
    recipe_basename: str,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    repo = repo_root or REPO_ROOT
    audit_dir = repo / ".omx" / "state" / "catalog202_sentinel_cleanliness"
    matches = sorted(audit_dir.glob(f"{recipe_basename}_*.json"))
    if not matches:
        return None
    return _read_catalog202_sentinel_audit(matches[-1], repo_root=repo)


def _env_catalog202_sentinel_audit(
    env: Mapping[str, str],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    """Read the audit artifact named by the current env, if any."""

    raw = env.get(CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR, "")
    if not raw.strip():
        return None
    repo = repo_root or REPO_ROOT
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo / path
    return _read_catalog202_sentinel_audit(path, repo_root=repo)


def _read_catalog202_sentinel_audit(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo = repo_root or REPO_ROOT
    try:
        rel = str(path.relative_to(repo))
    except ValueError:
        rel = str(path)
    if not path.is_file():
        return {"path": rel, "missing": True}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"path": rel, "parse_error": True}
    return {
        "path": rel,
        "schema": payload.get("schema"),
        "effective_sentinel_files": payload.get("effective_sentinel_files") or [],
        "sentinel_set_sha256": payload.get("sentinel_set_sha256"),
        "sentinel_set_clean_for_catalog202": payload.get(
            "sentinel_set_clean_for_catalog202"
        ),
        "ready_for_catalog202_audit_backed_dirty_sentinel_attestation": payload.get(
            "ready_for_catalog202_audit_backed_dirty_sentinel_attestation"
        ),
        "dirty_sentinel_path_count": payload.get("dirty_sentinel_path_count"),
        "dirty_sentinel_paths": payload.get("dirty_sentinel_paths") or [],
        "audit_backed_attestation_blockers": payload.get(
            "audit_backed_attestation_blockers"
        )
        or [],
    }


def _dry_run_command(
    recipe_basename: str,
    substrate_id: str,
    *,
    recipe_path: Path | None = None,
) -> str:
    """Return the non-spend smoke-before-full command for a queue row."""

    recipe = _parse_recipe(recipe_path) if recipe_path is not None else {}
    platform = str(recipe.get("platform", "modal")).strip().lower()
    if platform == "vastai":
        remote_driver = str(recipe.get("remote_driver") or "").strip()
        if not remote_driver:
            return (
                f"# Candidate {substrate_id} has platform=vastai but no "
                "remote_driver; dry-run unavailable"
            )
        cost_band = recipe.get("cost_band", {}) or {}
        predicted_band = recipe.get("predicted_band") or [0.0, 2.0]
        try:
            band_low = float(predicted_band[0])
            band_high = float(predicted_band[1])
        except (TypeError, ValueError, IndexError):
            band_low, band_high = 0.0, 2.0
        try:
            estimated_cost = float(
                cost_band.get("predicted_cost_usd")
                or cost_band.get("hand_calibrated_fallback_p50_usd")
                or 2.0
            )
        except (TypeError, ValueError):
            estimated_cost = 2.0
        vastai = recipe.get("vastai", {}) or {}
        if not isinstance(vastai, dict):
            vastai = {}
        label = str(vastai.get("label") or f"codex-{substrate_id}")
        max_dph = str(vastai.get("max_dph") or "0.50")
        min_disk_gb = str(vastai.get("min_disk_gb") or "60")
        return shlex.join(
            [
                ".venv/bin/python",
                "scripts/launch_lane_on_vastai.py",
                "--lane-script",
                remote_driver,
                "--label",
                label,
                "--predicted-band",
                f"{band_low:.6g}",
                f"{band_high:.6g}",
                "--estimated-cost",
                f"{estimated_cost:.6g}",
                "--max-dph",
                max_dph,
                "--min-disk-gb",
                min_disk_gb,
                "--dry-run",
            ]
        )
    return (
        ".venv/bin/python tools/run_modal_smoke_before_full.py "
        f"--recipe {recipe_basename} "
        f"--operator-handle codex:{substrate_id} "
        "--dry-run"
    )


def build_dispatch_sequence(
    assessment: ReadinessAssessment,
    *,
    repo_root: Path | None = None,
    dirty_path_count: int | None = None,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build an ordered dispatch sequence for each candidate.

    Per Catalog #167 smoke-before-full pattern: every candidate must
    smoke-first ($1) before full ($N), then paired-axis CPU eval (~$0.10) for
    contest-CPU verification.
    """
    sequence: list[dict[str, Any]] = []
    current_dirty_path_count = (
        count_dirty_paths(repo_root=repo_root)
        if dirty_path_count is None
        else max(0, dirty_path_count)
    )
    env_map = os.environ if env is None else env
    for sid in assessment.ranked_by_ev_per_dollar:
        c = next(x for x in assessment.candidates if x.substrate_id == sid)
        smoke_cost = 1.0  # Canonical $1 smoke ceiling
        paired_cost = 0.10
        # CandidateReadiness.estimated_dispatch_cost_usd is already the paired
        # CPU+CUDA estimate from _estimate_dispatch_cost(paired_axis=True).
        # Split it here so the explicit CPU axis stage is visible without
        # double-counting it in the rollup.
        full_cuda_cost = max(0.0, round(c.estimated_dispatch_cost_usd - paired_cost, 3))
        total_cost = smoke_cost + full_cuda_cost + paired_cost
        recipe_budget_floor = _recipe_session_budget_floor_usd(c.recipe_path)
        operator_session_budget_floor = max(round(total_cost, 3), recipe_budget_floor)
        ready_for_paid_dispatch = (
            c.readiness_verdict == "READY"
            and not c.blocking_issues
            and c.dispatch_enabled
            and not c.research_only
        )
        latest_sentinel_audit = _latest_catalog202_sentinel_audit(
            c.recipe_basename,
            repo_root=repo_root,
        )
        env_sentinel_audit = _env_catalog202_sentinel_audit(
            env_map,
            repo_root=repo_root,
        )
        current_sentinel_snapshot = _current_catalog202_sentinel_snapshot(
            c.recipe_path,
            repo_root=repo_root,
        )
        catalog202 = _catalog202_dirty_tree_attestation(
            dirty_path_count=current_dirty_path_count,
            env=env_map,
            latest_sentinel_audit=latest_sentinel_audit,
            env_sentinel_audit=env_sentinel_audit,
            current_sentinel_snapshot=current_sentinel_snapshot,
        )
        paid_launch_missing_preconditions = _paid_launch_missing_preconditions(
            ready_for_paid_dispatch=ready_for_paid_dispatch,
            catalog202=catalog202,
        )
        sequence.append(
            {
                "substrate_id": sid,
                "readiness_verdict": c.readiness_verdict,
                "ready_for_paid_dispatch": ready_for_paid_dispatch,
                "immediately_runnable_paid_launch": (
                    ready_for_paid_dispatch and not paid_launch_missing_preconditions
                ),
                "current_worktree_dirty_path_count": current_dirty_path_count,
                "paid_launch_missing_preconditions": paid_launch_missing_preconditions,
                "paid_launch_command": (
                    build_operator_authorize_command(c)
                    if ready_for_paid_dispatch
                    else None
                ),
                "audit_backed_paid_launch_command": (
                    (
                        f"{CATALOG202_BYPASS_INTENT_ENV_VAR}=1 "
                        f"{CATALOG202_BYPASS_ATTESTATION_ENV_VAR}="
                        f"catalog202_sentinel_audit:{latest_sentinel_audit['sentinel_set_sha256']} "
                        f"{CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR}="
                        f"{latest_sentinel_audit['path']} "
                        f"{build_operator_authorize_command(c)}"
                    )
                    if ready_for_paid_dispatch
                    and latest_sentinel_audit
                    and latest_sentinel_audit.get(
                        "ready_for_catalog202_audit_backed_dirty_sentinel_attestation"
                    )
                    and catalog202.get("latest_sentinel_audit_matches_current")
                    else None
                ),
                "dry_run_command": _dry_run_command(
                    c.recipe_basename,
                    sid,
                    recipe_path=c.recipe_path,
                ),
                "operator_session_authorization": {
                    "required_for_paid_dispatch": ready_for_paid_dispatch,
                    "session_directive_env_var": SESSION_DIRECTIVE_ENV_VAR,
                    "session_budget_env_var": SESSION_BUDGET_ENV_VAR,
                    "minimum_session_budget_usd": round(
                        operator_session_budget_floor, 3
                    ),
                    "budget_floor_basis": {
                        "queue_estimate_usd": round(total_cost, 3),
                        "recipe_declared_floor_usd": round(recipe_budget_floor, 3),
                    },
                    "catalog202_dirty_tree_attestation": catalog202,
                },
                "horizon_class": c.horizon_class,
                "predicted_delta_s_band": [c.predicted_delta_s_band_low, c.predicted_delta_s_band_high],
                "predicted_score_band": (
                    [c.predicted_delta_s_band_low, c.predicted_delta_s_band_high]
                    if c.predicted_band_kind == "predicted_score_band"
                    else None
                ),
                "predicted_band_kind": c.predicted_band_kind,
                "predicted_band_axis": c.predicted_band_axis,
                "predicted_band_validation_status": c.predicted_band_validation_status,
                "predicted_band_metadata_blockers": list(
                    c.predicted_band_metadata_blockers
                ),
                "stages": [
                    {
                        "stage": "smoke_100ep",
                        "estimated_cost_usd": smoke_cost,
                        "gpu": c.min_smoke_gpu,
                        "wall_clock_seconds": min(c.estimated_dispatch_wall_clock_seconds, 600),
                        "rationale": "Catalog #167 smoke-before-full pattern; refuses full dispatch on smoke failure",
                    },
                    {
                        "stage": "full_eval_contest_cuda",
                        "estimated_cost_usd": full_cuda_cost,
                        "gpu": c.gpu_class,
                        "wall_clock_seconds": c.estimated_dispatch_wall_clock_seconds,
                        "rationale": "Contest-CUDA full run; paired CPU axis is costed as a separate explicit stage below",
                    },
                    {
                        "stage": "paired_cpu_axis_verification",
                        "estimated_cost_usd": paired_cost,
                        "gpu": "CPU",
                        "wall_clock_seconds": 3600,
                        "rationale": "Catalog #316 frontier-scan drift detection on contest-CPU axis",
                    },
                ],
                "total_estimated_cost_usd": round(total_cost, 3),
                "operator_session_budget_floor_usd": round(
                    operator_session_budget_floor, 3
                ),
                "operator_session_budget_floor_basis": {
                    "queue_estimate_usd": round(total_cost, 3),
                    "recipe_declared_floor_usd": round(recipe_budget_floor, 3),
                },
                "local_identity_disambiguator_probe": {
                    "path": c.local_identity_disambiguator_probe_path,
                    "verdict": c.local_identity_disambiguator_probe_verdict,
                    "runtime_output_changed": (
                        c.local_identity_disambiguator_runtime_output_changed
                    ),
                    "custody": dict(c.local_identity_disambiguator_custody),
                    "blockers": list(c.local_identity_disambiguator_blockers),
                },
                "dispatch_blocker_supersessions": list(
                    c.dispatch_blocker_supersessions
                ),
                "blocking_issues": list(c.blocking_issues),
            }
        )
    return sequence


def compute_cost_band_rollup(sequence: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tier cost rollup per CLAUDE.md 'Production-hardened dispatch optimization protocol'."""
    ready = [s for s in sequence if s["readiness_verdict"] == "READY"]
    needs_fix = [s for s in sequence if s["readiness_verdict"] == "NEEDS_FIX"]
    defer = [s for s in sequence if s["readiness_verdict"] == "DEFER"]
    return {
        "ready_total_cost_usd_if_dispatched": round(
            sum(s["total_estimated_cost_usd"] for s in ready), 3
        ),
        "ready_total_session_budget_floor_usd": round(
            sum(s["operator_session_budget_floor_usd"] for s in ready), 3
        ),
        "needs_fix_total_cost_usd_if_unblocked_and_dispatched": round(
            sum(s["total_estimated_cost_usd"] for s in needs_fix), 3
        ),
        "needs_fix_total_session_budget_floor_usd": round(
            sum(s["operator_session_budget_floor_usd"] for s in needs_fix), 3
        ),
        "defer_total_cost_usd_if_phase_2_council_completes": round(
            sum(s["total_estimated_cost_usd"] for s in defer), 3
        ),
        "defer_total_session_budget_floor_usd": round(
            sum(s["operator_session_budget_floor_usd"] for s in defer), 3
        ),
        "ready_count": len(ready),
        "needs_fix_count": len(needs_fix),
        "defer_count": len(defer),
        "total_count": len(sequence),
    }


def compute_operator_attention_budget(
    sequence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per CLAUDE.md 'Council hierarchy 4-tier protocol' operator-attention
    budget per tier. Asymptotic-pursuit substrates typically need:
      - T2 sextet council deliberation per substrate before unblock
      - T3 grand council only if cross-cutting (e.g., Phase 2 lift)
    """
    # Per-substrate T2 deliberation: 1 per substrate; aggregate
    needs_fix_t2 = sum(1 for s in sequence if s["readiness_verdict"] == "NEEDS_FIX")
    defer_t2 = sum(1 for s in sequence if s["readiness_verdict"] == "DEFER")
    return {
        "t1_working_group_unbounded": "OK",
        "t2_sextet_council_per_substrate_to_unblock": needs_fix_t2 + defer_t2,
        "t2_per_30_day_budget": 90,
        "t2_within_budget": (needs_fix_t2 + defer_t2) <= 90,
        "t3_grand_council_for_phase_2_cross_cutting_unblock": defer_t2,
        "t3_per_30_day_budget": 13,
        "t3_within_budget": defer_t2 <= 13,
        "t4_symposium_only_for_strategic_pivots": 0,
    }


def build_payload(
    assessment: ReadinessAssessment,
    sequence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the machine-readable dispatch queue payload."""

    ready_rows = [row for row in sequence if row["ready_for_paid_dispatch"]]
    immediately_runnable_rows = [
        row for row in ready_rows if row["immediately_runnable_paid_launch"]
    ]
    top_ready = ready_rows[0] if ready_rows else None
    top_immediately_runnable = (
        immediately_runnable_rows[0] if immediately_runnable_rows else None
    )
    dirty_path_count = (
        int(sequence[0].get("current_worktree_dirty_path_count", 0))
        if sequence
        else 0
    )
    ready_requiring_catalog202 = [
        row
        for row in ready_rows
        if row["operator_session_authorization"]["catalog202_dirty_tree_attestation"][
            "required_for_paid_dispatch"
        ]
    ]
    return {
        "dispatch_sequence": sequence,
        "cost_band_rollup": compute_cost_band_rollup(sequence),
        "operator_attention_budget": compute_operator_attention_budget(sequence),
        "top_1_substrate": assessment.top_1_substrate,
        "top_1_readiness_verdict": assessment.top_1_readiness_verdict,
        "top_2_substrate_for_stage_2_stacking": assessment.top_2_substrate,
        "ready_for_paid_dispatch_count": len(ready_rows),
        "immediately_runnable_paid_dispatch_count": len(immediately_runnable_rows),
        "current_worktree_dirty_path_count": dirty_path_count,
        "ready_paid_rows_requiring_catalog202_dirty_tree_attestation_count": len(
            ready_requiring_catalog202
        ),
        "top_ready_substrate": top_ready["substrate_id"] if top_ready else None,
        "top_ready_paid_launch_command": (
            top_ready["paid_launch_command"] if top_ready else None
        ),
        "top_ready_audit_backed_paid_launch_command": (
            top_ready["audit_backed_paid_launch_command"] if top_ready else None
        ),
        "top_ready_paid_launch_missing_preconditions": (
            top_ready["paid_launch_missing_preconditions"] if top_ready else []
        ),
        "top_immediately_runnable_paid_launch_command": (
            top_immediately_runnable["paid_launch_command"]
            if top_immediately_runnable
            else None
        ),
        "top_ready_dry_run_command": top_ready["dry_run_command"] if top_ready else None,
        "assessment_utc": assessment.assessment_utc,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "predicted",
        "provenance_kind": "PREDICTED_FROM_MODEL",
        "result_review_blockers": [
            "dispatch_queue_is_planning_artifact_not_score_claim",
            "requires_operator_session_directive_budget_and_lane_claim_before_provider_dispatch",
            "requires_catalog202_dirty_tree_attestation_when_worktree_dirty_before_paid_provider_dispatch",
            "requires_paired_contest_cuda_cpu_harvest_before_score_or_promotion_claim",
        ],
    }


def write_artifact(
    payload: dict[str, Any], *, repo_root: Path | None = None
) -> Path:
    """Persist a timestamped dispatch-queue artifact for operator handoff."""

    repo = repo_root or REPO_ROOT
    artifact_dir = repo / ".omx" / "state" / "asymptotic_pursuit"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(payload["assessment_utc"]).replace(":", "").replace("-", "")
    path = artifact_dir / f"dispatch_queue_{stamp}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="Emit JSON output")
    p.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Comma-separated candidate IDs",
    )
    p.add_argument("--repo-root", type=Path, default=None, help="Override repo root.")
    p.add_argument(
        "--write-artifact",
        action="store_true",
        help="Persist queue JSON to .omx/state/asymptotic_pursuit/.",
    )
    args = p.parse_args(argv)

    candidates = (
        tuple(c.strip() for c in args.candidates.split(",")) if args.candidates else None
    )
    assessment = assess_candidates(candidates, repo_root=args.repo_root)
    sequence = build_dispatch_sequence(assessment)
    payload = build_payload(assessment, sequence)
    rollup = payload["cost_band_rollup"]
    budget = payload["operator_attention_budget"]

    if args.write_artifact:
        path = write_artifact(payload, repo_root=args.repo_root)
        print(f"[asymptotic-pursuit-queue] wrote artifact: {path}", file=sys.stderr)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"=== ASYMPTOTIC PURSUIT dispatch queue ===")
        print(f"  TOP-1: {assessment.top_1_substrate} ({assessment.top_1_readiness_verdict})")
        print(f"  TOP-2 (Stage 2 stacking): {assessment.top_2_substrate}")
        print(
            "  immediately_runnable_paid_dispatch_count: "
            f"{payload['immediately_runnable_paid_dispatch_count']}"
        )
        print(
            "  current_worktree_dirty_path_count: "
            f"{payload['current_worktree_dirty_path_count']}"
        )
        print()
        print(f"=== Cost-band rollup ===")
        for k, v in rollup.items():
            print(f"  {k}: {v}")
        print()
        print(f"=== Operator-attention budget ===")
        for k, v in budget.items():
            print(f"  {k}: {v}")
        print()
        print(f"=== Per-substrate dispatch sequence ===")
        for s in sequence:
            print(f"\n  {s['substrate_id']} [{s['readiness_verdict']}] horizon={s['horizon_class']}")
            print(f"    predicted_ΔS={s['predicted_delta_s_band']}")
            print(f"    total_cost=${s['total_estimated_cost_usd']}")
            print(
                "    operator_session_budget_floor="
                f"${s['operator_session_budget_floor_usd']}"
            )
            if s["blocking_issues"]:
                print(f"    blocking: {len(s['blocking_issues'])} issue(s)")
            if s["paid_launch_missing_preconditions"]:
                print(
                    "    paid_launch_missing_preconditions: "
                    f"{s['paid_launch_missing_preconditions']}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
