# SPDX-License-Identifier: MIT
"""Deterministic replay bundles for local repair stackability probes."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.optimization.repair_campaign_scorer import (
    REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
)
from tac.repo_io import json_text, sha256_bytes, sha256_file

REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA = (
    "repair_campaign_stackability_replay_bundle.v1"
)
REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_DIFF_SCHEMA = (
    "repair_campaign_stackability_replay_bundle_diff.v1"
)
SAFE_REPLAY_ENVIRONMENT_CAPTURE_SCHEMA = "safe_replay_environment_capture.v1"

_REDACT_TOKENS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
    "KEY",
)


class RepairCampaignReplayBundleError(ValueError):
    """Raised when a repair stackability replay bundle cannot be built."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path: str | Path, *, repo_root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else repo_root / value


def _repo_rel(path: str | Path, *, repo_root: Path) -> str:
    value = Path(path)
    try:
        return value.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _file_record(
    *,
    label: str,
    path: str | Path,
    repo_root: Path,
    required: bool = True,
) -> dict[str, Any]:
    resolved = _resolve(path, repo_root=repo_root)
    present = resolved.is_file()
    if required and not present:
        raise RepairCampaignReplayBundleError(f"required artifact missing: {label}={path}")
    record = {
        "label": label,
        "path": _repo_rel(resolved, repo_root=repo_root),
        "required": required,
        "present": present,
    }
    if present:
        record.update(
            {
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    return record


def _stable_json_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(json_text(payload).encode("utf-8"))


def _git_text(args: Sequence[str], *, repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def capture_safe_replay_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture environment values with credential-like keys redacted."""

    source = dict(os.environ if environment is None else environment)
    captured: dict[str, str] = {}
    redacted: list[str] = []
    for key, value in sorted(source.items()):
        upper = key.upper()
        if any(token in upper for token in _REDACT_TOKENS):
            captured[key] = "[REDACTED]"
            redacted.append(key)
        else:
            captured[key] = str(value)
    return {
        "schema": SAFE_REPLAY_ENVIRONMENT_CAPTURE_SCHEMA,
        "policy": (
            "all environment keys captured for replay diagnostics; "
            "credential-like values are redacted"
        ),
        "env": captured,
        "redacted_keys": redacted,
    }


def _custody_file_records(
    probe: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in probe.get("local_mlx_custody_paths") or []:
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        key = str(item.get("key") or "local_mlx_custody_artifact")
        records.append(
            _file_record(label=f"local_mlx_custody:{key}", path=path, repo_root=repo_root)
        )
    return records


def build_repair_campaign_stackability_replay_bundle(
    *,
    score_report_path: str | Path,
    probe_path: str | Path,
    score_report: Mapping[str, Any],
    probe: Mapping[str, Any],
    replay_argv: Sequence[str],
    invocation_argv: Sequence[str],
    repo_root: str | Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a hash-bound replay bundle for one local MLX stackability probe."""

    repo = Path(repo_root)
    if score_report.get("schema") != REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        raise RepairCampaignReplayBundleError(
            "score report must be repair_campaign_score_report.v1"
        )
    if probe.get("schema") != REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA:
        raise RepairCampaignReplayBundleError(
            "probe must be repair_campaign_stackability_probe.v1"
        )
    require_no_truthy_authority_fields(
        score_report,
        context="repair_stackability_replay_bundle_score_report",
    )
    require_no_truthy_authority_fields(
        probe,
        context="repair_stackability_replay_bundle_probe",
    )
    if probe.get("stackability_ready") is not True:
        raise RepairCampaignReplayBundleError("probe is not stackability_ready")

    source_records = [
        _file_record(
            label="repair_campaign_score_report",
            path=score_report_path,
            repo_root=repo,
        ),
        _file_record(
            label="repair_campaign_stackability_probe",
            path=probe_path,
            repo_root=repo,
        ),
        *_custody_file_records(probe, repo_root=repo),
    ]
    if len(source_records) < 4:
        raise RepairCampaignReplayBundleError(
            "probe missing required local MLX custody file records"
        )

    stable_hash_manifest = {
        "schema": "repair_campaign_stackability_replay_hash_manifest.v1",
        "typed_response_id": probe.get("typed_response_id"),
        "source_records": source_records,
        "replay_argv": list(replay_argv),
        "component_response_axis": probe.get("component_response_axis"),
        "entropy_position_label": probe.get("entropy_position_label"),
        "allocated_repair_bytes": probe.get("allocated_repair_bytes"),
    }
    python_context = {
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
    }
    environment_capture = capture_safe_replay_environment(environment)
    git_context = {
        "head": _git_text(["rev-parse", "HEAD"], repo_root=repo),
        "status_short": _git_text(["status", "--short"], repo_root=repo),
    }
    execution_context_manifest = {
        "schema": "repair_campaign_stackability_replay_execution_context.v1",
        "invocation_argv": list(invocation_argv),
        "python": python_context,
        "environment": environment_capture,
        "git": git_context,
    }
    payload = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "tool": "tools/build_repair_campaign_stackability_replay_bundle.py",
        "replay_target_tool": "tools/run_repair_campaign_stackability_probe.py",
        "cwd": _repo_rel(repo, repo_root=repo),
        "typed_response_id": probe.get("typed_response_id"),
        "component_response_axis": "[macOS-MLX research-signal]",
        "replay_argv": list(replay_argv),
        "invocation_argv": list(invocation_argv),
        "python": python_context,
        "environment": environment_capture,
        "git": git_context,
        "hash_manifest": stable_hash_manifest,
        "hash_manifest_sha256": _stable_json_sha256(stable_hash_manifest),
        "source_records_sha256": _stable_json_sha256(
            {
                "schema": "repair_campaign_stackability_replay_source_records.v1",
                "source_records": source_records,
            }
        ),
        "replay_argv_sha256": _stable_json_sha256(
            {
                "schema": "repair_campaign_stackability_replay_argv.v1",
                "replay_argv": list(replay_argv),
            }
        ),
        "invocation_argv_sha256": _stable_json_sha256(
            {
                "schema": "repair_campaign_stackability_invocation_argv.v1",
                "invocation_argv": list(invocation_argv),
            }
        ),
        "environment_sha256": _stable_json_sha256(environment_capture),
        "execution_context_manifest": execution_context_manifest,
        "execution_context_sha256": _stable_json_sha256(execution_context_manifest),
        "calibration_gate": {
            "schema": "repair_campaign_stackability_replay_calibration_gate.v1",
            "local_signal_axis": "[macOS-MLX research-signal]",
            "target_contest_axes": ["contest-CPU", "contest-CUDA"],
            "blockers": [
                "local_mlx_replay_bundle_is_not_score_authority",
                "exact_axis_component_response_required_before_budget_spend",
                "receiver_runtime_materialization_required_before_exact_dispatch",
                "exact_auth_eval_required_before_score_or_promotion_claim",
            ],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "deterministic_local_mlx_repair_stackability_replay_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="repair_campaign_stackability_replay_bundle",
    )
    return payload


def diff_repair_campaign_stackability_replay_bundles(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two stackability replay bundles without granting authority."""

    for label, payload in (("left", left), ("right", right)):
        if payload.get("schema") != REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA:
            raise RepairCampaignReplayBundleError(f"{label} bundle schema mismatch")
        require_no_truthy_authority_fields(
            payload,
            context=f"repair_campaign_stackability_replay_bundle_diff:{label}",
        )
    left_env = left.get("environment", {})
    right_env = right.get("environment", {})
    left_keys = set(left_env.get("env", {}) if isinstance(left_env, Mapping) else {})
    right_keys = set(right_env.get("env", {}) if isinstance(right_env, Mapping) else {})
    common = left_keys & right_keys
    changed_env = [
        key
        for key in sorted(common)
        if left_env.get("env", {}).get(key) != right_env.get("env", {}).get(key)
    ]
    stable_replay_identity_matched = (
        left.get("hash_manifest_sha256") == right.get("hash_manifest_sha256")
    )
    execution_context_matched = (
        left.get("execution_context_sha256") == right.get("execution_context_sha256")
    )
    diff = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_DIFF_SCHEMA,
        "left_hash_manifest_sha256": left.get("hash_manifest_sha256"),
        "right_hash_manifest_sha256": right.get("hash_manifest_sha256"),
        "left_execution_context_sha256": left.get("execution_context_sha256"),
        "right_execution_context_sha256": right.get("execution_context_sha256"),
        "left_source_records_sha256": left.get("source_records_sha256"),
        "right_source_records_sha256": right.get("source_records_sha256"),
        "left_replay_argv_sha256": left.get("replay_argv_sha256"),
        "right_replay_argv_sha256": right.get("replay_argv_sha256"),
        "left_environment_sha256": left.get("environment_sha256"),
        "right_environment_sha256": right.get("environment_sha256"),
        "stable_replay_identity_matched": stable_replay_identity_matched,
        "execution_context_matched": execution_context_matched,
        "matched": stable_replay_identity_matched and execution_context_matched,
        "changed_environment_keys": changed_env,
        "left_only_environment_keys": sorted(left_keys - right_keys),
        "right_only_environment_keys": sorted(right_keys - left_keys),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        diff,
        context="repair_campaign_stackability_replay_bundle_diff",
    )
    return diff


__all__ = [
    "REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_DIFF_SCHEMA",
    "REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA",
    "RepairCampaignReplayBundleError",
    "build_repair_campaign_stackability_replay_bundle",
    "capture_safe_replay_environment",
    "diff_repair_campaign_stackability_replay_bundles",
]
