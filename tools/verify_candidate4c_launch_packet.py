#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""No-spend launch packet doctor for Z6-v2 Candidate 4c.

This verifies the current Candidate 4c dispatch/handoff surface without opening
a lane claim or provider job. It checks:

* whether the live recipe is an intentionally diagnostic-only training surface
  or a paid-launch surface;
* if paid-launchable, the latest Race Mode queue points at Candidate 4c and
  carries an audit-backed paid command;
* if paid-launchable, the referenced Catalog #202 sentinel audit is hash-stable
  and audit-backed;
* lane claims are currently clean;
* required input files validate locally;
* smoke-before-full and operator-authorize dry-runs resolve;
* the Catalog #202 dirty-sentinel bypass helper accepts the audit JSON when a
  paid launch command is actually in scope;
* the exact-eval handoff requirements for promoting a harvested 600-pair
  Candidate 4c archive pair to paired Modal CUDA/CPU auth eval.

The output is a planning/launch artifact only: no score claim, no promotion
claim, no provider dispatch, and no lane claim.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE = "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch"
SUBSTRATE_ID = "z6_v2_candidate_4c_scorer_logit"
OPERATOR_HANDLE = "codex:z6_v2_candidate_4c_scorer_logit"
TRAINER = "experiments/train_substrate_time_traveler_l5_z6.py"
VIDEO_PATH = "upstream/videos/0.mkv"
INTENT_ENV = "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"
ATTESTATION_ENV = "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED"
AUDIT_JSON_ENV = "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON"
SESSION_DIRECTIVE_ENV = "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"
SESSION_BUDGET_ENV = "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD"
SESSION_BUDGET_USD = "1.250"
RECIPE_PATH = (
    ".omx/operator_authorize_recipes/"
    f"{RECIPE}.yaml"
)
FULL_600_PAIR_COUNT = 600
EXACT_EVAL_PAIR_GROUP_ID = "candidate4c_scorer_logit_full_vs_identity"
FULL_CUDA_LANE_ID = (
    "lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_full_contest_cuda"
)
IDENTITY_CUDA_LANE_ID = (
    "lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_identity_contest_cuda"
)
FULL_CPU_LANE_ID = (
    "lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_full_contest_cpu"
)
IDENTITY_CPU_LANE_ID = (
    "lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_identity_contest_cpu"
)
FULL_LANE_ID_BASE = (
    "lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_full"
)
IDENTITY_LANE_ID_BASE = (
    "lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_identity"
)
HEX_DIGITS = set("0123456789abcdefABCDEF")

CommandRunner = Callable[
    [Sequence[str], Path, Mapping[str, str] | None], subprocess.CompletedProcess[str]
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _latest_queue_path(repo_root: Path = REPO_ROOT) -> Path:
    queue_dir = repo_root / ".omx" / "state" / "asymptotic_pursuit"
    matches = sorted(queue_dir.glob("dispatch_queue_*.json"))
    if not matches:
        raise FileNotFoundError(f"no dispatch_queue_*.json under {queue_dir}")
    return matches[-1]


def _latest_codex_review_path(repo_root: Path = REPO_ROOT) -> Path | None:
    review_dir = repo_root / ".omx" / "state" / "candidate4c_launch_packet"
    matches = sorted(review_dir.glob("candidate4c_codex_pre_dispatch_review_*.json"))
    return matches[-1] if matches else None


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _default_runner(
    cmd: Sequence[str],
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        list(cmd),
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run(
    name: str,
    cmd: Sequence[str],
    *,
    repo_root: Path,
    runner: CommandRunner,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    proc = runner(cmd, repo_root, env)
    return {
        "name": name,
        "command": list(cmd),
        "env_overrides": dict(env or {}),
        "returncode": proc.returncode,
        "stdout": _truncate(proc.stdout or ""),
        "stderr": _truncate(proc.stderr or ""),
        "ok": proc.returncode == 0,
    }


def _claim_summary_active_clean(stdout: str) -> bool:
    return "CLAIM_SUMMARY" in stdout and "active=0" in stdout


def _catalog202_bypass_probe_accepted(stdout: str) -> bool:
    """The probe subprocess exits 0 even when it prints False."""

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    return bool(lines and lines[-1] == "True")


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_recipe_status(repo_root: Path) -> dict[str, Any]:
    """Read the live recipe without importing operator-authorize internals."""

    path = repo_root / RECIPE_PATH
    if not path.is_file():
        return {
            "path": RECIPE_PATH,
            "exists": False,
            "dispatch_enabled": None,
            "research_only": None,
            "smoke_only": None,
            "smoke_validation_contract": None,
            "dispatch_blockers": ["candidate4c_recipe_missing"],
            "current_mode": "recipe_missing",
        }
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive artifact path
        return {
            "path": RECIPE_PATH,
            "exists": True,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "dispatch_enabled": None,
            "research_only": None,
            "smoke_only": None,
            "smoke_validation_contract": None,
            "dispatch_blockers": ["candidate4c_recipe_unreadable"],
            "current_mode": "recipe_unreadable",
        }
    if not isinstance(raw, Mapping):
        return {
            "path": RECIPE_PATH,
            "exists": True,
            "parse_error": "recipe root is not a mapping",
            "dispatch_enabled": None,
            "research_only": None,
            "smoke_only": None,
            "smoke_validation_contract": None,
            "dispatch_blockers": ["candidate4c_recipe_not_mapping"],
            "current_mode": "recipe_unreadable",
        }
    blockers = raw.get("dispatch_blockers") or []
    if not isinstance(blockers, list):
        blockers = ["candidate4c_recipe_dispatch_blockers_not_list"]
    dispatch_enabled = bool(raw.get("dispatch_enabled", True))
    smoke_only = bool(raw.get("smoke_only", False))
    diagnostic_blocker = (
        "candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required"
        in blockers
    )
    if not dispatch_enabled and diagnostic_blocker:
        current_mode = "diagnostic_only_exact_eval_handoff_required"
    elif dispatch_enabled:
        current_mode = "paid_training_launch_surface"
    else:
        current_mode = "dispatch_disabled_other"
    return {
        "path": RECIPE_PATH,
        "exists": True,
        "name": raw.get("name"),
        "lane_id": raw.get("lane_id"),
        "dispatch_enabled": dispatch_enabled,
        "research_only": raw.get("research_only"),
        "smoke_only": smoke_only,
        "smoke_validation_contract": raw.get("smoke_validation_contract"),
        "dispatch_blockers": blockers,
        "current_mode": current_mode,
        "target_modes": raw.get("target_modes") or [],
    }


def _skipped_check(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "command": [],
        "env_overrides": {},
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "ok": True,
        "skipped": True,
        "skip_reason": reason,
    }


def _local_identity_disambiguator_probe_status(
    probe: Any,
    *,
    repo_root: Path,
) -> tuple[bool, list[str]]:
    """Validate the queue-carried local full-vs-identity no-op proof."""

    blockers: list[str] = []
    if not isinstance(probe, Mapping):
        return False, ["candidate4c_local_identity_disambiguator_probe_missing"]
    probe_path_raw = probe.get("path")
    if not probe_path_raw:
        blockers.append("candidate4c_local_identity_disambiguator_probe_path_missing")
    if probe.get("verdict") != "pending_paired_exact_eval_json":
        blockers.append(
            "candidate4c_local_identity_disambiguator_probe_verdict_not_pending_exact_eval"
        )
    if probe.get("runtime_output_changed") is not True:
        blockers.append(
            "candidate4c_local_identity_disambiguator_runtime_output_not_changed"
        )
    if probe.get("blockers") not in ([], ()):
        blockers.append("candidate4c_local_identity_disambiguator_probe_has_blockers")

    custody = probe.get("custody")
    if not isinstance(custody, Mapping):
        blockers.append("candidate4c_local_identity_disambiguator_custody_missing")
    else:
        for field in (
            "runtime_custody_aggregate_sha256",
            "full_output_aggregate_sha256",
            "identity_output_aggregate_sha256",
            "output_root",
        ):
            if not custody.get(field):
                blockers.append(
                    f"candidate4c_local_identity_disambiguator_{field}_missing"
                )
        total_byte_differences = custody.get("total_byte_differences")
        if (
            not isinstance(total_byte_differences, int)
            or total_byte_differences <= 0
        ):
            blockers.append(
                "candidate4c_local_identity_disambiguator_total_byte_differences_not_positive"
            )
    if probe_path_raw:
        probe_path = _resolve_repo_path(repo_root, str(probe_path_raw))
        if not probe_path.is_file():
            blockers.append("candidate4c_local_identity_disambiguator_artifact_missing")
        else:
            try:
                artifact = json.loads(probe_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                blockers.append("candidate4c_local_identity_disambiguator_artifact_unreadable")
                artifact = None
            if not isinstance(artifact, Mapping):
                blockers.append("candidate4c_local_identity_disambiguator_artifact_not_json_object")
            else:
                if artifact.get("verdict") != probe.get("verdict"):
                    blockers.append(
                        "candidate4c_local_identity_disambiguator_artifact_verdict_mismatch"
                    )
                comparison = artifact.get("inflate_output_comparison")
                if not isinstance(comparison, Mapping):
                    blockers.append(
                        "candidate4c_local_identity_disambiguator_artifact_comparison_missing"
                    )
                else:
                    if comparison.get("runtime_output_changed") is not True:
                        blockers.append(
                            "candidate4c_local_identity_disambiguator_artifact_runtime_output_not_changed"
                        )
                    if comparison.get("total_byte_differences") != (
                        custody.get("total_byte_differences")
                        if isinstance(custody, Mapping)
                        else None
                    ):
                        blockers.append(
                            "candidate4c_local_identity_disambiguator_total_byte_differences_artifact_mismatch"
                        )
                    artifact_custody_pairs = (
                        (
                            "runtime_custody_aggregate_sha256",
                            ("runtime_custody", "aggregate_sha256"),
                        ),
                        (
                            "full_output_aggregate_sha256",
                            ("full_output_tree", "aggregate_sha256"),
                        ),
                        (
                            "identity_output_aggregate_sha256",
                            ("identity_output_tree", "aggregate_sha256"),
                        ),
                        ("output_root", ("output_root",)),
                    )
                    for queue_field, artifact_path in artifact_custody_pairs:
                        expected = (
                            custody.get(queue_field)
                            if isinstance(custody, Mapping)
                            else None
                        )
                        cur: Any = comparison
                        for key in artifact_path:
                            if not isinstance(cur, Mapping):
                                cur = None
                                break
                            cur = cur.get(key)
                        if expected != cur:
                            blockers.append(
                                "candidate4c_local_identity_disambiguator_"
                                f"{queue_field}_artifact_mismatch"
                            )
    return not blockers, blockers


def _paid_command_status(
    command: Any,
    *,
    audit_rel: str | None,
    sentinel_hash: str,
) -> tuple[bool, list[str]]:
    """Validate that the handoff paid command names the audited sentinel state."""

    blockers: list[str] = []
    if not isinstance(command, str) or not command.strip():
        return False, ["candidate4c_next_paid_command_missing"]
    required_snippets = {
        "catalog202 intent env": f"{INTENT_ENV}=1",
        "catalog202 audit env": (
            f"{AUDIT_JSON_ENV}={audit_rel}" if audit_rel else None
        ),
        "catalog202 sentinel hash": (
            f"{ATTESTATION_ENV}=catalog202_sentinel_audit:{sentinel_hash}"
            if sentinel_hash
            else None
        ),
        "session directive env": f"{SESSION_DIRECTIVE_ENV}=1",
        "session budget env": f"{SESSION_BUDGET_ENV}={SESSION_BUDGET_USD}",
        "smoke-before-full tool": "tools/run_modal_smoke_before_full.py",
        "recipe": f"--recipe {RECIPE}",
        "operator handle": f"--operator-handle {OPERATOR_HANDLE}",
    }
    for label, snippet in required_snippets.items():
        if snippet is None or snippet not in command:
            key = label.replace(" ", "_").replace("-", "_")
            blockers.append(f"candidate4c_next_paid_command_missing_{key}")
    if "--dry-run" in command:
        blockers.append("candidate4c_next_paid_command_contains_dry_run")
    return not blockers, blockers


def _zip_member_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            for info in infos:
                blob = zf.read(info.filename)
                rows.append(
                    {
                        "name": info.filename,
                        "file_size": info.file_size,
                        "compress_size": info.compress_size,
                        "crc": f"{info.CRC:08x}",
                        "sha256": __import__("hashlib").sha256(blob).hexdigest(),
                        "compression_method": info.compress_type,
                    }
                )
    except (OSError, zipfile.BadZipFile) as exc:
        blockers.append(f"zip_unreadable:{type(exc).__name__}")
    return rows, blockers


def _latest_pair_probe_artifact(
    local_probe: Any,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if not isinstance(local_probe, Mapping) or not local_probe.get("path"):
        return {"path": None, "exists": False, "payload": None}
    path = _resolve_repo_path(repo_root, str(local_probe["path"]))
    if not path.is_file():
        return {"path": str(path), "exists": False, "payload": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": str(path),
            "exists": True,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "payload": None,
        }
    return {
        "path": str(path.relative_to(repo_root))
        if path.is_relative_to(repo_root)
        else str(path),
        "exists": True,
        "payload": payload if isinstance(payload, Mapping) else None,
    }


def _archive_source_row_status(row: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    path = _resolve_repo_path(repo_root, str(row.get("zip_path") or ""))
    exists = path.is_file()
    zip_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    if not exists:
        blockers.append("archive_zip_missing")
    else:
        zip_rows, zip_blockers = _zip_member_rows(path)
        blockers.extend(zip_blockers)
        if len(zip_rows) != 1:
            blockers.append("archive_zip_not_single_member")
        if zip_rows and zip_rows[0].get("name") != "0.bin":
            blockers.append("archive_zip_member_not_0_bin")
    return {
        "mode": row.get("mode"),
        "identity_predictor": row.get("identity_predictor"),
        "zip_path": row.get("zip_path"),
        "zip_exists": exists,
        "zip_bytes": path.stat().st_size if exists else None,
        "zip_sha256": row.get("zip_sha256") or row.get("contest_archive_sha256_basis"),
        "num_pairs": row.get("num_pairs"),
        "zip_member_rows": zip_rows,
        "blockers": blockers,
    }


def _paired_dispatch_command(
    *,
    archive_zip: str,
    archive_sha: str,
    mode: str,
    lane_id_base: str,
    submission_dir: str,
) -> list[str]:
    short_sha = _archive_sha_prefix(archive_sha, mode=mode)
    safe_mode = mode.replace("_", "-")
    return [
        ".venv/bin/python",
        "tools/dispatch_modal_paired_auth_eval.py",
        "--archive",
        archive_zip,
        "--expected-archive-sha256",
        archive_sha,
        "--submission-dir",
        submission_dir,
        "--inflate-sh",
        "inflate.sh",
        "--label",
        f"candidate4c_{safe_mode}",
        "--run-id",
        f"candidate4c_{safe_mode}_{short_sha}",
        "--pair-group-id",
        f"{EXACT_EVAL_PAIR_GROUP_ID}_{mode}_{short_sha}",
        "--lane-id-base",
        lane_id_base,
        "--output-root",
        "experiments/results",
        "--modal-bin",
        ".venv/bin/modal",
        "--gpu",
        "T4",
        "--claim-agent",
        "codex:candidate4c_exact_eval_handoff",
        "--claim-notes",
        (
            "Candidate 4c paired exact-eval handoff; "
            f"mode={mode}; archive_sha={archive_sha}; "
            f"full_vs_identity_group={EXACT_EVAL_PAIR_GROUP_ID}"
        ),
        "--expected-runtime-tree-sha256",
        "auto",
        "--skip-axis-if-promotable-anchor-exists",
    ]


def _archive_sha_prefix(archive_sha: str, *, mode: str) -> str:
    sha = str(archive_sha or "").strip()
    if len(sha) >= 12 and all(char in HEX_DIGITS for char in sha[:12]):
        return sha[:12].lower()
    return f"<{mode}_archive_sha256_prefix>"


def _paired_dispatch_command_pair(
    *,
    full_zip: str,
    full_sha: str,
    identity_zip: str,
    identity_sha: str,
    submission_dir: str,
    execute: bool,
) -> dict[str, str]:
    commands = {
        "full_paired_contest_cpu_cuda": _paired_dispatch_command(
            archive_zip=full_zip,
            archive_sha=full_sha,
            mode="full",
            lane_id_base=FULL_LANE_ID_BASE,
            submission_dir=submission_dir,
        ),
        "identity_paired_contest_cpu_cuda": _paired_dispatch_command(
            archive_zip=identity_zip,
            archive_sha=identity_sha,
            mode="identity",
            lane_id_base=IDENTITY_LANE_ID_BASE,
            submission_dir=submission_dir,
        ),
    }
    if execute:
        commands = {key: [*value, "--execute"] for key, value in commands.items()}
    return {key: shlex.join(value) for key, value in commands.items()}


def _modal_exact_eval_commands(
    *,
    full_zip: str,
    full_sha: str,
    identity_zip: str,
    identity_sha: str,
    submission_dir: str,
) -> dict[str, str]:
    """Return canonical paired dispatcher commands, not single-axis wrappers."""

    return _paired_dispatch_command_pair(
        full_zip=full_zip,
        full_sha=full_sha,
        identity_zip=identity_zip,
        identity_sha=identity_sha,
        submission_dir=submission_dir,
        execute=True,
    )


def _modal_exact_eval_command_templates() -> dict[str, str]:
    return _paired_dispatch_command_pair(
        full_zip="<harvested_full_600_pair_run_dir>/archive.zip",
        full_sha="<full_archive_zip_sha256>",
        identity_zip=(
            "<harvested_full_600_pair_run_dir>/"
            "archive_identity_predictor_disambiguator.zip"
        ),
        identity_sha="<identity_archive_zip_sha256>",
        submission_dir="<harvested_full_600_pair_run_dir>/submission_dir",
        execute=True,
    )


def _exact_eval_handoff_status(
    local_probe: Any,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Classify whether the latest archive pair can enter paired exact eval."""

    artifact = _latest_pair_probe_artifact(local_probe, repo_root=repo_root)
    payload = artifact.get("payload")
    blockers: list[str] = []
    source_rows: list[dict[str, Any]] = []
    if not isinstance(payload, Mapping):
        blockers.append("candidate4c_pair_probe_artifact_missing_or_unreadable")
        payload = {}
    raw_source_rows = payload.get("source_archives") or []
    if not isinstance(raw_source_rows, list):
        raw_source_rows = []
    for row in raw_source_rows:
        if isinstance(row, Mapping):
            source_rows.append(_archive_source_row_status(row, repo_root=repo_root))
    full_rows = [row for row in source_rows if row.get("identity_predictor") is False]
    identity_rows = [row for row in source_rows if row.get("identity_predictor") is True]
    if len(full_rows) != 1:
        blockers.append("candidate4c_exact_handoff_full_archive_row_missing")
    if len(identity_rows) != 1:
        blockers.append("candidate4c_exact_handoff_identity_archive_row_missing")
    for row in source_rows:
        blockers.extend(
            f"candidate4c_exact_handoff_{row.get('mode')}_{blocker}"
            for blocker in row.get("blockers", [])
        )
    max_pairs = max(
        (int(row.get("num_pairs") or 0) for row in source_rows),
        default=0,
    )
    if max_pairs != FULL_600_PAIR_COUNT:
        blockers.append(
            f"candidate4c_exact_handoff_latest_archive_pair_not_{FULL_600_PAIR_COUNT}_pairs"
        )
    submission_dir = None
    if full_rows:
        full_zip = _resolve_repo_path(repo_root, str(full_rows[0].get("zip_path") or ""))
        candidate = full_zip.parent / "submission_dir"
        if candidate.is_dir() and (candidate / "inflate.sh").is_file():
            submission_dir = (
                str(candidate.relative_to(repo_root))
                if candidate.is_relative_to(repo_root)
                else str(candidate)
            )
        else:
            blockers.append("candidate4c_exact_handoff_submission_dir_or_inflate_sh_missing")
    ready = not blockers
    commands: dict[str, str] = {}
    plan_commands: dict[str, str] = {}
    if ready and full_rows and identity_rows and submission_dir:
        commands = _modal_exact_eval_commands(
            full_zip=str(full_rows[0].get("zip_path")),
            full_sha=str(full_rows[0].get("zip_sha256") or ""),
            identity_zip=str(identity_rows[0].get("zip_path")),
            identity_sha=str(identity_rows[0].get("zip_sha256") or ""),
            submission_dir=submission_dir,
        )
        plan_commands = _paired_dispatch_command_pair(
            full_zip=str(full_rows[0].get("zip_path")),
            full_sha=str(full_rows[0].get("zip_sha256") or ""),
            identity_zip=str(identity_rows[0].get("zip_path")),
            identity_sha=str(identity_rows[0].get("zip_sha256") or ""),
            submission_dir=submission_dir,
            execute=False,
        )
    return {
        "schema": "candidate4c_exact_eval_handoff_v1",
        "ready_for_exact_eval_handoff": ready,
        "provider_dispatch_attempted": False,
        "lane_claim_opened": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pair_group_id": EXACT_EVAL_PAIR_GROUP_ID,
        "required_pair_count": FULL_600_PAIR_COUNT,
        "latest_pair_count": max_pairs or None,
        "latest_pair_probe_artifact": {
            "path": artifact.get("path"),
            "exists": artifact.get("exists"),
            "parse_error": artifact.get("parse_error"),
            "verdict": payload.get("verdict") if isinstance(payload, Mapping) else None,
            "evidence_grade": payload.get("evidence_grade")
            if isinstance(payload, Mapping)
            else None,
        },
        "source_archive_rows": source_rows,
        "required_files": [
            "archive.zip",
            "archive_identity_predictor_disambiguator.zip",
            "submission_dir/inflate.sh",
            "stats.json",
            "manifest.json",
            "provenance.json",
        ],
        "axis_plan": [
            {"axis": "contest-CUDA", "mode": "full", "lane_id": FULL_CUDA_LANE_ID},
            {
                "axis": "contest-CUDA",
                "mode": "identity",
                "lane_id": IDENTITY_CUDA_LANE_ID,
            },
            {"axis": "contest-CPU", "mode": "full", "lane_id": FULL_CPU_LANE_ID},
            {
                "axis": "contest-CPU",
                "mode": "identity",
                "lane_id": IDENTITY_CPU_LANE_ID,
            },
        ],
        "paired_dispatch_plan": [
            {
                "mode": "full",
                "lane_id_base": FULL_LANE_ID_BASE,
                "archive_axis_pair": ["contest-CUDA", "contest-CPU"],
            },
            {
                "mode": "identity",
                "lane_id_base": IDENTITY_LANE_ID_BASE,
                "archive_axis_pair": ["contest-CUDA", "contest-CPU"],
            },
        ],
        "modal_commands_after_full_600_pair_packet": commands,
        "modal_plan_commands_after_full_600_pair_packet": plan_commands,
        "modal_command_templates_after_full_600_pair_packet": (
            _modal_exact_eval_command_templates()
        ),
        "blockers": [] if ready else blockers,
    }


def _queue_immediate_launch_status(
    queue: Mapping[str, Any],
    sequence_row: Mapping[str, Any],
    *,
    top_ready: Any,
    command: Any,
) -> tuple[bool, list[str]]:
    """Require the queue's richer immediate-launch gate, not only a command."""

    blockers: list[str] = []
    if top_ready != SUBSTRATE_ID:
        blockers.append("candidate4c_queue_top_ready_substrate_mismatch")
    if not isinstance(command, str) or not command.strip():
        blockers.append("candidate4c_queue_audit_backed_paid_launch_command_missing")
    if not sequence_row:
        blockers.append("candidate4c_queue_sequence_row_missing")
        return False, blockers

    if sequence_row.get("ready_for_paid_dispatch") is not True:
        blockers.append("candidate4c_queue_row_not_ready_for_paid_dispatch")
    if sequence_row.get("immediately_runnable_paid_launch") is not True:
        blockers.append("candidate4c_queue_row_not_immediately_runnable")

    top_missing = queue.get("top_ready_paid_launch_missing_preconditions")
    if top_missing not in ([], ()):
        blockers.append("candidate4c_queue_top_ready_paid_launch_preconditions_missing")

    row_missing = sequence_row.get("paid_launch_missing_preconditions")
    if row_missing not in ([], ()):
        blockers.append("candidate4c_queue_row_paid_launch_preconditions_missing")

    immediate_count = queue.get("immediately_runnable_paid_dispatch_count")
    if not isinstance(immediate_count, int) or immediate_count <= 0:
        blockers.append("candidate4c_queue_no_immediately_runnable_paid_dispatch")

    if not queue.get("top_immediately_runnable_paid_launch_command"):
        blockers.append("candidate4c_queue_top_immediately_runnable_command_missing")

    catalog202 = (
        sequence_row.get("operator_session_authorization", {})
        .get("catalog202_dirty_tree_attestation", {})
    )
    if isinstance(catalog202, Mapping):
        if (
            catalog202.get("required_for_paid_dispatch")
            and catalog202.get("satisfied_in_current_environment") is not True
        ):
            blockers.append("candidate4c_queue_catalog202_env_attestation_not_satisfied")
        if (
            catalog202.get("dirty_sentinel_audit_required")
            and catalog202.get("env_sentinel_audit_matches_current") is not True
        ):
            blockers.append("candidate4c_queue_catalog202_env_sentinel_audit_not_current")
        if (
            catalog202.get("required_for_paid_dispatch")
            and catalog202.get("current_sentinel_snapshot_valid") is not True
        ):
            blockers.append(
                "candidate4c_queue_catalog202_current_sentinel_snapshot_not_valid"
            )
    else:
        blockers.append("candidate4c_queue_catalog202_attestation_missing")

    return not blockers, blockers


def _codex_pre_dispatch_review_status(
    *,
    repo_root: Path,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Require the latest codex pre-dispatch review to be non-blocking."""

    path = _latest_codex_review_path(repo_root)
    if path is None:
        return (
            False,
            ["candidate4c_codex_pre_dispatch_review_missing"],
            {"path": None, "verdict": None, "findings": []},
        )
    rel_path = (
        str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (
            False,
            ["candidate4c_codex_pre_dispatch_review_unreadable"],
            {"path": rel_path, "verdict": None, "findings": []},
        )
    verdict = str(payload.get("verdict") or "")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        findings = []
    if not findings and payload.get("raw_output_excerpt"):
        try:
            from run_codex_review_for_dispatch import parse_verdict_from_codex_output

            _parsed_verdict, parsed_findings = parse_verdict_from_codex_output(
                str(payload.get("raw_output_excerpt") or "")
            )
            findings = parsed_findings
        except Exception:
            findings = []
    status = {
        "path": rel_path,
        "verdict": verdict,
        "findings": findings,
        "cache_hit": payload.get("cache_hit"),
        "cache_key": payload.get("cache_key"),
        "invoked_at_utc": payload.get("invoked_at_utc"),
        "elapsed_sec": payload.get("elapsed_sec"),
    }
    if verdict not in {"approve", "advisory"}:
        return (
            False,
            [f"candidate4c_codex_pre_dispatch_review_blocking_{verdict or 'missing'}"],
            status,
        )
    return True, [], status


def build_packet(
    *,
    repo_root: Path = REPO_ROOT,
    queue_path: Path | None = None,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    queue_path = queue_path or _latest_queue_path(repo_root)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    recipe_status = _load_recipe_status(repo_root)
    recipe_dispatch_enabled = recipe_status.get("dispatch_enabled") is True
    top_ready = queue.get("top_ready_substrate")
    command = queue.get("top_ready_audit_backed_paid_launch_command")
    sequence_row = next(
        (
            row
            for row in queue.get("dispatch_sequence", [])
            if row.get("substrate_id") == SUBSTRATE_ID
        ),
        {},
    )
    catalog202 = (
        sequence_row.get("operator_session_authorization", {})
        .get("catalog202_dirty_tree_attestation", {})
    )
    audit_ref = catalog202.get("latest_sentinel_audit") or {}
    audit_rel = audit_ref.get("path")
    audit_path = _resolve_repo_path(repo_root, audit_rel) if audit_rel else None
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path and audit_path.is_file()
        else {}
    )
    local_probe = sequence_row.get("local_identity_disambiguator_probe")
    local_probe_ready, local_probe_blockers = (
        _local_identity_disambiguator_probe_status(local_probe, repo_root=repo_root)
    )
    exact_eval_handoff = _exact_eval_handoff_status(
        local_probe,
        repo_root=repo_root,
    )
    sentinel_hash = str(audit.get("sentinel_set_sha256") or "")
    if recipe_dispatch_enabled:
        command_ready, command_blockers = _paid_command_status(
            command,
            audit_rel=audit_rel,
            sentinel_hash=sentinel_hash,
        )
        queue_immediate_ready, queue_immediate_blockers = _queue_immediate_launch_status(
            queue,
            sequence_row,
            top_ready=top_ready,
            command=command,
        )
    else:
        command_ready = False
        command_blockers = [
            "candidate4c_paid_training_launch_not_in_scope_recipe_dispatch_disabled"
        ]
        queue_immediate_ready = False
        queue_immediate_blockers = [
            "candidate4c_recipe_dispatch_disabled_exact_eval_handoff_required"
        ]
    codex_review_ready, codex_review_blockers, codex_review_status = (
        _codex_pre_dispatch_review_status(repo_root=repo_root)
    )
    env = {
        INTENT_ENV: "1",
        ATTESTATION_ENV: f"catalog202_sentinel_audit:{sentinel_hash}",
        AUDIT_JSON_ENV: str(audit_path.relative_to(repo_root))
        if audit_path and audit_path.is_relative_to(repo_root)
        else str(audit_path or ""),
        SESSION_DIRECTIVE_ENV: "1",
        SESSION_BUDGET_ENV: SESSION_BUDGET_USD,
    }
    checks = [
        _run(
            "lane_claim_summary",
            [sys.executable, "tools/claim_lane_dispatch.py", "summary"],
            repo_root=repo_root,
            runner=runner,
        ),
        _run(
            "required_input_validation",
            [
                sys.executable,
                "tools/validate_dispatch_required_inputs.py",
                "--trainer",
                TRAINER,
                f"--flag-value=--video-path={VIDEO_PATH}",
            ],
            repo_root=repo_root,
            runner=runner,
        ),
        _run(
            "smoke_before_full_dry_run",
            [
                sys.executable,
                "tools/run_modal_smoke_before_full.py",
                "--recipe",
                RECIPE,
                "--operator-handle",
                OPERATOR_HANDLE,
                "--dry-run",
            ],
            repo_root=repo_root,
            runner=runner,
            env=env,
        ),
        _run(
            "operator_authorize_dry_run",
            [
                sys.executable,
                "tools/operator_authorize.py",
                "--recipe",
                RECIPE,
                "--dry-run",
            ],
            repo_root=repo_root,
            runner=runner,
            env=env,
        ),
    ]
    if recipe_dispatch_enabled:
        checks.append(
            _run(
                "catalog202_audit_backed_bypass_probe",
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; sys.path.insert(0, 'tools'); "
                        "import operator_authorize as oa; "
                        f"recipe = oa._load_recipe({RECIPE!r}); "
                        "print(oa._whole_tree_clean_check_bypass_active(recipe))"
                    ),
                ],
                repo_root=repo_root,
                runner=runner,
                env=env,
            )
        )
    else:
        checks.append(
            _skipped_check(
                "catalog202_audit_backed_bypass_probe",
                "recipe dispatch_enabled=false; no paid training command is in scope",
            )
        )
    claim_check = next(c for c in checks if c["name"] == "lane_claim_summary")
    active_claims_clean = _claim_summary_active_clean(claim_check["stdout"])
    bypass_check = next(
        c for c in checks if c["name"] == "catalog202_audit_backed_bypass_probe"
    )
    bypass_probe_accepted = bool(
        bypass_check.get("skipped")
        or _catalog202_bypass_probe_accepted(bypass_check["stdout"])
    )
    bypass_check["ok"] = bool(bypass_check["ok"] and bypass_probe_accepted)
    audit_ready = bool(
        audit.get("ready_for_catalog202_audit_backed_dirty_sentinel_attestation")
    )
    checks_ok = all(c["ok"] for c in checks)
    ready = bool(
        recipe_dispatch_enabled
        and queue_immediate_ready
        and audit_ready
        and active_claims_clean
        and checks_ok
        and local_probe_ready
        and command_ready
        and codex_review_ready
    )
    result_review_blockers = [
        *queue_immediate_blockers,
        *local_probe_blockers,
        *command_blockers,
        *codex_review_blockers,
    ]
    if not recipe_dispatch_enabled:
        result_review_blockers.extend(exact_eval_handoff.get("blockers", []))
    if not bypass_probe_accepted:
        result_review_blockers.append(
            "candidate4c_catalog202_audit_backed_bypass_probe_not_true"
        )
    if not ready and not result_review_blockers:
        result_review_blockers.append(
            "candidate4c_no_spend_launch_packet_checks_not_all_green"
        )
    return {
        "schema": "candidate4c_no_spend_launch_packet_v1",
        "generated_utc": _utc_now(),
        "substrate_id": SUBSTRATE_ID,
        "recipe": RECIPE,
        "recipe_status": recipe_status,
        "paid_training_launch_in_scope": recipe_dispatch_enabled,
        "current_mode": recipe_status.get("current_mode"),
        "queue_path": str(queue_path.relative_to(repo_root)),
        "audit_path": str(audit_path.relative_to(repo_root))
        if audit_path and audit_path.is_relative_to(repo_root)
        else str(audit_path),
        "queue_top_ready_substrate": top_ready,
        "queue_current_worktree_dirty_path_count": queue.get(
            "current_worktree_dirty_path_count"
        ),
        "queue_immediately_runnable_paid_dispatch_count": queue.get(
            "immediately_runnable_paid_dispatch_count"
        ),
        "queue_immediate_launch_ready": queue_immediate_ready,
        "queue_immediate_launch_blockers": queue_immediate_blockers,
        "queue_top_ready_paid_launch_missing_preconditions": queue.get(
            "top_ready_paid_launch_missing_preconditions"
        ),
        "local_identity_disambiguator_probe": local_probe,
        "local_identity_disambiguator_probe_ready": local_probe_ready,
        "local_identity_disambiguator_probe_blockers": local_probe_blockers,
        "next_paid_command_ready": command_ready,
        "next_paid_command_blockers": command_blockers,
        "codex_pre_dispatch_review_ready": codex_review_ready,
        "codex_pre_dispatch_review_blockers": codex_review_blockers,
        "codex_pre_dispatch_review": codex_review_status,
        "audit_sentinel_set_sha256": sentinel_hash,
        "audit_ready_for_dirty_sentinel_attestation": audit_ready,
        "audit_dirty_sentinel_paths": audit.get("dirty_sentinel_paths") or [],
        "catalog202_audit_backed_bypass_probe_accepted": bypass_probe_accepted,
        "active_lane_claims_clean": active_claims_clean,
        "checks": checks,
        "checks_ok": checks_ok,
        "diagnostic_smoke_dry_run_ready": next(
            c for c in checks if c["name"] == "smoke_before_full_dry_run"
        )["ok"],
        "operator_authorize_dry_run_returned": next(
            c for c in checks if c["name"] == "operator_authorize_dry_run"
        )["returncode"],
        "exact_eval_handoff": exact_eval_handoff,
        "ready_for_operator_paid_execution": ready,
        "operator_action_required": True,
        "next_paid_command": command,
        "provider_dispatch_attempted": False,
        "lane_claim_opened": False,
        "score_claim": False,
        "promotion_eligible": False,
        "result_review_blockers": [] if ready else result_review_blockers,
    }


def write_artifact(payload: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> Path:
    out_dir = repo_root / ".omx" / "state" / "candidate4c_launch_packet"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(payload["generated_utc"]).replace(":", "").replace("-", "")
    path = out_dir / f"candidate4c_no_spend_launch_packet_{stamp}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--queue-path", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-artifact", action="store_true")
    args = parser.parse_args(argv)

    queue_path = args.queue_path
    if queue_path is not None and not queue_path.is_absolute():
        queue_path = args.repo_root / queue_path
    payload = build_packet(repo_root=args.repo_root, queue_path=queue_path)
    if args.write_artifact:
        path = write_artifact(payload, repo_root=args.repo_root)
        print(f"[candidate4c-launch-packet] wrote artifact: {path}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("=== Candidate 4c no-spend launch packet ===")
        print(f"ready_for_operator_paid_execution: {payload['ready_for_operator_paid_execution']}")
        print(f"active_lane_claims_clean: {payload['active_lane_claims_clean']}")
        print(f"checks_ok: {payload['checks_ok']}")
        for check in payload["checks"]:
            print(f"  {check['name']}: rc={check['returncode']}")
    return 0 if payload["ready_for_operator_paid_execution"] else 1


if __name__ == "__main__":
    sys.exit(main())
