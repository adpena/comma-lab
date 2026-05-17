# SPDX-License-Identifier: MIT
"""Generated TT5L Lightning route-unblock packet.

This packet sits above the paired-axis plan, execution bundle, dry-run
verification, harvest cells, and effect curve. Its job is to preserve the
current route blocker as executable operator evidence without turning
dry-run/parser custody into score or dispatch authority. Architecture lock is a
downstream refresh target, not an input artifact, because hashing it here creates
a circular custody dependency with the architecture-lock packet's route status.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH,
    T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run import (
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_ARTIFACT_PATH,
)

L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_SCHEMA = (
    "l5_v2_tt5l_lightning_route_unblock_packet_v1"
)
L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_TOOL_PATH = (
    "tools/build_l5_v2_tt5l_lightning_route_unblock_packet.py"
)
L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_lightning_route_unblock_packet_20260517_codex.json"
)
L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_REPORT_PATH = (
    ".omx/research/l5_v2_tt5l_lightning_route_unblock_packet_20260517_codex.md"
)

L5V2_PROVIDER_READINESS_REFRESH_ARTIFACT_PATH = (
    ".omx/research/l5_v2_provider_readiness_refresh_20260517_codex.json"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_lightning_execution_preflight_20260517_codex.json"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json"
)
L5V2_TT5L_LIGHTNING_ALT_PROVIDER_PLAN_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_lightning_alt_provider_plan_20260517_codex.json"
)

_FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_provider_dispatch": False,
    "dispatch_attempted": False,
    "provider_spend_attempted": False,
}
_AUTHORITY_FLAGS = {
    "planning_only": True,
    **_FALSE_AUTHORITY_FLAGS,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _load_mapping(path: Path) -> tuple[Mapping[str, Any], list[str]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, ["artifact_missing"]
    except (OSError, json.JSONDecodeError):
        return {}, ["artifact_json_invalid"]
    if not isinstance(loaded, Mapping):
        return {}, ["artifact_not_object"]
    return loaded, []


def _artifact_record(
    *,
    repo_root: Path,
    path: str,
    expected_schema: str = "",
    required: bool = True,
) -> tuple[dict[str, Any], Mapping[str, Any], list[str]]:
    artifact_path = _resolve_repo_path(path, repo_root)
    payload, blockers = _load_mapping(artifact_path)
    exists = artifact_path.is_file()
    record: dict[str, Any] = {
        "path": _repo_relative(artifact_path, repo_root),
        "exists": exists,
        "sha256": _sha256_file(artifact_path) if exists else "",
        "schema": payload.get("schema", "") if payload else "",
    }
    if expected_schema:
        record["expected_schema"] = expected_schema
        if payload and payload.get("schema") != expected_schema:
            blockers.append("artifact_schema_mismatch")
    if not required:
        blockers = [blocker for blocker in blockers if blocker != "artifact_missing"]
    record["blockers"] = blockers
    return record, payload, blockers


def _provider_by_name(
    payload: Mapping[str, Any],
    provider_name: str,
) -> Mapping[str, Any]:
    for row in _mapping_rows(payload.get("providers")):
        if row.get("provider") == provider_name:
            return row
    return {}


def _human_lightning_blockers(lightning_provider: Mapping[str, Any]) -> list[str]:
    raw = {str(blocker) for blocker in lightning_provider.get("blockers", [])}
    blocker_map = {
        "lightning_teamspace_missing": "LIGHTNING_TEAMSPACE missing",
        "lightning_owner_missing": "LIGHTNING_SDK_USER or LIGHTNING_ORG missing",
        "lightning_ssh_target_missing": "LIGHTNING_SSH_TARGET missing",
        "credits_or_quota_not_checked": "Lightning credits or quota not checked",
        "no_dispatch_claim": "active dispatch claims not created for non-dry-run cells",
    }
    out = [blocker_map.get(blocker, blocker) for blocker in sorted(raw)]
    for blocker in (
        "Lightning machine inventory not checked",
        "source manifest not staged to remote Lightning workspace",
        "remote CUDA runtime not probed",
    ):
        out.append(blocker)
    return _dedupe(out)


def _all_cells_have_stdout_state_match(dry_run: Mapping[str, Any]) -> bool:
    rows = _mapping_rows(dry_run.get("cells"))
    return bool(rows) and all(
        isinstance(row.get("dry_run_state_file"), Mapping)
        and row["dry_run_state_file"].get("stdout_core_matched") is True
        for row in rows
    )


def _all_cells_have_executable_inflate(dry_run: Mapping[str, Any]) -> bool:
    rows = _mapping_rows(dry_run.get("cells"))
    return bool(rows) and all(
        isinstance(row.get("inflate_runtime"), Mapping)
        and row["inflate_runtime"].get("exists") is True
        and row["inflate_runtime"].get("executable") is True
        for row in rows
    )


def _bundle_embeds_t4_runtime_env(bundle: Mapping[str, Any]) -> bool:
    for cell in _mapping_rows(bundle.get("cells")):
        dry_run_command = str(cell.get("dry_run_submit_command") or "")
        non_dry_run_command = str(cell.get("non_dry_run_submit_command_template") or "")
        if not all(item in dry_run_command for item in T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV):
            return False
        if not all(item in non_dry_run_command for item in T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV):
            return False
    return bool(_mapping_rows(bundle.get("cells")))


def _source_relevant_paths_match(
    *,
    source_commit: str,
    current_head_commit: str,
    source_relevant_diff_paths: Iterable[str],
) -> bool:
    if not source_commit:
        return False
    if list(source_relevant_diff_paths):
        return False
    return True


def build_l5_v2_tt5l_lightning_route_unblock_packet(
    *,
    repo_root: str | Path,
    current_head_commit: str = "",
    source_relevant_diff_paths: Iterable[str] = (),
    generated_at_utc: str | None = None,
    provider_readiness_path: str = L5V2_PROVIDER_READINESS_REFRESH_ARTIFACT_PATH,
    execution_preflight_path: str = L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_ARTIFACT_PATH,
    execution_bundle_path: str = L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH,
    dry_run_verification_path: str = L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_ARTIFACT_PATH,
    paired_axis_plan_path: str = L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH,
    harvest_cells_path: str = L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH,
    sideinfo_effect_curve_path: str = L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH,
    legacy_alt_provider_plan_path: str = L5V2_TT5L_LIGHTNING_ALT_PROVIDER_PLAN_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Return a generated false-authority-safe TT5L Lightning route packet."""

    root = Path(repo_root).resolve()
    generated = generated_at_utc or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    artifacts: dict[str, dict[str, Any]] = {}
    payloads: dict[str, Mapping[str, Any]] = {}
    all_blockers: list[str] = []

    artifact_specs = {
        "provider_readiness": (provider_readiness_path, "cloud_provider_readiness_v1", True),
        "sideinfo_execution_preflight": (
            execution_preflight_path,
            "l5_v2_tt5l_sideinfo_lightning_execution_preflight_v1",
            True,
        ),
        "sideinfo_execution_bundle": (
            execution_bundle_path,
            "l5_v2_tt5l_sideinfo_lightning_execution_bundle_v1",
            True,
        ),
        "sideinfo_execution_bundle_dry_run_verification": (
            dry_run_verification_path,
            "l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification_v1",
            True,
        ),
        "sideinfo_lightning_paired_axis_plan": (
            paired_axis_plan_path,
            "l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_v1",
            True,
        ),
        "sideinfo_harvest_cells": (
            harvest_cells_path,
            "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_v1",
            True,
        ),
        "sideinfo_effect_curve": (
            sideinfo_effect_curve_path,
            "l5_v2_sideinfo_effect_curve_v1",
            True,
        ),
        "legacy_cuda_only_alt_provider_plan": (
            legacy_alt_provider_plan_path,
            "",
            False,
        ),
    }
    for name, (path, schema, required) in artifact_specs.items():
        record, payload, blockers = _artifact_record(
            repo_root=root,
            path=path,
            expected_schema=schema,
            required=required,
        )
        artifacts[name] = record
        payloads[name] = payload
        all_blockers.extend(f"{name}:{blocker}" for blocker in blockers)

    provider = payloads["provider_readiness"]
    lightning_provider = _provider_by_name(provider, "lightning")
    bundle = payloads["sideinfo_execution_bundle"]
    dry_run = payloads["sideinfo_execution_bundle_dry_run_verification"]
    plan = payloads["sideinfo_lightning_paired_axis_plan"]
    harvest = payloads["sideinfo_harvest_cells"]
    effect_curve = payloads["sideinfo_effect_curve"]

    source_commit = str(plan.get("source_commit") or "")
    source_relevant_diff_path_list = [str(path) for path in source_relevant_diff_paths]
    source_paths_match = _source_relevant_paths_match(
        source_commit=source_commit,
        current_head_commit=current_head_commit,
        source_relevant_diff_paths=source_relevant_diff_path_list,
    )

    artifacts["sideinfo_execution_bundle_dry_run_verification"].update(
        {
            "all_dry_runs_passed": dry_run.get("all_dry_runs_passed") is True,
            "passed_cell_count": dry_run.get("passed_cell_count", 0),
            "cell_count": dry_run.get("cell_count", 0),
        }
    )
    artifacts["sideinfo_lightning_paired_axis_plan"].update(
        {
            "source_commit": source_commit,
            "source_commit_matches_current_head": source_commit == current_head_commit,
            "source_relevant_paths_match_current_head": source_paths_match,
            "source_relevant_diff_paths": source_relevant_diff_path_list,
        }
    )
    artifacts["sideinfo_harvest_cells"].update(
        {
            "harvested_exact_eval_artifact_count": harvest.get(
                "harvested_exact_eval_artifact_count", 0
            ),
            "missing_exact_eval_artifact_count": harvest.get(
                "missing_exact_eval_artifact_count", 0
            ),
        }
    )
    artifacts["sideinfo_effect_curve"].update(
        {"predicate_passed": effect_curve.get("predicate_passed") is True}
    )
    artifacts["legacy_cuda_only_alt_provider_plan"].update(
        {
            "role": (
                "historical single-cell provider-blocker evidence; superseded "
                "for sideinfo effect-curve command source by the 10-cell bundle"
            )
        }
    )

    if not lightning_provider:
        all_blockers.append("provider_readiness:lightning_provider_row_missing")
    if source_commit and not source_paths_match:
        all_blockers.append("paired_axis_plan:source_relevant_paths_changed")
    if dry_run.get("all_dry_runs_passed") is not True:
        all_blockers.append("dry_run_verification:not_all_dry_runs_passed")
    if not _all_cells_have_stdout_state_match(dry_run):
        all_blockers.append("dry_run_verification:stdout_state_core_not_all_matched")
    if not _all_cells_have_executable_inflate(dry_run):
        all_blockers.append("dry_run_verification:inflate_runtime_not_all_executable")
    if not _bundle_embeds_t4_runtime_env(bundle):
        all_blockers.append("execution_bundle:t4_runtime_env_not_embedded_for_all_cells")

    remaining_blockers = _human_lightning_blockers(lightning_provider)

    return {
        **_AUTHORITY_FLAGS,
        "schema": L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_SCHEMA,
        "tool": L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_TOOL_PATH,
        "generated_at_utc": generated,
        "current_head_commit": current_head_commit,
        "purpose": (
            "Turn the L5 v2 TT5L Lightning alternate-provider blocker into an "
            "executable, false-authority-safe operator unblock checklist."
        ),
        "source_artifacts": artifacts,
        "current_route_verdict": {
            "provider": "lightning",
            "status": "blocked_on_operator_route_configuration",
            "method_blocker": False,
            "provider_blocker": True,
            "execution_ready": False,
            "ready_for_operator_dispatch": False,
            **_FALSE_AUTHORITY_FLAGS,
        },
        "remaining_blockers": remaining_blockers,
        "verified_before_this_packet": {
            "local_lightning_sdk_version": str(
                lightning_provider.get("stdout_excerpt") or ""
            ).strip(),
            "provider_readiness_probe_completed": bool(lightning_provider),
            "paired_axis_plan_source_relevant_paths_match_current_head": source_paths_match,
            "bundle_all_10_dry_run_commands_parse": dry_run.get("all_dry_runs_passed")
            is True,
            "bundle_all_10_dry_run_state_files_match_stdout_core": (
                _all_cells_have_stdout_state_match(dry_run)
            ),
            "bundle_all_10_inflate_runtimes_exist_and_are_executable": (
                _all_cells_have_executable_inflate(dry_run)
            ),
            "bundle_axes": list(bundle.get("required_axes") or plan.get("required_axes") or []),
            "bundle_t4_runtime_env_pins_embedded": _bundle_embeds_t4_runtime_env(bundle),
            "bundle_t4_runtime_env_pins": list(T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV),
            "bundle_variants": list(
                bundle.get("required_variants") or plan.get("required_variants") or []
            ),
            "dry_run_scope": (
                "local parser and queue-spec custody only; no provider job, no score"
            ),
        },
        "operator_supplied_fields_required": {
            "LIGHTNING_TEAMSPACE": "<lightning-teamspace>",
            "LIGHTNING_SDK_USER_or_LIGHTNING_ORG": "<lightning-user-or-org>",
            "LIGHTNING_SSH_TARGET": "<lightning-ssh-target>",
            "LIGHTNING_STUDIO": "<lightning-studio>",
            "identity_mode": (
                "choose exactly one of --user <lightning-user> or "
                "--org <lightning-org>"
            ),
        },
        "pre_dispatch_command_order": [
            {
                "step": 1,
                "name": "configure_lightning_route",
                "command_template": (
                    "export LIGHTNING_TEAMSPACE='<lightning-teamspace>'; "
                    "export LIGHTNING_SSH_TARGET='<lightning-ssh-target>'; "
                    "export LIGHTNING_SDK_USER='<lightning-user>'  # or export "
                    "LIGHTNING_ORG='<lightning-org>'"
                ),
            },
            {
                "step": 2,
                "name": "required_doctor",
                "command_template": (
                    ".venv/bin/python scripts/launch_lightning_batch_job.py doctor "
                    "--json-out .omx/research/"
                    "l5_v2_tt5l_lightning_required_doctor_20260517_codex.json "
                    "--run-id l5_v2_tt5l_lightning_required_doctor_20260517 "
                    "--strict --ssh-target \"$LIGHTNING_SSH_TARGET\" --require-ssh "
                    "--remote-supply-chain --require-remote-supply-chain "
                    "--repo-dir /teamspace/studios/this_studio/pact "
                    "--python-bin .venv/bin/python "
                    "--teamspace \"$LIGHTNING_TEAMSPACE\" "
                    "<--user-or---org> '<lightning-user-or-org>' "
                    "--machine-inventory --require-machine-inventory "
                    "--machine T4 --gpu-only"
                ),
            },
            {
                "step": 3,
                "name": "restage_source_manifest_per_cell",
                "command_source": (
                    f"{execution_bundle_path}:"
                    "cells[*].stage_source_manifest_command_template"
                ),
                "required_replacements": {"<lightning-ssh-target>": "$LIGHTNING_SSH_TARGET"},
            },
            {
                "step": 4,
                "name": "claim_each_lane_before_non_dry_run",
                "command_source": f"{execution_bundle_path}:cells[*].claim_command",
            },
            {
                "step": 5,
                "name": "submit_non_dry_run_after_placeholders_removed",
                "command_source": (
                    f"{execution_bundle_path}:"
                    "cells[*].non_dry_run_submit_command_template"
                ),
                "required_replacements": {
                    "<lightning-studio>": "$LIGHTNING_STUDIO",
                    "<lightning-teamspace>": "$LIGHTNING_TEAMSPACE",
                    "<--user-or---org>": "--user or --org",
                    "<lightning-user-or-org>": "$LIGHTNING_SDK_USER or $LIGHTNING_ORG",
                    "<lightning-ssh-target>": "$LIGHTNING_SSH_TARGET",
                },
            },
            {
                "step": 6,
                "name": "harvest_and_close_claims",
                "command_source": (
                    f"{execution_bundle_path}:cells[*].harvest_probe_command_template "
                    "plus terminal_success_claim_template or terminal_failure_claim_template"
                ),
            },
            {
                "step": 7,
                "name": "refresh_effect_curve_and_architecture_packet",
                "command_template": (
                    ".venv/bin/python tools/"
                    "build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py "
                    f"--lightning-plan-json {paired_axis_plan_path} "
                    f"--output-json {harvest_cells_path} --repo-root . && "
                    ".venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py "
                    f"--cell-json {harvest_cells_path} "
                    f"--output-json {sideinfo_effect_curve_path} --repo-root . && "
                    ".venv/bin/python tools/build_l5_v2_architecture_lock_packet.py "
                    "--repo-root ."
                ),
            },
        ],
        "authority": dict(_AUTHORITY_FLAGS),
        "blockers": _dedupe(all_blockers),
        "no_signal_loss_notes": [
            "This packet is generated from live artifact SHA-256s, not hand-edited.",
            "Architecture lock is deliberately downstream of this packet; do not hash it here.",
            "The old single-cell Lightning alternate-provider plan remains preserved as historical provider-blocker evidence.",
            "The 10-cell bundle and dry-run verifier are the current sideinfo effect-curve command authority.",
            "No CPU/CUDA axis may be promoted from this packet; harvested contest_auth_eval artifacts are still required.",
        ],
    }


def l5_v2_tt5l_lightning_route_unblock_packet_json(
    payload: Mapping[str, Any],
) -> str:
    """Return canonical JSON text for the route-unblock packet."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_l5_v2_tt5l_lightning_route_unblock_packet_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing route-unblock packet."""

    artifacts = (
        payload.get("source_artifacts")
        if isinstance(payload.get("source_artifacts"), Mapping)
        else {}
    )
    verified = (
        payload.get("verified_before_this_packet")
        if isinstance(payload.get("verified_before_this_packet"), Mapping)
        else {}
    )
    axis_labels = {
        "contest_cpu": "[contest-CPU]",
        "contest_cuda": "[contest-CUDA]",
    }
    bundle_axis_labels = [
        axis_labels.get(str(axis), str(axis))
        for axis in verified.get("bundle_axes", [])
    ]
    lines = [
        "# L5 v2 TT5L Lightning route unblock packet",
        "",
        f"**Generated:** {payload.get('generated_at_utc')}",
        f"**Commit:** `{payload.get('current_head_commit')}`",
        "",
        "This packet is generated from live artifact hashes. It turns the "
        "current TT5L Lightning blocker into an executable operator checklist. "
        "It is not a dispatch, score claim, or promotion artifact.",
        "",
        "## Verdict",
        "",
        "The TT5L method path is not blocked here. The route is blocked on "
        "provider configuration:",
        "",
    ]
    for blocker in payload.get("remaining_blockers", []):
        lines.append(f"- {blocker}")
    lines.extend(["", "## Current Evidence", ""])
    evidence_order = [
        ("provider_readiness", "Provider readiness refresh"),
        ("sideinfo_execution_preflight", "10-cell sideinfo execution preflight"),
        ("sideinfo_execution_bundle", "10-cell sideinfo execution bundle"),
        (
            "sideinfo_execution_bundle_dry_run_verification",
            "10-cell dry-run verification",
        ),
        ("sideinfo_lightning_paired_axis_plan", "10-cell paired-axis plan"),
        ("sideinfo_harvest_cells", "Sideinfo harvest cells"),
        ("sideinfo_effect_curve", "Sideinfo effect curve"),
    ]
    for key, label in evidence_order:
        record = artifacts.get(key) if isinstance(artifacts.get(key), Mapping) else {}
        lines.append(f"- {label}: `{record.get('path', '')}`")
        lines.append(f"  - exists: `{record.get('exists')}`")
        lines.append(f"  - SHA-256: `{record.get('sha256', '')}`")
        if key == "sideinfo_execution_bundle_dry_run_verification":
            lines.append(
                "  - all dry-runs passed: "
                f"`{record.get('all_dry_runs_passed')}`"
            )
            lines.append(
                "  - cells passed: "
                f"`{record.get('passed_cell_count')}`/`{record.get('cell_count')}`"
            )
        if key == "sideinfo_lightning_paired_axis_plan":
            lines.append(f"  - source commit: `{record.get('source_commit', '')}`")
            lines.append(
                "  - source-relevant paths match current HEAD: "
                f"`{record.get('source_relevant_paths_match_current_head')}`"
            )
        if key == "sideinfo_harvest_cells":
            lines.append(
                "  - harvested exact-eval artifacts: "
                f"`{record.get('harvested_exact_eval_artifact_count')}`"
            )
            lines.append(
                "  - missing exact-eval artifacts: "
                f"`{record.get('missing_exact_eval_artifact_count')}`"
            )
        if key == "sideinfo_effect_curve":
            lines.append(f"  - predicate passed: `{record.get('predicate_passed')}`")
    lines.extend(
        [
            "",
            "The refreshed bundle embeds the T4/g4dn exact-eval runtime pins "
            "required by `scripts/launch_lightning_batch_job.py`: "
            + ", ".join(f"`{item}`" for item in T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV)
            + ".",
            "",
            "Dry-run scope: "
            f"`{verified.get('dry_run_scope', '')}`.",
            "",
            "Bundle axes preserved: "
            + ", ".join(f"`{axis}`" for axis in bundle_axis_labels),
            "",
            "## Command Order",
            "",
        ]
    )
    for step in payload.get("pre_dispatch_command_order", []):
        if not isinstance(step, Mapping):
            continue
        lines.append(f"{step.get('step')}. {step.get('name')}")
        if step.get("command_template"):
            lines.extend(["", "```bash", str(step["command_template"]), "```", ""])
        elif step.get("command_source"):
            lines.append(f"   - command source: `{step.get('command_source')}`")
    lines.extend(
        [
            "## Authority",
            "",
            "- `planning_only=true`",
            "- `score_claim=false`",
            "- `promotion_eligible=false`",
            "- `rank_or_kill_eligible=false`",
            "- `ready_for_exact_eval_dispatch=false`",
            "- `ready_for_provider_dispatch=false`",
            "- `dispatch_attempted=false`",
            "- `provider_spend_attempted=false`",
            "",
            "No CPU/CUDA axis may be promoted from this packet. Promotion "
            "requires harvested contest-auth-eval artifacts with custody, "
            "adjudication, and terminal lane claims.",
            "",
            f"Blockers: `{payload.get('blockers', [])}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_ARTIFACT_PATH",
    "L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_REPORT_PATH",
    "L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_SCHEMA",
    "L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_TOOL_PATH",
    "build_l5_v2_tt5l_lightning_route_unblock_packet",
    "l5_v2_tt5l_lightning_route_unblock_packet_json",
    "render_l5_v2_tt5l_lightning_route_unblock_packet_markdown",
]
