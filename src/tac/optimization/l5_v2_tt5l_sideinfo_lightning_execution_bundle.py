# SPDX-License-Identifier: MIT
"""Dry-run execution bundle for TT5L side-info Lightning paired cells.

The paired-axis plan proves the ten exact-eval cells can be described. The
execution preflight proves lane claims and harvest templates are coherent. This
module binds those two surfaces into a dry-run-default operator bundle without
launching provider work or turning a plan into score authority.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_preflight import (
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_SCHEMA,
)

L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA = (
    "l5_v2_tt5l_sideinfo_lightning_execution_bundle_v1"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_TOOL_PATH = (
    "tools/build_l5_v2_tt5l_sideinfo_lightning_execution_bundle.py"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_REPORT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.md"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_CLAIMS_PATH = (
    ".omx/state/active_lane_dispatch_claims.md"
)

_FALSE_AUTHORITY_FLAGS = {
    "score_claim_valid": False,
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "ready_for_provider_dispatch": False,
    "dispatch_attempted": False,
}
_PLANNING_AUTHORITY_FLAGS = {
    "planning_only": True,
    **_FALSE_AUTHORITY_FLAGS,
}
TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH = (
    ".omx/state/dykstra_feasibility_time_traveler_l5.json"
)
TT5L_DYKSTRA_FEASIBILITY_SCHEMA = "dykstra_feasibility_verdict_v1"

_PROMOTION_ADJACENT_SURFACES = {
    "sideinfo_effect_claim": (
        "complete_harvested_exact_eval_cells_required",
        "passing_sideinfo_effect_curve_artifact_required",
    ),
    "timing_smoke_authority": (
        "complete_harvested_exact_eval_cells_required",
        "paired_cpu_cuda_runtime_timing_artifacts_required",
    ),
    "paired_anchor_claim": (
        "complete_paired_cpu_cuda_exact_eval_cells_required",
        "adjudicated_exact_eval_artifacts_required",
    ),
}

_SOURCE_DIRS = (
    "src",
    "experiments",
    "submissions",
    "scripts",
    "upstream",
    "tools",
    "pyproject.toml",
)
T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV = (
    "INFLATE_TORCH_SPEC=torch==2.5.1+cu124",
    "INFLATE_TORCHVISION_SPEC=torchvision==0.20.1+cu124",
    "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124",
    "UV_INDEX_STRATEGY=unsafe-best-match",
)


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


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _axis_eval_device(axis: str) -> str:
    if axis == "contest_cpu":
        return "cpu"
    if axis == "contest_cuda":
        return "cuda"
    raise ValueError(f"unsupported TT5L side-info axis: {axis!r}")


def _axis_label(axis: str) -> str:
    return "[contest-CPU]" if axis == "contest_cpu" else "[contest-CUDA]"


def _q(value: str) -> str:
    return shlex.quote(value)


def _cell_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("variant") or "").strip(), str(row.get("axis") or "").strip()


def _lane_id_for_cell(*, variant: str, axis: str) -> str:
    return f"lane_l5_v2_tt5l_sideinfo_effect_curve_{_slug(variant)}_{_slug(axis)}"


def _variant_rows_by_name(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in _mapping_rows(manifest.get("variants")):
        variant = str(row.get("variant") or "").strip()
        if variant:
            out[variant] = row
    return out


def _source_manifest_path(job_name: str) -> str:
    return f"experiments/results/lightning_batch/{job_name}/source_manifest.json"


def _source_manifest_receipt_path(job_name: str) -> str:
    return f"experiments/results/lightning_batch/{job_name}/source_manifest_receipt.json"


def _stage_source_manifest_command(
    *,
    job_name: str,
    archive_path: str,
    remote_repo_dir: str,
) -> str:
    command = [
        ".venv/bin/python",
        "scripts/lightning_repro_workspace.py",
        "--remote",
        "<lightning-ssh-target>",
        "--remote-pact",
        remote_repo_dir,
        "--run-id",
        job_name,
        "--manifest-out",
        _source_manifest_path(job_name),
        "--receipt-out",
        _source_manifest_receipt_path(job_name),
    ]
    for source in _SOURCE_DIRS:
        command.extend(["--source", source])
    command.extend(
        [
            "--artifact",
            archive_path,
            "--requirements-mode",
            "no-install",
            "--no-install",
            "--ssh-connect-timeout",
            "30",
        ]
    )
    return shlex.join(command)


def _adjudication_args(adjudication: Mapping[str, Any], archive_size_bytes: int) -> list[str]:
    baseline_score = adjudication.get("baseline_score", 0.1928)
    predicted_low = adjudication.get("predicted_band_low", 0.0)
    predicted_high = adjudication.get("predicted_band_high", 200.0)
    regression_threshold = adjudication.get("regression_threshold", 200.0)
    max_sane_score = adjudication.get("max_sane_score", 200.0)
    baseline_archive = adjudication.get("baseline_archive_size_bytes")
    if not isinstance(baseline_archive, int):
        baseline_archive = archive_size_bytes
    return [
        "--adjudicate",
        "--baseline-score",
        str(baseline_score),
        "--baseline-archive-bytes",
        str(baseline_archive),
        "--predicted-band",
        str(predicted_low),
        str(predicted_high),
        "--regression-threshold",
        str(regression_threshold),
        "--max-sane-score",
        str(max_sane_score),
    ]


def _runtime_env_args_for_machine(machine: str) -> list[str]:
    """Return launch-script env pins required by the selected machine class."""

    machine_l = machine.lower()
    if "t4" not in machine_l and "g4dn" not in machine_l:
        return []
    args: list[str] = []
    for item in T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV:
        args.extend(["--env", item])
    return args


def _launcher_base_command(
    *,
    spec: Mapping[str, Any],
    variant: str,
    axis: str,
    lane_id: str,
    archive_path: str,
    archive_sha256: str,
    archive_size_bytes: int,
    remote_repo_dir: str,
    submission_dir: str,
    local_artifact_dir: str,
    pair_group_id: str,
    run_id: str,
    source_spec_command_sha256: str,
) -> list[str]:
    eval_device = _axis_eval_device(axis)
    output_dir = str(spec.get("remote_output_dir") or "").strip()
    job_name = str(spec.get("name") or "").strip()
    machine = str(spec.get("machine") or "T4").strip()
    max_runtime = spec.get("max_runtime")
    adjudication = (
        spec.get("adjudication") if isinstance(spec.get("adjudication"), Mapping) else {}
    )
    command = [
        ".venv/bin/python",
        "scripts/launch_lightning_batch_job.py",
        "exact-eval",
        "--job-name",
        job_name,
        "--archive",
        f"{remote_repo_dir.rstrip('/')}/{archive_path}",
        "--repo-dir",
        remote_repo_dir.rstrip("/"),
        "--upstream-dir",
        f"{remote_repo_dir.rstrip('/')}/upstream",
        "--output-dir",
        output_dir,
        "--machine",
        machine,
        "--python-bin",
        ".venv/bin/python",
        *_runtime_env_args_for_machine(machine),
        "--inflate-sh",
        f"{submission_dir.rstrip('/')}/inflate.sh",
        "--expected-archive-sha256",
        archive_sha256,
        "--expected-archive-size-bytes",
        str(archive_size_bytes),
        "--local-artifact-dir",
        local_artifact_dir,
        "--dispatch-lane-id",
        lane_id,
        "--dispatch-claims-path",
        L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_CLAIMS_PATH,
        "--eval-device",
        eval_device,
        "--source-manifest",
        _source_manifest_path(job_name),
        "--queue-metadata",
        f"variant={variant}",
        "--queue-metadata",
        f"axis={axis}",
        "--queue-metadata",
        f"pair_group_id={pair_group_id}",
        "--queue-metadata",
        f"run_id={run_id}",
        "--queue-metadata",
        f"lane_id={lane_id}",
        "--queue-metadata",
        f"archive_sha256={archive_sha256}",
        "--queue-metadata",
        "source_plan="
        f"{L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH}",
        "--queue-metadata",
        f"source_spec_command_sha256={source_spec_command_sha256}",
    ]
    if isinstance(max_runtime, int):
        command.extend(["--max-runtime", str(max_runtime)])
    command.extend(_adjudication_args(adjudication, archive_size_bytes))
    return command


def _dry_run_submit_command(base_command: list[str], *, state_path: str) -> str:
    command = list(base_command)
    command[3:3] = ["--state-path", state_path]
    command.append("--dry-run")
    return shlex.join(command)


def _non_dry_run_submit_command_template(base_command: list[str]) -> str:
    command = list(base_command)
    command.extend(
        [
            "--studio",
            "<lightning-studio>",
            "--teamspace",
            "<lightning-teamspace>",
            "<--user-or---org>",
            "<lightning-user-or-org>",
            "--remote-preflight-ssh-target",
            "<lightning-ssh-target>",
        ]
    )
    return shlex.join(command)


def _variant_manifest_blockers(
    *,
    manifest: Mapping[str, Any],
    manifest_path: str,
) -> list[str]:
    blockers: list[str] = []
    if not manifest:
        return ["source_variant_manifest_missing"]
    if manifest.get("schema") != L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA:
        blockers.append("source_variant_manifest_schema_mismatch")
    if manifest.get("score_claim") is not False:
        blockers.append("source_variant_manifest_score_claim_not_false")
    if manifest.get("promotion_eligible") is not False:
        blockers.append("source_variant_manifest_promotion_eligible_not_false")
    if manifest.get("dispatch_attempted") is not False:
        blockers.append("source_variant_manifest_dispatch_attempted_not_false")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        blockers.append("source_variant_manifest_runtime_missing")
    if not manifest_path:
        blockers.append("source_variant_manifest_path_missing")
    return blockers


def _dykstra_status(repo_root: Path) -> dict[str, Any]:
    path = repo_root / TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH
    blockers: list[str] = []
    payload: Mapping[str, Any] = {}
    if not path.is_file():
        return {
            "artifact_path": TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH,
            "artifact_exists": False,
            "artifact_valid": False,
            "schema": "",
            "sha256": "",
            "verdict": "",
            "blockers": ["dykstra_feasibility_artifact_missing"],
        }
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        blockers.append("dykstra_feasibility_json_invalid")
    else:
        if isinstance(loaded, Mapping):
            payload = loaded
        else:
            blockers.append("dykstra_feasibility_not_object")
    if payload and payload.get("schema") != TT5L_DYKSTRA_FEASIBILITY_SCHEMA:
        blockers.append("dykstra_feasibility_schema_mismatch")
    verdict = str(payload.get("verdict") or "").strip() if payload else ""
    if payload and verdict != "FEASIBLE":
        blockers.append("dykstra_feasibility_verdict_not_feasible")
    return {
        "artifact_path": TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH,
        "artifact_exists": path.is_file(),
        "artifact_valid": bool(payload) and not blockers,
        "schema": payload.get("schema", "") if payload else "",
        "sha256": _sha256_file(path) if path.is_file() else "",
        "verdict": verdict,
        "blockers": blockers,
    }


def _promotion_adjacent_readiness(dykstra: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit false-authority blockers for post-dry-run claim surfaces."""

    dykstra_valid = dykstra.get("artifact_valid") is True
    surfaces: dict[str, dict[str, Any]] = {}
    for surface, evidence_blockers in _PROMOTION_ADJACENT_SURFACES.items():
        blockers: list[str] = []
        if not dykstra_valid:
            blockers.append(f"{surface}_requires_valid_dykstra_feasibility_artifact")
        blockers.extend(f"{surface}_{blocker}" for blocker in evidence_blockers)
        surfaces[surface] = {
            "ready": False,
            "blockers": blockers,
        }
    return surfaces


def build_l5_v2_tt5l_sideinfo_lightning_execution_bundle(
    *,
    preflight: Mapping[str, Any],
    preflight_path: str | Path,
    lightning_plan: Mapping[str, Any],
    lightning_plan_path: str | Path,
    variant_manifest: Mapping[str, Any],
    variant_manifest_path: str | Path,
    repo_root: str | Path,
    current_head_commit: str = "",
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Return a dry-run-default execution bundle for ten TT5L Lightning cells."""

    root = Path(repo_root).resolve()
    preflight_file = _resolve_repo_path(preflight_path, root)
    plan_file = _resolve_repo_path(lightning_plan_path, root)
    variant_manifest_file = _resolve_repo_path(variant_manifest_path, root)
    preflight_sha256 = _sha256_file(preflight_file) if preflight_file.is_file() else ""
    plan_sha256 = _sha256_file(plan_file) if plan_file.is_file() else ""
    variant_manifest_sha256 = (
        _sha256_file(variant_manifest_file) if variant_manifest_file.is_file() else ""
    )
    generated = generated_at_utc or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    expected_cell_count = (
        len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS)
        * len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES)
    )
    blockers: list[str] = []
    dykstra = _dykstra_status(root)
    promotion_adjacent_readiness = _promotion_adjacent_readiness(dykstra)
    if preflight.get("schema") != L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_SCHEMA:
        blockers.append("execution_preflight_schema_mismatch")
    if (
        lightning_plan.get("schema")
        != L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA
    ):
        blockers.append("lightning_paired_axis_plan_schema_mismatch")
    for field in _FALSE_AUTHORITY_FLAGS:
        if field != "score_claim_valid" and preflight.get(field) is not False:
            blockers.append(f"execution_preflight_{field}_not_false")
        if lightning_plan.get(field) is not False:
            blockers.append(f"lightning_paired_axis_plan_{field}_not_false")
    if preflight.get("ready_for_operator_claiming") is not True:
        blockers.append("execution_preflight_not_ready_for_operator_claiming")
    if lightning_plan.get("all_cells_dry_run_ready") is not True:
        blockers.append("lightning_paired_axis_plan_cells_not_dry_run_ready")
    if str(preflight.get("source_plan_sha256") or "") != plan_sha256:
        blockers.append("execution_preflight_source_plan_sha_mismatch")

    variant_manifest_rel = _repo_relative(variant_manifest_file, root)
    blockers.extend(
        _variant_manifest_blockers(
            manifest=variant_manifest,
            manifest_path=variant_manifest_rel,
        )
    )
    variant_rows = _variant_rows_by_name(variant_manifest)
    runtime = (
        variant_manifest.get("runtime")
        if isinstance(variant_manifest.get("runtime"), Mapping)
        else {}
    )
    source_dispatch_plan = str(lightning_plan.get("source_dispatch_plan") or "").strip()
    source_dispatch_plan_path = (
        _resolve_repo_path(source_dispatch_plan, root) if source_dispatch_plan else None
    )
    source_dispatch_plan_sha256 = (
        _sha256_file(source_dispatch_plan_path)
        if source_dispatch_plan_path is not None and source_dispatch_plan_path.is_file()
        else ""
    )
    submission_dir = str(runtime.get("submission_dir") or "").strip()
    remote_repo_dir = str(lightning_plan.get("remote_repo_dir") or "").strip()
    if not submission_dir:
        blockers.append("source_variant_manifest_submission_dir_missing")
    if not remote_repo_dir:
        blockers.append("lightning_paired_axis_plan_remote_repo_dir_missing")

    preflight_by_key = {_cell_key(row): row for row in _mapping_rows(preflight.get("cells"))}
    plan_by_key = {_cell_key(row): row for row in _mapping_rows(lightning_plan.get("cells"))}
    cells: list[dict[str, Any]] = []

    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
            key = (variant, axis)
            preflight_cell = preflight_by_key.get(key, {})
            plan_cell = plan_by_key.get(key, {})
            variant_row = variant_rows.get(variant, {})
            lane_id = _lane_id_for_cell(variant=variant, axis=axis)
            job_name = str(preflight_cell.get("job_name") or plan_cell.get("job_name") or "")
            local_artifact_dir = str(
                preflight_cell.get("local_artifact_dir")
                or plan_cell.get("local_artifact_dir")
                or ""
            )
            archive_path = str(variant_row.get("archive_path") or "").strip()
            archive_sha256 = str(preflight_cell.get("archive_sha256") or "").strip()
            archive_bytes = preflight_cell.get("archive_size_bytes")
            if not isinstance(archive_bytes, int):
                archive_bytes = None
            pair_group_id = str(preflight_cell.get("pair_group_id") or "").strip()
            run_id = str(preflight_cell.get("run_id") or "").strip()
            source_command_sha256 = str(
                preflight_cell.get("source_spec_command_sha256") or ""
            ).strip()
            plan_command_sha256 = str(plan_cell.get("command_sha256") or "").strip()
            spec = plan_cell.get("spec") if isinstance(plan_cell.get("spec"), Mapping) else {}
            cell_blockers: list[str] = []
            if not preflight_cell:
                cell_blockers.append("execution_preflight_cell_missing")
            if not plan_cell:
                cell_blockers.append("lightning_paired_axis_plan_cell_missing")
            if not variant_row:
                cell_blockers.append("source_variant_manifest_row_missing")
            if preflight_cell.get("ready_for_operator_claiming") is not True:
                cell_blockers.append("execution_preflight_cell_not_ready")
            if plan_cell.get("ready_for_operator_dispatch") is not True:
                cell_blockers.append("lightning_paired_axis_plan_cell_not_ready")
            if preflight_cell.get("lane_id") != lane_id:
                cell_blockers.append("execution_preflight_lane_id_mismatch")
            if source_command_sha256 != plan_command_sha256:
                cell_blockers.append("source_spec_command_sha256_mismatch")
            for field_name, value in {
                "job_name": job_name,
                "local_artifact_dir": local_artifact_dir,
                "archive_path": archive_path,
                "archive_sha256": archive_sha256,
                "pair_group_id": pair_group_id,
                "run_id": run_id,
                "source_spec_command_sha256": source_command_sha256,
                "submission_dir": submission_dir,
                "remote_repo_dir": remote_repo_dir,
            }.items():
                if not value:
                    cell_blockers.append(f"{field_name}_missing")
            if archive_bytes is None:
                cell_blockers.append("archive_size_bytes_missing")
            stage_command = ""
            dry_run_command = ""
            non_dry_run_command = ""
            dry_run_state_path = (
                f"{local_artifact_dir.rstrip('/')}/launcher_dry_run_state.json"
                if local_artifact_dir
                else ""
            )
            if not cell_blockers and archive_bytes is not None:
                base_command = _launcher_base_command(
                    spec=spec,
                    variant=variant,
                    axis=axis,
                    lane_id=lane_id,
                    archive_path=archive_path,
                    archive_sha256=archive_sha256,
                    archive_size_bytes=archive_bytes,
                    remote_repo_dir=remote_repo_dir,
                    submission_dir=submission_dir,
                    local_artifact_dir=local_artifact_dir,
                    pair_group_id=pair_group_id,
                    run_id=run_id,
                    source_spec_command_sha256=source_command_sha256,
                )
                stage_command = _stage_source_manifest_command(
                    job_name=job_name,
                    archive_path=archive_path,
                    remote_repo_dir=remote_repo_dir,
                )
                dry_run_command = _dry_run_submit_command(
                    base_command,
                    state_path=dry_run_state_path,
                )
                non_dry_run_command = _non_dry_run_submit_command_template(
                    base_command
                )
            cells.append(
                {
                    **_PLANNING_AUTHORITY_FLAGS,
                    "variant": variant,
                    "axis": axis,
                    "axis_label": _axis_label(axis),
                    "eval_device": _axis_eval_device(axis),
                    "lane_id": lane_id,
                    "platform": "lightning",
                    "job_name": job_name,
                    "archive_path": archive_path,
                    "archive_path_in_lightning_workspace": (
                        f"{remote_repo_dir.rstrip('/')}/{archive_path}"
                        if remote_repo_dir and archive_path
                        else ""
                    ),
                    "archive_sha256": archive_sha256,
                    "archive_size_bytes": archive_bytes,
                    "pair_group_id": pair_group_id,
                    "run_id": run_id,
                    "local_artifact_dir": local_artifact_dir,
                    "expected_result_json": (
                        f"{local_artifact_dir.rstrip('/')}/contest_auth_eval.json"
                        if local_artifact_dir
                        else ""
                    ),
                    "stage_source_manifest_receipt_path": (
                        _source_manifest_receipt_path(job_name) if job_name else ""
                    ),
                    "source_spec_command_sha256": source_command_sha256,
                    "source_spec_command_is_authoritative": True,
                    "claim_command": preflight_cell.get("claim_command", ""),
                    "stage_source_manifest_command_template": stage_command,
                    "dry_run_submit_command": dry_run_command,
                    "dry_run_state_path": dry_run_state_path,
                    "non_dry_run_submit_command_template": non_dry_run_command,
                    "terminal_success_claim_template": preflight_cell.get(
                        "terminal_success_claim_template",
                        "",
                    ),
                    "terminal_failure_claim_template": preflight_cell.get(
                        "terminal_failure_claim_template",
                        "",
                    ),
                    "harvest_probe_command_template": preflight_cell.get(
                        "harvest_probe_command_template",
                        "",
                    ),
                    "ready_for_dry_run_submit": not cell_blockers,
                    "ready_for_provider_dispatch": False,
                    "ready_for_non_dry_run_submit": False,
                    "non_dry_run_submit_blockers": [
                        "claim_command_must_be_run_first",
                        "source_manifest_must_be_staged_by_stage_command",
                        "lightning_identity_and_workspace_required",
                        "remote_preflight_ssh_target_required",
                        "operator_must_remove_template_placeholders",
                    ],
                    "blockers": _dedupe(cell_blockers),
                }
            )
            blockers.extend(cell_blockers)

    ready_dry_run_cells = sum(1 for cell in cells if cell["ready_for_dry_run_submit"])
    return {
        **_PLANNING_AUTHORITY_FLAGS,
        "schema": L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA,
        "tool": L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_TOOL_PATH,
        "generated_at_utc": generated,
        "current_head_commit": current_head_commit,
        "source_preflight": _repo_relative(preflight_file, root),
        "source_preflight_sha256": preflight_sha256,
        "source_plan": _repo_relative(plan_file, root),
        "source_plan_sha256": plan_sha256,
        "source_plan_commit": lightning_plan.get("source_commit", ""),
        "source_dispatch_plan": source_dispatch_plan,
        "source_dispatch_plan_sha256": source_dispatch_plan_sha256,
        "source_variant_manifest": variant_manifest_rel,
        "source_variant_manifest_sha256": variant_manifest_sha256,
        "runtime_tree_sha256": runtime.get("runtime_tree_sha256"),
        "runtime_content_tree_sha256": runtime.get("runtime_content_tree_sha256"),
        "runtime_file_count": runtime.get("runtime_file_count"),
        "dykstra_feasibility_status": dykstra,
        "ready_for_sideinfo_effect_claim": False,
        "ready_for_timing_smoke_authority": False,
        "ready_for_paired_anchor_claim": False,
        "promotion_adjacent_readiness": promotion_adjacent_readiness,
        "promotion_adjacent_blockers": {
            surface: list(status["blockers"])
            for surface, status in promotion_adjacent_readiness.items()
        },
        "remote_repo_dir": remote_repo_dir,
        "submission_dir": submission_dir,
        "dry_run_default": True,
        "provider_submit_default": "refuse_non_dry_run",
        "operator_execute_required": True,
        "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
        "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
        "cell_count": len(cells),
        "expected_cell_count": expected_cell_count,
        "ready_dry_run_cell_count": ready_dry_run_cells,
        "cells": cells,
        "ready_for_dry_run_submit": (
            len(cells) == expected_cell_count
            and ready_dry_run_cells == expected_cell_count
            and not blockers
        ),
        "ready_for_non_dry_run_submit": False,
        "ready_for_provider_dispatch": False,
        "execution_order": [
            "run_each_stage_source_manifest_command_template",
            "run_each_claim_command",
            "run_each_dry_run_submit_command_and_compare_generated_command_sha",
            "replace_non_dry_run_template_placeholders_after_identity_preflight",
            "run_each_non_dry_run_submit_command_template",
            "harvest_contest_auth_eval_json_for_each_cell",
            "run_each_terminal_claim_template",
            "build_harvest_cells_artifact",
            "build_sideinfo_effect_curve_artifact",
            "refresh_l5_v2_architecture_lock_packet",
        ],
        "global_blockers": [
            *[
                f"dykstra:{blocker}"
                for blocker in dykstra.get("blockers", [])
                if str(blocker)
            ],
            "non_dry_run_submit_requires_lightning_identity_and_workspace",
            "non_dry_run_submit_requires_remote_preflight_ssh_target",
            "non_dry_run_submit_requires_staged_source_manifest",
            "non_dry_run_submit_requires_active_lane_claim_per_cell",
            "score_claim_forbidden_until_effect_curve_artifact_passes",
            *[
                f"{surface}:{blocker}"
                for surface, status in promotion_adjacent_readiness.items()
                for blocker in status["blockers"]
            ],
        ],
        "blockers": _dedupe(blockers),
    }


def l5_v2_tt5l_sideinfo_lightning_execution_bundle_json(
    payload: Mapping[str, Any],
) -> str:
    """Return canonical JSON text for the execution-bundle artifact."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render a compact operator bundle for TT5L Lightning execution."""

    lines = [
        "# L5 v2 TT5L side-info Lightning execution bundle",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        "",
        "This bundle is dry-run/default and fail-closed. It converts the TT5L "
        "side-info paired-axis plan plus execution preflight into concrete "
        "Lightning launch-script commands while keeping the source plan command "
        "SHA as the authoritative runtime payload. It does not submit provider "
        "work and does not claim score movement.",
        "",
        "## Status",
        "",
        f"- Source plan: `{payload.get('source_plan')}`",
        f"- Source preflight: `{payload.get('source_preflight')}`",
        f"- Source variant manifest: `{payload.get('source_variant_manifest')}`",
        f"- Cells ready for dry-run submit: `{payload.get('ready_dry_run_cell_count')}`/`{payload.get('cell_count')}`",
        f"- ready_for_dry_run_submit: `{payload.get('ready_for_dry_run_submit')}`",
        "- ready_for_non_dry_run_submit: `false`",
        "- ready_for_provider_dispatch: `false`",
        "- ready_for_sideinfo_effect_claim: `false`",
        "- ready_for_timing_smoke_authority: `false`",
        "- ready_for_paired_anchor_claim: `false`",
        "- dispatch_attempted: `false`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        f"- Promotion-adjacent blockers: `{payload.get('promotion_adjacent_blockers', {})}`",
        f"- Blockers: `{payload.get('blockers', [])}`",
        f"- Global blockers: `{payload.get('global_blockers', [])}`",
        "",
        "## Cells",
        "",
        "| variant | axis | lane_id | job_name | dry-run ready | archive bytes | command SHA-256 |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for cell in _mapping_rows(payload.get("cells")):
        lines.append(
            f"| `{cell.get('variant')}` | `{cell.get('axis_label')}` | "
            f"`{cell.get('lane_id')}` | `{cell.get('job_name')}` | "
            f"`{cell.get('ready_for_dry_run_submit')}` | "
            f"`{cell.get('archive_size_bytes')}` | "
            f"`{cell.get('source_spec_command_sha256')}` |"
        )
    lines.extend(
        [
            "",
            "## Command Bundle",
            "",
            "For each cell: stage the workspace source manifest, claim the lane, run "
            "the dry-run submit command, then only after identity and remote "
            "preflight replace placeholders in the non-dry-run template. The "
            "source plan's `spec.command` remains authoritative; this bundle "
            "keeps its SHA-256 adjacent to the launcher command.",
            "",
        ]
    )
    for cell in _mapping_rows(payload.get("cells")):
        lines.extend(
            [
                f"### `{cell.get('variant')}` `{cell.get('axis_label')}`",
                "",
                f"- source_spec_command_sha256: `{cell.get('source_spec_command_sha256')}`",
                f"- claim_command: `{cell.get('claim_command')}`",
                f"- stage_source_manifest_command_template: `{cell.get('stage_source_manifest_command_template')}`",
                f"- stage_source_manifest_receipt_path: `{cell.get('stage_source_manifest_receipt_path')}`",
                f"- dry_run_submit_command: `{cell.get('dry_run_submit_command')}`",
                f"- non_dry_run_submit_command_template: `{cell.get('non_dry_run_submit_command_template')}`",
                f"- non_dry_run_submit_blockers: `{cell.get('non_dry_run_submit_blockers', [])}`",
                f"- terminal_success_claim_template: `{cell.get('terminal_success_claim_template')}`",
                f"- terminal_failure_claim_template: `{cell.get('terminal_failure_claim_template')}`",
                f"- harvest_probe_command_template: `{cell.get('harvest_probe_command_template')}`",
                f"- blockers: `{cell.get('blockers', [])}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_ARTIFACT_PATH",
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_REPORT_PATH",
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA",
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_TOOL_PATH",
    "T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV",
    "build_l5_v2_tt5l_sideinfo_lightning_execution_bundle",
    "l5_v2_tt5l_sideinfo_lightning_execution_bundle_json",
    "render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_markdown",
]
