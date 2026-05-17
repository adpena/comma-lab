# SPDX-License-Identifier: MIT
"""Dispatch plan for TT5L side-info effect-curve measurements.

The L5-v2 lattice schedule is intentionally first-match: it keeps earlier
architecture-lock blockers visible before the TT5L side-info curve. This
module is the focused escape hatch for the already materialized TT5L variant
archives: it turns the byte-closed variant manifest into reviewable paired
CPU/CUDA work units without claiming score movement or launching provider work.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.deploy.modal.paired_dispatch import (
    PAIRED_AUTH_EVAL_DISPATCH_TOOL,
    paired_auth_eval_dispatch_command_template,
)
from tac.deploy.modal.paired_dispatch_contract import (
    paired_auth_eval_dispatch_command_blockers,
)
from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_REPORT_PATH,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA,
)

L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_SCHEMA = (
    "l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_v1"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_TOOL_PATH = (
    "tools/build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.json"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_REPORT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.md"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_OUTPUT_ROOT = (
    "experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_MEASUREMENT_ID = (
    "measure_tt5l_sideinfo_effect_curve"
)
_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
_FALSE_AUTHORITY_FLAGS = {
    "planning_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "dispatch_attempted": False,
    "adjudication_required": True,
}


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repo root: {resolved}") from exc


def _slug(value: object) -> str:
    text = str(value or "").strip().lower().replace("/", "_")
    text = _SAFE_TOKEN_RE.sub("_", text)
    text = text.strip("_.:-")
    return text or "unknown"


def _as_mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _as_text_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _variant_rows_by_name(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in _as_mapping_rows(manifest.get("variants")):
        variant = str(row.get("variant") or "").strip()
        if variant:
            out[variant] = row
    return out


def _manifest_blockers(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    repo_root: Path,
) -> list[str]:
    blockers: list[str] = []
    if manifest.get("schema") != L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA:
        blockers.append("tt5l_sideinfo_variant_manifest_schema_mismatch")
    if manifest.get("score_claim") is not False:
        blockers.append("variant_manifest_score_claim_not_false")
    if manifest.get("promotion_eligible") is not False:
        blockers.append("variant_manifest_promotion_eligible_not_false")
    if manifest.get("dispatch_attempted") is not False:
        blockers.append("variant_manifest_dispatch_attempted_not_false")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        blockers.append("variant_manifest_runtime_missing")
    else:
        submission_dir = str(runtime.get("submission_dir") or "").strip()
        if not runtime.get("available"):
            blockers.append("variant_manifest_runtime_unavailable")
        if not submission_dir:
            blockers.append("variant_manifest_submission_dir_missing")
        else:
            runtime_dir = _resolve_repo_path(submission_dir, repo_root)
            if not runtime_dir.is_dir():
                blockers.append("variant_manifest_submission_dir_not_found")
            elif not (runtime_dir / "inflate.sh").is_file():
                blockers.append("variant_manifest_inflate_sh_not_found")
    manifest_parent = _repo_relative(manifest_path.parent, repo_root)
    if manifest_parent != ".omx/research":
        blockers.append("variant_manifest_not_in_omx_research")
    missing = [
        variant
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
        if variant not in _variant_rows_by_name(manifest)
    ]
    if missing:
        blockers.append("variant_manifest_required_variants_missing:" + ",".join(missing))
    extra = [
        variant
        for variant in _variant_rows_by_name(manifest)
        if variant not in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    ]
    if extra:
        blockers.append("variant_manifest_extra_variants:" + ",".join(sorted(extra)))
    return blockers


def _variant_archive_status(
    *,
    row: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    variant = str(row.get("variant") or "").strip()
    blockers: list[str] = []
    archive_rel = str(row.get("archive_path") or "").strip()
    expected_sha = str(row.get("archive_sha256") or "").strip().lower()
    archive_path = None
    observed_sha = ""
    archive_bytes = None
    if not archive_rel:
        blockers.append(f"variant_archive_path_missing:{variant}")
    else:
        archive_path = _resolve_repo_path(archive_rel, repo_root)
        if not archive_path.is_file():
            blockers.append(f"variant_archive_missing:{variant}")
        else:
            observed_sha = _sha256_file(archive_path)
            archive_bytes = archive_path.stat().st_size
            if expected_sha and expected_sha != observed_sha:
                blockers.append(f"variant_archive_sha_mismatch:{variant}")
            if not expected_sha:
                blockers.append(f"variant_archive_sha_missing:{variant}")
    declared_bytes = row.get("archive_bytes")
    if isinstance(declared_bytes, int) and archive_bytes is not None:
        if declared_bytes != archive_bytes:
            blockers.append(f"variant_archive_bytes_mismatch:{variant}")
    elif archive_bytes is not None:
        blockers.append(f"variant_archive_bytes_missing:{variant}")
    archive = {
        "path": archive_rel,
        "bytes": archive_bytes,
        "declared_bytes": declared_bytes if isinstance(declared_bytes, int) else None,
        "sha256": observed_sha or expected_sha,
        "expected_sha256": expected_sha,
        "expected_sha256_match": bool(
            observed_sha and expected_sha and observed_sha == expected_sha
        ),
    }
    return archive, blockers


def _variant_lane_id_base(variant: str) -> str:
    return f"lane_l5_v2_tt5l_sideinfo_effect_curve_{_slug(variant)}"


def _variant_pair_group_id(variant: str, archive_sha256: str) -> str:
    return (
        f"pair_l5_v2_tt5l_sideinfo_effect_curve_"
        f"{_slug(variant)}_{archive_sha256[:12]}"
    )


def _variant_run_id(variant: str, archive_sha256: str) -> str:
    return (
        f"l5_v2_tt5l_sideinfo_effect_curve_"
        f"{_slug(variant)}_{archive_sha256[:12]}"
    )


def _variant_output_root(variant: str) -> str:
    return (
        f"{L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_OUTPUT_ROOT}/"
        f"{_slug(variant)}"
    )


def _axis_output_dirs(*, variant: str, run_id: str) -> dict[str, str]:
    root = _variant_output_root(variant)
    return {
        "contest_cuda": f"{root}/modal_auth_eval/{run_id}_cuda",
        "contest_cpu": f"{root}/modal_auth_eval_cpu/{run_id}_cpu",
    }


def _harvest_commands(*, variant: str, run_id: str) -> dict[str, str]:
    return {
        axis: f".venv/bin/python tools/recover_modal_auth_eval.py --output-dir {out_dir}"
        for axis, out_dir in _axis_output_dirs(variant=variant, run_id=run_id).items()
    }


def _paired_command(
    *,
    archive_path: str,
    archive_sha256: str,
    submission_dir: str,
    variant: str,
    modal_bin: str,
    gpu: str,
) -> list[str]:
    run_id = _variant_run_id(variant, archive_sha256)
    pair_group_id = _variant_pair_group_id(variant, archive_sha256)
    command = paired_auth_eval_dispatch_command_template(
        archive_path=archive_path,
        submission_dir=submission_dir,
        lane_id_base=_variant_lane_id_base(variant),
        archive_sha256=archive_sha256,
        execute=False,
        label=f"l5_v2_tt5l_sideinfo_effect_curve_{_slug(variant)}",
        run_id=run_id,
        inflate_sh="inflate.sh",
        output_root=_variant_output_root(variant),
        modal_bin=modal_bin,
        gpu=gpu,
        claim_agent="codex:l5_v2_tt5l_sideinfo_effect_curve",
        claim_notes=(
            "l5_v2_tt5l_sideinfo_effect_curve;"
            f"variant={variant};pair_group_id={pair_group_id};"
            f"archive_sha={archive_sha256}"
        ),
    )
    pair_index = command.index("--pair-group-id") + 1
    command[pair_index] = pair_group_id
    return command


def _command_blockers(command: list[str]) -> list[str]:
    return paired_auth_eval_dispatch_command_blockers(
        paired_dispatch_tool=PAIRED_AUTH_EVAL_DISPATCH_TOOL,
        command_template=shlex.join(command),
        require_command=True,
    )


def _required_cells(variant: str) -> list[dict[str, str]]:
    return [
        {"axis": axis, "variant": variant}
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
    ]


def build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan(
    *,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    repo_root: str | Path,
    modal_bin: str = ".venv/bin/modal",
    gpu: str = "T4",
) -> dict[str, Any]:
    """Return a non-promotional paired dispatch plan for all TT5L variants."""

    root = Path(repo_root).resolve()
    manifest_file = _resolve_repo_path(manifest_path, root)
    manifest_sha = _sha256_file(manifest_file) if manifest_file.is_file() else ""
    manifest_blockers = _manifest_blockers(
        manifest=manifest,
        manifest_path=manifest_file,
        repo_root=root,
    )
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), Mapping) else {}
    submission_dir = str(runtime.get("submission_dir") or "").strip()
    variants_by_name = _variant_rows_by_name(manifest)
    work_units: list[dict[str, Any]] = []
    top_blockers: list[str] = list(manifest_blockers)

    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        row = variants_by_name.get(variant, {})
        archive, archive_blockers = _variant_archive_status(row=row, repo_root=root)
        archive_sha = str(archive.get("sha256") or "")
        command: list[str] = []
        command_blockers: list[str] = []
        if archive_sha and not archive_blockers and submission_dir:
            command = _paired_command(
                archive_path=str(archive["path"]),
                archive_sha256=archive_sha,
                submission_dir=submission_dir,
                variant=variant,
                modal_bin=modal_bin,
                gpu=gpu,
            )
            command_blockers = _command_blockers(command)
        else:
            command_blockers.append("paired_dispatch_command_not_materialized")
        dispatch_blockers = _dedupe(
            list(manifest_blockers) + archive_blockers + command_blockers
        )
        score_claim_blockers = _dedupe(
            _as_text_list(manifest.get("blockers")) + _as_text_list(row.get("blockers"))
        )
        run_id = _variant_run_id(variant, archive_sha) if archive_sha else ""
        pair_group_id = (
            _variant_pair_group_id(variant, archive_sha) if archive_sha else ""
        )
        operator_execute_command = [*command, "--execute"] if command else []
        lanes = {
            "contest_cuda": f"{_variant_lane_id_base(variant)}_contest_cuda",
            "contest_cpu": f"{_variant_lane_id_base(variant)}_contest_cpu",
        }
        ready_for_operator = not dispatch_blockers
        work_units.append(
            {
                **_FALSE_AUTHORITY_FLAGS,
                "measurement_id": (
                    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_MEASUREMENT_ID
                ),
                "work_unit_id": (
                    f"{L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_MEASUREMENT_ID}"
                    f"__{_slug(variant)}"
                ),
                "variant": variant,
                "sideinfo_variant": variant,
                "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
                "required_cells": _required_cells(variant),
                "archive": archive,
                "runtime": {
                    "submission_dir": submission_dir,
                    "runtime_tree_sha256": runtime.get("runtime_tree_sha256"),
                    "runtime_content_tree_sha256": runtime.get(
                        "runtime_content_tree_sha256"
                    ),
                    "runtime_file_count": runtime.get("runtime_file_count"),
                },
                "lane_id": _variant_lane_id_base(variant),
                "lanes": lanes,
                "pair_group_id": pair_group_id,
                "run_id": run_id,
                "provider": "modal",
                "paired_dispatch_tool": PAIRED_AUTH_EVAL_DISPATCH_TOOL,
                "dispatch_command": command,
                "dispatch_command_template": shlex.join(command),
                "dispatch_command_executable": False,
                "operator_execute_required": True,
                "operator_execute_command_after_review": operator_execute_command,
                "operator_execute_command_template_after_review": (
                    shlex.join(operator_execute_command) if operator_execute_command else ""
                ),
                "ready_for_operator_dispatch": ready_for_operator,
                "ready_for_provider_dispatch": False,
                "preclaim_forbidden": True,
                "claim_lifecycle_owner": (
                    "tools/dispatch_modal_paired_auth_eval.py via per-axis "
                    "Modal auth-eval wrappers"
                ),
                "axis_output_dirs": (
                    _axis_output_dirs(variant=variant, run_id=run_id)
                    if run_id
                    else {}
                ),
                "harvest_commands": (
                    _harvest_commands(variant=variant, run_id=run_id)
                    if run_id
                    else {}
                ),
                "variant_manifest_row": dict(row),
                "score_claim_blockers": score_claim_blockers,
                "dispatch_blockers": dispatch_blockers,
                "readiness_blockers": _dedupe(
                    [
                        *dispatch_blockers,
                        "requires_operator_review_and_execute_flag",
                        *score_claim_blockers,
                    ]
                ),
                "blockers": _dedupe([*dispatch_blockers, *score_claim_blockers]),
            }
        )
        top_blockers.extend(dispatch_blockers)
        top_blockers.extend(score_claim_blockers)

    plan_id_payload = {
        "schema": manifest.get("schema"),
        "manifest_sha256": manifest_sha,
        "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
        "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
        "archive_shas": [
            str(row.get("archive", {}).get("sha256") or "")
            for row in work_units
            if isinstance(row.get("archive"), Mapping)
        ],
    }
    return {
        **_FALSE_AUTHORITY_FLAGS,
        "schema": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_SCHEMA,
        "tool": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_TOOL_PATH,
        "generated_at_utc": (
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
        "plan_id": "l5_v2_tt5l_sideinfo_effect_curve_dispatch_"
        + hashlib.sha256(_canonical_json_bytes(plan_id_payload)).hexdigest()[:16],
        "source_manifest_path": _repo_relative(manifest_file, root),
        "source_manifest_sha256": manifest_sha,
        "source_manifest_schema": str(manifest.get("schema") or ""),
        "source_variant_report_path": L5V2_TT5L_SIDEINFO_VARIANT_PACKET_REPORT_PATH,
        "measurement_id": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_MEASUREMENT_ID,
        "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
        "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
        "work_unit_count": len(work_units),
        "ready_work_unit_count": sum(
            1 for row in work_units if row.get("ready_for_operator_dispatch") is True
        ),
        "work_units": work_units,
        "ready_for_operator_dispatch": all(
            row.get("ready_for_operator_dispatch") is True for row in work_units
        ),
        "ready_for_provider_dispatch": False,
        "operator_execute_required": True,
        "manifest_blockers": manifest_blockers,
        "blockers": _dedupe(top_blockers),
        "classification": (
            "byte_closed_dispatch_plan_no_score_claim; paired CPU/CUDA exact-eval "
            "cells still required before TT5L side-info usefulness claims"
        ),
    }


def l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_json(
    payload: Mapping[str, Any],
) -> str:
    """Return canonical JSON text for durable dispatch-plan artifacts."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing side-info dispatch plan."""

    lines = [
        "# L5 v2 TT5L side-info effect-curve dispatch plan",
        "",
        f"- schema: `{payload.get('schema')}`",
        f"- plan_id: `{payload.get('plan_id')}`",
        f"- source_manifest_path: `{payload.get('source_manifest_path')}`",
        f"- source_manifest_sha256: `{payload.get('source_manifest_sha256')}`",
        f"- measurement_id: `{payload.get('measurement_id')}`",
        f"- required_axes: `{payload.get('required_axes')}`",
        f"- required_variants: `{payload.get('required_variants')}`",
        f"- work_unit_count: `{payload.get('work_unit_count')}`",
        f"- ready_work_unit_count: `{payload.get('ready_work_unit_count')}`",
        "- planning_only: `true`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "- dispatch_attempted: `false`",
        f"- ready_for_operator_dispatch: `{payload.get('ready_for_operator_dispatch')}`",
        "- ready_for_provider_dispatch: `false`",
        "- operator_execute_required: `true`",
        f"- blockers: `{payload.get('blockers')}`",
        "",
        "## Work Units",
    ]
    for row in _as_mapping_rows(payload.get("work_units")):
        archive = row.get("archive") if isinstance(row.get("archive"), Mapping) else {}
        runtime = row.get("runtime") if isinstance(row.get("runtime"), Mapping) else {}
        lines.extend(
            [
                "",
                f"### {row.get('variant')}",
                "",
                f"- work_unit_id: `{row.get('work_unit_id')}`",
                f"- archive path: `{archive.get('path')}`",
                f"- archive bytes: `{archive.get('bytes')}`",
                f"- archive sha256: `{archive.get('sha256')}`",
                f"- submission runtime: `{runtime.get('submission_dir')}`",
                f"- lane_id: `{row.get('lane_id')}`",
                f"- lanes: `{row.get('lanes')}`",
                f"- pair_group_id: `{row.get('pair_group_id')}`",
                f"- required_cells: `{row.get('required_cells')}`",
                f"- ready_for_operator_dispatch: `{row.get('ready_for_operator_dispatch')}`",
                "- ready_for_provider_dispatch: `false`",
                f"- dispatch_blockers: `{row.get('dispatch_blockers')}`",
                f"- score_claim_blockers: `{row.get('score_claim_blockers')}`",
                "- dispatch_command: "
                f"`{row.get('dispatch_command_template')}`",
                "- operator_execute_command_after_review: "
                f"`{row.get('operator_execute_command_template_after_review')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            "This is a byte-closed operator dispatch plan for the TT5L side-info "
            "effect curve. It does not launch provider work, does not create lane "
            "claims, and does not claim score movement. Each variant still needs "
            "paired `[contest-CPU]` and `[contest-CUDA]` exact-eval cells harvested "
            "through the canonical Modal recovery path before the side-info "
            "usefulness predicate can be evaluated.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_ARTIFACT_PATH",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_REPORT_PATH",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_SCHEMA",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_TOOL_PATH",
    "build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan",
    "l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_json",
    "render_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_markdown",
]
