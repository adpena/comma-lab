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

from tac.deploy.lightning.batch_jobs import (
    LightningAdjudicationSpec,
    LightningBatchJobsClient,
    make_exact_eval_spec,
)
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
from tac.optimizer.exact_dispatch_authority import exact_dispatch_authority

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
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA = (
    "l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_v1"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_TOOL_PATH = (
    "tools/build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan.py"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_REPORT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.md"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_ARTIFACT_ROOT = (
    "experiments/results/lightning_batch/"
    "l5_v2_tt5l_sideinfo_effect_curve_paired_axes"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_CUDA_ONLY_REPORT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_lightning_alt_provider_plan_20260517_codex.md"
)
DEFAULT_LIGHTNING_REMOTE_REPO_DIR = "/teamspace/studios/this_studio/pact"
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
_REQUIRED_VARIANT_ROW_CUSTODY_FIELDS = (
    "generation_rule",
    "variant_seed",
    "source_archive_sha256",
    "source_archive_member_sha256",
    "source_sideinfo_section_sha256",
    "sideinfo_changed_from_source",
    "archive_sha_changed_from_source",
    "archive_member_sha_changed_from_source",
    "sideinfo_section_sha_changed_from_source",
)
_REQUIRED_VARIANT_ROW_BOOL_FIELDS = (
    "sideinfo_changed_from_source",
    "archive_sha_changed_from_source",
    "archive_member_sha_changed_from_source",
    "sideinfo_section_sha_changed_from_source",
)
_REQUIRED_VARIANT_ROW_SHA_FIELDS = (
    "source_archive_sha256",
    "source_archive_member_sha256",
    "source_sideinfo_section_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _variant_custody_blockers(row: Mapping[str, Any]) -> list[str]:
    variant = str(row.get("variant") or "unknown").strip() or "unknown"
    blockers: list[str] = []
    for field in _REQUIRED_VARIANT_ROW_CUSTODY_FIELDS:
        if field not in row:
            blockers.append(f"variant_custody_field_missing:{variant}:{field}")
    for field in _REQUIRED_VARIANT_ROW_BOOL_FIELDS:
        if field in row and not isinstance(row.get(field), bool):
            blockers.append(f"variant_custody_bool_field_invalid:{variant}:{field}")
    for field in _REQUIRED_VARIANT_ROW_SHA_FIELDS:
        value = str(row.get(field) or "").strip().lower()
        if field in row and not _SHA256_RE.fullmatch(value):
            blockers.append(f"variant_custody_sha_field_invalid:{variant}:{field}")
    return blockers


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


def _axis_eval_device(axis: str) -> str:
    if axis == "contest_cpu":
        return "cpu"
    if axis == "contest_cuda":
        return "cuda"
    raise ValueError(f"unsupported TT5L side-info effect-curve axis: {axis!r}")


def _axis_job_suffix(axis: str) -> str:
    return _axis_eval_device(axis)


def _lightning_job_name(*, variant: str, axis: str) -> str:
    variant_slug = _slug(variant).replace("_", "-")
    return f"l5-v2-tt5l-sideinfo-{variant_slug}-{_axis_job_suffix(axis)}-20260517"


def _lightning_cell_local_dir(*, variant: str, axis: str, artifact_root: str) -> str:
    return f"{artifact_root.rstrip('/')}/{_slug(variant)}/{axis}"


def _hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_lightning_dry_run_artifacts(
    *,
    spec: Any,
    state_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    reset_state: bool,
) -> dict[str, Any]:
    if reset_state:
        for path in (state_path, stdout_path, stderr_path):
            if path.exists():
                path.unlink()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    record = LightningBatchJobsClient(state_path=state_path).submit(spec, dry_run=True)
    stdout_text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text(stdout_path, stdout_text)
    _write_text(stderr_path, "")
    return record


def _lightning_adjudication(*, archive_bytes: int, eval_device: str) -> LightningAdjudicationSpec:
    return LightningAdjudicationSpec(
        baseline_score=0.1928,
        predicted_band_low=0.0,
        predicted_band_high=200.0,
        regression_threshold=200.0,
        baseline_archive_size_bytes=archive_bytes,
        max_sane_score=200.0,
        required_samples=600,
        required_device=eval_device,
        allow_component_gate_forensic_success=True,
        allow_sane_score_forensic_success=True,
    )


def _dry_run_state_snapshot(path: Path, *, repo_root: Path) -> dict[str, str | int]:
    return {
        "path": _repo_relative(path, repo_root),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
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


def _variant_archive_manifest_path(
    *,
    archive_path: str,
    row: Mapping[str, Any],
    repo_root: Path,
) -> str:
    declared = str(row.get("archive_manifest_path") or "").strip()
    if declared:
        return declared
    if not archive_path:
        return ""
    archive = _resolve_repo_path(archive_path, repo_root)
    return _repo_relative(archive.parent / "archive_manifest.json", repo_root)


def _exact_dispatch_authority_for_variant(
    *,
    variant: str,
    row: Mapping[str, Any],
    archive: Mapping[str, Any],
    runtime: Mapping[str, Any],
    submission_dir: str,
    lanes: Mapping[str, str],
    repo_root: Path,
) -> dict[str, Any]:
    """Run shared exact-dispatch custody on both paired TT5L axes."""

    archive_path = str(archive.get("path") or "").strip()
    archive_sha256 = str(archive.get("sha256") or "").strip()
    archive_bytes = archive.get("bytes")
    archive_manifest_path = _variant_archive_manifest_path(
        archive_path=archive_path,
        row=row,
        repo_root=repo_root,
    )
    base_authority_row = {
        "candidate_id": f"l5_v2_tt5l_sideinfo_effect_curve_{_slug(variant)}",
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "modal_paired_cpu_cuda_auth_eval",
        "ready_for_exact_eval_dispatch": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "archive_path": archive_path,
        "candidate_archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "candidate_archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "candidate_archive_bytes": archive_bytes,
        "submission_dir": submission_dir,
        "archive_manifest_path": archive_manifest_path,
        "score_affecting_payload_changed": row.get("archive_sha_changed_from_source"),
        "charged_bits_changed": row.get("archive_member_sha_changed_from_source"),
        "source_archive_sha256": row.get("source_archive_sha256"),
        "source_payload_sha256": row.get("source_sideinfo_section_sha256"),
        "candidate_payload_sha256": archive_sha256,
        "runtime_tree_sha256": runtime.get("runtime_tree_sha256"),
        "runtime_content_tree_sha256": runtime.get("runtime_content_tree_sha256"),
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": "present",
        "runtime_consumption_proof_path": str(
            row.get("runtime_consumption_proof_path")
            or (Path(submission_dir) / "runtime_consumption_proof.json").as_posix()
        ),
    }
    verdicts: dict[str, Any] = {}
    combined_blockers: list[str] = []
    claims_path = repo_root / ".omx" / "state" / "active_lane_dispatch_claims.md"
    for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
        axis_row = {
            **base_authority_row,
            "lane_id": str(lanes.get(axis) or ""),
            "exact_eval_axis": axis,
        }
        verdict = exact_dispatch_authority(
            axis_row,
            repo_root=repo_root,
            source=(
                "l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan:"
                f"{variant}:{axis}"
            ),
            dispatch_claims_path=claims_path,
            claim_policy="preclaim_conflict_check",
            required_score_axis=axis,
        ).as_dict()
        verdicts[axis] = verdict
        combined_blockers.extend(
            f"exact_dispatch_authority:{axis}:{blocker}"
            for blocker in verdict.get("blockers", [])
            if str(blocker)
        )
    return {
        "schema": "l5_v2_tt5l_sideinfo_exact_dispatch_authority_v1",
        "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
        "authorized": all(
            verdict.get("authorized") is True for verdict in verdicts.values()
        ),
        "archive_manifest_path": archive_manifest_path,
        "dispatch_claims_path": _repo_relative(claims_path, repo_root),
        "axis_verdicts": verdicts,
        "blockers": _dedupe(combined_blockers),
    }


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
        custody_blockers = _variant_custody_blockers(row)
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
            list(manifest_blockers)
            + archive_blockers
            + custody_blockers
            + command_blockers
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
        exact_authority = _exact_dispatch_authority_for_variant(
            variant=variant,
            row=row,
            archive=archive,
            runtime=runtime,
            submission_dir=submission_dir,
            lanes=lanes,
            repo_root=root,
        )
        dispatch_blockers = _dedupe(
            dispatch_blockers + _as_text_list(exact_authority.get("blockers"))
        )
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
                "exact_dispatch_authority": exact_authority,
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


def build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan(
    *,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    repo_root: str | Path,
    remote_repo_dir: str = DEFAULT_LIGHTNING_REMOTE_REPO_DIR,
    artifact_root: str = L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_ARTIFACT_ROOT,
    machine: str = "T4",
    python_bin: str = ".venv/bin/python",
    source_dispatch_plan: str = L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_ARTIFACT_PATH,
    modal_billing_blocker: str = (
        ".omx/research/"
        "l5_v2_tt5l_sideinfo_effect_curve_modal_billing_blocker_20260517_codex.json"
    ),
    source_commit: str = "",
    materialize_dry_runs: bool = True,
    reset_state: bool = True,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the Lightning 5x2 dry-run surface for TT5L side-info variants.

    The function never submits provider work. It uses Lightning's dry-run
    client to materialize exact-eval state records when requested, then records
    hashes of those ignored result-tree artifacts in a durable `.omx` ledger.
    """

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
    generated = generated_at_utc or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    cells: list[dict[str, Any]] = []
    blockers: list[str] = list(manifest_blockers)

    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        row = variants_by_name.get(variant, {})
        archive, archive_blockers = _variant_archive_status(row=row, repo_root=root)
        custody_blockers = _variant_custody_blockers(row)
        archive_sha = str(archive.get("sha256") or "")
        archive_bytes = archive.get("bytes")
        if not isinstance(archive_bytes, int):
            archive_bytes = None
        pair_group_id = _variant_pair_group_id(variant, archive_sha) if archive_sha else ""
        run_id = _variant_run_id(variant, archive_sha) if archive_sha else ""
        variant_blockers = _dedupe(
            list(manifest_blockers) + archive_blockers + custody_blockers
        )
        if not submission_dir:
            variant_blockers.append("variant_manifest_submission_dir_missing")
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
            eval_device = _axis_eval_device(axis)
            local_dir_rel = _lightning_cell_local_dir(
                variant=variant,
                axis=axis,
                artifact_root=artifact_root,
            )
            local_dir = _resolve_repo_path(local_dir_rel, root)
            state_path = local_dir / "state.json"
            stdout_path = local_dir / "dry_run.stdout"
            stderr_path = local_dir / "dry_run.stderr"
            cell_blockers = list(variant_blockers)
            spec_dict: dict[str, Any] | None = None
            queue_dict: dict[str, Any] | None = None
            record_status = "not_materialized"
            invariants: dict[str, bool] = {}
            command_sha256 = ""
            if archive_sha and archive_bytes is not None and not cell_blockers:
                job_name = _lightning_job_name(variant=variant, axis=axis)
                spec = make_exact_eval_spec(
                    name=job_name,
                    archive_path=f"{remote_repo_dir.rstrip('/')}/{archive['path']}",
                    repo_dir=remote_repo_dir.rstrip("/"),
                    upstream_dir=f"{remote_repo_dir.rstrip('/')}/upstream",
                    output_dir=f"{remote_repo_dir.rstrip('/')}/{local_dir_rel}",
                    machine=machine,
                    python_bin=python_bin,
                    inflate_sh=f"{submission_dir}/inflate.sh",
                    expected_archive_sha256=archive_sha,
                    expected_archive_size_bytes=archive_bytes,
                    queue_metadata={
                        "variant": variant,
                        "axis": axis,
                        "pair_group_id": pair_group_id,
                        "run_id": run_id,
                        "archive_sha256": archive_sha,
                        "source_plan": source_dispatch_plan,
                    },
                    local_artifact_dir=local_dir_rel,
                    adjudication=_lightning_adjudication(
                        archive_bytes=archive_bytes,
                        eval_device=eval_device,
                    ),
                    eval_device=eval_device,
                )
                spec_dict = spec.asdict()
                command_sha256 = _hash_bytes(spec.command.encode("utf-8"))
                if materialize_dry_runs:
                    record = _write_lightning_dry_run_artifacts(
                        spec=spec,
                        state_path=state_path,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        reset_state=reset_state,
                    )
                    record_status = str(record.get("status") or "")
                    queue_dict = (
                        dict(record["queue"]) if isinstance(record.get("queue"), Mapping) else None
                    )
                invariants = {
                    "status_dry_run": (not materialize_dry_runs) or record_status == "DRY_RUN",
                    "role_matches_axis": spec.role == f"exact_{eval_device}_eval",
                    "adjudication_required_device_matches_axis": (
                        spec.adjudication is not None
                        and spec.adjudication.required_device == eval_device
                    ),
                    "command_contains_expected_device": f"--device {eval_device}" in spec.command,
                    "command_omits_opposite_device": (
                        f"--device {'cuda' if eval_device == 'cpu' else 'cpu'}"
                        not in spec.command
                    ),
                    "cuda_require_marker_matches_axis": (
                        ("INFLATE_REQUIRE_CUDA=1" in spec.command)
                        == (eval_device == "cuda")
                    ),
                    "queue_metadata_axis_matches_cell": (
                        spec.queue_metadata.get("axis") == axis
                    ),
                    "queue_metadata_pair_group_matches_cell": (
                        spec.queue_metadata.get("pair_group_id") == pair_group_id
                    ),
                    "queue_metadata_run_id_matches_cell": (
                        spec.queue_metadata.get("run_id") == run_id
                    ),
                    "stderr_empty": (
                        (not materialize_dry_runs)
                        or (stderr_path.is_file() and stderr_path.stat().st_size == 0)
                    ),
                }
                failed = [key for key, ok in invariants.items() if not ok]
                if failed:
                    cell_blockers.extend(
                        f"lightning_paired_axis_invariant_failed:{key}"
                        for key in failed
                    )
            else:
                cell_blockers.append("lightning_exact_eval_spec_not_materialized")

            state_snapshot = (
                _dry_run_state_snapshot(state_path, repo_root=root)
                if materialize_dry_runs and state_path.is_file()
                else None
            )
            stdout_snapshot = (
                _dry_run_state_snapshot(stdout_path, repo_root=root)
                if materialize_dry_runs and stdout_path.is_file()
                else None
            )
            stderr_snapshot = (
                _dry_run_state_snapshot(stderr_path, repo_root=root)
                if materialize_dry_runs and stderr_path.is_file()
                else None
            )
            cell = {
                **_FALSE_AUTHORITY_FLAGS,
                "variant": variant,
                "axis": axis,
                "role": f"exact_{eval_device}_eval",
                "required_device": eval_device,
                "archive_sha256": archive_sha,
                "archive_size_bytes": archive_bytes,
                "pair_group_id": pair_group_id,
                "run_id": run_id,
                "local_artifact_dir": local_dir_rel,
                "state_path": str(state_path.relative_to(root)),
                "dry_run_stdout_path": str(stdout_path.relative_to(root)),
                "dry_run_stderr_path": str(stderr_path.relative_to(root)),
                "state": state_snapshot,
                "dry_run_stdout": stdout_snapshot,
                "dry_run_stderr": stderr_snapshot,
                "state_sha256": state_snapshot["sha256"] if state_snapshot else "",
                "dry_run_stdout_sha256": (
                    stdout_snapshot["sha256"] if stdout_snapshot else ""
                ),
                "dry_run_stderr_sha256": (
                    stderr_snapshot["sha256"] if stderr_snapshot else ""
                ),
                "command_sha256": command_sha256,
                "job_name": spec_dict.get("name") if spec_dict else "",
                "spec": spec_dict,
                "queue": queue_dict,
                "invariants": invariants,
                "ready_for_operator_dispatch": not cell_blockers,
                "ready_for_provider_dispatch": False,
                "blockers": _dedupe(cell_blockers),
            }
            cells.append(cell)
            blockers.extend(cell["blockers"])

    all_cells_ready = all(cell["ready_for_operator_dispatch"] is True for cell in cells)
    role_set = {str(cell["role"]) for cell in cells}
    return {
        **_FALSE_AUTHORITY_FLAGS,
        "schema": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA,
        "tool": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_TOOL_PATH,
        "generated_at_utc": generated,
        "source_commit": source_commit,
        "source_variant_manifest": _repo_relative(manifest_file, root),
        "source_variant_manifest_sha256": manifest_sha,
        "source_dispatch_plan": source_dispatch_plan,
        "modal_billing_blocker": modal_billing_blocker,
        "supersedes_cuda_only_lightning_plan": (
            L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_CUDA_ONLY_REPORT_PATH
        ),
        "artifact_root": artifact_root,
        "remote_repo_dir": remote_repo_dir.rstrip("/"),
        "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
        "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
        "cell_count": len(cells),
        "cells": cells,
        "all_cells_dry_run_ready": all_cells_ready,
        "ready_for_operator_dispatch_after_identity_and_claims": all_cells_ready,
        "ready_for_provider_dispatch": False,
        "blockers": _dedupe(
            [
                *blockers,
                "dry_run_only_no_provider_job_launched",
                "requires_lightning_identity_and_workspace_preflight_before_submit",
                "requires_source_manifest_staged_to_lightning_workspace_before_submit",
                "requires_per_axis_lane_claim_before_non_dry_run_submit",
                "requires_harvested_contest_cpu_and_contest_cuda_cells_before_sideinfo_effect_claim",
                "score_claim_forbidden_until_effect_curve_artifact_passes",
            ]
        ),
        "verification": {
            "paired_axis_dry_run_semantics": (
                "PASS" if all_cells_ready else "BLOCKED"
            ),
            "stderr_files_empty": all(
                bool(cell.get("invariants", {}).get("stderr_empty")) for cell in cells
            ),
            "cpu_cells_exact_cpu_eval": all(
                cell["role"] == "exact_cpu_eval"
                for cell in cells
                if cell["axis"] == "contest_cpu"
            ),
            "cuda_cells_exact_cuda_eval": all(
                cell["role"] == "exact_cuda_eval"
                for cell in cells
                if cell["axis"] == "contest_cuda"
            ),
            "cpu_commands_omit_cuda_device_and_inflate_require_cuda": all(
                cell.get("invariants", {}).get("command_omits_opposite_device") is True
                and cell.get("invariants", {}).get("cuda_require_marker_matches_axis")
                is True
                for cell in cells
                if cell["axis"] == "contest_cpu"
            ),
            "cuda_commands_retain_cuda_device_and_inflate_require_cuda": all(
                cell.get("invariants", {}).get("command_contains_expected_device") is True
                and cell.get("invariants", {}).get("cuda_require_marker_matches_axis")
                is True
                for cell in cells
                if cell["axis"] == "contest_cuda"
            ),
            "roles": sorted(role_set),
        },
    }


def l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_json(
    payload: Mapping[str, Any],
) -> str:
    """Return canonical JSON text for the Lightning paired-axis plan."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render the Lightning paired-axis dry-run surface for operators."""

    lines = [
        "# L5 v2 TT5L side-info effect curve Lightning paired-axis plan",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        "",
        "This memo supersedes the CUDA-only Lightning alternate-provider memo for "
        "the TT5L side-info effect curve. It does not launch provider work and "
        "does not claim score movement. It records ten dry-run exact-eval cells: "
        "five side-info variants times `[contest-CPU]` and `[contest-CUDA]`.",
        "",
        "## Status",
        "",
        f"- Source commit: `{payload.get('source_commit')}`",
        f"- Source variant manifest: `{payload.get('source_variant_manifest')}`",
        f"- Source dispatch plan: `{payload.get('source_dispatch_plan')}`",
        f"- Modal blocker still recorded at: `{payload.get('modal_billing_blocker')}`",
        f"- Raw dry-run artifact root: `{payload.get('artifact_root')}` "
        "(ignored result-tree state; hashes recorded below)",
        "- Dispatch attempted: `false`",
        "- Score claim: `false`",
        "- Promotion eligible: `false`",
        "- Required axes: `[contest-CPU]`, `[contest-CUDA]`",
        "- Required variants: `zero`, `random_lsb`, `shuffled`, `trained`, `ablated`",
        "",
        "## Paired-Axis Dry-Run Cells",
        "",
        "| variant | axis | role | required device | bytes | archive SHA-256 | command SHA-256 | state SHA-256 |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for cell in _as_mapping_rows(payload.get("cells")):
        state = cell.get("state") if isinstance(cell.get("state"), Mapping) else {}
        lines.append(
            f"| `{cell.get('variant')}` | `{cell.get('axis')}` | "
            f"`{cell.get('role')}` | `{cell.get('required_device')}` | "
            f"{cell.get('archive_size_bytes')} | `{cell.get('archive_sha256')}` | "
            f"`{cell.get('command_sha256')}` | `{state.get('sha256')}` |"
        )
    lines.extend(
        [
            "",
            "## Axis Invariants Verified",
            "",
            "- CPU cells are `exact_cpu_eval`, adjudicate with `required_device=cpu`, "
            "contain `--device cpu`, omit `--device cuda`, and omit "
            "`INFLATE_REQUIRE_CUDA=1`.",
            "- CUDA cells are `exact_cuda_eval`, adjudicate with "
            "`required_device=cuda`, contain `--device cuda`, and retain "
            "`INFLATE_REQUIRE_CUDA=1`.",
            "- All ten `dry_run.stderr` files are zero bytes when the plan is ready.",
            "- Queue metadata carries the expected `axis`, `variant`, `pair_group_id`, "
            "`run_id`, `archive_sha256`, and source dispatch-plan pointer for every cell.",
            "",
            "## Reactivation Criteria",
            "",
            "Before any non-dry-run submit, configure Lightning identity/workspace "
            "variables, stage a fresh source manifest, run the remote preflight, and "
            "claim the per-axis lane. The effect-curve predicate remains blocked until "
            "all ten cells are harvested and the aggregate side-info effect artifact "
            "passes its paired-axis validation.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_ARTIFACT_PATH",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_REPORT_PATH",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_SCHEMA",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_TOOL_PATH",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_REPORT_PATH",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_TOOL_PATH",
    "build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan",
    "build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan",
    "l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_json",
    "l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_json",
    "render_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_markdown",
    "render_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_markdown",
]
