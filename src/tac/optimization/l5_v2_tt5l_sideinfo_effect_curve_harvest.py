# SPDX-License-Identifier: MIT
"""Harvest TT5L side-info effect-curve cells from a paired-axis plan.

The Lightning paired-axis plan is the identity source of truth for TT5L
side-info effect-curve cells. This module converts that plan plus local
``contest_auth_eval`` artifacts into builder-ready cells without allowing a
manual harvest step to drop or remix ``pair_group_id`` / ``run_id``.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.exact_eval_custody import (
    extract_observed_runtime_content_tree_sha256,
)
from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
)
from tac.optimization.l5_v2_probe_intake import (
    exact_eval_evidence_from_auth_eval_artifact,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA,
)

L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_SCHEMA = (
    "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_v1"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_TOOL_PATH = (
    "tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_REPORT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.md"
)

_FALSE_AUTHORITY_FLAGS = {
    "planning_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_provider_dispatch": False,
    "rank_or_kill_eligible": False,
    "dispatch_attempted": False,
}
_AUTH_EVAL_FILENAMES = (
    "contest_auth_eval.adjudicated.json",
    "contest_auth_eval.json",
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


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _read_json_mapping_or_none(path: Path) -> dict[str, Any] | None:
    try:
        return _read_json_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _command_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value or "")


def _command_flag_value(command: str, flag: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    for idx, part in enumerate(parts[:-1]):
        if part == flag:
            return str(parts[idx + 1]).strip()
    prefix = f"{flag}="
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix) :].strip()
    return ""


def _payload_command(payload: Mapping[str, Any]) -> str:
    return _command_text(
        _nested(payload, "custody", "command")
        or _nested(payload, "provenance", "sys_argv")
        or payload.get("sys_argv")
        or _nested(payload, "provenance", "command")
        or payload.get("auth_eval_command")
    )


def _payload_pair_group_id(payload: Mapping[str, Any]) -> str:
    command = _payload_command(payload)
    return str(
        payload.get("pair_group_id")
        or _nested(payload, "custody", "pair_group_id")
        or _nested(payload, "provenance", "pair_group_id")
        or _command_flag_value(command, "--pair-group-id")
        or ""
    ).strip()


def _payload_run_id(payload: Mapping[str, Any]) -> str:
    command = _payload_command(payload)
    return str(
        payload.get("run_id")
        or _nested(payload, "custody", "run_id")
        or _nested(payload, "provenance", "run_id")
        or _command_flag_value(command, "--run-id")
        or ""
    ).strip()


def _payload_axis(payload: Mapping[str, Any]) -> str:
    return str(
        payload.get("score_axis")
        or payload.get("axis")
        or _nested(payload, "custody", "axis")
        or _nested(payload, "provenance", "axis")
        or ""
    ).strip()


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _variant_rows_by_name(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("variant") or ""): row
        for row in _mapping_rows(manifest.get("variants"))
        if str(row.get("variant") or "")
    }


def _variant_liveness_by_name(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for variant, row in _variant_rows_by_name(manifest).items():
        liveness = row.get("sideinfo_liveness")
        if isinstance(liveness, Mapping):
            out[variant] = dict(liveness)
    return out


def _source_variant_manifest(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], str, str]:
    raw_path = str(plan.get("source_variant_manifest") or "").strip()
    if not raw_path:
        return {}, "", ""
    resolved = _resolve_repo_path(raw_path, repo_root)
    if not resolved.is_file():
        return {}, _repo_relative(resolved, repo_root), ""
    manifest = _read_json_mapping(resolved)
    recorded_sha = str(plan.get("source_variant_manifest_sha256") or "").strip()
    actual_sha = _sha256_file(resolved)
    if recorded_sha and recorded_sha != actual_sha:
        manifest = {
            **manifest,
            "_harvest_blocker": "source_variant_manifest_sha_mismatch",
        }
    return manifest, _repo_relative(resolved, repo_root), actual_sha


def _planned_cells(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _mapping_rows(plan.get("cells"))


def _auth_eval_artifact_path(cell: Mapping[str, Any], *, repo_root: Path) -> Path | None:
    raw_dir = str(cell.get("local_artifact_dir") or "").strip()
    if not raw_dir:
        return None
    local_dir = _resolve_repo_path(raw_dir, repo_root)
    for filename in _AUTH_EVAL_FILENAMES:
        candidate = local_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _plan_cell_identity(cell: Mapping[str, Any]) -> dict[str, str]:
    queue = cell.get("spec")
    queue_metadata = {}
    if isinstance(queue, Mapping) and isinstance(queue.get("queue_metadata"), Mapping):
        queue_metadata = dict(queue["queue_metadata"])
    return {
        "axis": str(cell.get("axis") or queue_metadata.get("axis") or "").strip(),
        "variant": str(cell.get("variant") or queue_metadata.get("variant") or "").strip(),
        "archive_sha256": str(
            cell.get("archive_sha256") or queue_metadata.get("archive_sha256") or ""
        ).strip(),
        "pair_group_id": str(
            cell.get("pair_group_id") or queue_metadata.get("pair_group_id") or ""
        ).strip(),
        "run_id": str(cell.get("run_id") or queue_metadata.get("run_id") or "").strip(),
    }


def _missing_evidence(
    *,
    identity: Mapping[str, str],
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    local_dir = str(cell.get("local_artifact_dir") or "").strip()
    artifact_path = str(Path(local_dir) / "contest_auth_eval.json") if local_dir else ""
    return {
        "axis": identity["axis"],
        "archive_sha256": identity["archive_sha256"],
        "pair_group_id": identity["pair_group_id"],
        "run_id": identity["run_id"],
        "artifact_path": artifact_path,
    }


def _cell_blockers(
    *,
    identity: Mapping[str, str],
    cell: Mapping[str, Any],
    artifact_path: Path | None,
    liveness: Mapping[str, Any] | None,
    manifest_blocker: str,
) -> list[str]:
    blockers: list[str] = []
    for key in ("axis", "variant", "archive_sha256", "pair_group_id", "run_id"):
        if not identity.get(key):
            blockers.append(f"plan_cell_{key}_missing")
    spec = cell.get("spec")
    queue_metadata = (
        spec.get("queue_metadata")
        if isinstance(spec, Mapping) and isinstance(spec.get("queue_metadata"), Mapping)
        else {}
    )
    for key in ("axis", "variant", "archive_sha256", "pair_group_id", "run_id"):
        expected = identity.get(key, "")
        observed = str(queue_metadata.get(key) or "").strip()
        if observed and expected and observed != expected:
            blockers.append(f"plan_cell_queue_metadata_{key}_mismatch")
    if artifact_path is None:
        blockers.append(
            "harvested_exact_eval_artifact_missing:"
            f"{identity.get('variant') or '<missing>'}:"
            f"{identity.get('axis') or '<missing>'}"
        )
    if liveness is None:
        blockers.append(
            "source_variant_sideinfo_liveness_missing:"
            f"{identity.get('variant') or '<missing>'}"
        )
    if manifest_blocker:
        blockers.append(manifest_blocker)
    return list(dict.fromkeys(blockers))


def build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
    *,
    plan: Mapping[str, Any],
    plan_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Return builder-ready effect-curve cells from a Lightning plan.

    Missing harvested artifacts are preserved as blocked cells, not dropped,
    so the aggregate effect-curve builder can report the exact missing
    ``variant``/``axis`` cells without weakening the 5x2 paired requirement.
    """

    root = Path(repo_root).resolve()
    resolved_plan = _resolve_repo_path(plan_path, root)
    plan_sha = _sha256_file(resolved_plan) if resolved_plan.is_file() else ""
    variant_manifest, variant_manifest_path, variant_manifest_sha = (
        _source_variant_manifest(plan=plan, plan_path=resolved_plan, repo_root=root)
    )
    manifest_blocker = str(variant_manifest.get("_harvest_blocker") or "")
    liveness_by_variant = _variant_liveness_by_name(variant_manifest)
    cells: list[dict[str, Any]] = []
    blockers: list[str] = []
    artifact_count = 0
    if plan.get("schema") != L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA:
        blockers.append("lightning_paired_axis_plan_schema_mismatch")
    for flag in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "ready_for_provider_dispatch",
        "dispatch_attempted",
    ):
        if plan.get(flag) is True:
            blockers.append(f"lightning_paired_axis_plan_{flag}_true")

    for cell in _planned_cells(plan):
        identity = _plan_cell_identity(cell)
        variant = identity["variant"]
        axis = identity["axis"]
        artifact_path = _auth_eval_artifact_path(cell, repo_root=root)
        liveness = liveness_by_variant.get(variant)
        cell_blockers = _cell_blockers(
            identity=identity,
            cell=cell,
            artifact_path=artifact_path,
            liveness=liveness,
            manifest_blocker=manifest_blocker,
        )
        if artifact_path is not None:
            raw_payload = _read_json_mapping_or_none(artifact_path)
            if raw_payload is None:
                cell_blockers.append(
                    f"harvested_exact_eval_artifact_unreadable:{variant}:{axis}"
                )
            else:
                raw_axis = _payload_axis(raw_payload)
                if raw_axis and raw_axis != axis:
                    cell_blockers.append(
                        f"harvested_exact_eval_axis_mismatch:{variant}:{axis}"
                    )
                raw_pair_group_id = _payload_pair_group_id(raw_payload)
                if raw_pair_group_id and raw_pair_group_id != identity["pair_group_id"]:
                    cell_blockers.append(
                        "harvested_exact_eval_pair_group_id_mismatch:"
                        f"{variant}:{axis}"
                    )
                raw_run_id = _payload_run_id(raw_payload)
                if raw_run_id and raw_run_id != identity["run_id"]:
                    cell_blockers.append(
                        f"harvested_exact_eval_run_id_mismatch:{variant}:{axis}"
                    )
            evidence = exact_eval_evidence_from_auth_eval_artifact(
                artifact_path,
                axis=axis,
                repo_root=root,
                source_metadata={
                    "pair_group_id": identity["pair_group_id"],
                    "run_id": identity["run_id"],
                    "source_lightning_paired_axis_plan_path": _repo_relative(
                        resolved_plan,
                        root,
                    ),
                },
            )
            artifact_count += 1
            if evidence is None:
                evidence = _missing_evidence(identity=identity, cell=cell)
                if raw_payload is not None:
                    cell_blockers.append(
                        f"harvested_exact_eval_artifact_unreadable:{variant}:{axis}"
                    )
            else:
                if raw_payload is not None:
                    runtime_content_sha = extract_observed_runtime_content_tree_sha256(
                        raw_payload
                    )
                    if runtime_content_sha:
                        evidence["runtime_content_tree_sha256"] = runtime_content_sha
                observed_archive_sha = str(evidence.get("archive_sha256") or "").strip()
                if observed_archive_sha != identity["archive_sha256"]:
                    cell_blockers.append(
                        "harvested_exact_eval_archive_sha_mismatch:"
                        f"{variant}:{axis}"
                    )
                observed_archive_bytes = evidence.get("archive_bytes")
                expected_archive_bytes = cell.get("archive_size_bytes")
                if (
                    isinstance(expected_archive_bytes, int)
                    and observed_archive_bytes != expected_archive_bytes
                ):
                    cell_blockers.append(
                        "harvested_exact_eval_archive_bytes_mismatch:"
                        f"{variant}:{axis}"
                    )
        else:
            evidence = _missing_evidence(identity=identity, cell=cell)

        cells.append(
            {
                **_FALSE_AUTHORITY_FLAGS,
                "axis": axis,
                "variant": variant,
                "archive_sha256": identity["archive_sha256"],
                "archive_size_bytes": cell.get("archive_size_bytes"),
                "pair_group_id": identity["pair_group_id"],
                "run_id": identity["run_id"],
                "source_lightning_plan": _repo_relative(resolved_plan, root),
                "source_lightning_plan_sha256": plan_sha,
                "source_lightning_job_name": str(cell.get("job_name") or ""),
                "source_plan_cell": {
                    "local_artifact_dir": str(cell.get("local_artifact_dir") or ""),
                    "state_path": str(cell.get("state_path") or ""),
                    "job_name": str(cell.get("job_name") or ""),
                    "command_sha256": str(cell.get("command_sha256") or ""),
                },
                "source_variant_manifest": variant_manifest_path,
                "source_variant_manifest_sha256": variant_manifest_sha,
                "sideinfo_liveness": dict(liveness or {}),
                "evidence": evidence,
                "ready_for_effect_curve_build": not cell_blockers,
                "blockers": list(dict.fromkeys(cell_blockers)),
            }
        )
        blockers.extend(cell_blockers)

    expected_cells = {
        f"{axis}:{variant}"
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    }
    observed_cells = [f"{row['axis']}:{row['variant']}" for row in cells]
    observed_cell_set = set(observed_cells)
    duplicate_cells = sorted(
        cell for cell in observed_cell_set if observed_cells.count(cell) > 1
    )
    extra_cells = sorted(observed_cell_set - expected_cells)
    missing_expected = sorted(expected_cells - observed_cell_set)
    blockers.extend(f"planned_cell_missing:{cell}" for cell in missing_expected)
    blockers.extend(f"planned_cell_extra:{cell}" for cell in extra_cells)
    blockers.extend(f"planned_cell_duplicate:{cell}" for cell in duplicate_cells)

    return {
        **_FALSE_AUTHORITY_FLAGS,
        "schema": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_SCHEMA,
        "tool": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_TOOL_PATH,
        "source_plan": _repo_relative(resolved_plan, root),
        "source_plan_sha256": plan_sha,
        "source_variant_manifest": variant_manifest_path,
        "source_variant_manifest_sha256": variant_manifest_sha,
        "source_plan_schema": str(plan.get("schema") or ""),
        "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
        "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
        "cell_count": len(cells),
        "ready_cell_count": sum(
            1 for cell in cells if cell.get("ready_for_effect_curve_build") is True
        ),
        "harvested_exact_eval_artifact_count": artifact_count,
        "missing_exact_eval_artifact_count": max(0, len(cells) - artifact_count),
        "cells": cells,
        "observed_cells": cells,
        "sideinfo_effect_curve_builder_tool": "tools/build_l5_v2_sideinfo_effect_curve.py",
        "ready_for_effect_curve_build": (
            len(cells)
            == len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES)
            * len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS)
            and not blockers
        ),
        "effect_curve_builder_command": (
            ".venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py "
            "--cell-json "
            ".omx/research/"
            "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json"
        ),
        "blockers": list(dict.fromkeys(blockers)),
    }


def l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_json(
    payload: Mapping[str, Any],
) -> str:
    """Return deterministic harvest-cell JSON text."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render the TT5L harvest-cell bridge status for `.omx` ledgers."""

    lines = [
        "# L5 v2 TT5L side-info effect curve harvest cells",
        "",
        "This memo is the post-harvest bridge from the Lightning paired-axis "
        "plan to the TT5L side-info effect-curve builder. It does not launch "
        "provider work and does not claim score movement.",
        "",
        "## Authority",
        "",
        "- Score claim: `false`",
        "- Promotion eligible: `false`",
        "- Ready for exact eval dispatch: `false`",
        "- Ready for provider dispatch: `false`",
        "- Dispatch attempted: `false`",
        "",
        "## Sources",
        "",
        f"- Lightning paired-axis plan: `{payload.get('source_plan')}`",
        f"- Lightning plan SHA-256: `{payload.get('source_plan_sha256')}`",
        f"- Variant manifest: `{payload.get('source_variant_manifest')}`",
        f"- Variant manifest SHA-256: `{payload.get('source_variant_manifest_sha256')}`",
        "",
        "## Cell Status",
        "",
        "| variant | axis | expected artifact | ready | blockers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell in _mapping_rows(payload.get("cells")):
        evidence = cell.get("evidence") if isinstance(cell.get("evidence"), Mapping) else {}
        blockers = cell.get("blockers")
        blocker_text = (
            ", ".join(str(item) for item in blockers)
            if isinstance(blockers, list)
            else ""
        )
        lines.append(
            f"| `{cell.get('variant')}` | `{cell.get('axis')}` | "
            f"`{evidence.get('artifact_path')}` | "
            f"`{str(cell.get('ready_for_effect_curve_build')).lower()}` | "
            f"{blocker_text or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            f"- Ready for effect-curve build: `{str(payload.get('ready_for_effect_curve_build')).lower()}`",
            f"- Ready cells: `{payload.get('ready_cell_count')}/{payload.get('cell_count')}`",
            f"- Harvested exact-eval artifacts: `{payload.get('harvested_exact_eval_artifact_count')}`",
            f"- Missing exact-eval artifacts: `{payload.get('missing_exact_eval_artifact_count')}`",
            f"- Effect-curve command: `{payload.get('effect_curve_builder_command')}`",
            "",
            "The downstream effect-curve artifact remains the only surface that "
            "can satisfy the side-info predicate. This bridge is custody plumbing, "
            "not a promotion artifact.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_ARTIFACT_PATH",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_REPORT_PATH",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_SCHEMA",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_TOOL_PATH",
    "build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan",
    "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_json",
    "render_l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_markdown",
]
