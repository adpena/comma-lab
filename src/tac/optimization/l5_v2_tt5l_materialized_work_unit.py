# SPDX-License-Identifier: MIT
"""Build the L5 v2 TT5L materialized paired auth-eval work unit.

The output is a reviewable Modal CPU/CUDA dispatch plan. It is never a score
claim and never provider-dispatch authority by itself; operators must still add
``--execute`` to the paired dispatcher after reviewing custody.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.deploy.modal.auth_eval import modal_uploaded_submission_dir_runtime_manifest
from tac.deploy.modal.paired_dispatch import (
    MODAL_AUTH_EVAL_CPU_REMOTE_SUBMISSION_DIR,
    MODAL_AUTH_EVAL_CUDA_REMOTE_SUBMISSION_DIR,
    paired_auth_eval_axis_command,
)
from tac.optimization.l5_staircase_v2 import (
    TT5L_MATERIALIZED_PAIRED_WORK_UNIT_LANES,
    TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PAIR_GROUP_ID,
    TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_OUTPUT_ROOT,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_REPORT_PATH,
)
from tac.optimizer.exact_readiness import runtime_dependency_manifest

TT5L_MATERIALIZED_PAIRED_WORK_UNIT_REPORT_PATH = (
    ".omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.md"
)
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_SCHEMA = "modal_paired_auth_eval_dispatch_plan_v2"
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_TOOL = (
    "tools/build_l5_v2_tt5l_materialized_paired_work_unit.py"
)
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_DEFAULT_VARIANT = "random_lsb"
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_RUN_ID = (
    "l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement"
)
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_OUTPUT_ROOT = (
    "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact"
)
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_LABEL = "l5_v2_time_traveler_l5_autonomy"
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_LANE_ID_BASE = (
    "lane_l5_v2_measure_tt5l_autonomy_paired_exact"
)
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_CLAIM_AGENT = (
    "codex:l5_v2_paired_measurement_dispatch"
)
TT5L_MATERIALIZED_PAIRED_WORK_UNIT_CLAIM_NOTES = (
    "l5_v2_paired_measurement:"
    "pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda"
)
_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repo root: {resolved}") from exc


def _resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _slug(value: object) -> str:
    text = str(value or "").strip().lower().replace("/", "_")
    text = _SAFE_TOKEN_RE.sub("_", text).strip("._-")
    return text or "unknown"


def _materialized_run_id(
    *,
    materialized_from: Mapping[str, Any] | None,
    archive_sha256: str,
) -> str:
    variant = ""
    if isinstance(materialized_from, Mapping):
        variant = str(materialized_from.get("variant") or "").strip()
    suffix_parts = []
    if variant:
        suffix_parts.append(_slug(variant))
    suffix_parts.append(str(archive_sha256 or "")[:12])
    return (
        TT5L_MATERIALIZED_PAIRED_WORK_UNIT_RUN_ID
        + "_"
        + "_".join(part for part in suffix_parts if part)
    )


def _variant_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("variants")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def select_tt5l_variant_archive(
    *,
    variant_manifest: str | Path,
    variant: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Return archive custody for one TT5L side-info variant manifest row."""

    root = Path(repo_root).resolve()
    manifest_path = _resolve_repo_path(variant_manifest, root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"variant manifest is not a JSON object: {manifest_path}")
    for row in _variant_rows(payload):
        if str(row.get("variant") or "") != variant:
            continue
        archive_rel = str(row.get("archive_path") or "").strip()
        archive_sha = str(row.get("archive_sha256") or "").strip()
        if not archive_rel:
            raise ValueError(f"variant row lacks archive_path: {variant}")
        archive_path = _resolve_repo_path(archive_rel, root)
        if not archive_path.is_file():
            raise FileNotFoundError(f"variant archive missing: {archive_path}")
        observed_sha = _sha256_file(archive_path)
        if archive_sha and archive_sha != observed_sha:
            raise ValueError(
                "variant archive sha mismatch: "
                f"variant={variant} manifest={archive_sha} observed={observed_sha}"
            )
        return {
            "variant_manifest_path": _repo_relative(manifest_path, root),
            "variant_manifest_sha256": _sha256_file(manifest_path),
            "variant_report_path": L5V2_TT5L_SIDEINFO_VARIANT_PACKET_REPORT_PATH,
            "variant": variant,
            "archive_path": _repo_relative(archive_path, root),
            "archive_sha256": observed_sha,
            "archive_bytes": archive_path.stat().st_size,
            "variant_row": dict(row),
            "source_archive_path": payload.get("source_archive_path"),
            "source_archive_sha256": payload.get("source_archive_sha256"),
            "source_sideinfo_liveness": payload.get("source_sideinfo_liveness"),
        }
    raise ValueError(f"variant {variant!r} not found in {manifest_path}")


def default_tt5l_variant_archive_path(*, variant: str) -> Path:
    return Path(L5V2_TT5L_SIDEINFO_VARIANT_PACKET_OUTPUT_ROOT) / variant / "archive.zip"


def _runtime_manifest_by_axis(
    *,
    submission_dir: Path,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    local_manifest = runtime_dependency_manifest(submission_dir, repo_root)
    return {
        "contest_cuda": modal_uploaded_submission_dir_runtime_manifest(
            local_manifest,
            remote_submission_dir=MODAL_AUTH_EVAL_CUDA_REMOTE_SUBMISSION_DIR,
        ),
        "contest_cpu": modal_uploaded_submission_dir_runtime_manifest(
            local_manifest,
            remote_submission_dir=MODAL_AUTH_EVAL_CPU_REMOTE_SUBMISSION_DIR,
        ),
    }


def build_tt5l_materialized_paired_work_unit_plan(
    *,
    archive: str | Path,
    submission_dir: str | Path,
    repo_root: str | Path,
    materialized_from: Mapping[str, Any] | None = None,
    modal_bin: str = ".venv/bin/modal",
    gpu: str = "T4",
    inflate_sh: str = "inflate.sh",
) -> dict[str, Any]:
    """Build the byte-closed TT5L paired CPU/CUDA work-unit plan."""

    root = Path(repo_root).resolve()
    archive_path = _resolve_repo_path(archive, root)
    runtime_dir = _resolve_repo_path(submission_dir, root)
    if not archive_path.is_file():
        raise FileNotFoundError(f"TT5L archive missing: {archive_path}")
    if not runtime_dir.is_dir():
        raise FileNotFoundError(f"TT5L runtime dir missing: {runtime_dir}")
    if not (runtime_dir / inflate_sh).is_file():
        raise FileNotFoundError(f"TT5L runtime inflate entry missing: {runtime_dir / inflate_sh}")

    archive_rel = _repo_relative(archive_path, root)
    runtime_rel = _repo_relative(runtime_dir, root)
    archive_sha = _sha256_file(archive_path)
    archive_bytes = archive_path.stat().st_size
    run_id = _materialized_run_id(
        materialized_from=materialized_from,
        archive_sha256=archive_sha,
    )
    runtime_by_axis = _runtime_manifest_by_axis(
        submission_dir=runtime_dir,
        repo_root=root,
    )
    runtime_tree_by_axis = {
        axis: str(runtime_by_axis[axis].get("runtime_tree_sha256") or "")
        for axis in ("contest_cpu", "contest_cuda")
    }
    runtime_content_by_axis = {
        axis: str(runtime_by_axis[axis].get("runtime_content_tree_sha256") or "")
        for axis in ("contest_cpu", "contest_cuda")
    }
    output_root = Path(TT5L_MATERIALIZED_PAIRED_WORK_UNIT_OUTPUT_ROOT)
    outputs = {
        "contest_cuda": (
            output_root
            / "modal_auth_eval"
            / f"{run_id}_cuda"
        ).as_posix(),
        "contest_cpu": (
            output_root
            / "modal_auth_eval_cpu"
            / f"{run_id}_cpu"
        ).as_posix(),
    }
    notes_by_axis = {
        axis: (
            f"{TT5L_MATERIALIZED_PAIRED_WORK_UNIT_CLAIM_NOTES}; "
            f"pair_group_id={TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PAIR_GROUP_ID}; "
            f"axis={axis}; archive_sha={archive_sha}; bytes={archive_bytes}"
        )
        for axis in ("contest_cpu", "contest_cuda")
    }
    commands = {
        "contest_cpu": paired_auth_eval_axis_command(
            axis="contest_cpu",
            modal_bin=modal_bin,
            archive_path=archive_rel,
            archive_sha256=archive_sha,
            inflate_sh=inflate_sh,
            output_dir=outputs["contest_cpu"],
            pair_group_id=TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PAIR_GROUP_ID,
            lane_id=TT5L_MATERIALIZED_PAIRED_WORK_UNIT_LANES["contest_cpu"],
            instance_job_id=f"{run_id}_cpu",
            claim_agent=TT5L_MATERIALIZED_PAIRED_WORK_UNIT_CLAIM_AGENT,
            claim_notes=notes_by_axis["contest_cpu"],
            submission_dir=runtime_rel,
            expected_runtime_tree_sha256=runtime_tree_by_axis["contest_cpu"],
        ),
        "contest_cuda": paired_auth_eval_axis_command(
            axis="contest_cuda",
            modal_bin=modal_bin,
            archive_path=archive_rel,
            archive_sha256=archive_sha,
            inflate_sh=inflate_sh,
            output_dir=outputs["contest_cuda"],
            gpu=gpu,
            pair_group_id=TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PAIR_GROUP_ID,
            lane_id=TT5L_MATERIALIZED_PAIRED_WORK_UNIT_LANES["contest_cuda"],
            instance_job_id=f"{run_id}_cuda",
            claim_agent=TT5L_MATERIALIZED_PAIRED_WORK_UNIT_CLAIM_AGENT,
            claim_notes=notes_by_axis["contest_cuda"],
            submission_dir=runtime_rel,
            expected_runtime_tree_sha256=runtime_tree_by_axis["contest_cuda"],
        ),
    }
    return {
        "schema": TT5L_MATERIALIZED_PAIRED_WORK_UNIT_SCHEMA,
        "created_at_utc": (
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
        "tool": TT5L_MATERIALIZED_PAIRED_WORK_UNIT_TOOL,
        "run_id": run_id,
        "output_root": TT5L_MATERIALIZED_PAIRED_WORK_UNIT_OUTPUT_ROOT,
        "pair_group_id": TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PAIR_GROUP_ID,
        "required_axes": ["contest_cuda", "contest_cpu"],
        "archive": {
            "path": archive_rel,
            "bytes": archive_bytes,
            "sha256": archive_sha,
            "expected_sha256": archive_sha,
            "expected_sha256_match": True,
        },
        "runtime": {
            "submission_dir": runtime_rel,
            "inflate_sh": inflate_sh,
            "expected_runtime_tree_sha256": None,
            "expected_runtime_tree_sha256_by_axis": runtime_tree_by_axis,
            "expected_runtime_content_tree_sha256_by_axis": runtime_content_by_axis,
            "modal_uploaded_runtime_manifest_by_axis": runtime_by_axis,
        },
        "outputs": outputs,
        "lanes": dict(TT5L_MATERIALIZED_PAIRED_WORK_UNIT_LANES),
        "commands": commands,
        "skip_axis_if_promotable_anchor_exists": False,
        "axes_skipped_due_to_existing_anchor": {
            "contest_cuda": False,
            "contest_cpu": False,
        },
        "existing_anchors_reused": {
            "contest_cuda": None,
            "contest_cpu": None,
        },
        "materialized_from": dict(materialized_from or {}),
        "planning_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "ready_for_operator_dispatch": True,
        "ready_for_provider_dispatch": False,
        "operator_execute_required": True,
        "notes": [
            "This materialized TT5L work unit is a review artifact, not a score claim.",
            "Both axes carry the same archive SHA and pair_group_id.",
            "Provider dispatch still requires operator review and explicit --execute.",
        ],
    }


def tt5l_materialized_paired_work_unit_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_tt5l_materialized_paired_work_unit_markdown(
    payload: Mapping[str, Any],
) -> str:
    archive = payload.get("archive") if isinstance(payload.get("archive"), Mapping) else {}
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), Mapping) else {}
    materialized_from = (
        payload.get("materialized_from")
        if isinstance(payload.get("materialized_from"), Mapping)
        else {}
    )
    runtime_tree = runtime.get("expected_runtime_tree_sha256_by_axis") or {}
    runtime_content = runtime.get("expected_runtime_content_tree_sha256_by_axis") or {}
    lines = [
        "# L5 v2 TT5L materialized paired work-unit plan",
        "",
        f"- schema: `{payload.get('schema')}`",
        f"- tool: `{payload.get('tool')}`",
        f"- materialized artifact: `{TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH}`",
        f"- run_id: `{payload.get('run_id')}`",
        f"- score_claim: `{str(payload.get('score_claim')).lower()}`",
        f"- promotion_eligible: `{str(payload.get('promotion_eligible')).lower()}`",
        (
            "- ready_for_exact_eval_dispatch: "
            f"`{str(payload.get('ready_for_exact_eval_dispatch')).lower()}`"
        ),
        f"- dispatch_attempted: `{str(payload.get('dispatch_attempted')).lower()}`",
        f"- materialized_variant: `{materialized_from.get('variant', '')}`",
        "",
        "## Materialized Custody",
        "",
        f"- archive path: `{archive.get('path')}`",
        f"- archive bytes: `{archive.get('bytes')}`",
        f"- archive sha256: `{archive.get('sha256')}`",
        f"- submission runtime: `{runtime.get('submission_dir')}`",
        f"- pair group: `{payload.get('pair_group_id')}`",
        f"- CPU lane: `{payload.get('lanes', {}).get('contest_cpu')}`",
        f"- CUDA lane: `{payload.get('lanes', {}).get('contest_cuda')}`",
        f"- CPU expected Modal uploaded runtime tree: `{runtime_tree.get('contest_cpu')}`",
        f"- CUDA expected Modal uploaded runtime tree: `{runtime_tree.get('contest_cuda')}`",
        f"- CPU runtime content tree: `{runtime_content.get('contest_cpu')}`",
        f"- CUDA runtime content tree: `{runtime_content.get('contest_cuda')}`",
        "",
        "## Materialization Source",
        "",
        f"- variant manifest: `{materialized_from.get('variant_manifest_path', '')}`",
        f"- variant manifest sha256: `{materialized_from.get('variant_manifest_sha256', '')}`",
        f"- variant report: `{materialized_from.get('variant_report_path', '')}`",
        f"- source archive: `{materialized_from.get('source_archive_path', '')}`",
        f"- source archive sha256: `{materialized_from.get('source_archive_sha256', '')}`",
        "",
        "## Operator Next Action",
        "",
        "Review this materialized CPU/CUDA work unit, then execute through the "
        "canonical paired dispatcher only if the archive/runtime custody is still "
        "accepted. This packet intentionally stays `ready_for_provider_dispatch=false` "
        "until that explicit operator step.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "TT5L_MATERIALIZED_PAIRED_WORK_UNIT_DEFAULT_VARIANT",
    "TT5L_MATERIALIZED_PAIRED_WORK_UNIT_REPORT_PATH",
    "build_tt5l_materialized_paired_work_unit_plan",
    "default_tt5l_variant_archive_path",
    "render_tt5l_materialized_paired_work_unit_markdown",
    "select_tt5l_variant_archive",
    "tt5l_materialized_paired_work_unit_json",
]
