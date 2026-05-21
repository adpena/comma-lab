# SPDX-License-Identifier: MIT
"""Adjudicate DP1 procedural-codebook paired CPU/CUDA harvests.

The DP1 procedural-codebook lane is intentionally split into:

1. train/export byte-closed baseline and procedural packets;
2. paired Modal CPU/CUDA auth eval for each packet;
3. this post-harvest comparison step.

This module does not dispatch provider work and does not create score or
promotion authority. It verifies that both variants were evaluated on both
contest axes, compares procedural-vs-baseline deltas, and optionally registers
the comparison in the probe-outcomes ledger when the caller explicitly asks.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.auth_eval_result import parse_finite_auth_eval_score
from tac.probe_outcomes_ledger import register_probe_outcome

SCHEMA = "dp1_procedural_paired_adjudication_v1"
TOOL_PATH = "tools/adjudicate_dp1_procedural_paired_harvest.py"
EXPECTED_AXES = ("contest_cpu", "contest_cuda")
VARIANTS = ("baseline", "procedural")
AUTH_EVAL_JSON_NAMES = (
    "contest_auth_eval.adjudicated.json",
    "contest_auth_eval.json",
)
FALSE_AUTHORITY_FLAGS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)
FALSE_AUTHORITY_NUMERIC_SCORE_FIELDS = (
    "contest_cuda_score",
    "contest_cpu_score",
    "auth_eval_cuda_score",
    "auth_eval_cpu_score",
    "score",
    "score_recomputed_from_components",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _find_auth_eval_json(axis_dir: Path) -> Path | None:
    for name in AUTH_EVAL_JSON_NAMES:
        candidate = axis_dir / name
        if candidate.is_file():
            return candidate
    return None


def _payload_archive_sha256(payload: Mapping[str, Any]) -> str:
    direct = payload.get("archive_sha256")
    if isinstance(direct, str) and direct:
        return direct
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        nested = provenance.get("archive_sha256")
        if isinstance(nested, str) and nested:
            return nested
    return ""


def _payload_archive_bytes(payload: Mapping[str, Any]) -> int | None:
    for key in ("archive_size_bytes", "archive_bytes", "compressed_size", "bytes"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("archive_size_bytes")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _false_authority_blockers(payload: Mapping[str, Any], *, source: str) -> list[str]:
    blockers: list[str] = []
    for key in FALSE_AUTHORITY_FLAGS:
        if payload.get(key) is True:
            blockers.append(f"{source}_{key}_true")
    for key in FALSE_AUTHORITY_NUMERIC_SCORE_FIELDS:
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            blockers.append(f"{source}_{key}_present")
    return blockers


def _candidate_summary(
    *,
    variant: str,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    archive = output_dir / "archive.zip"
    summary: dict[str, Any] = {
        "variant": variant,
        "output_dir": _repo_rel(output_dir, repo_root),
        "archive_path": _repo_rel(archive, repo_root),
        "archive_exists": archive.is_file(),
    }
    if archive.is_file():
        summary["archive_bytes"] = archive.stat().st_size
        summary["archive_sha256"] = _sha256_file(archive)
    else:
        blockers.append(f"{variant}_archive_zip_missing")

    for filename, source in (
        ("manifest.json", "manifest"),
        ("provenance.json", "provenance"),
    ):
        path = output_dir / filename
        if not path.is_file():
            blockers.append(f"{variant}_{source}_json_missing")
            continue
        payload = _read_json(path)
        blockers.extend(
            _false_authority_blockers(payload, source=f"{variant}_{source}")
        )

    procedural_path = output_dir / "procedural_variant_provenance.json"
    if variant == "procedural":
        if not procedural_path.is_file():
            blockers.append("procedural_variant_provenance_missing")
        else:
            payload = _read_json(procedural_path)
            blockers.extend(
                _false_authority_blockers(
                    payload, source="procedural_variant_provenance"
                )
            )
            if payload.get("null_exploit_control") is True:
                blockers.append("procedural_variant_marked_null_control")
    elif procedural_path.exists():
        warnings.append("baseline_has_unexpected_procedural_variant_provenance")

    summary["blockers"] = sorted(set(blockers))
    summary["warnings"] = sorted(set(warnings))
    return summary


def _axis_eval_summary(
    *,
    variant: str,
    axis: str,
    axis_dir: Path,
    expected_archive_sha256: str,
    expected_archive_bytes: int | None,
    repo_root: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    eval_path = _find_auth_eval_json(axis_dir)
    summary: dict[str, Any] = {
        "variant": variant,
        "axis": axis,
        "axis_dir": _repo_rel(axis_dir, repo_root),
        "eval_json_path": _repo_rel(eval_path, repo_root) if eval_path else None,
        "eval_json_exists": eval_path is not None,
    }
    if eval_path is None:
        summary["blockers"] = [f"{variant}_{axis}_auth_eval_json_missing"]
        return summary

    payload = _read_json(eval_path)
    observed_axis = str(payload.get("score_axis") or "")
    if observed_axis != axis:
        blockers.append(f"{variant}_{axis}_score_axis_mismatch:{observed_axis or 'missing'}")

    parsed = parse_finite_auth_eval_score(
        payload,
        score_keys=(
            "score_recomputed_from_components",
            "canonical_score",
            "score",
            "final_score",
        ),
        require_component_recompute=True,
    )
    if parsed is None:
        blockers.append(f"{variant}_{axis}_finite_component_recomputed_score_missing")
    else:
        summary["score"] = parsed.score
        summary["score_source_key"] = parsed.source_key
        summary["score_recomputed_matches_components"] = parsed.recomputed_matches

    payload_sha = _payload_archive_sha256(payload)
    payload_bytes = _payload_archive_bytes(payload)
    summary["archive_sha256"] = payload_sha or None
    summary["archive_bytes"] = payload_bytes
    if expected_archive_sha256 and payload_sha != expected_archive_sha256:
        blockers.append(f"{variant}_{axis}_archive_sha256_mismatch")
    if expected_archive_bytes is not None and payload_bytes != expected_archive_bytes:
        blockers.append(f"{variant}_{axis}_archive_bytes_mismatch")

    for key in (
        "avg_posenet_dist",
        "avg_segnet_dist",
        "rate_unscaled",
        "score_pose_contribution",
        "score_seg_contribution",
        "score_rate_contribution",
        "evidence_grade",
        "lane_tag",
    ):
        if key in payload:
            summary[key] = payload.get(key)
    summary["blockers"] = sorted(set(blockers))
    return summary


def _score_delta(
    evals: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    axis: str,
) -> float | None:
    try:
        baseline = float(evals["baseline"][axis]["score"])
        procedural = float(evals["procedural"][axis]["score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(baseline) or not math.isfinite(procedural):
        return None
    return procedural - baseline


def _verdict_from_deltas(deltas: Mapping[str, float | None]) -> tuple[str, str]:
    if any(value is None for value in deltas.values()):
        return (
            "OPERATOR_REVIEW_REQUIRED",
            "wait_for_complete_paired_cpu_cuda_harvest_before_updating_dp1_policy",
        )
    cpu_delta = float(deltas["contest_cpu"])
    cuda_delta = float(deltas["contest_cuda"])
    if cpu_delta < 0.0 and cuda_delta < 0.0:
        return (
            "PROCEED",
            "procedural_codebook_improves_both_axes; queue next byte_closed_variant_or_exact_replay_review",
        )
    if cpu_delta <= 0.0 or cuda_delta <= 0.0:
        return (
            "PARTIAL",
            "axes_disagree_or_one_axis_flat; inspect component_deltas_before_next_dispatch",
        )
    return (
        "DEFER",
        "procedural_codebook_regresses_both_axes; redesign_or_retarget_before_repeating_same_probe",
    )


def build_dp1_procedural_paired_adjudication(
    *,
    baseline_output_dir: str | Path,
    procedural_output_dir: str | Path,
    baseline_cpu_dir: str | Path,
    baseline_cuda_dir: str | Path,
    procedural_cpu_dir: str | Path,
    procedural_cuda_dir: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return a fail-closed paired CPU/CUDA comparison packet."""

    root = Path(repo_root).resolve()
    candidate_dirs = {
        "baseline": Path(baseline_output_dir).resolve(),
        "procedural": Path(procedural_output_dir).resolve(),
    }
    axis_dirs = {
        "baseline": {
            "contest_cpu": Path(baseline_cpu_dir).resolve(),
            "contest_cuda": Path(baseline_cuda_dir).resolve(),
        },
        "procedural": {
            "contest_cpu": Path(procedural_cpu_dir).resolve(),
            "contest_cuda": Path(procedural_cuda_dir).resolve(),
        },
    }

    candidates = {
        variant: _candidate_summary(
            variant=variant,
            output_dir=output_dir,
            repo_root=root,
        )
        for variant, output_dir in candidate_dirs.items()
    }

    evals: dict[str, dict[str, dict[str, Any]]] = {}
    blockers: list[str] = []
    warnings: list[str] = []
    for variant, candidate in candidates.items():
        blockers.extend(candidate.get("blockers") or [])
        warnings.extend(candidate.get("warnings") or [])
        expected_sha = str(candidate.get("archive_sha256") or "")
        expected_bytes = candidate.get("archive_bytes")
        evals[variant] = {}
        for axis, axis_dir in axis_dirs[variant].items():
            eval_summary = _axis_eval_summary(
                variant=variant,
                axis=axis,
                axis_dir=axis_dir,
                expected_archive_sha256=expected_sha,
                expected_archive_bytes=(
                    int(expected_bytes)
                    if isinstance(expected_bytes, int) and not isinstance(expected_bytes, bool)
                    else None
                ),
                repo_root=root,
            )
            blockers.extend(eval_summary.get("blockers") or [])
            evals[variant][axis] = eval_summary

    deltas = {axis: _score_delta(evals, axis=axis) for axis in EXPECTED_AXES}
    archive_byte_delta = None
    if all("archive_bytes" in candidates[v] for v in VARIANTS):
        archive_byte_delta = (
            int(candidates["procedural"]["archive_bytes"])
            - int(candidates["baseline"]["archive_bytes"])
        )

    if blockers:
        verdict = "OPERATOR_REVIEW_REQUIRED"
        next_action = "repair_missing_or_inconsistent_harvest_artifacts_before_score_delta_use"
        metric_value = None
    else:
        verdict, next_action = _verdict_from_deltas(deltas)
        metric_value = max(float(value) for value in deltas.values() if value is not None)

    return {
        "schema": SCHEMA,
        "tool": TOOL_PATH,
        "repo_root": str(root),
        "planning_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "required_axes": list(EXPECTED_AXES),
        "variants": list(VARIANTS),
        "all_required_evidence_valid": not blockers,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "candidates": candidates,
        "evals": evals,
        "score_deltas_procedural_minus_baseline": deltas,
        "archive_byte_delta_procedural_minus_baseline": archive_byte_delta,
        "metric_name": "max_score_delta_procedural_minus_baseline_lower_is_better",
        "metric_value": metric_value,
        "threshold": 0.0,
        "threshold_token": "procedural_must_improve_or_tie_each_contest_axis",
        "verdict": verdict,
        "blocker_status": "advisory",
        "next_action": next_action,
        "notes": [
            "Lower score is better; negative deltas mean procedural beat baseline on that axis.",
            "This packet compares harvested auth-eval artifacts only; it does not promote either candidate.",
            "Probe-outcomes ledger registration is intentionally opt-in via the CLI flag.",
        ],
    }


def register_dp1_procedural_paired_adjudication(
    report: Mapping[str, Any],
    *,
    evidence_path: str | Path | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str = "codex:dp1_procedural_paired_adjudication",
) -> dict[str, Any]:
    """Append the adjudication result to the canonical probe-outcomes ledger."""

    verdict = str(report.get("verdict") or "OPERATOR_REVIEW_REQUIRED")
    metric_value = report.get("metric_value")
    if not isinstance(metric_value, int | float) or not math.isfinite(float(metric_value)):
        metric_value = 1_000_000_000.0
    procedural = report.get("candidates", {}).get("procedural", {})
    procedural_sha = ""
    if isinstance(procedural, Mapping):
        procedural_sha = str(procedural.get("archive_sha256") or "")
    probe_id = (
        "dp1_procedural_codebook_paired_cpu_cuda_"
        f"{procedural_sha[:12] or 'unknown'}"
    )
    return register_probe_outcome(
        probe_id=probe_id,
        substrate="pretrained_driving_prior_dp1",
        recipe_path=".omx/operator_authorize_recipes/substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch.yaml",
        probe_kind="dp1_procedural_codebook_paired_cpu_cuda_auth_eval",
        verdict=verdict,
        metric_name=str(report.get("metric_name") or "score_delta"),
        metric_value=float(metric_value),
        threshold=(
            float(report["threshold"])
            if isinstance(report.get("threshold"), int | float)
            else None
        ),
        threshold_token=str(report.get("threshold_token") or ""),
        evidence_path=str(evidence_path) if evidence_path else None,
        next_action=str(report.get("next_action") or ""),
        blocker_status=str(report.get("blocker_status") or "advisory"),
        agent=agent,
        path=path,
        lock_path=lock_path,
        score_deltas_procedural_minus_baseline=report.get(
            "score_deltas_procedural_minus_baseline"
        ),
        archive_byte_delta_procedural_minus_baseline=report.get(
            "archive_byte_delta_procedural_minus_baseline"
        ),
        all_required_evidence_valid=report.get("all_required_evidence_valid"),
        blockers=report.get("blockers"),
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = [
        "# DP1 Procedural Paired-Harvest Adjudication",
        "",
        f"- Schema: `{report.get('schema')}`",
        f"- Evidence valid: `{report.get('all_required_evidence_valid')}`",
        f"- Verdict: `{report.get('verdict')}`",
        f"- Metric: `{report.get('metric_name')}` = `{report.get('metric_value')}`",
        f"- Blockers: `{', '.join(report.get('blockers') or []) or 'none'}`",
        "",
        "| axis | baseline score | procedural score | delta procedural-baseline |",
        "|---|---:|---:|---:|",
    ]
    evals = report.get("evals") if isinstance(report.get("evals"), Mapping) else {}
    deltas = (
        report.get("score_deltas_procedural_minus_baseline")
        if isinstance(report.get("score_deltas_procedural_minus_baseline"), Mapping)
        else {}
    )
    for axis in EXPECTED_AXES:
        baseline = evals.get("baseline", {}).get(axis, {}) if isinstance(evals, Mapping) else {}
        procedural = evals.get("procedural", {}).get(axis, {}) if isinstance(evals, Mapping) else {}
        rows.append(
            "| {axis} | {baseline} | {procedural} | {delta} |".format(
                axis=axis,
                baseline=baseline.get("score", ""),
                procedural=procedural.get("score", ""),
                delta=deltas.get(axis, ""),
            )
        )
    rows.extend(
        [
            "",
            f"- Archive byte delta: `{report.get('archive_byte_delta_procedural_minus_baseline')}`",
            f"- Next action: `{report.get('next_action')}`",
        ]
    )
    return "\n".join(rows) + "\n"


__all__ = [
    "SCHEMA",
    "TOOL_PATH",
    "build_dp1_procedural_paired_adjudication",
    "register_dp1_procedural_paired_adjudication",
    "render_markdown",
]
