# SPDX-License-Identifier: MIT
"""Canonicalize DQS1 local-first harvests as planning observations.

The DQS1 autopilot writes harvest JSON and macOS-CPU advisory auth-eval JSON.
Those artifacts are useful search signal, but they are not contest authority.
This module converts them into the shared dynamic-sweep observation schema with
explicit false-authority fields and component deltas against the matching
compact top32 baseline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA,
    build_archive_bound_candidate_contract_surface,
)
from tac.optimization.macos_cpu_advisory_signal import EVIDENCE_GRADE, EVIDENCE_TAG
from tac.optimization.mlx_dynamic_sweep_observations import (
    FALSE_AUTHORITY,
    MLXDynamicSweepObservationError,
    build_observation_row,
    file_sha256,
    json_text,
    load_observation_rows,
    summarize_observations,
)
from tac.optimization.pairset_component_marginal import (
    CONTEST_RATE_DENOMINATOR_BYTES,
    CONTEST_RATE_MULTIPLIER,
)

SCHEMA = "dqs1_local_first_harvest_observations.v1"
ROW_SOURCE_SCHEMA = "dqs1_local_first_harvest.v1"
TOOL = "tac.optimization.dqs1_local_first_harvest_observations"
OBSERVED_AXIS = "macos_cpu_advisory"
SWEEP_CONFIG_ID = "dqs1_local_first_macos_cpu_advisory"
OPTIMIZATION_PASS_ID = "local_cpu_advisory_harvest"
BASELINE_POLICY = (
    "local_top32_advisory_scorer_components_with_gap_uleb_archive_size_override"
)


class DQS1LocalHarvestObservationError(ValueError):
    """Raised when a local harvest cannot be safely canonicalized."""


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DQS1LocalHarvestObservationError(f"{path}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise DQS1LocalHarvestObservationError(f"{path}: expected JSON object")
    return payload


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise DQS1LocalHarvestObservationError(f"{label} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise DQS1LocalHarvestObservationError(f"{label} must be numeric") from exc
    if not math.isfinite(out):
        raise DQS1LocalHarvestObservationError(f"{label} must be finite")
    return out


def _optional_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DQS1LocalHarvestObservationError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DQS1LocalHarvestObservationError(f"{label} must be an integer") from exc


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if key in payload and payload.get(key) is not False:
            raise DQS1LocalHarvestObservationError(f"{label} {key} must be false")
    for key in ("score_claim_eligible", "score_claim_valid"):
        if key in payload and payload.get(key) is not False:
            raise DQS1LocalHarvestObservationError(f"{label} {key} must be false")


def _sha_from_advisory(advisory: Mapping[str, Any], *, label: str) -> str:
    provenance = advisory.get("provenance")
    if not isinstance(provenance, Mapping):
        raise DQS1LocalHarvestObservationError(f"{label} provenance is required")
    value = provenance.get("archive_sha256")
    text = str(value or "").lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise DQS1LocalHarvestObservationError(f"{label} archive_sha256 is invalid")
    return text


def _runtime_sha_from_advisory(advisory: Mapping[str, Any], *, label: str) -> str:
    provenance = advisory.get("provenance")
    manifest = (
        provenance.get("inflate_runtime_manifest")
        if isinstance(provenance, Mapping)
        else None
    )
    value = manifest.get("runtime_tree_sha256") if isinstance(manifest, Mapping) else None
    text = str(value or "").lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise DQS1LocalHarvestObservationError(
            f"{label} runtime_tree_sha256 is invalid"
        )
    return text


def _raw_output_sha_from_advisory(advisory: Mapping[str, Any], *, label: str) -> str:
    provenance = advisory.get("provenance")
    manifest = (
        provenance.get("inflated_output_manifest")
        if isinstance(provenance, Mapping)
        else None
    )
    payload = manifest.get("payload") if isinstance(manifest, Mapping) else None
    value = payload.get("aggregate_sha256") if isinstance(payload, Mapping) else None
    text = str(value or "").lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise DQS1LocalHarvestObservationError(
            f"{label} inflated aggregate_sha256 is invalid"
        )
    return text


def _rate_contribution(archive_size_bytes: int) -> float:
    return (
        CONTEST_RATE_MULTIPLIER
        * float(archive_size_bytes)
        / float(CONTEST_RATE_DENOMINATOR_BYTES)
    )


def _score_components(
    advisory: Mapping[str, Any],
    *,
    label: str,
    archive_size_bytes_override: int | None = None,
) -> dict[str, float]:
    archive_size = archive_size_bytes_override
    if archive_size is None:
        archive_size = _optional_int(
            advisory.get("archive_size_bytes"),
            label=f"{label}.archive_size_bytes",
        )
    if archive_size is None:
        provenance = advisory.get("provenance")
        if isinstance(provenance, Mapping):
            archive_size = _optional_int(
                provenance.get("archive_size_bytes"),
                label=f"{label}.provenance.archive_size_bytes",
            )
    if archive_size is None:
        raise DQS1LocalHarvestObservationError(
            f"{label} archive_size_bytes is required"
        )
    if archive_size <= 0:
        raise DQS1LocalHarvestObservationError(
            f"{label} archive_size_bytes must be positive"
        )
    seg = _finite_float(
        advisory.get("score_seg_contribution"),
        label=f"{label}.score_seg_contribution",
    )
    pose = _finite_float(
        advisory.get("score_pose_contribution"),
        label=f"{label}.score_pose_contribution",
    )
    rate = (
        _rate_contribution(archive_size)
        if archive_size_bytes_override is not None
        else _finite_float(
            advisory.get("score_rate_contribution"),
            label=f"{label}.score_rate_contribution",
        )
    )
    return {
        "archive_size_bytes": float(archive_size),
        "score_seg_contribution": seg,
        "score_pose_contribution": pose,
        "score_rate_contribution": rate,
        "score": seg + pose + rate,
    }


def _score_from_advisory(advisory: Mapping[str, Any], *, label: str) -> float:
    for key in ("canonical_score", "score_recomputed_from_components", "final_score"):
        if advisory.get(key) is not None:
            return _finite_float(advisory.get(key), label=f"{label}.{key}")
    components = _score_components(advisory, label=label)
    return components["score"]


def _normalize_harvest_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.endswith("Z") and "T" in text and "-" in text:
        return text
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).isoformat(timespec="seconds") + "Z"
        except ValueError:
            continue
    return text


def _candidate_id_from_acquisition(row: Mapping[str, Any]) -> str:
    for key in ("candidate_id", "acquisition_id", "selector_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise DQS1LocalHarvestObservationError("acquisition candidate id is required")


def load_pairset_acquisition_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json_object(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise DQS1LocalHarvestObservationError(f"{path}: candidates list is required")
    index: dict[str, dict[str, Any]] = {}
    for raw in candidates:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        canonical_id = _candidate_id_from_acquisition(row)
        for key in (
            canonical_id,
            str(row.get("candidate_id") or ""),
            str(row.get("acquisition_id") or ""),
            str(row.get("selector_id") or ""),
        ):
            if key:
                index[key] = row
    return index


def _family_from_acquisition(acquisition: Mapping[str, Any]) -> str:
    operation = acquisition.get("acquisition_operation")
    op = str(operation.get("op") if isinstance(operation, Mapping) else "").strip()
    if op == "drop_one":
        return "decoder_q_pairset_drop_one"
    if op == "drop_two":
        return "decoder_q_pairset_drop_two"
    if op == "diversity_spaced":
        return "decoder_q_pairset_diversity"
    if op:
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in op)
        return f"decoder_q_pairset_{safe}"
    return "decoder_q_selective_dqs1"


def _selected_pair_indices(acquisition: Mapping[str, Any]) -> list[int]:
    raw = acquisition.get("selected_pair_indices")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DQS1LocalHarvestObservationError(
            f"{_candidate_id_from_acquisition(acquisition)} selected_pair_indices is required"
        )
    out: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            raise DQS1LocalHarvestObservationError(
                "selected_pair_indices values must be integers"
            )
        try:
            out.append(int(value))
        except (TypeError, ValueError) as exc:
            raise DQS1LocalHarvestObservationError(
                "selected_pair_indices values must be integers"
            ) from exc
    return out


def _archive_sha_from_harvest(
    harvest: Mapping[str, Any],
    advisory: Mapping[str, Any],
    *,
    label: str,
) -> str:
    harvest_sha = str(harvest.get("candidate_archive_sha256") or "").lower()
    advisory_sha = _sha_from_advisory(advisory, label=label)
    if harvest_sha and harvest_sha != advisory_sha:
        raise DQS1LocalHarvestObservationError(
            f"{label} harvest/advisory archive_sha256 mismatch"
        )
    return advisory_sha


def build_observation_row_from_harvest(
    harvest_path: Path,
    *,
    repo_root: Path,
    acquisition_index: Mapping[str, Mapping[str, Any]],
    baseline_advisory_path: Path,
    baseline_advisory: Mapping[str, Any],
    baseline_archive_size_bytes: int,
    baseline_candidate_id: str,
) -> dict[str, Any]:
    harvest = _load_json_object(harvest_path)
    if harvest.get("schema") != ROW_SOURCE_SCHEMA:
        raise DQS1LocalHarvestObservationError(
            f"{harvest_path}: schema must be {ROW_SOURCE_SCHEMA}"
        )
    _require_false_authority(harvest, label=f"{harvest_path}")
    candidate_id = str(harvest.get("candidate_id") or "").strip()
    if not candidate_id:
        raise DQS1LocalHarvestObservationError(f"{harvest_path}: candidate_id is required")
    acquisition = acquisition_index.get(candidate_id)
    if acquisition is None:
        raise DQS1LocalHarvestObservationError(
            f"{harvest_path}: no pairset acquisition candidate for {candidate_id}"
        )

    advisory_path_raw = harvest.get("local_cpu_advisory_path")
    if not advisory_path_raw:
        raise DQS1LocalHarvestObservationError(
            f"{harvest_path}: local_cpu_advisory_path is required"
        )
    advisory_path = _resolve_path(str(advisory_path_raw), repo_root)
    advisory = _load_json_object(advisory_path)
    _require_false_authority(advisory, label=f"{advisory_path}")
    archive_sha = _archive_sha_from_harvest(
        harvest,
        advisory,
        label=f"{advisory_path}",
    )
    candidate_components = _score_components(advisory, label=str(advisory_path))
    baseline_components = _score_components(
        baseline_advisory,
        label=str(baseline_advisory_path),
        archive_size_bytes_override=baseline_archive_size_bytes,
    )
    component_deltas = {
        "segnet_delta": candidate_components["score_seg_contribution"]
        - baseline_components["score_seg_contribution"],
        "posenet_delta": candidate_components["score_pose_contribution"]
        - baseline_components["score_pose_contribution"],
        "rate_delta": candidate_components["score_rate_contribution"]
        - baseline_components["score_rate_contribution"],
    }
    observed_score = _score_from_advisory(advisory, label=str(advisory_path))
    score_delta_vs_baseline = observed_score - baseline_components["score"]
    archive_bound_surface = _dqs1_archive_bound_candidate_contract_surface(
        harvest=harvest,
        candidate_components=candidate_components,
        baseline_components=baseline_components,
        archive_sha=archive_sha,
        repo_root=repo_root,
        candidate_id=candidate_id,
    )
    selected = _selected_pair_indices(acquisition)
    operation = acquisition.get("acquisition_operation")
    operation_payload = dict(operation) if isinstance(operation, Mapping) else {}
    observed_at = _normalize_harvest_timestamp(harvest.get("harvested_at_utc"))
    extra = {
        "producer": TOOL,
        "evidence_grade": EVIDENCE_GRADE,
        "selected_pair_indices": selected,
        "selected_pair_count": len(selected),
        "selector_kind": acquisition.get("selector_kind"),
        "acquisition_operation": operation_payload,
        "source_schema": ROW_SOURCE_SCHEMA,
        "planner_artifact_path": _repo_rel(harvest_path, repo_root),
        "planner_artifact_sha256": file_sha256(harvest_path),
        "baseline_candidate_id": baseline_candidate_id,
        "baseline_artifact_path": _repo_rel(baseline_advisory_path, repo_root),
        "baseline_artifact_sha256": file_sha256(baseline_advisory_path),
        "baseline_score": baseline_components["score"],
        "baseline_archive_size_bytes": baseline_archive_size_bytes,
        "score_delta_vs_baseline": score_delta_vs_baseline,
        "archive_byte_delta_vs_baseline": int(candidate_components["archive_size_bytes"])
        - int(baseline_components["archive_size_bytes"]),
        "component_delta_baseline_policy": BASELINE_POLICY,
        "archive_bound_candidate_contract_schema": (
            ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
        ),
        "archive_bound_candidate_contract": archive_bound_surface[
            "selected_candidate_contract"
        ],
        "archive_bound_candidate_contract_surface_schema": (
            ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA
        ),
        "archive_bound_candidate_contract_surface": archive_bound_surface,
        "run_id": f"{candidate_id}_{harvest.get('harvested_at_utc')}",
        "notes": (
            "macOS-CPU advisory local-first harvest; planning signal only, "
            "not score/rank/promotion authority."
        ),
        "source_row": {
            "candidate_id": candidate_id,
            "local_score": harvest.get("local_score"),
            "projected_contest_score": harvest.get("projected_contest_score"),
            "conservative_projected_contest_score": harvest.get(
                "conservative_projected_contest_score"
            ),
            "recommended_action": harvest.get("recommended_action"),
            "eureka_trigger": harvest.get("eureka_trigger"),
            "eureka_margin": harvest.get("eureka_margin"),
            "dispatch_blockers": harvest.get("dispatch_blockers"),
            "authority": harvest.get("authority"),
        },
    }
    return build_observation_row(
        candidate_id=candidate_id,
        sweep_config_id=SWEEP_CONFIG_ID,
        optimization_pass_id=OPTIMIZATION_PASS_ID,
        family=_family_from_acquisition(acquisition),
        observed_axis=OBSERVED_AXIS,
        evidence_tag=EVIDENCE_TAG,
        observed_score_or_delta=observed_score,
        archive_sha256=archive_sha,
        runtime_sha256=_runtime_sha_from_advisory(advisory, label=str(advisory_path)),
        raw_output_or_cache_sha256=_raw_output_sha_from_advisory(
            advisory,
            label=str(advisory_path),
        ),
        component_deltas=component_deltas,
        source_artifact_path=_repo_rel(advisory_path, repo_root),
        source_artifact_sha256=file_sha256(advisory_path),
        observed_at_utc=observed_at or None,
        extra=extra,
    )


def _dqs1_archive_bound_candidate_contract_surface(
    *,
    harvest: Mapping[str, Any],
    candidate_components: Mapping[str, Any],
    baseline_components: Mapping[str, Any],
    archive_sha: str,
    repo_root: Path,
    candidate_id: str,
) -> dict[str, Any]:
    archive_path = str(harvest.get("candidate_archive_path") or "").strip()
    candidate_bytes = int(candidate_components["archive_size_bytes"])
    source_bytes = int(baseline_components["archive_size_bytes"])
    blockers = [
        "dqs1_local_first_observation_is_macos_cpu_advisory",
        "dqs1_candidate_requires_receiver_runtime_proof",
        "dqs1_candidate_requires_exact_cpu_or_cuda_replay",
    ]
    if not archive_path:
        blockers.append("dqs1_candidate_archive_path_missing")
    return build_archive_bound_candidate_contract_surface(
        candidates=[
            {
                "archive_native_transform_kind": (
                    "dqs1_pairset_drop_pair_local_advisory"
                ),
                "materialized": bool(archive_path and archive_sha),
                "path": archive_path,
                "sha256": archive_sha,
                "bytes": candidate_bytes,
                "source_archive_bytes": source_bytes,
                "runtime_consumption_proof_ready": False,
                "receiver_contract_kind": "archive_charged_pairset_runtime_selector",
                "receiver_contract_satisfied": False,
                "semantic_payload_changed": True,
                "score_affecting_payload_changed": True,
                "exact_axis_score_affecting_adjudication_required": True,
                "charged_bits_changed": candidate_bytes != source_bytes,
                "blockers": blockers,
            }
        ],
        selected_transform_kind="dqs1_pairset_drop_pair_local_advisory",
        repo_root=repo_root,
        family_id="dqs1_local_first",
        typed_response_id=candidate_id,
        candidate_chain_id=archive_sha,
        entropy_position_label="before_entropy_coder_selector",
    )


def build_observation_rows_from_harvests(
    harvest_paths: Iterable[Path],
    *,
    repo_root: Path,
    pairset_acquisition_path: Path,
    baseline_advisory_path: Path,
    baseline_archive_size_bytes: int,
    baseline_candidate_id: str = "dqs1_top32_gap_uleb",
) -> list[dict[str, Any]]:
    acquisition_index = load_pairset_acquisition_index(pairset_acquisition_path)
    baseline_advisory = _load_json_object(baseline_advisory_path)
    _require_false_authority(baseline_advisory, label=str(baseline_advisory_path))
    rows = [
        build_observation_row_from_harvest(
            path,
            repo_root=repo_root,
            acquisition_index=acquisition_index,
            baseline_advisory_path=baseline_advisory_path,
            baseline_advisory=baseline_advisory,
            baseline_archive_size_bytes=baseline_archive_size_bytes,
            baseline_candidate_id=baseline_candidate_id,
        )
        for path in sorted(harvest_paths)
    ]
    return sorted(rows, key=lambda row: (str(row["observed_at_utc"]), str(row["candidate_id"])))


def write_observation_jsonl(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    replace: bool = False,
) -> None:
    if output_path.exists() and not replace:
        raise DQS1LocalHarvestObservationError(
            f"{output_path} already exists; pass replace=True to overwrite"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), sort_keys=True, allow_nan=False) for row in rows]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    try:
        loaded = load_observation_rows(output_path)
    except MLXDynamicSweepObservationError as exc:
        raise DQS1LocalHarvestObservationError(str(exc)) from exc
    if len(loaded) != len(rows):
        raise DQS1LocalHarvestObservationError("observation JSONL validation row-count mismatch")


def build_harvest_observation_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    jsonl_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    normalized_summary = summarize_observations(rows)
    score_rows = sorted(
        rows,
        key=lambda row: (
            float(row["observed_score_or_delta"]),
            str(row["candidate_id"]),
        ),
    )
    best = score_rows[0] if score_rows else None
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "allowed_use": "macos_cpu_advisory_replanning_signal_only",
        "row_count": len(rows),
        "observed_axis": OBSERVED_AXIS,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_tag": EVIDENCE_TAG,
        "component_delta_baseline_policy": BASELINE_POLICY,
        "jsonl_path": (
            None
            if jsonl_path is None
            else _repo_rel(jsonl_path, repo_root or Path.cwd())
        ),
        "jsonl_sha256": None if jsonl_path is None or not jsonl_path.exists() else file_sha256(jsonl_path),
        "best_local_advisory": None
        if best is None
        else {
            "candidate_id": best["candidate_id"],
            "observed_score": best["observed_score_or_delta"],
            "score_delta_vs_baseline": best.get("score_delta_vs_baseline"),
            "component_deltas": best.get("component_deltas"),
            "source_artifact_path": best.get("source_artifact_path"),
        },
        "observation_summary": normalized_summary,
    }


def render_markdown_summary(summary: Mapping[str, Any]) -> str:
    best = summary.get("best_local_advisory")
    lines = [
        "# DQS1 Local-First Harvest Observations",
        "",
        f"- schema: `{summary.get('schema')}`",
        f"- row_count: `{summary.get('row_count')}`",
        f"- observed_axis: `{summary.get('observed_axis')}`",
        f"- evidence: `{summary.get('evidence_tag')}`",
        f"- allowed_use: `{summary.get('allowed_use')}`",
        f"- score_claim: `{summary.get('score_claim')}`",
        f"- promotion_eligible: `{summary.get('promotion_eligible')}`",
        f"- ready_for_exact_eval_dispatch: `{summary.get('ready_for_exact_eval_dispatch')}`",
        f"- component_delta_baseline_policy: `{summary.get('component_delta_baseline_policy')}`",
    ]
    if isinstance(best, Mapping):
        lines.extend(
            [
                "",
                "## Best Local Advisory",
                "",
                f"- candidate_id: `{best.get('candidate_id')}`",
                f"- observed_score: `{best.get('observed_score')}`",
                f"- score_delta_vs_baseline: `{best.get('score_delta_vs_baseline')}`",
                f"- source_artifact_path: `{best.get('source_artifact_path')}`",
            ]
        )
    lines.extend(
        [
            "",
            "This artifact is planning signal only. Exact contest CPU/CUDA auth eval is still required for any score, rank, promotion, or submission claim.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "BASELINE_POLICY",
    "OBSERVED_AXIS",
    "SCHEMA",
    "DQS1LocalHarvestObservationError",
    "build_harvest_observation_summary",
    "build_observation_row_from_harvest",
    "build_observation_rows_from_harvests",
    "json_text",
    "load_pairset_acquisition_index",
    "render_markdown_summary",
    "write_observation_jsonl",
]
