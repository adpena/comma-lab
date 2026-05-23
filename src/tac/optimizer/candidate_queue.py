# SPDX-License-Identifier: MIT
"""Fail-closed candidate queue adapter for optimizer and sweep outputs.

The queue schema here is intentionally smaller than a new optimizer. It turns
heterogeneous local search artifacts into the ``top_k`` shape already consumed
by dispatch/eval actuators, while preserving the evidence boundary: local CPU,
proxy, and forensic rows remain planning candidates until a separate readiness
gate proves byte-closed archive/runtime custody.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.exact_eval_custody import CONTEST_EXACT_SAMPLE_COUNT
from tac.optimization.local_cpu_contest_drift import (
    EUREKA_SIGNAL_SCHEMA,
    LocalCPUContestDriftError,
    require_eureka_false_authority,
)
from tac.optimization.local_training_runtime_profile import (
    SCHEMA as TRAINER_RUNTIME_PROFILE_SCHEMA,
)
from tac.optimization.local_training_runtime_profile import (
    adapt_runtime_profile_observation_to_candidate,
)
from tac.optimization.pr95_muon_local_training_integration import (
    PLAN_SCHEMA as PR95_LOCAL_TRAINING_PLAN_SCHEMA,
)
from tac.optimization.pr95_muon_local_training_integration import (
    SCHEMA as PR95_LOCAL_TRAINING_SCHEMA,
)
from tac.optimization.pr95_muon_local_training_integration import (
    adapt_pr95_local_training_manifest_to_candidate,
)
from tac.optimization.proxy_candidate_contract import (
    PROXY_DEPLOYMENT_TARGET,
    PROXY_DISPATCH_BLOCKERS,
    PROXY_TARGET_MODES,
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)
from tac.optimization.representation_training_probe_integration import (
    PLAN_SCHEMA as REPRESENTATION_TRAINING_PLAN_SCHEMA,
)
from tac.optimization.representation_training_probe_integration import (
    SCHEMA as REPRESENTATION_TRAINING_SCHEMA,
)
from tac.optimization.representation_training_probe_integration import (
    adapt_representation_training_manifest_to_candidate,
)
from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
    serialized_archive_delta_blockers,
)

QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
TOOL_NAME = "tools/build_optimizer_candidate_queue.py"
PLANNING_TARGET_MODES = list(PROXY_TARGET_MODES)
PLANNING_DEPLOYMENT_TARGET = PROXY_DEPLOYMENT_TARGET
BASE_DISPATCH_BLOCKERS = list(PROXY_DISPATCH_BLOCKERS)
CANDIDATE_IDENTITY_FIELDS: tuple[str, ...] = (
    "candidate_family",
    "representation_family",
    "substrate_family",
    "param_schema",
)
SCORE_AFFECTING_BOOLEAN_FIELDS: frozenset[str] = frozenset(
    {
        "score_affecting_payload_changed",
        "charged_bits_changed",
        "score_affecting_runtime_changed",
    }
)
UNKNOWN_CANDIDATES_REASON = (
    "unsupported_candidates_schema_requires_explicit_adapter_or_"
    "codec_op_param_sweep_manifest_v1"
)


@dataclass(frozen=True)
class SourceExtraction:
    schema: str
    rows: list[dict[str, Any]]
    unsupported_reason: str | None = None


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ValueError(f"YAML source requires PyYAML: {path}") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _shaish(value: Any) -> str | None:
    if isinstance(value, str) and len(value.strip()) == 64:
        text = value.strip().lower()
        if all(ch in "0123456789abcdef" for ch in text):
            return text
    return None


def _validate_mlx_dynamic_normalized_gain_sum(
    row: Mapping[str, Any],
    *,
    context: str,
) -> None:
    normalized_raw = row.get("non_authoritative_normalized_full_video_gain_sum")
    mlx_alias_raw = row.get("non_authoritative_mlx_gain_sum")
    window_raw = row.get("non_authoritative_mlx_window_gain_sum")
    if normalized_raw is None and mlx_alias_raw is None and window_raw is None:
        return
    denominator = _as_int(row.get("full_video_denominator"))
    if denominator != CONTEST_EXACT_SAMPLE_COUNT:
        raise ValueError(
            f"{context}.full_video_denominator must be {CONTEST_EXACT_SAMPLE_COUNT}"
        )
    normalized = _as_float(normalized_raw)
    window = _as_float(window_raw)
    if normalized is None:
        raise ValueError(
            f"{context}.non_authoritative_normalized_full_video_gain_sum must be finite"
        )
    if window is None:
        raise ValueError(
            f"{context}.non_authoritative_mlx_window_gain_sum must be finite"
        )
    expected = window / float(denominator)
    if not math.isclose(normalized, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError(
            f"{context}.non_authoritative_normalized_full_video_gain_sum mismatch"
        )
    if mlx_alias_raw is not None:
        mlx_alias = _as_float(mlx_alias_raw)
        if mlx_alias is None:
            raise ValueError(f"{context}.non_authoritative_mlx_gain_sum must be finite")
        if not math.isclose(mlx_alias, normalized, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError(
                f"{context}.non_authoritative_mlx_gain_sum must equal normalized full-video gain sum"
            )


def _mlx_dynamic_parent_queue_candidate_id(source: Mapping[str, Any]) -> str | None:
    source_candidate_id = str(source.get("candidate_id") or "")
    config_id = str(source.get("sweep_config_id") or "")
    pass_id = str(source.get("optimization_pass_id") or "")
    if not source_candidate_id or not config_id or not pass_id:
        return None
    return f"{source_candidate_id}::{config_id}::{pass_id}"


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _source_schema(payload: Any) -> str:
    if isinstance(payload, Mapping):
        schema = payload.get("schema") or payload.get("schema_version")
        if schema is not None:
            return str(schema)
        tool = payload.get("tool")
        if tool is not None:
            return str(tool)
    if isinstance(payload, list):
        return "json_list"
    return type(payload).__name__


def _base_candidate(candidate_id: str, *, source_path: Path, repo_root: Path) -> dict[str, Any]:
    return apply_proxy_evidence_boundary({
        "candidate_id": candidate_id,
        "source_paths": [_repo_rel(source_path, repo_root)],
    })


def _add_blockers(row: dict[str, Any], blockers: Iterable[str]) -> None:
    row["dispatch_blockers"] = _ordered_unique(
        [*row.get("dispatch_blockers", []), *[str(b) for b in blockers if b]]
    )


def _merge_candidate(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key == "source_paths" or key == "dispatch_blockers":
            merged[key] = _ordered_unique([*merged.get(key, []), *value])
        elif key in SCORE_AFFECTING_BOOLEAN_FIELDS:
            if merged.get(key) is True or value is True:
                merged[key] = True
            elif key not in merged and value is False:
                merged[key] = False
        elif key in {"rank_score", "predicted_contest_cpu_gha", "fitness"}:
            old = _as_float(merged.get(key))
            new = _as_float(value)
            if old is None or (new is not None and new < old):
                merged[key] = value
        elif value is not None and (
            key not in merged or merged.get(key) in (None, "", [], {}) or key in {
                "macos_cpu_score",
                "rank_score_field",
                "queue_priority",
                "ranking_evidence_grade",
            }
        ):
            merged[key] = value
    return merged


def _candidate_identity_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the merge key for one candidate queue row.

    ``candidate_id`` alone is not globally unique once generic representation
    training manifests enter the queue; distinct substrate families may reuse a
    human-readable id such as ``seed17``.  Legacy rows keep merging when these
    family/schema fields are absent or identical.
    """

    family_or_schema_present = any(
        str(row.get(field) or "") for field in CANDIDATE_IDENTITY_FIELDS
    )
    params = row.get("candidate_params")
    if family_or_schema_present and isinstance(params, Mapping):
        params_key = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    else:
        params_key = ""
    return tuple(
        [str(row.get("candidate_id") or "")]
        + [str(row.get(field) or "") for field in CANDIDATE_IDENTITY_FIELDS]
        + [params_key]
    )


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    priority = str(row.get("queue_priority") or "")
    if priority == "auto_promote_gha":
        class_rank = 0
    elif priority == "operator_decision":
        class_rank = 1
    elif row.get("archive_candidate_verified") is True:
        class_rank = 2
    elif row.get("materialized_payload_path"):
        class_rank = 3
    else:
        class_rank = 4
    for key in ("predicted_contest_cpu_gha", "rank_score", "fitness"):
        value = _as_float(row.get(key))
        if value is not None:
            return (class_rank, 0, value, str(row.get("candidate_id") or ""))
    return (class_rank, 1, str(row.get("candidate_id") or ""))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _resolve_repo_path(path_value: Any, repo_root: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _annotate_archive_candidate_verification(row: dict[str, Any], repo_root: Path) -> None:
    archive_path = _resolve_repo_path(row.get("candidate_archive_path") or row.get("archive_path"), repo_root)
    submission_dir = _resolve_repo_path(row.get("submission_dir"), repo_root)
    if archive_path is None and submission_dir is not None:
        archive_path = submission_dir / "archive.zip"
    if archive_path is None and submission_dir is None:
        return

    blockers: list[str] = []
    archive_sha = _shaish(row.get("candidate_archive_sha256") or row.get("archive_sha256"))
    archive_bytes = _as_int(row.get("candidate_archive_bytes") or row.get("archive_bytes") or row.get("archive_size_bytes"))
    if archive_path is None or not archive_path.is_file():
        blockers.append("candidate_archive_path_unverified")
    if archive_sha is None:
        blockers.append("candidate_archive_sha256_missing")
    elif archive_path is not None and archive_path.is_file():
        actual_sha = _sha256_file(archive_path)
        row["candidate_archive_sha256_observed"] = actual_sha
        if actual_sha != archive_sha:
            blockers.append("candidate_archive_sha256_mismatch")
    if archive_bytes is None:
        blockers.append("candidate_archive_bytes_missing")
    elif archive_path is not None and archive_path.is_file() and archive_path.stat().st_size != archive_bytes:
        blockers.append("candidate_archive_bytes_mismatch")

    if blockers:
        row["archive_candidate_verified"] = False
        row["candidate_archive_path_unverified"] = True
        _add_blockers(row, blockers)
        return
    row["archive_candidate_verified"] = True
    row["candidate_archive_path_unverified"] = False


def _read_optional_manifest(path_value: Any, repo_root: Path) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value:
        return {}
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        return {}
    try:
        payload = _load_json(path)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _a1_rollup_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in payload.get("variants") or []:
        if not isinstance(variant, Mapping):
            continue
        candidate_id = str(variant.get("variant_id") or variant.get("candidate_id") or "")
        if not candidate_id:
            continue
        manifest = _read_optional_manifest(variant.get("build_manifest_relpath"), repo_root)
        row = _base_candidate(candidate_id, source_path=source_path, repo_root=repo_root)
        archive_path = manifest.get("archive_path")
        archive_sha = _shaish(manifest.get("archive_sha256") or variant.get("archive_sha256"))
        archive_bytes = _as_int(manifest.get("archive_size_bytes"))
        submission_dir = None
        if isinstance(archive_path, str) and archive_path:
            archive = Path(archive_path)
            submission_dir = archive.parent.as_posix()
        row.update(
            {
                "lane_id": payload.get("lane_id", "lane_pr101_bias_constrained_coord_search"),
                "lane_class": "a1_pr101_bias_coord_search",
                "candidate_family": "a1_inflate_time_bias_coordinate_search",
                "evidence_semantics": "a1_runtime_variant_planning_only",
                "evidence_grade": payload.get("evidence_grade")
                or "[predicted; constrained coord search on A1 substrate]",
                "coords": dict(variant.get("coords") or {}),
                "submission_name": variant.get("submission_name") or candidate_id,
                "submission_dir": submission_dir,
                "archive_path": archive_path,
                "candidate_archive_path": archive_path,
                "archive_sha256": archive_sha,
                "candidate_archive_sha256": archive_sha,
                "archive_bytes": archive_bytes,
                "archive_size_bytes": archive_bytes,
                "candidate_archive_bytes": archive_bytes,
                "archive_unchanged_from_a1": manifest.get("archive_unchanged_from_a1", True),
                "inflate_py_sha256": manifest.get("inflate_py_sha256_new")
                or variant.get("inflate_py_sha256"),
                "score_affecting_runtime_changed": True,
                "runtime_smoke_checked": manifest.get("runtime_smoke_checked", False),
                "source_manifest_path": variant.get("build_manifest_relpath"),
            }
        )
        _add_blockers(
            row,
            [
                "a1_runtime_variant_requires_cpu_or_cuda_eval",
                "archive_bytes_unchanged_score_depends_on_inflate_runtime",
                "runtime_tree_sha_required_before_exact_dispatch",
            ],
        )
        rows.append(row)
    return rows


def _m5max_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    queues = (
        ("auto_promote_gha_queue", "auto_promote_gha"),
        ("operator_decision_queue", "operator_decision"),
    )
    for queue_key, priority in queues:
        for item in summary.get(queue_key) or []:
            if not isinstance(item, Mapping):
                continue
            candidate_id = str(item.get("candidate_id") or "")
            if not candidate_id:
                continue
            rank_score = _as_float(item.get("predicted_contest_cpu_gha"))
            row = _base_candidate(candidate_id, source_path=source_path, repo_root=repo_root)
            row.update(
                {
                    "lane_class": "a1_pr101_bias_coord_search",
                    "candidate_family": "a1_inflate_time_bias_coordinate_search",
                    "queue_priority": priority,
                    "rank_score": rank_score,
                    "rank_score_field": "predicted_contest_cpu_gha",
                    "macos_cpu_score": _as_float(item.get("macos_cpu_score")),
                    "predicted_contest_cpu_gha": rank_score,
                    "ranking_evidence_grade": item.get("tag") or "[macOS-CPU calibrated]",
                    "evidence_semantics": "macos_cpu_calibrated_ranking_not_dispatch_evidence",
                }
            )
            blocker = (
                "operator_decision_required_before_gha_or_exact_cuda"
                if priority == "operator_decision"
                else "gha_eval_required_before_exact_cuda_promotion"
            )
            _add_blockers(row, [blocker, "macos_cpu_is_not_contest_cuda_evidence"])
            rows.append(row)
    return rows


def _codec_search_report_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    op_class = str(payload.get("op_class") or "codec_op")
    op_module = str(payload.get("op_module") or "")
    evidence_semantics = str(payload.get("evidence_semantics") or "cpu_codec_op_search_forensic")
    evidence_grade = str(payload.get("evidence_grade") or "[CPU-prep]")
    tool = str(payload.get("tool") or source_path.name)
    for ev in payload.get("all_evaluations") or []:
        if not isinstance(ev, Mapping):
            continue
        eval_idx = _as_int(ev.get("eval_idx"))
        if eval_idx is None:
            continue
        candidate_id = f"{op_class.lower()}_eval_{eval_idx:05d}"
        row = _base_candidate(candidate_id, source_path=source_path, repo_root=repo_root)
        row.update(
            {
                "lane_class": op_class.lower(),
                "candidate_family": "codec_op_param_search",
                "optimizer_tool": tool,
                "op_module": op_module,
                "op_class": op_class,
                "op_params": dict(ev.get("params") or {}),
                "eval_idx": eval_idx,
                "bytes_out": _as_int(ev.get("bytes_out")),
                "candidate_substream_bytes": _as_int(ev.get("bytes_out")),
                "reconstruction_rms": _as_float(ev.get("reconstruction_rms")),
                "fitness": _as_float(ev.get("fitness")),
                "rank_score": _as_float(ev.get("fitness")),
                "rank_score_field": "fitness",
                "pareto_frontier": bool(ev.get("pareto_frontier")),
                "materialized_payload_path": ev.get("materialized_payload_path"),
                "materialized_payload_bytes": ev.get("materialized_payload_bytes"),
                "materialized_payload_sha256": ev.get("materialized_payload_sha256"),
                "materialized_payload_contract": ev.get("materialized_payload_contract"),
                "decode_coverage_status": ev.get("decode_coverage_status"),
                "evidence_semantics": evidence_semantics,
                "evidence_grade": evidence_grade,
                "error": ev.get("error"),
            }
        )
        _add_blockers(
            row,
            [
                "codec_op_payload_not_archive_zip",
                "archive_substitution_surgery_required",
                "exact_cuda_auth_eval_missing",
            ],
        )
        if ev.get("error"):
            _add_blockers(row, ["optimizer_eval_failed"])
        rows.append(row)
    return rows


def _codec_param_manifest_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in payload.get("candidates") or []:
        if not isinstance(cand, Mapping):
            continue
        candidate_id = str(cand.get("candidate_id") or "")
        if not candidate_id:
            continue
        row = _base_candidate(candidate_id, source_path=source_path, repo_root=repo_root)
        row.update(dict(cand))
        row.update(
            {
                "target_modes": list(PLANNING_TARGET_MODES),
                "deployment_target": PLANNING_DEPLOYMENT_TARGET,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_score": _as_float(cand.get("predicted_score")),  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
                "rank_score_field": "predicted_score",  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
                "evidence_semantics": cand.get("evidence_semantics")
                or payload.get("evidence_semantics")
                or "cpu_substrate_predicted_band",
            }
        )
        _add_blockers(
            row,
            [
                "predicted_score_is_not_score_evidence",
                "archive_substitution_surgery_required",
                "exact_cuda_auth_eval_missing",
            ],
        )
        row = apply_proxy_evidence_boundary(row)
        rows.append(row)
    return rows


def _meta_lagrangian_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    source_rows = (
        payload.get("top_k_forensic")
        or payload.get("engine_top_k_local_only")
        or payload.get("all_evaluations")
        or []
    )
    rows: list[dict[str, Any]] = []
    for cand in source_rows:
        if not isinstance(cand, Mapping):
            continue
        candidate_id = str(cand.get("candidate_id") or "")
        if not candidate_id:
            continue
        row = _base_candidate(candidate_id, source_path=source_path, repo_root=repo_root)
        row.update(dict(cand))
        row.update(
            {
                "target_modes": list(PLANNING_TARGET_MODES),
                "deployment_target": PLANNING_DEPLOYMENT_TARGET,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_score": _as_float(cand.get("proxy_score") or cand.get("lagrangian")),
                "rank_score_field": "proxy_score",
                "evidence_semantics": payload.get("evidence_semantics")
                or "local_proxy_prediction_forensic",
            }
        )
        _add_blockers(row, ["proxy_score_is_not_score_evidence", "candidate_archive_missing"])
        row = apply_proxy_evidence_boundary(row)
        rows.append(row)
    return rows


def _optimizer_guided_queue_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    source_rows = payload.get("top_k_forensic")
    if not isinstance(source_rows, list):
        source_rows = payload.get("top_k")
    if not isinstance(source_rows, list):
        return []

    rows: list[dict[str, Any]] = []
    source_profile = str(payload.get("profile") or "")
    source_optimizer = str(payload.get("optimizer") or "")
    source_optimizer_status = str(payload.get("optimizer_status") or "")
    source_contract = payload.get("profile_contract")
    profile_contract = source_contract if isinstance(source_contract, Mapping) else {}
    profile_blockers = [
        str(item)
        for item in profile_contract.get("dispatch_blockers", [])
        if str(item)
    ]
    for cand in source_rows:
        if not isinstance(cand, Mapping):
            continue
        candidate_id = str(cand.get("candidate_id") or "")
        if not candidate_id:
            continue
        candidate_params = cand.get("candidate_params") or cand.get("op_params") or {}
        if not isinstance(candidate_params, Mapping):
            candidate_params = {}

        row = dict(cand)
        row.update(
            {
                "candidate_id": candidate_id,
                "source_paths": [_repo_rel(source_path, repo_root)],
                "profile": cand.get("profile") or source_profile,
                "optimizer": cand.get("optimizer") or source_optimizer,
                "optimizer_status": cand.get("optimizer_status") or source_optimizer_status,
                "optimizer_tool": payload.get("tool") or "tools/build_optimizer_guided_candidate_queue.py",
                "candidate_params": dict(candidate_params),
                "op_params": dict(cand.get("op_params") or candidate_params),
                "rank_score": _as_float(cand.get("rank_score") or cand.get("proxy_objective")),
                "rank_score_field": cand.get("rank_score_field") or "proxy_objective_not_score",
                "evidence_semantics": cand.get("evidence_semantics")
                or "offline_optimizer_guided_proxy_queue_not_exact_auth_eval",
                "evidence_grade": cand.get("evidence_grade") or "[offline-proxy-planning-only]",
            }
        )
        row = apply_proxy_evidence_boundary(
            row,
            dispatch_blockers=[
                *profile_blockers,
                "optimizer_guided_queue_requires_archive_materialization",
                "optimizer_guided_row_has_no_runtime_consumption_proof",
            ],
        )
        rows.append(row)
    return rows


def _mlx_dynamic_learned_sweep_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ranked_sources = [
        source
        for source in payload.get("ranked_sweep_rows") or []
        if isinstance(source, Mapping)
    ]
    accepted_parent_queue_ids: set[str] = set()
    for index, source in enumerate(ranked_sources):
        if source.get("schema") != "mlx_dynamic_learned_sweep_row.v1":
            continue
        _validate_mlx_dynamic_normalized_gain_sum(
            source,
            context=f"ranked_sweep_rows[{index}]",
        )
        parent_id = _mlx_dynamic_parent_queue_candidate_id(source)
        if parent_id is not None:
            accepted_parent_queue_ids.add(parent_id)
    for recipe in payload.get("optimizer_scheduler_candidates") or []:
        if not isinstance(recipe, Mapping):
            continue
        require_no_truthy_authority_fields(
            recipe,
            context="mlx_dynamic_learned_sweep_optimizer_scheduler_candidate",
        )
        if recipe.get("schema") != "optimizer_scheduler_descriptor.v1":
            continue
        descriptor_id = str(recipe.get("descriptor_id") or "")
        if not descriptor_id:
            continue
        row = {
            "candidate_id": f"optimizer_scheduler::{descriptor_id}",
            "source_paths": [_repo_rel(source_path, repo_root)],
            "lane_id": "optimizer_scheduler_registry_planning",
            "lane_class": "optimizer_scheduler_recipe",
            "candidate_family": "optimizer_scheduler_recipe",
            "optimizer_tool": payload.get("tool") or "tools/plan_mlx_dynamic_learned_sweep.py",
            "descriptor_id": descriptor_id,
            "optimizer": recipe.get("optimizer"),
            "scheduler": recipe.get("scheduler"),
            "config_sha256": recipe.get("config_sha256"),
            "parameter_group_lr_policy_id": recipe.get("parameter_group_lr_policy_id"),
            "parameter_group_lr_policy_sha256": recipe.get(
                "parameter_group_lr_policy_sha256"
            ),
            "rank_score": None,
            "rank_score_field": recipe.get("rank_score_field")
            or "planner_priority_not_score",
            "evidence_semantics": (
                "optimizer_scheduler_registry_recipe_proxy_not_exact_auth_eval"
            ),
            "evidence_grade": "[offline-proxy-planning-only]",
            "consumer_payload": {
                "schema": "optimizer_scheduler_recipe_candidate_payload.v1",
                "optimizer_scheduler_recipe": {
                    "descriptor_id": descriptor_id,
                    "optimizer": recipe.get("optimizer"),
                    "scheduler": recipe.get("scheduler"),
                    "config_sha256": recipe.get("config_sha256"),
                    "parameter_group_lr_policy": recipe.get("parameter_group_lr_policy"),
                    "parameter_group_lr_policy_sha256": recipe.get(
                        "parameter_group_lr_policy_sha256"
                    ),
                    "allowed_axis_tags": list(recipe.get("allowed_axis_tags") or []),
                    "allowed_target_modes": list(recipe.get("allowed_target_modes") or []),
                    "solver_stack_wire_in": recipe.get("solver_stack_wire_in"),
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "promotable": False,
            },
        }
        row = apply_proxy_evidence_boundary(
            row,
            dispatch_blockers=[
                "optimizer_scheduler_recipe_is_planning_only",
                "requires_training_telemetry_before_candidate_selection",
                "requires_byte_closed_archive_export_before_dispatch_readiness",
                "requires_exact_auth_eval_result_before_score_claim",
            ],
        )
        rows.append(row)
    for source in payload.get("optimizer_scheduler_pairings") or []:
        if not isinstance(source, Mapping):
            continue
        require_no_truthy_authority_fields(
            source,
            context="mlx_dynamic_learned_sweep_optimizer_scheduler_pairing",
        )
        if source.get("schema") != "mlx_dynamic_learned_sweep_optimizer_scheduler_pairing.v1":
            continue
        queue_candidate_id = str(source.get("queue_candidate_id") or "")
        if not queue_candidate_id:
            continue
        parent_queue_candidate_id = str(source.get("parent_queue_candidate_id") or "")
        if parent_queue_candidate_id not in accepted_parent_queue_ids:
            raise ValueError(
                "mlx_dynamic_learned_sweep_optimizer_scheduler_pairing "
                f"parent_queue_candidate_id not accepted: {parent_queue_candidate_id}"
            )
        row = dict(source)
        row.update(
            {
                "candidate_id": queue_candidate_id,
                "source_candidate_id": source.get("candidate_id"),
                "source_paths": [_repo_rel(source_path, repo_root)],
                "lane_id": source.get("lane_id") or "mlx_dynamic_optimizer_scheduler_pairing",
                "lane_class": source.get("family") or "optimizer_scheduler_pairing",
                "candidate_family": "optimizer_scheduler_paired_sweep_recipe",
                "optimizer_tool": payload.get("tool") or "tools/plan_mlx_dynamic_learned_sweep.py",
                "descriptor_id": source.get("optimizer_scheduler_descriptor_id"),
                "optimizer": source.get("optimizer"),
                "scheduler": source.get("scheduler"),
                "config_sha256": source.get("optimizer_scheduler_config_sha256"),
                "parameter_group_lr_policy_id": source.get("parameter_group_lr_policy_id"),
                "parameter_group_lr_policy_sha256": source.get(
                    "parameter_group_lr_policy_sha256"
                ),
                "rank_score": _as_float(source.get("rank_score")),
                "rank_score_field": source.get("rank_score_field")
                or "parent_negative_acquisition_value_plus_recipe_tiebreak_not_score",
                "evidence_semantics": (
                    "optimizer_scheduler_pairing_proxy_not_exact_auth_eval"
                ),
                "evidence_grade": payload.get("evidence_grade")
                or "[offline-proxy-planning-only]",
                "consumer_payload": {
                    "schema": "optimizer_scheduler_pairing_candidate_payload.v1",
                    "optimizer_scheduler_pairing": {
                        "parent_queue_candidate_id": source.get("parent_queue_candidate_id"),
                        "source_candidate_id": source.get("candidate_id"),
                        "sweep_config_id": source.get("sweep_config_id"),
                        "optimization_pass_id": source.get("optimization_pass_id"),
                        "optimizer_scheduler_descriptor_id": source.get(
                            "optimizer_scheduler_descriptor_id"
                        ),
                        "optimizer_scheduler_config_sha256": source.get(
                            "optimizer_scheduler_config_sha256"
                        ),
                        "parameter_group_lr_policy_id": source.get(
                            "parameter_group_lr_policy_id"
                        ),
                        "parameter_group_lr_policy_sha256": source.get(
                            "parameter_group_lr_policy_sha256"
                        ),
                        "paired_ablation_contract": source.get("paired_ablation_contract"),
                        "solver_stack_wire_in": source.get("solver_stack_wire_in"),
                    },
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "promotable": False,
                },
            }
        )
        row = apply_proxy_evidence_boundary(
            row,
            dispatch_blockers=[
                "optimizer_scheduler_pairing_is_planning_only",
                "requires_same_seed_local_ablation_before_recipe_posterior_update",
                "requires_training_telemetry_before_candidate_selection",
                "requires_byte_closed_archive_export_before_dispatch_readiness",
                "requires_exact_auth_eval_result_before_score_claim",
            ],
        )
        rows.append(row)
    for index, source in enumerate(ranked_sources):
        if not isinstance(source, Mapping):
            continue
        require_no_truthy_authority_fields(
            source,
            context="mlx_dynamic_learned_sweep_row",
        )
        _validate_mlx_dynamic_normalized_gain_sum(
            source,
            context=f"ranked_sweep_rows[{index}]",
        )
        source_candidate_id = str(source.get("candidate_id") or "")
        config_id = str(source.get("sweep_config_id") or "")
        pass_id = str(source.get("optimization_pass_id") or "")
        if not source_candidate_id or not config_id or not pass_id:
            continue
        acquisition_value = _as_float(source.get("acquisition_value"))
        row = dict(source)
        row.update(
            {
                "candidate_id": f"{source_candidate_id}::{config_id}::{pass_id}",
                "source_candidate_id": source_candidate_id,
                "source_paths": [_repo_rel(source_path, repo_root)],
                "lane_id": source.get("lane_id") or "mlx_dynamic_learned_sweep_planning",
                "lane_class": source.get("family") or "mlx_dynamic_learned_sweep",
                "candidate_family": source.get("family") or "mlx_dynamic_learned_sweep",
                "optimizer_tool": payload.get("tool") or "tools/plan_mlx_dynamic_learned_sweep.py",
                "rank_score": -acquisition_value if acquisition_value is not None else None,
                "rank_score_field": "negative_acquisition_value_proxy_not_score",
                "evidence_semantics": (
                    "mlx_dynamic_learned_sweep_plan_proxy_not_exact_auth_eval"
                ),
                "evidence_grade": payload.get("evidence_grade")
                or "[macOS-MLX research-signal]",
                "consumer_payload": {
                    "schema": "mlx_dynamic_learned_sweep_candidate_payload.v1",
                    "mlx_dynamic_learned_sweep": {
                        "source_candidate_id": source_candidate_id,
                        "sweep_config_id": config_id,
                        "optimization_pass_id": pass_id,
                        "family": source.get("family"),
                        "component_axis_context": source.get("component_axis_context"),
                        "canonical_equation_provenance": source.get(
                            "canonical_equation_provenance"
                        ),
                        "master_gradient_provenance": source.get(
                            "master_gradient_provenance"
                        ),
                        "acquisition_value": acquisition_value,
                        "ready_for_local_sweep": source.get("ready_for_local_sweep"),
                        "exact_eval_candidate": source.get("exact_eval_candidate"),
                    },
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "promotable": False,
                },
            }
        )
        row = apply_proxy_evidence_boundary(
            row,
            dispatch_blockers=[
                "mlx_dynamic_learned_sweep_plan_is_proxy_signal",
                "requires_observation_append_before_promotion",
                "requires_byte_closed_materialization_before_exact_eval",
                "requires_lane_claim_before_dispatch",
            ],
        )
        rows.append(row)
    return rows


def _local_cpu_drift_eureka_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    candidate_id = str(payload.get("candidate_id") or "")
    if not candidate_id:
        return []
    try:
        require_eureka_false_authority(
            payload,
            context=f"{source_path} local CPU eureka",
        )
    except LocalCPUContestDriftError as exc:
        raise ValueError(str(exc)) from exc
    require_no_truthy_authority_fields(payload, context="local_cpu_drift_eureka")
    eureka_trigger = payload.get("eureka_trigger") is True
    recommended_action = str(payload.get("recommended_action") or "")
    rank_score = _as_float(
        payload.get("conservative_projected_contest_score")
        or payload.get("projected_contest_score")
        or payload.get("local_score")
    )
    row = {
        "candidate_id": f"{candidate_id}::local_cpu_contest_drift_eureka",
        "source_candidate_id": candidate_id,
        "source_paths": [_repo_rel(source_path, repo_root)],
        "lane_id": f"local_cpu_drift_eureka_{candidate_id}",
        "lane_class": "local_cpu_contest_drift_eureka",
        "candidate_family": "dqs1_local_cpu_drift_eureka",
        "candidate_archive_sha256": _shaish(payload.get("candidate_archive_sha256")),
        "local_score": _as_float(payload.get("local_score")),
        "projected_contest_score": _as_float(payload.get("projected_contest_score")),
        "conservative_projected_contest_score": _as_float(
            payload.get("conservative_projected_contest_score")
        ),
        "auth_frontier_score": _as_float(payload.get("auth_frontier_score")),
        "eureka_margin": _as_float(payload.get("eureka_margin")),
        "eureka_trigger": eureka_trigger,
        "recommended_action": recommended_action,
        "exact_auth_anchor_requested": (
            eureka_trigger and recommended_action == "dispatch_exact_auth_anchor"
        ),
        "rank_score": rank_score,
        "rank_score_field": "conservative_projected_contest_score_false_authority",
        "queue_priority": "operator_decision" if eureka_trigger else "observe_only",
        "evidence_semantics": "local_cpu_drift_eureka_spend_triage_not_score_authority",
        "evidence_grade": "[contest-CPU drift-projected; false authority]",
        "consumer_payload": {
            "schema": "local_cpu_drift_eureka_candidate_payload.v1",
            "source_schema": payload.get("schema"),
            "candidate_id": candidate_id,
            "source_artifact": payload.get("source_artifact"),
            "candidate_trust_region_blockers": list(
                payload.get("candidate_trust_region_blockers") or []
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        },
    }
    row = apply_proxy_evidence_boundary(
        row,
        dispatch_blockers=[
            "local_cpu_drift_eureka_is_spend_triage_only",
            "exact_auth_anchor_requires_lane_claim_and_dispatch",
            "exact_auth_eval_result_required_before_score_claim",
            *(
                ["positive_eureka_requires_manual_exact_auth_anchor_claim"]
                if eureka_trigger
                else ["observe_only_eureka_not_dispatch_candidate"]
            ),
        ],
    )
    return [row]


def _byte_shaving_campaign_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    require_no_truthy_authority_fields(payload, context="byte_shaving_campaign_plan")
    campaign_id = str(payload.get("campaign_id") or "byte_shaving_campaign")
    lane_id = str(payload.get("lane_id") or "byte_shaving_campaign")
    source_candidate_id = str(payload.get("candidate_id") or campaign_id)
    source_refs = list(payload.get("source_signal_refs") or [])
    auth_eval_refs = list(payload.get("auth_eval_refs") or [])
    mlx_calibration_refs = list(payload.get("mlx_calibration_refs") or [])
    scorer_response_refs = list(payload.get("scorer_response_refs") or [])
    plan_blockers = [str(item) for item in payload.get("dispatch_blockers", []) if str(item)]
    rows: list[dict[str, Any]] = []

    def append_plan_row(kind: str, item: Mapping[str, Any]) -> None:
        row_id = str(item.get("combo_id") or item.get("sweep_id") or "")
        if not row_id:
            return
        require_no_truthy_authority_fields(
            item,
            context=f"byte_shaving_campaign_plan.{kind}.{row_id}",
        )
        selected_operations = list(item.get("selected_operations") or [])
        selected_unit_ids = [str(value) for value in item.get("selected_unit_ids", []) if str(value)]
        candidate_id = f"{campaign_id}::{kind}::{row_id}"
        expected_delta = _as_float(item.get("expected_delta_score"))
        candidate_saved_bytes = _as_int(item.get("candidate_saved_bytes"))
        serialized_archive_delta = build_serialized_archive_delta_contract(
            modeled_saved_bytes=candidate_saved_bytes,
            require_realized_saving=True,
        )
        row = {
            "candidate_id": candidate_id,
            "source_candidate_id": source_candidate_id,
            "source_paths": [_repo_rel(source_path, repo_root)],
            "lane_id": lane_id,
            "lane_class": "byte_shaving_campaign",
            "candidate_family": "post_training_byte_shaving_plan",
            "param_schema": "byte_shaving_campaign_operation_selection_v1",
            "optimizer_tool": payload.get("tool") or "tools/plan_byte_shaving_campaign.py",
            "selection_kind": kind,
            "selection_id": row_id,
            "candidate_params": {
                "selection_kind": kind,
                "selection_id": row_id,
                "selected_unit_ids": selected_unit_ids,
                "selected_operations": selected_operations,
                "active_interactions": list(item.get("active_interactions") or []),
            },
            "op_params": {
                "selected_operations": selected_operations,
                "operation_families": list(item.get("operation_families") or []),
            },
            "selected_unit_ids": selected_unit_ids,
            "selected_operations": selected_operations,
            "active_interactions": list(item.get("active_interactions") or []),
            "operation_families": list(item.get("operation_families") or []),
            "unit_count": _as_int(item.get("unit_count")),
            "candidate_saved_bytes": candidate_saved_bytes,
            "predicted_saved_bytes": candidate_saved_bytes,
            "predicted_saved_bytes_semantics": (
                "planner_model_only_not_serialized_archive_delta"
            ),
            "serialized_archive_delta": serialized_archive_delta,
            "expected_delta_score": expected_delta,
            "expected_score_gain": _as_float(item.get("expected_score_gain")),
            "confidence": _as_float(item.get("confidence")),
            "confidence_adjusted_gain": _as_float(item.get("confidence_adjusted_gain")),
            "rank_score": expected_delta,
            "rank_score_field": "expected_delta_score_planning_only",
            "evidence_semantics": (
                "byte_shaving_campaign_plan_proxy_not_exact_auth_eval"
            ),
            "evidence_grade": payload.get("frontier_axis") or "[planning-only]",
            "planned_score_affecting_payload_change": True,
            "source_signal_refs": source_refs,
            "auth_eval_refs": auth_eval_refs,
            "mlx_calibration_refs": mlx_calibration_refs,
            "scorer_response_refs": scorer_response_refs,
            "consumer_payload": {
                "schema": "byte_shaving_campaign_candidate_payload.v1",
                "campaign_id": campaign_id,
                "selection_kind": kind,
                "selection_id": row_id,
                "source_candidate_id": source_candidate_id,
                "selected_unit_ids": selected_unit_ids,
                "selected_operations": selected_operations,
                "active_interactions": list(item.get("active_interactions") or []),
                "source_signal_refs": source_refs,
                "auth_eval_refs": auth_eval_refs,
                "mlx_calibration_refs": mlx_calibration_refs,
                "scorer_response_refs": scorer_response_refs,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "promotable": False,
            },
        }
        row = apply_proxy_evidence_boundary(
            row,
            dispatch_blockers=[
                *plan_blockers,
                *[str(item) for item in item.get("dispatch_blockers", []) if str(item)],
                *serialized_archive_delta_blockers(serialized_archive_delta),
                "byte_shaving_campaign_plan_is_planning_only",
                "selected_operations_require_materializer",
                "materialized_archive_runtime_custody_required",
                "locality_controls_required_before_exact_eval",
                "exact_auth_eval_result_required_before_score_claim",
            ],
        )
        rows.append(row)

    for item in payload.get("sweep_ladder") or []:
        if isinstance(item, Mapping):
            append_plan_row("prefix", item)
    for item in payload.get("combination_ladder") or []:
        if isinstance(item, Mapping):
            append_plan_row("combo", item)
    return rows


def _kaggle_proxy_sweep_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    best = payload.get("best_candidate")
    if not isinstance(best, Mapping):
        return []
    candidate_id = str(best.get("candidate_id") or "kaggle_proxy_best")
    row = _base_candidate(candidate_id, source_path=source_path, repo_root=repo_root)
    proxy_objective = _as_float(best.get("proxy_objective"))
    row.update(
        {
            "lane_id": payload.get("lane_id") or "kaggle_pr101_proxy_sweep",
            "lane_class": payload.get("lane_class")
            or best.get("lane_class")
            or "pr101_kaggle_proxy_sweep",
            "candidate_family": payload.get("candidate_family")
            or best.get("candidate_family")
            or "pr101_proxy_config_search",
            "param_schema": payload.get("param_schema") or best.get("param_schema"),
            "optimizer_tool": payload.get("tool") or "tools/build_kaggle_proxy_sweep_kernel.py",
            "optimizer": payload.get("optimizer") or best.get("optimizer"),
            "optimizer_status": payload.get("optimizer_status") or best.get("optimizer_status"),
            "trial_index": _as_int(best.get("trial_index")),
            "op_params": dict(best.get("params") or {}),
            "proxy_components": dict(best.get("proxy_components") or {}),
            "proxy_score": proxy_objective,
            "rank_score": proxy_objective,
            "rank_score_field": "proxy_objective",
            "evidence_semantics": payload.get("evidence_semantics")
            or "kaggle_gpu_proxy_config_search_only_not_exact_auth_eval",
            "evidence_grade": "[Kaggle-proxy-only]",
            "proxy_only": True,
            "source_manifest_path": _repo_rel(source_path, repo_root),
        }
    )
    _add_blockers(
        row,
        [
            *[
                str(item)
                for item in payload.get("dispatch_blockers", [])
                if str(item)
            ],
            "kaggle_proxy_output_requires_archive_builder_promotion",
            "kaggle_proxy_result_is_not_rank_or_kill_evidence",
        ],
    )
    return [row]


def _pr101_kaggle_proxy_runtime_packet_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    packet_archive = payload.get("packet_archive")
    runtime_custody = payload.get("runtime_custody")
    runtime_patch = payload.get("runtime_patch")
    if not isinstance(packet_archive, Mapping) or not isinstance(runtime_custody, Mapping):
        return []

    candidate_id = str(payload.get("candidate_id") or "")
    if not candidate_id:
        return []
    packet_dir_value = payload.get("packet_dir")
    if isinstance(packet_dir_value, str) and packet_dir_value:
        packet_dir = Path(packet_dir_value)
        if not packet_dir.is_absolute():
            packet_dir = repo_root / packet_dir
    else:
        packet_dir = source_path.parent
    archive_rel = str(packet_archive.get("relpath") or "archive.zip")
    archive_path = packet_dir / archive_rel
    archive_sha = _shaish(packet_archive.get("sha256"))
    archive_bytes = _as_int(packet_archive.get("bytes"))
    source_archive = payload.get("source_archive")
    source_archive_sha = (
        _shaish(source_archive.get("sha256"))
        if isinstance(source_archive, Mapping)
        else _shaish(payload.get("archive_unchanged_sha256"))
    )
    source_archive_bytes = (
        _as_int(source_archive.get("bytes"))
        if isinstance(source_archive, Mapping)
        else None
    )
    runtime_tree_sha = _shaish(runtime_custody.get("runtime_tree_sha256"))
    runtime_consumption_proof_path = source_path.parent / "runtime_consumption_proof.json"
    runtime_consumption_proof_present = runtime_consumption_proof_path.is_file()
    row = _base_candidate(
        f"{candidate_id}_pr101_proxy_runtime_packet",
        source_path=source_path,
        repo_root=repo_root,
    )
    row.update(
        {
            "lane_id": "pr101_kaggle_proxy_runtime_packet_exact_eval",
            "lane_class": "pr101_kaggle_proxy_runtime_packet",
            "candidate_family": "pr101_inflate_time_bias_runtime_packet",
            "source_candidate_id": candidate_id,
            "submission_dir": _repo_rel(packet_dir, repo_root),
            "archive_path": _repo_rel(archive_path, repo_root),
            "candidate_archive_path": _repo_rel(archive_path, repo_root),
            "archive_sha256": archive_sha,
            "candidate_archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "archive_size_bytes": archive_bytes,
            "candidate_archive_bytes": archive_bytes,
            "source_archive_sha256": source_archive_sha,
            "source_archive_bytes": source_archive_bytes,
            "archive_unchanged_from_source": bool(payload.get("archive_changed") is False),
            "runtime_tree_sha256": runtime_tree_sha,
            "runtime_manifest_path": _repo_rel(source_path, repo_root),
            "runtime_consumption_proof_required": True,
            "runtime_consumption_proof_path": _repo_rel(
                runtime_consumption_proof_path,
                repo_root,
            ),
            "runtime_consumption_proof_status": "present"
            if runtime_consumption_proof_present
            else "missing",
            "candidate_params": dict(payload.get("runtime_consumed_params") or {}),
            "runtime_patch": dict(runtime_patch) if isinstance(runtime_patch, Mapping) else {},
            "score_affecting_payload_changed": False,
            "charged_bits_changed": False,
            "score_affecting_runtime_changed": True,
            "evidence_semantics": "byte_closed_proxy_runtime_packet_pending_exact_eval",
            "evidence_grade": "[byte-closed-runtime-packet-no-score]",
            "source_manifest_path": _repo_rel(source_path, repo_root),
        }
    )
    _add_blockers(
        row,
        [
            "exact_cuda_auth_eval_missing",
            "requires_exact_eval_readiness_gate",
            "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
            *([] if runtime_consumption_proof_present else ["runtime_consumption_proof_missing"]),
        ],
    )
    return [row]


def _archive_record_sha(record: Any) -> str | None:
    if isinstance(record, Mapping):
        return _shaish(record.get("archive_sha256") or record.get("sha256"))
    return None


def _archive_record_bytes(record: Any) -> int | None:
    if isinstance(record, Mapping):
        return _as_int(record.get("archive_bytes") or record.get("bytes"))
    return None


def _hnerv_lowlevel_exact_eval_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    candidate_id = str(payload.get("candidate_id") or "")
    if not candidate_id:
        return []
    lane_id = str(payload.get("lane_id") or candidate_id)
    release_surface = source_path.parent
    if source_path.name != "archive_manifest.json":
        release_surface = source_path.parent / "release_surface"
    archive_path = release_surface / "archive.zip"
    source_archive_bytes = _as_int(payload.get("source_archive_bytes"))
    source_archive_sha = _shaish(payload.get("source_archive_sha256"))
    row = _base_candidate(candidate_id, source_path=source_path, repo_root=repo_root)
    archive_sha = _shaish(
        payload.get("candidate_archive_sha256")
        or payload.get("archive_sha256")
        or (payload.get("archive") or {}).get("sha256")
    )
    archive_bytes = _as_int(
        payload.get("candidate_archive_bytes")
        or payload.get("archive_bytes")
        or (payload.get("archive") or {}).get("bytes")
    )
    row.update(
        {
            "lane_id": lane_id,
            "lane_class": "hnerv_lowlevel_exact_eval_candidate",
            "candidate_family": "hnerv_lowlevel_byte_repack",
            "submission_dir": _repo_rel(release_surface, repo_root),
            "archive_path": _repo_rel(archive_path, repo_root),
            "candidate_archive_path": _repo_rel(archive_path, repo_root),
            "archive_sha256": archive_sha,
            "candidate_archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "archive_size_bytes": archive_bytes,
            "candidate_archive_bytes": archive_bytes,
            "source_archive_sha256": source_archive_sha,
            "source_archive_bytes": source_archive_bytes,
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "evidence_semantics": "byte_closed_hnerv_lowlevel_candidate_pending_exact_eval",
            "evidence_grade": "[byte-closed-no-score]",
            "source_manifest_path": _repo_rel(source_path, repo_root),
        }
    )
    _add_blockers(
        row,
        [
            "exact_cuda_auth_eval_missing",
            "requires_exact_eval_readiness_gate",
            "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
        ],
    )
    return [row]


def _pr103_hidden_gem_candidates(
    payload: Mapping[str, Any], *, source_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    candidate_id = str(payload.get("candidate_id") or "")
    if not candidate_id:
        return []
    candidate_archive = payload.get("candidate_archive")
    source_archive = payload.get("source_archive")
    release_surface = source_path.parent
    if source_path.name != "archive_manifest.json":
        release_surface = source_path.parent / "release_surface"
    archive_path = release_surface / "archive.zip"
    row = _base_candidate(candidate_id, source_path=source_path, repo_root=repo_root)
    archive_sha = _archive_record_sha(candidate_archive)
    archive_bytes = _archive_record_bytes(candidate_archive)
    row.update(
        {
            "lane_id": str(payload.get("lane_id") or "pr103_ac_hidden_gem"),
            "lane_class": "pr103_ac_hidden_gem",
            "candidate_family": "pr103_ac_byte_hidden_gem",
            "submission_dir": _repo_rel(release_surface, repo_root),
            "archive_path": _repo_rel(archive_path, repo_root),
            "candidate_archive_path": _repo_rel(archive_path, repo_root),
            "archive_sha256": archive_sha,
            "candidate_archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "archive_size_bytes": archive_bytes,
            "candidate_archive_bytes": archive_bytes,
            "source_archive_sha256": _archive_record_sha(source_archive),
            "source_archive_bytes": _archive_record_bytes(source_archive),
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "runtime_consumed_section_changed": bool(
                (payload.get("section_sha256_proof") or {}).get(
                    "runtime_consumed_section_changed"
                )
            ),
            "decoded_state_changed": bool(
                (payload.get("runtime_consumption_no_op_proof") or {}).get(
                    "state_dict_changed_vs_source"
                )
            ),
            "evidence_semantics": "byte_closed_pr103_hidden_gem_pending_exact_eval",
            "evidence_grade": "[byte-closed-no-score]",
            "source_manifest_path": _repo_rel(source_path, repo_root),
        }
    )
    _add_blockers(
        row,
        [
            "exact_cuda_auth_eval_missing",
            "requires_exact_eval_readiness_gate",
            "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
        ],
    )
    return [row]


def extract_candidates_from_source(path: Path, *, repo_root: Path) -> SourceExtraction:
    payload = _load_json(path)
    schema = _source_schema(payload)
    if not isinstance(payload, Mapping):
        return SourceExtraction(schema=schema, rows=[])
    if schema == "constrained_coord_search_rollup_v1":
        return SourceExtraction(
            schema=schema,
            rows=_a1_rollup_candidates(payload, source_path=path, repo_root=repo_root),
        )
    if payload.get("tool") == "tools/sweep_m5max_hnerv_cluster.py":
        return SourceExtraction(
            schema=schema,
            rows=_m5max_candidates(payload, source_path=path, repo_root=repo_root),
        )
    if schema in {"codec_op_cma_search_report_v1", "codec_op_optuna_search_report_v1"}:
        return SourceExtraction(
            schema=schema,
            rows=_codec_search_report_candidates(payload, source_path=path, repo_root=repo_root),
        )
    if schema == "codec_op_param_sweep_manifest.v1":
        return SourceExtraction(
            schema=schema,
            rows=_codec_param_manifest_candidates(payload, source_path=path, repo_root=repo_root),
        )
    if schema == "meta_lagrangian_search_v1":
        return SourceExtraction(
            schema=schema,
            rows=_meta_lagrangian_candidates(payload, source_path=path, repo_root=repo_root),
        )
    if schema == "optimizer_guided_candidate_queue_v1":
        return SourceExtraction(
            schema=schema,
            rows=_optimizer_guided_queue_candidates(
                payload,
                source_path=path,
                repo_root=repo_root,
            ),
        )
    if schema == "mlx_dynamic_learned_sweep_plan.v1":
        return SourceExtraction(
            schema=schema,
            rows=_mlx_dynamic_learned_sweep_candidates(
                payload,
                source_path=path,
                repo_root=repo_root,
            ),
        )
    if schema == "byte_shaving_campaign_plan.v1":
        return SourceExtraction(
            schema=schema,
            rows=_byte_shaving_campaign_candidates(
                payload,
                source_path=path,
                repo_root=repo_root,
            ),
        )
    if schema == EUREKA_SIGNAL_SCHEMA:
        return SourceExtraction(
            schema=schema,
            rows=_local_cpu_drift_eureka_candidates(
                payload,
                source_path=path,
                repo_root=repo_root,
            ),
        )
    if schema in {PR95_LOCAL_TRAINING_SCHEMA, PR95_LOCAL_TRAINING_PLAN_SCHEMA}:
        return SourceExtraction(
            schema=schema,
            rows=[
                adapt_pr95_local_training_manifest_to_candidate(
                    payload,
                    source_path=path,
                    repo_root=repo_root,
                )
            ],
        )
    if schema in {REPRESENTATION_TRAINING_SCHEMA, REPRESENTATION_TRAINING_PLAN_SCHEMA}:
        return SourceExtraction(
            schema=schema,
            rows=[
                adapt_representation_training_manifest_to_candidate(
                    payload,
                    source_path=path,
                    repo_root=repo_root,
                )
            ],
        )
    if schema == TRAINER_RUNTIME_PROFILE_SCHEMA:
        return SourceExtraction(
            schema=schema,
            rows=[
                adapt_runtime_profile_observation_to_candidate(
                    payload,
                    source_path=path,
                    repo_root=repo_root,
                )
            ],
        )
    if schema == "pr101_kaggle_proxy_sweep_v1":
        return SourceExtraction(
            schema=schema,
            rows=_kaggle_proxy_sweep_candidates(payload, source_path=path, repo_root=repo_root),
        )
    if schema == "pr101_kaggle_proxy_runtime_packet_v1":
        return SourceExtraction(
            schema=schema,
            rows=_pr101_kaggle_proxy_runtime_packet_candidates(
                payload,
                source_path=path,
                repo_root=repo_root,
            ),
        )
    if schema in {
        "hnerv_lowlevel_exact_eval_candidate_manifest_v1",
        "hnerv_lowlevel_exact_eval_operator_packet_v1",
        "hnerv_lowlevel_release_surface_manifest_v1",
    }:
        return SourceExtraction(
            schema=schema,
            rows=_hnerv_lowlevel_exact_eval_candidates(
                payload,
                source_path=path,
                repo_root=repo_root,
            ),
        )
    if (
        schema == "pr103_hidden_gem_release_surface_manifest_v1"
        or payload.get("tool") == "inline_local_materialize_pr103_ac_hidden_gem_candidate"
    ):
        return SourceExtraction(
            schema=schema,
            rows=_pr103_hidden_gem_candidates(
                payload,
                source_path=path,
                repo_root=repo_root,
            ),
        )
    if isinstance(payload.get("candidates"), list):
        return SourceExtraction(
            schema=schema,
            rows=[],
            unsupported_reason=UNKNOWN_CANDIDATES_REASON,
        )
    return SourceExtraction(schema=schema, rows=[])


def build_candidate_queue(
    source_paths: Iterable[Path],
    *,
    repo_root: Path,
    top_k: int | None = None,
) -> dict[str, Any]:
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    source_schemas: list[dict[str, Any]] = []
    unsupported_sources: list[dict[str, str]] = []
    for path in source_paths:
        extraction = extract_candidates_from_source(path, repo_root=repo_root)
        source_entry: dict[str, Any] = {
            "path": _repo_rel(path, repo_root),
            "schema": extraction.schema,
            "extracted_candidate_count": len(extraction.rows),
        }
        if extraction.unsupported_reason:
            source_entry["status"] = "unsupported"
            source_entry["unsupported_reason"] = extraction.unsupported_reason
            unsupported_sources.append({
                "path": _repo_rel(path, repo_root),
                "schema": extraction.schema,
                "reason": extraction.unsupported_reason,
            })
        else:
            source_entry["status"] = "supported"
        source_schemas.append(source_entry)
        for row in extraction.rows:
            cid = str(row.get("candidate_id") or "")
            if not cid:
                continue
            identity_key = _candidate_identity_key(row)
            if identity_key in merged:
                merged[identity_key] = _merge_candidate(merged[identity_key], row)
            else:
                merged[identity_key] = row

    for row in merged.values():
        _annotate_archive_candidate_verification(row, repo_root)

    sorted_rows = sorted(merged.values(), key=_candidate_sort_key)
    if top_k is not None:
        sorted_rows = sorted_rows[:top_k]
    dispatch_ready = [
        row for row in sorted_rows if row.get("ready_for_exact_eval_dispatch") is True
    ]
    return _json_safe({
        "schema": QUEUE_SCHEMA,
        "tool": TOOL_NAME,
        "generated_at_utc": _utc_now(),
        "source_schemas": source_schemas,
        "unsupported_sources": unsupported_sources,
        "n_candidates": len(merged),
        "top_k_count": len(sorted_rows),
        "dispatch_ready_count": len(dispatch_ready),
        "dispatch_ready": dispatch_ready,
        "top_k": sorted_rows,
        "top_k_forensic": sorted_rows,
        "evidence_boundary": {
            "planning_only_by_default": True,
            "ready_for_exact_eval_dispatch_default": False,
            "proxy_or_macos_cpu_rows_must_not_promote_score": True,
            "next_gate": "materialize byte-closed archive/runtime custody, then exact eval readiness gate",
        },
    })


__all__ = [
    "BASE_DISPATCH_BLOCKERS",
    "QUEUE_SCHEMA",
    "build_candidate_queue",
    "extract_candidates_from_source",
]
