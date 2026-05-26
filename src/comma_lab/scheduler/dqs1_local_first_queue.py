from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from comma_lab.operator_storage_waterfall import operator_cold_store_roots
from comma_lab.storage_tiers import DEFAULT_RESERVE_FREE_GB
from tac.optimization.decoder_q_constants import FEC6_PAIR_COUNT
from tac.optimization.dqs1_materializer_feedback_bridge import (
    DQS1_OBSERVATION_SOURCE_SCHEMA,
    DQS1_OBSERVATION_SWEEP_CONFIG_ID,
    build_dqs1_materializer_feedback_bridge,
)
from tac.optimization.local_cpu_contest_drift import (
    EUREKA_FALSE_AUTHORITY_FIELDS,
    EUREKA_SIGNAL_SCHEMA,
    LocalCPUContestDriftError,
    require_eureka_false_authority,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition
from .storage_preflight import (
    build_scheduler_storage_preflight_experiment,
    validate_scheduler_storage_preflight_config,
)

DEFAULT_QUEUE_ID = "dqs1_pairset_local_first"
DEFAULT_RESULTS_ROOT = (
    "experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb"
)
DEFAULT_MLX_EFFECTIVE_SELECTION = (
    "experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/"
    "mlx_effective_spend_triage_observed_window_selection_top32.json"
)
DEFAULT_DECODER_Q_CANDIDATE_MANIFEST = (
    "experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/"
    "contest_oracle_search/op3v3_decoder_q_selected_candidates_20260520_codex/"
    "d1f1e56e042692f2/mutation_manifest.json"
)
DEFAULT_BASE_SUBMISSION_DIR = (
    "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "submission_dir"
)
DEFAULT_GLOBAL_MUTATED_ARCHIVE = (
    "experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/"
    "contest_oracle_search/op3v3_decoder_q_selected_candidates_20260520_codex/"
    "d1f1e56e042692f2/archive.zip"
)
DEFAULT_UPSTREAM_DIR = "upstream"
DEFAULT_VIDEO_NAMES_FILE = "upstream/public_test_video_names.txt"
DEFAULT_FRAME_POLICY = "pair_all_frames"
DEFAULT_DRIFT_CALIBRATION_JSON = (
    ".omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json"
)
DEFAULT_EUREKA_OUTPUT_DIR = ".omx/research"
DEFAULT_MLX_REFERENCE_CACHE_DIR = "experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"
DEFAULT_SCHEDULER_PREFLIGHT_EXPERIMENT_ID = "dqs1_scheduler_preflight"
SAFE_OPERATOR_ACTION = "materialize_pairset_archive_and_run_local_controls"
LOCAL_CPU_CONTEST_DRIFT_EUREKA_SCHEMA = EUREKA_SIGNAL_SCHEMA
REPO_ROOT = Path(__file__).resolve().parents[3]

_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotable",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "dispatch_attempted",
    "gpu_launched",
)
_REQUIRED_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
)
_TIMESTAMP_RE = re.compile(r"(20\d{6}T\d{4,6}Z)")
_DROP_ONE_CANDIDATE_RE = re.compile(r"^pairset_drop_one_(rank\d{3}_pair\d{4})$")
_PAIRSET_CANDIDATE_RE = re.compile(r"^pairset_[a-z0-9][a-z0-9_]*$")
_SAFE_ACTION_SUMMARY_SCHEMAS = {
    "pairset_component_marginal_canonicalization_summary.v1",
    "cross_family_candidate_portfolio_action_summary.v1",
}
PAIR_FRAME_GEOMETRY_QUEUE_REQUEST_SCHEMA = (
    "pair_frame_geometry_queue_executable_drop_request.v1"
)
SELECTED_PAIRSET_ACQUISITION_SCHEMA = "dqs1_selected_pairset_acquisition.v1"


@dataclass(frozen=True)
class Dqs1QueueSelection:
    candidate_id: str
    candidate_slug: str
    selected_pair_indices: tuple[int, ...]
    action_summary_path: Path
    portfolio_path: Path
    operator_action_rank: int | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    skipped_candidates: tuple[dict[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Dqs1QueueBuildResult:
    queue: dict[str, Any]
    selection: Dqs1QueueSelection
    selections: tuple[Dqs1QueueSelection, ...] = field(default_factory=tuple)
    materializer_feedback_bridge: dict[str, Any] | None = None
    selected_pairset_acquisition: dict[str, Any] | None = None


def _json_load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ExperimentQueueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ExperimentQueueError(f"{path}: JSON root must be an object")
    return payload


def _timestamp_key(path: Path) -> tuple[str, float, str]:
    match = _TIMESTAMP_RE.search(str(path))
    timestamp = match.group(1) if match else ""
    return (timestamp, path.stat().st_mtime, str(path))


def _action_summary_embedded_timestamp(summary: Mapping[str, Any]) -> str:
    for key in ("generated_at_utc", "created_at_utc", "captured_at_utc", "stamp"):
        value = summary.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = re.sub(r"[^0-9]", "", value)
        if len(normalized) >= 12:
            return normalized
    return ""


def _action_summary_sort_key(path: Path, summary: Mapping[str, Any]) -> tuple[str, str, float, str]:
    path_timestamp, mtime, path_key = _timestamp_key(path)
    return (_action_summary_embedded_timestamp(summary), path_timestamp, mtime, path_key)


def find_latest_cross_family_action_summary(repo_root: str | Path) -> Path:
    repo = Path(repo_root)
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for root in (repo / "experiments" / "results", repo / ".omx" / "research"):
        if not root.exists():
            continue
        for path in root.rglob("action_summary.json"):
            try:
                summary = _json_load(path)
            except ExperimentQueueError:
                continue
            if _is_dqs1_safe_action_summary(summary):
                candidates.append((path, summary))
    if not candidates:
        raise ExperimentQueueError("no DQS1-safe cross-family action_summary.json files found")
    return max(candidates, key=lambda item: _action_summary_sort_key(item[0], item[1]))[0]


def candidate_slug(candidate_id: str) -> str:
    drop_one_match = _DROP_ONE_CANDIDATE_RE.match(candidate_id)
    if drop_one_match:
        return f"drop_{drop_one_match.group(1)}"
    if not _PAIRSET_CANDIDATE_RE.match(candidate_id):
        raise ExperimentQueueError(f"unsupported DQS1 pairset candidate id: {candidate_id!r}")
    return candidate_id.removeprefix("pairset_")


def _candidate_priority(candidate_id: str, operator_action_rank: int | None) -> int:
    if operator_action_rank is not None:
        return operator_action_rank
    for pattern in (r"rank(\d{3})", r"_r(\d{3})(?:_|$)", r"_k(\d{3})(?:_|$)"):
        match = re.search(pattern, candidate_id)
        if match:
            return int(match.group(1))
    return 100


def _scheduler_preflight_experiment(
    *,
    repo_root: str | Path,
    date: str,
    results_root: str,
    storage_tiers: tuple[str, ...] = (),
    storage_workload_subdir: str | None = None,
    storage_expected_workload_root: str | None = None,
    storage_reserve_free_gb: float = DEFAULT_RESERVE_FREE_GB,
    storage_expected_bytes: int = 0,
    proactive_cleanup_roots: tuple[str, ...] = (),
    proactive_cleanup_execute: bool = False,
    proactive_cleanup_action: str = "move",
    proactive_cleanup_min_bytes: str = "1",
    proactive_cleanup_cold_store_roots: tuple[str, ...] = (),
    proactive_cleanup_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
) -> dict[str, Any]:
    return build_scheduler_storage_preflight_experiment(
        experiment_id=DEFAULT_SCHEDULER_PREFLIGHT_EXPERIMENT_ID,
        lane_id=f"lane_dqs1_scheduler_preflight_{date}",
        tags=["dqs1", "scheduler-preflight", "storage", "cleanup", "no-score-authority"],
        artifact_prefix="dqs1_local_first",
        date=date,
        results_root=results_root,
        repo_root=repo_root,
        storage_tiers=storage_tiers,
        storage_workload_subdir=storage_workload_subdir,
        storage_expected_workload_root=storage_expected_workload_root,
        storage_reserve_free_gb=storage_reserve_free_gb,
        storage_expected_bytes=storage_expected_bytes,
        proactive_cleanup_roots=proactive_cleanup_roots,
        proactive_cleanup_execute=proactive_cleanup_execute,
        proactive_cleanup_action=proactive_cleanup_action,
        proactive_cleanup_min_bytes=proactive_cleanup_min_bytes,
        proactive_cleanup_cold_store_roots=proactive_cleanup_cold_store_roots,
        proactive_cleanup_cold_store_reserve_gb=proactive_cleanup_cold_store_reserve_gb,
    )


def _resolve_output_root(path_value: str, *, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def _path_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _expected_dqs1_workload_root(
    *,
    repo_root: Path,
    results_root: str,
    expected_workload_root: str | None,
) -> Path | None:
    if expected_workload_root is not None:
        return _resolve_output_root(expected_workload_root, repo_root=repo_root)
    path = Path(results_root).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return None


def _require_dqs1_outputs_under_storage_root(
    *,
    repo_root: Path,
    results_root: str,
    expected_workload_root: str | None,
) -> None:
    root = _expected_dqs1_workload_root(
        repo_root=repo_root,
        results_root=results_root,
        expected_workload_root=expected_workload_root,
    )
    if root is None:
        raise ExperimentQueueError(
            "scheduler_storage_expected_workload_root is required when "
            "results_root is relative and scheduler preflight gates DQS1 queue execution"
        )
    resolved_results_root = _resolve_output_root(results_root, repo_root=repo_root)
    if not _path_under_root(resolved_results_root, root):
        raise ExperimentQueueError(
            "DQS1 results_root outside scheduler workload root: "
            f"{resolved_results_root} not under {root}"
        )


def _require_false_authority(
    row: dict[str, Any],
    *,
    label: str,
    require_all: bool = False,
) -> None:
    required = _FALSE_AUTHORITY_FIELDS if require_all else _REQUIRED_FALSE_AUTHORITY_FIELDS
    bad_fields = [
        field
        for field in required
        if field not in row or row.get(field) is not False
    ]
    bad_fields.extend(
        field
        for field in _FALSE_AUTHORITY_FIELDS
        if field not in required and field in row and row.get(field) is not False
    )
    if bad_fields:
        raise ExperimentQueueError(
            f"{label} must set false-authority field(s) exactly false: {', '.join(bad_fields)}"
        )


def _is_dqs1_safe_action_summary(summary: dict[str, Any]) -> bool:
    if summary.get("schema") not in _SAFE_ACTION_SUMMARY_SCHEMAS:
        return False
    try:
        _require_false_authority(summary, label="action summary", require_all=True)
    except ExperimentQueueError:
        return False
    actions = summary.get("top_operator_actions")
    if not isinstance(actions, list):
        return False
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("operator_next_action") != SAFE_OPERATOR_ACTION:
            continue
        candidate_id_value = action.get("candidate_id")
        if not isinstance(candidate_id_value, str) or not _PAIRSET_CANDIDATE_RE.match(candidate_id_value):
            continue
        try:
            _require_false_authority(
                action,
                label=f"{candidate_id_value} top action",
                require_all=True,
            )
        except ExperimentQueueError:
            continue
        return True
    return False


def _resolve_portfolio_path(
    action_summary_path: Path,
    summary: dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    portfolio_json = summary.get("portfolio_json") or summary.get("json_out")
    if not isinstance(portfolio_json, str) or not portfolio_json.strip():
        raise ExperimentQueueError(f"{action_summary_path}: missing portfolio_json/json_out")
    portfolio_path = Path(portfolio_json)
    candidates = [portfolio_path] if portfolio_path.is_absolute() else [
        repo_root / portfolio_path,
        action_summary_path.parent / portfolio_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ExperimentQueueError(f"{action_summary_path}: portfolio_json not found: {portfolio_json}")


def _load_portfolio_row(portfolio_path: Path, candidate_id: str) -> dict[str, Any]:
    portfolio = _json_load(portfolio_path)
    rows = portfolio.get("operator_action_rows")
    if not isinstance(rows, list):
        raise ExperimentQueueError(f"{portfolio_path}: operator_action_rows must be a list")
    for row in rows:
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id:
            return row
    raise ExperimentQueueError(f"{portfolio_path}: missing operator action row for {candidate_id}")


def _selected_pair_indices(row: dict[str, Any], *, candidate_id: str) -> tuple[int, ...]:
    metadata = row.get("source_metadata")
    if not isinstance(metadata, dict):
        raise ExperimentQueueError(f"{candidate_id}: missing source_metadata")
    selected = metadata.get("selected_pair_indices")
    if (
        not isinstance(selected, list)
        or not selected
        or any(isinstance(item, bool) or not isinstance(item, int) for item in selected)
    ):
        raise ExperimentQueueError(f"{candidate_id}: selected_pair_indices must be a non-empty int list")
    selected_count = metadata.get("selected_pair_count")
    if isinstance(selected_count, int) and selected_count != len(selected):
        raise ExperimentQueueError(
            f"{candidate_id}: selected_pair_count={selected_count} but got {len(selected)} indices"
        )
    pairs = tuple(int(item) for item in selected)
    if len(set(pairs)) != len(pairs):
        raise ExperimentQueueError(f"{candidate_id}: selected_pair_indices contains duplicates")
    if tuple(sorted(pairs)) != pairs:
        raise ExperimentQueueError(f"{candidate_id}: selected_pair_indices must be sorted ascending")
    out_of_range = [pair for pair in pairs if not 0 <= pair < FEC6_PAIR_COUNT]
    if out_of_range:
        raise ExperimentQueueError(
            f"{candidate_id}: selected_pair_indices out of range 0..{FEC6_PAIR_COUNT - 1}: {out_of_range}"
        )
    return pairs


def _selection_acquisition_operation(selection: Dqs1QueueSelection) -> dict[str, Any]:
    metadata = selection.source_metadata
    operation = metadata.get("acquisition_operation")
    if isinstance(operation, Mapping):
        return dict(operation)
    request = metadata.get("pair_frame_geometry_request")
    if isinstance(request, Mapping):
        request_operation = request.get("acquisition_operation")
        if isinstance(request_operation, Mapping):
            return dict(request_operation)
        selector_kind = str(
            request.get("selector_kind") or "pair_frame_geometry_queue_request"
        )
        return {
            "op": selector_kind,
            "source_schema": request.get("schema"),
            "source_lattice_path": metadata.get(
                "pair_frame_geometry_request_source_path"
            ),
        }
    selector_kind = str(metadata.get("selector_kind") or "queue_action_summary")
    return {
        "op": selector_kind,
        "source_schema": metadata.get("schema"),
    }


def _selection_selector_kind(selection: Dqs1QueueSelection) -> str:
    metadata = selection.source_metadata
    request = metadata.get("pair_frame_geometry_request")
    if isinstance(request, Mapping) and isinstance(request.get("selector_kind"), str):
        return str(request["selector_kind"])
    if isinstance(metadata.get("selector_kind"), str):
        return str(metadata["selector_kind"])
    if isinstance(metadata.get("queue_source_kind"), str):
        return str(metadata["queue_source_kind"])
    operation = _selection_acquisition_operation(selection)
    if isinstance(operation.get("op"), str) and operation["op"]:
        return str(operation["op"])
    return "queue_action_summary"


def _repo_rel_path(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve()
    if _path_under_root(resolved, repo):
        return resolved.relative_to(repo).as_posix()
    return path.as_posix()


def build_selected_pairset_acquisition(
    selections: Sequence[Dqs1QueueSelection],
    *,
    repo_root: str | Path,
    action_summary_path: str | Path,
) -> dict[str, Any]:
    """Build the acquisition sidecar harvest canonicalization expects."""

    repo = Path(repo_root)
    summary_path = Path(action_summary_path)
    if not summary_path.is_absolute():
        summary_path = repo / summary_path
    rows: list[dict[str, Any]] = []
    for selection in selections:
        rows.append(
            {
                "candidate_id": selection.candidate_id,
                "acquisition_id": selection.candidate_id,
                "selector_id": selection.candidate_id,
                "selector_kind": _selection_selector_kind(selection),
                "selected_pair_count": len(selection.selected_pair_indices),
                "selected_pair_indices": list(selection.selected_pair_indices),
                "acquisition_operation": _selection_acquisition_operation(selection),
                "source_metadata": dict(selection.source_metadata),
                "source_action_summary_path": _repo_rel_path(
                    summary_path,
                    repo_root=repo,
                ),
                "source_portfolio_path": _repo_rel_path(
                    selection.portfolio_path,
                    repo_root=repo,
                ),
                "allowed_use": "dqs1_local_first_harvest_observation_canonicalization",
                "forbidden_use": (
                    "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    return {
        "schema": SELECTED_PAIRSET_ACQUISITION_SCHEMA,
        "compatible_schema": "decoder_q_pairset_acquisition.v1",
        "candidate_count": len(rows),
        "candidates": rows,
        "source_action_summary_path": _repo_rel_path(summary_path, repo_root=repo),
        "allowed_use": "dqs1_local_first_harvest_observation_canonicalization",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _pair_frame_geometry_selection_metadata(
    request: Mapping[str, Any],
    *,
    repo_root: Path,
    source_path: Path,
    request_index: int,
) -> tuple[str, dict[str, Any], tuple[int, ...]]:
    """Validate a pair-frame geometry request and adapt it to queue metadata."""

    if request.get("schema") != PAIR_FRAME_GEOMETRY_QUEUE_REQUEST_SCHEMA:
        raise ExperimentQueueError(
            "pair-frame geometry request schema mismatch: "
            f"{request.get('schema')!r}"
        )
    require_no_truthy_authority_fields(
        request,
        context=f"pair_frame_geometry_request[{request_index}]",
    )
    if request.get("queue_executable") is not True:
        raise ExperimentQueueError(
            f"pair-frame geometry request[{request_index}] is not queue_executable"
        )
    if request.get("operator_next_action") != SAFE_OPERATOR_ACTION:
        raise ExperimentQueueError(
            f"pair-frame geometry request[{request_index}] has unsupported "
            "operator_next_action"
        )
    queue_family = request.get("queue_family")
    if queue_family not in {None, DEFAULT_QUEUE_ID, "dqs1_pairset_local_first"}:
        raise ExperimentQueueError(
            f"pair-frame geometry request[{request_index}] targets unsupported "
            f"queue_family={queue_family!r}"
        )
    candidate_id = request.get("candidate_id")
    if not isinstance(candidate_id, str) or not _PAIRSET_CANDIDATE_RE.match(candidate_id):
        raise ExperimentQueueError(
            f"pair-frame geometry request[{request_index}] has unsupported "
            f"candidate_id={candidate_id!r}"
        )
    selected = request.get("selected_pair_indices")
    metadata_probe = {
        "source_metadata": {
            "selected_pair_indices": selected,
            "selected_pair_count": request.get("selected_pair_count"),
        }
    }
    pairs = _selected_pair_indices(metadata_probe, candidate_id=candidate_id)
    source_rel = (
        source_path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
        if _path_under_root(source_path.resolve(strict=False), repo_root.resolve())
        else source_path.as_posix()
    )
    metadata = {
        "schema": "dqs1_pair_frame_geometry_queue_request_metadata.v1",
        "queue_source_kind": "pair_frame_scorer_geometry_lattice",
        "pair_frame_geometry_request_source_path": source_rel,
        "pair_frame_geometry_request": dict(request),
        "selected_pair_count": len(pairs),
        "selected_pair_indices": list(pairs),
        "allowed_use": "local_first_queue_selection_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return candidate_id, metadata, pairs


def _candidate_completed_locally(
    *,
    repo_root: Path,
    results_root: str,
    candidate_id: str,
) -> bool:
    advisory_path = (
        repo_root
        / results_root
        / "materialized"
        / candidate_slug(candidate_id)
        / "local_cpu_advisory.json"
    )
    if not advisory_path.exists():
        return False
    advisory = _json_load(advisory_path)
    _require_false_authority(advisory, label=f"{candidate_id} local advisory")
    if advisory.get("score_axis") != "cpu_advisory":
        return False
    if advisory.get("evidence_semantics") != "non_contest_cpu_auth_eval_advisory":
        return False
    try:
        score = float(advisory.get("canonical_score"))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(score):
        return False
    if advisory.get("n_samples") != 600:
        return False
    archive_size = advisory.get("archive_size_bytes")
    if isinstance(archive_size, bool) or not isinstance(archive_size, int) or archive_size <= 0:
        return False
    provenance = advisory.get("provenance")
    if not isinstance(provenance, dict):
        return False
    archive_sha = provenance.get("archive_sha256")
    if not isinstance(archive_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", archive_sha):
        return False
    if provenance.get("archive_size_bytes") != archive_size:
        return False
    archive_path_value = provenance.get("archive_path")
    if not isinstance(archive_path_value, str) or not archive_path_value.strip():
        return False
    expected_archive_path = (
        repo_root
        / results_root
        / "materialized"
        / candidate_slug(candidate_id)
        / "submission_dir"
        / "archive.zip"
    )
    actual_archive_path = Path(archive_path_value)
    if not actual_archive_path.is_absolute():
        actual_archive_path = repo_root / actual_archive_path
    if actual_archive_path.resolve(strict=False) != expected_archive_path.resolve(strict=False):
        return False
    if not expected_archive_path.exists():
        return False
    if expected_archive_path.stat().st_size != archive_size:
        return False
    digest = sha256(expected_archive_path.read_bytes()).hexdigest()
    if digest != archive_sha:
        return False
    eureka_action = _candidate_eureka_signal_action(
        repo_root=repo_root,
        candidate_id=candidate_id,
        advisory_path=advisory_path,
        advisory=advisory,
        archive_sha=archive_sha,
        local_score=score,
    )
    if eureka_action == "dispatch_exact_auth_anchor":
        raise ExperimentQueueError(
            f"{candidate_id}: local CPU drift/eureka signal requests exact auth dispatch; "
            "refusing to reroute the local-first queue past this candidate"
        )
    return eureka_action == "observe_only"


def _completion_roots(results_root: str, completed_results_roots: tuple[str, ...]) -> tuple[str, ...]:
    roots: list[str] = [results_root]
    roots.extend(completed_results_roots)
    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _dqs1_observation_outcome_label(row: Mapping[str, Any]) -> str:
    score_delta = row.get("score_delta_vs_baseline")
    if (
        isinstance(score_delta, int | float)
        and not isinstance(score_delta, bool)
        and math.isfinite(float(score_delta))
    ):
        if float(score_delta) < 0.0:
            return "local_advisory_improved"
        if float(score_delta) > 0.0:
            return "local_advisory_regressed"
    byte_delta = row.get("archive_byte_delta_vs_baseline")
    if isinstance(byte_delta, int) and not isinstance(byte_delta, bool):
        if byte_delta < 0:
            return "archive_bytes_reduced_no_score_improvement"
        if byte_delta > 0:
            return "archive_bytes_increased_no_score_improvement"
    return "flat_local_advisory"


def _observed_dqs1_candidate_skips(
    dqs1_observations: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for row in dqs1_observations:
        if not isinstance(row, Mapping):
            continue
        require_no_truthy_authority_fields(
            row,
            context="dqs1_local_first_queue.observation_skip",
        )
        if (
            row.get("source_schema") != DQS1_OBSERVATION_SOURCE_SCHEMA
            or row.get("sweep_config_id") != DQS1_OBSERVATION_SWEEP_CONFIG_ID
        ):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        observed.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "reason": "dqs1_harvest_observation_exists",
                "observation_outcome": _dqs1_observation_outcome_label(row),
            },
        )
    return observed


def _candidate_completed_in_roots(
    *,
    repo_root: Path,
    results_roots: tuple[str, ...],
    candidate_id: str,
) -> str | None:
    for root in results_roots:
        if _candidate_completed_locally(
            repo_root=repo_root,
            results_root=root,
            candidate_id=candidate_id,
        ):
            return root
    return None


def _candidate_eureka_signal_action(
    *,
    repo_root: Path,
    candidate_id: str,
    advisory_path: Path,
    advisory: dict[str, Any],
    archive_sha: str,
    local_score: float,
) -> str | None:
    research_root = repo_root / ".omx" / "research"
    if not research_root.exists():
        return None
    expected_advisory = advisory_path.resolve(strict=False)
    for signal_path in sorted(
        research_root.glob(f"local_cpu_contest_drift_eureka_{candidate_id}_*.json"),
        reverse=True,
    ):
        try:
            signal = _json_load(signal_path)
        except ExperimentQueueError:
            continue
        if signal.get("schema") != LOCAL_CPU_CONTEST_DRIFT_EUREKA_SCHEMA:
            continue
        if signal.get("candidate_id") != candidate_id:
            continue
        if signal.get("candidate_archive_sha256") != archive_sha:
            continue
        try:
            signal_local_score = float(signal.get("local_score"))
        except (TypeError, ValueError):
            continue
        if not math.isclose(signal_local_score, local_score, rel_tol=0.0, abs_tol=1e-15):
            continue
        source_artifact = signal.get("source_artifact")
        if not isinstance(source_artifact, str) or not source_artifact.strip():
            continue
        source_path = Path(source_artifact)
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        if source_path.resolve(strict=False) != expected_advisory:
            continue
        if signal.get("local_axis") not in {
            advisory.get("evidence_grade"),
            advisory.get("score_axis"),
            "[macOS-CPU advisory]",
            "macOS-CPU advisory",
            "cpu_advisory",
        }:
            continue
        if signal.get("target_axis") != "contest-CPU":
            continue
        try:
            require_eureka_false_authority(
                signal,
                context=f"{signal_path} {candidate_id} eureka signal",
            )
        except LocalCPUContestDriftError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        eureka_trigger = signal.get("eureka_trigger")
        recommended_action = signal.get("recommended_action")
        if eureka_trigger is False and recommended_action == "observe_only":
            return "observe_only"
        if eureka_trigger is True and recommended_action == "dispatch_exact_auth_anchor":
            return "dispatch_exact_auth_anchor"
        if eureka_trigger not in {True, False} or recommended_action not in {
            "observe_only",
            "dispatch_exact_auth_anchor",
        }:
            continue
    return None


def _summary_timestamp_from_path(action_summary_path: Path) -> str:
    match = _TIMESTAMP_RE.search(str(action_summary_path))
    if match:
        return match.group(1)
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _lane_date_from_summary_path(action_summary_path: Path) -> str:
    return _summary_timestamp_from_path(action_summary_path)[:8]


def _eureka_signal_path(
    selection: Dqs1QueueSelection,
    *,
    eureka_output_dir: str,
    eureka_run_id: str | None = None,
) -> str:
    timestamp = eureka_run_id or _summary_timestamp_from_path(selection.action_summary_path)
    filename = f"local_cpu_contest_drift_eureka_{selection.candidate_id}_{timestamp}.json"
    return f"{eureka_output_dir.rstrip('/')}/{filename}"


def _mlx_local_advisory_cache_audit_path(
    selection: Dqs1QueueSelection,
    *,
    eureka_output_dir: str,
    eureka_run_id: str | None = None,
) -> str:
    timestamp = eureka_run_id or _summary_timestamp_from_path(selection.action_summary_path)
    filename = f"mlx_delta_cache_local_cpu_advisory_identity_{selection.candidate_slug}_{timestamp}.json"
    return f"{eureka_output_dir.rstrip('/')}/{filename}"


def _false_authority_postcondition(
    path: str,
    *,
    axis_key: str | None = None,
    axis_equals: str | None = None,
    false_or_missing: list[str] | None = None,
) -> dict[str, Any]:
    condition: dict[str, Any] = {
        "type": "json_false_authority",
        "path": path,
        "required_false": ["score_claim", "promotion_eligible", "rank_or_kill_eligible"],
        "false_or_missing": false_or_missing
        or ["ready_for_exact_eval_dispatch", "dispatch_attempted", "gpu_launched"],
    }
    if axis_key is not None:
        condition["axis_key"] = axis_key
        condition["axis_equals"] = axis_equals
    return condition


def _retention_command(
    *,
    roots: list[str],
    json_output: str,
    include_kind: str | None = None,
    execute: bool = False,
    action: str = "move",
    cold_store_roots: tuple[str, ...] = (),
    cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
) -> list[str]:
    if action not in {"move", "delete"}:
        raise ExperimentQueueError("retention action must be move or delete")
    if cold_store_roots:
        effective_cold_roots = cold_store_roots
    else:
        repo_device = REPO_ROOT.stat().st_dev
        existing_external_roots: list[str] = []
        for root in operator_cold_store_roots():
            path = Path(root)
            try:
                if path.is_dir() and path.stat().st_dev != repo_device:
                    existing_external_roots.append(root)
            except OSError:
                continue
        effective_cold_roots = tuple(existing_external_roots)
    if execute and action == "move" and not effective_cold_roots:
        raise ExperimentQueueError(
            "retention cold-store roots are required for executed move retention"
        )
    command = [
        ".venv/bin/python",
        "tools/compact_experiment_artifacts.py",
        *roots,
    ]
    if include_kind is not None:
        command.extend(["--include-kind", include_kind])
    command.extend(
        [
            "--min-bytes",
            "1",
            "--json-output",
            json_output,
        ]
    )
    if execute:
        command.extend(["--execute", "--action", action])
        command.extend(["--cold-store-reserve-gb", str(cold_store_reserve_gb)])
        if action == "move":
            for cold_store_root in effective_cold_roots:
                command.extend(["--cold-store-root", cold_store_root])
    return command


def _eureka_false_authority_postcondition(path: str) -> dict[str, Any]:
    return {
        "type": "json_false_authority",
        "path": path,
        "required_false": list(EUREKA_FALSE_AUTHORITY_FIELDS),
        "false_or_missing": [],
    }


def select_dqs1_local_first_candidate(
    action_summary_path: str | Path,
    *,
    repo_root: str | Path,
    results_root: str = DEFAULT_RESULTS_ROOT,
    completed_results_roots: tuple[str, ...] = (),
    exclude_candidate_ids: set[str] | None = None,
    skip_completed_local_advisory: bool = True,
    dqs1_observations: tuple[dict[str, Any], ...] = (),
    skip_observed_dqs1_candidates: bool = True,
) -> Dqs1QueueSelection:
    return select_dqs1_local_first_candidates(
        action_summary_path,
        repo_root=repo_root,
        results_root=results_root,
        completed_results_roots=completed_results_roots,
        exclude_candidate_ids=exclude_candidate_ids,
        skip_completed_local_advisory=skip_completed_local_advisory,
        candidate_limit=1,
        dqs1_observations=dqs1_observations,
        skip_observed_dqs1_candidates=skip_observed_dqs1_candidates,
    )[0]


def select_dqs1_local_first_candidates(
    action_summary_path: str | Path,
    *,
    repo_root: str | Path,
    results_root: str = DEFAULT_RESULTS_ROOT,
    completed_results_roots: tuple[str, ...] = (),
    exclude_candidate_ids: set[str] | None = None,
    skip_completed_local_advisory: bool = True,
    candidate_limit: int = 1,
    dqs1_observations: tuple[dict[str, Any], ...] = (),
    skip_observed_dqs1_candidates: bool = True,
    additional_queue_requests: tuple[dict[str, Any], ...] = (),
    additional_queue_request_source_paths: tuple[str, ...] = (),
) -> tuple[Dqs1QueueSelection, ...]:
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or candidate_limit <= 0:
        raise ExperimentQueueError("candidate_limit must be a positive integer")
    summary_path = Path(action_summary_path)
    if not summary_path.is_absolute():
        summary_path = Path(repo_root) / summary_path
    summary = _json_load(summary_path)
    _require_false_authority(summary, label=f"{summary_path} summary", require_all=True)

    actions = summary.get("top_operator_actions")
    if not isinstance(actions, list) or not actions:
        raise ExperimentQueueError(f"{summary_path}: top_operator_actions must be a non-empty list")

    portfolio_path = _resolve_portfolio_path(summary_path, summary, repo_root=Path(repo_root))
    excluded = exclude_candidate_ids or set()
    completion_roots = _completion_roots(results_root, completed_results_roots)
    observed_candidates = (
        _observed_dqs1_candidate_skips(dqs1_observations)
        if skip_observed_dqs1_candidates
        else {}
    )
    skipped: list[dict[str, str]] = []

    sorted_actions = sorted(
        (action for action in actions if isinstance(action, dict)),
        key=lambda action: int(action.get("operator_action_rank", 10**9)),
    )
    selections: list[Dqs1QueueSelection] = []
    for request_index, request in enumerate(additional_queue_requests):
        source_path_value = (
            additional_queue_request_source_paths[request_index]
            if request_index < len(additional_queue_request_source_paths)
            else request.get("source_geometry_lattice_path")
        )
        source_path = (
            Path(repo_root) / source_path_value
            if isinstance(source_path_value, str) and not Path(source_path_value).is_absolute()
            else Path(source_path_value or summary_path)
        )
        current_candidate_id, selection_metadata, selected_pairs = (
            _pair_frame_geometry_selection_metadata(
                request,
                repo_root=Path(repo_root),
                source_path=source_path,
                request_index=request_index,
            )
        )
        if current_candidate_id in excluded:
            skipped.append({"candidate_id": current_candidate_id, "reason": "explicitly_excluded"})
            continue
        if current_candidate_id in observed_candidates:
            skipped.append(dict(observed_candidates[current_candidate_id]))
            continue
        if skip_completed_local_advisory:
            completed_root = _candidate_completed_in_roots(
                repo_root=Path(repo_root),
                results_roots=completion_roots,
                candidate_id=current_candidate_id,
            )
            if completed_root is not None:
                skip_row = {"candidate_id": current_candidate_id, "reason": "local_advisory_exists"}
                if len(completion_roots) > 1:
                    skip_row["completed_results_root"] = completed_root
                skipped.append(skip_row)
                continue
        selections.append(
            Dqs1QueueSelection(
                candidate_id=current_candidate_id,
                candidate_slug=candidate_slug(current_candidate_id),
                selected_pair_indices=selected_pairs,
                action_summary_path=summary_path,
                portfolio_path=source_path,
                source_metadata=selection_metadata,
                operator_action_rank=None,
                skipped_candidates=tuple(skipped),
            )
        )
        if len(selections) >= candidate_limit:
            return tuple(selections)

    for action in sorted_actions:
        candidate_id_value = action.get("candidate_id")
        if not isinstance(candidate_id_value, str):
            skipped.append({"candidate_id": "", "reason": "missing_candidate_id"})
            continue
        current_candidate_id = candidate_id_value
        _require_false_authority(
            action,
            label=f"{current_candidate_id} top action",
            require_all=True,
        )
        if action.get("operator_next_action") != SAFE_OPERATOR_ACTION:
            skipped.append({"candidate_id": current_candidate_id, "reason": "unsupported_operator_action"})
            continue
        if current_candidate_id in excluded:
            skipped.append({"candidate_id": current_candidate_id, "reason": "explicitly_excluded"})
            continue
        if current_candidate_id in observed_candidates:
            skipped.append(dict(observed_candidates[current_candidate_id]))
            continue
        if skip_completed_local_advisory:
            completed_root = _candidate_completed_in_roots(
                repo_root=Path(repo_root),
                results_roots=completion_roots,
                candidate_id=current_candidate_id,
            )
            if completed_root is not None:
                skip_row = {"candidate_id": current_candidate_id, "reason": "local_advisory_exists"}
                if len(completion_roots) > 1:
                    skip_row["completed_results_root"] = completed_root
                skipped.append(skip_row)
                continue
        row = _load_portfolio_row(portfolio_path, current_candidate_id)
        _require_false_authority(
            row,
            label=f"{current_candidate_id} portfolio row",
            require_all=True,
        )
        metadata = row.get("source_metadata")
        if not isinstance(metadata, dict):
            raise ExperimentQueueError(f"{current_candidate_id}: missing source_metadata")
        selection_metadata = dict(metadata)
        if observed_candidates:
            selection_metadata["dqs1_observation_acquisition_skip"] = {
                "schema": "dqs1_observation_acquisition_skip.v1",
                "active": True,
                "observed_candidate_count": len(observed_candidates),
                "skip_reason": "dqs1_harvest_observation_exists",
                "allowed_use": "local_first_queue_rerun_suppression_only",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        selections.append(
            Dqs1QueueSelection(
                candidate_id=current_candidate_id,
                candidate_slug=candidate_slug(current_candidate_id),
                selected_pair_indices=_selected_pair_indices(
                    row,
                    candidate_id=current_candidate_id,
                ),
                action_summary_path=summary_path,
                portfolio_path=portfolio_path,
                source_metadata=selection_metadata,
                operator_action_rank=(
                    int(action["operator_action_rank"])
                    if isinstance(action.get("operator_action_rank"), int)
                    and not isinstance(action.get("operator_action_rank"), bool)
                    else None
                ),
                skipped_candidates=tuple(skipped),
            )
        )
        if len(selections) >= candidate_limit:
            return tuple(selections)

    if selections:
        return tuple(selections)

    reasons = ", ".join(f"{item['candidate_id']}:{item['reason']}" for item in skipped)
    raise ExperimentQueueError(f"no safe DQS1 local-first candidate found; skipped {reasons}")


def build_dqs1_local_first_queue(
    selection: Dqs1QueueSelection,
    *,
    lane_date: str | None = None,
    queue_id: str = DEFAULT_QUEUE_ID,
    results_root: str = DEFAULT_RESULTS_ROOT,
    mlx_effective_selection: str = DEFAULT_MLX_EFFECTIVE_SELECTION,
    decoder_q_candidate_manifest: str = DEFAULT_DECODER_Q_CANDIDATE_MANIFEST,
    base_submission_dir: str = DEFAULT_BASE_SUBMISSION_DIR,
    global_mutated_archive: str = DEFAULT_GLOBAL_MUTATED_ARCHIVE,
    upstream_dir: str = DEFAULT_UPSTREAM_DIR,
    video_names_file: str = DEFAULT_VIDEO_NAMES_FILE,
    frame_policy: str = DEFAULT_FRAME_POLICY,
    drift_calibration_json: str = DEFAULT_DRIFT_CALIBRATION_JSON,
    eureka_output_dir: str = DEFAULT_EUREKA_OUTPUT_DIR,
    eureka_run_id: str | None = None,
    local_cpu_concurrency: int = 1,
    local_io_concurrency: int = 1,
    include_mlx_local_advisory_debug: bool = False,
    allow_large_mlx_cache: bool = False,
    mlx_reference_cache_dir: str = DEFAULT_MLX_REFERENCE_CACHE_DIR,
    mlx_device: str = "gpu",
    mlx_batch_pairs: int = 1,
    mlx_cache_batch_pairs: int = 8,
    include_raw_retention_plan: bool = True,
    raw_retention_execute: bool = False,
    raw_retention_action: str = "move",
    raw_retention_cold_store_roots: tuple[str, ...] = (),
    raw_retention_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    include_mlx_retention_plan: bool = True,
    mlx_retention_execute: bool = False,
    mlx_retention_action: str = "move",
    mlx_retention_cold_store_roots: tuple[str, ...] = (),
    mlx_retention_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    preflight_dependency: str | None = None,
    materializer_feedback_bridge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        isinstance(local_cpu_concurrency, bool)
        or not isinstance(local_cpu_concurrency, int)
        or local_cpu_concurrency <= 0
    ):
        raise ExperimentQueueError("local_cpu_concurrency must be a positive integer")
    if (
        isinstance(local_io_concurrency, bool)
        or not isinstance(local_io_concurrency, int)
        or local_io_concurrency <= 0
    ):
        raise ExperimentQueueError("local_io_concurrency must be a positive integer")
    if mlx_device not in {"cpu", "gpu"}:
        raise ExperimentQueueError("mlx_device must be 'cpu' or 'gpu'")
    if isinstance(mlx_batch_pairs, bool) or not isinstance(mlx_batch_pairs, int) or mlx_batch_pairs <= 0:
        raise ExperimentQueueError("mlx_batch_pairs must be a positive integer")
    if mlx_batch_pairs != 1:
        raise ExperimentQueueError(
            "mlx_batch_pairs must remain 1 until the MLX batch-shape invariance gate passes"
        )
    if (
        isinstance(mlx_cache_batch_pairs, bool)
        or not isinstance(mlx_cache_batch_pairs, int)
        or mlx_cache_batch_pairs <= 0
    ):
        raise ExperimentQueueError("mlx_cache_batch_pairs must be a positive integer")
    if include_mlx_local_advisory_debug and not allow_large_mlx_cache:
        raise ExperimentQueueError(
            "include_mlx_local_advisory_debug requires allow_large_mlx_cache=True "
            "because DQS1 full-sample tensor caches are multi-GB artifacts"
        )
    for label, action in {
        "raw_retention_action": raw_retention_action,
        "mlx_retention_action": mlx_retention_action,
    }.items():
        if action not in {"move", "delete"}:
            raise ExperimentQueueError(f"{label} must be move or delete")
    date = lane_date or _lane_date_from_summary_path(selection.action_summary_path)
    selected_pairs = ",".join(str(index) for index in selection.selected_pair_indices)
    materialized_root = f"{results_root}/materialized/{selection.candidate_slug}"
    bridge_plan = f"{materialized_root}/decoder_q_selective_window_bridge_plan.json"
    bridge_plan_md = f"{materialized_root}/decoder_q_selective_window_bridge_plan.md"
    packet_plan = f"{results_root}/selector_pareto/packet_plans/{selection.candidate_slug}.json"
    packet_plan_md = f"{results_root}/selector_pareto/packet_plans/{selection.candidate_slug}.md"
    eureka_signal = _eureka_signal_path(
        selection,
        eureka_output_dir=eureka_output_dir,
        eureka_run_id=eureka_run_id,
    )
    mlx_cache_audit = _mlx_local_advisory_cache_audit_path(
        selection,
        eureka_output_dir=eureka_output_dir,
        eureka_run_id=eureka_run_id,
    )
    experiment_id = selection.candidate_id
    mlx_cache_dir = f"{materialized_root}/mlx_delta_cache"
    mlx_response = f"{materialized_root}/mlx_delta_response_{mlx_device}_b{mlx_batch_pairs}_full600.json"
    mlx_components_dir = f"{materialized_root}/mlx_delta_components_{mlx_device}_b{mlx_batch_pairs}_full600"
    raw_retention_plan = f"{materialized_root}/raw_artifact_retention_plan.json"
    mlx_retention_plan = f"{materialized_root}/mlx_delta_cache_retention_plan.json"

    steps: list[dict[str, Any]] = [
        {
            "id": "build_bridge_plan",
            **({"requires": [preflight_dependency]} if preflight_dependency else {}),
            "command": [
                ".venv/bin/python",
                "tools/build_decoder_q_selective_window_bridge_plan.py",
                "--selection",
                mlx_effective_selection,
                "--candidate-manifest",
                decoder_q_candidate_manifest,
                "--lane-id",
                f"lane_dqs1_{experiment_id}_local_first_{date}",
                "--json-out",
                bridge_plan,
                "--md-out",
                bridge_plan_md,
            ],
            "resources": {"kind": "local_cpu"},
            "postconditions": [
                _false_authority_postcondition(bridge_plan),
            ],
        },
        {
            "id": "plan_packet",
            "requires": ["build_bridge_plan"],
            "command": [
                ".venv/bin/python",
                "tools/plan_decoder_q_selective_runtime_packet.py",
                "--bridge-plan",
                bridge_plan,
                "--base-archive",
                f"{base_submission_dir}/archive.zip",
                "--json-out",
                packet_plan,
                "--md-out",
                packet_plan_md,
                "--frame-policy",
                frame_policy,
                "--selected-pairs",
                selected_pairs,
            ],
            "resources": {"kind": "local_cpu"},
            "postconditions": [
                _false_authority_postcondition(packet_plan)
            ],
        },
        {
            "id": "materialize",
            "requires": ["plan_packet"],
            "command": [
                ".venv/bin/python",
                "tools/materialize_decoder_q_selective_runtime_candidate.py",
                "--plan",
                packet_plan,
                "--base-submission-dir",
                base_submission_dir,
                "--output-dir",
                f"{materialized_root}/submission_dir",
                "--frame-policy",
                frame_policy,
                "--manifest-output",
                f"{materialized_root}/materialization_manifest.json",
            ],
            "resources": {"kind": "local_cpu"},
            "postconditions": [
                _false_authority_postcondition(
                    f"{materialized_root}/materialization_manifest.json"
                )
            ],
        },
        {
            "id": "locality_controls",
            "requires": ["materialize"],
            "timeout_seconds": 960,
            "command": [
                ".venv/bin/python",
                "tools/run_decoder_q_selective_runtime_locality_controls.py",
                "--parent-submission-dir",
                base_submission_dir,
                "--global-mutated-archive",
                global_mutated_archive,
                "--selective-submission-dir",
                f"{materialized_root}/submission_dir",
                "--selected-pairs",
                selected_pairs,
                "--frame-policy",
                frame_policy,
                "--timeout-seconds",
                "540",
                "--global-timeout-seconds",
                "840",
                "--max-inflate-parallelism",
                "3",
                "--reuse-existing-inflates",
                "--work-dir",
                f"{materialized_root}/locality_work",
                "--json-out",
                f"{materialized_root}/locality_controls.json",
            ],
            "resources": {"kind": "local_io_heavy"},
            "postconditions": [
                {
                    "type": "json_equals",
                    "path": f"{materialized_root}/locality_controls.json",
                    "key": "locality_controls_passed",
                    "equals": True,
                },
                _false_authority_postcondition(
                    f"{materialized_root}/locality_controls.json",
                    axis_key="score_axis",
                    axis_equals="[locality-control no-score]",
                ),
            ],
        },
        {
            "id": "local_cpu_advisory",
            "requires": ["locality_controls"],
            "timeout_seconds": 3600,
            "command": [
                ".venv/bin/python",
                "experiments/contest_auth_eval.py",
                "--archive",
                f"{materialized_root}/submission_dir/archive.zip",
                "--inflate-sh",
                f"{materialized_root}/submission_dir/inflate.sh",
                "--upstream-dir",
                upstream_dir,
                "--video-names-file",
                video_names_file,
                "--device",
                "cpu",
                "--work-dir",
                f"{materialized_root}/local_cpu_advisory_work",
                "--json-out",
                f"{materialized_root}/local_cpu_advisory.json",
                "--inflate-timeout",
                "1800",
                "--evaluate-timeout",
                "1800",
                "--keep-work-dir",
            ],
            "resources": {"kind": "local_cpu"},
            "postconditions": [
                _false_authority_postcondition(
                    f"{materialized_root}/local_cpu_advisory.json",
                    axis_key="score_axis",
                    axis_equals="cpu_advisory",
                )
            ],
        },
    ]
    if include_raw_retention_plan:
        steps.append(
            {
                "id": "plan_raw_artifact_retention",
                "requires": ["local_cpu_advisory"],
                "timeout_seconds": 1200,
                "command": _retention_command(
                    roots=[materialized_root],
                    json_output=raw_retention_plan,
                    execute=raw_retention_execute,
                    action=raw_retention_action,
                    cold_store_roots=raw_retention_cold_store_roots,
                    cold_store_reserve_gb=raw_retention_cold_store_reserve_gb,
                ),
                "resources": {"kind": "local_io_heavy"},
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": raw_retention_plan,
                        "key": "plan.blocked_candidate_count",
                        "equals": 0,
                    },
                    {
                        "type": "json_false_authority",
                        "path": raw_retention_plan,
                        "required_false": [
                            "plan.score_claim",
                            "plan.promotion_eligible",
                            "plan.ready_for_exact_eval_dispatch",
                        ],
                        "false_or_missing": [],
                    },
                ],
            }
        )
    if include_mlx_local_advisory_debug:
        build_mlx_cache_command = [
            ".venv/bin/python",
            "tools/build_mlx_scorer_input_cache_from_local_advisory.py",
            "--local-cpu-advisory",
            f"{materialized_root}/local_cpu_advisory.json",
            "--output-cache-dir",
            mlx_cache_dir,
            "--audit-output",
            mlx_cache_audit,
            "--expected-pair-count",
            "600",
            "--batch-pairs",
            str(mlx_cache_batch_pairs),
            "--allow-large-tensor-cache",
            "--stamp-cache-manifest-on-pass",
        ]
        run_mlx_response_command = [
            ".venv/bin/python",
            "tools/run_mlx_scorer_response_from_local_advisory.py",
            "--local-cpu-advisory",
            f"{materialized_root}/local_cpu_advisory.json",
            "--reference-cache-dir",
            mlx_reference_cache_dir,
            "--candidate-cache-dir",
            mlx_cache_dir,
            "--output",
            mlx_response,
            "--repo-root",
            ".",
            "--batch-pairs",
            str(mlx_batch_pairs),
            "--device",
            mlx_device,
            "--allow-local-cpu-advisory-cache-identity",
            "--components-dir",
            mlx_components_dir,
            "--response-family",
            "dqs1_local_advisory_debug",
        ]
        if mlx_device == "gpu":
            run_mlx_response_command.append("--allow-gpu-research-signal")
        steps.extend(
            [
                {
                    "id": "build_mlx_local_advisory_cache",
                    "requires": ["local_cpu_advisory"],
                    "timeout_seconds": 1800,
                    "command": build_mlx_cache_command,
                    "resources": {"kind": "local_cpu"},
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": mlx_cache_audit,
                            "key": "passed",
                            "equals": True,
                        },
                        _false_authority_postcondition(mlx_cache_audit),
                        _false_authority_postcondition(f"{mlx_cache_dir}/manifest.json"),
                    ],
                },
                {
                    "id": "local_mlx_advisory_response",
                    "requires": ["build_mlx_local_advisory_cache"],
                    "timeout_seconds": 600,
                    "command": run_mlx_response_command,
                    "resources": {
                        "kind": "local_mlx" if mlx_device == "gpu" else "local_cpu"
                    },
                    "postconditions": [
                        _false_authority_postcondition(
                            mlx_response,
                            axis_key="score_axis",
                            axis_equals="[macOS-MLX research-signal]",
                        )
                    ],
                },
            ]
        )
        if include_mlx_retention_plan:
            steps.append(
                {
                    "id": "plan_mlx_delta_cache_retention",
                    "requires": ["local_mlx_advisory_response"],
                    "timeout_seconds": 1200,
                    "command": _retention_command(
                        roots=[materialized_root],
                        json_output=mlx_retention_plan,
                        include_kind="mlx_scorer_input_cache",
                        execute=mlx_retention_execute,
                        action=mlx_retention_action,
                        cold_store_roots=mlx_retention_cold_store_roots,
                        cold_store_reserve_gb=mlx_retention_cold_store_reserve_gb,
                    ),
                    "resources": {"kind": "local_io_heavy"},
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": mlx_retention_plan,
                            "key": "plan.candidate_count",
                            "equals": 1,
                        },
                        {
                            "type": "json_false_authority",
                            "path": mlx_retention_plan,
                            "required_false": [
                                "plan.score_claim",
                                "plan.promotion_eligible",
                                "plan.ready_for_exact_eval_dispatch",
                            ],
                            "false_or_missing": [],
                        },
                    ],
                }
            )
    steps.append(
        {
            "id": "local_cpu_contest_drift_eureka",
            "requires": ["local_cpu_advisory"],
            "timeout_seconds": 120,
            "command": [
                ".venv/bin/python",
                "tools/calibrate_local_cpu_contest_drift.py",
                "--calibration-json",
                drift_calibration_json,
                "--candidate-id",
                experiment_id,
                "--candidate-local-json",
                f"{materialized_root}/local_cpu_advisory.json",
                "--auth-frontier-score-from-pointer",
                "--eureka-out",
                eureka_signal,
                "--min-margin",
                "0.0",
            ],
            "resources": {"kind": "local_cpu"},
            "postconditions": [
                {
                    "type": "json_equals",
                    "path": eureka_signal,
                    "key": "schema",
                    "equals": LOCAL_CPU_CONTEST_DRIFT_EUREKA_SCHEMA,
                },
                _eureka_false_authority_postcondition(eureka_signal),
            ],
        }
    )

    queue = {
        "schema": QUEUE_SCHEMA,
        "queue_id": queue_id,
        "controls": {
            "mode": "running",
            "local_first": True,
            "max_concurrency": {
                "local_cpu": local_cpu_concurrency,
                "local_io_heavy": local_io_concurrency,
                "local_mlx": 1,
                "modal_cpu": 0,
                "modal_gpu": 0,
            },
        },
        "experiments": [
            {
                "id": experiment_id,
                "priority": _candidate_priority(experiment_id, selection.operator_action_rank),
                "lane_id": f"lane_dqs1_{experiment_id}_local_first_{date}",
                "tags": ["dqs1", "pairset", "local-first", "no-score-authority"],
                "metadata": {
                    "schema": "dqs1_local_first_experiment_metadata.v1",
                    "source_metadata": selection.source_metadata,
                    "skipped_candidates": list(selection.skipped_candidates),
                    "action_summary_path": str(selection.action_summary_path),
                    "portfolio_path": str(selection.portfolio_path),
                    "selected_pair_indices": list(selection.selected_pair_indices),
                    "source_custody_preserved": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    **(
                        {"materializer_feedback_bridge": materializer_feedback_bridge}
                        if materializer_feedback_bridge is not None
                        else {}
                    ),
                },
                "steps": steps,
            }
        ],
    }
    if preflight_dependency:
        return queue
    return normalize_queue_definition(queue)


def build_dqs1_local_first_queue_from_selections(
    selections: tuple[Dqs1QueueSelection, ...],
    *,
    repo_root: str | Path = ".",
    lane_date: str | None = None,
    queue_id: str = DEFAULT_QUEUE_ID,
    results_root: str = DEFAULT_RESULTS_ROOT,
    mlx_effective_selection: str = DEFAULT_MLX_EFFECTIVE_SELECTION,
    decoder_q_candidate_manifest: str = DEFAULT_DECODER_Q_CANDIDATE_MANIFEST,
    base_submission_dir: str = DEFAULT_BASE_SUBMISSION_DIR,
    global_mutated_archive: str = DEFAULT_GLOBAL_MUTATED_ARCHIVE,
    upstream_dir: str = DEFAULT_UPSTREAM_DIR,
    video_names_file: str = DEFAULT_VIDEO_NAMES_FILE,
    frame_policy: str = DEFAULT_FRAME_POLICY,
    drift_calibration_json: str = DEFAULT_DRIFT_CALIBRATION_JSON,
    eureka_output_dir: str = DEFAULT_EUREKA_OUTPUT_DIR,
    eureka_run_id: str | None = None,
    local_cpu_concurrency: int = 1,
    local_io_concurrency: int = 1,
    include_mlx_local_advisory_debug: bool = False,
    allow_large_mlx_cache: bool = False,
    mlx_reference_cache_dir: str = DEFAULT_MLX_REFERENCE_CACHE_DIR,
    mlx_device: str = "gpu",
    mlx_batch_pairs: int = 1,
    mlx_cache_batch_pairs: int = 8,
    include_raw_retention_plan: bool = True,
    raw_retention_execute: bool = False,
    raw_retention_action: str = "move",
    raw_retention_cold_store_roots: tuple[str, ...] = (),
    raw_retention_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    include_mlx_retention_plan: bool = True,
    mlx_retention_execute: bool = False,
    mlx_retention_action: str = "move",
    mlx_retention_cold_store_roots: tuple[str, ...] = (),
    mlx_retention_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    include_scheduler_preflight: bool = False,
    scheduler_storage_tiers: tuple[str, ...] = (),
    scheduler_storage_workload_subdir: str | None = None,
    scheduler_storage_expected_workload_root: str | None = None,
    scheduler_storage_reserve_free_gb: float = DEFAULT_RESERVE_FREE_GB,
    scheduler_storage_expected_bytes: int = 0,
    scheduler_proactive_cleanup_roots: tuple[str, ...] = (),
    scheduler_proactive_cleanup_execute: bool = False,
    scheduler_proactive_cleanup_action: str = "move",
    scheduler_proactive_cleanup_min_bytes: str = "1",
    scheduler_proactive_cleanup_cold_store_roots: tuple[str, ...] = (),
    scheduler_proactive_cleanup_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    materializer_feedback_bridge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not selections:
        raise ExperimentQueueError("at least one DQS1 queue selection is required")
    if include_scheduler_preflight and not scheduler_proactive_cleanup_execute:
        raise ExperimentQueueError(
            "scheduler_proactive_cleanup_execute must be true when "
            "scheduler preflight gates DQS1 queue execution"
        )
    if include_scheduler_preflight:
        try:
            validate_scheduler_storage_preflight_config(
                proactive_cleanup_execute=scheduler_proactive_cleanup_execute,
                proactive_cleanup_action=scheduler_proactive_cleanup_action,
                proactive_cleanup_cold_store_roots=scheduler_proactive_cleanup_cold_store_roots,
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        _require_dqs1_outputs_under_storage_root(
            repo_root=Path(repo_root),
            results_root=results_root,
            expected_workload_root=scheduler_storage_expected_workload_root,
        )
    date = lane_date or _lane_date_from_summary_path(selections[0].action_summary_path)
    preflight_artifact_id = eureka_run_id or date
    preflight_dependency = (
        f"{DEFAULT_SCHEDULER_PREFLIGHT_EXPERIMENT_ID}.proactive_cleanup"
        if include_scheduler_preflight
        else None
    )
    queues = [
        build_dqs1_local_first_queue(
            selection,
            lane_date=date,
            queue_id=queue_id,
            results_root=results_root,
            mlx_effective_selection=mlx_effective_selection,
            decoder_q_candidate_manifest=decoder_q_candidate_manifest,
            base_submission_dir=base_submission_dir,
            global_mutated_archive=global_mutated_archive,
            upstream_dir=upstream_dir,
            video_names_file=video_names_file,
            frame_policy=frame_policy,
            drift_calibration_json=drift_calibration_json,
            eureka_output_dir=eureka_output_dir,
            eureka_run_id=eureka_run_id,
            local_cpu_concurrency=local_cpu_concurrency,
            local_io_concurrency=local_io_concurrency,
            include_mlx_local_advisory_debug=include_mlx_local_advisory_debug,
            allow_large_mlx_cache=allow_large_mlx_cache,
            mlx_reference_cache_dir=mlx_reference_cache_dir,
            mlx_device=mlx_device,
            mlx_batch_pairs=mlx_batch_pairs,
            mlx_cache_batch_pairs=mlx_cache_batch_pairs,
            include_raw_retention_plan=include_raw_retention_plan,
            raw_retention_execute=raw_retention_execute,
            raw_retention_action=raw_retention_action,
            raw_retention_cold_store_roots=raw_retention_cold_store_roots,
            raw_retention_cold_store_reserve_gb=raw_retention_cold_store_reserve_gb,
            include_mlx_retention_plan=include_mlx_retention_plan,
            mlx_retention_execute=mlx_retention_execute,
            mlx_retention_action=mlx_retention_action,
            mlx_retention_cold_store_roots=mlx_retention_cold_store_roots,
            mlx_retention_cold_store_reserve_gb=mlx_retention_cold_store_reserve_gb,
            preflight_dependency=preflight_dependency,
            materializer_feedback_bridge=materializer_feedback_bridge,
        )
        for selection in selections
    ]
    queue = dict(queues[0])
    preflight_experiments = [
        _scheduler_preflight_experiment(
            repo_root=repo_root,
            date=preflight_artifact_id,
            results_root=results_root,
            storage_tiers=scheduler_storage_tiers,
            storage_workload_subdir=scheduler_storage_workload_subdir,
            storage_expected_workload_root=scheduler_storage_expected_workload_root,
            storage_reserve_free_gb=scheduler_storage_reserve_free_gb,
            storage_expected_bytes=scheduler_storage_expected_bytes,
            proactive_cleanup_roots=scheduler_proactive_cleanup_roots,
            proactive_cleanup_execute=scheduler_proactive_cleanup_execute,
            proactive_cleanup_action=scheduler_proactive_cleanup_action,
            proactive_cleanup_min_bytes=scheduler_proactive_cleanup_min_bytes,
            proactive_cleanup_cold_store_roots=scheduler_proactive_cleanup_cold_store_roots,
            proactive_cleanup_cold_store_reserve_gb=scheduler_proactive_cleanup_cold_store_reserve_gb,
        )
    ] if include_scheduler_preflight else []
    queue["experiments"] = list(preflight_experiments)
    queue["experiments"].extend(
        experiment
        for candidate_queue in queues
        for experiment in candidate_queue["experiments"]
    )
    return normalize_queue_definition(queue)


def build_queue_from_action_summary(
    action_summary_path: str | Path,
    *,
    repo_root: str | Path,
    results_root: str = DEFAULT_RESULTS_ROOT,
    queue_id: str = DEFAULT_QUEUE_ID,
    completed_results_roots: tuple[str, ...] = (),
    mlx_effective_selection: str = DEFAULT_MLX_EFFECTIVE_SELECTION,
    decoder_q_candidate_manifest: str = DEFAULT_DECODER_Q_CANDIDATE_MANIFEST,
    base_submission_dir: str = DEFAULT_BASE_SUBMISSION_DIR,
    global_mutated_archive: str = DEFAULT_GLOBAL_MUTATED_ARCHIVE,
    upstream_dir: str = DEFAULT_UPSTREAM_DIR,
    video_names_file: str = DEFAULT_VIDEO_NAMES_FILE,
    frame_policy: str = DEFAULT_FRAME_POLICY,
    drift_calibration_json: str = DEFAULT_DRIFT_CALIBRATION_JSON,
    eureka_output_dir: str = DEFAULT_EUREKA_OUTPUT_DIR,
    eureka_run_id: str | None = None,
    exclude_candidate_ids: set[str] | None = None,
    skip_completed_local_advisory: bool = True,
    candidate_limit: int = 1,
    local_cpu_concurrency: int = 1,
    local_io_concurrency: int = 1,
    include_mlx_local_advisory_debug: bool = False,
    allow_large_mlx_cache: bool = False,
    mlx_reference_cache_dir: str = DEFAULT_MLX_REFERENCE_CACHE_DIR,
    mlx_device: str = "gpu",
    mlx_batch_pairs: int = 1,
    mlx_cache_batch_pairs: int = 8,
    include_raw_retention_plan: bool = True,
    raw_retention_execute: bool = False,
    raw_retention_action: str = "move",
    raw_retention_cold_store_roots: tuple[str, ...] = (),
    raw_retention_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    include_mlx_retention_plan: bool = True,
    mlx_retention_execute: bool = False,
    mlx_retention_action: str = "move",
    mlx_retention_cold_store_roots: tuple[str, ...] = (),
    mlx_retention_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    include_scheduler_preflight: bool = False,
    scheduler_storage_tiers: tuple[str, ...] = (),
    scheduler_storage_workload_subdir: str | None = None,
    scheduler_storage_expected_workload_root: str | None = None,
    scheduler_storage_reserve_free_gb: float = DEFAULT_RESERVE_FREE_GB,
    scheduler_storage_expected_bytes: int = 0,
    scheduler_proactive_cleanup_roots: tuple[str, ...] = (),
    scheduler_proactive_cleanup_execute: bool = False,
    scheduler_proactive_cleanup_action: str = "move",
    scheduler_proactive_cleanup_min_bytes: str = "1",
    scheduler_proactive_cleanup_cold_store_roots: tuple[str, ...] = (),
    scheduler_proactive_cleanup_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    materializer_feedback_payloads: tuple[dict[str, Any], ...] = (),
    materializer_feedback_source_paths: tuple[str, ...] = (),
    dqs1_observations: tuple[dict[str, Any], ...] = (),
    dqs1_observation_source_paths: tuple[str, ...] = (),
    skip_observed_dqs1_candidates: bool = True,
    additional_queue_requests: tuple[dict[str, Any], ...] = (),
    additional_queue_request_source_paths: tuple[str, ...] = (),
) -> Dqs1QueueBuildResult:
    try:
        selections = select_dqs1_local_first_candidates(
            action_summary_path,
            repo_root=repo_root,
            results_root=results_root,
            completed_results_roots=completed_results_roots,
            exclude_candidate_ids=exclude_candidate_ids,
            skip_completed_local_advisory=skip_completed_local_advisory,
            candidate_limit=candidate_limit,
            dqs1_observations=dqs1_observations,
            skip_observed_dqs1_candidates=skip_observed_dqs1_candidates,
            additional_queue_requests=additional_queue_requests,
            additional_queue_request_source_paths=additional_queue_request_source_paths,
        )
        materializer_feedback_bridge = build_dqs1_materializer_feedback_bridge(
            materializer_feedback_payloads=materializer_feedback_payloads,
            materializer_feedback_source_paths=materializer_feedback_source_paths,
            planned_dqs1_candidate_ids=tuple(selection.candidate_id for selection in selections),
            candidate_limit=candidate_limit,
            dqs1_observations=dqs1_observations,
            dqs1_observation_source_paths=dqs1_observation_source_paths,
        )
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    selected_pairset_acquisition = build_selected_pairset_acquisition(
        selections,
        repo_root=repo_root,
        action_summary_path=action_summary_path,
    )
    return Dqs1QueueBuildResult(
        queue=build_dqs1_local_first_queue_from_selections(
            selections,
            repo_root=repo_root,
            queue_id=queue_id,
            results_root=results_root,
            mlx_effective_selection=mlx_effective_selection,
            decoder_q_candidate_manifest=decoder_q_candidate_manifest,
            base_submission_dir=base_submission_dir,
            global_mutated_archive=global_mutated_archive,
            upstream_dir=upstream_dir,
            video_names_file=video_names_file,
            frame_policy=frame_policy,
            drift_calibration_json=drift_calibration_json,
            eureka_output_dir=eureka_output_dir,
            eureka_run_id=eureka_run_id,
            local_cpu_concurrency=local_cpu_concurrency,
            local_io_concurrency=local_io_concurrency,
            include_mlx_local_advisory_debug=include_mlx_local_advisory_debug,
            allow_large_mlx_cache=allow_large_mlx_cache,
            mlx_reference_cache_dir=mlx_reference_cache_dir,
            mlx_device=mlx_device,
            mlx_batch_pairs=mlx_batch_pairs,
            mlx_cache_batch_pairs=mlx_cache_batch_pairs,
            include_raw_retention_plan=include_raw_retention_plan,
            raw_retention_execute=raw_retention_execute,
            raw_retention_action=raw_retention_action,
            raw_retention_cold_store_roots=raw_retention_cold_store_roots,
            raw_retention_cold_store_reserve_gb=raw_retention_cold_store_reserve_gb,
            include_mlx_retention_plan=include_mlx_retention_plan,
            mlx_retention_execute=mlx_retention_execute,
            mlx_retention_action=mlx_retention_action,
            mlx_retention_cold_store_roots=mlx_retention_cold_store_roots,
            mlx_retention_cold_store_reserve_gb=mlx_retention_cold_store_reserve_gb,
            include_scheduler_preflight=include_scheduler_preflight,
            scheduler_storage_tiers=scheduler_storage_tiers,
            scheduler_storage_workload_subdir=scheduler_storage_workload_subdir,
            scheduler_storage_expected_workload_root=scheduler_storage_expected_workload_root,
            scheduler_storage_reserve_free_gb=scheduler_storage_reserve_free_gb,
            scheduler_storage_expected_bytes=scheduler_storage_expected_bytes,
            scheduler_proactive_cleanup_roots=scheduler_proactive_cleanup_roots,
            scheduler_proactive_cleanup_execute=scheduler_proactive_cleanup_execute,
            scheduler_proactive_cleanup_action=scheduler_proactive_cleanup_action,
            scheduler_proactive_cleanup_min_bytes=scheduler_proactive_cleanup_min_bytes,
            scheduler_proactive_cleanup_cold_store_roots=scheduler_proactive_cleanup_cold_store_roots,
            scheduler_proactive_cleanup_cold_store_reserve_gb=scheduler_proactive_cleanup_cold_store_reserve_gb,
            materializer_feedback_bridge=materializer_feedback_bridge,
        ),
        selection=selections[0],
        selections=selections,
        materializer_feedback_bridge=materializer_feedback_bridge,
        selected_pairset_acquisition=selected_pairset_acquisition,
    )
