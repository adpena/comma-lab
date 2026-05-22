from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition

DEFAULT_QUEUE_ID = "dqs1_pairset_local_first"
DEFAULT_RESULTS_ROOT = (
    "experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb"
)
DEFAULT_BRIDGE_PLAN = (
    "experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/"
    "decoder_q_selective_window_bridge_plan_top32.json"
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
SAFE_OPERATOR_ACTION = "materialize_pairset_archive_and_run_local_controls"

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


@dataclass(frozen=True)
class Dqs1QueueSelection:
    candidate_id: str
    candidate_slug: str
    selected_pair_indices: tuple[int, ...]
    action_summary_path: Path
    portfolio_path: Path
    operator_action_rank: int | None = None
    skipped_candidates: tuple[dict[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Dqs1QueueBuildResult:
    queue: dict[str, Any]
    selection: Dqs1QueueSelection


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


def find_latest_cross_family_action_summary(repo_root: str | Path) -> Path:
    repo = Path(repo_root)
    candidates = []
    for path in (repo / "experiments" / "results").rglob("action_summary.json"):
        if "cross_family_candidate_portfolio" not in str(path):
            continue
        try:
            summary = _json_load(path)
        except ExperimentQueueError:
            continue
        if _is_dqs1_safe_action_summary(summary):
            candidates.append(path)
    if not candidates:
        raise ExperimentQueueError("no DQS1-safe cross-family action_summary.json files found")
    return max(candidates, key=_timestamp_key)


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
    if summary.get("schema") != "pairset_component_marginal_canonicalization_summary.v1":
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
    portfolio_json = summary.get("portfolio_json")
    if not isinstance(portfolio_json, str) or not portfolio_json.strip():
        raise ExperimentQueueError(f"{action_summary_path}: missing portfolio_json")
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
    return tuple(int(item) for item in selected)


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
    return digest == archive_sha


def _lane_date_from_summary_path(action_summary_path: Path) -> str:
    match = _TIMESTAMP_RE.search(str(action_summary_path))
    if match:
        return match.group(1)[:8]
    return datetime.now(UTC).strftime("%Y%m%d")


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


def select_dqs1_local_first_candidate(
    action_summary_path: str | Path,
    *,
    repo_root: str | Path,
    results_root: str = DEFAULT_RESULTS_ROOT,
    exclude_candidate_ids: set[str] | None = None,
    skip_completed_local_advisory: bool = True,
) -> Dqs1QueueSelection:
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
    skipped: list[dict[str, str]] = []

    sorted_actions = sorted(
        (action for action in actions if isinstance(action, dict)),
        key=lambda action: int(action.get("operator_action_rank", 10**9)),
    )
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
        if skip_completed_local_advisory and _candidate_completed_locally(
            repo_root=Path(repo_root),
            results_root=results_root,
            candidate_id=current_candidate_id,
        ):
            skipped.append({"candidate_id": current_candidate_id, "reason": "local_advisory_exists"})
            continue
        row = _load_portfolio_row(portfolio_path, current_candidate_id)
        _require_false_authority(
            row,
            label=f"{current_candidate_id} portfolio row",
            require_all=True,
        )
        return Dqs1QueueSelection(
            candidate_id=current_candidate_id,
            candidate_slug=candidate_slug(current_candidate_id),
            selected_pair_indices=_selected_pair_indices(row, candidate_id=current_candidate_id),
            action_summary_path=summary_path,
            portfolio_path=portfolio_path,
            operator_action_rank=(
                int(action["operator_action_rank"])
                if isinstance(action.get("operator_action_rank"), int)
                and not isinstance(action.get("operator_action_rank"), bool)
                else None
            ),
            skipped_candidates=tuple(skipped),
        )

    reasons = ", ".join(f"{item['candidate_id']}:{item['reason']}" for item in skipped)
    raise ExperimentQueueError(f"no safe DQS1 local-first candidate found; skipped {reasons}")


def build_dqs1_local_first_queue(
    selection: Dqs1QueueSelection,
    *,
    lane_date: str | None = None,
    queue_id: str = DEFAULT_QUEUE_ID,
    results_root: str = DEFAULT_RESULTS_ROOT,
    bridge_plan: str = DEFAULT_BRIDGE_PLAN,
    base_submission_dir: str = DEFAULT_BASE_SUBMISSION_DIR,
    global_mutated_archive: str = DEFAULT_GLOBAL_MUTATED_ARCHIVE,
    upstream_dir: str = DEFAULT_UPSTREAM_DIR,
    video_names_file: str = DEFAULT_VIDEO_NAMES_FILE,
    frame_policy: str = DEFAULT_FRAME_POLICY,
) -> dict[str, Any]:
    date = lane_date or _lane_date_from_summary_path(selection.action_summary_path)
    selected_pairs = ",".join(str(index) for index in selection.selected_pair_indices)
    materialized_root = f"{results_root}/materialized/{selection.candidate_slug}"
    packet_plan = f"{results_root}/selector_pareto/packet_plans/{selection.candidate_slug}.json"
    packet_plan_md = f"{results_root}/selector_pareto/packet_plans/{selection.candidate_slug}.md"
    experiment_id = selection.candidate_id

    queue = {
        "schema": QUEUE_SCHEMA,
        "queue_id": queue_id,
        "controls": {
            "mode": "running",
            "local_first": True,
            "max_concurrency": {
                "local_cpu": 1,
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
                "steps": [
                    {
                        "id": "plan_packet",
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
                            "--force",
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
                        "timeout_seconds": 900,
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
                            "600",
                            "--work-dir",
                            f"{materialized_root}/locality_work",
                            "--json-out",
                            f"{materialized_root}/locality_controls.json",
                        ],
                        "resources": {"kind": "local_cpu"},
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
                        "timeout_seconds": 1200,
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
                            "600",
                            "--evaluate-timeout",
                            "900",
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
                ],
            }
        ],
    }
    return normalize_queue_definition(queue)


def build_queue_from_action_summary(
    action_summary_path: str | Path,
    *,
    repo_root: str | Path,
    results_root: str = DEFAULT_RESULTS_ROOT,
    bridge_plan: str = DEFAULT_BRIDGE_PLAN,
    base_submission_dir: str = DEFAULT_BASE_SUBMISSION_DIR,
    global_mutated_archive: str = DEFAULT_GLOBAL_MUTATED_ARCHIVE,
    upstream_dir: str = DEFAULT_UPSTREAM_DIR,
    video_names_file: str = DEFAULT_VIDEO_NAMES_FILE,
    frame_policy: str = DEFAULT_FRAME_POLICY,
    exclude_candidate_ids: set[str] | None = None,
    skip_completed_local_advisory: bool = True,
) -> Dqs1QueueBuildResult:
    selection = select_dqs1_local_first_candidate(
        action_summary_path,
        repo_root=repo_root,
        results_root=results_root,
        exclude_candidate_ids=exclude_candidate_ids,
        skip_completed_local_advisory=skip_completed_local_advisory,
    )
    return Dqs1QueueBuildResult(
        queue=build_dqs1_local_first_queue(
            selection,
            results_root=results_root,
            bridge_plan=bridge_plan,
            base_submission_dir=base_submission_dir,
            global_mutated_archive=global_mutated_archive,
            upstream_dir=upstream_dir,
            video_names_file=video_names_file,
            frame_policy=frame_policy,
        ),
        selection=selection,
    )
