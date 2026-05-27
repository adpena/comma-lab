# SPDX-License-Identifier: MIT
"""Build queue-owned repair campaign scoring rows from waterfill queues."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA, normalize_queue_definition
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
)
from tac.optimization.repair_campaign_posterior import (
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    REPAIR_CAMPAIGN_BLOCKED_POSTERIOR_APPEND_REPORT_SCHEMA,
)
from tac.optimization.repair_campaign_scorer import REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA

REPAIR_CAMPAIGN_SCORE_QUEUE_METADATA_SCHEMA = (
    "repair_campaign_score_queue_metadata.v1"
)
REPAIR_CAMPAIGN_SCORE_EXPERIMENT_METADATA_SCHEMA = (
    "repair_campaign_score_experiment_metadata.v1"
)
DEFAULT_STACKABILITY_WORKER_MAX_STEPS = 8
DEFAULT_STACKABILITY_WORKER_MAX_EXPERIMENTS = 2
DEFAULT_STACKABILITY_WORKER_MAX_PARALLEL = 1
DEFAULT_STACKABILITY_WORKER_TIMEOUT_SECONDS = 900


class RepairCampaignScoreQueueError(ValueError):
    """Raised when a repair campaign score queue cannot be built."""


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "unknown"


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    resolved = Path(path)
    repo = Path(repo_root)
    try:
        return str(resolved.resolve(strict=False).relative_to(repo.resolve(strict=False)))
    except ValueError:
        return str(resolved)


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(repo_root) / value


def _command_arg(command: Sequence[Any], flag: str) -> str:
    values = [str(item) for item in command]
    try:
        index = values.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(values):
        return ""
    return values[index + 1]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else 0


def _work_order_path_from_experiment(experiment: Mapping[str, Any]) -> str:
    metadata = experiment.get("metadata")
    if isinstance(metadata, Mapping):
        for key in (
            "repair_budget_waterfill_work_order_path",
            "waterfill_work_order_path",
            "work_order_path",
        ):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
    for step in experiment.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        command = step.get("command")
        if isinstance(command, Sequence) and not isinstance(
            command,
            (str, bytes, bytearray),
        ):
            command_text = " ".join(str(item) for item in command)
            if (
                step.get("id") != "emit_repair_budget_waterfill_work_order"
                and "build_frontier_repair_budget_waterfill_work_order.py"
                not in command_text
                and "--work-order-out" not in command_text
            ):
                continue
            value = _command_arg(command, "--work-order-out")
            if value:
                return value
    return ""


def _score_experiment(
    *,
    source_experiment: Mapping[str, Any],
    source_queue_path: str | Path,
    repo_root: str | Path,
    results_root: str | Path,
    queue_root: Path,
    priority: int,
    posterior_path: str | Path | None,
) -> dict[str, Any]:
    metadata = (
        source_experiment.get("metadata")
        if isinstance(source_experiment.get("metadata"), Mapping)
        else {}
    )
    chain_id = str(
        metadata.get("chain_id")
        or source_experiment.get("id")
        or f"repair_campaign_{priority}"
    )
    work_order_ref = _work_order_path_from_experiment(source_experiment)
    score_report_path = (
        queue_root / _slug(chain_id) / "repair_campaign_score_report.json"
    )
    score_report_ref = _repo_rel(score_report_path, repo_root)
    stackability_queue_path = (
        queue_root / _slug(chain_id) / "repair_campaign_stackability_queue.json"
    )
    stackability_queue_ref = _repo_rel(stackability_queue_path, repo_root)
    stackability_worker_result_path = (
        queue_root / _slug(chain_id) / "repair_campaign_stackability_worker_result.json"
    )
    stackability_worker_result_ref = _repo_rel(
        stackability_worker_result_path,
        repo_root,
    )
    blocked_learning_signal_report_path = (
        queue_root / _slug(chain_id) / "repair_campaign_blocked_learning_signal_report.json"
    )
    blocked_learning_signal_report_ref = _repo_rel(
        blocked_learning_signal_report_path,
        repo_root,
    )
    blocked_posterior_append_report_path = (
        queue_root / _slug(chain_id) / "repair_campaign_blocked_posterior_append_report.json"
    )
    blocked_posterior_append_report_ref = _repo_rel(
        blocked_posterior_append_report_path,
        repo_root,
    )
    posterior_ref = (
        _repo_rel(_resolve(posterior_path, repo_root), repo_root)
        if posterior_path is not None
        else None
    )
    blockers = []
    if not work_order_ref:
        blockers.append("repair_budget_waterfill_work_order_path_missing")
        if _positive_int(metadata.get("typed_response_row_count")) == 0:
            blockers.append("repair_budget_waterfill_typed_response_ledger_empty")
        if metadata.get("queue_actuation_ready") is False:
            blockers.append("source_repair_budget_waterfill_queue_not_actuation_ready")
        blockers.extend(
            f"source_queue_actuation_blocker:{blocker}"
            for blocker in _string_list(metadata.get("queue_actuation_blockers"))
        )
        blockers.extend(
            f"source_missing_prerequisite_artifact:{key}"
            for key in _string_list(metadata.get("missing_prerequisite_artifact_keys"))
        )
    work_order_exists_at_build = (
        bool(work_order_ref) and _resolve(work_order_ref, repo_root).is_file()
    )
    status = "queued" if not blockers else "frozen"
    experiment_metadata = {
        "schema": REPAIR_CAMPAIGN_SCORE_EXPERIMENT_METADATA_SCHEMA,
        "source_queue_path": str(source_queue_path),
        "source_experiment_id": source_experiment.get("id"),
        "chain_id": chain_id,
        "repair_budget_waterfill_work_order_path": work_order_ref or None,
        "repair_budget_waterfill_work_order_exists_at_build": (
            work_order_exists_at_build
        ),
        "repair_campaign_score_report_path": score_report_ref,
        "repair_campaign_stackability_queue_path": stackability_queue_ref,
        "repair_campaign_stackability_worker_result_path": (
            stackability_worker_result_ref
        ),
        "repair_campaign_blocked_learning_signal_report_path": (
            blocked_learning_signal_report_ref
        ),
        "repair_campaign_blocked_posterior_append_report_path": (
            blocked_posterior_append_report_ref
        ),
        "repair_campaign_stackability_posterior_path": posterior_ref,
        "campaign_scorer_default": True,
        "campaign_scorer_uses_posterior_priors": posterior_ref is not None,
        "stackability_followup_default": True,
        "continual_learning_followup_default": True,
        "blocked_learning_followup_default": True,
        "stackability_worker_limits": {
            "schema": "repair_campaign_stackability_worker_limits.v1",
            "max_steps": DEFAULT_STACKABILITY_WORKER_MAX_STEPS,
            "max_experiments": DEFAULT_STACKABILITY_WORKER_MAX_EXPERIMENTS,
            "max_parallel": DEFAULT_STACKABILITY_WORKER_MAX_PARALLEL,
            "timeout_seconds": DEFAULT_STACKABILITY_WORKER_TIMEOUT_SECONDS,
        },
        "queue_actuation_ready": not blockers,
        "queue_actuation_blockers": ordered_unique(blockers),
        "source_queue_actuation_blockers": _string_list(
            metadata.get("queue_actuation_blockers")
        ),
        "source_missing_prerequisite_artifact_keys": _string_list(
            metadata.get("missing_prerequisite_artifact_keys")
        ),
        "score_report_schema": REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "queue_owned_repair_campaign_scoring_metadata",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        experiment_metadata,
        context=f"repair_campaign_score_experiment_metadata:{chain_id}",
    )
    score_command = [
        ".venv/bin/python",
        "tools/score_repair_campaign.py",
        "--work-order",
        work_order_ref,
        "--score-report-out",
        score_report_ref,
        "--overwrite",
    ]
    if posterior_ref is not None:
        score_command.extend(["--posterior", posterior_ref])
    steps: list[dict[str, Any]]
    if blockers:
        steps = [
            {
                "id": "inspect_missing_repair_campaign_score_prerequisites",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "-c",
                    (
                        "import json; print(json.dumps({"
                        "'ready_for_exact_eval_dispatch': False,"
                        "'budget_spend_allowed': False,"
                        f"'blockers': {ordered_unique(blockers)!r}"
                        "}, sort_keys=True))"
                    ),
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 30,
                "telemetry": {"include_postcondition_paths": True},
            }
        ]
    else:
        steps = [
            {
                "id": "assert_repair_budget_waterfill_work_order_materialized",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "-c",
                    (
                        "import json, sys; from pathlib import Path; "
                        "p = Path(sys.argv[1]); ok = p.is_file(); "
                        "print(json.dumps({"
                        "'schema': 'repair_campaign_score_prerequisite_check.v1', "
                        "'work_order': str(p), "
                        "'work_order_exists': ok, "
                        "'budget_spend_allowed': False, "
                        "'ready_for_exact_eval_dispatch': False, "
                        "'score_claim': False, "
                        "'promotion_eligible': False, "
                        "'rank_or_kill_eligible': False"
                        "}, sort_keys=True)); "
                        "raise SystemExit(0 if ok else 2)"
                    ),
                    work_order_ref,
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 30,
                "telemetry": {"input_artifact_paths": [work_order_ref]},
            },
            {
                "id": "score_repair_campaign_from_typed_ledger",
                "kind": "command",
                "requires": [
                    "assert_repair_budget_waterfill_work_order_materialized"
                ],
                "command": score_command,
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": score_report_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": score_report_ref,
                    },
                    {
                        "type": "json_equals",
                        "path": score_report_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                    {
                        "type": "json_equals",
                        "path": score_report_ref,
                        "key": "budget_spend_allowed",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [score_report_ref],
                    "input_artifact_paths": [work_order_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "build_repair_campaign_blocked_learning_signals",
                "kind": "command",
                "requires": ["score_repair_campaign_from_typed_ledger"],
                "command": [
                    ".venv/bin/python",
                    "tools/build_repair_campaign_blocked_learning_signals.py",
                    "--score-report",
                    score_report_ref,
                    "--blocked-signal-report-out",
                    blocked_learning_signal_report_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": blocked_learning_signal_report_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": blocked_learning_signal_report_ref,
                    },
                    {
                        "type": "json_equals",
                        "path": blocked_learning_signal_report_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [blocked_learning_signal_report_ref],
                    "input_artifact_paths": [score_report_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "append_blocked_repair_campaign_learning_posterior",
                "kind": "command",
                "requires": ["build_repair_campaign_blocked_learning_signals"],
                "command": [
                    ".venv/bin/python",
                    "tools/append_repair_campaign_blocked_posterior.py",
                    "--blocked-learning-signal-report",
                    blocked_learning_signal_report_ref,
                    "--report-out",
                    blocked_posterior_append_report_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": blocked_posterior_append_report_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_BLOCKED_POSTERIOR_APPEND_REPORT_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": blocked_posterior_append_report_ref,
                    },
                    {
                        "type": "json_equals",
                        "path": blocked_posterior_append_report_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [blocked_posterior_append_report_ref],
                    "input_artifact_paths": [blocked_learning_signal_report_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "build_repair_campaign_stackability_queue",
                "kind": "command",
                "requires": ["score_repair_campaign_from_typed_ledger"],
                "command": [
                    ".venv/bin/python",
                    "tools/build_repair_campaign_stackability_queue.py",
                    "--score-report",
                    score_report_ref,
                    "--stackability-queue-out",
                    stackability_queue_ref,
                    "--results-root",
                    str(results_root),
                    "--queue-id",
                    f"{_slug(chain_id)}_repair_stackability",
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": stackability_queue_ref,
                        "key": "schema",
                        "equals": QUEUE_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": stackability_queue_ref,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [stackability_queue_ref],
                    "input_artifact_paths": [score_report_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "validate_repair_campaign_stackability_queue",
                "kind": "command",
                "requires": ["build_repair_campaign_stackability_queue"],
                "command": [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    stackability_queue_ref,
                    "validate",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "telemetry": {
                    "input_artifact_paths": [stackability_queue_ref],
                },
            },
            {
                "id": "run_repair_campaign_stackability_queue_bounded_local",
                "kind": "command",
                "requires": ["validate_repair_campaign_stackability_queue"],
                "command": [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    stackability_queue_ref,
                    "run-worker",
                    "--execute",
                    "--max-steps",
                    str(DEFAULT_STACKABILITY_WORKER_MAX_STEPS),
                    "--max-experiments",
                    str(DEFAULT_STACKABILITY_WORKER_MAX_EXPERIMENTS),
                    "--max-parallel",
                    str(DEFAULT_STACKABILITY_WORKER_MAX_PARALLEL),
                    "--output",
                    stackability_worker_result_ref,
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": DEFAULT_STACKABILITY_WORKER_TIMEOUT_SECONDS,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": stackability_worker_result_ref,
                        "key": "schema",
                        "equals": "experiment_queue_worker_result.v1",
                    },
                    {
                        "type": "json_equals",
                        "path": stackability_worker_result_ref,
                        "key": "failure_count",
                        "equals": 0,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [stackability_worker_result_ref],
                    "input_artifact_paths": [stackability_queue_ref, score_report_ref],
                    "include_postcondition_paths": True,
                },
            },
        ]
    return {
        "id": f"score_repair_campaign_{_slug(chain_id)}",
        "priority": priority,
        "status": status,
        "tags": [
            "frontier-rate-attack",
            "repair-campaign-scorer",
            "default-campaign-scorer",
            "no-score-authority",
        ],
        "metadata": experiment_metadata,
        "steps": steps,
    }


def build_repair_campaign_score_queue(
    *,
    repo_root: str | Path,
    repair_budget_waterfill_queue: Mapping[str, Any],
    repair_budget_waterfill_queue_path: str | Path,
    results_root: str | Path,
    queue_id: str = "repair_campaign_score_queue",
    experiment_limit: int | None = None,
    posterior_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a queue that scores repair-waterfill work orders."""

    if repair_budget_waterfill_queue.get("schema") != QUEUE_SCHEMA:
        raise RepairCampaignScoreQueueError("repair waterfill input must be experiment_queue.v1")
    require_no_truthy_authority_fields(
        repair_budget_waterfill_queue,
        context="repair_campaign_score_queue_input",
    )
    source_experiments = [
        experiment
        for experiment in repair_budget_waterfill_queue.get("experiments") or []
        if isinstance(experiment, Mapping)
    ]
    if experiment_limit is not None:
        source_experiments = source_experiments[: max(0, int(experiment_limit))]
    queue_root = (
        _resolve(results_root, repo_root)
        / "repair_campaign_score_queue"
        / _slug(queue_id)
    )
    experiments = [
        _score_experiment(
            source_experiment=experiment,
            source_queue_path=repair_budget_waterfill_queue_path,
            repo_root=repo_root,
            results_root=results_root,
            queue_root=queue_root,
            priority=priority,
            posterior_path=posterior_path,
        )
        for priority, experiment in enumerate(source_experiments, start=1)
    ]
    ready_count = sum(
        1
        for experiment in experiments
        if experiment["metadata"].get("queue_actuation_ready") is True
    )
    metadata = {
        "schema": REPAIR_CAMPAIGN_SCORE_QUEUE_METADATA_SCHEMA,
        "source_queue_path": str(repair_budget_waterfill_queue_path),
        "source_queue_id": repair_budget_waterfill_queue.get("queue_id"),
        "campaign_scorer_default": True,
        "campaign_scorer_uses_posterior_priors": posterior_path is not None,
        "repair_campaign_stackability_posterior_path": (
            _repo_rel(_resolve(posterior_path, repo_root), repo_root)
            if posterior_path is not None
            else None
        ),
        "score_report_schema": REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
        "experiment_count": len(experiments),
        "ready_experiment_count": ready_count,
        "blocked_experiment_count": len(experiments) - ready_count,
        "results_root": _repo_rel(queue_root, repo_root),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_campaign_score_queue_for_local_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context="repair_campaign_score_queue_metadata",
    )
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {"local_io_heavy": 1, "local_cpu": 1},
            },
            "metadata": metadata,
            "experiments": experiments,
        }
    )


__all__ = [
    "DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH",
    "REPAIR_CAMPAIGN_SCORE_EXPERIMENT_METADATA_SCHEMA",
    "REPAIR_CAMPAIGN_SCORE_QUEUE_METADATA_SCHEMA",
    "RepairCampaignScoreQueueError",
    "build_repair_campaign_score_queue",
]
