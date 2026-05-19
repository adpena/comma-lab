# SPDX-License-Identifier: MIT
"""HF Jobs routing-target consumer for cathedral autopilot.

Per Catalog #335 this package exposes the canonical cathedral consumer
contract so the autopilot auto-discovery loop sees HF Jobs beside the existing
Modal / Lightning / Vast.ai / local routing surfaces.

This consumer is observability-only. It never imports ``huggingface_hub``,
never calls the HF Jobs dispatcher, and never registers a ledger row. Actual
spend remains behind the canonical operator-authorize / dispatcher path:
``tools/dispatch_hf_jobs_vision_training.py`` plus
``tac.deploy.hf_jobs.job_id_ledger``.

Hook assignments per Catalog #125:

* #4 cathedral autopilot dispatch — ACTIVE. The consumer emits a
  non-promotable routing annotation when a candidate contains a minimal HF
  Jobs plan and the canonical HF Jobs helper surfaces are present.
* #1, #2, #3, #5, #6 — N/A. This package is not a sensitivity, Pareto,
  bit-allocation, posterior, or probe-disambiguator surface.
"""
from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "hf_jobs_dispatcher_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)

ROUTE_HF_JOBS = "hf_jobs"
ROUTE_NONE = "none"

HF_JOBS_HELPER_MODULE = "tac.deploy.hf_jobs.job_id_ledger"
HF_JOBS_DISPATCHER_CLI = "tools/dispatch_hf_jobs_vision_training.py"
DEFAULT_HF_JOBS_SCRIPT = "experiments/hf_jobs_segnet_surrogate_distillation.py"
DEFAULT_HF_JOBS_HUB_DATASET_REPO = "adpena/comma-video-segnet-image-level-600pairs"
DEFAULT_HF_JOBS_HUB_MODEL_REPO = (
    "adpena/segnet-image-level-surrogate-mobilenet-v3-small-200ep"
)
DEFAULT_HF_JOBS_FLAVOR = "t4-small"

_REQUIRED_HELPER_SYMBOLS = (
    "HF_JOBS_CALL_ID_LEDGER_PATH",
    "SCHEMA_VERSION",
    "register_dispatched_hf_jobs_id_fail_closed",
    "poll_ledger_for_hf_jobs_id",
    "query_by_lane",
)

_HF_JOBS_CONFIG_FIELDS = ("hf_jobs", "hf_jobs_config", "hf_jobs_plan")
_SCRIPT_FIELDS = ("script", "script_path", "training_script", "hf_jobs_script")
_DATASET_FIELDS = ("hub_dataset_repo", "dataset_repo", "dataset")
_MODEL_REPO_FIELDS = ("hub_model_repo", "model_repo", "hub_model_id")
_FLAVOR_FIELDS = ("flavor", "hf_jobs_flavor", "hardware_flavor")
_LANE_FIELDS = ("lane_id", "candidate_id", "id")
_ROUTE_MARKER_FIELDS = (
    "platform",
    "provider",
    "route",
    "recommended_route",
    "routing_target",
    "candidate_id",
    "lane_id",
)

_ROUTE_COMPETITORS = ("modal", "lightning", "vastai", "local_mps", "local_cpu")


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 callable surface.

    HF Jobs dispatch lifecycle updates are owned by the canonical
    ``tac.deploy.hf_jobs.job_id_ledger`` helper. This routing consumer keeps no
    separate posterior and deliberately performs no state mutation.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — emit an HF Jobs routing annotation.

    The returned mapping is always non-promotable:
    ``predicted_delta_adjustment=0.0``, ``axis_tag="[predicted]"``, and
    ``promotable=False``. A valid candidate only means the autopilot can show
    HF Jobs as a target option; the actual dispatcher remains gated by the
    operator-authorize path and ledger fail-closed helpers.
    """
    if not isinstance(candidate, Mapping):
        return _fail_closed(
            "candidate is not a mapping; HF Jobs routing refused",
            failure_class="candidate_not_mapping",
        )

    helper_status = hf_jobs_helper_status()
    if not helper_status["available"]:
        missing = ", ".join(helper_status["missing_helpers"])
        return _fail_closed(
            "HF Jobs routing refused because canonical helper surface is "
            f"incomplete: {missing}",
            failure_class="missing_hf_jobs_helper",
            helper_status=helper_status,
        )

    plan = build_hf_jobs_route_candidate(candidate)
    if not plan["accepted"]:
        return _fail_closed(
            str(plan["reason"]),
            failure_class=str(plan["failure_class"]),
            helper_status=helper_status,
            route_candidate=plan,
        )

    routing_notes = {
        "accepted": True,
        "route": ROUTE_HF_JOBS,
        "route_competes_with": list(_ROUTE_COMPETITORS),
        "dispatcher_cli": HF_JOBS_DISPATCHER_CLI,
        "ledger_helper": HF_JOBS_HELPER_MODULE,
        "ledger_schema_version": helper_status["schema_version"],
        "operator_authorize_required": True,
        "consumer_dispatches_jobs": False,
        "dry_run_only_from_consumer": True,
        "dispatch_ready": False,
        "ready_for_exact_eval_dispatch": False,
        "plan": plan["plan"],
    }
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "HF Jobs route candidate available via canonical dispatcher "
            f"{HF_JOBS_DISPATCHER_CLI} and ledger helper "
            f"{HF_JOBS_HELPER_MODULE}; observability-only route annotation "
            "requires operator-authorize before spend [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_ready": False,
        "confidence": 0.0,
        "recommended_route": ROUTE_HF_JOBS,
        "routing_target": ROUTE_HF_JOBS,
        "routing_target_candidate": routing_notes,
        "notes": {"hf_jobs_dispatcher_consumer": routing_notes},
    }


def hf_jobs_helper_status() -> dict[str, Any]:
    """Return canonical HF Jobs helper availability without dispatch side effects."""
    missing_helpers: list[str] = []
    schema_version: str | None = None
    try:
        helper = importlib.import_module(HF_JOBS_HELPER_MODULE)
    except ImportError:
        helper = None
        missing_helpers.append(f"module:{HF_JOBS_HELPER_MODULE}")

    if helper is not None:
        for symbol in _REQUIRED_HELPER_SYMBOLS:
            if not hasattr(helper, symbol):
                missing_helpers.append(f"{HF_JOBS_HELPER_MODULE}.{symbol}")
        raw_schema = getattr(helper, "SCHEMA_VERSION", None)
        if isinstance(raw_schema, str) and raw_schema:
            schema_version = raw_schema

    dispatcher_path = _repo_root() / HF_JOBS_DISPATCHER_CLI
    if not dispatcher_path.is_file():
        missing_helpers.append(HF_JOBS_DISPATCHER_CLI)

    return {
        "available": not missing_helpers,
        "missing_helpers": missing_helpers,
        "helper_module": HF_JOBS_HELPER_MODULE,
        "dispatcher_cli": HF_JOBS_DISPATCHER_CLI,
        "schema_version": schema_version,
    }


def build_hf_jobs_route_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a minimal HF Jobs routing plan from a cathedral candidate."""
    config = _extract_hf_jobs_config(candidate)
    script = _string_from_any(config, _SCRIPT_FIELDS)
    dataset_repo = _string_from_any(config, _DATASET_FIELDS)
    model_repo = _string_from_any(config, _MODEL_REPO_FIELDS)
    flavor = _string_from_any(config, _FLAVOR_FIELDS) or DEFAULT_HF_JOBS_FLAVOR
    lane_id = _string_from_any(config, _LANE_FIELDS)

    defaulted_fields: list[str] = []
    if _candidate_requests_hf_jobs_route(config):
        if not script:
            script = DEFAULT_HF_JOBS_SCRIPT
            defaulted_fields.append("script")
        if not dataset_repo:
            dataset_repo = DEFAULT_HF_JOBS_HUB_DATASET_REPO
            defaulted_fields.append("hub_dataset_repo")
        if not model_repo:
            model_repo = DEFAULT_HF_JOBS_HUB_MODEL_REPO
            defaulted_fields.append("hub_model_repo")

    missing_fields = [
        name
        for name, value in (
            ("script", script),
            ("hub_dataset_repo", dataset_repo),
            ("hub_model_repo", model_repo),
        )
        if not value
    ]
    if missing_fields:
        return {
            "accepted": False,
            "failure_class": "missing_required_hf_jobs_fields",
            "reason": (
                "HF Jobs route candidate missing required field(s): "
                + ", ".join(missing_fields)
            ),
            "missing_fields": missing_fields,
        }

    script_path = _repo_root() / str(script)
    if not script_path.is_file():
        return {
            "accepted": False,
            "failure_class": "hf_jobs_script_missing",
            "reason": f"HF Jobs training script missing: {script}",
            "missing_fields": ["script_path_exists"],
            "script": script,
        }

    return {
        "accepted": True,
        "failure_class": None,
        "reason": "minimal HF Jobs route candidate accepted",
        "plan": {
            "platform": ROUTE_HF_JOBS,
            "script": str(script),
            "hub_dataset_repo": str(dataset_repo),
            "hub_model_repo": str(model_repo),
            "flavor": str(flavor),
            "lane_id": str(lane_id) if lane_id else None,
            "defaulted_fields": defaulted_fields,
        },
    }


def _extract_hf_jobs_config(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return nested HF Jobs config if present, otherwise the candidate itself."""
    for field in _HF_JOBS_CONFIG_FIELDS:
        value = candidate.get(field)
        if isinstance(value, Mapping):
            merged = dict(candidate)
            merged.update(value)
            return merged
    return candidate


def _string_from_any(candidate: Mapping[str, Any], fields: tuple[str, ...]) -> str | None:
    """Return the first non-empty string-like field from ``candidate``."""
    for field in fields:
        value = candidate.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Path):
            return str(value)
    return None


def _candidate_requests_hf_jobs_route(candidate: Mapping[str, Any]) -> bool:
    """Return True when a candidate explicitly names the HF Jobs route family."""
    for field in _ROUTE_MARKER_FIELDS:
        value = candidate.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized == ROUTE_HF_JOBS or normalized.startswith("hf_jobs"):
            return True
        if normalized.startswith("lane_hf_jobs") or "_hf_jobs_" in normalized:
            return True
    return False


def _fail_closed(
    rationale: str,
    *,
    failure_class: str,
    helper_status: Mapping[str, Any] | None = None,
    route_candidate: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return a non-promotable no-route verdict with diagnostic notes."""
    notes: dict[str, Any] = {
        "accepted": False,
        "route": ROUTE_NONE,
        "failure_class": failure_class,
        "consumer_dispatches_jobs": False,
        "dispatch_ready": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if helper_status is not None:
        notes["helper_status"] = dict(helper_status)
    if route_candidate is not None:
        notes["route_candidate"] = dict(route_candidate)
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": f"{rationale}; no dispatch and no score claim [predicted]",
        "axis_tag": "[predicted]",
        "promotable": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_ready": False,
        "confidence": 0.0,
        "recommended_route": ROUTE_NONE,
        "routing_target": ROUTE_NONE,
        "notes": {"hf_jobs_dispatcher_consumer": notes},
    }


def _repo_root() -> Path:
    """Locate repo root from this package without importing operator tools."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists() and (parent / ".omx").exists():
            return parent
    return Path.cwd()


__all__ = [
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "DEFAULT_HF_JOBS_FLAVOR",
    "DEFAULT_HF_JOBS_HUB_DATASET_REPO",
    "DEFAULT_HF_JOBS_HUB_MODEL_REPO",
    "DEFAULT_HF_JOBS_SCRIPT",
    "HF_JOBS_DISPATCHER_CLI",
    "HF_JOBS_HELPER_MODULE",
    "ROUTE_HF_JOBS",
    "ROUTE_NONE",
    "build_hf_jobs_route_candidate",
    "consume_candidate",
    "hf_jobs_helper_status",
    "update_from_anchor",
]
