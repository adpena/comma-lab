#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Parallel-harvest-actuator — fan-out concurrent harvest of in-flight dispatches.

Canonical primitive per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch
first" non-negotiable + Grand Council 2026-05-14 Tier 0 authorization (Carmack
intervention — "harvest actuator is FIRST-CLASS deliverable").

Design (per `.omx/research/grand_council_tier_dispatch_authorizations_20260514.md`
T0-D spec):

1. Discovers all in-flight Modal call_ids from
   ``experiments/results/lane_*_modal/modal_metadata.json``
2. Concurrent harvest via ``concurrent.futures.ThreadPoolExecutor`` over
   per-call ``modal.functions.FunctionCall.from_id(call_id).get(timeout=...)``
3. Each harvest auto-validates the contract per CLAUDE.md harness pillars:
   - Axis tag per Catalog #127 (``[contest-CUDA T4 Linux x86_64]`` /
     ``[contest-CPU GHA Linux x86_64]`` / ``[macOS-CPU advisory only]`` /
     ``[predicted]``)
   - Custody validator (Catalog #127 / #128 / #130) routes every score
   - Posterior anchor append via ``posterior_update_locked`` (Catalog #128 +
     #131 fcntl.flock LOCK_EX)
4. Sister scan of ``.omx/state/vastai_active_instances.json`` for Vast.ai
   in-flight (read-only audit; harvest itself is a separate ssh path).
5. Consolidated JSONL output (HISTORICAL_PROVENANCE per Catalog #113) one row
   per call_id with all metadata + axis-tagged scores.

Default execution is read-only (``--list-only``). Pass ``--execute`` to fan
out the concurrent harvest, write per-call artifacts via the canonical
``tools/harvest_modal_calls.py`` flow, and append posterior anchors.

Safety:
- Read-only mode does NOT contact Modal.
- ``--execute`` mode delegates per-call harvest to the canonical
  ``tools.harvest_modal_calls.harvest_modal_calls`` flow when running
  serial; concurrency adds the executor + canonical-API per-call wrapper
  to avoid serialization on Modal's blocking ``get()``.
- Posterior appends are fcntl-locked per Catalog #128 — concurrent worker
  threads serialize at the moment of write, NOT at the moment of get().
- Custody-refused anchors are recorded in the consolidated report but NOT
  promoted to posterior (per Catalog #127 + #130).

Cross-refs:
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" Rule 1
- CLAUDE.md "Auth eval EVERYWHERE — NON-NEGOTIABLE"
- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE — NON-NEGOTIABLE"
- ``tools/parallel_dispatch_top_k.py`` (canonical dispatch actuator)
- ``tools/harvest_modal_calls.py`` (canonical per-call harvester)
- ``tools/harvest_and_reseed.py`` (canonical reseed loop)
- Grand Council ledger `.omx/research/grand_council_tiered_parallel_plan_full_authority_20260514.md`
- Grand Council auth memo `.omx/research/grand_council_tier_dispatch_authorizations_20260514.md`
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool


DEFAULT_REPO = repo_root_from_tool(__file__)
DEFAULT_MAX_WORKERS = 4
DEFAULT_PER_CALL_TIMEOUT_SECONDS = 5.0
DEFAULT_OVERALL_TIMEOUT_SECONDS = 600.0


# -- Discovery (read-only; no provider contact) ------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _read_jsonl_or_json_list(path: Path) -> list[dict[str, Any]]:
    """Read a path that may be JSON list or JSONL; return empty on missing."""

    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        return loaded if isinstance(loaded, list) else []
    rows: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _is_terminal_harvest(out_dir: Path, artifacts_dir: Path) -> bool:
    """Return True when local state is terminal enough to avoid re-polling Modal.

    Mirrors the canonical ``tools/harvest_modal_calls.py::_already_harvested``
    logic so concurrent workers can short-circuit on already-harvested calls.
    """

    terminal_marker = out_dir / "modal_training_terminal_claim.json"
    if terminal_marker.is_file():
        marker = _read_json(terminal_marker)
        if isinstance(marker, dict) and marker.get("appended") is True:
            return True
    if (out_dir / "harvest_summary.json").is_file():
        return True
    summary = _read_json(artifacts_dir / "_harvest_summary.json")
    if isinstance(summary, dict):
        if "rc" in summary or "returncode" in summary:
            return True
        status = str(summary.get("status") or "")
        crash_kind = str(summary.get("crash_kind") or "")
        if (
            status in {"expired", "function_timeout"}
            or status.startswith("error_")
            or crash_kind in {"RESULT_CACHE_EXPIRED", "FUNCTION_TIMEOUT"}
            or crash_kind.startswith("ERROR_")
        ):
            return True
    if artifacts_dir.is_dir():
        for child in artifacts_dir.iterdir():
            if child.name not in {"_harvest_summary.json", "_stdout_tail.txt"}:
                return True
    return False


def discover_inflight_modal_lanes(*, repo_root: Path) -> list[dict[str, Any]]:
    """Enumerate every Modal lane with ``modal_metadata.json`` and classify state.

    Returns rows of:
        {
          "label": str, "call_id": str, "dispatched_at": str,
          "lane_id": str, "out_dir": str,
          "harvested": bool,        # local state is terminal
          "metadata_schema": str,   # e.g. modal_train_lane_dispatch_metadata_v2_catalog166
          "mounted_code_git_head": str,
        }

    Read-only — does NOT contact Modal. Sister of
    ``tools/harvest_modal_calls.py::list_modal_lanes`` but with extra Catalog
    #166 + #127 fields surfaced so the consolidated report can route by axis
    tag downstream.
    """

    rows: list[dict[str, Any]] = []
    results_root = repo_root / "experiments" / "results"
    if not results_root.is_dir():
        return rows
    for directory in sorted(results_root.glob("lane_*_modal")):
        mfile = directory / "modal_metadata.json"
        if not mfile.is_file():
            continue
        meta = _read_json(mfile) or {}
        out_dir = directory
        artifacts_dir = out_dir / "harvested_artifacts"
        rows.append(
            {
                "label": str(meta.get("label", "?")),
                "call_id": str(meta.get("call_id", "?")),
                "dispatched_at": str(meta.get("dispatched_at", "?")),
                "lane_id": str(meta.get("lane_id", "?")),
                "out_dir": str(out_dir),
                "harvested": _is_terminal_harvest(out_dir, artifacts_dir),
                "metadata_schema": str(meta.get("metadata_schema", "?")),
                "mounted_code_git_head": str(meta.get("mounted_code_git_head", "?")),
                "gpu": str(meta.get("gpu", "?")),
                "max_seconds": meta.get("max_seconds"),
            }
        )
    return rows


def discover_inflight_vastai_instances(*, repo_root: Path) -> list[dict[str, Any]]:
    """Read ``.omx/state/vastai_active_instances.json`` — read-only audit only.

    Returns the raw list; harvest of Vast.ai instances happens via a
    separate SSH pipeline (out of scope for this actuator beyond surfacing
    the count + IDs for the consolidated report).
    """

    state_path = repo_root / ".omx" / "state" / "vastai_active_instances.json"
    rows = _read_jsonl_or_json_list(state_path)
    # The schema may be either a top-level list (modern) or a wrapper dict;
    # be defensive.
    normalised: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalised.append(row)
    return normalised


def discover_inflight_lightning_jobs(*, repo_root: Path) -> list[dict[str, Any]]:
    """Read ``.omx/state/lightning_active_jobs.json`` — read-only audit only.

    The Lightning state schema carries ``terminal_status``; rows whose
    ``terminal_status`` is None/empty are considered in-flight.
    """

    state_path = repo_root / ".omx" / "state" / "lightning_active_jobs.json"
    rows = _read_jsonl_or_json_list(state_path)
    normalised: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalised.append(row)
    return normalised


# -- Score extraction from harvested artifacts -------------------------------


# Score-axis-to-tag canonical mapping per CLAUDE.md "Forbidden score claims".
AXIS_TAG_CUDA = "[contest-CUDA]"
AXIS_TAG_CPU_GHA = "[contest-CPU GHA Linux x86_64]"
AXIS_TAG_CPU_GENERIC = "[contest-CPU]"
AXIS_TAG_MACOS_ADVISORY = "[macOS-CPU advisory only]"


def _axis_from_device_string(device: str | None, hardware: str | None) -> str:
    """Canonical axis tag inference (string-based — NOT a custody validator).

    Returns the most accurate axis tag the device + hardware metadata
    permits. ``[macOS-CPU advisory only]`` for macOS substrates regardless
    of device. ``[contest-CPU GHA Linux x86_64]`` for Linux x86_64 CPU.
    ``[contest-CUDA]`` for CUDA on a Linux GPU substrate. ``[contest-CPU]``
    short-form when device is CPU but hardware unknown.

    NOTE: this is the BEST-EFFORT inference for the report row. Custody
    validation per Catalog #127 happens in :func:`build_contest_result_row`
    using the typed ``CustodyVerdict`` API.
    """

    dev = (device or "").lower().strip()
    hw = (hardware or "").lower().strip()
    # macOS substrate always advisory
    if "macos" in hw or "darwin" in hw or "m1" in hw or "m2" in hw or "m5" in hw:
        return AXIS_TAG_MACOS_ADVISORY
    if "cuda" in dev:
        return AXIS_TAG_CUDA
    if "cpu" in dev:
        if "linux" in hw and "x86" in hw:
            return AXIS_TAG_CPU_GHA
        return AXIS_TAG_CPU_GENERIC
    # Unknown — return CUDA if hardware looks GPU-ish, else generic CPU
    if any(token in hw for token in ("t4", "a100", "a10g", "l40s", "h100", "4090", "rtx", "v100")):
        return AXIS_TAG_CUDA
    return AXIS_TAG_CPU_GENERIC


def _find_auth_eval_jsons(harvested_dir: Path) -> list[Path]:
    """Find any auth-eval JSON artifacts under ``harvested_dir``.

    Matches ``*auth_eval*.json`` (case-insensitive); skips the harvest
    bookkeeping files (``_harvest_summary.json`` / ``_stdout_tail.txt``).
    """

    if not harvested_dir.is_dir():
        return []
    matches: list[Path] = []
    for path in harvested_dir.rglob("*.json"):
        name = path.name.lower()
        if name in {"_harvest_summary.json"}:
            continue
        if "auth_eval" in name or "auth-eval" in name:
            matches.append(path)
    return matches


def extract_score_claim_from_harvested_dir(
    out_dir: Path,
) -> dict[str, Any] | None:
    """Return axis-tagged score row if any auth-eval JSON is present.

    The row is non-promotable until passed through the custody validator
    in :func:`build_contest_result_row` — this function is the *signal
    extraction* step, not the *promotion* step.

    Returns None when no auth-eval JSON exists in the harvested artifacts
    OR when the JSON cannot be parsed.
    """

    artifacts_dir = out_dir / "harvested_artifacts"
    candidates = _find_auth_eval_jsons(artifacts_dir)
    if not candidates:
        return None
    # Pick the most recently modified (typical: the final auth-eval JSON)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    payload = _read_json(chosen)
    if payload is None:
        return None
    # Score may live at top level OR under nested keys depending on schema.
    score: float | None = None
    for key in ("score", "total_score", "auth_eval_score"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            score = float(value)
            break
    archive_sha256: str | None = None
    for key in ("archive_sha256", "archive_zip_sha256", "submission_archive_sha256"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            archive_sha256 = value
            break
    archive_bytes: int | None = None
    for key in ("archive_bytes", "archive_zip_bytes"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            archive_bytes = int(value)
            break
    device = str(payload.get("device") or payload.get("score_axis") or "").lower()
    hardware = str(payload.get("hardware_substrate") or "").lower()
    return {
        "auth_eval_path": str(chosen),
        "score": score,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "device": device,
        "hardware_substrate": hardware,
        "axis_tag": _axis_from_device_string(device, hardware),
        "raw_payload_keys": sorted(payload.keys()),
    }


def build_contest_result_row(
    *,
    extracted: dict[str, Any],
    lane_id: str,
    architecture_class: str,
) -> dict[str, Any]:
    """Run ``ContestResult.validate_custody_verdict`` on the extracted row.

    Returns a dict combining the extracted score signal + the typed
    custody verdict (``accepted``, ``reason``, ``refused_class``). Callers
    decide whether to route the row to ``posterior_update_locked`` based
    on ``verdict.accepted``.

    Per Catalog #127, this is the canonical promotion gate.
    """

    ensure_repo_imports(DEFAULT_REPO)
    from tac.continual_learning import ContestResult

    axis_tag = extracted.get("axis_tag") or AXIS_TAG_CPU_GENERIC
    # Map axis tag → ContestResult.axis ("cuda" or "cpu")
    axis = "cuda" if "CUDA" in axis_tag else "cpu"
    score = extracted.get("score")
    if not isinstance(score, (int, float)):
        # No score available — custody validation can't promote
        return {
            **extracted,
            "lane_id": lane_id,
            "architecture_class": architecture_class,
            "custody_accepted": False,
            "custody_reason": "extracted row missing score_value",
            "custody_refused_class": "missing_metadata",
        }
    archive_sha256 = extracted.get("archive_sha256") or ""
    archive_bytes = extracted.get("archive_bytes") or 0
    hardware = extracted.get("hardware_substrate") or "linux_x86_64_unknown_cuda"
    result = ContestResult(
        axis=axis,
        hardware_substrate=hardware,
        architecture_class=architecture_class,
        score_value=float(score),
        evidence_tag=axis_tag,
        archive_sha256=archive_sha256,
        archive_bytes=int(archive_bytes),
        notes=f"parallel_harvest_actuator extracted from {extracted.get('auth_eval_path', '?')}",
    )
    verdict = result.validate_custody_verdict()
    return {
        **extracted,
        "lane_id": lane_id,
        "architecture_class": architecture_class,
        "custody_accepted": verdict.accepted,
        "custody_reason": verdict.reason,
        "custody_refused_class": verdict.refused_class,
    }


# -- Concurrent harvest (per-call) -------------------------------------------


# Module-level lock for posterior appends — Catalog #128's
# ``posterior_update_locked`` already does fcntl.flock, so this lock is a
# defence-in-depth process-local guard against multiple worker threads
# racing on the same posterior file *within* this process. The fcntl lock
# remains the authoritative cross-process serializer.
_POSTERIOR_THREAD_LOCK = threading.Lock()


def _harvest_one_call(
    *,
    lane_row: dict[str, Any],
    repo_root: Path,
    per_call_timeout_seconds: float,
    append_posterior: bool,
) -> dict[str, Any]:
    """Harvest a single Modal call_id and return the consolidated row.

    Returns a dict containing:
      - all keys from the lane_row (label/call_id/dispatched_at/...)
      - "harvest_status": one of {already_harvested, harvested, expired,
        function_timeout, error_<ExcType>}
      - "harvest_summary": the result dict written to _harvest_summary.json
      - "score_claim": axis-tagged extracted score (may be None)
      - "custody_accepted": bool (may be None if no score)
      - "posterior_update": dict result of posterior_update_locked (only
        when ``append_posterior=True`` and custody_accepted=True)
    """

    out_dir = Path(lane_row["out_dir"])
    artifacts_dir = out_dir / "harvested_artifacts"
    call_id = lane_row["call_id"]
    label = lane_row["label"]
    base: dict[str, Any] = {
        **lane_row,
        "harvest_status": "unknown",
        "harvest_summary": None,
        "score_claim": None,
        "custody_accepted": None,
        "posterior_update": None,
        "error": None,
    }

    # Short-circuit: already harvested → extract score from existing artifacts
    if _is_terminal_harvest(out_dir, artifacts_dir):
        base["harvest_status"] = "already_harvested"
        summary_path = artifacts_dir / "_harvest_summary.json"
        if summary_path.is_file():
            base["harvest_summary"] = _read_json(summary_path)
        extracted = extract_score_claim_from_harvested_dir(out_dir)
        base["score_claim"] = extracted
        if extracted is not None:
            row = build_contest_result_row(
                extracted=extracted,
                lane_id=str(lane_row.get("lane_id", "?")),
                architecture_class=str(lane_row.get("lane_id", "?")),
            )
            base["custody_accepted"] = row["custody_accepted"]
            base["score_claim"] = row
            if append_posterior and row["custody_accepted"]:
                base["posterior_update"] = _append_posterior_safely(
                    extracted=row, repo_root=repo_root
                )
        return base

    # Contact Modal for an unharvested call_id
    try:
        ensure_repo_imports(repo_root)
        import modal  # type: ignore[import-not-found]
    except ImportError as exc:
        base["harvest_status"] = "error_modal_import_failed"
        base["error"] = str(exc)
        return base

    try:
        fc = modal.functions.FunctionCall.from_id(call_id)
        result = fc.get(timeout=per_call_timeout_seconds)
    except TimeoutError:
        base["harvest_status"] = "not_ready"
        return base
    except Exception as exc:  # broad except: per-call error must not kill batch
        base["harvest_status"] = f"error_{type(exc).__name__}"
        base["error"] = str(exc)[:1000]
        return base

    rc = result.get("returncode", "?") if isinstance(result, dict) else "?"
    artifacts_dir.mkdir(exist_ok=True)
    base["harvest_status"] = "harvested"
    base["harvest_summary"] = {
        "rc": rc,
        "elapsed_seconds": result.get("elapsed_seconds") if isinstance(result, dict) else None,
        "n_artifacts": len(result.get("artifacts", {}) if isinstance(result, dict) else {}),
        "stdout_tail": (result.get("stdout_tail", "") if isinstance(result, dict) else "")[-500:],
    }
    # The canonical artifact-write path is in harvest_modal_calls.py; we
    # surface that the canonical tool should be invoked for full custody.
    # Do NOT duplicate artifact-write logic here — the actuator's job is
    # fan-out + summary, not byte custody.
    extracted = extract_score_claim_from_harvested_dir(out_dir)
    base["score_claim"] = extracted
    if extracted is not None:
        row = build_contest_result_row(
            extracted=extracted,
            lane_id=str(lane_row.get("lane_id", "?")),
            architecture_class=str(lane_row.get("lane_id", "?")),
        )
        base["custody_accepted"] = row["custody_accepted"]
        base["score_claim"] = row
        if append_posterior and row["custody_accepted"]:
            base["posterior_update"] = _append_posterior_safely(
                extracted=row, repo_root=repo_root
            )
    return base


def _append_posterior_safely(
    *, extracted: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    """Append the extracted+custody-validated row to the posterior under lock.

    Uses the canonical ``posterior_update_locked`` (Catalog #128 fcntl
    serialization) inside a thread-local guard (defence-in-depth). Returns
    a dict with at minimum ``accepted`` + ``refusal_reason``.
    """

    ensure_repo_imports(repo_root)
    try:
        from tac.continual_learning import ContestResult, posterior_update_locked
    except ImportError as exc:
        return {"accepted": False, "refusal_reason": f"posterior import failed: {exc}"}

    axis_tag = extracted.get("axis_tag") or AXIS_TAG_CPU_GENERIC
    axis = "cuda" if "CUDA" in axis_tag else "cpu"
    score = extracted.get("score")
    if not isinstance(score, (int, float)):
        return {"accepted": False, "refusal_reason": "no score_value to promote"}
    result = ContestResult(
        axis=axis,
        hardware_substrate=str(extracted.get("hardware_substrate") or "linux_x86_64_unknown_cuda"),
        architecture_class=str(extracted.get("architecture_class") or "?"),
        score_value=float(score),
        evidence_tag=axis_tag,
        archive_sha256=str(extracted.get("archive_sha256") or ""),
        archive_bytes=int(extracted.get("archive_bytes") or 0),
        notes=f"parallel_harvest_actuator: {extracted.get('auth_eval_path', '?')}",
    )
    with _POSTERIOR_THREAD_LOCK:
        try:
            update = posterior_update_locked(result)
        except Exception as exc:  # defensive
            return {"accepted": False, "refusal_reason": f"posterior_update_locked raised: {exc}"}
    return {
        "accepted": update.accepted,
        "refusal_reason": update.refusal_reason,
        "score_value": update.score_value,
        "evidence_tag": update.evidence_tag,
        "archive_sha256": update.archive_sha256,
        "posterior_n_anchors_after": update.posterior_n_anchors_after,
    }


def fan_out_harvest(
    *,
    lane_rows: list[dict[str, Any]],
    repo_root: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
    per_call_timeout_seconds: float = DEFAULT_PER_CALL_TIMEOUT_SECONDS,
    overall_timeout_seconds: float = DEFAULT_OVERALL_TIMEOUT_SECONDS,
    append_posterior: bool = True,
) -> list[dict[str, Any]]:
    """Run the harvest concurrently over the lane rows. Returns ordered results.

    The order of the returned list matches ``lane_rows``. Each row is the
    output of :func:`_harvest_one_call`. Per-call errors are captured in
    the row (``harvest_status`` and ``error`` fields) and do NOT kill the
    batch.
    """

    if not lane_rows:
        return []
    results: list[dict[str, Any] | None] = [None] * len(lane_rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        future_to_index: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
        for i, row in enumerate(lane_rows):
            future = pool.submit(
                _harvest_one_call,
                lane_row=row,
                repo_root=repo_root,
                per_call_timeout_seconds=per_call_timeout_seconds,
                append_posterior=append_posterior,
            )
            future_to_index[future] = i
        try:
            for future in concurrent.futures.as_completed(
                future_to_index, timeout=overall_timeout_seconds
            ):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # defensive: per-future error
                    row = lane_rows[index]
                    results[index] = {
                        **row,
                        "harvest_status": f"error_executor_{type(exc).__name__}",
                        "error": str(exc)[:1000],
                    }
        except concurrent.futures.TimeoutError:
            # Mark all unfinished as "timeout_executor"
            for future, index in future_to_index.items():
                if results[index] is None:
                    row = lane_rows[index]
                    results[index] = {
                        **row,
                        "harvest_status": "timeout_executor",
                        "error": f"overall executor timeout {overall_timeout_seconds}s",
                    }
    # Final fallback (should not trigger)
    return [r if r is not None else {"harvest_status": "no_result"} for r in results]


# -- CLI surface -------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _build_report(
    *,
    repo_root: Path,
    harvest_results: list[dict[str, Any]],
    vastai_inflight: list[dict[str, Any]],
    lightning_inflight: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    max_workers: int,
    per_call_timeout_seconds: float,
    overall_timeout_seconds: float,
    appended_posterior: bool,
) -> dict[str, Any]:
    """Build the consolidated report dict written to JSON/JSONL."""

    in_flight = sum(1 for r in harvest_results if r.get("harvest_status") == "not_ready")
    expired = sum(1 for r in harvest_results if r.get("harvest_status") == "expired")
    harvested = sum(1 for r in harvest_results if r.get("harvest_status") == "harvested")
    already = sum(1 for r in harvest_results if r.get("harvest_status") == "already_harvested")
    errors = sum(
        1 for r in harvest_results if str(r.get("harvest_status") or "").startswith("error_")
    )
    score_rows = [
        r for r in harvest_results if isinstance(r.get("score_claim"), dict)
    ]
    custody_accepted = sum(
        1 for r in score_rows if r["score_claim"].get("custody_accepted") is True
    )
    return {
        "schema": "parallel_harvest_actuator_report_v1_catalog_206_206",
        "generated_at_utc": finished_at,
        "started_at_utc": started_at,
        "repo_root": str(repo_root),
        "config": {
            "max_workers": max_workers,
            "per_call_timeout_seconds": per_call_timeout_seconds,
            "overall_timeout_seconds": overall_timeout_seconds,
            "appended_posterior": appended_posterior,
        },
        "counts": {
            "total_modal_lanes": len(harvest_results),
            "harvested_this_run": harvested,
            "already_harvested": already,
            "expired": expired,
            "not_ready_in_flight": in_flight,
            "errors": errors,
            "score_rows_extracted": len(score_rows),
            "custody_accepted_rows": custody_accepted,
            "vastai_inflight": len(vastai_inflight),
            "lightning_inflight": len(lightning_inflight),
        },
        "modal_harvest_rows": harvest_results,
        "vastai_inflight_rows": vastai_inflight,
        "lightning_inflight_rows": lightning_inflight,
        # CLAUDE.md non-negotiables this report participates in:
        "non_negotiable_references": [
            "race_mode_rigor_inversion_parallel_dispatch_first",
            "auth_eval_everywhere",
            "modal_spawn_harvest_or_lose",
            "subagent_coherence_by_default",
            "operator_gates_must_be_wired_and_used",
        ],
        # Per Catalog #113 artifact-kind: this is HISTORICAL_PROVENANCE
        # (append-only consolidated harvest record).
        "artifact_kind": "HISTORICAL_PROVENANCE",
        # Per CLAUDE.md "Forbidden score claims": this report is a fan-out
        # SUMMARY, NOT a score-claim authority. Individual rows carry their
        # axis-tagged claim; the report aggregates them.
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }


def _print_plan(
    *,
    lane_rows: list[dict[str, Any]],
    vastai: list[dict[str, Any]],
    lightning: list[dict[str, Any]],
) -> None:
    inflight_modal = sum(1 for r in lane_rows if not r["harvested"])
    print(
        f"=== parallel_harvest_actuator (read-only plan; pass --execute to fan out) ==="
    )
    print(
        f"Modal lanes:          {len(lane_rows)} total ({inflight_modal} pending, "
        f"{len(lane_rows) - inflight_modal} already-harvested)"
    )
    print(f"Vast.ai instances:    {len(vastai)} active")
    print(f"Lightning jobs:       {len(lightning)} tracked")
    print()
    for row in lane_rows[:50]:
        state = "harvested" if row["harvested"] else "pending"
        print(
            f"  {state:9s} {row['label'][:60]:60s} call={row['call_id'][:28]:28s} "
            f"gpu={row['gpu']:6s}"
        )
    if len(lane_rows) > 50:
        print(f"  ... {len(lane_rows) - 50} more rows omitted from plan view")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO,
        help="Repository root used to discover experiments/results/lane_*_modal.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Contact Modal, fan out concurrent harvests, append posterior. "
            "Without this flag the actuator only lists in-flight + already-harvested."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Max concurrent Modal FunctionCall.get() workers.",
    )
    parser.add_argument(
        "--per-call-timeout-seconds",
        type=float,
        default=DEFAULT_PER_CALL_TIMEOUT_SECONDS,
        help="Per-call Modal FunctionCall.get() timeout.",
    )
    parser.add_argument(
        "--overall-timeout-seconds",
        type=float,
        default=DEFAULT_OVERALL_TIMEOUT_SECONDS,
        help="Overall fan-out executor timeout.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help=(
            "Optional consolidated report JSON path. Defaults to "
            "reports/parallel_harvest_report_<utc>.json when --execute is set."
        ),
    )
    parser.add_argument(
        "--no-posterior-append",
        action="store_true",
        help=(
            "Skip the posterior_update_locked append step (still extracts + "
            "validates the score row, but does not promote)."
        ),
    )
    parser.add_argument(
        "--filter-label-substr",
        type=str,
        default=None,
        help="Only harvest rows whose label contains this substring.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout summary table (still writes report).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    started_at = _utc_now_iso()
    lane_rows = discover_inflight_modal_lanes(repo_root=repo_root)
    if args.filter_label_substr:
        sub = args.filter_label_substr.lower()
        lane_rows = [r for r in lane_rows if sub in r["label"].lower()]
    vastai = discover_inflight_vastai_instances(repo_root=repo_root)
    lightning = discover_inflight_lightning_jobs(repo_root=repo_root)
    if not args.execute:
        if not args.quiet:
            _print_plan(lane_rows=lane_rows, vastai=vastai, lightning=lightning)
        return 0
    harvest_results = fan_out_harvest(
        lane_rows=lane_rows,
        repo_root=repo_root,
        max_workers=args.max_workers,
        per_call_timeout_seconds=args.per_call_timeout_seconds,
        overall_timeout_seconds=args.overall_timeout_seconds,
        append_posterior=not args.no_posterior_append,
    )
    finished_at = _utc_now_iso()
    report = _build_report(
        repo_root=repo_root,
        harvest_results=harvest_results,
        vastai_inflight=vastai,
        lightning_inflight=lightning,
        started_at=started_at,
        finished_at=finished_at,
        max_workers=args.max_workers,
        per_call_timeout_seconds=args.per_call_timeout_seconds,
        overall_timeout_seconds=args.overall_timeout_seconds,
        appended_posterior=not args.no_posterior_append,
    )
    if args.report_output is None:
        ts = finished_at.replace(":", "").replace("-", "")
        output = repo_root / "reports" / f"parallel_harvest_report_{ts}.json"
    else:
        output = args.report_output
        if not output.is_absolute():
            output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not args.quiet:
        print(f"\n=== consolidated report written: {output} ===")
        counts = report["counts"]
        print(
            f"  modal_lanes_total={counts['total_modal_lanes']} "
            f"harvested_now={counts['harvested_this_run']} "
            f"already={counts['already_harvested']} "
            f"in_flight={counts['not_ready_in_flight']} "
            f"errors={counts['errors']} "
            f"score_rows={counts['score_rows_extracted']} "
            f"custody_ok={counts['custody_accepted_rows']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
