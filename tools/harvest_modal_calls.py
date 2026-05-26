#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest dispatched Modal call IDs before their result cache expires.

Default execution is read-only. Pass ``--execute`` to contact Modal, write
harvest artifacts, append terminal claims, and merge-refresh the summary file.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool


DEFAULT_REPO = repo_root_from_tool(__file__)
GENERATED_HARVEST_FILENAMES = frozenset({"_harvest_summary.json", "_stdout_tail.txt"})
CATHEDRAL_EVIDENCE_REL = Path("reports") / "cathedral_autopilot_evidence.jsonl"


class UnsafeModalArtifactPath(ValueError):
    """Raised when a Modal artifact key would escape the harvest directory."""


class ModalArtifactWriteError(RuntimeError):
    """Raised when one or more Modal artifacts could not be written locally."""

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__(f"failed to write {len(errors)} Modal artifact(s)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Contact Modal, harvest artifacts, append terminal claims, append cost "
            "anchors, and merge-refresh the harvest summary. Without this flag "
            "the tool only lists known lanes and exits without mutation."
        ),
    )
    parser.add_argument(
        "--from-ledger",
        action="store_true",
        help=(
            "Catalog #245 — consume the canonical Modal call_id ledger at "
            ".omx/state/modal_call_id_ledger.jsonl as the primary discovery "
            "surface (instead of glob'ing experiments/results/lane_*_modal/). "
            "Lists every dispatched call_id whose latest event is non-terminal."
        ),
    )
    parser.add_argument(
        "--call-id",
        action="append",
        default=[],
        help=(
            "Restrict list/ledger/harvest work to this Modal FunctionCall ID. "
            "May be passed more than once."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO,
        help="Repository root used to discover experiments/results/lane_*_modal.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help=(
            "Optional summary JSON path. Defaults to "
            "experiments/results/_modal_harvest_summary.json when --execute is set."
        ),
    )
    parser.add_argument(
        "--get-timeout-seconds",
        type=float,
        default=2.0,
        help="Timeout passed to Modal FunctionCall.get for each unharvested call.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []
    return [dict(item) for item in loaded if isinstance(item, dict)]


def _summary_entry_key(row: dict[str, Any]) -> tuple[str, str]:
    call_id = str(row.get("call_id") or "").strip()
    label = str(row.get("label") or "").strip()
    if call_id:
        return ("call_id", call_id)
    if label:
        return ("label", label)
    return ("object", json.dumps(row, sort_keys=True, default=str))


def merge_modal_harvest_summary_rows(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a non-lossy summary refresh.

    A narrowed ``--call-id`` harvest should update just those call entries in
    the global ``_modal_harvest_summary.json`` file, not replace unrelated
    historical harvest rows.
    """

    updates_by_key = {_summary_entry_key(row): row for row in updates}
    emitted: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for row in existing:
        key = _summary_entry_key(row)
        replacement = updates_by_key.get(key)
        if replacement is not None:
            merged.append(replacement)
            emitted.add(key)
        else:
            merged.append(row)
    for row in updates:
        key = _summary_entry_key(row)
        if key in emitted:
            continue
        merged.append(row)
        emitted.add(key)
    return merged


def _metadata_files(repo_root: Path) -> list[Path]:
    result_dirs = list((repo_root / "experiments" / "results").glob("lane_*_modal"))
    return sorted(
        directory / "modal_metadata.json"
        for directory in result_dirs
        if (directory / "modal_metadata.json").exists()
    )


def _normalise_call_ids(
    call_ids: list[str] | set[str] | frozenset[str] | None,
) -> frozenset[str]:
    return frozenset(
        str(call_id).strip()
        for call_id in (call_ids or [])
        if str(call_id).strip()
    )


def _metadata_files_for_call_ids(
    repo_root: Path,
    *,
    call_ids: list[str] | set[str] | frozenset[str] | None = None,
) -> list[Path]:
    """Return Modal metadata files, optionally narrowed to explicit call IDs."""

    filters = _normalise_call_ids(call_ids)
    metadata_files = _metadata_files(repo_root)
    if not filters:
        return metadata_files
    filtered: list[Path] = []
    for mfile in metadata_files:
        meta = _read_json(mfile) or {}
        if str(meta.get("call_id") or "") in filters:
            filtered.append(mfile)
    return filtered


def _safe_harvest_artifact_path(artifacts_dir: Path, relpath: str) -> Path:
    """Return a safe artifact target path under ``artifacts_dir``.

    Modal artifact keys are remote-controlled strings from the worker result.
    Treat them as untrusted: absolute paths, ``..`` traversal, empty parts, and
    resolved paths outside the harvest root are refused before any write.
    """

    raw = str(relpath).replace("\\", "/").strip()
    path = Path(raw)
    if (
        not raw
        or raw in {".", ".."}
        or path.is_absolute()
        or any(part in {"", ".."} for part in path.parts)
    ):
        raise UnsafeModalArtifactPath(f"unsafe Modal artifact path: {relpath!r}")
    root = artifacts_dir.resolve(strict=False)
    target = (artifacts_dir / path).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise UnsafeModalArtifactPath(
            f"Modal artifact path escapes harvest root: {relpath!r}"
        ) from exc
    return target


def _write_modal_artifacts(
    *,
    artifacts_dir: Path,
    artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    """Write Modal artifact payloads and refuse partial-success harvests."""

    planned: list[tuple[str, Path, bytes]] = []
    errors: list[dict[str, Any]] = []
    for relpath, data in artifacts.items():
        try:
            if not isinstance(relpath, str):
                raise TypeError(
                    f"Modal artifact keys must be str, got {type(relpath).__name__}"
                )
            target = _safe_harvest_artifact_path(artifacts_dir, relpath)
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise TypeError(
                    "Modal artifact values must be bytes-like, "
                    f"got {type(data).__name__}"
                )
            planned.append((relpath, target, bytes(data)))
        except Exception as exc:
            errors.append(
                {
                    "relative_path": str(relpath),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                }
            )
            print(f"    FAILED {relpath}: {exc}")
    if errors:
        raise ModalArtifactWriteError(errors)

    written_paths: list[Path] = []
    written: list[dict[str, Any]] = []
    for relpath, target, payload in planned:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        except Exception as exc:  # pragma: no cover - filesystem edge
            for path in written_paths:
                try:
                    path.unlink()
                except OSError:
                    pass
            print(f"    FAILED {relpath}: {exc}")
            raise ModalArtifactWriteError(
                [
                    {
                        "relative_path": relpath,
                        "target": str(target),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:1000],
                    }
                ]
            ) from exc
        written_paths.append(target)
        written.append(
            {
                "relative_path": relpath,
                "path": str(target),
                "bytes": len(payload),
            }
        )
    return written


def _terminal_harvest_summary(summary: dict[str, Any] | None) -> bool:
    """Return whether a generated harvest summary is terminal enough to skip get()."""

    if not isinstance(summary, dict):
        return False
    if "rc" in summary or "returncode" in summary:
        return True
    status = str(summary.get("status") or "")
    crash_kind = str(summary.get("crash_kind") or "")
    return (
        status in {"expired", "function_timeout"}
        or status.startswith("error_")
        or crash_kind in {
            "RESULT_CACHE_EXPIRED",
            "FUNCTION_TIMEOUT",
        }
        or crash_kind.startswith("ERROR_")
    )


def _has_harvest_payload_files(artifacts_dir: Path) -> bool:
    """Return True when harvested_artifacts contains real remote payload files."""

    if not artifacts_dir.exists():
        return False
    return any(path.name not in GENERATED_HARVEST_FILENAMES for path in artifacts_dir.iterdir())


def _already_harvested(out_dir: Path, artifacts_dir: Path) -> bool:
    """Return whether local state is terminal enough to avoid re-polling Modal."""

    terminal_marker = out_dir / "modal_training_terminal_claim.json"
    if terminal_marker.is_file():
        marker = _read_json(terminal_marker)
        if isinstance(marker, dict) and marker.get("appended") is True:
            return True
        # A failed/incomplete claim marker is weaker than harvested local result
        # evidence. Keep checking so a provider-GC refresh cannot erase rc/artifact
        # custody that was already materialized before the claim helper existed.
    if (out_dir / "harvest_summary.json").is_file():
        return True
    if _has_harvest_payload_files(artifacts_dir):
        return True
    summary = _read_json(artifacts_dir / "_harvest_summary.json")
    return _terminal_harvest_summary(summary)


def _covered_terminal_claims(evidence_path: Path) -> set[tuple[str, str, str]]:
    covered: set[tuple[str, str, str]] = set()
    if not evidence_path.is_file():
        return covered
    for line in evidence_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        claims = payload.get("covered_terminal_claims")
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            lane_id = str(claim.get("lane_id") or "")
            job_id = str(claim.get("instance_job_id") or "")
            status = str(claim.get("status") or "")
            if lane_id and job_id and status:
                covered.add((lane_id, job_id, status))
    return covered


def _append_terminal_claim_evidence(
    *,
    repo_root: Path,
    out_dir: Path,
    terminal_claim: dict[str, Any] | None,
) -> dict[str, Any]:
    """Append no-score cathedral evidence for a recovered terminal claim."""

    if not isinstance(terminal_claim, dict):
        return {"appended": False, "reason": "missing_terminal_claim"}
    lane_id = str(terminal_claim.get("lane_id") or "")
    job_id = str(terminal_claim.get("instance_job_id") or "")
    status = str(terminal_claim.get("status") or "")
    if not lane_id or not job_id or not status:
        return {"appended": False, "reason": "incomplete_terminal_claim"}
    if (
        terminal_claim.get("appended") is not True
        and terminal_claim.get("already_appended") is not True
    ):
        return {
            "appended": False,
            "reason": "terminal_claim_not_appended",
            "lane_id": lane_id,
            "instance_job_id": job_id,
            "status": status,
        }

    evidence_path = repo_root / CATHEDRAL_EVIDENCE_REL
    claim_key = (lane_id, job_id, status)
    if claim_key in _covered_terminal_claims(evidence_path):
        return {
            "appended": False,
            "already_covered": True,
            "lane_id": lane_id,
            "instance_job_id": job_id,
            "status": status,
        }

    recovered_auth_eval = terminal_claim.get("recovered_auth_eval")
    row = {
        "schema": "cathedral_autopilot_terminal_claim_evidence_v1",
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "evidence_grade": "[infrastructure terminal dispatch coverage]",
        "evidence_marker": "[infrastructure terminal dispatch coverage]",
        "evidence_semantics": "terminal_modal_training_harvest_no_signal_loss",
        "covered_terminal_claim_count": 1,
        "covered_terminal_claims": [
            {
                "lane_id": lane_id,
                "instance_job_id": job_id,
                "status": status,
            }
        ],
        "dispatch_attempted": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "recovered_auth_eval": recovered_auth_eval if isinstance(recovered_auth_eval, dict) else None,
        "proxy_row": True,
        "family_falsified": False,
        "method_family_retired": False,
        "contest_dispatch_verdict": "no_signal_loss_terminal_claim_coverage",
        "measured_config_status": (
            "terminal_dispatch_claim_preserved_with_inline_auth_eval"
            if isinstance(recovered_auth_eval, dict)
            else "terminal_dispatch_claim_preserved_no_score"
        ),
        "dispatch_blockers": [
            (
                "terminal_training_harvest_preserves_inline_auth_eval_but_is_not_rank_authority"
                if isinstance(recovered_auth_eval, dict)
                else "terminal_training_harvest_has_no_score_authority"
            ),
            "exact_result_review_required_before_promotion_or_rank_kill",
        ],
        "reactivation_criteria": [
            "classify any archive/auth-eval artifacts through an exact result-review packet",
            "do not infer model success or failure from terminal training infrastructure rows",
        ],
        "source": str(out_dir / "modal_training_terminal_claim.json"),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "appended": True,
        "evidence_path": str(evidence_path),
        "lane_id": lane_id,
        "instance_job_id": job_id,
        "status": status,
    }


def _coerce_int(value: Any) -> int | None:
    if type(value) is int:
        return value
    return None


def _coerce_float(value: Any) -> float | None:
    if type(value) in {int, float}:
        return float(value)
    return None


def _call_ledger_status_from_terminal(
    *,
    harvested: dict[str, Any] | None,
    terminal_claim: dict[str, Any] | None,
) -> str | None:
    try:
        from tac.deploy.modal.call_id_ledger import (
            STATUS_FAILED,
            STATUS_HARVESTED,
            STATUS_STALE,
        )
    except Exception:  # pragma: no cover - import failure handled by caller
        return None

    claim_status = str((terminal_claim or {}).get("status") or "")
    harvest_status = str((harvested or {}).get("status") or "")
    crash_kind = str((harvested or {}).get("crash_kind") or "")
    if claim_status == "failed_modal_training_result_cache_expired":
        return STATUS_STALE
    if harvest_status == "expired" or crash_kind == "RESULT_CACHE_EXPIRED":
        return STATUS_STALE
    if claim_status.startswith("failed"):
        return STATUS_FAILED
    if claim_status.startswith("completed"):
        return STATUS_HARVESTED

    rc = None
    if harvested is not None:
        rc = _coerce_int(harvested.get("rc"))
        if rc is None:
            rc = _coerce_int(harvested.get("returncode"))
    if rc == 0:
        return STATUS_HARVESTED
    if rc is not None:
        return STATUS_FAILED
    if harvest_status.startswith("error_") or harvest_status == "function_timeout":
        return STATUS_FAILED
    return None


def _append_call_id_ledger_terminal_event(
    *,
    repo_root: Path,
    metadata: dict[str, Any],
    harvested: dict[str, Any] | None,
    terminal_claim: dict[str, Any] | None,
    agent: str,
) -> dict[str, Any]:
    """Mirror a terminal harvest into the canonical Modal call_id ledger."""

    from tac.deploy.modal.harvest_outcomes import append_terminal_call_id_ledger_event

    return append_terminal_call_id_ledger_event(
        repo_root=repo_root,
        metadata=metadata,
        harvested=harvested,
        terminal_claim=terminal_claim,
        agent=agent,
    )


def list_modal_lanes(
    *,
    repo_root: Path,
    call_ids: list[str] | set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Return read-only Modal lane metadata for operator status probes."""

    rows: list[dict[str, Any]] = []
    for mfile in _metadata_files_for_call_ids(repo_root, call_ids=call_ids):
        meta = _read_json(mfile) or {}
        out_dir = mfile.parent
        artifacts_dir = out_dir / "harvested_artifacts"
        harvested = _already_harvested(out_dir, artifacts_dir)
        rows.append(
            {
                "label": meta.get("label", "?"),
                "call_id": meta.get("call_id", "?"),
                "dispatched_at": meta.get("dispatched_at", "?"),
                "out_dir": str(out_dir),
                "harvested": harvested,
            }
        )
    return rows


def _print_plan(rows: list[dict[str, Any]]) -> None:
    print(f"Found {len(rows)} dispatched lanes with modal_metadata.json")
    print("PLAN ONLY: pass --execute to contact Modal and mutate harvest state.")
    print()
    for row in rows:
        state = "harvested" if row["harvested"] else "pending"
        call_id = str(row["call_id"])
        print(
            f"  {state:9s} {str(row['label'])[:60]:60s} "
            f"call_id={call_id[:30]}"
        )


def harvest_modal_calls(
    *,
    repo_root: Path,
    summary_output: Path,
    get_timeout_seconds: float,
    call_ids: list[str] | set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute the Modal harvest flow and return the summary rows."""

    import modal

    ensure_repo_imports(repo_root)
    from tac.deploy.modal.harvest_summary import (
        enrich_modal_training_result_summary,
        modal_training_summary_entry,
        normalise_modal_training_result_summary,
        partial_modal_training_result_summary,
    )
    from tac.deploy.modal.training_claims import (
        append_modal_training_terminal_claim,
        recovered_inline_contest_cuda_auth_eval,
    )
    from tac.deploy.modal.training_cost import append_modal_training_cost_anchor

    filters = _normalise_call_ids(call_ids)
    metadata_files = _metadata_files_for_call_ids(repo_root, call_ids=filters)
    print(f"Found {len(metadata_files)} dispatched lanes with modal_metadata.json")
    if filters:
        print(f"Restricted to {len(filters)} explicit Modal call_id filter(s)")
    print()

    summary: list[dict[str, Any]] = []
    for mfile in metadata_files:
        meta = _read_json(mfile) or {}
        call_id = str(meta.get("call_id", "?"))
        label = str(meta.get("label", "?"))
        dispatched = meta.get("dispatched_at", "?")
        out_dir = mfile.parent

        artifacts_dir = out_dir / "harvested_artifacts"
        if _already_harvested(out_dir, artifacts_dir):
            print(f"[SKIP-already-harvested] {label:30s} call_id={call_id[:30]}")
            artifacts_dir.mkdir(exist_ok=True)
            harvest_summary = artifacts_dir / "_harvest_summary.json"
            harvested: dict[str, Any] | None = None
            if harvest_summary.is_file():
                harvested = _read_json(harvest_summary)
                if harvested is None:
                    harvested = {"crash_kind": "INVALID_HARVEST_SUMMARY"}
            else:
                root_harvest_summary = out_dir / "harvest_summary.json"
                if root_harvest_summary.is_file():
                    loaded = _read_json(root_harvest_summary)
                    if loaded is not None:
                        harvested = normalise_modal_training_result_summary(
                            loaded,
                            artifacts_dir=artifacts_dir,
                            source_summary=root_harvest_summary,
                        )
                        (artifacts_dir / "_harvest_summary.json").write_text(
                            json.dumps(harvested, indent=2, default=str),
                            encoding="utf-8",
                        )
                    else:
                        harvested = {"crash_kind": "INVALID_ROOT_HARVEST_SUMMARY"}
                else:
                    harvested = partial_modal_training_result_summary(
                        artifacts_dir=artifacts_dir
                    )
                    (artifacts_dir / "_harvest_summary.json").write_text(
                        json.dumps(harvested, indent=2, default=str),
                        encoding="utf-8",
                    )
            cost_marker = out_dir / "cost_band_anchor_appended.json"
            cost_anchor = _read_json(cost_marker) if cost_marker.is_file() else None
            if cost_anchor is None and cost_marker.is_file():
                cost_anchor = {
                    "appended": False,
                    "reason": "invalid_existing_cost_anchor_marker",
                }
            elif cost_anchor is None and harvested is not None:
                try:
                    cost_anchor = append_modal_training_cost_anchor(
                        out_dir=out_dir,
                        metadata=meta,
                        result=dict(harvested),
                    )
                except Exception as exc:  # pragma: no cover - defensive provider IO
                    cost_anchor = {
                        "appended": False,
                        "reason": (
                            "already_harvested_append_failed:"
                            f"{type(exc).__name__}:{exc}"
                        ),
                    }
            terminal_marker = out_dir / "modal_training_terminal_claim.json"
            terminal_claim = (
                _read_json(terminal_marker) if terminal_marker.is_file() else None
            )
            if terminal_claim is None and terminal_marker.is_file():
                terminal_claim = {
                    "appended": False,
                    "reason": "invalid_existing_terminal_claim_marker",
                }
            elif terminal_claim is None and harvested is not None:
                try:
                    terminal_claim = append_modal_training_terminal_claim(
                        repo_root=repo_root,
                        out_dir=out_dir,
                        metadata=meta,
                        result=dict(harvested),
                        agent="codex:harvest_modal_calls",
                    )
                except Exception as exc:  # pragma: no cover - defensive provider IO
                    terminal_claim = {
                        "appended": False,
                        "reason": (
                            "already_harvested_terminal_claim_failed:"
                            f"{type(exc).__name__}:{exc}"
                        ),
                    }
            elif isinstance(terminal_claim, dict) and not isinstance(
                terminal_claim.get("recovered_auth_eval"), dict
            ):
                recovered_auth_eval = recovered_inline_contest_cuda_auth_eval(out_dir)
                if recovered_auth_eval is not None:
                    terminal_claim = {
                        **terminal_claim,
                        "recovered_auth_eval": recovered_auth_eval,
                    }
                    terminal_marker.write_text(
                        json.dumps(terminal_claim, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
            if isinstance(harvested, dict):
                enriched_harvested = enrich_modal_training_result_summary(
                    harvested,
                    artifacts_dir=artifacts_dir,
                    out_dir=out_dir,
                    cost_anchor=cost_anchor if isinstance(cost_anchor, dict) else None,
                    terminal_claim=terminal_claim
                    if isinstance(terminal_claim, dict)
                    else None,
                )
                if enriched_harvested != harvested:
                    harvested = enriched_harvested
                    (artifacts_dir / "_harvest_summary.json").write_text(
                        json.dumps(harvested, indent=2, default=str),
                        encoding="utf-8",
                    )
            terminal_evidence = _append_terminal_claim_evidence(
                repo_root=repo_root,
                out_dir=out_dir,
                terminal_claim=terminal_claim,
            )
            call_id_ledger = _append_call_id_ledger_terminal_event(
                repo_root=repo_root,
                metadata=meta,
                harvested=harvested,
                terminal_claim=terminal_claim,
                agent="codex:harvest_modal_calls",
            )
            entry = modal_training_summary_entry(
                label=label,
                status="already_harvested",
                call_id=call_id,
                harvested=harvested,
                cost_anchor=cost_anchor,
                terminal_claim=terminal_claim,
                terminal_evidence=terminal_evidence,
            )
            entry["call_id_ledger"] = call_id_ledger
            summary.append(entry)
            continue

        print(f"\n=== {label} ({call_id[:30]}, dispatched {dispatched}) ===")
        try:
            fc = modal.functions.FunctionCall.from_id(call_id)
            result = fc.get(timeout=get_timeout_seconds)
            rc = result.get("returncode", "?")
            elapsed_raw = result.get("elapsed_seconds")
            elapsed = elapsed_raw if isinstance(elapsed_raw, (int, float)) else 0
            timed_out = result.get("timed_out", False)
            n_artifacts = len(result.get("artifacts", {}))
            tail = result.get("stdout_tail", "") or ""
            tail = tail[-500:]

            crash_kind = "OK"
            if "OutOfMemoryError" in tail or "CUDA out of memory" in tail:
                crash_kind = "CUDA_OOM"
            elif "FATAL" in tail or "Traceback" in tail:
                crash_kind = "ERROR"
            elif timed_out:
                crash_kind = "TIMEOUT"
            elif rc != 0:
                crash_kind = f"RC_{rc}"

            print(
                f"  rc={rc} elapsed={elapsed:.0f}s timed_out={timed_out} "
                f"artifacts={n_artifacts} crash={crash_kind}"
            )

            artifacts_dir.mkdir(exist_ok=True)
            artifact_files = _write_modal_artifacts(
                artifacts_dir=artifacts_dir,
                artifacts=dict(result.get("artifacts", {})),
            )

            (artifacts_dir / "_stdout_tail.txt").write_text(
                result.get("stdout_tail", "") or "",
                encoding="utf-8",
            )
            # OP-8 fix (codex chunk 5/10, 2026-05-15): n_artifacts==0
            # mislabeling. When a Modal dispatch returns rc=0 + not timed_out
            # but produces ZERO artifacts, the canonical
            # ``append_platform_training_anchor`` (cost_band_calibration.py)
            # would derive ``outcome=successful_dispatch`` from the rc=0
            # signal alone — corrupting the cost-band posterior with anchors
            # that look successful but produced no usable evidence (no
            # archive, no auth-eval JSON, no checkpoint). The downstream
            # cathedral-autopilot ranker then trusts these zero-evidence
            # anchors as proof-of-feasibility and re-prioritizes the same
            # config that just failed.
            #
            # The fix is to override the outcome to ``harvested_partial`` for
            # the zero-artifact-rc-0 case so:
            #   1. ``predict()`` excludes the anchor by default
            #      (``successful_dispatch`` is the only default-included
            #      outcome per VALID_OUTCOMES).
            #   2. The cost-band posterior preserves the row for forensic
            #      audit (HARVESTED_PARTIAL is in VALID_OUTCOMES).
            #   3. Catalog #245 Modal call_id ledger keeps its own
            #      EVENT_HARVESTED status independently (no schema change).
            #   4. Catalog #185 live-count drift detection is no longer
            #      poisoned by false-positive successful_dispatch counts.
            #
            # Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog
            # #221 (`check_auth_eval_result_artifacts_fail_closed_for_score_claims`)
            # the rc=0+zero-artifacts row is by definition non-promotable:
            # there is nothing to score.
            cost_meta_for_anchor = meta
            if n_artifacts == 0 and rc == 0 and not timed_out:
                base_cost_meta = dict(meta.get("cost_band_anchor") or {})
                base_cost_meta["outcome"] = "harvested_partial"
                base_notes = base_cost_meta.get("notes", "")
                op8_note = (
                    "OP-8 (2026-05-15): n_artifacts==0 with rc=0 → "
                    "outcome overridden to harvested_partial; row preserved "
                    "for forensic audit but excluded from predict() default."
                )
                base_cost_meta["notes"] = (
                    f"{base_notes}; {op8_note}" if base_notes else op8_note
                )
                cost_meta_for_anchor = {**meta, "cost_band_anchor": base_cost_meta}
                outcome_classification = "partial_harvest_zero_artifacts"
            elif rc != 0:
                outcome_classification = f"failed_rc_{rc}"
            elif timed_out:
                outcome_classification = "timed_out"
            else:
                outcome_classification = "successful_dispatch"
            cost_anchor = append_modal_training_cost_anchor(
                out_dir=out_dir,
                metadata=cost_meta_for_anchor,
                result=result,
            )
            terminal_claim = append_modal_training_terminal_claim(
                repo_root=repo_root,
                out_dir=out_dir,
                metadata=meta,
                result=result,
                agent="codex:harvest_modal_calls",
            )
            terminal_evidence = _append_terminal_claim_evidence(
                repo_root=repo_root,
                out_dir=out_dir,
                terminal_claim=terminal_claim,
            )
            harvest_summary = {
                "rc": rc,
                "elapsed_seconds": elapsed,
                "timed_out": timed_out,
                "n_artifacts": n_artifacts,
                "artifact_files": artifact_files,
                "crash_kind": crash_kind,
                "outcome_classification": outcome_classification,
                "cost_band_anchor": cost_anchor,
                "terminal_claim": terminal_claim,
                "terminal_evidence": terminal_evidence,
            }
            call_id_ledger = _append_call_id_ledger_terminal_event(
                repo_root=repo_root,
                metadata=meta,
                harvested=harvest_summary,
                terminal_claim=terminal_claim,
                agent="codex:harvest_modal_calls",
            )
            harvest_summary["call_id_ledger"] = call_id_ledger
            (artifacts_dir / "_harvest_summary.json").write_text(
                json.dumps(harvest_summary, indent=2, default=str),
                encoding="utf-8",
            )

            summary.append(
                {
                    "label": label,
                    "call_id": call_id,
                    "rc": rc,
                    "elapsed_seconds": elapsed,
                    "timed_out": timed_out,
                    "n_artifacts": n_artifacts,
                    "artifact_files": artifact_files,
                    "crash_kind": crash_kind,
                    "outcome_classification": outcome_classification,
                    "cost_band_anchor": cost_anchor,
                    "terminal_claim": terminal_claim,
                    "terminal_evidence": terminal_evidence,
                    "call_id_ledger": call_id_ledger,
                }
            )

        except modal.exception.OutputExpiredError:
            print("  EXPIRED (>24h old, GC'd)")
            terminal_claim = append_modal_training_terminal_claim(
                repo_root=repo_root,
                out_dir=out_dir,
                metadata=meta,
                result=None,
                status="failed_modal_training_result_cache_expired",
                agent="codex:harvest_modal_calls",
            )
            terminal_evidence = _append_terminal_claim_evidence(
                repo_root=repo_root,
                out_dir=out_dir,
                terminal_claim=terminal_claim,
            )
            artifacts_dir.mkdir(exist_ok=True)
            expired_summary = {
                "status": "expired",
                "crash_kind": "RESULT_CACHE_EXPIRED",
                "n_artifacts": 0,
                "terminal_claim": terminal_claim,
                "terminal_evidence": terminal_evidence,
            }
            call_id_ledger = _append_call_id_ledger_terminal_event(
                repo_root=repo_root,
                metadata=meta,
                harvested=expired_summary,
                terminal_claim=terminal_claim,
                agent="codex:harvest_modal_calls",
            )
            expired_summary["call_id_ledger"] = call_id_ledger
            (artifacts_dir / "_harvest_summary.json").write_text(
                json.dumps(expired_summary, indent=2, default=str),
                encoding="utf-8",
            )
            entry = modal_training_summary_entry(
                label=label,
                status="expired",
                call_id=call_id,
                harvested=expired_summary,
                terminal_claim=terminal_claim,
                terminal_evidence=terminal_evidence,
            )
            entry["call_id_ledger"] = call_id_ledger
            summary.append(entry)
        except modal.exception.FunctionTimeoutError as exc:
            print(f"  FUNCTION TIMEOUT: {exc}")
            terminal_claim = append_modal_training_terminal_claim(
                repo_root=repo_root,
                out_dir=out_dir,
                metadata=meta,
                result=None,
                status="failed_modal_training_function_timeout",
                agent="codex:harvest_modal_calls",
            )
            terminal_evidence = _append_terminal_claim_evidence(
                repo_root=repo_root,
                out_dir=out_dir,
                terminal_claim=terminal_claim,
            )
            artifacts_dir.mkdir(exist_ok=True)
            timeout_summary = {
                "status": "function_timeout",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "crash_kind": "FUNCTION_TIMEOUT",
                "n_artifacts": 0,
                "terminal_claim": terminal_claim,
                "terminal_evidence": terminal_evidence,
            }
            call_id_ledger = _append_call_id_ledger_terminal_event(
                repo_root=repo_root,
                metadata=meta,
                harvested=timeout_summary,
                terminal_claim=terminal_claim,
                agent="codex:harvest_modal_calls",
            )
            timeout_summary["call_id_ledger"] = call_id_ledger
            (artifacts_dir / "_harvest_summary.json").write_text(
                json.dumps(timeout_summary, indent=2, default=str),
                encoding="utf-8",
            )
            entry = modal_training_summary_entry(
                label=label,
                status="function_timeout",
                call_id=call_id,
                harvested=timeout_summary,
                terminal_claim=terminal_claim,
                terminal_evidence=terminal_evidence,
            )
            entry["call_id_ledger"] = call_id_ledger
            summary.append(entry)
        except UnsafeModalArtifactPath as exc:
            print(f"  UNSAFE ARTIFACT PATH: {exc}")
            terminal_claim = append_modal_training_terminal_claim(
                repo_root=repo_root,
                out_dir=out_dir,
                metadata=meta,
                result=None,
                status="failed_modal_training_unsafe_artifact_path",
                agent="codex:harvest_modal_calls",
            )
            terminal_evidence = _append_terminal_claim_evidence(
                repo_root=repo_root,
                out_dir=out_dir,
                terminal_claim=terminal_claim,
            )
            artifacts_dir.mkdir(exist_ok=True)
            unsafe_summary = {
                "status": "unsafe_artifact_path",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "crash_kind": "UNSAFE_ARTIFACT_PATH",
                "n_artifacts": 0,
                "terminal_claim": terminal_claim,
                "terminal_evidence": terminal_evidence,
            }
            call_id_ledger = _append_call_id_ledger_terminal_event(
                repo_root=repo_root,
                metadata=meta,
                harvested=unsafe_summary,
                terminal_claim=terminal_claim,
                agent="codex:harvest_modal_calls",
            )
            unsafe_summary["call_id_ledger"] = call_id_ledger
            (artifacts_dir / "_harvest_summary.json").write_text(
                json.dumps(unsafe_summary, indent=2, default=str),
                encoding="utf-8",
            )
            entry = modal_training_summary_entry(
                label=label,
                status="unsafe_artifact_path",
                call_id=call_id,
                harvested=unsafe_summary,
                terminal_claim=terminal_claim,
                terminal_evidence=terminal_evidence,
            )
            entry["call_id_ledger"] = call_id_ledger
            summary.append(entry)
        except ModalArtifactWriteError as exc:
            print(f"  ARTIFACT WRITE FAILURE: {exc}")
            terminal_claim = append_modal_training_terminal_claim(
                repo_root=repo_root,
                out_dir=out_dir,
                metadata=meta,
                result=None,
                status="failed_modal_training_invalid_artifacts",
                agent="codex:harvest_modal_calls",
            )
            terminal_evidence = _append_terminal_claim_evidence(
                repo_root=repo_root,
                out_dir=out_dir,
                terminal_claim=terminal_claim,
            )
            artifacts_dir.mkdir(exist_ok=True)
            write_error_summary = {
                "status": "invalid_artifacts",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "artifact_write_errors": exc.errors,
                "crash_kind": "INVALID_MODAL_ARTIFACTS",
                "n_artifacts": n_artifacts,
                "terminal_claim": terminal_claim,
                "terminal_evidence": terminal_evidence,
            }
            call_id_ledger = _append_call_id_ledger_terminal_event(
                repo_root=repo_root,
                metadata=meta,
                harvested=write_error_summary,
                terminal_claim=terminal_claim,
                agent="codex:harvest_modal_calls",
            )
            write_error_summary["call_id_ledger"] = call_id_ledger
            (artifacts_dir / "_harvest_summary.json").write_text(
                json.dumps(write_error_summary, indent=2, default=str),
                encoding="utf-8",
            )
            entry = modal_training_summary_entry(
                label=label,
                status="invalid_artifacts",
                call_id=call_id,
                harvested=write_error_summary,
                terminal_claim=terminal_claim,
                terminal_evidence=terminal_evidence,
            )
            entry["call_id_ledger"] = call_id_ledger
            summary.append(entry)
        except TimeoutError:
            print("  NOT READY (still queued or running)")
            summary.append({"label": label, "status": "not_ready", "call_id": call_id})
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {str(exc)[:200]}")
            status = f"failed_modal_training_recovery_error_{type(exc).__name__.lower()}"
            terminal_claim = append_modal_training_terminal_claim(
                repo_root=repo_root,
                out_dir=out_dir,
                metadata=meta,
                result=None,
                status=status,
                agent="codex:harvest_modal_calls",
            )
            terminal_evidence = _append_terminal_claim_evidence(
                repo_root=repo_root,
                out_dir=out_dir,
                terminal_claim=terminal_claim,
            )
            artifacts_dir.mkdir(exist_ok=True)
            error_summary = {
                "status": f"error_{type(exc).__name__}",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "crash_kind": f"ERROR_{type(exc).__name__}",
                "n_artifacts": 0,
                "terminal_claim": terminal_claim,
                "terminal_evidence": terminal_evidence,
            }
            call_id_ledger = _append_call_id_ledger_terminal_event(
                repo_root=repo_root,
                metadata=meta,
                harvested=error_summary,
                terminal_claim=terminal_claim,
                agent="codex:harvest_modal_calls",
            )
            error_summary["call_id_ledger"] = call_id_ledger
            (artifacts_dir / "_harvest_summary.json").write_text(
                json.dumps(error_summary, indent=2, default=str),
                encoding="utf-8",
            )
            entry = modal_training_summary_entry(
                label=label,
                status=f"error_{type(exc).__name__}",
                call_id=call_id,
                harvested=error_summary,
                terminal_claim=terminal_claim,
                terminal_evidence=terminal_evidence,
            )
            entry["call_id_ledger"] = call_id_ledger
            summary.append(entry)

    print("\n\n=== SUMMARY ===")
    for row in summary:
        print(
            f"  {row.get('label', '?')!s:30s}  "
            f"{row.get('crash_kind', row.get('status', '?'))!s:20s}  "
            f"rc={row.get('rc', '?')}  elapsed={row.get('elapsed_seconds', '?')}"
        )

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    existing_summary = _read_json_list(summary_output)
    merged_summary = merge_modal_harvest_summary_rows(existing_summary, summary)
    summary_output.write_text(
        json.dumps(merged_summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSummary saved: {summary_output}")
    return merged_summary


def _print_from_ledger_view(
    repo_root: Path,
    *,
    call_ids: list[str] | set[str] | frozenset[str] | None = None,
) -> None:
    """Catalog #245 — print the canonical ledger view of unharvested call_ids."""
    try:
        from tac.deploy.modal.call_id_ledger import (
            MODAL_CALL_ID_LEDGER_PATH,
            latest_status_by_call_id,
            query_unharvested,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[from-ledger] FAILED to import call_id_ledger: {exc}")
        return
    ledger_path = MODAL_CALL_ID_LEDGER_PATH
    print(f"[from-ledger] canonical ledger: {ledger_path}")
    statuses = latest_status_by_call_id(path=ledger_path)
    print(f"[from-ledger] total call_ids in ledger: {len(statuses)}")
    unharvested = query_unharvested(path=ledger_path)
    filters = _normalise_call_ids(call_ids)
    if filters:
        unharvested = [
            row
            for row in unharvested
            if str(row.get("call_id") or "") in filters
        ]
        print(f"[from-ledger] restricted to call_id filters: {len(filters)}")
    print(f"[from-ledger] unharvested call_ids: {len(unharvested)}")
    for row in unharvested[:20]:
        print(
            f"  call_id={row.get('call_id', '?')} "
            f"lane={row.get('lane_id', '?')} "
            f"gpu={row.get('gpu', '?')} "
            f"dispatched_at={row.get('dispatched_at_utc', '?')}"
        )
    if len(unharvested) > 20:
        print(f"  ... and {len(unharvested) - 20} more")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    call_ids = _normalise_call_ids(args.call_id)
    if args.from_ledger:
        # Catalog #245 canonical ledger consumer surface. Without --execute it
        # remains a read-only status view; with --execute we print the canonical
        # index first, then run the normal harvest path and mirror terminal
        # outcomes back into the ledger.
        _print_from_ledger_view(repo_root, call_ids=call_ids)
        if not args.execute:
            return 0
    if not args.execute:
        _print_plan(list_modal_lanes(repo_root=repo_root, call_ids=call_ids))
        return 0
    summary_output = args.summary_output
    if summary_output is None:
        summary_output = repo_root / "experiments" / "results" / "_modal_harvest_summary.json"
    elif not summary_output.is_absolute():
        summary_output = repo_root / summary_output
    harvest_modal_calls(
        repo_root=repo_root,
        summary_output=summary_output,
        get_timeout_seconds=args.get_timeout_seconds,
        call_ids=call_ids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
