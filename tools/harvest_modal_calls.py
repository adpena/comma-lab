#!/usr/bin/env python3
"""Harvest dispatched Modal call IDs before their result cache expires.

Default execution is read-only. Pass ``--execute`` to contact Modal, write
harvest artifacts, append terminal claims, and refresh the summary file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool


DEFAULT_REPO = repo_root_from_tool(__file__)
GENERATED_HARVEST_FILENAMES = frozenset({"_harvest_summary.json", "_stdout_tail.txt"})


class UnsafeModalArtifactPath(ValueError):
    """Raised when a Modal artifact key would escape the harvest directory."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Contact Modal, harvest artifacts, append terminal claims, append cost "
            "anchors, and rewrite the harvest summary. Without this flag the tool "
            "only lists known lanes and exits without mutation."
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


def _metadata_files(repo_root: Path) -> list[Path]:
    result_dirs = list((repo_root / "experiments" / "results").glob("lane_*_modal"))
    return sorted(
        directory / "modal_metadata.json"
        for directory in result_dirs
        if (directory / "modal_metadata.json").exists()
    )


def _safe_harvest_artifact_path(artifacts_dir: Path, relpath: str) -> Path:
    """Return a safe artifact target path under ``artifacts_dir``.

    Modal artifact keys are remote-controlled strings from the worker result.
    Treat them as untrusted: absolute paths, ``..`` traversal, empty parts, and
    resolved paths outside the harvest root are refused before any write.
    """

    raw = str(relpath).replace("\\", "/").strip()
    path = Path(raw)
    if not raw or path.is_absolute() or any(part in {"", ".."} for part in path.parts):
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
        or crash_kind in {"RESULT_CACHE_EXPIRED", "FUNCTION_TIMEOUT"}
        or crash_kind.startswith("ERROR_")
    )


def _has_harvest_payload_files(artifacts_dir: Path) -> bool:
    """Return True when harvested_artifacts contains real remote payload files."""

    if not artifacts_dir.exists():
        return False
    return any(path.name not in GENERATED_HARVEST_FILENAMES for path in artifacts_dir.iterdir())


def _already_harvested(out_dir: Path, artifacts_dir: Path) -> bool:
    """Return whether local state is terminal enough to avoid re-polling Modal."""

    if (out_dir / "modal_training_terminal_claim.json").is_file():
        return True
    if (out_dir / "harvest_summary.json").is_file():
        return True
    if _has_harvest_payload_files(artifacts_dir):
        return True
    summary = _read_json(artifacts_dir / "_harvest_summary.json")
    return _terminal_harvest_summary(summary)


def list_modal_lanes(*, repo_root: Path) -> list[dict[str, Any]]:
    """Return read-only Modal lane metadata for operator status probes."""

    rows: list[dict[str, Any]] = []
    for mfile in _metadata_files(repo_root):
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
) -> list[dict[str, Any]]:
    """Execute the Modal harvest flow and return the summary rows."""

    import modal

    ensure_repo_imports(repo_root)
    from tac.deploy.modal.harvest_summary import (
        modal_training_summary_entry,
        normalise_modal_training_result_summary,
        partial_modal_training_result_summary,
    )
    from tac.deploy.modal.training_claims import append_modal_training_terminal_claim
    from tac.deploy.modal.training_cost import append_modal_training_cost_anchor

    metadata_files = _metadata_files(repo_root)
    print(f"Found {len(metadata_files)} dispatched lanes with modal_metadata.json")
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
            summary.append(
                modal_training_summary_entry(
                    label=label,
                    status="already_harvested",
                    call_id=call_id,
                    harvested=harvested,
                    cost_anchor=cost_anchor,
                    terminal_claim=terminal_claim,
                )
            )
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
            for relpath, data in result.get("artifacts", {}).items():
                target = _safe_harvest_artifact_path(artifacts_dir, str(relpath))
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    target.write_bytes(data)
                except Exception as exc:  # pragma: no cover - filesystem edge
                    print(f"    SKIP {relpath}: {exc}")

            (artifacts_dir / "_stdout_tail.txt").write_text(
                result.get("stdout_tail", "") or "",
                encoding="utf-8",
            )
            cost_anchor = append_modal_training_cost_anchor(
                out_dir=out_dir,
                metadata=meta,
                result=result,
            )
            terminal_claim = append_modal_training_terminal_claim(
                repo_root=repo_root,
                out_dir=out_dir,
                metadata=meta,
                result=result,
                agent="codex:harvest_modal_calls",
            )
            harvest_summary = {
                "rc": rc,
                "elapsed_seconds": elapsed,
                "timed_out": timed_out,
                "n_artifacts": n_artifacts,
                "crash_kind": crash_kind,
                "cost_band_anchor": cost_anchor,
                "terminal_claim": terminal_claim,
            }
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
                    "crash_kind": crash_kind,
                    "cost_band_anchor": cost_anchor,
                    "terminal_claim": terminal_claim,
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
            artifacts_dir.mkdir(exist_ok=True)
            expired_summary = {
                "status": "expired",
                "crash_kind": "RESULT_CACHE_EXPIRED",
                "n_artifacts": 0,
                "terminal_claim": terminal_claim,
            }
            (artifacts_dir / "_harvest_summary.json").write_text(
                json.dumps(expired_summary, indent=2, default=str),
                encoding="utf-8",
            )
            summary.append(
                modal_training_summary_entry(
                    label=label,
                    status="expired",
                    call_id=call_id,
                    harvested=expired_summary,
                    terminal_claim=terminal_claim,
                )
            )
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
            artifacts_dir.mkdir(exist_ok=True)
            timeout_summary = {
                "status": "function_timeout",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "crash_kind": "FUNCTION_TIMEOUT",
                "n_artifacts": 0,
                "terminal_claim": terminal_claim,
            }
            (artifacts_dir / "_harvest_summary.json").write_text(
                json.dumps(timeout_summary, indent=2, default=str),
                encoding="utf-8",
            )
            summary.append(
                modal_training_summary_entry(
                    label=label,
                    status="function_timeout",
                    call_id=call_id,
                    harvested=timeout_summary,
                    terminal_claim=terminal_claim,
                )
            )
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
            artifacts_dir.mkdir(exist_ok=True)
            unsafe_summary = {
                "status": "unsafe_artifact_path",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "crash_kind": "UNSAFE_ARTIFACT_PATH",
                "n_artifacts": 0,
                "terminal_claim": terminal_claim,
            }
            (artifacts_dir / "_harvest_summary.json").write_text(
                json.dumps(unsafe_summary, indent=2, default=str),
                encoding="utf-8",
            )
            summary.append(
                modal_training_summary_entry(
                    label=label,
                    status="unsafe_artifact_path",
                    call_id=call_id,
                    harvested=unsafe_summary,
                    terminal_claim=terminal_claim,
                )
            )
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
            artifacts_dir.mkdir(exist_ok=True)
            error_summary = {
                "status": f"error_{type(exc).__name__}",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "crash_kind": f"ERROR_{type(exc).__name__}",
                "n_artifacts": 0,
                "terminal_claim": terminal_claim,
            }
            (artifacts_dir / "_harvest_summary.json").write_text(
                json.dumps(error_summary, indent=2, default=str),
                encoding="utf-8",
            )
            summary.append(
                modal_training_summary_entry(
                    label=label,
                    status=f"error_{type(exc).__name__}",
                    call_id=call_id,
                    harvested=error_summary,
                    terminal_claim=terminal_claim,
                )
            )

    print("\n\n=== SUMMARY ===")
    for row in summary:
        print(
            f"  {row.get('label', '?')!s:30s}  "
            f"{row.get('crash_kind', row.get('status', '?'))!s:20s}  "
            f"rc={row.get('rc', '?')}  elapsed={row.get('elapsed_seconds', '?')}"
        )

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary saved: {summary_output}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if not args.execute:
        _print_plan(list_modal_lanes(repo_root=repo_root))
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
