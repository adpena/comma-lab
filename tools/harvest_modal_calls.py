"""Harvest all dispatched Modal call_ids before their result cache expires (~24h)."""
import json

import modal

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.deploy.modal.training_claims import append_modal_training_terminal_claim  # noqa: E402
from tac.deploy.modal.training_cost import append_modal_training_cost_anchor  # noqa: E402
from tac.deploy.modal.harvest_summary import (  # noqa: E402
    modal_training_summary_entry,
    normalise_modal_training_result_summary,
    partial_modal_training_result_summary,
)

result_dirs = list((REPO / "experiments" / "results").glob("lane_*_modal"))
metadata_files = [d / "modal_metadata.json" for d in result_dirs if (d / "modal_metadata.json").exists()]

print(f"Found {len(metadata_files)} dispatched lanes with modal_metadata.json")
print()

summary = []
for mfile in sorted(metadata_files):
    meta = json.loads(mfile.read_text())
    call_id = meta.get("call_id", "?")
    label = meta.get("label", "?")
    dispatched = meta.get("dispatched_at", "?")
    out_dir = mfile.parent

    # Skip if we already have an artifacts/ directory (already harvested)
    artifacts_dir = out_dir / "harvested_artifacts"
    if artifacts_dir.exists() and any(artifacts_dir.iterdir()):
        print(f"[SKIP-already-harvested] {label:30s} call_id={call_id[:30]}")
        harvest_summary = artifacts_dir / "_harvest_summary.json"
        harvested = None
        if harvest_summary.is_file():
            try:
                loaded = json.loads(harvest_summary.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    harvested = loaded
            except json.JSONDecodeError:
                harvested = {"crash_kind": "INVALID_HARVEST_SUMMARY"}
        else:
            root_harvest_summary = out_dir / "harvest_summary.json"
            if root_harvest_summary.is_file():
                try:
                    loaded = json.loads(root_harvest_summary.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        harvested = normalise_modal_training_result_summary(
                            loaded,
                            artifacts_dir=artifacts_dir,
                            source_summary=root_harvest_summary,
                        )
                        (artifacts_dir / "_harvest_summary.json").write_text(
                            json.dumps(harvested, indent=2, default=str),
                            encoding="utf-8",
                        )
                except json.JSONDecodeError:
                    harvested = {"crash_kind": "INVALID_ROOT_HARVEST_SUMMARY"}
            else:
                harvested = partial_modal_training_result_summary(artifacts_dir=artifacts_dir)
                (artifacts_dir / "_harvest_summary.json").write_text(
                    json.dumps(harvested, indent=2, default=str),
                    encoding="utf-8",
                )
        cost_marker = out_dir / "cost_band_anchor_appended.json"
        cost_anchor = None
        if cost_marker.is_file():
            try:
                cost_anchor = json.loads(cost_marker.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cost_anchor = {"appended": False, "reason": "invalid_existing_cost_anchor_marker"}
        elif harvested is not None:
            try:
                cost_anchor = append_modal_training_cost_anchor(
                    out_dir=out_dir,
                    metadata=meta,
                    result=dict(harvested),
                )
            except Exception as exc:
                cost_anchor = {
                    "appended": False,
                    "reason": f"already_harvested_append_failed:{type(exc).__name__}:{exc}",
                }
        terminal_claim = None
        terminal_marker = out_dir / "modal_training_terminal_claim.json"
        if terminal_marker.is_file():
            try:
                terminal_claim = json.loads(terminal_marker.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                terminal_claim = {"appended": False, "reason": "invalid_existing_terminal_claim_marker"}
        elif harvested is not None:
            try:
                terminal_claim = append_modal_training_terminal_claim(
                    repo_root=REPO,
                    out_dir=out_dir,
                    metadata=meta,
                    result=dict(harvested),
                    agent="codex:harvest_modal_calls",
                )
            except Exception as exc:
                terminal_claim = {
                    "appended": False,
                    "reason": f"already_harvested_terminal_claim_failed:{type(exc).__name__}:{exc}",
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
        result = fc.get(timeout=2)
        rc = result.get("returncode", "?")
        elapsed_raw = result.get("elapsed_seconds")
        elapsed = elapsed_raw if isinstance(elapsed_raw, (int, float)) else 0
        timed_out = result.get("timed_out", False)
        n_artifacts = len(result.get("artifacts", {}))
        tail = result.get("stdout_tail", "") or ""
        tail = tail[-500:]

        # Heuristic crash classification
        crash_kind = "OK"
        if "OutOfMemoryError" in tail or "CUDA out of memory" in tail:
            crash_kind = "CUDA_OOM"
        elif "FATAL" in tail or "Traceback" in tail:
            crash_kind = "ERROR"
        elif timed_out:
            crash_kind = "TIMEOUT"
        elif rc != 0:
            crash_kind = f"RC_{rc}"

        print(f"  rc={rc} elapsed={elapsed:.0f}s timed_out={timed_out} artifacts={n_artifacts} crash={crash_kind}")

        # Save artifacts to disk
        artifacts_dir.mkdir(exist_ok=True)
        for relpath, data in result.get("artifacts", {}).items():
            target = artifacts_dir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.write_bytes(data)
            except Exception as e:
                print(f"    SKIP {relpath}: {e}")

        # Save full stdout tail too
        (artifacts_dir / "_stdout_tail.txt").write_text(result.get("stdout_tail", "") or "")
        cost_anchor = append_modal_training_cost_anchor(
            out_dir=out_dir,
            metadata=meta,
            result=result,
        )
        terminal_claim = append_modal_training_terminal_claim(
            repo_root=REPO,
            out_dir=out_dir,
            metadata=meta,
            result=result,
            agent="codex:harvest_modal_calls",
        )
        (artifacts_dir / "_harvest_summary.json").write_text(json.dumps({
            "rc": rc, "elapsed_seconds": elapsed, "timed_out": timed_out,
            "n_artifacts": n_artifacts, "crash_kind": crash_kind,
            "cost_band_anchor": cost_anchor,
            "terminal_claim": terminal_claim,
        }, indent=2))

        summary.append({
            "label": label, "call_id": call_id, "rc": rc, "elapsed_seconds": elapsed,
            "n_artifacts": n_artifacts, "crash_kind": crash_kind,
            "cost_band_anchor": cost_anchor,
            "terminal_claim": terminal_claim,
        })

    except modal.exception.OutputExpiredError:
        print("  EXPIRED (>24h old, GC'd)")
        terminal_claim = append_modal_training_terminal_claim(
            repo_root=REPO,
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
            json.dumps(expired_summary, indent=2, default=str), encoding="utf-8"
        )
        summary.append(modal_training_summary_entry(
            label=label,
            status="expired",
            call_id=call_id,
            harvested=expired_summary,
            terminal_claim=terminal_claim,
        ))
    except modal.exception.FunctionTimeoutError as e:
        print(f"  FUNCTION TIMEOUT: {e}")
        summary.append({"label": label, "status": "function_timeout", "call_id": call_id})
    except TimeoutError:
        print("  NOT READY (still queued or running)")
        summary.append({"label": label, "status": "not_ready", "call_id": call_id})
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")
        status = f"failed_modal_training_recovery_error_{type(e).__name__.lower()}"
        terminal_claim = append_modal_training_terminal_claim(
            repo_root=REPO,
            out_dir=out_dir,
            metadata=meta,
            result=None,
            status=status,
            agent="codex:harvest_modal_calls",
        )
        artifacts_dir.mkdir(exist_ok=True)
        error_summary = {
            "status": f"error_{type(e).__name__}",
            "error_type": type(e).__name__,
            "error_message": str(e)[:1000],
            "crash_kind": f"ERROR_{type(e).__name__}",
            "n_artifacts": 0,
            "terminal_claim": terminal_claim,
        }
        (artifacts_dir / "_harvest_summary.json").write_text(
            json.dumps(error_summary, indent=2, default=str), encoding="utf-8"
        )
        summary.append(modal_training_summary_entry(
            label=label,
            status=f"error_{type(e).__name__}",
            call_id=call_id,
            harvested=error_summary,
            terminal_claim=terminal_claim,
        ))

print("\n\n=== SUMMARY ===")
for s in summary:
    print(f"  {s.get('label', '?'):30s}  {s.get('crash_kind', s.get('status', '?')):20s}  rc={s.get('rc', '?')}  elapsed={s.get('elapsed_seconds', '?')}")

# Write summary to file
out = REPO / "experiments" / "results" / "_modal_harvest_summary.json"
out.write_text(json.dumps(summary, indent=2, default=str))
print(f"\nSummary saved: {out}")
