"""Harvest all dispatched Modal call_ids before their result cache expires (~24h)."""
import json
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parents[1]
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
        summary.append({"label": label, "status": "already_harvested", "call_id": call_id})
        continue

    print(f"\n=== {label} ({call_id[:30]}, dispatched {dispatched}) ===")
    try:
        fc = modal.functions.FunctionCall.from_id(call_id)
        result = fc.get(timeout=2)
        rc = result.get("returncode", "?")
        elapsed = result.get("elapsed_seconds", 0)
        timed_out = result.get("timed_out", False)
        n_artifacts = len(result.get("artifacts", {}))
        tail = result.get("stdout_tail", "")[-500:]

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
        (artifacts_dir / "_stdout_tail.txt").write_text(result.get("stdout_tail", ""))
        (artifacts_dir / "_harvest_summary.json").write_text(json.dumps({
            "rc": rc, "elapsed_seconds": elapsed, "timed_out": timed_out,
            "n_artifacts": n_artifacts, "crash_kind": crash_kind,
        }, indent=2))

        summary.append({
            "label": label, "call_id": call_id, "rc": rc, "elapsed_seconds": elapsed,
            "n_artifacts": n_artifacts, "crash_kind": crash_kind,
        })

    except modal.exception.OutputExpiredError:
        print(f"  EXPIRED (>24h old, GC'd)")
        summary.append({"label": label, "status": "expired", "call_id": call_id})
    except modal.exception.FunctionTimeoutError as e:
        print(f"  FUNCTION TIMEOUT: {e}")
        summary.append({"label": label, "status": "function_timeout", "call_id": call_id})
    except TimeoutError:
        print(f"  NOT READY (still queued or running)")
        summary.append({"label": label, "status": "not_ready", "call_id": call_id})
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")
        summary.append({"label": label, "status": f"error_{type(e).__name__}", "call_id": call_id})

print("\n\n=== SUMMARY ===")
for s in summary:
    print(f"  {s.get('label', '?'):30s}  {s.get('crash_kind', s.get('status', '?')):20s}  rc={s.get('rc', '?')}  elapsed={s.get('elapsed_seconds', '?')}")

# Write summary to file
out = REPO / "experiments" / "results" / "_modal_harvest_summary.json"
out.write_text(json.dumps(summary, indent=2, default=str))
print(f"\nSummary saved: {out}")
