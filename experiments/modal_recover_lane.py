"""Recover artifacts from a spawned Modal lane training run.

Companion to `experiments/modal_train_lane.py`. When a lane is dispatched
via `.spawn()` (the only way to survive local CLI disconnect), the function
runs detached and the call_id is saved to
`experiments/results/lane_<label>_modal/modal_call_id.txt`.

Use this script to:
1. Poll the function call status
2. Once complete, download artifacts to local
3. Auto-extract auth score

Usage:
    .venv/bin/python experiments/modal_recover_lane.py --label lane_omega_hessian
    .venv/bin/python experiments/modal_recover_lane.py --call-id fc-abc123...
    .venv/bin/python experiments/modal_recover_lane.py --all  # poll every lane_*_modal/

Reference: project_modal_pipeline_trusted_lane_g_v3_1_04_20260429
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def recover_one(label: str | None, call_id: str | None) -> int:
    import modal

    if label and not call_id:
        sentinel = REPO_ROOT / "experiments" / "results" / f"lane_{label}_modal" / "modal_call_id.txt"
        if not sentinel.exists():
            print(f"FATAL: no call_id sentinel at {sentinel}", file=sys.stderr)
            return 2
        call_id = sentinel.read_text().strip()
    if not call_id:
        print("FATAL: must provide --label or --call-id", file=sys.stderr)
        return 2

    print(f"=== Polling Modal call_id={call_id} ===")
    fc = modal.FunctionCall.from_id(call_id)

    # `get(timeout=0)` returns immediately if done; raises TimeoutError if not.
    try:
        result = fc.get(timeout=0)
    except TimeoutError:
        print(f"  STILL RUNNING — re-run later. modal call get {call_id}")
        return 0
    except modal.exception.OutputExpiredError:
        print(f"  OUTPUT EXPIRED (Modal expires output after 7 days). Lane logs may still be available.", file=sys.stderr)
        return 3

    if not isinstance(result, dict):
        print(f"  unexpected result type: {type(result)}", file=sys.stderr)
        return 4

    # Save artifacts
    if not label:
        # Derive label from a hint in the result, else use call_id
        label = "unknown"
        for k in result.get("artifacts", {}):
            m = re.search(r"lane_([\w_-]+)_results/", k)
            if m:
                label = m.group(1).split("_")[0]
                break
    out_dir = REPO_ROOT / "experiments" / "results" / f"lane_{label}_modal"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for path, data in result.get("artifacts", {}).items():
        full = out_dir / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)
        n_saved += 1
    print(f"  saved {n_saved} artifacts → {out_dir}")

    rc = result.get("returncode")
    timed_out = result.get("timed_out", False)
    elapsed = result.get("elapsed_seconds")
    print(f"  returncode={rc}  timed_out={timed_out}  elapsed={elapsed:.0f}s" if elapsed else f"  returncode={rc}")

    # Extract score from artifacts
    score_found = False
    for path, data_bytes in result.get("artifacts", {}).items():
        if score_found:
            break
        if path.endswith(".json"):
            try:
                d = json.loads(data_bytes.decode())
                if isinstance(d, dict):
                    score = d.get("score") or d.get("final_score")
                    if score is not None:
                        print(f"\n=== AUTH SCORE: {score} (label={label}) ===")
                        print(f"  source:  {path}")
                        print(f"  PoseNet: {d.get('pose') or d.get('pose_dist')}")
                        print(f"  SegNet:  {d.get('seg') or d.get('seg_dist')}")
                        print(f"  Rate:    {d.get('rate')}")
                        score_found = True
            except Exception:
                pass
        elif path.endswith(".log"):
            try:
                text = data_bytes.decode(errors="ignore")
                m = re.search(r"RESULT_JSON:\s*(\{[^\n]+\})", text)
                if m:
                    d = json.loads(m.group(1))
                    score = d.get("score") or d.get("final_score")
                    if score is not None:
                        print(f"\n=== AUTH SCORE: {score} (label={label}) ===")
                        print(f"  source:  {path} (RESULT_JSON line)")
                        score_found = True
            except Exception:
                pass

    if not score_found:
        print(f"  WARNING: no auth score in artifacts. Check {out_dir}/ for run.log + auth_eval.log.")
    if rc != 0:
        return rc
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", help="Lane label (read call_id from sentinel)")
    p.add_argument("--call-id", help="Modal function call ID directly")
    p.add_argument("--all", action="store_true",
                   help="Poll every lane_*_modal/ directory with a sentinel")
    args = p.parse_args()

    if args.all:
        results_dir = REPO_ROOT / "experiments" / "results"
        labels = []
        for d in sorted(results_dir.glob("lane_*_modal")):
            sentinel = d / "modal_call_id.txt"
            if sentinel.exists():
                labels.append(d.name.replace("lane_", "").replace("_modal", ""))
        print(f"Polling {len(labels)} lanes: {labels}")
        rcs = [recover_one(label=lbl, call_id=None) for lbl in labels]
        return max(rcs) if rcs else 0
    return recover_one(args.label, args.call_id)


if __name__ == "__main__":
    sys.exit(main())
