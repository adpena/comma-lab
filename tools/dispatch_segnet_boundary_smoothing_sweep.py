#!/usr/bin/env python3
r"""Dispatch all 4 SegNet boundary smoothing variants to GHA CPU eval in
parallel. Wraps ``tools/dispatch_cpu_eval_via_github_actions.py`` per variant.

Usage:
  .venv/bin/python tools/dispatch_segnet_boundary_smoothing_sweep.py \\
      --rollup experiments/results/segnet_boundary_smoothing_rollup_<ts>.json \\
      [--max-concurrent 4]

Per CLAUDE.md:
  - HIGH 1 fix in dispatcher (codex round-2) landed: submission_name matching
    is now exact-identity, so distinct submission names per variant are still
    used for clarity but are no longer load-bearing for custody.
  - Per "Submission auth eval — BOTH CPU AND CUDA": this is the CPU axis only.
    For any variant that lands sub-0.190, a paired CUDA dispatch is required
    before promotion.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DISPATCHER = REPO_ROOT / "tools" / "dispatch_cpu_eval_via_github_actions.py"


def dispatch_one(variant: dict[str, Any]) -> dict[str, Any]:
    """Run the dispatcher for one variant; return per-variant status."""
    submission_name = variant["submission_name"]
    archive_path = REPO_ROOT / variant["archive_path"]
    archive_sha = variant["archive_sha256"]
    submission_dir = archive_path.parent
    out_dir = submission_dir.parent / "gha_dispatch"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(DISPATCHER),
        "--archive-path", str(archive_path),
        "--archive-sha", archive_sha,
        "--submission-name", submission_name,
        "--submission-dir", str(submission_dir),
        "--output-dir", str(out_dir),
        "--auto-create-fork-pr",
    ]
    print(f"[dispatch] {submission_name}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    return {
        "variant_id": variant["variant_id"],
        "submission_name": submission_name,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:],
        "out_dir": str(out_dir.relative_to(REPO_ROOT)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rollup", type=Path, required=True, help="path to rollup.json")
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help="subset of variant_ids to dispatch (default: all)",
    )
    args = p.parse_args()

    rollup = json.loads(args.rollup.read_text())
    variants = rollup["variants"]
    if args.variants is not None:
        wanted = set(args.variants)
        variants = [v for v in variants if v["variant_id"] in wanted]
        if len(variants) != len(wanted):
            sys.stderr.write(
                f"[fatal] unknown variant_ids; available: "
                f"{[v['variant_id'] for v in rollup['variants']]}\n"
            )
            return 2

    print(
        f"[start] dispatching {len(variants)} variants concurrent={args.max_concurrent}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_concurrent
    ) as ex:
        fs = {ex.submit(dispatch_one, v): v for v in variants}
        for f in concurrent.futures.as_completed(fs):
            results.append(f.result())
            r = results[-1]
            print(
                f"[done] {r['variant_id']:<28} rc={r['returncode']} "
                f"out={r['out_dir']}",
                flush=True,
            )

    # Write results next to rollup
    out_path = (
        args.rollup.parent
        / args.rollup.name.replace("_rollup_", "_dispatch_results_")
    )
    out_path.write_text(json.dumps(
        {
            "schema_version": "segnet_boundary_smoothing_dispatch_v1",
            "rollup_input": str(args.rollup.relative_to(REPO_ROOT))
            if args.rollup.is_relative_to(REPO_ROOT) else str(args.rollup),
            "n_variants": len(variants),
            "results": results,
        },
        indent=2, sort_keys=True,
    ) + "\n")
    print(f"\n[ok] dispatch results -> {out_path.relative_to(REPO_ROOT)}", flush=True)

    rc_sum = sum(1 for r in results if r["returncode"] != 0)
    if rc_sum > 0:
        print(f"\n[warn] {rc_sum} variants returned non-zero", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
