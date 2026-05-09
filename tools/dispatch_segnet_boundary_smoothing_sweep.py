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
CLAIM_HELPER = REPO_ROOT / "tools" / "claim_lane_dispatch.py"
DEFAULT_AGENT = "codex:segnet_boundary_smoothing_sweep"
DEFAULT_PLATFORM = "github_actions"


def _repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def dispatch_results_path(rollup_path: Path) -> Path:
    """Return a dispatch-results path without risking rollup overwrite."""
    if "_rollup_" in rollup_path.name:
        return rollup_path.parent / rollup_path.name.replace(
            "_rollup_", "_dispatch_results_"
        )
    return rollup_path.with_name(
        f"{rollup_path.stem}_dispatch_results{rollup_path.suffix}"
    )


def build_claim_cmd(
    *,
    lane_id: str,
    instance_job_id: str,
    agent: str,
    platform: str,
    status: str,
    notes: str,
    child_of: str | None = None,
    parallel_reason: str = "",
) -> list[str]:
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(CLAIM_HELPER),
        "claim",
        "--lane-id",
        lane_id,
        "--platform",
        platform,
        "--instance-job-id",
        instance_job_id,
        "--agent",
        agent,
        "--status",
        status,
        "--notes",
        notes,
    ]
    if child_of:
        cmd.extend(
            [
                "--allow-parallel",
                "--child-of",
                child_of,
                "--parallel-reason",
                parallel_reason,
            ]
        )
    return cmd


def run_claim(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def dispatch_one(
    variant: dict[str, Any],
    *,
    lane_id: str,
    parent_job_id: str,
    agent: str,
    platform: str,
    skip_dispatch: bool,
) -> dict[str, Any]:
    """Run the dispatcher for one variant; return per-variant status."""
    submission_name = variant["submission_name"]
    archive_path = REPO_ROOT / variant["archive_path"]
    archive_sha = variant["archive_sha256"]
    submission_dir = archive_path.parent
    out_dir = submission_dir.parent / "gha_dispatch"
    out_dir.mkdir(parents=True, exist_ok=True)

    claim_proc = run_claim(
        build_claim_cmd(
            lane_id=lane_id,
            instance_job_id=submission_name,
            agent=agent,
            platform=platform,
            status="eval_cpu",
            notes=(
                f"child of {parent_job_id}; SegNet smoothing variant "
                f"{variant['variant_id']}"
            ),
            child_of=parent_job_id,
            parallel_reason="bounded SegNet boundary smoothing sweep variants",
        )
    )
    if claim_proc.returncode != 0:
        return {
            "variant_id": variant["variant_id"],
            "submission_name": submission_name,
            "returncode": claim_proc.returncode,
            "claim_failed": True,
            "stdout_tail": claim_proc.stdout[-1500:],
            "stderr_tail": claim_proc.stderr[-1500:],
            "out_dir": _repo_rel(out_dir),
        }

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
    if skip_dispatch:
        cmd.append("--skip-dispatch")
    print(f"[dispatch] {submission_name}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    terminal_proc = run_claim(
        build_claim_cmd(
            lane_id=lane_id,
            instance_job_id=submission_name,
            agent=agent,
            platform=platform,
            status="completed_gha_cpu_eval"
            if proc.returncode == 0
            else "failed_gha_cpu_eval",
            notes=(
                f"SegNet smoothing variant {variant['variant_id']} "
                f"rc={proc.returncode}"
            ),
        )
    )
    return {
        "variant_id": variant["variant_id"],
        "submission_name": submission_name,
        "returncode": proc.returncode,
        "claim_failed": False,
        "terminal_claim_returncode": terminal_proc.returncode,
        "terminal_claim_stdout_tail": terminal_proc.stdout[-1500:],
        "terminal_claim_stderr_tail": terminal_proc.stderr[-1500:],
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:],
        "out_dir": _repo_rel(out_dir),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rollup", type=Path, required=True, help="path to rollup.json")
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument("--agent", default=DEFAULT_AGENT)
    p.add_argument("--claim-platform", default=DEFAULT_PLATFORM)
    p.add_argument(
        "--skip-dispatch",
        action="store_true",
        help=(
            "pass through to the CPU dispatcher; validates upload/runtime prep "
            "without launching eval"
        ),
    )
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
    lane_id = str(
        rollup.get("lane_id") or "lane_a1_segnet_boundary_smoothing_inflate"
    )
    parent_job_id = f"{lane_id}_sweep_{rollup.get('timestamp', args.rollup.stem)}"
    if args.max_concurrent < 1:
        sys.stderr.write("[fatal] --max-concurrent must be >= 1\n")
        return 2
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
    parent_claim = run_claim(
        build_claim_cmd(
            lane_id=lane_id,
            instance_job_id=parent_job_id,
            agent=args.agent,
            platform=args.claim_platform,
            status="eval_cpu_sweep",
            notes=f"SegNet boundary smoothing sweep from {_repo_rel(args.rollup)}",
        )
    )
    if parent_claim.returncode != 0:
        sys.stderr.write(
            "[fatal] parent dispatch claim failed; refusing to dispatch\n"
            f"stdout={parent_claim.stdout[-1500:]}\n"
            f"stderr={parent_claim.stderr[-1500:]}\n"
        )
        return parent_claim.returncode

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_concurrent
    ) as ex:
        fs = {
            ex.submit(
                dispatch_one,
                v,
                lane_id=lane_id,
                parent_job_id=parent_job_id,
                agent=args.agent,
                platform=args.claim_platform,
                skip_dispatch=args.skip_dispatch,
            ): v
            for v in variants
        }
        for f in concurrent.futures.as_completed(fs):
            results.append(f.result())
            r = results[-1]
            print(
                f"[done] {r['variant_id']:<28} rc={r['returncode']} "
                f"out={r['out_dir']}",
                flush=True,
            )

    # Write results next to rollup
    out_path = dispatch_results_path(args.rollup)
    out_path.write_text(json.dumps(
        {
            "schema_version": "segnet_boundary_smoothing_dispatch_v1",
            "rollup_input": _repo_rel(args.rollup),
            "lane_id": lane_id,
            "parent_job_id": parent_job_id,
            "n_variants": len(variants),
            "results": results,
        },
        indent=2, sort_keys=True,
    ) + "\n")
    print(f"\n[ok] dispatch results -> {_repo_rel(out_path)}", flush=True)

    rc_sum = sum(1 for r in results if r["returncode"] != 0)
    if rc_sum > 0:
        print(f"\n[warn] {rc_sum} variants returned non-zero", flush=True)
    parent_terminal = run_claim(
        build_claim_cmd(
            lane_id=lane_id,
            instance_job_id=parent_job_id,
            agent=args.agent,
            platform=args.claim_platform,
            status="completed_gha_cpu_sweep" if rc_sum == 0 else "failed_gha_cpu_sweep",
            notes=(
                f"SegNet boundary smoothing sweep finished nonzero={rc_sum}; "
                f"results={_repo_rel(out_path)}"
            ),
        )
    )
    if parent_terminal.returncode != 0:
        print(
            "[warn] parent terminal claim failed "
            f"rc={parent_terminal.returncode} "
            f"stderr={parent_terminal.stderr[-500:]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
