#!/usr/bin/env python3
"""Dispatch Phase A1 (track1_phase_a1_score_gradient) to a remote GPU.

Phase A1: score-gradient supervision PR101 fine-tune. Council priority: 22/22
ENDORSE, UNANIMOUS HIGHEST PRIORITY (Quantizr/Carmack/Hinton). Reference:
``.omx/research/grand_council_extreme_rigor_track_1_20260508.md``.

This is the ACTUATOR per CLAUDE.md "parallel-dispatch is a FIRST-CLASS
DELIVERABLE". It runs the full chain on the remote GPU:

    1. Stage workspace via lightning_repro_workspace.py (or scp on Vast.ai)
    2. Open lane claim via tools/claim_lane_dispatch.py
    3. Submit dispatch via Lightning SSH (calling
       scripts/remote_track1_phase_a1_score_gradient_pr101.sh) — that script
       chains: bootstrap → train → build_archive → contest_auth_eval --device cuda
    4. Write heartbeat scaffold + manifest stub at
       experiments/results/<lane_id>_<timestamp>/

Returns when the job is FIRED (does not wait for completion). Harvest via
launch_lane_lightning.py status / harvest commands using the printed session_id.

Cost: Lightning T4 ~$0.66/hr × ~3.5h = ~$2.30. Hard cap: $8 per --cost-cap.

Usage:
    .venv/bin/python tools/dispatch_phase_a1_score_gradient_pr101.py \\
        --lane-id track1_phase_a1_score_gradient \\
        --pr101-archive experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/archive.zip \\
        --video-path upstream/videos/0.mkv \\
        --pr101-source-dir experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/src \\
        --provider lightning \\
        --gpu-tier T4 \\
        --epochs 200 \\
        --cost-cap 8.0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Cost gates
# ---------------------------------------------------------------------------

HOURLY_RATES = {
    "lightning_t4": 0.66,
    "lightning_a10g": 1.10,
    "lightning_l40s": 1.95,
    "lightning_a100": 3.00,
    "vastai_4090": 0.30,
    "vastai_a100": 1.40,
}


def estimated_cost_usd(provider: str, gpu_tier: str, hours: float) -> float:
    key = f"{provider}_{gpu_tier.lower()}"
    rate = HOURLY_RATES.get(key)
    if rate is None:
        # Fallback to Lightning T4 rate if unknown.
        rate = HOURLY_RATES["lightning_t4"]
    return float(rate) * float(hours)


# ---------------------------------------------------------------------------
# Lane claim
# ---------------------------------------------------------------------------

def claim_lane(
    *,
    lane_id: str,
    platform: str,
    instance_job_id: str,
    predicted_eta_utc: str,
    notes: str = "",
    status: str = "active_dispatching",
    force: bool = False,
) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--lane-id", lane_id,
        "--platform", platform,
        "--instance-job-id", instance_job_id,
        "--agent", "claude:dispatch_phase_a1_score_gradient_pr101",
        "--predicted-eta-utc", predicted_eta_utc,
        "--status", status,
    ]
    if notes:
        cmd.extend(["--notes", notes])
    if force:
        cmd.append("--force")
    print(f"[claim] {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    return proc.returncode


# ---------------------------------------------------------------------------
# Provenance + heartbeat
# ---------------------------------------------------------------------------

def write_heartbeat(lane_dir: Path, message: str) -> None:
    lane_dir.mkdir(parents=True, exist_ok=True)
    with (lane_dir / "heartbeat.txt").open("a") as f:
        f.write(f"{dt.datetime.now(tz=dt.UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')} {message}\n")


def write_dispatch_manifest(
    lane_dir: Path,
    *,
    lane_id: str,
    instance_job_id: str,
    provider: str,
    gpu_tier: str,
    estimated_cost_usd: float,
    predicted_low: float,
    predicted_high: float,
    pr101_archive: str,
    video_path: str,
    pr101_source_dir: str,
    epochs: int,
    timeout_hours: float,
    session_id: str | None,
    dispatch_attempted: bool,
    dispatch_status: str,
    started_at_utc: str,
    predicted_eta_utc: str,
    args_dict: dict,
) -> Path:
    in_flight = dispatch_attempted and dispatch_status.startswith("fired")
    manifest = {
        "lane_id": lane_id,
        "schema_version": "phase_a1_dispatch_manifest_v1",
        "council_memo_ref": ".omx/research/grand_council_extreme_rigor_track_1_20260508.md",
        "council_decision": "A1 — score-gradient supervision (UNANIMOUS HIGHEST PRIORITY)",
        "instance_job_id": instance_job_id,
        "session_id": session_id,
        "provider": provider,
        "gpu_tier": gpu_tier,
        "estimated_cost_usd": estimated_cost_usd,
        "estimated_duration_hours": timeout_hours,
        "predicted_band": [predicted_low, predicted_high],
        "started_at_utc": started_at_utc,
        "predicted_eta_utc": predicted_eta_utc,
        "remote_lane_script": "scripts/remote_track1_phase_a1_score_gradient_pr101.sh",
        "training_script": "experiments/train_score_gradient_pr101_finetune.py",
        "archive_build_tool": "tools/build_pr101_finetuned_archive.py",
        "pr101_archive": pr101_archive,
        "video_path": video_path,
        "pr101_source_dir": pr101_source_dir,
        "epochs": epochs,
        "evidence_grade": (
            "[advisory only — dispatch in flight]"
            if in_flight
            else "[advisory only — no dispatch]"
        ),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_blockers": ["dispatch_in_flight"] if in_flight else [dispatch_status],
        "dispatch_attempted": dispatch_attempted,
        "dispatch_status": dispatch_status,
        "harvest_command_hint": (
            f".venv/bin/python scripts/launch_lane_lightning.py harvest "
            f"--session-id {session_id} --local-dir {lane_dir}/harvested"
            if session_id
            else (
                "use platform-specific status command with the instance_job_id"
                if in_flight
                else None
            )
        ),
        "args": args_dict,
    }
    manifest_path = lane_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# ---------------------------------------------------------------------------
# Lightning dispatch
# ---------------------------------------------------------------------------

def dispatch_lightning(
    *,
    lane_id: str,
    instance_job_id: str,
    pr101_archive: Path,
    video_path: Path,
    pr101_source_dir: Path,
    epochs: int,
    predicted_low: float,
    predicted_high: float,
    estimated_cost_usd: float,
    gpu_tier: str,
    allow_gpu_mismatch: bool,
    print_only: bool,
) -> tuple[str | None, bool]:
    """Submit the lane to Lightning via launch_lane_lightning.py dispatch.

    The remote script reads PR101_ARCHIVE_PATH / VIDEO_PATH / etc. from env.
    Returns the session_id if the dispatch was fired, else None.
    """
    lane_script = "scripts/remote_track1_phase_a1_score_gradient_pr101.sh"

    env_args: list[str] = []
    env_args += ["--env", f"LANE_ID={lane_id}"]
    env_args += ["--env", f"PR101_ARCHIVE_PATH={pr101_archive.relative_to(REPO_ROOT)}"]
    env_args += ["--env", f"PR101_SOURCE_DIR={pr101_source_dir.relative_to(REPO_ROOT)}"]
    env_args += ["--env", f"VIDEO_PATH={video_path.relative_to(REPO_ROOT)}"]
    env_args += ["--env", f"EPOCHS={epochs}"]
    env_args += ["--env", f"PRED_LOW={predicted_low}"]
    env_args += ["--env", f"PRED_HIGH={predicted_high}"]

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "launch_lane_lightning.py"),
        "dispatch",
        "--lane-script", lane_script,
        "--label", instance_job_id,
        "--predicted-band", str(predicted_low), str(predicted_high),
        "--estimated-cost", str(estimated_cost_usd),
        "--gpu-tier", gpu_tier,
        "--kill-criteria",
        f"max_cost_usd={estimated_cost_usd*1.2:.2f}; "
        f"min(seg_delta_pct,pose_delta_pct)<5%→retire; "
        f"NaN_in_loss→abort",
        *env_args,
    ]
    if allow_gpu_mismatch:
        cmd.append("--allow-gpu-mismatch")

    print(f"[dispatch] {' '.join(shlex.quote(c) for c in cmd)}")
    if print_only:
        print("[dispatch] --print-only: not actually launching")
        return None, False

    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"[dispatch] FAILED rc={proc.returncode}", file=sys.stderr)
        return None, False

    # Parse session_id from JSON output.
    try:
        # launch_lane_lightning.py prints JSON via _print_json; find first {...} block
        out = proc.stdout
        first_brace = out.find("{")
        last_brace = out.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            payload = json.loads(out[first_brace : last_brace + 1])
            return payload.get("session_id"), True
    except Exception as exc:
        print(f"[dispatch] WARN: failed to parse session_id: {exc}", file=sys.stderr)
    return None, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lane-id", default="track1_phase_a1_score_gradient",
                   help="Base lane id (a timestamp suffix is appended for instance_job_id)")
    p.add_argument("--pr101-archive", type=Path, required=True,
                   help="Repo-relative path to PR101 source archive.zip")
    p.add_argument("--pr101-source-dir", type=Path, required=True,
                   help="Repo-relative path to PR101 hnerv_ft_microcodec/src dir")
    p.add_argument("--video-path", type=Path, required=True,
                   help="Repo-relative path to upstream/videos/0.mkv")
    p.add_argument("--provider", choices=["lightning", "vastai"], default="lightning")
    p.add_argument("--gpu-tier", default="T4",
                   help="GPU tier (T4/A10G/L40S/A100/H100). Lightning g4dn.2xlarge maps to T4.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--predicted-low", type=float, default=0.150,
                   help="Predicted score band lower bound [contest-CUDA]")
    p.add_argument("--predicted-high", type=float, default=0.220,
                   help="Predicted score band upper bound [contest-CUDA]")
    p.add_argument("--timeout-hours", type=float, default=4.0,
                   help="Wall-clock cap for the remote chain (default 4h)")
    p.add_argument("--cost-cap", type=float, default=8.0,
                   help="Hard cap on estimated $; abort dispatch if exceeded")
    p.add_argument("--allow-gpu-mismatch", action="store_true",
                   help="Allow dispatch even if Lightning Studio's GPU doesn't match")
    p.add_argument("--print-only", action="store_true",
                   help="Print resolved invocation without claiming or dispatching")
    p.add_argument("--force-claim", action="store_true",
                   help="Pass --force to lane claim (only for explicit conflict resolution)")
    p.add_argument("--output-root", type=Path,
                   default=REPO_ROOT / "experiments" / "results")
    args = p.parse_args(argv)

    # Path resolution.
    pr101_archive = args.pr101_archive
    if not pr101_archive.is_absolute():
        pr101_archive = REPO_ROOT / pr101_archive
    if not pr101_archive.is_file():
        print(f"FATAL: --pr101-archive not found: {pr101_archive}", file=sys.stderr)
        return 2
    try:
        pr101_archive.relative_to(REPO_ROOT)
    except ValueError:
        print(f"FATAL: --pr101-archive must be inside repo root: {pr101_archive}", file=sys.stderr)
        return 2

    video_path = args.video_path
    if not video_path.is_absolute():
        video_path = REPO_ROOT / video_path
    if not video_path.is_file():
        print(f"FATAL: --video-path not found: {video_path}", file=sys.stderr)
        return 2
    try:
        video_path.relative_to(REPO_ROOT)
    except ValueError:
        print(f"FATAL: --video-path must be inside repo root: {video_path}", file=sys.stderr)
        return 2

    pr101_source_dir = args.pr101_source_dir
    if not pr101_source_dir.is_absolute():
        pr101_source_dir = REPO_ROOT / pr101_source_dir
    if not pr101_source_dir.is_dir():
        print(f"FATAL: --pr101-source-dir not found: {pr101_source_dir}", file=sys.stderr)
        return 2
    try:
        pr101_source_dir.relative_to(REPO_ROOT)
    except ValueError:
        print(f"FATAL: --pr101-source-dir must be inside repo root: {pr101_source_dir}", file=sys.stderr)
        return 2

    # Cost gate.
    cost = estimated_cost_usd(args.provider, args.gpu_tier, args.timeout_hours)
    print(f"[cost-gate] estimated ${cost:.2f} for {args.provider} {args.gpu_tier} × {args.timeout_hours}h "
          f"(cap ${args.cost_cap:.2f})")
    if cost > args.cost_cap:
        print(f"FATAL: estimated cost ${cost:.2f} exceeds cap ${args.cost_cap:.2f}; abort.", file=sys.stderr)
        return 3

    # Lane dir and instance_job_id.
    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    instance_job_id = f"{args.lane_id}_{timestamp}"
    lane_dir = args.output_root / instance_job_id
    lane_dir.mkdir(parents=True, exist_ok=True)

    started_at_utc = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    predicted_eta_utc = (
        dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=args.timeout_hours)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")

    # Heartbeat (pre-dispatch).
    write_heartbeat(lane_dir, f"DISPATCH_START provider={args.provider} gpu={args.gpu_tier} "
                              f"eta={predicted_eta_utc} cost=${cost:.2f}")

    if args.print_only:
        print("[print-only] skipping lane claim and remote dispatch")
        # Still write a manifest so operator can audit the resolved invocation.
        write_dispatch_manifest(
            lane_dir,
            lane_id=args.lane_id,
            instance_job_id=instance_job_id,
            provider=args.provider,
            gpu_tier=args.gpu_tier,
            estimated_cost_usd=cost,
            predicted_low=args.predicted_low,
            predicted_high=args.predicted_high,
            pr101_archive=str(pr101_archive),
            video_path=str(video_path),
            pr101_source_dir=str(pr101_source_dir),
            epochs=args.epochs,
            timeout_hours=args.timeout_hours,
            session_id=None,
            dispatch_attempted=False,
            dispatch_status="print_only_no_dispatch",
            started_at_utc=started_at_utc,
            predicted_eta_utc=predicted_eta_utc,
            args_dict={k: str(v) for k, v in vars(args).items()},
        )
        # Still print the invocation for audit.
        dispatch_lightning(
            lane_id=args.lane_id,
            instance_job_id=instance_job_id,
            pr101_archive=pr101_archive,
            video_path=video_path,
            pr101_source_dir=pr101_source_dir,
            epochs=args.epochs,
            predicted_low=args.predicted_low,
            predicted_high=args.predicted_high,
            estimated_cost_usd=cost,
            gpu_tier=args.gpu_tier,
            allow_gpu_mismatch=args.allow_gpu_mismatch,
            print_only=True,
        )
        return 0

    if args.provider != "lightning":
        print(
            f"FATAL: provider {args.provider!r} not yet supported in this dispatcher; "
            "use --provider lightning",
            file=sys.stderr,
        )
        return 5

    # Open lane claim BEFORE GPU spend (CLAUDE.md NON-NEGOTIABLE).
    platform_for_claim = "lightning"
    claim_rc = claim_lane(
        lane_id=args.lane_id,
        platform=platform_for_claim,
        instance_job_id=instance_job_id,
        predicted_eta_utc=predicted_eta_utc,
        notes=(
            f"Phase A1 score-gradient supervision PR101 fine-tune; "
            f"council UNANIMOUS HIGHEST PRIORITY; predicted=[{args.predicted_low}, {args.predicted_high}]; "
            f"cost=${cost:.2f}; archive={pr101_archive.name}"
        ),
        force=args.force_claim,
    )
    if claim_rc != 0:
        print(f"FATAL: lane claim failed rc={claim_rc}; aborting before GPU spend", file=sys.stderr)
        return 4

    # Submit dispatch.
    session_id, dispatch_ok = dispatch_lightning(
        lane_id=args.lane_id,
        instance_job_id=instance_job_id,
        pr101_archive=pr101_archive,
        video_path=video_path,
        pr101_source_dir=pr101_source_dir,
        epochs=args.epochs,
        predicted_low=args.predicted_low,
        predicted_high=args.predicted_high,
        estimated_cost_usd=cost,
        gpu_tier=args.gpu_tier,
        allow_gpu_mismatch=args.allow_gpu_mismatch,
        print_only=False,
    )
    if not dispatch_ok:
        claim_lane(
            lane_id=args.lane_id,
            platform=platform_for_claim,
            instance_job_id=instance_job_id,
            predicted_eta_utc=started_at_utc,
            notes="Phase A1 dispatch submission failed before remote job fired",
            status="failed_dispatch_submission",
            force=True,
        )
        write_heartbeat(lane_dir, "DISPATCH_SUBMISSION_FAILED")
        write_dispatch_manifest(
            lane_dir,
            lane_id=args.lane_id,
            instance_job_id=instance_job_id,
            provider=args.provider,
            gpu_tier=args.gpu_tier,
            estimated_cost_usd=cost,
            predicted_low=args.predicted_low,
            predicted_high=args.predicted_high,
            pr101_archive=str(pr101_archive),
            video_path=str(video_path),
            pr101_source_dir=str(pr101_source_dir),
            epochs=args.epochs,
            timeout_hours=args.timeout_hours,
            session_id=None,
            dispatch_attempted=True,
            dispatch_status="failed_dispatch_submission",
            started_at_utc=started_at_utc,
            predicted_eta_utc=predicted_eta_utc,
            args_dict={k: str(v) for k, v in vars(args).items()},
        )
        return 6

    write_heartbeat(lane_dir, f"DISPATCH_FIRED session_id={session_id or 'unknown'}")

    manifest_path = write_dispatch_manifest(
        lane_dir,
        lane_id=args.lane_id,
        instance_job_id=instance_job_id,
        provider=args.provider,
        gpu_tier=args.gpu_tier,
        estimated_cost_usd=cost,
        predicted_low=args.predicted_low,
        predicted_high=args.predicted_high,
        pr101_archive=str(pr101_archive),
        video_path=str(video_path),
        pr101_source_dir=str(pr101_source_dir),
        epochs=args.epochs,
        timeout_hours=args.timeout_hours,
        session_id=session_id,
        dispatch_attempted=True,
        dispatch_status="fired" if session_id else "fired_session_id_unknown",
        started_at_utc=started_at_utc,
        predicted_eta_utc=predicted_eta_utc,
        args_dict={k: str(v) for k, v in vars(args).items()},
    )
    print(f"[dispatch] FIRED session_id={session_id} manifest={manifest_path}")
    print(f"[harvest-hint] .venv/bin/python scripts/launch_lane_lightning.py status --session-id {session_id}")
    print(f"[harvest-hint] .venv/bin/python scripts/launch_lane_lightning.py harvest --session-id {session_id} --local-dir {lane_dir}/harvested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
