#!/usr/bin/env python3
"""One-liner Lightning Studio dispatch for PR106-anchor stack lanes.

Wraps the 6-gate `scripts/launch_lightning_batch_job.py exact-eval` boilerplate
into a single-arg invocation. Encodes the canonical Lightning settings
discovered the hard way 2026-05-05 (see
`reference_lightning_studio_canonical_dispatch_recipe_20260505.md`).

Usage:
    .venv/bin/python tools/lightning_dispatch_pr106_stack.py \
        --lane apogee_int4 \
        --archive experiments/results/apogee_int4_repack_20260504_claude/apogee_int4_archive.zip \
        --predicted-low 0.155 --predicted-high 0.180

Optional:
    --inflate-sh PATH   (default: derived from --lane name)
    --skip-stage        (skip lightning_repro_workspace.py if already staged)
    --ssh-target        (default: s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai)
    --print-only        (print the resolved invocation without running)

Workflow:
    1. Stages workspace via lightning_repro_workspace.py (sync src/, experiments/,
       submissions/, scripts/, upstream/, tools/, pyproject.toml + the archive)
    2. Generates source manifest at experiments/results/lightning_batch/<job>/source_manifest.json
    3. Files dispatch claim with platform=lightning canonical
    4. Submits exact-eval via launch_lightning_batch_job.py with all 6 gates pre-filled
    5. Returns the job-name for harvest

Encodes:
    - studio=lossy-compression-challenge, teamspace=comma-lab, user=adpena
    - platform=lightning (claim must be exact lowercase)
    - INFLATE_TORCH_SPEC=torch==2.5.1+cu124 (driver<580 cu13 trap)
    - --allow-skip-remote-preflight-reason for the launcher path-lowercase bug
    - PR106 baseline anchor (0.20945673 / 186239 bytes)

Failure modes handled:
    - Lightning SSH unreachable → fail loud
    - Already-staged workspace → reuse manifest
    - Existing terminal claim → re-file with --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SSH_TARGET = "s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai"
DEFAULT_REMOTE_PACT = "/teamspace/studios/this_studio/pact"
PR106_BASELINE_SCORE = 0.20945673
PR106_BASELINE_BYTES = 186239
LIGHTNING_STUDIO = "lossy-compression-challenge"
LIGHTNING_TEAMSPACE = "comma-lab"
LIGHTNING_USER = "adpena"
INFLATE_TORCH_SPEC = "torch==2.5.1+cu124"
UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
UV_INDEX_STRATEGY = "unsafe-best-match"


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_plus_1h_iso() -> str:
    return (dt.datetime.now(tz=dt.timezone.utc) + dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_name(lane: str) -> str:
    ts = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"lane_{lane}_pr106_{ts}"


def stage_workspace(*, lane: str, job_name: str, archive: Path,
                    ssh_target: str, remote_pact: str) -> Path:
    manifest_dir = REPO_ROOT / "experiments" / "results" / "lightning_batch" / job_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = manifest_dir / "source_manifest.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "lightning_repro_workspace.py"),
        "--remote", ssh_target,
        "--remote-pact", remote_pact,
        "--run-id", job_name,
        "--manifest-out", str(manifest_out),
        "--source", "src",
        "--source", "experiments",
        "--source", "submissions",
        "--source", "scripts",
        "--source", "upstream",
        "--source", "tools",
        "--source", "pyproject.toml",
        "--artifact", str(archive),
        "--requirements-mode", "no-install",
        "--no-install",
        "--ssh-connect-timeout", "30",
    ]
    print(f"[stage] {' '.join(cmd[:6])} ... ({len(cmd)} args)")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        sys.exit(f"FATAL: lightning_repro_workspace.py failed (rc={result.returncode})")
    return manifest_out


def claim_lane(*, lane: str, job_name: str) -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"), "claim",
        "--lane-id", f"lane_{lane}",
        "--agent", "claude_lab",
        "--platform", "lightning",
        "--instance-job-id", job_name,
        "--predicted-eta-utc", _utc_plus_1h_iso(),
        "--status", "active_dispatching",
        "--notes", f"canonical Lightning dispatch via tools/lightning_dispatch_pr106_stack.py {_utc_now_iso()}",
        "--force",
    ]
    print(f"[claim] platform=lightning lane=lane_{lane} job={job_name}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        sys.exit(f"FATAL: claim_lane_dispatch.py failed (rc={result.returncode})")


def submit_dispatch(*, lane: str, job_name: str, archive: Path, manifest: Path,
                    inflate_sh: Path, predicted_low: float, predicted_high: float,
                    ssh_target: str, machine: str, print_only: bool) -> int:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "launch_lightning_batch_job.py"), "exact-eval",
        "--job-name", job_name,
        "--archive", str(archive),
        "--repo-dir", str(REPO_ROOT),
        "--upstream-dir", str(REPO_ROOT / "upstream"),
        "--teamspace", LIGHTNING_TEAMSPACE,
        "--studio", LIGHTNING_STUDIO,
        "--user", LIGHTNING_USER,
        "--machine", machine,
        "--inflate-sh", str(inflate_sh),
        "--predicted-band", str(predicted_low), str(predicted_high),
        "--baseline-score", str(PR106_BASELINE_SCORE),
        "--baseline-archive-bytes", str(PR106_BASELINE_BYTES),
        "--infer-expected-archive",
        "--adjudicate",
        "--regression-threshold", "0.05",
        "--dispatch-lane-id", f"lane_{lane}",
        "--source-manifest", str(manifest),
        # NOTE: do NOT pass --remote-preflight-ssh-target here.
        # The launcher's _run_remote_supply_chain_preflight runs unconditionally
        # when ssh-target is set AND lowercases the local repo path which fails
        # on macOS Mixed-Case repos. The skip-reason short-circuits the SHAPE
        # check but not the actual preflight execution. Cleanest path is to
        # not provide the ssh-target and rely on lightning_repro_workspace
        # having already done the supply-chain verification.
        "--allow-skip-remote-preflight-reason",
            "manually-staged-via-lightning_repro_workspace.py pre-submit "
            "(launcher path-lowercase bug + redundant preflight)",
        "--env", f"INFLATE_TORCH_SPEC={INFLATE_TORCH_SPEC}",
        "--env", f"UV_EXTRA_INDEX_URL={UV_EXTRA_INDEX_URL}",
        "--env", f"UV_INDEX_STRATEGY={UV_INDEX_STRATEGY}",
    ]
    if print_only:
        print("=== resolved invocation (would run) ===")
        print(" \\\n  ".join(cmd))
        return 0
    print(f"[submit] launching Lightning exact-eval for {job_name}...")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True,
                        help="lane id (e.g. apogee_int4, pr106_latent_sidecar)")
    parser.add_argument("--archive", required=True, type=Path,
                        help="path to archive.zip (must be inside repo root)")
    parser.add_argument("--predicted-low", required=True, type=float)
    parser.add_argument("--predicted-high", required=True, type=float)
    parser.add_argument("--inflate-sh", type=Path, default=None,
                        help="defaults to submissions/<derived-from-lane>/inflate.sh")
    parser.add_argument("--skip-stage", action="store_true",
                        help="skip lightning_repro_workspace.py (workspace already staged)")
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--remote-pact", default=DEFAULT_REMOTE_PACT)
    parser.add_argument("--machine", default="g4dn.2xlarge",
                        help="Lightning machine class (default g4dn.2xlarge — "
                             "AWS T4 instance class that codex used yesterday for "
                             "exact_eval_pr95_hnerv_muon_repacked etc. NOT 'T4' literal — "
                             "the abbreviation fails with 'accelerator T4 not found for AWS cluster').")
    parser.add_argument("--print-only", action="store_true",
                        help="print resolved invocation without running")
    parser.add_argument("--job-name", default=None,
                        help="override auto-generated job name")
    args = parser.parse_args(argv)

    if not args.archive.is_absolute():
        args.archive = REPO_ROOT / args.archive
    if not args.archive.is_file():
        sys.exit(f"FATAL: archive not found: {args.archive}")
    try:
        args.archive.relative_to(REPO_ROOT)
    except ValueError:
        sys.exit(f"FATAL: archive must be inside repo root: {args.archive}")

    job_name = args.job_name or _job_name(args.lane)

    if args.inflate_sh is None:
        # derive: apogee_int4 → submissions/apogee_intN/inflate.sh
        # pr106_latent_sidecar → submissions/pr106_latent_sidecar/inflate.sh
        if args.lane.startswith("apogee_int"):
            inflate_dir = "apogee_intN"
        else:
            inflate_dir = args.lane
        args.inflate_sh = REPO_ROOT / "submissions" / inflate_dir / "inflate.sh"
    if not args.inflate_sh.is_file():
        sys.exit(f"FATAL: inflate.sh not found: {args.inflate_sh}")

    if args.skip_stage:
        manifest = REPO_ROOT / "experiments" / "results" / "lightning_batch" / job_name / "source_manifest.json"
        if not manifest.is_file():
            sys.exit(f"FATAL: --skip-stage but manifest not found: {manifest}")
    else:
        manifest = stage_workspace(
            lane=args.lane,
            job_name=job_name,
            archive=args.archive,
            ssh_target=args.ssh_target,
            remote_pact=args.remote_pact,
        )

    claim_lane(lane=args.lane, job_name=job_name)

    return submit_dispatch(
        lane=args.lane,
        job_name=job_name,
        archive=args.archive,
        manifest=manifest,
        inflate_sh=args.inflate_sh,
        predicted_low=args.predicted_low,
        predicted_high=args.predicted_high,
        ssh_target=args.ssh_target,
        machine=args.machine,
        print_only=args.print_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
