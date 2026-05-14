#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Canonical Vast.ai deployment — ONE command, profile-driven, tmux-managed.

Replaces all ad-hoc launch_*.sh scripts. Enforces:
  - All experiments via pipeline.py --profile (no CLI overrides)
  - Remote process via tmux (survives SSH disconnects, NOT nohup)
  - Preflight: profile exists, instance reachable, tmux available, code synced
  - Postflight: tmux session alive, training process running, watcher armed

Usage:
    python scripts/deploy_vastai.py --profile shiraz_v2 --instance-id 35562151
    python scripts/deploy_vastai.py --profile shiraz_v2 --offer-id 28665084 --create
    python scripts/deploy_vastai.py --status   # check all instances
    python scripts/deploy_vastai.py --kill 35562151

The pipeline runs:
    cd /workspace/pact
    python experiments/pipeline.py --profile <profile> --device cuda --output-dir results/<profile>

inside a named tmux session. Self-healing: if the tmux session dies, this script
reports it on next --status check.
"""
from __future__ import annotations
import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_PY = REPO_ROOT / "src" / "tac" / "profiles.py"
PIPELINE_PY = REPO_ROOT / "experiments" / "pipeline.py"


def vastai(*args) -> str:
    """Run vastai CLI via venv binary."""
    cmd = [str(REPO_ROOT / ".venv" / "bin" / "vastai"), *args]
    return subprocess.check_output(cmd, text=True)


def get_instance(instance_id: int) -> dict:
    """Fetch instance metadata. Raises if not found or not running."""
    raw = vastai("show", "instances", "--raw")
    data = json.loads(raw)
    for d in data:
        if d["id"] == instance_id:
            return d
    raise RuntimeError(f"Instance {instance_id} not found")


def ssh_url(instance_id: int) -> tuple[str, int]:
    """Get (host, port) from vastai ssh-url (uses direct IP, not proxy)."""
    url = vastai("ssh-url", str(instance_id)).strip()
    # ssh://root@HOST:PORT
    body = url.replace("ssh://root@", "")
    host, port = body.rsplit(":", 1)
    return host, int(port)


def ssh(host: str, port: int, cmd: str, timeout: int = 30,
        allow_remote_failure: bool = False) -> str:
    """Run a remote command via SSH, return stdout.

    R38 fix: prior version silently swallowed non-zero remote exit codes
    (capture_output without check, no returncode inspection). Every preflight
    "X in out" check could false-pass on SSH failure. Now raises RuntimeError
    on non-zero exit unless allow_remote_failure=True (for probes that
    legitimately expect non-zero like `which tmux` on missing binary).
    """
    full = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={min(timeout, 10)}",
        "-p", str(port), f"root@{host}", cmd,
    ]
    result = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 and not allow_remote_failure:
        raise RuntimeError(
            f"SSH command failed (exit {result.returncode}) on {host}:{port}\n"
            f"  cmd: {cmd[:120]}{'…' if len(cmd) > 120 else ''}\n"
            f"  stderr: {(result.stderr or '').strip()[:300]}"
        )
    return result.stdout


def preflight(profile: str, instance_id: int) -> None:
    """Validate everything BEFORE launching. Raise if anything is wrong.

    Runs FIVE layers of preflight via tac.preflight.preflight_all:
      1. check_codebase_drift: AST scan blocks ad-hoc patterns (no nohup, no launch_*.sh)
      2. preflight_arity: every subprocess.run flag matches target argparse signature
      3. preflight_profiles: every renderer profile has required arch keys + eval_roundtrip
      4. preflight_training_inputs: TTO range, profile arch, eval_roundtrip
      5. preflight_check: artifact validation if relevant
    Plus deployment-specific checks: profile registered, instance running,
    ssh+tmux ready, NO stale ad-hoc run_pipeline.sh on remote.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))

    # 1. Profile exists locally
    from tac.profiles import PROFILES
    if profile not in PROFILES:
        raise RuntimeError(f"Profile '{profile}' not in PROFILES. Available: {list(PROFILES.keys())}")
    profile_dict = PROFILES[profile]
    print(f"[preflight] OK profile '{profile}' registered")

    # 2. Run all-layers preflight (codebase drift + training inputs)
    # CLAUDE.md non-negotiable: TTO frames must be in TTO range, not GT range
    # (the WILDE failure mode). Don't silently skip when the artifact is
    # absent — refuse to deploy. (R26 finding.)
    from tac.preflight import preflight_all
    print(f"[preflight] Running tac.preflight.preflight_all ...")
    tto_path = REPO_ROOT / "experiments/results/tto_v7_hinge_500/tto_frames.pt"
    gt_poses = REPO_ROOT / "experiments/results/gt_poses.pt"
    masks = REPO_ROOT / "submissions/robust_current/masks_crf50.mkv"
    missing_artifacts = [name for name, p in [
        ("tto_frames", tto_path), ("gt_poses", gt_poses), ("masks", masks),
    ] if not p.exists()]
    if missing_artifacts:
        raise RuntimeError(
            f"Missing local training artifacts: {missing_artifacts}. "
            f"Cannot validate WILDE failure mode (TTO frames at GT range) without them. "
            f"Generate them locally before deploying, or pass --skip-input-preflight "
            f"only if you know the remote already has correct artifacts."
        )
    preflight_all(
        profile_name=profile,
        profile_arch=profile_dict,
        tto_frames_path=tto_path,
        gt_poses_path=gt_poses,
        masks_path=masks,
        check_codebase=True,
        verbose=True,
    )

    # 2. pipeline.py exists
    if not PIPELINE_PY.exists():
        raise RuntimeError(f"pipeline.py missing at {PIPELINE_PY}")
    print(f"[preflight] OK pipeline.py exists")

    # 3. Instance is running
    inst = get_instance(instance_id)
    if inst.get("actual_status") != "running":
        raise RuntimeError(f"Instance {instance_id} status={inst.get('actual_status')}")
    print(f"[preflight] OK instance {instance_id} running")

    # 4. SSH reachable
    host, port = ssh_url(instance_id)
    out = ssh(host, port, "echo OK && which tmux && which python3", timeout=15)
    if "OK" not in out:
        raise RuntimeError(f"SSH to {host}:{port} failed")
    if "tmux" not in out.lower():
        raise RuntimeError("tmux not installed on remote")
    if "python3" not in out:
        raise RuntimeError("python3 not installed on remote")
    print(f"[preflight] OK ssh + tmux + python3 ready on {host}:{port}")

    # 5. Repo on remote
    out = ssh(host, port, "ls /workspace/pact/experiments/pipeline.py", timeout=10)
    if "pipeline.py" not in out:
        raise RuntimeError("/workspace/pact/experiments/pipeline.py missing on remote — sync code first")
    print(f"[preflight] OK pipeline.py exists on remote")

    # 6. NO stale ad-hoc run_pipeline.sh on remote.
    # rsync excludes experiments/results/, so any shell script left in
    # results/ from a prior ad-hoc deploy will silently survive code syncs
    # and keep invoking stale CLI args. (Round 23 finding.)
    # Compound check emits a sentinel token so we can distinguish "no script
    # found" from "SSH failed" — a silent SSH failure would otherwise return
    # empty string and let the check pass. (Round 24 finding.)
    sentinel = "__PREFLIGHT_CHECK_OK__"
    cmd = (
        "ls /workspace/pact/experiments/results/*/run_pipeline.sh 2>/dev/null; "
        f"echo {sentinel}"
    )
    out = ssh(host, port, cmd, timeout=15)
    if sentinel not in out:
        raise RuntimeError(
            f"SSH check 6 (stale run_pipeline.sh) did not return sentinel — "
            f"SSH may have failed. Cannot certify remote state. Output: {out!r}"
        )
    stale = out.replace(sentinel, "").strip()
    if stale:
        raise RuntimeError(
            "Stale ad-hoc run_pipeline.sh found on remote — these survive "
            "code sync (results/ is rsync-excluded) and bypass the canonical "
            "pipeline.py. Remove them before deploying:\n"
            f"  {stale}\n"
            f"Run: ssh -p {port} root@{host} 'rm /workspace/pact/experiments/results/*/run_pipeline.sh'"
        )
    print(f"[preflight] OK no stale run_pipeline.sh on remote")

    # 7. Remote code parity (R-remote-parity 2026-04-25). Per CLAUDE.md
    # HIGHEST-EMPHASIS rule "Remote code parity required before launch":
    # SHIRAZ wasted 16h + $10 because the deployed auth_eval_renderer.py
    # had a NameError that we'd fixed locally that morning. The deploy
    # script must verify the critical files on remote match local HEAD.
    crit_files = [
        "experiments/auth_eval_renderer.py",
        "experiments/pipeline.py",
        "src/tac/preflight.py",
        "src/tac/submission_archive.py",
        "src/tac/experiments/train_renderer.py",
        "src/tac/fp4_quantize.py",
        "src/tac/losses.py",
        "src/tac/renderer_export.py",
    ]
    parity_violations = []
    for rel in crit_files:
        local = REPO_ROOT / rel
        if not local.exists():
            continue  # local missing — preflight 5 (artifact validation) catches
        local_md5 = subprocess.check_output(["md5", "-q", str(local)],
                                             stderr=subprocess.DEVNULL).decode().strip()
        # macOS has md5 -q; linux has md5sum
        if not local_md5:
            local_md5 = subprocess.check_output(["md5sum", str(local)]).decode().split()[0]
        remote_md5 = ssh(host, port,
                         f"md5sum /workspace/pact/{rel} 2>/dev/null | cut -d' ' -f1",
                         timeout=10).strip()
        if not remote_md5:
            parity_violations.append(f"  {rel}: MISSING on remote")
        elif remote_md5 != local_md5:
            parity_violations.append(f"  {rel}: REMOTE={remote_md5[:8]}.. LOCAL={local_md5[:8]}.. (DRIFT)")
    if parity_violations:
        raise RuntimeError(
            "Remote code parity check FAILED — critical files differ from local HEAD.\n"
            "This is the SHIRAZ-class failure mode that wasted 16h+$10 on 2026-04-25.\n"
            "Run sync_code() to fix, OR pass --allow-stale-remote to bypass (NOT recommended):\n"
            + "\n".join(parity_violations)
        )
    print(f"[preflight] OK remote code parity verified ({len(crit_files)} critical files)")


def sync_code(instance_id: int) -> None:
    """Rsync src + experiments + upstream to remote."""
    host, port = ssh_url(instance_id)
    print(f"[sync] Rsyncing code to {host}:{port}...")
    for src in ["src/", "experiments/", "upstream/"]:
        cmd = [
            "rsync", "-az", "--exclude", "__pycache__", "--exclude", "*.pyc",
            "--exclude", "experiments/results", "--exclude", "reports",
            "-e", f"ssh -o StrictHostKeyChecking=no -p {port}",
            f"{REPO_ROOT}/{src}",
            f"root@{host}:/workspace/pact/{src}",
        ]
        # R38 fix: surface rsync stderr on failure (was silenced by
        # capture_output=True; CalledProcessError stack trace gave no
        # actionable diagnostic).
        rsync_result = subprocess.run(cmd, capture_output=True, text=True)
        if rsync_result.returncode != 0:
            raise RuntimeError(
                f"rsync failed (exit {rsync_result.returncode}) syncing {src}\n"
                f"  stderr: {(rsync_result.stderr or '').strip()[:500]}"
            )
    print(f"[sync] OK")


def launch(profile: str, instance_id: int, sync: bool = True) -> str:
    """Start canonical pipeline.py in a tmux session on the remote.

    Returns the tmux session name.
    """
    preflight(profile, instance_id)
    if sync:
        sync_code(instance_id)

    host, port = ssh_url(instance_id)
    session_name = f"pact_{profile}"

    # Kill any existing session with this name
    ssh(host, port, f"tmux kill-session -t {shlex.quote(session_name)} 2>/dev/null || true")

    # Start canonical pipeline in tmux. pipeline.py uses subcommands —
    # `compress` is the experiment-running entry point. (R25 finding.)
    # R28 fix: the prior approach used `shlex.quote(pipeline_cmd)` which
    # wrapped the whole command in single quotes, suppressing every `$VAR`
    # and `$(...)` expansion → bash exec'd literal `$PYBIN` and failed.
    # Resolve the remote python path NOW (before tmux), then embed the
    # absolute path literally — no shell expansion needed at runtime.
    output_dir = f"experiments/results/{profile}"
    video = "/workspace/pact/upstream/videos/0.mkv"
    # R38+R39 fix: probe remote for actual checkpoint name with whitelist
    # protection. Prior version (R38) hardcoded distill_phase3_best.pt and
    # then `ls -t | head -1` would silently pick the most-recent .pt — could
    # be a stale partial save from a failed Fridrich/QAT run. R39 restricts
    # to canonical training-output basenames + logs the choice loudly.
    # 2026-04-26 hardening: canonical name registry now lives in
    # tac.checkpoint_names so deploy_vastai, remote_train_bootstrap.sh,
    # and preflight all share ONE source of truth. Adding a new training
    # script that emits a different name? Update the registry, not here.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tac.checkpoint_names import canonical_checkpoint_names
    CANONICAL_CHECKPOINT_NAMES = canonical_checkpoint_names(profile=profile)
    # Try each canonical name in priority order; first hit wins.
    checkpoint_glob = None
    for name in CANONICAL_CHECKPOINT_NAMES:
        candidate = f"/workspace/pact/{output_dir}/{name}"
        check = ssh(host, port, f"test -f {shlex.quote(candidate)} && echo OK || echo MISS",
                    timeout=10, allow_remote_failure=False).strip()
        if "OK" in check:
            checkpoint_glob = candidate
            print(f"[preflight] OK probed checkpoint: {candidate}")
            break
    if checkpoint_glob is None:
        raise RuntimeError(
            f"No canonical checkpoint found in /workspace/pact/{output_dir}/ on remote.\n"
            f"  Expected one of: {', '.join(CANONICAL_CHECKPOINT_NAMES)}\n"
            f"  Training must complete and save under a canonical name before deploy.\n"
            f"  If the actual filename differs, add it to CANONICAL_CHECKPOINT_NAMES."
        )
    # R39 fix: pass allow_remote_failure=True since `test -x` exits non-zero
    # when the venv path doesn't exist. The `|| echo SYS` branch makes the
    # outer command exit 0, but a future simplification could remove that
    # branch and silently start crashing in preflight.
    venv_check = ssh(host, port,
                     "test -x /workspace/pact/.venv/bin/python3 && echo VENV || echo SYS",
                     timeout=15, allow_remote_failure=True).strip()
    if "VENV" in venv_check:
        python_bin = "/workspace/pact/.venv/bin/python3"
    elif "SYS" in venv_check:
        python_bin = "python3"
    else:
        raise RuntimeError(
            f"Cannot determine remote python — SSH probe returned {venv_check!r}. "
            f"SSH may have failed."
        )
    pipeline_cmd = (
        f"cd /workspace/pact && "
        f"PYTHONPATH=/workspace/pact/src:/workspace/pact/upstream:/workspace/pact "
        f"{python_bin} -u experiments/pipeline.py compress "
        f"--profile {shlex.quote(profile)} "
        f"--video {shlex.quote(video)} "
        f"--checkpoint {shlex.quote(checkpoint_glob)} "
        f"--device cuda "
        f"--output-dir {shlex.quote(output_dir)} "
        f"2>&1 | tee {output_dir}/deploy.log"
    )
    # pipeline_cmd contains no single quotes (shlex.quote safe args + plain
    # paths). Wrap in literal single quotes for bash -c so no further
    # expansion happens to our embedded args.
    if "'" in pipeline_cmd:
        raise RuntimeError(
            f"pipeline_cmd contains a single quote — would break bash -c wrapping. "
            f"Profile or path contains a special char that shlex.quote escaped: "
            f"{pipeline_cmd!r}"
        )
    tmux_cmd = (
        f"mkdir -p /workspace/pact/{output_dir} && "
        f"tmux new-session -d -s {shlex.quote(session_name)} -- "
        f"bash -c '{pipeline_cmd}'"
    )
    ssh(host, port, tmux_cmd)

    # Postflight: verify tmux session is alive and process is running
    time.sleep(2)
    # tmux ls and pgrep both exit non-zero on no-match; tolerate that.
    sessions = ssh(host, port, "tmux ls 2>/dev/null", timeout=10,
                   allow_remote_failure=True)
    if session_name not in sessions:
        raise RuntimeError(f"tmux session {session_name} did not start. Output: {sessions}")
    procs = ssh(host, port, "pgrep -af pipeline.py", timeout=10,
                allow_remote_failure=True)
    if "pipeline.py" not in procs:
        raise RuntimeError(f"pipeline.py not running. tmux says: {sessions}")
    print(f"[launch] OK tmux session '{session_name}' running pipeline.py --profile {profile}")
    print(f"[launch] OK pipeline.py PID(s): {procs.strip()}")
    print(f"[launch] To monitor: ssh -p {port} root@{host} 'tmux attach -t {session_name}'")
    return session_name


def status() -> None:
    """Report status of all running instances."""
    raw = vastai("show", "instances", "--raw")
    data = json.loads(raw)
    if not data:
        print("No instances running")
        return
    for d in data:
        iid = d["id"]
        try:
            host, port = ssh_url(iid)
            sessions = ssh(host, port, "tmux ls 2>/dev/null || echo no-tmux", timeout=10).strip()
            procs = ssh(host, port, "pgrep -af pipeline.py | head -1 || echo no-pipeline", timeout=10).strip()
            print(f"#{iid} {d.get('gpu_name','?')} ${d.get('dph_total',0):.3f}/hr at {host}:{port}")
            print(f"  tmux: {sessions}")
            print(f"  pipeline: {procs[:120]}")
        except Exception as e:
            print(f"#{iid} ERROR: {e}")


def kill(instance_id: int) -> None:
    """Destroy a Vast.ai instance."""
    print(f"Destroying instance {instance_id}...")
    proc = subprocess.run(  # subprocess-no-check-OK: best-effort destroy — already-destroyed instances exit non-zero but that's OK; stdout/stderr printed so operator sees outcome
        [str(REPO_ROOT / ".venv" / "bin" / "vastai"), "destroy", "instance", str(instance_id)],
        input="y\n", capture_output=True, text=True,
    )
    print(proc.stdout)
    print(proc.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", help="Profile name from src/tac/profiles.py")
    p.add_argument("--instance-id", type=int, help="Existing Vast.ai instance ID")
    p.add_argument("--no-sync", action="store_true", help="Skip code sync to remote")
    p.add_argument("--status", action="store_true", help="Show all instance status")
    p.add_argument("--kill", type=int, metavar="INSTANCE_ID", help="Destroy instance")
    args = p.parse_args()

    if args.status:
        status(); return
    if args.kill:
        kill(args.kill); return
    if args.profile and args.instance_id:
        launch(args.profile, args.instance_id, sync=not args.no_sync); return
    p.print_help()


if __name__ == "__main__":
    main()
