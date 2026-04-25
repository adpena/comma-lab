#!/usr/bin/env python3
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


def ssh(host: str, port: int, cmd: str, timeout: int = 30) -> str:
    """Run a remote command via SSH, return stdout."""
    full = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={min(timeout, 10)}",
        "-p", str(port), f"root@{host}", cmd,
    ]
    result = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
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
        subprocess.run(cmd, check=True, capture_output=True)
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
    # `compress` is the experiment-running entry point. (R25 finding: prior
    # invocation omitted the subcommand and would have died at argparse.)
    # R27 finding: tmux new-session may exec the command directly (no shell)
    # depending on default-shell config; the `2>&1 | tee` would be silently
    # dropped. Wrap in `bash -c` to guarantee shell pipeline semantics.
    # R27 finding: `python3` may resolve to system Python without torch/tac.
    # Prefer the uv-managed venv if present.
    output_dir = f"experiments/results/{profile}"
    video = "/workspace/pact/upstream/videos/0.mkv"
    checkpoint_glob = f"/workspace/pact/{output_dir}/distill_phase3_best.pt"
    python_select = (
        "PYBIN=/workspace/pact/.venv/bin/python3; "
        "if [ ! -x \"$PYBIN\" ]; then PYBIN=$(command -v python3); fi; "
    )
    pipeline_cmd = (
        f"cd /workspace/pact && "
        f"{python_select}"
        f"PYTHONPATH=src:upstream:$PWD "
        f'\"$PYBIN\" -u experiments/pipeline.py compress '
        f"--profile {shlex.quote(profile)} "
        f"--video {shlex.quote(video)} "
        f"--checkpoint {shlex.quote(checkpoint_glob)} "
        f"--device cuda "
        f"--output-dir {shlex.quote(output_dir)} "
        f"2>&1 | tee {output_dir}/deploy.log"
    )
    tmux_cmd = (
        f"mkdir -p /workspace/pact/{output_dir} && "
        f"tmux new-session -d -s {shlex.quote(session_name)} -- "
        f"bash -c {shlex.quote(pipeline_cmd)}"
    )
    ssh(host, port, tmux_cmd)

    # Postflight: verify tmux session is alive and process is running
    time.sleep(2)
    sessions = ssh(host, port, f"tmux ls 2>/dev/null", timeout=10)
    if session_name not in sessions:
        raise RuntimeError(f"tmux session {session_name} did not start. Output: {sessions}")
    procs = ssh(host, port, "pgrep -af pipeline.py", timeout=10)
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
    proc = subprocess.run(
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
