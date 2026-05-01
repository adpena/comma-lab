"""Lightning AI Studio dispatch - persistent SSH + tmux + SCP harvest.

This is the legacy/manual Lightning path. Official Lightning Batch Jobs should
be preferred for new auditable lane/eval queues because they provide snapshots
and job artifact directories. Keep this dispatcher for emergency/manual Studio
work and for harvesting existing tmux sessions.

Why this exists (2026-04-30):
- User activated Lightning AI Pro account ($240 annual credits, $20/mo
  effective via annual plan; H100 access at $3.50/hr).
- Lane 17 IMP 10-cycle on Vast.ai 4090 ETA ~80h; on Lightning H100 ~25h.
- Wall-clock priority mandate (per
  feedback_priority_time_to_floor_with_final_approval_20260430.md) makes
  H100 the right device for IMP and other multi-hour training jobs.

Architecture vs Modal vs Vast.ai:
- Modal: ephemeral container per call; artifacts live in FunctionCall result
  cache (24h TTL); spawn-and-harvest pattern with `experiments/modal_train_lane.py`.
- Vast.ai: per-lane spin-up; ssh + tmux on bare PyTorch container; ~85%
  NVDEC-bad-host roulette mitigated via `scripts/launch_lane_with_retry.py`.
- Lightning: PERSISTENT Studio; the workspace at
  /teamspace/studios/this_studio/<repo> survives across sessions; the GPU
  tier (T4 / L40S / A100 / H100) is a per-Studio setting changed via the
  Lightning UI or `lightning_sdk` Python API. Multiple Studios can run in
  parallel.

Lane lifecycle on Lightning:
1. dispatch_lane(): SSH in, write env.sh, launch lane script in tmux session,
   record session_id + start_time + provenance to `.omx/state/lightning_active_sessions.json`
2. poll_status(session_id): SSH in, check tmux session live + tail log
3. harvest(session_id, local_dir): SCP artifacts from remote workspace into
   local_dir; read lane-local contest_auth_eval.json when present
4. tear_down(session_id): kill the tmux session (Studio stays up; this just
   ends the lane)

Strategic Secrecy (CLAUDE.md): the Studio URL itself is not shared publicly
until the user approves disclosure. This module never logs or prints the
Studio URL to stdout — it operates by SSH credentials only.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
LIGHTNING_STATE = REPO_ROOT / ".omx/state/lightning_active_sessions.json"


@dataclasses.dataclass
class DispatchResult:
    """Returned by dispatch_lane()."""

    session_id: str
    label: str
    lane_script: str
    remote_log_path: str
    remote_workspace: str
    started_at_utc: str
    gpu_tier_observed: str
    env_overrides: dict[str, str]


class LightningDispatcher:
    """SSH-based dispatcher for Lightning AI Studios.

    Studios are persistent so this class operates in a "ssh in, do stuff,
    come back later" model — there's no spin-up cost per lane.
    """

    def __init__(
        self,
        *,
        ssh_user: str | None = None,
        ssh_host: str = "ssh.lightning.ai",
        ssh_target: str | None = None,
        remote_workspace: str = "/teamspace/studios/this_studio/pact",
        ssh_key: str | None = None,
        connect_timeout: int = 15,
    ) -> None:
        if ssh_target is not None and ssh_user:
            raise ValueError("pass either ssh_target or ssh_user/ssh_host, not both")
        if ssh_target is None:
            if not ssh_user:
                raise ValueError("ssh_user is required when ssh_target is not provided")
            ssh_target = f"{ssh_user}@{ssh_host}"
        self.ssh_user = ssh_user or ""
        self.ssh_host = ssh_host
        self.ssh_target = ssh_target
        self.remote_workspace = remote_workspace
        self.ssh_key = ssh_key  # Optional path to identity file
        self.connect_timeout = connect_timeout

    # ------------------------------------------------------------------
    # SSH plumbing
    # ------------------------------------------------------------------
    def _ssh_args(self) -> list[str]:
        args = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
        ]
        if self.ssh_key:
            args += ["-i", self.ssh_key]
        args += [self.ssh_target]
        return args

    def _scp_args(self, src: str, dst: str) -> list[str]:
        args = [
            "scp",
            "-r",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
        ]
        if self.ssh_key:
            args += ["-i", self.ssh_key]
        args += [src, dst]
        return args

    def _run_ssh(self, remote_cmd: str, timeout: int = 60) -> tuple[int, str, str]:
        """Run a remote command synchronously. Returns (rc, stdout, stderr)."""
        full = self._ssh_args() + [remote_cmd]
        proc = subprocess.run(
            full,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    # ------------------------------------------------------------------
    # GPU-tier introspection
    # ------------------------------------------------------------------
    def get_gpu_tier(self) -> str:
        """Query nvidia-smi on the Studio. Returns observed GPU name string."""
        rc, out, err = self._run_ssh(
            "nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1",
            timeout=30,
        )
        if rc != 0:
            return f"UNKNOWN (rc={rc}, err={err.strip()[:80]})"
        return out.strip().splitlines()[0] if out.strip() else "UNKNOWN (empty)"

    @staticmethod
    def _gpu_tier_normalize(name: str) -> str:
        """Normalize GPU name to a canonical tier string."""
        n = name.upper()
        if "H100" in n:
            return "H100"
        if "A100" in n:
            return "A100"
        if "L40" in n:
            return "L40S"
        if "L4" in n:
            return "L4"
        if "A10G" in n:
            return "A10G"
        if "T4" in n:
            return "T4"
        if "V100" in n:
            return "V100"
        return name

    # ------------------------------------------------------------------
    # Active-session state I/O
    # ------------------------------------------------------------------
    @classmethod
    def _load_state(cls) -> list[dict[str, Any]]:
        if not LIGHTNING_STATE.exists():
            return []
        try:
            return json.loads(LIGHTNING_STATE.read_text())
        except json.JSONDecodeError:
            return []

    @classmethod
    def _save_state(cls, sessions: list[dict[str, Any]]) -> None:
        LIGHTNING_STATE.parent.mkdir(parents=True, exist_ok=True)
        LIGHTNING_STATE.write_text(json.dumps(sessions, indent=2, sort_keys=True))

    @classmethod
    def register_session(cls, session_meta: dict[str, Any]) -> None:
        sessions = cls._load_state()
        sessions.append(session_meta)
        cls._save_state(sessions)

    @classmethod
    def list_sessions(cls) -> list[dict[str, Any]]:
        return cls._load_state()

    @classmethod
    def remove_session(cls, session_id: str) -> bool:
        sessions = cls._load_state()
        new = [s for s in sessions if s.get("session_id") != session_id]
        if len(new) == len(sessions):
            return False
        cls._save_state(new)
        return True

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def dispatch_lane(
        self,
        *,
        lane_script: str,
        label: str,
        gpu_tier_required: str | None = None,
        env_overrides: dict[str, str] | None = None,
        allow_gpu_mismatch: bool = False,
    ) -> DispatchResult:
        """Launch a remote_lane_*.sh script on the Studio inside a tmux session.

        Args:
            lane_script: path relative to repo root, e.g.
                'scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh'.
            label: short label used as tmux session name + log dir suffix.
            gpu_tier_required: 'H100', 'A100', 'L40S', 'A10G', 'T4', etc. If
                set, the dispatch raises RuntimeError when the Studio's
                attached GPU doesn't match (unless allow_gpu_mismatch=True).
            env_overrides: extra env vars to export before running the lane.
            allow_gpu_mismatch: silence the GPU-tier check (useful for smoke
                runs on T4 where the lane script accepts CPU/T4 fallback).

        Returns:
            DispatchResult with session metadata. Also written to
            `.omx/state/lightning_active_sessions.json`.

        Raises:
            RuntimeError: if SSH fails, the tmux session creation fails, or
            GPU tier doesn't match (unless allow_gpu_mismatch).
        """
        env_overrides = dict(env_overrides or {})

        # --- GPU tier check ------------------------------------------------
        observed = self.get_gpu_tier()
        observed_tier = self._gpu_tier_normalize(observed)
        if gpu_tier_required and not allow_gpu_mismatch:
            req = self._gpu_tier_normalize(gpu_tier_required)
            if req != observed_tier:
                raise RuntimeError(
                    f"GPU tier mismatch: required {req}, Studio has {observed_tier} "
                    f"(raw='{observed}'). Change Studio GPU via Lightning UI or "
                    f"pass allow_gpu_mismatch=True for fallback runs."
                )

        # --- tmux session id ----------------------------------------------
        session_id = f"lightning_{label}_{int(time.time())}"
        log_path = f"{self.remote_workspace}/lightning_dispatch_{label}.log"
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # --- pre-flight: ensure remote workspace + lane script exist ------
        rc, out, err = self._run_ssh(
            f"test -d {shlex.quote(self.remote_workspace)} && "
            f"test -f {shlex.quote(self.remote_workspace + '/' + lane_script)} && echo OK",
            timeout=30,
        )
        if rc != 0 or "OK" not in out:
            raise RuntimeError(
                f"remote workspace or lane script missing: "
                f"workspace={self.remote_workspace}, lane_script={lane_script}, "
                f"rc={rc}, err={err.strip()[:200]}"
            )

        # --- compose env exports ------------------------------------------
        env_lines = [
            f"export WORKSPACE={shlex.quote(self.remote_workspace)}",
            f"export TAC_UPSTREAM_DIR={shlex.quote(self.remote_workspace + '/upstream')}",
            f"export PYTHONPATH={shlex.quote(self.remote_workspace + '/src')}:"
            f"{shlex.quote(self.remote_workspace + '/upstream')}:{shlex.quote(self.remote_workspace)}",
            "export PYBIN=$WORKSPACE/.venv/bin/python",
            "export PYTHONUNBUFFERED=1",
        ]
        for key, val in env_overrides.items():
            env_lines.append(f"export {key}={shlex.quote(val)}")
        env_block = "\n".join(env_lines)

        # --- compose the launch command (tmux new-session detached) -------
        # Pattern: tmux new-session -d -s <session_id> 'bash -c "cd /workspace; <env>; bash <lane> >log 2>&1; echo DONE >> log"'
        # Wrap in a heredoc-friendly inner-script via printf so quoting stays sane.
        inner_script = (
            f"set -euo pipefail\n"
            f"cd {shlex.quote(self.remote_workspace)}\n"
            f"{env_block}\n"
            f"echo \"[lightning-dispatch] starting {label} at $(date -u +%FT%TZ)\" "
            f"| tee -a {shlex.quote(log_path)}\n"
            f"bash {shlex.quote(lane_script)} 2>&1 | tee -a {shlex.quote(log_path)}\n"
            f"RC=${{PIPESTATUS[0]}}\n"
            f"echo \"[lightning-dispatch] DONE rc=$RC at $(date -u +%FT%TZ)\" "
            f"| tee -a {shlex.quote(log_path)}\n"
            f"echo \"$RC\" > {shlex.quote(self.remote_workspace + '/lightning_dispatch_' + label + '.rc')}\n"
        )

        # SSH-side: write the inner script to a tmpfile, then tmux it.
        inner_path = f"/tmp/lightning_inner_{session_id}.sh"
        b64 = __import__("base64").b64encode(inner_script.encode()).decode()
        compose_cmd = (
            f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(inner_path)} "
            f"&& chmod +x {shlex.quote(inner_path)} "
            f"&& tmux new-session -d -s {shlex.quote(session_id)} "
            f"  'bash {shlex.quote(inner_path)}' "
            f"&& tmux ls 2>&1 | grep -F {shlex.quote(session_id)} > /dev/null "
            f"&& echo DISPATCHED"
        )

        rc, out, err = self._run_ssh(compose_cmd, timeout=60)
        if rc != 0 or "DISPATCHED" not in out:
            raise RuntimeError(
                f"failed to launch tmux session: rc={rc}, "
                f"stdout={out.strip()[:200]}, stderr={err.strip()[:200]}"
            )

        result = DispatchResult(
            session_id=session_id,
            label=label,
            lane_script=lane_script,
            remote_log_path=log_path,
            remote_workspace=self.remote_workspace,
            started_at_utc=started_at,
            gpu_tier_observed=observed_tier,
            env_overrides=env_overrides,
        )
        self.register_session(
            {
                "session_id": session_id,
                "label": label,
                "lane_script": lane_script,
                "remote_log_path": log_path,
                "remote_workspace": self.remote_workspace,
                "ssh_user": self.ssh_user,
                "ssh_host": self.ssh_host,
                "ssh_target": self.ssh_target,
                "started_at_utc": started_at,
                "gpu_tier_observed": observed_tier,
                "env_overrides": env_overrides,
                "status": "running",
            }
        )
        return result

    # ------------------------------------------------------------------
    # Status + harvest
    # ------------------------------------------------------------------
    def poll_status(self, session_id: str) -> dict[str, Any]:
        """Check a tmux session's liveness + tail the log."""
        rc_alive, out_alive, _ = self._run_ssh(
            f"tmux has-session -t {shlex.quote(session_id)} 2>/dev/null && "
            f"echo ALIVE || echo DEAD",
            timeout=20,
        )
        alive = "ALIVE" in out_alive

        # Find the matching session metadata for log path / rc file
        sessions = self.list_sessions()
        meta = next((s for s in sessions if s.get("session_id") == session_id), None)
        if meta is None:
            return {"alive": alive, "session_id": session_id, "meta_missing": True}

        # Tail log + check rc
        log_path = meta["remote_log_path"]
        label = meta["label"]
        rc_file = f"{meta['remote_workspace']}/lightning_dispatch_{label}.rc"
        rc_log, out_log, _ = self._run_ssh(
            f"tail -50 {shlex.quote(log_path)} 2>/dev/null; echo '--RC--'; "
            f"cat {shlex.quote(rc_file)} 2>/dev/null || echo NONE",
            timeout=30,
        )
        log_tail = ""
        rc_value: int | None = None
        if rc_log == 0:
            parts = out_log.split("--RC--")
            log_tail = parts[0]
            if len(parts) > 1:
                rc_str = parts[1].strip()
                if rc_str and rc_str != "NONE":
                    try:
                        rc_value = int(rc_str)
                    except ValueError:
                        pass
        return {
            "session_id": session_id,
            "alive": alive,
            "log_tail": log_tail,
            "remote_rc": rc_value,
            "label": label,
        }

    def harvest(
        self,
        session_id: str,
        *,
        local_dir: Path | str,
        remote_subdir: str | None = None,
    ) -> dict[str, Any]:
        """SCP artifacts from the Studio to a local directory.

        Args:
            session_id: as returned by dispatch_lane().
            local_dir: where to write artifacts on the local filesystem.
            remote_subdir: relative to remote_workspace. If None, defaults
                to the lane's expected results dir based on lane label.

        Returns:
            dict with 'artifacts' (list of relative paths transferred) and,
            when present, score fields read from contest_auth_eval.json. Human
            logs are not parsed for score authority.
        """
        sessions = self.list_sessions()
        meta = next((s for s in sessions if s.get("session_id") == session_id), None)
        if meta is None:
            raise RuntimeError(f"unknown session_id: {session_id}")

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        if remote_subdir is None:
            # Heuristic: most lane scripts write to <workspace>/<lane>_results/
            # or <workspace>/results/<label>/. Try a glob-based search remotely.
            label = meta["label"]
            rc, out, _ = self._run_ssh(
                f"cd {shlex.quote(meta['remote_workspace'])} && "
                f"ls -d *{label}*_results 2>/dev/null | head -1",
                timeout=30,
            )
            if rc == 0 and out.strip():
                remote_subdir = out.strip().splitlines()[0]
            else:
                # Fall back to results/<label>
                remote_subdir = f"results/{label}"

        remote_full = f"{meta['remote_workspace']}/{remote_subdir}"
        # Verify it exists
        rc, _, err = self._run_ssh(
            f"test -d {shlex.quote(remote_full)} && echo OK", timeout=20
        )
        if rc != 0:
            return {
                "artifacts": [],
                "auth_eval_score": None,
                "error": f"remote dir missing: {remote_full}",
            }

        # SCP recursively
        src = f"{self.ssh_target}:{remote_full}"
        dst = str(local_dir)
        scp_args = self._scp_args(src, dst)
        proc = subprocess.run(scp_args, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            return {
                "artifacts": [],
                "auth_eval_score": None,
                "error": f"scp failed: rc={proc.returncode}, "
                f"stderr={proc.stderr[:200]}",
            }

        # Walk local copy, collect artifact list + parse authoritative JSON.
        copied = []
        score: float | None = None
        result_json_relpath: str | None = None
        result_json: dict[str, Any] | None = None
        copy_root = local_dir / Path(remote_full).name
        if copy_root.exists():
            for fp in copy_root.rglob("*"):
                if fp.is_file():
                    copied.append(str(fp.relative_to(local_dir)))
                    if fp.name == "contest_auth_eval.json":
                        try:
                            parsed = json.loads(fp.read_text())
                        except (OSError, json.JSONDecodeError):
                            continue
                        result_json = parsed
                        result_json_relpath = str(fp.relative_to(local_dir))
                        value = (
                            parsed.get("score_recomputed_from_components")
                            or parsed.get("final_score")
                        )
                        if isinstance(value, (int, float)):
                            score = float(value)

        return {
            "artifacts": copied,
            "auth_eval_score": score,
            "contest_auth_eval_json": result_json_relpath,
            "contest_auth_eval": result_json,
            "remote_subdir": remote_subdir,
            "local_root": str(copy_root),
        }

    def tear_down(self, session_id: str) -> bool:
        """Kill the tmux session (Studio stays up).

        Returns True if a session was killed, False if it was already gone.
        """
        rc, out, _ = self._run_ssh(
            f"tmux kill-session -t {shlex.quote(session_id)} 2>&1 && echo KILLED",
            timeout=20,
        )
        killed = "KILLED" in out
        # Always remove from state (idempotent cleanup)
        self.remove_session(session_id)
        return killed
