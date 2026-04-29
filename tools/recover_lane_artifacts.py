"""Recover archive-relevant artifacts from a Vast.ai instance BEFORE destroying it.

PROBLEM (2026-04-28): Lane RM-d crashed at the auth-eval stage AFTER 3.5h of
training. The instance was destroyed by the canonical fail-loud path before any
artifact recovery, losing the trained renderer + masks + poses + checkpoints.
$1.16 wasted with nothing to show. Memory:
``feedback_artifact_recovery_canonical_workflow_20260428``.

CANONICAL WORKFLOW (this module enforces it):

    recovered = recover_artifacts(
        instance_id=12345,
        lane_label="lane_rm_d",
        ssh_host="ssh.vast.ai", ssh_port=12345,
    )
    if recovered.archive_zip is not None:
        # We have a complete archive — can run auth eval locally / on Modal.
        # Operator can decide before destroy.
        ...

The launcher's ``destroy_instance()`` calls ``recover_before_destroy()`` first
(best-effort, with a 5-minute total timeout). On unreachable instances or
explicit ``--no-recover``, we skip recovery and destroy straight away.

USAGE (operator CLI):

    # Standalone recovery (no destroy):
    python tools/recover_lane_artifacts.py 12345 --lane-label lane_rm_d

    # Recovery + destroy after:
    python tools/recover_lane_artifacts.py 12345 --lane-label lane_rm_d --then-destroy

    # Force-skip recovery (instance unreachable):
    python tools/recover_lane_artifacts.py 12345 --no-recover --then-destroy

DESIGN NOTES:
* Idempotent — re-running just refreshes the recovery dir + metadata. The SCP
  layer overwrites by mtime; the metadata file is rewritten each run.
* Best-effort by design — every SSH/SCP call has a tight per-call timeout, and
  a single failure does not abort the whole recovery (we report what we got).
* Per-instance recovery dir: ``experiments/results/recovered_<instance>_<label>/``.
  The label is sanitized to keep filesystem characters only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Re-use launcher constants so behaviour stays consistent.
REPO_ROOT = Path(__file__).resolve().parent.parent
RECOVERY_BASE = REPO_ROOT / "experiments" / "results"
VASTAI = REPO_ROOT / ".venv/bin/vastai"

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "LogLevel=ERROR",
]

# Patterns we recover from the remote workspace. Each pattern is a single
# `find -path` glob fragment relative to /workspace. We deliberately scope to
# /workspace so we never recurse into /opt/conda or container guts.
RECOVERY_PATTERNS: tuple[str, ...] = (
    "renderer.bin",
    "renderer.bin.br",
    "renderer.bin.zst",
    "renderer.bin.gz",
    "masks.mkv",
    "optimized_poses.pt",
    "optimized_poses.pt.br",
    "archive.zip",
    "archive_*.zip",
    "*best*.pt",
    "*BEST*.pt",
    "best.ckpt",
    "checkpoint*.pt",
    "ckpt_*.pt",
    "run.log",
    "train.log",
    "setup.log",
    "heartbeat.log",
    "provenance.json",
    "report.txt",
    "RESULT_JSON*.json",
    "auth_eval*.json",
    "training.log",
)

# Per-call SSH/SCP timeouts (seconds). Recovery is best-effort; if the instance
# has died mid-recovery we want to surface that quickly, not block forever.
SSH_TIMEOUT_S = 30
SCP_TIMEOUT_S = 300  # 5 min per file batch — large checkpoints + masks
LIST_TIMEOUT_S = 60


@dataclass
class RecoveredArtifact:
    """One file pulled from the remote instance."""

    remote_path: str
    local_path: str
    size_bytes: int


@dataclass
class RecoveryReport:
    """Summary of what `recover_artifacts` collected."""

    instance_id: int
    lane_label: str
    recovery_dir: str
    started_at_utc: str
    elapsed_seconds: float
    ssh_reachable: bool
    artifacts: list[RecoveredArtifact] = field(default_factory=list)
    skipped_patterns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    archive_zip: str | None = None  # Path of the most-likely-canonical archive
    renderer_bin: str | None = None
    masks_mkv: str | None = None
    poses_pt: str | None = None

    def total_bytes(self) -> int:
        return sum(a.size_bytes for a in self.artifacts)

    def summary(self) -> str:
        lines = [
            f"=== Recovery report: instance={self.instance_id} label={self.lane_label} ===",
            f"  recovery_dir: {self.recovery_dir}",
            f"  ssh_reachable: {self.ssh_reachable}",
            f"  artifacts_found: {len(self.artifacts)}",
            f"  total_bytes: {self.total_bytes():,}",
            f"  elapsed: {self.elapsed_seconds:.1f}s",
        ]
        if self.archive_zip:
            lines.append(f"  archive_zip:  {self.archive_zip}")
        if self.renderer_bin:
            lines.append(f"  renderer_bin: {self.renderer_bin}")
        if self.masks_mkv:
            lines.append(f"  masks_mkv:    {self.masks_mkv}")
        if self.poses_pt:
            lines.append(f"  poses_pt:     {self.poses_pt}")
        if self.notes:
            lines.append("  notes:")
            for n in self.notes:
                lines.append(f"    - {n}")
        return "\n".join(lines)


def _sanitize_label(label: str) -> str:
    """Keep only filesystem-safe chars; collapse runs of underscores."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()) or "unlabeled"
    return safe.strip("_") or "unlabeled"


def _recovery_dir_for(instance_id: int, lane_label: str) -> Path:
    return RECOVERY_BASE / f"recovered_{int(instance_id)}_{_sanitize_label(lane_label)}"


def _run(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    """Run a subprocess; return (rc, stdout, stderr). Never raises on timeout."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return -2, "", f"command not found: {e}"


def _resolve_ssh_details(
    instance_id: int,
    explicit_host: str | None,
    explicit_port: int | None,
) -> tuple[str | None, int | None]:
    """Resolve (host, port). Prefer caller-provided values; else query vastai."""
    if explicit_host and explicit_port:
        return explicit_host, int(explicit_port)
    if not VASTAI.exists():
        return None, None
    rc, out, _ = _run(
        [str(VASTAI), "show", "instance", str(instance_id), "--raw"],
        timeout=SSH_TIMEOUT_S,
    )
    if rc != 0 or not out.strip():
        return None, None
    try:
        d = json.loads(out)
        d = d[0] if isinstance(d, list) else d
    except (json.JSONDecodeError, IndexError):
        return None, None
    host = d.get("ssh_host")
    port = d.get("ssh_port")
    if not host or not port:
        return None, None
    return str(host), int(port)


def _ssh_check(host: str, port: int) -> bool:
    """Cheap reachability probe."""
    cmd = [
        "ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
        "echo recover_ok",
    ]
    rc, out, _ = _run(cmd, timeout=SSH_TIMEOUT_S)
    return rc == 0 and "recover_ok" in out


def _list_remote_artifacts(host: str, port: int) -> list[tuple[str, int]]:
    """Find candidate artifact files on the remote.

    Returns list of (path, size_bytes). Uses a single `find` call so we don't
    spin up SSH per pattern. Caller filters and SCPs.
    """
    # Build a `find` expression: `\( -name X -o -name Y ... \) -type f -printf`
    name_clauses = " -o ".join(f"-name {shlex.quote(p)}" for p in RECOVERY_PATTERNS)
    # `-printf` returns "<size>\t<path>\n" so we can parse without re-statting.
    # Fall back to `stat` if `find -printf` is unavailable (BSD `find`).
    remote_cmd = (
        # Two candidate roots; we union results. /workspace is canonical;
        # /root is occasionally where pip installs land.
        "set -u; "
        "for ROOT in /workspace /root; do "
        "  [ -d \"$ROOT\" ] || continue; "
        f"  find \"$ROOT\" -maxdepth 6 \\( {name_clauses} \\) "
        "    -type f -printf '%s\\t%p\\n' 2>/dev/null || "
        f"    find \"$ROOT\" -maxdepth 6 \\( {name_clauses} \\) "
        "    -type f -exec stat -c '%s\\t%n' {} +; "
        "done"
    )
    cmd = [
        "ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", remote_cmd,
    ]
    rc, out, _err = _run(cmd, timeout=LIST_TIMEOUT_S)
    if rc != 0:
        return []
    results: list[tuple[str, int]] = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        size_s, path = line.split("\t", 1)
        try:
            size = int(size_s)
        except ValueError:
            continue
        results.append((path, size))
    # Dedup by path (a file under both /workspace and /root would appear twice
    # only if it really is in both places — keep both; SCP target is unique).
    seen: set[str] = set()
    deduped: list[tuple[str, int]] = []
    for p, s in results:
        if p in seen:
            continue
        seen.add(p)
        deduped.append((p, s))
    return deduped


def _scp_one(
    host: str, port: int, remote_path: str, local_path: Path,
) -> tuple[bool, str]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp", *SSH_OPTS, "-P", str(port),
        f"root@{host}:{remote_path}", str(local_path),
    ]
    rc, _out, err = _run(cmd, timeout=SCP_TIMEOUT_S)
    if rc != 0:
        return False, err.strip().splitlines()[-1] if err.strip() else f"scp rc={rc}"
    return True, ""


def _classify(local_paths: list[Path]) -> dict[str, str | None]:
    """Pick the most-likely-canonical archive / renderer / masks / poses.

    Heuristics:
      * archive_zip: largest non-zero archive*.zip.
      * renderer_bin: largest *renderer.bin*.
      * masks_mkv: any masks.mkv.
      * poses_pt: largest optimized_poses.pt*.
    """
    arch = [p for p in local_paths if p.name.endswith(".zip") and p.name.startswith("archive")]
    rend = [p for p in local_paths if "renderer.bin" in p.name]
    mask = [p for p in local_paths if p.name == "masks.mkv"]
    pose = [p for p in local_paths if p.name.startswith("optimized_poses.pt")]

    def _largest(paths: list[Path]) -> str | None:
        if not paths:
            return None
        try:
            return str(max(paths, key=lambda p: p.stat().st_size))
        except OSError:
            return None

    return {
        "archive_zip": _largest(arch),
        "renderer_bin": _largest(rend),
        "masks_mkv": str(mask[0]) if mask else None,
        "poses_pt": _largest(pose),
    }


def recover_artifacts(
    instance_id: int,
    lane_label: str,
    ssh_host: str | None = None,
    ssh_port: int | None = None,
    overall_timeout_s: int = 600,
) -> RecoveryReport:
    """Pull all archive-relevant artifacts from a Vast.ai instance.

    Best-effort: every individual SCP / SSH call has a tight per-call timeout;
    failures on individual files do not abort the whole pass. Returns a
    RecoveryReport with whatever we could collect plus any notes about failures.

    Args:
        instance_id: Vast.ai instance ID.
        lane_label: Human-readable label (used in recovery dir name).
        ssh_host, ssh_port: Optional explicit SSH details. If omitted, we query
          ``vastai show instance`` to resolve them.
        overall_timeout_s: Soft cap on the whole recovery pass (default 10 min).
    """
    t0 = time.monotonic()
    rec_dir = _recovery_dir_for(instance_id, lane_label)
    rec_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    report = RecoveryReport(
        instance_id=int(instance_id),
        lane_label=str(lane_label),
        recovery_dir=str(rec_dir),
        started_at_utc=started,
        elapsed_seconds=0.0,
        ssh_reachable=False,
    )

    host, port = _resolve_ssh_details(instance_id, ssh_host, ssh_port)
    if host is None or port is None:
        report.notes.append(
            "Could not resolve SSH details (vastai show failed or instance "
            "already gone). Skipping recovery."
        )
        report.elapsed_seconds = time.monotonic() - t0
        _write_report(rec_dir, report)
        return report

    if not _ssh_check(host, port):
        report.notes.append(
            f"SSH reachability probe to root@{host}:{port} failed. "
            "Instance may be unreachable. Skipping recovery."
        )
        report.elapsed_seconds = time.monotonic() - t0
        _write_report(rec_dir, report)
        return report

    report.ssh_reachable = True
    candidates = _list_remote_artifacts(host, port)
    if not candidates:
        report.notes.append(
            "Remote `find` returned no candidate files matching recovery "
            "patterns. (Either nothing was produced, or the workspace "
            "structure differs from /workspace and /root.)"
        )
        report.elapsed_seconds = time.monotonic() - t0
        _write_report(rec_dir, report)
        return report

    deadline = t0 + overall_timeout_s
    local_paths: list[Path] = []
    for remote_path, size in candidates:
        if time.monotonic() >= deadline:
            report.notes.append(
                f"Overall timeout {overall_timeout_s}s reached; "
                f"skipping remaining {len(candidates) - len(local_paths)} files."
            )
            break
        # Mirror the remote path under the recovery dir so two files with the
        # same basename in different dirs don't collide.
        rel = remote_path.lstrip("/")
        local_path = rec_dir / rel
        ok, err = _scp_one(host, port, remote_path, local_path)
        if not ok:
            report.notes.append(f"scp failed: {remote_path}: {err}")
            report.skipped_patterns.append(remote_path)
            continue
        if local_path.exists():
            try:
                actual = local_path.stat().st_size
            except OSError:
                actual = size
            report.artifacts.append(RecoveredArtifact(
                remote_path=remote_path,
                local_path=str(local_path),
                size_bytes=int(actual),
            ))
            local_paths.append(local_path)

    classified = _classify(local_paths)
    report.archive_zip = classified["archive_zip"]
    report.renderer_bin = classified["renderer_bin"]
    report.masks_mkv = classified["masks_mkv"]
    report.poses_pt = classified["poses_pt"]
    report.elapsed_seconds = time.monotonic() - t0
    _write_report(rec_dir, report)
    return report


def _write_report(rec_dir: Path, report: RecoveryReport) -> None:
    rec_dir.mkdir(parents=True, exist_ok=True)
    meta_path = rec_dir / "recovery_metadata.json"
    payload = asdict(report)
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def recover_before_destroy(
    instance_id: int,
    lane_label: str,
    *,
    enabled: bool = True,
    ssh_host: str | None = None,
    ssh_port: int | None = None,
    overall_timeout_s: int = 600,
) -> RecoveryReport | None:
    """Wrapper used by the launcher's destroy path.

    Returns the RecoveryReport on success, or None if recovery was disabled
    via ``enabled=False`` (the ``--no-recover`` operator opt-out).

    NEVER raises — destroying a Vast.ai instance must not be blocked by an
    unrelated recovery hiccup. Errors are surfaced via the report's ``notes``.
    """
    if not enabled:
        return None
    try:
        return recover_artifacts(
            instance_id=instance_id,
            lane_label=lane_label,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            overall_timeout_s=overall_timeout_s,
        )
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as e:
        # Best-effort: log + swallow so destroy can proceed. The destroy path
        # MUST NOT be blocked by an unrelated recovery hiccup (Lane RM-d
        # incident). RuntimeError is included because some downstream
        # subprocess wrappers raise it on auth / DNS / quota issues.
        sys.stderr.write(
            f"[recover_lane_artifacts] non-fatal error during recovery: {e}\n"
        )
        return None


def _cli_destroy(instance_id: int) -> int:
    """Best-effort destroy via the vastai CLI. Returns subprocess rc."""
    if not VASTAI.exists():
        sys.stderr.write(
            "[recover_lane_artifacts] vastai CLI not found; skipping destroy.\n"
        )
        return -2
    cmd = [
        "bash", "-c",
        f"echo y | {shlex.quote(str(VASTAI))} destroy instance {instance_id}",
    ]
    rc, _out, _err = _run(cmd, timeout=60)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Recover archive-relevant artifacts from a Vast.ai instance "
            "BEFORE destroying it. Idempotent. See module docstring."
        )
    )
    ap.add_argument("instance_id", type=int, help="Vast.ai instance ID")
    ap.add_argument(
        "--lane-label", default="unlabeled",
        help="Human-readable lane label (used in recovery dir name)",
    )
    ap.add_argument("--ssh-host", default=None, help="Override ssh_host")
    ap.add_argument("--ssh-port", type=int, default=None, help="Override ssh_port")
    ap.add_argument(
        "--no-recover", action="store_true",
        help="Skip artifact recovery (force-destroy mode)",
    )
    ap.add_argument(
        "--then-destroy", action="store_true",
        help="After recovery, destroy the instance via vastai CLI",
    )
    ap.add_argument(
        "--overall-timeout", type=int, default=600,
        help="Soft cap on the whole recovery pass (seconds, default 600)",
    )
    args = ap.parse_args(argv)

    if args.no_recover:
        print(f"[recover_lane_artifacts] --no-recover; skipping recovery.")
    else:
        report = recover_before_destroy(
            instance_id=args.instance_id,
            lane_label=args.lane_label,
            ssh_host=args.ssh_host,
            ssh_port=args.ssh_port,
            overall_timeout_s=args.overall_timeout,
        )
        if report is None:
            print("[recover_lane_artifacts] recovery returned None (disabled or fatal error)")
        else:
            print(report.summary())

    if args.then_destroy:
        print(f"[recover_lane_artifacts] destroying instance {args.instance_id}…")
        rc = _cli_destroy(args.instance_id)
        if rc != 0:
            sys.stderr.write(
                f"[recover_lane_artifacts] destroy returned rc={rc}; verify manually.\n"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
