#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Unified-memory OOM guard — NEVER let training OOM the machine, NEVER kill the
control plane.

Operator directive 2026-06-25 (binding, safety-critical): a naive/blind
fleet-launch OOM'd the M5 Max (128GB unified) and killed the machine + all work.
THREE binding constraints:
  (1) "always protect at least 30 GB" free — OOM must NEVER happen again;
  (2) vendor molt's extensive memory-guard tooling;
  (3) "it is actually a very nuanced problem ... must NEVER kill control plane
      claude or codex app or codex cli processes."

This is the canonical guard (sister of the "Local Disk/SSD fail-closed-if-no-
space" non-negotiable, applied to RAM/unified memory). It is DEFENSE-IN-DEPTH
with THREE independent layers:
  - launch-preflight (`--check`): refuse to START a job that would breach the
    30 GB floor (wired into tools/spawn_durable_daemon.py);
  - whole-machine watchdog (`--watch`): if available memory drops below the
    floor, shed the LARGEST *training arm* — and NEVER the control plane;
  - per-arm RSS cap (tools/safe_run.py): each arm SIGKILLs ITSELF if its own
    process-group RSS exceeds a cap (bounds each arm, not just the whole machine).

THE CONTROL-PLANE PROTECTION (the nuanced core). The watchdog kill-selector is
CUSTODY-GATED + belt-and-suspenders — a process is killable ONLY if it passes
ALL of these independent gates (custody FIRST, the PRIMARY gate):
  (0) EXPLICIT CUSTODY: it is in the durable-daemon registry — something WE
      launched/own — AND passes the IDENTITY gate (live pgid + live command
      match the recorded launch, defeating PID recycling). Per the operator +
      molt 3b1e49b18: kill ONLY what we have explicit custody of; NEVER kill
      anything outside custody, EVEN IF it is the largest RSS and matches a
      training pattern; AND
  (1) it is NOT the guard, NOT an ancestor of the guard; AND
  (2) it is NOT a host control-plane app (claude / codex / app-server /
      node_repl / Code Helper); AND
  (3) it has NO external host control-plane LINEAGE (a Codex/Claude-spawned
      shell/node/git helper is protected even under the guard); AND
  (4) it does NOT match the broad direct-kill denylist (ssh/tmux/shell/...); AND
  (5) its pgid is NOT in ``protected_process_group_ids``; AND
  (6) it is its OWN process-group leader (detached daemon — group-kill scope is
      exactly its subtree, never a shared shell job); AND
  (7) its command matches the training ALLOWLIST.
If NO killable training arm UNDER CUSTODY matches when memory is critical, the
guard LOGS A LOUD ALERT and KILLS NOTHING (better to alert than kill the control
plane).

Borrowed-substrate accounting: the control-plane token/executable sets,
launcher-command detection, ``ProcessSample``/ps-parsing,
``is_host_control_plane_process``, ``_ancestor_pids``, ``descendant_pids``,
``has_external_host_control_plane_lineage`` and ``protected_process_group_ids``
are VENDORED from molt's CUSTODY-GATED, control-plane-safe HEAD
(``/Users/adpena/Projects/molt/tools/memory_guard_core/process_model.py``,
3b1e49b18) and adapted to a self-contained dependency-free form (molt's original
is ~6000 LOC coupled to ``molt.dx``). NOTE: the earlier molt HEAD 353784a1a had
a bug class that killed Codex/host-control processes repeatedly; this guard is
derived from the FIXED 3b1e49b18 (commits f516406ed / 89c54bd92 / c311f8a38 /
1f2837098 / 95dceaa1b). The vm_stat ``available_gb`` 30 GB floor, the training
ALLOWLIST, the registry-as-custody-ledger gate, and the launch-preflight are
OURS (this repo's stopgap, hardened here).

Modes:
  --free
      Print available GB (free + reclaimable inactive) and strict-free GB, JSON.
  --check [--min-free-gb 30] [--projected-gb N]
      Exit 0 if (available - projected) >= min_free; else exit 3 (REFUSE).
      Launch preflight: refuse to start a job that would breach the floor.
  --watch [--min-free-gb 30] [--kill-pattern RE] [--interval 5] [--warn-margin 8]
      Loop forever: if available < min_free, SIGTERM the process-GROUP of the
      LARGEST-RSS process that passes the belt-and-suspenders selector above.
      If available < min_free + warn_margin, log a WARN (early signal).
      Logs every action to .omx/state/memory_guard.log.
  --select-victim-dry-run [--kill-pattern RE]
      Print (without killing) the victim the watchdog WOULD select right now —
      for auditing the control-plane safety on the live process table.

macOS-only (vm_stat). "available" = (Pages free + Pages inactive) * page_size —
inactive is reclaimable before swap pressure; a conservative 30 GB floor leaves
ample margin above the OOM cliff.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOG = _REPO_ROOT / ".omx" / "state" / "memory_guard.log"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_FREE_GB = 30.0
DEFAULT_WARN_MARGIN_GB = 8.0
DEFAULT_INTERVAL_SEC = 5.0

# POSITIVE ALLOWLIST (training arms only): a process is a kill CANDIDATE only if
# its command matches this regex. OURS — the contest training-arm scripts.
DEFAULT_TRAINING_KILL_PATTERN = (
    r"train_witness_realized_through_R"
    r"|train_witness"
    r"|witness_capstone_deepmath_smoke"
    r"|witness_capstone"
    r"|train_substrate_"
    r"|train_renderer"
)

# ---------------------------------------------------------------------------
# CONTROL-PLANE PROTECTION (VENDORED from molt 3b1e49b18 — the CUSTODY-GATED,
# control-plane-safe HEAD, NOT the earlier 353784a1a which had a bug class that
# killed Codex/host-control processes repeatedly; fixed by molt commits
# f516406ed "Protect Codex lineage behind explicit guard custody",
# 89c54bd92 "Gate single-PID guard termination by identity",
# c311f8a38/a8c83ec3f "Harden host/Codex control process custody",
# 1f2837098/95dceaa1b "Require explicit custody for termination").
#
# HOST_CONTROL_PLANE_TOKENS / _EXECUTABLE_NAMES classify the actual control-plane
# APPLICATIONS (claude / codex / app-server / node_repl / Code Helper). Their
# *descendants* are also protected via lineage (claude spawning Code Helper,
# codex spawning node_repl/git/shells). This set deliberately EXCLUDES
# init/launchd — every process descends from launchd, so classifying it as
# control-plane would protect the entire tree (nothing killable).
#
# LAUNCHER_NAMES + _host_control_plane_launcher_command catch `node .../@anthropic
# -ai/claude-code/cli.js`, `bash -c codex ...`, etc. — a launcher executing a
# control-plane arg IS control plane.
# ---------------------------------------------------------------------------
HOST_CONTROL_PLANE_TOKENS: tuple[str, ...] = (
    "/Applications/Codex.app/",
    "Codex.app/Contents/",
    "Codex (Renderer)",
    "Codex Helper",
    "OpenAI.Codex_",
    "/codex.app/",
    "\\app\\Codex.exe",
    "\\app\\resources\\codex.exe",
    "codex.cmd",
    "codex --",
    'codex.exe" app-server',
    "codex app-server",
    "codex-app-server",
    "codex-linux-sandbox",
    "codex-macos-sandbox",
    "codex-win32-sandbox",
    "codex.ps1",
    "codex_chronicle",
    "/.codex/",
    "/appdata/local/codex/",
    "/appdata/local/openai/codex/",
    "/appdata/local/temp/codex/",
    "/appdata/roaming/codex/",
    "/node_modules/@openai/codex/",
    "\\node_modules\\@openai\\codex\\",
    "@openai/codex",
    "/cua_node/bin/node_repl",
    "\\runtimes\\cua_node\\",
    "node_repl",
    "node_repl.exe",
    "/Applications/Claude.app/",
    "claude --",
    "\\claude.exe",
    "\\claude.cmd",
    "\\claude-code.exe",
    "\\node_modules\\@anthropic-ai\\claude-code\\",
    "Claude.app/Contents/",
    "/.claude/",
    "/appdata/local/temp/claude/",
    "@anthropic-ai/claude-code",
    "CLAUDE_PLUGIN_DATA=",
)
HOST_CONTROL_PLANE_EXECUTABLE_NAMES: frozenset[str] = frozenset(
    {
        "claude",
        "claude-code",
        "claude-code.exe",
        "claude.cmd",
        "claude.exe",
        "codex",
        "codex.appimage",
        "codex-cli",
        "codex-cli.exe",
        "codex.cmd",
        "codex.exe",
        "codex.ps1",
        "codex-app-server",
        "codex-linux-sandbox",
        "codex-macos-sandbox",
        "codex-win32-sandbox",
        "node_repl",
        "node_repl.exe",
    }
)
HOST_CONTROL_PLANE_ARG_EXECUTABLE_NAMES: frozenset[str] = (
    HOST_CONTROL_PLANE_EXECUTABLE_NAMES | frozenset({"claude.js", "codex.js"})
)
HOST_CONTROL_PLANE_LAUNCHER_NAMES: frozenset[str] = frozenset(
    {
        "bun", "bun.exe", "bash", "cmd", "cmd.exe", "deno", "deno.exe", "env",
        "fish", "node", "node.exe", "npm", "npm.cmd", "npx", "npx.cmd",
        "powershell", "powershell.exe", "pwsh", "pwsh.exe", "sh", "zsh",
    }
)
HOST_CONTROL_PLANE_LINEAGE_PROTECTED_EXECUTABLE_NAMES: frozenset[str] = (
    HOST_CONTROL_PLANE_LAUNCHER_NAMES
    | frozenset(
        {
            "conhost.exe", "git", "git.exe", "git-remote-https",
            "git-remote-https.exe", "openconsole.exe",
        }
    )
)

# BROAD direct-kill denylist (operator's explicit list: ssh/tmux/shells/guard).
# Used ONLY as an extra direct-kill refusal (layer b); does NOT propagate to
# descendants (would re-introduce the universal-launchd protection bug). Never
# contains "python" — training arms are python.
EXTRA_PROTECTED_TOKENS: tuple[str, ...] = (
    "/usr/bin/ssh", " ssh ", "sshd", "tmux", "/sbin/launchd", "WindowServer",
    "loginwindow", "memory_guard.py", "safe_run.py", "spawn_durable_daemon.py",
)
EXTRA_PROTECTED_EXECUTABLE_NAMES: frozenset[str] = frozenset(
    {"ssh", "sshd", "tmux", "launchd", "login", "-bash", "-zsh", "-sh",
     "bash", "zsh", "sh", "windowserver"}
)


# ---------------------------------------------------------------------------
# vm_stat available-memory floor (OURS). TWO metrics, deliberately:
#   * CONSERVATIVE (free + inactive) — used by the REFUSE launch-preflight.
#     Undercounts real availability (ignores purgeable + clean file cache), so
#     refusing is FAIL-CLOSED = SAFE (we never over-commit a launch).
#   * GENEROUS / reclaimable-aware (free + inactive + purgeable + file-backed)
#     — used ONLY by the SHED watchdog. Under macOS memory compression the
#     conservative metric can undercount, which would false-positive-shed a
#     custody training arm when memory is actually fine (M3 review finding: an
#     availability bug, never a control-plane kill, but wastes training). The
#     generous metric + debounce (see watch_loop) prevents that.
# ---------------------------------------------------------------------------
def _parse_vm_stat(out: str) -> dict[str, int]:
    """Parse vm_stat output into a {field: pages} dict (+ 'page_size')."""
    ps = 16384
    m = re.search(r"page size of (\d+)", out)
    if m:
        ps = int(m.group(1))
    fields: dict[str, int] = {"page_size": ps}

    def _grab(label: str) -> int:
        for line in out.splitlines():
            if line.startswith(label):
                mm = re.search(r"(\d+)", line.split(":", 1)[1])
                return int(mm.group(1)) if mm else 0
        return 0

    fields["free"] = _grab("Pages free")
    fields["inactive"] = _grab("Pages inactive")
    fields["purgeable"] = _grab("Pages purgeable")
    fields["file_backed"] = _grab("File-backed pages")
    return fields


def _vm_stat_fields() -> dict[str, int]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    return _parse_vm_stat(out)


def _vm_stat_bytes() -> tuple[float, float]:
    """Return (conservative_available_bytes, strict_free_bytes) from vm_stat."""
    f = _vm_stat_fields()
    ps = f["page_size"]
    available = (f["free"] + f["inactive"]) * ps
    return float(available), float(f["free"] * ps)


def available_gb() -> float:
    """CONSERVATIVE available GB (free + reclaimable inactive). Used by REFUSE."""
    avail, _ = _vm_stat_bytes()
    return avail / 1e9


def available_generous_gb() -> float:
    """GENEROUS / reclaimable-aware available GB (free + inactive + purgeable +
    clean file-backed cache). Used by the SHED watchdog ONLY (M3 fix) so memory
    compression cannot false-positive-shed a healthy training arm."""
    f = _vm_stat_fields()
    ps = f["page_size"]
    return (f["free"] + f["inactive"] + f["purgeable"] + f["file_backed"]) * ps / 1e9


def strict_free_gb() -> float:
    _, sf = _vm_stat_bytes()
    return sf / 1e9


# ---------------------------------------------------------------------------
# Process sampling (vendored from molt, trimmed)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessSample:
    pid: int
    ppid: int
    rss_kb: int
    command: str
    pgid: int | None = None


def parse_ps_table(text: str) -> dict[int, ProcessSample]:
    """Parse ``ps -axo pid=,ppid=,pgid=,rss=,command=`` output.

    Vendored/trimmed from molt ``parse_process_table`` (we drop the etime column
    — the OOM guard does not need process age). Falls back to a 4-column
    (no-pgid) parse if pgid is absent.
    """
    samples: dict[int, ProcessSample] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 4)
        pgid: int | None
        if len(parts) >= 5:
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
                pgid = int(parts[2])
                rss_kb = int(parts[3])
                command = parts[4]
            except ValueError:
                # legacy 4-col (no pgid): pid ppid rss command
                legacy = line.split(None, 3)
                if len(legacy) < 4:
                    continue
                try:
                    pid = int(legacy[0])
                    ppid = int(legacy[1])
                    rss_kb = int(legacy[2])
                except ValueError:
                    continue
                command = legacy[3]
                pgid = None
        else:
            legacy = line.split(None, 3)
            if len(legacy) < 4:
                continue
            try:
                pid = int(legacy[0])
                ppid = int(legacy[1])
                rss_kb = int(legacy[2])
            except ValueError:
                continue
            command = legacy[3]
            pgid = None
        samples[pid] = ProcessSample(
            pid=pid, ppid=ppid, rss_kb=rss_kb, command=command, pgid=pgid
        )
    return samples


def sample_processes() -> dict[int, ProcessSample]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,rss=,command="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    return parse_ps_table(result.stdout)


def _sample_pgid(sample: ProcessSample) -> int:
    return sample.pgid if sample.pgid is not None else sample.pid


# ---------------------------------------------------------------------------
# Control-plane classification (vendored from molt)
# ---------------------------------------------------------------------------
def _command_executable_name(command: str) -> str:
    text = command.strip()
    if not text:
        return ""
    if text[0] in {"'", '"'}:
        quote = text[0]
        end = text.find(quote, 1)
        token = text[1:end] if end > 0 else text[1:]
    else:
        token = text.split(None, 1)[0]
    return token.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _command_arg_executable_names(command: str) -> tuple[str, ...]:
    """Executable basenames of every whitespace/quote-delimited arg. Vendored."""
    names: list[str] = []
    for match in re.finditer(r'''(?:"([^"]+)"|'([^']+)'|(\S+))''', command.strip()):
        token = next(group for group in match.groups() if group is not None)
        normalized = token.replace("\\", "/").rstrip("/")
        name = normalized.rsplit("/", 1)[-1].casefold()
        if name:
            names.append(name)
    return tuple(names)


# Real installed-binary path shapes for codex/claude that the token list alone
# misses (HIGH review finding 1): the homebrew cask installs codex at
# /opt/homebrew/Caskroom/codex/<ver>/codex-aarch64-apple-darwin — basename
# `codex-aarch64-apple-darwin` (NOT a literal in the exec-name set) and path
# token `/caskroom/codex/` (not `/.codex/` or `/codex.app/`). These regexes
# (case-insensitive) close that escape: the cask dir AND the generic
# /codex/<version>/codex(-arch) binary shape, for both codex and claude.
_CONTROL_PLANE_PATH_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"/caskroom/(codex|claude)/", re.IGNORECASE),
    re.compile(r"/(codex|claude)/[^/]+/(codex|claude)(-[^/\s]*)?(\s|$|\")", re.IGNORECASE),
    re.compile(r"/(codex|claude)-[a-z0-9_]+-apple-darwin(\s|$|\")", re.IGNORECASE),
)


def _is_control_plane_exec_name(name: str) -> bool:
    """True iff the executable basename is a control-plane binary — the literal
    set OR a real cask/versioned form (``codex``, ``codex-aarch64-apple-darwin``,
    ``codex-cli``, ``claude``, ``claude-code``, ...). HIGH review finding 1: the
    real ``codex-aarch64-apple-darwin`` basename was not in the literal set."""
    name = name.casefold()
    if name in HOST_CONTROL_PLANE_EXECUTABLE_NAMES:
        return True
    for stem in ("codex", "claude"):
        # exact, or stem followed by a -/_/. separator (codex-*, claude-code, ...)
        if name == stem or name.startswith(stem + "-") or name.startswith(stem + "_") or name.startswith(stem + "."):
            return True
    return False


def _host_control_plane_launcher_command(command: str) -> bool:
    """True iff command is a launcher (bash/node/npm/...) running a control-plane
    arg, e.g. ``node .../@anthropic-ai/claude-code/cli.js`` or any launcher whose
    argv references a codex/claude path (HIGH review finding 1(c): the real entry
    is ``cli.js``, not ``claude.js``/``codex.js`` — match the path, not just the
    arg basename). Extends molt's launcher-command check."""
    names = _command_arg_executable_names(command)
    if not names or names[0] not in HOST_CONTROL_PLANE_LAUNCHER_NAMES:
        return False
    # (a) molt's original: a control-plane arg basename (claude.js/codex.js/...)
    if any(name in HOST_CONTROL_PLANE_ARG_EXECUTABLE_NAMES for name in names[1:]):
        return True
    # (b) the launcher's argv references a codex/claude install path / token.
    lowered = command.casefold()
    if any(rx.search(command) for rx in _CONTROL_PLANE_PATH_RES):
        return True
    return any(
        marker in lowered
        for marker in ("@anthropic-ai/claude", "@openai/codex", "/.codex/", "/.claude/", "claude-code")
    )


def is_host_control_plane_process(sample: ProcessSample) -> bool:
    """True iff the process IS a control-plane application (vendored from molt,
    hardened per HIGH review finding 1 for real cask/versioned codex/claude
    binaries + node cli.js launchers)."""
    command = sample.command.casefold()
    normalized_command = command.replace("\\", "/")
    return (
        any(
            token.casefold() in command
            or token.casefold().replace("\\", "/") in normalized_command
            for token in HOST_CONTROL_PLANE_TOKENS
        )
        or any(rx.search(sample.command) for rx in _CONTROL_PLANE_PATH_RES)
        or _is_control_plane_exec_name(_command_executable_name(sample.command))
        or _host_control_plane_launcher_command(sample.command)
    )


def _matches_extra_protected(command: str) -> bool:
    lowered = command.casefold()
    for token in EXTRA_PROTECTED_TOKENS:
        if token.casefold() in lowered:
            return True
    return _command_executable_name(command) in EXTRA_PROTECTED_EXECUTABLE_NAMES


def is_protected_command(command: str) -> bool:
    """BROAD direct-kill denylist (layer b): control-plane app OR ssh/tmux/shell/
    guard. Used as an extra refusal on top of the custody gate."""
    return is_host_control_plane_process(
        ProcessSample(pid=0, ppid=0, rss_kb=0, command=command)
    ) or _matches_extra_protected(command)


def _ancestor_pids(samples: Mapping[int, ProcessSample], pid: int | None) -> set[int]:
    """All PIDs on the parent chain of ``pid`` (inclusive). Vendored from molt."""
    if pid is None or pid <= 0:
        return set()
    ancestors: set[int] = set()
    current = pid
    while current > 0 and current not in ancestors:
        ancestors.add(current)
        sample = samples.get(current)
        if sample is None or sample.ppid <= 0 or sample.ppid == current:
            break
        current = sample.ppid
    return ancestors


def descendant_pids(samples: Mapping[int, ProcessSample], root_pid: int | None) -> set[int]:
    """All PIDs descended from ``root_pid`` (inclusive). Vendored from molt."""
    if root_pid is None or root_pid <= 0:
        return set()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for sample in samples.values():
            if sample.pid in descendants:
                continue
            if sample.ppid in descendants:
                descendants.add(sample.pid)
                changed = True
    return descendants


def _host_control_plane_ancestor_pids(
    samples: Mapping[int, ProcessSample], pid: int | None, *, include_self: bool = False
) -> set[int]:
    ancestors = _ancestor_pids(samples, pid)
    if not include_self and pid is not None:
        ancestors.discard(pid)
    return {
        a for a in ancestors
        if (s := samples.get(a)) is not None and is_host_control_plane_process(s)
    }


def has_external_host_control_plane_lineage(
    samples: Mapping[int, ProcessSample],
    pid: int | None,
    *,
    current_pid: int | None = None,
    include_self: bool = True,
    owned_pids: "set[int] | frozenset[int] | tuple[int, ...]" = (),
) -> bool:
    """Return True when ``pid`` belongs to PROTECTED host-control lineage.

    Vendored from molt 3b1e49b18 (the custody fix). Codex/Claude/app-server/
    renderer/node-repl processes are the operator control plane; their
    descendants are protected UNLESS the caller proves ownership via an explicit
    ``owned_pids`` set AND the pid is a descendant of the running guard AND its
    executable is not a protected launcher AND it is not itself control-plane.
    Being a descendant of the guard is NOT ownership by itself.
    """
    if pid is None or pid <= 0:
        return False
    sample = samples.get(pid)
    if sample is None:
        return False
    if not _host_control_plane_ancestor_pids(samples, pid, include_self=include_self):
        return False
    if pid not in owned_pids:
        return True
    if current_pid is None or current_pid <= 0:
        return True
    if pid not in descendant_pids(samples, current_pid):
        return True
    if _command_executable_name(sample.command) in HOST_CONTROL_PLANE_LINEAGE_PROTECTED_EXECUTABLE_NAMES:
        return True
    return is_host_control_plane_process(sample)


def protected_process_group_ids(
    samples: Mapping[int, ProcessSample],
    *,
    self_pid: int | None = None,
    self_pgid: int | None = None,
) -> set[int]:
    """Process-group ids that must NEVER be killed (vendored from molt 3b1e49b18).

    Protects: (1) the guard's own pgid; (2) the guard's ancestors; (3) every
    host control-plane process; (4) descendants of any host control-plane
    process EXCEPT descendants of the guard itself.
    """
    protected: set[int] = set()
    if self_pgid is not None and self_pgid > 0:
        protected.add(self_pgid)
    ancestor_ids = _ancestor_pids(samples, self_pid)
    self_descendant_ids = descendant_pids(samples, self_pid) if self_pid else set()
    host_control_plane_pids = {
        sample.pid for sample in samples.values() if is_host_control_plane_process(sample)
    }
    for sample in samples.values():
        if sample.pid in ancestor_ids or is_host_control_plane_process(sample):
            protected.add(_sample_pgid(sample))
            continue
        sample_ancestors = _ancestor_pids(samples, sample.pid)
        if (
            host_control_plane_pids.intersection(sample_ancestors)
            and sample.pid not in self_descendant_ids
        ):
            protected.add(_sample_pgid(sample))
    return protected


def _safe_getpgrp() -> int | None:
    try:
        return os.getpgrp()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# EXPLICIT CUSTODY (the primary gate). The durable-daemon registry records
# exactly what WE launched (pid / pgid / cmd / status) — it IS the custody
# ledger. We kill ONLY processes under custody, identity-gated against PID
# recycling. Per molt 3b1e49b18 + operator directive: "kill ONLY processes the
# guard has explicit CUSTODY of ... NEVER terminate anything not under explicit
# custody — even if it's the largest RSS and matches a training pattern."
# ---------------------------------------------------------------------------
_DURABLE_DAEMON_REGISTRY = _REPO_ROOT / ".omx" / "state" / "durable_daemons.json"


@dataclass(frozen=True)
class CustodyRecord:
    label: str
    pid: int
    pgid: int
    cmd: str  # joined command string as recorded at launch


def load_custody_records(path: Path | None = None) -> list[CustodyRecord]:
    """Load RUNNING daemons from the durable-daemon registry = our custody set."""
    p = path or _DURABLE_DAEMON_REGISTRY
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(rows, list):
        return []
    out: list[CustodyRecord] = []
    for r in rows:
        if not isinstance(r, dict) or r.get("status") != "running":
            continue
        try:
            pid = int(r.get("pid", 0))
            pgid = int(r.get("pgid", 0)) or pid
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        cmd = r.get("cmd", "")
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        out.append(CustodyRecord(label=str(r.get("label", "")), pid=pid, pgid=pgid, cmd=cmd_str))
    return out


def _identity_matches(sample: ProcessSample, record: CustodyRecord) -> bool:
    """Identity gate against PID recycling: the LIVE process at this pid must
    still be the daemon we recorded — same process group AND the recorded launch
    command's distinctive script token must appear in the live command line.
    """
    # process-group must match what we recorded (a recycled pid in a different
    # group is not our daemon).
    live_pgid = _sample_pgid(sample)
    if record.pgid > 0 and live_pgid != record.pgid:
        return False
    # the recorded command must share a distinctive script-path token with the
    # live command (so an unrelated process that reused the pid is rejected).
    rec_tokens = [t for t in _command_arg_executable_names(record.cmd) if t.endswith(".py")]
    if not rec_tokens:
        rec_tokens = list(_command_arg_executable_names(record.cmd))
    live = sample.command.casefold()
    return any(tok in live for tok in rec_tokens)


# ---------------------------------------------------------------------------
# The make-or-break kill selector — CUSTODY-GATED, belt-and-suspenders.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KillVictim:
    pid: int
    pgid: int
    rss_gb: float
    command: str
    label: str = ""


def matches_training_allowlist(command: str, pattern: str) -> bool:
    """POSITIVE ALLOWLIST: True iff command matches a training-arm pattern."""
    try:
        return re.search(pattern, command) is not None
    except re.error:
        return False


def select_kill_victim(
    samples: Mapping[int, ProcessSample],
    *,
    custody_records: Sequence[CustodyRecord] | None = None,
    kill_pattern: str = DEFAULT_TRAINING_KILL_PATTERN,
    self_pid: int | None = None,
    self_pgid: int | None = None,
) -> KillVictim | None:
    """Return the largest-RSS *killable training arm UNDER CUSTODY*, or None.

    A process is killable ONLY if it passes ALL of these independent gates
    (custody FIRST — the primary gate per molt 3b1e49b18 + operator directive):
      (0) CUSTODY: it is in ``custody_records`` (the durable-daemon registry —
          something WE launched/own), AND passes the IDENTITY gate (live pgid +
          live command match the recorded launch, defeating PID recycling);
      (1) it is NOT the guard, NOT an ancestor of the guard;
      (2) it is NOT a host control-plane app (``is_host_control_plane_process``);
      (3) it has NO external host control-plane LINEAGE
          (``has_external_host_control_plane_lineage``);
      (4) it does NOT match the broad direct-kill denylist (ssh/tmux/shell/...);
      (5) its pgid is NOT in ``protected_process_group_ids``;
      (6) it is its OWN process-group leader (pgid == pid) — i.e. a detached
          daemon, so the group-kill scope is exactly its subtree, never a shared
          shell job's group;
      (7) its command matches the training ALLOWLIST.

    Returns None (→ caller LOGS A LOUD ALERT and kills NOTHING) when nothing is
    safely killable: "better to alert than kill the control plane."
    """
    if self_pid is None:
        self_pid = os.getpid()
    if self_pgid is None:
        self_pgid = _safe_getpgrp()
    if custody_records is None:
        custody_records = load_custody_records()

    protected_pgids = protected_process_group_ids(
        samples, self_pid=self_pid, self_pgid=self_pgid
    )
    guard_ancestors = _ancestor_pids(samples, self_pid)
    owned_pids = frozenset(r.pid for r in custody_records)

    best: KillVictim | None = None
    for record in custody_records:
        sample = samples.get(record.pid)
        if sample is None:  # daemon already dead
            continue
        # (0) identity gate (PID-recycling defense)
        if not _identity_matches(sample, record):
            continue
        cmd = sample.command
        # (1) never the guard or its ancestors
        if sample.pid == self_pid or sample.pid in guard_ancestors:
            continue
        # (2) never a control-plane app
        if is_host_control_plane_process(sample):
            continue
        # (3) never external control-plane lineage (custody-aware)
        if has_external_host_control_plane_lineage(
            samples, sample.pid, current_pid=self_pid, owned_pids=owned_pids
        ):
            continue
        # (4) broad direct-kill denylist
        if _matches_extra_protected(cmd):
            continue
        # (5) process-group protection
        pgid = _sample_pgid(sample)
        if pgid in protected_pgids:
            continue
        # (6) must be its own group leader (detached daemon)
        if pgid != sample.pid:
            continue
        # (7) positive training allowlist
        if not matches_training_allowlist(cmd, kill_pattern):
            continue
        rss_gb = sample.rss_kb / 1e6
        if best is None or rss_gb > best.rss_gb:
            best = KillVictim(
                pid=sample.pid, pgid=pgid, rss_gb=rss_gb, command=cmd, label=record.label
            )
    return best


# ---------------------------------------------------------------------------
# Launch preflight
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LaunchDecision:
    available_gb: float
    projected_gb: float
    headroom_after_gb: float
    min_free_gb: float
    ok: bool

    def to_json(self) -> dict[str, object]:
        return {
            "available_gb": round(self.available_gb, 2),
            "projected_gb": round(self.projected_gb, 2),
            "headroom_after_gb": round(self.headroom_after_gb, 2),
            "min_free_gb": self.min_free_gb,
            "decision": "OK" if self.ok else "REFUSE",
        }


def check_launch_ok(
    *,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    projected_gb: float = 0.0,
    available: float | None = None,
) -> LaunchDecision:
    """Return whether launching a job needing ``projected_gb`` keeps >= the floor.

    ``available`` is injectable for testing; defaults to a live vm_stat read.
    """
    avail = available_gb() if available is None else available
    headroom = avail - projected_gb
    return LaunchDecision(
        available_gb=avail,
        projected_gb=projected_gb,
        headroom_after_gb=headroom,
        min_free_gb=min_free_gb,
        ok=headroom >= min_free_gb,
    )


# ---------------------------------------------------------------------------
# Logging + kill
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts} {msg}"
    try:
        with open(_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def _kill_pgrp(pgid: int, pid: int) -> None:
    """SIGTERM the process group (no orphan); fall back to the pid."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


DEFAULT_SHED_CONSECUTIVE = 3
DEFAULT_SHED_MARGIN_GB = 2.0


def shed_decision(
    *,
    consecutive_subfloor: int,
    avail_generous: float,
    min_free_gb: float,
    shed_margin_gb: float = DEFAULT_SHED_MARGIN_GB,
    shed_consecutive: int = DEFAULT_SHED_CONSECUTIVE,
) -> str:
    """Pure SHED policy (M3 fix): debounced + near-boundary-ALERT.

    Returns one of:
      * "ok"    — generous (reclaimable-aware) available is at/above the floor;
      * "alert" — below the floor but NOT yet a shed (debounce not satisfied, OR
                  the reading is only within ``shed_margin_gb`` of the floor —
                  near-boundary noise gets an ALERT, never a kill);
      * "shed"  — sustained (>= ``shed_consecutive`` consecutive polls) AND
                  clearly below (avail < floor - margin) → shed the largest
                  custody training arm.

    ``consecutive_subfloor`` is the count of consecutive polls (including this
    one) where the generous metric was below the floor. The REFUSE launch-
    preflight is intentionally NOT routed through this (it stays conservative +
    fail-closed); only the watchdog SHED path uses this debounced policy so
    macOS memory compression cannot false-positive-shed a healthy arm.
    """
    if avail_generous >= min_free_gb:
        return "ok"
    if consecutive_subfloor >= shed_consecutive and avail_generous < (min_free_gb - shed_margin_gb):
        return "shed"
    return "alert"


# ---------------------------------------------------------------------------
# Watchdog loop
# ---------------------------------------------------------------------------
def watch_loop(
    *,
    min_free_gb: float,
    kill_pattern: str,
    interval: float,
    warn_margin: float,
    shed_consecutive: int = DEFAULT_SHED_CONSECUTIVE,
    shed_margin_gb: float = DEFAULT_SHED_MARGIN_GB,
    max_iterations: int | None = None,
) -> int:
    """Whole-machine watchdog. Loops forever (or ``max_iterations`` for tests).

    SHED decisions use the GENEROUS / reclaimable-aware metric + a debounce
    (M3 fix): a custody arm is shed only after ``shed_consecutive`` consecutive
    sub-floor polls AND when clearly below the floor (> ``shed_margin_gb``).
    Near-boundary / transient sub-floor readings get a (non-killing) ALERT.
    """
    self_pid = os.getpid()
    self_pgid = _safe_getpgrp()
    _log(
        f"WATCH start min_free={min_free_gb}GB warn_margin={warn_margin}GB "
        f"interval={interval}s shed_consecutive={shed_consecutive} "
        f"shed_margin={shed_margin_gb}GB pattern={kill_pattern!r} "
        f"self_pid={self_pid} self_pgid={self_pgid}"
    )
    it = 0
    consecutive_subfloor = 0
    while max_iterations is None or it < max_iterations:
        it += 1
        try:
            avail_generous = available_generous_gb()
            avail_conservative = available_gb()
            if avail_generous < min_free_gb:
                consecutive_subfloor += 1
            else:
                consecutive_subfloor = 0
            decision = shed_decision(
                consecutive_subfloor=consecutive_subfloor,
                avail_generous=avail_generous,
                min_free_gb=min_free_gb,
                shed_margin_gb=shed_margin_gb,
                shed_consecutive=shed_consecutive,
            )
            if decision == "shed":
                samples = sample_processes()
                custody = load_custody_records()
                victim = select_kill_victim(
                    samples,
                    custody_records=custody,
                    kill_pattern=kill_pattern,
                    self_pid=self_pid,
                    self_pgid=self_pgid,
                )
                if victim is not None:
                    _log(
                        f"CRITICAL avail_generous={avail_generous:.1f}GB "
                        f"(conservative={avail_conservative:.1f}GB) < {min_free_gb}GB "
                        f"sustained {consecutive_subfloor} polls → KILL custody-arm "
                        f"label={victim.label!r} pid={victim.pid} pgid={victim.pgid} "
                        f"rss={victim.rss_gb:.1f}GB cmd={victim.command[:140]}"
                    )
                    _kill_pgrp(victim.pgid, victim.pid)
                    consecutive_subfloor = 0  # reset after acting; re-evaluate next poll
                else:
                    _log(
                        f"ALERT avail_generous={avail_generous:.1f}GB < {min_free_gb}GB "
                        f"sustained {consecutive_subfloor} polls but NO killable training "
                        f"arm UNDER CUSTODY (custody_count={len(custody)}) — killing "
                        "NOTHING (control plane + non-custody protected); manual "
                        "intervention may be needed."
                    )
            elif decision == "alert":
                _log(
                    f"ALERT avail_generous={avail_generous:.1f}GB "
                    f"(conservative={avail_conservative:.1f}GB) < floor {min_free_gb}GB "
                    f"but NOT shedding yet (consecutive={consecutive_subfloor}/"
                    f"{shed_consecutive}, margin={shed_margin_gb}GB — debounce/near-"
                    "boundary; prefer ALERT over kill). Free memory if sustained."
                )
            elif avail_generous < min_free_gb + warn_margin:
                _log(
                    f"WARN avail_generous={avail_generous:.1f}GB approaching floor "
                    f"{min_free_gb}GB"
                )
        except Exception as e:  # never let the guard die silently
            _log(f"ERROR guard loop: {e!r}")
        if max_iterations is not None and it >= max_iterations:
            break
        time.sleep(interval)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Unified-memory OOM guard (protect >=30GB free; never kill control plane)."
    )
    ap.add_argument("--free", action="store_true", help="print available + strict-free GB (JSON)")
    ap.add_argument("--check", action="store_true", help="exit 3 if (available - projected) < min_free")
    ap.add_argument("--watch", action="store_true", help="loop + shed largest training arm if available < min_free")
    ap.add_argument(
        "--select-victim-dry-run",
        action="store_true",
        help="print (without killing) the victim the watchdog WOULD select now",
    )
    ap.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    ap.add_argument("--projected-gb", type=float, default=0.0, help="memory the to-be-launched job will need")
    ap.add_argument("--kill-pattern", type=str, default=DEFAULT_TRAINING_KILL_PATTERN,
                    help="regex of TRAINING-ARM commands eligible to be killed (allowlist)")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC)
    ap.add_argument("--warn-margin", type=float, default=DEFAULT_WARN_MARGIN_GB)
    ap.add_argument("--shed-consecutive", type=int, default=DEFAULT_SHED_CONSECUTIVE,
                    help="consecutive sub-floor polls required before the watchdog sheds (debounce)")
    ap.add_argument("--shed-margin-gb", type=float, default=DEFAULT_SHED_MARGIN_GB,
                    help="only shed when generous-available < (min_free - this); else ALERT (near-boundary)")
    args = ap.parse_args(argv)

    if args.free:
        avail, sf = _vm_stat_bytes()
        print(json.dumps({
            "available_gb": round(avail / 1e9, 2),
            "available_generous_gb": round(available_generous_gb(), 2),
            "strict_free_gb": round(sf / 1e9, 2),
        }))
        return 0

    if args.check:
        decision = check_launch_ok(min_free_gb=args.min_free_gb, projected_gb=args.projected_gb)
        print(json.dumps(decision.to_json()))
        return 0 if decision.ok else 3

    if args.select_victim_dry_run:
        samples = sample_processes()
        custody = load_custody_records()
        victim = select_kill_victim(samples, custody_records=custody, kill_pattern=args.kill_pattern)
        if victim is None:
            print(json.dumps({
                "victim": None,
                "custody_count": len(custody),
                "note": "no killable training arm under custody; control plane + non-custody protected",
            }))
        else:
            print(json.dumps({
                "victim": {
                    "label": victim.label, "pid": victim.pid, "pgid": victim.pgid,
                    "rss_gb": round(victim.rss_gb, 2), "command": victim.command[:200],
                },
                "custody_count": len(custody),
            }))
        return 0

    if args.watch:
        return watch_loop(
            min_free_gb=args.min_free_gb,
            kill_pattern=args.kill_pattern,
            interval=args.interval,
            warn_margin=args.warn_margin,
            shed_consecutive=args.shed_consecutive,
            shed_margin_gb=args.shed_margin_gb,
        )

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
