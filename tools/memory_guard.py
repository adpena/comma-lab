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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
DEFAULT_TRAINING_KILL_PATTERN = (  # OBSERVER_ROLE_OK: kill selection is custody-gated by SIX independent conditions (positive training allowlist AND explicit custody AND control-plane protection AND direct-kill denylist AND own-process-group-leader), not by token presence; regex is one necessary condition, never sufficient. ddm_gh1 #829 file-scope fallback FP -- flagged for a dedicated review, not silently blessed.
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
# NOTE: ``safe_run.py`` is deliberately NOT in this denylist. A training arm
# wrapped in safe_run (the layer-3 per-arm cap) records the safe_run WRAPPER as
# its custody pgid; the watchdog sheds the arm by killpg(wrapper), which cascades
# to the inner trainer (safe_run's SIGTERM handler). Protecting "safe_run.py"
# here would make every wrapped training arm UN-SHEDDABLE (HIGH-1 corollary). A
# safe_run wrapping a NON-training command is still safe — it would not match the
# training allowlist (gate 7) nor be in custody (gate 0).
EXTRA_PROTECTED_TOKENS: tuple[str, ...] = (
    "/usr/bin/ssh", " ssh ", "sshd", "tmux", "/sbin/launchd", "WindowServer",
    "loginwindow", "memory_guard.py", "spawn_durable_daemon.py",
    # Memory-protection infra: the black-box recorder + system governor MUST never be shed/paused —
    # they ARE the protection layer (a governor that pauses itself cannot recover the machine).
    "memory_blackbox.py", "system_memory_governor.py",
)
EXTRA_PROTECTED_EXECUTABLE_NAMES: frozenset[str] = frozenset(
    {"ssh", "sshd", "tmux", "launchd", "login", "-bash", "-zsh", "-sh",
     "bash", "zsh", "sh", "windowserver"}
)


# ---------------------------------------------------------------------------
# vm_stat available-memory floor (OURS).
#
# A SINGLE, CONSERVATIVE metric drives BOTH the REFUSE launch-preflight AND the
# SHED watchdog: ``available_gb`` = (free + inactive) * page_size.
#
# HIGH-2 review fix (the OOM-dangerous overcount): the earlier "generous" metric
# (free + inactive + purgeable + FILE-BACKED) OVERCOUNTS true reclaimable memory.
# On macOS, ``File-backed pages`` includes ACTIVE in-use file pages (the identity
# ``file_backed + anonymous == active + inactive + speculative`` holds), so under
# real pressure file_backed stays large while free collapses → the generous
# metric reads ~40GB when true available is ~10GB → the watchdog returns "ok" and
# NEVER sheds → OOM. We therefore DROP file_backed entirely and drive SHED off
# the conservative metric: it is provably ``<= true_available`` (it omits some
# genuinely-reclaimable purgeable/clean cache), so it FAILS TOWARD shedding =
# OOM-SAFE. Over-eager shedding is bounded by the debounce + margin in watch_loop
# and only ever sheds a recoverable, resumable CUSTODY training arm (never the
# control plane). REFUSE stays conservative too (fail-closed = never over-commit).
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
    fields["speculative"] = _grab("Pages speculative")
    fields["purgeable"] = _grab("Pages purgeable")
    fields["file_backed"] = _grab("File-backed pages")
    return fields


def _vm_stat_fields() -> dict[str, int]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout  # subprocess-no-check-OK: rc!=0 yields empty fields -> available reads 0 -> fails toward shedding, the module's declared OOM-safe direction
    return _parse_vm_stat(out)


def _available_gb_from_fields(fields: dict[str, int]) -> float:
    """CONSERVATIVE available GB from parsed fields (free + inactive). Pure +
    testable. Deliberately EXCLUDES file_backed (HIGH-2: it overcounts)."""
    ps = fields["page_size"]
    return (fields.get("free", 0) + fields.get("inactive", 0)) * ps / 1e9


def _broadest_available_estimate_gb_from_fields(fields: dict[str, int]) -> float:
    """The most-generous conceivable available estimate (free + inactive +
    speculative + purgeable + file_backed). USED ONLY by the no-overcount test:
    the shed metric must be <= this for all inputs (it never claims more memory
    than could possibly be reclaimable → never falsely 'ok' → never skips a
    needed shed). NOT used to drive any decision (it overcounts)."""
    ps = fields["page_size"]
    return (
        fields.get("free", 0) + fields.get("inactive", 0) + fields.get("speculative", 0)
        + fields.get("purgeable", 0) + fields.get("file_backed", 0)
    ) * ps / 1e9


def _vm_stat_bytes() -> tuple[float, float]:
    """Return (conservative_available_bytes, strict_free_bytes) from vm_stat."""
    f = _vm_stat_fields()
    ps = f["page_size"]
    available = (f["free"] + f["inactive"]) * ps
    return float(available), float(f["free"] * ps)


def available_gb() -> float:
    """CONSERVATIVE available GB (free + inactive). Drives BOTH REFUSE and SHED
    (HIGH-2 fix: fail toward shedding = OOM-safe; never overcounts)."""
    return _available_gb_from_fields(_vm_stat_fields())


def _broadest_available_estimate_gb() -> float:
    """Live broadest estimate — for the no-overcount invariant test only."""
    return _broadest_available_estimate_gb_from_fields(_vm_stat_fields())


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
    # Kernel-reported process start identity (``ps lstart``). PIDs recycle; any persisted per-PID
    # safety history must bind to this value, never to PID alone. Optional only for legacy fixtures.
    start_identity: str | None = None


def parse_ps_table(text: str) -> dict[int, ProcessSample]:
    """Parse ``ps -axo pid=,ppid=,pgid=,rss=,lstart=,command=`` output.

    ``lstart`` is five tokens on BSD/macOS. It is retained as process-start identity so a recycled
    PID cannot inherit the prior process's safety history. Legacy 5-column and 4-column fixtures
    remain accepted with ``start_identity=None``; consumers must fail conservative when identity is
    unavailable.
    """
    samples: dict[int, ProcessSample] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        extended = line.split(None, 9)
        start_identity: str | None = None
        pgid: int | None
        looks_extended = (
            len(extended) >= 10
            and extended[4] in {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
            and re.fullmatch(r"\d{1,2}", extended[6]) is not None
            and re.fullmatch(r"\d{2}:\d{2}:\d{2}", extended[7]) is not None
            and re.fullmatch(r"\d{4}", extended[8]) is not None
        )
        if looks_extended:
            try:
                pid = int(extended[0])
                ppid = int(extended[1])
                pgid = int(extended[2])
                rss_kb = int(extended[3])
                start_identity = " ".join(extended[4:9])
                command = extended[9]
            except ValueError:
                continue
        else:
            parts = line.split(None, 4)
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
            pid=pid, ppid=ppid, rss_kb=rss_kb, command=command, pgid=pgid,
            start_identity=start_identity,
        )
    return samples


def sample_processes() -> dict[int, ProcessSample]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,rss=,lstart=,command="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
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


def group_rss_gb(samples: Mapping[int, ProcessSample], root_pid: int) -> float:
    """REAL footprint of a custody arm: RSS summed over ``root_pid`` AND all its
    descendants (HIGH-1 corollary fix). Under safe_run wrapping the custody pid is
    the ~10MB wrapper; the inner trainer is a descendant (child by ppid even
    though it is its own session leader), so summing descendants attributes the
    true memory — making 'shed the LARGEST arm' selection + the CRITICAL log
    correct. Works for bare (unwrapped) arms too (sums the arm + its children)."""
    total_kb = 0
    for pid in descendant_pids(samples, root_pid):
        s = samples.get(pid)
        if s is not None:
            total_kb += s.rss_kb
    return total_kb / 1e6


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


def _record_passes_kill_gates(
    sample: ProcessSample,
    record: CustodyRecord,
    *,
    samples: Mapping[int, ProcessSample],
    protected_pgids: set[int] | frozenset[int],
    guard_ancestors: set[int] | frozenset[int],
    owned_pids: set[int] | frozenset[int] | tuple[int, ...],
    self_pid: int,
    kill_pattern: str,
) -> bool:
    """The 8-gate control-plane-safety predicate (SINGLE SOURCE OF TRUTH).

    True iff ``sample`` (the LIVE process at ``record.pid``) is a killable
    training arm UNDER CUSTODY — passing ALL of gates (0)-(7) documented on
    ``select_kill_victim``. Extracted verbatim so BOTH the selector AND the
    SIGKILL-escalation re-check (``pgid_is_killable_custody_arm``) apply the
    EXACT same gates — a recycled control-plane group can never pass one path and
    fail the other. Weakening any gate here weakens BOTH, which is exactly the
    molt CP-kill class this guard exists to prevent.
    """
    # (0) identity gate (PID-recycling defense)
    if not _identity_matches(sample, record):
        return False
    cmd = sample.command
    # (1) never the guard or its ancestors
    if sample.pid == self_pid or sample.pid in guard_ancestors:
        return False
    # (2) never a control-plane app
    if is_host_control_plane_process(sample):
        return False
    # (3) never external control-plane lineage (custody-aware)
    if has_external_host_control_plane_lineage(
        samples, sample.pid, current_pid=self_pid, owned_pids=owned_pids
    ):
        return False
    # (4) broad direct-kill denylist
    if _matches_extra_protected(cmd):
        return False
    # (5) process-group protection
    pgid = _sample_pgid(sample)
    if pgid in protected_pgids:
        return False
    # (6) must be its own group leader (detached daemon)
    if pgid != sample.pid:
        return False
    # (7) positive training allowlist
    return matches_training_allowlist(cmd, kill_pattern)


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
        if not _record_passes_kill_gates(
            sample,
            record,
            samples=samples,
            protected_pgids=protected_pgids,
            guard_ancestors=guard_ancestors,
            owned_pids=owned_pids,
            self_pid=self_pid,
            kill_pattern=kill_pattern,
        ):
            continue
        pgid = _sample_pgid(sample)
        # REAL footprint = the arm's whole process group/descendants (HIGH-1
        # corollary: under safe_run wrapping sample.rss_kb is the ~10MB wrapper).
        rss_gb = group_rss_gb(samples, sample.pid)
        if best is None or rss_gb > best.rss_gb:
            best = KillVictim(
                pid=sample.pid, pgid=pgid, rss_gb=rss_gb, command=sample.command, label=record.label
            )
    return best


def pgid_is_killable_custody_arm(
    pgid: int,
    *,
    samples: Mapping[int, ProcessSample] | None = None,
    custody_records: Sequence[CustodyRecord] | None = None,
    kill_pattern: str = DEFAULT_TRAINING_KILL_PATTERN,
    self_pid: int | None = None,
    self_pgid: int | None = None,
) -> bool:
    """FRESH re-verification that ``pgid`` STILL identifies a control-plane-safe
    custody training arm — the SIGKILL-escalation guard against PID/PGID recycling
    during the SIGTERM grace window.

    The escalation cascade SIGTERMs a vetted victim, waits a grace window, and (if
    the group is still alive) SIGKILLs. But a process group is a REUSABLE kernel
    id: if the training arm exits during the grace window and the kernel recycles
    its pgid onto a control-plane group, a blind SIGKILL(pgid) would kill the
    control plane — the EXACT molt CP-kill class. This predicate re-samples the
    LIVE process table and re-runs the SAME 8-gate suite as ``select_kill_victim``
    (``_record_passes_kill_gates`` — single source of truth) restricted to
    ``pgid``. It returns True ONLY if the group STILL contains a custody arm that
    passes every gate.

    Fails SAFE (returns False → NO SIGKILL) when: the pgid is non-positive; the
    process table cannot be sampled; the arm exited (pid gone); the pgid was
    recycled onto a control-plane / protected / non-custody group; or nothing
    under custody still lives in the group.
    """
    if pgid <= 0:
        return False
    if self_pid is None:
        self_pid = os.getpid()
    if self_pgid is None:
        self_pgid = _safe_getpgrp()
    if samples is None:
        samples = sample_processes()
    if not samples:  # cannot verify the live table → fail safe (no SIGKILL)
        return False
    if custody_records is None:
        custody_records = load_custody_records()

    protected_pgids = protected_process_group_ids(
        samples, self_pid=self_pid, self_pgid=self_pgid
    )
    guard_ancestors = _ancestor_pids(samples, self_pid)
    owned_pids = frozenset(r.pid for r in custody_records)

    for record in custody_records:
        sample = samples.get(record.pid)
        if sample is None:
            continue
        if _sample_pgid(sample) != pgid:
            continue
        if _record_passes_kill_gates(
            sample,
            record,
            samples=samples,
            protected_pgids=protected_pgids,
            guard_ancestors=guard_ancestors,
            owned_pids=owned_pids,
            self_pid=self_pid,
            kill_pattern=kill_pattern,
        ):
            return True
    return False


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


# SIGKILL escalation defaults. The grace window is DERIVED from the watch
# interval (one full poll cadence, floored) so a well-behaved arm's SIGTERM
# handler (safe_run checkpoints + exits) has time to finish before we escalate;
# it is NOT a magic constant. ``watch_loop`` passes ``max(GRACE_MIN, interval)``.
DEFAULT_SIGKILL_GRACE_MIN_SEC = 3.0
DEFAULT_SIGKILL_GRACE_SEC = 5.0
_SIGKILL_POLL_SEC = 0.2


def _signal_pgrp(pgid: int, pid: int, sig: int, *, pid_fallback: bool = True) -> bool:
    """Deliver ``sig`` to the process GROUP (optionally fall back to the ``pid``).

    Returns True iff the signal reached SOMETHING, False if the target(s) were
    already gone / not permitted (a pure no-op — an already-dead group never
    raises out of here; no ESRCH escapes). ``pgid`` must be positive: ``killpg``
    with a non-positive pgid has dangerous broadcast semantics, so we route
    straight to the pid instead.

    ``pid_fallback`` (default True, preserving the original SIGTERM behavior): if
    the group is gone/not-permitted, try ``os.kill(pid, sig)``. The SIGKILL
    escalation passes ``pid_fallback=False`` DELIBERATELY: victim.pid == pgid
    (group leader), so a dead group means the leader is gone too — the only thing
    ``os.kill(pid)`` could still hit is a RECYCLED pid (a fresh, possibly
    control-plane, process reusing that pid number). For an unstoppable SIGKILL
    that CP-kill vector is unacceptable and valueless (a dead group is a dead
    arm), so the SIGKILL path is group-only.
    """
    if pgid > 0:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            pass  # group gone → try the single pid below (if allowed)
        except PermissionError:
            pass  # not permitted to signal the group → try the single pid below
    if pid_fallback and pid > 0:
        try:
            os.kill(pid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False
    return False


def _pgrp_alive(pgid: int) -> bool:
    """True iff at least one process remains in the group ``pgid``.

    ``os.killpg(pgid, 0)`` is the POSIX existence probe (signal 0 delivers
    nothing, only performs the existence + permission check). A PermissionError
    means the group EXISTS but we may not signal it — treat as alive (we could
    not have SIGKILLed it anyway, and 'alive' is the fail-safe direction here).
    A non-positive pgid is treated as not-alive (never broadcast-probe).
    """
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _kill_pgrp(
    pgid: int,
    pid: int,
    *,
    grace_sec: float = DEFAULT_SIGKILL_GRACE_SEC,
    poll_sec: float = _SIGKILL_POLL_SEC,
    still_killable: Callable[[int, int], bool] | None = None,
) -> None:
    """Escalating shed of a VETTED custody training arm: SIGTERM → grace → SIGKILL.

    This is the whole-machine watchdog's LAST-RESORT kill. Its caller
    (``select_kill_victim``) has already proven the ``(pgid, pid)`` is a
    control-plane-safe custody training arm through the full 8-gate suite. The
    escalation adds defense-in-depth for the rare arm that ignores SIGTERM:

      (1) SIGTERM the process GROUP first — let the arm (and safe_run's SIGTERM
          handler) checkpoint and exit cleanly. If the SIGTERM was a pure no-op
          (group + pid already gone), there is nothing to escalate → return.
      (2) GRACE WINDOW — poll for the group to exit. As soon as it is gone,
          return (no SIGKILL needed — the common, clean case).
      (3) SIGKILL ONLY IF the group is STILL alive after the grace window AND
          ``still_killable(pgid, pid)`` re-confirms the group is STILL a
          control-plane-safe custody arm. A process group id is a REUSABLE kernel
          resource: if the arm exited mid-grace and the pgid was recycled onto a
          control-plane group, a blind SIGKILL would be the molt CP-kill class.
          The re-check (``pgid_is_killable_custody_arm``, fresh-sampled) gates the
          SIGKILL exactly as the selector gated the SIGTERM — the exempt set gates
          BOTH signals.

    Fail-safe by construction: if ``still_killable`` is None (no re-verifier
    supplied — e.g. an unexpected direct caller), the SIGKILL is SKIPPED and the
    function is SIGTERM-only, exactly as before. You cannot reach SIGKILL without
    a passing safety re-check. An already-dead group is a no-op at every step.
    """
    # (1) SIGTERM first.
    if not _signal_pgrp(pgid, pid, signal.SIGTERM):
        return  # already gone / not permitted → nothing to escalate

    # (2) grace window — poll for a clean exit on SIGTERM.
    deadline = time.monotonic() + max(0.0, grace_sec)
    while time.monotonic() < deadline:
        if not _pgrp_alive(pgid):
            return  # exited on SIGTERM — no SIGKILL needed (the clean path)
        time.sleep(max(0.0, poll_sec))

    # (3) still alive after grace → SIGKILL, but ONLY after re-verifying safety.
    if not _pgrp_alive(pgid):
        return  # exited right at the deadline
    if still_killable is None:
        _log(
            f"ESCALATE-SKIP pgid={pgid} pid={pid} survived SIGTERM+{grace_sec}s "
            "grace but NO safety re-check supplied → SIGTERM-only (no SIGKILL)"
        )
        return
    if not still_killable(pgid, pid):
        _log(
            f"ESCALATE-ABORT pgid={pgid} pid={pid} still alive after {grace_sec}s "
            "grace but FAILED the control-plane-safety re-check (pgid recycled or "
            "now protected/non-custody) → NOT sending SIGKILL"
        )
        return
    _log(f"ESCALATE pgid={pgid} pid={pid} survived SIGTERM+{grace_sec}s grace → SIGKILL")
    # Group-only SIGKILL (no pid fallback): a dead group means the leader pid is
    # gone; os.kill(pid) could then only hit a RECYCLED pid (possible CP). See
    # _signal_pgrp's ``pid_fallback`` rationale.
    _signal_pgrp(pgid, pid, signal.SIGKILL, pid_fallback=False)


DEFAULT_SHED_CONSECUTIVE = 3
DEFAULT_SHED_MARGIN_GB = 2.0


def shed_decision(
    *,
    consecutive_subfloor: int,
    avail: float,
    min_free_gb: float,
    shed_margin_gb: float = DEFAULT_SHED_MARGIN_GB,
    shed_consecutive: int = DEFAULT_SHED_CONSECUTIVE,
) -> str:
    """Pure SHED policy: debounced + near-boundary-ALERT.

    ``avail`` is the CONSERVATIVE metric (free + inactive) — HIGH-2 fix: it never
    overcounts (provably <= true available), so the watchdog can never falsely
    read 'ok' and skip a needed shed → OOM-safe. Over-eager shedding from the
    conservative undercount is bounded by the debounce + margin below, and only
    ever sheds a recoverable CUSTODY training arm (never the control plane).

    Returns one of:
      * "ok"    — conservative available is at/above the floor;
      * "alert" — below the floor but NOT yet a shed (debounce not satisfied, OR
                  the reading is only within ``shed_margin_gb`` of the floor —
                  near-boundary noise gets an ALERT, never a kill);
      * "shed"  — sustained (>= ``shed_consecutive`` consecutive polls) AND
                  clearly below (avail < floor - margin) → shed the largest
                  custody training arm.

    ``consecutive_subfloor`` is the count of consecutive polls (including this
    one) where ``avail`` was below the floor. The REFUSE launch-preflight uses
    the same conservative metric (fail-closed = never over-commit).
    """
    if avail >= min_free_gb:
        return "ok"
    if consecutive_subfloor >= shed_consecutive and avail < (min_free_gb - shed_margin_gb):
        return "shed"
    return "alert"


def _warn_custody_arms_classified_control_plane(kill_pattern: str) -> list[str]:
    """LOW-4: warn if any RUNNING custody arm that matches the training allowlist
    ALSO classifies as control-plane (its argv coincidentally contains /.claude/
    /.codex/ etc.) — such an arm would be PROTECTED → un-sheddable (OOM-adjacent).
    Returns the list of offending labels (also logged). Observability only."""
    offending: list[str] = []
    samples = sample_processes()
    for rec in load_custody_records():
        s = samples.get(rec.pid)
        cmd = s.command if s is not None else rec.cmd
        if matches_training_allowlist(cmd, kill_pattern) and is_protected_command(cmd):
            offending.append(rec.label)
            _log(
                f"WARN custody training arm label={rec.label!r} CLASSIFIES "
                "control-plane (argv contains a control-plane token) → it would be "
                "PROTECTED and UN-SHEDDABLE. Keep training-arm paths free of "
                "claude/codex/anthropic tokens."
            )
    return offending


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
    sigkill_grace_sec: float | None = None,
    max_iterations: int | None = None,
) -> int:
    """Whole-machine watchdog. Loops forever (or ``max_iterations`` for tests).

    SHED decisions use the CONSERVATIVE metric (free + inactive; HIGH-2 fix —
    never overcounts → never skips a needed shed → OOM-safe) + a debounce: a
    custody arm is shed only after ``shed_consecutive`` consecutive sub-floor
    polls AND when clearly below the floor (> ``shed_margin_gb``). Near-boundary /
    transient sub-floor readings get a (non-killing) ALERT.
    """
    self_pid = os.getpid()
    self_pgid = _safe_getpgrp()
    # DERIVED grace: one full poll cadence, floored — NOT a magic number.
    if sigkill_grace_sec is None:
        sigkill_grace_sec = max(DEFAULT_SIGKILL_GRACE_MIN_SEC, interval)
    _log(
        f"WATCH start min_free={min_free_gb}GB warn_margin={warn_margin}GB "
        f"interval={interval}s shed_consecutive={shed_consecutive} "
        f"shed_margin={shed_margin_gb}GB sigkill_grace={sigkill_grace_sec}s "
        f"pattern={kill_pattern!r} self_pid={self_pid} self_pgid={self_pgid}"
    )
    # LOW-4 startup WARN: surface any custody training arm that would (mis)classify
    # as control-plane — it would be protected/un-sheddable (OOM-adjacent).
    try:
        _warn_custody_arms_classified_control_plane(kill_pattern)
    except Exception:  # observability only — never block the watchdog
        pass
    it = 0
    consecutive_subfloor = 0
    while max_iterations is None or it < max_iterations:
        it += 1
        try:
            avail = available_gb()  # CONSERVATIVE — drives shed (HIGH-2 fix)
            if avail < min_free_gb:
                consecutive_subfloor += 1
            else:
                consecutive_subfloor = 0
            decision = shed_decision(
                consecutive_subfloor=consecutive_subfloor,
                avail=avail,
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
                        f"CRITICAL available={avail:.1f}GB < {min_free_gb}GB sustained "
                        f"{consecutive_subfloor} polls → KILL custody-arm "
                        f"label={victim.label!r} pid={victim.pid} pgid={victim.pgid} "
                        f"group_rss={victim.rss_gb:.1f}GB cmd={victim.command[:140]}"
                    )
                    # SIGTERM → grace → SIGKILL, with a FRESH-sampled control-plane
                    # re-check gating the SIGKILL (pgid-recycling defense — the
                    # exempt set gates BOTH signals, never just the SIGTERM).
                    _kill_pgrp(
                        victim.pgid,
                        victim.pid,
                        grace_sec=sigkill_grace_sec,
                        still_killable=lambda pg, pi: pgid_is_killable_custody_arm(
                            pg,
                            kill_pattern=kill_pattern,
                            self_pid=self_pid,
                            self_pgid=self_pgid,
                        ),
                    )
                    consecutive_subfloor = 0  # reset after acting; re-evaluate next poll
                else:
                    _log(
                        f"ALERT available={avail:.1f}GB < {min_free_gb}GB sustained "
                        f"{consecutive_subfloor} polls but NO killable training arm "
                        f"UNDER CUSTODY (custody_count={len(custody)}) — killing NOTHING "
                        "(control plane + non-custody protected); manual intervention "
                        "may be needed."
                    )
            elif decision == "alert":
                _log(
                    f"ALERT available={avail:.1f}GB < floor {min_free_gb}GB but NOT "
                    f"shedding yet (consecutive={consecutive_subfloor}/{shed_consecutive}, "
                    f"margin={shed_margin_gb}GB — debounce/near-boundary; prefer ALERT "
                    "over kill). Free memory if sustained."
                )
            elif avail < min_free_gb + warn_margin:
                _log(f"WARN available={avail:.1f}GB approaching floor {min_free_gb}GB")
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
                    help="only shed when conservative-available < (min_free - this); else ALERT (near-boundary)")
    ap.add_argument("--sigkill-grace-sec", type=float, default=None,
                    help="grace window after SIGTERM before escalating to SIGKILL a still-alive "
                         "shed arm (default: max(3s, interval)); SIGKILL is re-gated by the "
                         "control-plane-safety check so a recycled pgid is never killed")
    args = ap.parse_args(argv)

    if args.free:
        avail, sf = _vm_stat_bytes()
        print(json.dumps({
            "available_gb": round(avail / 1e9, 2),  # conservative; drives REFUSE + SHED
            "broadest_estimate_gb": round(_broadest_available_estimate_gb(), 2),  # observability only (overcounts)
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
            sigkill_grace_sec=args.sigkill_grace_sec,
        )

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
