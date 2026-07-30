#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""main_hot_state — the orchestrator's retained-reasoning manifest (compaction-cliff cure).

Operator directive 2026-07-29 (binding): "We can also enhance our apparatus and harness
engineering to make you smarter and more capable and less forgetful and more coherent."
External anchor (ARC-AGI-3, Bigio & Sanders 2026-07-29): two harness settings — retained
reasoning + context compaction — took GPT-5.6 Sol 13.3%->38.3% at 6x fewer output tokens.
This tool is OUR *retained-reasoning* equivalent for the MAIN orchestrator: a tiny durable
manifest of the live campaign state that survives a context-compaction cliff, so MAIN never
has to re-guess a memo slug / a live pid / an open operator decision after compaction.

Measured seam this cures (07-31 session): after a compaction, MAIN mis-guessed the pi1 memo
slug and wrongly assumed a helper file was missing (stale-worktree assumption). Both were
recorded facts; neither was loaded at decision time. The manifest holds them explicitly.

TYPED SECTIONS (fixed set; a section is a queue, never a free-form scratchpad):
  pointer_line          — the exact-score pointer (means/ends firewall: the END first)
  live_processes        — pid / out-dir / ETA of any long-running job (never touch its dir)
  live_arms             — task-id / name / one-line charter of dispatched subagent arms
  open_operator_decisions — decisions awaiting the operator (GO/no-go, config, promotion)
  monitor_tasks         — things MAIN is actively watching (a burn's knee, an arm's landing)
  next_boundaries       — the next boundary charter / the next action MAIN takes
  freshest_receipts     — paths + SHAs of the freshest load-bearing artifacts

STORAGE: .omx/state/main_hot_state.md — human-readable markdown, machine-parseable by the
``## NAME`` section headers. Writes are atomic (tmp + os.replace) under an fcntl LOCK_EX on a
sidecar lock file (Catalog #131 pattern), so a concurrent writer never interleaves a partial
manifest. Dependency-light (stdlib only) so the SessionStart digest wire-in stays fast +
fail-open.

USAGE:
  .venv/bin/python tools/main_hot_state.py                       # read (print the manifest)
  .venv/bin/python tools/main_hot_state.py --json                # machine-readable sections
  .venv/bin/python tools/main_hot_state.py --set-section live_arms --content-file F
  echo "...content..." | tools/main_hot_state.py --set-section pointer_line
  .venv/bin/python tools/main_hot_state.py --seed                # write initial state if absent
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = _REPO / ".omx" / "state" / "main_hot_state.md"
_LOCK_PATH = _REPO / ".omx" / "state" / ".main_hot_state.lock"

#: The canonical typed sections, in render order, each with a one-line purpose used as the
#: seed placeholder. Editing a section that is NOT in this set is refused (a section is a
#: typed queue, never an ad-hoc key-space — P1 one fact/one store/one key).
SECTIONS: dict[str, str] = {
    "pointer_line": "the exact-score pointer (means/ends firewall — the END first)",
    "live_processes": "pid / out-dir / ETA of any long-running job (do NOT touch its dir)",
    "live_arms": "task-id / name / one-line charter of dispatched subagent arms",
    "open_operator_decisions": "decisions awaiting the operator (GO / config / promotion)",
    "monitor_tasks": "what MAIN is actively watching (a burn knee, an arm landing)",
    "next_boundaries": "the next boundary charter / the next action MAIN takes",
    "freshest_receipts": "paths + SHAs of the freshest load-bearing artifacts",
}

_TITLE = "# MAIN HOT STATE — the orchestrator's retained-reasoning manifest"
_MANAGED = (
    "<!-- managed by tools/main_hot_state.py; sections are typed; "
    "set via --set-section NAME (--content-file F | stdin) -->"
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _header(name: str) -> str:
    return f"## {name.upper()}"


def parse_manifest(text: str) -> dict[str, str]:
    """Parse the manifest markdown into ``{section_name: body_text}`` (lenient).

    A section body is every line after its ``## NAME`` header up to the next ``## `` header
    (or EOF), with surrounding blank lines stripped. Unknown headers are ignored so a manual
    note never corrupts the typed read. Missing sections are absent from the returned dict.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    known_upper = {name.upper(): name for name in SECTIONS}
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("## "):
            key = stripped[3:].strip().upper()
            current = known_upper.get(key)  # None for unknown headers -> body dropped
            if current is not None:
                sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(raw)
    return {name: "\n".join(body).strip() for name, body in sections.items()}


def render_manifest(sections: dict[str, str], *, updated: str | None = None) -> str:
    """Render the full manifest markdown from a section map (missing sections seeded)."""
    parts: list[str] = [_TITLE, _MANAGED, f"_updated: {updated or _now_iso()}_", ""]
    for name, purpose in SECTIONS.items():
        parts.append(_header(name))
        body = (sections.get(name) or "").strip()
        if not body:
            body = f"_(empty — {purpose})_"
        parts.append(body)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _read_text() -> str:
    try:
        return MANIFEST_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def read_sections() -> dict[str, str]:
    """Read + parse the current manifest (empty dict if absent/unreadable). Fail-open."""
    text = _read_text()
    return parse_manifest(text) if text else {}


def _write_atomic(text: str) -> None:
    """Atomic tmp+rename write of the full manifest text (no lock; callers hold it)."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_name(MANIFEST_PATH.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, MANIFEST_PATH)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _locked_read_modify_write(mutate) -> None:
    """Run ``mutate(sections)`` and persist, with the ENTIRE read-modify-write under one
    exclusive fcntl lock (Catalog #131 pattern) so two concurrent ``--set-section`` calls
    cannot both read the old state and clobber each other's section. Off-POSIX (no fcntl)
    falls back to an unlocked read-modify-write (single-writer assumption)."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl

        with open(_LOCK_PATH, "a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                sections = read_sections()  # inside the lock — no TOCTOU
                mutate(sections)
                _write_atomic(render_manifest(sections))
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover - non-POSIX fallback
        sections = read_sections()
        mutate(sections)
        _write_atomic(render_manifest(sections))


def set_section(name: str, content: str) -> None:
    """Replace one typed section's body, preserving all others; atomic + locked RMW."""
    key = name.strip().lower()
    if key not in SECTIONS:
        raise ValueError(
            f"unknown section {name!r} — typed sections are: {', '.join(SECTIONS)}"
        )
    _locked_read_modify_write(lambda sections: sections.__setitem__(key, content.strip()))


def seed_if_absent(force: bool = False) -> bool:
    """Write the initial manifest (all sections placeholder) if it does not exist.

    Returns True if a file was written. ``force`` re-seeds even when present (placeholders
    only overwrite EMPTY sections, so a live manifest is never clobbered)."""
    if MANIFEST_PATH.exists() and not force:
        return False
    # preserve any existing non-empty bodies under force; RMW under the lock.
    _locked_read_modify_write(lambda _sections: None)
    return True


def digest_block(max_lines: int = 40) -> str:
    """Compact manifest block for the SessionStart digest (truncated). Fail-open: on ANY
    error returns "" so the caller (costate_digest) can print nothing without a crash."""
    try:
        text = _read_text()
        if not text.strip():
            return ""
        lines = text.splitlines()
        head = lines[:max_lines]
        out = ["[main-hot-state] MAIN retained-reasoning manifest (tools/main_hot_state.py):"]
        out.extend(head)
        if len(lines) > max_lines:
            out.append(f"  ... (+{len(lines) - max_lines} more lines; read the file)")
        return "\n".join(out)
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable section map")
    ap.add_argument(
        "--set-section",
        metavar="NAME",
        help=f"replace one typed section ({', '.join(SECTIONS)})",
    )
    ap.add_argument(
        "--content-file",
        metavar="F",
        help="file with the new section content (else read stdin)",
    )
    ap.add_argument("--seed", action="store_true", help="write the initial manifest if absent")
    ap.add_argument(
        "--force-seed",
        action="store_true",
        help="re-seed empty sections even when the manifest exists",
    )
    args = ap.parse_args(argv)

    if args.seed or args.force_seed:
        wrote = seed_if_absent(force=args.force_seed)
        print(f"[main-hot-state] {'seeded' if wrote else 'already present'}: {MANIFEST_PATH}")
        return 0

    if args.set_section:
        content = (
            Path(args.content_file).read_text(encoding="utf-8")
            if args.content_file
            else sys.stdin.read()
        )
        try:
            set_section(args.set_section, content)
        except ValueError as exc:
            ap.error(str(exc))
        print(f"[main-hot-state] set section {args.set_section.lower()!r} in {MANIFEST_PATH}")
        return 0

    if args.json:
        print(json.dumps(read_sections(), indent=2, sort_keys=True))
        return 0

    text = _read_text()
    if not text.strip():
        print(
            "[main-hot-state] manifest is empty/absent — seed it with "
            "`tools/main_hot_state.py --seed`"
        )
        return 0
    print(text.rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
