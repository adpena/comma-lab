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

#: Per-section staleness sidecar (ea1 arc, 2026-08-31): LIVE_PROCESSES sat 4 days past its
#: consumption undetected because the manifest carries ONE file-level `_updated` — any write
#: to any section refreshed the global stamp while sibling sections rotted silently. The
#: sidecar records each section's OWN last-write time; thresholds are derived from each
#: section's natural cadence (live process/arm/monitor state churns daily; the pointer and
#: boundary charters at pointer-move cadence; receipts and operator decisions persist).
_SECTION_TIMES_PATH = _REPO / ".omx" / "state" / ".main_hot_state.section_times.json"
_STALE_DAYS: dict[str, float] = {
    "pointer_line": 3.0,
    "live_processes": 1.0,
    "live_arms": 1.0,
    "open_operator_decisions": 7.0,
    "monitor_tasks": 1.0,
    "next_boundaries": 2.0,
    "freshest_receipts": 7.0,
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_section_times() -> dict[str, str]:
    """Read the per-section timestamp sidecar (empty dict if absent/corrupt). Fail-open."""
    try:
        data = json.loads(_SECTION_TIMES_PATH.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in SECTIONS and isinstance(v, str)}
    except Exception:
        return {}


def _save_section_times(times: dict[str, str]) -> None:
    """Atomic write of the sidecar (callers hold the manifest lock). Fail-open."""
    try:
        tmp = _SECTION_TIMES_PATH.with_name(_SECTION_TIMES_PATH.name + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(times, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, _SECTION_TIMES_PATH)
    except Exception:
        pass


def _age_days(stamp_iso: str) -> float | None:
    try:
        stamp = datetime.fromisoformat(stamp_iso)
        return (datetime.now(UTC) - stamp).total_seconds() / 86400.0
    except Exception:
        return None


def staleness_rows() -> list[dict]:
    """One row per typed section: {section, age_days|None, threshold_days, stale|None}.

    ``stale`` is None for a section never stamped since the sidecar landed — age unknown,
    honestly reported as such rather than assumed fresh (it gets a stamp on its next write).
    """
    times = _load_section_times()
    rows: list[dict] = []
    for name in SECTIONS:
        threshold = _STALE_DAYS.get(name, 7.0)
        stamp = times.get(name)
        age = _age_days(stamp) if stamp else None
        rows.append(
            {
                "section": name,
                "age_days": round(age, 2) if age is not None else None,
                "threshold_days": threshold,
                "stale": (age > threshold) if age is not None else None,
            }
        )
    return rows


def _staleness_banner() -> str:
    """One-line banner naming stale/unstamped sections ('' when all fresh). Fail-open."""
    try:
        stale = [
            f"{r['section']}({r['age_days']}d>{r['threshold_days']}d)"
            for r in staleness_rows()
            if r["stale"]
        ]
        unstamped = [r["section"] for r in staleness_rows() if r["stale"] is None]
        parts: list[str] = []
        if stale:
            parts.append("STALE: " + ", ".join(stale))
        if unstamped:
            parts.append("unstamped (age unknown): " + ", ".join(unstamped))
        return " · ".join(parts)
    except Exception:
        return ""


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
    parts: list[str] = [_TITLE, _MANAGED, f"_updated: {updated or _now_iso()}_"]
    banner = _staleness_banner()
    if banner:
        parts.append(f"_staleness: {banner}_")
    parts.append("")
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
    def _mutate(sections: dict[str, str]) -> None:
        sections[key] = content.strip()
        times = _load_section_times()
        times[key] = _now_iso()
        _save_section_times(times)

    _locked_read_modify_write(_mutate)


def seed_if_absent(force: bool = False) -> bool:
    """Write the initial manifest (all sections placeholder) if it does not exist.

    Returns True if a file was written. ``force`` re-seeds even when present (placeholders
    only overwrite EMPTY sections, so a live manifest is never clobbered)."""
    if MANIFEST_PATH.exists() and not force:
        return False
    # preserve any existing non-empty bodies under force; RMW under the lock.
    _locked_read_modify_write(lambda _sections: None)
    return True


def digest_block(max_lines: int = 40, max_line_chars: int | None = None) -> str:
    """Compact manifest block for the SessionStart digest (truncated). Fail-open: on ANY
    error returns "" so the caller (costate_digest) can print nothing without a crash.

    ``max_line_chars`` (ddm_gh2, 2026-07-31) caps each line's WIDTH.  ``max_lines``
    alone bounded nothing that mattered: this manifest holds free-text paragraphs
    (MEASURED 2026-07-31: single lines of 1,047 / 1,011 / 635 chars, 3,851 B total
    for 40 lines), so a line-COUNT budget on a file with mega-lines is a cap that
    looks binding and is not.  ``None`` (the default) preserves the historical
    behaviour exactly, so existing callers are unaffected; the SessionStart hook
    passes a width so the recurring payload is actually bounded.  Truncated lines
    are marked with a visible ellipsis + the dropped-char count — never silently."""
    try:
        text = _read_text()
        if not text.strip():
            return ""
        lines = text.splitlines()
        head = lines[:max_lines]
        if max_line_chars is not None and max_line_chars > 0:
            head = [
                ln
                if len(ln) <= max_line_chars
                else f"{ln[:max_line_chars]}… (+{len(ln) - max_line_chars} chars; read the file)"
                for ln in head
            ]
        out = ["[main-hot-state] MAIN retained-reasoning manifest (tools/main_hot_state.py):"]
        live_banner = _staleness_banner()
        if "STALE:" in live_banner:
            out.append(f"  ⚠ section staleness — {live_banner}")
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
    ap.add_argument(
        "--staleness",
        action="store_true",
        help="per-section age vs cadence threshold; rc=1 when any section is stale",
    )
    ap.add_argument("--seed", action="store_true", help="write the initial manifest if absent")
    ap.add_argument(
        "--force-seed",
        action="store_true",
        help="re-seed empty sections even when the manifest exists",
    )
    args = ap.parse_args(argv)

    if args.staleness:
        rows = staleness_rows()
        any_stale = False
        for r in rows:
            if r["stale"]:
                mark, any_stale = "STALE", True
            elif r["stale"] is None:
                mark = "unstamped"
            else:
                mark = "fresh"
            age = f"{r['age_days']}d" if r["age_days"] is not None else "?"
            print(f"[main-hot-state] {r['section']:24s} {age:>8s} / {r['threshold_days']}d  {mark}")
        return 1 if any_stale else 0

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
