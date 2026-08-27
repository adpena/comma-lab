#!/usr/bin/env python
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Stop hook: keep AGENTS.md a byte-identical copy of CLAUDE.md.

Operator directive 2026-07-25 ("make it a stop hook that recopies on CLAUDE.md
change"). Rationale: subagent pre-read mandates BOTH files; codex arms read
AGENTS.md as PLAIN markdown, so the @CLAUDE.md import-pointer form (Claude-Code
memory loader only) left them a stub, and a manual copy drifted 14 days stale
(Jul-11 177KB vs Jul-24 424KB). This hook extincts the drift class structurally.

Behavior (fail-open — a Stop hook must NEVER block the turn):
  - identical            -> exit 0, silent
  - drifted              -> copy CLAUDE.md over AGENTS.md, commit via the
                            canonical serializer (Catalog #117/#157), print one line
  - any error            -> print a WARN line, exit 0
Ordering contract: registered BEFORE tools/auto_push_main.py in the Stop array
so the resync commit rides the same push cycle.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLAUDE_MD = REPO / "CLAUDE.md"
AGENTS_MD = REPO / "AGENTS.md"
SERIALIZER = REPO / "tools" / "subagent_commit_serializer.py"
COMMIT_MSG = (
    "AGENTS.md: auto-resync to CLAUDE.md (sync_agents_md Stop hook; operator "
    "2026-07-25) [no-triality] [p0-ledger-ok]"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    try:
        if not CLAUDE_MD.is_file():
            print(f"[sync-agents-md] WARN: {CLAUDE_MD} missing; no-op")
            return 0
        src_sha = _sha256(CLAUDE_MD)
        if AGENTS_MD.is_file() and _sha256(AGENTS_MD) == src_sha:
            # In sync on disk — but retry the commit if a prior run's serializer
            # refusal left the (correct) copy uncommitted.
            status = subprocess.run(
                ["git", "status", "--porcelain", "--", "AGENTS.md"],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=30,
            )
            # rc gate: a FAILED git status must not masquerade as "in sync +
            # committed" — fall through to the serializer, which fails loudly.
            if status.returncode == 0 and not status.stdout.strip():
                return 0  # in sync + committed — silent
        else:
            shutil.copyfile(CLAUDE_MD, AGENTS_MD)
        if _sha256(AGENTS_MD) != src_sha:
            print("[sync-agents-md] WARN: copy verification mismatch; leaving working tree as-is")
            return 0
        proc = subprocess.run(
            [
                sys.executable,
                str(SERIALIZER),
                "--message",
                COMMIT_MSG,
                "--files",
                "AGENTS.md",
                "--expected-content-sha256",
                f"AGENTS.md={src_sha}",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            print(f"[sync-agents-md] AGENTS.md re-copied from CLAUDE.md + committed (sha {src_sha[:12]})")
        else:
            # Copy is on disk (correct content); commit refusal is surfaced, not fatal.
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            print(
                "[sync-agents-md] WARN: re-copied but serializer commit refused "
                f"(rc={proc.returncode}): {tail[-1] if tail else 'no output'}"
            )
        return 0
    except Exception as exc:  # fail-open by contract: a Stop hook must never block
        print(f"[sync-agents-md] WARN: {type(exc).__name__}: {exc}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
