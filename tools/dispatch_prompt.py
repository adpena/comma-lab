#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compose a subagent dispatch prompt from the canonical contract (#338).

Makes ``tac.subagent_contract.standard_contract()`` the path of least
resistance for dispatchers: task text + the harvested contract blocks (which
end with the operating-manual citation) + the serializer commit-discipline
block. Import of the real contract module is REQUIRED — if it fails, this
tool fails LOUD (no silent fallback prompt; a drifted hand-typed prompt is
the exact failure mode the contract module exists to extinct).

Usage:
  tools/dispatch_prompt.py --task-text "Build X ..." [--no-review] [--no-triality]
  echo "Build X ..." | tools/dispatch_prompt.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _import_standard_contract():
    try:
        from tac.subagent_contract import standard_contract
    except ImportError:
        # Not running under the repo venv: give the real module one honest
        # chance via the source tree, then let any failure propagate LOUDLY.
        sys.path.insert(0, str(_REPO / "src"))
        from tac.subagent_contract import standard_contract
    return standard_contract


SERIALIZER_BLOCK = """\
COMMIT DISCIPLINE: commit ONLY the exact files you touched, via
  python tools/subagent_commit_serializer.py --message "<what changed>: <why>" \\
      --files <file1> <file2> ... \\
      --expected-content-sha256 <file>=<post-edit-working-tree-sha256>
Compute each sha AFTER all your edits (shasum -a 256 <file> — the POST-EDIT
working-tree content, NOT the HEAD sha). If the serializer refuses with rc=4,
a sister landed first: re-read the file, re-base your edit, re-hash, retry.
NEVER bare `git commit`. NEVER REVIEW_GATE_OVERRIDE=1 on .py files — run
`python tools/review_tracker.py mark-file <f> --status reviewed` instead."""


def compose(task_text: str, *, review: bool = True, triality: bool = True) -> str:
    standard_contract = _import_standard_contract()
    contract = standard_contract(review=review, triality=triality)
    return f"{task_text.rstrip()}\n\n{contract}\n\n{SERIALIZER_BLOCK}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-text", default=None, help="task body (default: read stdin)")
    ap.add_argument("--no-review", action="store_true", help="omit the own-round-1 review block")
    ap.add_argument("--no-triality", action="store_true", help="omit the triality-wiring block")
    args = ap.parse_args(argv)

    task_text = args.task_text if args.task_text is not None else sys.stdin.read()
    if not task_text.strip():
        ap.error("empty task text (pass --task-text or pipe the task body on stdin)")
    print(compose(task_text, review=not args.no_review, triality=not args.no_triality))
    return 0


if __name__ == "__main__":
    sys.exit(main())
