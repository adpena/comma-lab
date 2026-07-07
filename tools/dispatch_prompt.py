#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compose a subagent dispatch prompt from the canonical contract (#338).

Makes ``tac.subagent_contract.standard_contract()`` the path of least
resistance for dispatchers: task text + the harvested contract blocks (which
end with the operating-manual citation) + the serializer commit-discipline
block. Import of the real contract module is REQUIRED — if it fails, this
tool fails LOUD (no silent fallback prompt; a drifted hand-typed prompt is
the exact failure mode the contract module exists to extinct).

``--review`` composes a REVIEW dispatch instead: task text +
``review_contract()`` (manual §3/§5/§6/§8 as structure). Review dispatches are
REPORT-ONLY by default (the emitted prompt says so); pass ``--allow-commits``
to append the serializer block for the rare reviewer that also lands fixes.

Usage:
  tools/dispatch_prompt.py --task-text "Build X ..." [--no-review] [--no-triality]
  tools/dispatch_prompt.py --review --task-text "Review commits A B ..." \\
      [--counter-context "surface X: 2 consecutive CLEAN rounds"] [--allow-commits]
  echo "Build X ..." | tools/dispatch_prompt.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _import_contract_module():
    try:
        from tac import subagent_contract
    except ImportError:
        # Not running under the repo venv: give the real module one honest
        # chance via the source tree, then let any failure propagate LOUDLY.
        sys.path.insert(0, str(_REPO / "src"))
        from tac import subagent_contract
    return subagent_contract


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


REPORT_ONLY_BLOCK = """\
REPORT-ONLY DISPATCH: this is a review dispatch — you do NOT commit. Report
findings (each labeled CONFIRMED or PLAUSIBLE, risk-ranked) as your final
message; the author lands any fixes and those fixes get their own review round."""


def compose(task_text: str, *, review: bool = True, triality: bool = True) -> str:
    sc = _import_contract_module()
    contract = sc.standard_contract(review=review, triality=triality)
    return f"{task_text.rstrip()}\n\n{contract}\n\n{SERIALIZER_BLOCK}"


def compose_review(
    task_text: str, *, counter_context: str = "", allow_commits: bool = False
) -> str:
    """Review-dispatch prompt: task + review_contract(); report-only unless allowed."""
    sc = _import_contract_module()
    addendum = sc.review_contract(counter_context=counter_context)
    tail = SERIALIZER_BLOCK if allow_commits else REPORT_ONLY_BLOCK
    return f"{task_text.rstrip()}\n\n{addendum}\n\n{tail}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-text", default=None, help="task body (default: read stdin)")
    ap.add_argument("--no-review", action="store_true", help="omit the own-round-1 review block")
    ap.add_argument("--no-triality", action="store_true", help="omit the triality-wiring block")
    ap.add_argument(
        "--review", action="store_true",
        help="compose a REVIEW dispatch (review_contract(); report-only by default)")
    ap.add_argument(
        "--counter-context", default="",
        help="[--review] clean-pass counter state to prepend (e.g. from tac.review_counter)")
    ap.add_argument(
        "--allow-commits", action="store_true",
        help="[--review] append the serializer block (default review dispatches are report-only)")
    args = ap.parse_args(argv)

    task_text = args.task_text if args.task_text is not None else sys.stdin.read()
    if not task_text.strip():
        ap.error("empty task text (pass --task-text or pipe the task body on stdin)")
    if args.review:
        print(compose_review(
            task_text,
            counter_context=args.counter_context,
            allow_commits=args.allow_commits,
        ))
    else:
        print(compose(task_text, review=not args.no_review, triality=not args.no_triality))
    return 0


if __name__ == "__main__":
    sys.exit(main())
