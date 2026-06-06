#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build a PR110 pairwise commutator ledger from ActionEffect v1 JSONL.

Consumes the THIN ``tac.action_effect.v1`` ledger rows (as written by
``tac.analysis.action_effect.append_action_effect`` / produced by swarm-D) and
emits the pairwise non-commutativity ledger: which ordered pairs of actions are
synergistic (macro-action candidates), which conflict, and which still need a
measured composite before any commutator can be computed.

This is an ANALYSIS tool: it computes commutators from REAL measured deltas and
emits typed needs-measurement rows for unmeasured pairs.  It never invents a
commutator value and never claims a contest score (every output carries the
canonical false-authority markers).

Usage::

    run_pr110_commutator_ledger.py \\
        --action-effects path/to/single_effects.jsonl \\
        [--pair-effects path/to/pair_effects.jsonl] \\
        [--output experiments/results/pr110_commutator_ledger_<ts>] \\
        [--eps 1e-9] [--top-k 16]

Outputs (under ``--output``, default ``experiments/results/...`` per CLAUDE.md
"Forbidden /tmp paths in any persisted artifact"):

* ``commutator_ledger.jsonl`` — one JSON row per measured commutator + one per
  needs-measurement entry (schema-tagged; round-trippable).
* ``commutator_summary.json`` — counts + the macro-action / conflict / queue
  views.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from tac.analysis.action_commutator import (
    ACTION_COMMUTATOR_LEDGER_SCHEMA,
    build_commutator_ledger,
)
from tac.analysis.action_effect import ActionEffect, read_action_effects


def _read_effects(path: Path) -> list[ActionEffect]:
    """Read ActionEffect v1 rows from a JSONL ledger.

    Delegates to the canonical reader (which skips malformed lines and rebuilds
    via ``ActionEffect.from_dict``) so the input contract matches exactly what
    the producer ledger emits.
    """

    if not path.exists():
        raise FileNotFoundError(f"action-effects ledger not found: {path}")
    return read_action_effects(path)


def _write_ledger_jsonl(ledger: dict, path: Path) -> int:
    """Write measured commutator rows + needs-measurement rows as JSONL.

    Returns the number of rows written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as out:
        for row in ledger["rows"]:
            out.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
        for row in ledger["measurement_queue"]:
            out.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
    return written


def _write_summary_json(ledger: dict, path: Path, *, top_k: int) -> None:
    """Write the counts + capped macro/conflict/queue views as a summary JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": ACTION_COMMUTATOR_LEDGER_SCHEMA,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "single_effect_count": ledger["single_effect_count"],
        "pair_effect_count": ledger["pair_effect_count"],
        "ordered_pair_count": ledger["ordered_pair_count"],
        "measured_commutator_count": ledger["measured_commutator_count"],
        "synergistic_count": ledger["synergistic_count"],
        "conflicting_count": ledger["conflicting_count"],
        "additive_count": ledger["additive_count"],
        "needs_measurement_count": ledger["needs_measurement_count"],
        "eps": ledger["eps"],
        "macro_action_candidates": ledger["macro_action_candidates"][: max(0, top_k)],
        "conflict_pairs": ledger["conflict_pairs"][: max(0, top_k)],
        "measurement_queue": ledger["measurement_queue"][: max(0, top_k)],
        "policy": ledger["policy"],
        # Carry the false-authority markers onto the summary so a downstream
        # reader cannot mistake this for a score-claim artifact.
        **{k: v for k, v in ledger.items() if isinstance(v, bool)},
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_output_dir() -> Path:
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("experiments/results") / f"pr110_commutator_ledger_{ts}"


def _print_human(ledger: dict, out_dir: Path) -> None:
    print(f"[pr110-commutator] single effects: {ledger['single_effect_count']}")
    print(f"[pr110-commutator] pair (composite) effects: {ledger['pair_effect_count']}")
    print(f"[pr110-commutator] ordered pairs possible: {ledger['ordered_pair_count']}")
    print(
        "[pr110-commutator] measured commutators: "
        f"{ledger['measured_commutator_count']} "
        f"(synergistic={ledger['synergistic_count']} "
        f"conflicting={ledger['conflicting_count']} "
        f"additive={ledger['additive_count']})"
    )
    print(f"[pr110-commutator] needs-measurement queue: {ledger['needs_measurement_count']}")
    macros = ledger["macro_action_candidates"]
    if macros:
        print("[pr110-commutator] top macro-action candidates (most synergistic):")
        for row in macros[:10]:
            print(
                f"    {row['first_action_id']} -> {row['second_action_id']}  "
                f"comm={row['comm']:+.6g} synergy={row['synergy_score_units']:+.6g} "
                f"basis={row['basis']} authority={row['authority']}"
            )
    conflicts = ledger["conflict_pairs"]
    if conflicts:
        print("[pr110-commutator] top conflict pairs:")
        for row in conflicts[:10]:
            print(
                f"    {row['first_action_id']} -> {row['second_action_id']}  "
                f"comm={row['comm']:+.6g} basis={row['basis']} authority={row['authority']}"
            )
    print(f"[pr110-commutator] outputs written under: {out_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--action-effects",
        required=True,
        type=Path,
        help="JSONL of single-action tac.action_effect.v1 rows.",
    )
    parser.add_argument(
        "--pair-effects",
        type=Path,
        default=None,
        help="Optional JSONL of measured composite (pair) tac.action_effect.v1 rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default experiments/results/pr110_commutator_ledger_<utc>).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-9,
        help="Additive band (score units) for classification (default 1e-9).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Cap on macro-action / conflict / queue views in the summary (default 16).",
    )
    args = parser.parse_args(argv)

    if args.eps < 0.0:
        parser.error("--eps must be non-negative")

    singles = _read_effects(args.action_effects)
    pairs = _read_effects(args.pair_effects) if args.pair_effects is not None else []

    ledger = build_commutator_ledger(
        singles,
        pairs,
        eps=args.eps,
        macro_action_limit=max(args.top_k, 0),
        conflict_pair_limit=max(args.top_k, 0),
    )

    out_dir = args.output if args.output is not None else _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "commutator_ledger.jsonl"
    summary_path = out_dir / "commutator_summary.json"
    _write_ledger_jsonl(ledger, jsonl_path)
    _write_summary_json(ledger, summary_path, top_k=args.top_k)
    _print_human(ledger, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
