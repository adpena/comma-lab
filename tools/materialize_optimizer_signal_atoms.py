#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize optimizer planning signals as canonical Atom rows.

The tool consumes an optimizer-guided or optimizer-candidate queue and writes
typed atoms for downstream meta-Lagrangian, Pareto, cathedral/autopilot, and
continual-learning consumers.  It preserves the proxy evidence boundary: no
score, promotion, rank/kill, or dispatch authority is emitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.atom.ledger import append_atoms_batch  # noqa: E402
from tac.optimization.optimizer_signal_atoms import (  # noqa: E402
    OptimizerSignalAtomError,
    build_atoms_from_optimizer_signal_source,
    build_optimizer_signal_atom_ledger,
)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
            f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-queue",
        type=Path,
        required=True,
        help="optimizer-guided or optimizer-candidate queue JSON.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON ledger payload with atoms and meta-lagrangian projections.",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=None,
        help="Optional canonical atom-row JSONL output, one Atom per line.",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Optional cap for smoke materialization.",
    )
    parser.add_argument(
        "--append-canonical-atom-ledger",
        action="store_true",
        help="Append atoms to .omx/state/atom_ledger.jsonl via tac.atom.ledger.",
    )
    args = parser.parse_args(argv)

    try:
        payload = _load_json(args.candidate_queue)
        if not isinstance(payload, dict):
            raise OptimizerSignalAtomError("candidate queue JSON must be an object")
        source_path = args.candidate_queue.as_posix()
        ledger = build_optimizer_signal_atom_ledger(
            payload,
            source_path=source_path,
            max_atoms=args.max_atoms,
        )
        atoms = build_atoms_from_optimizer_signal_source(
            payload,
            source_path=source_path,
            max_atoms=args.max_atoms,
        )
    except (OSError, json.JSONDecodeError, OptimizerSignalAtomError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    if args.json_out is not None:
        _write_json(args.json_out, ledger)
    if args.jsonl_out is not None:
        _write_jsonl(args.jsonl_out, [atom.to_jsonl_row() for atom in atoms])
    appended = []
    if args.append_canonical_atom_ledger:
        appended = append_atoms_batch(atoms)

    destinations = []
    if args.json_out is not None:
        destinations.append(f"json={args.json_out}")
    if args.jsonl_out is not None:
        destinations.append(f"jsonl={args.jsonl_out}")
    if args.append_canonical_atom_ledger:
        destinations.append(f"canonical_ledger_rows={len(appended)}")
    print(
        "materialized optimizer signal atoms "
        f"count={len(atoms)} "
        "score_claim=false promotion_eligible=false "
        + (" ".join(destinations) if destinations else "dry_run=true")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
