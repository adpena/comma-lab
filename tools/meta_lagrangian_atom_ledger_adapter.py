"""Meta-Lagrangian ↔ atom ledger adapter (priority #5).

Bridge between `experiments/results/bilevel_atom_ledger.jsonl` (continual-
learning ledger written by `tools.run_bilevel_optimization` and
`tools.auto_promote_contest_cuda`) and the existing meta-Lagrangian
search engine (`tools.meta_lagrangian_search_cli`).

The bilevel-optimization driver appends one record per phase iteration:

    {
      "phase": int,
      "substrate_label": str,
      "contest_cuda_score": float | None,
      "evidence_grade": "[contest-CUDA]" | "[CPU-prep]",
      "cathedral_op": str,
      "archive_bytes": int | None,
      ...
    }

The meta-Lagrangian search engine consumes "atoms" (candidate codec /
architecture / paradigm modifications) and produces a ranking. This
adapter:

  1. Reads all ledger records.
  2. Converts each `[contest-CUDA]` entry into a meta-Lagrangian "atom"
     with empirical (rate, score) coordinates.
  3. Emits an atom-ledger JSON file in the format
     `tools.meta_lagrangian_search_cli` expects.
  4. (Optional) provides a Pareto-non-dominated filter so the meta-
     Lagrangian only considers atoms on the empirical frontier.

Strict-scorer-rule: pure CPU + json. No scorer load. All atoms tagged
with their original evidence_grade.

Cross-references:

- Bilevel driver: `tools/run_bilevel_optimization.py`
- Auto-promotion: `tools/auto_promote_contest_cuda.py`
- Meta-Lagrangian search: `tools/meta_lagrangian_search_cli.py`
- Atom ledger location: `experiments/results/bilevel_atom_ledger.jsonl`
- Council prescription: `.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md`
"""

import argparse
import datetime as _dt
import json
import pathlib
import sys
from dataclasses import asdict, dataclass


# ---------------------------------------------------------------------------
# Atom record (the meta-Lagrangian's input format)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetaLagrangianAtom:
    """One atom in the meta-Lagrangian search space.

    Fields:
        atom_id: unique identifier (derived from ledger record).
        cathedral_op: which CodecOp / paradigm produced this atom.
        substrate_label: which substrate it operates on.
        rate_bytes: empirical archive bytes.
        score: empirical contest-CUDA score (if available).
        evidence_grade: ``[contest-CUDA]`` or ``[CPU-prep]`` or ``[predicted]``.
        archive_sha256: byte-content reference for replay.
        notes: free-form forensic context.
    """
    atom_id: str
    cathedral_op: str
    substrate_label: str
    rate_bytes: int | None
    score: float | None
    evidence_grade: str
    archive_sha256: str | None
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Reader / converter
# ---------------------------------------------------------------------------

def read_atom_ledger(ledger_path: pathlib.Path) -> list[dict]:
    """Read JSONL ledger; return list of records (one per line)."""
    if not ledger_path.exists():
        return []
    records: list[dict] = []
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def record_to_atom(record: dict, idx: int) -> MetaLagrangianAtom:
    """Convert a ledger record to a meta-Lagrangian atom."""
    phase = record.get("phase", "?")
    substrate = (
        record.get("substrate_label")
        or record.get("substrate_path")
        or record.get("lane_id")
        or "unknown"
    )
    cathedral_op = record.get("cathedral_op") or record.get("tool") or "unknown"
    score = record.get("contest_cuda_score")
    rate = record.get("archive_bytes")
    sha = record.get("archive_sha256")
    grade = record.get("evidence_grade", "[predicted]")
    atom_id = f"phase{phase}_{substrate.replace(' ', '_').replace('/', '_')[:40]}_{idx}"
    return MetaLagrangianAtom(
        atom_id=atom_id,
        cathedral_op=str(cathedral_op),
        substrate_label=str(substrate)[:100],
        rate_bytes=int(rate) if isinstance(rate, (int, float)) else None,
        score=float(score) if isinstance(score, (int, float)) else None,
        evidence_grade=str(grade),
        archive_sha256=str(sha) if sha else None,
        notes=str(record.get("notes", ""))[:200],
    )


def ledger_to_atoms(ledger_path: pathlib.Path) -> list[MetaLagrangianAtom]:
    return [
        record_to_atom(rec, idx)
        for idx, rec in enumerate(read_atom_ledger(ledger_path))
    ]


# ---------------------------------------------------------------------------
# Pareto-non-dominated filter
# ---------------------------------------------------------------------------

def filter_pareto_non_dominated(
    atoms: list[MetaLagrangianAtom],
) -> list[MetaLagrangianAtom]:
    """Filter atoms to the Pareto-non-dominated set in (rate, score) space.

    An atom A is DOMINATED by B if both:
      - B.rate_bytes <= A.rate_bytes (smaller or equal archive)
      - B.score      <= A.score      (smaller or equal score is better)
    AND at least one is strictly less.

    Atoms with missing rate or score are excluded from the comparison
    (they cannot be placed on the (rate, score) plane).
    """
    valid = [
        a for a in atoms
        if a.rate_bytes is not None and a.score is not None
    ]
    out: list[MetaLagrangianAtom] = []
    for i, a in enumerate(valid):
        dominated = False
        for j, b in enumerate(valid):
            if i == j:
                continue
            if (
                b.rate_bytes <= a.rate_bytes
                and b.score <= a.score
                and (b.rate_bytes < a.rate_bytes or b.score < a.score)
            ):
                dominated = True
                break
        if not dominated:
            out.append(a)
    return out


# ---------------------------------------------------------------------------
# Output (the format `meta_lagrangian_search_cli` expects)
# ---------------------------------------------------------------------------

def emit_meta_lagrangian_input(
    atoms: list[MetaLagrangianAtom],
    output_path: pathlib.Path,
    *,
    pareto_only: bool = False,
) -> dict[str, object]:
    """Emit the atoms in the meta-Lagrangian-search format."""
    if pareto_only:
        atoms = filter_pareto_non_dominated(atoms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": "tools/meta_lagrangian_atom_ledger_adapter",
        "evidence_grade": "[empirical]",
        "score_claim": False,
        "pareto_only": pareto_only,
        "n_atoms": len(atoms),
        "atoms": [a.to_dict() for a in atoms],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Meta-Lagrangian atom ledger adapter")
    p.add_argument(
        "--ledger",
        default=None,
        help="path to bilevel_atom_ledger.jsonl (default: experiments/results/bilevel_atom_ledger.jsonl)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="output path for meta-Lagrangian-format JSON (default: lane_meta_lagrangian_atoms_<UTC>/atoms.json)",
    )
    p.add_argument(
        "--pareto-only",
        action="store_true",
        help="emit only Pareto-non-dominated atoms",
    )
    p.add_argument(
        "--repo-root",
        default=None,
    )
    args = p.parse_args(argv)

    repo_root = pathlib.Path(args.repo_root) if args.repo_root else pathlib.Path(__file__).resolve().parent.parent
    ledger_path = pathlib.Path(args.ledger) if args.ledger else (
        repo_root / "experiments/results/bilevel_atom_ledger.jsonl"
    )
    if args.output is None:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = repo_root / f"experiments/results/lane_meta_lagrangian_atoms_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "atoms.json"
    else:
        output_path = pathlib.Path(args.output)

    atoms = ledger_to_atoms(ledger_path)
    print(f"[adapter] read {len(atoms)} atoms from {ledger_path}")
    if args.pareto_only:
        non_dom = filter_pareto_non_dominated(atoms)
        print(f"[adapter] Pareto-non-dominated subset: {len(non_dom)} of {len(atoms)}")

    payload = emit_meta_lagrangian_input(atoms, output_path, pareto_only=args.pareto_only)
    print(f"[adapter] emitted {payload['n_atoms']} atoms to {output_path}")
    print()
    print("Top 10 atoms (by score, breaking ties by rate):")
    sortable = [a for a in payload["atoms"] if a.get("score") is not None]
    sortable.sort(key=lambda a: (a["score"], a.get("rate_bytes") or float("inf")))
    for a in sortable[:10]:
        print(f"  {a['atom_id']:<60s}  score={a['score']:<10}  rate={a.get('rate_bytes', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
