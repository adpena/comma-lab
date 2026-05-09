#!/usr/bin/env python3
"""Adapt bilevel atom-ledger JSONL into meta-Lagrangian candidate rows.

This is a narrow bridge between ``experiments/results/bilevel_atom_ledger.jsonl``
and ``tools/meta_lagrangian_search_cli.py``.  It does not run a scorer, spend
GPU, or claim scores.  It reads ledger records, preserves evidence/custody
metadata in an auditable row envelope, and can write a strict JSON list accepted
by ``meta_lagrangian_search_cli.py --candidates-json``.

Fail-closed policy:
  * malformed JSONL raises ``LedgerFormatError`` by default;
  * rows missing the fields required by the search CLI are retained in the
    report but excluded from the candidate JSON;
  * local archive custody is checked when an archive path is present.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import pathlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = pathlib.Path("experiments/results/bilevel_atom_ledger.jsonl")
TOOL_NAME = "tools/meta_lagrangian_atom_ledger_adapter"
SEARCH_CANDIDATE_KEYS = (
    "candidate_id",
    "archive_bytes",
    "rel_err_pct",
    "n_layers",
    "lane_class",
    "archive_path",
)
EXPLICIT_BYTE_DELTA_KEYS = (
    "byte_delta_vs_substrate_anchor",
    "archive_byte_delta_vs_substrate_anchor",
    "archive_byte_delta",
    "byte_delta",
    "delta_bytes_vs_substrate",
    "delta_bytes_vs_source_archive",
)


class LedgerFormatError(ValueError):
    """Raised when a JSONL ledger cannot be trusted as adapter input."""


@dataclass(frozen=True)
class MetaLagrangianAtom:
    """One ledger-derived row plus its strict search-candidate projection.

    The first eight fields preserve the earlier adapter's public shape.  The
    extra fields carry the fail-closed bridge metadata requested by the operator.
    """

    atom_id: str
    cathedral_op: str
    substrate_label: str
    rate_bytes: int | None
    score: float | None
    evidence_grade: str
    archive_sha256: str | None
    notes: str
    phase: int | str | None = None
    substrate_path: str | None = None
    lane_id: str | None = None
    archive_path: str | None = None
    evidence_path: str | None = None
    rel_err_pct: float | None = None
    n_layers: int | None = None
    lane_class: str | None = None
    substrate_score_anchor: float | None = None
    substrate_archive_bytes: int | None = None
    score_delta_vs_substrate_anchor: float | None = None
    byte_delta_vs_substrate_anchor: int | None = None
    archive_custody: dict[str, object] = field(default_factory=dict)
    fail_closed: bool = True
    fail_closed_status: str = "blocked"
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    ready_for_meta_lagrangian_search: bool = False
    ready_for_exact_eval_dispatch: bool = False
    score_claim: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_search_candidate(self) -> dict[str, object]:
        """Return the exact key set accepted by ``MetaLagrangianSearch``."""
        if not self.ready_for_meta_lagrangian_search:
            raise ValueError(
                f"{self.atom_id} is fail-closed and cannot be emitted as a "
                "meta-lagrangian search candidate"
            )
        if self.rate_bytes is None or self.rel_err_pct is None or self.n_layers is None:
            raise ValueError(f"{self.atom_id} is missing search-candidate fields")
        candidate: dict[str, object] = {
            "candidate_id": self.atom_id,
            "archive_bytes": self.rate_bytes,
            "rel_err_pct": self.rel_err_pct,
            "n_layers": self.n_layers,
            "lane_class": self.lane_class or "bilevel_atom_ledger",
        }
        if self.archive_path:
            candidate["archive_path"] = self.archive_path
        return candidate


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _first_present(record: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _optional_float(record: dict[str, Any], *keys: str) -> float | None:
    value = _first_present(record, *keys)
    if _is_finite_number(value):
        return float(value)
    return None


def _optional_int(record: dict[str, Any], *keys: str) -> int | None:
    value = _first_present(record, *keys)
    if not _is_finite_number(value):
        return None
    as_float = float(value)
    if not as_float.is_integer():
        return None
    return int(as_float)


def _require_float(
    record: dict[str, Any],
    field_name: str,
    blockers: list[str],
    *keys: str,
) -> float | None:
    value = _first_present(record, *keys)
    if value is None:
        blockers.append(f"missing_{field_name}")
        return None
    if not _is_finite_number(value):
        blockers.append(f"invalid_{field_name}")
        return None
    return float(value)


def _require_int(
    record: dict[str, Any],
    field_name: str,
    blockers: list[str],
    *keys: str,
) -> int | None:
    value = _first_present(record, *keys)
    if value is None:
        blockers.append(f"missing_{field_name}")
        return None
    if not _is_finite_number(value) or not float(value).is_integer():
        blockers.append(f"invalid_{field_name}")
        return None
    return int(float(value))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(path_value: str, repo_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: pathlib.Path, repo_root: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _slug(value: object, *, fallback: str) -> str:
    text = str(value or fallback)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return slug[:80] or fallback


def _archive_custody(
    record: dict[str, Any],
    *,
    repo_root: pathlib.Path,
    archive_path: str | None,
    archive_bytes: int | None,
    archive_sha256: str | None,
    blockers: list[str],
) -> dict[str, object]:
    custody: dict[str, object] = {
        "archive_path": archive_path,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "evidence_path": _string_or_none(record.get("evidence_path")),
        "lane_dir": _string_or_none(record.get("lane_dir")),
        "has_archive_path": archive_path is not None,
        "has_archive_bytes": archive_bytes is not None,
        "has_archive_sha256": archive_sha256 is not None,
        "local_archive_checked": False,
        "local_archive_exists": None,
        "local_archive_sha256": None,
        "local_archive_bytes": None,
    }

    if archive_bytes is None:
        blockers.append("missing_archive_custody_bytes")
    if archive_sha256 is None:
        blockers.append("missing_archive_custody_sha256")
    elif not re.fullmatch(r"[0-9a-fA-F]{64}", archive_sha256):
        blockers.append("invalid_archive_custody_sha256")

    if archive_path is None:
        blockers.append("missing_archive_custody_path")
        return custody

    resolved = _resolve_repo_path(archive_path, repo_root)
    custody["resolved_archive_path"] = _display_path(resolved, repo_root)
    custody["local_archive_checked"] = True
    custody["local_archive_exists"] = resolved.is_file()
    if not resolved.is_file():
        blockers.append("archive_custody_path_not_found")
        return custody

    local_bytes = resolved.stat().st_size
    custody["local_archive_bytes"] = local_bytes
    if archive_bytes is not None and archive_bytes != local_bytes:
        blockers.append("archive_custody_bytes_mismatch")

    local_sha = _sha256_file(resolved)
    custody["local_archive_sha256"] = local_sha
    if archive_sha256 is not None and archive_sha256.lower() != local_sha:
        blockers.append("archive_custody_sha256_mismatch")
    return custody


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _score_delta(record: dict[str, Any], score: float | None) -> float | None:
    explicit = _optional_float(
        record,
        "score_delta_vs_substrate_anchor",
        "score_delta",
        "contest_cuda_score_delta",
    )
    if explicit is not None:
        return explicit
    anchor = _optional_float(record, "substrate_score_anchor", "baseline_score")
    if score is None or anchor is None:
        return None
    return score - anchor


def _byte_delta(
    record: dict[str, Any],
    archive_bytes: int | None,
) -> tuple[int | None, int | None]:
    explicit = _optional_int(record, *EXPLICIT_BYTE_DELTA_KEYS)
    anchor_bytes = _optional_int(
        record,
        "substrate_archive_bytes",
        "source_archive_bytes",
        "baseline_archive_bytes",
    )
    if explicit is not None:
        return explicit, anchor_bytes
    if archive_bytes is None or anchor_bytes is None:
        return None, anchor_bytes
    return archive_bytes - anchor_bytes, anchor_bytes


def read_atom_ledger(
    ledger_path: pathlib.Path,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Read a JSONL atom ledger.

    ``strict=True`` is fail-closed: corrupt lines or non-object JSON records
    raise ``LedgerFormatError`` instead of being silently skipped.
    """
    if not ledger_path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(ledger_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise LedgerFormatError(
                    f"{ledger_path}:{line_no}: invalid JSONL record: {exc.msg}"
                ) from exc
            continue
        if not isinstance(raw, dict):
            if strict:
                raise LedgerFormatError(
                    f"{ledger_path}:{line_no}: expected JSON object, got "
                    f"{type(raw).__name__}"
                )
            continue
        records.append(raw)
    return records


def record_to_atom(
    record: dict[str, Any],
    idx: int,
    *,
    repo_root: pathlib.Path | None = None,
) -> MetaLagrangianAtom:
    """Convert one bilevel ledger record into a fail-closed adapter row."""
    root = repo_root or REPO_ROOT
    blockers: list[str] = []
    warnings: list[str] = []

    phase = record.get("phase")
    lane_id = _string_or_none(record.get("lane_id"))
    substrate = _string_or_none(
        _first_present(record, "substrate_label", "substrate_path", "lane_id")
    ) or "unknown_substrate"
    substrate_path = _string_or_none(record.get("substrate_path"))
    cathedral_op = _string_or_none(
        _first_present(record, "cathedral_op", "tool")
    ) or "unknown"
    notes = (_string_or_none(record.get("notes")) or "")[:500]
    evidence_grade = _string_or_none(record.get("evidence_grade"))
    if evidence_grade is None:
        blockers.append("missing_evidence_grade")
        evidence_grade = "[unknown]"

    archive_sha256 = _string_or_none(record.get("archive_sha256"))
    archive_path = _string_or_none(record.get("archive_path"))

    score = _optional_float(record, "contest_cuda_score", "score")
    if (
        evidence_grade.lower() in {"[contest-cuda]", "contest-cuda", "a", "a++"}
        and score is None
    ):
        blockers.append("missing_contest_cuda_score_for_evidence_grade")
    if evidence_grade.lower() in {"[contest-cuda]", "contest-cuda", "a", "a++"}:
        if archive_sha256 is None:
            blockers.append("missing_archive_sha256_for_evidence_grade")

    archive_bytes = _require_int(
        record,
        "archive_bytes",
        blockers,
        "archive_bytes",
        "rate_bytes",
    )
    rel_err_pct = _require_float(
        record,
        "rel_err_pct",
        blockers,
        "rel_err_pct",
        "rel_err_pct_per_weight",
        "relative_error_pct",
    )
    n_layers = _require_int(
        record,
        "n_layers",
        blockers,
        "n_layers",
        "n_intn_layers",
        "n_quantized_layers",
    )
    lane_class = _string_or_none(record.get("lane_class")) or "bilevel_atom_ledger"

    score_delta = _score_delta(record, score)
    if score is not None and _optional_float(record, "substrate_score_anchor") is None:
        warnings.append("missing_substrate_score_anchor_for_score_delta")
    byte_delta, substrate_archive_bytes = _byte_delta(record, archive_bytes)
    if archive_bytes is not None and byte_delta is None:
        warnings.append("missing_substrate_archive_bytes_for_byte_delta")

    custody = _archive_custody(
        record,
        repo_root=root,
        archive_path=archive_path,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        blockers=blockers,
    )

    atom_id = _string_or_none(record.get("candidate_id")) or (
        f"phase{phase if phase is not None else 'unknown'}_"
        f"{_slug(lane_id or substrate, fallback='substrate')}_{idx}"
    )
    fail_closed = bool(blockers)
    status = "ready_for_meta_lagrangian_search" if not fail_closed else "blocked"

    return MetaLagrangianAtom(
        atom_id=atom_id,
        cathedral_op=cathedral_op,
        substrate_label=substrate[:200],
        rate_bytes=archive_bytes,
        score=score,
        evidence_grade=evidence_grade,
        archive_sha256=archive_sha256,
        notes=notes,
        phase=phase,
        substrate_path=substrate_path,
        lane_id=lane_id,
        archive_path=archive_path,
        evidence_path=_string_or_none(record.get("evidence_path")),
        rel_err_pct=rel_err_pct,
        n_layers=n_layers,
        lane_class=lane_class,
        substrate_score_anchor=_optional_float(record, "substrate_score_anchor"),
        substrate_archive_bytes=substrate_archive_bytes,
        score_delta_vs_substrate_anchor=score_delta,
        byte_delta_vs_substrate_anchor=byte_delta,
        archive_custody=custody,
        fail_closed=fail_closed,
        fail_closed_status=status,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        ready_for_meta_lagrangian_search=not fail_closed,
        ready_for_exact_eval_dispatch=False,
        score_claim=False,
    )


def ledger_to_atoms(
    ledger_path: pathlib.Path,
    *,
    repo_root: pathlib.Path | None = None,
    strict: bool = True,
) -> list[MetaLagrangianAtom]:
    return [
        record_to_atom(rec, idx, repo_root=repo_root)
        for idx, rec in enumerate(read_atom_ledger(ledger_path, strict=strict))
    ]


def candidates_for_meta_lagrangian_search(
    atoms: list[MetaLagrangianAtom],
) -> list[dict[str, object]]:
    """Return only fail-open rows in the strict CLI candidate shape."""
    candidates: list[dict[str, object]] = []
    for atom in atoms:
        if atom.ready_for_meta_lagrangian_search:
            candidates.append(atom.to_search_candidate())
    return candidates


def filter_pareto_non_dominated(
    atoms: list[MetaLagrangianAtom],
) -> list[MetaLagrangianAtom]:
    """Filter atoms to the Pareto-non-dominated set in (bytes, score)."""
    valid = [
        atom
        for atom in atoms
        if atom.rate_bytes is not None and atom.score is not None
    ]
    out: list[MetaLagrangianAtom] = []
    for i, atom in enumerate(valid):
        dominated = False
        for j, other in enumerate(valid):
            if i == j:
                continue
            if (
                other.rate_bytes <= atom.rate_bytes
                and other.score <= atom.score
                and (
                    other.rate_bytes < atom.rate_bytes
                    or other.score < atom.score
                )
            ):
                dominated = True
                break
        if not dominated:
            out.append(atom)
    return out


def emit_meta_lagrangian_input(
    atoms: list[MetaLagrangianAtom],
    output_path: pathlib.Path,
    *,
    pareto_only: bool = False,
    source_ledger: pathlib.Path | None = None,
    candidates_output_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Emit an auditable row envelope and optional strict CLI candidates."""
    rows = filter_pareto_non_dominated(atoms) if pareto_only else list(atoms)
    candidates = candidates_for_meta_lagrangian_search(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if candidates_output_path is not None:
        candidates_output_path.parent.mkdir(parents=True, exist_ok=True)
        candidates_output_path.write_text(json.dumps(candidates, indent=2))

    payload: dict[str, object] = {
        "schema": "bilevel_atom_ledger_meta_lagrangian_rows_v1",
        "started_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": TOOL_NAME,
        "source_ledger": str(source_ledger) if source_ledger is not None else None,
        "score_claim": False,
        "evidence_grade": "[adapter:source_rows_only]",
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "adapter_never_dispatches",
            "meta_lagrangian_search_is_forensic_until_exact_cuda",
        ],
        "pareto_only": pareto_only,
        "n_atoms": len(rows),
        "n_rows": len(rows),
        "ready_candidate_count": len(candidates),
        "fail_closed_count": len([row for row in rows if row.fail_closed]),
        "candidate_output_format": (
            "JSON list accepted by tools/meta_lagrangian_search_cli.py "
            "--candidates-json"
        ),
        "candidate_keys": list(SEARCH_CANDIDATE_KEYS),
        "candidates_for_meta_lagrangian_search_cli": candidates,
        "atoms": [row.to_dict() for row in rows],
        "rows": [row.to_dict() for row in rows],
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def _default_output_paths(
    repo_root: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = repo_root / f"experiments/results/lane_meta_lagrangian_atoms_{ts}"
    return out_dir / "atoms.json", out_dir / "candidates.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adapt bilevel_atom_ledger.jsonl for meta-Lagrangian search",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help=(
            "path to bilevel_atom_ledger.jsonl "
            "(default: experiments/results/bilevel_atom_ledger.jsonl)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "auditable row-envelope JSON output "
            "(default: lane_meta_lagrangian_atoms_<UTC>/atoms.json)"
        ),
    )
    parser.add_argument(
        "--candidates-output",
        default=None,
        help=(
            "strict candidate JSON list accepted by "
            "tools/meta_lagrangian_search_cli.py --candidates-json"
        ),
    )
    parser.add_argument("--pareto-only", action="store_true")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--allow-corrupt-lines",
        action="store_true",
        help="skip corrupt/non-object JSONL records instead of failing closed",
    )
    args = parser.parse_args(argv)

    repo_root = pathlib.Path(args.repo_root) if args.repo_root else REPO_ROOT
    ledger_path = pathlib.Path(args.ledger) if args.ledger else repo_root / DEFAULT_LEDGER

    default_output, default_candidates = _default_output_paths(repo_root)
    output_path = pathlib.Path(args.output) if args.output else default_output
    candidates_output = (
        pathlib.Path(args.candidates_output)
        if args.candidates_output
        else default_candidates
    )

    atoms = ledger_to_atoms(
        ledger_path,
        repo_root=repo_root,
        strict=not args.allow_corrupt_lines,
    )
    payload = emit_meta_lagrangian_input(
        atoms,
        output_path,
        pareto_only=args.pareto_only,
        source_ledger=ledger_path,
        candidates_output_path=candidates_output,
    )

    print(f"[adapter] read {len(atoms)} rows from {ledger_path}")
    print(
        f"[adapter] ready candidates: {payload['ready_candidate_count']} "
        f"(fail-closed rows: {payload['fail_closed_count']})"
    )
    print(f"[adapter] rows -> {output_path}")
    print(f"[adapter] candidates -> {candidates_output}")
    if int(payload["ready_candidate_count"]) == 0:
        print("[adapter] no search candidates emitted; all rows are fail-closed")
    else:
        print(
            "[adapter] next local forensic step: "
            "tools/meta_lagrangian_search_cli.py --candidates-json "
            f"{candidates_output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
