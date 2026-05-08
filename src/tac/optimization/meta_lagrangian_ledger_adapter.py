"""Adapters from existing ledger artifacts into meta-Lagrangian atom rows.

The adapter is intentionally conservative: it reads local evidence ledgers and
candidate-packet manifests, emits canonical atom dictionaries accepted by
``tac.optimization.meta_lagrangian_allocator.build_atom_ledger``, and never
turns local/planning evidence into a score claim or dispatch-ready row.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.repo_io import sha256_file

ADAPTER_TOOL = "tac.optimization.meta_lagrangian_ledger_adapter"
DEFAULT_FAMILY_GROUP = "bilevel_atom_ledger"
DEFAULT_PACKET_FAMILY_GROUP = "candidate_packet_manifest"
MPS_RESEARCH_SIGNAL_SCHEMA = "mps_research_signal_manifest.v1"
EXPLICIT_BYTE_DELTA_KEYS = (
    "byte_delta_vs_substrate_anchor",
    "archive_byte_delta_vs_substrate_anchor",
    "archive_byte_delta",
    "byte_delta",
    "delta_bytes_vs_substrate",
    "delta_bytes_vs_source_archive",
    "estimated_byte_delta",
)
EXACT_CUDA_GRADE_TOKENS = {
    "a",
    "a++",
    "contest-cuda",
    "contest_cuda",
    "[contest-cuda]",
    "[contest_cuda]",
}
ADAPTER_DISPATCH_BLOCKERS = (
    "ledger_adapter_never_dispatches",
    "requires_exact_cuda_auth_eval",
)
PROXY_SEARCH_EVIDENCE_MARKERS = (
    "mps",
    "proxy",
    "research-signal",
    "research_signal",
)


class LedgerAdapterError(ValueError):
    """Raised when a ledger artifact cannot be adapted fail-closed."""


@dataclass(frozen=True)
class LedgerAdapterResult:
    """Canonical atom rows plus auditable adapter metadata."""

    source_path: str
    source_sha256: str
    source_format: str
    atoms: tuple[dict[str, Any], ...]
    dispatch_blockers: tuple[str, ...]
    score_claim: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "meta_lagrangian_ledger_adapter_result_v1",
            "tool": ADAPTER_TOOL,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_format": self.source_format,
            "score_claim": self.score_claim,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "dispatch_blockers": list(self.dispatch_blockers),
            "atom_count": len(self.atoms),
            "atoms": list(self.atoms),
        }


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _unique_ordered(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _optional_int(value: Any) -> int | None:
    if not _is_finite_number(value):
        return None
    number = float(value)
    if not number.is_integer():
        return None
    return int(number)


def _optional_float(value: Any) -> float | None:
    if not _is_finite_number(value):
        return None
    return float(value)


def _first_present(record: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _slug(value: Any, *, fallback: str) -> str:
    text = str(value or fallback)
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text).strip("_")
    return slug[:120] or fallback


def _display_path(path: Path, repo_root: Path | None) -> str:
    if repo_root is not None:
        try:
            return str(path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            pass
    return str(path)


def _source_sha256(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def _evidence_grade(record: Mapping[str, Any], *, fallback: str = "invalid") -> str:
    return str(record.get("evidence_grade") or record.get("score_evidence_grade") or fallback)


def _has_exact_cuda_evidence(record: Mapping[str, Any], evidence_grade: str) -> bool:
    grade = evidence_grade.strip().lower()
    if record.get("exact_positive_cuda_evidence") is True:
        return True
    if _optional_float(record.get("contest_cuda_score")) is not None:
        return grade in EXACT_CUDA_GRADE_TOKENS or "contest-cuda" in grade or "contest_cuda" in grade
    return grade in {"a", "a++"}


def _score_claim_for_source(
    record: Mapping[str, Any],
    *,
    exact_cuda_evidence: bool,
    blockers: list[str],
) -> bool:
    requested = record.get("score_claim") is True
    if requested and not exact_cuda_evidence:
        blockers.append("source_score_claim_true_without_exact_cuda_evidence")
    return bool(requested and exact_cuda_evidence)


def _extract_byte_delta(
    record: Mapping[str, Any],
    *,
    archive_bytes: int | None,
    blockers: list[str],
) -> tuple[int, str]:
    explicit = _optional_int(_first_present(record, EXPLICIT_BYTE_DELTA_KEYS))
    if explicit is not None:
        return explicit, "explicit"
    substrate_bytes = _optional_int(
        _first_present(
            record,
            (
                "substrate_archive_bytes",
                "source_archive_bytes",
                "baseline_archive_bytes",
            ),
        )
    )
    if archive_bytes is not None and substrate_bytes is not None:
        return archive_bytes - substrate_bytes, "archive_bytes_minus_substrate_archive_bytes"
    blockers.append("missing_byte_delta_vs_anchor")
    return 0, "missing"


def read_jsonl_records(path: Path, *, strict: bool = True) -> list[dict[str, Any]]:
    """Read a JSONL object ledger, failing closed on malformed rows by default."""

    records: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise LedgerAdapterError(f"{path}:{line_no}: invalid JSONL: {exc.msg}") from exc
            continue
        if not isinstance(payload, dict):
            if strict:
                raise LedgerAdapterError(
                    f"{path}:{line_no}: expected JSON object, got {type(payload).__name__}"
                )
            continue
        records.append(payload)
    return records


def bilevel_record_to_atom(
    record: Mapping[str, Any],
    *,
    idx: int,
    source_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Convert one ``bilevel_atom_ledger.jsonl`` record to a canonical atom."""

    blockers = [
        *ADAPTER_DISPATCH_BLOCKERS,
        *_string_list(record.get("dispatch_blockers")),
    ]
    evidence_grade = _evidence_grade(record, fallback="invalid_missing_evidence_grade")
    exact_cuda = _has_exact_cuda_evidence(record, evidence_grade)
    score_claim = _score_claim_for_source(
        record,
        exact_cuda_evidence=exact_cuda,
        blockers=blockers,
    )
    if not exact_cuda:
        blockers.append("non_exact_cuda_source_evidence")

    archive_bytes = _optional_int(record.get("archive_bytes"))
    if "archive_bytes" in record and record.get("archive_bytes") is not None and archive_bytes is None:
        blockers.append("invalid_archive_bytes")
    if archive_bytes is None:
        blockers.append("missing_archive_bytes")
    archive_sha256 = _string_or_none(record.get("archive_sha256"))
    if not archive_sha256:
        blockers.append("missing_archive_sha256")

    byte_delta, byte_delta_source = _extract_byte_delta(
        record,
        archive_bytes=archive_bytes,
        blockers=blockers,
    )
    phase = record.get("phase", "unknown")
    op = _string_or_none(record.get("cathedral_op")) or "unknown_op"
    atom_id = _string_or_none(record.get("atom_id") or record.get("candidate_id")) or (
        f"bilevel_phase{phase}:{_slug(op, fallback='op')}:{idx}"
    )
    family = _string_or_none(record.get("family") or record.get("atom_family")) or (
        f"bilevel_phase_{phase}"
    )
    family_group = _string_or_none(record.get("family_group")) or DEFAULT_FAMILY_GROUP
    rel_err_pct = _optional_float(
        _first_present(record, ("rel_err_pct", "rel_err_pct_per_weight", "relative_error_pct"))
    )
    n_layers = _optional_int(
        _first_present(record, ("n_layers", "n_intn_layers", "n_quantized_layers"))
    )
    confidence = _optional_float(record.get("confidence"))
    if confidence is None:
        confidence = 1.0 if exact_cuda else 0.0

    atom = {
        "atom_id": atom_id,
        "family": family,
        "family_group": family_group,
        "pareto_scope": _string_or_none(record.get("pareto_scope")) or family_group,
        "byte_delta": byte_delta,
        "byte_delta_source": byte_delta_source,
        "expected_seg_dist_delta": _optional_float(record.get("expected_seg_dist_delta")) or 0.0,
        "expected_pose_dist_delta": _optional_float(record.get("expected_pose_dist_delta")) or 0.0,
        "confidence": confidence,
        "evidence_grade": evidence_grade,
        "raw_equal": bool(record.get("raw_equal") is True or exact_cuda),
        "score_claim": score_claim,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": _unique_ordered(blockers),
        "archive_bytes": archive_bytes,
        "source_artifact_bytes": archive_bytes,
        "archive_sha256": archive_sha256 or "",
        "source_archive_sha256": archive_sha256 or "",
        "archive_path": _string_or_none(record.get("archive_path")) or "",
        "archive_manifest_path": _string_or_none(record.get("archive_manifest_path")) or "",
        "archive_manifest_sha256": _string_or_none(record.get("archive_manifest_sha256")) or "",
        "evidence_source_path": _display_path(source_path, repo_root),
        "evidence_source_sha256": _source_sha256(source_path),
        "source_record_index": idx,
        "source_record_kind": "bilevel_atom_ledger_jsonl",
        "substrate_label": _string_or_none(record.get("substrate_label")) or "",
        "substrate_path": _string_or_none(record.get("substrate_path")) or "",
        "substrate_score_anchor": _optional_float(record.get("substrate_score_anchor")),
        "contest_cuda_score": _optional_float(record.get("contest_cuda_score")),
        "cathedral_op": op,
        "notes": _string_or_none(record.get("notes")) or "",
        "interaction_assumptions": _string_list(record.get("interaction_assumptions")),
    }
    if rel_err_pct is not None:
        atom["rel_err_pct"] = rel_err_pct
    if n_layers is not None:
        atom["n_layers"] = n_layers
    if record.get("lane_class"):
        atom["lane_class"] = str(record["lane_class"])
    return atom


def _candidate_packet_row(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = payload.get("selected_target")
    if isinstance(selected, Mapping):
        row = selected.get("row")
        if isinstance(row, Mapping):
            return row
        return selected
    return {}


def candidate_packet_to_atoms(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert a candidate-packet manifest's selected target into atoms."""

    row = _candidate_packet_row(payload)
    export = row.get("meta_lagrangian_atom_export") if isinstance(row, Mapping) else None
    export_map = export if isinstance(export, Mapping) else {}
    template = export_map.get("atom_template")
    template_map = template if isinstance(template, Mapping) else {}

    blockers = [
        *ADAPTER_DISPATCH_BLOCKERS,
        "candidate_packet_manifest_is_planning_only",
        *_string_list(payload.get("dispatch_blockers")),
        *_string_list(row.get("dispatch_blockers") if isinstance(row, Mapping) else None),
        *_string_list(export_map.get("export_blockers")),
    ]
    blockers.extend(f"missing_artifact:{item}" for item in _string_list(payload.get("missing_artifacts")))
    evidence_grade = _evidence_grade(
        template_map,
        fallback=_evidence_grade(payload, fallback="invalid_candidate_packet"),
    )
    exact_cuda = _has_exact_cuda_evidence(template_map, evidence_grade) or _has_exact_cuda_evidence(
        payload,
        evidence_grade,
    )
    score_claim = _score_claim_for_source(
        template_map or payload,
        exact_cuda_evidence=exact_cuda,
        blockers=blockers,
    )
    if not exact_cuda:
        blockers.append("non_exact_cuda_source_evidence")

    if not template_map:
        blockers.append("missing_meta_lagrangian_atom_export_template")
    byte_delta, byte_delta_source = _extract_byte_delta(
        template_map or row,
        archive_bytes=None,
        blockers=blockers,
    )
    source_bytes = _optional_int(row.get("actual_bytes") if isinstance(row, Mapping) else None)
    if source_bytes is None:
        audit_summary = payload.get("audit_summary")
        if isinstance(audit_summary, Mapping):
            source_bytes = _optional_int(audit_summary.get("total_actual_bytes"))
    target_bytes = _optional_int(template_map.get("target_bytes") or row.get("target_bytes"))
    label = str(
        template_map.get("atom_id")
        or row.get("label")
        or payload.get("source_path")
        or source_path.stem
    )
    atom = {
        **dict(template_map),
        "atom_id": str(template_map.get("atom_id") or f"candidate_packet:{_slug(label, fallback='target')}"),
        "family": str(template_map.get("family") or row.get("target_kind") or DEFAULT_PACKET_FAMILY_GROUP),
        "family_group": str(template_map.get("family_group") or DEFAULT_PACKET_FAMILY_GROUP),
        "pareto_scope": str(
            template_map.get("pareto_scope")
            or template_map.get("family_group")
            or DEFAULT_PACKET_FAMILY_GROUP
        ),
        "byte_delta": byte_delta,
        "byte_delta_source": byte_delta_source,
        "expected_seg_dist_delta": _optional_float(template_map.get("expected_seg_dist_delta")) or 0.0,
        "expected_pose_dist_delta": _optional_float(template_map.get("expected_pose_dist_delta")) or 0.0,
        "confidence": _optional_float(template_map.get("confidence")) or 0.0,
        "evidence_grade": evidence_grade,
        "raw_equal": bool(template_map.get("raw_equal") is True),
        "score_claim": score_claim,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": _unique_ordered(blockers),
        "source_artifact_bytes": source_bytes,
        "target_bytes": target_bytes,
        "archive_manifest_path": str(template_map.get("archive_manifest_path") or ""),
        "archive_manifest_sha256": str(template_map.get("archive_manifest_sha256") or ""),
        "source_archive_sha256": str(row.get("source_archive_sha256") or ""),
        "evidence_source_path": _display_path(source_path, repo_root),
        "evidence_source_sha256": _source_sha256(source_path),
        "source_record_index": 0,
        "source_record_kind": "candidate_packet_manifest",
        "candidate_packet_label": label,
        "candidate_packet_ready_for_exact_eval_dispatch": bool(
            payload.get("ready_for_exact_eval_dispatch") is True
        ),
        "candidate_packet_score_evidence_grade": str(payload.get("score_evidence_grade") or ""),
    }
    return [atom]


def adapt_bilevel_jsonl(
    ledger_path: Path,
    *,
    repo_root: Path | None = None,
    strict: bool = True,
) -> LedgerAdapterResult:
    records = read_jsonl_records(ledger_path, strict=strict)
    atoms = tuple(
        bilevel_record_to_atom(
            record,
            idx=idx,
            source_path=ledger_path,
            repo_root=repo_root,
        )
        for idx, record in enumerate(records)
    )
    blockers = _unique_ordered(
        [
            *ADAPTER_DISPATCH_BLOCKERS,
            *(blocker for atom in atoms for blocker in _string_list(atom.get("dispatch_blockers"))),
        ]
    )
    return LedgerAdapterResult(
        source_path=_display_path(ledger_path, repo_root),
        source_sha256=_source_sha256(ledger_path),
        source_format="bilevel_atom_ledger_jsonl",
        atoms=atoms,
        dispatch_blockers=tuple(blockers),
    )


def adapt_candidate_packet_json(
    packet_path: Path,
    *,
    repo_root: Path | None = None,
) -> LedgerAdapterResult:
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise LedgerAdapterError(f"{packet_path}: candidate packet JSON must be an object")
    atoms = tuple(candidate_packet_to_atoms(payload, source_path=packet_path, repo_root=repo_root))
    blockers = _unique_ordered(
        [
            *ADAPTER_DISPATCH_BLOCKERS,
            *(blocker for atom in atoms for blocker in _string_list(atom.get("dispatch_blockers"))),
        ]
    )
    return LedgerAdapterResult(
        source_path=_display_path(packet_path, repo_root),
        source_sha256=_source_sha256(packet_path),
        source_format="candidate_packet_manifest_json",
        atoms=atoms,
        dispatch_blockers=tuple(blockers),
    )


def adapt_mps_research_signal_manifest(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
) -> LedgerAdapterResult:
    """Adapt an MPS research-signal manifest as proxy atoms only."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise LedgerAdapterError(f"{manifest_path}: MPS manifest must be an object")
    if payload.get("schema") != MPS_RESEARCH_SIGNAL_SCHEMA:
        raise LedgerAdapterError(
            f"{manifest_path}: expected schema {MPS_RESEARCH_SIGNAL_SCHEMA}"
        )
    raw_atoms = payload.get("meta_lagrangian_atoms")
    if not isinstance(raw_atoms, list):
        raise LedgerAdapterError(f"{manifest_path}: missing meta_lagrangian_atoms[]")
    atoms: list[dict[str, Any]] = []
    for idx, raw_atom in enumerate(raw_atoms):
        if not isinstance(raw_atom, Mapping):
            raise LedgerAdapterError(f"{manifest_path}: atom {idx} is not an object")
        atom = dict(raw_atom)
        atom["score_claim"] = False
        atom["ready_for_exact_eval_dispatch"] = False
        atom["dispatchable"] = False
        atom["proxy_row"] = True
        atom["rankable"] = False
        atom["evidence_grade"] = str(
            atom.get("evidence_grade") or payload.get("evidence_grade") or "MPS-research-signal"
        )
        atom["evidence_semantics"] = str(
            atom.get("evidence_semantics")
            or payload.get("evidence_semantics")
            or "mps_proxy_curve_shape_only"
        )
        atom["dispatch_blockers"] = _unique_ordered(
            [
                *ADAPTER_DISPATCH_BLOCKERS,
                *_string_list(payload.get("dispatch_blockers")),
                *_string_list(atom.get("dispatch_blockers")),
                "mps_research_signal_not_search_candidate",
            ]
        )
        atoms.append(atom)
    blockers = _unique_ordered(
        [
            *ADAPTER_DISPATCH_BLOCKERS,
            *_string_list(payload.get("dispatch_blockers")),
            *(blocker for atom in atoms for blocker in _string_list(atom.get("dispatch_blockers"))),
        ]
    )
    return LedgerAdapterResult(
        source_path=_display_path(manifest_path, repo_root),
        source_sha256=_source_sha256(manifest_path),
        source_format="mps_research_signal_manifest_json",
        atoms=tuple(atoms),
        dispatch_blockers=tuple(blockers),
    )


def adapt_artifact_to_atoms(
    artifact_path: Path,
    *,
    repo_root: Path | None = None,
    strict: bool = True,
) -> LedgerAdapterResult:
    """Adapt a supported local artifact into canonical atom dictionaries."""

    if artifact_path.suffix == ".jsonl":
        return adapt_bilevel_jsonl(artifact_path, repo_root=repo_root, strict=strict)

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and payload.get("schema") == MPS_RESEARCH_SIGNAL_SCHEMA:
        return adapt_mps_research_signal_manifest(artifact_path, repo_root=repo_root)
    if isinstance(payload, Mapping) and "selected_target" in payload:
        return adapt_candidate_packet_json(artifact_path, repo_root=repo_root)
    raise LedgerAdapterError(f"{artifact_path}: unsupported meta-Lagrangian ledger artifact")


def search_candidates_from_atoms(atoms: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project only fully parameterized atoms to the strict search-CLI schema."""

    candidates: list[dict[str, Any]] = []
    for atom in atoms:
        evidence_text = " ".join(
            str(atom.get(key) or "").strip().lower()
            for key in ("evidence_grade", "evidence_semantics")
        )
        if atom.get("proxy_row") is True:
            continue
        if atom.get("rankable") is False:
            continue
        if any(marker in evidence_text for marker in PROXY_SEARCH_EVIDENCE_MARKERS):
            continue
        archive_bytes = _optional_int(atom.get("archive_bytes"))
        rel_err_pct = _optional_float(atom.get("rel_err_pct"))
        n_layers = _optional_int(atom.get("n_layers"))
        if archive_bytes is None or archive_bytes <= 0:
            continue
        if rel_err_pct is None or n_layers is None:
            continue
        candidate: dict[str, Any] = {
            "candidate_id": str(atom["atom_id"]),
            "archive_bytes": archive_bytes,
            "rel_err_pct": rel_err_pct,
            "n_layers": n_layers,
            "lane_class": str(atom.get("lane_class") or atom.get("family_group") or DEFAULT_FAMILY_GROUP),
        }
        archive_path = _string_or_none(atom.get("archive_path"))
        if archive_path:
            candidate["archive_path"] = archive_path
        candidates.append(candidate)
    return candidates


__all__ = [
    "ADAPTER_TOOL",
    "LedgerAdapterError",
    "LedgerAdapterResult",
    "adapt_artifact_to_atoms",
    "adapt_bilevel_jsonl",
    "adapt_candidate_packet_json",
    "adapt_mps_research_signal_manifest",
    "bilevel_record_to_atom",
    "candidate_packet_to_atoms",
    "read_jsonl_records",
    "search_candidates_from_atoms",
]
