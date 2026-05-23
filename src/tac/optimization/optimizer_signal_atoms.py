# SPDX-License-Identifier: MIT
"""Convert optimizer planning signals into canonical ``tac.atom`` rows.

Optimizer queues and learned-sweep pairings are useful only after downstream
solver surfaces can consume them as typed, provenance-bearing records.  This
adapter takes the existing optimizer ``solver_stack_wire_in`` contract and
emits canonical Atoms without upgrading proxy objectives into score authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from tac.atom.atom import Atom
from tac.atom.types import AtomKind, ResolutionPath
from tac.optimization.optimizer_training_signal_bridge import (
    OPTIMIZER_TRAINING_SIGNAL_WIRE_IN_SCHEMA,
    validate_optimizer_training_signal_wire_in,
)
from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    require_no_truthy_authority_fields,
    validate_proxy_candidate,
)
from tac.provenance import build_provenance_for_predicted, provenance_to_dict

OPTIMIZER_SIGNAL_ATOM_LEDGER_SCHEMA = "optimizer_signal_atom_ledger_v1"
OPTIMIZER_SIGNAL_ATOM_SOURCE_SCHEMA = "optimizer_signal_atom_source_v1"
SUPPORTED_QUEUE_SCHEMAS = frozenset(
    {
        "optimizer_guided_candidate_queue_v1",
        "optimizer_candidate_queue_v1",
    }
)
ATOM_HELPER_LINK = "src/tac/optimization/optimizer_signal_atoms.py"
DEFAULT_LITERATURE_CITATION = (
    "Boyd and Vandenberghe 2004 Lagrangian composition; "
    "tac.optimization.optimizer_training_signal_bridge canonical wire-in"
)


class OptimizerSignalAtomError(ValueError):
    """Raised when an optimizer signal cannot become a canonical Atom."""


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for hashing and persisted metadata."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_payload(value: Any) -> str:
    """Return SHA-256 over the canonical JSON representation of ``value``."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _false_authority_metadata() -> dict[str, bool]:
    return dict(PROXY_FALSE_AUTHORITY_FIELDS)


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OptimizerSignalAtomError(f"{label} must be a mapping")
    return value


def _validate_wire_in(payload: Mapping[str, Any]) -> None:
    violations = validate_optimizer_training_signal_wire_in(payload)
    if violations:
        raise OptimizerSignalAtomError(
            "optimizer wire-in contract violations: " + ", ".join(violations)
        )
    require_no_truthy_authority_fields(payload, context="optimizer_signal_atom_wire_in")


def build_atom_from_optimizer_wire_in(
    payload: Mapping[str, Any],
    *,
    source_record: Mapping[str, Any] | None = None,
    source_path: str | None = None,
    cost_envelope_usd: float = 0.0,
) -> Atom:
    """Build one canonical planning Atom from an optimizer wire-in payload.

    The returned Atom deliberately uses a neutral score-impact interval.  Any
    proxy objective, local CPU, or MLX signal remains in metadata until a
    byte-closed archive and exact CPU/CUDA auth artifact can promote it.
    """

    _validate_wire_in(payload)
    record = dict(source_record or {})
    require_no_truthy_authority_fields(record, context="optimizer_signal_atom_source_record")

    atom_wire_in = _require_mapping(payload.get("atom_wire_in"), label="atom_wire_in")
    candidate_id = str(payload.get("candidate_id") or "").strip()
    profile_id = str(payload.get("profile_id") or "").strip()
    lane_id = str(payload.get("lane_id") or "").strip()
    if not candidate_id:
        raise OptimizerSignalAtomError("candidate_id is required")
    if not profile_id:
        raise OptimizerSignalAtomError("profile_id is required")
    if not lane_id:
        raise OptimizerSignalAtomError("lane_id is required")

    requested_atom_kind = str(atom_wire_in.get("atom_kind") or "")
    requested_resolution_path = str(atom_wire_in.get("resolution_path") or "")
    if requested_atom_kind != AtomKind.META_LAGRANGIAN.value:
        raise OptimizerSignalAtomError(
            f"unsupported optimizer atom_kind {requested_atom_kind!r}; "
            "expected meta_lagrangian"
        )
    if requested_resolution_path != ResolutionPath.LEARNED.value:
        raise OptimizerSignalAtomError(
            f"unsupported optimizer resolution_path {requested_resolution_path!r}; "
            "expected learned"
        )

    candidate_atom_id = str(atom_wire_in.get("candidate_atom_id") or candidate_id).strip()
    atom_id = f"optimizer_signal:{profile_id}:{candidate_atom_id}"
    source_sha256 = sha256_payload(
        {
            "wire_in": payload,
            "source_record": record,
            "source_path": source_path,
        }
    )
    provenance = build_provenance_for_predicted(
        model_id="tac.optimization.optimizer_signal_atoms",
        inputs_sha256=source_sha256,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )

    metadata = {
        "source_schema": OPTIMIZER_SIGNAL_ATOM_SOURCE_SCHEMA,
        "candidate_id": candidate_id,
        "profile_id": profile_id,
        "lane_id": lane_id,
        "lane_class": payload.get("lane_class"),
        "candidate_family": payload.get("candidate_family"),
        "representation_family": payload.get("representation_family"),
        "substrate_family": payload.get("substrate_family"),
        "training_signal_kind": payload.get("training_signal_kind"),
        "param_schema": payload.get("param_schema"),
        "candidate_params": dict(_require_mapping(payload.get("candidate_params"), label="candidate_params")),
        "source_anchor": payload.get("source_anchor"),
        "score_lowering_hypothesis": payload.get("score_lowering_hypothesis"),
        "variant_axes": list(payload.get("variant_axes") or []),
        "canonical_equation_refs": list(payload.get("canonical_equation_refs") or []),
        "master_gradient_wire_in": dict(
            _require_mapping(payload.get("master_gradient_wire_in"), label="master_gradient_wire_in")
        ),
        "pareto_wire_in": dict(_require_mapping(payload.get("pareto_wire_in"), label="pareto_wire_in")),
        "bit_allocator_wire_in": dict(
            _require_mapping(payload.get("bit_allocator_wire_in"), label="bit_allocator_wire_in")
        ),
        "cathedral_autopilot_wire_in": dict(
            _require_mapping(payload.get("cathedral_autopilot_wire_in"), label="cathedral_autopilot_wire_in")
        ),
        "continual_learning_wire_in": dict(
            _require_mapping(payload.get("continual_learning_wire_in"), label="continual_learning_wire_in")
        ),
        "probe_disambiguator_wire_in": dict(
            _require_mapping(payload.get("probe_disambiguator_wire_in"), label="probe_disambiguator_wire_in")
        ),
        "xray_wire_in": dict(_require_mapping(payload.get("xray_wire_in"), label="xray_wire_in")),
        "deterministic_solution_wire_in": dict(
            _require_mapping(payload.get("deterministic_solution_wire_in"), label="deterministic_solution_wire_in")
        ),
        "false_authority": _false_authority_metadata(),
        "score_impact_band_status": "neutral_until_exact_auth_or_calibrated_posterior",
        "proxy_objective": record.get("proxy_objective"),
        "rank_score": record.get("rank_score"),
        "rank_score_field": record.get("rank_score_field"),
        "proxy_components": record.get("proxy_components"),
        "optimizer": record.get("optimizer"),
        "optimizer_status": record.get("optimizer_status"),
        "parameter_group_lr_policy_id": record.get("parameter_group_lr_policy_id"),
        "parameter_group_lr_policy_sha256": record.get("parameter_group_lr_policy_sha256"),
        "parameter_group_fingerprint_sha256": record.get("parameter_group_fingerprint_sha256"),
        "source_path": source_path,
        "source_record_sha256": sha256_payload(record) if record else None,
        "wire_in_sha256": sha256_payload(payload),
    }

    return Atom(
        atom_id=atom_id,
        kind=AtomKind.META_LAGRANGIAN,
        resolution_path=ResolutionPath.LEARNED,
        predicted_impact_delta_s_lower=0.0,
        predicted_impact_delta_s_upper=0.0,
        cost_envelope_usd=float(cost_envelope_usd),
        provenance=provenance_to_dict(provenance),
        wired_hooks=_ordered_unique(atom_wire_in.get("wired_hooks") or []),
        observability_surface=(
            "inspectable_per_layer",
            "decomposable_per_signal",
            "diff_able_across_runs",
            "queryable_post_hoc",
            "cite_able",
            "counterfactual_able",
        ),
        literature_citation=DEFAULT_LITERATURE_CITATION,
        canonical_helper_repo_link=ATOM_HELPER_LINK,
        metadata=metadata,
    )


def extract_optimizer_wire_in_records(
    payload: Mapping[str, Any],
    *,
    source_path: str | None = None,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Return ``(wire_in, source_record)`` pairs from a queue or direct payload."""

    schema = str(payload.get("schema") or "")
    if schema == OPTIMIZER_TRAINING_SIGNAL_WIRE_IN_SCHEMA:
        return [(payload, {})]
    if schema not in SUPPORTED_QUEUE_SCHEMAS:
        raise OptimizerSignalAtomError(
            f"unsupported optimizer signal source schema {schema!r}; "
            f"expected {sorted(SUPPORTED_QUEUE_SCHEMAS | {OPTIMIZER_TRAINING_SIGNAL_WIRE_IN_SCHEMA})}"
        )

    rows = payload.get("top_k")
    if not isinstance(rows, list):
        raise OptimizerSignalAtomError("optimizer queue payload must contain top_k list")

    records: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for idx, row_any in enumerate(rows):
        row = _require_mapping(row_any, label=f"top_k[{idx}]")
        violations = validate_proxy_candidate(row)
        if violations:
            raise OptimizerSignalAtomError(
                f"top_k[{idx}] proxy contract violations: " + ", ".join(violations)
            )
        require_no_truthy_authority_fields(row, context=f"optimizer_signal_atom_top_k[{idx}]")
        wire = _require_mapping(row.get("solver_stack_wire_in"), label=f"top_k[{idx}].solver_stack_wire_in")
        records.append((wire, {**row, "source_queue_path": source_path}))
    return records


def build_atoms_from_optimizer_signal_source(
    payload: Mapping[str, Any],
    *,
    source_path: str | None = None,
    max_atoms: int | None = None,
) -> list[Atom]:
    """Build canonical Atoms from an optimizer queue or direct wire-in payload."""

    records = extract_optimizer_wire_in_records(payload, source_path=source_path)
    if max_atoms is not None:
        if max_atoms < 1:
            raise OptimizerSignalAtomError("max_atoms must be positive when supplied")
        records = records[:max_atoms]
    atoms = [
        build_atom_from_optimizer_wire_in(
            wire,
            source_record=record,
            source_path=source_path,
        )
        for wire, record in records
    ]
    atom_ids = [atom.atom_id for atom in atoms]
    if len(atom_ids) != len(set(atom_ids)):
        raise OptimizerSignalAtomError("optimizer signal source produced duplicate atom ids")
    return atoms


def build_optimizer_signal_atom_ledger(
    payload: Mapping[str, Any],
    *,
    source_path: str | None = None,
    max_atoms: int | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe Atom ledger payload for optimizer signal ingestion."""

    atoms = build_atoms_from_optimizer_signal_source(
        payload,
        source_path=source_path,
        max_atoms=max_atoms,
    )
    return {
        "schema": OPTIMIZER_SIGNAL_ATOM_LEDGER_SCHEMA,
        "source_schema": str(payload.get("schema") or ""),
        "source_path": source_path,
        "atom_count": len(atoms),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_semantics": "optimizer_signal_atoms_proxy_planning_only",
        "promotion_gate": "byte_closed_archive_plus_claimed_exact_cpu_cuda_auth_eval",
        "atoms": [atom.to_jsonl_row() for atom in atoms],
        "meta_lagrangian_atoms": [atom.to_meta_lagrangian_atom() for atom in atoms],
    }


__all__ = [
    "ATOM_HELPER_LINK",
    "DEFAULT_LITERATURE_CITATION",
    "OPTIMIZER_SIGNAL_ATOM_LEDGER_SCHEMA",
    "OPTIMIZER_SIGNAL_ATOM_SOURCE_SCHEMA",
    "SUPPORTED_QUEUE_SCHEMAS",
    "OptimizerSignalAtomError",
    "build_atom_from_optimizer_wire_in",
    "build_atoms_from_optimizer_signal_source",
    "build_optimizer_signal_atom_ledger",
    "canonical_json",
    "extract_optimizer_wire_in_records",
    "sha256_payload",
]
