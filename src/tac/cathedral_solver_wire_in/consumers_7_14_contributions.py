# SPDX-License-Identifier: MIT
"""Per-consumer × per-hook solver-surface contributions for Cable D 7-14.

Slot HH 2026-05-20 — closes the canonical 6-hook producer → sidecar → ranker
(FF) → solver (HH) loop per Catalog #125 for the 6 Cable D consumers landed
2026-05-19 (commit `6a1e94a63`).

Per Catalog #303 cargo-cult audit per assumption: the operator prompt asks
"wire 3 hooks × 6 consumers (target: 18)". The HARD-EARNED inventory per the
producer headers' wire-in declarations is 9 (consumer × hook) cells, NOT 18:

    | Consumer                            | #1 Sensitivity | #2 Pareto | #3 BitAlloc |
    |-------------------------------------|----------------|-----------|-------------|
    | #7  per_pair_pareto_envelope        |       N/A      |  ACTIVE   |     N/A     |
    | #8  per_pair_lagrangian_lambda      |     ACTIVE     |  ACTIVE   |     N/A     |
    | #9  per_pair_lora_supervision       |       N/A      |    N/A    |   ACTIVE    |
    | #10 per_pair_coding_budget          |       N/A      |    N/A    |   ACTIVE    |
    | #12 per_pair_kkt_residuals          |     ACTIVE     |  ACTIVE   |     N/A     |
    | #13 per_pair_volterra_cross_terms   |     ACTIVE     |  ACTIVE   |     N/A     |

9 ACTIVE + 9 N/A = 18 cells total. The N/A cells are NOT silent omissions:
each is explicitly classified per the producer header's hook declarations
(an honest map of where the consumer's signal type genuinely lives in the
solver-surface taxonomy). Forcing the full 18 via "all × all" would create
phantom contributions whose existence the producer never declared — that IS
the canonical orphan-signal-WITH-FAKE-CLAIM bug class per CLAUDE.md
"Apples-to-apples evidence discipline" non-negotiable.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog
#287/#323/#341: every ACTIVE contribution is OBSERVABILITY-ONLY:

    - `score_claim_valid=False` (per canonical Provenance)
    - `promotion_eligible=False` (per Catalog #127/#192/#317/#341 promotion-leak guard)
    - canonical `[predicted]` axis tag (per Catalog #287)
    - `predicted_delta_adjustment=0.0` (the SLOT FF cascade ALREADY applies the
       1.01× per-sidecar multiplicative reward at the RANKER surface; this
       SOLVER-surface contribution is an observability annotation, NOT a
       second-order score adjustment)

Per Catalog #318 raw-byte master-gradient guard: contributions NEVER return
raw archive-byte tensors; only canonical typed structural-signal markers.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# ─── Canonical solver-surface observability axis tag ────────────────────────
SOLVER_WIRE_IN_OBSERVABILITY_AXIS: str = "[predicted]"

# ─── Cable D consumers 7-14 canonical sidecar registry ──────────────────────
# Per Slot FF landing: (consumer_id, expected_sidecar_schema)
CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS: tuple[tuple[str, str], ...] = (
    (
        "per_pair_pareto_envelope",
        "master_gradient_consumer_per_pair_pareto_envelope_v1",
    ),
    (
        "per_pair_lagrangian_lambda_bisection",
        "master_gradient_consumer_per_pair_lambda_bisection_v1",
    ),
    (
        "per_pair_lora_supervision_signal",
        "master_gradient_consumer_per_pair_lora_supervision_v1",
    ),
    (
        "per_pair_coding_budget_allocation",
        "master_gradient_consumer_per_pair_coding_budget_v1",
    ),
    (
        "per_pair_kkt_residuals",
        "master_gradient_consumer_per_pair_kkt_residuals_v1",
    ),
    (
        "per_pair_volterra_cross_terms",
        "master_gradient_consumer_per_pair_volterra_v1",
    ),
)

# ─── Per-cell hook registry per Catalog #303 cargo-cult audit ──────────────
# 9 HARD-EARNED ACTIVE cells (per producer-header declarations). 9 N/A cells.
# Hook names per src/tac/cathedral/consumer_contract.py HookNumber enum:
#   "sensitivity_map"     = HookNumber.SENSITIVITY_MAP
#   "pareto_constraint"   = HookNumber.PARETO_CONSTRAINT
#   "bit_allocator"       = HookNumber.BIT_ALLOCATOR
CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY: Mapping[str, frozenset[str]] = {
    "per_pair_pareto_envelope": frozenset({"pareto_constraint"}),
    "per_pair_lagrangian_lambda_bisection": frozenset(
        {"sensitivity_map", "pareto_constraint"}
    ),
    "per_pair_lora_supervision_signal": frozenset({"bit_allocator"}),
    "per_pair_coding_budget_allocation": frozenset({"bit_allocator"}),
    "per_pair_kkt_residuals": frozenset(
        {"sensitivity_map", "pareto_constraint"}
    ),
    "per_pair_volterra_cross_terms": frozenset(
        {"sensitivity_map", "pareto_constraint"}
    ),
}

# Enumerated (consumer_id, hook_name) pairs for the 9 ACTIVE cells. Exposed
# for downstream iteration (e.g. `collect_all_solver_contributions_for_archive`).
CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (consumer_id, hook_name)
        for consumer_id, hooks in CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY.items()
        for hook_name in hooks
    )
)

# Legal hook names — gate against typos in caller code.
_LEGAL_HOOK_NAMES: frozenset[str] = frozenset(
    {"sensitivity_map", "pareto_constraint", "bit_allocator"}
)


@dataclass(frozen=True)
class SolverHookContribution:
    """Typed observability-only contribution to a canonical solver surface.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323/#341:
    every contribution is OBSERVABILITY-ONLY:

    - ``score_claim_valid=False``
    - ``promotion_eligible=False``
    - canonical ``[predicted]`` ``axis_tag``
    - ``predicted_delta_adjustment=0.0``

    Per Catalog #305 6-facet observability: every field is inspectable +
    decomposable + diff-able + queryable + cite-able + counterfactual-able.

    Per Catalog #318 master-gradient raw-byte-authority guard: the payload
    field is restricted to typed structural-signal markers (canonical sidecar
    path + schema + sha + n_pairs/n_bytes counts), NEVER raw archive-byte
    tensors.
    """

    consumer_id: str
    hook_name: str
    archive_sha256: str
    sidecar_present: bool
    sidecar_path: str | None  # canonical sidecar path for cite-chain
    sidecar_schema: str  # expected schema tag (canonical)
    n_pairs: int  # structural signal: 0 if sidecar absent/invalid
    n_bytes: int  # structural signal: 0 if sidecar absent/invalid

    # Canonical non-promotable markers (per Catalog #287/#323/#341)
    score_claim_valid: bool = False
    promotion_eligible: bool = False
    axis_tag: str = SOLVER_WIRE_IN_OBSERVABILITY_AXIS

    # Ranker-surface adjustment: ALWAYS 0.0 because the Slot FF cascade
    # ALREADY applies the multiplicative reward at the ranker surface.
    predicted_delta_adjustment: float = 0.0

    # Per-hook narrative annotation (human-readable cite-chain).
    rationale: str = ""

    # Canonical Provenance citation for the contribution.
    provenance_canonical_helper: str = (
        "tac.cathedral_solver_wire_in.consumers_7_14_contributions"
    )

    def __post_init__(self) -> None:
        if self.consumer_id not in CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY:
            raise ValueError(
                f"consumer_id={self.consumer_id!r} not in canonical Cable D 7-14 "
                f"set ({sorted(CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY.keys())})"
            )
        if self.hook_name not in _LEGAL_HOOK_NAMES:
            raise ValueError(
                f"hook_name={self.hook_name!r} not in legal hook set "
                f"({sorted(_LEGAL_HOOK_NAMES)})"
            )
        # Per Catalog #303: refuse N/A cells at construction time
        if self.hook_name not in CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[
            self.consumer_id
        ]:
            raise ValueError(
                f"consumer_id={self.consumer_id!r} is N/A for "
                f"hook_name={self.hook_name!r} per Catalog #303 cargo-cult audit; "
                f"this consumer's HARD-EARNED hooks are "
                f"{sorted(CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[self.consumer_id])}. "
                "Constructing this N/A cell would create a phantom contribution "
                "per CLAUDE.md 'Apples-to-apples evidence discipline'."
            )
        # Canonical non-promotable invariants per Catalog #287/#323/#341
        if self.score_claim_valid:
            raise ValueError(
                "score_claim_valid=True forbidden per CLAUDE.md "
                "'Apples-to-apples evidence discipline' + Catalog #287/#323 "
                "canonical Provenance; solver-surface contributions are "
                "observability-only."
            )
        if self.promotion_eligible:
            raise ValueError(
                "promotion_eligible=True forbidden per Catalog #127/#192/#317/#341 "
                "promotion-leak guard; solver-surface contributions never "
                "leak into promotion."
            )
        if self.axis_tag != SOLVER_WIRE_IN_OBSERVABILITY_AXIS:
            raise ValueError(
                f"axis_tag={self.axis_tag!r} must equal "
                f"{SOLVER_WIRE_IN_OBSERVABILITY_AXIS!r} per Catalog #287 "
                "canonical Provenance for [predicted] axis."
            )
        if self.predicted_delta_adjustment != 0.0:
            raise ValueError(
                f"predicted_delta_adjustment={self.predicted_delta_adjustment} "
                "must be 0.0; the FF ranker cascade already applies the "
                "multiplicative reward at the ranker surface, so the SOLVER "
                "contribution MUST be a zero-adjustment observability annotation."
            )


def consumer_owns_hook(consumer_id: str, hook_name: str) -> bool:
    """True iff (consumer_id, hook_name) is in the HARD-EARNED registry.

    Per Catalog #303: this is the canonical disambiguator between HARD-EARNED
    ACTIVE cells (per producer-header declarations) and CARGO-CULTED N/A cells
    (per "all consumers × all hooks" reflex).
    """
    if consumer_id not in CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY:
        return False
    if hook_name not in _LEGAL_HOOK_NAMES:
        return False
    return hook_name in CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[consumer_id]


def is_solver_contribution_promotable(
    contribution: SolverHookContribution,
) -> bool:
    """Per Catalog #341: solver contributions are NEVER promotable.

    Returns False by construction; provided as a public guard so downstream
    consumers (e.g. autopilot ranker, dispatch wrappers) can call this
    explicitly per CLAUDE.md "Apples-to-apples evidence discipline" instead
    of relying on the contribution's `promotion_eligible` field directly.
    """
    return False  # always False by Catalog #341 invariant


# ─── FF helper import shim (cross-callable from canonical module) ──────────
# The FF cascade landed `_latest_cable_d_consumer_sidecar_for_archive` +
# `_cable_d_consumer_sidecar_carries_structural_signal` inside
# `tools/cathedral_autopilot_autonomous_loop.py`. We import them by
# spec-loader (the tools/ namespace is not a canonical Python package).
#
# Per Catalog #110/#113 APPEND-ONLY discipline: we DO NOT mutate the FF
# helpers; we read them via a lazy importlib loader.


def _load_ff_helpers() -> tuple[Any, Any]:
    """Lazy-load the FF cascade helpers from the canonical autopilot module.

    Returns ``(_latest_helper, _structural_signal_helper)``.

    Per Catalog #340 sister-checkpoint guard + Catalog #314 absorption-pattern
    discipline: we DO NOT mutate `tools/cathedral_autopilot_autonomous_loop.py`;
    we read its public helpers via importlib.
    """
    repo_root = _resolve_repo_root()
    target = repo_root / "tools" / "cathedral_autopilot_autonomous_loop.py"
    if not target.exists():
        raise RuntimeError(
            f"FF cascade module not found at {target}; expected per Slot FF "
            "landing memo `.omx/research/cable_d_consumers_7_14_autopilot_"
            "cascade_wire_in_landed_20260519.md`"
        )
    module_name = "_cathedral_solver_wire_in_ff_helpers"
    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, target)
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"could not load spec for FF cascade module at {target}"
            )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    return (
        mod._latest_cable_d_consumer_sidecar_for_archive,
        mod._cable_d_consumer_sidecar_carries_structural_signal,
    )


def _resolve_repo_root() -> Path:
    """Resolve the canonical repo root for FF helper import.

    Searches upward from this file for the canonical marker `pyproject.toml`.
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "pyproject.toml").is_file() and (
            parent / "tools" / "cathedral_autopilot_autonomous_loop.py"
        ).is_file():
            return parent
    # Allow env override for test fixtures
    env_root = os.environ.get("PACT_REPO_ROOT")
    if env_root:
        return Path(env_root)
    raise RuntimeError(
        "could not resolve canonical repo root; expected pyproject.toml + "
        "tools/cathedral_autopilot_autonomous_loop.py marker"
    )


def _build_contribution_payload(
    consumer_id: str, archive_sha256: str
) -> tuple[bool, str | None, int, int]:
    """Read canonical sidecar for (consumer_id, archive); return structural payload.

    Returns ``(sidecar_present, sidecar_path_str, n_pairs, n_bytes)``.

    A sidecar is considered "present" per the FF helper's 6-condition
    `_cable_d_consumer_sidecar_carries_structural_signal` gate (schema +
    custody + non-trivial signal). Absent / malformed / cross-archive /
    score-claim-leak / trivial-signal sidecars return
    ``(False, None, 0, 0)`` per Catalog #341 promotion-leak guard.
    """
    expected_schema = _expected_schema_for_consumer(consumer_id)
    latest_helper, signal_helper = _load_ff_helpers()
    sidecar = latest_helper(consumer_id, archive_sha256)
    if sidecar is None:
        return (False, None, 0, 0)
    if not signal_helper(sidecar, archive_sha256, expected_schema):
        return (False, None, 0, 0)
    # Sidecar passed all 6 FF gates; safe to read structural counts.
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (False, None, 0, 0)
    n_pairs = int(payload.get("n_pairs", 0) or 0)
    n_bytes = int(payload.get("n_bytes", 0) or 0)
    return (True, str(sidecar), n_pairs, n_bytes)


def _expected_schema_for_consumer(consumer_id: str) -> str:
    for cid, schema in CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS:
        if cid == consumer_id:
            return schema
    raise ValueError(
        f"consumer_id={consumer_id!r} not in canonical sidecar registry"
    )


def _rationale_for(
    consumer_id: str, hook_name: str, sidecar_present: bool
) -> str:
    """Per-cell human-readable cite-chain rationale.

    The rationale documents the producer-header declaration that justifies
    this (consumer × hook) cell being HARD-EARNED ACTIVE per Catalog #303.
    """
    base = (
        f"Cable D consumer {consumer_id!r} → solver-surface hook "
        f"{hook_name!r} per producer header declaration"
    )
    presence = (
        "canonical sidecar PRESENT + schema-valid + custody-clean + "
        "non-trivial structural signal"
        if sidecar_present
        else "canonical sidecar ABSENT or non-structural"
    )
    return f"{base}; {presence} [predicted]"


def sensitivity_map_contribution_for_consumer(
    consumer_id: str, archive_sha256: str
) -> SolverHookContribution:
    """Hook #1 sensitivity-map contribution for a Cable D consumer.

    Per Catalog #303 HARD-EARNED cells (producer headers' declared wirings):
    - #8  per_pair_lagrangian_lambda_bisection (per-pair λ_R as axis weight)
    - #12 per_pair_kkt_residuals (KKT residual as confidence weight)
    - #13 per_pair_volterra_cross_terms (second-order Volterra kernel)

    Per Catalog #303 N/A cells (CARGO-CULTED per "all × all"):
    - #7  per_pair_pareto_envelope (Pareto-only per header)
    - #9  per_pair_lora_supervision_signal (BitAllocator-only per header)
    - #10 per_pair_coding_budget_allocation (BitAllocator-only per header)

    Raises ValueError if (consumer_id, "sensitivity_map") is N/A per Catalog #303.
    """
    if not consumer_owns_hook(consumer_id, "sensitivity_map"):
        raise ValueError(
            f"consumer {consumer_id!r} does NOT own hook 'sensitivity_map' "
            f"per Catalog #303 HARD-EARNED registry; legal hooks for this "
            f"consumer are "
            f"{sorted(CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY.get(consumer_id, frozenset()))}"
        )
    expected_schema = _expected_schema_for_consumer(consumer_id)
    sidecar_present, sidecar_path, n_pairs, n_bytes = _build_contribution_payload(
        consumer_id, archive_sha256
    )
    return SolverHookContribution(
        consumer_id=consumer_id,
        hook_name="sensitivity_map",
        archive_sha256=archive_sha256,
        sidecar_present=sidecar_present,
        sidecar_path=sidecar_path,
        sidecar_schema=expected_schema,
        n_pairs=n_pairs,
        n_bytes=n_bytes,
        rationale=_rationale_for(consumer_id, "sensitivity_map", sidecar_present),
    )


def pareto_constraint_contribution_for_consumer(
    consumer_id: str, archive_sha256: str
) -> SolverHookContribution:
    """Hook #2 Pareto-constraint contribution for a Cable D consumer.

    Per Catalog #303 HARD-EARNED cells:
    - #7  per_pair_pareto_envelope (the canonical Pareto envelope)
    - #8  per_pair_lagrangian_lambda_bisection (per-pair λ_R Pareto-frontier sample)
    - #12 per_pair_kkt_residuals (Pareto stationarity certificate)
    - #13 per_pair_volterra_cross_terms (Pareto interaction effects)

    Per Catalog #303 N/A cells (CARGO-CULTED):
    - #9  per_pair_lora_supervision_signal
    - #10 per_pair_coding_budget_allocation
    """
    if not consumer_owns_hook(consumer_id, "pareto_constraint"):
        raise ValueError(
            f"consumer {consumer_id!r} does NOT own hook 'pareto_constraint' "
            f"per Catalog #303 HARD-EARNED registry; legal hooks for this "
            f"consumer are "
            f"{sorted(CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY.get(consumer_id, frozenset()))}"
        )
    expected_schema = _expected_schema_for_consumer(consumer_id)
    sidecar_present, sidecar_path, n_pairs, n_bytes = _build_contribution_payload(
        consumer_id, archive_sha256
    )
    return SolverHookContribution(
        consumer_id=consumer_id,
        hook_name="pareto_constraint",
        archive_sha256=archive_sha256,
        sidecar_present=sidecar_present,
        sidecar_path=sidecar_path,
        sidecar_schema=expected_schema,
        n_pairs=n_pairs,
        n_bytes=n_bytes,
        rationale=_rationale_for(consumer_id, "pareto_constraint", sidecar_present),
    )


def bit_allocator_contribution_for_consumer(
    consumer_id: str, archive_sha256: str
) -> SolverHookContribution:
    """Hook #3 bit-allocator contribution for a Cable D consumer.

    Per Catalog #303 HARD-EARNED cells:
    - #9  per_pair_lora_supervision_signal (per-pair LoRA target injection)
    - #10 per_pair_coding_budget_allocation (canonical per-pair byte allocation)

    Per Catalog #303 N/A cells (CARGO-CULTED):
    - #7  per_pair_pareto_envelope
    - #8  per_pair_lagrangian_lambda_bisection
    - #12 per_pair_kkt_residuals
    - #13 per_pair_volterra_cross_terms
    """
    if not consumer_owns_hook(consumer_id, "bit_allocator"):
        raise ValueError(
            f"consumer {consumer_id!r} does NOT own hook 'bit_allocator' "
            f"per Catalog #303 HARD-EARNED registry; legal hooks for this "
            f"consumer are "
            f"{sorted(CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY.get(consumer_id, frozenset()))}"
        )
    expected_schema = _expected_schema_for_consumer(consumer_id)
    sidecar_present, sidecar_path, n_pairs, n_bytes = _build_contribution_payload(
        consumer_id, archive_sha256
    )
    return SolverHookContribution(
        consumer_id=consumer_id,
        hook_name="bit_allocator",
        archive_sha256=archive_sha256,
        sidecar_present=sidecar_present,
        sidecar_path=sidecar_path,
        sidecar_schema=expected_schema,
        n_pairs=n_pairs,
        n_bytes=n_bytes,
        rationale=_rationale_for(consumer_id, "bit_allocator", sidecar_present),
    )


# Mapping from canonical hook_name to the dispatcher accessor.
_HOOK_DISPATCHERS = {
    "sensitivity_map": sensitivity_map_contribution_for_consumer,
    "pareto_constraint": pareto_constraint_contribution_for_consumer,
    "bit_allocator": bit_allocator_contribution_for_consumer,
}


def collect_all_solver_contributions_for_archive(
    archive_sha256: str,
) -> tuple[SolverHookContribution, ...]:
    """Collect every HARD-EARNED (consumer × hook) contribution for an archive.

    Returns the 9 ACTIVE contributions per Catalog #303 cargo-cult audit.
    N/A cells are NOT included; per CLAUDE.md "Apples-to-apples evidence
    discipline" forcing N/A cells would create phantom contributions.

    Per Catalog #305 6-facet observability: the return value is the canonical
    queryable snapshot of every solver-surface contribution for this archive,
    diff-able across runs (stateless), inspectable per (consumer × hook),
    cite-able via each contribution's sidecar_path.
    """
    if not archive_sha256:
        raise ValueError("archive_sha256 must be non-empty")
    contributions: list[SolverHookContribution] = []
    for consumer_id, hook_name in CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS:
        dispatcher = _HOOK_DISPATCHERS[hook_name]
        contributions.append(dispatcher(consumer_id, archive_sha256))
    return tuple(contributions)
