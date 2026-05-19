# SPDX-License-Identifier: MIT
"""Row #1: VRAM-aware analytical-solve batch_size helper.

Replaces hand-picked `batch_size in {1, 4, 8, 16, 32}` per substrate trainer with
the closed-form solve

    batch_size = floor((vram_budget_gb - model_size_gb - activation_overhead_gb)
                       / per_sample_activation_gb)

Then applies the Goyal et al 2017 ("Accurate, Large Minibatch SGD"
arxiv:1706.02677) linear LR scaling rule so a 2x batch implies 2x base LR.

Per CLAUDE.md Catalog #170 (substrate recipes declare `min_vram_gb`) +
Catalog #218 (mini-batch reconstruct) we already KNOW per-substrate VRAM
footprint; the canonical 1/4/8/16/32 grid is cargo-cult.

Canonical-vs-unique decision per layer
--------------------------------------
- Data type / framework: ADOPT_CANONICAL (Python floats; no numpy needed)
- Solver pattern: UNIQUE (closed-form linear-budget arithmetic)
- Atom emission: ADOPT_CANONICAL (tac.atom.builders.build_arbitrary_value_atom)

9-dimension success checklist evidence
--------------------------------------
- Uniqueness: VRAM-budget-aware batch sizing is per-substrate; cargo-cult grid is shared
- Beauty + elegance: 1 multiplication + 1 floor in core math
- Distinctness: derives batch from physical VRAM, not opinion
- Rigor: bounded by floor + max_batch_size cap; refuses negative budget
- Optimization per technique: solves the actual constraint
- Stack-of-stacks composability: emits Atom + Provenance
- Deterministic reproducibility: pure function; no RNG
- Extreme optimization: O(1)
- Optimal minimal contest score: predicted ΔS [-0.004, -0.001]

Observability surface (6 facets)
--------------------------------
- inspectable per layer: every input + intermediate computation exposed in result
- decomposable per signal: vram_remaining_for_activations split from solve
- diff-able across runs: pure function; identical inputs → identical output
- queryable post-hoc: result is a frozen dataclass
- cite-able: literature_citation + canonical_helper_invocation
- counterfactual-able: change inputs → observe output deltas

6-hook wire-in declaration per Catalog #125
-------------------------------------------
1. Sensitivity-map: N/A — defensive optimization, not a signal contribution
2. Pareto constraint: ACTIVE via memory-budget feasibility set
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE via Atom emission for substrate ranking
5. Continual-learning posterior: ACTIVE via canonical Provenance
6. Probe-disambiguator: N/A
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from tac.atom.atom import Atom


@dataclass(frozen=True)
class BatchSizeSolverInput:
    """Inputs to the VRAM-aware analytical-solve batch_size helper."""

    vram_budget_gb: float
    model_size_gb: float
    activation_overhead_gb: float
    per_sample_activation_gb: float
    base_lr_at_reference_batch: float
    reference_batch_size: int = 1
    min_batch_size: int = 1
    max_batch_size: int = 256

    def __post_init__(self) -> None:
        if self.vram_budget_gb <= 0:
            raise ValueError(f"vram_budget_gb must be positive; got {self.vram_budget_gb}")
        if self.model_size_gb < 0:
            raise ValueError(f"model_size_gb must be non-negative; got {self.model_size_gb}")
        if self.per_sample_activation_gb <= 0:
            raise ValueError(
                f"per_sample_activation_gb must be positive; got {self.per_sample_activation_gb}"
            )
        if self.reference_batch_size < 1:
            raise ValueError(
                f"reference_batch_size must be >= 1; got {self.reference_batch_size}"
            )
        if self.min_batch_size < 1:
            raise ValueError(f"min_batch_size must be >= 1; got {self.min_batch_size}")
        if self.max_batch_size < self.min_batch_size:
            raise ValueError(
                f"max_batch_size {self.max_batch_size} < min_batch_size {self.min_batch_size}"
            )
        if self.base_lr_at_reference_batch <= 0:
            raise ValueError(
                f"base_lr_at_reference_batch must be positive; got {self.base_lr_at_reference_batch}"
            )


@dataclass(frozen=True)
class AnalyticalSolveResult:
    """Generic structured result for every Wave 2A analytical-solve helper.

    Fields
    ------
    solved_value : Any
        The closed-form solution (batch_size int, K int, threshold float, etc.).
    intermediate_values : Mapping[str, Any]
        Per-step intermediates (vram_remaining, R_D_curve_knee, etc.) for
        the 6-facet observability surface.
    literature_citation : str
        Arxiv / ISBN / DOI / paper-name citation per CLAUDE.md "Apples-to-apples
        evidence discipline" + operator citation standing directive.
    canonical_helper_invocation : str
        Self-identification string for HNeRV parity L9 runtime closure.
    coupled_adjustments : Mapping[str, Any]
        Downstream values that should also change (e.g. LR scaling per Goyal
        2017 linear rule alongside batch_size).
    notes : str
        Free-text notes / caveats.
    """

    solved_value: Any
    intermediate_values: Mapping[str, Any] = field(default_factory=dict)
    literature_citation: str = ""
    canonical_helper_invocation: str = ""
    coupled_adjustments: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""


_LITERATURE_CITATION = (
    "Goyal et al 2017 'Accurate, Large Minibatch SGD' arxiv:1706.02677 (linear-LR scaling rule)"
)


def solve_vram_aware_batch_size(
    inputs: BatchSizeSolverInput,
    *,
    emit_arbitrariness_atom: bool = False,
    substrate_id: str = "<unknown_substrate>",
) -> AnalyticalSolveResult:
    """Closed-form VRAM-aware batch_size solve + Goyal linear LR scaling.

    Parameters
    ----------
    inputs : BatchSizeSolverInput
        Frozen dataclass with VRAM budget + per-sample activation cost
        + reference batch LR for linear scaling.
    emit_arbitrariness_atom : bool
        When True, also emit a canonical ``tac.atom.Atom`` instance via
        ``tac.atom.builders.build_arbitrary_value_atom``. Stored in
        ``coupled_adjustments["atom"]``.
    substrate_id : str
        Substrate id for atom file_path resolution (used only when
        ``emit_arbitrariness_atom=True``).

    Returns
    -------
    AnalyticalSolveResult
        ``solved_value`` is the integer batch_size in
        [``min_batch_size``, ``max_batch_size``]. ``coupled_adjustments``
        includes ``"linear_scaled_lr"`` per Goyal 2017.

    Raises
    ------
    ValueError
        When the remaining VRAM after model + activation overhead is
        insufficient for even one sample (the caller should reduce model
        size or pick a larger-VRAM GPU class).
    """
    vram_remaining = (
        inputs.vram_budget_gb - inputs.model_size_gb - inputs.activation_overhead_gb
    )
    if vram_remaining <= 0:
        raise ValueError(
            f"VRAM budget {inputs.vram_budget_gb}GB insufficient for "
            f"model {inputs.model_size_gb}GB + overhead {inputs.activation_overhead_gb}GB; "
            "increase budget or shrink model"
        )
    raw_batch = vram_remaining / inputs.per_sample_activation_gb
    if raw_batch < 1.0:
        raise ValueError(
            f"VRAM remaining {vram_remaining:.3f}GB cannot fit a single sample "
            f"of {inputs.per_sample_activation_gb}GB; increase budget or shrink model"
        )
    batch_size = max(inputs.min_batch_size, min(inputs.max_batch_size, math.floor(raw_batch)))
    linear_scaled_lr = inputs.base_lr_at_reference_batch * (batch_size / inputs.reference_batch_size)

    intermediate: dict[str, Any] = {
        "vram_remaining_gb": vram_remaining,
        "raw_unrounded_batch_size": raw_batch,
        "ceiling_capped": batch_size == inputs.max_batch_size,
        "floor_capped": batch_size == inputs.min_batch_size,
    }
    coupled: dict[str, Any] = {
        "linear_scaled_lr": linear_scaled_lr,
        "lr_scaling_factor": batch_size / inputs.reference_batch_size,
    }

    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, batch_size, linear_scaled_lr, substrate_id)

    return AnalyticalSolveResult(
        solved_value=batch_size,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.analytical_solve_extinctions.vram_aware_batch_size.solve_vram_aware_batch_size"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"Substrate {substrate_id}: batch={batch_size} (raw {raw_batch:.3f}); "
            f"LR scaled {inputs.reference_batch_size}->{batch_size} per Goyal 2017."
        ),
    )


def _emit_atom(
    inputs: BatchSizeSolverInput,
    batch_size: int,
    linear_scaled_lr: float,
    substrate_id: str,
) -> "Atom":
    """Lazy-import atom builder to avoid circulars + minimize import surface."""
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="analytical_solve_extinctions.vram_aware_batch_size.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id=f"batch_size_solved_for_{substrate_id}",
        file_path=f"experiments/train_substrate_{substrate_id}.py",
        current_value="hand_picked_grid_1_4_8_16_32",
        predicted_replacement={"batch_size": batch_size, "linear_scaled_lr": linear_scaled_lr},
        resolution_path=ResolutionPath.ANALYTICAL_SOLVE,
        predicted_ev_delta_s=(-0.004, -0.001),
        cost_envelope_usd=2.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/analytical_solve_extinctions/vram_aware_batch_size.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518",
    )
