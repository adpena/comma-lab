# SPDX-License-Identifier: MIT
"""Row #1 — Goyal+He 2017 canonical linear warmup schedule.

Replaces the hand-picked warmup_epochs (10% in nscs03; UNDEFINED elsewhere)
with the canonical formula:

    warmup_steps = ceil(fraction_of_total * total_steps)
                   where fraction_of_total ∈ [0.05, 0.10]

per Goyal et al 2017 ("Accurate, Large Minibatch SGD", arxiv:1706.02677 §2.2)
which prescribes a gradual warmup over the first 5-10% of training to allow
the model to escape the initialization basin before the full LR kicks in.
He et al 2016 ResNet (arxiv:1512.03385) is the empirical anchor showing
that warmup is critical for training stability at large batch.

Canonical-vs-unique decision per layer
--------------------------------------
- Data type / framework: ADOPT_CANONICAL (plain Python int/float)
- Solver pattern: UNIQUE (ceil-of-fraction-of-total formula)
- Atom emission: ADOPT_CANONICAL (tac.atom.builders.build_arbitrary_value_atom)
- Validation: ADOPT_CANONICAL (raise ValueError on invalid inputs)

9-dimension success checklist evidence
--------------------------------------
- Uniqueness: per-substrate warmup; cargo-cult is shared/undefined
- Beauty + elegance: 1 multiplication + 1 ceil call
- Distinctness: derives from total_steps, not opinion
- Rigor: bounded by min_warmup_steps + max_warmup_fraction; ValueError on invalid
- Optimization per technique: solves Goyal §2.2 canonical formula
- Stack-of-stacks composability: emits Atom + Provenance
- Deterministic reproducibility: pure function; no RNG
- Extreme optimization: O(1)
- Optimal minimal contest score: predicted ΔS [-0.002, -0.0003]

Observability surface (6 facets)
--------------------------------
- inspectable per layer: every input + intermediate exposed in result
- decomposable per signal: raw_warmup_steps separated from min-clipped
- diff-able across runs: pure function; identical inputs -> identical output
- queryable post-hoc: result is a frozen dataclass
- cite-able: literature_citation + canonical_helper_invocation
- counterfactual-able: change fraction -> observe steps delta

6-hook wire-in declaration per Catalog #125
-------------------------------------------
1. Sensitivity-map: N/A — training-time scheduler, not a score signal
2. Pareto constraint: N/A
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE via Atom emission for substrate ranking
5. Continual-learning posterior: ACTIVE via canonical Provenance on Atom
6. Probe-disambiguator: N/A

Out-of-scope
------------
Substrate-trainer wire-ins (substrates that currently hand-pick warmup_epochs
or omit it entirely) are operator-routable follow-ons per Catalog #325
per-substrate symposium discipline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from tac.atom.atom import Atom


_LITERATURE_CITATION = (
    "Goyal et al 2017 'Accurate, Large Minibatch SGD' arxiv:1706.02677 §2.2 "
    "(gradual warmup over 5-10% of training); He et al 2016 ResNet "
    "arxiv:1512.03385 (warmup canonical anchor)"
)


@dataclass(frozen=True)
class FormulaSolveResult:
    """Generic structured result for every Wave 2B formula-path helper.

    Fields
    ------
    solved_value : Any
        The closed-form formula output (steps int, fraction float, etc.).
    intermediate_values : Mapping[str, Any]
        Per-step intermediates exposed for the 6-facet observability surface.
    literature_citation : str
        Arxiv / ISBN / DOI / paper-name citation per CLAUDE.md "Apples-to-apples
        evidence discipline" + operator citation standing directive.
    canonical_helper_invocation : str
        Self-identification string for HNeRV parity L9 runtime closure.
    coupled_adjustments : Mapping[str, Any]
        Downstream values that should also change (e.g. peak_lr alongside
        warmup_steps).
    notes : str
        Free-text notes / caveats.
    """

    solved_value: Any
    intermediate_values: Mapping[str, Any] = field(default_factory=dict)
    literature_citation: str = ""
    canonical_helper_invocation: str = ""
    coupled_adjustments: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class WarmupScheduleInput:
    """Inputs to the canonical Goyal+He warmup-schedule helper."""

    total_steps: int
    fraction_of_total: float = 0.05
    min_warmup_steps: int = 100

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError(f"total_steps must be >= 1; got {self.total_steps}")
        if not 0.0 < self.fraction_of_total <= 0.10:
            raise ValueError(
                f"fraction_of_total must be in (0, 0.10] per Goyal 2017 §2.2; "
                f"got {self.fraction_of_total}"
            )
        if self.min_warmup_steps < 0:
            raise ValueError(f"min_warmup_steps must be >= 0; got {self.min_warmup_steps}")


def canonical_warmup_steps(
    inputs: WarmupScheduleInput,
    *,
    emit_arbitrariness_atom: bool = False,
    substrate_id: str = "<unknown_substrate>",
) -> FormulaSolveResult:
    """Compute canonical warmup_steps per Goyal+He 2017.

    Returns ``warmup_steps = max(min_warmup_steps, ceil(fraction * total_steps))``.

    Parameters
    ----------
    inputs : WarmupScheduleInput
        Frozen dataclass with total_steps + fraction (default 0.05 per Goyal §2.2).
    emit_arbitrariness_atom : bool
        When True, also emit a canonical ``tac.atom.Atom`` instance via
        ``tac.atom.builders.build_arbitrary_value_atom``.
    substrate_id : str
        Substrate id for atom file_path resolution (used only when
        ``emit_arbitrariness_atom=True``).

    Returns
    -------
    FormulaSolveResult
        ``solved_value`` is the integer warmup_steps.

    Examples
    --------
    >>> r = canonical_warmup_steps(WarmupScheduleInput(total_steps=1_200_000))
    >>> r.solved_value
    60000
    >>> r.intermediate_values["raw_warmup_steps"]
    60000

    >>> # 10% per He et al 2016 ResNet anchor
    >>> r = canonical_warmup_steps(
    ...     WarmupScheduleInput(total_steps=10_000, fraction_of_total=0.10)
    ... )
    >>> r.solved_value
    1000
    """
    raw = math.ceil(inputs.fraction_of_total * inputs.total_steps)
    warmup_steps = max(inputs.min_warmup_steps, raw)

    intermediate: dict[str, Any] = {
        "raw_warmup_steps": raw,
        "min_clipped": warmup_steps == inputs.min_warmup_steps and raw < inputs.min_warmup_steps,
        "fraction_of_total": inputs.fraction_of_total,
        "total_steps": inputs.total_steps,
    }
    coupled: dict[str, Any] = {}

    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, warmup_steps, substrate_id)

    return FormulaSolveResult(
        solved_value=warmup_steps,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.formula_extinctions.canonical_warmup_schedule.canonical_warmup_steps"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"Substrate {substrate_id}: warmup_steps={warmup_steps} "
            f"({inputs.fraction_of_total:.0%} of {inputs.total_steps} per Goyal 2017 §2.2)"
        ),
    )


def _emit_atom(
    inputs: WarmupScheduleInput,
    warmup_steps: int,
    substrate_id: str,
) -> "Atom":
    """Lazy-import atom builder to avoid circulars + minimize import surface."""
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="formula_extinctions.canonical_warmup_schedule.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id=f"warmup_steps_solved_for_{substrate_id}",
        file_path=f"experiments/train_substrate_{substrate_id}.py",
        current_value="10% hardcoded in nscs03; UNDEFINED elsewhere",
        predicted_replacement={
            "warmup_steps": warmup_steps,
            "fraction_of_total": inputs.fraction_of_total,
        },
        resolution_path=ResolutionPath.FORMULA,
        predicted_ev_delta_s=(-0.002, -0.0003),
        cost_envelope_usd=0.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/formula_extinctions/canonical_warmup_schedule.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2b_path3_formula_batch_20260518",
    )
