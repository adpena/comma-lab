# SPDX-License-Identifier: MIT
"""Historical Row #8 compatibility surface for the retired inflate.py LOC gate.

Operator directive 2026-07-21 permanently removed every ``inflate.py`` and
bolt-on source-line restriction.  The public names in this module remain
importable for old manifests and callers, but they have no acceptance,
dispatch, Pareto, atom-emission, or score authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tac.formula_extinctions.canonical_warmup_schedule import FormulaSolveResult


_LITERATURE_CITATION = (
    "Operator directive 2026-07-21: inflate.py and bolt-on LOC restrictions "
    "permanently removed; historical constants retained for compatibility"
)

# Historical values retained solely so older imports and serialized records load.
LOC_BUDGET_AT_30SEC: int = 200
COMPLEXITY_BUDGET_MCCABE: int = 10
DEPENDENCY_BUDGET_HNERV: int = 2
WEIGHT_LOC: float = 0.4
WEIGHT_COMPLEXITY: float = 0.4
WEIGHT_DEPENDENCIES: float = 0.2


@dataclass(frozen=True)
class LOCBudgetInput:
    """Historical telemetry inputs; none can block or authorize a submission."""

    loc: int
    cyclomatic_complexity: int
    external_dependencies: int

    def __post_init__(self) -> None:
        if self.loc < 0:
            raise ValueError(f"loc must be >= 0; got {self.loc}")
        if self.cyclomatic_complexity < 0:
            raise ValueError(
                f"cyclomatic_complexity must be >= 0; got {self.cyclomatic_complexity}"
            )
        if self.external_dependencies < 0:
            raise ValueError(
                f"external_dependencies must be >= 0; got {self.external_dependencies}"
            )


def canonical_inflate_py_loc_budget(
    inputs: LOCBudgetInput,
    *,
    emit_arbitrariness_atom: bool = False,
    submission_id: str = "<unknown_submission>",
) -> FormulaSolveResult:
    """Return a permanent no-op result while preserving raw telemetry.

    ``emit_arbitrariness_atom`` is intentionally ignored: the retired
    restriction must not re-enter the cathedral/autopilot action surface.
    """
    _ = emit_arbitrariness_atom
    intermediate: dict[str, Any] = {
        "restriction_active": False,
        "informational_only": True,
        "loc": inputs.loc,
        "cyclomatic_complexity": inputs.cyclomatic_complexity,
        "external_dependencies": inputs.external_dependencies,
        "historical_loc_budget_at_30sec": LOC_BUDGET_AT_30SEC,
        "historical_complexity_budget_mccabe": COMPLEXITY_BUDGET_MCCABE,
        "historical_dependency_budget_hnerv": DEPENDENCY_BUDGET_HNERV,
    }
    return FormulaSolveResult(
        solved_value=0.0,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.formula_extinctions.inflate_py_loc_budget_derivation."
            "canonical_inflate_py_loc_budget"
        ),
        coupled_adjustments={
            "dominant_factor": "none",
            "operator_decision": "permanently_removed_2026-07-21",
        },
        notes=(
            f"Submission {submission_id}: historical LOC telemetry only; "
            "restriction_active=False"
        ),
    )
