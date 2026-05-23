# SPDX-License-Identifier: MIT
"""Row #6 — Frontier threshold auto-derived from canonical state (Catalog #316).

The cathedral_autopilot ranker historically used a hardcoded
``frontier_threshold_cpu = 0.192`` (PR101 GOLD anchor). Per the audit row this
literal is stale by 0.00005 TODAY (current live frontier 0.19205 per
``pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean``) and grows
staler every day a new frontier anchor lands.

Per Catalog #316 the canonical answer is to derive the threshold from
``tac.frontier_scan.collect_all_anchors`` + ``best_per_axis`` at runtime.
This module is the DERIVATION helper (NOT a modification to
``tools/cathedral_autopilot_autonomous_loop.py``; the autopilot already wires
``_resolve_canonical_frontier_threshold_cpu`` via the canonical helper).

The formula:

    best = best_per_axis(all_anchors)
    threshold_cpu = best["contest_cpu"][0].score
    threshold_cuda = best["contest_cuda"][0].score

with 1:1 contest-compliant hardware filter (Linux x86_64 for CPU; NVIDIA
T4/A100/4090/H100/A10G/L40S for CUDA) per CLAUDE.md "Submission auth eval —
BOTH CPU AND CUDA" non-negotiable.

Canonical-vs-unique decision per layer
--------------------------------------
- Data type / framework: ADOPT_CANONICAL (Python float)
- Solver pattern: UNIQUE (canonical-state-best lookup)
- Atom emission: ADOPT_CANONICAL (tac.atom.builders.build_arbitrary_value_atom)

9-dimension success checklist evidence
--------------------------------------
- Uniqueness: per-axis canonical-state lookup; cargo-cult was hardcoded literal
- Beauty + elegance: thin wrapper over tac.frontier_scan
- Distinctness: derives from canonical state, never goes stale
- Rigor: refuses if no qualifying anchors exist
- Optimization per technique: composes with Catalog #316 canonical helper
- Stack-of-stacks composability: emits Atom + Provenance
- Deterministic reproducibility: pure function of canonical state
- Extreme optimization: O(N_anchors) in best_per_axis
- Optimal minimal contest score: predicted ΔS [-0.0001, 0.0]

Observability surface (6 facets)
--------------------------------
- inspectable per layer: per-axis best anchor exposed
- decomposable per signal: anchor.score / .lane_id / .hardware_substrate exposed
- diff-able across runs: deterministic given canonical state
- queryable post-hoc: result is a frozen dataclass
- cite-able: Catalog #316 + tac.frontier_scan
- counterfactual-able: state change -> threshold change

6-hook wire-in declaration per Catalog #125
-------------------------------------------
1. Sensitivity-map: N/A — autopilot threshold derivation
2. Pareto constraint: ACTIVE — frontier IS the Pareto-optimal threshold
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE — consumed by autopilot ranker
5. Continual-learning posterior: ACTIVE via canonical Provenance on Atom
6. Probe-disambiguator: N/A
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tac.formula_extinctions.canonical_warmup_schedule import FormulaSolveResult

if TYPE_CHECKING:
    from tac.atom.atom import Atom


_LITERATURE_CITATION = (
    "Catalog #316 'reports/latest.md not stale vs canonical frontier' + "
    "tac.frontier_scan canonical helper (frontier auto-derived from "
    "best qualifying anchor per axis per CLAUDE.md 'Submission auth eval — "
    "BOTH CPU AND CUDA' non-negotiable)"
)


@dataclass(frozen=True)
class FrontierThresholdInput:
    """Inputs to the canonical frontier-threshold-from-state helper."""

    repo_root: Path
    axis: Literal["cpu", "cuda"] = "cpu"

    def __post_init__(self) -> None:
        if self.axis not in ("cpu", "cuda"):
            raise ValueError(f"axis must be 'cpu' or 'cuda'; got {self.axis!r}")
        if not isinstance(self.repo_root, Path):
            raise ValueError(
                f"repo_root must be a Path; got {type(self.repo_root).__name__}"
            )


def canonical_frontier_threshold_from_state(
    inputs: FrontierThresholdInput,
    *,
    emit_arbitrariness_atom: bool = False,
) -> FormulaSolveResult:
    """Auto-derive the contest-axis frontier threshold from canonical state.

    Delegates to ``tac.frontier_scan.collect_all_anchors`` + ``best_per_axis``
    per Catalog #316. Returns the best (lowest score) qualifying anchor's
    score for the requested axis.

    Parameters
    ----------
    inputs : FrontierThresholdInput
        Frozen dataclass with repo_root + axis ('cpu' or 'cuda').
    emit_arbitrariness_atom : bool
        When True, also emit a canonical ``tac.atom.Atom`` instance.

    Returns
    -------
    FormulaSolveResult
        ``solved_value`` is the float threshold (best qualifying score).
        ``intermediate_values`` includes the anchor source.

    Raises
    ------
    RuntimeError
        If no qualifying anchors exist for the requested axis.
    """
    # Lazy-import to avoid circulars
    from tac.frontier_scan import best_per_axis, collect_all_anchors

    anchors = collect_all_anchors(inputs.repo_root)
    best_by_axis = best_per_axis(anchors)
    axis_key = "contest_cpu" if inputs.axis == "cpu" else "contest_cuda"
    best_list = best_by_axis.get(axis_key) or []
    best = best_list[0] if best_list else None

    if best is None:
        raise RuntimeError(
            f"No qualifying contest-{inputs.axis.upper()} anchors found in "
            f"canonical state under {inputs.repo_root}; cannot derive frontier "
            f"threshold (per Catalog #316)"
        )

    threshold = best.score

    intermediate: dict[str, Any] = {
        "axis": inputs.axis,
        "anchor_score": best.score,
        "anchor_lane_id": getattr(best, "lane_id", "<unknown>"),
        "anchor_hardware_substrate": getattr(best, "hardware_substrate", "<unknown>"),
        "anchor_archive_sha256": getattr(best, "archive_sha256", "<unknown>"),
        "total_anchors_scanned": len(anchors),
    }
    coupled: dict[str, Any] = {}

    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, threshold, best)

    return FormulaSolveResult(
        solved_value=threshold,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.formula_extinctions.canonical_frontier_threshold_from_state."
            "canonical_frontier_threshold_from_state"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"axis={inputs.axis}: threshold={threshold} from "
            f"{getattr(best, 'lane_id', '<unknown>')} "
            f"(scanned {len(anchors)} anchors)"
        ),
    )


def _emit_atom(
    inputs: FrontierThresholdInput,
    threshold: float,
    best: Any,
) -> Atom:
    """Lazy-import atom builder."""
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="formula_extinctions.canonical_frontier_threshold_from_state.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id=f"frontier_threshold_{inputs.axis}_from_canonical_state",
        file_path="tools/cathedral_autopilot_autonomous_loop.py",
        current_value="0.192 hardcoded (PR101 GOLD anchor; stale)",
        predicted_replacement={
            "axis": inputs.axis,
            "threshold": threshold,
            "anchor_lane_id": getattr(best, "lane_id", "<unknown>"),
        },
        resolution_path=ResolutionPath.FORMULA,
        predicted_ev_delta_s=(-0.0001, 0.0),
        cost_envelope_usd=0.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/formula_extinctions/canonical_frontier_threshold_from_state.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2b_path3_formula_batch_20260518",
    )
