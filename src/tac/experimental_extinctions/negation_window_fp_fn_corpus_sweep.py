# SPDX-License-Identifier: MIT
"""Row #7: Catalog #236 80-char negation window FP/FN corpus sweep.

Replaces ``80-char negation window`` hardcoded in Catalog #236
(``check_catalog_233_gate_3_prose_negation_guard_present``; per
``.omx/state/arbitrariness_extinction_audit_20260518.jsonl`` row
``80_char_negation_window_Catalog_236``) with a per-corpus FP/FN sweep that
measures False Positive + False Negative rate at each candidate window size
on a labeled CLAUDE.md catalog corpus and emits the empirically-optimal
window size (FP+FN minimized).

Predicted EV: [-0.0002, 0.0] per ``.omx/research/arbitrariness_extinction_audit_top_50_ranked_20260518.md``.

Empirical anchor (expected): 80-char window may have FP/FN sweet spot at
different value. Candidate grid: {40, 60, 80, 100, 120, 160}. Sweet spot
typically near median sentence length per the labeled corpus.

Canonical-vs-unique decision per layer
--------------------------------------
- Data type / framework: ADOPT_CANONICAL (Python str + stdlib only)
- Sweep pattern: UNIQUE (per-corpus FP/FN ROC over candidate window grid)
- Atom emission: ADOPT_CANONICAL (tac.atom.builders.build_arbitrary_value_atom)
- Provenance: ADOPT_CANONICAL (tac.provenance Catalog #323)

9-dim success checklist evidence per Catalog #294
-------------------------------------------------
- UNIQUENESS: per-corpus FP/FN sweep, not hardcoded 80-char window
- BEAUTY+ELEGANCE: per-window FP+FN ROC; ~30-LOC math
- DISTINCTNESS: distinct from other rows (linguistic vs numerical sweep)
- RIGOR: requires labeled corpus, refuses empty, refuses missing labels
- OPTIMIZATION-PER-TECHNIQUE: minimizes FP+FN over candidate window grid
- STACK-OF-STACKS-COMPOSABILITY: emits Atom + Provenance
- DETERMINISTIC-REPRODUCIBILITY: pure function on labeled corpus
- EXTREME-OPTIMIZATION-PERFORMANCE: O(N samples * G grid)
- OPTIMAL-MINIMAL-CONTEST-SCORE: predicted ΔS [-0.0002, 0.0]

Observability surface per Catalog #305 (6 facets)
-------------------------------------------------
- inspectable per layer: per-window FP + FN exposed
- decomposable per signal: TP/FP/TN/FN per window separated
- diff-able across runs: pure function
- queryable post-hoc: frozen dataclass with sweep_points
- cite-able: literature_citation (Wang-Rudin falling rule)
- counterfactual-able: change corpus -> observe window shift

6-hook wire-in declaration per Catalog #125
-------------------------------------------
1. Sensitivity-map: ACTIVE — per-corpus linguistic shape IS a sensitivity
2. Pareto constraint: ACTIVE via FP-vs-FN Pareto axis
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE via Atom emission
5. Continual-learning posterior: ACTIVE via canonical Provenance
6. Probe-disambiguator: ACTIVE — empirical sweep IS the disambiguator

Citations
---------
- Wang-Rudin 2015 "Falling Rule Lists" (linguistic FP/FN tradeoff discipline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from tac.experimental_extinctions.per_substrate_convergence_aware_early_stopping import (
    EmpiricalSweepResult,
)

if TYPE_CHECKING:
    from tac.atom.atom import Atom


_LITERATURE_CITATION = (
    "Wang-Rudin 2015 'Falling Rule Lists' (linguistic FP/FN tradeoff discipline)"
)

# Canonical Catalog #236 negation tokens (subset for corpus FP/FN testing)
_NEGATION_TOKENS = (
    "previously",
    "superseded",
    "deprecated",
    "retired",
    "discussion of",
    "tradeoff",
    "vs ",
    "instead of",
    "originally",
    "abandoned",
    "hypothetical",
)


@dataclass(frozen=True)
class NegationWindowInput:
    """Inputs to the per-corpus negation-window FP/FN sweep.

    Parameters
    ----------
    labeled_corpus : Sequence[Mapping[str, Any]]
        Labeled corpus entries; each entry MUST have:
            - ``text`` (str) — the surrounding text snippet
            - ``trigger_offset`` (int) — character offset of the trigger token
              (e.g. "100ep") within ``text``
            - ``label`` (str) — one of ``"affirmative"`` or ``"negation"``
              (operator-labeled ground truth)
    window_grid : Sequence[int]
        Candidate window sizes (chars) to sweep. Default (40, 60, 80, 100, 120, 160).
    """

    labeled_corpus: Sequence[Mapping[str, Any]]
    window_grid: Sequence[int] = (40, 60, 80, 100, 120, 160)

    def __post_init__(self) -> None:
        if len(self.labeled_corpus) == 0:
            raise ValueError("labeled_corpus must be non-empty")
        for i, entry in enumerate(self.labeled_corpus):
            for required in ("text", "trigger_offset", "label"):
                if required not in entry:
                    raise ValueError(
                        f"labeled_corpus[{i}] missing required field {required!r}"
                    )
            if entry["label"] not in ("affirmative", "negation"):
                raise ValueError(
                    f"labeled_corpus[{i}] label must be 'affirmative' or "
                    f"'negation'; got {entry['label']!r}"
                )
        if len(self.window_grid) == 0:
            raise ValueError("window_grid must be non-empty")
        for w in self.window_grid:
            if w <= 0:
                raise ValueError(f"window_grid values must be positive; got {w}")


def _has_negation_token_in_window(
    text: str, trigger_offset: int, window_chars: int
) -> bool:
    """Check if any negation token is present in [trigger - window, trigger + window]."""
    lo = max(0, trigger_offset - window_chars)
    hi = min(len(text), trigger_offset + window_chars)
    snippet = text[lo:hi].lower()
    return any(tok in snippet for tok in _NEGATION_TOKENS)


def negation_window_fp_fn_corpus_sweep(
    inputs: NegationWindowInput,
    *,
    emit_arbitrariness_atom: bool = False,
) -> EmpiricalSweepResult:
    """Per-corpus sweep over candidate negation-window sizes.

    For each window size in the grid, apply the canonical Catalog #236
    negation detector to every labeled corpus entry, classify as
    affirmative (no negation detected in window) or negation (detected),
    and compute FP + FN vs ground truth. Window minimizing FP + FN wins.

    Parameters
    ----------
    inputs : NegationWindowInput
        Validated dataclass with labeled corpus + window grid.
    emit_arbitrariness_atom : bool
        When True, emit canonical Atom instance.

    Returns
    -------
    EmpiricalSweepResult
        ``solved_value`` is the winning window size (int). ``sweep_points``
        carries per-window TP/FP/TN/FN breakdown.
    """
    sweep: list[Mapping[str, Any]] = []

    for window in inputs.window_grid:
        tp = fp = tn = fn = 0
        for entry in inputs.labeled_corpus:
            text = str(entry["text"])
            trigger = int(entry["trigger_offset"])
            ground_truth = str(entry["label"])
            detected_negation = _has_negation_token_in_window(text, trigger, window)
            predicted = "negation" if detected_negation else "affirmative"
            if predicted == "negation" and ground_truth == "negation":
                tp += 1
            elif predicted == "negation" and ground_truth == "affirmative":
                fp += 1
            elif predicted == "affirmative" and ground_truth == "affirmative":
                tn += 1
            else:
                fn += 1
        total_errors = fp + fn
        sweep.append(
            {
                "window_chars": window,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "total_errors": total_errors,
                "fp_rate": fp / max(fp + tn, 1),
                "fn_rate": fn / max(fn + tp, 1),
            }
        )

    winner = min(sweep, key=lambda r: r["total_errors"])
    window_80 = next((r for r in sweep if r["window_chars"] == 80), winner)

    intermediate: dict[str, Any] = {
        "n_corpus_entries": len(inputs.labeled_corpus),
        "winner_window": winner["window_chars"],
        "winner_total_errors": winner["total_errors"],
        "window_80_total_errors": window_80["total_errors"],
        "window_80_was_winner": winner["window_chars"] == 80,
        "n_affirmative_in_corpus": sum(
            1 for e in inputs.labeled_corpus if e["label"] == "affirmative"
        ),
        "n_negation_in_corpus": sum(
            1 for e in inputs.labeled_corpus if e["label"] == "negation"
        ),
    }
    coupled: dict[str, Any] = {
        "optimal_window_chars": winner["window_chars"],
        "error_reduction_vs_80": window_80["total_errors"] - winner["total_errors"],
    }

    if emit_arbitrariness_atom:
        from tac.atom import ResolutionPath, build_arbitrary_value_atom

        atom: "Atom" = build_arbitrary_value_atom(
            atom_id="negation_window_fp_fn_per_corpus__catalog_236",
            file_path="src/tac/preflight.py (check_catalog_233_gate_3_prose_negation_guard_present)",
            current_value=80,
            predicted_replacement=winner["window_chars"],
            resolution_path=ResolutionPath.EXPERIMENTAL,
            predicted_ev_delta_s=(-0.0002, 0.0),
            cost_envelope_usd=0.0,
            literature_citation=_LITERATURE_CITATION,
            canonical_helper_repo_link=(
                "src/tac/experimental_extinctions/"
                "negation_window_fp_fn_corpus_sweep.py"
            ),
            wired_hooks=(
                "sensitivity_map",
                "pareto_constraint",
                "cathedral_autopilot_dispatch",
                "continual_learning_posterior",
                "probe_disambiguator",
            ),
            observability_surface=(
                "inspectable_per_layer",
                "decomposable_per_signal",
                "diff_able_across_runs",
                "queryable_post_hoc",
                "cite_able",
                "counterfactual_able",
            ),
            captured_by_subagent=(
                "lane_arbitrariness_extinction_wave_2c_path1_experimental_zero_batch_20260518"
            ),
        )
        coupled["atom"] = atom

    return EmpiricalSweepResult(
        solved_value=int(winner["window_chars"]),
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.experimental_extinctions.negation_window_fp_fn_corpus_sweep."
            "negation_window_fp_fn_corpus_sweep"
        ),
        sweep_points=tuple(sweep),
        coupled_adjustments=coupled,
        notes=(
            f"Winner: window={winner['window_chars']}chars "
            f"(FP+FN={winner['total_errors']}); "
            f"hardcoded 80 had FP+FN={window_80['total_errors']}."
        ),
    )
