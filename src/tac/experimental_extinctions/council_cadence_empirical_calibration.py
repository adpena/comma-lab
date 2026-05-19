# SPDX-License-Identifier: MIT
"""Row #5: Council 4-tier cadence empirical calibration from posterior.

Replaces ``council_4_tier_cadence in {3/day, 3/week, ...}`` hardcoded
arbitrary defaults (per ``.omx/state/arbitrariness_extinction_audit_20260518.jsonl``
row ``council_4_tier_cadence_3_per_day_3_per_week_arbitrary``) with an
empirical calibration that reads ``.omx/state/council_deliberation_posterior.jsonl``
(canonical helper ``tac.council_continual_learning``; READ-ONLY consumer per
sister-coordination map) and computes the actual per-tier deliberation throughput,
then emits the empirically-justified cadence.

Predicted EV: [-0.001, 0.0] per ``.omx/research/arbitrariness_extinction_audit_top_50_ranked_20260518.md``.

Empirical anchor (expected): hardcoded cadence may match or diverge from
actual T2/T3/T4 throughput. Either result is signal: if actual matches
hardcoded, the hardcoded value is validated; if actual diverges, the
hardcoded value should be replaced with the empirical anchor.

Canonical-vs-unique decision per layer
--------------------------------------
- Data type / framework: ADOPT_CANONICAL (Python dict + stdlib only)
- Sweep pattern: UNIQUE (per-tier throughput analytics over time window)
- Atom emission: ADOPT_CANONICAL (tac.atom.builders.build_arbitrary_value_atom)
- Provenance: ADOPT_CANONICAL (tac.provenance Catalog #323)
- Reader: ADOPT_CANONICAL (tac.council_continual_learning READ-ONLY)

9-dim success checklist evidence per Catalog #294
-------------------------------------------------
- UNIQUENESS: per-tier empirical throughput, not hardcoded budget
- BEAUTY+ELEGANCE: per-tier count / window-days; ~20-LOC math
- DISTINCTNESS: distinct from other rows (calibration on real state, not synth)
- RIGOR: refuses invalid window, handles empty posterior gracefully
- OPTIMIZATION-PER-TECHNIQUE: matches budget to actual throughput
- STACK-OF-STACKS-COMPOSABILITY: emits Atom + Provenance
- DETERMINISTIC-REPRODUCIBILITY: pure function on input anchors list
- EXTREME-OPTIMIZATION-PERFORMANCE: O(N anchors)
- OPTIMAL-MINIMAL-CONTEST-SCORE: predicted ΔS [-0.001, 0.0]

Observability surface per Catalog #305 (6 facets)
-------------------------------------------------
- inspectable per layer: per-tier count + window split exposed
- decomposable per signal: per-tier T1/T2/T3/T4 throughput separated
- diff-able across runs: pure function on input anchors
- queryable post-hoc: frozen dataclass + sweep_points (per-tier rates)
- cite-able: literature_citation (Surowiecki + Kemeny-Snell)
- counterfactual-able: change anchors -> observe cadence shift

6-hook wire-in declaration per Catalog #125
-------------------------------------------
1. Sensitivity-map: N/A (apparatus-governance, not score-contribution)
2. Pareto constraint: N/A
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE via Atom emission
5. Continual-learning posterior: ACTIVE — reads from canonical posterior
6. Probe-disambiguator: ACTIVE — empirical IS the disambiguator vs hardcoded

Citations
---------
- Surowiecki 2004 "The Wisdom of Crowds" (group decision throughput)
- Kemeny-Snell 1962 "Mathematical Models in the Social Sciences" (preferences)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from tac.experimental_extinctions.per_substrate_convergence_aware_early_stopping import (
    EmpiricalSweepResult,
)

if TYPE_CHECKING:
    from tac.atom.atom import Atom


_LITERATURE_CITATION = (
    "Surowiecki 2004 'The Wisdom of Crowds' (group decision throughput); "
    "Kemeny-Snell 1962 'Mathematical Models in the Social Sciences' "
    "(preference aggregation)"
)

# Canonical CLAUDE.md hardcoded cadence budgets per "Council hierarchy: 4-tier protocol"
_HARDCODED_CADENCE_BUDGETS = {
    "T1": {"per_day": -1, "per_30d": -1},  # unbounded
    "T2": {"per_day": 3, "per_30d": 90},
    "T3": {"per_week": 3, "per_30d": 13},
    "T4": {"per_30d": 2},
}


@dataclass(frozen=True)
class CouncilCadenceInput:
    """Inputs to the empirical council cadence calibration.

    Parameters
    ----------
    deliberation_anchors : Sequence[Mapping[str, Any]]
        Council deliberation records loaded from
        ``.omx/state/council_deliberation_posterior.jsonl``. MUST have
        ``council_tier`` field (one of T1/T2/T3/T4) and a timestamp field
        (``written_at_utc`` or ``deliberation_id`` containing YYYYMMDD).
    window_days : int
        Time window in days for throughput calibration. Default 30 matches
        CLAUDE.md cadence-budget window.
    end_utc : str | None
        End of the time window (ISO UTC). When None, uses now().
    """

    deliberation_anchors: Sequence[Mapping[str, Any]]
    window_days: int = 30
    end_utc: str | None = None

    def __post_init__(self) -> None:
        if self.window_days <= 0:
            raise ValueError(f"window_days must be positive; got {self.window_days}")


def _parse_anchor_timestamp(anchor: Mapping[str, Any]) -> datetime | None:
    """Best-effort timestamp parse from a council deliberation anchor."""
    if "written_at_utc" in anchor:
        try:
            return datetime.fromisoformat(str(anchor["written_at_utc"]).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    delib_id = str(anchor.get("deliberation_id", ""))
    # Look for trailing _YYYYMMDD or _<YYYY-MM-DD>
    for part in delib_id.split("_"):
        if len(part) == 8 and part.isdigit():
            try:
                return datetime(
                    year=int(part[:4]),
                    month=int(part[4:6]),
                    day=int(part[6:8]),
                    tzinfo=timezone.utc,
                )
            except (ValueError, TypeError):
                pass
    return None


def council_cadence_empirical_calibration(
    inputs: CouncilCadenceInput,
    *,
    emit_arbitrariness_atom: bool = False,
) -> EmpiricalSweepResult:
    """Compute per-tier empirical deliberation throughput vs hardcoded budgets.

    Reads the deliberation anchors (canonically loaded from
    ``.omx/state/council_deliberation_posterior.jsonl`` via
    ``tac.council_continual_learning``), filters to the time window, computes
    per-tier counts + per-tier per-day / per-30d rates, and emits the
    empirical anchor for the operator + autopilot to consume.

    Parameters
    ----------
    inputs : CouncilCadenceInput
        Validated dataclass with deliberation anchors + window.
    emit_arbitrariness_atom : bool
        When True, emit canonical Atom instance.

    Returns
    -------
    EmpiricalSweepResult
        ``solved_value`` is a dict mapping tier -> empirical per-day rate.
        ``sweep_points`` carries per-tier observed counts + budget compliance.
    """
    end = (
        datetime.fromisoformat(inputs.end_utc.replace("Z", "+00:00"))
        if inputs.end_utc
        else datetime.now(timezone.utc)
    )
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=inputs.window_days)

    in_window: list[Mapping[str, Any]] = []
    skipped_no_timestamp = 0
    for anchor in inputs.deliberation_anchors:
        ts = _parse_anchor_timestamp(anchor)
        if ts is None:
            skipped_no_timestamp += 1
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if start <= ts <= end:
            in_window.append(anchor)

    tier_counts = Counter(str(a.get("council_tier", "<unknown>")) for a in in_window)

    sweep: list[Mapping[str, Any]] = []
    for tier in ("T1", "T2", "T3", "T4"):
        observed = tier_counts.get(tier, 0)
        empirical_per_day = observed / max(inputs.window_days, 1)
        budgets = _HARDCODED_CADENCE_BUDGETS.get(tier, {})
        compliance: Mapping[str, Any] = {
            f"hardcoded_{k}": v for k, v in budgets.items()
        }
        sweep.append(
            {
                "tier": tier,
                "observed_count_in_window": observed,
                "empirical_per_day": empirical_per_day,
                "empirical_per_30d": observed
                * (30.0 / max(inputs.window_days, 1)),
                **compliance,
            }
        )

    empirical_solved = {row["tier"]: row["empirical_per_day"] for row in sweep}

    intermediate: dict[str, Any] = {
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "window_days": inputs.window_days,
        "n_anchors_total": len(inputs.deliberation_anchors),
        "n_anchors_in_window": len(in_window),
        "n_anchors_skipped_no_timestamp": skipped_no_timestamp,
        "tier_counts": dict(tier_counts),
    }
    coupled: dict[str, Any] = {
        "empirical_per_tier_per_day": empirical_solved,
        "hardcoded_vs_empirical_T2_per_day": {
            "hardcoded": 3,
            "empirical": empirical_solved.get("T2", 0.0),
        },
        "hardcoded_vs_empirical_T3_per_week": {
            "hardcoded": 3,
            "empirical": empirical_solved.get("T3", 0.0) * 7.0,
        },
    }

    if emit_arbitrariness_atom:
        from tac.atom import ResolutionPath, build_arbitrary_value_atom

        atom: "Atom" = build_arbitrary_value_atom(
            atom_id="council_4_tier_cadence_empirical_calibration",
            file_path="CLAUDE.md (Council hierarchy: 4-tier protocol)",
            current_value="T2 3/day + T3 3/week + T4 2/30d (hardcoded)",
            predicted_replacement=empirical_solved,
            resolution_path=ResolutionPath.EXPERIMENTAL,
            predicted_ev_delta_s=(-0.001, 0.0),
            cost_envelope_usd=0.0,
            literature_citation=_LITERATURE_CITATION,
            canonical_helper_repo_link=(
                "src/tac/experimental_extinctions/"
                "council_cadence_empirical_calibration.py"
            ),
            wired_hooks=(
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
        solved_value=empirical_solved,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.experimental_extinctions.council_cadence_empirical_calibration."
            "council_cadence_empirical_calibration"
        ),
        sweep_points=tuple(sweep),
        coupled_adjustments=coupled,
        notes=(
            f"Observed {len(in_window)} anchors in {inputs.window_days}-day window "
            f"(skipped {skipped_no_timestamp} missing-timestamp anchors)."
        ),
    )
