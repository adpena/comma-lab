#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Select next K dispatches via compressive-sensing coherence-minimization +
Bayesian sequential experimental design.

Thin CLI wrapper around
:class:`tac.autopilot_rudin_daubechies.CoherenceMinimizingSelector` (Tropp 2004 RIP)
+ :class:`tac.autopilot_rudin_daubechies.BayesianSequentialKSelector`
(Snoek-Larochelle-Adams 2012 + Ji-Xue-Carin 2008).

Two modes per operator approval 2026-05-16
------------------------------------------
* ``--mode coherence`` — Tropp 2004 RIP-preserving greedy selection.
  Use EARLY (few anchors) when the posterior is uninformative.  Enforces
  the horizon-class budget distribution (<=30% plateau, >=40%
  frontier-pursuit, >=20% asymptotic-pursuit per the standing directive
  2026-05-16).
* ``--mode bayesian`` — Snoek-Larochelle-Adams EIG-maximizing selection
  conditional on a posterior from the sister
  ``tools/compressive_sensing_lattice_recovery.py`` helper.  Use LATE
  (many anchors).

Observability surface (per max-observability standing directive 2026-05-16)
---------------------------------------------------------------------------
* Input snapshot: lattice JSON + optional posterior JSON + K + mode + budget
* Output snapshot: ranked K-selection JSON with per-substrate rationale +
  pairwise-coherence matrix + horizon-class budget consumption
* Decision-path: which mode fired (coherence vs bayesian) + which fallback
  (Thompson sampling when posterior missing)
* Cite-chain: every selection tagged with the canonical
  ``[compressive-sensing-K-selection; mode={mode}; K={k}; basis_from_posterior={basis}]``

Sample command::

    python tools/select_next_K_dispatches_per_compressive_sensing.py \\
        --lattice-json my_lattice.json \\
        --K 8 --mode coherence \\
        --output-json my_K_selection.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make canonical helper importable without installed package via repo layout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Make the sibling tool importable when this script is invoked directly
# without ``tools`` already on sys.path.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tac.autopilot_rudin_daubechies import (  # noqa: E402
    BayesianSequentialKSelector,
    CoherenceMinimizingSelector,
    FrontierPursuitClass,
    SparseSignalPosterior,
    SubstrateLatticeNode,
    compute_pairwise_coherence,
)
from tools.compressive_sensing_lattice_recovery import _parse_lattice_json  # noqa: E402


def _load_posterior(path: Path) -> SparseSignalPosterior:
    """Load a SparseSignalPosterior from its observability JSON record."""
    with path.open("r", encoding="utf-8") as fh:
        rec = json.load(fh)
    return SparseSignalPosterior(
        n_substrates=int(rec.get("n_substrates", 0)),
        n_anchors=int(rec.get("n_anchors", 0)),
        sparsity_recovered=int(rec.get("sparsity_recovered", 0)),
        basis=str(rec.get("basis", "haar_db1")),
        tree_depth_max=int(rec.get("tree_depth_max", 0)),
        posterior_frontier_probability=tuple(
            (r["node_id"], r["probability"])
            for r in rec.get("posterior_frontier_probability", [])
        ),
        posterior_score_band=tuple(
            (r["node_id"], r["low"], r["high"])
            for r in rec.get("posterior_score_band", [])
        ),
        recovery_uncertainty=tuple(
            (r["node_id"], r["uncertainty"])
            for r in rec.get("recovery_uncertainty", [])
        ),
        confidence_tag=str(rec.get("confidence_tag", "")),
    )


def _format_selection_record(
    selected: list[SubstrateLatticeNode],
    *,
    mode: str,
    K: int,
    posterior: SparseSignalPosterior | None,
    coherence: dict[tuple[str, str], float],
    plateau_budget_max: float,
    frontier_pursuit_budget_min: float,
    asymptotic_budget_min: float,
) -> dict[str, Any]:
    """Format the K-selection as a JSON-serializable observability record."""
    plateau = sum(
        1 for s in selected if s.frontier_pursuit_class == FrontierPursuitClass.PLATEAU_ADJACENT
    )
    frontier = sum(
        1 for s in selected if s.frontier_pursuit_class == FrontierPursuitClass.FRONTIER_PURSUIT
    )
    asymptotic = sum(
        1 for s in selected if s.frontier_pursuit_class == FrontierPursuitClass.ASYMPTOTIC_PURSUIT
    )
    probe = sum(
        1 for s in selected if s.frontier_pursuit_class == FrontierPursuitClass.DISAMBIGUATOR_PROBE
    )
    total = max(1, len(selected))
    # Per-pair coherence among the selection.
    selected_ids = {s.node_id for s in selected}
    pairwise: list[dict[str, Any]] = []
    for (a, b), c in coherence.items():
        if a in selected_ids and b in selected_ids:
            pairwise.append({"node_a": a, "node_b": b, "coherence": c})
    max_coh = max((p["coherence"] for p in pairwise), default=0.0)
    mean_coh = (
        sum(p["coherence"] for p in pairwise) / len(pairwise)
        if pairwise
        else 0.0
    )
    basis_from_posterior = posterior.basis if posterior is not None else "none"
    return {
        "schema": "compressive_sensing_K_selection_v1",
        "mode": mode,
        "K_requested": K,
        "K_selected": len(selected),
        "selected_substrates": [
            {
                "node_id": s.node_id,
                "parent_id": s.parent_id,
                "support_level": s.support_level,
                "predicted_band_low": s.predicted_band_low,
                "predicted_band_high": s.predicted_band_high,
                "predicted_midpoint": s.predicted_midpoint,
                "frontier_pursuit_class": s.frontier_pursuit_class.value,
            }
            for s in selected
        ],
        "horizon_class_budget_consumption": {
            "plateau_adjacent": {
                "count": plateau,
                "fraction": plateau / total,
                "max_allowed_fraction": plateau_budget_max,
                "within_budget": plateau / total <= plateau_budget_max + 1e-6,
            },
            "frontier_pursuit": {
                "count": frontier,
                "fraction": frontier / total,
                "min_required_fraction": frontier_pursuit_budget_min,
                "meets_minimum": frontier / total >= frontier_pursuit_budget_min - 1e-6,
            },
            "asymptotic_pursuit": {
                "count": asymptotic,
                "fraction": asymptotic / total,
                "min_required_fraction": asymptotic_budget_min,
                "meets_minimum": asymptotic / total >= asymptotic_budget_min - 1e-6,
            },
            "disambiguator_probe": {"count": probe, "fraction": probe / total},
        },
        "pairwise_coherence_among_selected": pairwise,
        "pairwise_coherence_max": max_coh,
        "pairwise_coherence_mean": mean_coh,
        "confidence_tag": (
            f"[compressive-sensing-K-selection; mode={mode}; K={K}; "
            f"basis_from_posterior={basis_from_posterior}]"
        ),
    }


def run(
    *,
    lattice_json: Path,
    K: int,
    mode: str,
    output_json: Path,
    posterior_json: Path | None,
    plateau_budget_max: float,
    frontier_pursuit_budget_min: float,
    asymptotic_budget_min: float,
) -> int:
    nodes, _config = _parse_lattice_json(lattice_json)
    coherence = compute_pairwise_coherence(nodes)
    if mode == "coherence":
        selector = CoherenceMinimizingSelector(
            K=K,
            plateau_budget_max=plateau_budget_max,
            frontier_pursuit_budget_min=frontier_pursuit_budget_min,
            asymptotic_budget_min=asymptotic_budget_min,
        )
        selected = selector.select(nodes)
        posterior = None
    elif mode == "bayesian":
        posterior = _load_posterior(posterior_json) if posterior_json else None
        selector = BayesianSequentialKSelector(K=K, posterior=posterior)
        selected = selector.select_next_K(nodes)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'coherence' or 'bayesian'")
    record = _format_selection_record(
        selected,
        mode=mode,
        K=K,
        posterior=posterior,
        coherence=coherence,
        plateau_budget_max=plateau_budget_max,
        frontier_pursuit_budget_min=frontier_pursuit_budget_min,
        asymptotic_budget_min=asymptotic_budget_min,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"[K-selection] mode={mode} K_selected={len(selected)} "
        f"max_coherence={record['pairwise_coherence_max']:.3f} "
        f"mean_coherence={record['pairwise_coherence_mean']:.3f}",
        file=sys.stderr,
    )
    print(f"[K-selection] {record['confidence_tag']}", file=sys.stderr)
    print(f"[K-selection] wrote {output_json}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lattice-json", type=Path, required=True)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=["coherence", "bayesian"],
        default="coherence",
        help="coherence = Tropp 2004 RIP; bayesian = Snoek-Larochelle-Adams 2012",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--posterior-json",
        type=Path,
        default=None,
        help="Required when --mode bayesian; output of tools/compressive_sensing_lattice_recovery.py",
    )
    parser.add_argument(
        "--plateau-budget-max",
        type=float,
        default=0.30,
        help="Max fraction of K-selection that can be plateau_adjacent (default 0.30 per horizon-class directive 2026-05-16)",
    )
    parser.add_argument(
        "--frontier-pursuit-budget-min",
        type=float,
        default=0.40,
        help="Min fraction of K-selection that should be frontier_pursuit (default 0.40)",
    )
    parser.add_argument(
        "--asymptotic-budget-min",
        type=float,
        default=0.20,
        help="Min fraction of K-selection that should be asymptotic_pursuit (default 0.20)",
    )
    args = parser.parse_args(argv)
    if not args.lattice_json.is_file():
        print(f"lattice JSON not found: {args.lattice_json}", file=sys.stderr)
        return 2
    if args.K <= 0:
        print(f"--K must be > 0, got {args.K}", file=sys.stderr)
        return 2
    if args.mode == "bayesian" and (
        args.posterior_json is None or not args.posterior_json.is_file()
    ):
        print(
            "--mode bayesian requires --posterior-json pointing to an existing file",
            file=sys.stderr,
        )
        return 2
    return run(
        lattice_json=args.lattice_json,
        K=args.K,
        mode=args.mode,
        output_json=args.output_json,
        posterior_json=args.posterior_json,
        plateau_budget_max=args.plateau_budget_max,
        frontier_pursuit_budget_min=args.frontier_pursuit_budget_min,
        asymptotic_budget_min=args.asymptotic_budget_min,
    )


if __name__ == "__main__":
    sys.exit(main())
