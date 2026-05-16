#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lattice phase-transition monitor (Donoho-Tanner 2009).

Thin CLI wrapper around
:class:`tac.autopilot_rudin_daubechies.LatticePhaseTransitionMonitor` per
operator approval 2026-05-16.  Alerts when K/N approaches the Donoho-Tanner
sparsity-undersampling threshold (~0.2-0.3 at delta = 0.25 per the canonical
weak-transition curve); prevents under-sampling misleading recovery.

Mathematical foundation
-----------------------
Per Donoho-Tanner 2009 "Counting faces of randomly-projected polytopes
when the projection radically lowers dimension":

  For Gaussian sensing matrices, L1 recovery transitions from EXACT to
  FAILED at the curve rho_S(delta) where delta = K/N and rho = s/K.
  Below the curve: exact recovery with high probability.  Above the
  curve: recovery fails.

Operationalized as an OPERATOR-FACING GUARD per CLAUDE.md "Operator gates
must be wired and used".  Sister of Catalog #270 dispatch-optimization-
protocol verdict pattern: when K/N drops below the safe regime, the
monitor REFUSES to claim confident recovery and surfaces the
under-sampling diagnostic.

Observability surface (per max-observability standing directive 2026-05-16)
---------------------------------------------------------------------------
* Input snapshot: K + N + sparsity_estimate + safety_margin
* Output snapshot: 5-field diagnostic record + recommended_K
* Decision-path: regime classification (EXACT / AT_THRESHOLD / FAILED)
* Cite-chain: ``[phase-transition-monitor; donoho-tanner-2009; K={k}; N={n}; s={s}; regime={regime}]``

Exit codes
----------
* 0 if regime == EXACT and ``--strict`` is not set
* 0 if regime == EXACT and ``--strict`` is set
* 1 if regime in {AT_THRESHOLD, FAILED} and ``--strict`` is set
* 0 otherwise (advisory mode)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make canonical helper importable without installed package via repo layout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.autopilot_rudin_daubechies import (  # noqa: E402
    LatticePhaseTransitionMonitor,
)


def run(
    *,
    K: int,
    N: int,
    sparsity_estimate: int,
    safety_margin: float,
    output_json: Path | None,
    strict: bool,
) -> int:
    monitor = LatticePhaseTransitionMonitor(safety_margin=safety_margin)
    diagnostic = monitor.compute_undersampling_diagnostic(
        K=K, N=N, sparsity_estimate=sparsity_estimate
    )
    tag = (
        f"[phase-transition-monitor; donoho-tanner-2009; K={K}; N={N}; "
        f"s={sparsity_estimate}; regime={diagnostic['recovery_regime']}]"
    )
    diagnostic["confidence_tag"] = tag
    diagnostic["schema"] = "lattice_phase_transition_diagnostic_v1"
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as fh:
            json.dump(diagnostic, fh, indent=2, sort_keys=True)
            fh.write("\n")
    print(json.dumps(diagnostic, indent=2, sort_keys=True))
    print(tag, file=sys.stderr)
    if strict and diagnostic["recovery_regime"] != "EXACT":
        recommended = diagnostic.get("recommended_K")
        msg = (
            f"[phase-transition-monitor] STRICT REFUSE: regime="
            f"{diagnostic['recovery_regime']} (rho={diagnostic['rho']:.3f} "
            f"vs threshold {diagnostic['rho_threshold']:.3f})"
        )
        if recommended is not None:
            msg += f"; recommended_K={recommended}"
        print(msg, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--K", type=int, required=True, help="Number of empirical anchors"
    )
    parser.add_argument(
        "--N", type=int, required=True, help="Number of substrates in the lattice"
    )
    parser.add_argument(
        "--sparsity-estimate",
        type=int,
        required=True,
        help="Expected number of frontier-breaking substrates (s)",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.05,
        help="Safety margin below the Donoho-Tanner threshold (default 0.05)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the diagnostic JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit rc=1 when regime is AT_THRESHOLD or FAILED (default advisory)",
    )
    args = parser.parse_args(argv)
    return run(
        K=args.K,
        N=args.N,
        sparsity_estimate=args.sparsity_estimate,
        safety_margin=args.safety_margin,
        output_json=args.output_json,
        strict=args.strict,
    )


if __name__ == "__main__":
    sys.exit(main())
