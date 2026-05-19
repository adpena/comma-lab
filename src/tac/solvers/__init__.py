# SPDX-License-Identifier: MIT
"""More-optimal solver primitives for empirical paired comparison vs canonical helpers.

Per the synthesis memo amendment (commit ``7b231f4fa``) and operator standing
directive 2026-05-18 ("If there are more optimal algorithms or engineering or
meta — do that and pursue and test and digest and research and experiment"),
this subpackage hosts 5 candidate solvers paired against the canonical
``tac.water_filling_codec`` / ``tac.bit_allocator`` / Rashomon-ensemble paths
on local M5 Max CPU at zero GPU envelope.

Every result emitted from this subpackage is ``[macOS-CPU advisory]`` per
CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 + Catalog #317 — never
promoted to ``[contest-CPU]`` without a paired Linux x86_64 anchor.

Modules:

* :mod:`tac.solvers.fista` — Beck-Teboulle 2009 accelerated proximal gradient
* :mod:`tac.solvers.frank_wolfe` — Frank-Wolfe 1956 / Jaggi 2013 conditional-grad
* :mod:`tac.solvers.sinkhorn` — Cuturi 2013 entropic-regularized optimal transport
* :mod:`tac.solvers.riemannian_newton_stiefel` — Edelman-Arias-Smith 1998 manifold opt
* :mod:`tac.solvers.numba_jit_water_filling` — Numba-JIT canonical water-filling
* :mod:`tac.solvers.more_optimal_algorithms` — canonical shim package + per-algorithm
  metadata registry (PROCEED'd at council T3 finding #6 2026-05-18) for consumer-side
  adoption decisions per Catalog #290 canonical-vs-unique decision per layer.

Each module ≤ 200 LOC (the canonical shim is ≤ 280 incl. metadata) per operator's
"beautiful elegant composable" directive.
"""

from __future__ import annotations

__all__ = [
    "fista",
    "frank_wolfe",
    "sinkhorn",
    "riemannian_newton_stiefel",
    "numba_jit_water_filling",
    "more_optimal_algorithms",
]
