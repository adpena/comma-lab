# SPDX-License-Identifier: MIT
"""Canonical analytical-solve helpers for the 10 remaining Wave 2A extinctions.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290
canonical-vs-unique decision discipline: each module SOLVES ONE arbitrary
value via a closed-form formula or analytical optimization, replacing a
hand-picked default with a derived value.

Sister packages (DISJOINT scope; do NOT import to mutate):
    - ``tac.contest_oracle`` — canonical contest-oracle classifier (sister
      subagent ``contest_oracle_canonical_package_20260518``).
    - ``tac.atom`` — canonical atom ledger surfacing arbitrariness atoms.
    - ``tac.provenance`` — canonical provenance contract.
    - ``tac.score_lagrangian`` + ``tac.training`` — sister TOP-1+TOP-4
      analytical-solve landings (lambda multipliers + EMA decay).
    - ``tac.master_gradient*`` — sister master-gradient family (READ-ONLY here).

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer experience"):

    from tac.analytical_solve_extinctions import (
        solve_vram_aware_batch_size,
        solve_rd_theoretic_vq_codebook_K,
        solve_optimal_block_fp_block_size,
        solve_min_spanning_tree_frame_ordering,
        solve_roc_optimal_high_pair_invariant_threshold,
        solve_coupling_threshold_statistical,
        solve_sgld_t_final_welling_teh,
        solve_bootstrap_ci_rashomon_K,
        solve_greedy_tsp_per_pair_ordering,
    )

Each helper emits a structured ``AnalyticalSolveResult`` and optionally a
``tac.atom.Atom`` instance via ``emit_arbitrariness_atom=True`` so downstream
consumers (cathedral autopilot ranker / continual-learning posterior /
probe-disambiguator) see the canonical extinction record per Catalog #125
6-hook wire-in declaration.

Citations
---------
- Goyal et al 2017 "Accurate, Large Minibatch SGD" arxiv:1706.02677
- Gersho-Gray 1992 "Vector Quantization and Signal Compression"
- Ballé-Laparra-Simoncelli 2017 "End-to-end Optimized Image Compression" arxiv:1611.01704
- Welling-Teh 2011 "Bayesian Learning via Stochastic Gradient Langevin Dynamics" ICML
- Fisher-Rudin-Dominici 2019 "All Models are Wrong, but Many are Useful" arxiv:1801.01489
- Cover-Thomas "Elements of Information Theory" (R-D theoretic K derivation)
- Cormen-Leiserson-Rivest-Stein "Introduction to Algorithms" (MST + TSP)
- Boyd-Vandenberghe "Convex Optimization" (KKT conditions + closed-form solves)

Lane: ``lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518``
"""

from __future__ import annotations

from tac.analytical_solve_extinctions.vram_aware_batch_size import (  # noqa: F401
    AnalyticalSolveResult,
    BatchSizeSolverInput,
    solve_vram_aware_batch_size,
)
from tac.analytical_solve_extinctions.rd_theoretic_vq_codebook_K import (  # noqa: F401
    RDCodebookSolverInput,
    solve_rd_theoretic_vq_codebook_K,
)
from tac.analytical_solve_extinctions.optimal_block_fp_block_size import (  # noqa: F401
    BlockFPSolverInput,
    solve_optimal_block_fp_block_size,
)
from tac.analytical_solve_extinctions.min_spanning_tree_frame_ordering import (  # noqa: F401
    FrameOrderingInput,
    solve_min_spanning_tree_frame_ordering,
)
from tac.analytical_solve_extinctions.roc_optimal_high_pair_invariant_threshold import (  # noqa: F401
    ROCThresholdInput,
    solve_roc_optimal_high_pair_invariant_threshold,
)
from tac.analytical_solve_extinctions.coupling_threshold_statistical_derivation import (  # noqa: F401
    CouplingThresholdInput,
    solve_coupling_threshold_statistical,
)
from tac.analytical_solve_extinctions.sgld_t_final_welling_teh import (  # noqa: F401
    SGLDTFinalInput,
    solve_sgld_t_final_welling_teh,
)
from tac.analytical_solve_extinctions.bootstrap_ci_rashomon_K import (  # noqa: F401
    RashomonKInput,
    solve_bootstrap_ci_rashomon_K,
)
from tac.analytical_solve_extinctions.greedy_tsp_per_pair_ordering import (  # noqa: F401
    PairOrderingInput,
    solve_greedy_tsp_per_pair_ordering,
)

__all__ = [
    # Shared result dataclass
    "AnalyticalSolveResult",
    # Per-row helpers + inputs (10 total; rows #2 + #3 batched in rd_theoretic)
    "BatchSizeSolverInput",
    "solve_vram_aware_batch_size",
    "RDCodebookSolverInput",
    "solve_rd_theoretic_vq_codebook_K",
    "BlockFPSolverInput",
    "solve_optimal_block_fp_block_size",
    "FrameOrderingInput",
    "solve_min_spanning_tree_frame_ordering",
    "ROCThresholdInput",
    "solve_roc_optimal_high_pair_invariant_threshold",
    "CouplingThresholdInput",
    "solve_coupling_threshold_statistical",
    "SGLDTFinalInput",
    "solve_sgld_t_final_welling_teh",
    "RashomonKInput",
    "solve_bootstrap_ci_rashomon_K",
    "PairOrderingInput",
    "solve_greedy_tsp_per_pair_ordering",
]
