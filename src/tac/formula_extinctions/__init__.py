# SPDX-License-Identifier: MIT
"""Canonical formula-path helpers for the 11 remaining Wave 2B extinctions.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290
canonical-vs-unique decision discipline: each module REPLACES ONE arbitrary
value with a CLOSED-FORM FORMULA derived from a published canonical source.

Sister packages (DISJOINT scope; do NOT mutate):
    - ``tac.contest_oracle`` — canonical contest-oracle classifier (sister
      subagent ``contest_oracle_canonical_package_20260518`` LANDED).
    - ``tac.analytical_solve_extinctions`` — sister Wave 2A analytical-solve
      helpers (sister subagent ``wave_2a_analytical_solve_batch_20260518``).
    - ``tac.atom`` — canonical atom ledger surfacing arbitrariness atoms.
    - ``tac.provenance`` — canonical provenance contract.
    - ``tac.score_lagrangian`` + ``tac.training`` — sister TOP-1+TOP-4
      formula-path landings (lambda multipliers + EMA decay).
    - ``tac.frontier_scan`` — Catalog #316 canonical frontier helper.

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer experience"):

    from tac.formula_extinctions import (
        # Row #1 — Goyal+He 2017 linear warmup
        canonical_warmup_steps,
        # Row #2 — stratified k-fold across video chunks
        canonical_validation_split,
        # Row #3 — R-D-theoretic qint_max grid justification
        canonical_qint_max_grid_rd_proof,
        # Row #4 — per-archive inflate device-pin metadata
        canonical_inflate_device_pin_metadata,
        # Row #5 — Bayesian-aggregation quorum (Surowiecki + Kemeny-Snell)
        canonical_bayesian_aggregation_quorum,
        # Row #6 — frontier threshold from canonical state
        canonical_frontier_threshold_from_state,
        # Row #7 — Prechelt 1998 validation-slope early stopping
        canonical_early_stopping_patience,
        # Row #8 — HNeRV parity L4 inflate.py LOC budget derivation
        canonical_inflate_py_loc_budget,
        # Row #9 — Pascanu 2013 gradient-norm clipping
        canonical_gradient_clipping_norm,
        # Row #10 — Catalog quota from preflight time budget
        canonical_catalog_quota_from_preflight_budget,
        # Row #11 — Smith 2017 warmup init_lr factor
        canonical_lr_warmup_init_lr_factor,
        # Shared result dataclass
        FormulaSolveResult,
    )

Each helper emits a structured ``FormulaSolveResult`` and optionally a
``tac.atom.Atom`` instance via ``emit_arbitrariness_atom=True`` so downstream
consumers (cathedral autopilot ranker / continual-learning posterior /
probe-disambiguator) see the canonical extinction record per Catalog #125
6-hook wire-in declaration.

Citations
---------
- Goyal et al 2017 "Accurate, Large Minibatch SGD" arxiv:1706.02677 §2.2 (warmup)
- He et al 2016 "Deep Residual Learning for Image Recognition" arxiv:1512.03385 (warmup canonical)
- Bengio 2012 "Practical Recommendations for Gradient-Based Training" arxiv:1206.5533 §2.2 (k-fold)
- Cover-Thomas 1991 "Elements of Information Theory" Ch.13 (R-D theory + qint grid)
- Surowiecki 2004 "The Wisdom of Crowds" (Bayesian aggregation quorum)
- Kemeny-Snell 1962 "Mathematical Models in the Social Sciences" (preference aggregation)
- Prechelt 1998 "Early Stopping — But When?" Neural Networks: Tricks of the Trade
- Pascanu+Mikolov+Bengio 2013 "On the difficulty of training RNNs" arxiv:1211.5063
- Smith 2017 "Cyclical Learning Rates" arxiv:1506.01186 §3.2 (warmup init_lr)
- Hastie-Tibshirani-Friedman 2009 "Elements of Statistical Learning" Ch.16

Lane: ``lane_arbitrariness_extinction_wave_2b_path3_formula_batch_20260518``
"""

from __future__ import annotations

from tac.formula_extinctions.canonical_warmup_schedule import (  # noqa: F401
    FormulaSolveResult,
    WarmupScheduleInput,
    canonical_warmup_steps,
)
from tac.formula_extinctions.stratified_kfold_video_chunks import (  # noqa: F401
    ValidationSplitInput,
    canonical_validation_split,
)
from tac.formula_extinctions.qint_max_grid_rd_justification import (  # noqa: F401
    QintMaxGridInput,
    canonical_qint_max_grid_rd_proof,
)
from tac.formula_extinctions.inflate_device_pin_metadata import (  # noqa: F401
    InflateDevicePinInput,
    canonical_inflate_device_pin_metadata,
)
from tac.formula_extinctions.bayesian_aggregation_quorum import (  # noqa: F401
    QuorumInput,
    canonical_bayesian_aggregation_quorum,
)
from tac.formula_extinctions.canonical_frontier_threshold_from_state import (  # noqa: F401
    FrontierThresholdInput,
    canonical_frontier_threshold_from_state,
)
from tac.formula_extinctions.early_stopping_prechelt_slope import (  # noqa: F401
    EarlyStoppingInput,
    canonical_early_stopping_patience,
)
from tac.formula_extinctions.inflate_py_loc_budget_derivation import (  # noqa: F401
    LOCBudgetInput,
    canonical_inflate_py_loc_budget,
)
from tac.formula_extinctions.gradient_clipping_pascanu_canonical import (  # noqa: F401
    GradientClipInput,
    canonical_gradient_clipping_norm,
)
from tac.formula_extinctions.catalog_quota_from_preflight_time_budget import (  # noqa: F401
    CatalogQuotaInput,
    canonical_catalog_quota_from_preflight_budget,
)
from tac.formula_extinctions.lr_warmup_init_smith_canonical import (  # noqa: F401
    WarmupInitLRInput,
    canonical_lr_warmup_init_lr_factor,
)

__all__ = [
    # Shared result dataclass
    "FormulaSolveResult",
    # Row #1 Goyal+He warmup
    "WarmupScheduleInput",
    "canonical_warmup_steps",
    # Row #2 stratified k-fold
    "ValidationSplitInput",
    "canonical_validation_split",
    # Row #3 R-D qint grid
    "QintMaxGridInput",
    "canonical_qint_max_grid_rd_proof",
    # Row #4 inflate device pin
    "InflateDevicePinInput",
    "canonical_inflate_device_pin_metadata",
    # Row #5 Bayesian quorum
    "QuorumInput",
    "canonical_bayesian_aggregation_quorum",
    # Row #6 frontier threshold
    "FrontierThresholdInput",
    "canonical_frontier_threshold_from_state",
    # Row #7 Prechelt early stopping
    "EarlyStoppingInput",
    "canonical_early_stopping_patience",
    # Row #8 inflate.py LOC budget
    "LOCBudgetInput",
    "canonical_inflate_py_loc_budget",
    # Row #9 Pascanu gradient clipping
    "GradientClipInput",
    "canonical_gradient_clipping_norm",
    # Row #10 catalog quota
    "CatalogQuotaInput",
    "canonical_catalog_quota_from_preflight_budget",
    # Row #11 Smith warmup init_lr
    "WarmupInitLRInput",
    "canonical_lr_warmup_init_lr_factor",
]
