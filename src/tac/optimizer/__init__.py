"""Meta-Lagrangian optimizer for extreme automated Shannon-floor search.

Integrates:
  - tac.predictor.score_band     — score-band prediction with refusal modes
  - experiments.distortion_proxy_local — closed-form distortion estimator
  - tools/predispatch_sanity.py  — 5-gate ladder before any GPU dispatch

Produces a ranked candidate queue for paid-eval dispatch, gated by all three.
The search is automated (single CLI call walks a candidate generator) but
every advance to GPU spend goes through the same hardened gates that the
2026-05-05 catastrophe wave proved necessary.
"""
from tac.optimizer.meta_lagrangian import (
    CandidateEvaluation,
    LagrangianConstraints,
    MetaLagrangianSearch,
    contest_score,
)

__all__ = [
    "CandidateEvaluation",
    "LagrangianConstraints",
    "MetaLagrangianSearch",
    "contest_score",
]
