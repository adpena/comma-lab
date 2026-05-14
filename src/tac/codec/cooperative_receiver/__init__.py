# SPDX-License-Identifier: MIT
"""Cooperative-receiver primitive package.

Exports two distinct primitives that together capture the alien-tech
score-aware-training stack the time-traveler L5 substrate productionized:

- :class:`AtickRedlichWeights` + :func:`cooperative_receiver_loss`
  (efficient coding 1990/1992): scorer-conditional loss that maximizes
  ``MI(B; S(B))`` against a FIXED known scorer ``(SegNet, PoseNet)``.
- :class:`PredictiveCodingWeights` + :func:`predictive_coding_residual_term`
  (Rao-Ballard 1999): top-down predictive-coding hierarchy that penalizes
  the L2 norm of the per-pair residual the world model failed to predict.

The two primitives are mathematically distinct (cooperative-receiver is a
scorer-matched MI bound; predictive-coding is a top-down prediction-error
penalty) but compose orthogonally — the time-traveler L5 substrate stacks
both, and any new substrate that ships a learned world model + per-pair
side info should consider doing the same.

Per the floor-v3 commit ``27a7950fd`` strategy memo, cooperative-receiver
is alien-tech move #1 in the orthogonal Amdahl composition unlocking the
optimistic 0.10-0.13 floor.

Per Catalog #169 (canonical primitive inventory), both primitives are
registered in :func:`tac.composition.registry.canonical_primitive_inventory`
so the cathedral autopilot ranker + Pareto solver consume them.

Cross-references
----------------
- Source substrate (in-tree consumer):
  :mod:`tac.substrates.time_traveler_l5_autonomy.score_aware_loss`
- Canonical scorer-loss helper (delegated to internally):
  :func:`tac.substrates.score_aware_common.score_pair_components`
- Canonical eval-roundtrip pipeline (delegated to internally):
  :func:`tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`
- Sister cooperative-receiver concept (Wyner-Ziv 1976, sister-subagent
  scope, distinct primitive surface — kept separate per the
  probe-disambiguator pattern):
  :mod:`tac.packet_compiler.wyner_ziv` (when landed)

Lane: ``lane_cooperative_receiver_primitive_20260513``.
"""

from tac.codec.cooperative_receiver.atick_redlich import (
    AtickRedlichWeights,
    CooperativeReceiverOutput,
    cooperative_receiver_loss,
)
from tac.codec.cooperative_receiver.predictive_coding import (
    PredictiveCodingOutput,
    PredictiveCodingWeights,
    predictive_coding_residual_term,
)

__all__ = [
    "AtickRedlichWeights",
    "CooperativeReceiverOutput",
    "PredictiveCodingOutput",
    "PredictiveCodingWeights",
    "cooperative_receiver_loss",
    "predictive_coding_residual_term",
]
