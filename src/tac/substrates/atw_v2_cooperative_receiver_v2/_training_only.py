# SPDX-License-Identifier: MIT
"""ATW V2 cooperative-receiver V2 — training-only PyTorch loss (NON-PORTABLE).

Per Phase 3 design memo §9 (portability via numpy per primitive) + operator
binding directive #3 documented exception.

This module is explicitly NON-PORTABLE. It is the ISOLATED training loss
compute (PyTorch gradient path through SegNet+PoseNet for the Atick-Redlich
cooperative-receiver loss). Per Axis 3 portability discipline:

- FORWARD path + INFLATE path remain PORTABLE (numpy_reference.py +
  inflate.py + archive.py + mlx_renderer.py forward)
- TRAINING loss compute (THIS module) is PyTorch-only by design

The loss routing per Phase 3 §1:

    L = -I(X; R(decoder(z))) + alpha * rate(z|Y_ego_motion)
                             + beta * d_seg + gamma * sqrt(d_pose)

Where:
- ``-I(X; R(decoder(z)))`` = Atick-Redlich cooperative-receiver loss
  (routes through canonical ``tac.codec.cooperative_receiver.atick_redlich.
  cooperative_receiver_loss`` per Catalog #164)
- ``rate(z|Y_ego_motion)`` = conditional source coding rate term per
  Wyner-Cover R(D|Y) for known shared conditioning Y
- ``d_seg`` + ``d_pose`` = canonical contest distortion terms via
  canonical ``score_pair_components`` per Catalog #164

NOT YET WIRED at L0 SCAFFOLD
============================

Per Catalog #240(c): ``_full_main raises NotImplementedError`` until Phase 2+
council approval. This module provides the SKELETON for the training loss
but does NOT yet run a forward+backward pass (Phase 4 deliverable).

Cross-references
----------------

* Phase 3 design memo §1 + §7 + §9
* ``src/tac/codec/cooperative_receiver/atick_redlich.py`` (canonical primitive)
* ``src/tac/substrates/_shared/score_aware_common.py`` (canonical scorer routing)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ATWv2CR2LossWeights:
    """Loss term weights per Phase 3 design memo §1.

    Phase 3 L0 SCAFFOLD defaults; Phase 4 paired smoke determines substrate-
    optimal weights per Dykstra-feasibility framework.
    """

    alpha_rate: float = 25.0
    """Canonical contest rate-term coefficient per CLAUDE.md formula."""

    beta_seg: float = 100.0
    """Canonical contest seg-term coefficient per CLAUDE.md formula."""

    gamma_pose: float = 1.0
    """Canonical contest pose-term coefficient (multiplier on sqrt(10 * d_pose))."""

    cooperative_receiver_lambda: float = 0.1
    """Atick-Redlich cooperative-receiver MI term weight. Phase 4 paired
    smoke determines substrate-optimal value."""


def compute_atwv2cr2_loss_skeleton(*args, **kwargs):
    """Placeholder for the canonical cooperative-receiver loss compute.

    Per Catalog #240(c): raises NotImplementedError. Phase 4 lands the full
    PyTorch training forward+backward routing through:

    1. canonical ``score_pair_components`` per Catalog #164 (scorer routing)
    2. canonical ``cooperative_receiver_loss`` per Catalog #164 (Atick-Redlich)
    3. conditional source coding rate term R(D|Y_ego_motion)
    """
    raise NotImplementedError(
        "Phase 2+ council approval required to lift _full_main NotImplementedError "
        "per Catalog #240(c) + CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or "
        "RESEARCH-ONLY'. L0 SCAFFOLD provides MLX-native forward + portable numpy "
        "reference + PyTorch parity reference + archive grammar + inflate runtime; "
        "the training loss compute (this function) is the Phase 4 deliverable."
    )


__all__ = [
    "ATWv2CR2LossWeights",
    "compute_atwv2cr2_loss_skeleton",
]
