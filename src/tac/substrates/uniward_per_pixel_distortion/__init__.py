# SPDX-License-Identifier: MIT
"""UNIWARD per-pixel score-conditional sensitivity weighting substrate.

Fridrich-canonical pure-distortion attack (3-strategy framework PRIMARY = DISTORTION;
sub-axis = JOINT d_seg + d_pose) adapted for contest scorers per Yousfi grand-council
position. Per-pixel weight map `w[h,w] = 1 / (eps + (d_seg_grad[h,w])^2 +
(d_pose_grad[h,w])^2)` routes training-time perturbation budget toward
LOW-sensitivity zones where scorer-response is uninformative.

Entropy-position P2 loss-shape (TRAIN phase BEFORE entropy coder); MLX-first
training; numpy-portable inflate (compress-only weighting; weight map NOT
shipped per Carmack-preferred budget conservation); individually-fractal
substrate per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

Canonical equation target (proposed):
`uniward_per_pixel_score_conditional_sensitivity_weighting_distortion_savings_v1`.

References:
- Holub, Fridrich, Denemark 2014 "Universal Distortion Function for Steganography
  in an Arbitrary Domain" (UNIWARD canonical)
- Yousfi grand-council position on inverse-steganalysis applied to contest scorers
- CLAUDE.md "Fridrich inverse steganalysis" section
- 4 standing directives (2026-05-26): 3-strategy / MLX-first-numpy-portable /
  entropy-position / MLX↔CUDA bidirectional drift
"""

from __future__ import annotations

__all__ = [
    "SUBSTRATE_ID",
    "SUBSTRATE_VERSION",
    "compute_per_pixel_uniward_weight_map_numpy",
    "compose_uniward_weighted_score_loss",
]

SUBSTRATE_ID = "uniward_per_pixel_distortion"
SUBSTRATE_VERSION = "v1_2026-05-26"

# Canonical Provenance per Catalog #323 + Catalog #335 cathedral consumer contract
CONSUMER_NAME = "uniward_per_pixel_distortion_substrate"
CONSUMER_VERSION = SUBSTRATE_VERSION
CONSUMER_HOOK_NUMBERS = (1, 4, 5)  # sensitivity-map / autopilot dispatch / continual learning

# Lazy imports to avoid circular dependencies
from tac.substrates.uniward_per_pixel_distortion.weight_map import (  # noqa: E402
    compute_per_pixel_uniward_weight_map_numpy,
)
from tac.substrates.uniward_per_pixel_distortion.score_aware_loss import (  # noqa: E402
    compose_uniward_weighted_score_loss,
)
