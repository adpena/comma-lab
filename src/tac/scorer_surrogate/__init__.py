# SPDX-License-Identifier: MIT
"""Canonical scorer-surrogate package.

This is the canonical namespace for deployable scorer surrogates — small,
cheap-to-invert proxies of the canonical contest scorers (PoseNet
FastViT-T12 + SegNet EfficientNet-B2) that can be:

* Trained via the canonical Hinton-distilled training infrastructure at
  ``tac.substrates.hinton_distilled_scorer_surrogate`` (KL T=2.0 for
  SegNet + MSE for PoseNet);
* Deployed via numpy-portable inference (no MLX / no PyTorch runtime
  dependency) so per-byte gradient extraction + cathedral-consumer
  routing can consult them without bootstrapping the full MLX stack;
* Verified against the canonical scorers via Catalog #1265
  contest-equivalence drift discipline (max_abs < 3e-5 per Slot 1303 T3
  GRAND COUNCIL).

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable:
surrogates are NEVER promotable — they produce ``[macOS-CPU advisory]``
/ ``[macOS-MLX research-signal]`` artifacts per Catalog #192. Their role
is INFORMATION-PROVIDING (per-byte sensitivity, cost-discrimination,
attention prior) not SCORE-CLAIMING.

The first deployable surrogate landed under this namespace is the
PoseNet MAE-V (Multi-Axis Variant) — see
:mod:`tac.scorer_surrogate.posenet_mae_v` for the canonical PoseNet
surrogate sister of the existing Hinton-distilled SegNet surrogate at
``tac.residual_basis.hinton_distilled_scorer_surrogate``.
"""
from __future__ import annotations
