# SPDX-License-Identifier: MIT
"""Score-axis-actionable utility callables for the unified-Lagrangian action.

Per the synthesis memo amendment
`.omx/research/magic_codec_plus_water_filling_plus_lagrangian_redirection_unified_synthesis_cross_pollination_20260518.md`
§"OTHER APPLICATIONS" top-3, this subpackage exposes 3 canonical utility
callables that match the signature accepted by
:func:`tac.unified_action.make_action_from_track_callables` and can therefore
be wired as Action term-evaluators that flow through any of the 4 solver
routers (``evaluate_with_water_filling`` / ``evaluate_with_admm`` /
``evaluate_with_magic_codec`` / ``choose_solver``).

The 3 utilities are intentionally orthogonal — each indexes a DIFFERENT
score-axis surface (per-tensor rate-distortion / per-pixel inverse local
variance / per-byte master-gradient sensitivity) so the cross-domain
demonstration test can exercise a SINGLE ``Action`` factory across all 27
applications enumerated in the synthesis memo.

Each utility module exports ONE callable matching the contract:

    Callable[[torch.Tensor, DualVariables | None], torch.Tensor]

Returning a SCALAR tensor (with autograd grad_fn when input requires_grad)
that the Action sums into ``S_total``.

Catalog #125 hook 4 (cathedral autopilot dispatch): ACTIVE — every utility
in this subpackage is consumable by the autopilot ranker as a candidate
score-axis contribution.

CLAUDE.md compliance
--------------------

- Per "Beauty, simplicity, and developer experience" non-negotiable: the
  public surface is narrow (3 callables + ``__all__``).
- Per "Forbidden score claims": no utility ever claims a score. They are
  derivative-friendly proxies the unified action consumes. Authoritative
  scores still come from ``upstream/evaluate.py`` on exact archive bytes.
- Per "Forbidden device-selection defaults": no MPS fallback. Callers pass
  ``device`` via the input tensor; defaults remain CPU for the smoke path.
"""
from __future__ import annotations

from .per_byte_master_gradient import per_byte_master_gradient_utility
from .per_pixel_inverse_variance import per_pixel_inverse_variance_utility
from .per_tensor_rate_distortion import per_tensor_rate_distortion_utility

__all__ = [
    "per_byte_master_gradient_utility",
    "per_pixel_inverse_variance_utility",
    "per_tensor_rate_distortion_utility",
]
