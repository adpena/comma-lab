# SPDX-License-Identifier: MIT
"""Impl 8 -- canonical alias for the contest theoretical-floor estimate.

Thin wrapper over ``tac.symposium_impls.blahut_arimoto_theoretical_floor``
exposing the contest_oracle's namespaced surface so callers can:

    from tac.contest_oracle.theoretical_floor import compute_contest_floor

instead of reaching into ``symposium_impls`` directly. The alias preserves
the canonical signature; consumers get the canonical Tao+Boyd Blahut-Arimoto
floor estimate (rate-distortion lower bound on the contest score).

Per the design memo: S_floor ~= 25*0.05 + 0 + 0 ~= 0.05 -- current 0.193
is ~3.9x the analytical floor, so substantial frontier-EV remains.

Citations:
  - Blahut 1972 *Computation of channel capacity and rate-distortion functions*
    (IEEE IT) -- Blahut algorithm.
  - Arimoto 1972 *An algorithm for computing the capacity of arbitrary
    discrete memoryless channels* (IEEE IT).
  - Cover & Thomas 2006 *Elements of Information Theory* Ch.10 -- R(D).
  - ``tac.symposium_impls.blahut_arimoto_theoretical_floor.compute_contest_theoretical_floor``
    -- canonical sister this module re-exports.

Catalog #125 hook 2 (pareto_constraint): ACTIVE -- the theoretical floor IS
the lower-bound Pareto constraint.
Catalog #125 hook 5 (continual_learning_posterior): ACTIVE -- the canonical
Blahut-Arimoto state is persisted via the sister symposium_impls module.
Catalog #305 observability surface: cite_able.
"""
from __future__ import annotations

from typing import Any, Final

try:
    from tac.symposium_impls.blahut_arimoto_theoretical_floor import (
        BLAHUT_ARIMOTO_FLOOR_STATE_PATH,
        ContestTheoreticalFloor,
        compute_contest_theoretical_floor as _canonical_compute_floor,
    )

    CANONICAL_BLAHUT_AVAILABLE: Final[bool] = True
except ImportError:  # pragma: no cover -- shim for partial-checkout safety
    BLAHUT_ARIMOTO_FLOOR_STATE_PATH = None
    ContestTheoreticalFloor = None  # type: ignore
    _canonical_compute_floor = None
    CANONICAL_BLAHUT_AVAILABLE = False


def compute_contest_floor(
    *,
    target_d_seg: float,
    target_d_pose: float,
    operating_point_anchor: str = "A1 [contest-CPU GHA Linux x86_64] 0.1928",
    num_units_seg: int = 117_964_800,
    num_units_pose: int = 3_600,
) -> Any:
    """Canonical alias for the contest theoretical-floor estimate.

    Composes ``tac.symposium_impls.blahut_arimoto_theoretical_floor.compute_contest_theoretical_floor``.
    See the sister docstring for full parameter semantics.

    Args:
        target_d_seg: Target seg distortion at the floor (e.g. 0.0005).
        target_d_pose: Target pose distortion at the floor (e.g. 1e-6).
        operating_point_anchor: Canonical evidence-grade anchor string.
        num_units_seg: Seg units (default = per-archive pixel cells).
        num_units_pose: Pose units (default = 600 pairs * 6 dims).

    Returns:
        ``ContestTheoreticalFloor`` dataclass from the sister module.

    Raises:
        ImportError: if the sister Blahut-Arimoto module is unavailable.
    """
    if not CANONICAL_BLAHUT_AVAILABLE:
        raise ImportError(
            "tac.symposium_impls.blahut_arimoto_theoretical_floor not available; "
            "install symposium_impls package"
        )
    return _canonical_compute_floor(
        target_d_seg=target_d_seg,
        target_d_pose=target_d_pose,
        operating_point_anchor=operating_point_anchor,
        num_units_seg=num_units_seg,
        num_units_pose=num_units_pose,
    )


__all__ = [
    "BLAHUT_ARIMOTO_FLOOR_STATE_PATH",
    "CANONICAL_BLAHUT_AVAILABLE",
    "ContestTheoreticalFloor",
    "compute_contest_floor",
]
