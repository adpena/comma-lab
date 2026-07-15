# SPDX-License-Identifier: MIT
"""Canonical definitions for D24a SegNet nonlocality measurement."""

from __future__ import annotations

from typing import Any

LAW_ID = "segnet_margin_gradient_tail_block_jacobian_v1"


def segnet_margin_gradient_tail_block_jacobian() -> dict[str, Any]:
    """Return the definitions and the deliberately unfilled verdict boundary."""

    return {
        "law_id": LAW_ID,
        "tail_energy_fraction": {
            "equation": "T_r(q)=sum_{||u-q||_2>r} ||grad_x d_q(u)||_2^2 / sum_u ||grad_x d_q(u)||_2^2",
            "radii_px": [64, 128, 192],
            "label": "DERIVED_DEFINITION",
        },
        "block_jacobian": {
            "equation": "B_ab=||partial d(edge_a)/partial x(region_b)||_F^2",
            "relations": ["same_edge", "adjacent_edge", "remote_edge"],
            "label": "DERIVED_DEFINITION",
        },
        "measurement_contract": {
            "pairs": 600,
            "queries_per_pair": ["minimum_margin", "high_margin_control"],
            "bind": ["frozen_scorer_sha256", "source_sha256", "cache_sha256"],
            "scorer_batch_size": 32,
        },
        "negative_boundary": {
            "verdict": "NO_VERDICT_THRESHOLD_NOT_PREREGISTERED",
            "verdict_scope": "INSTANCE x N600 x FROZEN_SEGNET x LOCAL_JACOBIAN_FORMULATION",
            "reason": "A measured nonzero off-diagonal response falsifies exact independence, but no locality-admission threshold is derived here.",
        },
    }


__all__ = ["LAW_ID", "segnet_margin_gradient_tail_block_jacobian"]
