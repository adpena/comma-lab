# SPDX-License-Identifier: MIT
"""Canonical DDM joint-recursion costate and scheduling laws.

This module is intentionally pure.  It does not discover artifacts, mutate a
run, or dispatch work.  The live advisory organ binds measured DDM receipts to
these equations in :mod:`tac.ddm_costate_organ`.
"""

from tac.ddm_costate_law import (  # canonical implementation; dependency-light for SessionStart
    EQUATION_ID,
    RATE_BREAK_EVEN_SCORE_PER_BYTE,
    SCHEDULER_EQUATION_ID,
    ddm_joint_costate,
    gauss_southwell_validity_score,
    realized_pair_distortion_delta,
)

__all__ = [
    "EQUATION_ID",
    "RATE_BREAK_EVEN_SCORE_PER_BYTE",
    "SCHEDULER_EQUATION_ID",
    "ddm_joint_costate",
    "gauss_southwell_validity_score",
    "realized_pair_distortion_delta",
]
