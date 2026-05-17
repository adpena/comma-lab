# SPDX-License-Identifier: MIT
"""Canonical example boost stages.

Three stages demonstrate the namespace's composition + decorator pattern:

  - ``raw_decoder``: seed stage (passthrough); emits frames_v0
  - ``cascade_pose_residual_v1``: depth-1 additive pose residual cascade
  - ``cascade_seg_residual_v1``: depth-1 additive seg residual cascade
    (consumes the prior cascade's frames_v1; emits frames_v2)

These are EXAMPLES — they implement the contract surface but their
correction values are toy/zeros for testing. Real substrate consumers
provide real per-pair correction tensors via the same pattern.
"""

from tac.boosting.examples.example_stages import (
    cascade_pose_residual_v1,
    cascade_seg_residual_v1,
    raw_decoder,
)

__all__ = [
    "cascade_pose_residual_v1",
    "cascade_seg_residual_v1",
    "raw_decoder",
]
