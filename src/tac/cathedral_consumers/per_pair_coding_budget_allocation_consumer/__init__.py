# SPDX-License-Identifier: MIT
"""Cathedral consumer for `tac.master_gradient_consumers.per_pair_coding_budget_allocation`.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for the Cable D
per-pair coding-budget allocation consumer landed 2026-05-19 (commits
418698227 + 27fc83168) per `.omx/research/cable_d_master_gradient_extension_
batch_landed_20260519T055121Z.md`.

The producer allocates per-pair latent-byte budget from per-pair pose
gradient L1 norm: (1) total_budget = N_pairs × baseline_bytes_per_pair;
(2) each pair's relative_share = pose_norm / sum_pose_norms;
(3) allocated_bytes = round(total_budget × relative_share). Feeds
`tac.optimization.bit_allocator` per CLAUDE.md hook #3.

THIS wrapper exposes the canonical CathedralConsumerContract so the
auto-discovery loop ingests the surface. Cathedral contribution is
observability-only ([predicted] axis, promotable=False) per Catalog
#287/#323.

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":
at the PR106 frontier (pose_avg ~3.4e-5) pose marginal sensitivity is
2.71× SegNet's, so pose-gradient-L1-weighted allocation IS operating-point
optimal for the current frontier.

Hook numbers per Cable D landing memo:
- Hook #3 BIT_ALLOCATOR (consumed by tac.optimization.bit_allocator)
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this wrapper)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "per_pair_coding_budget_allocation_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per-pair coding-budget sidecar JSON is canonically persisted via
    `tac.master_gradient_consumers.consumer_output_path` at
    `.omx/state/master_gradient_consumers/`. NO-OP by design.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution."""
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.master_gradient_consumers.per_pair_coding_budget_allocation "
            "surface available (per-pair latent-byte budget from pose gradient L1; "
            "feeds tac.optimization.bit_allocator; operating-point-aware per CLAUDE.md "
            "SegNet vs PoseNet importance) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
