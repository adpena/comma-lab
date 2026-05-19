# SPDX-License-Identifier: MIT
"""Cathedral consumer for `tac.master_gradient_consumers.gradient_informed_decoder_pruning`.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for the Cable D
gradient-informed decoder-pruning consumer landed 2026-05-19 (commits
418698227 + 27fc83168) per `.omx/research/cable_d_master_gradient_extension_
batch_landed_20260519T055121Z.md`.

The producer identifies DEAD bytes via joint criterion:
- aggregate L1 magnitude < aggregate_floor_relative × max_aggregate_L1
- per-pair L1 variance < per_pair_variance_floor

Joint-criterion is stricter than aggregate-alone: bytes with low aggregate
mean but HIGH per-pair variance are KEPT (they have per-pair leverage even
if they cancel in the mean). Per Cable D HARD-EARNED audit, this IS the
canonical disambiguator for "dead byte vs hidden per-pair leverage" (hook
#6 PROBE_DISAMBIGUATOR per Cable D landing memo). Feeds
`tac.optimization.bit_allocator` per CLAUDE.md hook #3.

THIS wrapper exposes the canonical CathedralConsumerContract so the
auto-discovery loop ingests the surface. Cathedral contribution is
observability-only ([predicted] axis, promotable=False) per Catalog
#287/#323.

Hook numbers per Cable D landing memo:
- Hook #3 BIT_ALLOCATOR (decoder pruning feeds tac.optimization.bit_allocator)
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this wrapper)
- Hook #6 PROBE_DISAMBIGUATOR (dead byte vs hidden per-pair leverage)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "gradient_informed_decoder_pruning_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Decoder-pruning sidecar JSON is canonically persisted via
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
            "tac.master_gradient_consumers.gradient_informed_decoder_pruning "
            "surface available (joint aggregate L1 AND per-pair variance floor; "
            "canonical disambiguator for dead-byte vs hidden-per-pair-leverage; "
            "feeds tac.optimization.bit_allocator) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
