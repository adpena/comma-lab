# SPDX-License-Identifier: MIT
"""Cathedral consumer for `tac.master_gradient_consumers.per_pair_lora_supervision_signal`.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for the Cable D
per-pair LoRA supervision consumer landed 2026-05-19 (commits 418698227 +
27fc83168) per `.omx/research/cable_d_master_gradient_extension_batch_landed_
20260519T055121Z.md`.

The producer ranks bytes by (per_pair_distortion_mean × per_pair_distortion_std)
— HIGH mean = byte matters for distortion; HIGH std = matters DIFFERENTLY
per pair (LoRA's per-pair adapter is the canonical absorber). Feeds
`tac.optimization.bit_allocator` per CLAUDE.md "Subagent coherence-by-default"
hook #3 via per-pair LoRA target injection.

THIS wrapper exposes the canonical CathedralConsumerContract so the
auto-discovery loop ingests the surface. Cathedral contribution is
observability-only ([predicted] axis, promotable=False) per Catalog
#287/#323.

Hook numbers per Cable D landing memo:
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this wrapper)
- Hook #5 CONTINUAL_LEARNING_POSTERIOR (sidecar persists to canonical store)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "per_pair_lora_supervision_signal_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per-pair LoRA supervision sidecar JSON is canonically persisted via
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
            "tac.master_gradient_consumers.per_pair_lora_supervision_signal "
            "surface available (per-pair LoRA targets: HIGH mean × HIGH std bytes; "
            "feeds tac.optimization.bit_allocator) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
