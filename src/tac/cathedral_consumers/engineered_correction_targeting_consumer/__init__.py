# SPDX-License-Identifier: MIT
"""Cathedral consumer for `tac.master_gradient_consumers.engineered_correction_targeting`.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for the Cable D
engineered-correction targeting consumer landed 2026-05-19 (commits
418698227 + 27fc83168) per `.omx/research/cable_d_master_gradient_extension_
batch_landed_20260519T055121Z.md`. The modern canonical sink is
`tac.optimization.byte_shaving_signal_surface_builder`, which subsumes the
legacy sidecar JSON as planning-only `correction_target` units.

For each pair, the producer selects the top-K bytes by per-pair distortion
magnitude × high per-pair variance (HIGH = PAIR_SPECIFIC class per
consumer 1). Feeds `tac.optimization.bit_allocator` per CLAUDE.md hook #3
via per-pair-per-byte engineered-correction sidecar injection.

THIS wrapper exposes the canonical CathedralConsumerContract so the
auto-discovery loop ingests the surface. Cathedral contribution is
observability-only ([predicted] axis, promotable=False) per Catalog
#287/#323.

Hook numbers per Cable D landing memo:
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this wrapper)
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "engineered_correction_targeting_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.BIT_ALLOCATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Engineered-correction-targeting sidecar JSON is canonically persisted
    via `tac.master_gradient_consumers.consumer_output_path` at
    `.omx/state/master_gradient_consumers/`. NO-OP by design.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution."""
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.master_gradient_consumers.engineered_correction_targeting "
            "surface available (top-K bytes by per-pair distortion × variance; "
            "PAIR_SPECIFIC class per consumer 1; feeds tac.optimization.bit_allocator) "
            "and is subsumed by byte_shaving_signal_surface_builder as "
            "planning-only correction_target units) [predicted]"
        ),
        "canonical_sink": (
            "tac.optimization.byte_shaving_signal_surface_builder:"
            "engineered_correction_targeting_paths"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
