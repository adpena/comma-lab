# SPDX-License-Identifier: MIT
"""Cathedral consumer for `tac.master_gradient` aggregate score-impact gradient.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the canonical aggregate master-gradient surface into cathedral autopilot
auto-discovery per the Cable D wire-in batch landing 2026-05-19 (lane
``lane_cable_d_wire_in_batch_d2_d3_20260519``).

The producer (`tac.master_gradient_consumers.load_aggregate_gradient_from_anchor`)
emits a (N_bytes, 3) Jacobian tensor representing the per-archive-byte
aggregate score-axis sensitivity (seg / pose / rate) averaged across pairs.
This consumer exposes that signal to the cathedral autopilot ranker as a
DIAGNOSTIC [predicted] observability annotation.

Sister of:
- `master_gradient_per_pair_consumer` (paired per-pair-gradient surface)
- `per_byte_sensitivity_consumer` (consumes aggregate gradient via DuckDB)
- `engineered_correction_targeting_consumer` (consumes aggregate gradient)
- `gradient_informed_decoder_pruning_consumer` (consumes aggregate gradient)

Per Catalog #318 raw-byte-authority-guard: this consumer NEVER returns raw
byte-level finite-difference response tensors. The cathedral contribution is
an axis-presence + shape annotation only; downstream consumers consume the
gradient via the canonical typed loaders.

Per Catalog #327 master-gradient contest-axis custody: the consumer reads
the latest authoritative anchor for the candidate's archive via
`load_aggregate_gradient_from_anchor` which internally filters via
`is_authoritative_axis_anchor`. Anchors that fail the contest-axis
authority filter are treated as ABSENT.

Per Catalog #341 cathedral consumer routing markers: every return value
carries `predicted_delta_adjustment=0.0` + `promotable=False` +
`axis_tag="[predicted]"` so the routing recommendation cannot leak into a
score signal.

Hook numbers per Catalog #125 6-hook wire-in:
- Hook #1 SENSITIVITY_MAP (aggregate Jacobian feeds sensitivity_map.axis_weights)
- Hook #3 BIT_ALLOCATOR (aggregate per-byte importance ranks bit budget)
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this consumer)
- Hook #5 CONTINUAL_LEARNING_POSTERIOR (anchor updates trigger recomputation)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "master_gradient_aggregate_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Aggregate master-gradient anchors are persisted via
    `tac.master_gradient.append_anchor_locked` to the canonical
    `.omx/state/master_gradient_anchors.jsonl` ledger (fcntl-locked per
    Catalog #131/#138/#245). This consumer is STATELESS (reads anchor
    fresh per candidate) so the hook is a no-op by design.

    Per Catalog #327 contest-axis custody: this hook does NOT promote
    diagnostic anchors. The authority filter is applied at read time via
    `load_aggregate_gradient_from_anchor`.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Looks up the latest aggregate master-gradient anchor for the candidate's
    archive and returns a [predicted] observability annotation citing the
    anchor's presence + shape + measurement axis. No score adjustment per
    Catalog #341 canonical non-promotable markers.

    Per Catalog #318 raw-byte-authority-guard: never returns raw bytes or
    byte modifications. The annotation surfaces presence + shape only;
    downstream sister consumers (per_byte_sensitivity /
    engineered_correction_targeting / gradient_informed_decoder_pruning)
    consume the gradient via the canonical typed loaders.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": the
    `axis_tag` is ALWAYS `[predicted]` — the aggregate gradient is a
    planning signal, not a contest-axis score claim, even when derived
    from a contest-CUDA or contest-CPU anchor.
    """
    archive_sha256 = (
        candidate.get("archive_sha256") if isinstance(candidate, Mapping) else None
    )

    if not isinstance(archive_sha256, str) or len(archive_sha256) < 8:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "tac.master_gradient aggregate gradient surface available; "
                "candidate carries no archive_sha256 so no anchor lookup "
                "attempted [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    try:
        from tac.master_gradient_consumers import (
            load_aggregate_gradient_from_anchor,
        )
    except (ImportError, ModuleNotFoundError):
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "tac.master_gradient_consumers import unavailable; "
                "aggregate gradient surface absent [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    try:
        array, anchor = load_aggregate_gradient_from_anchor(
            archive_sha256=archive_sha256
        )
    except (FileNotFoundError, ValueError, OSError):
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                f"no authoritative aggregate master-gradient anchor for "
                f"archive {archive_sha256[:12]} [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    n_bytes, n_axes = array.shape
    measurement_axis = anchor.get("measurement_axis", "unknown")
    measurement_hardware = anchor.get("measurement_hardware", "unknown")

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            f"tac.master_gradient aggregate anchor available for "
            f"archive {archive_sha256[:12]}: (N_bytes={n_bytes}, "
            f"axes={n_axes}); measurement_axis={measurement_axis}, "
            f"measurement_hardware={measurement_hardware}; downstream "
            f"sister consumers (per_byte_sensitivity / "
            f"engineered_correction_targeting / "
            f"gradient_informed_decoder_pruning) consume the gradient "
            f"via canonical loaders [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
