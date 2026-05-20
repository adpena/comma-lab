# SPDX-License-Identifier: MIT
"""Cathedral consumer for `tac.master_gradient` per-pair score-impact gradient.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the canonical per-pair master-gradient surface into cathedral autopilot
auto-discovery per the Cable D wire-in batch landing 2026-05-19 (lane
``lane_cable_d_wire_in_batch_d2_d3_20260519``).

The producer (`tac.master_gradient.load_per_pair_gradient_from_anchor` via
`tac.master_gradient_consumers`) emits a (N_bytes, N_pairs, 3) Jacobian
tensor representing the per-archive-byte per-pair score-axis sensitivity
(seg / pose / rate). This consumer exposes that signal to the cathedral
autopilot ranker as a DIAGNOSTIC [predicted] observability annotation.

Sister of:
- `master_gradient_aggregate_consumer` (paired aggregate-gradient surface)
- `per_pair_pareto_envelope_consumer` (consumes per-pair gradient)
- `per_pair_lagrangian_lambda_bisection_consumer` (consumes per-pair gradient)
- `per_pair_kkt_residuals_consumer` (consumes per-pair gradient + lambda)
- `per_pair_volterra_cross_terms_consumer` (consumes per-pair gradient)

Per Catalog #318 raw-byte-authority-guard: this consumer NEVER returns raw
byte-level finite-difference response tensors. The cathedral contribution is
an axis-presence + n_pairs annotation only; downstream consumers (per_pair_*
package family) consume the gradient via the canonical typed loaders.

Per Catalog #327 master-gradient contest-axis custody: the consumer reads
the latest authoritative anchor for the candidate's archive via
`tac.master_gradient.latest_anchor_for_archive` which routes through
`is_authoritative_axis_anchor`. Anchors that fail the contest-axis authority
filter (CPU/CUDA axis mismatch / MPS / macOS-CPU advisory / etc.) are
treated as ABSENT.

Per Catalog #341 cathedral consumer routing markers: every return value
carries `predicted_delta_adjustment=0.0` + `promotable=False` +
`axis_tag="[predicted]"` so the routing recommendation cannot leak into a
score signal.

Hook numbers per Catalog #125 6-hook wire-in:
- Hook #1 SENSITIVITY_MAP (per-pair Jacobian feeds sensitivity_map.axis_weights)
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this consumer)
- Hook #5 CONTINUAL_LEARNING_POSTERIOR (anchor updates trigger recomputation)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "master_gradient_per_pair_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per-pair master-gradient anchors are persisted by the canonical helper
    `tac.master_gradient.append_anchor_locked` writing to
    `.omx/state/master_gradient_anchors.jsonl` (fcntl-locked per Catalog
    #131/#138/#245). The auto-discovery loop calls this hook when a NEW
    anchor lands so consumers can invalidate cached state; this consumer
    is STATELESS (reads anchor fresh per candidate) so the hook is a no-op
    by design.

    Per CLAUDE.md "Apples-to-apples evidence discipline": this hook does
    NOT promote diagnostic anchors to contest-grade. The
    `is_authoritative_axis_anchor` filter is applied at read time per
    Catalog #327.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Looks up the latest per-pair master-gradient anchor for the candidate's
    archive via `tac.master_gradient_consumers.load_per_pair_gradient_from_anchor`
    and returns a [predicted] observability annotation citing the anchor's
    presence + shape + measurement axis. No score adjustment per Catalog
    #341 canonical non-promotable markers.

    Per Catalog #318 raw-byte-authority-guard: never returns raw bytes or
    byte modifications. The annotation surfaces presence + shape only;
    downstream sister consumers (per_pair_pareto_envelope /
    per_pair_lagrangian_lambda_bisection / per_pair_kkt_residuals etc.)
    consume the gradient via the canonical typed loaders.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": the
    `axis_tag` is ALWAYS `[predicted]` — the per-pair gradient is a planning
    signal, not a contest-axis score claim, even when derived from a
    contest-CUDA or contest-CPU anchor (those anchors authorize the
    GRADIENT measurement; they do not authorize the cathedral routing
    recommendation built from it).
    """
    archive_sha256 = (
        candidate.get("archive_sha256") if isinstance(candidate, Mapping) else None
    )

    if not isinstance(archive_sha256, str) or len(archive_sha256) < 8:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "tac.master_gradient per-pair gradient surface available; "
                "candidate carries no archive_sha256 so no anchor lookup "
                "attempted [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    # Per Catalog #327 the loader routes through is_authoritative_axis_anchor;
    # any anchor failing the contest-axis authority filter is treated as
    # ABSENT. Per Catalog #318 the result NEVER includes raw byte tensors;
    # the cathedral contribution is shape + axis presence only.
    try:
        from tac.master_gradient_consumers import (
            load_per_pair_gradient_from_anchor,
        )
    except (ImportError, ModuleNotFoundError):
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "tac.master_gradient_consumers import unavailable; "
                "per-pair gradient surface absent [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    try:
        array, anchor = load_per_pair_gradient_from_anchor(
            archive_sha256=archive_sha256
        )
    except (FileNotFoundError, ValueError, OSError):
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                f"no authoritative per-pair master-gradient anchor for "
                f"archive {archive_sha256[:12]} [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    n_bytes, n_pairs, n_axes = array.shape
    measurement_axis = anchor.get("measurement_axis", "unknown")
    measurement_hardware = anchor.get("measurement_hardware", "unknown")

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            f"tac.master_gradient per-pair anchor available for "
            f"archive {archive_sha256[:12]}: (N_bytes={n_bytes}, "
            f"N_pairs={n_pairs}, axes={n_axes}); "
            f"measurement_axis={measurement_axis}, "
            f"measurement_hardware={measurement_hardware}; downstream "
            f"sister consumers (per_pair_pareto_envelope / "
            f"per_pair_lagrangian_lambda_bisection / per_pair_kkt_residuals) "
            f"consume the gradient via canonical loaders [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
