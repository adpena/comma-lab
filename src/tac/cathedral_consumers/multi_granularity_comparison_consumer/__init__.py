# SPDX-License-Identifier: MIT
"""Cathedral consumer for the canonical multi-granularity comparison surface.

Per Catalog #335 + ``tac.cathedral.consumer_contract.CathedralConsumerContract``.
Wires the canonical multi-granularity master-gradient comparison surface
(:mod:`tac.master_gradient_comparison.multi_granularity`) into cathedral
autopilot auto-discovery per the SLOT MG-3 landing 2026-05-19 (lane
``lane_slot_mg_3_multi_granularity_master_gradient_comparison_20260519``).

The producer module exposes (1) per-pixel scorer-axis gradient
(``M_contest`` and ``M_inflated``), (2) per-byte sensitivity derived via
chain rule (``M_archive`` per Catalog #318 fail-closed), (3) per-pair
difficulty atlas, (4) SegNet-class decomposition (NSCS06 anchor), and (5)
information-theoretic floor estimator. This consumer surfaces presence +
shape annotations to the cathedral autopilot ranker as DIAGNOSTIC
``[predicted]`` observability rows.

Sister of:

* ``master_gradient_aggregate_consumer`` (canonical aggregate surface)
* ``master_gradient_per_pair_consumer`` (canonical per-pair surface)
* ``per_byte_sensitivity_consumer`` (DuckDB-mediated per-byte sensitivity)
* ``per_pair_difficulty_atlas_consumer`` (canonical per-pair difficulty
  atlas - this consumer's
  :func:`tac.master_gradient_comparison.compute_per_pair_difficulty_atlas`
  output feeds it)
* ``gradient_informed_decoder_pruning_consumer``
* ``engineered_correction_targeting_consumer``

Per Catalog #318 raw-byte-authority-guard: this consumer NEVER returns raw
byte-level finite-difference response tensors. The cathedral contribution
is an axis-presence + shape + chain-rule-derivation annotation only;
downstream consumers consume the gradient via the canonical typed loaders.

Per Catalog #341 cathedral consumer routing markers: every return value
carries ``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
``axis_tag="[predicted]"`` so the routing recommendation cannot leak into
a score signal.

Hook numbers per Catalog #125 6-hook wire-in:

* Hook #1 SENSITIVITY_MAP (per-axis Jacobian feeds sensitivity_map)
* Hook #2 PARETO_CONSTRAINT (per-pair difficulty atlas informs Pareto)
* Hook #3 BIT_ALLOCATOR (top-K / bottom-K byte ranking input)
* Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this consumer)
* Hook #5 CONTINUAL_LEARNING_POSTERIOR (anchor updates trigger recomputation)
* Hook #6 PROBE_DISAMBIGUATOR (M_inflated vs M_contest substrate-fit diagnostic)
"""

from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "multi_granularity_comparison_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    The multi-granularity comparison surface persists comparison artifacts
    via :func:`tac.master_gradient_comparison.persist_comparison_artifact`
    to canonical sidecar JSON files under
    ``.omx/state/master_gradient_comparison/``. This consumer is STATELESS
    by design; per-archive recomputation routes through the canonical
    extractors when an anchor lands.

    Per Catalog #327 contest-axis custody: this hook does NOT promote
    diagnostic anchors. The comparison surface is always ``[predicted]``
    grade.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Surfaces presence + shape + chain-rule-derivation annotation for the
    multi-granularity comparison surface. No score adjustment per Catalog
    #341 canonical non-promotable markers.

    Per Catalog #318 raw-byte-authority-guard: never returns raw bytes or
    byte modifications. The annotation surfaces the chain-rule canonical
    helper invocation; downstream consumers consume the gradient via the
    canonical typed loaders.
    """
    archive_sha256 = (
        candidate.get("archive_sha256") if isinstance(candidate, Mapping) else None
    )

    if not isinstance(archive_sha256, str) or len(archive_sha256) < 8:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "tac.master_gradient_comparison multi-granularity surface "
                "available; candidate carries no archive_sha256 so no anchor "
                "lookup attempted [predicted]"
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
                "multi-granularity comparison surface absent [predicted]"
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
                f"archive {archive_sha256[:12]}; multi-granularity comparison "
                "surface depends on M_contest + M_inflated which are derived "
                "from the aggregate / per-pair anchors [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    n_bytes = array.shape[0]
    measurement_axis = anchor.get("measurement_axis", "unknown")
    measurement_hardware = anchor.get("measurement_hardware", "unknown")

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            f"tac.master_gradient_comparison multi-granularity surface "
            f"available for archive {archive_sha256[:12]}: "
            f"(N_bytes={n_bytes}, axes=3); measurement_axis={measurement_axis}, "
            f"measurement_hardware={measurement_hardware}; downstream M_contest "
            f"+ M_inflated + M_archive (chain-rule) extractors available via "
            "tac.master_gradient_comparison.extract_* helpers per Catalog #318 "
            "fail-closed [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
