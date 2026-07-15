# SPDX-License-Identifier: MIT
"""Executable LawRefs for the C0 event-conditional wall-clock telemetry.

The measured source is content-addressed by a small JSON receipt because the
original audit is Markdown and therefore cannot be resolved by ``InputRef``.
The evaluators perform only unit conversion and the preregistered midpoint;
they do not manufacture efficacy or score authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.canonical_equations.evaluators import register_evaluator
from tac.witness_dsl.lawref import (
    LADDER_DERIVED_AT_CONFIG,
    LADDER_MEASURED_ANCHOR,
    InputRef,
    LawRef,
)

ANCHOR_PATH = ".omx/research/l7_default_failloud_budget_eventlaw_anchor_20260715.json"
ANCHOR_SHA256 = "cc7edc3dc8b8b613242470cf453536f1abc875dd183d6eae80207cdd62f140a3"
CONFIG_TAGS = {"vehicle": "v9_cgauge_c0", "event": "lane_band_via_lane_nucleus"}

TRANSITION_EQUATION_ID = "wall_clock_event_transition_epoch_v1"
SECONDS_TO_MINUTES_EQUATION_ID = "wall_clock_seconds_to_minutes_v1"
MIDPOINT_TO_MINUTES_EQUATION_ID = "wall_clock_midpoint_seconds_to_minutes_v1"


def _transition_epoch(inputs: Mapping[str, Any]) -> int:
    epoch = int(inputs["epoch"])
    if epoch != inputs["epoch"] or epoch < 1:
        raise ValueError(f"event transition epoch must be a positive integer, got {inputs['epoch']!r}")
    return epoch


def _seconds_to_minutes(inputs: Mapping[str, Any]) -> float:
    seconds = float(inputs["seconds"])
    if not 0.0 < seconds < float("inf"):
        raise ValueError(f"seconds must be finite and positive, got {seconds!r}")
    return seconds / 60.0


def _midpoint_seconds_to_minutes(inputs: Mapping[str, Any]) -> float:
    lower = float(inputs["lower_seconds"])
    upper = float(inputs["upper_seconds"])
    if not (0.0 < lower <= upper < float("inf")):
        raise ValueError(f"seconds range must be finite, positive, and ordered, got {lower}/{upper}")
    return ((lower + upper) / 2.0) / 60.0


register_evaluator(TRANSITION_EQUATION_ID, _transition_epoch)
register_evaluator(SECONDS_TO_MINUTES_EQUATION_ID, _seconds_to_minutes)
register_evaluator(MIDPOINT_TO_MINUTES_EQUATION_ID, _midpoint_seconds_to_minutes)


def _anchor(extract: str, provenance: str) -> InputRef:
    return InputRef.anchor(
        ANCHOR_PATH,
        extract,
        provenance,
        expected_sha256=ANCHOR_SHA256,
        config_tags=CONFIG_TAGS,
    )


def canonical_event_wall_clock_lawrefs() -> dict[str, LawRef]:
    """Return the three executable constants defining the C0 stage profile."""

    return {
        "transition_epoch": LawRef(
            equation_id=TRANSITION_EQUATION_ID,
            inputs={
                "epoch": _anchor(
                    "telemetry/transition_epoch",
                    "MEASURED C0 lane-band event fired at epoch 33",
                )
            },
            ladder_class=LADDER_MEASURED_ANCHOR,
        ),
        "pre_event_min_per_ep": LawRef(
            equation_id=SECONDS_TO_MINUTES_EQUATION_ID,
            inputs={
                "seconds": _anchor(
                    "telemetry/pre_event/median_seconds_per_epoch",
                    "MEASURED C0 epochs 1..32 median seconds per epoch",
                )
            },
            ladder_class=LADDER_MEASURED_ANCHOR,
        ),
        "post_event_min_per_ep": LawRef(
            equation_id=MIDPOINT_TO_MINUTES_EQUATION_ID,
            inputs={
                "lower_seconds": _anchor(
                    "telemetry/post_event/observed_seconds_per_epoch_lower",
                    "MEASURED C0 post-event lower observed seconds per epoch",
                ),
                "upper_seconds": _anchor(
                    "telemetry/post_event/observed_seconds_per_epoch_upper",
                    "MEASURED C0 post-event upper observed seconds per epoch",
                ),
            },
            ladder_class=LADDER_DERIVED_AT_CONFIG,
        ),
    }


__all__ = [
    "ANCHOR_PATH",
    "ANCHOR_SHA256",
    "CONFIG_TAGS",
    "canonical_event_wall_clock_lawrefs",
]
