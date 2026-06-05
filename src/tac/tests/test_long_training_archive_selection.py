# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.training.long_training_canonical import (
    CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE,
    _archive_selection_health_sort_key,
)


def _row(occupied_fraction: float) -> dict[str, object]:
    return {
        "score_components": {
            "selection_health_segnet_direct_live_candidate_occupied_class_fraction": (
                occupied_fraction
            )
        }
    }


def test_archive_selection_treats_two_of_five_segnet_classes_as_collapsed() -> None:
    collapsed = _archive_selection_health_sort_key(
        _row(CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE - 1e-6)
    )
    healthy = _archive_selection_health_sort_key(
        _row(CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE)
    )

    assert collapsed[0] == 1
    assert healthy[0] == 0
    assert healthy < collapsed
