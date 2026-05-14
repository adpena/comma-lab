# SPDX-License-Identifier: MIT
"""Tests for canonical comma10k / contest SegNet semantic labels."""

from __future__ import annotations

from tac import camera
from tac.semantic_label_contract import (
    CONTEST_SEGNET_CLASS_NAME_TUPLE,
    CONTEST_SEGNET_CLASS_NAMES,
    NUM_CONTEST_SEGNET_CLASSES,
    SELFCOMP_CLASS_TO_GRAY,
    validate_contest_class_table,
)


def test_contest_label_contract_matches_camera_exports() -> None:
    validate_contest_class_table()
    assert camera.NUM_CLASSES == NUM_CONTEST_SEGNET_CLASSES
    assert camera.SEGNET_CLASS_NAMES == CONTEST_SEGNET_CLASS_NAME_TUPLE
    assert tuple(CONTEST_SEGNET_CLASS_NAMES.values()) == (
        "road",
        "lane_markings",
        "undrivable",
        "movable",
        "my_car",
    )


def test_selfcomp_gray_targets_are_wire_codebook_not_label_names() -> None:
    assert SELFCOMP_CLASS_TO_GRAY == {
        0: 0,
        1: 255,
        2: 64,
        3: 192,
        4: 128,
    }
    assert CONTEST_SEGNET_CLASS_NAMES[0] == "road"
    assert CONTEST_SEGNET_CLASS_NAMES[4] == "my_car"
