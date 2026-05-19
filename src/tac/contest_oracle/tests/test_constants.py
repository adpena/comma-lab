# SPDX-License-Identifier: MIT
"""Tests for ``tac.contest_oracle.constants`` -- contest_fixed value mirror."""
from __future__ import annotations

import math

from tac.contest_oracle.constants import (
    CONTEST_INPUT_HEIGHT,
    CONTEST_INPUT_WIDTH,
    CONTEST_NUM_PAIRS,
    CONTEST_PER_ARCHIVE_PER_CLASS_CELLS,
    CONTEST_PER_ARCHIVE_PIXEL_CELLS,
    CONTEST_PIXELS_PER_FRAME,
    CONTEST_POSE_SQRT_INNER,
    CONTEST_POSE_SQRT_WEIGHT,
    CONTEST_RATE_DENOM_BYTES,
    CONTEST_RATE_PER_BYTE,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
    SCORE_AXIS_LABELS,
    SEGNET_NUM_CLASSES,
)


def test_rate_denom_bytes_canonical_value():
    """Per upstream/evaluate.py + tac.master_gradient.CONTEST_RATE_DENOM_BYTES."""
    assert CONTEST_RATE_DENOM_BYTES == 37_545_489


def test_rate_per_byte_is_25_over_denom():
    """Closed-form analytical marginal dS/d(bytes)."""
    assert math.isclose(CONTEST_RATE_PER_BYTE, 25.0 / 37_545_489)
    # ~6.66e-7 per CLAUDE.md operating-point heuristic
    assert 6.0e-7 < CONTEST_RATE_PER_BYTE < 7.0e-7


def test_pose_sqrt_weight_is_sqrt_10():
    """Canonical pose coefficient."""
    assert math.isclose(CONTEST_POSE_SQRT_WEIGHT, math.sqrt(10))
    assert CONTEST_POSE_SQRT_INNER == 10


def test_score_weights_match_contest_formula():
    """S = 100*d_seg + sqrt(10*d_pose) + 25*R per upstream/evaluate.py."""
    assert CONTEST_SEG_WEIGHT == 100.0
    assert CONTEST_RATE_WEIGHT == 25.0


def test_segnet_num_classes_is_5():
    """Per upstream/modules.py smp.Unet(.., classes=5)."""
    assert SEGNET_NUM_CLASSES == 5


def test_contest_num_pairs_is_600():
    """600 non-overlapping pairs (NOT 1199 overlapping per CLAUDE.md lesson)."""
    assert CONTEST_NUM_PAIRS == 600


def test_input_resolution_384x512():
    """Per upstream/mask_extractor canonical output."""
    assert CONTEST_INPUT_HEIGHT == 384
    assert CONTEST_INPUT_WIDTH == 512
    assert CONTEST_PIXELS_PER_FRAME == 384 * 512 == 196_608


def test_per_archive_pixel_cells_canonical():
    """117_964_800 = 196_608 * 600."""
    assert CONTEST_PER_ARCHIVE_PIXEL_CELLS == 117_964_800


def test_per_archive_per_class_cells_canonical():
    """588M+ = 117_964_800 * 5."""
    assert CONTEST_PER_ARCHIVE_PER_CLASS_CELLS == 117_964_800 * 5
    assert CONTEST_PER_ARCHIVE_PER_CLASS_CELLS == 589_824_000


def test_score_axis_labels_canonical():
    """Mirror tac.master_gradient.SCORE_AXIS_LABELS."""
    assert SCORE_AXIS_LABELS == ("seg", "pose", "rate")


def test_canonical_helper_consistency_with_master_gradient():
    """Catalog #229 sister-check: our DENOM matches master_gradient's."""
    from tac.master_gradient import CONTEST_RATE_DENOM_BYTES as sister_denom

    assert CONTEST_RATE_DENOM_BYTES == sister_denom


def test_canonical_helper_consistency_with_score_aware_common():
    """Catalog #229 sister-check: pose sqrt weight matches sister."""
    from tac.substrates.score_aware_common import (
        CONTEST_POSE_SQRT_WEIGHT as sister_pose,
    )

    assert math.isclose(CONTEST_POSE_SQRT_WEIGHT, sister_pose)
