# SPDX-License-Identifier: MIT
"""Canonical contest-fixed values surfaced as Python constants with provenance.

The 5 contest_fixed values (per arbitrariness-extinction audit commit
``2d042f7e6``; classified ``resolution_path=contest_fixed`` so NOT arbitrary):

1. Rate term denominator: ``25 * archive_bytes / 37_545_489``
2. Pose coefficient: ``sqrt(10 * pose_avg)``  -> ``sqrt(10)`` constant
3. SegNet class count: 5
4. Non-overlapping pair count: 600
5. Input resolution: ``384 x 512``

Per CLAUDE.md "Non-Negotiable Upstream Rule" + "Apples-to-apples evidence
discipline": these values are MIRRORED from ``upstream/evaluate.py`` and
``upstream/modules.py``; they MUST track upstream snapshot changes.

Citations:
  - ``upstream/evaluate.py`` (pinned snapshot)
  - ``upstream/modules.py`` SegNet 5-class definition
  - ``tac.master_gradient.CONTEST_RATE_DENOM_BYTES`` -- canonical sister
  - ``tac.substrates.score_aware_common.CONTEST_POSE_SQRT_WEIGHT`` -- canonical sister
  - Cover & Thomas 2006 *Elements of Information Theory* Ch.10 -- R(D)
    framework that grounds the rate-denom dimensional analysis

Catalog #305 observability surface: inspectable_per_layer, cite_able.
Catalog #125 hook 1 (sensitivity_map): ACTIVE (the constants are the
analytical scaffolding for ``contest_oracle.gradient``).
"""
from __future__ import annotations

import math
from typing import Final

# ---------------------------------------------------------------------------
# Contest-fixed value 1 -- rate term denominator (BYTES)
# ---------------------------------------------------------------------------
# S_rate = 25 * archive_bytes / CONTEST_RATE_DENOM_BYTES
# Mirrored from upstream/evaluate.py via tac.master_gradient.CONTEST_RATE_DENOM_BYTES
CONTEST_RATE_DENOM_BYTES: Final[int] = 37_545_489
"""Bytes denominator for the contest rate term.

upstream/evaluate.py computes the rate term as ``25 * archive_bytes /
37_545_489``. The denominator is the 1080x1920 video bytecount for one of
the reference videos; it produces a contest-rate-term scaling such that
1 MB ~= 0.665 contest-score-rate-units.
"""

# Per-byte rate coefficient (analytical marginal): 25 / DENOM
CONTEST_RATE_PER_BYTE: Final[float] = 25.0 / CONTEST_RATE_DENOM_BYTES
"""Analytical marginal ``dS/d(archive_bytes) = 25 / CONTEST_RATE_DENOM_BYTES``.

Approximately ``6.66e-7`` per CLAUDE.md ``SegNet vs PoseNet importance``
section + arbitrariness audit row #15. This IS the gradient oracle for
the rate axis -- closed-form, byte-exact, no hand tuning required.
"""

# Bulk weight on the rate term (canonical alias)
CONTEST_RATE_WEIGHT: Final[float] = 25.0
"""Weight on the rate term: ``S_rate = CONTEST_RATE_WEIGHT * archive_bytes / DENOM``."""

# ---------------------------------------------------------------------------
# Contest-fixed value 2 -- pose coefficient (sqrt(10))
# ---------------------------------------------------------------------------
# S_pose = sqrt(10 * d_pose)  =  sqrt(10) * sqrt(d_pose)
CONTEST_POSE_SQRT_INNER: Final[int] = 10
"""Inner coefficient of the pose sqrt: ``sqrt(10 * d_pose)``."""

CONTEST_POSE_SQRT_WEIGHT: Final[float] = math.sqrt(CONTEST_POSE_SQRT_INNER)
"""``sqrt(10) ~= 3.1622776601683795``; canonical sister of
``tac.substrates.score_aware_common.CONTEST_POSE_SQRT_WEIGHT``.

The sqrt curvature implies the marginal dS/d(d_pose) DIVERGES as
d_pose -> 0: ``dS/d(d_pose) = 5 / sqrt(10 * d_pose)``. At PR106
frontier pose_avg=3.4e-5, the marginal is ~271 (vs constant 100 for seg).
This is why pose is the high-marginal-value axis at the frontier
operating point (CLAUDE.md operating-point-dependent rule).
"""

# ---------------------------------------------------------------------------
# Contest-fixed value 3 -- SegNet class count
# ---------------------------------------------------------------------------
SEGNET_NUM_CLASSES: Final[int] = 5
"""Number of SegNet output classes; from upstream/modules.py ``smp.Unet(..,
classes=5)``. Drives per-class Lagrangian (Impl 5) + class-conditional CDF
(Impl 14). NOT arbitrary; mandated by the contest's scorer architecture.
"""

# Per-axis weight (bulk) on segmentation distortion
CONTEST_SEG_WEIGHT: Final[float] = 100.0
"""Weight on the seg term: ``S_seg = 100 * d_seg`` (constant marginal)."""

# ---------------------------------------------------------------------------
# Contest-fixed value 4 -- non-overlapping pair count
# ---------------------------------------------------------------------------
CONTEST_NUM_PAIRS: Final[int] = 600
"""Number of non-overlapping (frame_t, frame_{t+1}) pairs.

upstream/evaluate.py uses ``seq_len=2`` non-overlapping batching over a
1200-frame video -> 600 pairs. NOT 1199 overlapping; see CLAUDE.md
"CATASTROPHIC FAILURES" the 1199-vs-600 lesson. Drives per-pair
decomposition (Impl 4) and bandit-style per-pair Thompson sampling
(Impl 13).
"""

# ---------------------------------------------------------------------------
# Contest-fixed value 5 -- input resolution
# ---------------------------------------------------------------------------
CONTEST_INPUT_HEIGHT: Final[int] = 384
"""Renderer input/output height in pixels."""

CONTEST_INPUT_WIDTH: Final[int] = 512
"""Renderer input/output width in pixels."""

CONTEST_PIXELS_PER_FRAME: Final[int] = CONTEST_INPUT_HEIGHT * CONTEST_INPUT_WIDTH
"""``384 * 512 = 196_608`` pixels per frame."""

# Per-archive cell count (per Impl 10): pixels * pairs * classes_per_pixel
# (with class assignment one-hot, ``CONTEST_PIXELS_PER_FRAME * CONTEST_NUM_PAIRS``
# is the effective per-archive cell count for the 588M figure; multiply by
# 5 for full per-class cell count = ``588_879_360``).
CONTEST_PER_ARCHIVE_PIXEL_CELLS: Final[int] = (
    CONTEST_PIXELS_PER_FRAME * CONTEST_NUM_PAIRS
)
"""``196_608 * 600 = 117_964_800`` pixel cells per archive (per-class scaling
multiplies by 5 = 589_824_000 ~= 588M per design memo)."""

CONTEST_PER_ARCHIVE_PER_CLASS_CELLS: Final[int] = (
    CONTEST_PER_ARCHIVE_PIXEL_CELLS * SEGNET_NUM_CLASSES
)
"""``117_964_800 * 5 = 589_824_000`` per-class pixel cells; the 588M-cell
sparse water-filling decomposition surface for Impl 10."""

# ---------------------------------------------------------------------------
# Convenience: canonical tuple of all axis labels
# ---------------------------------------------------------------------------
SCORE_AXIS_LABELS: Final[tuple[str, ...]] = ("seg", "pose", "rate")
"""Canonical tuple of the 3 score axes; mirrors
``tac.master_gradient.SCORE_AXIS_LABELS``."""


__all__ = [
    "CONTEST_RATE_DENOM_BYTES",
    "CONTEST_RATE_PER_BYTE",
    "CONTEST_RATE_WEIGHT",
    "CONTEST_POSE_SQRT_INNER",
    "CONTEST_POSE_SQRT_WEIGHT",
    "CONTEST_SEG_WEIGHT",
    "SEGNET_NUM_CLASSES",
    "CONTEST_NUM_PAIRS",
    "CONTEST_INPUT_HEIGHT",
    "CONTEST_INPUT_WIDTH",
    "CONTEST_PIXELS_PER_FRAME",
    "CONTEST_PER_ARCHIVE_PIXEL_CELLS",
    "CONTEST_PER_ARCHIVE_PER_CLASS_CELLS",
    "SCORE_AXIS_LABELS",
]
