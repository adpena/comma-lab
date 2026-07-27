# SPDX-License-Identifier: MIT
"""Research-only frame-0 owner used until the final conditional Y0 plug-in."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

VARIANT_ID: Final = "tac.semantic_root_y0.duplicate_y1_research_only.v1"


def prepare(packet: bytes, archive_root: Path) -> None:
    if type(packet) is not bytes or not packet or archive_root.is_symlink():
        raise ValueError("duplicate-Y1 frame0 requires a nonempty counted semantic packet")
    return None


def render_camera_y0(
    state: None,
    pair_id: int,
    scorer_y1: np.ndarray,
    camera_y1: np.ndarray,
) -> np.ndarray:
    del state, scorer_y1
    if type(pair_id) is not int or not 0 <= pair_id < 600:
        raise ValueError("pair_id is outside exact n600")
    if camera_y1.dtype != np.uint8 or camera_y1.shape != (874, 1164, 3):
        raise ValueError("camera_y1 shape/dtype differs")
    return camera_y1
