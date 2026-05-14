# SPDX-License-Identifier: MIT
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.stbm1br_mask_codec import (
    FEAT_DIAG_TLTL,
    FEAT_PEEL_DIST42,
    FEAT_PREV_BOTTOM2,
    FEAT_PREV_RIGHT2,
    FEAT_X_BIN5_SHIFT,
    STBM1BR_MAGIC,
    STBM1BRError,
    _decode_frame_topband,
    parse_stbm1br_metadata,
    sha256_bytes,
)


REPO = Path(__file__).resolve().parents[3]
PR90_ARCHIVE = REPO / "experiments/results/public_pr90_intake_20260504_worker/archive.zip"


def test_stbm1br_metadata_rejects_missing_magic() -> None:
    with pytest.raises(STBM1BRError, match="bad STBM1BR magic"):
        parse_stbm1br_metadata(b"QMA9" + b"\0" * 32)


def test_real_pr90_stbm1br_segment_metadata_is_self_describing() -> None:
    if not PR90_ARCHIVE.is_file():
        pytest.skip("public PR90 intake archive is not present")
    with zipfile.ZipFile(PR90_ARCHIVE, "r") as zf:
        payload = zf.read("p")
    mask_body = payload[:152_431]
    segment = STBM1BR_MAGIC + mask_body

    metadata = parse_stbm1br_metadata(segment)

    assert metadata.segment_bytes == 152_439
    assert metadata.brotli_body_bytes == 152_431
    assert metadata.brotli_body_sha256 == "420f74e5a02b7d559954c2920e2617846e52ad9d75d46111a3e224cc7d2c14ee"
    assert metadata.segment_sha256 == sha256_bytes(segment)
    assert metadata.qtbm_magic == "QTBM5\0"
    assert (metadata.n_pairs, metadata.height, metadata.width) == (600, 384, 512)
    assert metadata.residual_order is not None
    assert sorted(metadata.residual_order) == [0, 1, 2, 3]


class _FakeRangeDecoder:
    def __init__(self) -> None:
        self.advance_count = 0

    def decode_target(self, total: int) -> int:
        assert total == 4
        return 0

    def advance(self, cum_low: int, cum_high: int, total: int) -> None:
        assert (cum_low, cum_high, total) == (0, 1, 4)
        self.advance_count += 1


def test_decode_frame_topband_skips_prefilled_top_and_road_pixels() -> None:
    height, width = 4, 5
    top = np.zeros((height, width), dtype=np.uint8)
    top[0, :] = 1
    top[1, 0] = 1
    road = np.zeros((height, width), dtype=np.uint8)
    road[3, :] = 1
    road[2, 4] = 1
    frame = np.empty((height, width), dtype=np.uint8)
    decoder = _FakeRangeDecoder()
    cdf_row = [0, 1, 2, 3, 4]

    frame_list = _decode_frame_topband(
        decoder,
        frame,
        [[0] * width for _ in range(height)],
        [[0] * width for _ in range(height)],
        None,
        top,
        road,
        [cdf_row] * (5**4),
        [cdf_row] * (5**5),
        [cdf_row],
        {},
        (
            FEAT_DIAG_TLTL,
            FEAT_PREV_RIGHT2,
            FEAT_PREV_BOTTOM2,
            FEAT_X_BIN5_SHIFT,
            FEAT_PEEL_DIST42,
        ),
        4,
        height,
        width,
        1,
        1,
        [0, 1, 2, 3],
    )

    active = int((~(top.astype(bool) | road.astype(bool))).sum())
    assert decoder.advance_count == active
    assert frame_list == frame.tolist()
    assert (frame[top.astype(bool)] == 2).all()
    assert (frame[(road.astype(bool) & ~top.astype(bool))] == 4).all()
    assert (frame[~(top.astype(bool) | road.astype(bool))] == 0).all()
