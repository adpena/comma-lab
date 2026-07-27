# SPDX-License-Identifier: MIT
"""Full-n600 exact G1 OpenCV/Pillow parity blocker fixture."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from tac.witness_dsl import taskspace_g85_g1_rasterizer_parity_v1 as g85r

_ROOT = Path(__file__).resolve().parents[4]
_SEMANTIC_PATH = (
    _ROOT
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "fresh_v15_semantic_base_n600_20260726"
    / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
)
_G1_SHA256 = "1066081727229e605462e67b8fdd26937d5e3552c13cb66a7444ea3b7360366f"


def _g1() -> bytes:
    if not _SEMANTIC_PATH.is_file():
        pytest.skip("retained fresh V15 semantic custody is absent")
    with zipfile.ZipFile(io.BytesIO(_SEMANTIC_PATH.read_bytes()), "r") as reader:
        payload = reader.read("predict/movable_polygon_worldsheet.g1s")
    assert len(payload) == 29_810
    assert hashlib.sha256(payload).hexdigest() == _G1_SHA256
    return payload


def test_exact_all_n600_pillow_candidate_is_not_fillpoly_parity() -> None:
    receipt = g85r.measure_pillow_g1_rasterizer_parity(_g1())
    assert receipt.pair_count == 600
    assert receipt.max_slots == 10
    assert receipt.polygon_count == 2_197
    assert receipt.vertex_count == 19_150
    assert receipt.one_vertex_polygon_count == 42
    assert receipt.two_vertex_polygon_count == 101
    assert receipt.three_or_more_vertex_polygon_count == 2_054
    assert receipt.canonical_mask_bytes == receipt.portable_mask_bytes == 117_964_800
    assert receipt.canonical_mask_sha256 == ("42ecdc1d82402cd27ce7c54b198087389f51c114c4f691324d12d433c8d8acd4")
    assert receipt.portable_mask_sha256 == ("826d3a9df72fd09e1ec84e9ddc959c6b46a81fe69e0c3247c3c4939dd0dc0825")
    assert receipt.differing_pixels == 28_648
    assert receipt.differing_frames == 600
    assert receipt.maximum_differing_pixels_per_frame == 134
    assert receipt.exact_mask_equality is False
    assert receipt.public_receiver_closed is False
    assert receipt.open_blocker == g85r.OPEN_BLOCKER


def test_subset_measurement_cannot_be_misrepresented_as_authority() -> None:
    with pytest.raises(g85r.G1RasterizerParityError, match="all 600"):
        g85r.measure_pillow_g1_rasterizer_parity(_g1(), expected_pairs=96)
