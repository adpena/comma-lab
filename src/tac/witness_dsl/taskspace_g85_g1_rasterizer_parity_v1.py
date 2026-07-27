# SPDX-License-Identifier: MIT
"""Exact n600 OpenCV/Pillow parity classifier for the counted G1 worldsheet.

The retained G1 semantic stream is decoded by the canonical OpenCV receiver and
also by a portable Pillow candidate over the exact typed polygon lift.  This
module records whole-population mask hashes and disagreement, so an approximate
replacement cannot be mistaken for public receiver closure.

This is a blocker classifier, not a public decoder.  Importing the canonical G1
module still imports OpenCV, and a nonzero disagreement keeps the public-runtime
claim false.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
from PIL import Image, ImageDraw

from tac.optimization.direct_description_g1_worldsheet import (
    HEIGHT,
    WIDTH,
    G1WorldsheetParameterLiftV1,
    decode_g1_movable_worldsheet,
    lift_g1_movable_worldsheet,
)

RECEIPT_SCHEMA: Final = "tac.g85_g1_rasterizer_parity_receipt.v1"
OPEN_BLOCKER: Final = "G1_OPENCV_FILLPOLY_BIT_EXACT_GENERIC_RASTERIZER_OWED"


class G1RasterizerParityError(ValueError):
    """The G1 input, typed lift, or parity receipt violated its contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rasterize_lift_with_pillow(lift: G1WorldsheetParameterLiftV1) -> np.ndarray:
    """Rasterize the exact lifted polygons with Pillow's integer primitives."""

    if not 1 <= lift.pair_count <= 600:
        raise G1RasterizerParityError("G1 Pillow raster pair count is invalid")
    templates = {row.template_ref: row.relative_vertices_xy for row in lift.templates}
    if len(templates) != len(lift.templates):
        raise G1RasterizerParityError("G1 Pillow raster template refs are not unique")
    rendered = np.zeros((lift.pair_count, HEIGHT, WIDTH), dtype=bool)
    for knot in lift.knots:
        relative = templates.get(knot.template_ref)
        if relative is None:
            raise G1RasterizerParityError("G1 Pillow raster knot template is absent")
        points = tuple((knot.center_x + x, knot.center_y + y) for x, y in relative)
        if (
            not points
            or knot.pair_index < 0
            or knot.pair_index >= lift.pair_count
            or any(x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT for x, y in points)
        ):
            raise G1RasterizerParityError("G1 Pillow raster polygon escaped scorer geometry")
        image = Image.new("L", (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(image)
        if len(points) == 1:
            draw.point(points, fill=1)
        elif len(points) == 2:
            draw.line(points, fill=1, width=1)
        else:
            draw.polygon(points, fill=1)
        rendered[knot.pair_index] |= np.asarray(image, dtype=np.uint8).astype(bool)
    return rendered


@dataclass(frozen=True, slots=True)
class G1RasterizerParityReceiptV1:
    source_g1_bytes: int
    source_g1_sha256: str
    pair_count: int
    max_slots: int
    polygon_count: int
    vertex_count: int
    one_vertex_polygon_count: int
    two_vertex_polygon_count: int
    three_or_more_vertex_polygon_count: int
    canonical_mask_bytes: int
    canonical_mask_sha256: str
    portable_mask_bytes: int
    portable_mask_sha256: str
    differing_pixels: int
    differing_frames: int
    maximum_differing_pixels_per_frame: int
    exact_mask_equality: bool
    schema: str = RECEIPT_SCHEMA
    source_population: str = "retained_fresh_v15_semantic_P_full_n600"
    canonical_rasterizer: str = "opencv_cv2_fillPoly_LINE_8_shift_0"
    portable_rasterizer: str = "pillow_ImageDraw_integer_point_line_polygon"
    verdict_scope: str = "this_exact_pillow_integer_rasterizer_on_retained_G1_n600"
    public_receiver_closed: bool = False
    evaluator_invoked: bool = False
    score_claim: bool = False
    candidate_claim: bool = False
    research_only: bool = True
    open_blocker: str = OPEN_BLOCKER

    def __post_init__(self) -> None:
        if (
            self.schema != RECEIPT_SCHEMA
            or re.fullmatch(r"[0-9a-f]{64}", self.source_g1_sha256) is None
            or re.fullmatch(r"[0-9a-f]{64}", self.canonical_mask_sha256) is None
            or re.fullmatch(r"[0-9a-f]{64}", self.portable_mask_sha256) is None
            or self.pair_count != 600
            or self.polygon_count
            != (self.one_vertex_polygon_count + self.two_vertex_polygon_count + self.three_or_more_vertex_polygon_count)
            or self.canonical_mask_bytes != self.portable_mask_bytes
            or self.exact_mask_equality != (self.differing_pixels == 0)
            or self.public_receiver_closed is not False
            or self.evaluator_invoked is not False
            or self.score_claim is not False
            or self.candidate_claim is not False
            or self.research_only is not True
            or self.open_blocker != OPEN_BLOCKER
        ):
            raise G1RasterizerParityError("G1 rasterizer parity receipt truth differs")

    def to_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")


def measure_pillow_g1_rasterizer_parity(
    payload: bytes,
    *,
    expected_pairs: int = 600,
) -> G1RasterizerParityReceiptV1:
    """Measure whole-population mask equality against canonical ``fillPoly``."""

    if expected_pairs != 600:
        raise G1RasterizerParityError("G1 public parity authority requires all 600 pairs")
    canonical, metadata = decode_g1_movable_worldsheet(payload, expected_pairs=expected_pairs)
    lift = lift_g1_movable_worldsheet(payload)
    if lift.pair_count != expected_pairs or metadata.pair_count != expected_pairs:
        raise G1RasterizerParityError("G1 rasterizer population custody differs")
    portable = rasterize_lift_with_pillow(lift)
    if canonical.shape != portable.shape or canonical.dtype != np.bool_ or portable.dtype != np.bool_:
        raise G1RasterizerParityError("G1 rasterizer output ABI differs")

    per_frame = np.count_nonzero(canonical != portable, axis=(1, 2))
    templates = {row.template_ref: row.relative_vertices_xy for row in lift.templates}
    vertex_counts = tuple(len(templates[knot.template_ref]) for knot in lift.knots)
    return G1RasterizerParityReceiptV1(
        source_g1_bytes=len(payload),
        source_g1_sha256=_sha256(payload),
        pair_count=lift.pair_count,
        max_slots=lift.max_slots,
        polygon_count=len(lift.knots),
        vertex_count=sum(vertex_counts),
        one_vertex_polygon_count=sum(count == 1 for count in vertex_counts),
        two_vertex_polygon_count=sum(count == 2 for count in vertex_counts),
        three_or_more_vertex_polygon_count=sum(count >= 3 for count in vertex_counts),
        canonical_mask_bytes=canonical.nbytes,
        canonical_mask_sha256=_sha256(canonical.tobytes(order="C")),
        portable_mask_bytes=portable.nbytes,
        portable_mask_sha256=_sha256(portable.tobytes(order="C")),
        differing_pixels=int(per_frame.sum()),
        differing_frames=int(np.count_nonzero(per_frame)),
        maximum_differing_pixels_per_frame=int(per_frame.max(initial=0)),
        exact_mask_equality=bool(np.array_equal(canonical, portable)),
    )


__all__ = [
    "OPEN_BLOCKER",
    "G1RasterizerParityError",
    "G1RasterizerParityReceiptV1",
    "measure_pillow_g1_rasterizer_parity",
    "rasterize_lift_with_pillow",
]
