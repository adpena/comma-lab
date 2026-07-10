# SPDX-License-Identifier: MIT
"""Tests for tac.through_r.roadlane_texture_generator (#394 UNIT A texture-fill generator).

$0 / no SegNet / no gt cache: exercises the composition + byte-accounting logic on synthetic
label maps + explicit fill plans. The through-R d_seg path (run_composed_generator_arm) needs the
frozen SegNet + gt_n600 and is a governed n600 measurement, exercised by the driver, not here.
"""
from __future__ import annotations

import typing

import numpy as np
import pytest

from tac.through_r.resolution_chain import SEG_H, SEG_W
from tac.through_r.roadlane_texture_generator import (
    BASIN_CLASSES,
    TEXTURE_CLASSES,
    ClassFill,
    RoadLaneTextureError,
    TextureFillPlan,
    byte_account_texture_fill,
    default_roadlane_grating_specs,
    fill_partition_texture,
    fit_texture_fill_plan,
)
from tac.through_r.stem_perception import TextureSpec, texture_dl_bits


def _explicit_plan(color_quant: int = 5) -> TextureFillPlan:
    """A fully-explicit plan (no SegNet): flat basins + default gratings."""
    fills: dict[int, ClassFill] = {}
    names = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
    grat = default_roadlane_grating_specs()
    for c in BASIN_CLASSES:
        spec = TextureSpec(family="flat", c_a=(120.0, 120.0, 120.0))
        fills[c] = ClassFill(c, names[c], spec, texture_dl_bits(spec, color_quant=color_quant), "given")
    for c in TEXTURE_CLASSES:
        spec = grat[c]
        fills[c] = ClassFill(c, names[c], spec, texture_dl_bits(spec, color_quant=color_quant), "given")
    return byte_account_texture_fill(
        TextureFillPlan(fills=fills, total_texture_bits=0, total_texture_bytes=0.0, color_quant=color_quant)
    )


def test_default_grating_polarity_reversed():
    g = default_roadlane_grating_specs()
    road, lane = g[0], g[1]
    # Road = bright-on-dark, Lane = dark-on-bright -> the two-colour order is swapped.
    assert road.c_a == lane.c_b
    assert road.c_b == lane.c_a
    assert road.period == 4 and lane.period == 4  # stem-Nyquist


def test_default_grating_period_is_stem_nyquist():
    g = default_roadlane_grating_specs(period=4)
    assert all(s.period == 4 for s in g.values())


def test_byte_account_sums_perclass_bits():
    plan = _explicit_plan()
    assert plan.total_texture_bits == sum(f.bits for f in plan.fills.values())
    assert plan.total_texture_bytes == pytest.approx(plan.total_texture_bits / 8.0)
    # 3 flats (15 bits) + 2 stripes (1+15+15+4+2+3 = 40 bits) = 45 + 80 = 125 bits.
    assert plan.total_texture_bits == 3 * 15 + 2 * 40


def test_texture_rate_is_tiny_whole_video():
    plan = _explicit_plan()
    # the whole-video texture carrier is < 20 bytes (near-free; the rate is the geometry carrier).
    assert plan.total_texture_bytes < 20.0


def test_fill_partition_shapes_and_dtype():
    plan = _explicit_plan()
    lab = np.zeros((SEG_H, SEG_W), dtype=np.int64)
    lab[100:200, 100:200] = 1  # a Lane block
    lab[:50, :] = 2            # Undrivable top
    frame = fill_partition_texture(lab, plan)
    assert frame.shape == (SEG_H, SEG_W, 3)
    assert frame.dtype == np.float64


def test_fill_partition_basin_is_flat():
    plan = _explicit_plan()
    lab = np.full((SEG_H, SEG_W), 2, dtype=np.int64)  # all Undrivable (a basin)
    frame = fill_partition_texture(lab, plan)
    # a flat basin -> every pixel identical.
    assert np.allclose(frame, frame[0, 0][None, None, :])


def test_fill_partition_road_is_textured_not_flat():
    plan = _explicit_plan()
    lab = np.zeros((SEG_H, SEG_W), dtype=np.int64)  # all Road (grating)
    frame = fill_partition_texture(lab, plan)
    # a grating -> NOT flat (has period-4 variation).
    assert frame.std() > 1.0


def test_fill_partition_grating_period_present():
    # Use an EXPLICIT orientation-0 grating (the default winner is diagonal 135°) so the
    # period-4 repeat is checkable along the x-axis.
    fills = dict(_explicit_plan().fills)
    fills[0] = ClassFill(
        0, "Road",
        TextureSpec(family="stripe", c_a=(160.0, 160.0, 160.0), c_b=(0.0, 0.0, 0.0), period=4, orientation=0.0),
        40, "given",
    )
    plan = byte_account_texture_fill(
        TextureFillPlan(fills=fills, total_texture_bits=0, total_texture_bytes=0.0, color_quant=5)
    )
    lab = np.zeros((SEG_H, SEG_W), dtype=np.int64)  # all Road, orientation 0 -> vertical stripes
    frame = fill_partition_texture(lab, plan)
    row = frame[SEG_H // 2, :, 0]
    # period-4 square wave: value repeats every 4 px.
    assert np.allclose(row[0:4], row[4:8])


def test_fill_partition_missing_class_raises():
    # a plan missing class 4 (MyCar) but the lab contains it.
    plan = _explicit_plan()
    del plan.fills[4]
    lab = np.full((SEG_H, SEG_W), 4, dtype=np.int64)
    with pytest.raises(RoadLaneTextureError):
        fill_partition_texture(lab, plan)


def test_fill_partition_wrong_shape_raises():
    plan = _explicit_plan()
    with pytest.raises(RoadLaneTextureError):
        fill_partition_texture(np.zeros((10, 10), dtype=np.int64), plan)


def test_fit_plan_from_explicit_inputs_no_segnet():
    # given basin colours + roadlane specs -> no SegNet probe needed.
    basin = {2: (10.0, 20.0, 30.0), 3: (40.0, 50.0, 60.0), 4: (70.0, 80.0, 90.0)}
    grat = default_roadlane_grating_specs()
    plan = fit_texture_fill_plan(basin_flat_colors=basin, roadlane_specs=grat)
    assert set(plan.fills) == {0, 1, 2, 3, 4}
    assert plan.fills[2].source == "given"
    assert plan.fills[0].source == "given"
    assert plan.fills[2].spec.c_a == (10.0, 20.0, 30.0)
    assert plan.total_texture_bits > 0


def test_fit_plan_basins_are_flat_texture_classes_are_gratings():
    basin = {2: (10.0, 20.0, 30.0), 3: (40.0, 50.0, 60.0), 4: (70.0, 80.0, 90.0)}
    plan = fit_texture_fill_plan(basin_flat_colors=basin, roadlane_specs=default_roadlane_grating_specs())
    for c in BASIN_CLASSES:
        assert plan.fills[c].spec.family == "flat"
    for c in TEXTURE_CLASSES:
        assert plan.fills[c].spec.family in ("stripe", "gabor", "checker")


def test_fit_plan_uses_price_list_flat_for_basins():
    # a fake price-list object with a cheapest flat for Undrivable.
    class _Pt:
        def __init__(self, spec):
            self.spec = spec

    class _PL:
        per_class_cheapest: typing.ClassVar[dict] = {
            "Undrivable": _Pt(TextureSpec(family="flat", c_a=(5.0, 5.0, 5.0))),
            "Movable": _Pt(TextureSpec(family="flat", c_a=(6.0, 6.0, 6.0))),
            "MyCar": _Pt(TextureSpec(family="flat", c_a=(7.0, 7.0, 7.0))),
            "Road": _Pt(TextureSpec(family="stripe", c_a=(0.0, 0.0, 0.0), c_b=(128.0, 128.0, 128.0), period=4)),
            "Lane": _Pt(TextureSpec(family="stripe", c_a=(128.0, 128.0, 128.0), c_b=(0.0, 0.0, 0.0), period=4)),
        }

    plan = fit_texture_fill_plan(price_list=_PL())
    assert plan.fills[2].source == "price_list"
    assert plan.fills[2].spec.c_a == (5.0, 5.0, 5.0)
    assert plan.fills[0].source == "price_list"
    assert plan.fills[0].spec.family == "stripe"


def test_period_lt_one_raises():
    with pytest.raises(RoadLaneTextureError):
        default_roadlane_grating_specs(period=0)


def test_all_five_classes_covered():
    plan = _explicit_plan()
    assert set(plan.fills) == set(BASIN_CLASSES) | set(TEXTURE_CLASSES) == {0, 1, 2, 3, 4}
