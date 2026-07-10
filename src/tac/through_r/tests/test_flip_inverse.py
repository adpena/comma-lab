# SPDX-License-Identifier: MIT
"""Tests for the resize-exploit flip solver (#391, tac.through_r.flip_inverse).

The RIGOR is concentrated on the EXACT parts (kernel/adjoint/dead-zone), validated
NUMERICALLY against the REAL torch ``interpolate`` (the operating-manual "re-derive from
the primary artifact, not your own derivation" discipline) — a convention/transposition
bug MUST fail these. The ledger/solve/verify are exercised end-to-end on the real frozen
CPU SegNet using the small ``gt_n6.npz`` cache (scorer-gated skip if unavailable).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.through_r import flip_inverse as fi
from tac.through_r.flip_inverse import (
    CAMERA_H,
    CAMERA_W,
    SEG_H,
    SEG_PIXELS,
    SEG_W,
    FlipInverseError,
    ResizeComposite,
    build_flip_ledger,
    delta_s_per_flip,
    free_flip_fraction,
    make_resize_degraded_candidate,
    resize_matrix_1d,
    solve_flip_costs,
    uint8_dead_zone,
    verify_targeted_fix,
)

_GT6 = Path("experiments/results/mlx_fleet_gt_cache/gt_n6.npz")


# ----------------------------------------------------------- 1D resize matrix (vs real torch)
def test_resize_matrix_1d_bilinear_matches_torch_bit_exact():
    torch = pytest.importorskip("torch")
    n_in, n_out = 40, 17
    m = resize_matrix_1d(n_in, n_out, "bilinear", align_corners=False)
    assert m.shape == (n_out, n_in)
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 255, size=n_in)
    y_mine = m @ x
    with torch.inference_mode():
        y_torch = (
            torch.nn.functional.interpolate(
                torch.from_numpy(x).reshape(1, 1, 1, n_in).double(),
                size=(1, n_out), mode="bilinear", align_corners=False,
            )[0, 0, 0].numpy()
        )
    assert np.max(np.abs(y_mine - y_torch)) < 1e-12


def test_resize_matrix_1d_bicubic_matches_torch_bit_exact():
    torch = pytest.importorskip("torch")
    n_in, n_out = 12, 55  # upscale (bicubic up path)
    m = resize_matrix_1d(n_in, n_out, "bicubic", align_corners=False)
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 255, size=n_in)
    with torch.inference_mode():
        y_torch = (
            torch.nn.functional.interpolate(
                torch.from_numpy(x).reshape(1, 1, 1, n_in).double(),
                size=(1, n_out), mode="bicubic", align_corners=False,
            )[0, 0, 0].numpy()
        )
    assert np.max(np.abs(m @ x - y_torch)) < 1e-11


def test_resize_matrix_1d_rejects_bad_mode_and_sizes():
    with pytest.raises(FlipInverseError):
        resize_matrix_1d(10, 5, "nearest")
    with pytest.raises(FlipInverseError):
        resize_matrix_1d(0, 5, "bilinear")


# ------------------------------------------------------------- ResizeComposite (exact + adjoint)
def test_camera_composite_down_matches_torch():
    pytest.importorskip("torch")
    comp = ResizeComposite.build()
    errs = comp.validate_against_torch()
    assert errs["down_maxabs"] < 1e-9
    assert "up_maxabs" not in errs  # camera-native: no up op


def test_render_composite_up_and_down_match_torch():
    pytest.importorskip("torch")
    comp = ResizeComposite.build(render_hw=(96, 128))
    errs = comp.validate_against_torch()
    assert errs["down_maxabs"] < 1e-9
    assert errs["up_maxabs"] < 1e-8


def test_row_sums_are_partition_of_unity():
    comp = ResizeComposite.build(render_hw=(96, 128))
    rs = comp.row_sums()
    # bilinear + Keys-bicubic are both normalised (rows sum to 1).
    for k, v in rs.items():
        assert v < 1e-9, (k, v)


def test_down_adjoint_is_exact_transpose():
    # <down_apply(x), y> == <x, down_adjoint(y)> to machine precision (adjoint identity).
    comp = ResizeComposite.build()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((CAMERA_H, CAMERA_W, 1))
    y = rng.standard_normal((SEG_H, SEG_W, 1))
    lhs = float(np.sum(comp.down_apply(x) * y))
    rhs = float(np.sum(x * comp.down_adjoint(y)))
    assert abs(lhs - rhs) <= 1e-6 * (1 + abs(lhs))


def test_seg_pixel_footprint_is_bilinear_stencil():
    comp = ResizeComposite.build()
    cy, cx, w = comp.seg_pixel_footprint(200, 300)
    assert w.shape == (cy.size, cx.size)
    assert cy.size <= 2 and cx.size <= 2  # antialias=False => <=2 taps/axis
    assert abs(float(w.sum()) - 1.0) < 1e-9  # weights are a partition of unity
    assert cy.min() >= 0 and cy.max() < CAMERA_H
    assert cx.min() >= 0 and cx.max() < CAMERA_W


def test_down_apply_shape_validation():
    comp = ResizeComposite.build()
    with pytest.raises(FlipInverseError):
        comp.down_apply(np.zeros((10, 10, 3)))  # not camera grid
    with pytest.raises(FlipInverseError):
        comp.down_adjoint(np.zeros((10, 10, 3)))  # not seg grid


def test_up_apply_requires_render_composite():
    comp = ResizeComposite.build()  # camera-native, no up op
    with pytest.raises(FlipInverseError):
        comp.up_apply(np.zeros((96, 128, 3)))


def test_render_to_camera_uint8_matches_resolution_chain():
    pytest.importorskip("torch")
    from tac.through_r.resolution_chain import render_grid_to_camera_uint8

    comp = ResizeComposite.build(render_hw=(64, 96))
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 255, size=(64, 96, 3))
    a = comp.render_to_camera_uint8(x)
    b = render_grid_to_camera_uint8(x)
    assert a.shape == (CAMERA_H, CAMERA_W, 3) and a.dtype == np.uint8
    # The extracted operator matches the pinned chain to <=1 LSB: the chain runs bicubic in
    # fp32 (`.float()`), my matrix in fp64, so ~0.02% of pixels differ by exactly 1 near a
    # rounding boundary (fp32/fp64 jitter). The scored DOWN path is validated bit-exact
    # separately (validate_against_torch); candidates are built via the fp32 chain itself.
    diff = np.abs(a.astype(np.int64) - b.astype(np.int64))
    assert diff.max() <= 1
    assert float(np.mean(diff > 0)) < 1e-3


# ---------------------------------------------------------------------------- uint8 dead-zone
def test_uint8_dead_zone_range_and_integer_field():
    rng = np.random.default_rng(5)
    camf = rng.uniform(0, 255, (50, 60, 3))
    hz = uint8_dead_zone(camf)
    assert hz.min() >= 0.0 and hz.max() <= 0.5 + 1e-12
    u8 = np.round(camf).astype(np.uint8).astype(np.float64)
    assert np.allclose(uint8_dead_zone(u8), 0.5)  # integer field: full headroom, LSB granularity


def test_uint8_dead_zone_at_half_boundary_is_zero():
    a = np.array([[[10.5, 3.5, 200.5]]])  # exactly on rounding boundaries
    assert np.allclose(uint8_dead_zone(a), 0.0)


def test_free_flip_fraction_uniform_field_is_about_two_eps():
    rng = np.random.default_rng(6)
    camf = rng.uniform(0, 255, (400, 400))
    frac = free_flip_fraction(camf, eps=0.05)
    assert abs(frac - 0.10) < 0.02  # headroom uniform in [0,0.5] => P(<eps)=2*eps


# ---------------------------------------------------------------------------- ΔS + boundary dist
def test_delta_s_per_flip_is_exact():
    assert delta_s_per_flip(600) == pytest.approx(100.0 / (600 * SEG_PIXELS))
    assert delta_s_per_flip(6) == pytest.approx(100.0 / (6 * SEG_PIXELS))


def test_boundary_distance_zero_on_boundary_large_on_uniform():
    lab = np.zeros((20, 20), dtype=np.int64)
    lab[:, 10:] = 1  # a vertical class boundary at x=10
    d = fi._boundary_distance(lab)
    assert d[5, 9] == 0.0 and d[5, 10] == 0.0  # boundary pixels
    assert d[5, 0] > 3.0  # far interior
    uniform = np.zeros((20, 20), dtype=np.int64)
    du = fi._boundary_distance(uniform)
    assert du.min() >= 20.0  # no boundary => all far


# ------------------------------------------------------------------ solve_flip_costs (buckets)
def _toy_ledger(n_flips=1000, n_pairs=600, seed=0):
    rng = np.random.default_rng(seed)
    return fi.FlipLedger(
        candidate_class="toy", n_pairs=n_pairs, total_flips=n_flips,
        total_pixels=n_pairs * SEG_PIXELS, d_seg=n_flips / (n_pairs * SEG_PIXELS),
        pair_idx=rng.integers(0, n_pairs, n_flips).astype(np.int32),
        y=rng.integers(0, SEG_H, n_flips).astype(np.int16),
        x=rng.integers(0, SEG_W, n_flips).astype(np.int16),
        c_wrong=rng.integers(0, 5, n_flips).astype(np.int16),
        c_gt=rng.integers(0, 5, n_flips).astype(np.int16),
        deficit=rng.uniform(0, 3, n_flips).astype(np.float32),
        annulus_dist=rng.uniform(0, 20, n_flips).astype(np.float32),
        persistence=rng.integers(0, 3, n_flips).astype(np.int16),
        is_n600=n_pairs == 600,
    )


def test_solve_flip_costs_buckets_partition_and_monotone():
    led = _toy_ledger()
    fr = solve_flip_costs(led, ResizeComposite.build())
    assert fr.n_free + fr.n_cheap + fr.n_costed + fr.n_unreachable == led.total_flips
    # cumulative Δd_seg / ΔS strictly increasing; last == n_flips/total_pixels
    assert np.all(np.diff(fr.cum_delta_s) > 0)
    assert fr.cum_delta_dseg[-1] == pytest.approx(led.total_flips / led.total_pixels)
    # cost is sorted ascending; cheapest-first frontier
    assert np.all(np.diff(fr.cost) >= 0)
    assert fr.extra["delta_s_per_flip"] == pytest.approx(delta_s_per_flip(led.n_pairs))


def test_solve_flip_costs_empty_ledger():
    led = _toy_ledger(n_flips=0)
    fr = solve_flip_costs(led, ResizeComposite.build())
    assert fr.n_flips == 0 and fr.n_free == 0 and fr.cum_delta_s.size == 0


def test_solve_flip_costs_free_bucket_is_zero_deficit():
    led = _toy_ledger()
    led.deficit[:50] = 0.0  # 50 zero-deficit flips
    led.annulus_dist[:50] = 1.0  # in-annulus so not counted unreachable
    fr = solve_flip_costs(led, ResizeComposite.build())
    assert fr.n_free >= 50


# --------------------------------------------------------- n600 discipline + candidate builder
def test_build_flip_ledger_refuses_non_n600_without_reason():
    frames = [np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8) for _ in range(3)]
    lst = np.zeros((3, SEG_H, SEG_W), dtype=np.int64)
    with pytest.raises(FlipInverseError):
        build_flip_ledger(frames, lstars=lst, candidate_class="toy")


def test_make_resize_degraded_candidate_shape_and_differs():
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    gt = rng.integers(0, 256, (CAMERA_H, CAMERA_W, 3)).astype(np.uint8)
    cand = make_resize_degraded_candidate(gt, (96, 128))
    assert cand.shape == (CAMERA_H, CAMERA_W, 3) and cand.dtype == np.uint8
    assert not np.array_equal(cand, gt)  # degradation actually changed bytes


def test_make_resize_degraded_candidate_rejects_non_camera():
    with pytest.raises(FlipInverseError):
        make_resize_degraded_candidate(np.zeros((10, 10, 3), dtype=np.uint8), (5, 5))


# ---------------------------------------------------------------- scorer-gated end-to-end (n6)
def _load_segnet_or_skip():
    try:
        from tac.through_r.harness import load_frozen_segnet

        return load_frozen_segnet("cpu-torch")
    except Exception as exc:
        pytest.skip(f"frozen SegNet unavailable: {exc}")


@pytest.mark.skipif(not _GT6.exists(), reason="gt_n6 cache absent")
def test_gt_frame_reproduces_lstars_zero_flips():
    # feeding the pristine GT camera frames => realized == lstars => ZERO flips (definition).
    seg = _load_segnet_or_skip()
    z = np.load(_GT6, mmap_mode="r")
    frames = [np.asarray(z["gt_f1"][i]) for i in range(6)]
    lst = np.asarray(z["lstars"])
    led = build_flip_ledger(
        frames, lstars=lst, candidate_class="pristine_gt", segnet=seg,
        allow_subset_reason="n6 identity check",
    )
    assert led.total_flips == 0
    assert led.d_seg == 0.0


@pytest.mark.skipif(not _GT6.exists(), reason="gt_n6 cache absent")
def test_resize_degraded_ledger_and_verify_end_to_end():
    seg = _load_segnet_or_skip()
    z = np.load(_GT6, mmap_mode="r")
    gt_f1 = z["gt_f1"]
    lst = np.asarray(z["lstars"])
    comp_r = ResizeComposite.build(render_hw=(192, 256))
    cand = [make_resize_degraded_candidate(np.asarray(gt_f1[i]), (192, 256), comp_r) for i in range(6)]
    led = build_flip_ledger(
        cand, lstars=lst, candidate_class="resize_degraded_gt(192x256)", segnet=seg,
        allow_subset_reason="n6 end-to-end",
    )
    assert led.total_flips > 0
    # d_seg identity: total_flips / total_pixels
    assert led.d_seg == pytest.approx(led.total_flips / (6 * SEG_PIXELS))
    # class-pair histogram sums to total flips (partition)
    assert sum(led.class_pair_counts.values()) <= led.total_flips  # top-12 truncation
    # per-flip arrays are aligned
    assert led.deficit.shape == (led.total_flips,)
    assert led.annulus_dist.shape == (led.total_flips,)
    assert np.all(led.deficit >= 0.0)

    fr = solve_flip_costs(led, ResizeComposite.build())
    assert fr.n_free + fr.n_cheap + fr.n_costed + fr.n_unreachable == led.total_flips

    gt_cam = [np.asarray(gt_f1[i]) for i in range(6)]
    ver = verify_targeted_fix(
        led, fr, ResizeComposite.build(), cand, gt_cam, lstars=lst,
        top_k=64, step_lsb=16.0, segnet=seg,
    )
    assert ver["predicted"] == min(64, fr.n_flips)
    assert 0 <= ver["realized_fixed"] <= ver["predicted"]
    assert 0.0 <= ver["prediction_vs_realized"] <= 1.0
    # the resize-footprint perturbation removes net flips (the exploit works on this candidate)
    assert ver["net_flips_removed"] >= 0
