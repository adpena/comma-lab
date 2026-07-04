"""Tests for the ``paint`` mode of ``inject_lane_sdf`` (paint-then-SDF, task #291).

The MEASURED nucleation failure (facet-3, 2026-07-04): ``mode="replace"`` writes the thin
lane SDF (max ~+halfwidth) into the lane channel, but that value LOSES the argmax to the deep
road/static-core SDF (~+region-radius), so ``argmax`` never picks the lane and the seed
``part_frac[lane] == 0`` — a zero-mass class the tau/MCF stage then erodes. ``mode="paint"``
paints the lane label into the argmax FIRST then rebuilds all K SDFs, so the lane WINS by
construction. These tests PROVE (not assert) the two behaviors on a synthetic-but-faithful
field: replace nucleates NOTHING, paint nucleates the lane, and the rebuilt field satisfies
the exact SDF invariant. NO-FAKE: real scipy EDT, real argmax, no stub.

Run: ``.venv/bin/python -m pytest src/tac/boundary_math/tests/test_inject_lane_sdf_paint_nucleation.py``
"""
from __future__ import annotations

import numpy as np

from tac.boundary_math.lane_sdf_component import inject_lane_sdf
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields

_H, _W, _K = 48, 64, 5
_LANE = 1


def _road_dominated_phi() -> np.ndarray:
    """A K-field whose argmax is road (0) everywhere EXCEPT a top band = sky (2) and a
    bottom band = hood (4) — i.e. the static-core seed with NO lane and NO movable (exactly
    the #205 structured-init: part_frac = {road, sky, hood} > 0, lane == movable == 0)."""
    labels = np.zeros((_H, _W), np.int64)          # road everywhere
    labels[: _H // 4, :] = 2                        # top -> sky/undrivable
    labels[3 * _H // 4:, :] = 4                     # bottom -> hood
    return signed_distance_fields(labels, n_classes=_K)


def _thin_lane_band_sdf() -> np.ndarray:
    """A thin vertical lane band SDF (+EDT inside a 3px-wide column, -EDT outside) — the
    ``build_structured_lane_sdf`` output shape/sign, small enough to lose a naive argmax."""
    band = np.zeros((_H, _W), bool)
    c = _W // 2
    band[_H // 4: 3 * _H // 4, c - 1: c + 2] = True   # 3px wide, mid-field (over road)
    from scipy import ndimage
    d_in = ndimage.distance_transform_edt(band)
    d_out = ndimage.distance_transform_edt(~band)
    return (d_in - d_out).astype(np.float32)


def test_replace_is_a_nucleation_no_op():
    """MEASURED: mode='replace' leaves part_frac[lane] == 0 — the thin lane SDF loses the
    argmax to the deep road static-core (the exact #205 failure)."""
    phi = _road_dominated_phi()
    lane_sdf = _thin_lane_band_sdf()
    out = inject_lane_sdf(phi, lane_sdf, lane_cls=_LANE, mode="replace")
    part = out.argmax(-1)
    lane_frac = float(np.mean(part == _LANE))
    assert lane_frac == 0.0, f"replace should NOT nucleate the lane, got part_frac[lane]={lane_frac}"


def test_paint_nucleates_the_lane():
    """MEASURED: mode='paint' makes part_frac[lane] > 0 — the lane wins the argmax at the
    band pixels (the nucleation fix)."""
    phi = _road_dominated_phi()
    lane_sdf = _thin_lane_band_sdf()
    band = lane_sdf > 0.0
    out = inject_lane_sdf(phi, lane_sdf, lane_cls=_LANE, mode="paint")
    part = out.argmax(-1)
    lane_frac = float(np.mean(part == _LANE))
    assert lane_frac > 0.0, "paint MUST nucleate the lane"
    # Every band pixel is now lane in the argmax (won by construction).
    assert np.all(part[band] == _LANE), "paint must make the lane win at EVERY band pixel"
    # And it added exactly the band mass (nothing spurious elsewhere vs the painted target).
    assert lane_frac == float(np.mean(band)), (lane_frac, float(np.mean(band)))


def test_paint_rebuilds_a_valid_sdf_field():
    """The rebuilt K-field satisfies the exact SDF invariant argmax_k phi_k == painted labels
    (signed_distance_fields contract) — a CONSISTENT field, not a spiked channel."""
    phi = _road_dominated_phi()
    lane_sdf = _thin_lane_band_sdf()
    band = lane_sdf > 0.0
    out = inject_lane_sdf(phi, lane_sdf, lane_cls=_LANE, mode="paint")
    painted = np.where(band, _LANE, phi.argmax(-1))
    assert np.array_equal(out.argmax(-1), painted), "paint must produce argmax == painted labels"
    # Inside the lane band the lane channel is strictly the max (>0, others <0 near it).
    assert np.all(out[band, _LANE] > 0.0), "lane channel must be +inside its own class"


def test_paint_preserves_the_other_static_classes():
    """Painting the (thin) lane does not destroy the road/sky/hood majority — only the band
    pixels flip to lane; road stays dominant."""
    phi = _road_dominated_phi()
    lane_sdf = _thin_lane_band_sdf()
    before = phi.argmax(-1)
    out = inject_lane_sdf(phi, lane_sdf, lane_cls=_LANE, mode="paint")
    after = out.argmax(-1)
    band = lane_sdf > 0.0
    # Only band pixels changed class.
    changed = before != after
    assert np.array_equal(changed, band), "ONLY the band pixels may change class"
    # Road still the plurality.
    assert float(np.mean(after == 0)) > 0.4, "road must remain the dominant class"


def test_unknown_mode_still_raises():
    phi = _road_dominated_phi()
    lane_sdf = _thin_lane_band_sdf()
    try:
        inject_lane_sdf(phi, lane_sdf, lane_cls=_LANE, mode="bogus")
    except ValueError as e:
        assert "paint" in str(e), "error message should list the valid modes incl paint"
    else:
        raise AssertionError("unknown mode must raise ValueError")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
