"""Tests for the ddm_ar1 AA-render price instrument.

The load-bearing invariants are structural, not statistical:

* ``ss == 1`` must be the trainer's current point-sampled render BIT-FOR-BIT, otherwise the AA
  delta is measuring two changes at once.
* the ``ss > 1`` path must be the exact ``ss x ss`` block mean of the fine render, because that
  block mean IS the pixel-footprint integral the AA law names.
* the per-pair d_seg / d_pose arithmetic must be the burn's own arithmetic
  (``ddm_qbt1_qbflow_trainer._retain_eval_outputs``), otherwise the calibration gate is vacuous.
* the B/H/W split must partition the frame by target class and must be sign-correct.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments import ddm_ar1_aa_render_price as ar1
from experiments import ddm_qbt1_qbflow_trainer as qbt
from tac.boundary_math.aa_sdf_observation_render import (
    build_render_coords,
    build_supersampled_coords,
)


# ---------------------------------------------------------------------------
# pair selections
# ---------------------------------------------------------------------------
def test_seeded_random_32_is_deterministic_and_in_range() -> None:
    first = ar1.seeded_random_32()
    assert first == ar1.seeded_random_32()
    assert len(first) == 32 == len(set(first))
    assert all(0 <= pair_id < qbt.N for pair_id in first)
    assert list(first) == sorted(first)


def test_resolve_pairs_spellings() -> None:
    assert ar1.resolve_pairs("all") == tuple(range(qbt.N))
    assert ar1.resolve_pairs("selection") == tuple(qbt.SELECTION_IDS)
    assert ar1.resolve_pairs("seeded32") == ar1.seeded_random_32()
    assert ar1.resolve_pairs("7,3,3, 11") == (3, 7, 11)


# ---------------------------------------------------------------------------
# the fine lattice IS the AA module's lattice (the premise the whole arm rests on)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ss", [1, 2, 3])
def test_trainer_lattice_matches_aa_module_coords_to_half_a_float32_ulp(ss: int) -> None:
    """Same lattice, NOT bit-identical: torch and numpy round ``linspace``'s last bit differently.

    MEASURED max gap 5.960464e-08 (0.5 ULP of float32) at every grid this arm uses -- 1.5e-05 of
    one base-grid pixel pitch.  The instrument builds both the ss=1 and ss>1 grids through the
    trainer's own ``forward``, so this gap never enters the AA delta; the test pins the bound so a
    real lattice divergence cannot hide behind it.
    """
    height, width = 12, 16
    _base, x, y = qbt._base_features(
        torch.tensor([0], dtype=torch.long), height=height * ss, width=width * ss
    )
    trainer_coords = np.stack(
        [x[0].numpy().ravel(), y[0].numpy().ravel()], axis=-1
    ).astype(np.float32)
    module_coords = build_supersampled_coords(height, width, ss)
    assert trainer_coords.shape == module_coords.shape
    assert np.abs(trainer_coords - module_coords).max() <= 5.960464477539063e-08


def test_supersampled_coords_at_ss1_is_the_base_grid() -> None:
    assert np.array_equal(build_supersampled_coords(9, 7, 1), build_render_coords(9, 7))


# ---------------------------------------------------------------------------
# the module lattice's registration defect, and the corrected lattice
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("ss", "expected_pixels"), [(2, 0.2497), (3, 0.3328), (4, 0.3743)]
)
def test_module_lattice_block_centres_drift_from_the_coarse_samples(
    ss: int, expected_pixels: float
) -> None:
    """MEASURED defect: ``linspace(-1,1,n*ss)`` blocks are not centred on ``linspace(-1,1,n)``.

    The drift is inward at both frame edges and exactly zero at the centre, so the module's AA
    image is a slightly contracted copy of the field.  Pinning the magnitude keeps the memo's
    registration-vs-blur decomposition honest.
    """
    n = 384
    coarse = np.linspace(-1.0, 1.0, n)
    block_means = np.linspace(-1.0, 1.0, n * ss).reshape(n, ss).mean(axis=1)
    pitch = 2.0 / (n - 1)
    drift_pixels = np.abs(block_means - coarse) / pitch
    assert drift_pixels.max() == pytest.approx(expected_pixels, abs=1e-4)
    assert drift_pixels[n // 2] < 1e-3  # zero at the centre
    assert (block_means - coarse)[0] > 0 and (block_means - coarse)[-1] < 0  # inward at both edges


@pytest.mark.parametrize("ss", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n", [8, 384, 512])
def test_footprint_centred_span_blocks_are_exactly_the_coarse_samples(n: int, ss: int) -> None:
    lo, hi = ar1.footprint_centred_span(n, ss)
    fine = np.linspace(lo, hi, n * ss)
    block_means = fine.reshape(n, ss).mean(axis=1)
    np.testing.assert_allclose(block_means, np.linspace(-1.0, 1.0, n), atol=1e-12)


def test_footprint_centred_span_is_the_identity_at_ss1() -> None:
    assert ar1.footprint_centred_span(384, 1) == (-1.0, 1.0)


def test_footprint_centred_span_refuses_degenerate_input() -> None:
    with pytest.raises(ar1.AR1Error):
        ar1.footprint_centred_span(1, 2)
    with pytest.raises(ar1.AR1Error):
        ar1.footprint_centred_span(8, 0)


def test_centred_shim_respans_only_the_two_render_axes_and_is_restored() -> None:
    saved = torch.linspace
    shim = ar1._centred_linspace_shim(6, 8, 2)
    lo_h, hi_h = ar1.footprint_centred_span(6, 2)
    assert torch.allclose(shim(-1.0, 1.0, 12), torch.linspace(lo_h, hi_h, 12))
    lo_w, hi_w = ar1.footprint_centred_span(8, 2)
    assert torch.allclose(shim(-1.0, 1.0, 16), torch.linspace(lo_w, hi_w, 16))
    # Any other call passes straight through, including a same-length non-(-1,1) span.
    assert torch.equal(shim(0.0, 5.0, 12), torch.linspace(0.0, 5.0, 12))
    assert torch.equal(shim(-1.0, 1.0, 7), torch.linspace(-1.0, 1.0, 7))
    assert torch.linspace is saved  # the shim never installs itself


def test_render_rgb_pair_restores_torch_linspace_even_when_the_model_raises() -> None:
    class _Boom:
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
            raise RuntimeError("render exploded")

    saved = torch.linspace
    with pytest.raises(RuntimeError, match="render exploded"):
        ar1.render_rgb_pair(_Boom(), 0, 2, ar1.LATTICE_CENTRED)  # type: ignore[arg-type]
    assert torch.linspace is saved


# ---------------------------------------------------------------------------
# footprint integral == exact ss x ss block mean
# ---------------------------------------------------------------------------
class _StubModel:
    """Minimal stand-in for ``QBFLOWTorch`` that returns a known deterministic field."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.calls: list[tuple[int, int]] = []

    def __call__(self, pair_ids: torch.Tensor, *, height: int, width: int) -> dict[str, torch.Tensor]:
        self.calls.append((height, width))
        generator = torch.Generator().manual_seed(int(pair_ids[0]) * 1000 + height)
        rgb = torch.rand(1, 2, 3, height, width, generator=generator)
        return {"rgb_pair_01": rgb}


@pytest.mark.parametrize("ss", [2, 3, 4])
def test_render_rgb_pair_is_the_exact_block_mean(monkeypatch: pytest.MonkeyPatch, ss: int) -> None:
    monkeypatch.setattr(ar1.qbt, "EVAL_H", 6)
    monkeypatch.setattr(ar1.qbt, "EVAL_W", 8)
    model = _StubModel(6, 8)
    coarse = ar1.render_rgb_pair(model, 3, ss)  # type: ignore[arg-type]
    assert model.calls == [(6 * ss, 8 * ss)]
    assert coarse.shape == (1, 2, 3, 6, 8)

    generator = torch.Generator().manual_seed(3 * 1000 + 6 * ss)
    fine = torch.rand(1, 2, 3, 6 * ss, 8 * ss, generator=generator)
    expected = fine.reshape(1, 2, 3, 6, ss, 8, ss).mean(dim=(4, 6))
    torch.testing.assert_close(coarse, expected, rtol=1e-6, atol=1e-6)


def test_render_rgb_pair_at_ss1_is_the_untouched_point_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ar1.qbt, "EVAL_H", 6)
    monkeypatch.setattr(ar1.qbt, "EVAL_W", 8)
    model = _StubModel(6, 8)
    coarse = ar1.render_rgb_pair(model, 3, 1)  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(3 * 1000 + 6)
    expected = torch.rand(1, 2, 3, 6, 8, generator=generator)
    assert torch.equal(coarse, expected)
    assert model.calls == [(6, 8)]


def test_render_rgb_pair_refuses_an_unknown_lattice() -> None:
    with pytest.raises(ar1.AR1Error):
        ar1.render_rgb_pair(_StubModel(4, 4), 0, 2, "nearest_neighbour")  # type: ignore[arg-type]


def test_render_rgb_pair_refuses_ss_below_one() -> None:
    with pytest.raises(ar1.AR1Error):
        ar1.render_rgb_pair(_StubModel(4, 4), 0, 0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# d_seg / d_pose arithmetic == the burn's own arithmetic
# ---------------------------------------------------------------------------
def test_d_seg_d_pose_matches_the_trainer_retain_eval_arithmetic() -> None:
    rng = np.random.default_rng(7)
    argmax = rng.integers(0, 5, size=(9, 11)).astype(np.uint8)
    target = rng.integers(0, 5, size=(9, 11)).astype(np.uint8)
    pose = rng.normal(size=6).astype(np.float32)
    target_pose = rng.normal(size=6).astype(np.float32)
    d_seg, d_pose = ar1.d_seg_d_pose(argmax, pose, target, target_pose)
    assert d_seg == pytest.approx(float(np.mean(argmax != target)))
    assert d_pose == pytest.approx(
        float(np.mean(np.square(pose.astype(np.float64) - target_pose)))
    )


def test_d_seg_is_zero_on_a_perfect_prediction() -> None:
    argmax = np.zeros((4, 4), dtype=np.uint8)
    pose = np.zeros(6, dtype=np.float32)
    assert ar1.d_seg_d_pose(argmax, pose, argmax, pose) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# B/H/W split
# ---------------------------------------------------------------------------
def test_bhw_split_counts_fixed_broken_and_net() -> None:
    target = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    base = np.array([[0, 2, 2, 1]], dtype=np.uint8)  # Road: 1 wrong; Lane: 1 wrong
    aa = np.array([[2, 0, 1, 1]], dtype=np.uint8)  # Road: broke one, fixed one; Lane: fixed one
    rows = {row["class_name"]: row for row in ar1.bhw_split(base, aa, target)}
    assert rows["Road"]["target_sites"] == 2
    assert rows["Road"]["fixed"] == 1 and rows["Road"]["broken"] == 1 and rows["Road"]["net"] == 0
    assert rows["Lane"]["target_sites"] == 2
    assert rows["Lane"]["fixed"] == 1 and rows["Lane"]["broken"] == 0 and rows["Lane"]["net"] == 1
    assert rows["Undrivable"]["target_sites"] == 0


def test_bhw_split_partitions_the_frame_and_reconciles_with_d_seg() -> None:
    rng = np.random.default_rng(11)
    target = rng.integers(0, 5, size=(16, 20)).astype(np.uint8)
    base = rng.integers(0, 5, size=(16, 20)).astype(np.uint8)
    aa = rng.integers(0, 5, size=(16, 20)).astype(np.uint8)
    rows = ar1.bhw_split(base, aa, target)
    assert sum(row["target_sites"] for row in rows) == target.size
    assert sum(row["base_wrong"] for row in rows) == int((base != target).sum())
    assert sum(row["aa_wrong"] for row in rows) == int((aa != target).sum())
    # net = fixed - broken must equal the drop in wrong sites, per class and in total.
    for row in rows:
        assert row["net"] == row["base_wrong"] - row["aa_wrong"]
    total_net = sum(row["net"] for row in rows)
    assert total_net / target.size == pytest.approx(
        ar1.d_seg_d_pose(base, np.zeros(6, np.float32), target, np.zeros(6, np.float32))[0]
        - ar1.d_seg_d_pose(aa, np.zeros(6, np.float32), target, np.zeros(6, np.float32))[0]
    )


def test_class_names_are_the_canonical_comma10k_order() -> None:
    assert ar1.CLASS_NAMES == ("Road", "Lane", "Undrivable", "Movable", "MyCar")


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def test_ht_mean_matches_the_burn_estimator() -> None:
    lookup = {pair_id: float(index) for index, pair_id in enumerate(qbt.SELECTION_IDS)}
    expected = (
        sum(
            weight * lookup[pair_id]
            for pair_id, weight in zip(qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS, strict=True)
        )
        / qbt.N
    )
    assert ar1._ht_mean(lookup) == pytest.approx(expected)


def test_ht_mean_refuses_a_pair_set_it_does_not_recognise() -> None:
    assert ar1._ht_mean({0: 1.0}) is None


def test_s_hat_is_the_contest_arithmetic() -> None:
    value = ar1._s_hat(0.002, 6.0e-4, 106643)
    assert value == pytest.approx(
        100.0 * 0.002 + float(np.sqrt(10.0 * 6.0e-4)) + 25.0 * 106643 / 37_545_489
    )


def test_rate_slope_is_the_published_exchange_rate() -> None:
    """25 / 37,545,489 S-per-byte is the exchange rate every byte claim is priced against."""
    assert pytest.approx(6.658589531221714e-7, rel=0, abs=1e-22) == 25.0 / qbt.RATE_DENOMINATOR


def test_mean_of_empty_is_nan_not_a_silent_zero() -> None:
    assert np.isnan(ar1._mean([]))


# ---------------------------------------------------------------------------
# subset aggregation
# ---------------------------------------------------------------------------
def _row(pair_id: int, ss: int, d_seg: float, d_pose: float, wall: float) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "ss": ss,
        "d_seg_dali_authority": d_seg,
        "d_pose_dali_authority": d_pose,
        "wall_total_s": wall,
        "wall_render_s": wall / 2.0,
    }


def test_subset_block_reports_direction_sign_census_and_byte_equivalent() -> None:
    """AA worse on two pairs, better on one: the block must say so in every field."""
    rows_by_ss = {
        1: {
            0: _row(0, 1, 0.010, 1.0e-4, 1.0),
            1: _row(1, 1, 0.020, 1.0e-4, 1.0),
            2: _row(2, 1, 0.030, 1.0e-4, 1.0),
        },
        2: {
            0: _row(0, 2, 0.020, 1.0e-4, 2.0),  # worse
            1: _row(1, 2, 0.040, 1.0e-4, 2.0),  # worse
            2: _row(2, 2, 0.015, 1.0e-4, 2.0),  # better
        },
    }
    block = ar1._subset_block(rows_by_ss, [0, 1, 2], 106_643, "dali_authority")
    assert block["n"] == 3
    base = block["per_ss"]["1"]
    aa = block["per_ss"]["2"]
    assert base["d_seg_mean"] == pytest.approx(0.020)
    assert aa["d_seg_mean"] == pytest.approx(0.025)

    delta = block["deltas_vs_ss1"]["2"]
    # ratio < 1 means the AA render is WORSE than the point-sampled baseline.
    assert delta["d_seg_ratio_base_over_aa"] == pytest.approx(0.020 / 0.025)
    assert delta["d_seg_delta"] == pytest.approx(0.005)
    assert delta["pairs_aa_worse"] == 2
    assert delta["pairs_aa_better"] == 1
    assert delta["pairs_unchanged"] == 0
    assert delta["d_seg_ratio_median_over_pairs"] == pytest.approx(0.5)
    assert delta["wall_cost_multiple"] == pytest.approx(2.0)
    # d_pose is unchanged here, so delta_S is the d_seg term alone: 100 * 0.005.
    assert delta["delta_S_mean_estimator"] == pytest.approx(0.5)
    assert delta["delta_S_equivalent_bytes"] == pytest.approx(0.5 / (25.0 / qbt.RATE_DENOMINATOR))


def test_subset_block_skips_an_ss_that_does_not_cover_the_subset() -> None:
    rows_by_ss = {
        1: {0: _row(0, 1, 0.01, 1e-4, 1.0), 1: _row(1, 1, 0.02, 1e-4, 1.0)},
        2: {0: _row(0, 2, 0.01, 1e-4, 2.0)},
    }
    block = ar1._subset_block(rows_by_ss, [0, 1], 106_643, "dali_authority")
    assert set(block["per_ss"]) == {"1"}
    assert block["deltas_vs_ss1"] == {}
