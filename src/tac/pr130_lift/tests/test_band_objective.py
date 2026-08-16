"""Tests for the debt-proportional band objective.

The two controls that make the term admissible:

1. ``weight=None`` DELEGATES to the lifted ``curriculum_loss`` -- bit-identical
   by construction, in all three curriculum phases.
2. ``weight=ones`` exercises the reimplemented PER-PIXEL path and must reproduce
   the oracle.  Its residual is MEASURED and reported, not assumed: the test
   asserts a bound and prints the realised value so a regression shows up as a
   number rather than a pass/fail.

Everything else guards the invariants that keep the term from confounding the
``ddm_lr1`` measurement: ``mean(weight) == 1`` for every alpha, scale neutrality
of the reducer, and the geometric identity of the band.
"""

from __future__ import annotations

import json

import pytest
import torch

from tac.pr130_lift.band_objective import (
    BAND_WEIGHT_TABLE_PATH,
    EDGE_PAIRS,
    N_SEMANTIC_CLASSES,
    BandObjectiveError,
    BandWeightTable,
    _oracle,
    band_weight_field,
    band_weight_stats,
    band_weight_table_sha256,
    curriculum_loss_weighted,
    label_boundary,
    load_band_weight_table,
    pair_debt_field,
)

TOTAL_STEPS = 600
CE_FRACTION = 0.50
SOFTPLUS_FRACTION = 0.85
# One step inside each of the three curriculum phases.
PHASE_STEPS = ((0, "ce"), (400, "softplus_margin"), (560, "expected_flip"))


@pytest.fixture(scope="module")
def table() -> BandWeightTable:
    return load_band_weight_table()


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260816)
    logits = torch.randn(2, N_SEMANTIC_CLASSES, 24, 32, generator=generator)
    target = torch.randint(
        0, N_SEMANTIC_CLASSES, (2, 24, 32), generator=generator, dtype=torch.long
    )
    return logits, target


# ---------------------------------------------------------------------------
# control 1: weight=None is the stock objective, exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("step", "expected_phase"), PHASE_STEPS)
def test_weight_none_is_bit_identical_to_oracle(batch, step, expected_phase):
    logits, target = batch
    oracle = _oracle()
    reference, reference_phase = oracle.curriculum_loss(
        logits, target, step, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION
    )
    value, phase = curriculum_loss_weighted(
        logits, target, step, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION, weight=None
    )
    assert phase == reference_phase == expected_phase
    assert value.item() == reference.item()
    assert torch.equal(value, reference)


def test_weight_none_default_argument_matches_explicit_none(batch):
    logits, target = batch
    positional = curriculum_loss_weighted(
        logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION
    )
    explicit = curriculum_loss_weighted(
        logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION, None
    )
    assert torch.equal(positional[0], explicit[0])
    assert positional[1] == explicit[1]


# ---------------------------------------------------------------------------
# control 2: the per-pixel path reproduces the oracle under a uniform weight
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("step", "expected_phase"), PHASE_STEPS)
def test_uniform_weight_reproduces_oracle(batch, step, expected_phase, capsys):
    logits, target = batch
    oracle = _oracle()
    reference, _ = oracle.curriculum_loss(
        logits, target, step, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION
    )
    ones = torch.ones_like(target, dtype=torch.float32)
    value, phase = curriculum_loss_weighted(
        logits, target, step, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION, ones
    )
    assert phase == expected_phase
    relative = abs(value.item() - reference.item()) / max(abs(reference.item()), 1e-12)
    # MEASURED, not assumed: print so a regression reads as a number.
    with capsys.disabled():
        print(
            f"\n[uniform-weight control] phase={phase} oracle={reference.item():.12e} "
            f"per_pixel={value.item():.12e} rel={relative:.3e}"
        )
    assert relative < 1e-6


def test_uniform_weight_scale_is_irrelevant(batch):
    """A weighted mean cannot be rescaled by the weight -- the lr-neutrality guard."""

    logits, target = batch
    base = torch.ones_like(target, dtype=torch.float32)
    one, _ = curriculum_loss_weighted(
        logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION, base
    )
    scaled, _ = curriculum_loss_weighted(
        logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION, base * 137.0
    )
    assert one.item() == pytest.approx(scaled.item(), rel=1e-12)


def test_weighted_mean_moves_toward_the_weighted_region(batch):
    """A non-uniform weight must actually change the loss, or the term is inert."""

    logits, target = batch
    weight = torch.ones_like(target, dtype=torch.float32)
    weight[:, :12, :] = 9.0
    uniform, _ = curriculum_loss_weighted(
        logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION,
        torch.ones_like(weight),
    )
    weighted, _ = curriculum_loss_weighted(
        logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION, weight
    )
    assert weighted.item() != uniform.item()


def test_negative_weight_refuses(batch):
    logits, target = batch
    weight = torch.ones_like(target, dtype=torch.float32)
    weight[0, 0, 0] = -1.0
    with pytest.raises(BandObjectiveError, match="non-negative"):
        curriculum_loss_weighted(
            logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION, weight
        )


def test_zero_weight_field_refuses(batch):
    logits, target = batch
    with pytest.raises(BandObjectiveError, match="sums to zero"):
        curriculum_loss_weighted(
            logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION,
            torch.zeros_like(target, dtype=torch.float32),
        )


def test_gradient_flows_through_the_weighted_loss(batch):
    logits, target = batch
    logits = logits.clone().requires_grad_(True)
    weight = torch.rand_like(target, dtype=torch.float32) + 0.5
    value, _ = curriculum_loss_weighted(
        logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION, weight
    )
    value.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# the band geometry
# ---------------------------------------------------------------------------
def test_label_boundary_matches_the_rt1_convention():
    """rt1 `boundary()`: BOTH sides of a 4-neighbour label change are ring 0."""

    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :, 2:] = 1
    band = label_boundary(target)
    expected = torch.zeros(1, 4, 4, dtype=torch.bool)
    expected[0, :, 1:3] = True
    assert torch.equal(band, expected)


def test_label_boundary_is_empty_on_a_constant_field():
    assert not label_boundary(torch.full((1, 5, 5), 3, dtype=torch.long)).any()


def test_label_boundary_refuses_wrong_rank():
    with pytest.raises(BandObjectiveError, match=r"\(B, H, W\)"):
        label_boundary(torch.zeros(4, 4, dtype=torch.long))


def test_debt_field_is_off_band_weight_away_from_the_boundary(table):
    target = torch.zeros(1, 6, 6, dtype=torch.long)
    target[0, :, 3:] = 1
    debt, band = pair_debt_field(target, table)
    assert torch.equal(band, label_boundary(target))
    assert float(debt[0, 0, 0]) == pytest.approx(table.off_band_weight)
    road_lane = table.weights_by_pair["0-1"]
    assert float(debt[0, 0, 2]) == pytest.approx(road_lane)
    assert float(debt[0, 0, 3]) == pytest.approx(road_lane)


def test_junction_pixel_takes_the_max_incident_weight(table):
    """The declared DOF, exercised: three classes meeting at one pixel."""

    target = torch.zeros(1, 3, 3, dtype=torch.long)
    target[0, 0, :] = 0
    target[0, 1, :] = 3  # Movable
    target[0, 2, :] = 1  # Lane
    debt, _ = pair_debt_field(target, table)
    weights = table.weights_by_pair
    expected = max(weights["0-3"], weights["1-3"])
    assert float(debt[0, 1, 1]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# the mean(w) == 1 invariant -- the thing that keeps lr uncofounded
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_weight_field_has_unit_mean_for_every_alpha(table, alpha):
    generator = torch.Generator().manual_seed(7)
    target = torch.randint(
        0, N_SEMANTIC_CLASSES, (3, 32, 40), generator=generator, dtype=torch.long
    )
    weight = band_weight_field(target, table, alpha)
    assert float(weight.mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(weight.min()) >= 0.0


def test_alpha_zero_is_exactly_uniform(table):
    generator = torch.Generator().manual_seed(11)
    target = torch.randint(
        0, N_SEMANTIC_CLASSES, (2, 16, 16), generator=generator, dtype=torch.long
    )
    weight = band_weight_field(target, table, 0.0)
    assert torch.equal(weight, torch.ones_like(weight))


def test_alpha_zero_loss_equals_the_stock_objective(table, batch):
    """alpha=0 must reproduce the trainer's own loss through the weighted path."""

    logits, target = batch
    weight = band_weight_field(target, table, 0.0)
    weighted, phase = curriculum_loss_weighted(
        logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION, weight
    )
    stock, stock_phase = curriculum_loss_weighted(
        logits, target, 400, TOTAL_STEPS, CE_FRACTION, SOFTPLUS_FRACTION, None
    )
    assert phase == stock_phase
    assert weighted.item() == pytest.approx(stock.item(), rel=1e-6)


def test_alpha_one_concentrates_weight_on_the_band(table):
    generator = torch.Generator().manual_seed(13)
    target = torch.randint(
        0, N_SEMANTIC_CLASSES, (2, 32, 32), generator=generator, dtype=torch.long
    )
    weight = band_weight_field(target, table, 1.0)
    band = label_boundary(target)
    assert float(weight[band].mean()) > float(weight[~band].mean())


def test_alpha_out_of_range_refuses(table):
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    for alpha in (-0.1, 1.1):
        with pytest.raises(BandObjectiveError, match=r"\[0, 1\]"):
            band_weight_field(target, table, alpha)


def test_constant_field_refuses_rather_than_dividing_by_zero(table):
    """No band and a zero off-band weight would be a silent 0/0."""

    zero_off = BandWeightTable(
        basis=table.basis,
        flips_by_pair=dict(table.flips_by_pair),
        band_px_by_pair=dict(table.band_px_by_pair),
        off_band_flips=0,
        off_band_px=table.off_band_px,
        provenance={},
    )
    with pytest.raises(BandObjectiveError, match="non-positive mean"):
        band_weight_field(torch.zeros(1, 4, 4, dtype=torch.long), zero_off, 1.0)


# ---------------------------------------------------------------------------
# the committed table
# ---------------------------------------------------------------------------
def test_committed_table_reproduces_the_rt1_receipts(table):
    provenance = table.provenance
    # rt1 RT1_GEOMETRY.json::ring_population[0]
    assert provenance["band_px_ring0"] == 2_551_464
    # rt1's measured round trip: render argmax vs shipped labels
    assert provenance["total_flips"] == 33_743
    # rt1's headline: 99.22% of the seg axis is ON the boundary
    assert provenance["on_band_flip_share"] == pytest.approx(0.9922, abs=5e-5)
    assert provenance["on_band_flips"] + table.off_band_flips == provenance["total_flips"]
    assert provenance["instrument_control_free_band_mask_parity"] is True
    assert provenance["total_px"] == 600 * 384 * 512
    assert table.off_band_px == 600 * 384 * 512 - 2_551_464


def test_committed_table_confusion_cross_check_reproduces_rt1_shape(table):
    """rc2 section 2.1 worked Road<->Lane = 43.4% from the retained confusion."""

    confusion = table.provenance["confusion_by_pair_cross_check"]
    total = sum(confusion.values())
    assert total == 33_743
    assert confusion["0-1"] / total == pytest.approx(0.4399, abs=5e-4)


def test_committed_table_is_debt_density_not_flip_share(table):
    """The measured point of the rule: length and debt rank DIFFERENTLY."""

    weights = table.weights_by_pair
    flips = table.flips_by_pair
    # Road/Lane carries the most flips ...
    assert max(flips, key=lambda k: flips[k]) == "0-1"
    # ... but is NOT the densest debt, because its edge is the longest.
    assert weights["0-1"] < weights["2-3"]
    assert weights["0-1"] < weights["3-4"]


def test_committed_table_has_no_band_pixels_for_the_zero_weight_pair(table):
    """A zero weight may never be selectable at runtime."""

    for key, weight in table.weights_by_pair.items():
        if weight == 0.0:
            assert table.band_px_by_pair[key] == 0


def test_lookup_tensor_is_symmetric_with_zero_diagonal(table):
    lut = table.lookup_tensor()
    assert lut.shape == (N_SEMANTIC_CLASSES, N_SEMANTIC_CLASSES)
    assert torch.equal(lut, lut.T)
    assert float(lut.diagonal().abs().sum()) == 0.0
    for class_a, class_b in EDGE_PAIRS:
        assert float(lut[class_a, class_b]) == pytest.approx(
            table.weights_by_pair[f"{class_a}-{class_b}"]
        )


def test_table_round_trips_through_json(table):
    restored = BandWeightTable.from_dict(json.loads(json.dumps(table.to_dict())))
    assert restored.weights_by_pair == table.weights_by_pair
    assert restored.off_band_weight == table.off_band_weight
    assert restored.basis == table.basis


def test_table_refuses_a_foreign_schema(table):
    payload = table.to_dict()
    payload["schema"] = "something.else.v1"
    with pytest.raises(BandObjectiveError, match="schema"):
        BandWeightTable.from_dict(payload)


def test_table_refuses_incomplete_pair_coverage(table):
    payload = table.to_dict()
    del payload["flips_by_pair"]["0-1"]
    with pytest.raises(BandObjectiveError, match="exactly the 10 unordered pairs"):
        BandWeightTable.from_dict(payload)


def test_table_sha_is_stable_and_matches_the_file(table):
    digest = band_weight_table_sha256()
    assert len(digest) == 64
    assert digest == band_weight_table_sha256(BAND_WEIGHT_TABLE_PATH)


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------
def test_band_weight_stats_reports_the_concentration(table):
    generator = torch.Generator().manual_seed(17)
    target = torch.randint(
        0, N_SEMANTIC_CLASSES, (2, 24, 24), generator=generator, dtype=torch.long
    )
    weight = band_weight_field(target, table, 1.0)
    stats = band_weight_stats(weight, label_boundary(target))
    assert stats["mean"] == pytest.approx(1.0, abs=1e-5)
    assert 0.0 < stats["band_fraction"] <= 1.0
    assert stats["band_mean"] > stats["off_band_mean"]
    assert 0.0 < stats["band_weight_mass_fraction"] <= 1.0


def test_band_weight_stats_without_a_mask_is_still_valid(table):
    weight = torch.ones(4, 4)
    stats = band_weight_stats(weight)
    assert stats["mean"] == pytest.approx(1.0)
    assert "band_px" not in stats
