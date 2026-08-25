"""Tests for the ddm_ds1 cheap-to-shrink objective.

The load-bearing tests are the BINDING-VS-INERT proofs: `test_inert_returns_the_same
_object` and `test_enabled_changes_the_value_and_the_gradient`. Together they show the
lever cannot change a byte when off, and provably does change the descent when on.
An unfired lever is an orphan; a lever that cannot be shown to fire is worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_ds1_cheap_to_shrink import (  # noqa: E402
    DEFAULT_CONFIG,
    CheapToShrinkConfig,
    DS1Error,
    apply,
    derive_rung_ladder,
    derive_uniform_rung_ladder,
    is_inert,
    rungs_for_step,
    select_rung_for_step,
)

torch = pytest.importorskip("torch")


# ── binding-vs-inert proof ────────────────────────────────────────────────────


def test_default_config_is_inert() -> None:
    assert is_inert(DEFAULT_CONFIG)
    assert DEFAULT_CONFIG.mode == "off"


def test_inert_returns_the_same_object() -> None:
    """Byte-identity when off: the caller's own loss object comes back unchanged."""

    loss = torch.tensor(1.25, requires_grad=True)
    total, telemetry = apply(base_loss=loss, rung_losses=(), config=DEFAULT_CONFIG)
    assert total is loss
    assert telemetry["ds1_active"] is False
    assert telemetry["ds1_rungs_evaluated"] == 0


def test_inert_leaves_the_gradient_bit_identical() -> None:
    weight = torch.tensor([2.0], requires_grad=True)
    (weight.square().sum()).backward()
    reference = weight.grad.clone()

    weight2 = torch.tensor([2.0], requires_grad=True)
    total, _ = apply(base_loss=weight2.square().sum(), rung_losses=(), config=DEFAULT_CONFIG)
    total.backward()
    assert torch.equal(weight2.grad, reference)


def test_enabled_changes_the_value_and_the_gradient() -> None:
    """The lever provably BINDS: value and descent direction both move."""

    config = CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,))
    assert not is_inert(config)

    weight = torch.tensor([3.0], requires_grad=True)
    base = weight.square().sum()
    rung = (weight * 2.0).square().sum()
    total, telemetry = apply(base_loss=base, rung_losses=((0, rung),), config=config)
    total.backward()

    assert telemetry["ds1_active"] is True
    assert total.detach().item() == pytest.approx(9.0 + 36.0)
    # d/dw [w^2 + (2w)^2] = 2w + 8w = 10w = 30, versus 2w = 6 without the lever.
    assert float(weight.grad) == pytest.approx(30.0)


def test_zero_weight_rung_is_inert_even_when_declared() -> None:
    config = CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,), rung_weights=(0.0,))
    assert is_inert(config)
    loss = torch.tensor(1.0)
    total, _ = apply(base_loss=loss, rung_losses=(), config=config)
    assert total is loss


# ── config validation (fail-closed) ───────────────────────────────────────────


def test_unknown_mode_refuses() -> None:
    with pytest.raises(DS1Error, match="mode must be one of"):
        CheapToShrinkConfig(mode="matryoshka")


def test_off_mode_with_rungs_refuses() -> None:
    with pytest.raises(DS1Error, match="must declare no rungs"):
        CheapToShrinkConfig(mode="off", ceiling_multipliers=(2.0,))


def test_enabled_mode_without_rungs_refuses() -> None:
    with pytest.raises(DS1Error, match="requires at least one ceiling multiplier"):
        CheapToShrinkConfig(mode="sandwich")


def test_multiplier_at_or_below_one_refuses() -> None:
    """A multiplier <= 1 is not a cheaper rung; silently training on it is the bug."""

    with pytest.raises(DS1Error, match=r"must be > 1\.0"):
        CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(1.0,))
    with pytest.raises(DS1Error, match=r"must be > 1\.0"):
        CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(0.5,))


def test_unordered_multipliers_refuse() -> None:
    with pytest.raises(DS1Error, match="ordered cheapest-last"):
        CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(4.0, 2.0))


def test_duplicate_multipliers_refuse() -> None:
    with pytest.raises(DS1Error, match="distinct"):
        CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0, 2.0))


def test_weight_length_mismatch_refuses() -> None:
    with pytest.raises(DS1Error, match="length differs"):
        CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,), rung_weights=(1.0, 1.0))


def test_negative_weight_refuses() -> None:
    with pytest.raises(DS1Error, match="non-negative"):
        CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,), rung_weights=(-1.0,))


def test_reference_uniform_weighting_is_the_default() -> None:
    config = CheapToShrinkConfig(mode="all", ceiling_multipliers=(2.0, 4.0))
    assert config.weights == (1.0, 1.0)
    assert config.base_weight == 1.0


def test_uniform_ladder_requires_declared_uniform_rungs() -> None:
    with pytest.raises(DS1Error, match="requires at least one uniform bit rung"):
        CheapToShrinkConfig(mode="sampled", allocation_family="uniform_bits")


def test_uniform_ladder_refuses_waterfill_multipliers() -> None:
    with pytest.raises(DS1Error, match="may not declare waterfill"):
        CheapToShrinkConfig(
            mode="sampled",
            allocation_family="uniform_bits",
            uniform_bits=(3, 2),
            ceiling_multipliers=(2.0, 4.0),
        )


def test_uniform_ladder_is_real_byte_ordered() -> None:
    config = CheapToShrinkConfig(
        mode="sampled",
        allocation_family="uniform_bits",
        uniform_bits=(3, 2),
        seed=20260815,
    )
    costs = {"uniform4": 400, "uniform3": 320, "uniform2": 240}
    ladder = derive_uniform_rung_ladder(
        base_allocation="uniform4",
        allocation_for_bits=lambda bits: f"uniform{bits}",
        config=config,
        byte_cost=lambda allocation: costs[allocation],
    )
    assert ladder.cheaper_allocations == ("uniform3", "uniform2")
    assert ladder.diagnostics["rung_byte_savings"] == [80, 160]


def test_uniform_ladder_refuses_nonprogressive_real_bytes() -> None:
    config = CheapToShrinkConfig(
        mode="sampled",
        allocation_family="uniform_bits",
        uniform_bits=(3, 2),
    )
    with pytest.raises(DS1Error, match="not progressively cheaper"):
        derive_uniform_rung_ladder(
            base_allocation="uniform4",
            allocation_for_bits=lambda bits: f"uniform{bits}",
            config=config,
            byte_cost=lambda allocation: {"uniform4": 400, "uniform3": 250, "uniform2": 260}[allocation],
        )


# ── ladder derivation reuses the trainer's own waterfill ──────────────────────


def _fake_waterfill(*, maximum_predicted_error: float) -> str:
    return f"alloc@{maximum_predicted_error:.6f}"


def test_ladder_is_inert_when_config_is_inert() -> None:
    ladder = derive_rung_ladder(
        base_allocation="base",
        base_ceiling=1e-3,
        waterfill=_fake_waterfill,
        config=DEFAULT_CONFIG,
    )
    assert len(ladder) == 1
    assert ladder.cheaper_allocations == ()


def test_ladder_calls_waterfill_at_loosened_ceilings() -> None:
    config = CheapToShrinkConfig(mode="all", ceiling_multipliers=(2.0, 4.0))
    ladder = derive_rung_ladder(
        base_allocation="base",
        base_ceiling=1e-3,
        waterfill=_fake_waterfill,
        config=config,
    )
    assert len(ladder) == 3
    assert ladder.ceilings == (2e-3, 4e-3)
    assert ladder.cheaper_allocations == ("alloc@0.002000", "alloc@0.004000")


def test_ladder_refuses_a_non_positive_base_ceiling() -> None:
    with pytest.raises(DS1Error, match="must be positive"):
        derive_rung_ladder(
            base_allocation="base",
            base_ceiling=0.0,
            waterfill=_fake_waterfill,
            config=DEFAULT_CONFIG,
        )


def test_ladder_refuses_a_rung_that_saves_no_bytes() -> None:
    """A rung that is not cheaper teaches nothing; fail closed rather than train on it."""

    config = CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,))
    with pytest.raises(DS1Error, match="not cheaper than the shipped allocation"):
        derive_rung_ladder(
            base_allocation="base",
            base_ceiling=1e-3,
            waterfill=_fake_waterfill,
            config=config,
            byte_cost=lambda _allocation: 1000,
        )


def test_ladder_records_byte_savings_when_costs_are_available() -> None:
    config = CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,))
    costs = {"base": 30856, "alloc@0.002000": 27000}
    ladder = derive_rung_ladder(
        base_allocation="base",
        base_ceiling=1e-3,
        waterfill=_fake_waterfill,
        config=config,
        byte_cost=lambda allocation: costs[allocation],
    )
    assert ladder.diagnostics["base_bytes"] == 30856
    assert ladder.diagnostics["rung_byte_savings"] == [3856]


# ── stateless sampler (resumability P0) ───────────────────────────────────────


def test_sampler_is_none_when_inert() -> None:
    assert select_rung_for_step(DEFAULT_CONFIG, 0, 2) is None


def test_sandwich_always_takes_the_cheapest_rung() -> None:
    config = CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,))
    assert [select_rung_for_step(config, step, 1) for step in range(5)] == [0] * 5


def test_sampled_selection_is_deterministic_in_step() -> None:
    """Stateless by construction: resume at any step reproduces the same schedule."""

    config = CheapToShrinkConfig(mode="sampled", ceiling_multipliers=(2.0, 4.0), seed=7)
    first = [select_rung_for_step(config, step, 2) for step in range(64)]
    second = [select_rung_for_step(config, step, 2) for step in range(64)]
    assert first == second
    # Resuming mid-run reproduces the tail exactly.
    assert [select_rung_for_step(config, step, 2) for step in range(32, 64)] == first[32:]


def test_sampled_selection_depends_on_the_seed() -> None:
    a = CheapToShrinkConfig(mode="sampled", ceiling_multipliers=(2.0, 4.0), seed=1)
    b = CheapToShrinkConfig(mode="sampled", ceiling_multipliers=(2.0, 4.0), seed=2)
    assert [select_rung_for_step(a, s, 2) for s in range(64)] != [select_rung_for_step(b, s, 2) for s in range(64)]


def test_sampled_selection_covers_every_rung() -> None:
    config = CheapToShrinkConfig(mode="sampled", ceiling_multipliers=(2.0, 4.0, 8.0), seed=3)
    seen = {select_rung_for_step(config, step, 3) for step in range(256)}
    assert seen == {0, 1, 2}


def test_sampled_selection_refuses_a_negative_step() -> None:
    config = CheapToShrinkConfig(mode="sampled", ceiling_multipliers=(2.0,), seed=0)
    with pytest.raises(DS1Error, match="non-negative"):
        select_rung_for_step(config, -1, 1)


# ── apply() contract ──────────────────────────────────────────────────────────


def test_inert_apply_refuses_rung_losses() -> None:
    with pytest.raises(DS1Error, match="must not be given rung losses"):
        apply(base_loss=torch.tensor(1.0), rung_losses=((0, torch.tensor(1.0)),), config=DEFAULT_CONFIG)


def test_apply_refuses_an_out_of_range_rung_index() -> None:
    config = CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,))
    with pytest.raises(DS1Error, match="outside the declared ladder"):
        apply(base_loss=torch.tensor(1.0), rung_losses=((3, torch.tensor(1.0)),), config=config)


def test_sampled_mode_refuses_more_than_one_rung_per_step() -> None:
    config = CheapToShrinkConfig(mode="sampled", ceiling_multipliers=(2.0, 4.0))
    with pytest.raises(DS1Error, match="exactly one rung per step"):
        apply(
            base_loss=torch.tensor(1.0),
            rung_losses=((0, torch.tensor(1.0)), (1, torch.tensor(1.0))),
            config=config,
        )


def test_sampled_estimator_is_unbiased_for_the_full_sum() -> None:
    """FjORD's estimator: one rung per step, rescaled, matches the deterministic sum."""

    multipliers = (2.0, 4.0)
    deterministic_all = CheapToShrinkConfig(mode="all", ceiling_multipliers=multipliers)
    sampled = CheapToShrinkConfig(mode="sampled", ceiling_multipliers=multipliers, seed=11)

    base = torch.tensor(1.0)
    rungs = (torch.tensor(2.0), torch.tensor(5.0))
    deterministic = float(base) + sum(float(value) for value in rungs)

    total = 0.0
    trials = 20000
    for step in range(trials):
        index = select_rung_for_step(sampled, step, len(rungs))
        assert index is not None
        value, _ = apply(base_loss=base, rung_losses=((index, rungs[index]),), config=sampled)
        total += float(value)
    assert total / trials == pytest.approx(deterministic, rel=0.02)
    # The deterministic sandwich form over both rungs reproduces it exactly.
    exact, _ = apply(
        base_loss=base,
        rung_losses=tuple(enumerate(rungs)),
        config=deterministic_all,
    )
    assert float(exact) == pytest.approx(deterministic)


def test_base_weight_scales_the_shipped_rung() -> None:
    config = CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,), base_weight=0.5)
    total, _ = apply(base_loss=torch.tensor(4.0), rung_losses=((0, torch.tensor(3.0)),), config=config)
    assert float(total) == pytest.approx(0.5 * 4.0 + 3.0)


def test_provenance_declares_the_reference_form_and_inertness() -> None:
    provenance = DEFAULT_CONFIG.provenance()
    assert provenance["inert"] is True
    assert provenance["lever"] == "ddm_ds1_cheap_to_shrink"
    assert "matryoshka" in provenance["reference_form"]
    assert provenance["perturbation_operator"].endswith("exact_not_surrogate")
    assert len(provenance["reference_citations"]) == 3

    live = CheapToShrinkConfig(mode="sampled", ceiling_multipliers=(2.0,), seed=5).provenance()
    assert live["inert"] is False
    assert live["rung_weights"] == [1.0]


# ── mode='all' and the general rungs_for_step API ─────────────────────────────


def test_sandwich_refuses_more_than_one_rung() -> None:
    """The two-end form cannot carry middle rungs; refusing beats orphaning them."""

    with pytest.raises(DS1Error, match="admits exactly one cheaper rung"):
        CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0, 4.0))


def test_all_mode_returns_every_rung() -> None:
    config = CheapToShrinkConfig(mode="all", ceiling_multipliers=(2.0, 4.0, 8.0))
    assert rungs_for_step(config, 0, 3) == (0, 1, 2)
    assert rungs_for_step(config, 99, 3) == (0, 1, 2)


def test_rungs_for_step_is_empty_when_inert() -> None:
    assert rungs_for_step(DEFAULT_CONFIG, 0, 3) == ()


def test_rungs_for_step_wraps_the_single_rung_modes() -> None:
    sandwich = CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,))
    assert rungs_for_step(sandwich, 0, 1) == (0,)
    sampled = CheapToShrinkConfig(mode="sampled", ceiling_multipliers=(2.0, 4.0), seed=4)
    selected = rungs_for_step(sampled, 12, 2)
    assert len(selected) == 1
    assert selected[0] in (0, 1)


def test_select_rung_refuses_mode_all() -> None:
    config = CheapToShrinkConfig(mode="all", ceiling_multipliers=(2.0, 4.0))
    with pytest.raises(DS1Error, match="no single rung"):
        select_rung_for_step(config, 0, 2)


def test_all_mode_sums_every_rung_deterministically() -> None:
    config = CheapToShrinkConfig(mode="all", ceiling_multipliers=(2.0, 4.0))
    total, telemetry = apply(
        base_loss=torch.tensor(1.0),
        rung_losses=((0, torch.tensor(2.0)), (1, torch.tensor(5.0))),
        config=config,
    )
    assert total.item() == pytest.approx(8.0)
    assert telemetry["ds1_rungs_evaluated"] == 2
    assert telemetry["ds1_rung_scale"] == [1.0, 1.0]


def test_zero_base_weight_refuses() -> None:
    """base_weight=0 trains only the cheap rungs and abandons the shipped one."""

    with pytest.raises(DS1Error, match="abandons the SHIPPED allocation"):
        CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,), base_weight=0.0)


def test_ladder_records_whether_the_byte_check_ran() -> None:
    """A silently skipped cheaper-rung check must not look like a passed one."""

    config = CheapToShrinkConfig(mode="sandwich", ceiling_multipliers=(2.0,))
    unchecked = derive_rung_ladder(
        base_allocation="base",
        base_ceiling=1e-3,
        waterfill=_fake_waterfill,
        config=config,
    )
    assert unchecked.diagnostics["byte_cost_checked"] is False

    costs = {"base": 30856, "alloc@0.002000": 27000}
    checked = derive_rung_ladder(
        base_allocation="base",
        base_ceiling=1e-3,
        waterfill=_fake_waterfill,
        config=config,
        byte_cost=lambda allocation: costs[allocation],
    )
    assert checked.diagnostics["byte_cost_checked"] is True
