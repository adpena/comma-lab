"""Tests for the B2E train-for-editability levers.

These verify BEHAVIOUR, not constants: every test would fail if the lever body
were replaced by a marker-returning stub.  The two load-bearing ones are

* :func:`test_deployed_fake_quant_matches_shipped_quantizer` -- F2 must train on
  the EXACT grid the deployed packer uses, or the whole regime is a proxy; and
* :func:`test_inactive_config_draws_no_randomness` -- the byte-identity contract.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.pr130_lift.editability_levers import (  # noqa: E402
    DEFAULT_FILM_CRITICAL_MULTIPLIER,
    FILM_ROW_FAMILY,
    POSE_CRITICAL_TENSORS,
    SELECTED_MIXED_Q3_NAMES,
    EditabilityLeverConfig,
    EditabilityLevers,
    LeverError,
    deployed_fake_quant,
    film_row_order,
    mixed_bit_allocation,
)


def _toy_model() -> torch.nn.Module:
    """A module whose parameter names mirror the SemanticTokenRenderer families."""
    model = torch.nn.Module()
    model.add_module("blocks", torch.nn.Module())
    for index in (1, 2, 3):
        block = torch.nn.Module()
        block.register_parameter(
            "weight", torch.nn.Parameter(torch.randn(8, 4, generator=_gen()))
        )
        film = torch.nn.Module()
        film.register_parameter(
            "weight", torch.nn.Parameter(torch.randn(8, 4, generator=_gen()))
        )
        film.register_parameter("bias", torch.nn.Parameter(torch.randn(8, generator=_gen())))
        block.add_module("film", film)
        model.blocks.add_module(str(index), block)
    frame = torch.nn.Module()
    frame.register_parameter(
        "weight", torch.nn.Parameter(torch.randn(16, 6, generator=_gen()))
    )
    model.add_module("frame_embed", frame)
    return model


def _gen() -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(4242)
    return generator


# ---------------------------------------------------------------------------
# Contract: the pinned edit sets still match the shipped definitions.
# ---------------------------------------------------------------------------


def test_constants_match_shipped_sets() -> None:
    """If the shipped edit set moves, this fails loudly instead of drifting."""
    sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")
    assert SELECTED_MIXED_Q3_NAMES == frozenset(sm3.SELECTED_MIXED_Q3_NAMES)
    assert FILM_ROW_FAMILY == frozenset(sm3.PRUNE_NAMES)
    assert POSE_CRITICAL_TENSORS == FILM_ROW_FAMILY


def test_mixed_bit_allocation_matches_mp2_rule() -> None:
    names = ["frame_embed.weight", "blocks.1.film.weight", "blocks.0.dw.weight"]
    allocation = mixed_bit_allocation(names)
    assert allocation["frame_embed.weight"] == 3
    assert allocation["blocks.1.film.weight"] == 3
    assert allocation["blocks.0.dw.weight"] == 4


def test_mixed_bit_allocation_rejects_out_of_range_depth() -> None:
    with pytest.raises(LeverError):
        mixed_bit_allocation(["a.weight"], low_bits=1)


# ---------------------------------------------------------------------------
# Contract: F2 trains on the DEPLOYED grid.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize(
    "name,shape",
    [
        ("blocks.1.film.weight", (12, 5)),
        ("frame_embed.weight", (20, 8)),
        ("blocks.0.dw.weight", (6, 1, 3, 3)),
    ],
)
def test_deployed_fake_quant_matches_shipped_quantizer(
    name: str, shape: tuple[int, ...], bits: int
) -> None:
    """Forward output must equal ``sd1.quantized_tensor``'s restored tensor."""
    sd1 = importlib.import_module("experiments.ddm_sd1_semantic_rd_curve")
    generator = torch.Generator()
    generator.manual_seed(11)
    value = torch.randn(shape, generator=generator)

    expected, _ = sd1.quantized_tensor(name, value, bits)
    actual = deployed_fake_quant(name, value, bits)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_deployed_fake_quant_is_straight_through() -> None:
    value = torch.nn.Parameter(torch.randn(6, 3))
    out = deployed_fake_quant("blocks.1.film.weight", value, 4)
    out.sum().backward()
    assert value.grad is not None
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_deployed_fake_quant_rejects_bad_bits() -> None:
    with pytest.raises(LeverError):
        deployed_fake_quant("a.weight", torch.randn(4, 4), 9)


def test_deployed_fake_quant_1d_uses_fp16_path() -> None:
    value = torch.tensor([0.1234567, -2.5])
    out = deployed_fake_quant("blocks.1.film.bias", value, 4)
    torch.testing.assert_close(out, value.to(torch.float16).float())


# ---------------------------------------------------------------------------
# Contract: row ordering matches the shipped keep-ladder.
# ---------------------------------------------------------------------------


def test_film_row_order_is_descending_squared_l2() -> None:
    value = torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
    assert film_row_order(value) == [1, 2, 0]


def test_film_row_order_breaks_ties_by_index() -> None:
    value = torch.ones(4, 3)
    assert film_row_order(value) == [0, 1, 2, 3]


def test_film_row_order_rejects_vectors() -> None:
    with pytest.raises(LeverError):
        film_row_order(torch.ones(5))


# ---------------------------------------------------------------------------
# Contract: default is inert (the byte-identity guarantee).
# ---------------------------------------------------------------------------


def test_default_config_is_inert() -> None:
    config = EditabilityLeverConfig()
    assert not config.any_active
    assert not config.f1_active
    assert not config.f2_active
    assert not config.f3_active
    assert not config.f4_active


def test_inactive_levers_leave_parameters_untouched() -> None:
    model = _toy_model()
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    levers = EditabilityLevers(EditabilityLeverConfig())
    with levers.applied(model):
        during = {k: v.detach().clone() for k, v in model.named_parameters()}
    for key, value in before.items():
        torch.testing.assert_close(during[key], value)
        torch.testing.assert_close(dict(model.named_parameters())[key].detach(), value)
    assert levers.steps_applied == 0


def test_inactive_config_draws_no_randomness() -> None:
    """An all-off run must consume the identical global RNG stream."""
    model = _toy_model()
    levers = EditabilityLevers(EditabilityLeverConfig())

    torch.manual_seed(7)
    reference = torch.randn(5)

    torch.manual_seed(7)
    with levers.applied(model):
        pass
    after = torch.randn(5)

    torch.testing.assert_close(after, reference)


def test_inactive_rank_penalty_is_exactly_zero_and_graph_free() -> None:
    model = _toy_model()
    levers = EditabilityLevers(EditabilityLeverConfig())
    penalty = levers.rank_penalty(model)
    assert penalty.item() == 0.0
    assert not penalty.requires_grad


def test_active_levers_do_not_disturb_the_global_rng_stream() -> None:
    """Active levers use a dedicated generator, so an A/B stays a clean 2x2."""
    model = _toy_model()
    levers = EditabilityLevers(
        EditabilityLeverConfig(weight_perturb_robustness=0.5, film_row_dropout=0.25)
    )

    torch.manual_seed(7)
    reference = torch.randn(5)

    torch.manual_seed(7)
    with levers.applied(model):
        pass
    after = torch.randn(5)

    torch.testing.assert_close(after, reference)


# ---------------------------------------------------------------------------
# Contract: active levers actually do the work they name.
# ---------------------------------------------------------------------------


def test_f1_perturbs_and_upweights_the_pose_critical_subspace() -> None:
    config = EditabilityLeverConfig(weight_perturb_robustness=0.25)
    levers = EditabilityLevers(config)
    generator = torch.Generator()
    generator.manual_seed(3)
    value = torch.nn.Parameter(torch.randn(64, 16, generator=generator))

    film = levers.transform("blocks.1.film.weight", value)
    plain = levers.transform("blocks.0.dw.weight", value)

    film_shift = (film - value).abs().mean().item()
    plain_shift = (plain - value).abs().mean().item()
    assert film_shift > 0.0 and plain_shift > 0.0
    # The FiLM family must be perturbed materially harder (~sqrt(93.7)x).
    ratio = film_shift / plain_shift
    assert ratio > 5.0, ratio
    assert DEFAULT_FILM_CRITICAL_MULTIPLIER > 9.0


def test_f1_is_straight_through_for_gradients() -> None:
    levers = EditabilityLevers(
        EditabilityLeverConfig(weight_perturb_robustness=0.5)
    )
    value = torch.nn.Parameter(torch.randn(8, 4))
    levers.transform("blocks.1.film.weight", value).sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_f2_quantizes_to_the_mixed_map() -> None:
    levers = EditabilityLevers(EditabilityLeverConfig(weight_qat_q3q4=True))
    generator = torch.Generator()
    generator.manual_seed(5)
    value = torch.randn(32, 8, generator=generator)

    q3 = levers.transform("blocks.1.film.weight", value)
    q4 = levers.transform("blocks.0.dw.weight", value)

    torch.testing.assert_close(q3, deployed_fake_quant("blocks.1.film.weight", value, 3))
    torch.testing.assert_close(q4, deployed_fake_quant("blocks.0.dw.weight", value, 4))
    # 3-bit is strictly coarser, so it must sit further from the original.
    assert (q3 - value).abs().mean() > (q4 - value).abs().mean()


def test_f3_drops_whole_rows_only_in_the_film_family() -> None:
    levers = EditabilityLevers(EditabilityLeverConfig(film_row_dropout=0.5))
    value = torch.ones(64, 5)

    dropped = levers.transform("blocks.2.film.weight", value)
    untouched = levers.transform("blocks.0.pw.weight", value)

    torch.testing.assert_close(untouched, value)
    row_values = dropped.abs().sum(dim=1)
    zero_rows = int((row_values == 0).sum())
    assert 0 < zero_rows < 64, zero_rows
    # Whole rows, never partial ones.
    for row in dropped:
        assert row.eq(0).all() or row.ne(0).all()


def test_f3_protects_the_top_rows() -> None:
    levers = EditabilityLevers(
        EditabilityLeverConfig(film_row_dropout=0.99, film_row_dropout_protect_top=3)
    )
    value = torch.arange(1.0, 41.0).reshape(8, 5)
    protected = film_row_order(value)[:3]
    dropped = levers.transform("blocks.3.film.weight", value)
    for index in protected:
        assert dropped[index].ne(0).all()


def test_f4_penalizes_spectral_spread() -> None:
    config = EditabilityLeverConfig(
        carrier_rank_penalty=1.0, carrier_tensors=("carrier.weight",)
    )
    levers = EditabilityLevers(config)

    model = torch.nn.Module()
    carrier = torch.nn.Module()
    # Rank-1: nuclear/frobenius ratio == 1, the minimum.
    carrier.register_parameter(
        "weight", torch.nn.Parameter(torch.outer(torch.arange(1.0, 9.0), torch.ones(4)))
    )
    model.add_module("carrier", carrier)
    low = levers.rank_penalty(model)

    full = torch.nn.Module()
    full_carrier = torch.nn.Module()
    full_carrier.register_parameter("weight", torch.nn.Parameter(torch.eye(8, 4)))
    full.add_module("carrier", full_carrier)
    high = levers.rank_penalty(full)

    assert high.item() > low.item()
    assert low.item() == pytest.approx(1.0, abs=1e-4)


def test_f4_requires_named_tensors_to_exist() -> None:
    levers = EditabilityLevers(
        EditabilityLeverConfig(carrier_rank_penalty=1.0, carrier_tensors=("nope.weight",))
    )
    with pytest.raises(LeverError):
        levers.rank_penalty(_toy_model())


# ---------------------------------------------------------------------------
# Contract: application is reversible, even on failure.
# ---------------------------------------------------------------------------


def test_applied_restores_parameters_after_exception() -> None:
    model = _toy_model()
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    levers = EditabilityLevers(EditabilityLeverConfig(weight_qat_q3q4=True))

    with pytest.raises(ValueError):
        with levers.applied(model):
            raise ValueError("boom")

    after = dict(model.named_parameters())
    assert set(after) == set(before)
    for key, value in before.items():
        assert isinstance(after[key], torch.nn.Parameter)
        torch.testing.assert_close(after[key].detach(), value)


def test_applied_changes_forward_values_while_active() -> None:
    model = _toy_model()
    levers = EditabilityLevers(EditabilityLeverConfig(weight_qat_q3q4=True))
    original = model.blocks._modules["1"].film.weight.detach().clone()
    with levers.applied(model):
        inside = model.blocks._modules["1"].film.weight.detach().clone()
    assert not torch.equal(inside, original)
    torch.testing.assert_close(
        model.blocks._modules["1"].film.weight.detach(), original
    )
    assert levers.steps_applied == 1


def test_applied_does_not_disturb_buffers() -> None:
    """Review-pass regression: the parameter swap must not touch ``_buffers``."""
    model = torch.nn.Module()
    blocks = torch.nn.Module()
    block = torch.nn.Module()
    film = torch.nn.Module()
    film.register_parameter("weight", torch.nn.Parameter(torch.randn(8, 4)))
    film.register_buffer("running", torch.arange(8.0))
    block.add_module("film", film)
    blocks.add_module("1", block)
    model.add_module("blocks", blocks)

    buffer_before = film.running.clone()
    levers = EditabilityLevers(EditabilityLeverConfig(weight_qat_q3q4=True))
    with levers.applied(model):
        torch.testing.assert_close(film.running, buffer_before)
    torch.testing.assert_close(film.running, buffer_before)


def test_applied_restores_the_parameter_type_and_clean_state_dict() -> None:
    """Review-pass regression: exiting must leave real Parameters and a clean state dict."""
    model = _toy_model()
    keys_before = set(model.state_dict())
    levers = EditabilityLevers(EditabilityLeverConfig(weight_qat_q3q4=True))
    with levers.applied(model):
        pass
    for name, param in model.named_parameters():
        assert isinstance(param, torch.nn.Parameter), name
    assert set(model.state_dict()) == keys_before


def test_nested_applied_scopes_restore_cleanly() -> None:
    """Review-pass regression: re-entrant use must not orphan a swapped tensor."""
    model = _toy_model()
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    levers = EditabilityLevers(EditabilityLeverConfig(weight_qat_q3q4=True))
    with levers.applied(model):
        with levers.applied(model):
            pass
    for key, value in before.items():
        param = dict(model.named_parameters())[key]
        assert isinstance(param, torch.nn.Parameter)
        torch.testing.assert_close(param.detach(), value)


def test_gradients_reach_the_original_parameters_through_applied() -> None:
    model = _toy_model()
    levers = EditabilityLevers(
        EditabilityLeverConfig(weight_perturb_robustness=0.1, weight_qat_q3q4=True)
    )
    with levers.applied(model):
        model.blocks._modules["1"].film.weight.sum().backward()
    grad = dict(model.named_parameters())["blocks.1.film.weight"].grad
    assert grad is not None
    assert torch.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Contract: configuration is validated, and "off" is a tracked, reasoned state.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_perturb_robustness": -0.1},
        {"weight_perturb_shape": "triangular"},
        {"film_row_dropout": 1.0},
        {"film_row_dropout": -0.1},
        {"carrier_rank_penalty": -1.0},
        {"weight_qat_low_bits": 1},
        {"film_row_dropout_protect_top": -1},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(LeverError):
        EditabilityLeverConfig(**kwargs)


def test_activation_ledger_reports_every_lever_with_a_reason() -> None:
    ledger = EditabilityLeverConfig().activation_ledger()
    assert ledger["any_active"] is False
    levers = ledger["levers"]
    assert set(levers) == {
        "F1_weight_perturb_robustness",
        "F2_weight_qat_q3q4",
        "F3_film_row_dropout",
        "F4_carrier_rank_penalty",
        "F5_gate_aware_conditioning",
    }
    for name, row in levers.items():
        assert row["active"] is False
        assert row["reason_if_off"], f"{name} is off without a recorded reason"
    assert levers["F5_gate_aware_conditioning"]["state"] == "DECLARED_UNBUILT_FOLLOW_ON"


def test_activation_ledger_clears_reason_when_active() -> None:
    ledger = EditabilityLeverConfig(weight_qat_q3q4=True).activation_ledger()
    assert ledger["any_active"] is True
    assert ledger["levers"]["F2_weight_qat_q3q4"]["active"] is True
    assert ledger["levers"]["F2_weight_qat_q3q4"]["reason_if_off"] is None


# ---------------------------------------------------------------------------
# ddm_ren2: F2's low-bit tensor set is an INPUT, sourced from the packed bytes.
#
# The default set mirrors mp2's MODE_ROW_PRUNE candidate family.  The object that
# SHIPS at move 48 is packed by ``pack_prune_mixed_candidate`` and its measured depth
# table is ``sm3.PRUNE_MIXED_Q3_NAMES`` -- a different set.  These tests verify that
# (a) the default is unchanged, so every pre-existing config stays byte-identical, and
# (b) an explicit set actually changes which tensor is coarsened, which is the whole
# point: a flag and a constant must not be able to disagree silently.
# ---------------------------------------------------------------------------


def test_effective_q3_names_defaults_to_the_mp2_set() -> None:
    assert EditabilityLeverConfig().effective_q3_names == SELECTED_MIXED_Q3_NAMES
    assert (
        EditabilityLeverConfig(weight_qat_q3q4=True).effective_q3_names
        == SELECTED_MIXED_Q3_NAMES
    )


def test_explicit_q3_names_change_the_realized_depth() -> None:
    """The shipped table coarsens blocks.0.film and leaves blocks.1.film at q4."""
    shipped = ("blocks.0.film.weight", "frame_embed.weight")
    levers = EditabilityLevers(
        EditabilityLeverConfig(weight_qat_q3q4=True, weight_qat_q3_names=shipped)
    )
    generator = torch.Generator()
    generator.manual_seed(11)
    value = torch.randn(32, 8, generator=generator)

    block0 = levers.transform("blocks.0.film.weight", value)
    block1 = levers.transform("blocks.1.film.weight", value)

    torch.testing.assert_close(
        block0, deployed_fake_quant("blocks.0.film.weight", value, 3)
    )
    torch.testing.assert_close(
        block1, deployed_fake_quant("blocks.1.film.weight", value, 4)
    )
    # Under the DEFAULT set the two depths are swapped; if the override were inert
    # this assertion would fail, so the test cannot pass on a stub.
    assert (block0 - value).abs().mean() > (block1 - value).abs().mean()


def test_explicit_q3_names_flow_through_parameter_overrides() -> None:
    """The primary integration point honours the override, not just ``transform``."""
    shipped = ("blocks.0.film.weight", "frame_embed.weight")
    model = _toy_model()
    default_overrides = EditabilityLevers(
        EditabilityLeverConfig(weight_qat_q3q4=True)
    ).parameter_overrides(model, base_bits=4)
    shipped_overrides = EditabilityLevers(
        EditabilityLeverConfig(weight_qat_q3q4=True, weight_qat_q3_names=shipped)
    ).parameter_overrides(model, base_bits=4)
    name = "blocks.1.film.weight"
    assert name in default_overrides and name in shipped_overrides
    assert not torch.equal(default_overrides[name], shipped_overrides[name])


def test_explicit_q3_names_reject_degenerate_inputs() -> None:
    for bad in ((), ("a", "a"), ("",), ["a"]):
        with pytest.raises(LeverError):
            EditabilityLeverConfig(weight_qat_q3_names=bad)  # type: ignore[arg-type]


def test_activation_ledger_reports_the_effective_q3_set() -> None:
    shipped = ("blocks.0.film.weight", "frame_embed.weight")
    ledger = EditabilityLeverConfig(
        weight_qat_q3q4=True, weight_qat_q3_names=shipped
    ).activation_ledger()["levers"]["F2_weight_qat_q3q4"]
    assert ledger["q3_names"] == sorted(shipped)
    assert ledger["q3_names_are_default"] is False
    assert ledger["q3_names_default"] == sorted(SELECTED_MIXED_Q3_NAMES)
    default_ledger = EditabilityLeverConfig(weight_qat_q3q4=True).activation_ledger()
    assert default_ledger["levers"]["F2_weight_qat_q3q4"]["q3_names_are_default"] is True


def test_shipped_depth_table_differs_from_the_default_set() -> None:
    """The measured reason this input exists.  If mp2's sets ever converge, say so."""
    sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")
    assert frozenset(sm3.PRUNE_MIXED_Q3_NAMES) != frozenset(sm3.SELECTED_MIXED_Q3_NAMES)
    assert frozenset(sm3.PRUNE_MIXED_Q3_NAMES) == {
        "frame_embed.weight",
        "blocks.0.film.weight",
    }


def test_f3_row_dropout_degenerates_to_a_constant_gain_on_a_pruned_tensor() -> None:
    """ddm_ren2, MEASURED on the shipped renderer: F3 is inert-but-not-harmless here.

    The deployed object row-prunes ``blocks.{1,2,3}.film.weight`` to 2 of 192 rows, so
    190 rows are EXACTLY zero.  Dropping a zero row is a no-op, and ``protect_top=2``
    selects precisely the two surviving rows -- so every draw keeps the same rows and
    the inverted-dropout rescale ``1/(1-p)`` lands on them DETERMINISTICALLY.  F3 then
    carries no stochastic content at all; it is a constant multiplicative gain that the
    receiver does not apply, i.e. exactly the trained-vs-realized divergence F3 exists
    to remove, reintroduced by a different route.

    This test fails if the rescale is ever made conditional on a row actually being
    dropped, which is the cure -- so it is a live guard, not a monument.
    """
    value = torch.zeros(192, 8)
    value[11] = torch.arange(1.0, 9.0)
    value[13] = torch.arange(2.0, 10.0)
    assert film_row_order(value)[:2] == [13, 11]
    live = [11, 13]
    for probability in (0.1, 0.5, 0.9):
        levers = EditabilityLevers(
            EditabilityLeverConfig(
                film_row_dropout=probability, film_row_dropout_protect_top=2
            )
        )
        for _ in range(4):
            out = levers.transform("blocks.1.film.weight", value)
            gain = out[live] / value[live]
            torch.testing.assert_close(
                gain, torch.full_like(gain, 1.0 / (1.0 - probability))
            )
            assert float(out[[i for i in range(192) if i not in live]].abs().sum()) == 0.0
