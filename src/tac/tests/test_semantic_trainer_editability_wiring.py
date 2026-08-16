"""Tests for the B2E lever wiring inside the semantic QAT trainer.

The load-bearing property is that the lever flags are ADDITIVE: a run with every
lever off must be indistinguishable from the pre-lever trainer, and a resume
checkpoint written before the flags existed must still resume -- but only while
the levers stay inert.  If a lever is ON against a pre-lever parent, the run
genuinely differs and the guard must refuse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.pr130_lift.editability_levers import (  # noqa: E402
    EditabilityLeverConfig,
    EditabilityLevers,
    deployed_fake_quant,
)
from tac.pr130_lift.train_semantic_quantized_resumable import (  # noqa: E402
    ADDITIVE_LEVER_CONFIG_DEFAULTS,
    _reconcile_additive_resume_config,
    _stable_training_config,
    parse_args,
)

BASE_ARGV = [
    "--challenge-root", "cr",
    "--init", "init.pt",
    "--out", "out",
    "--save", "save",
    "--cache", "cache.pt",
]


# ---------------------------------------------------------------------------
# CLI contract: every lever flag exists and defaults inert.
# ---------------------------------------------------------------------------


def test_every_lever_flag_parses_and_defaults_inert() -> None:
    args = parse_args(BASE_ARGV)
    assert args.weight_perturb_robustness == 0.0
    assert args.weight_perturb_shape == "quantization"
    assert args.weight_qat_q3q4 is False
    assert args.film_row_dropout == 0.0
    assert args.film_row_dropout_protect_top == 0
    assert args.carrier_rank_penalty == 0.0
    assert args.carrier_tensors == []


def test_default_args_build_an_inert_lever_config() -> None:
    args = parse_args(BASE_ARGV)
    config = EditabilityLeverConfig(
        weight_perturb_robustness=args.weight_perturb_robustness,
        weight_perturb_shape=args.weight_perturb_shape,
        film_critical_multiplier=args.film_critical_multiplier,
        weight_qat_q3q4=args.weight_qat_q3q4,
        weight_qat_high_bits=args.bits,
        film_row_dropout=args.film_row_dropout,
        film_row_dropout_protect_top=args.film_row_dropout_protect_top,
        carrier_rank_penalty=args.carrier_rank_penalty,
        carrier_tensors=tuple(args.carrier_tensors),
        seed=args.lever_seed,
    )
    assert not config.any_active


def test_lever_keys_are_causal_config() -> None:
    """Levers change training, so they must be inside the causal config."""
    stable = _stable_training_config(parse_args(BASE_ARGV))
    for key in ADDITIVE_LEVER_CONFIG_DEFAULTS:
        assert key in stable, key


def test_additive_defaults_match_the_argparse_defaults() -> None:
    """The declared inert values must be the real defaults, or resume drifts."""
    args = parse_args(BASE_ARGV)
    stable = _stable_training_config(args)
    for key, inert in ADDITIVE_LEVER_CONFIG_DEFAULTS.items():
        assert stable[key] == inert, key


# ---------------------------------------------------------------------------
# The additive resume guard.
# ---------------------------------------------------------------------------


def _pre_lever_config() -> dict:
    """A producing-stage config as written before the lever flags existed."""
    stable = _stable_training_config(parse_args(BASE_ARGV))
    return {k: v for k, v in stable.items() if k not in ADDITIVE_LEVER_CONFIG_DEFAULTS}


def test_pre_lever_checkpoint_resumes_when_levers_are_inert() -> None:
    prior, current = _reconcile_additive_resume_config(
        _pre_lever_config(), _stable_training_config(parse_args(BASE_ARGV))
    )
    assert prior == current


def test_pre_lever_checkpoint_refuses_when_a_lever_is_active() -> None:
    active = _stable_training_config(parse_args([*BASE_ARGV, "--weight-qat-q3q4"]))
    prior, current = _reconcile_additive_resume_config(_pre_lever_config(), active)
    assert prior != current
    assert current["weight_qat_q3q4"] is True


@pytest.mark.parametrize(
    "argv_extra,key",
    [
        (["--weight-perturb-robustness", "0.25"], "weight_perturb_robustness"),
        (["--film-row-dropout", "0.1"], "film_row_dropout"),
        (["--carrier-rank-penalty", "1.0"], "carrier_rank_penalty"),
        (["--film-row-dropout-protect-top", "4"], "film_row_dropout_protect_top"),
    ],
)
def test_every_active_lever_refuses_a_pre_lever_parent(argv_extra: list, key: str) -> None:
    active = _stable_training_config(parse_args([*BASE_ARGV, *argv_extra]))
    prior, current = _reconcile_additive_resume_config(_pre_lever_config(), active)
    assert prior != current
    assert key in current


def test_lever_aware_checkpoint_is_compared_exactly() -> None:
    """Once a checkpoint knows the keys, no reconciliation may hide a difference."""
    prior = _stable_training_config(parse_args([*BASE_ARGV, "--weight-qat-q3q4"]))
    current = _stable_training_config(parse_args(BASE_ARGV))
    reconciled_prior, reconciled_current = _reconcile_additive_resume_config(prior, current)
    assert reconciled_prior != reconciled_current


def test_reconciliation_never_drops_a_non_lever_difference() -> None:
    prior = _pre_lever_config()
    prior["lr"] = 1.0
    current = _stable_training_config(parse_args(BASE_ARGV))
    reconciled_prior, reconciled_current = _reconcile_additive_resume_config(prior, current)
    assert reconciled_prior != reconciled_current


def test_reconciliation_does_not_invent_keys() -> None:
    prior, current = _reconcile_additive_resume_config(
        _pre_lever_config(), _stable_training_config(parse_args(BASE_ARGV))
    )
    for key in ADDITIVE_LEVER_CONFIG_DEFAULTS:
        assert key not in prior
        assert key not in current


# ---------------------------------------------------------------------------
# The render substitution: levers-off must reproduce uniform QAT exactly.
# ---------------------------------------------------------------------------


def _toy() -> torch.nn.Module:
    torch.manual_seed(3)
    model = torch.nn.Module()
    block = torch.nn.Module()
    block.register_parameter("weight", torch.nn.Parameter(torch.randn(12, 6)))
    block.register_parameter("bias", torch.nn.Parameter(torch.randn(12)))
    embed = torch.nn.Module()
    embed.register_parameter("weight", torch.nn.Parameter(torch.randn(20, 4)))
    model.add_module("conv", block)
    model.add_module("frame_embed", embed)
    return model


def test_parameter_overrides_with_levers_off_is_uniform_fake_quant() -> None:
    """Levers-off + base_bits must equal PR130's uniform quantized_forward params."""
    model = _toy()
    levers = EditabilityLevers(EditabilityLeverConfig())
    overrides = levers.parameter_overrides(model, base_bits=4)
    for name, param in model.named_parameters():
        torch.testing.assert_close(
            overrides[name], deployed_fake_quant(name, param, 4), rtol=0.0, atol=0.0
        )


def test_applied_with_base_bits_quantizes_even_with_every_lever_off() -> None:
    """Guards the catastrophic silent case: F1-only must not drop QAT."""
    model = _toy()
    levers = EditabilityLevers(EditabilityLeverConfig())
    original = model.conv.weight.detach().clone()
    with levers.applied(model, base_bits=4):
        inside = model.conv.weight.detach().clone()
    assert not torch.equal(inside, original), "base_bits path must still quantize"
    torch.testing.assert_close(
        inside, deployed_fake_quant("conv.weight", original, 4), rtol=0.0, atol=0.0
    )


def test_f1_only_still_quantizes_at_base_bits() -> None:
    model = _toy()
    levers = EditabilityLevers(
        EditabilityLeverConfig(weight_perturb_robustness=0.1)
    )
    overrides = levers.parameter_overrides(model, base_bits=4)
    # Perturbed then quantized: still lands on the 4-bit grid of the perturbed value.
    for name, value in overrides.items():
        assert torch.isfinite(value).all(), name
    assert not torch.equal(
        overrides["conv.weight"], deployed_fake_quant("conv.weight", model.conv.weight, 4)
    )


def test_f2_changes_only_the_selected_tensors_relative_to_uniform() -> None:
    model = _toy()
    uniform = EditabilityLevers(EditabilityLeverConfig()).parameter_overrides(
        model, base_bits=4
    )
    mixed = EditabilityLevers(
        EditabilityLeverConfig(weight_qat_q3q4=True)
    ).parameter_overrides(model, base_bits=4)
    # frame_embed.weight is in the mp2 q3 set; conv.weight is not.
    assert not torch.equal(mixed["frame_embed.weight"], uniform["frame_embed.weight"])
    torch.testing.assert_close(
        mixed["conv.weight"], uniform["conv.weight"], rtol=0.0, atol=0.0
    )
