"""Tests for the variation gate.

The load-bearing test is `test_the_r1a_seed_path_is_refused`: the gate must refuse
the exact design that cost 3.4 h of authorized time, using only the trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_ds1_variation_gate import (  # noqa: E402
    Threshold,
    VariationClaim,
    VariationGateError,
    assert_can_vary,
    trace_is_broken,
)

LIVE = VariationClaim(
    quantity="renderer packet bytes",
    knob="ceiling_multiplier",
    path=("ceiling_multiplier", "adaptive_allocation_from_sensitivity", "packet_bytes"),
    evidence="ddm_wd3_scorer_aware_width_distillation.py:884-949",
)


# ── the instance that motivated the gate ──────────────────────────────────────


def test_the_r1a_seed_path_is_refused() -> None:
    """The real R1a design, refused from the trace alone. This is the whole point."""

    claim = VariationClaim(
        quantity="exchange ratio r",
        knob="config['seed']",
        path=(
            "config['seed']",
            "seed_everything",
            "torch.Generator",
            "_restore_rng(checkpoint) OVERWRITES the generator",
            "training trajectory",
        ),
        evidence="experiments/ddm_wd3_scorer_aware_width_distillation.py:1139,2213",
    )
    with pytest.raises(VariationGateError, match="cannot vary"):
        assert_can_vary(claim)


def test_the_refusal_names_the_destroying_hop() -> None:
    hop = trace_is_broken(("seed", "generator", "_restore_rng OVERWRITES it", "trajectory"))
    assert hop is not None and "OVERWRITES" in hop


def test_observation_beats_a_broken_trace() -> None:
    """If it has actually been SEEN to vary, the trace heuristic must not veto."""

    claim = VariationClaim(
        quantity="r",
        knob="seed",
        path=("seed", "restore path", "r"),
        evidence="receipt.json sha256 abc",
        observed_to_vary=True,
    )
    assert assert_can_vary(claim)["passed_by"] == "observation"


# ── live traces pass ──────────────────────────────────────────────────────────


def test_live_trace_passes() -> None:
    result = assert_can_vary(LIVE)
    assert result["trace_live"] is True
    assert result["passed_by"] == "live_trace"
    assert result["hops"] == 3


def test_every_overwrite_marker_is_caught() -> None:
    for marker in ("overwrite", "overwritten", "restore", "restored", "reset", "clobber", "ignored", "discard"):
        assert trace_is_broken(("knob", f"the value is {marker} here", "quantity")) is not None


def test_a_marker_in_the_knob_itself_is_not_a_break() -> None:
    """Only hops AFTER the knob can destroy it; the knob may legitimately be named 'reset'."""

    assert trace_is_broken(("reset_ramp_divisor", "optimizer", "trajectory")) is None


# ── claim validation ──────────────────────────────────────────────────────────


def test_one_hop_path_refuses() -> None:
    with pytest.raises(VariationGateError, match="at least knob"):
        VariationClaim(quantity="q", knob="k", path=("q",), evidence="e.py:1")


@pytest.mark.parametrize("bad", ["", "TBD", "todo", "n/a", "<rationale>", "  "])
def test_placeholder_evidence_refuses(bad: str) -> None:
    with pytest.raises(VariationGateError, match="placeholder"):
        VariationClaim(quantity="q", knob="k", path=("k", "q"), evidence=bad)


def test_placeholder_quantity_refuses() -> None:
    with pytest.raises(VariationGateError, match="quantity"):
        VariationClaim(quantity="unknown", knob="k", path=("k", "q"), evidence="e.py:1")


def test_placeholder_hop_refuses() -> None:
    with pytest.raises(VariationGateError, match="placeholder hop"):
        VariationClaim(quantity="q", knob="k", path=("k", "TBD", "q"), evidence="e.py:1")


# ── Threshold cannot exist without a passing claim ────────────────────────────


def test_threshold_requires_a_passing_claim() -> None:
    dead = VariationClaim(
        quantity="r", knob="seed", path=("seed", "_restore_rng overwrites", "r"), evidence="x.py:1"
    )
    with pytest.raises(VariationGateError, match="cannot vary"):
        Threshold(name="KILL", value=1.2, units="x improvement", derivation="from the cost table", claim=dead)


def test_conventional_derivation_refuses() -> None:
    """Instance 2, verbatim: rho >= 0.90 because 0.90 is conventionally good."""

    with pytest.raises(VariationGateError, match="convention"):
        Threshold(
            name="proxy_sound",
            value=0.90,
            units="spearman rho",
            derivation="0.90 is the conventional threshold for good correlation",
            claim=LIVE,
        )


def test_default_derivation_refuses() -> None:
    with pytest.raises(VariationGateError, match="convention"):
        Threshold(name="t", value=1.2, units="x", derivation="the default value everyone uses", claim=LIVE)


def test_derived_threshold_is_admitted() -> None:
    t = Threshold(
        name="proxy_mis_rank_cost",
        value=0.001027,
        units="S",
        derivation="1,543 B forgone on D56 x 6.658590e-07 S/B = 3.64% of the remaining gap",
        claim=LIVE,
    )
    assert t.value == pytest.approx(0.001027)


@pytest.mark.parametrize("field", ["name", "units", "derivation"])
def test_placeholder_threshold_fields_refuse(field: str) -> None:
    kwargs = {"name": "t", "value": 1.0, "units": "S", "derivation": "measured", "claim": LIVE}
    kwargs[field] = "TBD"
    with pytest.raises(VariationGateError, match=field):
        Threshold(**kwargs)


def test_receipt_shape_is_machine_readable() -> None:
    r = assert_can_vary(LIVE)
    assert r["gate"] == "ddm_ds1_variation_gate"
    assert set(r) == {"gate", "quantity", "knob", "hops", "trace_live", "observed_to_vary", "evidence", "passed_by"}


def test_honest_derivation_mentioning_default_is_admitted() -> None:
    """The bare-word check false-refused this; phrases must not."""

    t = Threshold(
        name="floor",
        value=0.0,
        units="S",
        derivation="measured on the default config's own baseline, 1,543 B forgone",
        claim=LIVE,
    )
    assert t.units == "S"


@pytest.mark.parametrize(
    "phrase",
    ["conventional", "by convention", "the default value", "rule of thumb", "everyone uses", "commonly used"],
)
def test_convention_phrases_all_refuse(phrase: str) -> None:
    with pytest.raises(VariationGateError, match="convention"):
        Threshold(name="t", value=1.0, units="S", derivation=f"picked because it is {phrase}", claim=LIVE)
