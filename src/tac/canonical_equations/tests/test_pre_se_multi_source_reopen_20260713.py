# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.pre_se_multi_source_reopen_20260713 import (
    EQUATION_ID,
    build_pre_se_multi_source_reopen_v1,
    cheap_global_tile_flops,
    joint_reopen_admitted,
    populate_pre_se_multi_source_reopen_v1,
    retained_mass_fraction,
)
from tac.canonical_equations.registry import query_equations


def test_retained_mass_matches_measured_nonlinear_aggregate() -> None:
    assert retained_mass_fraction(
        retained_mass=0.0003940449215155513,
        total_mass=0.001248472641573917,
    ) == pytest.approx(0.31562159104967574)


def test_true_per_tile_cost_includes_amortized_globals() -> None:
    assert cheap_global_tile_flops(
        local_conv_forward_macs_sum=1_045_272_384,
        global_forward_plus_vjp_flops=16_864_000,
        tile_count=4,
    ) == pytest.approx(1_049_488_384.0)


def test_both_reopen_bars_are_required() -> None:
    assert joint_reopen_admitted(
        retained_mass=0.47,
        retained_mass_bar=0.47,
        tileable_modulo_cheap_globals=True,
    )
    assert not joint_reopen_admitted(
        retained_mass=0.31562159104967574,
        retained_mass_bar=0.47,
        tileable_modulo_cheap_globals=True,
    )
    assert not joint_reopen_admitted(
        retained_mass=0.9,
        retained_mass_bar=0.47,
        tileable_modulo_cheap_globals=False,
    )


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (retained_mass_fraction, {"retained_mass": 2.0, "total_mass": 1.0}),
        (
            cheap_global_tile_flops,
            {
                "local_conv_forward_macs_sum": 0,
                "global_forward_plus_vjp_flops": 1,
                "tile_count": 1,
            },
        ),
        (
            joint_reopen_admitted,
            {
                "retained_mass": 0.5,
                "retained_mass_bar": 0.47,
                "tileable_modulo_cheap_globals": 1,
            },
        ),
    ],
)
def test_laws_fail_closed(function, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        function(**kwargs)


def test_equation_preserves_scope_and_measured_anchor() -> None:
    equation = build_pre_se_multi_source_reopen_v1()
    anchor = equation.empirical_anchors[0]
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["scope_level"] == (
        "family x frozen PRE-SE feature charts x fixed replay"
    )
    assert anchor.empirical_output["verdict"] == "RETAINED-MASS-FAMILY-KILL"
    assert anchor.empirical_output["tileable_modulo_cheap_globals"] is True
    assert anchor.empirical_output["nonlinear_multi_source_retained_mass"] == pytest.approx(
        0.31562159104967574
    )
    assert equation.provenance.score_claim_valid is False
    assert equation.provenance.promotion_eligible is False


def test_population_round_trips_through_isolated_registry(tmp_path: Path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_pre_se_multi_source_reopen_v1(
        path=registry,
        lock_path=lock,
        agent="pytest",
        subagent_id="pre_se_multi_source",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line]
    loaded = query_equations(path=registry)
    assert populated.equation_id == EQUATION_ID
    assert [row.equation_id for row in loaded] == [EQUATION_ID]
    assert rows[0]["notes"] == (
        "FEED-484; retained-mass-family-kill; cheap-globals tileability pass; research-only"
    )
