# SPDX-License-Identifier: MIT
"""Behaviour tests for the ddm_os1 retroactive termination-census law.

The load-bearing ones are the CONTROLS: a synthetic solve whose termination is known by
construction must be reconstructed correctly, and a solve that genuinely converged must
NOT be reported as bound-stopped.  Everything else is contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.canonical_equations.ddm_os1_termination_census_from_cost_proxy_20260802 import (
    BOUND_EITHER,
    BOUND_LADDER,
    BOUND_RELIN,
    CONVERGED,
    INFEASIBLE,
    build_ddm_os1_termination_census_from_cost_proxy_v1,
    termination_census,
)

REPO = Path(__file__).resolve().parents[3]

# A synthetic damped-GN shape used for the contract/control tests. NOTE: this is NOT
# ddm_pfs1's shape -- pinning that one is `test_the_pfs1_anchor_shape_is_the_receipt_
# producing_revisions` below. Naming it PFS1 in the first cut of this file was how the
# wrong-revision reading survived review; the name is kept generic on purpose now.
SHAPE = {"relin_bound": 4, "fd_per_relin": 6, "ladder_levels": 4,
         "line_search_points": 2, "init_cost": 1}
L_MAX = 8


def _simulate(relins_entered: int, accepting_costs: list[int], *, ladder_exhausted: bool,
              init: int = 1, fd: int = 6) -> int:
    """Forward count a solve of this shape WOULD record — the ground-truth generator."""
    assert len(accepting_costs) == relins_entered - (1 if ladder_exhausted else 0)
    n = init + fd * relins_entered + sum(accepting_costs)
    if ladder_exhausted:
        n += L_MAX
    return n


# --------------------------------------------------------------------------- controls


@pytest.mark.parametrize("relins,accepting", [
    (1, []), (2, [1]), (3, [4, 4]), (4, [8, 1, 3]),
])
def test_ladder_exhausted_solves_are_never_reported_as_converged(relins, accepting):
    """POSITIVE CONTROL: a solve that ran out of damping must read as a BOUND."""
    n = _simulate(relins, accepting, ladder_exhausted=True)
    out = termination_census([n], **SHAPE)
    assert out["states"][0] in {BOUND_LADDER, BOUND_EITHER}
    assert out["census"]["stopped_on_a_bound"]["count"] == 1
    assert out["census"][CONVERGED]["count"] == 0


@pytest.mark.parametrize("accepting", [[1, 1, 1, 1], [8, 8, 8, 8], [2, 5, 1, 7]])
def test_relin_exhausted_solves_are_never_reported_as_converged(accepting):
    """POSITIVE CONTROL: every relin accepted and the cap ran out — still a BOUND."""
    n = _simulate(4, accepting, ladder_exhausted=False)
    out = termination_census([n], **SHAPE)
    assert out["states"][0] in {BOUND_RELIN, BOUND_EITHER}
    assert out["census"][CONVERGED]["count"] == 0


def test_a_genuinely_converged_solve_is_not_called_bound_stopped():
    """NEGATIVE CONTROL: the census must not manufacture censoring where none exists."""
    n = _simulate(2, [3], ladder_exhausted=True)
    out = termination_census([n, n], objective=[1e-9, 5.0], tolerance=1e-6, **SHAPE)
    assert out["states"][0] == CONVERGED
    assert out["states"][1] != CONVERGED
    assert out["census"][CONVERGED]["count"] == 1
    assert out["verdict"] == "MIXED"


def test_convergence_is_read_from_the_objective_not_the_cost():
    """Two identical costs, different objectives — only the objective may decide."""
    n = _simulate(3, [2, 2], ladder_exhausted=True)
    below = termination_census([n], objective=[1e-12], tolerance=1e-6, **SHAPE)
    above = termination_census([n], objective=[1e-3], tolerance=1e-6, **SHAPE)
    assert below["states"][0] == CONVERGED
    assert above["states"][0] != CONVERGED


# ------------------------------------------------------------------- refusal contract


def test_an_empty_population_is_vacuous_never_a_clean_bill():
    out = termination_census([], **SHAPE)
    assert out["verdict"] == "UNDETERMINED_EMPTY"
    assert out["sufficient_for_verdict"] is False
    assert "denominator" in out["insufficiency_reason"]


def test_without_an_objective_the_convergence_leg_is_declared_undecidable():
    n = _simulate(4, [1, 1, 1, 1], ladder_exhausted=False)
    out = termination_census([n], **SHAPE)
    assert out["convergence_decidable"] is False
    assert out["sufficient_for_verdict"] is False
    assert out["insufficiency_reason"] == (
        "no_objective_and_tolerance_convergence_leg_undecidable"
    )


def test_a_cost_outside_every_feasible_interval_refuses_rather_than_guessing():
    """A cost the model cannot explain is INFEASIBLE — the model refusing, not a census."""
    out = termination_census([10_000], objective=[1.0], tolerance=1e-6, **SHAPE)
    assert out["states"][0] == INFEASIBLE
    assert out["sufficient_for_verdict"] is False
    assert "singular" in out["insufficiency_reason"]


def test_tolerance_without_objective_is_refused():
    with pytest.raises(ValueError, match="must be supplied together"):
        termination_census([30], tolerance=1e-6, **SHAPE)


def test_misaligned_objective_is_refused():
    with pytest.raises(ValueError, match="align 1:1"):
        termination_census([30, 31], objective=[1.0], tolerance=1e-6, **SHAPE)


@pytest.mark.parametrize("bad", ["relin_bound", "fd_per_relin", "ladder_levels",
                                 "line_search_points"])
def test_nonpositive_shape_parameters_are_refused(bad):
    kw = dict(SHAPE)
    kw[bad] = 0
    with pytest.raises(ValueError, match=bad):
        termination_census([30], **kw)


# ------------------------------------------------------------------------- aggregation


def test_objective_mass_fractions_are_reported_alongside_counts():
    """Mass, not count, is what the ranking consumes — pw1's measured lesson."""
    heavy = _simulate(1, [], ladder_exhausted=True)      # cheap solve, huge residual
    light = _simulate(4, [8, 8, 8, 8], ladder_exhausted=False)
    out = termination_census([heavy, light], objective=[99.0, 1.0], tolerance=1e-6, **SHAPE)
    bound = out["census"]["stopped_on_a_bound"]
    assert bound["count"] == 2
    assert bound["objective_mass_fraction"] == pytest.approx(1.0)
    ladder = out["census"][BOUND_LADDER]
    assert ladder["count"] == 1
    assert ladder["objective_mass_fraction"] == pytest.approx(0.99)


def test_max_possible_cost_matches_the_shape():
    out = termination_census([30], **SHAPE)
    assert out["l_max_per_relin"] == L_MAX
    assert out["max_possible_cost"] == 1 + 6 * 4 + 4 * 8


def test_a_solve_at_relin_one_is_pinned_exactly():
    """n = 15 admits ONLY ladder-exhaustion at R=1 — the sharpest reading available."""
    out = termination_census([15], objective=[1.0], tolerance=1e-6, **SHAPE)
    assert out["states"][0] == BOUND_LADDER


# ----------------------------------------------------------- the measured n600 anchor


def test_the_cost_lattice_has_a_gap_and_the_model_reports_it():
    """The model's REACHABLE SET is not an interval, and the gap is the real check.

    A solve of this shape can record 15 (ladder-exhausted after one relinearization) or
    >= 22, but NOTHING in 16..21 — a second relinearization costs at least 6 forwards plus
    a line search, so the lattice has a hole.  Rows landing in a hole are what make
    ``n_infeasible`` a real guard: on the ddm_pfs1 anchor it fired at 88/600 once the
    shape was read from the receipt-producing revision instead of the working tree.

    A tautological version of this test (force every objective above tolerance, assert
    nothing converged) would pass against a broken implementation; this one does not.
    """
    out = termination_census(list(range(1, 60)), **SHAPE)
    costs = list(range(1, 60))
    infeasible = {n for n, s in zip(costs, out["states"], strict=True) if s == INFEASIBLE}
    assert infeasible == set(range(1, 15)) | set(range(16, 22)) | {58, 59}
    feasible = {n for n, s in zip(costs, out["states"], strict=True) if s != INFEASIBLE}
    assert feasible == {15} | set(range(22, 58))


def test_the_equation_builds_and_carries_its_measured_anchor():
    eq = build_ddm_os1_termination_census_from_cost_proxy_v1()
    assert eq.equation_id == "ddm_os1_termination_census_from_cost_proxy_v1"
    assert len(eq.empirical_anchors) == 1
    anchor = eq.empirical_anchors[0]
    assert anchor.empirical_output["converged"] == 0
    assert anchor.empirical_output["stopped_on_a_bound_at_least"] == 512
    assert anchor.empirical_output["n_infeasible"] == 88
    assert anchor.inputs["new_scorer_evaluations"] == 0
    assert anchor.inputs["receipt_producing_revision"].startswith("8eb3d14594")


def test_the_equation_excludes_loop_shape_inference_explicitly():
    """The measured negative must travel with the law, not only with the memo."""
    eq = build_ddm_os1_termination_census_from_cost_proxy_v1()
    excluded = " ".join(eq.domain_of_validity["excluded"])
    assert "shape" in excluded.lower()
    assert eq.domain_of_validity["score_claim"] is False


def test_registered_evaluator_matches_the_direct_call():
    from tac.canonical_equations.evaluators import get_evaluator

    fn = get_evaluator("ddm_os1_termination_census_from_cost_proxy_v1")
    costs = [15, 22, 30, 44]
    direct = termination_census(costs, objective=[1.0] * 4, tolerance=1e-6, **SHAPE)
    via = fn({"cost_counts": costs, "objective": [1.0] * 4, "tolerance": 1e-6, **SHAPE})
    assert via["states"] == direct["states"]
    assert via["census"] == direct["census"]


def test_evaluator_refuses_inputs_outside_the_canonical_contract():
    from tac.canonical_equations.evaluators import get_evaluator

    fn = get_evaluator("ddm_os1_termination_census_from_cost_proxy_v1")
    with pytest.raises(ValueError, match="canonical callable contract"):
        fn({"cost_counts": [30], "relin_bound": 4, "fd_per_relin": 6,
            "ladder_levels": 4, "line_search_points": 2, "unexpected": 1})


def test_numpy_integer_costs_are_accepted():
    """Receipts round-trip through numpy; int64 must not be refused."""
    out = termination_census(np.array([15, 30], dtype=np.int64),
                             objective=np.array([1.0, 2.0]), tolerance=1e-6, **SHAPE)
    assert out["n_items"] == 2
    assert out["census"]["stopped_on_a_bound"]["count"] == 2


def test_non_finite_objective_is_refused_rather_than_averaged_in():
    """A diverged solve must not poison every mass fraction silently."""
    with pytest.raises(ValueError, match="non-finite"):
        termination_census([15, 30], objective=[1.0, float("nan")], tolerance=1e-6, **SHAPE)
    with pytest.raises(ValueError, match="non-finite"):
        termination_census([15, 30], objective=[1.0, float("inf")], tolerance=1e-6, **SHAPE)


def test_the_mass_key_is_always_present_even_when_undecidable():
    """Absent-vs-None: a mass-ranking consumer must fail loudly, not KeyError sometimes."""
    without = termination_census([15, 30], **SHAPE)
    assert without["census"]["stopped_on_a_bound"]["objective_mass_fraction"] is None
    all_zero = termination_census([15, 30], objective=[0.0, 0.0], tolerance=-1.0, **SHAPE)
    assert all_zero["census"]["stopped_on_a_bound"]["objective_mass_fraction"] is None


# ------------------------------------------------------- report tool (round-2 defects)


def test_report_tool_parses_pretty_printed_json_not_only_jsonl(tmp_path):
    """A pretty-printed .json contains "\\n{" — suffix/substring sniffing read it as empty."""
    from tools.os1_termination_census_report import load_rows

    p = tmp_path / "receipt.json"
    p.write_text(json.dumps([{"n": 15, "d": 1.0}, {"n": 30, "d": 2.0}], indent=1))
    rows, bad = load_rows(p)
    assert bad == 0
    assert [r["n"] for r in rows] == [15, 30]


def test_report_tool_parses_jsonl(tmp_path):
    from tools.os1_termination_census_report import load_rows

    p = tmp_path / "receipt.jsonl"
    p.write_text('{"n": 15}\n\n{"n": 30}\nnot-json\n')
    rows, bad = load_rows(p)
    assert [r["n"] for r in rows] == [15, 30]
    assert bad == 1


def test_report_tool_refuses_a_tolerance_without_an_objective(tmp_path):
    from tools.os1_termination_census_report import main

    p = tmp_path / "r.jsonl"
    p.write_text('{"n_forwards": 15}\n')
    with pytest.raises(SystemExit):
        main(["--receipt", str(p), "--cost-field", "n_forwards", "--tolerance", "1e-6",
              "--relin-bound", "4", "--fd-per-relin", "6", "--ladder-levels", "4",
              "--line-search-points", "2"])


def test_report_tool_counts_a_null_cost_as_missing_not_present(tmp_path):
    from tools.os1_termination_census_report import load_rows

    p = tmp_path / "r.jsonl"
    p.write_text('{"n_forwards": 15}\n{"n_forwards": null}\n')
    rows, _ = load_rows(p)
    usable = [r for r in rows if r.get("n_forwards") is not None]
    assert len(rows) == 2
    assert len(usable) == 1


def test_report_tool_reports_a_vacuous_receipt_as_vacuous(tmp_path, capsys):
    """Empty scope must not read as a clean bill."""
    from tools.os1_termination_census_report import main

    p = tmp_path / "r.jsonl"
    p.write_text('{"something_else": 1}\n')
    rc = main(["--receipt", str(p), "--cost-field", "n_forwards",
               "--relin-bound", "4", "--fd-per-relin", "6", "--ladder-levels", "4",
               "--line-search-points", "2"])
    assert rc == 3
    assert "VACUOUS" in capsys.readouterr().err


def test_the_equation_does_not_claim_a_consumer_that_never_calls_it():
    """canonical_consumers is a claim the code must honour (NO-FAKE)."""
    eq = build_ddm_os1_termination_census_from_cost_proxy_v1()
    for consumer in eq.canonical_consumers:
        text = (REPO / consumer).read_text(encoding="utf-8")
        assert "termination_census" in text, f"{consumer} never calls the law"


def test_the_persisted_registry_row_carries_no_false_consumer_claim():
    """The BUILDER guard above is not enough — the queryable row is the authority.

    Registering before a later edit to canonical_consumers leaves a stale claim on disk
    that the builder test cannot see. That happened once during this landing.
    """
    from tac.canonical_equations import query_equations

    rows = [e for e in query_equations()
            if e.equation_id == "ddm_os1_termination_census_from_cost_proxy_v1"]
    if not rows:
        pytest.skip("equation not registered in this checkout")
    persisted = rows[-1]          # latest-row-wins
    built = build_ddm_os1_termination_census_from_cost_proxy_v1()
    assert tuple(persisted.canonical_consumers) == tuple(built.canonical_consumers)
    for consumer in persisted.canonical_consumers:
        assert "termination_census" in (REPO / consumer).read_text(encoding="utf-8")


def test_the_pfs1_anchor_shape_matches_the_receipt_producing_revision():
    """The anchor's shape must come from the revision that WROTE the receipt.

    The first cut of this law read init=1/fd=6 off the WORKING TREE, which carried an
    uncommitted sibling rewrite, and produced a confident wrong census. The real shape at
    revision 8eb3d14594 (== HEAD) is n = 2 + 7R + sum(L_i): init=2 (one initial pose6_of
    plus one trailing d_pose_shipped) and fd=7 (six pose FD columns plus the s_t column).
    """
    import subprocess

    from tac.canonical_equations.ddm_os1_termination_census_from_cost_proxy_20260802 import (
        PFS1_FD_PER_RELIN,
        PFS1_INIT_COST,
    )

    assert (PFS1_INIT_COST, PFS1_FD_PER_RELIN) == (2, 7)
    src = subprocess.run(
        ["git", "show", "8eb3d14594:experiments/ddm_pfs1_ep_warp_pose_solve.py"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    body = src.split("def solve_pair_gn", 1)[1].split("\ndef ", 1)[0]
    assert f"n_fwd += {PFS1_FD_PER_RELIN}" in body, "fd_per_relin must match the source"
    assert body.count("n_fwd += 1") == 2, "init=2 means one leading and one trailing count"
    assert "for _damp in range(4)" in body
    assert "for scale in (1.0, 0.5)" in body


def test_the_anchor_reports_the_bound_share_as_a_lower_bound_not_a_census():
    """88 infeasible rows mean the law REFUSED; the anchor must not read as a full census."""
    eq = build_ddm_os1_termination_census_from_cost_proxy_v1()
    out = eq.empirical_anchors[0].empirical_output
    assert out["converged"] == 0                       # model-independent, load-bearing
    assert out["n_infeasible"] == 88
    assert out["sufficient_for_verdict"] is False
    assert "stopped_on_a_bound_at_least" in out
    assert "stopped_on_a_bound" not in out             # no unqualified census claim
