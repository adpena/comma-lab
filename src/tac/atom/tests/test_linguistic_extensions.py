# SPDX-License-Identifier: MIT
"""Tests for ``tac.atom.linguistic_extensions`` -- TOP-3 design memo APPENDIX B."""
from __future__ import annotations

import math
import time

import pytest

from tac.atom import Atom, ArbitrarinessClassification
from tac.atom.builders import build_arbitrary_value_atom
from tac.atom.linguistic_extensions import (
    AtomAlgebraError,
    TemporalLogicViolationError,
    always_invariant,
    classify_atom_arbitrariness,
    complement_atom,
    compose_atoms,
    eventually_extincted_within,
    intersect_atoms,
    union_atoms,
    valid_until,
)
from tac.atom.types import ResolutionPath


def _make_synthetic_atom(
    atom_id: str = "syn",
    resolution_path: ResolutionPath = ResolutionPath.EXPERIMENTAL,
    ev_lower: float = -0.005,
    ev_upper: float = -0.001,
) -> Atom:
    return build_arbitrary_value_atom(
        atom_id=atom_id,
        file_path="test.py",
        current_value=0.5,
        predicted_replacement=0.997,
        resolution_path=resolution_path,
        predicted_ev_delta_s=(ev_lower, ev_upper),
        cost_envelope_usd=1.0,
        literature_citation="test citation",
    )


# ---------------------------------------------------------------------------
# TOP-1: ArbitrarinessClassification modal-logic enum
# ---------------------------------------------------------------------------
class TestArbitrarinessClassification:
    def test_enum_has_4_modal_members(self):
        assert len(list(ArbitrarinessClassification)) == 4

    def test_canonical_members(self):
        assert ArbitrarinessClassification.NECESSARILY_CONTEST_FIXED.value == "necessarily_contest_fixed"
        assert ArbitrarinessClassification.POSSIBLY_ARBITRARY.value == "possibly_arbitrary"
        assert ArbitrarinessClassification.NECESSARILY_EMPIRICAL.value == "necessarily_empirical"
        assert ArbitrarinessClassification.INDETERMINATE_PENDING_EVIDENCE.value == "indeterminate_pending_evidence"

    def test_classify_contest_fixed_atom(self):
        atom = _make_synthetic_atom(resolution_path=ResolutionPath.CONTEST_FIXED)
        assert classify_atom_arbitrariness(atom) == ArbitrarinessClassification.NECESSARILY_CONTEST_FIXED

    def test_classify_formula_atom(self):
        atom = _make_synthetic_atom(resolution_path=ResolutionPath.FORMULA)
        assert classify_atom_arbitrariness(atom) == ArbitrarinessClassification.NECESSARILY_CONTEST_FIXED

    def test_classify_analytical_solve_atom(self):
        atom = _make_synthetic_atom(resolution_path=ResolutionPath.ANALYTICAL_SOLVE)
        assert classify_atom_arbitrariness(atom) == ArbitrarinessClassification.NECESSARILY_CONTEST_FIXED

    def test_classify_experimental_atom(self):
        atom = _make_synthetic_atom(resolution_path=ResolutionPath.EXPERIMENTAL)
        assert classify_atom_arbitrariness(atom) == ArbitrarinessClassification.NECESSARILY_EMPIRICAL

    def test_classify_learned_atom(self):
        atom = _make_synthetic_atom(resolution_path=ResolutionPath.LEARNED)
        assert classify_atom_arbitrariness(atom) == ArbitrarinessClassification.NECESSARILY_EMPIRICAL

    def test_classify_self_alien_tech_atom(self):
        atom = _make_synthetic_atom(resolution_path=ResolutionPath.SELF_ALIEN_TECH)
        assert classify_atom_arbitrariness(atom) == ArbitrarinessClassification.NECESSARILY_EMPIRICAL

    def test_explicit_metadata_overrides_inference(self):
        """If metadata carries explicit classification, it overrides inferred."""
        atom = build_arbitrary_value_atom(
            atom_id="explicit",
            file_path="test.py",
            current_value=0.5,
            predicted_replacement=0.997,
            resolution_path=ResolutionPath.FORMULA,  # would infer NECESSARILY_CONTEST_FIXED
            predicted_ev_delta_s=(-0.005, -0.001),
            cost_envelope_usd=1.0,
            literature_citation="test",
            extra_metadata={"arbitrariness_classification": "possibly_arbitrary"},
        )
        assert classify_atom_arbitrariness(atom) == ArbitrarinessClassification.POSSIBLY_ARBITRARY


# ---------------------------------------------------------------------------
# TOP-2: Atom-algebra
# ---------------------------------------------------------------------------
class TestAtomAlgebra:
    def test_compose_and_sums_predicted_impacts(self):
        left = _make_synthetic_atom(atom_id="L", ev_lower=-0.005, ev_upper=-0.001)
        right = _make_synthetic_atom(atom_id="R", ev_lower=-0.003, ev_upper=-0.001)
        composed = compose_atoms(left=left, right=right, op="and")
        assert math.isclose(composed.predicted_impact_delta_s_lower, -0.008)
        assert math.isclose(composed.predicted_impact_delta_s_upper, -0.002)
        assert "and" in composed.atom_id

    def test_compose_or_uses_extrema(self):
        left = _make_synthetic_atom(atom_id="L", ev_lower=-0.005, ev_upper=-0.001)
        right = _make_synthetic_atom(atom_id="R", ev_lower=-0.003, ev_upper=-0.002)
        composed = compose_atoms(left=left, right=right, op="or")
        # min of lowers = -0.005, max of uppers = -0.001
        assert composed.predicted_impact_delta_s_lower == -0.005
        assert composed.predicted_impact_delta_s_upper == -0.001

    def test_compose_xor_returns_abs_diff(self):
        left = _make_synthetic_atom(atom_id="L", ev_lower=-0.005, ev_upper=-0.001)
        right = _make_synthetic_atom(atom_id="R", ev_lower=-0.003, ev_upper=-0.001)
        composed = compose_atoms(left=left, right=right, op="xor")
        # |abs of lowers| = 0.002, |abs of uppers| = 0.0
        assert math.isclose(composed.predicted_impact_delta_s_lower, 0.0)
        assert math.isclose(composed.predicted_impact_delta_s_upper, 0.002)

    def test_compose_minus_clamped_to_zero(self):
        left = _make_synthetic_atom(atom_id="L", ev_lower=0.01, ev_upper=0.02)
        right = _make_synthetic_atom(atom_id="R", ev_lower=0.05, ev_upper=0.06)
        composed = compose_atoms(left=left, right=right, op="minus")
        # right > left -> minus clamps to 0
        assert composed.predicted_impact_delta_s_lower == 0.0
        assert composed.predicted_impact_delta_s_upper == 0.0

    def test_intersect_atoms_aliases_compose_and(self):
        left = _make_synthetic_atom(atom_id="L", ev_lower=-0.005, ev_upper=-0.001)
        right = _make_synthetic_atom(atom_id="R", ev_lower=-0.003, ev_upper=-0.001)
        i = intersect_atoms(left=left, right=right)
        c = compose_atoms(left=left, right=right, op="and")
        assert i.predicted_impact_delta_s_lower == c.predicted_impact_delta_s_lower
        assert i.predicted_impact_delta_s_upper == c.predicted_impact_delta_s_upper

    def test_union_atoms_aliases_compose_or(self):
        left = _make_synthetic_atom(atom_id="L", ev_lower=-0.005, ev_upper=-0.001)
        right = _make_synthetic_atom(atom_id="R", ev_lower=-0.003, ev_upper=-0.001)
        u = union_atoms(left=left, right=right)
        c = compose_atoms(left=left, right=right, op="or")
        assert u.predicted_impact_delta_s_lower == c.predicted_impact_delta_s_lower

    def test_complement_negates_predicted_impact(self):
        atom = _make_synthetic_atom(ev_lower=-0.005, ev_upper=-0.001)
        comp = complement_atom(atom=atom)
        # original [-0.005, -0.001] -> complement [0.001, 0.005]
        assert math.isclose(comp.predicted_impact_delta_s_lower, 0.001)
        assert math.isclose(comp.predicted_impact_delta_s_upper, 0.005)
        # complement has zero cost (inaction is free)
        assert comp.cost_envelope_usd == 0.0
        assert "complement" in comp.atom_id

    def test_complement_records_history(self):
        atom = _make_synthetic_atom()
        comp = complement_atom(atom=atom)
        assert "complement" in comp.metadata.get("algebra_history", [])

    def test_compose_rejects_unknown_op(self):
        left = _make_synthetic_atom(atom_id="L")
        right = _make_synthetic_atom(atom_id="R")
        with pytest.raises(AtomAlgebraError):
            compose_atoms(left=left, right=right, op="fubar")

    def test_compose_rejects_non_atom_operands(self):
        with pytest.raises(AtomAlgebraError):
            compose_atoms(left="not an atom", right=_make_synthetic_atom())  # type: ignore[arg-type]

    def test_complement_rejects_non_atom(self):
        with pytest.raises(AtomAlgebraError):
            complement_atom(atom="not an atom")  # type: ignore[arg-type]

    def test_compose_cost_envelopes_summed(self):
        left = _make_synthetic_atom(atom_id="L")  # cost=1.0
        right = _make_synthetic_atom(atom_id="R")  # cost=1.0
        composed = compose_atoms(left=left, right=right, op="and")
        assert composed.cost_envelope_usd == 2.0


# ---------------------------------------------------------------------------
# TOP-3: Temporal-logic decorators
# ---------------------------------------------------------------------------
class TestTemporalLogicDecorators:
    def test_always_invariant_predicate_holds_lets_fn_run(self):
        @always_invariant(lambda *a, **kw: kw.get("x", 0) >= 0)
        def fn(*, x: float) -> float:
            return x * 2

        assert fn(x=5) == 10

    def test_always_invariant_predicate_fails_raises(self):
        @always_invariant(lambda *a, **kw: kw.get("x", 0) >= 0)
        def fn(*, x: float) -> float:
            return x * 2

        with pytest.raises(TemporalLogicViolationError):
            fn(x=-1)

    def test_always_invariant_preserves_fn_signature(self):
        @always_invariant(lambda *a, **kw: True)
        def fn(*, x: float, y: float) -> float:
            """Original docstring."""
            return x + y

        assert fn.__name__ == "fn"
        assert fn.__doc__ == "Original docstring."

    def test_eventually_extincted_within_stamps_deadline(self):
        @eventually_extincted_within(days=30)
        def helper():
            return 1

        assert hasattr(helper, "_eventually_extincted_within_deadline_utc")
        assert hasattr(helper, "_eventually_extincted_within_decoration_utc")
        assert helper._eventually_extincted_within_days == 30

    def test_eventually_extincted_within_runtime_works(self):
        """The decorator does NOT block runtime; it records deadline."""
        @eventually_extincted_within(days=30)
        def helper(x: int) -> int:
            return x * 2

        assert helper(5) == 10

    def test_eventually_extincted_within_rejects_zero_days(self):
        with pytest.raises(ValueError):
            @eventually_extincted_within(days=0)
            def fn():
                pass

    def test_eventually_extincted_within_rejects_negative_days(self):
        with pytest.raises(ValueError):
            @eventually_extincted_within(days=-1)
            def fn():
                pass

    def test_valid_until_condition_not_fired_lets_fn_run(self):
        @valid_until(lambda: False)  # condition never fires
        def fn() -> int:
            return 42

        assert fn() == 42

    def test_valid_until_condition_fired_raises(self):
        @valid_until(lambda: True)  # condition always True -> invalid
        def fn() -> int:
            return 42

        with pytest.raises(TemporalLogicViolationError):
            fn()

    def test_valid_until_preserves_fn_signature(self):
        @valid_until(lambda: False)
        def fn(*, x: int) -> int:
            """Doc."""
            return x

        assert fn.__name__ == "fn"
        assert fn.__doc__ == "Doc."
