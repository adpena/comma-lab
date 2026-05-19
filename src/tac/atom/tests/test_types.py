# SPDX-License-Identifier: MIT
"""Tests for ``tac.atom.types``."""
from __future__ import annotations

import pytest

from tac.atom.types import (
    AtomKind,
    AtomProtocol,
    AtomValidationError,
    ResolutionPath,
)


class TestAtomKind:
    def test_has_all_seven_canonical_members(self) -> None:
        expected = {
            "arbitrary_value",
            "meta_lagrangian",
            "cargo_cult_assumption",
            "premise_verification",
            "probe_outcome",
            "council_deliberation",
            "dispatch_claim",
        }
        actual = {m.value for m in AtomKind}
        assert actual == expected

    def test_is_str_enum(self) -> None:
        assert AtomKind.PROBE_OUTCOME == "probe_outcome"
        assert isinstance(AtomKind.PROBE_OUTCOME.value, str)


class TestResolutionPath:
    def test_has_canonical_five_plus_contest_fixed(self) -> None:
        expected = {
            "experimental",
            "analytical_solve",
            "formula",
            "learned",
            "self_alien_tech",
            "contest_fixed",
        }
        actual = {m.value for m in ResolutionPath}
        assert actual == expected

    def test_is_str_enum(self) -> None:
        assert ResolutionPath.EXPERIMENTAL == "experimental"


class TestAtomProtocol:
    def test_runtime_checkable_minimum_shape(self) -> None:
        # A minimal duck-typed instance satisfies the Protocol
        class Dummy:
            atom_id = "x"
            kind = AtomKind.PROBE_OUTCOME
            resolution_path = ResolutionPath.EXPERIMENTAL
            predicted_impact_delta_s_lower = 0.0
            predicted_impact_delta_s_upper = 0.0
            cost_envelope_usd = 0.0

            def to_jsonl_row(self):
                return {}

            def to_meta_lagrangian_atom(self):
                return {}

        assert isinstance(Dummy(), AtomProtocol)

    def test_runtime_checkable_rejects_missing_methods(self) -> None:
        class Incomplete:
            atom_id = "x"
            kind = AtomKind.PROBE_OUTCOME
            resolution_path = ResolutionPath.EXPERIMENTAL
            predicted_impact_delta_s_lower = 0.0
            predicted_impact_delta_s_upper = 0.0
            cost_envelope_usd = 0.0
            # missing to_jsonl_row + to_meta_lagrangian_atom

        assert not isinstance(Incomplete(), AtomProtocol)


class TestAtomValidationError:
    def test_is_valueerror_subclass(self) -> None:
        assert issubclass(AtomValidationError, ValueError)

    def test_raisable(self) -> None:
        with pytest.raises(AtomValidationError, match="test message"):
            raise AtomValidationError("test message")
