# SPDX-License-Identifier: MIT
"""Task #852 — a red developer scope must report the DENOMINATOR of red gates.

Why this exists (MEASURED 2026-08-01): `tac.preflight --scope dev` was red on SIX
gates, but `_ParallelPreflightRunner.run` re-raised the first failure and
cancelled the rest. The standing plan to give the commit hook real gate coverage
was therefore written against "4 custody bypasses" — the violation count of the
FIRST red gate — as though that were the whole blocker. Reporting a first failure
as a debt total is the failure-side sibling of "vacuity is indistinguishable from
PASS": both answer with a symbol where the caller needed a count.

These tests pin the three behaviours the aggregation must have, including the one
its own round-2 review caught: control-flow exceptions must ABORT, never
aggregate.
"""
from __future__ import annotations

import pytest

from tac.preflight import (
    CodebaseDriftError,
    PreflightError,
    PreflightTimeoutError,
    _is_non_collectable_control_exception,
    _raise_aggregated_dev_gate_failures,
)


class TestAggregation:
    def test_no_failures_returns(self) -> None:
        assert _raise_aggregated_dev_gate_failures([], declared=25) is None

    def test_single_failure_reraises_the_original_exception_object(self) -> None:
        """Type preservation is load-bearing: the CLI selects handlers on it."""
        original = CodebaseDriftError("CODEBASE DRIFT DETECTED - ad-hoc patterns")
        with pytest.raises(CodebaseDriftError) as info:
            _raise_aggregated_dev_gate_failures(
                [("check_codebase_drift", original)], declared=25
            )
        assert info.value is original

    def test_many_failures_name_every_red_gate_and_the_denominator(self) -> None:
        failures = [
            ("check_codebase_drift", CodebaseDriftError("drift banner")),
            ("check_lane_pre_registered_before_work_starts",
             PreflightError("92 reference(s) to unregistered lane_id(s)")),
            ("check_subagent_landing_has_solver_wire_in",
             PreflightError("124 landing memo(s) missing wire-in")),
        ]
        with pytest.raises(PreflightError) as info:
            _raise_aggregated_dev_gate_failures(failures, declared=25)
        message = str(info.value)
        assert "3 of 25 declared developer gates are RED" in message
        for name, _exc in failures:
            assert name in message, f"{name} was not named in the aggregate report"
        # The counts each gate already computed must survive into the summary --
        # a reader plans against them.
        assert "92 reference(s)" in message
        assert "124 landing memo(s)" in message

    def test_aggregate_does_not_silently_downgrade_to_a_pass(self) -> None:
        """Whatever raised before must still raise. Aggregation is not amnesty."""
        with pytest.raises(PreflightError):
            _raise_aggregated_dev_gate_failures(
                [("a", PreflightError("x")), ("b", PreflightError("y"))], declared=2
            )


class TestControlExceptionsAbortInsteadOfAggregating:
    """Round-2 catch on this landing's OWN fix.

    The first version collected `BaseException`, which swallowed the wall-clock
    budget breach: the CLI lost the exception type it needs to report the hot
    step, AND a run that had already blown its deadline would keep executing the
    remaining gates. Aggregate what the DEVELOPER must fix; propagate what the
    RUN must obey.
    """

    @pytest.mark.parametrize(
        "exc",
        [
            PreflightTimeoutError("preflight exceeded 0.01s wall-clock budget"),
            KeyboardInterrupt(),
            SystemExit(1),
            MemoryError(),
        ],
    )
    def test_control_exceptions_are_not_collectable(self, exc: BaseException) -> None:
        assert _is_non_collectable_control_exception(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            PreflightError("check_x: 3 violation(s)"),
            CodebaseDriftError("drift"),
        ],
    )
    def test_gate_violations_are_collectable(self, exc: BaseException) -> None:
        assert _is_non_collectable_control_exception(exc) is False

    def test_timeout_subclasses_preflight_error_but_still_aborts(self) -> None:
        """The discriminator is the SUBCLASS, not the base type.

        `PreflightTimeoutError` inherits `PreflightError`, so a naive
        `isinstance(exc, PreflightError) -> collect` would have swallowed it.
        """
        timeout = PreflightTimeoutError("budget")
        assert isinstance(timeout, PreflightError)
        assert _is_non_collectable_control_exception(timeout) is True
