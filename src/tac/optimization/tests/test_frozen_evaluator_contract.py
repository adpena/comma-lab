"""Tests for the FrozenEvaluatorContract harness — the evaluator-PLUGGABLE V6 abstraction.

The headline test PROVES off-the-shelf generality: the SAME `synthesize_frontier` op that
will drive the comma contest finds the KNOWN optimum of a SYNTHETIC goal it has never seen
(task-conditioned MDL on a toy signal). That is the operator's "use it off the shelf for
this contest, then create a frontier score for a synthetic/literature goal we invent".
"""
from __future__ import annotations

from typing import Any

import pytest

from tac.optimization.frozen_evaluator_contract import (
    FrozenEvaluator,
    FrozenEvaluatorContract,
    comma_video_compression_contract,
    synthesize_frontier,
    verify_evaluator_satisfies_contract,
    verify_frozen_evaluator_contract,
)


# --- a SYNTHETIC frozen evaluator (a non-comma goal with a known optimum) -----------
class _SyntheticTopKReadoutEvaluator:
    """Toy task-conditioned MDL: keep the top-k coefficients (by magnitude) of a fixed
    sparse signal; a downstream readout = the sum. Distortion = relative readout error;
    'byte_units' = k. objective = 100*distortion + 0.5*k (minimize).

    Known optimum: the 3 large coeffs carry ~98.4% of the readout; the 4 tiny ones each
    cost 0.5 byte-unit to save ~0.41 distortion-points (100*0.1/24.4) = NOT worth it. So
    k*=3 is the analytic frontier (verified in the test below).
    """

    _X = (10.0, 8.0, 6.0, 0.1, 0.1, 0.1, 0.1)
    _RATE_COEF = 0.5

    def _recon(self, k: int) -> list[float]:
        order = sorted(range(len(self._X)), key=lambda i: -abs(self._X[i]))
        keep = set(order[: max(0, k)])
        return [self._X[i] if i in keep else 0.0 for i in range(len(self._X))]

    def score_terms(self, *, source_ref: str, candidate: Any) -> dict[str, float]:
        k = int(candidate)
        recon = self._recon(k)
        sx, sr = sum(self._X), sum(recon)
        dist = abs(sx - sr) / abs(sx) if sx else 0.0
        return {"task_distortion": dist, "byte_units": float(k)}

    def score(self, *, source_ref: str, candidate: Any) -> float:
        t = self.score_terms(source_ref=source_ref, candidate=candidate)
        return 100.0 * t["task_distortion"] + self._RATE_COEF * t["byte_units"]

    def within_eval_budget(self, *, candidate: Any) -> bool:
        return 0 <= int(candidate) <= len(self._X)


def _synthetic_contract() -> FrozenEvaluatorContract:
    return FrozenEvaluatorContract(
        name="synthetic_topk_readout",
        source_ref="synthetic://sparse_signal_v1",
        objective_formula="100*task_distortion + 0.5*byte_units",
        objective_terms=("task_distortion", "byte_units"),
        minimize=True,
        eval_runtime_budget_seconds=1.0,
        submission_boundary="k (top-k coefficient count)",
        overfit_authorized=True,
    )


def test_synthesize_frontier_finds_known_synthetic_optimum() -> None:
    contract = _synthetic_contract()
    ev = _SyntheticTopKReadoutEvaluator()
    res = synthesize_frontier(contract, ev, candidates=range(0, 8), candidate_id=str)
    # The harness it has never seen finds k*=3 (the analytic frontier).
    assert res.best_candidate_id == "3"
    assert res.n_within_budget == 8 and res.n_candidates_evaluated == 8
    assert res.minimize is True
    # Independent confirmation: k=3 beats its neighbors under the objective.
    s2 = ev.score(source_ref=contract.source_ref, candidate=2)
    s3 = ev.score(source_ref=contract.source_ref, candidate=3)
    s4 = ev.score(source_ref=contract.source_ref, candidate=4)
    s7 = ev.score(source_ref=contract.source_ref, candidate=7)
    assert s3 < s2 and s3 < s4 and s3 < s7
    assert res.best_objective == pytest.approx(s3)


def test_synthetic_evaluator_satisfies_protocol() -> None:
    ev = _SyntheticTopKReadoutEvaluator()
    assert isinstance(ev, FrozenEvaluator)
    assert verify_evaluator_satisfies_contract(ev)["satisfies"] is True


def test_comma_contract_is_well_formed() -> None:
    c = comma_video_compression_contract()
    v = verify_frozen_evaluator_contract(c)
    assert v["well_formed"] is True, v["issues"]
    assert c.objective_terms == ("d_seg", "d_pose", "rate")
    assert c.eval_runtime_budget_seconds == 1800.0
    assert c.rate_denominator_bytes and c.rate_denominator_bytes > 0
    assert c.overfit_authorized is True


def test_malformed_contract_rejected() -> None:
    with pytest.raises(ValueError):
        FrozenEvaluatorContract(  # empty objective_terms
            name="bad", source_ref="x", objective_formula="?", objective_terms=(),
            minimize=True, eval_runtime_budget_seconds=1.0, submission_boundary="?",
        )
    # rate term without a denominator is flagged not-well-formed (comma-style coupling).
    c = FrozenEvaluatorContract(
        name="rate_no_denom", source_ref="x", objective_formula="rate", objective_terms=("rate",),
        minimize=True, eval_runtime_budget_seconds=1.0, submission_boundary="?",
    )
    assert verify_frozen_evaluator_contract(c)["well_formed"] is False


def test_evaluator_missing_surface_is_caught() -> None:
    class _NotAnEvaluator:
        def score(self, **_: Any) -> float:  # missing score_terms + within_eval_budget
            return 0.0

    v = verify_evaluator_satisfies_contract(_NotAnEvaluator())
    assert v["satisfies"] is False
    assert "score_terms" in v["missing_methods"]
    with pytest.raises(ValueError):
        synthesize_frontier(_synthetic_contract(), _NotAnEvaluator(), candidates=range(3))  # type: ignore[arg-type]
