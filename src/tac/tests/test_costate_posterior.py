"""Tests for the #247 cross-run costate POSTERIOR (tac.witness_control.costate_posterior) — the
continual-learning loop. NO-FAKE: only IDENTIFIABLE costates accumulate; combination is
inverse-variance; UNIDENTIFIABLE is never recorded."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from tac.witness_control import costate_posterior as cp


@pytest.fixture
def store(tmp_path):
    return tmp_path / "costate_posterior.jsonl"


# --- record: only identifiable, finite ------------------------------------
def test_record_measured_is_written(store):
    r = cp.record_costate_observation("dS_depoch[muon]", -1e-4, 2e-5, tier="MEASURED",
                                      run_ref="/runs/a", path=store)
    assert r is not None and r["value"] == -1e-4


def test_unidentifiable_is_never_recorded(store):
    assert cp.record_costate_observation("x", 0.0, 0.0, tier="UNIDENTIFIABLE",
                                         run_ref="/runs/a", path=store) is None
    assert cp._read(store) == []


def test_nonfinite_dropped(store):
    assert cp.record_costate_observation("x", float("nan"), 1.0, tier="MEASURED",
                                         run_ref="/runs/a", path=store) is None
    assert cp.record_costate_observation("x", 1.0, float("inf"), tier="MEASURED",
                                         run_ref="/runs/a", path=store) is None


# --- posterior: inverse-variance combination ------------------------------
def test_posterior_inverse_variance_combination(store):
    # two runs measure the same costate; tighter stderr dominates the mean
    cp.record_costate_observation("L", 1.0, 1.0, tier="MEASURED", run_ref="/runs/a", path=store)
    cp.record_costate_observation("L", 3.0, 0.1, tier="MEASURED", run_ref="/runs/b", path=store)
    post = cp.posterior_for("L", path=store)
    assert post is not None and post.n_runs == 2
    # precision-weighted mean is pulled hard toward the tight (3.0 ± 0.1) estimate
    assert 2.9 < post.mean <= 3.0
    assert post.stderr < 0.1  # combined precision is higher than either alone


def test_posterior_latest_per_run_wins(store):
    # a run re-measures the same costate later (more converged); latest wins for that run
    cp.record_costate_observation("L", 5.0, 1.0, tier="MEASURED", run_ref="/runs/a", path=store)
    cp.record_costate_observation("L", 2.0, 0.5, tier="MEASURED", run_ref="/runs/a", path=store)
    post = cp.posterior_for("L", path=store)
    assert post.n_runs == 1 and abs(post.mean - 2.0) < 1e-9  # only the latest /runs/a estimate


def test_posterior_none_when_unmeasured(store):
    assert cp.posterior_for("never", path=store) is None


def test_all_posteriors_one_row_per_name(store):
    cp.record_costate_observation("A", 1.0, 1.0, tier="MEASURED", run_ref="/runs/a", path=store)
    cp.record_costate_observation("B", 2.0, 1.0, tier="ANALYTIC", run_ref="/runs/a", path=store)
    rows = cp.all_posteriors(path=store)
    assert {r["name"] for r in rows} == {"A", "B"}


# --- record_run_costates (CLOSE side) -------------------------------------
@dataclass
class _FakeCostate:
    name: str
    value: float
    stderr: float
    tier: str
    evidence: tuple


def test_record_run_costates_filters_unidentifiable(store):
    costates = [
        _FakeCostate("good", -1e-4, 1e-5, "MEASURED", ("verdict rows",)),
        _FakeCostate("bad", 0.0, 0.0, "UNIDENTIFIABLE", ()),  # must be dropped
    ]
    rows = cp.record_run_costates(costates, "/runs/z", path=store)
    assert {r["name"] for r in rows} == {"good"}
    assert cp.posterior_for("bad", path=store) is None


def test_shadow_report_carries_costate_prior():
    """The live controller SEES the cross-run posterior via ShadowReport.costate_prior + JSONL row."""
    from tac.witness_control.shadow_controller import RunInputs, build_shadow_report
    rep = build_shadow_report(RunInputs(run_dir="/tmp/cp_smoke", verdicts=[], stage_rows={}, flags={}))
    assert isinstance(rep.costate_prior, list)          # present (may be empty if no run recorded yet)
    assert "costate_prior" in rep.to_row()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
