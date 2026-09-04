"""Tests for ``tools/register_ddm_pr1_coupling_refinement_20260904``.

The equations leg is where a wrong number becomes permanent, so the guards here
are about what the tool would carry into the registry:

* it must refuse a report that is not a ddm_pr1 report (a foreign schema would
  silently register another arm's numbers under this anchor id);
* every registered value must be READ from the report, never retyped -- the test
  perturbs the report and asserts the anchor follows;
* the residual must be computed against the PRE-re-solve k, because the law's
  registered band was fitted to that quantity and the POST-re-solve k is a
  different one. Scoring a residual against a centre fitted to another quantity
  is the arithmetic version of a cross-instrument comparison.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_MODULE_PATH = REPO / "tools" / "register_ddm_pr1_coupling_refinement_20260904.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pr1_eq_undertest", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reg = _load_module()

_REPORT = {
    "schema": "tac.ddm_pr1.report.v1",
    "pairs": 600,
    "pair_selection": "full n600",
    "batch_size": 8,
    "instrument": {
        "archive_sha256": (
            "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
        )
    },
    "d_pose": {
        "base_as_shipped": 6.3656845e-06,
        "candidate_stale_carrier": 0.015482,
        "candidate_re_solved": 3.0e-05,
        "base_vs_t4_relative": -0.00068,
    },
    "recovery": {"mean_based": 516.0, "median_per_pair": 1200.0},
    "coupling": {
        "post_re_solve": 0.34,
        "pre_re_solve": 223.3,
        "delta_d_seg_used": 6.93e-05,
        "delta_d_seg_source": "test",
    },
    "closing_arithmetic": {
        "seg_cut_fraction": 0.25,
        "payable_pose_ceiling": 1.694e-05,
        "k_post_payable_bar": 0.2098,
        "overshoot_multiple": 1.4,
        "payable": False,
    },
    "charter_prediction": {"prediction_holds": True, "falsifier_fired": False},
}


def _report_path(tmp_path: Path, **overrides) -> Path:
    doc = json.loads(json.dumps(_REPORT))
    for dotted, value in overrides.items():
        head, _, tail = dotted.partition(".")
        if tail:
            doc[head][tail] = value
        else:
            doc[head] = value
    path = tmp_path / "report.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class TestAnchorIsReadNotRetyped:
    def test_every_headline_value_follows_the_report(self, tmp_path):
        path = _report_path(
            tmp_path, **{"coupling.post_re_solve": 42.0, "recovery.mean_based": 7.5}
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        anchor = reg.build_anchor(doc, path)
        assert anchor.empirical_output["coupling_post_re_solve"] == 42.0
        assert anchor.empirical_output["carrier_recovery_mean_based"] == 7.5
        assert anchor.inputs["n_pairs"] == 600
        assert anchor.inputs["delta_d_seg"] == 6.93e-05

    def test_residual_is_scored_against_the_pre_re_solve_k(self, tmp_path):
        centre = 190.38926383452008
        path = _report_path(tmp_path, **{"coupling.pre_re_solve": centre})
        doc = json.loads(path.read_text(encoding="utf-8"))
        anchor = reg.build_anchor(doc, path)
        assert anchor.residual == pytest.approx(0.0, abs=1e-12), (
            "the band was fitted to the PRE-re-solve k; scoring the residual "
            "against the POST-re-solve k would compare different quantities"
        )

    def test_residual_moves_with_the_pre_not_the_post_coupling(self, tmp_path):
        low = _report_path(tmp_path, **{"coupling.post_re_solve": 0.1})
        doc_low = json.loads(low.read_text(encoding="utf-8"))
        high = _report_path(tmp_path, **{"coupling.post_re_solve": 900.0})
        doc_high = json.loads(high.read_text(encoding="utf-8"))
        a = reg.build_anchor(doc_low, low)
        b = reg.build_anchor(doc_high, high)
        assert a.residual == b.residual

    def test_anchor_is_non_promotable_research_only(self, tmp_path):
        path = _report_path(tmp_path)
        anchor = reg.build_anchor(json.loads(path.read_text(encoding="utf-8")), path)
        assert anchor.provenance.promotion_eligible is False
        assert anchor.provenance.score_claim_valid is False
        assert anchor.provenance.measurement_axis == "[macOS-CPU advisory]"


class TestDomainExtension:
    def test_excludes_the_transferred_token_edit_recovery(self, tmp_path):
        doc = json.loads(_report_path(tmp_path).read_text(encoding="utf-8"))
        extension = reg.build_domain_extension(doc)
        excluded = " ".join(extension["domain_of_validity_excluded"])
        assert "TOKEN-edit carrier recovery" in excluded
        assert "8.0x" in excluded and "5.87x" in excluded

    def test_separates_pre_from_post_re_solve(self, tmp_path):
        doc = json.loads(_report_path(tmp_path).read_text(encoding="utf-8"))
        extension = reg.build_domain_extension(doc)
        included = " ".join(extension["domain_of_validity_included"])
        assert "PRE-re-solve" in included
        excluded = " ".join(extension["domain_of_validity_excluded"])
        assert "WITHOUT running the terminal pose re-solve" in excluded

    def test_carries_the_measured_renderer_recovery_not_a_transferred_one(self, tmp_path):
        doc = json.loads(_report_path(tmp_path).read_text(encoding="utf-8"))
        extension = reg.build_domain_extension(doc)
        got = extension["carrier_recovery_measured_renderer_change"]
        assert got["mean_based"] == 516.0
        assert got["median_per_pair"] == 1200.0
        assert got["source"] == "ddm_pr1"

    def test_carries_the_payability_bar_with_its_derivation(self, tmp_path):
        doc = json.loads(_report_path(tmp_path).read_text(encoding="utf-8"))
        bar = reg.build_domain_extension(doc)["post_re_solve_payability_bar"]
        assert bar["k_post_must_be_at_most"] == 0.2098
        assert "promotion condition" in bar["derivation"]


class TestSchemaGate:
    def test_refuses_a_foreign_report(self, tmp_path):
        path = tmp_path / "foreign.json"
        path.write_text(json.dumps({"schema": "tac.ddm_ft1.candidate_verdict.v1"}),
                        encoding="utf-8")
        with pytest.raises(SystemExit, match="not a ddm_pr1 report"):
            reg.main(["--report", str(path), "--dry-run"])

    def test_dry_run_does_not_touch_the_registry(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(reg, "update_equation_with_domain_refinement",
                            lambda *a, **k: called.append("domain"))
        monkeypatch.setattr(reg, "update_equation_with_empirical_anchor",
                            lambda *a, **k: called.append("anchor"))
        path = _report_path(tmp_path)
        assert reg.main(["--report", str(path), "--dry-run"]) == 0
        assert called == []

    def test_live_run_refines_then_appends_in_that_order(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(reg, "update_equation_with_domain_refinement",
                            lambda *a, **k: called.append("domain"))
        monkeypatch.setattr(reg, "update_equation_with_empirical_anchor",
                            lambda *a, **k: called.append("anchor"))
        path = _report_path(tmp_path)
        assert reg.main(["--report", str(path)]) == 0
        assert called == ["domain", "anchor"]

    def test_it_targets_the_registered_law_id(self):
        assert reg.EQUATION_ID == "renderer_seg_pose_coupling_shipped_object_v1"
