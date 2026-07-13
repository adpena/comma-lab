from __future__ import annotations

import importlib
import json
from pathlib import Path


def test_direct_module_discovery_and_authority_boundary() -> None:
    module = importlib.import_module("tac.canonical_equations.segnet_margin_trust_region_20260713")
    equation = module.build_segnet_margin_trust_region_v1()
    assert equation.equation_id == "segnet_margin_trust_region_v1"
    assert "local Jacobian" in equation.domain_of_validity["excluded"][0]
    assert "no rigorous positive radius" in equation.domain_of_validity["missing_bound"]
    assert "PROXY_ACCEPT" in equation.domain_of_validity["empirical_proxy"]
    assert equation.provenance.score_claim_valid is False
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_output["review_status"] == "fresh-eyes-reviewed(1)-CLEAN"
    assert anchor.empirical_output["verdict"] == "NO_GO"


def test_measured_anchor_matches_terminal_receipt() -> None:
    root = Path(__file__).resolve().parents[4]
    receipt = json.loads(
        (root / "experiments/results/segnet_validation_certificate_20260713T015633Z/receipt.json").read_text()
    )
    module = importlib.import_module("tac.canonical_equations.segnet_margin_trust_region_20260713")
    anchor = module.build_segnet_margin_trust_region_v1().empirical_anchors[0]
    accepts = sum(
        row["proxy_decision"]["status"] == "PROXY_ACCEPT"
        for regime in receipt["regimes"]
        for row in regime["holdout"]
    )
    assert anchor.empirical_output["proxy_accepts"] == accepts == 3
    assert anchor.empirical_output["joint_unsafe_accepts"] == 2
    assert anchor.empirical_output["derived_speedup_k2"] == receipt["economics"]["cadences"]["2"]["derived_speedup"]
    assert anchor.empirical_output["derived_speedup_k4"] == receipt["economics"]["cadences"]["4"]["derived_speedup"]
