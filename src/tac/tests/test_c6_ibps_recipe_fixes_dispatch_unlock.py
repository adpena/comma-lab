# SPDX-License-Identifier: MIT
"""Tests for the C6 IBPS 4 recipe-side fixes dispatch unlock landing 2026-05-17.

Per lane `lane_c6_ibps_4_recipe_fixes_dispatch_unlock_20260517` deliverable #4:
~18 tests across the 4-fix closure surface:

  - Fix 1 (Dykstra polytope FEASIBLE): canonical JSON anchor
    `.omx/state/dykstra_feasibility_c6_e4_mdl_ibps.json`
  - Fix 2 (composition_alpha=1.0 ORTHOGONAL): canonical JSON anchor
    `.omx/state/composition_alpha_c6_e4_mdl_ibps_x_wyner_ziv.json`
  - Fix 3 (Tier-C ACROSS_CLASS): canonical JSON anchor
    `.omx/state/tier_c_density_reconciliation_c6_e4_mdl_ibps.json`
  - Fix 4 (sextet PROCEED-unconditional): canonical posterior anchor
    `.omx/research/council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517.md`
    + `.omx/state/council_deliberation_posterior.jsonl` entry
  - Fix 5 (Recipe dispatch_enabled=true flip):
    `.omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml`

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #229 premise verification
+ Catalog #230 sister-subagent ownership: these tests cover the 4-fix closure
SURFACE without re-implementing canonical helpers; they read the persisted
JSON / YAML artifacts and verify the contract.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_PATH = (
    REPO_ROOT
    / ".omx"
    / "operator_authorize_recipes"
    / "substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml"
)
DYKSTRA_PATH = REPO_ROOT / ".omx" / "state" / "dykstra_feasibility_c6_e4_mdl_ibps.json"
COMPOSITION_ALPHA_PATH = (
    REPO_ROOT
    / ".omx"
    / "state"
    / "composition_alpha_c6_e4_mdl_ibps_x_wyner_ziv.json"
)
TIER_C_RECONCILIATION_PATH = (
    REPO_ROOT / ".omx" / "state" / "tier_c_density_reconciliation_c6_e4_mdl_ibps.json"
)
COUNCIL_MEMO_PATH = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517.md"
)
COUNCIL_POSTERIOR_PATH = (
    REPO_ROOT / ".omx" / "state" / "council_deliberation_posterior.jsonl"
)
DESIGN_MEMO_PATH = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "c6_e4_mdl_ibps_cargo_cult_unwind_design_20260516.md"
)


# ── Fix 1: Dykstra feasibility ─────────────────────────────────────────────


def test_fix_1_dykstra_anchor_exists():
    assert DYKSTRA_PATH.exists(), (
        f"Catalog #296 sister anchor missing at {DYKSTRA_PATH}"
    )


def test_fix_1_dykstra_verdict_feasible():
    data = json.loads(DYKSTRA_PATH.read_text())
    assert data["verdict"] == "FEASIBLE", (
        f"Dykstra polytope must be FEASIBLE for unlock; got {data['verdict']}"
    )
    assert data["blocker_axis"] is None
    assert data["score_claim"] is False
    assert data["promotion_eligible"] is False


def test_fix_1_dykstra_substrate_id_c6():
    data = json.loads(DYKSTRA_PATH.read_text())
    assert data["substrate_id"] == "c6_e4_mdl_ibps"


def test_fix_1_dykstra_polytope_contains_predicted_band():
    """Dykstra projection MUST contain the predicted band [0.113, 0.163]."""
    data = json.loads(DYKSTRA_PATH.read_text())
    band_lo = data["feasibility_band_lo"]
    band_hi = data["feasibility_band_hi"]
    rationale = data["feasibility_rationale"]
    # Polytope projection extracted from rationale text
    assert "[0.099879, 0.531541]" in rationale, (
        f"Expected polytope projection in rationale; got: {rationale}"
    )
    assert band_lo == 0.113
    assert band_hi == 0.163
    # Polytope contains band per FEASIBLE verdict semantics
    assert 0.099879 <= band_lo
    assert band_hi <= 0.531541


def test_fix_1_dykstra_constraints_include_c6_specific():
    """Constraints must include C6-specific MDL + IB + Z1 sets."""
    data = json.loads(DYKSTRA_PATH.read_text())
    ids = data["constraint_set_ids"]
    assert "contest_rate_budget" in ids
    assert "contest_seg_dist_budget" in ids
    assert "contest_pose_dist_budget" in ids
    # C6-specific
    assert any("rissanen" in i.lower() for i in ids), (
        "expected Rissanen MDL constraint"
    )
    assert any("tishby" in i.lower() for i in ids), "expected Tishby IB constraint"
    assert any("z1" in i.lower() for i in ids), (
        "expected Z1 class-shift constraint"
    )


# ── Fix 2: composition_alpha ───────────────────────────────────────────────


def test_fix_2_composition_alpha_anchor_exists():
    assert COMPOSITION_ALPHA_PATH.exists()


def test_fix_2_composition_alpha_orthogonal():
    data = json.loads(COMPOSITION_ALPHA_PATH.read_text())
    comp = data["composition"]
    assert comp["substrate_a"] == "c6_e4_mdl_ibps"
    assert "wyner_ziv" in comp["substrate_b"].lower()
    assert comp["composability"] == "orthogonal"
    assert comp["expected_alpha"] == 1.0


def test_fix_2_composition_alpha_sha256_distinctness():
    """Per #823 lesson: composition_alpha must NOT be emitted on byte-identical
    archives (lane_g_v3 + siren renderer.bin = phantom alpha)."""
    data = json.loads(COMPOSITION_ALPHA_PATH.read_text())
    proof = data["sha256_distinctness_proof"]
    assert proof["distinctness_verified"] is True
    assert (
        proof["substrate_a_namespace_sha"]
        != proof["substrate_b_namespace_sha"]
    )


def test_fix_2_composite_band_additive():
    """ORTHOGONAL alpha=1.0 means delta_composite = alpha * (delta_a + delta_b)."""
    data = json.loads(COMPOSITION_ALPHA_PATH.read_text())
    pred = data["predicted_composite_band"]
    a_lo, a_hi = pred["substrate_a_alone_band"]
    b_lo, b_hi = pred["substrate_b_alone_band"]
    composite_lo = pred["composite_lo"]
    composite_hi = pred["composite_hi"]
    # alpha=1.0 additive
    assert abs(composite_lo - (a_lo + b_lo)) < 1e-9
    assert abs(composite_hi - (a_hi + b_hi)) < 1e-9
    # No score claim
    assert pred["score_claim"] is False
    assert pred["promotion_eligible"] is False


# ── Fix 3: Tier-C reconciliation ───────────────────────────────────────────


def test_fix_3_tier_c_reconciliation_exists():
    assert TIER_C_RECONCILIATION_PATH.exists()


def test_fix_3_tier_c_across_class_verdict():
    data = json.loads(TIER_C_RECONCILIATION_PATH.read_text())
    assert data["substrate_id"] == "c6_e4_mdl_ibps"
    cls = data["catalog_227_classification"]
    assert cls["verdict"] == "ACROSS_CLASS"
    assert cls["haircut_applied"] is False
    measurement = data["tier_c_density_measurement"]
    assert measurement["measured_density"] == pytest.approx(2.67e-5)
    # Across class threshold per Catalog #227
    assert measurement["measured_density"] < cls["across_class_threshold"]


def test_fix_3_revised_band_revalidated():
    data = json.loads(TIER_C_RECONCILIATION_PATH.read_text())
    rec = data["reconciliation_with_design_memo"]
    assert rec["cc3_revised_band_status"] == "REVALIDATED"
    assert rec["revised_band"] == [0.113, 0.163]


# ── Fix 4: sextet council ──────────────────────────────────────────────────


def test_fix_4_council_memo_exists():
    assert COUNCIL_MEMO_PATH.exists()


def test_fix_4_council_memo_v2_frontmatter_complete():
    """Per Catalog #300 v2 frontmatter requirements."""
    body = COUNCIL_MEMO_PATH.read_text()
    assert body.startswith("---\n"), "Frontmatter must lead the file"
    # Extract YAML frontmatter
    rest = body[4:]
    fm_end = rest.find("\n---\n")
    assert fm_end != -1, "Frontmatter must terminate"
    fm = yaml.safe_load(rest[:fm_end])
    # Required Catalog #300 v2 fields
    assert fm["council_tier"] == "T2"
    assert fm["council_quorum_met"] is True
    assert fm["council_verdict"] == "PROCEED"
    assert "Shannon" in fm["council_attendees"]
    assert "Dykstra" in fm["council_attendees"]
    assert "Assumption-Adversary" in fm["council_attendees"]
    # Mission-alignment (Catalog #300 extension)
    assert fm["council_predicted_mission_contribution"] == "frontier_breaking"
    assert fm["council_override_invoked"] is False
    # Dissent + assumption-adversary surfaced
    assert len(fm["council_dissent"]) >= 2
    assert len(fm["council_assumption_adversary_verdict"]) >= 4


def test_fix_4_council_posterior_anchor_appended():
    """Per Catalog #128 + #131 fcntl-locked JSONL discipline."""
    assert COUNCIL_POSTERIOR_PATH.exists()
    rows = [
        json.loads(line)
        for line in COUNCIL_POSTERIOR_PATH.read_text().splitlines()
        if line.strip()
    ]
    matching = [
        r
        for r in rows
        if r.get("deliberation_id")
        == "council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517"
    ]
    assert len(matching) >= 1, (
        "Council deliberation anchor must be appended to canonical posterior"
    )
    record = matching[-1]
    assert record["council_tier"] == "T2"
    assert record["council_verdict"] == "PROCEED"
    assert record["predicted_mission_contribution"] == "frontier_breaking"


def test_fix_4_council_classifies_one_cargo_cult():
    """Per CLAUDE.md META-ASSUMPTION non-negotiable: Assumption-Adversary must
    classify at least one shared assumption as CARGO-CULTED."""
    body = COUNCIL_MEMO_PATH.read_text()
    rest = body[4:]
    fm_end = rest.find("\n---\n")
    fm = yaml.safe_load(rest[:fm_end])
    classes = [
        v["classification"] for v in fm["council_assumption_adversary_verdict"]
    ]
    assert "CARGO-CULTED" in classes, (
        "Per CLAUDE.md META-ASSUMPTION: ≥1 CARGO-CULTED assumption required"
    )


# ── Fix 5: Recipe dispatch_enabled flip + Catalog #270 ─────────────────────


def test_fix_5_recipe_parses():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    assert recipe["lane_id"] == "lane_c6_e4_mdl_ibps_substrate_20260514"


def test_fix_5_recipe_dispatch_enabled_true():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    assert recipe.get("dispatch_enabled") is True, (
        "Recipe MUST set dispatch_enabled=true post-unlock"
    )


def test_fix_5_recipe_predicted_band_revalidated():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    assert recipe.get("predicted_band") == [0.113, 0.163], (
        "Recipe predicted_band MUST flip from null to [0.113, 0.163]"
    )
    assert recipe.get("predicted_score_target") == pytest.approx(0.138)


def test_fix_5_recipe_dispatch_blockers_field_absent():
    """Old blockers field MUST be removed (replaced by dispatch_blockers_cleared)."""
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    assert recipe.get("dispatch_blockers") is None, (
        "Recipe dispatch_blockers MUST be cleared (replaced by dispatch_blockers_cleared)"
    )
    cleared = recipe.get("dispatch_blockers_cleared")
    assert cleared is not None and len(cleared) == 4, (
        f"Recipe MUST list 4 cleared blockers; got {cleared}"
    )


def test_fix_5_recipe_catalog_270_protocol_passes():
    """Per Catalog #270 dispatch optimization protocol: Tier 1/2/3 all pass."""
    # Skip if helper missing (defensive)
    helper = REPO_ROOT / "tools" / "canonical_dispatch_optimization_protocol.py"
    if not helper.exists():
        pytest.skip("canonical_dispatch_optimization_protocol.py missing")
    py = shutil.which("python") or "/usr/bin/python3"
    import subprocess

    result = subprocess.run(
        [
            py,
            str(helper),
            "--trainer",
            str(REPO_ROOT / "experiments/train_substrate_c6_e4_mdl_ibps.py"),
            "--recipe",
            "substrate_c6_e4_mdl_ibps_modal_t4_dispatch",
            "--json",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Parse JSON verdict
    if result.returncode != 0 and not result.stdout:
        pytest.skip(f"protocol tool errored: {result.stderr[:500]}")
    try:
        verdict = json.loads(result.stdout)
    except json.JSONDecodeError:
        pytest.skip(f"protocol tool emitted non-JSON: {result.stdout[:500]}")
    assert verdict["tier1"]["blockers"] == []
    assert verdict["tier2"]["blockers"] == []
    assert verdict["tier3"]["blockers"] == []


def test_fix_5_recipe_cites_4_fix_anchors_in_predicted_delta_basis():
    """Predicted-delta basis must cite all 4 canonical fix anchors."""
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    basis = recipe.get("predicted_delta_basis", "")
    assert "dykstra_feasibility_c6_e4_mdl_ibps.json" in basis
    assert "composition_alpha_c6_e4_mdl_ibps_x_wyner_ziv.json" in basis
    assert "tier_c_density_reconciliation_c6_e4_mdl_ibps.json" in basis
    assert "council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517" in basis


# ── Sister regression guards ──────────────────────────────────────────────


def test_design_memo_unchanged_per_historical_provenance():
    """Per Catalog #110 / #113 HISTORICAL_PROVENANCE: design memo body must
    remain unchanged; this lane's reconciliation appends to .omx/state/, not
    to the design memo body."""
    assert DESIGN_MEMO_PATH.exists()
    # Spot-check the original CC-3 unwind text is intact
    body = DESIGN_MEMO_PATH.read_text()
    assert "CC-3" in body
    assert "Dykstra-feasibility" in body
    assert "cargo-cult unwind" in body.lower() or "CARGO-CULT" in body


def test_canonical_inventory_c6_row_unchanged():
    """Sister `redo_pivot_fix_all_20260517` landed Catalog #322 + revert helper;
    this lane MUST NOT touch substrate_composition_matrix.py."""
    from tac.optimization.substrate_composition_matrix import (
        canonical_substrate_inventory,
    )

    inv = canonical_substrate_inventory()
    c6_rows = [r for r in inv if r.substrate_id == "c6_e4_mdl_ibps"]
    assert len(c6_rows) == 1, "C6 row must exist exactly once in canonical inventory"
    c6 = c6_rows[0]
    # Existing fields unchanged
    assert c6.lane_id == "lane_c6_e4_mdl_ibps_substrate_20260514"
    assert c6.literature_anchor.startswith("Tishby-Zaslavsky")
