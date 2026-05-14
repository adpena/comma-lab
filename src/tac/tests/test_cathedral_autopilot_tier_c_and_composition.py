# SPDX-License-Identifier: MIT
"""Tests for the Catalog #227 autopilot ranker wire-in:

  1. `mdl_tier_c_density` field on CandidateRow + the
     `adjust_predicted_delta_for_mdl_tier_c_density` helper
  2. `composition_alpha` field + the
     `adjust_predicted_delta_for_composition_alpha` helper
  3. `load_substrate_composition_alpha_index` consumer of the canonical
     `.omx/state/substrate_composition_matrix.json` posterior surface
  4. `apply_substrate_composition_matrix_to_candidates` populator
  5. Class-shift literature tokens extension (Ha-Schmidhuber / Hafner /
     DreamerV3 retention per C1 council "RETAIN" decision)
  6. Integration: full `apply_z1_empirical_revision_to_candidate_delta`
     stacking the Tier A + Tier C + class-shift + composition adjustments
  7. JSONL loader backward-compat for the new fields

Memory: feedback_autopilot_tier_c_integration_catalog_227_landed_20260514.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


import importlib.util


def _load_autopilot_module():
    """Load the cathedral_autopilot_autonomous_loop.py module."""
    name = "cathedral_autopilot_autonomous_loop"
    if name in sys.modules:
        return sys.modules[name]
    script = TOOLS / "cathedral_autopilot_autonomous_loop.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── adjust_predicted_delta_for_mdl_tier_c_density ────────────────────────


def test_tier_c_density_within_class_floors_delta() -> None:
    """Tier C density >= 0.70 (within-class) caps predicted delta near zero."""
    mod = _load_autopilot_module()
    # density = 0.85 (within-class). Original predicted is strong improvement.
    adjusted = mod.adjust_predicted_delta_for_mdl_tier_c_density(
        base_delta=-0.05, mdl_tier_c_density=0.85,
    )
    assert adjusted == mod.MDL_TIER_C_WITHIN_CLASS_SATURATED_DELTA_FLOOR
    assert adjusted == -0.005


def test_tier_c_density_within_class_trending_halves_delta() -> None:
    """Tier C density 0.50-0.70 (trending) halves predicted savings."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_mdl_tier_c_density(
        base_delta=-0.04, mdl_tier_c_density=0.60,
    )
    assert adjusted == pytest.approx(-0.02)


def test_tier_c_density_across_class_applies_bonus() -> None:
    """Tier C density <= 0.30 (across-class) subtracts bonus from delta."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_mdl_tier_c_density(
        base_delta=-0.02, mdl_tier_c_density=0.13,  # C6 5ep value
    )
    # bonus 0.01 subtracted → more negative
    assert adjusted == pytest.approx(-0.03)


def test_tier_c_density_indeterminate_no_adjustment() -> None:
    """0.30 < density < 0.50 → no adjustment."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_mdl_tier_c_density(
        base_delta=-0.03, mdl_tier_c_density=0.40,
    )
    assert adjusted == pytest.approx(-0.03)


def test_tier_c_density_none_no_adjustment() -> None:
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_mdl_tier_c_density(
        base_delta=-0.03, mdl_tier_c_density=None,
    )
    assert adjusted == pytest.approx(-0.03)


def test_tier_c_density_non_numeric_no_adjustment() -> None:
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_mdl_tier_c_density(
        base_delta=-0.03, mdl_tier_c_density="not_a_number",  # type: ignore[arg-type]
    )
    assert adjusted == pytest.approx(-0.03)


def test_tier_c_density_threshold_boundary_at_0_70() -> None:
    """density exactly 0.70 should trigger the within-class floor."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_mdl_tier_c_density(
        base_delta=-0.05, mdl_tier_c_density=0.70,
    )
    assert adjusted == mod.MDL_TIER_C_WITHIN_CLASS_SATURATED_DELTA_FLOOR


def test_tier_c_density_threshold_boundary_at_0_30() -> None:
    """density exactly 0.30 should still trigger the across-class bonus."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_mdl_tier_c_density(
        base_delta=-0.02, mdl_tier_c_density=0.30,
    )
    assert adjusted == pytest.approx(-0.03)


# ── adjust_predicted_delta_for_composition_alpha ─────────────────────────


def test_composition_alpha_additive_no_adjustment() -> None:
    """α > 0.7 = ADDITIVE → no penalty."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_composition_alpha(
        base_delta=-0.05, composition_alpha=0.95,
    )
    assert adjusted == pytest.approx(-0.05)


def test_composition_alpha_sub_additive_halves_savings() -> None:
    """0.3 < α <= 0.7 = SUB-ADDITIVE → halve."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_composition_alpha(
        base_delta=-0.04, composition_alpha=0.5,
    )
    assert adjusted == pytest.approx(-0.02)


def test_composition_alpha_saturating_floors_delta() -> None:
    """α <= 0.3 = SATURATING → floor at -0.005."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_composition_alpha(
        base_delta=-0.05, composition_alpha=0.2,
    )
    assert adjusted == mod.COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR


def test_composition_alpha_none_no_adjustment() -> None:
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_composition_alpha(
        base_delta=-0.03, composition_alpha=None,
    )
    assert adjusted == pytest.approx(-0.03)


def test_composition_alpha_non_numeric_no_adjustment() -> None:
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_composition_alpha(
        base_delta=-0.03, composition_alpha="bad",  # type: ignore[arg-type]
    )
    assert adjusted == pytest.approx(-0.03)


def test_composition_alpha_t1_f_z3xc6_realistic_value() -> None:
    """T1-F Z3×C6 probe returned α=1.0 (structural additive). Verify no
    penalty applied at that value."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_composition_alpha(
        base_delta=-0.0457, composition_alpha=1.0,
    )
    assert adjusted == pytest.approx(-0.0457)


# ── load_substrate_composition_alpha_index ───────────────────────────────


def test_load_composition_index_missing_file(tmp_path: Path) -> None:
    mod = _load_autopilot_module()
    p = tmp_path / "no_such_matrix.json"
    idx = mod.load_substrate_composition_alpha_index(p)
    assert idx == {}


def test_load_composition_index_canonical_schema(tmp_path: Path) -> None:
    mod = _load_autopilot_module()
    p = tmp_path / "substrate_composition_matrix.json"
    p.write_text(
        json.dumps({
            "entries": {
                "z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps": [
                    {
                        "alpha": 1.0,
                        "verdict": "additive",
                        "score_claim": False,
                        "written_at_utc": "2026-05-14T16:59:36Z",
                    },
                ],
            },
        }),
        encoding="utf-8",
    )
    idx = mod.load_substrate_composition_alpha_index(p)
    assert "z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps" in idx
    assert idx["z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps"] == 1.0


def test_load_composition_index_most_recent_wins(tmp_path: Path) -> None:
    """Multiple entries per pair → most-recent timestamp wins."""
    mod = _load_autopilot_module()
    p = tmp_path / "substrate_composition_matrix.json"
    p.write_text(
        json.dumps({
            "entries": {
                "a__x__b": [
                    {
                        "alpha": 0.5,
                        "score_claim": False,
                        "written_at_utc": "2026-05-13T00:00:00Z",
                    },
                    {
                        "alpha": 0.9,
                        "score_claim": False,
                        "written_at_utc": "2026-05-14T00:00:00Z",  # newer
                    },
                ],
            },
        }),
        encoding="utf-8",
    )
    idx = mod.load_substrate_composition_alpha_index(p)
    assert idx["a__x__b"] == 0.9


def test_load_composition_index_refuses_score_claim(tmp_path: Path) -> None:
    """A score_claim=True entry must raise ValueError."""
    mod = _load_autopilot_module()
    p = tmp_path / "substrate_composition_matrix.json"
    p.write_text(
        json.dumps({
            "entries": {
                "a__x__b": [
                    {
                        "alpha": 0.5,
                        "score_claim": True,
                        "written_at_utc": "2026-05-14T00:00:00Z",
                    },
                ],
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="score_claim=True"):
        mod.load_substrate_composition_alpha_index(p)


def test_load_composition_index_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON → empty dict (silent skip)."""
    mod = _load_autopilot_module()
    p = tmp_path / "substrate_composition_matrix.json"
    p.write_text("{not valid", encoding="utf-8")
    idx = mod.load_substrate_composition_alpha_index(p)
    assert idx == {}


def test_load_composition_index_real_t1_f_artifact() -> None:
    """Smoke: the live T1-F-emitted matrix loads without error if present."""
    mod = _load_autopilot_module()
    p = REPO / ".omx" / "state" / "substrate_composition_matrix.json"
    if not p.is_file():
        pytest.skip("substrate_composition_matrix.json not present")
    idx = mod.load_substrate_composition_alpha_index(p)
    # The T1-F probe emitted z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps
    # with α=1.0. Verify it's present and parseable.
    assert isinstance(idx, dict)
    if "z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps" in idx:
        assert idx["z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps"] == 1.0


# ── apply_substrate_composition_matrix_to_candidates ──────────────────────


def test_apply_composition_matrix_populates_alpha(tmp_path: Path) -> None:
    mod = _load_autopilot_module()
    p = tmp_path / "matrix.json"
    p.write_text(
        json.dumps({
            "entries": {
                "z3__x__c6": [
                    {
                        "alpha": 0.85,
                        "score_claim": False,
                        "written_at_utc": "2026-05-14T00:00:00Z",
                    },
                ],
            },
        }),
        encoding="utf-8",
    )
    c = mod.CandidateRow(
        candidate_id="z3_x_c6_stack",
        family="stack",
        predicted_score_delta=-0.05,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
    )
    assert c.composition_alpha is None
    mod.apply_substrate_composition_matrix_to_candidates(
        [c],
        substrate_ids_by_candidate={"z3_x_c6_stack": ("z3", "c6")},
        matrix_path=p,
    )
    assert c.composition_alpha == 0.85


def test_apply_composition_matrix_handles_sorted_key_lookup(tmp_path: Path) -> None:
    """The matrix key may be stored in non-sorted order; lookup must
    handle both orderings."""
    mod = _load_autopilot_module()
    p = tmp_path / "matrix.json"
    # Stored as c6__x__z3 (reverse-alpha)
    p.write_text(
        json.dumps({
            "entries": {
                "c6__x__z3": [
                    {
                        "alpha": 0.7,
                        "score_claim": False,
                        "written_at_utc": "2026-05-14T00:00:00Z",
                    },
                ],
            },
        }),
        encoding="utf-8",
    )
    c = mod.CandidateRow(
        candidate_id="z3_x_c6_stack",
        family="stack",
        predicted_score_delta=-0.05,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
    )
    # substrate_ids passed in sorted alpha order (z3, c6 sorted → c6, z3)
    mod.apply_substrate_composition_matrix_to_candidates(
        [c],
        substrate_ids_by_candidate={"z3_x_c6_stack": ("z3", "c6")},
        matrix_path=p,
    )
    assert c.composition_alpha == 0.7


def test_apply_composition_matrix_skips_single_substrate(tmp_path: Path) -> None:
    """Single-substrate candidates leave composition_alpha = None."""
    mod = _load_autopilot_module()
    p = tmp_path / "matrix.json"
    p.write_text(
        json.dumps({"entries": {"z3__x__c6": [
            {"alpha": 0.9, "score_claim": False, "written_at_utc": "2026-05-14T00:00:00Z"}
        ]}}),
        encoding="utf-8",
    )
    c = mod.CandidateRow(
        candidate_id="solo",
        family="single",
        predicted_score_delta=-0.05,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
    )
    mod.apply_substrate_composition_matrix_to_candidates(
        [c],
        substrate_ids_by_candidate={"solo": ("z3",)},  # only one substrate
        matrix_path=p,
    )
    assert c.composition_alpha is None


def test_apply_composition_matrix_handles_empty_index(tmp_path: Path) -> None:
    mod = _load_autopilot_module()
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps({"entries": {}}), encoding="utf-8")
    c = mod.CandidateRow(
        candidate_id="z3_x_c6",
        family="stack",
        predicted_score_delta=-0.05,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
    )
    mod.apply_substrate_composition_matrix_to_candidates(
        [c],
        substrate_ids_by_candidate={"z3_x_c6": ("z3", "c6")},
        matrix_path=p,
    )
    assert c.composition_alpha is None


# ── Class-shift literature tokens retention (C1 council RETAIN decision) ──


def test_class_shift_literature_tokens_include_hafner() -> None:
    mod = _load_autopilot_module()
    assert "Hafner" in mod._CLASS_SHIFT_LITERATURE_TOKENS


def test_class_shift_literature_tokens_include_dreamer_v3() -> None:
    mod = _load_autopilot_module()
    assert "DreamerV3" in mod._CLASS_SHIFT_LITERATURE_TOKENS


def test_class_shift_literature_tokens_include_ha_schmidhuber() -> None:
    mod = _load_autopilot_module()
    assert "Ha-Schmidhuber" in mod._CLASS_SHIFT_LITERATURE_TOKENS


def test_class_shift_reward_applied_for_hafner_anchor() -> None:
    """Per C1 council `RETAIN` decision the Hafner DreamerV3 lineage stays in
    `_CLASS_SHIFT_LITERATURE_TOKENS`; per the 2026-05-14 council reconvening
    Decision 6 HALF-MEASURE the literature-anchor reward for C1-class tokens
    (Hafner / Ha-Schmidhuber / DreamerV3) is HALVED 0.01 -> 0.005."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_class_shift(
        base_delta=-0.03,
        literature_anchor="Hafner DreamerV3 2023",
    )
    # Halved bonus 0.005 subtracted → -0.03 - 0.005 = -0.035
    assert adjusted == pytest.approx(-0.035)


def test_class_shift_reward_applied_for_ha_schmidhuber_anchor() -> None:
    """Per Decision 6 HALF-MEASURE the Ha-Schmidhuber 2018 anchor receives
    the halved literature-anchor reward (0.005 instead of 0.01)."""
    mod = _load_autopilot_module()
    adjusted = mod.adjust_predicted_delta_for_class_shift(
        base_delta=-0.03,
        literature_anchor="Ha-Schmidhuber 2018 world model",
    )
    # Halved bonus 0.005 subtracted → -0.03 - 0.005 = -0.035
    assert adjusted == pytest.approx(-0.035)


# ── apply_z1_empirical_revision_to_candidate_delta stack-up ──────────────


def test_full_stack_within_class_tier_a_AND_tier_c() -> None:
    """A candidate that's within-class on BOTH Tier A and Tier C should
    have the floor applied (the stricter of the two)."""
    mod = _load_autopilot_module()
    c = mod.CandidateRow(
        candidate_id="hnerv_clone",
        family="hnerv_lc_v2",
        predicted_score_delta=-0.05,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
        mdl_density=0.99,           # Tier A within-class saturated
        mdl_tier_c_density=0.85,    # Tier C within-class
        lane_class=None,
        literature_anchor="",
    )
    rank_key = mod.apply_z1_empirical_revision_to_candidate_delta(c)
    # Both gates apply their floor; the floor value is -0.005.
    assert rank_key == mod.MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR


def test_full_stack_across_class_via_tier_c_PLUS_literature() -> None:
    """A candidate with Tier A within-class trending (0.92) but Tier C
    across-class (0.13) + class-shift literature anchor."""
    mod = _load_autopilot_module()
    c = mod.CandidateRow(
        candidate_id="c6_ibps",
        family="mdl_ibps",
        predicted_score_delta=-0.04,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
        mdl_density=0.92,           # Tier A trending → 50% penalty
        mdl_tier_c_density=0.13,    # Tier C across-class → -0.01 bonus
        lane_class="substrate_class_shift",
        literature_anchor="MDL-IBPS",
    )
    rank_key = mod.apply_z1_empirical_revision_to_candidate_delta(c)
    # Step 1: Tier A trending → -0.04 * 0.5 = -0.02
    # Step 2: Tier C across-class → -0.02 - 0.01 = -0.03
    # Step 3: class-shift lane_class reward → -0.03 - 0.02 = -0.05
    # Step 4: literature_anchor reward → -0.05 - 0.01 = -0.06
    # Step 5: composition_alpha = None → no change
    assert rank_key == pytest.approx(-0.06)


def test_full_stack_composition_saturating_floor() -> None:
    """A candidate with composition_alpha=0.2 (SATURATING) should be
    floored regardless of other adjustments."""
    mod = _load_autopilot_module()
    c = mod.CandidateRow(
        candidate_id="bad_stack",
        family="stack",
        predicted_score_delta=-0.10,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
        composition_alpha=0.2,  # SATURATING
    )
    rank_key = mod.apply_z1_empirical_revision_to_candidate_delta(c)
    assert rank_key == mod.COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR


def test_full_stack_composition_additive_no_penalty() -> None:
    """α > 0.7 (ADDITIVE) leaves the stack at full magnitude."""
    mod = _load_autopilot_module()
    c = mod.CandidateRow(
        candidate_id="good_stack",
        family="stack",
        predicted_score_delta=-0.05,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
        composition_alpha=0.95,
    )
    rank_key = mod.apply_z1_empirical_revision_to_candidate_delta(c)
    assert rank_key == pytest.approx(-0.05)


def test_full_stack_all_signals_compose_correctly() -> None:
    """Verify the canonical composition order:
       Tier A → Tier C → class-shift → composition."""
    mod = _load_autopilot_module()
    c = mod.CandidateRow(
        candidate_id="all_signals",
        family="multi",
        predicted_score_delta=-0.10,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
        mdl_density=0.50,  # Tier A: no penalty
        mdl_tier_c_density=0.13,  # Tier C: -0.01 bonus
        lane_class="cooperative_receiver",  # class-shift: -0.02
        literature_anchor="Wyner-Ziv",  # literature: -0.01
        composition_alpha=0.5,  # SUB-ADDITIVE: halve
    )
    rank_key = mod.apply_z1_empirical_revision_to_candidate_delta(c)
    # Step 1: Tier A 0.50 = no penalty → -0.10
    # Step 2: Tier C 0.13 = across → -0.10 - 0.01 = -0.11
    # Step 3: class-shift lane → -0.11 - 0.02 = -0.13
    # Step 4: literature → -0.13 - 0.01 = -0.14
    # Step 5: composition_alpha=0.5 (sub-additive) → -0.14 * 0.5 = -0.07
    assert rank_key == pytest.approx(-0.07)


# ── JSONL loader backward-compat ──────────────────────────────────────────


def test_jsonl_loader_loads_tier_c_density_field(tmp_path: Path) -> None:
    mod = _load_autopilot_module()
    p = tmp_path / "queue.jsonl"
    p.write_text(
        json.dumps({
            "candidate_id": "c6_smoke",
            "family": "mdl_ibps",
            "predicted_score_delta": -0.04,
            "expected_information_gain": 5.0,
            "estimated_dispatch_cost_usd": 2.0,
            "mdl_tier_c_density": 0.13,
        }) + "\n",
        encoding="utf-8",
    )
    rows = mod.load_candidates_from_jsonl(p)
    assert len(rows) == 1
    assert rows[0].mdl_tier_c_density == pytest.approx(0.13)


def test_jsonl_loader_loads_composition_alpha_field(tmp_path: Path) -> None:
    mod = _load_autopilot_module()
    p = tmp_path / "queue.jsonl"
    p.write_text(
        json.dumps({
            "candidate_id": "z3_x_c6_stack",
            "family": "stack",
            "predicted_score_delta": -0.05,
            "expected_information_gain": 10.0,
            "estimated_dispatch_cost_usd": 5.0,
            "composition_alpha": 0.85,
        }) + "\n",
        encoding="utf-8",
    )
    rows = mod.load_candidates_from_jsonl(p)
    assert len(rows) == 1
    assert rows[0].composition_alpha == pytest.approx(0.85)


def test_jsonl_loader_backward_compat_no_new_fields(tmp_path: Path) -> None:
    """Rows lacking the new fields use the None defaults."""
    mod = _load_autopilot_module()
    p = tmp_path / "queue.jsonl"
    p.write_text(
        json.dumps({
            "candidate_id": "legacy",
            "family": "legacy",
            "predicted_score_delta": -0.03,
            "expected_information_gain": 1.0,
            "estimated_dispatch_cost_usd": 1.0,
        }) + "\n",
        encoding="utf-8",
    )
    rows = mod.load_candidates_from_jsonl(p)
    assert rows[0].mdl_tier_c_density is None
    assert rows[0].composition_alpha is None


def test_jsonl_loader_invalid_tier_c_density_falls_back_to_none(tmp_path: Path) -> None:
    mod = _load_autopilot_module()
    p = tmp_path / "queue.jsonl"
    p.write_text(
        json.dumps({
            "candidate_id": "bad_tier_c",
            "family": "x",
            "predicted_score_delta": -0.03,
            "expected_information_gain": 1.0,
            "estimated_dispatch_cost_usd": 1.0,
            "mdl_tier_c_density": "not_a_float",
        }) + "\n",
        encoding="utf-8",
    )
    rows = mod.load_candidates_from_jsonl(p)
    assert rows[0].mdl_tier_c_density is None


# ── rank_candidates integration ───────────────────────────────────────────


def test_rank_candidates_within_class_tier_c_loses_to_across_class() -> None:
    """In ranking, a Tier C within-class candidate should rank LOWER than
    an across-class candidate even when their raw predicted_score_delta
    is identical."""
    mod = _load_autopilot_module()
    within = mod.CandidateRow(
        candidate_id="within",
        family="hnerv",
        predicted_score_delta=-0.05,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
        mdl_tier_c_density=0.85,  # within-class
    )
    across = mod.CandidateRow(
        candidate_id="across",
        family="mdl_ibps",
        predicted_score_delta=-0.05,
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
        mdl_tier_c_density=0.13,  # across-class
    )
    ranked = mod.rank_candidates(
        [within, across],
        rank_axis="predicted_score_delta",
        apply_z1_empirical_revision=True,
    )
    # `across` should be first (more-negative effective delta after bonus).
    assert ranked[0].candidate_id == "across"
    assert ranked[1].candidate_id == "within"


def test_rank_candidates_disable_z1_revision_preserves_legacy_order() -> None:
    """With apply_z1_empirical_revision=False, ranking ignores tier_c."""
    mod = _load_autopilot_module()
    within = mod.CandidateRow(
        candidate_id="within",
        family="hnerv",
        predicted_score_delta=-0.06,  # MORE negative raw
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
        mdl_tier_c_density=0.85,
    )
    across = mod.CandidateRow(
        candidate_id="across",
        family="mdl_ibps",
        predicted_score_delta=-0.05,  # less negative raw
        expected_information_gain=10.0,
        estimated_dispatch_cost_usd=5.0,
        mdl_tier_c_density=0.13,
    )
    ranked = mod.rank_candidates(
        [within, across],
        rank_axis="predicted_score_delta",
        apply_z1_empirical_revision=False,
    )
    # With Z1 revision OFF, raw predicted_score_delta wins → within first.
    assert ranked[0].candidate_id == "within"


# ── Live regression guards ────────────────────────────────────────────────


def test_live_z1_queue_v2_still_loads() -> None:
    """The live re-ranked v2 queue must still load through the updated
    JSONL reader after the Tier C / composition fields were added."""
    mod = _load_autopilot_module()
    p = REPO / ".omx" / "state" / "autopilot_candidate_queue_v2_post_z1_revision_20260514.jsonl"
    if not p.is_file():
        pytest.skip("v2 queue not present")
    rows = mod.load_candidates_from_jsonl(p)
    assert len(rows) > 0


def test_constants_remain_pinned_for_council_visibility() -> None:
    """The threshold constants must remain accessible (not renamed) so
    operator-facing review tools can introspect them."""
    mod = _load_autopilot_module()
    assert hasattr(mod, "MDL_TIER_C_WITHIN_CLASS_SATURATED_THRESHOLD")
    assert hasattr(mod, "MDL_TIER_C_WITHIN_CLASS_TRENDING_THRESHOLD")
    assert hasattr(mod, "MDL_TIER_C_ACROSS_CLASS_THRESHOLD")
    assert hasattr(mod, "MDL_TIER_C_WITHIN_CLASS_SATURATED_DELTA_FLOOR")
    assert hasattr(mod, "MDL_TIER_C_WITHIN_CLASS_TRENDING_PENALTY_FACTOR")
    assert hasattr(mod, "MDL_TIER_C_ACROSS_CLASS_BONUS")
    assert hasattr(mod, "COMPOSITION_ALPHA_ADDITIVE_THRESHOLD")
    assert hasattr(mod, "COMPOSITION_ALPHA_SUB_ADDITIVE_THRESHOLD")
    assert hasattr(mod, "COMPOSITION_ALPHA_SUB_ADDITIVE_PENALTY_FACTOR")
    assert hasattr(mod, "COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR")
