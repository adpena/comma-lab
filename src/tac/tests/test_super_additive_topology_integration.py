# SPDX-License-Identifier: MIT
"""Tests for SUPER_ADDITIVE topology integration (lane_super_additive_..._20260517).

Per `feedback_super_additive_lane_g_v3_siren_topology_integration_landed_
20260517.md` + the Q6 OP-3 extended sweep finding (2026-05-17).

Covers:
- v2 cascade `adjust_predicted_delta_for_composition_alpha_v2` correctness
  across all 5 bands (SATURATING / SUB_ADDITIVE / ADDITIVE / SUPER_ADDITIVE / None)
- SUPER_ADDITIVE reward bounded at 2.0× (no runaway)
- Empirical α=4.74 produces reward factor 2.0× (cap)
- SUPER_ADDITIVE only rewards when base_delta < 0 (no false-promotion on +Δ)
- v2 is strict superset of v1 for alpha <= 1.05
- Sister regression: existing v1 composition_alpha tests still pass
- Mechanism investigation tool produces hypothesis report
- canonical_inventory + matrix posterior schema invariants
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"


# ── Load the cathedral autopilot module directly (it's a tools/ file). ──────


_AUTOPILOT_MODULE_NAME = "cathedral_autopilot_autonomous_loop_test_load"
_INVESTIGATION_MODULE_NAME = "investigate_super_additive_lane_g_v3_siren_mechanism_test_load"


def _load_autopilot_module():
    """Load tools/cathedral_autopilot_autonomous_loop.py as a module.

    Pre-registers the module in sys.modules BEFORE executing because @dataclass
    introspects ``sys.modules.get(cls.__module__).__dict__`` during processing.
    """
    if _AUTOPILOT_MODULE_NAME in sys.modules:
        return sys.modules[_AUTOPILOT_MODULE_NAME]
    autopilot_path = TOOLS_DIR / "cathedral_autopilot_autonomous_loop.py"
    spec = importlib.util.spec_from_file_location(
        _AUTOPILOT_MODULE_NAME,
        autopilot_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_AUTOPILOT_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_AUTOPILOT_MODULE_NAME, None)
        raise
    return module


def _load_investigation_module():
    """Load tools/investigate_super_additive_lane_g_v3_siren_mechanism.py as a module."""
    if _INVESTIGATION_MODULE_NAME in sys.modules:
        return sys.modules[_INVESTIGATION_MODULE_NAME]
    invest_path = TOOLS_DIR / "investigate_super_additive_lane_g_v3_siren_mechanism.py"
    spec = importlib.util.spec_from_file_location(
        _INVESTIGATION_MODULE_NAME,
        invest_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_INVESTIGATION_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_INVESTIGATION_MODULE_NAME, None)
        raise
    return module


# ── v2 cascade correctness tests ─────────────────────────────────────────────


def test_v2_cascade_super_additive_alpha_4_74_caps_reward_at_2x() -> None:
    """Empirical α=4.74 from Q6 OP-3 sweep produces reward factor 2.0× (cap)."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    base = -0.01
    # alpha=4.74 -> clamped to reward cap 2.0 -> base * 2.0 = -0.02
    assert fn(base, 4.743780383185587) == pytest.approx(-0.02, rel=1e-9)


def test_v2_cascade_super_additive_alpha_2_0_at_cap() -> None:
    """α=2.0 (exactly at cap) produces reward factor 2.0×."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    assert fn(-0.01, 2.0) == pytest.approx(-0.02, rel=1e-9)


def test_v2_cascade_super_additive_alpha_1_5_intermediate_reward() -> None:
    """α=1.5 (intermediate) produces reward factor 1.5×."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    assert fn(-0.01, 1.5) == pytest.approx(-0.015, rel=1e-9)


def test_v2_cascade_super_additive_alpha_1_05_threshold_boundary() -> None:
    """α=1.05 is exactly the threshold; should be ADDITIVE (no SUPER reward)."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    # alpha == 1.05 is NOT > 1.05 -> falls into ADDITIVE branch -> no adjustment
    assert fn(-0.01, 1.05) == pytest.approx(-0.01, rel=1e-9)


def test_v2_cascade_super_additive_alpha_just_above_threshold() -> None:
    """α slightly > 1.05 (e.g. 1.10) enters SUPER_ADDITIVE branch with reward factor 1.10×."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    assert fn(-0.01, 1.10) == pytest.approx(-0.011, rel=1e-9)


def test_v2_cascade_super_additive_alpha_runaway_capped() -> None:
    """α=100 (false-signal runaway) is capped at 2.0× reward, no runaway promotion."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    assert fn(-0.01, 100.0) == pytest.approx(-0.02, rel=1e-9)


def test_v2_cascade_super_additive_only_rewards_negative_base() -> None:
    """SUPER_ADDITIVE reward only fires when base_delta < 0 (candidate improves score)."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    # base_delta >= 0 means the candidate is NOT improving; no reward applies.
    assert fn(0.0, 4.74) == 0.0
    assert fn(0.01, 4.74) == pytest.approx(0.01, rel=1e-9)


def test_v2_cascade_additive_alpha_0_8_no_adjustment() -> None:
    """α=0.8 (ADDITIVE band) -> no adjustment."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    assert fn(-0.01, 0.8) == pytest.approx(-0.01, rel=1e-9)


def test_v2_cascade_sub_additive_alpha_0_5_halves() -> None:
    """α=0.5 (SUB_ADDITIVE band) -> 50% penalty."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    assert fn(-0.01, 0.5) == pytest.approx(-0.005, rel=1e-9)


def test_v2_cascade_saturating_alpha_0_2_floors_at_minus_005() -> None:
    """α=0.2 (SATURATING band) -> floor at -0.005."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    # base_delta = -0.01 floored at -0.005 (more-negative loses).
    assert fn(-0.01, 0.2) == pytest.approx(-0.005, rel=1e-9)
    # base_delta = -0.001 (already above floor) is unchanged.
    assert fn(-0.001, 0.2) == pytest.approx(-0.001, rel=1e-9)


def test_v2_cascade_none_alpha_no_adjustment() -> None:
    """alpha=None -> no adjustment."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    assert fn(-0.01, None) == pytest.approx(-0.01, rel=1e-9)


def test_v2_cascade_non_finite_alpha_rejected() -> None:
    """alpha = inf or nan rejected (no adjustment) per conservative-on-bad-input rule."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    assert fn(-0.01, float("inf")) == pytest.approx(-0.01, rel=1e-9)
    assert fn(-0.01, float("nan")) == pytest.approx(-0.01, rel=1e-9)
    assert fn(-0.01, float("-inf")) == pytest.approx(-0.01, rel=1e-9)


def test_v2_cascade_invalid_type_alpha_rejected() -> None:
    """alpha = non-numeric (str, list) rejected (no adjustment)."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    assert fn(-0.01, "not_a_number") == pytest.approx(-0.01, rel=1e-9)


# ── v2 is strict superset of v1 for alpha <= 1.05 ────────────────────────────


def test_v2_strict_superset_of_v1_below_super_threshold() -> None:
    """For alpha <= 1.05, v2 returns identical value as v1."""
    ap = _load_autopilot_module()
    v1 = ap.adjust_predicted_delta_for_composition_alpha
    v2 = ap.adjust_predicted_delta_for_composition_alpha_v2
    base = -0.01
    for alpha in [None, 0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.05]:
        v1_result = v1(base, alpha)
        v2_result = v2(base, alpha)
        assert v1_result == pytest.approx(v2_result, rel=1e-9), (
            f"v1/v2 mismatch at alpha={alpha}: v1={v1_result} v2={v2_result}"
        )


def test_v1_does_not_recognize_super_additive() -> None:
    """v1 ABSORBS alpha > 1.05 as ADDITIVE (no extra reward); v2 does NOT."""
    ap = _load_autopilot_module()
    v1 = ap.adjust_predicted_delta_for_composition_alpha
    v2 = ap.adjust_predicted_delta_for_composition_alpha_v2
    # alpha=4.74: v1 returns base unchanged; v2 returns base * 2.0 (reward).
    assert v1(-0.01, 4.74) == pytest.approx(-0.01, rel=1e-9)
    assert v2(-0.01, 4.74) == pytest.approx(-0.02, rel=1e-9)


# ── v2 constants pinned ──────────────────────────────────────────────────────


def test_v2_constants_pinned() -> None:
    """v2 cascade thresholds + reward bounds are at canonical values."""
    ap = _load_autopilot_module()
    assert ap.COMPOSITION_ALPHA_SUPER_ADDITIVE_THRESHOLD == 1.05
    assert ap.COMPOSITION_ALPHA_SUPER_ADDITIVE_REWARD_CAP == 2.0
    assert ap.COMPOSITION_ALPHA_SUPER_ADDITIVE_REWARD_FLOOR == 1.0
    # Sister v1 constants unchanged.
    assert ap.COMPOSITION_ALPHA_ADDITIVE_THRESHOLD == 0.7
    assert ap.COMPOSITION_ALPHA_SUB_ADDITIVE_THRESHOLD == 0.3
    assert ap.COMPOSITION_ALPHA_SUB_ADDITIVE_PENALTY_FACTOR == 0.5
    assert ap.COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR == -0.005


# ── Mechanism investigation tool tests ───────────────────────────────────────


def test_mechanism_investigation_produces_h2_byte_identity_verdict() -> None:
    """Live invocation: lane_g_v3 + siren renderer.bin are byte-identical."""
    inv = _load_investigation_module()
    verdict = inv.investigate()
    # The canonical files on disk are byte-identical sha256 08f12d72... per
    # the SIREN smoke timeout placeholder-copy artifact.
    assert verdict.candidate_a.exists is True
    assert verdict.candidate_b.exists is True
    assert verdict.candidate_a.sha256 == verdict.candidate_b.sha256
    assert verdict.candidate_a.sha256 == (
        "08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529"
    )
    assert verdict.primary_hypothesis == "H2"
    assert verdict.h2_byte_identity_artifact_supported is True
    assert verdict.h1_byte_level_structure_sharing_supported is False
    assert verdict.byte_identity_holds is True
    assert verdict.sha256_identity_holds is True
    # Score/promotion fail-closed per Catalog #127.
    assert verdict.score_claim is False
    assert verdict.promotion_eligible is False
    assert verdict.ready_for_exact_eval_dispatch is False


def test_mechanism_investigation_kl_divergence_is_zero_for_identical_files() -> None:
    """Identical byte histograms produce KL divergence = 0."""
    inv = _load_investigation_module()
    hist_a = {0: 10, 1: 20, 2: 30}
    hist_b = {0: 10, 1: 20, 2: 30}
    kl = inv.compute_byte_histogram_kl_divergence(hist_a, hist_b)
    assert kl == pytest.approx(0.0, abs=1e-12)


def test_mechanism_investigation_overlap_fraction_is_one_for_identical_histograms() -> None:
    """Identical byte histograms produce overlap fraction = 1.0."""
    inv = _load_investigation_module()
    hist_a = {0: 10, 1: 20, 2: 30}
    hist_b = {0: 10, 1: 20, 2: 30}
    overlap = inv.compute_byte_histogram_overlap_fraction(hist_a, hist_b)
    assert overlap == pytest.approx(1.0, rel=1e-9)


def test_mechanism_investigation_overlap_zero_for_disjoint_histograms() -> None:
    """Disjoint byte histograms (no shared byte values) produce overlap = 0.0."""
    inv = _load_investigation_module()
    hist_a = {0: 100}
    hist_b = {255: 100}
    overlap = inv.compute_byte_histogram_overlap_fraction(hist_a, hist_b)
    assert overlap == pytest.approx(0.0, abs=1e-12)


def test_mechanism_investigation_writes_markdown_report() -> None:
    """CLI subprocess writes a markdown report at the expected path."""
    inv = _load_investigation_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_mechanism_report.md"
        rc = inv.main(["--output", str(output_path)])
        assert rc == 0
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        # Must include the primary hypothesis verdict + canonical sha256.
        assert "primary hypothesis" in content.lower()
        assert "H2" in content
        assert (
            "08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529" in content
        )
        # Must include all 9-dim checklist evidence per Catalog #294.
        assert "## 9-dimension success checklist evidence" in content
        # Must include cargo-cult audit section per Catalog #303.
        assert "## Cargo-cult audit per assumption" in content
        # Must include observability surface section per Catalog #305.
        assert "## Observability surface" in content


def test_mechanism_investigation_writes_json_verdict() -> None:
    """CLI subprocess writes a JSON verdict with the canonical fields."""
    inv = _load_investigation_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = Path(tmpdir) / "test_mechanism_report.md"
        json_path = Path(tmpdir) / "test_mechanism_verdict.json"
        rc = inv.main(["--output", str(md_path), "--json-out", str(json_path)])
        assert rc == 0
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Verify canonical fields.
        assert data["primary_hypothesis"] == "H2"
        assert data["sha256_identity_holds"] is True
        assert data["score_claim"] is False
        assert data["promotion_eligible"] is False
        assert "candidate_a" in data
        assert "candidate_b" in data
        assert data["candidate_a"]["sha256"] == data["candidate_b"]["sha256"]


# ── Canonical posterior surface (matrix.json) regression tests ───────────────


def test_matrix_posterior_includes_super_additive_row_with_false_signal_blockers() -> None:
    """The canonical matrix posterior has the SUPER_ADDITIVE row with blockers."""
    matrix_path = REPO_ROOT / ".omx" / "state" / "substrate_composition_matrix.json"
    assert matrix_path.is_file(), "Canonical matrix posterior must exist at landing"
    data = json.loads(matrix_path.read_text(encoding="utf-8"))
    entries = data.get("entries", {})
    # Pair key is alphabetically sorted per canonical loader convention.
    pair_key = "lane_g_v3_renderer__x__siren_renderer"
    assert pair_key in entries, f"Missing canonical pair_key {pair_key!r} in matrix posterior"
    rows = entries[pair_key]
    assert len(rows) >= 1, "Must have at least one row for the SUPER_ADDITIVE pair"
    # Find the row matching alpha=4.74 (our landing row).
    target_row = None
    for row in rows:
        if abs(row.get("alpha", 0) - 4.743780383185587) < 1e-9:
            target_row = row
            break
    assert target_row is not None, "Must have the alpha=4.74 SUPER_ADDITIVE row landed"
    # Verify FALSE-SIGNAL discipline.
    assert target_row["score_claim"] is False
    assert target_row["promotion_eligible"] is False
    assert target_row["ready_for_exact_eval_dispatch"] is False
    assert "FALSE_SIGNAL" in target_row["verdict"]
    blockers = target_row.get("result_review_blockers", [])
    # At least 5 blockers documenting the false-signal class.
    assert len(blockers) >= 5
    # Specific anchors must be present.
    blocker_text = " ".join(blockers).lower()
    assert "byte_identity" in blocker_text or "byte-identity" in blocker_text
    assert "siren_smoke" in blocker_text or "siren-smoke" in blocker_text
    assert "08f12d72" in blocker_text


def test_matrix_posterior_loader_consumes_super_additive_row() -> None:
    """The autopilot's load_substrate_composition_alpha_index reads the SUPER_ADDITIVE row."""
    ap = _load_autopilot_module()
    matrix_path = REPO_ROOT / ".omx" / "state" / "substrate_composition_matrix.json"
    alpha_index = ap.load_substrate_composition_alpha_index(matrix_path)
    pair_key = "lane_g_v3_renderer__x__siren_renderer"
    assert pair_key in alpha_index
    # Most-recent wins; the landing row has alpha=4.74.
    assert alpha_index[pair_key] == pytest.approx(4.743780383185587, rel=1e-9)


# ── End-to-end regression: SUPER_ADDITIVE pair routed through v2 cascade ─────


def test_end_to_end_super_additive_pair_routes_through_v2_cascade() -> None:
    """For a candidate stacked from the SUPER_ADDITIVE pair, v2 cascade rewards correctly."""
    ap = _load_autopilot_module()
    fn = ap.adjust_predicted_delta_for_composition_alpha_v2
    # Simulated SUPER_ADDITIVE candidate: predicted delta -0.005, alpha=4.74.
    base = -0.005
    adjusted = fn(base, 4.743780383185587)
    # Reward factor 2.0× (capped), so adjusted = -0.010.
    assert adjusted == pytest.approx(-0.010, rel=1e-9)


# ── Sister regression: existing v1 composition_alpha tests still pass ───────


def test_v1_composition_alpha_existing_contract_unchanged() -> None:
    """Existing v1 contract is untouched (back-compat invariant)."""
    ap = _load_autopilot_module()
    v1 = ap.adjust_predicted_delta_for_composition_alpha
    # ADDITIVE: alpha > 0.7 -> no adjustment
    assert v1(-0.01, 0.8) == pytest.approx(-0.01, rel=1e-9)
    # SUB-ADDITIVE: 0.3 < alpha <= 0.7 -> halve
    assert v1(-0.01, 0.5) == pytest.approx(-0.005, rel=1e-9)
    # SATURATING: alpha <= 0.3 -> floor
    assert v1(-0.01, 0.2) == pytest.approx(-0.005, rel=1e-9)
    # None -> no adjustment
    assert v1(-0.01, None) == pytest.approx(-0.01, rel=1e-9)
