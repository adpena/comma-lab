# SPDX-License-Identifier: MIT
"""Tests for the Cable D consumers 7-14 cathedral autopilot cascade wire-in.

Slot FF 2026-05-20 — `lane_cable_d_consumers_7_14_autopilot_cascade_wire_in_20260519`.

Per `.omx/research/cable_d_wire_in_batch_landed_20260519.md` highest-EV
op-routable: wire the 6 sister Cable D per-pair canonical sidecars (NOT yet
read by the cathedral cascade) into the ranker so per-pair Pareto envelope +
Lagrangian lambda + KKT residuals + Volterra cross-terms + LoRA supervision +
coding-budget signals influence candidate ordering.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + sister Q2+Q3
v2 cascade discipline: sidecar ABSENT → 1.0× passthrough (no fake reward);
sidecar PRESENT + canonical-SCHEMA-valid + custody-clean + carries non-trivial
structural signal → 1.01× reward per sidecar (composed multiplicatively to
~1.0615× when all 6 present).

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
this is PLANNING-ONLY reweighting; never creates a score claim or dispatch
authority. Per Catalog #318 master-gradient raw-byte-authority guard: never
returns raw byte tensors — only multiplicative factors derived from canonical
sidecar SCHEMA-validated presence + structural-signal markers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_autopilot_module():
    """Load the autopilot loop as a module so we can call its internal helpers.

    Mirrors `_load_autopilot_module` from the sister sister-#817 BUCKET C
    test file to keep the test harness identical.
    """
    spec = importlib.util.spec_from_file_location(
        "autopilot_loop",
        REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("autopilot_loop", mod)
    spec.loader.exec_module(mod)
    return mod


CANONICAL_CABLE_D_SIDECARS = (
    (
        "per_pair_pareto_envelope",
        "master_gradient_consumer_per_pair_pareto_envelope_v1",
    ),
    (
        "per_pair_lagrangian_lambda_bisection",
        "master_gradient_consumer_per_pair_lambda_bisection_v1",
    ),
    (
        "per_pair_lora_supervision_signal",
        "master_gradient_consumer_per_pair_lora_supervision_v1",
    ),
    (
        "per_pair_coding_budget_allocation",
        "master_gradient_consumer_per_pair_coding_budget_v1",
    ),
    (
        "per_pair_kkt_residuals",
        "master_gradient_consumer_per_pair_kkt_residuals_v1",
    ),
    (
        "per_pair_volterra_cross_terms",
        "master_gradient_consumer_per_pair_volterra_v1",
    ),
)


def _write_canonical_sidecar(
    root: Path,
    consumer_id: str,
    schema: str,
    archive_sha256: str,
    *,
    score_claim: bool = False,
    promotion_eligible: bool = False,
    n_pairs: int = 8,
    n_bytes: int = 32,
    sha_in_payload: str | None = None,
    schema_override: str | None = None,
    utc_suffix: str = "20260520T013100",
) -> Path:
    """Stage a synthetic canonical sidecar at the canonical path."""
    root.mkdir(parents=True, exist_ok=True)
    sha_short = archive_sha256[:12]
    path = root / f"{consumer_id}_{sha_short}_{utc_suffix}.json"
    payload = {
        "schema": schema_override if schema_override is not None else schema,
        "consumer_id": consumer_id,
        "archive_sha256": sha_in_payload if sha_in_payload is not None else archive_sha256,
        "measurement_axis": "contest_cpu",
        "measurement_hardware": "linux_x86_64_cpu",
        "n_pairs": n_pairs,
        "n_bytes": n_bytes,
        "score_claim": score_claim,
        "promotion_eligible": promotion_eligible,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": f"[diagnostic; {consumer_id}; sister Cable D test fixture]",
        "interpretation_notes": "synthetic test fixture per Slot FF cascade wire-in",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ── Helper unit tests ────────────────────────────────────────────────────────


def test_constants_pinned():
    """Reward constants are pinned at the documented conservative values."""
    mod = _load_autopilot_module()
    assert mod._CABLE_D_CONSUMERS_7_14_SIDECAR_REWARD_FACTOR_PER_PRESENT == 1.01
    assert mod._CABLE_D_CONSUMERS_7_14_SIDECAR_REWARD_FACTOR_ABSENT == 1.0


def test_canonical_sidecar_list_pinned():
    """The 6 canonical sidecars are pinned with the canonical schema tags."""
    mod = _load_autopilot_module()
    assert len(mod._CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS) == 6
    actual = set(mod._CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS)
    expected = set(CANONICAL_CABLE_D_SIDECARS)
    assert actual == expected
    # Specifically: must NOT include atlas (wired separately via ITEM_7) and
    # must NOT include optimal_treatment_plan (PRIMARY in Catalog #319 CASCADE 1).
    consumer_ids = {c[0] for c in mod._CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS}
    assert "per_pair_difficulty_atlas" not in consumer_ids
    assert "per_pair_optimal_treatment_plan_via_lagrangian_dual" not in consumer_ids


def test_empty_sha_returns_passthrough():
    """Empty archive sha → 1.0× passthrough (defensive)."""
    mod = _load_autopilot_module()
    out = mod.adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars(-0.05, "")
    assert abs(out - (-0.05)) < 1e-12


def test_no_sidecars_returns_passthrough(tmp_path, monkeypatch):
    """No sidecars present → 1.0× passthrough (NO FAKE REWARD per Q2+Q3 v2 discipline)."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    out = mod.adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars(
        -0.05, "deadbeef1234567890abcdef"
    )
    assert abs(out - (-0.05)) < 1e-12


def test_factor_helper_zero_sidecars_returns_1_0(tmp_path, monkeypatch):
    """Factor helper returns 1.0 when no sidecars present."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(
        "deadbeef1234567890abcdef"
    )
    assert factor == 1.0


# ── Per-consumer read-path tests (1 per consumer; 6 tests minimum) ─────────


@pytest.mark.parametrize("consumer_id,schema", CANONICAL_CABLE_D_SIDECARS)
def test_single_consumer_sidecar_present_grants_one_step_reward(
    tmp_path, monkeypatch, consumer_id, schema
):
    """Each of the 6 canonical sidecars grants a single 1.01× reward step."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4  # 64-char synthetic sha
    _write_canonical_sidecar(tmp_path, consumer_id, schema, sha)
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(sha)
    assert factor == pytest.approx(1.01, abs=1e-12)


@pytest.mark.parametrize("consumer_id,schema", CANONICAL_CABLE_D_SIDECARS)
def test_single_consumer_sidecar_apply_to_predicted_delta(
    tmp_path, monkeypatch, consumer_id, schema
):
    """End-to-end: each consumer applies 1.01× to predicted_delta."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    _write_canonical_sidecar(tmp_path, consumer_id, schema, sha)
    out = mod.adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars(-0.05, sha)
    # negative × 1.01 = more negative = better-ranked
    assert out == pytest.approx(-0.05 * 1.01, abs=1e-12)


def test_all_six_consumer_sidecars_compose_multiplicatively(tmp_path, monkeypatch):
    """All 6 canonical sidecars present → factor = 1.01^6 = ~1.0615×."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    for consumer_id, schema in CANONICAL_CABLE_D_SIDECARS:
        _write_canonical_sidecar(tmp_path, consumer_id, schema, sha)
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(sha)
    expected = 1.01**6
    assert factor == pytest.approx(expected, abs=1e-12)
    out = mod.adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars(-0.10, sha)
    assert out == pytest.approx(-0.10 * expected, abs=1e-12)


def test_partial_subset_three_sidecars_compose_three_steps(tmp_path, monkeypatch):
    """Subset of 3 sidecars present → factor = 1.01^3."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    for consumer_id, schema in CANONICAL_CABLE_D_SIDECARS[:3]:
        _write_canonical_sidecar(tmp_path, consumer_id, schema, sha)
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(sha)
    assert factor == pytest.approx(1.01**3, abs=1e-12)


# ── Cascade-ordering tests (3) ───────────────────────────────────────────────


def test_apply_z1_empirical_revision_composes_new_cascade(tmp_path, monkeypatch):
    """The new sub-cascade is composed multiplicatively in
    apply_z1_empirical_revision_to_candidate_delta AFTER the venn v2 cascade
    + sister-817 + atlas, BEFORE realistic_stacking_correction.
    """
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    # Disable upstream venn classification sidecar so v2 venn cascade falls
    # through to passthrough (1.0×). The new Cable D cascade then applies.
    monkeypatch.setattr(
        mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", tmp_path / "venn_no_op"
    )
    sha = "abcdef1234567890" * 4
    # Stage one sister Cable D sidecar (per_pair_pareto_envelope) at sha
    _write_canonical_sidecar(
        tmp_path,
        "per_pair_pareto_envelope",
        "master_gradient_consumer_per_pair_pareto_envelope_v1",
        sha,
    )

    c = SimpleNamespace(
        predicted_score_delta=-0.10,
        mdl_density=None,
        mdl_tier_c_density=None,
        lane_class=None,
        literature_anchor="",
        notes="",
        composition_alpha=None,
        predicted_dispatch_risk=None,
        archive_sha256=sha,
    )
    monkeypatch.setattr(
        mod, "_candidate_literature_anchor_rank_reward_suppressed",
        lambda _c: False,
    )
    d = mod.apply_z1_empirical_revision_to_candidate_delta(c)
    # Expected: -0.10 → ... → -0.10 (venn passthrough) → -0.10 (sister-817
    # passthrough) → -0.10 (atlas passthrough) → -0.10 × 1.01 = -0.101 (one
    # Cable D sidecar present) → -0.101 (realistic_stacking_correction n=1)
    expected = -0.10 * 1.01
    assert d == pytest.approx(expected, abs=1e-9)


def test_lagrangian_planner_still_primary_in_v2_cascade(tmp_path, monkeypatch):
    """The Catalog #319 CASCADE 1 (Lagrangian planner REPLACE semantics) is
    NOT touched by the new sub-cascade — when an OptimalPerPairTreatmentPlan
    exists, it REPLACES predicted_delta BEFORE the new sub-cascade applies.
    """
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    # Mock the optimal-plan loader to return a plan with predicted_score_delta=-0.07
    import tac.master_gradient_consumers as mgc

    def fake_load_optimal_plan(archive_sha256, *, root=None):
        return {
            "schema": "master_gradient_consumer_optimal_per_pair_treatment_plan_v1",
            "consumer_id": "per_pair_optimal_treatment_plan_via_lagrangian_dual",
            "archive_sha256": archive_sha256,
            "predicted_score_delta": -0.07,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        mgc, "load_optimal_plan_for_archive", fake_load_optimal_plan
    )

    # Stage all 6 sister Cable D sidecars
    for consumer_id, schema in CANONICAL_CABLE_D_SIDECARS:
        _write_canonical_sidecar(tmp_path, consumer_id, schema, sha)

    # Call the v2 cascade DIRECTLY (not through full apply_z1) — Lagrangian
    # REPLACE semantics should still produce -0.07 as the venn v2 cascade
    # output. (The new Cable D cascade fires AFTER, in apply_z1_empirical_revision_to_candidate_delta.)
    venn_out = mod.adjust_predicted_delta_for_venn_classification_v2(
        -0.10, sha, panel_axis="contest_cpu"
    )
    assert venn_out == pytest.approx(-0.07, abs=1e-9)
    # The new Cable D cascade composes on top: -0.07 × 1.01^6 = ~-0.07431
    cascade_out = mod.adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars(
        venn_out, sha
    )
    assert cascade_out == pytest.approx(-0.07 * 1.01**6, abs=1e-9)


def test_cascade_passthrough_when_no_sidecars_for_archive(tmp_path, monkeypatch):
    """No sister Cable D sidecars exist for this archive → 1.0× passthrough."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    # No sidecars staged for `sha`
    out = mod.adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars(-0.05, sha)
    assert out == pytest.approx(-0.05, abs=1e-12)


# ── Canonical marker tests per Catalog #341 (3) ──────────────────────────────


def test_score_claim_true_payload_rejected(tmp_path, monkeypatch):
    """Sidecar with score_claim=True is REJECTED (phantom-score guard per
    Catalog #321/#322/#323). Returns 1.0× passthrough."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    _write_canonical_sidecar(
        tmp_path,
        "per_pair_pareto_envelope",
        "master_gradient_consumer_per_pair_pareto_envelope_v1",
        sha,
        score_claim=True,
    )
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(sha)
    assert factor == 1.0  # rejected


def test_promotion_eligible_true_payload_rejected(tmp_path, monkeypatch):
    """Sidecar with promotion_eligible=True is REJECTED (promotion-leak guard
    per Catalog #127/#317/#341). Returns 1.0× passthrough."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    _write_canonical_sidecar(
        tmp_path,
        "per_pair_pareto_envelope",
        "master_gradient_consumer_per_pair_pareto_envelope_v1",
        sha,
        promotion_eligible=True,
    )
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(sha)
    assert factor == 1.0  # rejected


def test_cross_archive_contamination_rejected(tmp_path, monkeypatch):
    """Sidecar whose payload archive_sha256 does NOT match the candidate's sha
    is REJECTED (cross-archive contamination guard). Returns 1.0× passthrough.

    Empirical anchor: a sidecar filename can be staged at sha-A's prefix while
    the JSON body claims sha-B; the structural-signal validator rejects.
    """
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha_a = "aaaa" + "1234567890abcdef" * 3 + "aaaa1234567890ab"
    sha_b = "bbbb" + "1234567890abcdef" * 3 + "bbbb1234567890ab"
    # File staged at sha_a's prefix but payload says sha_b
    _write_canonical_sidecar(
        tmp_path,
        "per_pair_pareto_envelope",
        "master_gradient_consumer_per_pair_pareto_envelope_v1",
        sha_a,
        sha_in_payload=sha_b,
    )
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(sha_a)
    assert factor == 1.0  # cross-archive contamination rejected


# ── Backwards-compat tests (2) ───────────────────────────────────────────────


def test_v1_wrapper_signature_preserved(tmp_path, monkeypatch):
    """The v1 wrapper signature (predicted_delta, archive_sha256) is preserved
    for backwards compat — existing callers + sister cascade tests still work."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    # Both v1-form callers (sister venn) and new cascade accept the same
    # (predicted_delta, archive_sha256) shape
    a = mod.adjust_predicted_delta_for_venn_classification(-0.05, sha)
    b = mod.adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars(-0.05, sha)
    # Both return float; both are -0.05 (no sidecars staged)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert a == pytest.approx(-0.05, abs=1e-12)
    assert b == pytest.approx(-0.05, abs=1e-12)


def test_existing_venn_v2_tests_still_pass_pattern(tmp_path, monkeypatch):
    """The new Cable D cascade does NOT touch the existing venn v2 cascade —
    confirms by asserting venn v2 behavior is unchanged."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    # Stage all 6 sister Cable D sidecars; venn v2 should STILL passthrough
    # because no venn classification sidecar exists for sha
    for consumer_id, schema in CANONICAL_CABLE_D_SIDECARS:
        _write_canonical_sidecar(tmp_path, consumer_id, schema, sha)
    venn_out = mod.adjust_predicted_delta_for_venn_classification_v2(-0.05, sha)
    # venn v2 ALONE returns -0.05 (cascade 3 passthrough)
    assert venn_out == pytest.approx(-0.05, abs=1e-12)


# ── Edge-case tests (2) ──────────────────────────────────────────────────────


def test_corrupt_sidecar_returns_passthrough(tmp_path, monkeypatch):
    """Malformed JSON sidecar should NOT crash; returns 1.0× per Catalog #138
    strict-load fail-closed sister discipline."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    sha_short = sha[:12]
    bad_path = (
        tmp_path
        / f"per_pair_pareto_envelope_{sha_short}_20260520T013100.json"
    )
    bad_path.write_text("not valid json {{{", encoding="utf-8")
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(sha)
    assert factor == 1.0  # safe passthrough


def test_trivial_signal_sidecar_rejected(tmp_path, monkeypatch):
    """Sidecar with both n_pairs=0 AND n_bytes=0 (no structural signal) is
    REJECTED. Returns 1.0× passthrough."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    _write_canonical_sidecar(
        tmp_path,
        "per_pair_pareto_envelope",
        "master_gradient_consumer_per_pair_pareto_envelope_v1",
        sha,
        n_pairs=0,
        n_bytes=0,
    )
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(sha)
    assert factor == 1.0  # trivial signal rejected


def test_wrong_schema_tag_rejected(tmp_path, monkeypatch):
    """Sidecar with a schema tag from a DIFFERENT consumer is REJECTED
    (orphan-sidecar guard). Returns 1.0× passthrough."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    _write_canonical_sidecar(
        tmp_path,
        "per_pair_pareto_envelope",
        "master_gradient_consumer_per_pair_pareto_envelope_v1",
        sha,
        schema_override="some_other_consumer_schema_v1",
    )
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(sha)
    assert factor == 1.0  # orphan-sidecar rejected


def test_non_dict_payload_rejected(tmp_path, monkeypatch):
    """Sidecar whose JSON payload is a list (not a dict) is REJECTED."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "abcdef1234567890" * 4
    sha_short = sha[:12]
    bad_path = (
        tmp_path
        / f"per_pair_pareto_envelope_{sha_short}_20260520T013100.json"
    )
    bad_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    factor = mod._cable_d_consumers_7_14_sidecar_reward_factor(sha)
    assert factor == 1.0


# ── Sister-regression tests (2) ──────────────────────────────────────────────


def test_sister_817_cascade_still_callable():
    """Sister sister-#817 cascade (adjust_predicted_delta_for_per_pair_sister_817_sidecars)
    remains callable + has same signature — the Cable D cascade does NOT
    break sister wire-ins."""
    mod = _load_autopilot_module()
    assert callable(mod.adjust_predicted_delta_for_per_pair_sister_817_sidecars)
    # Sister signature: (predicted_delta, archive_sha256)
    out = mod.adjust_predicted_delta_for_per_pair_sister_817_sidecars(
        -0.05, "deadbeef1234567890abcdef"
    )
    assert isinstance(out, float)


def test_atlas_cascade_still_callable():
    """Sister atlas cascade (adjust_predicted_delta_for_per_pair_difficulty_atlas)
    remains callable with the same signature."""
    mod = _load_autopilot_module()
    assert callable(mod.adjust_predicted_delta_for_per_pair_difficulty_atlas)
    out = mod.adjust_predicted_delta_for_per_pair_difficulty_atlas(
        -0.05, "deadbeef1234567890abcdef", panel_axis="contest_cpu"
    )
    assert isinstance(out, float)


# ── Sister-checkpoint regression: chain-position guard ────────────────────


def test_cable_d_cascade_runs_after_atlas_before_realistic_stacking(tmp_path, monkeypatch):
    """The new Cable D cascade is wired AFTER atlas (line 1107) and BEFORE
    realistic_stacking_correction (line 1124) per the chain comment.

    Verified by tracing the source-text ordering directly.
    """
    autopilot_path = REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py"
    source = autopilot_path.read_text(encoding="utf-8")
    atlas_pos = source.find("adjust_predicted_delta_for_per_pair_difficulty_atlas(\n        d,")
    cable_d_pos = source.find(
        "adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars("
    )
    realistic_pos = source.find("adjust_predicted_delta_for_realistic_stacking_correction(")
    assert atlas_pos != -1, "atlas cascade not found in source"
    assert cable_d_pos != -1, "Cable D cascade not found in source"
    assert realistic_pos != -1, "realistic_stacking_correction not found in source"
    assert atlas_pos < cable_d_pos < realistic_pos, (
        f"chain ordering violated: atlas={atlas_pos} cable_d={cable_d_pos} "
        f"realistic={realistic_pos}"
    )


# ── Catalog #185 sister regression ───────────────────────────────────────────


def test_cascade_callable_via_module_globals():
    """Per Catalog #185 sister discipline: the gate function MUST be callable
    via the autopilot module's globals so future drift detection works."""
    mod = _load_autopilot_module()
    # The cascade fn IS a public surface of the autopilot module
    assert hasattr(mod, "adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars")
    assert callable(
        getattr(mod, "adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars")
    )
    assert hasattr(mod, "_cable_d_consumers_7_14_sidecar_reward_factor")
    assert callable(
        getattr(mod, "_cable_d_consumers_7_14_sidecar_reward_factor")
    )
    assert hasattr(mod, "_CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS")
    assert hasattr(mod, "_CABLE_D_CONSUMERS_7_14_SIDECAR_REWARD_FACTOR_PER_PRESENT")
    assert hasattr(mod, "_CABLE_D_CONSUMERS_7_14_SIDECAR_REWARD_FACTOR_ABSENT")
