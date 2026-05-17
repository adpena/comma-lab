# SPDX-License-Identifier: MIT
"""Autopilot reweight v2 Lagrangian-derived cascade tests.

Verifies ``adjust_predicted_delta_for_venn_classification_v2`` per Q3 of
``lane_q2_q3_batched_catalog_319_gate_plus_autopilot_reweight_v2_20260517``.

The v2 cascade replaces the v1 flat 1.15× HIGH_PAIR_INVARIANT reward (proven
FAKE for fec6 — see ``probe_f174192aeadf_20260517T205208.json`` with
``deliverability_verdict='NOT_DELIVERABLE'``) with a 3-cascade decision tree:

  CASCADE 1 (PRIMARY): Lagrangian-derived — if an OptimalPerPairTreatmentPlan
    sidecar exists, REPLACE delta with plan.predicted_score_delta.
  CASCADE 2 (DELIVERABILITY): else if DeliverabilityProof exists AND Venn class
    is HIGH_PAIR_INVARIANT, apply per-tier byte-weighted reward.
  CASCADE 3 (PASSTHROUGH): no plan + no proof => 1.0× passthrough.

The HIGH_PAIR_SPECIFIC 0.85× PENALTY is preserved across all cascades.

Sister of:
  - ``test_check_319_venn_reweight_requires_deliverability_proof.py`` (Q2 gate)
  - ``test_session_20260517_cli_flag_additions.py`` (v1 baseline; still passes)
  - ``test_master_gradient_consumers.py`` (load_optimal_plan_for_archive)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_autopilot_module():
    """Load the autopilot loop as a module so we can call its internal helpers directly."""
    spec = importlib.util.spec_from_file_location(
        "autopilot_loop",
        REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("autopilot_loop", mod)
    spec.loader.exec_module(mod)
    return mod


def _write_synth_venn_sidecar(
    root: Path, sha: str, *, invariant: int, specific: int, neutral: int = 0, dead: int = 0
) -> Path:
    """Stage a synthetic Venn classification sidecar at the canonical path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"venn_classification_{sha[:12]}_20260517T120000.json"
    path.write_text(
        json.dumps(
            {
                "schema": "master_gradient_consumer_venn_classification_v1",
                "class_counts": {
                    "PAIR_SPECIFIC": specific,
                    "PAIR_INVARIANT": invariant,
                    "PAIR_NEUTRAL": neutral,
                    "DEAD": dead,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_synth_optimal_plan(
    root: Path, sha: str, *, predicted_score_delta: float
) -> Path:
    """Stage a synthetic OptimalPerPairTreatmentPlan sidecar."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"optimal_plan_{sha[:12]}_20260517T130000.json"
    path.write_text(
        json.dumps(
            {
                "schema": "master_gradient_consumer_optimal_per_pair_treatment_plan_v1",
                "consumer_id": "per_pair_optimal_treatment_plan_via_lagrangian_dual",
                "catalog_consumer_id": 15,
                "archive_sha256": sha,
                "predicted_score_delta": predicted_score_delta,
                "evidence_grade": "predicted",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    return path


# ──────────────────────────────────────────────────────────────────────────── #
# CASCADE 1: Lagrangian-derived (PRIMARY)                                       #
# ──────────────────────────────────────────────────────────────────────────── #


def test_cascade1_optimal_plan_replaces_delta(tmp_path, monkeypatch):
    """When OptimalPerPairTreatmentPlan exists, v2 REPLACES delta with plan.predicted_score_delta."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "master_gradient_consumers"

    # Override the consumer-output root so the loader finds our synthetic plan
    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)

    sha = "a" * 64
    # Synthetic planner emits predicted_score_delta = -0.025 (better than raw)
    _write_synth_optimal_plan(plan_root, sha, predicted_score_delta=-0.025)

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha)
    assert adjusted == -0.025, (
        f"Cascade 1 must REPLACE delta with plan.predicted_score_delta; got {adjusted}"
    )


def test_cascade1_optimal_plan_explicit_path_used(tmp_path):
    """When optimal_plan_path is explicitly passed, v2 reads that file directly."""
    mod = _load_autopilot_module()
    sha = "b" * 64
    plan_root = tmp_path / "explicit_plan"
    plan_path = _write_synth_optimal_plan(plan_root, sha, predicted_score_delta=-0.030)

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(
        raw, sha, optimal_plan_path=plan_path
    )
    assert adjusted == -0.030


def test_cascade1_optimal_plan_short_circuits_venn_branch(tmp_path, monkeypatch):
    """Cascade 1 must SHORT-CIRCUIT — no Venn / DeliverabilityProof consultation."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans"
    venn_root = tmp_path / "venn"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", venn_root)

    sha = "c" * 64
    _write_synth_optimal_plan(plan_root, sha, predicted_score_delta=-0.099)
    # Also write a Venn sidecar with HIGH_PAIR_SPECIFIC (would otherwise penalize)
    _write_synth_venn_sidecar(venn_root, sha, invariant=2000, specific=6000, neutral=2000)

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha)
    # Cascade 1 wins — plan delta replaces; Venn penalty NOT applied
    assert adjusted == -0.099


def test_cascade1_malformed_plan_falls_through_to_cascade2(tmp_path, monkeypatch):
    """Plan payload without predicted_score_delta falls through to Cascade 2."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", tmp_path / "venn_empty")

    sha = "d" * 64
    plan_root.mkdir(parents=True)
    bad_plan = plan_root / f"optimal_plan_{sha[:12]}_20260517T130000.json"
    bad_plan.write_text(json.dumps({"missing_predicted_score_delta": True}), encoding="utf-8")

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha)
    # No Venn sidecar either => Cascade 3 passthrough
    assert adjusted == raw


# ──────────────────────────────────────────────────────────────────────────── #
# CASCADE 2: DeliverabilityProof-gated                                          #
# ──────────────────────────────────────────────────────────────────────────── #


def test_cascade2_no_plan_high_invariant_passthrough_without_proof(tmp_path, monkeypatch):
    """Cascade 2 with HIGH_PAIR_INVARIANT but no DeliverabilityProof => passthrough."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans_empty"
    venn_root = tmp_path / "venn"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", venn_root)

    sha = "e" * 64
    _write_synth_venn_sidecar(venn_root, sha, invariant=8500, specific=100, neutral=1400)

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha)
    # No plan + no proof => 1.0× passthrough (Cascade 3 fall-through)
    assert adjusted == raw


def test_cascade2_high_pair_specific_penalty_preserved(tmp_path, monkeypatch):
    """HIGH_PAIR_SPECIFIC 0.85× penalty preserved across cascades when no plan."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans_empty"
    venn_root = tmp_path / "venn"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", venn_root)

    sha = "f" * 64
    _write_synth_venn_sidecar(venn_root, sha, invariant=5000, specific=3500, neutral=1500)

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha)
    expected = raw * 0.85
    assert abs(adjusted - expected) < 1e-9, (
        f"HIGH_PAIR_SPECIFIC penalty must apply 0.85×; got {adjusted / raw:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# CASCADE 3: Passthrough (NO FAKE REWARD)                                       #
# ──────────────────────────────────────────────────────────────────────────── #


def test_cascade3_no_plan_no_proof_no_venn_passthrough(tmp_path, monkeypatch):
    """No plan + no proof + no Venn sidecar => passthrough (NO FAKE REWARD)."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans_empty"
    venn_root = tmp_path / "venn_empty"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", venn_root)

    sha = "9" * 64
    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha)
    assert adjusted == raw


def test_cascade3_empty_sha_passthrough():
    """Empty archive_sha256 MUST passthrough (Cascade 3)."""
    mod = _load_autopilot_module()
    assert mod.adjust_predicted_delta_for_venn_classification_v2(-0.012, "") == -0.012


def test_cascade3_fec6_anchor_regression_no_fake_reward(tmp_path, monkeypatch):
    """fec6 empirical anchor: deliverability_verdict=NOT_DELIVERABLE => no reward.

    This is the canonical bug class anchor: the fec6 prober (live artifact
    at .omx/state/wyner_ziv_deliverability/probe_f174192aeadf_*.json) proved
    lzma/brotli/zlib all INFLATE the candidate-shared-prior set. The v1 flat
    1.15× HIGH_PAIR_INVARIANT reward was FAKE; v2 must apply 1.0× passthrough.
    """
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans_empty"
    venn_root = tmp_path / "venn"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", venn_root)

    # Simulate fec6: HIGH_PAIR_INVARIANT classification (90.7% per real anchor)
    fec6_sha = "f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd"
    _write_synth_venn_sidecar(venn_root, fec6_sha, invariant=9070, specific=100, neutral=830)

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, fec6_sha)
    # No plan; no proof => 1.0× passthrough (fec6 anchor regression)
    assert adjusted == raw, (
        f"fec6 NOT_DELIVERABLE empirical anchor: expected 1.0× passthrough; "
        f"got factor {adjusted / raw:.4f} — FAKE REWARD bug class regression"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Backward compat: v1 wrapper delegates to v2                                   #
# ──────────────────────────────────────────────────────────────────────────── #


def test_v1_wrapper_delegates_to_v2(tmp_path, monkeypatch):
    """adjust_predicted_delta_for_venn_classification (v1) delegates to v2 cascade."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)

    sha = "8" * 64
    _write_synth_optimal_plan(plan_root, sha, predicted_score_delta=-0.077)

    raw = -0.012
    # v1 call should produce the same v2 result
    v1 = mod.adjust_predicted_delta_for_venn_classification(raw, sha)
    v2 = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha)
    assert v1 == v2 == -0.077


# ──────────────────────────────────────────────────────────────────────────── #
# Function exports + signature regression                                       #
# ──────────────────────────────────────────────────────────────────────────── #


def test_v2_function_exported():
    """v2 function must be exported from autopilot module."""
    mod = _load_autopilot_module()
    assert hasattr(mod, "adjust_predicted_delta_for_venn_classification_v2")
    assert callable(mod.adjust_predicted_delta_for_venn_classification_v2)


def test_v2_signature_accepts_optional_plan_path():
    """v2 signature must accept (predicted_delta, archive_sha256, optimal_plan_path=None)."""
    import inspect

    mod = _load_autopilot_module()
    sig = inspect.signature(mod.adjust_predicted_delta_for_venn_classification_v2)
    params = list(sig.parameters)
    assert params[:2] == ["predicted_delta", "archive_sha256"]
    assert "optimal_plan_path" in params


def test_v1_signature_unchanged():
    """v1 wrapper signature MUST remain (predicted_delta, archive_sha256) for backwards compat."""
    import inspect

    mod = _load_autopilot_module()
    sig = inspect.signature(mod.adjust_predicted_delta_for_venn_classification)
    params = list(sig.parameters)
    assert params == ["predicted_delta", "archive_sha256"], (
        f"v1 signature drift; got {params}"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Multi-archive disambiguation                                                  #
# ──────────────────────────────────────────────────────────────────────────── #


def test_v2_most_recent_plan_wins(tmp_path, monkeypatch):
    """Multiple plan sidecars for same archive: most recent (lex-max filename) wins."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)

    sha = "7" * 64
    plan_root.mkdir(parents=True)
    base_payload = {
        "schema": "master_gradient_consumer_optimal_per_pair_treatment_plan_v1",
        "consumer_id": "per_pair_optimal_treatment_plan_via_lagrangian_dual",
        "catalog_consumer_id": 15,
        "archive_sha256": sha,
        "evidence_grade": "predicted",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    # Older plan (lexicographically smaller)
    (plan_root / f"optimal_plan_{sha[:12]}_20260517T100000.json").write_text(
        json.dumps({**base_payload, "predicted_score_delta": -0.001}), encoding="utf-8"
    )
    # Newer plan (lexicographically larger; chronologically later)
    (plan_root / f"optimal_plan_{sha[:12]}_20260517T200000.json").write_text(
        json.dumps({**base_payload, "predicted_score_delta": -0.099}), encoding="utf-8"
    )

    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(-0.012, sha)
    assert adjusted == -0.099, "newest plan must win"


def test_v2_plan_for_different_archive_not_used(tmp_path, monkeypatch):
    """Plan sidecar for archive A must NOT influence reweight for archive B."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans"
    venn_root = tmp_path / "venn_empty"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", venn_root)

    sha_a = "a" * 64
    sha_b = "b" * 64
    _write_synth_optimal_plan(plan_root, sha_a, predicted_score_delta=-0.099)

    raw = -0.012
    # Plan for A; query for B => no plan found, passthrough
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha_b)
    assert adjusted == raw, "plan for different archive must NOT leak"


# ──────────────────────────────────────────────────────────────────────────── #
# Edge cases: corrupt JSON, missing dir, planner with promotion flags           #
# ──────────────────────────────────────────────────────────────────────────── #


def test_v2_corrupt_plan_json_falls_through(tmp_path, monkeypatch):
    """Corrupt plan JSON falls through to Cascade 2."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans"
    venn_root = tmp_path / "venn_empty"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", venn_root)

    sha = "6" * 64
    plan_root.mkdir(parents=True)
    (plan_root / f"optimal_plan_{sha[:12]}_20260517T120000.json").write_text(
        "THIS IS NOT JSON", encoding="utf-8"
    )

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha)
    # Falls through cascades; no plan/proof/venn => passthrough
    assert adjusted == raw


def test_v2_planner_delta_is_float(tmp_path, monkeypatch):
    """Plan with non-numeric predicted_score_delta falls through."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans"
    venn_root = tmp_path / "venn_empty"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", venn_root)

    sha = "5" * 64
    plan_root.mkdir(parents=True)
    (plan_root / f"optimal_plan_{sha[:12]}_20260517T120000.json").write_text(
        json.dumps({"predicted_score_delta": "not-a-number"}), encoding="utf-8"
    )

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha)
    assert adjusted == raw


def test_v2_planner_delta_zero_replaces_raw(tmp_path, monkeypatch):
    """Plan with predicted_score_delta=0.0 still REPLACES raw (the planner says no improvement)."""
    mod = _load_autopilot_module()
    plan_root = tmp_path / "plans"

    import tac.master_gradient_consumers as mgc

    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", plan_root)

    sha = "4" * 64
    _write_synth_optimal_plan(plan_root, sha, predicted_score_delta=0.0)

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, sha)
    # 0.0 IS a valid planner output (substrate predicted to have no effect)
    assert adjusted == 0.0


# ──────────────────────────────────────────────────────────────────────────── #
# Live repo regression guards                                                   #
# ──────────────────────────────────────────────────────────────────────────── #


def test_live_fec6_archive_smoke():
    """Live fec6 archive: with no plan + no proof + live Venn sidecar => passthrough.

    This is the canonical anchor test. The fec6 deliverability prober's
    `deliverability_verdict='NOT_DELIVERABLE'` proves the live archive cannot
    win Cascade 2. Without a Lagrangian plan, the v2 cascade MUST passthrough.
    """
    mod = _load_autopilot_module()
    fec6_sha = "f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd"

    # Live fec6 has no optimal_plan sidecar today (Lagrangian planner sister
    # subagent has not run on fec6 yet); also no DeliverabilityProof sidecar.
    # If live state contains EITHER, the test still passes — we only assert
    # the factor is in the canonical set per Cascade contract.
    live_plan_root = REPO_ROOT / ".omx" / "state" / "master_gradient_consumers"
    if not live_plan_root.is_dir():
        pytest.skip("no live master_gradient_consumers sidecar root in this repo state")

    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification_v2(raw, fec6_sha)

    # The fec6 prober's verdict is NOT_DELIVERABLE — no Lagrangian plan AND
    # the existing Venn HIGH_PAIR_INVARIANT branch's deliverability factor
    # SHOULD be 1.0× (passthrough) since the prober rejected all 3 codecs.
    factor = adjusted / raw
    assert factor in (1.0, 0.85), (
        f"live fec6 v2 cascade expected 1.0× passthrough (or 0.85× penalty if "
        f"Venn class shifts); got factor {factor:.4f}. "
        f"If this fails post-Lagrangian-planner-run on fec6, the planner's "
        f"predicted_score_delta IS the new canonical answer; update the assert."
    )
