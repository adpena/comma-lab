# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_PATH = (
    REPO_ROOT
    / ".omx/operator_authorize_recipes/"
    "substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml"
)
PROBE_PATH = (
    REPO_ROOT
    / ".omx/research/"
    "probe_z7_mamba2_temporal_coherence_vs_static_capacity_disambiguator_20260519T155511Z_codex.json"
)


def _load_recipe() -> dict[str, object]:
    with RECIPE_PATH.open(encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    assert isinstance(payload, dict)
    return payload


def _load_probe() -> dict[str, object]:
    with PROBE_PATH.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    assert isinstance(payload, dict)
    return payload


def test_z7_mamba2_recipe_clears_only_validated_reference_torch_handoff_blocker():
    """The reference_torch exact-handoff blocker is cleared, not dispatch authority."""

    recipe = _load_recipe()
    blockers = recipe["dispatch_blockers"]
    assert isinstance(blockers, list)

    cleared_blocker = (
        "z7_mamba2_reference_torch_runtime_exact_handoff_must_validate_before_paid_dispatch"
    )
    assert cleared_blocker not in blockers
    assert "z7_mamba2_dispatch_requires_z7_gru_wave_2_disambiguator_outcome_per_revision_1" in blockers
    assert "z7_mamba2_beta_ib_parameter_requires_c6_ibps_phase_2_empirical_beta_anchor_per_revision_5" in blockers
    assert "z7_mamba2_requires_same_archive_bytes_identity_disambiguator_before_full_dispatch" in blockers
    assert recipe["research_only"] is True
    assert recipe["dispatch_enabled"] is False

    cleared = recipe["dispatch_blockers_cleared"]
    assert isinstance(cleared, list)
    assert any(
        isinstance(row, dict)
        and row.get("blocker") == cleared_blocker
        and isinstance(row.get("cleared_at_utc"), str)
        and "research_only and dispatch_enabled=false remain binding" in str(row.get("scope"))
        for row in cleared
    )


def test_z7_mamba2_handoff_probe_supports_cleared_blocker_without_promotion():
    """Current exact-eval evidence is handoff authority, not score authority."""

    probe = _load_probe()
    assert probe["schema"] == "z7_temporal_coherence_vs_static_capacity_disambiguator_v1"
    assert probe["substrate_id"] == "time_traveler_l5_z7_mamba2"
    assert probe["verdict"] == "recurrent_wins_same_bytes_contest_cuda_pair_but_packet_not_frontier"
    assert probe["score_claim"] is False
    assert probe["promotion_eligible"] is False
    assert probe["rank_or_kill_eligible"] is False
    assert probe["ready_for_paid_dispatch"] is False
    assert probe["ready_for_exact_eval_dispatch"] is False

    source_evals = probe["source_evals"]
    assert isinstance(source_evals, list)
    assert {row["score_axis"] for row in source_evals if isinstance(row, dict)} == {
        "contest_cuda"
    }
    assert all(
        isinstance(row, dict) and row["n_samples"] == 600 for row in source_evals
    )

    supplemental_cpu = probe["supplemental_cpu_pair"]
    assert isinstance(supplemental_cpu, dict)
    assert supplemental_cpu["axis"] == "contest_cpu"
    assert probe["comparability"] == {
        "same_archive_bytes": True,
        "same_n_samples": True,
        "same_score_axis": True,
    }
