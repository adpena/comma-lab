# SPDX-License-Identifier: MIT
"""Per-substrate wire-in tests for WAVE-1 canonical posterior emission.

Per WAVE-1-POSTERIOR-EMISSION-CANONICAL-WIRE-IN charter 2026-05-26 +
audit roadmap commit ``e757bb74c`` META #1 CRITICAL finding closure:
verifies each of the 8 LANDED Path 3 substrates exposes a canonical
``emit_landing_posterior_anchor()`` function that:

1. Calls the canonical helper at ``tac.substrates._shared.posterior_emission_helper``
2. Lifts the substrate's signal into the cathedral autopilot's 62 auto-
   discovered consumers via canonical posterior surfaces
3. Carries canonical non-promotable markers (Catalog #127/#192/#317/#341)
4. Carries canonical Provenance (Catalog #323)
5. Refuses promotion at the contest-axis posterior (custody validator
   advisory-grade refusal; bumps refused_anchor_count)
6. Appends a row to the MLX research-signal manifest with canonical
   substrate identifiers + canonical equation IDs

Each per-substrate test uses isolated tmp paths so the live canonical
state is not polluted by the test suite.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

# The 8 LANDED Path 3 substrates per the audit utility matrix (commit
# ``e757bb74c`` STEP 2). Each tuple is (substrate_id, module_path,
# expected_paradigm_token).
PATH_3_SUBSTRATES: tuple[tuple[str, str, str], ...] = (
    (
        "dreamer_v3_rssm",
        "tac.substrates.dreamer_v3_rssm",
        "categorical_posterior_substrate_class_shift",
    ),
    (
        "z7_mamba2_v2_fresh_substrate",
        "tac.substrates.z7_mamba2_v2_fresh_substrate",
        "temporal_predictive_coding_state_space",
    ),
    (
        "nscs06_v8_chroma_lut",
        "tac.substrates.nscs06_v8_chroma_lut",
        "procedural_codebook_chroma_lut_replacement",
    ),
    (
        "time_traveler_l5_z6",
        "tac.substrates.time_traveler_l5_z6",
        "predictive_coding_ego_motion_conditioned",
    ),
    (
        "boost_nerv",
        "tac.substrates.boost_nerv",
        "iterative_boosting_residual_chain_sidecar",
    ),
    (
        "z8_hierarchical_predictive_coding",
        "tac.substrates.z8_hierarchical_predictive_coding",
        "hierarchical_predictive_coding_canonical_quadruple",
    ),
    (
        "nirvana_cascading_nerv",
        "tac.substrates.nirvana_cascading_nerv",
        "hierarchical_cascading_residual_decoder",
    ),
    (
        "atw_v2_cooperative_receiver_v2",
        "tac.substrates.atw_v2_cooperative_receiver_v2",
        "cooperative_receiver_ego_motion_foe_projection",
    ),
)


@pytest.mark.parametrize(
    "substrate_id,module_path,expected_paradigm",
    PATH_3_SUBSTRATES,
    ids=[s[0] for s in PATH_3_SUBSTRATES],
)
def test_substrate_exposes_canonical_emit_landing_posterior_anchor(
    substrate_id: str, module_path: str, expected_paradigm: str, tmp_path: Path
) -> None:
    """Each Path 3 substrate exposes canonical emit_landing_posterior_anchor()."""
    module = importlib.import_module(module_path)

    # 1) Canonical public surface present
    assert hasattr(module, "emit_landing_posterior_anchor"), (
        f"{substrate_id} missing emit_landing_posterior_anchor per WAVE-1"
    )
    assert hasattr(module, "SUBSTRATE_ID"), f"{substrate_id} missing SUBSTRATE_ID constant per WAVE-1"
    assert hasattr(module, "ARCHITECTURE_CLASS"), f"{substrate_id} missing ARCHITECTURE_CLASS constant per WAVE-1"
    assert hasattr(module, "CANONICAL_EQUATION_IDS"), (
        f"{substrate_id} missing CANONICAL_EQUATION_IDS constant per WAVE-1"
    )

    # 2) Canonical identifiers match expected
    assert substrate_id == module.SUBSTRATE_ID


@pytest.mark.parametrize(
    "substrate_id,module_path,expected_paradigm",
    PATH_3_SUBSTRATES,
    ids=[s[0] for s in PATH_3_SUBSTRATES],
)
def test_substrate_emits_anchor_with_canonical_markers(
    substrate_id: str, module_path: str, expected_paradigm: str, tmp_path: Path
) -> None:
    """Each Path 3 substrate emit_landing_posterior_anchor() carries canonical markers."""
    module = importlib.import_module(module_path)
    post_p = tmp_path / "posterior.json"
    lock_p = tmp_path / "posterior.lock"
    manifest_p = tmp_path / "manifest.jsonl"

    anchor = module.emit_landing_posterior_anchor(
        posterior_path=post_p,
        posterior_lock_path=lock_p,
        manifest_path=manifest_p,
    )

    # Canonical substrate identifiers
    assert anchor.substrate_id == substrate_id

    # Canonical non-promotable markers per Catalog #127/#192/#317/#341
    assert anchor.score_claim is False
    assert anchor.promotion_eligible is False
    assert anchor.ready_for_exact_eval_dispatch is False
    assert anchor.rank_or_kill_eligible is False

    # Canonical evidence_tag + hardware_substrate
    assert anchor.evidence_tag == "[macOS-MLX research-signal]"
    assert anchor.hardware_substrate == "macos_arm64_mlx"

    # Canonical Provenance present
    assert anchor.provenance.promotion_eligible is False
    assert anchor.provenance.score_claim_valid is False
    assert anchor.provenance.canonical_helper_invocation == (
        "tac.provenance.builders.build_provenance_for_macos_mlx_research_signal"
    )


@pytest.mark.parametrize(
    "substrate_id,module_path,expected_paradigm",
    PATH_3_SUBSTRATES,
    ids=[s[0] for s in PATH_3_SUBSTRATES],
)
def test_substrate_posterior_refused_per_advisory_grade(
    substrate_id: str, module_path: str, expected_paradigm: str, tmp_path: Path
) -> None:
    """Each Path 3 substrate's posterior anchor is REFUSED (advisory-grade)."""
    module = importlib.import_module(module_path)
    post_p = tmp_path / "posterior.json"
    lock_p = tmp_path / "posterior.lock"
    manifest_p = tmp_path / "manifest.jsonl"

    anchor = module.emit_landing_posterior_anchor(
        posterior_path=post_p,
        posterior_lock_path=lock_p,
        manifest_path=manifest_p,
    )

    # Per CLAUDE.md local-substrate authority + custody validator:
    # [macOS-MLX research-signal] is NON_PROMOTABLE_TAGS -> REFUSED.
    assert anchor.posterior_update.accepted is False
    assert (
        "advisory" in anchor.posterior_update.refusal_reason.lower()
        or "non-authoritative" in anchor.posterior_update.refusal_reason.lower()
    )

    # Posterior state recorded refusal (bumped refused_anchor_count).
    posterior_data = json.loads(post_p.read_text())
    assert posterior_data["refused_anchor_count"] >= 1


@pytest.mark.parametrize(
    "substrate_id,module_path,expected_paradigm",
    PATH_3_SUBSTRATES,
    ids=[s[0] for s in PATH_3_SUBSTRATES],
)
def test_substrate_manifest_row_emitted_with_canonical_extras(
    substrate_id: str, module_path: str, expected_paradigm: str, tmp_path: Path
) -> None:
    """Each Path 3 substrate emits a canonical manifest row to MLX JSONL."""
    module = importlib.import_module(module_path)
    post_p = tmp_path / "posterior.json"
    lock_p = tmp_path / "posterior.lock"
    manifest_p = tmp_path / "manifest.jsonl"

    module.emit_landing_posterior_anchor(
        posterior_path=post_p,
        posterior_lock_path=lock_p,
        manifest_path=manifest_p,
    )

    # Manifest exists with exactly one row.
    assert manifest_p.exists()
    lines = manifest_p.read_text().strip().split("\n")
    assert len(lines) == 1
    row = json.loads(lines[0])

    # Canonical substrate identifiers
    assert row["substrate_id"] == substrate_id

    # Canonical non-promotable markers preserved
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["promotable"] is False
    assert row["predicted_delta_adjustment"] == 0.0
    assert row["axis_tag"] == "[macOS-MLX research-signal]"
    assert row["evidence_tag"] == "[macOS-MLX research-signal]"
    assert row["evidence_grade"] == "macOS-MLX-research-signal"
    assert row["hardware_substrate"] == "macos_arm64_mlx"

    # Substrate-specific paradigm token threaded through
    assert row["paradigm"] == expected_paradigm

    # Canonical equation IDs threaded through (every substrate references
    # at least one canonical or proposed-canonical equation)
    assert "canonical_equation_ids" in row
    assert isinstance(row["canonical_equation_ids"], list)
    assert len(row["canonical_equation_ids"]) >= 1


@pytest.mark.parametrize(
    "substrate_id,module_path,expected_paradigm",
    PATH_3_SUBSTRATES,
    ids=[s[0] for s in PATH_3_SUBSTRATES],
)
def test_substrate_emit_is_idempotent_at_manifest_jsonl(
    substrate_id: str, module_path: str, expected_paradigm: str, tmp_path: Path
) -> None:
    """Each Path 3 substrate can emit the anchor twice; JSONL is APPEND-ONLY."""
    module = importlib.import_module(module_path)
    post_p = tmp_path / "posterior.json"
    lock_p = tmp_path / "posterior.lock"
    manifest_p = tmp_path / "manifest.jsonl"

    for _ in range(2):
        module.emit_landing_posterior_anchor(
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
        )

    # Per Catalog #110/#113 APPEND-ONLY: manifest has 2 rows for 2 emissions.
    lines = manifest_p.read_text().strip().split("\n")
    assert len(lines) == 2


def test_all_8_path_3_substrates_emit_into_shared_manifest(tmp_path: Path) -> None:
    """ALL 8 Path 3 substrates emit cleanly into the SAME canonical manifest.

    This is the WAVE-1 audit META #1 CRITICAL finding closure test: a
    single test harness invokes every Path 3 substrate's canonical
    emit_landing_posterior_anchor() in turn and verifies all 8 rows
    land in the shared manifest JSONL with unique substrate_id values.
    """
    post_p = tmp_path / "posterior.json"
    lock_p = tmp_path / "posterior.lock"
    manifest_p = tmp_path / "manifest.jsonl"

    emitted_substrate_ids: set[str] = set()
    for _substrate_id, module_path, _ in PATH_3_SUBSTRATES:
        module = importlib.import_module(module_path)
        anchor = module.emit_landing_posterior_anchor(
            posterior_path=post_p,
            posterior_lock_path=lock_p,
            manifest_path=manifest_p,
        )
        emitted_substrate_ids.add(anchor.substrate_id)

    # All 8 substrates emitted
    assert len(emitted_substrate_ids) == 8
    expected_ids = {s[0] for s in PATH_3_SUBSTRATES}
    assert emitted_substrate_ids == expected_ids

    # Manifest JSONL has exactly 8 rows
    lines = manifest_p.read_text().strip().split("\n")
    assert len(lines) == 8

    # Posterior recorded all 8 refusals
    posterior_data = json.loads(post_p.read_text())
    assert posterior_data["refused_anchor_count"] >= 8


def test_canonical_equation_id_lineage_per_substrate(tmp_path: Path) -> None:
    """Canonical equation IDs per substrate match audit op-routable #3 mapping."""
    expected_lineage = {
        "dreamer_v3_rssm": "categorical_posterior_capacity_vs_continuous_gaussian_v1",
        "z7_mamba2_v2_fresh_substrate": "predictive_coding_residual_capacity_v1",  # PROPOSED
        "nscs06_v8_chroma_lut": "procedural_codebook_from_seed_compression_savings_v1",
        "time_traveler_l5_z6": "predictive_coding_residual_capacity_v1",  # PROPOSED (shared)
        "boost_nerv": "boosting_residual_score_lowering_per_stage_v1",  # PROPOSED
        "z8_hierarchical_predictive_coding": "categorical_posterior_capacity_vs_continuous_gaussian_v1",  # references 5 equations
        "nirvana_cascading_nerv": "cascading_nerv_per_stage_residual_v1",  # PROPOSED
        "atw_v2_cooperative_receiver_v2": "cooperative_receiver_atick_redlich_score_savings_v1",  # PROPOSED
    }

    for substrate_id, module_path, _ in PATH_3_SUBSTRATES:
        module = importlib.import_module(module_path)
        ids = module.CANONICAL_EQUATION_IDS
        assert isinstance(ids, tuple)
        assert len(ids) >= 1
        # At least one canonical equation reference includes the expected token
        expected_token = expected_lineage[substrate_id]
        match = any(expected_token in eq_id for eq_id in ids)
        assert match, f"{substrate_id} missing expected canonical equation lineage {expected_token!r} in {ids}"
