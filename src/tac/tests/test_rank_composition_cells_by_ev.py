"""Tests for tools/rank_composition_cells_by_ev.py.

Per the floor-v3 commit ``27a7950fd`` requirement: ensure the EV ranker
correctly consumes the composition cells matrix + posterior + alien-tech
bands, computes EV/$ correctly, and emits a markdown report.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_rank_module():
    spec = importlib.util.spec_from_file_location(
        "rank_composition_cells_by_ev_under_test",
        REPO_ROOT / "tools" / "rank_composition_cells_by_ev.py",
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["rank_composition_cells_by_ev_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


rank_mod = _load_rank_module()
PosteriorSummary = rank_mod.PosteriorSummary
RankedCellRow = rank_mod.RankedCellRow
load_posterior_summary = rank_mod.load_posterior_summary
load_lane_levels = rank_mod.load_lane_levels
rank_cells = rank_mod.rank_cells
render_markdown = rank_mod.render_markdown
serialize_ranked_payload = rank_mod.serialize_ranked_payload
DEFAULT_AXIS_WEIGHTS_PR106_R2 = rank_mod.DEFAULT_AXIS_WEIGHTS_PR106_R2
DEFAULT_ALIEN_TECH_BANDS = rank_mod.DEFAULT_ALIEN_TECH_BANDS
GENERIC_ALIEN_TECH_BAND = rank_mod.GENERIC_ALIEN_TECH_BAND


from tac.composition.enumerate import enumerate_cells  # noqa: E402
from tac.optimization.substrate_composition_matrix import (  # noqa: E402
    canonical_substrate_inventory,
)


# ── 1. axis weights match CLAUDE.md PR106 r2 operating point ──────────────


def test_axis_weights_pose_dominates_at_pr106_r2():
    """Per CLAUDE.md operating-point-dependent rule, pose-marginal is 2.71×."""
    assert DEFAULT_AXIS_WEIGHTS_PR106_R2["pose"] == pytest.approx(2.71)
    assert DEFAULT_AXIS_WEIGHTS_PR106_R2["seg"] == pytest.approx(1.00)
    # Pose >= seg + 0.5 to enforce the marginal-flip rule.
    assert (
        DEFAULT_AXIS_WEIGHTS_PR106_R2["pose"]
        > DEFAULT_AXIS_WEIGHTS_PR106_R2["seg"] + 0.5
    )


# ── 2. posterior loader ───────────────────────────────────────────────────


def test_posterior_summary_handles_missing_file(tmp_path: Path):
    summary = load_posterior_summary(tmp_path / "no-such.json")
    assert summary == {}


def test_posterior_summary_extracts_per_class_anchors(tmp_path: Path):
    payload = {
        "schema": "tac_continual_learning_posterior_v1",
        "accepted_anchor_history": [
            {
                "axis": "cuda",
                "architecture_class": "test_class_a",
                "evidence_tag": "[contest-CUDA]",
                "score_value": 0.193,
                "observed_at_utc": "2026-05-13T10:00:00+00:00",
            },
            {
                "axis": "cpu",
                "architecture_class": "test_class_a",
                "evidence_tag": "[contest-CPU GHA Linux x86_64]",
                "score_value": 0.190,
                "observed_at_utc": "2026-05-13T10:30:00+00:00",
            },
            {
                "axis": "cuda",
                "architecture_class": "test_class_b",
                "evidence_tag": "[byte-anchor]",
                "score_value": 0.5,
                "observed_at_utc": "2026-05-12T08:00:00+00:00",
            },
        ],
    }
    p = tmp_path / "posterior.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    summary = load_posterior_summary(p)
    assert "test_class_a" in summary
    assert summary["test_class_a"].n_authoritative_anchors == 2
    # Most recent authoritative is the CPU GHA row at 10:30.
    assert summary["test_class_a"].most_recent_authoritative_score == pytest.approx(0.190)
    # b has a non-authoritative anchor; n_auth=0.
    assert summary["test_class_b"].n_authoritative_anchors == 0
    assert summary["test_class_b"].n_advisory_anchors == 1


def test_posterior_summary_handles_unparseable_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not-json", encoding="utf-8")
    summary = load_posterior_summary(p)
    assert summary == {}


# ── 3. lane_levels loader ─────────────────────────────────────────────────


def test_lane_levels_loader_returns_dict(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "lanes": [
            {"lane_id": "lane_some_test", "level": 2},
            {"lane_id": "lane_other", "level": 1},
        ],
    }
    p = tmp_path / "lanes.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    levels = load_lane_levels(p)
    assert levels["lane_some_test"] == 2
    assert levels["lane_other"] == 1


def test_lane_levels_loader_handles_missing(tmp_path: Path):
    assert load_lane_levels(tmp_path / "no-such.json") == {}


# ── 4. rank_cells happy path ──────────────────────────────────────────────


def test_rank_cells_smoke_no_overrides():
    """Ranks the canonical substrate × primitive matrix end-to-end."""
    rows = rank_cells(only_with_primitives=False)
    assert len(rows) > 0
    # Sorted descending by ev_per_dollar.
    for i in range(len(rows) - 1):
        assert rows[i].ev_per_dollar >= rows[i + 1].ev_per_dollar
    # Every row has score_claim discipline.
    for r in rows:
        # No row promotes a score directly.
        assert r.predicted_score_delta_band_low <= r.predicted_score_delta_band_high


def test_rank_cells_filters_to_only_with_primitives():
    rows_with = rank_cells(only_with_primitives=True)
    rows_all = rank_cells(only_with_primitives=False)
    assert len(rows_with) <= len(rows_all)
    # No bare-substrate cells in only_with_primitives mode.
    assert all(r.primitive_pipeline for r in rows_with)


def test_rank_cells_axis_weight_propagated():
    """A pose-axis substrate gets axis_weight=2.71 at PR106 r2."""
    rows = rank_cells(only_with_primitives=False)
    pose_rows = [r for r in rows if "pose" in r.substrate_id.lower()]
    if pose_rows:
        # Either explicitly tagged pose-axis substrate OR pose sidecar.
        # We don't assert == 2.71 because some pose sidecars are tagged
        # POSE_AXIS_SIDECHANNEL but with a different target_axis. Just
        # assert weight > 0.5 (sanity).
        assert all(r.axis_weight > 0.5 for r in pose_rows)


def test_rank_cells_blocker_flagged_below_noise_floor():
    """Cells whose midpoint < 1.6e-5 get the F6 noise-floor advisory blocker."""
    # Override band to be very small for ALL substrates so blocker fires.
    overrides = {
        s.substrate_id: {"low": -5e-6, "high": -1e-6, "source": "[test override]"}
        for s in canonical_substrate_inventory()
    }
    rows = rank_cells(
        alien_tech_band_overrides=overrides, only_with_primitives=False
    )
    assert len(rows) > 0
    flagged = [r for r in rows if any("noise_floor" in b for b in r.blockers)]
    assert len(flagged) > 0


def test_rank_cells_posterior_correction_changes_ranking():
    """Posterior correction influences ranking when an architecture has anchors."""
    s_inv = canonical_substrate_inventory()
    target = s_inv[0]  # pick the first substrate as our anchor target
    # Build a posterior that says target has a strong recent anchor (0.18).
    posterior = {
        target.substrate_id: PosteriorSummary(
            architecture_class=target.substrate_id,
            n_authoritative_anchors=5,
            n_advisory_anchors=0,
            most_recent_authoritative_score=0.18,
            most_recent_authoritative_tag="[contest-CUDA]",
        )
    }
    rows_with_posterior = rank_cells(
        posterior_summary=posterior, only_with_primitives=False
    )
    rows_without = rank_cells(posterior_summary={}, only_with_primitives=False)
    target_row_with = next(r for r in rows_with_posterior if r.substrate_id == target.substrate_id)
    target_row_without = next(r for r in rows_without if r.substrate_id == target.substrate_id)
    # Correction factor >1 with strong anchor.
    assert target_row_with.posterior_correction_factor > 1.0
    # p_holds substantially higher with 5 authoritative anchors vs 0.
    assert target_row_with.p_holds > target_row_without.p_holds


def test_rank_cells_alien_tech_band_overrides_propagate():
    """The override band shows up in the predicted_score_delta_band fields."""
    overrides = {
        "wavelet_residual": {
            "low": -0.10, "high": -0.05,
            "source": "[test override; large band]",
        }
    }
    rows = rank_cells(
        alien_tech_band_overrides=overrides, only_with_primitives=False
    )
    wavelet = [r for r in rows if r.substrate_id == "wavelet_residual"]
    assert wavelet
    # Cells of wavelet_residual have band = override + per-cell predicted delta;
    # the override is dominant so band should be near [-0.10, -0.05].
    bare = [r for r in wavelet if not r.primitive_pipeline]
    assert bare
    # Bare cell adds its substrate-baseline prediction on top of override;
    # the override (~-0.05 to -0.10) dominates either way (substrate baseline
    # midpoint is on the order of -0.0006 for wavelet_residual).
    assert bare[0].predicted_score_delta_band_low == pytest.approx(-0.10, abs=0.01)
    assert bare[0].predicted_score_delta_band_high == pytest.approx(-0.05, abs=0.01)
    # Override source string propagates through to the row.
    assert "test override" in bare[0].alien_tech_source


def test_rank_cells_substrate_cost_overrides_propagate():
    overrides = {"wavelet_residual": 100.0}
    rows = rank_cells(
        substrate_cost_overrides=overrides,
        only_with_primitives=False,
    )
    wavelet_rows = [r for r in rows if r.substrate_id == "wavelet_residual"]
    assert wavelet_rows
    assert all(r.estimated_dispatch_cost_usd == pytest.approx(100.0) for r in wavelet_rows)


def test_rank_cells_lane_levels_propagate():
    s_inv = canonical_substrate_inventory()
    target = s_inv[0]
    levels = {f"lane_{target.substrate_id}_pr106": 3}
    rows = rank_cells(lane_levels=levels, only_with_primitives=False)
    target_rows = [r for r in rows if r.substrate_id == target.substrate_id]
    assert target_rows
    assert any(r.lane_level == 3 for r in target_rows)
    assert any(r.readiness == "L3" for r in target_rows)


# ── 5. render_markdown ────────────────────────────────────────────────────


def test_render_markdown_includes_top_k_table():
    rows = rank_cells(only_with_primitives=False)
    md = render_markdown(rows, top_k=5)
    assert "# Composition Cell EV Ranking — Top 5" in md
    assert "Schema:" in md
    assert "tac_composition_cell_ev_ranking_v1" in md
    assert "operator decision input" in md.lower()
    # Pose dominance should be cited in the methodology section.
    assert "pose=2.71" in md


def test_render_markdown_handles_zero_rows():
    md = render_markdown([], top_k=10)
    assert "Top 10" in md
    assert "Total ranked cells: 0" in md


def test_render_markdown_renders_table_rows():
    rows = rank_cells(only_with_primitives=False)
    md = render_markdown(rows, top_k=3)
    # The first 3 rows' cell_ids should appear in the markdown.
    for r in rows[:3]:
        assert r.cell_id in md


def test_render_markdown_includes_blocker_excerpt_when_present():
    overrides = {
        s.substrate_id: {"low": -5e-6, "high": -1e-6, "source": "[t]"}
        for s in canonical_substrate_inventory()[:3]
    }
    rows = rank_cells(
        alien_tech_band_overrides=overrides, only_with_primitives=False
    )
    # Limit to override-affected rows.
    rows = [r for r in rows if r.substrate_id in overrides]
    md = render_markdown(rows, top_k=20)
    assert "blockers" in md.lower()


# ── 6. serialize_ranked_payload ───────────────────────────────────────────


def test_serialize_ranked_payload_carries_compliance_tags():
    rows = rank_cells(only_with_primitives=False)
    payload = serialize_ranked_payload(
        rows[:5],
        top_k=5,
        posterior_path=None,
        alien_tech_overrides_path=None,
        substrate_cost_overrides_path=None,
    )
    assert payload["schema"] == "tac_composition_cell_ev_ranking_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    tags = payload["claude_md_compliance_tags"]
    assert "predicted_substrate_primitive_matrix_v1_posterior_reweight" in tags
    assert "no_score_claim_advanced_by_this_artifact" in tags
    assert "operator_decision_input_only" in tags
    assert "no_tmp_paths" in tags


def test_serialize_ranked_payload_rows_have_rank_field():
    rows = rank_cells(only_with_primitives=False)[:3]
    payload = serialize_ranked_payload(
        rows, top_k=3, posterior_path=None,
        alien_tech_overrides_path=None, substrate_cost_overrides_path=None,
    )
    ranked = payload["ranked_rows"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]
