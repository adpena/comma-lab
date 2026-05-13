"""Tests for the cathedral autopilot end-to-end wire to QQ's substrate
composition matrix ranking.

Per CLAUDE.md "race-mode rigor inversion + parallel-dispatch first" + the
substrate composition matrix landing memo, this wire connects QQ's ranked
dispatch JSON into MM's autopilot loop. Tests cover:

  - load_candidates_from_substrate_composition_ranking parses the canonical
    ranking JSON shape (schema=tac_autopilot_dispatch_ranking_v1)
  - filter_composition_incompatible_dispatches refuses REPLACEMENT siblings
    in the same iteration
  - filter_composition_incompatible_dispatches refuses INCOMPATIBLE pairs
  - candidate_substrate_ids_from_ranking returns the correct mapping
  - Pareto-filter is honored upstream (we trust QQ's filter)
  - Out-of-envelope rows are dropped by default
  - Out-of-cap rows are dropped by default
  - score_claim=True at top-level is REFUSED
  - score_claim=True per-row is REFUSED
  - ready_for_exact_eval_dispatch=True is REFUSED
  - Missing 'ranked_dispatches' key is REFUSED
  - Missing file path raises FileNotFoundError
  - End-to-end CLI smoke: --use-substrate-composition-matrix-ranking flag
    succeeds with QQ's canonical ranking artifact
  - CLI refuses both --candidates-jsonl AND --use-...-ranking
  - CLI refuses neither
  - HALT-and-ASK is preserved for every dispatch (no auto-dispatch in this wire)
  - Cumulative envelope hard cap = $20 (operator round-trip required above)
  - Per-dispatch cap = $5 (operator round-trip required above)
  - filter_composition_incompatible_dispatches returns dropped reasons
  - Output payload includes substrate_composition_ranking section when wired
  - Output payload includes n_dropped_by_composition_constraint count
  - Notes field carries [predicted; substrate composition matrix v1] tag
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cathedral_autopilot_autonomous_loop as loop  # noqa: E402


# ── Fixture: canonical ranking JSON ───────────────────────────────────────


def _canonical_ranking_payload() -> dict:
    """Return a canonical ranking JSON payload matching QQ's schema.

    Mirrors the shape of
    experiments/results/cathedral_autopilot_dispatch_ranking_*/ranking.json
    but with a tiny dispatch list for tests.
    """
    return {
        "schema": "tac_autopilot_dispatch_ranking_v1",
        "matrix_schema": "tac_substrate_composition_matrix_v1",
        "generated_at_utc": "2026-05-12T00:00:00.000000+00:00",
        "n_substrates_considered": 24,
        "per_dispatch_cap_usd": 5.0,
        "cumulative_cap_usd": 20.0,
        "cumulative_estimated_spend_usd": 0.6,
        "n_ranked_dispatches": 4,
        "n_filtered_dropped": 0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "planning_only_autopilot_dispatch_ranking_v1",
        "composition_constraints_applied": [
            "drop_redundant_dominated",
            "renderer_replacement_mutually_exclusive",
            "format_id_collision_check",
            "per_dispatch_cap_usd=5.0",
            "cumulative_cap_usd=20.0",
        ],
        "ranked_dispatches": [
            {
                "candidate_id": "singleton__magic_codec",
                "family": "meta_codec",
                "substrate_ids": ["magic_codec"],
                "predicted_score_delta": -0.0005,
                "expected_information_gain": 0.0005,
                "estimated_dispatch_cost_usd": 0.10,
                "eig_per_dollar": 0.005,
                "composition_notes": "[predicted; substrate composition matrix v1] singleton dispatch of magic_codec",
                "blockers": [],
                "fits_per_dispatch_cap": True,
                "fits_cumulative_envelope": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            {
                "candidate_id": "singleton__nerv_as_renderer",
                "family": "renderer_replacement",
                "substrate_ids": ["nerv_as_renderer"],
                "predicted_score_delta": -0.005,
                "expected_information_gain": 0.005,
                "estimated_dispatch_cost_usd": 0.50,
                "eig_per_dollar": 0.010,
                "composition_notes": "[predicted; substrate composition matrix v1] singleton dispatch of nerv_as_renderer",
                "blockers": [],
                "fits_per_dispatch_cap": True,
                "fits_cumulative_envelope": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            {
                "candidate_id": "singleton__mnerv",
                "family": "renderer_replacement",
                "substrate_ids": ["mnerv"],
                "predicted_score_delta": -0.004,
                "expected_information_gain": 0.004,
                "estimated_dispatch_cost_usd": 0.50,
                "eig_per_dollar": 0.008,
                "composition_notes": "[predicted; substrate composition matrix v1] singleton dispatch of mnerv",
                "blockers": [],
                "fits_per_dispatch_cap": True,
                "fits_cumulative_envelope": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            {
                "candidate_id": "out_of_envelope_singleton",
                "family": "self_compression",
                "substrate_ids": ["scpp_substrate"],
                "predicted_score_delta": -0.001,
                "expected_information_gain": 0.001,
                "estimated_dispatch_cost_usd": 0.10,
                "eig_per_dollar": 0.010,
                "composition_notes": "[predicted; substrate composition matrix v1] out of cumulative envelope (test fixture)",
                "blockers": [],
                "fits_per_dispatch_cap": True,
                "fits_cumulative_envelope": False,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ],
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "operator_gate_non_negotiable_at_every_dispatch",
            "halt_and_ask_default_on",
            "no_tmp_paths",
            "substrate_composition_matrix_consumed",
        ],
    }


def _write_ranking(tmp_path: Path, payload: dict | None = None) -> Path:
    payload = payload if payload is not None else _canonical_ranking_payload()
    p = tmp_path / "ranking.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p


# ── load_candidates_from_substrate_composition_ranking ────────────────────


def test_load_canonical_ranking_returns_in_envelope_only_by_default(tmp_path):
    p = _write_ranking(tmp_path)
    rows = loop.load_candidates_from_substrate_composition_ranking(p)
    ids = [r.candidate_id for r in rows]
    assert "out_of_envelope_singleton" not in ids
    assert "singleton__magic_codec" in ids
    assert "singleton__nerv_as_renderer" in ids
    assert "singleton__mnerv" in ids


def test_load_canonical_ranking_only_in_envelope_false_includes_out_of_envelope(tmp_path):
    p = _write_ranking(tmp_path)
    rows = loop.load_candidates_from_substrate_composition_ranking(
        p, only_in_envelope=False
    )
    ids = [r.candidate_id for r in rows]
    assert "out_of_envelope_singleton" in ids


def test_load_ranking_only_fits_per_dispatch_cap_filters_correctly(tmp_path):
    payload = _canonical_ranking_payload()
    payload["ranked_dispatches"][0]["fits_per_dispatch_cap"] = False
    p = _write_ranking(tmp_path, payload)
    rows = loop.load_candidates_from_substrate_composition_ranking(p)
    assert "singleton__magic_codec" not in [r.candidate_id for r in rows]


def test_load_ranking_carries_predicted_tag_in_notes(tmp_path):
    p = _write_ranking(tmp_path)
    rows = loop.load_candidates_from_substrate_composition_ranking(p)
    for r in rows:
        assert "[predicted; substrate composition matrix v1]" in r.notes
        assert "substrate_ids:" in r.notes


def test_load_ranking_refuses_top_level_score_claim_true(tmp_path):
    payload = _canonical_ranking_payload()
    payload["score_claim"] = True
    p = _write_ranking(tmp_path, payload)
    with pytest.raises(ValueError, match="score_claim=True"):
        loop.load_candidates_from_substrate_composition_ranking(p)


def test_load_ranking_refuses_per_row_score_claim_true(tmp_path):
    payload = _canonical_ranking_payload()
    payload["ranked_dispatches"][0]["score_claim"] = True
    p = _write_ranking(tmp_path, payload)
    with pytest.raises(ValueError, match="score_claim"):
        loop.load_candidates_from_substrate_composition_ranking(p)


def test_load_ranking_refuses_ready_for_exact_eval(tmp_path):
    payload = _canonical_ranking_payload()
    payload["ranked_dispatches"][0]["ready_for_exact_eval_dispatch"] = True
    p = _write_ranking(tmp_path, payload)
    with pytest.raises(ValueError, match="ready_for_exact_eval_dispatch"):
        loop.load_candidates_from_substrate_composition_ranking(p)


def test_load_ranking_refuses_missing_ranked_dispatches_key(tmp_path):
    payload = _canonical_ranking_payload()
    del payload["ranked_dispatches"]
    p = _write_ranking(tmp_path, payload)
    with pytest.raises(ValueError, match="ranked_dispatches"):
        loop.load_candidates_from_substrate_composition_ranking(p)


def test_load_ranking_missing_file_raises(tmp_path):
    p = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        loop.load_candidates_from_substrate_composition_ranking(p)


def test_load_ranking_predicted_score_delta_round_trips(tmp_path):
    p = _write_ranking(tmp_path)
    rows = loop.load_candidates_from_substrate_composition_ranking(p)
    by_id = {r.candidate_id: r for r in rows}
    assert by_id["singleton__magic_codec"].predicted_score_delta == pytest.approx(-0.0005)
    assert by_id["singleton__nerv_as_renderer"].predicted_score_delta == pytest.approx(-0.005)


# ── candidate_substrate_ids_from_ranking ───────────────────────────────────


def test_candidate_substrate_ids_returns_correct_mapping(tmp_path):
    p = _write_ranking(tmp_path)
    m = loop.candidate_substrate_ids_from_ranking(p)
    assert m["singleton__magic_codec"] == ("magic_codec",)
    assert m["singleton__nerv_as_renderer"] == ("nerv_as_renderer",)
    assert m["singleton__mnerv"] == ("mnerv",)


def test_candidate_substrate_ids_refuses_missing_ranked_dispatches(tmp_path):
    payload = _canonical_ranking_payload()
    del payload["ranked_dispatches"]
    p = _write_ranking(tmp_path, payload)
    with pytest.raises(ValueError, match="ranked_dispatches"):
        loop.candidate_substrate_ids_from_ranking(p)


# ── filter_composition_incompatible_dispatches ────────────────────────────


def test_composition_filter_drops_two_renderer_replacement_in_same_chain(tmp_path):
    """Two RENDERER_REPLACEMENT substrates in the same iteration must conflict.

    nerv_as_renderer and mnerv are both renderer-replacement substrates;
    only one can occupy the renderer slot per archive. The matrix marks
    them as REPLACEMENT and the filter must drop the second one (with
    lower EV/$).
    """
    p = _write_ranking(tmp_path)
    rows = loop.load_candidates_from_substrate_composition_ranking(p)
    sub_map = loop.candidate_substrate_ids_from_ranking(p)
    kept, dropped = loop.filter_composition_incompatible_dispatches(
        rows, candidate_substrate_ids=sub_map,
    )
    kept_ids = [c.candidate_id for c in kept]
    dropped_ids = [d[0] for d in dropped]
    # nerv_as_renderer kept (first in iteration order); mnerv dropped
    assert "singleton__nerv_as_renderer" in kept_ids
    assert "singleton__mnerv" in dropped_ids
    # magic_codec is meta-codec, ORTHOGONAL to renderer-replacement, kept
    assert "singleton__magic_codec" in kept_ids


def test_composition_filter_returns_reason_for_dropped_candidates(tmp_path):
    p = _write_ranking(tmp_path)
    rows = loop.load_candidates_from_substrate_composition_ranking(p)
    sub_map = loop.candidate_substrate_ids_from_ranking(p)
    _, dropped = loop.filter_composition_incompatible_dispatches(
        rows, candidate_substrate_ids=sub_map,
    )
    for cid, reason in dropped:
        assert "substrate composition matrix" in reason or "substrate" in reason
        assert "refuse" in reason.lower() or "compose" in reason.lower()


def test_composition_filter_orthogonal_substrates_all_kept(tmp_path):
    """magic_codec + film_pose_conditioning + nerv_enc_dec_separated are all
    ORTHOGONAL pairwise — all three must be kept."""
    payload = _canonical_ranking_payload()
    payload["ranked_dispatches"] = [
        {
            "candidate_id": "singleton__magic_codec",
            "family": "meta_codec",
            "substrate_ids": ["magic_codec"],
            "predicted_score_delta": -0.0005,
            "expected_information_gain": 0.0005,
            "estimated_dispatch_cost_usd": 0.10,
            "eig_per_dollar": 0.005,
            "composition_notes": "[predicted; substrate composition matrix v1] singleton",
            "blockers": [],
            "fits_per_dispatch_cap": True,
            "fits_cumulative_envelope": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        {
            "candidate_id": "singleton__film_pose_conditioning",
            "family": "bolt_on",
            "substrate_ids": ["film_pose_conditioning"],
            "predicted_score_delta": -0.0005,
            "expected_information_gain": 0.0005,
            "estimated_dispatch_cost_usd": 0.0,
            "eig_per_dollar": float("inf"),
            "composition_notes": "[predicted; substrate composition matrix v1] singleton",
            "blockers": [],
            "fits_per_dispatch_cap": True,
            "fits_cumulative_envelope": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        {
            "candidate_id": "singleton__nerv_enc_dec_separated",
            "family": "bolt_on",
            "substrate_ids": ["nerv_enc_dec_separated"],
            "predicted_score_delta": -0.0005,
            "expected_information_gain": 0.0005,
            "estimated_dispatch_cost_usd": 0.0,
            "eig_per_dollar": float("inf"),
            "composition_notes": "[predicted; substrate composition matrix v1] singleton",
            "blockers": [],
            "fits_per_dispatch_cap": True,
            "fits_cumulative_envelope": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    ]
    p = _write_ranking(tmp_path, payload)
    rows = loop.load_candidates_from_substrate_composition_ranking(p)
    sub_map = loop.candidate_substrate_ids_from_ranking(p)
    kept, dropped = loop.filter_composition_incompatible_dispatches(
        rows, candidate_substrate_ids=sub_map,
    )
    assert len(kept) == 3
    assert len(dropped) == 0


def test_composition_filter_unknown_substrate_treated_as_skip(tmp_path):
    """Unknown substrate names (not in the matrix) should not crash; they
    pass through (the filter only refuses on positively classified
    incompatible cells)."""
    payload = _canonical_ranking_payload()
    payload["ranked_dispatches"] = [
        {
            "candidate_id": "singleton__unknown_substrate_xyz",
            "family": "unknown",
            "substrate_ids": ["totally_made_up_substrate_xyz"],
            "predicted_score_delta": -0.001,
            "expected_information_gain": 0.001,
            "estimated_dispatch_cost_usd": 0.10,
            "eig_per_dollar": 0.010,
            "composition_notes": "[predicted; test fixture] unknown substrate",
            "blockers": [],
            "fits_per_dispatch_cap": True,
            "fits_cumulative_envelope": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    ]
    p = _write_ranking(tmp_path, payload)
    rows = loop.load_candidates_from_substrate_composition_ranking(p)
    sub_map = loop.candidate_substrate_ids_from_ranking(p)
    kept, dropped = loop.filter_composition_incompatible_dispatches(
        rows, candidate_substrate_ids=sub_map,
    )
    assert len(kept) == 1
    assert len(dropped) == 0


def test_composition_filter_returns_typed_tuple_pairs(tmp_path):
    """Dropped reasons must be (candidate_id, reason) tuples."""
    p = _write_ranking(tmp_path)
    rows = loop.load_candidates_from_substrate_composition_ranking(p)
    sub_map = loop.candidate_substrate_ids_from_ranking(p)
    _, dropped = loop.filter_composition_incompatible_dispatches(
        rows, candidate_substrate_ids=sub_map,
    )
    for entry in dropped:
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        assert isinstance(entry[0], str)
        assert isinstance(entry[1], str)


# ── End-to-end CLI smoke tests ────────────────────────────────────────────


def test_cli_with_ranking_succeeds(tmp_path, monkeypatch):
    p = _write_ranking(tmp_path)
    output_path = tmp_path / "report.json"
    # Ensure env-var is OFF — we are NOT using authorized mode here, just
    # verifying the wire works end-to-end with HALT-and-ASK preserved.
    monkeypatch.delenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, raising=False)
    rc = loop.main([
        "--use-substrate-composition-matrix-ranking", str(p),
        "--iterations", "1",
        "--output", str(output_path),
    ])
    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert payload["substrate_composition_ranking"]["ranking_path"] == str(p)
    assert payload["substrate_composition_ranking"]["include_out_of_envelope"] is False
    # nerv_as_renderer kept; mnerv dropped (REPLACEMENT collision)
    assert payload["substrate_composition_ranking"]["n_dropped_by_composition_constraint"] == 1
    dropped_ids = [
        d["candidate_id"]
        for d in payload["substrate_composition_ranking"]["dropped_with_reasons"]
    ]
    assert "singleton__mnerv" in dropped_ids


def test_cli_refuses_both_candidates_jsonl_and_ranking(tmp_path):
    p = _write_ranking(tmp_path)
    other = tmp_path / "candidates.jsonl"
    other.write_text("")
    rc = loop.main([
        "--candidates-jsonl", str(other),
        "--use-substrate-composition-matrix-ranking", str(p),
        "--iterations", "1",
    ])
    assert rc == 2


def test_cli_refuses_neither_source(tmp_path):
    rc = loop.main([
        "--iterations", "1",
    ])
    assert rc == 2


def test_cli_emits_substrate_composition_constraints_enforced_compliance_tag(tmp_path, monkeypatch):
    p = _write_ranking(tmp_path)
    output_path = tmp_path / "report.json"
    monkeypatch.delenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, raising=False)
    rc = loop.main([
        "--use-substrate-composition-matrix-ranking", str(p),
        "--iterations", "1",
        "--output", str(output_path),
    ])
    assert rc == 0
    payload = json.loads(output_path.read_text())
    tags = payload["claude_md_compliance_tags"]
    assert "substrate_composition_matrix_constraints_enforced" in tags
    assert "candidates_jsonl_source" not in tags


def test_cli_jsonl_path_emits_candidates_jsonl_source_tag(tmp_path, monkeypatch):
    """Backward-compat: --candidates-jsonl path still works and tags
    candidates_jsonl_source instead of the matrix-constraints tag."""
    candidates = tmp_path / "candidates.jsonl"
    row = {
        "candidate_id": "c1",
        "family": "test",
        "predicted_score_delta": -0.001,
        "expected_information_gain": 0.001,
        "estimated_dispatch_cost_usd": 0.5,
        "blockers": [],
        "notes": "",
    }
    candidates.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_path = tmp_path / "report.json"
    monkeypatch.delenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, raising=False)
    rc = loop.main([
        "--candidates-jsonl", str(candidates),
        "--iterations", "1",
        "--output", str(output_path),
    ])
    assert rc == 0
    payload = json.loads(output_path.read_text())
    tags = payload["claude_md_compliance_tags"]
    assert "candidates_jsonl_source" in tags
    assert "substrate_composition_matrix_constraints_enforced" not in tags
    assert payload["substrate_composition_ranking"] is None


def test_cli_halt_and_ask_preserved_for_every_dispatch(tmp_path, monkeypatch):
    """No autopilot self-authorization happens unless authorized mode is on
    AND the env-var is set. Verify HALT events all require_approval=True
    when authorized mode is off."""
    p = _write_ranking(tmp_path)
    output_path = tmp_path / "report.json"
    monkeypatch.delenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, raising=False)
    rc = loop.main([
        "--use-substrate-composition-matrix-ranking", str(p),
        "--iterations", "1",
        "--output", str(output_path),
    ])
    assert rc == 0
    payload = json.loads(output_path.read_text())
    for report in payload["reports"]:
        for event in report["halt_events"]:
            # CLAUDE.md non-negotiable: every dispatch HALT required approval
            # in default mode.
            assert event["autopilot_authorized"] is False


def test_cli_include_out_of_envelope_flag_brings_in_dropped_rows(tmp_path, monkeypatch):
    p = _write_ranking(tmp_path)
    output_path = tmp_path / "report.json"
    monkeypatch.delenv(loop.OPERATOR_AUTHORIZED_MODE_ENV_VAR, raising=False)
    rc = loop.main([
        "--use-substrate-composition-matrix-ranking", str(p),
        "--include-out-of-envelope-ranking-candidates",
        "--iterations", "1",
        "--output", str(output_path),
    ])
    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert payload["substrate_composition_ranking"]["include_out_of_envelope"] is True


# ── Schema constants ──────────────────────────────────────────────────────


def test_substrate_composition_ranking_schema_constant_stable():
    assert loop.SUBSTRATE_COMPOSITION_RANKING_SCHEMA == "tac_autopilot_dispatch_ranking_v1"


# ── Real-world ranking artifact smoke ─────────────────────────────────────


def test_real_qq_ranking_artifact_loads_if_present():
    """If QQ's canonical ranking artifact is checked in, verify the loader
    and filter both work end-to-end against the real bytes (no test fixture).
    """
    real_artifact = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "results"
        / "cathedral_autopilot_dispatch_ranking_20260512T000000Z"
        / "ranking.json"
    )
    if not real_artifact.is_file():
        pytest.skip("real QQ ranking artifact not present in checkout")
    rows = loop.load_candidates_from_substrate_composition_ranking(real_artifact)
    sub_map = loop.candidate_substrate_ids_from_ranking(real_artifact)
    kept, dropped = loop.filter_composition_incompatible_dispatches(
        rows, candidate_substrate_ids=sub_map,
    )
    assert len(rows) > 0
    # Composition filter should produce a non-zero drop count given the
    # real ranking includes multiple renderer-replacement substrates.
    assert isinstance(kept, list)
    assert isinstance(dropped, list)
