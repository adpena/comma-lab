# SPDX-License-Identifier: MIT
"""Tests for ``tools.build_composition_ranking_json``.

Per FIX-C operator scope 2026-05-12 + CLAUDE.md "Recursive adversarial
review protocol", these tests cover:

1. enumerate → filter → rank → JSON-emit roundtrip.
2. Schema compatibility with the autopilot's existing
   ``tac_autopilot_dispatch_ranking_v1`` consumer.
3. ``score_claim`` / ``promotion_eligible`` /
   ``ready_for_exact_eval_dispatch`` invariants (all False).
4. /tmp path refusal.
5. EV/$ descending sort (the autopilot's primary signal).
6. Envelope caps annotate (don't drop).
7. Axis-weight override.
8. Posterior + cost-band opt-out paths.
9. The composition_notes carry the [predicted; ...] tag.
10. The bridge-generated JSON loads through the autopilot's
    ``load_candidates_from_substrate_composition_ranking`` consumer
    WITHOUT modification.

Contrarian's binding challenge: "does the ranking actually reflect the
posterior-driven order, or does it default to alphabetical / by-
substrate?"  Test ``test_ranking_is_eig_per_dollar_descending_not_alpha``
pins that EV/$ — not alphabetical — drives the sort.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

# Repo-root import of the tools/ module.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRIDGE_PATH = _REPO_ROOT / "tools" / "build_composition_ranking_json.py"


def _load_bridge_module():
    import sys
    name = "build_composition_ranking_json"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(_BRIDGE_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses needs this for forward refs.
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bridge():
    return _load_bridge_module()


def test_enumerate_to_payload_smoke(bridge):
    """The bridge end-to-end produces a non-empty, schema-tagged payload."""
    payload = bridge.build_payload(max_primitives_per_cell=1)
    assert isinstance(payload, dict)
    assert payload["schema"] == "tac_autopilot_dispatch_ranking_v1"
    assert payload["bridge_schema"] == "tac_composition_cell_to_autopilot_bridge_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["n_ranked_dispatches"] > 0
    assert isinstance(payload["ranked_dispatches"], list)


def test_every_row_carries_score_claim_invariants(bridge):
    payload = bridge.build_payload(max_primitives_per_cell=1)
    for row in payload["ranked_dispatches"]:
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False


def test_composition_notes_carry_predicted_tag(bridge):
    payload = bridge.build_payload(max_primitives_per_cell=1)
    for row in payload["ranked_dispatches"]:
        assert "[predicted;" in row["composition_notes"], (
            f"row {row['candidate_id']} missing [predicted; ...] tag"
        )


def test_ranking_is_clean_first_then_eig_per_dollar_descending_not_alpha(bridge):
    """Contrarian challenge: clean rows rank first; each partition sorts by EV/$."""
    payload = bridge.build_payload(max_primitives_per_cell=1)
    rows = payload["ranked_dispatches"]
    assert len(rows) >= 2
    first_review = next(
        (i for i, row in enumerate(rows) if row["operator_review_required"]),
        len(rows),
    )
    assert all(not row["operator_review_required"] for row in rows[:first_review])
    assert all(row["operator_review_required"] for row in rows[first_review:])
    for partition in (rows[:first_review], rows[first_review:]):
        for i in range(len(partition) - 1):
            a = partition[i]["eig_per_dollar"]
            b = partition[i + 1]["eig_per_dollar"]
            assert a >= b, (
                f"rows not EV/$-descending inside partition at index {i}: "
                f"{partition[i]['candidate_id']}({a}) < "
                f"{partition[i + 1]['candidate_id']}({b})"
            )
    # If the sort were alphabetical, candidate_ids would be lexicographically
    # ascending. Verify the actual output is NOT alphabetical (catches a
    # silent regression where someone replaces the EV/$ key with str).
    cand_ids = [r["candidate_id"] for r in rows]
    assert cand_ids != sorted(cand_ids), (
        "rows are sorted alphabetically — EV/$ ranking failed"
    )


def test_envelope_caps_annotate_not_drop(bridge):
    """Envelope caps mark fit-flags but do NOT remove rows from the list."""
    payload = bridge.build_payload(
        max_primitives_per_cell=1,
        per_dispatch_cap_usd=0.01,    # Trivially low to trigger flags.
        cumulative_cap_usd=0.01,
    )
    rows = payload["ranked_dispatches"]
    assert len(rows) > 0
    # At least one row should be flagged as not-fits given the trivial cap.
    not_fits = [r for r in rows if not r["fits_per_dispatch_cap"]]
    not_envelope = [r for r in rows if not r["fits_cumulative_envelope"]]
    assert not_fits or not_envelope, (
        "trivially-low envelope did not annotate any rows"
    )


def test_axis_weight_override_changes_eig(bridge):
    """A higher pose-axis weight should reorder the ranking among pose substrates."""
    p1 = bridge.build_payload(
        max_primitives_per_cell=1,
        axis_weights={"pose": 1.0, "seg": 1.0, "rate": 1.0, "mixed": 1.0},
        apply_posterior=False,  # Pin posterior so axis-weight is the only variable.
    )
    p2 = bridge.build_payload(
        max_primitives_per_cell=1,
        axis_weights={"pose": 100.0, "seg": 1.0, "rate": 1.0, "mixed": 1.0},
        apply_posterior=False,
    )
    # Find a pose substrate cell in both.
    pose_ev_p1 = next(
        (
            r["expected_information_gain"]
            for r in p1["ranked_dispatches"]
            if "pose" in r["composition_notes"].lower()
        ),
        None,
    )
    pose_ev_p2 = next(
        (
            r["expected_information_gain"]
            for r in p2["ranked_dispatches"]
            if "pose" in r["composition_notes"].lower()
        ),
        None,
    )
    assert pose_ev_p1 is not None and pose_ev_p2 is not None
    assert pose_ev_p2 > pose_ev_p1, (
        f"axis weight 100x did not amplify pose EV: {pose_ev_p1} -> {pose_ev_p2}"
    )


def test_only_with_primitives_drops_bare(bridge):
    p_with_bare = bridge.build_payload(max_primitives_per_cell=1, only_with_primitives=False)
    p_no_bare = bridge.build_payload(max_primitives_per_cell=1, only_with_primitives=True)
    bare_cell_ids_with = {
        r["candidate_id"] for r in p_with_bare["ranked_dispatches"]
        if r["candidate_id"].endswith("__bare")
    }
    bare_cell_ids_without = {
        r["candidate_id"] for r in p_no_bare["ranked_dispatches"]
        if r["candidate_id"].endswith("__bare")
    }
    assert bare_cell_ids_with, "expected some bare cells in the with-bare payload"
    assert not bare_cell_ids_without, "bare cells leaked through only_with_primitives"


def test_max_total_caps_emitted_rows(bridge):
    p = bridge.build_payload(max_primitives_per_cell=1, max_total=5)
    assert len(p["ranked_dispatches"]) <= 5
    assert p["n_ranked_dispatches"] <= 5


def test_refuses_tmp_paths(bridge, tmp_path):
    """Per CLAUDE.md Forbidden /tmp paths non-negotiable."""
    payload = bridge.build_payload(max_primitives_per_cell=0)
    for bad in (Path("/tmp/foo.json"), Path("/var/tmp/foo.json"),
                Path("/private/tmp/foo.json")):
        with pytest.raises(ValueError, match="forbidden /tmp path"):
            bridge.write_payload(payload, bad)


def test_write_then_load_roundtrip(bridge, tmp_path):
    """The bridge-generated JSON loads through the autopilot's existing consumer."""
    payload = bridge.build_payload(max_primitives_per_cell=1, max_total=20)
    out = tmp_path / "ranking.json"
    bridge.write_payload(payload, out)
    assert out.is_file()
    # JSON-roundtrip via the file.
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["schema"] == "tac_autopilot_dispatch_ranking_v1"
    assert reloaded["n_ranked_dispatches"] == payload["n_ranked_dispatches"]
    # Now load through the autopilot's actual consumer to PROVE schema parity.
    # We import the loader as a regular module since tools/ is on PYTHONPATH.
    import importlib.util as iu
    import sys as _sys
    name = "cathedral_autopilot_autonomous_loop"
    if name in _sys.modules:
        autopilot = _sys.modules[name]
    else:
        spec = iu.spec_from_file_location(
            name,
            str(_REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py"),
        )
        assert spec is not None and spec.loader is not None
        autopilot = iu.module_from_spec(spec)
        _sys.modules[name] = autopilot
        spec.loader.exec_module(autopilot)
    rows = autopilot.load_candidates_from_substrate_composition_ranking(out)
    # Round-trip must produce at least one CandidateRow (rows pass envelope by default).
    assert len(rows) > 0
    # Every row's candidate_id matches a row in our payload.
    payload_ids = {r["candidate_id"] for r in payload["ranked_dispatches"]}
    for row in rows:
        assert row.candidate_id in payload_ids


def test_campaign_metadata_emitted_for_z3_bare_cell(bridge):
    payload = bridge.build_payload(
        max_primitives_per_cell=0,
        only_with_primitives=False,
        apply_posterior=False,
        apply_cost_band=False,
    )
    z3 = next(
        row for row in payload["ranked_dispatches"]
        if row["substrate_ids"] == ["z3_balle_hyperprior_bolton"]
    )
    assert z3["lane_class"] == "substrate_engineering substrate_class_shift"
    assert z3["literature_anchor"] == "balle_2018"
    assert "campaign_id=lane_z3_balle_hyperprior_bolton_campaign_20260514" in z3[
        "campaign_metadata"
    ]
    assert "literature_anchor=balle_2018" in z3["composition_notes"]
    assert z3["source_supports"].startswith("Scale-hyperprior")
    assert "not frozen-A1 contest latent replacement" in z3["paper_claim_scope"]
    assert "paired contest CPU/CUDA eval" in z3["pact_must_prove"]
    assert "T4 inflate-cost anchor" in z3["decode_complexity_evidence"]
    assert z3["prediction_band_verdict"]["valid_for_rank_reward"] is False
    assert "prediction_band_missing" in z3["blockers"]
    assert "prediction_band_blockers" in z3["composition_notes"]
    assert "source_supports=" in z3["composition_notes"]
    assert "paper_claim_scope=" in z3["composition_notes"]
    assert "pact_must_prove=" in z3["composition_notes"]
    assert "decode_complexity_evidence=" in z3["composition_notes"]
    assert any(
        item.startswith("source_supports=Scale-hyperprior")
        for item in z3["source_fidelity_metadata"]
    )


def test_blockers_propagate_from_cell(bridge):
    """Any cell-level blockers (e.g., PR101 GOLD on non-HNeRV) propagate through."""
    payload = bridge.build_payload(max_primitives_per_cell=4)
    any_blocked = [
        r for r in payload["ranked_dispatches"] if r["blockers"]
    ]
    assert any_blocked, (
        "expected at least one row to carry a propagated cell blocker "
        "(PR101 GOLD soft-blocker or cost_estimation_required)"
    )


def test_semantic_warning_requires_operator_review(bridge):
    payload = bridge.build_payload(max_primitives_per_cell=4)
    warned = [
        r
        for r in payload["ranked_dispatches"]
        if r["semantic_compatibility_warning"] is not None
    ]

    assert warned
    for row in warned:
        assert row["operator_review_required"] is True
        assert any("semantic_compatibility_warning" in b for b in row["blockers"])
        assert "semantic_compatibility_warning:" in row["composition_notes"]
    assert payload["n_operator_review_rows"] >= len(warned)
    assert payload["n_clean_dispatch_rows"] + payload["n_operator_review_rows"] == len(
        payload["ranked_dispatches"]
    )


def test_no_inf_or_nan_in_json(bridge, tmp_path):
    """allow_nan=False — verify no inf/nan leak into emitted JSON."""
    payload = bridge.build_payload(max_primitives_per_cell=1)
    out = tmp_path / "ranking.json"
    bridge.write_payload(payload, out)
    raw = out.read_text(encoding="utf-8")
    assert "Infinity" not in raw, "JSON output contains Infinity (RFC 8259 violation)"
    assert "NaN" not in raw, "JSON output contains NaN (RFC 8259 violation)"
    # And the round-tripped numerics are finite.
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    for row in reloaded["ranked_dispatches"]:
        assert math.isfinite(row["predicted_score_delta"])
        assert math.isfinite(row["expected_information_gain"])
        assert math.isfinite(row["eig_per_dollar"])
        assert math.isfinite(row["estimated_dispatch_cost_usd"])


def test_wire_in_hooks_declared(bridge):
    """The 6-hook wire-in declaration per CLAUDE.md Catalog #125."""
    payload = bridge.build_payload(max_primitives_per_cell=0)
    hooks = payload.get("wire_in_hooks_exercised", [])
    # The bridge declares it exercises hooks 1 (sensitivity-map axis weighting),
    # 4 (cathedral autopilot dispatch), 5 (continual-learning posterior).
    assert any("hook_1" in h for h in hooks), "hook 1 not declared"
    assert any("hook_4" in h for h in hooks), "hook 4 not declared"
    assert any("hook_5" in h for h in hooks), "hook 5 not declared"


def test_apply_posterior_off_zero_observations(bridge):
    """With apply_posterior=False, predicted_score_delta is unmodified by the posterior."""
    p_off = bridge.build_payload(
        max_primitives_per_cell=0, apply_posterior=False,
    )
    # The bare-substrate rows have empty primitives; their predicted_score_delta
    # equals substrate.predicted_delta_alone_midpoint exactly when posterior is off.
    from tac.optimization.substrate_composition_matrix import canonical_substrate_inventory
    sub_idx = {s.substrate_id: s for s in canonical_substrate_inventory()}
    for row in p_off["ranked_dispatches"]:
        sid = row["substrate_ids"][0]
        expected = sub_idx[sid].predicted_delta_alone_midpoint()
        assert math.isclose(
            row["predicted_score_delta"], expected, rel_tol=1e-9, abs_tol=1e-12,
        ), f"posterior-off delta mismatch for {sid}"


def test_evidence_grade_and_claude_md_tags(bridge):
    payload = bridge.build_payload(max_primitives_per_cell=0)
    assert "planning_only" in payload["evidence_grade"]
    tags = payload["claude_md_compliance_tags"]
    assert "planning_only_no_score_claim" in tags
    assert "no_tmp_paths" in tags
    assert "operator_gate_non_negotiable_at_every_dispatch" in tags


def test_cli_writes_file(bridge, tmp_path):
    """CLI smoke: --output writes a real file, exits 0."""
    out = tmp_path / "cli_ranking.json"
    rc = bridge.main([
        "--output", str(out),
        "--max-primitives-per-cell", "0",
        "--max-total", "10",
    ])
    assert rc == 0
    assert out.is_file()
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["schema"] == "tac_autopilot_dispatch_ranking_v1"
    assert len(reloaded["ranked_dispatches"]) <= 10


def test_cli_refuses_negative_max_primitives(bridge, tmp_path):
    out = tmp_path / "cli_ranking.json"
    rc = bridge.main([
        "--output", str(out),
        "--max-primitives-per-cell", "-1",
    ])
    assert rc == 2


def test_cli_refuses_negative_axis_weight(bridge, tmp_path):
    out = tmp_path / "cli_ranking.json"
    rc = bridge.main([
        "--output", str(out),
        "--max-primitives-per-cell", "0",
        "--pose-axis-weight", "-1.5",
    ])
    assert rc == 2
