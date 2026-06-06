# SPDX-License-Identifier: MIT
"""Behavior tests for the pairwise commutator ledger over ActionEffect v1.

These tests assert BEHAVIOR, not constants: a marker-stub that returned canonical
fields without computing the real commutator arithmetic would FAIL the
hand-checked-arithmetic, basis-consistency, authority-mismatch, classification,
and queue-emission tests below.

All numeric inputs are SYNTHETIC FIXTURES (hand-chosen distortion/byte endpoints)
exercised through the REAL ``ActionEffect`` + ``tac.score_geometry.contest_score``
scoring path; they carry no empirical / contest-score authority.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tac.analysis.action_commutator import (
    ACTION_COMMUTATOR_LEDGER_SCHEMA,
    ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA,
    ACTION_COMMUTATOR_SCHEMA,
    BASIS_NONRATE,
    BASIS_TOTAL,
    CLASSIFICATION_ADDITIVE,
    CLASSIFICATION_CONFLICTING,
    CLASSIFICATION_SYNERGISTIC,
    ActionCommutatorError,
    build_commutator_ledger,
    commutator_value,
    ledger_from_dict,
)
from tac.analysis.action_effect import (
    ACTION_EFFECT_V1_SCHEMA,
    ActionEffect,
    append_action_effect,
)
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS

# ── fixture builders (SYNTHETIC; no empirical authority) ────────────────────


def _seg_effect(
    action_id: str,
    *,
    authority: str = "fakequant_mlx",
    old_d_seg: float = 0.10,
    new_d_seg: float = 0.10,
    bytes_: int = 1000,
) -> ActionEffect:
    """A byte-priced (total-basis) effect that moves ONLY d_seg.

    pose held constant so ``delta_score_total == 100*(new_d_seg-old_d_seg)``
    (bytes unchanged ⇒ zero rate term), which makes commutator arithmetic
    hand-checkable.
    """

    return ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        authority=authority,
        producer="fixture",
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=0.10,
        new_d_pose=0.10,
        old_bytes=bytes_,
        new_bytes=bytes_,
    )


def _nonrate_only_effect(action_id: str, *, old_d_seg: float, new_d_seg: float) -> ActionEffect:
    """A distortion-only effect (no bytes ⇒ delta_score_total is None)."""

    return ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        authority="fakequant_mlx",
        producer="fixture",
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=0.10,
        new_d_pose=0.10,
        old_bytes=None,
        new_bytes=None,
    )


# ── 1. exact comm arithmetic (hand-checked, total basis) ────────────────────


def test_commutator_exact_synergistic_arithmetic_total_basis():
    # a: 100*(0.08-0.10) = -2.0 ; b: 100*(0.09-0.10) = -1.0 ; ab: -5.0
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    assert a.delta_score_total == pytest.approx(-2.0)
    assert b.delta_score_total == pytest.approx(-1.0)
    assert ab.delta_score_total == pytest.approx(-5.0)
    row = commutator_value(a, b, ab)
    # comm = -5.0 - (-2.0) - (-1.0) = -2.0
    assert row["comm"] == pytest.approx(-2.0)
    assert row["synergy_score_units"] == pytest.approx(2.0)
    assert row["basis"] == BASIS_TOTAL
    assert row["classification"] == CLASSIFICATION_SYNERGISTIC
    assert row["macro_action_recommended"] is True
    assert row["schema"] == ACTION_COMMUTATOR_SCHEMA


def test_commutator_exact_conflicting_arithmetic():
    # a: -2.0 ; b: -1.0 ; ab only -1.0 (composite gave back score the parts
    # promised) => comm = -1.0 - (-2.0) - (-1.0) = +2.0  => conflicting
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.09)
    row = commutator_value(a, b, ab)
    assert row["comm"] == pytest.approx(2.0)
    assert row["synergy_score_units"] == pytest.approx(-2.0)
    assert row["classification"] == CLASSIFICATION_CONFLICTING
    assert row["macro_action_recommended"] is False


def test_commutator_exact_additive_arithmetic():
    # ab delta == a + b exactly => comm == 0 => additive
    a = _seg_effect("A", new_d_seg=0.08)  # -2.0
    b = _seg_effect("B", new_d_seg=0.09)  # -1.0
    ab = _seg_effect("A__then__B", new_d_seg=0.07)  # -3.0 == (-2)+(-1)
    row = commutator_value(a, b, ab)
    assert row["comm"] == pytest.approx(0.0, abs=1e-9)
    assert row["classification"] == CLASSIFICATION_ADDITIVE
    assert row["macro_action_recommended"] is False


# ── 2. classification thresholds honor eps ──────────────────────────────────


def test_classification_thresholds_respect_eps():
    a = _seg_effect("A", new_d_seg=0.10)  # delta 0
    b = _seg_effect("B", new_d_seg=0.10)  # delta 0
    # ab moves seg by -0.0005 -> delta -0.05 ; comm = -0.05 - 0 - 0 = -0.05
    ab = _seg_effect("A__then__B", new_d_seg=0.0995)
    # with a small eps it is synergistic
    small = commutator_value(a, b, ab, eps=1e-3)
    assert small["comm"] == pytest.approx(-0.05)
    assert small["classification"] == CLASSIFICATION_SYNERGISTIC
    # with a large eps (band wider than |comm|) it reads additive
    large = commutator_value(a, b, ab, eps=1.0)
    assert large["classification"] == CLASSIFICATION_ADDITIVE


def test_negative_eps_rejected():
    a = _seg_effect("A")
    with pytest.raises(ValueError):
        commutator_value(a, a, a, eps=-1e-9)


# ── 3. basis consistency rule (NEVER mix total and nonrate) ─────────────────


def test_basis_falls_back_to_nonrate_when_any_row_lacks_bytes():
    # a and ab are byte-priced (total available); b is distortion-only (no bytes).
    # The rule: if ANY of the three lacks total, use nonrate for ALL THREE.
    a = _seg_effect("A", new_d_seg=0.08)  # total -2.0, nonrate -2.0
    b = _nonrate_only_effect("B", old_d_seg=0.10, new_d_seg=0.09)  # total None, nonrate -1.0
    ab = _seg_effect("A__then__B", new_d_seg=0.05)  # total -5.0, nonrate -5.0
    assert b.delta_score_total is None
    assert b.delta_score_nonrate == pytest.approx(-1.0)
    row = commutator_value(a, b, ab)
    assert row["basis"] == BASIS_NONRATE
    # nonrate deltas: a=-2.0, b=-1.0, ab=-5.0 => comm -2.0
    assert row["delta_a"] == pytest.approx(-2.0)
    assert row["delta_b"] == pytest.approx(-1.0)
    assert row["delta_ab"] == pytest.approx(-5.0)
    assert row["comm"] == pytest.approx(-2.0)


def test_basis_uses_total_when_all_three_byte_priced():
    # Add a real rate movement so total != nonrate, proving total is the basis used.
    a = ActionEffect.build(
        action_id="A", family="hinerv", authority="fakequant_mlx", producer="fixture",
        old_d_seg=0.10, new_d_seg=0.10, old_d_pose=0.10, new_d_pose=0.10,
        old_bytes=1000, new_bytes=2000,  # +1000 bytes -> positive rate delta
    )
    b = _seg_effect("B", new_d_seg=0.10)  # all-zero delta
    ab = ActionEffect.build(
        action_id="A__then__B", family="hinerv", authority="fakequant_mlx", producer="fixture",
        old_d_seg=0.10, new_d_seg=0.10, old_d_pose=0.10, new_d_pose=0.10,
        old_bytes=1000, new_bytes=2000,
    )
    row = commutator_value(a, b, ab)
    assert row["basis"] == BASIS_TOTAL
    # a total = rate delta only (nonzero); nonrate would be 0 -> proves total used
    assert a.delta_score_nonrate == pytest.approx(0.0)
    assert a.delta_score_total != pytest.approx(0.0)
    assert row["delta_a"] == pytest.approx(a.delta_score_total)


def test_commutator_undefined_when_no_consistent_basis_raises():
    # ab has neither total nor nonrate (no distortion endpoints, no bytes).
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = ActionEffect.build(
        action_id="A__then__B", family="hinerv", authority="fakequant_mlx", producer="fixture",
        old_d_seg=None, new_d_seg=None, old_d_pose=None, new_d_pose=None,
        old_bytes=None, new_bytes=None,
    )
    assert ab.delta_score_total is None
    assert ab.delta_score_nonrate is None
    with pytest.raises(ActionCommutatorError):
        commutator_value(a, b, ab)


# ── 4. authority is a type (mismatch raises) ────────────────────────────────


def test_authority_mismatch_raises():
    a = _seg_effect("A", new_d_seg=0.08, authority="fakequant_mlx")
    b = _seg_effect("B", new_d_seg=0.09, authority="inflated_torch_cpu")
    ab = _seg_effect("A__then__B", new_d_seg=0.05, authority="fakequant_mlx")
    with pytest.raises(ActionCommutatorError) as exc:
        commutator_value(a, b, ab)
    assert "authority" in str(exc.value).lower()


def test_authority_matches_returns_that_authority():
    a = _seg_effect("A", new_d_seg=0.08, authority="parseback_mlx")
    b = _seg_effect("B", new_d_seg=0.09, authority="parseback_mlx")
    ab = _seg_effect("A__then__B", new_d_seg=0.05, authority="parseback_mlx")
    row = commutator_value(a, b, ab)
    assert row["authority"] == "parseback_mlx"


# ── 5. false-authority markers present (planning row, never a score claim) ───


def test_commutator_row_carries_false_authority_markers():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    row = commutator_value(a, b, ab)
    for key, expected in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert row[key] is expected
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False


# ── 6. ledger: measured vs needs-measurement queue ──────────────────────────


def test_ledger_emits_measured_row_and_queue_for_missing_reverse_pair():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)  # only the (A,B) order measured
    ledger = build_commutator_ledger([a, b], [ab])
    assert ledger["schema"] == ACTION_COMMUTATOR_LEDGER_SCHEMA
    assert ledger["ordered_pair_count"] == 2  # (A,B) and (B,A)
    assert ledger["measured_commutator_count"] == 1
    assert ledger["needs_measurement_count"] == 1
    # the measured one is the synergistic (A,B)
    assert ledger["macro_action_candidates"][0]["first_action_id"] == "A"
    assert ledger["macro_action_candidates"][0]["second_action_id"] == "B"
    # the queued one is the (B,A) order
    q = ledger["measurement_queue"][0]
    assert q["schema"] == ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA
    assert q["first_action_id"] == "B"
    assert q["second_action_id"] == "A"
    assert q["proposed_composite_action_id"] == "B__then__A"
    assert q["comm"] is None  # NEVER fabricated


def test_ledger_queue_when_no_pair_effects_at_all():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    c = _seg_effect("C", new_d_seg=0.07)
    ledger = build_commutator_ledger([a, b, c], [])
    # 3 distinct singles -> 3*2 = 6 ordered pairs, all unmeasured
    assert ledger["ordered_pair_count"] == 6
    assert ledger["measured_commutator_count"] == 0
    assert ledger["needs_measurement_count"] == 6
    assert all(r["comm"] is None for r in ledger["measurement_queue"])


def test_ledger_queues_pair_with_incompatible_authority_never_fabricates():
    a = _seg_effect("A", new_d_seg=0.08, authority="fakequant_mlx")
    b = _seg_effect("B", new_d_seg=0.09, authority="inflated_torch_cpu")
    # measured composite exists but parts disagree on authority -> queued, not measured
    ab = _seg_effect("A__then__B", new_d_seg=0.05, authority="fakequant_mlx")
    ledger = build_commutator_ledger([a, b], [ab])
    assert ledger["measured_commutator_count"] == 0
    assert ledger["needs_measurement_count"] == 2
    incompat = [r for r in ledger["measurement_queue"] if r["first_action_id"] == "A" and r["second_action_id"] == "B"]
    assert len(incompat) == 1
    assert incompat[0]["authority_compatible"] is False
    assert "incompatible" in incompat[0]["reason"]


def test_ledger_separates_synergistic_and_conflicting_and_sorts():
    a = _seg_effect("A", new_d_seg=0.08)  # -2.0
    b = _seg_effect("B", new_d_seg=0.09)  # -1.0
    # (A,B) strongly synergistic: ab -6.0 -> comm -3.0
    ab = _seg_effect("A__then__B", new_d_seg=0.04)
    # (B,A) conflicting: ba -0.5 -> comm = -0.5 -(-1)-(-2) = +2.5
    ba = _seg_effect("B__then__A", new_d_seg=0.095)
    ledger = build_commutator_ledger([a, b], [ab, ba])
    assert ledger["measured_commutator_count"] == 2
    assert ledger["synergistic_count"] == 1
    assert ledger["conflicting_count"] == 1
    assert ledger["macro_action_candidates"][0]["comm"] == pytest.approx(-3.0)
    assert ledger["conflict_pairs"][0]["comm"] == pytest.approx(2.5)


def test_ledger_macro_candidates_sorted_most_synergistic_first():
    a = _seg_effect("A", new_d_seg=0.10)
    b = _seg_effect("B", new_d_seg=0.10)
    c = _seg_effect("C", new_d_seg=0.10)
    # A->B comm -1.0 ; A->C comm -3.0 (more synergistic) -> C should rank first
    ab = _seg_effect("A__then__B", new_d_seg=0.099)  # delta -0.1 ; comm -0.1
    ac = _seg_effect("A__then__C", new_d_seg=0.097)  # delta -0.3 ; comm -0.3
    ledger = build_commutator_ledger([a, b, c], [ab, ac])
    macros = ledger["macro_action_candidates"]
    assert len(macros) == 2
    assert macros[0]["comm"] <= macros[1]["comm"]
    assert macros[0]["second_action_id"] == "C"


def test_ledger_respects_top_k_caps():
    singles = [_seg_effect(f"A{i}", new_d_seg=0.10) for i in range(5)]
    # build several synergistic composites
    pairs = []
    for i in range(4):
        pairs.append(_seg_effect(f"A0__then__A{i + 1}", new_d_seg=0.10 - 0.001 * (i + 1)))
    ledger = build_commutator_ledger(singles, pairs, macro_action_limit=2, conflict_pair_limit=2)
    assert len(ledger["macro_action_candidates"]) <= 2


# ── 7. duplicate / self-composition handling ────────────────────────────────


def test_ledger_skips_self_and_duplicate_ids():
    a = _seg_effect("A", new_d_seg=0.08)
    a_dup = _seg_effect("A", new_d_seg=0.09)  # same id, different deltas
    ledger = build_commutator_ledger([a, a_dup], [])
    # one unique id -> zero ordered pairs
    assert ledger["ordered_pair_count"] == 0
    assert ledger["needs_measurement_count"] == 0


# ── 8. input type guards ────────────────────────────────────────────────────


def test_build_ledger_rejects_non_action_effect_inputs():
    with pytest.raises(TypeError):
        build_commutator_ledger([{"action_id": "A"}], [])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        build_commutator_ledger("not a sequence", [])  # type: ignore[arg-type]


def test_commutator_value_rejects_non_action_effect():
    a = _seg_effect("A", new_d_seg=0.08)
    with pytest.raises(TypeError):
        commutator_value(a, a, {"action_id": "A__then__A"})  # type: ignore[arg-type]


# ── 9. JSONL ledger round-trip ──────────────────────────────────────────────


def test_ledger_jsonl_round_trip(tmp_path: Path):
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    ledger = build_commutator_ledger([a, b], [ab])
    out = tmp_path / "ledger.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for row in ledger["rows"]:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        for row in ledger["measurement_queue"]:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    reloaded = []
    with open(out, encoding="utf-8") as fh:
        for line in fh:
            reloaded.append(json.loads(line))
    measured = [r for r in reloaded if r["schema"] == ACTION_COMMUTATOR_SCHEMA]
    queued = [r for r in reloaded if r["schema"] == ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA]
    assert len(measured) == 1
    assert measured[0]["comm"] == pytest.approx(-2.0)
    assert len(queued) == 1
    assert queued[0]["comm"] is None


def test_ledger_from_dict_validates_schema():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    ledger = build_commutator_ledger([a, b], [ab])
    round_tripped = ledger_from_dict(json.loads(json.dumps(ledger)))
    assert round_tripped["schema"] == ACTION_COMMUTATOR_LEDGER_SCHEMA
    with pytest.raises(ValueError):
        ledger_from_dict({"schema": "not_a_commutator_ledger"})


# ── 10. CLI smoke on tmp fixtures (fixtures labeled synthetic) ──────────────


def _write_effect_jsonl(effects: list[ActionEffect], path: Path) -> None:
    """Write ActionEffect rows via the canonical fcntl-locked appender."""

    for effect in effects:
        append_action_effect(effect, path)


def test_cli_smoke_emits_ledger_and_summary(tmp_path: Path):
    singles_path = tmp_path / "singles.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    out_dir = tmp_path / "out"

    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    _write_effect_jsonl([a, b], singles_path)
    _write_effect_jsonl([ab], pairs_path)

    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "run_pr110_commutator_ledger.py"),
        "--action-effects",
        str(singles_path),
        "--pair-effects",
        str(pairs_path),
        "--output",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
    assert proc.returncode == 0, proc.stderr

    summary = json.loads((out_dir / "commutator_summary.json").read_text())
    assert summary["schema"] == ACTION_COMMUTATOR_LEDGER_SCHEMA
    assert summary["measured_commutator_count"] == 1
    assert summary["needs_measurement_count"] == 1
    assert summary["macro_action_candidates"][0]["comm"] == pytest.approx(-2.0)
    # false-authority markers propagate to the summary
    assert summary["score_claim"] is False

    jsonl_rows = [
        json.loads(line) for line in (out_dir / "commutator_ledger.jsonl").read_text().splitlines() if line.strip()
    ]
    assert any(r["schema"] == ACTION_COMMUTATOR_SCHEMA for r in jsonl_rows)
    assert any(r["schema"] == ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA for r in jsonl_rows)


def test_cli_smoke_no_pair_effects_all_queued(tmp_path: Path):
    singles_path = tmp_path / "singles.jsonl"
    out_dir = tmp_path / "out2"
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    _write_effect_jsonl([a, b], singles_path)

    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "run_pr110_commutator_ledger.py"),
        "--action-effects",
        str(singles_path),
        "--output",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out_dir / "commutator_summary.json").read_text())
    assert summary["measured_commutator_count"] == 0
    assert summary["needs_measurement_count"] == 2


def test_cli_missing_action_effects_file_errors(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "run_pr110_commutator_ledger.py"),
        "--action-effects",
        str(tmp_path / "does_not_exist.jsonl"),
        "--output",
        str(tmp_path / "out3"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
    assert proc.returncode != 0


# ── 11. round-trip fidelity: schema constants are the real v1 names ──────────


def test_schema_constants_are_distinct_and_present():
    # guards against accidental schema collision with the ActionEffect surface
    assert ACTION_COMMUTATOR_SCHEMA == "tac.action_commutator.v1"
    assert ACTION_COMMUTATOR_LEDGER_SCHEMA == "tac.action_commutator_ledger.v1"
    assert ACTION_EFFECT_V1_SCHEMA == "tac.action_effect.v1"
    assert ACTION_COMMUTATOR_SCHEMA != ACTION_EFFECT_V1_SCHEMA


def test_synergy_is_exact_negative_of_comm():
    a = _seg_effect("A", new_d_seg=0.08)
    b = _seg_effect("B", new_d_seg=0.09)
    ab = _seg_effect("A__then__B", new_d_seg=0.05)
    row = commutator_value(a, b, ab)
    assert math.isclose(row["synergy_score_units"], -row["comm"], rel_tol=0, abs_tol=0)
