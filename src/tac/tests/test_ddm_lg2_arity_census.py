"""ddm_lg2 — tests for the arity-mismatch census.

Every measurement here has a POSITIVE CONTROL: a gate/meter never shown to
produce its non-default verdict is untrusted, so each subcommand is exercised
on a fixture that MUST flip it.  The empty-scope case is asserted to emit
``VACUOUS`` and never a clean verdict — the vacuity class these tests exist to
extinct.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "ddm_lg2_arity_census.py"


def _load():
    spec = importlib.util.spec_from_file_location("_ddm_lg2_arity_census", _TOOL)
    assert spec is not None and spec.loader is not None, f"cannot load {_TOOL}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_M = _load()
VACUOUS = _M.VACUOUS
bracket_direction_occupancy = _M.bracket_direction_occupancy
ladder_authority = _M.ladder_authority
smoke_proof_scope_census = _M.smoke_proof_scope_census


# --------------------------------------------------------------------------
# #822 — ladder authority
# --------------------------------------------------------------------------
def test_ladder_empty_series_is_vacuous_not_pass() -> None:
    """An empty scope must NOT emit a clean verdict."""
    out = ladder_authority(0.1, 66.2, 0.1, 5.0, [], 1.0)
    assert out["verdict"] == VACUOUS
    assert out["denominator_gates"] == 0


def test_ladder_zero_cap_is_vacuous() -> None:
    out = ladder_authority(0.1, 66.2, 0.0, 5.0, [-0.01, -0.02], 1.0)
    assert out["verdict"] == VACUOUS


def test_ladder_inert_when_rung_equals_cap_and_g_negative() -> None:
    """The measured burn-4 shape: rung == cap, every gate saturates, g < 0.

    The raise is erased at gate 1 and never revives.
    """
    g = [-0.0035, -0.0200, -0.0537, -0.0100] * 4
    out = ladder_authority(0.1, 66.2251655629139, 0.1, 5.0, g, 1.0)
    assert out["verdict"] == "LADDER_INERT"
    assert out["rung_over_cap"] == pytest.approx(1.0)
    assert out["gates_saturating_cap"] == len(g)
    assert out["lambda_after_first_gate"] == 0.0
    assert out["gates_with_lambda_gt_zero"] == 0


def test_ladder_positive_control_holds_when_rung_exceeds_one_gate_of_decay() -> None:
    """POSITIVE CONTROL: a ladder CAN climb — the meter is not stuck on INERT.

    Raise the rung above one step cap and the multiplier survives the first
    gate, so the verdict must flip.  Without this the INERT reading above
    would be unfalsifiable.
    """
    g = [-0.0035] * 8
    out = ladder_authority(0.35, 66.2251655629139, 0.1, 5.0, g, 1.0)
    assert out["verdict"] == "LADDER_HOLDS"
    assert out["gates_with_lambda_gt_zero"] > 0


def test_ladder_positive_g_lets_the_dual_climb_on_its_own() -> None:
    """When the constraint actually binds the dual ascends without any ladder."""
    out = ladder_authority(0.0, 66.2251655629139, 0.1, 5.0, [0.01] * 12, 1.0)
    assert out["gates_with_positive_g"] == 12
    assert out["verdict"] == "LADDER_HOLDS"


def test_ladder_escalation_rung_requires_cap_many_violation_gates() -> None:
    """The operator-escalation threshold is reachable only via sustained g > 0."""
    out = ladder_authority(0.1, 66.2, 0.1, 5.0, [-0.01] * 5, 1.0)
    assert out["consecutive_violation_gates_needed_to_escalate"] == pytest.approx(10.0)


# --------------------------------------------------------------------------
# #821 — smoke-proof scope census
# --------------------------------------------------------------------------
def _write_fixture_repo(tmp_path: Path, proofs: dict) -> Path:
    (tmp_path / "scripts").mkdir(parents=True)
    for name in proofs:
        (tmp_path / "scripts" / f"{name}.sh").write_text("#!/usr/bin/env bash\necho hi\n")
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / ".omx" / "state" / "lane_e2e_smoke_proofs.json").write_text(
        json.dumps(proofs)
    )
    return tmp_path


def _proof(sha: str, fixture: str = "fix.zip") -> dict:
    return {
        "fixture_archive": fixture,
        "archive_sha256": sha,
        "stages_passed": ["extract", "whitelist"],
        "submission_dir": "submissions/robust_current",
        "timestamp_utc": "2026-07-14T23:22:51Z",
        "elapsed_seconds": 0.04,
    }


def test_smoke_scope_flags_arity_mismatch_when_evidence_is_lane_independent(
    tmp_path: Path,
) -> None:
    root = _write_fixture_repo(
        tmp_path,
        {f"remote_lane_{i}": _proof("deadbeef") for i in range(5)},
    )
    out = smoke_proof_scope_census(repo_root=root)
    assert out["verdict"] == "ARITY_MISMATCH"
    assert out["denominator_lane_scripts"] == 5
    assert out["distinct_evidence_tuples"] == 1
    assert out["fan_out"] == pytest.approx(5.0)


def test_smoke_scope_positive_control_lane_specific_evidence_is_not_flagged(
    tmp_path: Path,
) -> None:
    """POSITIVE CONTROL: per-lane archives ⇒ the per-item iteration is warranted.

    Proves the census reports the mismatch because the evidence is constant,
    not because it always says ARITY_MISMATCH.
    """
    root = _write_fixture_repo(
        tmp_path,
        {f"remote_lane_{i}": _proof(f"sha{i}", f"lane{i}.zip") for i in range(5)},
    )
    out = smoke_proof_scope_census(repo_root=root)
    assert out["verdict"] == "SCOPE_VARIES_WITH_EVIDENCE"
    assert out["distinct_evidence_tuples"] == 5
    assert out["fan_out"] == pytest.approx(1.0)


def test_smoke_scope_empty_is_vacuous_not_pass(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir(parents=True)
    out = smoke_proof_scope_census(repo_root=tmp_path)
    assert out["verdict"] == VACUOUS
    assert out["denominator_lane_scripts"] == 0


def test_smoke_scope_single_proof_is_undetermined_not_clean(tmp_path: Path) -> None:
    """R1 self-review catch: n=1 cannot exhibit fan-out either way.

    Reporting SCOPE_VARIES_WITH_EVIDENCE there would be a clean verdict drawn
    from a population that cannot support one.
    """
    root = _write_fixture_repo(tmp_path, {"remote_lane_only": _proof("x")})
    out = smoke_proof_scope_census(repo_root=root)
    assert out["verdict"] == "SINGLE_PROOF_UNDETERMINED"


def test_smoke_scope_counts_opt_out_waivers(tmp_path: Path) -> None:
    root = _write_fixture_repo(tmp_path, {"remote_lane_a": _proof("x")})
    (root / "scripts" / "remote_lane_a.sh").write_text(
        "#!/usr/bin/env bash\n# E2E_SMOKE_OPT_OUT: needs 60GB GPU to build\n"
    )
    out = smoke_proof_scope_census(repo_root=root)
    assert out["waived_opt_out"] == 1


# --------------------------------------------------------------------------
# #871 — bracket-direction occupancy
# --------------------------------------------------------------------------
def _arms_row(pair: int, a_probes: list, b_probes: list, d: float = 0.01) -> dict:
    return {
        "pair": pair,
        "d_ctrl": 1.0,
        "arm_ab_d": d,
        "arm_a_probes": a_probes,
        "arm_b_probes": b_probes,
    }


def test_bracket_missing_receipt_is_vacuous(tmp_path: Path) -> None:
    out = bracket_direction_occupancy(tmp_path / "nope.jsonl")
    assert out["verdict"] == VACUOUS
    assert out["denominator_pairs"] == 0


def test_bracket_detects_untested_binary_commitment(tmp_path: Path) -> None:
    """One probe recorded ⇒ the opposite direction was never evaluated."""
    p = tmp_path / "arms.jsonl"
    rows = [
        _arms_row(0, [{"x": 1.0, "d": 0.5, "phase": "probe"}], []),
        _arms_row(
            1,
            [
                {"x": 1.0, "d": 2.0, "phase": "probe"},
                {"x": -1.0, "d": 0.5, "phase": "probe"},
            ],
            [],
        ),
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    out = bracket_direction_occupancy(p)
    assert out["verdict"] == "UNTESTED_BINARY_COMMITMENT"
    a = out["arms"]["arm_a_probes"]
    assert a["pairs_first_direction_committed_untested"] == 1
    assert a["pairs_both_directions_evaluated"] == 1
    assert a["of_those_only_second_direction_improved"] == 1
    assert out["union_untested_pairs"] == 1


def test_bracket_positive_control_symmetric_search_is_clean(tmp_path: Path) -> None:
    """POSITIVE CONTROL: when both directions are always evaluated the meter clears.

    Guards against a detector that reports the defect unconditionally.
    """
    p = tmp_path / "arms.jsonl"
    both = [
        {"x": 1.0, "d": 2.0, "phase": "probe"},
        {"x": -1.0, "d": 2.0, "phase": "probe"},
    ]
    p.write_text(
        "\n".join(json.dumps(_arms_row(i, both, both)) for i in range(4))
    )
    out = bracket_direction_occupancy(p)
    assert out["verdict"] == "BOTH_DIRECTIONS_ALWAYS_TESTED"
    assert out["union_untested_pairs"] == 0


def test_bracket_ignores_expand_phase_probes(tmp_path: Path) -> None:
    """Only the two INITIAL probes decide the direction; doubling probes must not
    make a short-circuited pair look symmetric."""
    p = tmp_path / "arms.jsonl"
    rows = [
        _arms_row(
            0,
            [
                {"x": 1.0, "d": 0.5, "phase": "probe"},
                {"x": 2.0, "d": 0.4, "phase": "expand"},
                {"x": 4.0, "d": 0.9, "phase": "expand"},
            ],
            [],
        )
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    out = bracket_direction_occupancy(p)
    assert out["arms"]["arm_a_probes"]["pairs_first_direction_committed_untested"] == 1


def test_bracket_threshold_proxy_selfcheck_flags_a_bad_threshold(tmp_path: Path) -> None:
    """R1 self-review catch: `d_ctrl` is a PROXY for the bracket's entry best_d.

    The `break` guarantees a first probe that truly improved leaves ONE probe,
    so a first-probe "improvement" inside the both-evaluated bucket proves the
    threshold is wrong.  The self-check must catch that, not silently report.
    """
    p = tmp_path / "arms.jsonl"
    rows = [
        _arms_row(
            0,
            [
                {"x": 1.0, "d": 0.1, "phase": "probe"},   # "improves" vs d_ctrl=1.0 …
                {"x": -1.0, "d": 0.2, "phase": "probe"},  # … yet BOTH were recorded
            ],
            [],
        )
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    out = bracket_direction_occupancy(p)
    a = out["arms"]["arm_a_probes"]
    assert a["consistency_first_improved_in_both_bucket"] == 1
    assert a["threshold_proxy_valid"] is False


def test_bracket_threshold_proxy_valid_on_consistent_receipt(tmp_path: Path) -> None:
    """POSITIVE CONTROL for the self-check: a consistent receipt reports valid."""
    p = tmp_path / "arms.jsonl"
    rows = [
        _arms_row(
            0,
            [
                {"x": 1.0, "d": 2.0, "phase": "probe"},
                {"x": -1.0, "d": 0.5, "phase": "probe"},
            ],
            [],
        )
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    a = bracket_direction_occupancy(p)["arms"]["arm_a_probes"]
    assert a["consistency_first_improved_in_both_bucket"] == 0
    assert a["threshold_proxy_valid"] is True


def test_bracket_refuses_malformed_receipt_loudly(tmp_path: Path) -> None:
    """R2 self-review: silently skipping bad lines would shrink the denominator."""
    p = tmp_path / "arms.jsonl"
    p.write_text('{"pair": 0}\nnot json at all\n')
    with pytest.raises(ValueError, match="not valid JSON"):
        bracket_direction_occupancy(p)


def test_smoke_scope_all_rows_unusable_is_vacuous(tmp_path: Path) -> None:
    """R2 self-review: zero usable evidence tuples is an EMPTY scope, not a finding."""
    root = _write_fixture_repo(tmp_path, {"remote_lane_a": {}, "remote_lane_b": {}})
    (root / ".omx" / "state" / "lane_e2e_smoke_proofs.json").write_text(
        json.dumps({"remote_lane_a": "bogus", "remote_lane_b": 7})
    )
    out = smoke_proof_scope_census(repo_root=root)
    assert out["verdict"] == VACUOUS
    assert out["distinct_evidence_tuples"] == 0


def test_bracket_mass_fraction_is_reported(tmp_path: Path) -> None:
    p = tmp_path / "arms.jsonl"
    rows = [
        _arms_row(0, [{"x": 1.0, "d": 0.5, "phase": "probe"}], [], d=3.0),
        _arms_row(
            1,
            [
                {"x": 1.0, "d": 2.0, "phase": "probe"},
                {"x": -1.0, "d": 2.0, "phase": "probe"},
            ],
            [],
            d=1.0,
        ),
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    out = bracket_direction_occupancy(p)
    assert out["union_pct_of_dpose_mass"] == pytest.approx(75.0)
