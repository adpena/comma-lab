# SPDX-License-Identifier: MIT
"""WALL-CLOCK-DEFAULT-ON + PERF-ENV CLASS GUARD (operator 2026-07-08:
"The wall clock stuff should be default on always and also that shouldn't have had to be caught
manually").

Guards two structural fixes on top of the compute audit (commit 8d9dabc92):
  * FIX 1 — wall-clock gating is DEFAULT-ON: the typed config carries a REQUIRED, DERIVED
    ``wall_clock_budget_days`` (forgetting it is a schema error); the launcher projects the MEASURED
    SegNet bench x epochs and REFUSES (rc=8) over the declared/derived budget with NO opt-in flag;
    ``--accept-wall-clock`` is the loud, stamped escape hatch.
  * FIX 2 — the perf-env CLASS guard: the required perf-env set is DERIVED from PERF_ENV_PREFIX (SoT,
    never a duplicate list); a launch.sh missing the ~17x var is a structural REFUSE naming the var;
    BOTH to_command paths consume the ONE prefix constant (drift-impossible).
  * FIX 3 — throughput-vs-budget coupling: a bench that passes the 700ms absolute gate but is slower
    than the budget-implied ceiling STILL projects over-budget (a non-env perf regression).

Pure/CPU — no GPU, no scorer weights, no launch (the live run is untouched).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "src"), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(_REPO / rel))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _budget(value=3.7, provenance=None):
    from tac.witness_dsl.typed_config import ProvenanceClass, Provenanced
    return Provenanced(value=value, unit="days",
                       provenance=provenance or ProvenanceClass.DERIVED_AT_CONFIG,
                       source="derive_wall_clock_budget_days(epochs)")


def _valid_typed(**over):
    from tac.witness_dsl.typed_config import (
        ProvenanceClass,
        Provenanced,
        TypedAnneal,
        TypedWitnessConfig,
    )
    kw = dict(
        name="t", out_dir="experiments/results/x", gt_cache="g.npz", num_pairs=600, epochs=1500,
        wall_clock_budget_days=_budget(),
        temp=TypedAnneal(
            start=Provenanced(value=1.0, provenance=ProvenanceClass.MEASURED_ANCHOR, unit="tau"),
            end=Provenanced(value=0.31, provenance=ProvenanceClass.MEASURED_ANCHOR, unit="tau")),
    )
    kw.update(over)
    return TypedWitnessConfig(**kw)


# ── FIX 1a: the REQUIRED typed field makes forgetting a schema error ─────────────────────
def test_missing_budget_is_a_validation_error():
    from tac.witness_dsl.typed_config import (
        ProvenanceClass,
        Provenanced,
        TypedAnneal,
        TypedWitnessConfig,
    )
    with pytest.raises(ValidationError):
        TypedWitnessConfig(
            name="t", out_dir="o", gt_cache="g", num_pairs=600, epochs=1000,
            temp=TypedAnneal(
                start=Provenanced(value=1.0, provenance=ProvenanceClass.MEASURED_ANCHOR),
                end=Provenanced(value=0.31, provenance=ProvenanceClass.MEASURED_ANCHOR)),
        )  # no wall_clock_budget_days -> forgetting is structurally impossible


def test_budget_must_be_positive():
    with pytest.raises(ValidationError):
        _valid_typed(wall_clock_budget_days=_budget(value=0.0))


def test_budget_must_be_derived_not_hardcoded():
    from tac.witness_dsl.typed_config import ProvenanceClass, Provenanced
    # a hand-picked hardcoded budget (even with a waiver) defeats the anchor-tracking ceiling.
    hard = Provenanced(value=5.0, provenance=ProvenanceClass.HARDCODED_WITH_WAIVER, unit="days",
                       waiver="magic number:re-derive never")
    with pytest.raises(ValidationError):
        _valid_typed(wall_clock_budget_days=hard)


def test_budget_days_value_is_float():
    c = _valid_typed(wall_clock_budget_days=_budget(value=7.4))
    assert c.budget_days_value() == pytest.approx(7.4)


def test_budget_is_in_the_config_hash():
    # two configs differing only in budget must hash differently (the field is manifest-covered).
    a = _valid_typed(wall_clock_budget_days=_budget(value=3.0))
    b = _valid_typed(wall_clock_budget_days=_budget(value=9.0))
    assert a.typed_config_hash() != b.typed_config_hash()


# ── FIX 1b: the DERIVED budget math (anchor x epochs x slack) ────────────────────────────
def test_derive_budget_math_matches_anchor_times_slack():
    from tac.local_acceleration.scorer_throughput_gate import (
        RUN1_MEASURED_MIN_PER_EP,
        WALL_CLOCK_SLACK_FACTOR,
        derive_wall_clock_budget_days,
        project_wall_clock_days,
    )
    got = derive_wall_clock_budget_days(3000)
    want = project_wall_clock_days(RUN1_MEASURED_MIN_PER_EP, 3000) * WALL_CLOCK_SLACK_FACTOR
    assert got == pytest.approx(want)
    # v7.3 delta 4 (S5-H2): 3.62 live incl-startup x 3000 / 1440 = 7.5417 anchor projection.
    assert got == pytest.approx(7.5417 * WALL_CLOCK_SLACK_FACTOR, rel=1e-3)  # ~8.67 days at 3000 ep


def test_derive_budget_rejects_nonpositive_epochs():
    from tac.local_acceleration.scorer_throughput_gate import derive_wall_clock_budget_days
    with pytest.raises(ValueError):
        derive_wall_clock_budget_days(0)


def test_v7_config_declares_a_derived_seven_day_budget():
    from tac.witness_autoconfig import derive_crucible_v7_config
    c = derive_crucible_v7_config(_GT, num_pairs=600, epochs=3000,
                                  out_dir="experiments/results/__v7_budget_test__")
    from tac.witness_dsl.typed_config import ProvenanceClass
    assert c.wall_clock_budget_days.provenance is ProvenanceClass.DERIVED_AT_CONFIG
    assert 8.4 <= c.budget_days_value() <= 8.9   # ~8.67 days at 3000 ep (v7.3 S5-H2 live-cadence re-anchor)


# ── FIX 1c: the launcher budget resolver (pure, default-on) ─────────────────────────────
_lw = _load("launch_witness_run_wallclock_test", "tools/launch_witness_run.py")


def test_resolve_prefers_operator_accept_and_marks_override():
    b, src, override = _lw.resolve_wall_clock_budget(accept_days=100.0, declared_days=7.4, epochs=3000)
    assert b == 100.0 and override is True and "accept" in src.lower()


def test_resolve_uses_config_declared_when_no_accept():
    b, src, override = _lw.resolve_wall_clock_budget(accept_days=None, declared_days=7.4, epochs=3000)
    assert b == pytest.approx(7.4) and override is False and "declared" in src.lower()


def test_resolve_falls_back_to_derived_when_nothing_declared():
    # DEFAULT-ON: a legacy config that declared no budget STILL gets a refuse ceiling.
    b, src, override = _lw.resolve_wall_clock_budget(accept_days=None, declared_days=None, epochs=3000)
    # v7.3 delta 4: derived fallback now ~8.67 days (3.62 live incl-startup anchor x 3000 x 1.15 slack).
    assert b == pytest.approx(8.67, rel=1e-2) and override is False and "fallback" in src.lower()


def test_resolve_none_only_when_epochs_unknown_and_nothing_supplied():
    b, src, override = _lw.resolve_wall_clock_budget(accept_days=None, declared_days=None, epochs=0)
    assert b is None and override is False


def test_config_wall_clock_budget_reads_provenanced_and_numeric_and_none():
    class _P:
        wall_clock_budget_days = _budget(value=4.2)

    class _N:
        wall_clock_budget_days = 5.0

    class _M:
        pass

    assert _lw._config_wall_clock_budget_days(_P()) == pytest.approx(4.2)
    assert _lw._config_wall_clock_budget_days(_N()) == pytest.approx(5.0)
    assert _lw._config_wall_clock_budget_days(_M()) is None


# ── FIX 1 + FIX 3: the gate REFUSES over-budget (rc=8), incl. a fast-but-over-budget bench ──
class _StubCfg:
    def __init__(self, epochs, budget=None):
        self.epochs = epochs
        self.wall_clock_budget_days = budget

    def to_trainer_flags(self, out_dir):
        return []  # no --compile* flags


def _fast_verdict(ms):
    from tac.local_acceleration.scorer_throughput_gate import ThroughputVerdict
    return ThroughputVerdict(status="fast", segnet_fwd_bwd_ms=ms, abs_threshold_ms=700.0,
                             baseline_ms=396.0, ceiling_mult=1.5, within_abs=True,
                             within_ceiling=True)


def test_gate_refuses_over_budget_with_rc8(monkeypatch, tmp_path):
    import tac.local_acceleration.scorer_throughput_gate as stg
    # a bench of 500ms PASSES the 700ms absolute throughput gate (env present, "fast") but at 3000 ep
    # projects ~8.15 days > the ~7.4-day derived fallback budget => REFUSE (fix #3 coupling).
    monkeypatch.setattr(stg, "evaluate_throughput", lambda **kw: _fast_verdict(500.0))
    rc = _lw._run_throughput_gate(_StubCfg(3000), tmp_path, threshold_ms=None,
                                  accept_wall_clock_days=None)
    assert rc == 8


def test_gate_proceeds_when_fast_and_under_budget(monkeypatch, tmp_path):
    import tac.local_acceleration.scorer_throughput_gate as stg
    monkeypatch.setattr(stg, "evaluate_throughput", lambda **kw: _fast_verdict(396.0))  # anchor speed
    rc = _lw._run_throughput_gate(_StubCfg(3000), tmp_path, threshold_ms=None,
                                  accept_wall_clock_days=None)
    assert rc == 0


def test_gate_accept_override_proceeds_and_stamps(monkeypatch, tmp_path):
    import tac.local_acceleration.scorer_throughput_gate as stg
    monkeypatch.setattr(stg, "evaluate_throughput", lambda **kw: _fast_verdict(500.0))
    rc = _lw._run_throughput_gate(_StubCfg(3000), tmp_path, threshold_ms=None,
                                  accept_wall_clock_days=100.0)  # knowingly accept a longer run
    assert rc == 0
    stamp = tmp_path / "wall_clock_accept.txt"
    assert stamp.exists() and "100.0" in stamp.read_text()


# ── FIX 3: the budget-implied bench ceiling inverts the projection ──────────────────────
def test_implied_ceiling_inverts_projection():
    from tac.local_acceleration.scorer_throughput_gate import (
        implied_segnet_ms_ceiling,
        project_launch_wall_clock,
    )
    budget, epochs = 7.4, 3000
    ceiling = implied_segnet_ms_ceiling(budget, epochs)
    # a bench 1ms below the ceiling is within budget; 1ms above is over.
    assert project_launch_wall_clock(ceiling - 1.0, epochs, budget_days=budget).over_budget is False
    assert project_launch_wall_clock(ceiling + 1.0, epochs, budget_days=budget).over_budget is True


# ── FIX 2: the perf-env CLASS guard (derived required set, missing var named) ────────────
def test_required_perf_env_is_derived_from_the_prefix():
    from tac.witness_dsl.typed_config import PERF_ENV_PREFIX, REQUIRED_PERF_ENV
    # v7.3 delta 1: the D16 persistence-pool env joins the required set (auto-derived from the prefix).
    assert REQUIRED_PERF_ENV == {"TAC_MLX_CUSTOM_GROUPED_BACKWARD": "1",
                                 "TAC_MLX_CUSTOM_PERSISTENCE_POOL": "1"}
    for name, val in REQUIRED_PERF_ENV.items():
        assert f"{name}={val}" in PERF_ENV_PREFIX  # SoT: parsed from the prefix, no duplicate list


def test_missing_perf_env_names_the_missing_var():
    from tac.witness_dsl.typed_config import missing_perf_env_vars
    # both required vars named when neither is present (sorted for a stable REFUSE message).
    assert missing_perf_env_vars("nothing here") == [
        "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1", "TAC_MLX_CUSTOM_PERSISTENCE_POOL=1"]
    # BOTH present => nothing missing.
    both = "... TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 TAC_MLX_CUSTOM_PERSISTENCE_POOL=1 ..."
    assert missing_perf_env_vars(both) == []
    # only grouped-backward present => the persistence-pool var is still named missing.
    assert missing_perf_env_vars("... TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 ...") == [
        "TAC_MLX_CUSTOM_PERSISTENCE_POOL=1"]
    # a bare NAME (no value) or WRONG value does NOT satisfy the required set.
    assert missing_perf_env_vars("TAC_MLX_CUSTOM_GROUPED_BACKWARD TAC_MLX_CUSTOM_PERSISTENCE_POOL") == [
        "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1", "TAC_MLX_CUSTOM_PERSISTENCE_POOL=1"]
    assert missing_perf_env_vars(
        "TAC_MLX_CUSTOM_GROUPED_BACKWARD=0 TAC_MLX_CUSTOM_PERSISTENCE_POOL=1") == [
        "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1"]


# ── FIX 2: prefix-parity — BOTH to_command paths consume the ONE prefix (drift-impossible) ──
def test_prefix_parity_both_to_command_paths_share_the_one_constant():
    import tac.witness_autoconfig as wac
    from tac.witness_dsl.typed_config import PERF_ENV_PREFIX
    # the autoconfig module CONSUMES the typed-config constant (same object) => drift-IMPOSSIBLE.
    assert wac.PERF_ENV_PREFIX is PERF_ENV_PREFIX
    # v6-family (WitnessConfig) to_command carries it.
    v6 = wac.derive_crucible_v6_config(_GT, num_pairs=600, epochs=1000)
    assert v6.to_command("RUNDIR", perf_env=True).startswith("TAC_MLX_CUSTOM_GROUPED_BACKWARD=1")
    # typed v7 to_command carries the same prefix.
    v7 = wac.derive_crucible_v7_config(_GT, num_pairs=600, epochs=3000,
                                       out_dir="experiments/results/__v7_parity_test__")
    assert v7.to_command("RUNDIR", perf_env=True).startswith(PERF_ENV_PREFIX)


def test_emitted_launch_command_satisfies_the_perf_env_guard():
    # the guard's pure core against a REAL emitted command (v7) — no missing vars.
    import tac.witness_autoconfig as wac
    from tac.witness_dsl.typed_config import missing_perf_env_vars
    v7 = wac.derive_crucible_v7_config(_GT, num_pairs=600, epochs=3000,
                                       out_dir="experiments/results/__v7_guard_test__")
    assert missing_perf_env_vars(v7.to_command("RUNDIR", perf_env=True)) == []
