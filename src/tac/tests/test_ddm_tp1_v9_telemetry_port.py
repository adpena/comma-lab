"""ddm_tp1 (#804) — tests for the v9-line confound-cure TELEMETRY PORT on the TR1 trainer.

Covers three axes the burn-4 §3.1 prerequisite demands:
  * SCHEMA — the per-term ``loss_terms`` row (#304), term_domination (#321), term_inert
    (#321), lever_engage (Q7), and the #404 positive-control sentinel produce the canonical
    v9 shapes with C6 liveness stamps.
  * ALARM-FIRE — term_domination is edge-triggered after the sustained window; term_inert
    fires for an engaged-but-inert term; the positive control clears.
  * OFF-IDENTITY — the flag is threaded via args ONLY (never TR1Config), so config_hash /
    canonical_json / asdict are FLAG-INVARIANT; the argparse default is 'off'; and the DSL
    Lever declares a flag the trainer argparse actually holds (never-invent-flags).

Pure-helper + reusable-producer tests only (MLX-free at import); the full-run byte-identity
receipt is a separate bounded smoke recorded in the memo (a real MLX/scorer run).
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
WORKTREE = _HERE.parents[3]
for _p in (str(WORKTREE), str(WORKTREE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.train_tr1_partition_renderer_mlx import (  # noqa: E402
    TR1_BASE_LOSS_TERM_KEYS,
    TR1_LOSS_TERM_KEYS,
    TR1_SCORED_FLOOR,
    TR1_SCORED_TERM,
    TR1_TERMDOM_FRAC,
    TR1_TERMDOM_MIN_ROWS,
    TR1Config,
    build_argparser,
    tr1_active_loss_term_keys,
    tr1_active_scored_terms,
    tr1_loss_terms_row,
    tr1_term_domination_alarms,
)
from tac.witness_control.telemetry_producers import (  # noqa: E402
    ProducerResumeState,
    deterministic_strata,
    lever_engage_row,
    term_inert_rows,
)
from tac.witness_control.verdict_trend_alarm import canary_suite  # noqa: E402
from tac.witness_dsl.spec_tr1_renderer_20260728 import (  # noqa: E402
    TR1RendererProgramV1,
    lever_telemetry_v9_port,
    lever_window,
    trainer_declared_flags,
)


# ----------------------------------------------------------------- schema ----
def test_loss_terms_row_schema_and_self_check():
    terms = {"seg": 10.0, "rate": 2.0, "delta_sparsity": 0.5}
    row = tr1_loss_terms_row(terms, 12.5, ep=7, accum_batch=3, accepted_frac=1.0,
                             weights_stepped=True, stage="seg_trunk_tau", seg_form="tau_softplus")
    assert row["stage"] == "loss_terms"
    # stable complete key set
    assert set(row["terms"]) == set(TR1_BASE_LOSS_TERM_KEYS)
    assert row["sum_terms"] == 12.5
    assert abs(row["sum_minus_total"]) < 1e-6  # self-checking addend breakdown
    # C6 LIVENESS stamps present (satisfies Catalog #402 within the emitter window)
    assert row["accepted_frac"] == 1.0
    assert row["weights_stepped"] is True
    assert row["score_neutral"] is True


def test_loss_terms_row_missing_terms_are_zero_stable_schema():
    row = tr1_loss_terms_row({"seg": 3.0}, 3.0, ep=0, accum_batch=1, accepted_frac=0.5,
                             weights_stepped=False, stage="seg_trunk_ce", seg_form="ce")
    assert row["terms"] == {"seg": 3.0, "rate": 0.0, "delta_sparsity": 0.0}
    assert row["weights_stepped"] is False
    assert row["accepted_frac"] == 0.5


def test_loss_terms_row_itemizes_active_jd1_pose_and_birth_amplify():
    keys = tr1_active_loss_term_keys(
        jd1_pose_finish_active=True,
        birth_amplify_active=True,
    )
    assert keys == TR1_LOSS_TERM_KEYS
    terms = {
        "seg": 10.0,
        "pose": 1.25,
        "rate": 2.0,
        "delta_sparsity": 0.5,
        "birth_amplify": 0.75,
    }
    total = sum(terms.values())
    row = tr1_loss_terms_row(
        terms, total, ep=1409, accum_batch=75, accepted_frac=1.0,
        weights_stepped=True, stage="joint_pose_finish", seg_form="tau_softplus",
        loss_term_keys=keys,
    )
    assert row["terms"] == terms
    assert row["sum_terms"] == total
    assert abs(row["sum_minus_total"]) < 1e-8


def test_term_domination_treats_jd1_pose_as_scored_when_active():
    keys = tr1_active_loss_term_keys(jd1_pose_finish_active=True)
    scored_terms = tr1_active_scored_terms(jd1_pose_finish_active=True)
    streaks: dict[str, int] = {}
    pose_heavy = {"seg": 2.0, "pose": 7.0, "rate": 0.5, "delta_sparsity": 0.5}
    for _ in range(TR1_TERMDOM_MIN_ROWS + 1):
        rows = tr1_term_domination_alarms(
            pose_heavy, 10.0, streaks,
            loss_term_keys=keys,
            scored_terms=scored_terms,
        )
        assert rows == []


def test_liveness_token_literal_present_for_catalog_402():
    # the #402 static gate greps the emitter window for a liveness token; prove one exists
    row = tr1_loss_terms_row({"seg": 1.0}, 1.0, ep=0, accum_batch=0, accepted_frac=1.0,
                             weights_stepped=True, stage="s", seg_form="ce")
    assert "accepted_frac" in row and "weights_stepped" in row


# --------------------------------------------------------- term_domination ----
def test_term_domination_scored_seg_dominant_no_alarm():
    """b4s 2026-07-31 first-fire calibration regression (MAIN adjudication): the SCORED
    seg term dominating a seg-only burn is the DESIGN, never an alarm.  Uses the real
    window_01 ep665 profile (seg 0.349 / rate 0.147 / ds 0.0 => seg share 0.6783) that
    produced the spurious fire under the pre-fix any-term>ceiling predicate."""
    streaks: dict[str, int] = {}
    ep665 = {"seg": 0.349, "rate": 0.147, "delta_sparsity": 0.0}
    total = sum(ep665.values())
    for _ in range(TR1_TERMDOM_MIN_ROWS + 3):
        assert tr1_term_domination_alarms(ep665, total, streaks) == []


def test_term_domination_nonscored_ceiling_fires_edge_triggered():
    """MAIN spec case: rate at 0.45 of the loss => alarm (a NON-scored term over the
    v9 caps-law single-term ceiling), edge-triggered after MIN_ROWS sustained rows.
    NOTE (derivation property): with shares summing to 1, rate > 0.40 IMPLIES
    seg < 0.60 = the floor — the two clauses CO-FIRE by construction (the floor is
    the exact complement of the caps-law non-scored aggregate cap), so BOTH rows are
    expected; the ceiling row is the primary attribution."""
    streaks: dict[str, int] = {}
    dom = {"seg": 5.5, "rate": 4.5, "delta_sparsity": 0.0}  # rate share 0.45 > 0.40
    for _ in range(TR1_TERMDOM_MIN_ROWS - 1):
        assert tr1_term_domination_alarms(dom, 10.0, streaks) == []
    rows = tr1_term_domination_alarms(dom, 10.0, streaks)
    fired = {(r["term"], r["predicate"]) for r in rows}
    assert fired == {("rate", "nonscored_above_ceiling"), ("seg", "scored_below_floor")}
    for r in rows:
        assert r["kind"] == "term_domination"
        assert r["event"] == "confound_alarm"
        assert r["sustained_rows"] == TR1_TERMDOM_MIN_ROWS
    # still violating one more row => streaks advance but do NOT re-fire (edge, not level)
    assert tr1_term_domination_alarms(dom, 10.0, streaks) == []


def test_term_domination_scored_floor_fires_when_seg_is_passenger():
    """MAIN spec case: seg at 0.25 of the loss => alarm (the SCORED share below the
    derived caps-law floor = seg-as-passenger, the original v9 meaning)."""
    streaks: dict[str, int] = {}
    passenger = {"seg": 2.5, "rate": 3.9, "delta_sparsity": 3.6}  # seg 0.25 < 0.60 floor;
    # rate 0.39 / ds 0.36 both under the per-term ceiling => ONLY the floor clause fires
    for _ in range(TR1_TERMDOM_MIN_ROWS - 1):
        assert tr1_term_domination_alarms(passenger, 10.0, streaks) == []
    rows = tr1_term_domination_alarms(passenger, 10.0, streaks)
    assert len(rows) == 1
    assert rows[0]["term"] == TR1_SCORED_TERM
    assert rows[0]["predicate"] == "scored_below_floor"


def test_term_domination_resets_when_profile_recovers():
    streaks: dict[str, int] = {}
    dom = {"seg": 5.5, "rate": 4.5, "delta_sparsity": 0.0}
    for _ in range(TR1_TERMDOM_MIN_ROWS - 1):
        tr1_term_domination_alarms(dom, 10.0, streaks)
    # a healthy row (seg dominant, rate under ceiling) resets the rate streak before it fires
    healthy = {"seg": 7.0, "rate": 3.0, "delta_sparsity": 0.0}
    assert tr1_term_domination_alarms(healthy, 10.0, streaks) == []
    assert streaks["rate"] == 0
    # threshold provenance sanity: ceiling 0.40 (v9 caps law), floor = its complement
    assert TR1_TERMDOM_FRAC == 0.40
    assert TR1_SCORED_FLOOR == 1.0 - TR1_TERMDOM_FRAC


# ------------------------------------------------------------- term_inert ----
def test_term_inert_fires_for_engaged_inert_term():
    # rate is ENGAGED but contributes ~0 share for the sustained window => inert alarm
    state = ProducerResumeState()
    terms = {"seg": 10.0, "rate": 1e-9, "delta_sparsity": 0.0}
    engaged = {"seg": True, "rate": True, "delta_sparsity": False}
    fired: list[dict] = []
    for ep in range(3):
        fired += term_inert_rows(terms, engaged=engaged, epoch=ep, state=state)
    assert any(r["term"] == "rate" and r["alarm"] == "term_inert" for r in fired)
    # seg is engaged AND material => never inert
    assert not any(r["term"] == "seg" for r in fired)


# ----------------------------------------------------------- lever_engage ----
def test_lever_engage_row_canonical_schema_with_extra():
    row = lever_engage_row("token_delta_group_sparsity", status="fired", epoch=42,
                           via="ce_tau_knee_base_stable",
                           extra={"source_event": "event_delta_sparsity_engage"})
    assert row["stage"] == "lever_engage"
    assert row["lever"] == "token_delta_group_sparsity"
    assert row["status"] == "fired"
    assert row["epoch"] == 42
    assert row["source_event"] == "event_delta_sparsity_engage"


# ------------------------------------------------------- positive control ----
def test_positive_control_canary_clears():
    pc = canary_suite()
    assert pc.passed is True
    assert pc.verdict_clearance() is True
    assert pc.descent_positive_registered is True
    assert pc.negative_fired is False


# ------------------------------------------------------------ off-identity ----
def test_flag_absent_from_config_so_checkpoint_bytes_are_flag_invariant():
    # the port flag must NOT be a TR1Config field (else asdict/config_hash/checkpoint bytes
    # would change even when off — breaking the sealed-lineage byte-identity guarantee).
    field_names = set(TR1Config.__dataclass_fields__)
    assert not any("telemetry" in f for f in field_names)
    cfg = TR1Config(
        variant="plain", num_pairs=4, grid_downsample=16, code_width=2, renderer_width=8,
        token_quant_levels=16, seed=3, lotto_seed=118, lotto_mask_density_init=0.5,
        seg_form_start="ce", w_seg=100.0, lr=1e-3, batch_pairs=2, epochs=2, gate_every=1,
        ema_decay=0.95, ema_decay_provenance="test", token_temporal_mode="shared_base",
        token_ste="round", class_weight_lane=1.0, margin_target=1.0)
    assert "telemetry_v9_port" not in asdict(cfg)
    assert "telemetry" not in cfg.canonical_json()


def test_argparse_default_is_on_and_choices_off_on():
    req = ["--variant", "plain", "--out-dir", "/tmp/tp1_test_unused"]
    ns = build_argparser().parse_args(req)
    assert ns.telemetry_v9_port == "on"
    ns_on = build_argparser().parse_args([*req, "--telemetry-v9-port", "on"])
    assert ns_on.telemetry_v9_port == "on"
    ns_off = build_argparser().parse_args([*req, "--telemetry-v9-port", "off"])
    assert ns_off.telemetry_v9_port == "off"


# --------------------------------------------------------------- DSL lever ----
def test_dsl_lever_declares_a_flag_the_trainer_holds():
    lv = lever_telemetry_v9_port("on")
    assert lv.overrides == {"--telemetry-v9-port": "on"}
    # never-invent-flags: the flag the lever emits must exist in the trainer argparse
    assert "--telemetry-v9-port" in trainer_declared_flags()


def test_dsl_program_with_telemetry_lever_validates_and_compiles():
    prog = TR1RendererProgramV1(
        levers=(lever_window(4, 90.0, batch_pairs=2, lr=1e-3), lever_telemetry_v9_port("on")),
        num_pairs=4, out_dir="/tmp/tp1_test_unused")
    prog.validate()  # fail-closed never-invent-flags
    argv = prog.compile_trainer_argv()
    assert "--telemetry-v9-port" in argv
    assert argv[argv.index("--telemetry-v9-port") + 1] == "on"


def test_dsl_lever_state_validated():
    with pytest.raises(ValueError, match="off|on"):
        lever_telemetry_v9_port("bogus")


# --------------------------------------------------------- strata producer ----
def test_deterministic_strata_canonical_and_caller_clamps_k():
    # the trainer calls deterministic_strata(num_pairs, min(8, num_pairs)); canonical n600/k8:
    assert deterministic_strata(600, 8) == (37, 112, 187, 262, 337, 412, 487, 562)
    # k must be <= n_pairs (the trainer clamps via min(); an un-clamped k>n is a caller error)
    assert deterministic_strata(4, 4) == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="k must be in"):
        deterministic_strata(4, 8)
