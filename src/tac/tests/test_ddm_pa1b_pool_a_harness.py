"""ddm_pa1b (#793) — Pool-A race harness tests: levers + theorem + seal + analyzer + DSL.

Scorer-free; every test is deterministic + numpy-only.  The real-QA80-field test skips cleanly
when the SSD custody tier is not mounted (CI), but the allocation LAW + off-identity + theorem +
analyzer + DSL coverage are exercised on synthetic + real data unconditionally.

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED; score_claim=False.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.ax1_pool_a_levers_20260730 import (
    QA80_FIELD_CUSTODY,
    QA80FieldError,
    apply_per_cell_quant_np,
    delta_group_sparsity_penalty,
    load_qa80_cell_field,
    margin_coupled_level_map,
    xi_informed_delta_weight,
)
from tac.witness_dsl.ax1_pool_a_race_20260730 import (
    HullCurvatureAnalyzer,
    RaceReceipt,
    enumerate_band_edge_theorem,
    per_row_flip_mass_from_field,
    seal_matched_bytes_race,
    smevr_bytes_of_field,
)


# ---------------------------------------------------------------------------
# 1. Exhaust theorem #2 — the complete band-edge enumeration + count derivation.
# ---------------------------------------------------------------------------
def test_theorem_count_is_300_superset_of_memo_276():
    prm = np.ones(48) * 0.1
    prm[20:30] = 5.0
    th = enumerate_band_edge_theorem(prm, fine_gw=64, coarse_factor=2)
    # DERIVED count = C(25,2) = 300 (25 coarse-aligned boundaries); memo C(24,2)=276.
    assert th.total_configs == 300
    assert th.memo_claimed_count == 276
    assert th.total_configs - th.memo_claimed_count == 24  # the single-D16-row bands
    assert len(th.configs) == 300
    assert "off-by-one" in th.count_derivation


def test_theorem_optimal_band_is_min_cells_covering_target():
    prm = np.ones(48) * 0.1
    prm[20:30] = 5.0                                  # flip mass concentrated rows 20-30 (D8)
    th = enumerate_band_edge_theorem(prm, fine_gw=64, coarse_factor=2,
                                     coverage_targets=(0.5, 0.9))
    opt50 = th.optimal_at_coverage["cov>=0.5"]
    assert opt50.flip_mass_covered >= 0.5
    # provably optimal: no OTHER config with >=0.5 coverage has fewer independent cells.
    feasible = [c for c in th.configs if c.flip_mass_covered >= 0.5 - 1e-9]
    assert opt50.independent_cells == min(c.independent_cells for c in feasible)


def test_theorem_pareto_frontier_monotone():
    prm = np.abs(np.sin(np.linspace(0, 3, 48))) + 0.05
    th = enumerate_band_edge_theorem(prm, fine_gw=64, coarse_factor=2)
    cells = [c.independent_cells for c in th.pareto_frontier]
    cov = [c.flip_mass_covered for c in th.pareto_frontier]
    assert cells == sorted(cells)          # non-decreasing cells
    assert cov == sorted(cov)              # non-decreasing coverage (non-dominated)


# ---------------------------------------------------------------------------
# 2. Allocation LAW (margin-coupled quant) — off-identity + field-derived monotonicity.
# ---------------------------------------------------------------------------
def test_allocation_off_identity_uniform():
    fm = np.random.default_rng(0).random((6, 8))
    off = margin_coupled_level_map(fm, base_levels=16, min_levels=16)
    assert (off == 16).all()               # min==base => uniform full levels (OFF)


def test_allocation_monotone_and_endpoints():
    fm = np.random.default_rng(1).random((6, 8))
    on = margin_coupled_level_map(fm, base_levels=16, min_levels=4)
    assert on.flat[fm.argmax()] == 16      # highest flip-mass cell => finest
    assert on.flat[fm.argmin()] == 4       # lowest flip-mass cell => coarsest
    assert on.min() >= 4 and on.max() <= 16


def test_allocation_uniform_field_gives_uniform_levels():
    fm = np.full((4, 5), 0.3)              # a tied (uniform) field
    on = margin_coupled_level_map(fm, base_levels=16, min_levels=4)
    assert (on == on.flat[0]).all()        # ties collapse => uniform allocation


def test_per_cell_quant_off_identity_vs_scalar():
    tok = (np.random.default_rng(2).random((6, 8, 4)) * 2 - 1).astype(np.float32)
    off = margin_coupled_level_map(np.random.default_rng(3).random((6, 8)),
                                   base_levels=16, min_levels=16)
    q_cell = apply_per_cell_quant_np(tok, off)
    L = 15.0
    q_scalar = np.round((np.clip(tok, -1, 1) + 1) * 0.5 * L) / L * 2 - 1
    assert np.allclose(q_cell, q_scalar)   # uniform level map == scalar-L control (byte-identical)


# ---------------------------------------------------------------------------
# 3. Delta group-sparsity penalty + xi-informed weight field.
# ---------------------------------------------------------------------------
def test_delta_penalty_zero_for_zero_deltas():
    d = np.zeros((5, 6, 8, 4))
    assert delta_group_sparsity_penalty(d) < 1e-3     # ~eps floor only


def test_delta_penalty_positive_and_grouped():
    d = np.zeros((5, 6, 8, 4))
    d[2, 3, 4] = 1.0
    assert delta_group_sparsity_penalty(d) > 0.0


def test_xi_weight_relaxes_dynamic_tightens_static():
    df = np.array([[0.0, 1.0], [0.5, 0.0]])           # static, dynamic, half, static
    w = xi_informed_delta_weight(df, floor=0.1)
    assert w[0, 0] == pytest.approx(1.0)              # static => full shrinkage
    assert w[0, 1] == pytest.approx(0.1)              # dynamic => floor (relaxed)
    assert w[1, 0] == pytest.approx(0.55)             # half => between


# ---------------------------------------------------------------------------
# 4. QA80 field loader — real custody consumption (skips if SSD absent) + fail-closed.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not Path(QA80_FIELD_CUSTODY).exists(),
                    reason="QA80 field custody (SSD tier) not mounted")
def test_qa80_field_consumption_real_bounded():
    agg = load_qa80_cell_field(24, 32, downsample=16, max_pairs=4)
    assert agg.flip_mass.shape == (24, 32)
    assert agg.dynamic_frac.shape == (24, 32)
    assert 1.5 < agg.q50_scale < 2.2                 # zb1 measured q50 ~1.8181
    assert 0.0 <= agg.dynamic_frac.mean() <= 1.0
    prm = per_row_flip_mass_from_field(agg.flip_mass)
    assert prm.shape == (24,)


@pytest.mark.skipif(not Path(QA80_FIELD_CUSTODY).exists(),
                    reason="QA80 field custody (SSD tier) not mounted")
def test_qa80_geometry_fail_closed():
    with pytest.raises(QA80FieldError):
        load_qa80_cell_field(48, 64, downsample=16, max_pairs=1)   # grid != field//ds


def test_qa80_missing_custody_fail_closed():
    with pytest.raises(QA80FieldError):
        load_qa80_cell_field(24, 32, downsample=16, field_custody="/nonexistent/path")


# ---------------------------------------------------------------------------
# 5. Matched-SMEVR-bytes seal (byte-matcher + argv-diff).
# ---------------------------------------------------------------------------
def _codes(rng, levels=16, shape=(8, 24, 32, 4)):
    return rng.integers(0, levels, size=shape, dtype=np.uint8)


def test_seal_matches_identical_fields_within_tol():
    rng = np.random.default_rng(7)
    ctrl = _codes(rng)
    arms = {"a": (["--x", "1"], ctrl.copy()), "b": (["--x", "2"], ctrl.copy())}
    v = seal_matched_bytes_race(["--x", "0"], ctrl, arms, levels=16)
    assert v.matched
    assert all(a.within_tol for a in v.arms)


def test_seal_refuses_byte_mismatch():
    rng = np.random.default_rng(8)
    ctrl = np.zeros((8, 24, 32, 4), dtype=np.uint8)       # ~free (all zeros)
    noisy = _codes(rng)                                   # high entropy => many more bytes
    arms = {"noisy": (["--y", "1"], noisy)}
    v = seal_matched_bytes_race(["--y", "0"], ctrl, arms, levels=16, tol=0.01)
    assert not v.matched
    assert v.refusal_reason and "outside" in v.refusal_reason


def test_seal_records_argv_diff():
    rng = np.random.default_rng(9)
    ctrl = _codes(rng)
    arms = {"arm": (["--grid-downsample", "8", "--new-flag", "on"], ctrl.copy())}
    v = seal_matched_bytes_race(["--grid-downsample", "16"], ctrl, arms, levels=16)
    diff = v.arms[0].argv_diff_vs_control
    assert diff["added"] == {"--new-flag": "on"}
    assert diff["changed"] == {"--grid-downsample": {"control": "16", "arm": "8"}}


def test_smevr_bytes_deterministic():
    rng = np.random.default_rng(10)
    c = _codes(rng)
    assert smevr_bytes_of_field(c, 16) == smevr_bytes_of_field(c, 16)


# ---------------------------------------------------------------------------
# 6. Hull-curvature analyzer — the gc10 verdict.
# ---------------------------------------------------------------------------
def test_analyzer_detects_hull_moved():
    recs = [
        RaceReceipt("control", "w", 0.00528, 250898, 0, 0, True),        # c ~ 0.088 (on iso-c)
        RaceReceipt("rowband", "w", 0.0040, 250000, -0.0013, -898, True),  # lower c => inside
        RaceReceipt("margin_quant", "w", 0.00528, 250898, 0, 0, True),   # same => on-contour
    ]
    v = HullCurvatureAnalyzer().analyze(recs)
    assert v.hull_moved
    assert "rowband" in v.inside_contour
    assert "margin_quant" in v.on_contour
    assert v.best_lever == "rowband"


def test_analyzer_line_stands_when_all_on_contour():
    recs = [
        RaceReceipt("control", "w", 0.00528, 250898, 0, 0, True),
        RaceReceipt("rowband", "w", 0.00528, 250898, 0, 0, True),
        RaceReceipt("delta_sparsity", "w", 0.00528, 250898, 0, 0, True),
    ]
    v = HullCurvatureAnalyzer().analyze(recs)
    assert not v.hull_moved
    assert set(v.on_contour) == {"rowband", "delta_sparsity"}
    assert "EXTENDS to the class" in v.curvature_note


def test_analyzer_ignores_unmatched_arms():
    recs = [
        RaceReceipt("control", "w", 0.00528, 250898, 0, 0, True),
        RaceReceipt("rowband", "w", 0.0040, 250000, -0.0013, -898, False),  # NOT matched
    ]
    v = HullCurvatureAnalyzer().analyze(recs)
    assert v.n_points == 0
    assert not v.hull_moved
    assert "UNMEASURED" in v.curvature_note


# ---------------------------------------------------------------------------
# 6b. ADDITIVE-S analyzer — the PRIMARY verdict (nv1 reframe: c is a multiplicative artifact).
# ---------------------------------------------------------------------------
def test_s_additive_terms_match_nv1_b_anchor():
    """RaceReceipt.s_additive reproduces nv1's MEASURED B-control accounting base (d_seg 0.005114,
    259,407 B => seg 0.5114, rate 0.17273, S 0.6842, c 0.08834)."""
    r = RaceReceipt("control", "w", 0.005114, 259407, 0, 0, True)
    assert r.seg_term() == pytest.approx(0.5114, abs=1e-4)
    assert r.rate_term() == pytest.approx(0.17273, abs=1e-4)
    assert r.s_additive() == pytest.approx(0.6842, abs=1e-3)
    assert r.c() == pytest.approx(0.08834, abs=1e-4)


def test_s_additive_catches_the_nv1_c_artifact():
    """THE decisive test: nv1's thr2 snap has LOWER c (0.0676 < 0.0883) but HIGHER additive S
    (0.925 > 0.684).  The c-analyzer calls it 'inside the contour' (a false hull move); the
    S-analyzer MUST classify it worse_s and report hull_moved_s=False."""
    control = RaceReceipt("control", "w", 0.005114, 259407, 0, 0, True)
    snap = RaceReceipt("delta_sparsity", "w", 0.008448, 120111, 0.003334, -139296, True)
    # c-telemetry (legacy) is fooled: snap.c < iso_c => "inside contour".
    assert snap.c() < control.c()
    cverd = HullCurvatureAnalyzer().analyze([control, snap])
    assert "delta_sparsity" in cverd.inside_contour            # c says (falsely) hull moved
    # S-additive (primary) is not fooled: S rose => worse, hull did NOT move.
    sverd = HullCurvatureAnalyzer().analyze_s_additive([control, snap])
    assert not sverd.hull_moved_s
    assert "delta_sparsity" in sverd.worse_s
    assert sverd.best_delta_s_vs_control > 0                   # best arm is still ABOVE control
    assert "DID NOT move" in sverd.verdict_note


def test_s_additive_axis_breakdown_exchange_rate_matches_nv1():
    """The reframe discriminator (FEED-reanchor): the per-arm axis breakdown exposes the EXCHANGE
    RATE, not just the net.  nv1 thr2 reported 'unfavorable× 3.59' = Δseg/(−Δrate); our
    s_favorability = (−Δrate)/Δseg must be its reciprocal ≈ 0.278 (<1 => unfavorable)."""
    control = RaceReceipt("control", "w", 0.005114, 259407, 0, 0, True)
    snap = RaceReceipt("delta_sparsity", "w", 0.008448, 120111, 0.003334, -139296, True)
    v = HullCurvatureAnalyzer().analyze_s_additive([control, snap])
    bd = v.arm_axis_breakdown["delta_sparsity"]
    assert bd["delta_seg_term"] == pytest.approx(0.3334, abs=1e-3)      # seg cost (S units)
    assert bd["delta_rate_term"] == pytest.approx(-0.09275, abs=1e-3)   # rate saved (S units)
    assert bd["delta_bytes"] == pytest.approx(-139296.0)
    assert bd["exchange_bytes_per_dseg"] > 0                            # paid d_seg for bytes
    assert bd["s_favorability"] == pytest.approx(1.0 / 3.59, abs=0.02)  # nv1 unfavorable 3.59×
    assert bd["s_favorability"] < 1.0                                   # unfavorable exchange


def test_s_additive_axis_breakdown_pareto_has_no_exchange():
    """A Pareto arm (lower d_seg AND bytes) has no exchange to price: exchange_bytes_per_dseg and
    s_favorability are None (d_seg fell => nothing traded)."""
    control = RaceReceipt("control", "w", 0.005114, 259407, 0, 0, True)
    win = RaceReceipt("delta_sparsity", "w", 0.005000, 250000, -1.14e-4, -9407, True)
    bd = HullCurvatureAnalyzer().analyze_s_additive([control, win]).arm_axis_breakdown["delta_sparsity"]
    assert bd["exchange_bytes_per_dseg"] is None
    assert bd["s_favorability"] is None
    assert bd["delta_rate_term"] < 0 and bd["delta_seg_term"] < 0        # both terms fell


def test_s_additive_detects_true_hull_move_pareto():
    """A genuine mover: lower d_seg AND lower bytes => Pareto-dominant + better_s + hull_moved_s."""
    control = RaceReceipt("control", "w", 0.005114, 259407, 0, 0, True)
    win = RaceReceipt("delta_sparsity", "w", 0.005000, 250000, -1.14e-4, -9407, True)
    v = HullCurvatureAnalyzer().analyze_s_additive([control, win])
    assert v.hull_moved_s
    assert "delta_sparsity" in v.better_s
    assert "delta_sparsity" in v.pareto_dominates_control
    assert v.best_lever == "delta_sparsity" and v.best_delta_s_vs_control < 0
    assert "hull MOVED" in v.verdict_note and "Pareto" in v.verdict_note


def test_s_additive_on_line_within_noise():
    """An arm within the d_seg noise band of control (bytes matched) sits on the S line."""
    control = RaceReceipt("control", "w", 0.005114, 259407, 0, 0, True)
    tie = RaceReceipt("margin_quant", "w", 0.005114 + 1e-5, 259407, 1e-5, 0, True)
    v = HullCurvatureAnalyzer().analyze_s_additive([control, tie])
    assert not v.hull_moved_s
    assert "margin_quant" in v.on_s_line


def test_s_additive_control_lever_fallback_to_rowband():
    """When no bare 'control' exists the analyzer falls back to a 'control*' lever
    (control_rowband is the local Pool-A control)."""
    control = RaceReceipt("control_rowband", "w", 0.005114, 250000, 0, 0, True)
    arm = RaceReceipt("delta_sparsity", "w", 0.005114, 250000, 0, 0, True)
    v = HullCurvatureAnalyzer().analyze_s_additive([control, arm])
    assert v.control_lever == "control_rowband"
    assert v.n_points == 1


def test_s_additive_json_entry_attaches_c_telemetry():
    import json as _json

    from tac.witness_dsl.ax1_pool_a_race_20260730 import analyze_race_s_additive_json
    recs = [
        {"lever": "control", "d_seg": 0.005114, "counted_bytes": 259407},
        {"lever": "delta_sparsity", "d_seg": 0.008448, "counted_bytes": 120111},
    ]
    out = analyze_race_s_additive_json(_json.dumps(recs))
    assert out["verdict_currency"].startswith("additive_S")
    assert out["hull_moved_s"] is False
    assert "c_telemetry" in out                                # c read attached, not the verdict


# ---------------------------------------------------------------------------
# 7. DSL — pool_a_race_programs compiles + validates; new levers emit declared flags;
#    fold-and-delete complete (stubs superseded).
# ---------------------------------------------------------------------------
_POOL_A_FLAGS = frozenset({
    "--token-quant-margin-coupling", "--token-quant-coupling-field",
    "--token-delta-group-sparsity", "--delta-sparsity-weight", "--delta-sparsity-engage",
    "--delta-sparsity-weight-field"})


def test_pool_a_programs_compile_and_validate():
    from tac.witness_dsl.spec_tr1_burn2_20260731 import pool_a_race_programs
    r = pool_a_race_programs("lotto", "/tmp/o", "/tmp/m.npy", field_custody=QA80_FIELD_CUSTODY)
    arms = {k: v for k, v in r.items() if k not in ("grammar", "non_additive_pools_law")}
    assert set(arms) == {"control_rowband", "margin_quant", "delta_sparsity",
                         "joint_quant_sparsity"}
    hashes = set()
    for prog in arms.values():
        prog.compile_trainer_argv()                # validate() fail-closes on never-invent-flags
        hashes.add(prog.sealed_ticket()["ticket_hash"])
    assert len(hashes) == 4                         # distinct sealed tickets


def test_pool_a_flags_declared_by_trainer_and_emitted():
    """tr1 DSL coverage (the "surface in completeness()" obligation for the tr1 surface): every
    Pool-A flag the DSL emits must be DECLARED by the trainer argparse (never-invent) AND the new
    lever factories must actually EMIT them."""
    from tac.witness_dsl.spec_tr1_burn2_20260731 import pool_a_race_programs
    from tac.witness_dsl.spec_tr1_renderer_20260728 import trainer_declared_flags
    declared = trainer_declared_flags()
    assert declared >= _POOL_A_FLAGS               # trainer declares all 6 (never-invent)
    r = pool_a_race_programs("plain", "/tmp/o", "/tmp/m.npy", field_custody=QA80_FIELD_CUSTODY)
    emitted = set(r["joint_quant_sparsity"].merged_overrides())
    assert emitted >= _POOL_A_FLAGS                # the joint arm holds all 6 (DSL coverage)


def test_fold_and_delete_stubs_superseded():
    from tac.witness_dsl.ax1_derived_levers_20260730 import AX1_DERIVED_STUB_LEVERS
    names = {f.__name__ for f in AX1_DERIVED_STUB_LEVERS}
    assert names == {"Ax1Frame0CarriedWarp"}       # Pool-A + joint stubs folded out
    from tac.witness_dsl import ax1_derived_levers_20260730 as m
    for gone in ("Ax1MarginCoupledTokenQuant", "Ax1DeltaGroupSparsity", "Ax1PoolAJointRace"):
        assert not hasattr(m, gone)


def test_new_lever_factories_have_provenance():
    from tac.witness_dsl.spec_tr1_renderer_20260728 import (
        lever_delta_group_sparsity,
        lever_token_quant_margin_coupling,
    )
    q = lever_token_quant_margin_coupling(QA80_FIELD_CUSTODY)
    assert "MEASURED_ANCHOR" in q.constant_manifest["--token-quant-coupling-field"]["rung"]
    s = lever_delta_group_sparsity(1e-3, weight_field="xi_informed")
    assert s.overrides["--delta-sparsity-weight-field"] == "xi_informed"


# ---------------------------------------------------------------------------
# 8. Trainer off-path — byte-identical control + fail-closed guards.
# ---------------------------------------------------------------------------
def _base_cfg(**kw):
    import experiments.train_tr1_partition_renderer_mlx as T
    cfg = T.TR1Config(
        variant="plain", num_pairs=600, grid_downsample=16, code_width=4, renderer_width=24,
        token_quant_levels=16, seed=0, lotto_seed=118, lotto_mask_density_init=0.5,
        seg_form_start="ce", w_seg=100.0, lr=2e-3, batch_pairs=8, epochs=60, gate_every=5,
        ema_decay=0.997, ema_decay_provenance="x", token_temporal_mode="shared_base",
        token_ste="round", class_weight_lane=1.0, margin_target=1.0)
    return replace(cfg, **kw) if kw else cfg


def test_trainer_off_path_byte_identical():
    import experiments.train_tr1_partition_renderer_mlx as T
    assert T._build_pool_a_banks(_base_cfg()) == (None, None)


def test_trainer_margin_coupling_requires_field():
    import experiments.train_tr1_partition_renderer_mlx as T
    with pytest.raises(ValueError, match="require --token-quant-coupling-field"):
        T._build_pool_a_banks(_base_cfg(token_quant_margin_coupling="on"))


def test_trainer_delta_sparsity_requires_shared_base():
    import experiments.train_tr1_partition_renderer_mlx as T
    cfg = _base_cfg(token_delta_group_sparsity="on", delta_sparsity_weight_field="xi_informed",
                    token_temporal_mode="independent", token_quant_coupling_field="/x")
    with pytest.raises(ValueError, match="shared_base"):
        T._build_pool_a_banks(cfg)
