# SPDX-License-Identifier: MIT
"""Tests for the PAIR-LOCAL DIAGONAL mode of the MC finisher (#400).

The rigor is on the DIAGONAL-EXPLOIT invariants that the borrowed-substrate sibling
(#399 ``tac.click_polish``) proved on HNeRV latents, re-established here on the witness
finisher seam: the fail-closed LOCALITY GUARD (cross-talk → refuse), diagonal ≡ sequential
per-pair scoring, monotone ratchet, the d_pose ROLLBACK FLOOR pinned to the banked R1 value
(value-provenance, not a bare literal), real re-encoded byte accounting, and resumability.

All oracles are TINY synthetic mocks that are PAIR-LOCAL by construction (per_pair[p]
depends only on table row p) so the exploit's preconditions hold and acceptance is
reachable; ONE nonlocal mock exercises the guard's fail-closed path. The 4c′ byte-close
wiring is verified at the SEAM (surfaces load, factory binds) — the real n600 d_pose
measurement fires post-launch at the terminal band (deferral D27b), never here.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tac.contest_score import pose_term, rate_term, seg_term
from tac.through_r.mc_finisher import (
    BANKED_R1_DXI_DPOSE_FLOOR,
    DiagonalObjective,
    DiagonalProblem,
    LocalityGuardError,
    MCFinisherError,
    PairLocalDiagonalFinisher,
    load_banked_r1_dxi_dpose_floor,
    load_byte_close_pose_surfaces,
    make_byte_close_xi_pose_measure,
    make_through_r_code_measure,
)

# ======================================================================================
# controllable PAIR-LOCAL mock: per_pair[p] = |row_p - target_p| / DEN (pure fn of row p).
# A click toward target lowers that pair's distortion by exactly 1/DEN — flip-shaped,
# reachable, and diagonal-batchable by construction.
# ======================================================================================
_DEN = 1000.0


def _target(shape, seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 8, size=shape).astype(np.int64)


def _local_measure(target, *, axis="d_seg", byte_cost_fn=None):
    tgt = np.asarray(target, dtype=np.int64)

    def render_measure_fn(table, *, confirm):
        t = np.asarray(table).astype(np.int64)
        per_pair = np.abs(t - tgt).sum(axis=1).astype(np.float64) / _DEN
        agg = float(per_pair.mean())
        bts = int(byte_cost_fn(table)) if byte_cost_fn is not None else 0
        return DiagonalObjective(
            per_pair=per_pair, agg=agg, archive_bytes=bts, confirm=confirm,
            axis=axis, n_pairs=t.shape[0],
        )

    return render_measure_fn


def _local_probe(table, pairs):
    t = np.asarray(table)
    return [t[p].copy() for p in pairs]  # frame p = row p ⇒ pair-local, no cross-talk


def _nonlocal_probe(table, pairs):
    t = np.asarray(table).astype(np.int64)
    n = t.shape[0]
    # frame p depends on the PREVIOUS row too ⇒ clicking pair a changes pair (a+1)'s frame.
    return [t[p] + t[(p - 1) % n] for p in pairs]


def _local_problem(target, *, axis="d_seg", lo=0, hi=7, byte_cost_fn=None, probe=_local_probe):
    return DiagonalProblem(
        render_measure_fn=_local_measure(target, axis=axis, byte_cost_fn=byte_cost_fn),
        n_pairs=int(np.asarray(target).shape[0]), lo=lo, hi=hi, axis=axis,
        probe_frames_fn=probe, byte_cost_fn=byte_cost_fn,
    )


# --------------------------------------------------------------------------------------
# 1. LOCALITY GUARD — negative test: a deliberately-nonlocal problem is REFUSED (fail-closed).
# --------------------------------------------------------------------------------------
def test_locality_guard_refuses_nonlocal_problem():
    tgt = _target((4, 3))
    tbl = np.zeros((4, 3), dtype=np.int64)
    prob = _local_problem(tgt, probe=_nonlocal_probe)
    fin = PairLocalDiagonalFinisher(tbl, prob)
    with pytest.raises(LocalityGuardError, match="CROSS-TALK"):
        fin.require_locality()


def test_locality_guard_passes_on_local_problem():
    tgt = _target((4, 3))
    tbl = np.zeros((4, 3), dtype=np.int64)
    fin = PairLocalDiagonalFinisher(tbl, _local_problem(tgt))
    rep = fin.require_locality()
    assert rep["locality_holds"] is True
    assert rep["pair_b_unchanged_by_pair_a_click"] and rep["pair_a_changed_by_its_click"]
    assert fin._locality_certified is True


def test_locality_guard_vacuous_probe_refuses():
    # a probe whose frames NEVER change (constant) ⇒ locality cannot be certified ⇒ refuse.
    tgt = _target((3, 2))
    tbl = np.zeros((3, 2), dtype=np.int64)

    def _const_probe(table, pairs):
        return [np.zeros(2, dtype=np.int64) for _ in pairs]

    fin = PairLocalDiagonalFinisher(tbl, _local_problem(tgt, probe=_const_probe))
    with pytest.raises(LocalityGuardError, match="VACUOUS"):
        fin.require_locality()


def test_probe_frames_missing_fn_raises():
    tgt = _target((3, 2))
    prob = DiagonalProblem(
        render_measure_fn=_local_measure(tgt), n_pairs=3, lo=0, hi=7, axis="d_seg",
        probe_frames_fn=None,
    )
    fin = PairLocalDiagonalFinisher(np.zeros((3, 2), np.int64), prob)
    with pytest.raises(LocalityGuardError, match="no probe_frames_fn"):
        fin.verify_locality()


# --------------------------------------------------------------------------------------
# 2. DIAGONAL ≡ SEQUENTIAL on n4 — one diagonal click scores 4 pairs == 4 single-pair renders.
# --------------------------------------------------------------------------------------
def test_diagonal_equals_sequential_n4():
    tgt = _target((4, 5))
    tbl = _target((4, 5), seed=9)
    prob = _local_problem(tgt)
    col, delta = 2, 1
    # diagonal: same click on EVERY pair, one render.
    diag = tbl.copy()
    diag[:, col] = np.clip(diag[:, col] + delta, 0, 7)
    diag_pp = prob.measure(diag, confirm=True).per_pair
    # sequential: click ONLY pair p, render it, read its per-pair distortion.
    seq = np.empty(4)
    for p in range(4):
        one = tbl.copy()
        one[p, col] = np.clip(one[p, col] + delta, 0, 7)
        seq[p] = prob.measure(one, confirm=True).per_pair[p]
    assert np.array_equal(diag_pp, seq)  # exact (pair-local by construction)


# --------------------------------------------------------------------------------------
# 3. MONOTONE RATCHET — confirmed S never increases; final ≤ start (d_seg axis).
# --------------------------------------------------------------------------------------
def test_ratchet_monotone_never_regresses_dseg():
    tgt = _target((6, 4))
    tbl = _target((6, 4), seed=17)
    fin = PairLocalDiagonalFinisher(tbl, _local_problem(tgt))
    start = fin.current_objective().axis_s_component()
    res = fin.run(max_rounds=30)
    s_hist = [start] + [o.s_after for o in res.outcomes if o.accepted]
    assert all(s_hist[i + 1] <= s_hist[i] + 1e-12 for i in range(len(s_hist) - 1))
    assert res.best_s_component <= res.start_s_component + 1e-12
    assert res.delta_s_total <= 0.0


def test_run_reaches_target_and_touches_pairs():
    tgt = _target((5, 4))
    tbl = _target((5, 4), seed=21)
    fin = PairLocalDiagonalFinisher(tbl, _local_problem(tgt))
    res = fin.run(max_rounds=60)
    # reachable landscape (±1/±2 clicks, grid [0,7]) ⇒ converges to target (agg 0).
    assert res.best_agg == pytest.approx(0.0, abs=1e-12)
    assert np.array_equal(res.best_table, tgt)
    assert res.pairs_touched >= 1
    assert res.locality_certified is True


# --------------------------------------------------------------------------------------
# 4. ROLLBACK FLOOR (d_pose) — never ACCEPT an aggregate d_pose worse than the pinned floor.
# --------------------------------------------------------------------------------------
def _floor_problem():
    # d_pose axis; per_pair rises with col0, bytes FALL with col0 → a +1 click lowers S via
    # rate but raises agg above the floor. Columns/deltas constrained to expose exactly that.
    base = 0.001610
    n = 3

    def render_measure_fn(table, *, confirm):
        t = np.asarray(table).astype(np.int64)
        per_pair = base + 0.0002 * t[:, 0].astype(np.float64)  # col0 up ⇒ d_pose up
        return DiagonalObjective(
            per_pair=per_pair, agg=float(per_pair.mean()),
            archive_bytes=100_000 - 5000 * int(t[:, 0].sum()),  # col0 up ⇒ bytes down
            confirm=confirm, axis="d_pose", n_pairs=n,
        )

    prob = DiagonalProblem(
        render_measure_fn=render_measure_fn, n_pairs=n, lo=0, hi=7, axis="d_pose",
        probe_frames_fn=lambda t, prs: [np.asarray(t)[p].copy() for p in prs],
    )
    tbl = np.zeros((n, 1), dtype=np.int64)  # start col0=0 ⇒ agg == floor exactly
    return tbl, prob


# The rollback floor is an ACCEPT-LAYER rail (the sweep ranks by per-pair DISTORTION, so it
# never *proposes* a pure byte-savings click; the floor guards the JOINT accept from ever
# shipping a d_pose worse than the banked guarantee — e.g. a byte-driven S win that sacrifices
# pose). So it is exercised at :meth:`_accept_joint` directly, its true locus (requirement d).
def test_rollback_floor_rejects_byte_win_that_worsens_dpose_below_guarantee():
    tbl, prob = _floor_problem()
    fin = PairLocalDiagonalFinisher(
        tbl, prob, columns=[0], deltas=(1,), rollback_floor=BANKED_R1_DXI_DPOSE_FLOOR,
    )
    base = fin.current_objective()  # at the floor (col0 == 0)
    clicks = [(p, 0, 1) for p in range(3)]  # +1 on col0: bytes DOWN (S win) but d_pose UP
    accepted, new_obj, _depth, _applied = fin._accept_joint(base, clicks)
    assert not accepted  # every subset lands agg > floor ⇒ floor guard refuses (rollback)
    assert np.array_equal(fin.table, tbl)  # table rolled back / unmoved


def test_without_floor_the_same_byte_win_is_accepted():
    tbl, prob = _floor_problem()
    fin = PairLocalDiagonalFinisher(tbl, prob, columns=[0], deltas=(1,), rollback_floor=None)
    base = fin.current_objective()
    clicks = [(p, 0, 1) for p in range(3)]
    accepted, new_obj, _depth, _applied = fin._accept_joint(base, clicks)
    assert accepted  # no floor ⇒ the byte-lowering (S-improving) move is accepted
    assert new_obj.axis_s_component() < base.axis_s_component()
    assert new_obj.agg > BANKED_R1_DXI_DPOSE_FLOOR  # d_pose sacrificed for bytes (unguarded)


# --------------------------------------------------------------------------------------
# 5. RATE ACCOUNTING == stat() of the REAL re-encoded bytes (never estimated).
# --------------------------------------------------------------------------------------
def test_rate_accounting_matches_real_encoded_file_bytes(tmp_path):
    tgt = _target((4, 3))
    tbl = _target((4, 3), seed=5)
    enc_path = tmp_path / "section.bin"

    def byte_cost_fn(table):
        # a REAL re-encode: write the mutated section to disk, return its stat() size.
        payload = np.ascontiguousarray(np.asarray(table).astype(np.int8)).tobytes()
        # a trivial "coder": drop trailing zero rows so bytes actually vary with the table.
        enc_path.write_bytes(payload)
        return os.path.getsize(enc_path)

    prob = _local_problem(tgt, byte_cost_fn=byte_cost_fn)
    obj = prob.measure(tbl, confirm=True)
    assert obj.archive_bytes == os.path.getsize(enc_path)
    # the s_component rate term uses the REAL byte count.
    assert obj.axis_s_component() == pytest.approx(
        seg_term(obj.agg) + rate_term(os.path.getsize(enc_path))
    )


# --------------------------------------------------------------------------------------
# 6. ξ FLOOR value-provenance — pinned to the byte-close artifact, not a bare literal.
# --------------------------------------------------------------------------------------
def test_xi_floor_matches_byte_close_artifact():
    # the constant equals the value re-read from the r1_dxi byte-close memo (ladder cross-check).
    assert load_banked_r1_dxi_dpose_floor() == pytest.approx(BANKED_R1_DXI_DPOSE_FLOOR, abs=1e-9)
    # sqrt(10·floor) == the banked 0.127 pose contribution (the fullstack anchor).
    assert pose_term(BANKED_R1_DXI_DPOSE_FLOOR) == pytest.approx(0.127, abs=1e-3)


def test_xi_floor_drift_is_caught(tmp_path):
    # a memo whose value disagrees with the cached constant RAISES (drift guard).
    bad = tmp_path / "bad_memo.md"
    bad.write_text("realized **d_pose = 0.009999** over all 600 inflated pairs\n", encoding="utf-8")
    with pytest.raises(MCFinisherError, match="DRIFT"):
        load_banked_r1_dxi_dpose_floor(bad)


def test_xi_floor_missing_memo_returns_cached_constant(tmp_path):
    missing = tmp_path / "nope.md"
    assert load_banked_r1_dxi_dpose_floor(missing) == BANKED_R1_DXI_DPOSE_FLOOR


# --------------------------------------------------------------------------------------
# 7. RESUMABILITY — replaying the accepted-moves JSONL reconstructs the exact table.
# --------------------------------------------------------------------------------------
def test_resume_from_ledger_round_trip(tmp_path):
    tgt = _target((5, 4))
    tbl = _target((5, 4), seed=33)
    log = tmp_path / "accepted.jsonl"
    fin = PairLocalDiagonalFinisher(tbl, _local_problem(tgt))
    res = fin.run(max_rounds=8, log_path=log)
    assert log.exists()
    # resume onto the ORIGINAL table by replaying accepted clicks.
    g = PairLocalDiagonalFinisher.resume_from_ledger(tbl, log, _local_problem(tgt))
    assert np.array_equal(g.table, fin.table)
    assert g._round_index == res.n_rounds


def test_atomic_snapshot_written_and_no_tmp(tmp_path):
    tgt = _target((4, 3))
    tbl = _target((4, 3), seed=7)
    snap = tmp_path / "sub" / "diag.npz"
    fin = PairLocalDiagonalFinisher(tbl, _local_problem(tgt))
    fin.run(max_rounds=6, snapshot_path=snap)
    assert snap.exists()
    d = np.load(snap, allow_pickle=True)
    assert "__diag_table" in d.files and "__diag_round" in d.files
    assert not (snap.parent / (snap.name + ".tmp.npz")).exists()


def test_jsonl_rows_carry_provenance_and_no_score_claim(tmp_path):
    tgt = _target((4, 3))
    tbl = _target((4, 3), seed=11)
    log = tmp_path / "l.jsonl"
    PairLocalDiagonalFinisher(tbl, _local_problem(tgt)).run(max_rounds=5, log_path=log)
    rows = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
    assert rows  # at least one accepted round
    for r in rows:
        assert r["score_claim"] is False and r["promotable"] is False
        assert "clicks" in r and "table_sha256" in r and "checkpoint_sha256" in r
        # waterfill-contract fold: SEPARATE per-component deltas (not just ΔS) for a future
        # interaction-aware selector (steps 4-5); #400 is the pair-local tier (steps 1-3,6).
        comp = r["components"]
        assert set(comp) == {"d_seg", "dist_term", "bytes", "rate_term"}
        assert {"before", "after", "delta"} <= set(comp["bytes"])
        # ΔS is the EXACT joint re-verified value (step 6), consistent with the dist+rate split.
        assert r["delta_s"] == pytest.approx(comp["dist_term"]["delta"] + comp["rate_term"]["delta"])


# --------------------------------------------------------------------------------------
# 8. CONFIRM AUTHORITY — a measure that returns confirm=False on a CONFIRM call RAISES (P9).
# --------------------------------------------------------------------------------------
def test_confirm_authority_asserted_when_measure_lies():
    def lying(table, *, confirm):
        return DiagonalObjective(
            per_pair=np.zeros(3), agg=0.0, archive_bytes=0, confirm=False,
            axis="d_seg", n_pairs=3,
        )

    prob = DiagonalProblem(
        render_measure_fn=lying, n_pairs=3, lo=0, hi=7, axis="d_seg",
        probe_frames_fn=_local_probe,
    )
    fin = PairLocalDiagonalFinisher(np.zeros((3, 2), np.int64), prob)
    with pytest.raises(MCFinisherError, match="confirm authority"):
        fin.current_objective()


# --------------------------------------------------------------------------------------
# 9. BISECT — a net-negative composition is salvaged; disabling it rejects the whole batch.
# --------------------------------------------------------------------------------------
def _bisect_problem():
    # pair 0: click helps (target reachable); pair 1: the only 'improving-by-proxy' click
    # actually overshoots and harms on exact recompute → the joint batch is net-negative,
    # the good half (pair 0) must be salvaged.
    n = 2

    def render_measure_fn(table, *, confirm):
        t = np.asarray(table).astype(np.int64)
        # pair0 distortion falls as col0 → 3; pair1 distortion is a V with min at col0==0,
        # so a +1 click (the sweep's proxy pick when starting below) can overshoot.
        d0 = abs(int(t[0, 0]) - 3) / 100.0
        d1 = abs(int(t[1, 0]) - 0) / 100.0
        per_pair = np.array([d0, d1], dtype=np.float64)
        return DiagonalObjective(
            per_pair=per_pair, agg=float(per_pair.mean()), archive_bytes=0,
            confirm=confirm, axis="d_seg", n_pairs=n,
        )

    prob = DiagonalProblem(
        render_measure_fn=render_measure_fn, n_pairs=n, lo=0, hi=7, axis="d_seg",
        probe_frames_fn=_local_probe,
    )
    tbl = np.array([[2], [0]], dtype=np.int64)  # pair0 wants +1 (good); pair1 at min already
    return tbl, prob


def test_bisect_salvages_good_half():
    tbl, prob = _bisect_problem()
    fin = PairLocalDiagonalFinisher(tbl, prob, deltas=(1,), bisect=True)
    res = fin.run(max_rounds=3)
    # pair0 advanced toward 3; pair1 never accepted a harmful click.
    assert res.best_table[0, 0] >= 3
    assert res.best_table[1, 0] == 0
    assert res.best_s_component <= res.start_s_component + 1e-12


# --------------------------------------------------------------------------------------
# 10. PLATEAU — no improving click stops cleanly (already at target).
# --------------------------------------------------------------------------------------
def test_plateau_stop_when_at_target():
    tgt = _target((4, 3))
    fin = PairLocalDiagonalFinisher(tgt.copy(), _local_problem(tgt))
    res = fin.run(max_rounds=10)
    assert res.stop_reason == "plateau"
    assert res.n_clicks_total == 0


# --------------------------------------------------------------------------------------
# 11. DETERMINISM — same table+problem ⇒ identical result (sweep is RNG-free).
# --------------------------------------------------------------------------------------
def test_determinism_identical_runs():
    tgt = _target((5, 4))
    tbl = _target((5, 4), seed=44)
    r1 = PairLocalDiagonalFinisher(tbl, _local_problem(tgt)).run(max_rounds=30)
    r2 = PairLocalDiagonalFinisher(tbl, _local_problem(tgt)).run(max_rounds=30)
    assert r1.best_table_sha256 == r2.best_table_sha256
    assert r1.n_clicks_total == r2.n_clicks_total
    assert [o.accepted for o in r1.outcomes] == [o.accepted for o in r2.outcomes]


# --------------------------------------------------------------------------------------
# 12. VALIDATION — table shape/dtype, axis, lo/hi, column bounds, per_pair shape.
# --------------------------------------------------------------------------------------
def test_table_validation():
    tgt = _target((3, 2))
    prob = _local_problem(tgt)
    with pytest.raises(MCFinisherError, match="2-D"):
        PairLocalDiagonalFinisher(np.zeros(3, np.int64), prob)
    with pytest.raises(MCFinisherError, match="rows"):
        PairLocalDiagonalFinisher(np.zeros((4, 2), np.int64), prob)
    with pytest.raises(MCFinisherError, match="integer"):
        PairLocalDiagonalFinisher(np.zeros((3, 2), np.float32), prob)
    with pytest.raises(MCFinisherError, match="out of range"):
        PairLocalDiagonalFinisher(np.zeros((3, 2), np.int64), prob, columns=[5])


def test_objective_and_problem_validation():
    with pytest.raises(MCFinisherError, match="axis must be"):
        DiagonalObjective(per_pair=np.zeros(2), agg=0.0, archive_bytes=0, confirm=True,
                          axis="bogus", n_pairs=2)
    with pytest.raises(MCFinisherError, match=r"must be <= hi"):
        DiagonalProblem(render_measure_fn=lambda t, *, confirm: None, n_pairs=2, lo=5, hi=1)
    # per_pair shape mismatch surfaces at measure time.
    prob = DiagonalProblem(
        render_measure_fn=lambda t, *, confirm: DiagonalObjective(
            per_pair=np.zeros(99), agg=0.0, archive_bytes=0, confirm=confirm, axis="d_seg", n_pairs=99,
        ),
        n_pairs=3, lo=0, hi=7, axis="d_seg", probe_frames_fn=_local_probe,
    )
    with pytest.raises(MCFinisherError, match="pairs"):
        prob.measure(np.zeros((3, 2), np.int64), confirm=True)


def test_axis_s_component_seg_vs_pose():
    seg = DiagonalObjective(per_pair=np.array([0.01, 0.02]), agg=0.015, archive_bytes=1000,
                            confirm=True, axis="d_seg", n_pairs=2)
    assert seg.axis_s_component() == pytest.approx(seg_term(0.015) + rate_term(1000))
    pos = DiagonalObjective(per_pair=np.array([0.001, 0.002]), agg=0.0015, archive_bytes=1000,
                            confirm=True, axis="d_pose", n_pairs=2)
    assert pos.axis_s_component() == pytest.approx(pose_term(0.0015) + rate_term(1000))


# --------------------------------------------------------------------------------------
# 13. COLUMN SUBSET — a restricted sweep only touches the given columns.
# --------------------------------------------------------------------------------------
def test_column_subset_restricts_sweep():
    tgt = _target((4, 3))
    tbl = _target((4, 3), seed=6)
    fin = PairLocalDiagonalFinisher(tbl, _local_problem(tgt), columns=[1])
    res = fin.run(max_rounds=20)
    # only column 1 may differ from the original; columns 0 and 2 are untouched.
    diff_cols = np.where((res.best_table != tbl).any(axis=0))[0].tolist()
    assert diff_cols in ([], [1])


# --------------------------------------------------------------------------------------
# 14. 4c′ SEAM — byte-close pose surfaces load; both measure factories bind a callable.
# --------------------------------------------------------------------------------------
def test_byte_close_pose_surfaces_load():
    ns = load_byte_close_pose_surfaces()
    for name in ("parse_pose_carrier", "serialize_pose_carrier", "pose_carrier_confirm"):
        assert callable(getattr(ns, name))


def test_make_byte_close_xi_pose_measure_binds_callable():
    fn = make_byte_close_xi_pose_measure(
        build_pose_carrier_bytes=lambda tbl: b"PCARSTUB",
        inflate_and_read_raw=lambda pc: "reports/does_not_run_at_build.raw",
        eval_pairs=600, gt_cache=None, num_pairs=600,
    )
    assert callable(fn)  # the seam binds; the heavy measure fires post-launch (D27b).


def test_make_through_r_code_measure_binds_callable():
    fn = make_through_r_code_measure(render_frame1_fn=lambda tbl: [], byte_cost_fn=lambda tbl: 0)
    assert callable(fn)  # wired to measure_through_r; per_pair_dseg IS the vector (no harness change).


# --------------------------------------------------------------------------------------
# 15. POSE-AXIS diagonal ≡ sequential + marginal ranking still descends.
# --------------------------------------------------------------------------------------
def _pose_local_measure(tgt):
    # d_pose in the ~1e-3 band (marginal weight large but finite); pair-local.
    tgt = np.asarray(tgt, dtype=np.int64)

    def render_measure_fn(table, *, confirm):
        t = np.asarray(table).astype(np.int64)
        per_pair = np.abs(t - tgt).sum(axis=1).astype(np.float64) / 1e5 + 1e-4
        return DiagonalObjective(
            per_pair=per_pair, agg=float(per_pair.mean()), archive_bytes=0,
            confirm=confirm, axis="d_pose", n_pairs=t.shape[0],
        )

    return render_measure_fn


def test_pose_axis_diagonal_equals_sequential_and_descends():
    tgt = _target((4, 3))
    tbl = _target((4, 3), seed=8)
    prob = DiagonalProblem(
        render_measure_fn=_pose_local_measure(tgt), n_pairs=4, lo=0, hi=7, axis="d_pose",
        probe_frames_fn=_local_probe,
    )
    # diagonal per-pair == sequential per-pair (pair-local).
    col, delta = 1, 1
    diag = tbl.copy()
    diag[:, col] = np.clip(diag[:, col] + delta, 0, 7)
    diag_pp = prob.measure(diag, confirm=True).per_pair
    seq = np.empty(4)
    for p in range(4):
        one = tbl.copy()
        one[p, col] = np.clip(one[p, col] + delta, 0, 7)
        seq[p] = prob.measure(one, confirm=True).per_pair[p]
    assert np.allclose(diag_pp, seq)
    res = PairLocalDiagonalFinisher(tbl, prob).run(max_rounds=40)
    assert res.best_s_component <= res.start_s_component + 1e-12
