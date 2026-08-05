"""ddm_bs3 (#909) — the FULL-SCOPE / WRONG-PROJECTION cures, with live controls.

THE GENUS (sibling of the vacuity genus, which is about EMPTY scope): a monitored
statistic runs correctly over EVERYTHING it was asked to cover and is
STRUCTURALLY INCAPABLE of registering the defect class it is trusted for,
because it is a contraction whose KERNEL is occupied on real data.

Two measured calibration instances the campaign already paid for:
  * ddm_dt1 (#903): the reported loss scalar was identical 5/5 runs while 26-28
    of 41 checkpoint arrays had already diverged. A mean annihilates the
    directions the divergence lived in.
  * ddm_ms8 (#873): a codebook "mode share" read 60.7% on a BROKEN codebook and
    51.5% on the RD-optimal one -- it ranked the defect HIGHER.

The two cures under test are EXACT PARTITIONS of the blind scalar, never new
proxies, so each carries an algebraic identity a test can pin.

Every test below pairs a POSITIVE control (the defect MUST be registered) with a
NEGATIVE control (a clean case MUST NOT fire), per design philosophy P4.
"""
from __future__ import annotations

import numpy as np
import pytest

import experiments.train_tr1_partition_renderer_mlx as T

H, W, NC = 8, 10, 5


def _maps(seed: int, n: int = 3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, NC, size=(n, H, W)).astype(np.int64)


def _dseg_mean(realized, gts):
    """The INCUMBENT scalar, verbatim from cpu_verdict_d_seg_argmax_batch."""
    return float(np.mean([np.count_nonzero(realized[i] != g) / g.size
                          for i, g in enumerate(gts)]))


# --------------------------------------------------------------------------
# 1. dseg_by_gt_class -- exactness of the partition
# --------------------------------------------------------------------------
def test_dseg_by_gt_class_sums_to_the_incumbent_mean():
    """The cure must be a PARTITION of the blind scalar, not a different number."""
    gts = list(_maps(1))
    realized = _maps(2)
    parts = T.dseg_by_gt_class(realized, gts)
    assert len(parts) == NC
    assert sum(parts) == pytest.approx(_dseg_mean(realized, gts), rel=1e-12, abs=1e-15)


def test_dseg_by_gt_class_zero_when_perfect():
    """NEGATIVE control: a perfect gate must give an all-zero class vector."""
    gts = list(_maps(3))
    assert T.dseg_by_gt_class(np.stack(gts), gts) == [0.0] * NC


def test_dseg_by_gt_class_refuses_empty_scope():
    """VACUITY: empty scope must RAISE, never return zeros (which read as perfect)."""
    with pytest.raises(ValueError, match="VACUOUS"):
        T.dseg_by_gt_class(np.zeros((0, H, W), np.int64), [])


def test_dseg_by_gt_class_refuses_mismatched_denominator():
    with pytest.raises(ValueError, match="realized maps vs"):
        T.dseg_by_gt_class(_maps(4, n=3), list(_maps(5, n=2)))


# --------------------------------------------------------------------------
# 2. THE POSITIVE CONTROL for the mean's kernel (the ms8 shape)
# --------------------------------------------------------------------------
def test_positive_control_mean_is_blind_to_class_redistribution():
    """KNOWN-POSITIVE: two states with the IDENTICAL blind scalar, one of which
    has moved its entire error mass onto a different class.

    This is the defect the incumbent gate cannot see: `realized_gate_dseg_mean`
    is equal to 12 significant figures across the two states, while the per-class
    error mass has moved completely. The campaign's binding structure IS
    per-class (lane erasure, the Undriv watch, the per-class floors), so this
    kernel is exactly where the live defects sit.

    Mutation check: if `dseg_by_gt_class` were replaced by anything that only
    depends on the total error count, the final assertion fails.
    """
    gt = np.zeros((1, H, W), np.int64)
    gt[0, :4, :] = 0          # top half is class 0
    gt[0, 4:, :] = 1          # bottom half is class 1
    gts = [gt[0]]

    # State A: 10 errors, ALL on class-0 territory.
    a = gt.copy()
    a[0, 0, :] = 2
    # State B: 10 errors, ALL on class-1 territory.
    b = gt.copy()
    b[0, 7, :] = 2

    mean_a, mean_b = _dseg_mean(a, gts), _dseg_mean(b, gts)
    # 1. the INCUMBENT statistic cannot separate them
    assert mean_a == mean_b, "fixture broken: the two states must tie on the mean"
    assert mean_a == pytest.approx(10 / (H * W))

    ca, cb = T.dseg_by_gt_class(a, gts), T.dseg_by_gt_class(b, gts)
    # 2. the CURE does separate them, and does so completely
    assert ca != cb
    assert ca[0] == pytest.approx(mean_a) and ca[1] == 0.0
    assert cb[1] == pytest.approx(mean_b) and cb[0] == 0.0
    # 3. and it is still a partition of the very scalar that was blind
    assert sum(ca) == pytest.approx(mean_a)
    assert sum(cb) == pytest.approx(mean_b)


# --------------------------------------------------------------------------
# 3. flip_direction_counts -- exactness of the 3-way partition
# --------------------------------------------------------------------------
def test_flip_partition_is_exact_and_net_equals_error_delta():
    """toward + away + lateral == the incumbent count, and away - toward is the
    EXACT error-pixel movement. Both identities are what make this a partition
    rather than a fourth proxy."""
    gts = list(_maps(10))
    prev, cur = _maps(11), _maps(12)
    d = T.flip_direction_counts(cur, prev, gts)
    incumbent = int(np.count_nonzero(cur != prev))
    assert (d["realized_flips_toward_gt"] + d["realized_flips_away_from_gt"]
            + d["realized_flips_lateral"]) == incumbent
    err_prev = sum(int(np.count_nonzero(prev[i] != g)) for i, g in enumerate(gts))
    err_cur = sum(int(np.count_nonzero(cur[i] != g)) for i, g in enumerate(gts))
    assert d["realized_flips_net_error_px"] == err_cur - err_prev


def test_flip_direction_negative_control_no_change():
    """NEGATIVE control: an unchanged gate must report zeros in every direction."""
    gts = list(_maps(13))
    same = _maps(14)
    d = T.flip_direction_counts(same, same.copy(), gts)
    assert all(v == 0 for v in d.values())


def test_flip_direction_refuses_empty_and_mismatched():
    gts = list(_maps(15))
    with pytest.raises(ValueError, match="VACUOUS"):
        T.flip_direction_counts(np.zeros((0, H, W), np.int64), np.zeros((0, H, W), np.int64), [])
    with pytest.raises(ValueError, match="shape"):
        T.flip_direction_counts(_maps(16, n=3), _maps(17, n=2), gts)


# --------------------------------------------------------------------------
# 4. THE POSITIVE CONTROL for the flip counter's kernel (the sign it drops)
# --------------------------------------------------------------------------
def test_positive_control_flip_count_is_blind_to_sign():
    """KNOWN-POSITIVE: an IMPROVING gate and a REGRESSING gate with the IDENTICAL
    incumbent flip count.

    `realized_flips_vs_prev_gate` is `count(realized != prev)`; wrong->right and
    right->wrong each add 1. MEASURED on the real burn-4 gate series (61
    consecutive gate pairs, 36-pair gate, 7,077,888 px compared per gate): the
    NET error movement is a MEDIAN 5.4% of the counted flips, so ~94.6% of what
    this counter reports cancels -- and it cannot say so.

    Mutation check: replace `flip_direction_counts` with anything computed from
    |change| alone and the direction assertions fail.
    """
    gt = np.zeros((1, H, W), np.int64)
    gts = [gt[0]]

    prev_imp = gt.copy()
    prev_imp[0, 0, :6] = 1      # 6 wrong
    cur_imp = gt.copy()         # ...all repaired

    prev_reg = gt.copy()        # 0 wrong
    cur_reg = gt.copy()
    cur_reg[0, 0, :6] = 1       # ...6 broken

    n_imp = int(np.count_nonzero(cur_imp != prev_imp))
    n_reg = int(np.count_nonzero(cur_reg != prev_reg))
    # 1. the INCUMBENT counter ties an improvement with a regression
    assert n_imp == n_reg == 6

    d_imp = T.flip_direction_counts(cur_imp, prev_imp, gts)
    d_reg = T.flip_direction_counts(cur_reg, prev_reg, gts)
    # 2. the CURE separates them, and gets the sign right
    assert d_imp["realized_flips_toward_gt"] == 6
    assert d_imp["realized_flips_away_from_gt"] == 0
    assert d_imp["realized_flips_net_error_px"] == -6      # improved
    assert d_reg["realized_flips_toward_gt"] == 0
    assert d_reg["realized_flips_away_from_gt"] == 6
    assert d_reg["realized_flips_net_error_px"] == +6      # regressed


def test_positive_control_lateral_flips_are_pure_cancellation():
    """The third bucket: wrong -> DIFFERENTLY wrong. The incumbent counts these
    as change; they move d_seg by exactly zero. This is the cancellation channel
    the burn-4 measurement is dominated by."""
    gt = np.zeros((1, H, W), np.int64)
    gts = [gt[0]]
    prev = gt.copy()
    prev[0, 0, :5] = 1
    cur = gt.copy()
    cur[0, 0, :5] = 2
    d = T.flip_direction_counts(cur, prev, gts)
    assert int(np.count_nonzero(cur != prev)) == 5      # incumbent: "5 flips"
    assert d["realized_flips_lateral"] == 5
    assert d["realized_flips_net_error_px"] == 0        # cure: nothing moved


# --------------------------------------------------------------------------
# 5. The GUARD -- a flat mean may not stand as "nothing moved"
# --------------------------------------------------------------------------
def test_guard_fires_when_flat_mean_hides_class_motion():
    """POSITIVE control for the guard, built on the ep454 shape (a flat realized
    mean read as evidence that nothing happened)."""
    prev = {"realized_gate_dseg_mean": 0.005064,
            "realized_gate_dseg_by_gt_class": [0.004, 0.001, 0.0, 0.0, 0.000064]}
    cur = {"realized_gate_dseg_mean": 0.005067,
           "realized_gate_dseg_by_gt_class": [0.001, 0.004, 0.0, 0.0, 0.000067]}
    rz_drop = (prev["realized_gate_dseg_mean"] - cur["realized_gate_dseg_mean"]) / abs(
        prev["realized_gate_dseg_mean"])
    assert abs(rz_drop) < T.A1_REALIZED_DROP_REL, "fixture: the mean must read FLAT"
    out = T.a1_class_motion_fields(prev, cur, rz_drop)
    assert out["realized_mean_hid_class_motion"] is True
    assert out["realized_class_l1_rel_since_prev_gate"] > abs(rz_drop)
    # the triangle-inequality relation that makes this principled, not a heuristic
    assert out["realized_class_l1_rel_since_prev_gate"] >= abs(rz_drop)


def test_guard_negative_control_genuinely_flat_state():
    """NEGATIVE control: a mean that is flat BECAUSE the state is flat must NOT
    fire. Without this the guard would be a constant-True field."""
    v = [0.004, 0.001, 0.0, 0.0, 0.000064]
    prev = {"realized_gate_dseg_mean": 0.005064, "realized_gate_dseg_by_gt_class": v}
    cur = {"realized_gate_dseg_mean": 0.005064, "realized_gate_dseg_by_gt_class": list(v)}
    out = T.a1_class_motion_fields(prev, cur, 0.0)
    assert out["realized_mean_hid_class_motion"] is False
    assert out["realized_class_l1_rel_since_prev_gate"] == 0.0


def test_guard_is_absent_not_false_on_legacy_rows():
    """A row predating the per-class vector must yield NO qualification at all --
    an absent field, never a `False` that reads as 'checked and clean'. This is
    the vacuity rule applied to a single field."""
    prev = {"realized_gate_dseg_mean": 0.005064}
    cur = {"realized_gate_dseg_mean": 0.005067}
    assert T.a1_class_motion_fields(prev, cur, 0.0) == {}
    # and a1_adjudicate must still work unchanged on legacy rows
    out = T.a1_adjudicate(prev, cur, 0.6178, 0.5478)
    assert "realized_mean_hid_class_motion" not in out
    assert out["a1_alarm"] is True          # the ep454 raw-signal behaviour, unchanged


def test_guard_never_changes_the_alarm_decision():
    """The guard is ADDITIVE. Adding the per-class vectors must not flip any
    existing verdict -- otherwise this landing silently re-scored history."""
    base_prev = {"realized_gate_dseg_mean": 0.50}
    base_cur = {"realized_gate_dseg_mean": 0.40}
    plain = T.a1_adjudicate(base_prev, base_cur, 1.0, 0.5)
    rich_prev = {**base_prev, "realized_gate_dseg_by_gt_class": [0.5, 0, 0, 0, 0]}
    rich_cur = {**base_cur, "realized_gate_dseg_by_gt_class": [0.0, 0.4, 0, 0, 0]}
    rich = T.a1_adjudicate(rich_prev, rich_cur, 1.0, 0.5)
    assert rich["a1_alarm"] == plain["a1_alarm"]
    assert rich["a1_classification"] == plain["a1_classification"]


# --------------------------------------------------------------------------
# 6. The v4d verifier conjunction -- the empty-projection class fix
# --------------------------------------------------------------------------
def test_conjoin_checks_picks_up_a_new_check_automatically():
    """POSITIVE control for the CLASS fix. The bug was a hand-written
    `A_ok and B_ok and C_ok` while the docstring advertised a (D) check that was
    not in it. A hand-written conjunction cannot see a newly added check; this
    one must."""
    from experiments.ddm_v4d_verify_decode import conjoin_checks

    ok, keys = conjoin_checks({"A_ok": True, "B_ok": True, "C_ok": True})
    assert ok is True and keys == ["A_ok", "B_ok", "C_ok"]
    # a NEW failing check must flip the verdict with no edit to the conjunction
    ok2, keys2 = conjoin_checks(
        {"A_ok": True, "B_ok": True, "C_ok": True, "E_ok": False})
    assert ok2 is False and "E_ok" in keys2


def test_conjoin_checks_ignores_non_check_records():
    """Recorded identities (D_archive_sha256) must NOT be mistaken for checks --
    they have no discriminating power and must not inflate the denominator."""
    from experiments.ddm_v4d_verify_decode import conjoin_checks

    ok, keys = conjoin_checks(
        {"A_ok": True, "D_archive_sha256": "deadbeef", "D_archive_bytes": 123})
    assert ok is True and keys == ["A_ok"]


def test_conjoin_checks_refuses_vacuous_conjunction():
    """`all([])` is True. An empty check set must RAISE, not report a pass --
    the exact empty-scope-reads-as-PASS failure this repo has been bitten by."""
    from experiments.ddm_v4d_verify_decode import conjoin_checks

    with pytest.raises(SystemExit, match="VACUOUS"):
        conjoin_checks({"D_archive_sha256": "deadbeef"})


# --------------------------------------------------------------------------
# 7. Round-1 self-review findings on THIS landing (a fix is unreviewed new code)
# --------------------------------------------------------------------------
def test_partition_refuses_out_of_range_gt_label():
    """A GT label >= n_classes would drop out of every bucket and the partition
    would UNDER-SUM the scalar it claims to decompose -- the cure reproducing the
    defect. Must refuse, not silently under-count."""
    gts = [np.full((H, W), 7, np.int64)]
    with pytest.raises(ValueError, match="n_classes"):
        T.dseg_by_gt_class(np.zeros((1, H, W), np.int64), gts)


def test_flip_counts_refuse_short_gt_denominator():
    """A short `gts` would count fewer pairs than the incumbent count covers, so
    the 3-way partition would stop summing to it -- silently."""
    with pytest.raises(ValueError, match="denominator"):
        T.flip_direction_counts(_maps(20, n=3), _maps(21, n=3), list(_maps(22, n=2)))


def test_bs3_keys_constant_covers_every_field_the_cures_emit():
    """CHECKPOINT-INVARIANCE PIN. The gate row is appended to `telemetry_tail`,
    which is baked into the checkpoint meta. The ddm_bs3 fields are stripped
    before that append, so this landing leaves checkpoint bytes untouched -- but
    only while the strip list stays complete. Derive the truth from the PRODUCING
    functions rather than re-typing it, so a future field cannot silently leak
    into the checkpoint."""
    gts = list(_maps(30))
    prev, cur = _maps(31), _maps(32)
    emitted = set(T.flip_direction_counts(cur, prev, gts))
    emitted |= {"realized_gate_dseg_by_gt_class", "realized_gate_dseg_per_pair_sd"}
    gd1_fields = T.gd1_realized_gate_dseg_fields(tuple(range(36)), [0.001] * 36, 36)
    emitted |= set(gd1_fields) & set(T.BS3_TELEMETRY_ONLY_KEYS)
    emitted |= set(T.a1_class_motion_fields(
        {"realized_gate_dseg_by_gt_class": [1.0, 0, 0, 0, 0]},
        {"realized_gate_dseg_by_gt_class": [0.0, 1.0, 0, 0, 0]}, 0.0))
    assert emitted == set(T.BS3_TELEMETRY_ONLY_KEYS), (
        "a ddm_bs3 field is missing from BS3_TELEMETRY_ONLY_KEYS and would leak "
        f"into the checkpoint: {emitted ^ set(T.BS3_TELEMETRY_ONLY_KEYS)}")


def test_checkpoint_safe_telemetry_row_strips_exactly_the_bs3_fields():
    row = {"epoch": 7, "realized_gate_dseg_mean": 0.1,
           "realized_gate_dseg_by_gt_class": [0.1, 0, 0, 0, 0],
           "realized_flips_toward_gt": 3, "topology_per_class": {}}
    safe = T.checkpoint_safe_telemetry_row(row)
    assert set(safe) == {"epoch", "realized_gate_dseg_mean", "topology_per_class"}
    assert safe["realized_gate_dseg_mean"] == 0.1


def test_ast_guard_telemetry_tail_append_actually_applies_the_strip():
    """ANTI-ORPHAN guard. A complete strip LIST that no call site applies is the
    declared-but-unwired defect. Assert STRUCTURALLY (on the parsed call node,
    not a substring) that every `telemetry_tail.append(...)` in the trainer
    passes its argument through `checkpoint_safe_telemetry_row`.

    Mutation check: revert the call site to a bare `dict(gate_row.items())` and
    this fails."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(T))
    appends = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "append"
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "telemetry_tail"
    ]
    assert appends, "did not find any telemetry_tail.append(...) call -- VACUOUS, not a pass"
    for call in appends:
        assert call.args, "telemetry_tail.append() with no argument"
        arg = call.args[0]
        assert (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                and arg.func.id == "checkpoint_safe_telemetry_row"), (
            f"telemetry_tail.append at line {call.lineno} does not route through "
            "checkpoint_safe_telemetry_row -- bs3 fields would leak into the checkpoint")


def test_incumbent_gate_row_keys_are_not_stripped():
    """NEGATIVE control for the strip list: it must not remove any field the
    checkpoint/resume path already depends on."""
    for k in ("realized_gate_dseg_mean", "realized_gate_dseg_per_pair_max",
              "realized_flips_vs_prev_gate", "topology_per_class", "gate_ids_n",
              "a1_alarm", "a1_classification", "epoch", "gate_params"):
        assert k not in T.BS3_TELEMETRY_ONLY_KEYS


def test_gd1_repaired_gate_observability_fields_are_checkpoint_stripped():
    """Per-pair/HT A1 repair fields go to telemetry.jsonl, not checkpoint meta."""
    for k in ("realized_gate_pair_ids", "realized_gate_dseg_per_pair",
              "realized_gate_dseg_mean_ht", "realized_gate_dseg_mean_ht_design",
              "realized_gate_dseg_per_pair_q95",
              "realized_gate_dseg_per_pair_gt_2x_mean_n"):
        assert k in T.BS3_TELEMETRY_ONLY_KEYS
