#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tests for #311 TropNNC witness trunk reduction (tac.boundary_math.tropnnc_witness_reduction).

Coverage: npz round-trip + cfg decode; forward-all-layers consistency with the canonical forward;
tropical influence ranking; uniform-width kept-set selection + guards; the mean-compensated
structured prune (shapes, film reshape validity, k=0 identity, bias-fold arithmetic on a tiny
hand-built net where the answer is analytic); byte accounting monotonicity; ReductionPlan JSON.

The tests build a TINY synthetic witness (no scorer, no big render) so they run in <1s and assert
the STRUCTURE + ARITHMETIC of the reduction, not a d_seg verdict (that is the n600 screen's job)."""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import tropnnc_witness_reduction as tr


# ---------------------------------------------------------------------------
# a tiny synthetic witness the tests can reason about analytically
# ---------------------------------------------------------------------------
def _tiny_witness(n_hidden=2, hidden_dim=4, in_feat=3, n_classes=5, mod_dim=2, n_pairs=3, seed=0):
    rng = np.random.default_rng(seed)
    p: dict[str, np.ndarray] = {}
    p["in_proj.weight"] = rng.standard_normal((hidden_dim, in_feat))
    p["in_proj.bias"] = rng.standard_normal(hidden_dim)
    p["film.weight"] = rng.standard_normal((n_hidden * 2 * hidden_dim, mod_dim)) * 0.1
    p["film.bias"] = rng.standard_normal(n_hidden * 2 * hidden_dim) * 0.1
    for li in range(n_hidden):
        p[f"hidden.{li}.weight"] = rng.standard_normal((hidden_dim, hidden_dim))
        p[f"hidden.{li}.bias"] = rng.standard_normal(hidden_dim)
    p["out_sdf.weight"] = rng.standard_normal((n_classes, hidden_dim))
    p["out_sdf.bias"] = rng.standard_normal(n_classes)
    p["out_tex.weight"] = rng.standard_normal((3, hidden_dim))
    p["out_tex.bias"] = rng.standard_normal(3)
    p["palette"] = rng.standard_normal((n_classes, 3))
    p["code"] = rng.standard_normal((2 * n_pairs, mod_dim))
    aux = {
        "__cfg_n_hidden": np.int64(n_hidden), "__cfg_hidden_dim": np.int64(hidden_dim),
        "__cfg_activation": np.array("hosc"), "__cfg_softmax_temp": np.float64(1.0),
        "__cfg_hosc_beta": np.float64(1.0), "__cfg_hosc_omega": np.float64(1.0),
        "__cfg_chroma": np.int64(1), "__render_hw": np.array([4, 4]),
        "pose_carrier.xi_stored": np.zeros((n_pairs, 6), np.float32),
    }
    cfg = {
        "n_hidden": n_hidden, "hidden_dim": hidden_dim, "n_classes": n_classes,
        "in_feat": in_feat, "mod_dim": mod_dim, "n_pairs": n_pairs, "activation": "hosc",
        "softmax_temp": 1.0, "hosc_beta": 1.0, "hosc_omega": 1.0, "wire_w0": 20.0,
        "wire_s0": 10.0, "chroma": True,
    }
    return tr.WitnessCheckpoint(params=p, aux=aux, cfg=cfg)


def _write_tiny(tmp_path, ck):
    out = {}
    for k, v in ck.params.items():
        out[k] = np.asarray(v, np.float32)
    out.update({k: np.asarray(v) for k, v in ck.aux.items()})
    path = tmp_path / "tiny_witness.npz"
    np.savez(path, **out)
    return path


# ---------------------------------------------------------------------------
# load / round-trip
# ---------------------------------------------------------------------------
def test_load_witness_decodes_cfg_and_splits_params(tmp_path):
    ck = _tiny_witness()
    path = _write_tiny(tmp_path, ck)
    loaded = tr.load_witness(path)
    assert loaded.n_hidden == 2 and loaded.hidden_dim == 4
    assert loaded.cfg["n_classes"] == 5 and loaded.cfg["in_feat"] == 3
    assert loaded.cfg["softmax_temp"] == 1.0 and loaded.cfg["activation"] == "hosc"
    assert "in_proj.weight" in loaded.params and "pose_carrier.xi_stored" in loaded.aux
    # pose_carrier / __cfg go to aux, not params
    assert not any(k.startswith("pose_carrier") for k in loaded.params)


def test_load_witness_missing_param_raises(tmp_path):
    ck = _tiny_witness()
    del ck.params["out_sdf.weight"]
    path = _write_tiny(tmp_path, ck)
    with pytest.raises(ValueError, match="out_sdf.weight"):
        tr.load_witness(path)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
def test_forward_all_layers_shapes_and_stack_len():
    ck = _tiny_witness(n_hidden=3, hidden_dim=6, in_feat=5)
    feats = np.random.default_rng(1).standard_normal((20, 5))
    stack = tr.forward_all_layers(ck, feats, ck.params["code"][1])
    assert len(stack) == ck.n_hidden + 1  # L0..L_nH
    for s in stack:
        assert s.shape == (20, 6)
        assert np.all(np.abs(s) <= 1.0 + 1e-9)  # tanh-bounded


def test_forward_matches_canonical_generator_phi():
    """forward_all_layers final activation @ out_sdf must reproduce the canonical forward's phi."""
    from tac.boundary_math.lever_b_levelset_generator import levelset_rgb_forward_numpy

    ck = _tiny_witness(n_hidden=2, hidden_dim=4, in_feat=3, n_classes=5)
    feats = np.random.default_rng(2).standard_normal((15, 3))
    code_row = ck.params["code"][1]
    stack = tr.forward_all_layers(ck, feats, code_row)
    phi_mine = stack[-1] @ ck.params["out_sdf.weight"].T + ck.params["out_sdf.bias"]
    pf32 = {k: np.asarray(v, np.float32) for k, v in ck.params.items()}
    _rgb, phi_canon = levelset_rgb_forward_numpy(
        pf32, feats, code_row, n_hidden=2, hidden_dim=4, n_classes=5, activation="hosc",
        softmax_temp=1.0, wire_w0=20.0, wire_s0=10.0, hosc_beta=1.0, hosc_omega=1.0, chroma=True)
    np.testing.assert_allclose(phi_mine, phi_canon, atol=1e-4)


# ---------------------------------------------------------------------------
# stats / ranking
# ---------------------------------------------------------------------------
def test_probe_layer_stats_shapes():
    ck = _tiny_witness(n_hidden=2, hidden_dim=4)
    feats = np.random.default_rng(3).standard_normal((30, ck.cfg["in_feat"]))
    stats = tr.probe_layer_stats(ck, feats, [1, 3, 5])
    assert len(stats.influence) == ck.n_hidden + 1
    for m in range(ck.n_hidden + 1):
        assert stats.mean[m].shape == (4,) and stats.std[m].shape == (4,)
        assert np.all(stats.std[m] >= 0.0) and np.all(stats.influence[m] >= 0.0)
    assert stats.n_probe_px == 90


# ---------------------------------------------------------------------------
# kept-set selection
# ---------------------------------------------------------------------------
def test_select_kept_sets_uniform_width_and_drops_lowest_influence():
    ck = _tiny_witness(n_hidden=2, hidden_dim=5)
    feats = np.random.default_rng(4).standard_normal((25, ck.cfg["in_feat"]))
    stats = tr.probe_layer_stats(ck, feats, [1])
    kept = tr.select_kept_sets(stats, ck.n_hidden, k=2)
    assert set(kept) == set(range(ck.n_hidden + 1))
    widths = {m: kept[m].shape[0] for m in kept}
    assert all(w == 3 for w in widths.values())  # 5 - 2
    # the dropped units are the lowest-influence ones at each layer
    for m in range(ck.n_hidden + 1):
        dropped = np.setdiff1d(np.arange(5), kept[m])
        thresh = stats.influence[m][kept[m]].min()
        assert np.all(stats.influence[m][dropped] <= thresh + 1e-12)


def test_select_kept_sets_k0_keeps_all():
    ck = _tiny_witness()
    feats = np.random.default_rng(5).standard_normal((10, ck.cfg["in_feat"]))
    stats = tr.probe_layer_stats(ck, feats, [1])
    kept = tr.select_kept_sets(stats, ck.n_hidden, k=0)
    for m in kept:
        np.testing.assert_array_equal(kept[m], np.arange(ck.hidden_dim))


def test_select_kept_sets_k_too_large_raises():
    ck = _tiny_witness(hidden_dim=4)
    feats = np.random.default_rng(6).standard_normal((10, ck.cfg["in_feat"]))
    stats = tr.probe_layer_stats(ck, feats, [1])
    with pytest.raises(ValueError, match="cannot drop all"):
        tr.select_kept_sets(stats, ck.n_hidden, k=4)


def test_select_kept_sets_negative_k_raises():
    ck = _tiny_witness()
    feats = np.random.default_rng(7).standard_normal((10, ck.cfg["in_feat"]))
    stats = tr.probe_layer_stats(ck, feats, [1])
    with pytest.raises(ValueError, match=">= 0"):
        tr.select_kept_sets(stats, ck.n_hidden, k=-1)


# ---------------------------------------------------------------------------
# apply_reduction: k=0 identity, shapes, film reshape validity, bias-fold
# ---------------------------------------------------------------------------
def test_apply_reduction_k0_is_identity():
    ck = _tiny_witness()
    feats = np.random.default_rng(8).standard_normal((20, ck.cfg["in_feat"]))
    stats = tr.probe_layer_stats(ck, feats, [1, 3])
    kept = tr.select_kept_sets(stats, ck.n_hidden, 0)
    reduced, w = tr.apply_reduction(ck, kept, stats)
    assert w == ck.hidden_dim
    for key in ck.params:
        np.testing.assert_allclose(reduced[key], ck.params[key], atol=0.0)


def test_apply_reduction_shapes_consistent_and_film_reshape_valid():
    ck = _tiny_witness(n_hidden=3, hidden_dim=6, in_feat=4, n_classes=5, mod_dim=2)
    feats = np.random.default_rng(9).standard_normal((30, 4))
    stats = tr.probe_layer_stats(ck, feats, [1, 3, 5])
    kept = tr.select_kept_sets(stats, ck.n_hidden, 2)
    reduced, w = tr.apply_reduction(ck, kept, stats)
    assert w == 4
    assert reduced["in_proj.weight"].shape == (4, 4)
    for li in range(ck.n_hidden):
        assert reduced[f"hidden.{li}.weight"].shape == (4, 4)
        assert reduced[f"hidden.{li}.bias"].shape == (4,)
    assert reduced["out_sdf.weight"].shape == (5, 4)
    assert reduced["out_tex.weight"].shape == (3, 4)
    # film must reshape to (n_hidden, 2, new_width) — the inflate invariant
    assert reduced["film.weight"].shape == (ck.n_hidden * 2 * w, 2)
    reduced["film.weight"].reshape(ck.n_hidden, 2, w, -1)  # raises if invalid
    reduced["film.bias"].reshape(ck.n_hidden, 2, w)


def test_apply_reduction_final_layer_bias_fold_is_exact():
    """Dropping a final-layer unit with mean-fold must EXACTLY reproduce the head output when the
    dropped unit's activation equals its folded mean (analytic check on out_sdf)."""
    ck = _tiny_witness(n_hidden=1, hidden_dim=4, in_feat=2, n_classes=5, mod_dim=1, n_pairs=2)
    feats = np.random.default_rng(10).standard_normal((12, 2))
    stats = tr.probe_layer_stats(ck, feats, [1])
    kept = tr.select_kept_sets(stats, ck.n_hidden, 1)
    reduced, w = tr.apply_reduction(ck, kept, stats)
    nH = ck.n_hidden
    dN = int(np.setdiff1d(np.arange(ck.hidden_dim), kept[nH])[0])
    # baseline head bias + dropped col * mean == reduced head bias (the fold identity)
    expected = ck.params["out_sdf.bias"] + ck.params["out_sdf.weight"][:, dN] * stats.mean[nH][dN]
    np.testing.assert_allclose(reduced["out_sdf.bias"], expected, atol=1e-12)


def test_apply_reduction_nonuniform_kept_raises():
    ck = _tiny_witness(n_hidden=2, hidden_dim=5)
    feats = np.random.default_rng(11).standard_normal((10, ck.cfg["in_feat"]))
    stats = tr.probe_layer_stats(ck, feats, [1])
    bad = {0: np.arange(3), 1: np.arange(4), 2: np.arange(3)}  # non-uniform widths
    with pytest.raises(ValueError, match="non-uniform"):
        tr.apply_reduction(ck, bad, stats)


# ---------------------------------------------------------------------------
# npz round-trip of a reduced checkpoint (byte-close compatible)
# ---------------------------------------------------------------------------
def test_write_reduced_npz_roundtrips_and_updates_hidden_dim(tmp_path):
    ck = _tiny_witness(n_hidden=2, hidden_dim=6, in_feat=3)
    feats = np.random.default_rng(12).standard_normal((20, 3))
    stats = tr.probe_layer_stats(ck, feats, [1, 3])
    kept = tr.select_kept_sets(stats, ck.n_hidden, 2)
    reduced, w = tr.apply_reduction(ck, kept, stats)
    out = tr.write_reduced_npz(ck, reduced, w, tmp_path / "reduced.npz")
    reloaded = tr.load_witness(out)
    assert reloaded.hidden_dim == w == 4
    assert int(reloaded.aux["__cfg_hidden_dim"]) == 4
    # pose_carrier preserved verbatim
    np.testing.assert_array_equal(reloaded.aux["pose_carrier.xi_stored"], ck.aux["pose_carrier.xi_stored"])
    # a reduced witness forward runs at the new width
    stack = tr.forward_all_layers(reloaded, feats, reloaded.params["code"][1])
    assert stack[-1].shape == (20, 4)


# ---------------------------------------------------------------------------
# byte accounting + plan
# ---------------------------------------------------------------------------
def test_trunk_blob_bytes_positive_and_monotone_decreasing():
    ck = _tiny_witness(n_hidden=3, hidden_dim=8, in_feat=6)
    feats = np.random.default_rng(13).standard_normal((40, 6))
    stats = tr.probe_layer_stats(ck, feats, [1, 3, 5])
    base = tr.trunk_blob_bytes(ck.params)
    assert base > 0
    prev = base
    for k in (1, 2, 3):
        _plan, reduced = tr.build_reduction_plan(ck, stats, k)
        b = tr.trunk_blob_bytes(reduced)
        assert b <= prev  # more units dropped -> fewer (or equal) bytes
        prev = b


def test_build_reduction_plan_accounting_and_json():
    ck = _tiny_witness(n_hidden=2, hidden_dim=8, in_feat=5)
    feats = np.random.default_rng(14).standard_normal((30, 5))
    stats = tr.probe_layer_stats(ck, feats, [1, 3])
    plan, reduced = tr.build_reduction_plan(ck, stats, 2)
    assert plan.new_width == 6
    assert plan.dropped_params > 0
    assert plan.bytes_saved == plan.baseline_blob_bytes - plan.reduced_blob_bytes
    d = plan.to_json_dict()
    assert d["k_per_layer"] == 2 and d["new_width"] == 6
    assert d["trunk_blob_bytes_saved"] == plan.bytes_saved
    assert set(d["kept_widths"].values()) == {6}
    # influence_dropped has k units per output layer
    assert len(plan.influence_dropped) == 2 * (ck.n_hidden + 1)


def test_dropped_params_matches_shape_delta():
    ck = _tiny_witness(n_hidden=2, hidden_dim=6, in_feat=4)
    feats = np.random.default_rng(15).standard_normal((20, 4))
    stats = tr.probe_layer_stats(ck, feats, [1])
    plan, reduced = tr.build_reduction_plan(ck, stats, 1)
    n_before = sum(int(np.prod(v.shape)) for v in ck.params.values())
    n_after = sum(int(np.prod(v.shape)) for v in reduced.values())
    assert plan.dropped_params == n_before - n_after
    assert plan.dropped_params > 0
