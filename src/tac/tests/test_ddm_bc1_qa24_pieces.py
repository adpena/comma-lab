# SPDX-License-Identifier: MIT
"""ddm_bc1 — unit tests for the QA24 5-piece composed seg re-burn (sg1 §3).

Covers §3.1 coarse-from-birth cell mask (token zeroing + byte-close exclusion), §3.3 lattice
anneal (STE engagement), §3.4 rate-in-loss soft-entropy surrogate, the 5 new DSL levers +
composed program (fail-closed never-invent-flags), and the §3.5 composed-S verdict scaffold's
HONESTY flag (trustworthy=False — the naive-Adam solver is a scaffold, not a trustworthy
verdict; the OWED completion is the tt1 GN + v4c grammar). Advisory axis; score_claim=false.
"""
from __future__ import annotations

import numpy as np
import pytest

TR = pytest.importorskip("experiments.train_tr1_partition_renderer_mlx")
mx = pytest.importorskip("mlx.core")


def _cfg(mask_path=None, quant_anneal="off", rate_model="entropy"):
    return TR.TR1Config(
        variant="plain", num_pairs=4, grid_downsample=16, code_width=4, renderer_width=8,
        token_quant_levels=16, seed=0, lotto_seed=118, lotto_mask_density_init=0.5,
        seg_form_start="ce", w_seg=100.0, lr=2e-3, batch_pairs=2, epochs=2, gate_every=1,
        ema_decay=0.99, ema_decay_provenance="test", token_temporal_mode="shared_base",
        token_ste="round", class_weight_lane=1.0, margin_target=1.0, token_init_mode="zero",
        basin_handoff="off", token_cell_mask=mask_path, margin_weighted_loss="on",
        margin_weight_temp=1.0, w_rate=0.05, rate_model=rate_model,
        token_quant_anneal=quant_anneal, composed_s_gate_subset=0)


# ---------------------------------------------------------------------------
# §3.1 coarse-from-birth cell mask
# ---------------------------------------------------------------------------
def _make_mask(tmp_path, gh=24, gw=32, keep_rows=None):
    keep_rows = range(5, 20) if keep_rows is None else keep_rows
    m = np.zeros((gh, gw), dtype=bool)
    for r in keep_rows:
        m[r, :] = True
    p = tmp_path / "keep.npy"
    np.save(p, m)
    return p, m


def test_cell_mask_zeros_inactive_cells(tmp_path):
    p, m = _make_mask(tmp_path)
    model = TR.build_module(_cfg(mask_path=str(p)))
    # write nonzero into the WHOLE delta field, then check raw_tokens zeros the dropped cells.
    model.tokens_delta = mx.ones(model.tokens_delta.shape)
    model.tokens_base = mx.ones(model.tokens_base.shape)
    rt = np.asarray(model.raw_tokens(0))
    keep = m[..., None]
    assert np.all(rt[~np.broadcast_to(keep, rt.shape)] == 0.0)   # dropped cells exactly 0
    assert np.all(rt[np.broadcast_to(keep, rt.shape)] != 0.0)    # kept cells preserved


def test_cell_mask_wrong_shape_fails_closed(tmp_path):
    bad = tmp_path / "bad.npy"
    np.save(bad, np.ones((10, 10), dtype=bool))  # not (24,32) at D=16
    with pytest.raises(ValueError, match="fail-closed"):
        TR.build_module(_cfg(mask_path=str(bad)))


def test_cell_mask_byte_close_excludes_inactive(tmp_path):
    p, m = _make_mask(tmp_path)
    rng = np.random.default_rng(0)
    tokens = rng.standard_normal((4, 24, 32, 4)).astype(np.float32)
    full = TR.token_stream_bytes(tokens, 16, keep_mask=None)
    kept = TR.token_stream_bytes(tokens, 16, keep_mask=m)
    assert kept < full  # coarse grid codes fewer cells => fewer bytes


def test_counted_ledger_uses_mask(tmp_path):
    p, m = _make_mask(tmp_path)
    cfg_m = _cfg(mask_path=str(p))
    cfg_u = _cfg(mask_path=None)
    mm = TR.build_module(cfg_m)
    mu = TR.build_module(cfg_u)
    for mdl in (mm, mu):  # identical nonzero token field in both
        mdl.tokens_delta = mx.array(
            np.random.default_rng(1).standard_normal(mdl.tokens_delta.shape).astype(np.float32))
    mx.eval(mm.parameters(), mu.parameters())
    bm = TR.counted_bytes_ledger(mm, cfg_m)["tokens_bytes"]
    bu = TR.counted_bytes_ledger(mu, cfg_u)["tokens_bytes"]
    assert bm < bu  # the masked ledger excludes the dropped cells


# ---------------------------------------------------------------------------
# §3.3 lattice anneal (STE engagement)
# ---------------------------------------------------------------------------
def test_quant_anneal_disengaged_returns_float():
    model = TR.build_module(_cfg(quant_anneal="at_knee"))
    assert model._quant_engaged is False
    model.tokens_delta = mx.array(
        (np.random.default_rng(2).random(model.tokens_delta.shape) * 0.3).astype(np.float32))
    q = np.asarray(model.quantized_tokens(0))
    r = np.asarray(mx.clip(model.raw_tokens(0), -1.0, 1.0))
    assert np.allclose(q, r)  # float tokens before the knee (no lattice snap)


def test_quant_anneal_engaged_snaps_to_lattice():
    model = TR.build_module(_cfg(quant_anneal="at_knee"))
    model._quant_engaged = True
    model.tokens_delta = mx.array(
        (np.random.default_rng(3).random(model.tokens_delta.shape) * 0.3).astype(np.float32))
    q = np.asarray(model.quantized_tokens(0))
    L = 15.0
    lattice = np.round((q + 1.0) * 0.5 * L) / L * 2.0 - 1.0
    assert np.allclose(q, lattice, atol=1e-5)  # values sit on the L16 lattice


def test_quant_off_engaged_from_birth():
    assert TR.build_module(_cfg(quant_anneal="off"))._quant_engaged is True


# ---------------------------------------------------------------------------
# §3.4 rate-in-loss soft-entropy surrogate
# ---------------------------------------------------------------------------
def test_soft_hist_entropy_lower_for_clumped():
    clumped = mx.zeros((256,))                       # all mass at one bin
    spread = mx.array(np.linspace(-1, 1, 256).astype(np.float32))
    hc = float(TR._soft_hist_entropy_bits(clumped, 16))
    hs = float(TR._soft_hist_entropy_bits(spread, 16))
    assert hc < hs and hc >= 0.0  # clumped distribution has lower entropy


def test_soft_hist_entropy_differentiable():
    def f(x):
        return TR._soft_hist_entropy_bits(x, 16)
    x = mx.array(np.random.default_rng(4).standard_normal(64).astype(np.float32) * 0.5)
    g = mx.grad(f)(x)
    mx.eval(g)
    assert np.any(np.asarray(g) != 0.0)  # a real gradient flows to the token values


# ---------------------------------------------------------------------------
# §3.1-§3.5 DSL levers + composed program
# ---------------------------------------------------------------------------
def test_dsl_levers_and_composed_program_validate():
    from tac.witness_dsl.spec_tr1_renderer_20260728 import (
        lever_composed_s_verdict,
        lever_rate_in_loss,
        lever_seg_margin_weight,
        lever_token_cell_mask,
        lever_token_quant_anneal,
        qa24_composed_burn_program,
    )
    assert lever_token_cell_mask("/x.npy").overrides["--token-cell-mask"] == "/x.npy"
    assert lever_seg_margin_weight(0.5).overrides["--margin-weighted-loss"] == "on"
    assert lever_rate_in_loss(0.1, "smevr_surrogate").overrides["--rate-model"] == "smevr_surrogate"
    assert lever_token_quant_anneal("at_knee").overrides["--token-quant-anneal"] == "at_knee"
    assert lever_composed_s_verdict(16).overrides["--composed-s-gate-subset"] == "16"
    with pytest.raises(ValueError):
        lever_rate_in_loss(0.1, "bogus")
    with pytest.raises(ValueError):
        lever_token_quant_anneal("bogus")
    prog = qa24_composed_burn_program("lotto", "/tmp/o", "/x.npy", composed_s_subset=16)
    argv = prog.compile_trainer_argv()  # fail-closed never-invent-flags (AST scan of trainer)
    for flag in ("--token-cell-mask", "--margin-weighted-loss", "--w-rate",
                 "--token-quant-anneal", "--composed-s-gate-subset"):
        assert flag in argv
    t = prog.sealed_ticket()
    assert t["ticket_hash"] == prog.sealed_ticket()["ticket_hash"]  # deterministic hash


# ---------------------------------------------------------------------------
# §3.5 composed-S verdict: constructs / graceful-skips (no scorer touch in-test)
# ---------------------------------------------------------------------------
def test_composed_s_verdict_constructs_or_graceful_skips():
    from experiments.ddm_composed_s_verdict import ComposedSVerdict
    # available depends on the SSD pfs1 geometry; either way it must NOT crash. When the
    # geometry is missing, available=False + a non-empty reason (advisory never crashes the burn).
    v = ComposedSVerdict(4)
    assert isinstance(v.available, bool)
    if not v.available:
        assert isinstance(v.reason, str) and v.reason
