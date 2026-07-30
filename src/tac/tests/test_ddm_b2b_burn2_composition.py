"""ddm_b2b burn-2 composition tests — QA86 config corrections + QA83 factorized head + DSL.

Scorer-free by construction: pure-numpy for the byte-ledger + EMA law paths, MLX-CPU (NEVER
Metal) for the tiny head forward. No SegNet/PoseNet, no Metal compute, no paid dispatch.
Pointer 0.1910828242 [contest-CPU] UNMOVED; every asserted number is a byte/shape fact, not
a score claim.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

TR = importlib.import_module("experiments.train_tr1_partition_renderer_mlx")


# ---------------------------------------------------------------------------
# QA86(c) — EMA clamp derives from run length, never a constant.
# ---------------------------------------------------------------------------
def test_ema_decay_burn_geometry_not_clamped():
    # 400 ep x 75 batches = 30,000 updates -> phi=0.5 law d = 1 - 2/(0.5*30000) = 0.99986667.
    d, prov = TR.derive_ema_decay(30_000)
    assert d == pytest.approx(0.99986667, abs=1e-8)
    # the old constant clamp would have capped at 0.9995 — it must NOT bind now.
    assert d > 0.9995
    assert "no constant clamp" in prov


def test_ema_decay_phi_half_warmup_is_half_run_at_all_scales():
    for u in (8, 75, 1500, 4000, 30_000):
        d, _ = TR.derive_ema_decay(u)
        warmup = 2.0 / (1.0 - d)
        assert warmup == pytest.approx(0.5 * u, rel=1e-6)  # phi=0.5 honored everywhere
        assert 0.0 < d < 1.0


def test_ema_decay_ceiling_is_run_length_derived_strictly_below_one():
    for u in (8, 30_000, 1_000_000):
        d, _ = TR.derive_ema_decay(u)
        assert d <= 1.0 - 2.0 / max(u, 8) + 1e-12  # <= run-geometry ceiling
        assert d < 1.0  # never a frozen shadow


# ---------------------------------------------------------------------------
# QA86(b) — SMEVR byte ledger (the shipped coder), lossless price + drop-savings.
# ---------------------------------------------------------------------------
def _fake_shared_base_model(P=6, gh=24, gw=32, c=4, keep=None):
    class _Bank:
        def __init__(self, t):
            self.tensors = t

    class _M:
        pass

    rng = np.random.default_rng(0)
    m = _M()
    m.tokens_base = (rng.random((gh, gw, c)) * 2 - 1).astype(np.float32) * 0.3
    m.tokens_delta = (rng.random((P, gh, gw, c)) * 2 - 1).astype(np.float32) * 0.1
    if keep is None:
        keep = np.ones((gh, gw, 1), np.float32)
    m._cell_mask = _Bank({"keep": keep})
    return m


def _cfg(**kw):
    base = dict(variant="plain", num_pairs=6, grid_downsample=16, code_width=4,
                renderer_width=4, token_quant_levels=16, token_temporal_mode="shared_base",
                seed=0, lotto_seed=118, lotto_mask_density_init=0.5, seg_form_start="ce",
                w_seg=100.0, lr=2e-3, batch_pairs=8, epochs=60, gate_every=5,
                ema_decay=0.997, ema_decay_provenance="test", token_ste="round",
                class_weight_lane=1.0, margin_target=1.0)
    base.update(kw)
    return TR.TR1Config(**base)


def test_smevr_token_price_is_lossless_and_beats_zlib():
    m = _fake_shared_base_model()
    cfg = _cfg()
    full = TR._full_token_field_np(m, cfg)
    assert full.shape == (6, 24, 32, 4)
    q = TR.quantize_tokens_np(full, 16)
    smevr = TR.token_stream_bytes_smevr(q, 16)
    zlib_b = TR.token_stream_bytes(full, 16)
    assert smevr > 0
    # the r7 SMEVR receipt: SMEVR decisively wins the token stream vs zlib-temporal-delta.
    assert smevr < zlib_b


def test_smevr_price_drops_when_cells_dropped():
    gh, gw = 24, 32
    keep = np.ones((gh, gw, 1), np.float32)
    keep[:, gw // 2:, :] = 0.0  # drop half the columns
    full_full = TR._full_token_field_np(_fake_shared_base_model(), _cfg())
    full_drop = TR._full_token_field_np(_fake_shared_base_model(keep=keep), _cfg())
    b_full = TR.token_stream_bytes_smevr(TR.quantize_tokens_np(full_full, 16), 16)
    b_drop = TR.token_stream_bytes_smevr(TR.quantize_tokens_np(full_drop, 16), 16)
    assert b_drop < b_full  # zeroed cells code ~free (keep-mask savings without restriction)


def test_full_token_field_matches_raw_tokens_reconstruction():
    m = _fake_shared_base_model()
    cfg = _cfg()
    full = TR._full_token_field_np(m, cfg)
    keep = np.asarray(m._cell_mask.tensors["keep"], np.float32)
    expect0 = (np.asarray(m.tokens_base) + np.asarray(m.tokens_delta)[0]) * keep
    np.testing.assert_allclose(full[0], expect0, rtol=1e-6)


# ---------------------------------------------------------------------------
# QA83 — factorized head: shapes, rgb control identity, monotone gray, ledger obs keys.
# ---------------------------------------------------------------------------
def test_head_out_channels_and_conv_shapes():
    assert TR._head_out_ch(_cfg(renderer_head_mode="rgb")) == 3
    assert TR._head_out_ch(_cfg(renderer_head_mode="class_field")) == 1
    assert TR._head_out_ch(_cfg(renderer_head_mode="class_field_photo")) == 2
    for mode, ch in (("rgb", 3), ("class_field", 1), ("class_field_photo", 2)):
        head = [s for s in TR._conv_shapes(_cfg(renderer_head_mode=mode)) if s[0] == "head"][0]
        assert head[1][0] == ch


def test_apply_head_shapes_and_rgb_control_identity():
    mx = importlib.import_module("mlx.core")
    mx.set_default_device(mx.cpu)  # NEVER Metal
    rng = np.random.default_rng(1)
    for mode, ch in (("rgb", 3), ("class_field", 1), ("class_field_photo", 2)):
        cfg = _cfg(renderer_head_mode=mode)
        x = mx.array(rng.standard_normal((1, 4, 4, ch)).astype(np.float32))
        out = TR._apply_head(mx, x, cfg)
        assert tuple(out.shape) == (1, 4, 4, 3)
        assert 0.0 <= float(out.min()) and float(out.max()) <= 255.0
    # rgb mode is EXACTLY the pre-QA83 behavior (sigmoid*255) — resume-safe control.
    x3 = mx.array(rng.standard_normal((1, 4, 4, 3)).astype(np.float32))
    ctrl = TR._apply_head(mx, x3, _cfg(renderer_head_mode="rgb"))
    assert float(mx.max(mx.abs(ctrl - mx.sigmoid(x3) * 255.0))) == 0.0


def test_class_field_is_monotone_gray():
    mx = importlib.import_module("mlx.core")
    mx.set_default_device(mx.cpu)
    cf = _cfg(renderer_head_mode="class_field")
    lo = TR._apply_head(mx, mx.array(np.full((1, 1, 1, 1), -4.0, np.float32)), cf)
    hi = TR._apply_head(mx, mx.array(np.full((1, 1, 1, 1), 4.0, np.float32)), cf)
    assert float(hi[0, 0, 0, 0]) > float(lo[0, 0, 0, 0])          # monotone in the class field
    assert float(lo[0, 0, 0, 0]) == float(lo[0, 0, 0, 1]) == float(lo[0, 0, 0, 2])  # gray


# ---------------------------------------------------------------------------
# counted_bytes_ledger: total excludes observability keys; coder selection honored.
# ---------------------------------------------------------------------------
def _fake_plain_model(cfg):
    m = _fake_shared_base_model(P=cfg.num_pairs, gh=cfg.grid_h, gw=cfg.grid_w, c=cfg.code_width)
    rng = np.random.default_rng(2)
    for name, shp in TR._conv_shapes(cfg):
        setattr(m, f"w_{name}", (rng.standard_normal(shp) * 0.1).astype(np.float32))
        setattr(m, f"b_{name}", np.zeros((shp[0],), np.float32))
    return m


def test_counted_ledger_total_excludes_observability_keys():
    cfg = _cfg(byte_ledger_coder="smevr")
    m = _fake_plain_model(cfg)
    led = TR.counted_bytes_ledger(m, cfg)
    assert led["total_counted_bytes"] == (
        led["tokens_bytes"] + led["renderer_bytes"] + led["selector_ledger_bytes"]
        + led["rowband_spec_bytes"])
    assert led["rowband_spec_bytes"] == 0  # no grammar on this fake model
    # observability keys are present but NOT summed into the total.
    assert led["token_ledger_coder"] == "smevr"
    assert "tokens_bytes_zlib" in led and "tokens_bytes_smevr" in led
    assert led["tokens_bytes"] == led["tokens_bytes_smevr"]


def test_counted_ledger_zlib_coder_matches_legacy_path():
    cfg = _cfg(byte_ledger_coder="zlib")
    m = _fake_plain_model(cfg)
    led = TR.counted_bytes_ledger(m, cfg)
    assert led["token_ledger_coder"] == "zlib"
    assert led["tokens_bytes"] == led["tokens_bytes_zlib"] == TR._token_bytes_zlib(m, cfg)


def test_class_field_head_reduces_renderer_bytes():
    rgb = _cfg(renderer_head_mode="rgb")
    cf = _cfg(renderer_head_mode="class_field")
    led_rgb = TR.counted_bytes_ledger(_fake_plain_model(rgb), rgb)
    led_cf = TR.counted_bytes_ledger(_fake_plain_model(cf), cf)
    # k=1 head has fewer output channels => strictly fewer head-conv weight bytes.
    assert led_cf["renderer_bytes"] < led_rgb["renderer_bytes"]


# ---------------------------------------------------------------------------
# DSL levers + burn-2 programs (never-invent-flags fail-closed) + provenance rungs.
# ---------------------------------------------------------------------------
def test_dsl_new_levers_validate_and_carry_provenance():
    spec = importlib.import_module("tac.witness_dsl.spec_tr1_renderer_20260728")
    # head lever
    assert spec.lever_renderer_head("class_field").overrides["--renderer-head-mode"] == "class_field"
    photo = spec.lever_renderer_head("class_field_photo")
    assert "--head-photo-slack-gain" in photo.overrides
    assert "band-lemma" in photo.constant_manifest["--head-photo-slack-gain"]["provenance"]
    # QA86d provenance rungs (no bare constants).
    r = spec.lever_rate_in_loss(0.05, "smevr_surrogate")
    assert r.constant_manifest["--w-rate"]["rung"].startswith("DERIVED-ESTIMATE")
    mw = spec.lever_seg_margin_weight(1.0)
    assert "RACED" in mw.constant_manifest["--margin-weight-temp"]["rung"]
    ema = spec.lever_ema_decay(0.99986667)
    assert ema.constant_manifest["--ema-decay"]["rung"].startswith("DERIVED")


def test_burn2_programs_compile_and_validate():
    b2 = importlib.import_module("tac.witness_dsl.spec_tr1_burn2_20260731")
    rr = b2.qa86_rate_surrogate_race_programs("lotto", "/tmp/o", "/tmp/m.npy")
    assert set(rr) == {"A_entropy", "B_smevr_surrogate"}
    assert rr["A_entropy"].merged_overrides()["--rate-model"] == "entropy"
    assert rr["B_smevr_surrogate"].merged_overrides()["--rate-model"] == "smevr_surrogate"
    hr = b2.qa83_head_race_programs("lotto", "/tmp/o", "/tmp/m.npy")
    assert set(hr) == {"A_rgb", "B_class_field", "C_class_field_photo"}
    assert hr["B_class_field"].merged_overrides()["--renderer-head-mode"] == "class_field"
    res = b2.qa86_mid_run_resume_program(
        "lotto", "/tmp/o", "/tmp/m.npy", "/tmp/ckpt.npz", ema_decay=0.99986667)
    assert res.resume_from == "/tmp/ckpt.npz"
    assert res.merged_overrides()["--ema-decay"] == "0.99986667"
    assert res.merged_overrides()["--byte-ledger-coder"] == "smevr"
    for prog in list(rr.values()) + list(hr.values()) + [res]:
        prog.validate()  # raises on any invented flag


def test_derived_w_rate_matches_exchange_rate():
    b2 = importlib.import_module("tac.witness_dsl.spec_tr1_burn2_20260731")
    n = b2.burn_geometry_n_counted_tokens()
    assert n == 384 * 4 + 600 * 384 * 4  # base + delta stream
    w, prov = b2.derive_w_rate_exchange_rate(n)
    assert w == pytest.approx((25 / 37_545_489) * n / 8.0)
    assert 0.05 < w < 0.10  # ~0.0768: the live 0.05 is ~65% of derived


# ---------------------------------------------------------------------------
# QA84 — the row-band variable-cell grammar (foveation).
# ---------------------------------------------------------------------------
def _grammar():
    g = importlib.import_module("tac.witness_dsl.qa84_rowband_grammar_20260731")
    return g.default_flip_band_grammar()


def test_rowband_grammar_is_foveated_between_the_uniform_lattices():
    dof = _grammar().dof_summary()
    # rowband spends more DOF than uniform coarse (D16) but far less than uniform fine (D8).
    assert dof["uniform_coarse_cells"] < dof["rowband_cells"] < dof["uniform_fine_cells"]
    assert dof["band_spec_bytes"] > 0


def test_rowband_tie_numpy_mlx_parity_and_structure():
    mx = importlib.import_module("mlx.core")
    mx.set_default_device(mx.cpu)
    g = _grammar()
    rng = np.random.default_rng(0)
    field = (rng.random((5, g.fine_gh, g.fine_gw, g.code_width)) * 2 - 1).astype(np.float32)
    tied_np = g.apply_tie_np(field)
    tied_mx = np.asarray(g.apply_tie_mx(mx, mx.array(field)))
    assert np.array_equal(tied_np, tied_mx)                       # backend parity (bit-identical)
    assert np.array_equal(tied_np[0, 0, 0], tied_np[0, 1, 1])     # bulk 2x2 block tied
    band_r = (g.band_row_lo + g.band_row_hi) // 2
    assert np.array_equal(tied_np[0, band_r, 3], field[0, band_r, 3])  # band cell free


def test_rowband_tie_single_frame_shape():  # the raw_tokens path
    mx = importlib.import_module("mlx.core")
    mx.set_default_device(mx.cpu)
    g = _grammar()
    frame = np.zeros((g.fine_gh, g.fine_gw, g.code_width), np.float32)
    assert g.apply_tie_np(frame).shape == (g.fine_gh, g.fine_gw, g.code_width)
    assert tuple(g.apply_tie_mx(mx, mx.array(frame)).shape) == (g.fine_gh, g.fine_gw, g.code_width)


def test_rowband_spec_json_roundtrip_and_render_snap():
    gm = importlib.import_module("tac.witness_dsl.qa84_rowband_grammar_20260731")
    g = gm.RowBandGrammar.from_render_rows(
        160, 240, fine_downsample=8, render_h=384, render_w=512, coarse_factor=2)
    assert (g.fine_gh, g.fine_gw) == (48, 64)
    assert g.band_row_lo % 2 == 0 and g.band_row_hi % 2 == 0          # coarse-aligned
    g2 = gm.RowBandGrammar.from_spec_json(g.spec_json())
    assert g2 == g


def test_rowband_grammar_fail_closed_on_unaligned_band():
    gm = importlib.import_module("tac.witness_dsl.qa84_rowband_grammar_20260731")
    with pytest.raises(ValueError):
        gm.RowBandGrammar(48, 64, band_row_lo=21, band_row_hi=30, coarse_factor=2)  # 21 % 2 != 0


def _fake_rowband_model(g, P=5):
    m = _fake_shared_base_model(P=P, gh=g.fine_gh, gw=g.fine_gw, c=g.code_width)
    m._rowband = g
    return m


def test_full_token_field_applies_rowband_tie():
    g = _grammar()
    cfg = _cfg(grid_downsample=8, token_rowband_spec=g.spec_json())
    m = _fake_rowband_model(g)
    full = TR._full_token_field_np(m, cfg)
    assert np.array_equal(full[0, 0, 0], full[0, 1, 1])              # bulk tied in the reconstruction
    band_r = (g.band_row_lo + g.band_row_hi) // 2
    raw0 = (np.asarray(m.tokens_base) + np.asarray(m.tokens_delta)[0])
    assert np.array_equal(full[0, band_r, 3], raw0[band_r, 3])       # band cell free


def test_rowband_ledger_counts_spec_bytes_in_total():
    g = _grammar()
    cfg = _cfg(grid_downsample=8, token_rowband_spec=g.spec_json())
    m = _fake_rowband_model(g)
    for name, shp in TR._conv_shapes(cfg):
        setattr(m, f"w_{name}", np.zeros(shp, np.float32))
        setattr(m, f"b_{name}", np.zeros((shp[0],), np.float32))
    led = TR.counted_bytes_ledger(m, cfg)
    assert led["rowband_spec_bytes"] == g.band_spec_bytes() > 0
    assert led["total_counted_bytes"] == (
        led["tokens_bytes"] + led["renderer_bytes"] + led["selector_ledger_bytes"]
        + led["rowband_spec_bytes"])


def test_qa84_grammar_race_programs_validate():
    b2 = importlib.import_module("tac.witness_dsl.spec_tr1_burn2_20260731")
    race = b2.qa84_grammar_race_programs("lotto", "/tmp/o", "/tmp/m.npy")
    a = race["A_uniform_D16_drop50"]
    b = race["B_rowband_D8"]
    assert a.merged_overrides()["--grid-downsample"] == "16"
    assert b.merged_overrides()["--grid-downsample"] == "8"
    assert "--token-rowband-spec" in b.merged_overrides()
    a.validate()
    b.validate()  # never-invent-flags fail-closed
