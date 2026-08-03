"""ddm_tb1 — tests for the tr1 trained partition→pixel renderer scaffold (T0).

Unit tests of the MECHANICS (quant STE, variants, ledger, A1 adjudication, DSL
compile/validate, checkpoint roundtrip). Synthetic tensors here verify code
behavior ONLY — no scorer-behavior claim is made from these tests (NO-FAKE #3);
realized-scorer evidence comes from the T1/T2 windows on real GT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
WORKTREE = _HERE.parents[3]
for _p in (str(WORKTREE), str(WORKTREE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.train_tr1_partition_renderer_mlx import (  # noqa: E402
    GATE_BLOCK_PAIRS,
    TR1Config,
    a1_adjudicate,
    build_module,
    counted_bytes_ledger,
    derive_ema_decay,
    load_checkpoint,
    resolve_gate_ids,
    save_checkpoint,
    token_stream_bytes,
)

mx = pytest.importorskip("mlx.core")


def _cfg(variant: str = "plain", **kw) -> TR1Config:
    base = {
        "variant": variant, "num_pairs": 4, "grid_downsample": 16, "code_width": 2,
        "renderer_width": 8, "token_quant_levels": 16, "seed": 3, "lotto_seed": 118,
        "lotto_mask_density_init": 0.5, "seg_form_start": "ce", "w_seg": 100.0, "lr": 1e-3,
        "batch_pairs": 2, "epochs": 2, "gate_every": 1, "ema_decay": 0.95,
        "ema_decay_provenance": "test", "token_temporal_mode": "shared_base",
        "token_ste": "round", "class_weight_lane": 1.0, "margin_target": 1.0}
    base.update(kw)
    return TR1Config(**base)


# ---------------------------------------------------------------- config ----
def test_grid_props_and_hash_stable():
    cfg = _cfg()
    assert (cfg.grid_h, cfg.grid_w, cfg.n_upsample) == (24, 32, 4)
    assert cfg.config_hash() == _cfg().config_hash()
    assert cfg.config_hash() != _cfg(seed=4).config_hash()


def test_d12_lattice_refused():
    with pytest.raises(ValueError, match="power of 2"):
        _ = _cfg(grid_downsample=12).n_upsample


def test_ema_decay_derived_from_run_geometry():
    d1, prov = derive_ema_decay(200)
    d2, _ = derive_ema_decay(2000)
    assert "DERIVED" in prov
    assert 0.9 <= d1 < d2 <= 0.9995  # monotone in U, clamped


# ---------------------------------------------------------------- model -----
def test_plain_render_shape_range_determinism():
    cfg = _cfg("plain")
    m1, m2 = build_module(cfg), build_module(cfg)
    r1 = np.asarray(m1.render_frame(0))
    r2 = np.asarray(m2.render_frame(0))
    assert r1.shape == (1, 384, 512, 3)
    assert float(r1.min()) >= 0.0 and float(r1.max()) <= 255.0
    np.testing.assert_array_equal(r1, r2)  # same seed => bit-identical build


def test_token_quant_ste_on_lattice_and_grad_flows():
    cfg = _cfg("plain", token_quant_levels=5)
    m = build_module(cfg)
    m.tokens_base = mx.array(np.linspace(-1.2, 1.2, 24 * 32 * 2, dtype=np.float32
                                         ).reshape(24, 32, 2))
    q = np.asarray(m.quantized_tokens(0))
    lattice = np.linspace(-1.0, 1.0, 5)
    assert np.allclose(np.min(np.abs(q[..., None] - lattice), axis=-1), 0.0, atol=1e-6)

    def f(model):
        return mx.sum(model.quantized_tokens(1))

    import mlx.nn as nn
    _, grads = nn.value_and_grad(m, f)(m)
    g = np.asarray(grads["tokens_delta"])
    assert np.count_nonzero(g[1]) > 0        # STE passes gradient to frame 1 delta
    assert np.count_nonzero(g[0]) == 0       # other frames untouched
    assert np.count_nonzero(np.asarray(grads["tokens_base"])) > 0


def test_dither_ste_differs_from_round_and_is_deterministic():
    cfg_r = _cfg("plain", token_ste="round")
    cfg_d = _cfg("plain", token_ste="dither")
    mr, md1, md2 = build_module(cfg_r), build_module(cfg_d), build_module(cfg_d)
    t = mx.array((np.random.default_rng(0).random((24, 32, 2)) * 1.6 - 0.8
                  ).astype(np.float32))
    mr.tokens_base = t
    md1.tokens_base = t
    md2.tokens_base = t
    qr, qd1, qd2 = (np.asarray(m.quantized_tokens(0)) for m in (mr, md1, md2))
    assert not np.array_equal(qr, qd1)
    np.testing.assert_array_equal(qd1, qd2)  # seeded dither = deterministic


def test_lotto_bank_hidden_and_regenerable():
    cfg = _cfg("lotto")
    m1, m2 = build_module(cfg), build_module(cfg)
    from mlx.utils import tree_flatten

    names = {k for k, _ in tree_flatten(m1.trainable_parameters())}
    assert not any(k.startswith("_bank") or "bank" in k for k in names)
    assert any(k.startswith("s_") for k in names) and any(k.startswith("g_") for k in names)
    for k in m1._bank.tensors:
        np.testing.assert_array_equal(np.asarray(m1._bank.tensors[k]),
                                      np.asarray(m2._bank.tensors[k]))
    m3 = build_module(_cfg("lotto", lotto_seed=119))
    assert not np.array_equal(np.asarray(m1._bank.tensors["conv0"]),
                              np.asarray(m3._bank.tensors["conv0"]))


def test_lotto_mask_is_hard_and_score_grad_flows():
    cfg = _cfg("lotto")
    m = build_module(cfg)
    s = np.asarray(m.s_conv0)
    w = np.asarray(m._weight("conv0"))
    bank = np.asarray(m._bank.tensors["conv0"])
    g = np.asarray(m.g_conv0).reshape(-1, 1, 1, 1)
    np.testing.assert_allclose(w, bank * (s > 0) * g, rtol=1e-6)

    import mlx.nn as nn

    def f(model):
        return mx.sum(model.render_frame(0))

    _, grads = nn.value_and_grad(m, f)(m)
    assert float(np.abs(np.asarray(grads["s_conv0"])).sum()) > 0.0  # STE through mask


# ---------------------------------------------------------------- ledger ----
# The rule-118 counted-vs-free boundary of ``counted_bytes_ledger``. EVERY key the
# ledger emits must be classified here:
#   COUNTED       -> archive.zip payload; MUST be summed into total_counted_bytes.
#   OBSERVABILITY -> decomposition telemetry; MUST NOT be summed (max-observability
#                    non-negotiable keeps them in the dict, never in the price).
# A new, unclassified key FAILS this test by construction — the author must decide
# which side of the boundary it is on. That decision is a compliance decision: a
# byte-bearing key left out of the total UNDER-prices the rate term.
COUNTED_LEDGER_KEYS = frozenset({
    "tokens_bytes",             # token stream (smevr or zlib per cfg.byte_ledger_coder)
    "renderer_bytes",           # plain: int8 weights | lotto: 1-bit mask + fp16 mods
    "selector_ledger_bytes",    # every decoder-visible VIDEO-SELECTED choice (eu1)
    "rowband_spec_bytes",       # QA84 §4.2 row-band grammar spec (decoder side-info)
})
OBSERVABILITY_LEDGER_KEYS = frozenset({
    "tokens_bytes_zlib",        # legacy temporal-delta price, kept for decomposition
    "tokens_bytes_smevr",       # r7 coder price (== tokens_bytes when smevr is used)
    "token_ledger_coder",       # which coder priced tokens_bytes (a str, not bytes)
})


def test_counted_ledger_keys_and_selector_counted_both_variants():
    for variant in ("plain", "lotto"):
        m = build_module(_cfg(variant))
        led = counted_bytes_ledger(m, _cfg(variant))
        assert COUNTED_LEDGER_KEYS <= set(led)
        # rule-118 boundary: the total is EXACTLY the counted streams, no more, no less.
        assert led["total_counted_bytes"] == sum(int(led[k]) for k in COUNTED_LEDGER_KEYS)
        unclassified = (set(led) - COUNTED_LEDGER_KEYS - OBSERVABILITY_LEDGER_KEYS
                        - {"total_counted_bytes"})
        assert not unclassified, (
            f"unclassified counted_bytes_ledger key(s) {sorted(unclassified)}: add each to "
            "COUNTED_LEDGER_KEYS (archive payload -> summed into total_counted_bytes) or to "
            "OBSERVABILITY_LEDGER_KEYS (telemetry -> never summed). Leaving a byte-bearing "
            "key out of the total UNDER-prices the rate term.")
        assert led["selector_ledger_bytes"] > 0  # rule-118: selection COUNTED for BOTH
        # No row-band grammar on these configs, so that term is degenerate HERE; the
        # nonzero case is covered by test_ddm_b2b_burn2_composition.py
        # ::test_rowband_ledger_counts_spec_bytes_in_total.
        assert led["rowband_spec_bytes"] == 0


def test_lotto_renderer_bytes_below_plain_int8():
    p = counted_bytes_ledger(build_module(_cfg("plain")), _cfg("plain"))
    lo = counted_bytes_ledger(build_module(_cfg("lotto")), _cfg("lotto"))
    assert lo["renderer_bytes"] < p["renderer_bytes"]  # 1-bit mask + fp16 mods < int8


def test_token_stream_temporal_delta_exploits_static_structure():
    rng = np.random.default_rng(0)
    frame = (rng.random((1, 24, 32, 2)) * 2 - 1).astype(np.float32)
    static = np.repeat(frame, 8, axis=0)
    indep = (rng.random((8, 24, 32, 2)) * 2 - 1).astype(np.float32)
    assert token_stream_bytes(static, 16) < token_stream_bytes(indep, 16) / 3


# ---------------------------------------------------------------- A1 gate ---
def test_a1_first_gate_and_coupled_descent():
    assert a1_adjudicate(None, {"realized_gate_dseg_mean": 0.5}, None, 1.0)[
        "a1_classification"] == "FIRST_GATE"
    prev = {"realized_gate_dseg_mean": 0.50}
    cur = {"realized_gate_dseg_mean": 0.40}
    out = a1_adjudicate(prev, cur, smooth_prev=10.0, smooth_cur=9.0)
    assert out["a1_classification"] == "COUPLED_DESCENT" and not out["a1_alarm"]


def test_a1_alarm_fires_on_smooth_only_descent():
    prev = {"realized_gate_dseg_mean": 0.500}
    cur = {"realized_gate_dseg_mean": 0.4999}  # realized flat
    out = a1_adjudicate(prev, cur, smooth_prev=10.0, smooth_cur=9.0)  # smooth -10%
    assert out["a1_alarm"] and out["a1_classification"] == "A1_REALIZATION_GAP_ALARM"


def test_gate_ids_geometry():
    assert resolve_gate_ids(24) == tuple(range(24))
    g = resolve_gate_ids(600)
    assert g[:4] == GATE_BLOCK_PAIRS and len(g) == 36 and len(set(g)) == 36
    assert resolve_gate_ids(600) == g  # deterministic (rng(0))


# ------------------------------------------------------------- checkpoint ---
def test_checkpoint_roundtrip(tmp_path):
    from mlx.utils import tree_flatten

    cfg = _cfg("plain")
    m = build_module(cfg)
    ema = {k: v + 1.0 for k, v in tree_flatten(m.trainable_parameters())}
    p = tmp_path / "stage_test.npz"
    save_checkpoint(p, model=m, ema=ema, opt_state_flat={}, epoch=7, stage="seg_trunk_ce",
                    cfg=cfg, telemetry_tail=[{"event": "epoch", "epoch": 7}])
    ref = {k: np.asarray(v) for k, v in tree_flatten(m.trainable_parameters())}
    m2 = build_module(_cfg("plain", seed=99))  # different init, then restored
    st = load_checkpoint(p, m2)
    for k, v in tree_flatten(m2.trainable_parameters()):
        np.testing.assert_array_equal(np.asarray(v), ref[k])
    assert st["epoch"] == 7 and st["meta"]["stage"] == "seg_trunk_ce"
    np.testing.assert_array_equal(np.asarray(st["ema"]["tokens_base"]),
                                  ref["tokens_base"] + 1.0)


# ------------------------------------------------------------------- DSL ----
def _dsl():
    import importlib.util

    spec_path = WORKTREE / "src/tac/witness_dsl/spec_tr1_renderer_20260728.py"
    spec = importlib.util.spec_from_file_location("spec_tr1_renderer_20260728", spec_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolves InitVar via sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_dsl_t1_program_compiles_with_real_flags(tmp_path):
    dsl = _dsl()
    prog = dsl.default_t1_smoke_program("lotto", str(tmp_path), num_pairs=24)
    argv = prog.compile_trainer_argv()
    assert argv[0] == dsl.TRAINER_RELPATH
    assert "--variant" in argv and "lotto" in argv
    ticket = prog.sealed_ticket()
    assert ticket["ticket_hash"] == prog.sealed_ticket()["ticket_hash"]  # stable


def test_dsl_validate_refuses_invented_flag(tmp_path):
    dsl = _dsl()
    from tac.witness_dsl.curriculum_dsl import Lever

    prog = dsl.TR1RendererProgramV1(
        levers=(Lever(name="bad", overrides={"--not-a-real-flag": "1"}),),
        num_pairs=4, out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="never-invent-flags"):
        prog.compile_trainer_argv()


def test_dsl_declared_flags_scan_finds_core_flags():
    dsl = _dsl()
    flags = dsl.trainer_declared_flags()
    for f in ("--variant", "--grid-downsample", "--token-ste", "--class-weight-lane",
              "--gate-every", "--max-wall-minutes", "--resume-from"):
        assert f in flags


# ------------------------------------------- ddm_gd4: grid_downsample=32 ----
# mt1 §5 #1 / gd3 §6.1: D=32 is the largest measured rate object on the vehicle
# (archive 101,636 B, ΔS_rate −0.1722397 = 23.72% of the gap). gd3 proved its
# d_seg half is unmeasurable at $0, so the row costs a training run — which means
# the ds=32 code path itself must be verified, not assumed.
def test_ds32_geometry_and_seven_conv_topology():
    cfg = _cfg("lotto", grid_downsample=32)
    assert (cfg.grid_h, cfg.grid_w, cfg.n_upsample) == (12, 16, 5)
    from experiments.train_tr1_partition_renderer_mlx import _conv_shapes

    names = [n for n, _ in _conv_shapes(cfg)]
    assert names == ["conv0", "up0", "up1", "up2", "up3", "up4", "head"]
    # the ds=16 incumbent stays a 6-conv decoder (no silent topology drift)
    assert [n for n, _ in _conv_shapes(_cfg("lotto"))] == [
        "conv0", "up0", "up1", "up2", "up3", "head"]


def test_ds32_render_reaches_full_seg_resolution():
    m = build_module(_cfg("lotto", grid_downsample=32))
    assert tuple(np.asarray(m.render_frame(0)).shape) == (1, 384, 512, 3)


def test_ds32_up4_is_trainable_and_not_inert():
    """POSITIVE CONTROL — fails if ``up4`` exists but never receives gradient.

    The inert-lever class: a layer that is constructed, checkpointed and reported
    but bypassed or gradient-starved. Three independent legs, each of which the
    inert case fails: registered as a trainable param, NONZERO loss gradient, and
    a causal effect on the rendered output.
    """
    from mlx.utils import tree_flatten

    cfg = _cfg("lotto", grid_downsample=32)
    m = build_module(cfg)
    names = {k for k, _ in tree_flatten(m.trainable_parameters())}
    # leg 1: registered (lotto trains score + per-channel gain + bias)
    assert {"s_up4", "g_up4", "b_up4"} <= names
    # the fixed bank must NOT be trainable (it is generic PRNG expansion, rule-118 free)
    assert not any(k.startswith("_bank") for k in names)

    # leg 2: nonzero gradient on EVERY upsample conv, up4 included
    def loss_fn(model):
        return mx.mean(model.render_frame(0) ** 2)

    _, grads = mx.value_and_grad(loss_fn)(m)
    g = {k: np.asarray(v) for k, v in tree_flatten(grads)}
    for k in range(cfg.n_upsample):
        assert np.abs(g[f"s_up{k}"]).max() > 0.0, f"up{k} score gradient is identically zero"
        assert np.abs(g[f"b_up{k}"]).max() > 0.0, f"up{k} bias gradient is identically zero"

    # leg 3: causal — perturbing ONLY up4 must move the render
    r0 = np.asarray(m.render_frame(0))
    m.b_up4 = m.b_up4 + 1.0
    assert np.abs(np.asarray(m.render_frame(0)) - r0).max() > 1e-3


def test_ds16_has_no_up4_layer():
    m = build_module(_cfg("lotto"))
    assert not hasattr(m, "s_up4") and not hasattr(m, "b_up4")


def test_dsl_token_grid_lever_admits_32_and_still_refuses_12():
    dsl = _dsl()
    lev = dsl.lever_token_grid(downsample=32, code_width=4)
    assert lev.overrides["--grid-downsample"] == "32"
    with pytest.raises(ValueError):
        dsl.lever_token_grid(downsample=12)


def test_argparse_admits_32_and_refuses_non_power_of_two():
    from experiments.train_tr1_partition_renderer_mlx import build_argparser

    ap = build_argparser()
    base = ["--variant", "lotto", "--out-dir", "/dev/null"]
    assert ap.parse_args([*base, "--grid-downsample", "32"]).grid_downsample == 32
    assert ap.parse_args([*base, "--grid-downsample", "16"]).grid_downsample == 16
    with pytest.raises(SystemExit):
        ap.parse_args([*base, "--grid-downsample", "12"])


# -------------------------------------- ddm_gd4: resume geometry fail-closed --
def test_resume_refuses_cross_grid_downsample_checkpoint(tmp_path):
    """MEASURED defect (gd4 G1): before this guard, a ds=16 checkpoint loaded into
    a ds=32 model WITHOUT raising — ``tokens_delta`` silently took the ds=16 shape
    and ``up4`` stayed at init, absorbed by the resume block's "new param since the
    checkpoint" backfill, which logs a clean-looking line. ``mlx.nn.Module.update``
    assigns a wrong-shaped array without complaint.
    """
    from experiments.train_tr1_partition_renderer_mlx import ResumeGeometryMismatch
    from mlx.utils import tree_flatten

    cfg16 = _cfg("lotto")
    m16 = build_module(cfg16)
    p = tmp_path / "ds16_stage_ce.npz"
    save_checkpoint(p, model=m16,
                    ema=dict(tree_flatten(m16.trainable_parameters())),
                    opt_state_flat={}, epoch=3, stage="seg_trunk_ce", cfg=cfg16,
                    telemetry_tail=[])
    m32 = build_module(_cfg("lotto", grid_downsample=32))
    with pytest.raises(ResumeGeometryMismatch, match="resume REFUSED"):
        load_checkpoint(p, m32)


@pytest.mark.parametrize("kw", [
    {"grid_downsample": 32}, {"code_width": 4}, {"renderer_width": 16}, {"num_pairs": 6},
])
def test_resume_refuses_every_geometry_bearing_field(tmp_path, kw):
    from experiments.train_tr1_partition_renderer_mlx import ResumeGeometryMismatch
    from mlx.utils import tree_flatten

    cfg = _cfg("lotto")
    m = build_module(cfg)
    p = tmp_path / "parent.npz"
    save_checkpoint(p, model=m, ema=dict(tree_flatten(m.trainable_parameters())),
                    opt_state_flat={}, epoch=1, stage="seg_trunk_ce", cfg=cfg,
                    telemetry_tail=[])
    with pytest.raises(ResumeGeometryMismatch):
        load_checkpoint(p, build_module(_cfg("lotto", **kw)))


def test_resume_still_allows_matched_geometry_and_reports_new_params(tmp_path):
    """The guard must not break the legitimate resume: same geometry, changed
    stage-level knobs (lr / w_seg / epochs / ema_decay), and a newly-introduced
    trainable param is REPORTED (for EMA backfill) rather than refused."""
    from mlx.utils import tree_flatten

    cfg = _cfg("lotto")
    m = build_module(cfg)
    p = tmp_path / "parent.npz"
    save_checkpoint(p, model=m, ema=dict(tree_flatten(m.trainable_parameters())),
                    opt_state_flat={}, epoch=5, stage="seg_trunk_ce", cfg=cfg,
                    telemetry_tail=[])
    child = _cfg("lotto", lr=5e-4, w_seg=250.0, epochs=9, ema_decay=0.997)
    st = load_checkpoint(p, build_module(child))
    assert st["epoch"] == 5
    assert st["params_new_since_checkpoint"] == []
    # a newly-introduced lever param is reported, not refused
    st2 = load_checkpoint(p, build_module(_cfg("lotto", head_range_relax="linear")))
    assert st2["params_new_since_checkpoint"] == ["head_relax_gain"]


def test_resume_geometry_helper_is_pure_and_symmetric():
    from experiments.train_tr1_partition_renderer_mlx import (
        ResumeGeometryMismatch,
        assert_resume_geometry_compatible,
    )

    ok = assert_resume_geometry_compatible({"a": (2, 3)}, {"a": (2, 3), "b": (4,)})
    assert ok == ["b"]                      # model-only param => reported for backfill
    with pytest.raises(ResumeGeometryMismatch, match="shape conflicts"):
        assert_resume_geometry_compatible({"a": (2, 3)}, {"a": (2, 4)})
    with pytest.raises(ResumeGeometryMismatch, match="absent from the model"):
        assert_resume_geometry_compatible({"a": (2, 3), "z": (1,)}, {"a": (2, 3)})
