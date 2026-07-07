"""FEED-07b BUILD-wave (#337) behavior tests: #310 FINER++ bias-init + #218 loss-time logit
adjustment + the #220 AA compose-after-downsample unblock — with the byte-identity proofs.

Byte-identity proofs pinned here (the review contract):
  * FINER OFF => the wire-in branch never runs => ZERO RNG draws (the values helper uses a
    DEDICATED ``np.random.default_rng(seed + _FINER_RNG_SALT)`` stream — calling it does not
    advance the shared ``np.random`` state, so even the ON path perturbs no other seeded draw).
  * logit-adjust tau=0.0 => ``_loss_adapter is adapter`` (the SAME object; source-level gate
    asserted) => the make_loss_fn closure/graph is byte-identical. The wrapper itself only ADDS a
    constant per-class offset to segnet logits; posenet passes through as the SAME object.
  * AA ss=1 with a compose_fn is byte-identical to ``render_through_R_mlx`` with the same
    compose_fn (identity downsample => compose-before == compose-after bit-for-bit); ss>1 now
    invokes compose_fn at the BASE grid (the #220 unblock), asserted by shape capture.

means != ends: build plumbing, NOT a score; pointer 0.19110 moves only via a byte-closed exact row.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_TRAINER_PATH = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO / "experiments"))

_TRAINER_SRC = _TRAINER_PATH.read_text()


def _load_trainer():
    spec = importlib.util.spec_from_file_location("tl_feed07b_build", str(_TRAINER_PATH))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ==========================================================================
# #310 FINER++ bias-init
# ==========================================================================
def test_finer_flags_declared_with_byte_identical_defaults():
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser

    ap = build_real_trainer_parser()
    assert ap.get_default("finer_bias_init") is False, "--finer-bias-init must default OFF"
    assert ap.get_default("finer_bias_k") == 10.0
    # BooleanOptionalAction => the --no- form exists (DSL C2/type-compat contract)
    opts = {o for a in ap._actions for o in a.option_strings}
    assert "--finer-bias-init" in opts and "--no-finer-bias-init" in opts


def test_finer_bias_values_deterministic_range_and_dtype():
    tl = _load_trainer()
    v1 = tl._finer_bias_init_values(7, 10.0, 96)
    v2 = tl._finer_bias_init_values(7, 10.0, 96)
    v3 = tl._finer_bias_init_values(8, 10.0, 96)
    assert v1.shape == (96,) and v1.dtype == np.float32
    assert np.array_equal(v1, v2), "same (seed,k,n) must be bit-identical (deterministic repro)"
    assert not np.array_equal(v1, v3), "a different seed must give a different draw"
    assert np.all(np.abs(v1) <= 10.0)
    # WIDE range actually used (the whole point: spread neurons across the period)
    assert v1.max() > 5.0 and v1.min() < -5.0


def test_finer_bias_values_use_dedicated_stream_not_shared_numpy():
    tl = _load_trainer()
    np.random.seed(1234)
    before = np.random.get_state()[1].copy()
    vals = tl._finer_bias_init_values(0, 10.0, 32)
    after = np.random.get_state()[1].copy()
    assert np.array_equal(before, after), "FINER draw must NOT advance the shared np.random stream"
    # and it IS exactly the dedicated stream (seed + salt), so the draw is auditable
    ref = np.random.default_rng(0 + tl._FINER_RNG_SALT).uniform(-10.0, 10.0, size=32).astype(np.float32)
    assert np.array_equal(vals, ref)


def test_finer_bias_values_fail_closed():
    tl = _load_trainer()
    with pytest.raises(ValueError, match="finer-bias-k"):
        tl._finer_bias_init_values(0, 0.0, 8)
    with pytest.raises(ValueError, match="finer-bias-k"):
        tl._finer_bias_init_values(0, -1.0, 8)
    with pytest.raises(ValueError, match="n > 0"):
        tl._finer_bias_init_values(0, 10.0, 0)


def test_finer_wire_in_gated_off_and_fails_closed_on_relu():
    # source-level byte-identity gate: the wire-in only runs under the flag, only touches
    # in_proj.bias, and refuses non-periodic activations.
    assert 'if bool(getattr(args, "finer_bias_init", False)):' in _TRAINER_SRC
    gate = _TRAINER_SRC.index('if bool(getattr(args, "finer_bias_init", False)):')
    block = _TRAINER_SRC[gate:gate + 1600]
    assert 'args.activation not in {"hosc", "wire"}' in block, "must fail closed on relu"
    assert "model.in_proj.bias = mx.array(_finer_bias_init_values(" in block
    assert '"stage": "finer_bias_init"' in block, "observability row required (default-on telemetry)"
    # applied:false stamp on resume (C10 init-lever confound discipline)
    assert "overwritten_by_resume" in block


# ==========================================================================
# #218 loss-time logit adjustment
# ==========================================================================
def test_logit_adjust_flag_declared_default_off():
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser

    ap = build_real_trainer_parser()
    assert ap.get_default("logit_adjust_loss_tau") == 0.0, "--logit-adjust-loss-tau defaults OFF"
    # the pre-existing facet-3 sister pair is UNTOUCHED (collision honestly resolved by a new name)
    assert ap.get_default("logit_adjust_tau") == 1.0
    assert ap.get_default("logit_adjust_per_class") is False


def test_equation_callable_matches_measured_anchor_and_fails_closed():
    from tac.canonical_equations.logit_adjustment_class_prior_20260707 import (
        MEASURED_GT_CLASS_PRIORS_N600,
        logit_adjust_offsets,
    )

    pri = np.asarray(MEASURED_GT_CLASS_PRIORS_N600)
    off = logit_adjust_offsets(pri, 1.0)
    assert off.dtype == np.float32 and off.shape == (5,)
    # tau * log(pi) exactly (normalization is a no-op change within fp tolerance here)
    assert np.allclose(off, np.log(pri / pri.sum()), atol=1e-6)
    # rare classes get the most-negative offsets (the whole mechanism)
    assert off[1] == off.min() and off[2] == off.max()
    # scaling in tau is linear
    assert np.allclose(logit_adjust_offsets(pri, 2.0), 2.0 * off, atol=1e-6)
    with pytest.raises(ValueError, match="K>=2"):
        logit_adjust_offsets(np.array([1.0]), 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        logit_adjust_offsets(np.array([0.5, -0.1]), 1.0)
    with pytest.raises(ValueError, match="sum must be > 0"):
        logit_adjust_offsets(np.zeros(3), 1.0)
    # absent class => floored, finite (never -inf)
    off0 = logit_adjust_offsets(np.array([1.0, 0.0, 1.0]), 1.0)
    assert np.isfinite(off0).all()


def test_trainer_offsets_helper_computes_priors_from_lstars():
    tl = _load_trainer()
    # toy L*: 3 maps, class areas 50% / 25% / 25% over classes {0,1,2}; classes 3,4 absent
    ls = np.zeros((4, 4), np.int64)
    ls[:2, :] = 0
    ls[2, :] = 1
    ls[3, :] = 2
    off, priors = tl._logit_adjust_offsets_np([ls, ls, ls], tau=1.0, n_classes=5)
    assert np.allclose(priors, [0.5, 0.25, 0.25, 0.0, 0.0])
    assert np.isfinite(off).all(), "absent classes must be prior-floored, never log(0)"
    assert np.isclose(off[0], np.log(0.5), atol=1e-6)
    with pytest.raises(ValueError, match="empty GT"):
        tl._logit_adjust_offsets_np([np.zeros((0, 0), np.int64)], tau=1.0)


def test_micro_batch_fail_closed_validator():
    tl = _load_trainer()
    tl._validate_logit_adjust_compat(0.0, 8)      # OFF composes with micro-batch
    tl._validate_logit_adjust_compat(1.0, 1)      # ON composes with the serial path
    with pytest.raises(ValueError, match="micro-batch-pairs"):
        tl._validate_logit_adjust_compat(1.0, 2)  # ON x batched twin => refuse


def test_logit_adjust_adapter_offsets_segnet_only_and_tau0_is_same_object():
    tl = _load_trainer()

    class _StubSeg:
        def __call__(self, f):
            return np.zeros((1, 2, 2, 5), np.float64) + 1.0

    class _StubInner:
        posenet = object()

        def __init__(self):
            self.segnet = _StubSeg()

    inner = _StubInner()
    off = np.array([-1.4603, -5.1321, -0.7025, -4.3894, -1.3697], np.float64)
    wrapped = tl._LogitAdjustSegAdapter(inner, off)
    out = wrapped.segnet(None)
    assert np.allclose(out, 1.0 + off[None, None, None, :])
    assert wrapped.posenet is inner.posenet, "pose path must be the SAME object (pass-through)"
    # tau=0 byte-identity is a SOURCE-LEVEL gate: the wrapper is only built under la_tau != 0.0
    # and the default keeps the ORIGINAL adapter object.
    assert "_loss_adapter = adapter" in _TRAINER_SRC
    assert "if la_tau != 0.0:" in _TRAINER_SRC
    assert "base_loss = make_loss_fn(\n        _loss_adapter," in _TRAINER_SRC
    # fail-closed validator is wired BEFORE the wrap (micro-batch guard cannot be skipped)
    assert "_validate_logit_adjust_compat(la_tau, int(getattr(args, \"micro_batch_pairs\", 1)))" \
        in _TRAINER_SRC


def test_constant_shift_invariance_documents_center_equivalence():
    # A GLOBAL constant on every logit changes neither softmax-CE nor the top1-top2 margin —
    # the documented equivalence between the un-centered train-time offsets here and the
    # mean-centered decode-time facet-3 offsets.
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((7, 5))
    y = rng.integers(0, 5, size=7)

    def ce(lg):
        lse = np.log(np.exp(lg).sum(-1))
        return (lse - lg[np.arange(7), y]).mean()

    assert np.isclose(ce(logits), ce(logits + 3.7), atol=1e-9)


def test_resume_divergence_guard_covers_logit_adjust():
    tl = _load_trainer()

    class _Args:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, k):  # unset attrs behave like argparse defaults via getattr fallbacks
            raise AttributeError(k)

    cfg = {"__cfg_logit_adjust_loss_tau": np.asarray(1.0)}
    div = tl._resume_lever_divergences(cfg, _Args())
    assert any("logit_adjust_loss_tau" in d for d in div), "dropping the lever on resume must flag"
    cfg0 = {"__cfg_logit_adjust_loss_tau": np.asarray(0.0)}
    assert tl._resume_lever_divergences(cfg0, _Args()) == []
    # sidecar persists the key (source-level)
    assert 'out["__cfg_logit_adjust_loss_tau"]' in _TRAINER_SRC


# ==========================================================================
# DSL levers (FEED-07b BUILD halves) — construction guards; composability/ledger
# membership auto-covered by test_feed07_dsl_wirein._NEW_LEVERS.
# ==========================================================================
def test_dsl_lever_guards():
    from tac.witness_dsl import curriculum_dsl as cd

    lv = cd.FinerBiasInit()
    assert lv.overrides == {"--finer-bias-init": True, "--finer-bias-k": 10.0}
    with pytest.raises(ValueError, match="k must be > 0"):
        cd.FinerBiasInit(k=0.0)
    la = cd.LogitAdjust()
    assert la.overrides == {"--logit-adjust-loss-tau": 1.0}
    with pytest.raises(ValueError, match="nonzero"):
        cd.LogitAdjust(tau=0.0)
    with pytest.raises(ValueError, match="nonzero"):
        cd.LogitAdjust(tau=float("nan"))
    # MarginFieldHead now HOLDS the facet-3 pair (default-off; store_true emitted True only)
    mfh = cd.MarginFieldHead()
    assert "--logit-adjust-per-class" not in mfh.overrides
    mfh3 = cd.MarginFieldHead(logit_adjust_per_class=True, logit_adjust_tau=0.5)
    assert mfh3.overrides["--logit-adjust-per-class"] is True
    assert mfh3.overrides["--logit-adjust-tau"] == 0.5


def test_dsl_levers_parse_through_real_argparse_composed():
    from tac.witness_dsl import curriculum_dsl as cd
    from tac.witness_dsl import lever_registry as LR

    prog = cd.BASELINE
    for name in ("FinerBiasInit", "LogitAdjust", "StepNativeActivation"):
        prog = prog.with_lever(LR.resolve_composable_lever(name))
    assert prog.validate() == []
    ap = cd.build_real_trainer_parser()
    argv: list[str] = []
    for f, v in prog.flag_dict().items():
        if v is True:
            argv.append(f)
        elif v is False:
            argv.append(f.replace("--", "--no-", 1))
        else:
            argv.extend([f, str(v)])
    ns = ap.parse_args(argv)
    assert ns.finer_bias_init is True and ns.finer_bias_k == 10.0
    assert ns.logit_adjust_loss_tau == 1.0


# ==========================================================================
# #220 AA compose-after-downsample (MLX)
# ==========================================================================
def test_aa_ss1_with_compose_fn_byte_identical_to_base_render():
    mx = pytest.importorskip("mlx.core")
    from train_witness_realized_through_R_mlx import render_through_R_mlx
    from tac.boundary_math.aa_sdf_observation_render import (
        build_render_coords,
        render_aa_through_R_mlx,
    )

    rh, rw = 12, 16

    class _StubWitness:
        def __call__(self, coord_feats, code_idx):
            rng = mx.random.uniform(shape=(coord_feats.shape[0], 3))
            return rng * 255.0

    def compose(rgb_nhwc, code_idx):
        return rgb_nhwc * 0.5 + 10.0  # any base-grid compose

    feats = mx.array(build_render_coords(rh, rw))
    mx.random.seed(3)
    ref = np.asarray(render_through_R_mlx(_StubWitness(), feats, 1, rh, rw, compose_fn=compose))
    mx.random.seed(3)
    got = np.asarray(render_aa_through_R_mlx(_StubWitness(), feats, 1, rh, rw, 1, compose_fn=compose))
    assert np.array_equal(got, ref), \
        "ss=1 AA render WITH a compose_fn must be byte-identical to render_through_R_mlx"


def test_aa_ss2_compose_fn_receives_base_grid_and_composes():
    mx = pytest.importorskip("mlx.core")
    from tac.boundary_math.aa_sdf_observation_render import (
        SEG_H,
        SEG_W,
        box_downsample_np,
        build_supersampled_coords,
        render_aa_through_R_mlx,
    )

    rh, rw, ss = 12, 16, 2
    seen: dict = {}

    class _CoordWitness:
        # deterministic function of the coords => the fine render is reproducible in numpy
        def __call__(self, coord_feats, code_idx):
            x = coord_feats[:, 0:1]
            y = coord_feats[:, 1:2]
            r = (x + 1.0) * 100.0
            g = (y + 1.0) * 80.0
            b = (x * y + 1.0) * 60.0
            return mx.concatenate([r, g, b], axis=-1)

    def compose(rgb_nhwc, code_idx):
        seen["shape"] = tuple(int(d) for d in rgb_nhwc.shape)
        m = mx.zeros_like(rgb_nhwc)  # base-grid (H,W)-shaped mask composes without broadcast error
        return rgb_nhwc * (1.0 - 0.25) + m
    feats_fine = mx.array(build_supersampled_coords(rh, rw, ss))
    out = np.asarray(render_aa_through_R_mlx(
        _CoordWitness(), feats_fine, 1, rh, rw, ss, compose_fn=compose))
    # (a) the #220 contract: compose_fn saw the BASE grid, not the fine grid
    assert seen["shape"] == (1, rh, rw, 3), f"compose must run at the base grid, saw {seen['shape']}"
    assert out.shape == (1, SEG_H, SEG_W, 3) and np.isfinite(out).all()
    # (b) equivalence: manual numpy pipeline (fine render -> box-down -> same compose) matches the
    #     pre-R composed tensor the MLX path produced (checked via the same roundtrip)
    coords = build_supersampled_coords(rh, rw, ss)
    fine = np.stack([(coords[:, 0] + 1.0) * 100.0,
                     (coords[:, 1] + 1.0) * 80.0,
                     (coords[:, 0] * coords[:, 1] + 1.0) * 60.0], axis=-1)
    base = box_downsample_np(fine.reshape(1, rh * ss, rw * ss, 3), ss) * 0.75
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )
    ref = np.asarray(apply_contest_faithful_roundtrip_nhwc(
        mx.array(base.astype(np.float32)), output_hw=(SEG_H, SEG_W), ste_round=True))
    assert np.allclose(out, ref, atol=1e-3), "AA ss=2 + base-grid compose must match the manual pipeline"
