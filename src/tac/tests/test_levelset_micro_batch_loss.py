"""Numerical-equivalence + correctness tests for the --micro-batch-pairs batched loss.

The batched loss (``tac.boundary_math.levelset_micro_batch_loss``) is the OPT-IN speed lever
for the LEVELSET witness trainer (the pointer-mover). The #1 correctness gate: the batched
grad over K pairs MUST equal the mean of the K per-pair grads within fp tolerance — a wrong
gradient corrupts training (NO-FAKE). Real EfficientNet-B2/FastViT scorer kernels are MEASURED
batch-dependent, so bit identity is not the training gate. These tests use a deliberately
batch-independent mock scorer to isolate the required functional loss/gradient parity; operator
policy admits the training-only lever only when that parity and measured end-to-end speedup both
pass. Exact byte-closed scoring authority is unchanged. The suite also checks that the extracted
per-pair base math matches the canonical importable ``make_loss_fn`` op-for-op.

# LOSS_CONVERGENCE_NOT_REQUIRED: this is a gradient-PARITY suite (batched twin vs canonical
# make_loss_fn, op-for-op, mock scorer by design) — a mock-scorer convergence assertion would
# be a toy; convergence of the canonical loss is owned by the real trainer runs (pf2x r78).
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
# Run the functional-parity matrix on MLX CPU with a batch-independent mock scorer, isolating the
# routed loss math from the real scorer's measured batch-dependent kernels. The dedicated fused-map
# test below explicitly enters a GPU context and verifies the actual Metal forwards and VJPs.
mx.set_default_device(mx.cpu)
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from tac.boundary_math.levelset_micro_batch_loss import (  # noqa: E402
    LeverConfig,
    batched_realized_loss,
    single_realized_loss,
)
from tac.local_acceleration import metal_micro_batch_v9_levers as _v9_kernels  # noqa: E402

# ---- import the trainer-local eikonal/nuclear helpers (module-level, cheap import ~0.2s) ----
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_TRAINER_PATH = os.path.join(_REPO, "experiments", "train_levelset_witness_realized_through_R_mlx.py")
_spec = importlib.util.spec_from_file_location("_lvl_trainer_for_test", _TRAINER_PATH)
_lvl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lvl)
_EIKONAL = _lvl._eikonal_length_mlx
_NUCLEAR = _lvl._nuclear_norm_smooth_mlx


# ===========================================================================
# Tiny witness + batch-independent mock scorer
# ===========================================================================
class _TinyWitness(nn.Module):
    def __init__(self, feat_dim, mod_dim, hidden, n_frames, n_classes=5, seed=0):
        super().__init__()
        self.n_hidden = 1
        self.hidden_dim = hidden
        self.in_proj = nn.Linear(feat_dim, hidden)
        self.film = nn.Linear(mod_dim, self.n_hidden * 2 * hidden)
        self.hidden = [nn.Linear(hidden, hidden)]
        self.out_sdf = nn.Linear(hidden, n_classes)
        self.out_tex = nn.Linear(hidden, 3)
        rng = np.random.default_rng(seed)
        self.code = mx.array(rng.standard_normal((n_frames, mod_dim)).astype(np.float32) * 0.5)

    def _trunk(self, cf, code_idx):
        h = nn.relu(self.in_proj(cf))
        film = mx.reshape(self.film(self.code[code_idx]), (self.n_hidden, 2, self.hidden_dim))
        for li, layer in enumerate(self.hidden):
            pre = layer(h) * (1.0 + film[li, 0]) + film[li, 1]
            h = nn.relu(pre)
        return h

    def sdf(self, cf, code_idx):
        return self.out_sdf(self._trunk(cf, code_idx))  # (n_px, n_classes)

    def __call__(self, cf, code_idx):
        h = self._trunk(cf, code_idx)
        return mx.sigmoid(self.out_tex(h)) * 255.0       # (n_px, 3)


class _MockAdapter:
    """Batch-independent mock SegNet/PoseNet (linear per-pixel / per-frame) -> the batching
    invariance we rely on (segnet(batch)[k] == segnet(batch[k:k+1])[0]) holds EXACTLY."""

    def __init__(self, pose_dims=6, seed=1):
        rng = np.random.default_rng(seed)
        self.seg_w = mx.array(rng.standard_normal((3, 5)).astype(np.float32))
        self.seg_b = mx.array(rng.standard_normal((5,)).astype(np.float32))
        self.pose_w = mx.array(rng.standard_normal((12, pose_dims)).astype(np.float32) * 0.01)

    def segnet(self, f):                 # f (B,H,W,3) -> (B,H,W,5)
        # ``f`` is 0..255 RGB. Feeding it RAW into an N(0,1) linear map yields logits spanning
        # ~[-107, +179], which SATURATES every downstream ``mx.softmax`` to an exact one-hot: the
        # class probabilities used by the V9 chroma/phase/temporal levers collapse to 0.0/1.0 and
        # any ``g1 - g0`` probability difference underflows to 0 in fp32. MEASURED 2026-08-01 on
        # the pre-fix fixture: ``max(softmax(logits)[..., 0:3]) == 5.0e-37`` and the temporal term
        # was IDENTICALLY ZERO, so its parity assertions (``|routed - oracle| / |oracle|``) passed
        # VACUOUSLY -- 0 == 0 -- and only the negative control noticed. The real scorer's logits
        # live at O(1..10) because SegNet normalizes its input; dividing by 255 restores that
        # regime so the probability-space levers are actually EXERCISED.
        return (f / 255.0) @ self.seg_w + self.seg_b

    def posenet(self, yuv):              # yuv (B,h2,w2,12) -> {"pose": (B, pose_dims)}
        return {"pose": mx.mean(yuv, axis=(1, 2)) @ self.pose_w}


def _render_fn(model, cf, code_idx, rh, rw):
    # deterministic render (no R): witness (n_px,3) -> (1,H,W,3). The equivalence we test is the
    # batched-scorer + per-pair-reduction identity, which is R-independent.
    return mx.reshape(model(cf, int(code_idx)), (1, rh, rw, 3))


def _build(K, seed=0):
    rh, rw = 8, 12
    n_px = rh * rw
    feat_dim, mod_dim, hidden = 7, 4, 6
    n_frames = 2 * K
    # DETERMINISM (added 2026-08-01). ``nn.Linear`` initializes from MLX's PROCESS-GLOBAL RNG, not
    # from the ``seed`` argument — only ``_TinyWitness.code`` and the numpy draws below were ever
    # seeded. So the trunk weights depended on how many MLX random draws happened EARLIER in the
    # process, i.e. on test ordering and on which ``-k`` filter was used. MEASURED 2026-08-01: the
    # temporal lever's share of the loss moved ~130x (6.5e-4 vs 8.6e-2) between a filtered run and
    # a full-suite run of the SAME test. That breaks the deterministic-reproducibility
    # non-negotiable ("all RNG from a single recorded seed") and makes every measured ratio in
    # this file order-dependent. Seeding the global stream here makes ``_build`` a pure function
    # of ``(K, seed)`` again.
    mx.random.seed(seed)
    rng = np.random.default_rng(seed + 7)
    model = _TinyWitness(feat_dim, mod_dim, hidden, n_frames, seed=seed)
    adapter = _MockAdapter(pose_dims=6, seed=seed + 1)
    cf = mx.array(rng.standard_normal((n_px, feat_dim)).astype(np.float32))
    cf_list = [cf for _ in range(K)]                     # shared feats (self-orient off case)
    c0_list = [2 * p + 0 for p in range(K)]
    c1_list = [2 * p + 1 for p in range(K)]
    oh_list, mg_list, pt_list = [], [], []
    for _k in range(K):
        arg = rng.integers(0, 5, size=(rh, rw))
        oh = np.eye(5, dtype=np.float32)[arg].reshape(1, rh, rw, 5)
        mg = (rng.random((1, rh, rw)).astype(np.float32) * 0.5)
        oh_list.append(mx.array(oh))
        mg_list.append(mx.array(mg))
        pt_list.append(mx.array(rng.standard_normal(6).astype(np.float32) * 0.01))
    mx.eval(model.parameters(), cf, *oh_list, *mg_list, *pt_list)
    return {
        "model": model,
        "adapter": adapter,
        "rh": rh,
        "rw": rw,
        "cf_list": cf_list,
        "c0_list": c0_list,
        "c1_list": c1_list,
        "oh_list": oh_list,
        "mg_list": mg_list,
        "pt_list": pt_list,
    }


def _base_lc(**over):
    lc = LeverConfig(seg_loss_default="ce", tau_use=0.3, l7_thr_use=0.42, l7_mult=4.0,
                     score_domain=True, pose_eps=1e-2,
                     eikonal_length=_EIKONAL, nuclear_norm_smooth=_NUCLEAR)
    for k, v in over.items():
        setattr(lc, k, v)
    return lc


_FP32_EPS = float(np.finfo(np.float32).eps)          # 1.1920929e-07
_FP32_U = _FP32_EPS / 2.0                            # half-ulp, 2^-24
# Slack over one ulp-of-the-field, for the maps whose fused-vs-reference disagreement really is
# reordered-summation rounding: MEASURED 2026-08-01, chroma value/grads and temporal value/grads
# all sit at <= 1.7 ulp of their field. 64 leaves ~37x headroom there while still refusing a real
# indexing/dispatch defect by ~5 orders (such a defect misplaces whole rows or tiles, i.e. an
# error of order the FIELD SCALE itself => ratio ~1/eps32 ~ 8.4e6, not ~2).
#
# The phase grad0 field is NOT in that regime and deliberately does NOT use this bound; see
# _assert_phase_signed_grad_parity. MEASURED 2026-08-01 over 24 (shape x distribution x seed)
# combinations, its field-ulp ratio ranged 0.53 .. 58.0 — a 110x spread that reaches 91% of this
# 64.0 threshold. A threshold calibrated on ONE seed's draw of a quantity with that spread is a
# hand-picked atol wearing a derivation, and would flip red on an innocent fixture change.
_FP32_FIELD_SLACK = 64.0

# Multiplier over the DERIVED per-pixel phase-VJP budget below. MEASURED 2026-08-01: the worst
# observed err/budget over the same 24 combinations was 0.53 (range 0.25..0.53, a 2.1x spread —
# versus 110x for the field-scale metric). 4.0 leaves ~7.5x headroom on a bound that is a
# FUNCTION OF THE PRIMALS, so it travels with shape, seed and distribution instead of being
# recalibrated whenever the fixture moves.
_PHASE_CANCELLATION_SLACK = 4.0
# The eps these tests exercise: the ``phase_squared_map`` signature default, since the fused/
# reference calls below pass no eps. Read from the signature so it cannot silently drift.
_PHASE_KERNEL_DEFAULT_EPS = float(
    inspect.signature(_v9_kernels.phase_squared_map).parameters["eps"].default
)


def _assert_phase_signed_grad_parity(label, fused, reference, signed, direction, ref_map, cot,
                                     eps):
    """Phase dL/dsigned parity at a bound DERIVED from the reference's cancellation floor.

    The two sides are not equally accurate and the test must not pretend otherwise. MEASURED
    2026-08-01 at (B,H,W)=(2,384,512), both against an independent float64 evaluation:
    the FUSED Metal kernel is 1.60 ulp-of-field from truth; the MLX REFERENCE is 25.6 ulp — the
    kernel is ~16x MORE accurate, and the disagreement is almost entirely the reference's error.
    (Ruled out as causes, both MEASURED: the MLX GPU stream — reference GPU vs reference CPU is
    bit-identical, 0/393216 elements differing — and any matmul precision floor, since this VJP
    contains no matmul.)

    MECHANISM, derived then confirmed: MLX forms ``dL/dm`` for ``tie = m/den`` as
    ``g/den - g*tie/den``, a difference of two O(g/den) terms that cancels catastrophically as
    ``tie -> 1`` (i.e. wherever the partner margin is small next to ``m``). The kernel instead
    evaluates the algebraically-identical closed form ``(partner + eps)/den**2``, which has no
    subtraction and therefore no cancellation. In float64 the two routes agree to 2.9e-15 and
    both match a central finite difference of the forward map to 3.3e-9, so neither is "the
    kernel's own answer" — they are the same number, and only the fp32 conditioning differs.
    Binned by ``1 - tie``: for pixels with ``1-tie`` in [1e-6, 1e-4) the autodiff route's error is
    3.6e-6 while the kernel route's is 5.7e-10, a 6300x gap that closes as ``tie`` leaves 1.

    So the honest tolerance is the REFERENCE's error floor, ~u*|g|/den per pixel (absolute, set by
    the cancellation) plus the kernel's own ~u*|g|*A relative term, propagated through the same
    three-term stencil the VJP uses. DO NOT "fix" the kernel toward the reference: that would make
    the shipped gradient strictly worse.
    """
    f = np.asarray(fused, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    assert f.shape == r.shape, (label, f.shape, r.shape)
    s = np.asarray(signed, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    m = np.maximum(s, 0.0)
    right = np.pad(m[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
    down = np.pad(m[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
    partner = np.where(d < 0.5, right, down)
    den = m + partner + float(eps)
    tie = m / den
    # |chain-rule multiplier| = |dL/dtie| at each pixel.
    g = np.abs(2.0 * (tie - np.asarray(ref_map, dtype=np.float64))
               * np.asarray(cot, dtype=np.float64))
    pos = s > 0.0
    # 8 roundings is the op count of each route (the two divides, the multiply chain, the
    # subtraction) rounded up; it is an op-count, not a fitted constant.
    own = np.where(pos, _FP32_U * g * (8.0 / den + 8.0 * (partner + float(eps)) / (den * den)),
                   0.0)
    pred = np.where(pos, _FP32_U * g * (8.0 / den + 8.0 * m / (den * den)), 0.0)
    budget = own.copy()
    budget[:, :, 1:] += np.where(d[:, :, :-1] < 0.5, pred[:, :, :-1], 0.0)
    budget[:, 1:, :] += np.where(~(d[:, :-1, :] < 0.5), pred[:, :-1, :], 0.0)
    budget = np.where(pos, budget, 0.0) * _PHASE_CANCELLATION_SLACK
    err = np.abs(f - r)
    over = err > budget
    if over.any():
        # A structurally-zero budget (signed <= 0, so the pixel contributes no gradient) with a
        # nonzero error is an infinite ratio; report it as such rather than dividing by `tiny`.
        pos_budget = budget[over] > 0.0
        ratio = (float((err[over][pos_budget] / budget[over][pos_budget]).max())
                 if pos_budget.any() else float("inf"))
        raise AssertionError(
            f"{label}: {int(over.sum())} pixel(s) exceed the derived fp32 cancellation budget; "
            f"worst err {float(err[over].max()):.6g} vs budget {float(budget[over].max()):.6g} "
            f"(worst ratio {ratio:.4g}). This budget scales with the primals, so a breach means "
            "an indexing/dispatch defect, not a fixture change."
        )
    assert float(np.abs(r).max()) > 0.0, (
        f"{label}: reference dL/dsigned is identically zero — the gradient under test is absent, "
        "so the comparison above is vacuous."
    )


def _assert_fp32_field_parity(label, fused, reference, *, expect_nonzero):
    """Fused-kernel vs reference parity at a FIELD-SCALE fp32 bound.

    A fixed ``atol`` is the wrong instrument for these maps: they span many decades (the phase VJP
    field runs 0 .. 3.3e3, the chroma map 0 .. 2e5), so a single absolute floor is simultaneously
    far too loose for the large elements and unsatisfiable for the small ones. Reordered fp32
    arithmetic perturbs a field by ~ulp-of-the-LARGEST-term, not ulp-of-each-element, so the
    honest bound scales with ``max|reference|``.

    ``expect_nonzero`` is REQUIRED and is the caller's DECLARATION, never an inference from the
    data. That distinction is the whole point: an all-zero field and a correctly-computed field
    are both "no assertion failure", so a helper that INFERS the zero branch cannot tell a static
    provider (phase ``direction``, temporal ``class_mask`` — structurally zero, and asserting
    EXACT zero on them is a real check) from a theta-bearing gradient that COLLAPSED to zero on
    both sides because the term under test went missing. The second case is
    ``[[vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801]]``: an instrument
    that examined nothing emits the same symbol as one that examined everything cleanly. Declaring
    the expectation up front makes the collapse a FAILURE instead of a silent pass.
    """
    f = np.asarray(fused, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    assert f.shape == r.shape, (label, f.shape, r.shape)
    scale = float(np.abs(r).max())
    if expect_nonzero:
        assert scale > 0.0, (
            f"{label}: caller declared this field theta-bearing (expect_nonzero=True) but the "
            "reference is IDENTICALLY ZERO. A scale-relative parity check would then compare "
            "0 to 0 and pass vacuously — the term under test is absent, not correct."
        )
    else:
        assert scale == 0.0, (
            f"{label}: caller declared this field a structurally-zero static provider "
            f"(expect_nonzero=False) but the reference is nonzero (maxabs {scale:.6g}); "
            "re-derive which primals are theta-bearing rather than relaxing this."
        )
        stray = float(np.abs(f).max())
        assert stray == 0.0, (
            f"{label}: reference gradient is structurally zero (static provider) but the fused "
            f"kernel wrote a nonzero field (maxabs {stray:.6g})"
        )
        return
    atol = _FP32_FIELD_SLACK * _FP32_EPS * scale
    maxabs = float(np.abs(f - r).max())
    assert maxabs <= atol, (
        f"{label}: fused-vs-reference maxabs {maxabs:.6g} exceeds the fp32 field bound "
        f"{atol:.6g} (field max {scale:.6g}, ratio {maxabs / (_FP32_EPS * scale):.3g} ulp). "
        "A ratio of order 1e6 means an indexing/dispatch defect, not rounding."
    )


def _max_rel_grad_err(g_batched, g_mean):
    """Global relative L2 error between two grad pytrees — the standard gradient-check metric
    (``||g_b - g_m||_2 / ||g_m||_2``). This is the honest fp32 measure: a per-leaf ratio with a
    tiny denom floor blows up to ~1e-3 on near-zero grad leaves (catastrophic cancellation) even
    when the two gradients agree to machine precision globally; the global L2 rel err is ~1e-7
    here (batched-grad == mean-of-per-pair-grad to fp32 precision)."""
    fb = dict(tree_flatten(g_batched))
    fm = dict(tree_flatten(g_mean))
    assert set(fb.keys()) == set(fm.keys()), (set(fb) ^ set(fm))
    diff = np.concatenate([(np.asarray(fb[k], np.float64) - np.asarray(fm[k], np.float64)).ravel()
                           for k in fb])
    ref = np.concatenate([np.asarray(fm[k], np.float64).ravel() for k in fm])
    return float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12))


def _batched_and_meanpairs(env, lc, *, w_seg=100.0, w_pose=1.0, hinge=4.0, mtgt=0.5,
                           seg_form=None, eik_w=0.0, len_w=0.0, render_fn_wa=None,
                           render_fn=_render_fn):
    """Return (loss_batched, grad_batched, mean_loss_pairs, mean_grad_pairs). ``render_fn_wa``
    routes the island levers (amplify/persistence) through a distinct (seed-excluded) forward in
    BOTH the batched and the mean-of-per-pair paths — the equivalence gate for the wa leg."""
    model = env["model"]

    def _bfn(m):
        return batched_realized_loss(
            m, env["adapter"], render_fn, env["rh"], env["rw"],
            env["cf_list"], env["c0_list"], env["c1_list"],
            env["oh_list"], env["mg_list"], env["pt_list"],
            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc, render_fn_wa=render_fn_wa)

    lb, gb = nn.value_and_grad(model, _bfn)(model)
    mx.eval(lb, gb)

    K = len(env["c1_list"])
    from mlx.utils import tree_map
    accum = None
    lsum = 0.0

    def _sfn(m, k):
        return single_realized_loss(
            m, env["adapter"], render_fn, env["rh"], env["rw"],
            env["cf_list"][k], env["c0_list"][k], env["c1_list"][k],
            env["oh_list"][k], env["mg_list"][k], env["pt_list"][k],
            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc, render_fn_wa=render_fn_wa)

    for k in range(K):
        ls, gs = nn.value_and_grad(model, _sfn)(model, k)
        mx.eval(ls, gs)
        lsum += float(ls)
        accum = gs if accum is None else tree_map(lambda a, b: a + b, accum, gs)
        mx.eval(accum)
    mean_grad = tree_map(lambda g, c=float(K): g / c, accum)
    mx.eval(mean_grad)
    return float(lb), gb, lsum / K, mean_grad


# ===========================================================================
# TEST 1 — batched grad == mean of per-pair grads (the #1 gate), several regimes
# ===========================================================================
@pytest.mark.parametrize("seg_form", ["ce", "tau_softplus", "l7_softplus", "margin_hinge"])
@pytest.mark.parametrize("K", [2, 4])
def test_batched_grad_equals_mean_of_per_pair_grads_base(seg_form, K):
    env = _build(K, seed=K)
    lc = _base_lc(seg_loss_default=seg_form)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form=seg_form, eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (seg_form, K, lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"{seg_form} K={K} max rel grad err {err:.2e} >= 1e-4"


# ===========================================================================
# TEST 2 — with weighted-mean levers (msal +UNIWARD, lane-edge, mfh) still matches.
#          (global sum(x*w)/sum(w) over K would BREAK this; per-pair mean must be used.)
# ===========================================================================
@pytest.mark.parametrize("K", [3, 4])
def test_batched_grad_matches_with_weighted_mean_levers(K):
    env = _build(K, seed=100 + K)
    lc = _base_lc(
        lane_w=0.7, lane_cls=1, lane_tgt=0.5, lane_gate={"on": True},
        msal_w=0.9, msal_tau=0.3, msal_tgt=0.5, msal_uni=True, msal_uni_beta=1.0,
        msal_gate={"on": True},
        mfh_w=0.4, mfh_target_mx=mx.array(np.full((1, 1, 1, 5), 0.3, np.float32)),
    )
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="tau_softplus", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"weighted-mean levers K={K} max rel grad err {err:.2e}"


# ===========================================================================
# TEST 3 — with a per-MODEL once-term (code-nuclear) still matches (added once in both paths).
# ===========================================================================
@pytest.mark.parametrize("K", [2, 4])
def test_batched_grad_matches_with_once_term(K):
    env = _build(K, seed=200 + K)
    lc = _base_lc(code_nuc_w=0.05, code_nuc_eps=1e-3, code_nuc_iters=15)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"once-term K={K} max rel grad err {err:.2e}"


# ===========================================================================
# TEST 4 — the score-domain pose sqrt is per-pair (NOT sqrt-of-mean): the batched loss must
#          MATCH mean-of-per-pair (this is why we do NOT reuse make_loss_fn_batch verbatim).
# ===========================================================================
def test_pose_sqrt_is_per_pair_not_sqrt_of_mean():
    K = 4
    env = _build(K, seed=321)
    lc = _base_lc(score_domain=True)
    # pose-heavy so the sqrt nonlinearity dominates.
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, w_seg=0.0, w_pose=10.0, seg_form="ce",
                                            eik_w=0.0, len_w=0.0)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"pose sqrt per-pair max rel grad err {err:.2e}"


# ===========================================================================
# TEST 5 — the extracted per-pair BASE math (levers/eik off) matches the canonical importable
#          make_loss_fn op-for-op (validates the seg-form + pose replication vs the real loss).
# ===========================================================================
@pytest.mark.parametrize("seg_form", ["ce", "tau_softplus", "l7_softplus", "margin_hinge"])
@pytest.mark.parametrize("score_domain", [True, False])
def test_single_base_matches_canonical_make_loss_fn(seg_form, score_domain):
    from experiments.train_witness_realized_through_R_mlx import make_loss_fn

    env = _build(1, seed=55)
    lc = _base_lc(seg_loss_default=seg_form, score_domain=score_domain)
    w_seg, w_pose, hinge, mtgt = 100.0, 1.0, 4.0, 0.5

    def _sfn(m):
        return single_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
            env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
            w_seg, w_pose, hinge, mtgt, seg_form, 0.0, 0.0, lc)

    ls, gs = nn.value_and_grad(env["model"], _sfn)(env["model"])
    mx.eval(ls, gs)

    canonical = make_loss_fn(env["adapter"], env["rh"], env["rw"], score_domain=score_domain,
                             pose_eps=lc.pose_eps, seg_loss=seg_form, tau_softplus_tau=lc.tau_use,
                             l7_mult=lc.l7_mult, l7_threshold=lc.l7_thr_use, render_fn=_render_fn)

    def _cfn(m):
        return canonical(m, env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
                         env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
                         w_seg, w_pose, hinge, mtgt)

    lc_val, gc = nn.value_and_grad(env["model"], _cfn)(env["model"])
    mx.eval(lc_val, gc)
    assert abs(float(ls) - float(lc_val)) / (abs(float(lc_val)) + 1e-6) < 1e-4, (seg_form, score_domain, float(ls), float(lc_val))
    err = _max_rel_grad_err(gs, gc)
    assert err < 1e-4, f"{seg_form} score_domain={score_domain} base-vs-canonical grad err {err:.2e}"


# ===========================================================================
# MB-TWIN #313 — newly-ROUTED legs (were fail-closed): focal / boundary-distance /
# eik-stab (ViscoReg + StEik) / witness-alone-island routing. Each pins the SAME #1 gate:
# batched grad == mean-of-per-pair grad within fp tol (MLX-CPU ~1e-7). The fail-closed
# remainder (msal-reachability, spike-reweight) is a trainer-level gate, tested separately below.
# ===========================================================================
_FOCAL = _lvl.focal_pixel_weight_mlx
_BDTERM = _lvl.boundary_distance_term_mlx
_BDBAND = _lvl.boundary_distance_band_map
_VISCO = _lvl._eikonal_visco_mlx
_STEIK = _lvl._eikonal_steik_mlx


def _island_weight_prov(env, val=1.0):
    """dict[pair -> (1,H,W) island weight] keyed by pi == c1//2 (same key the levers use)."""
    prov = {}
    for k in range(len(env["c1_list"])):
        pi = int(env["c1_list"][k]) // 2
        prov[pi] = mx.array(np.full((1, env["rh"], env["rw"]), val, np.float32))
    return prov


def _bd_band_prov(env, band_px=2.0):
    """dict[pair -> (1,H,W) GT-boundary band map] built from each pair's GT argmax."""
    prov = {}
    for k in range(len(env["c1_list"])):
        pi = int(env["c1_list"][k]) // 2
        arg = np.asarray(env["oh_list"][k])[0].argmax(-1).astype(np.int64)  # (H,W)
        prov[pi] = mx.array(_BDBAND(arg, band_px=band_px)[None].astype(np.float32))
    return prov


# ---- LEG 1: FOCAL ---------------------------------------------------------
@pytest.mark.parametrize("seg_form", ["ce", "tau_softplus", "l7_softplus", "margin_hinge"])
@pytest.mark.parametrize("K", [2, 3])
def test_leg_focal_batched_grad_equals_mean_of_pairs(seg_form, K):
    env = _build(K, seed=400 + K)
    lc = _base_lc(seg_loss_default=seg_form, focal_gamma=2.0, focal_pixel_weight=_FOCAL)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form=seg_form, eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (seg_form, K, lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"focal {seg_form} K={K} grad err {err:.2e}"


def test_leg_focal_single_matches_canonical_make_loss_fn_with_gamma():
    """The batched twin's focal must be BIT-consistent with the canonical make_loss_fn focal path
    (same focal_pixel_weight callable, same multiplicative fold) — the one-math-one-backend gate."""
    from experiments.train_witness_realized_through_R_mlx import make_loss_fn

    env = _build(1, seed=411)
    gamma = 2.5
    lc = _base_lc(seg_loss_default="ce", focal_gamma=gamma, focal_pixel_weight=_FOCAL)
    w_seg, w_pose, hinge, mtgt = 100.0, 1.0, 4.0, 0.5

    def _sfn(m):
        return single_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
            env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
            w_seg, w_pose, hinge, mtgt, "ce", 0.0, 0.0, lc)

    ls, gs = nn.value_and_grad(env["model"], _sfn)(env["model"])
    mx.eval(ls, gs)
    canonical = make_loss_fn(env["adapter"], env["rh"], env["rw"], score_domain=True,
                             pose_eps=lc.pose_eps, seg_loss="ce", tau_softplus_tau=lc.tau_use,
                             l7_mult=lc.l7_mult, l7_threshold=lc.l7_thr_use, render_fn=_render_fn,
                             focal_gamma=gamma)

    def _cfn(m):
        return canonical(m, env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
                         env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
                         w_seg, w_pose, hinge, mtgt)

    lcv, gc = nn.value_and_grad(env["model"], _cfn)(env["model"])
    mx.eval(lcv, gc)
    assert abs(float(ls) - float(lcv)) / (abs(float(lcv)) + 1e-6) < 1e-4, (float(ls), float(lcv))
    assert _max_rel_grad_err(gs, gc) < 1e-4


def test_leg_focal_actually_changes_the_loss():
    """NO-FAKE: focal>0 must MOVE the loss vs focal=0 (else it's a silent no-op)."""
    env = _build(2, seed=412)
    l_off = _batched_and_meanpairs(env, _base_lc(seg_loss_default="ce"), seg_form="ce")[0]
    l_on = _batched_and_meanpairs(env, _base_lc(seg_loss_default="ce", focal_gamma=2.0,
                                                focal_pixel_weight=_FOCAL), seg_form="ce")[0]
    assert abs(l_on - l_off) > 1e-6, (l_on, l_off)


# ---- LEG 2: BOUNDARY-DISTANCE --------------------------------------------
@pytest.mark.parametrize("K", [2, 4])
def test_leg_boundary_distance_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=500 + K)
    lc = _base_lc(bd_w=0.3, bd_band_prov=_bd_band_prov(env), boundary_distance_term=_BDTERM)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


def test_leg_boundary_distance_actually_changes_the_loss():
    env = _build(2, seed=511)
    l_off = _batched_and_meanpairs(env, _base_lc(), seg_form="ce")[0]
    l_on = _batched_and_meanpairs(env, _base_lc(bd_w=0.3, bd_band_prov=_bd_band_prov(env),
                                                boundary_distance_term=_BDTERM), seg_form="ce")[0]
    assert abs(l_on - l_off) > 1e-6, (l_on, l_off)


# ---- LEG 3: EIK-STAB (ViscoReg + StEik) ----------------------------------
@pytest.mark.parametrize("K", [2, 3])
def test_leg_eik_viscosity_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=600 + K)
    lc = _base_lc(eik_stab={"visco_eps": 0.05, "steik_w": 0.0}, eikonal_visco=_VISCO,
                  eikonal_steik=_STEIK)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


@pytest.mark.parametrize("K", [2, 3])
def test_leg_eik_steik_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=650 + K)
    lc = _base_lc(eik_stab={"visco_eps": 0.0, "steik_w": 0.02}, eikonal_visco=_VISCO,
                  eikonal_steik=_STEIK)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


def test_leg_eik_viscosity_replaces_residual_not_double_counts():
    """ViscoReg REPLACES the eikonal residual (not additive): visco-on differs from visco-off. The
    eikonal term is ISOLATED (w_seg=w_pose=len_w=0, eik_w=1) so its magnitude is not swamped by the
    ~1e4 seg/pose loss (fp32 resolution)."""
    env = _build(2, seed=611)
    l_legacy = _batched_and_meanpairs(env, _base_lc(), seg_form="ce",
                                      w_seg=0.0, w_pose=0.0, eik_w=1.0, len_w=0.0)[0]
    l_visco = _batched_and_meanpairs(
        env, _base_lc(eik_stab={"visco_eps": 0.05, "steik_w": 0.0}, eikonal_visco=_VISCO,
                      eikonal_steik=_STEIK), seg_form="ce",
        w_seg=0.0, w_pose=0.0, eik_w=1.0, len_w=0.0)[0]
    assert abs(l_visco - l_legacy) > 1e-6, (l_visco, l_legacy)
    # eik_w=0 kills BOTH the legacy residual and the visco replacement -> visco branch rides eik_w
    # (replaces the SAME term) -> with only the length term left, visco-on == visco-off.
    l_visco_noeik = _batched_and_meanpairs(
        env, _base_lc(eik_stab={"visco_eps": 0.05, "steik_w": 0.0}, eikonal_visco=_VISCO,
                      eikonal_steik=_STEIK), seg_form="ce",
        w_seg=0.0, w_pose=0.0, eik_w=0.0, len_w=1.0)[0]
    l_legacy_noeik = _batched_and_meanpairs(env, _base_lc(), seg_form="ce",
                                            w_seg=0.0, w_pose=0.0, eik_w=0.0, len_w=1.0)[0]
    assert abs(l_visco_noeik - l_legacy_noeik) / (abs(l_legacy_noeik) + 1e-9) < 1e-5


# ---- LEG 4: WITNESS-ALONE ISLAND ROUTING ---------------------------------
def _render_fn_wa(model, cf, code_idx, rh, rw):
    # DISTINCT (seed-excluded analog) render: a genuine perturbation so sl_wa != sl and the
    # routing is OBSERVABLE. Both batched + single use this SAME fn -> equivalence must hold.
    return mx.reshape(model(cf, int(code_idx)) * 0.85 + 6.0, (1, rh, rw, 3))


@pytest.mark.parametrize("K", [2, 3])
def test_leg_wa_island_amplify_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=700 + K)
    lc = _base_lc(wa_island=True, amplify_w=0.6, amplify_mtgt=1.0, amplify_form="hinge",
                  island_weight_mx=_island_weight_prov(env))
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3,
                                            render_fn_wa=_render_fn_wa)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


@pytest.mark.parametrize("K", [2, 3])
def test_leg_wa_island_persist_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=750 + K)
    lc = _base_lc(wa_island=True, persist_gate={"w": 0.4}, persist_classes=(1, 3),
                  persist_cldice_iters=3, persist_recall_w=1.0)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3,
                                            render_fn_wa=_render_fn_wa)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


def test_leg_wa_island_routing_is_real_not_aliased():
    """NO-FAKE: with wa_island ON and a DISTINCT render_fn_wa, the amplify term must read the
    seed-excluded forward -> loss differs from the aliased (render_fn_wa=None) case. If the two
    were equal, the wa routing would be a silent no-op."""
    env = _build(2, seed=711)
    lc = _base_lc(wa_island=True, amplify_w=0.6, amplify_mtgt=1.0, amplify_form="hinge",
                  island_weight_mx=_island_weight_prov(env))
    l_wa = _batched_and_meanpairs(env, lc, seg_form="ce", render_fn_wa=_render_fn_wa)[0]
    l_aliased = _batched_and_meanpairs(env, lc, seg_form="ce", render_fn_wa=None)[0]
    assert abs(l_wa - l_aliased) > 1e-6, (l_wa, l_aliased)


def test_leg_wa_off_is_byte_identical_regardless_of_render_fn_wa():
    """When wa_island is False, passing render_fn_wa must NOT change anything (no 2nd forward):
    the island levers alias the composed forward -> byte-identical to render_fn_wa=None."""
    env = _build(2, seed=712)
    lc = _base_lc(wa_island=False, amplify_w=0.6, amplify_mtgt=1.0, amplify_form="hinge",
                  island_weight_mx=_island_weight_prov(env))
    l_passed = _batched_and_meanpairs(env, lc, seg_form="ce", render_fn_wa=_render_fn_wa)[0]
    l_none = _batched_and_meanpairs(env, lc, seg_form="ce", render_fn_wa=None)[0]
    assert abs(l_passed - l_none) < 1e-9, (l_passed, l_none)


def test_leg_wa_no_island_lever_no_second_forward():
    """wa_island True but NO island lever engaged -> _wa_route_active False -> render_fn_wa ignored
    (base + surgical levers never read a wa forward) -> byte-identical to wa off."""
    from tac.boundary_math.levelset_micro_batch_loss import _wa_route_active
    lc = _base_lc(wa_island=True)  # no amplify / persist
    assert _wa_route_active(lc, _render_fn_wa) is False


# ---- COMBINED: all four legs stacked still equivalent ---------------------
def test_all_four_new_legs_stacked_batched_grad_equals_mean_of_pairs():
    K = 3
    env = _build(K, seed=800)
    lc = _base_lc(
        seg_loss_default="ce", focal_gamma=2.0, focal_pixel_weight=_FOCAL,
        bd_w=0.3, bd_band_prov=_bd_band_prov(env), boundary_distance_term=_BDTERM,
        eik_stab={"visco_eps": 0.05, "steik_w": 0.02}, eikonal_visco=_VISCO, eikonal_steik=_STEIK,
        wa_island=True, amplify_w=0.6, amplify_mtgt=1.0, amplify_form="hinge",
        island_weight_mx=_island_weight_prov(env),
        persist_gate={"w": 0.3}, persist_classes=(1, 3), persist_cldice_iters=3,
    )
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3,
                                            render_fn_wa=_render_fn_wa)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"all-four-legs grad err {err:.2e}"


# ===========================================================================
# #D15 — LOGIT-ADJUST (--logit-adjust-loss-tau) + UNIFY-τ (--seg-form-unify-tau) routing.
# The two v7-active seg-form legs the twin previously fail-closed on. Each pins the SAME #1 gate
# (batched grad == mean-of-per-pair grad within fp tol) PLUS a canonical-match vs make_loss_fn so
# the base-form-adjusted / levers-raw split matches the SERIAL trainer semantics op-for-op.
# ===========================================================================
_LA_ADAPTER = _lvl._LogitAdjustSegAdapter
_SEG_UNIFY = _lvl._seg_unify_tau_perpixel
# A fixed nonzero per-class offset (Menon tau*log(prior) shape; the specific values do not matter
# to the routing identity, only that it is a (5,) constant broadcast over (K,H,W,5)).
_LA_OFFSET_NP = np.array([-1.46, -5.13, -0.70, -4.39, -1.36], np.float32)


# ---- LEG 5: LOGIT-ADJUST --------------------------------------------------
@pytest.mark.parametrize("seg_form", ["ce", "tau_softplus", "l7_softplus", "margin_hinge"])
@pytest.mark.parametrize("K", [2, 3])
def test_leg_logit_adjust_batched_grad_equals_mean_of_pairs(seg_form, K):
    env = _build(K, seed=900 + K)
    lc = _base_lc(seg_loss_default=seg_form, logit_adjust_offset=mx.array(_LA_OFFSET_NP))
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form=seg_form, eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (seg_form, K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4, f"logit-adjust {seg_form} K={K}"


def test_leg_logit_adjust_actually_changes_the_loss():
    """NO-FAKE: a nonzero offset must MOVE the base seg-form loss vs offset=None."""
    env = _build(2, seed=911)
    l_off = _batched_and_meanpairs(env, _base_lc(seg_loss_default="ce"), seg_form="ce")[0]
    l_on = _batched_and_meanpairs(
        env, _base_lc(seg_loss_default="ce", logit_adjust_offset=mx.array(_LA_OFFSET_NP)),
        seg_form="ce")[0]
    assert abs(l_on - l_off) > 1e-6, (l_on, l_off)


@pytest.mark.parametrize("seg_form", ["ce", "tau_softplus", "margin_hinge"])
def test_leg_logit_adjust_single_matches_canonical_wrapped_adapter(seg_form):
    """The strongest NO-FAKE gate: the twin (RAW adapter + offset on the BASE seg-form only) must
    equal the SERIAL make_loss_fn given the WRAPPED _LogitAdjustSegAdapter (levers off, so the base
    form is the whole seg loss and both apply the SAME per-class offset). Proves the base-adjusted /
    levers-raw split matches the serial semantics op-for-op."""
    from experiments.train_witness_realized_through_R_mlx import make_loss_fn

    env = _build(1, seed=920)
    w_seg, w_pose, hinge, mtgt = 100.0, 1.0, 4.0, 0.5
    lc = _base_lc(seg_loss_default=seg_form, logit_adjust_offset=mx.array(_LA_OFFSET_NP))

    def _sfn(m):
        return single_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
            env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
            w_seg, w_pose, hinge, mtgt, seg_form, 0.0, 0.0, lc)

    ls, gs = nn.value_and_grad(env["model"], _sfn)(env["model"])
    mx.eval(ls, gs)

    wrapped = _LA_ADAPTER(env["adapter"], mx.array(_LA_OFFSET_NP))
    canonical = make_loss_fn(wrapped, env["rh"], env["rw"], score_domain=True,
                             pose_eps=lc.pose_eps, seg_loss=seg_form, tau_softplus_tau=lc.tau_use,
                             l7_mult=lc.l7_mult, l7_threshold=lc.l7_thr_use, render_fn=_render_fn)

    def _cfn(m):
        return canonical(m, env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
                         env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
                         w_seg, w_pose, hinge, mtgt)

    lcv, gc = nn.value_and_grad(env["model"], _cfn)(env["model"])
    mx.eval(lcv, gc)
    assert abs(float(ls) - float(lcv)) / (abs(float(lcv)) + 1e-6) < 1e-4, (seg_form, float(ls), float(lcv))
    assert _max_rel_grad_err(gs, gc) < 1e-4, f"logit-adjust {seg_form} base-vs-canonical"


def test_leg_logit_adjust_offset_none_is_byte_identical():
    """offset None must be byte-identical to the pre-#D15 path (sl_base aliases sl)."""
    env = _build(2, seed=921)
    l_a = _batched_and_meanpairs(env, _base_lc(seg_loss_default="ce", logit_adjust_offset=None),
                                 seg_form="ce")[0]
    l_b = _batched_and_meanpairs(env, _base_lc(seg_loss_default="ce"), seg_form="ce")[0]
    assert abs(l_a - l_b) < 1e-12, (l_a, l_b)


def test_leg_logit_adjust_composes_with_wa_island_ladder():
    """LADDER composition: logit-adjust (base seg-form) + the #224 island levers (amplify/persist,
    which read the WITNESS-ALONE RAW forward) must STILL satisfy batched == mean-of-pairs. This is
    the micro-batch x seed-islands x LADDER-mask stack the seal flagged as the risky composition —
    the base form adjusted, the island levers raw, both equivalent under batching."""
    K = 3
    env = _build(K, seed=930)
    lc = _base_lc(
        seg_loss_default="ce", logit_adjust_offset=mx.array(_LA_OFFSET_NP),
        wa_island=True, amplify_w=0.6, amplify_mtgt=1.0, amplify_form="hinge",
        island_weight_mx=_island_weight_prov(env),
        persist_gate={"w": 0.3}, persist_classes=(1, 3), persist_cldice_iters=3,
    )
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3,
                                            render_fn_wa=_render_fn_wa)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


# ---- LEG 6: UNIFY-τ (--seg-form-unify-tau) --------------------------------
@pytest.mark.parametrize("K", [2, 3])
def test_leg_unify_tau_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=1000 + K)
    lc = _base_lc(unify_tau_state={"tau": 0.7}, seg_unify_tau_perpixel=_SEG_UNIFY)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="unify_tau", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4, f"unify_tau K={K}"


@pytest.mark.parametrize("tau", [1.0, 0.5, 0.3])
def test_leg_unify_tau_single_matches_canonical_make_loss_fn(tau):
    """The twin's unify_tau branch must equal make_loss_fn's unify_tau branch op-for-op at the LIVE
    τ (passed via unify_tau_state here, via tau_override in the serial trainer). At τ=1 both reduce
    to CE (documented)."""
    from experiments.train_witness_realized_through_R_mlx import make_loss_fn

    env = _build(1, seed=1010)
    w_seg, w_pose, hinge, mtgt = 100.0, 1.0, 4.0, 0.5
    lc = _base_lc(unify_tau_state={"tau": tau}, seg_unify_tau_perpixel=_SEG_UNIFY)

    def _sfn(m):
        return single_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
            env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
            w_seg, w_pose, hinge, mtgt, "unify_tau", 0.0, 0.0, lc)

    ls, gs = nn.value_and_grad(env["model"], _sfn)(env["model"])
    mx.eval(ls, gs)
    canonical = make_loss_fn(env["adapter"], env["rh"], env["rw"], score_domain=True,
                             pose_eps=lc.pose_eps, seg_loss="ce", tau_softplus_tau=lc.tau_use,
                             l7_mult=lc.l7_mult, l7_threshold=lc.l7_thr_use, render_fn=_render_fn)

    def _cfn(m):
        return canonical(m, env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
                         env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
                         w_seg, w_pose, hinge, mtgt, seg_form="unify_tau", tau_override=tau)

    lcv, gc = nn.value_and_grad(env["model"], _cfn)(env["model"])
    mx.eval(lcv, gc)
    assert abs(float(ls) - float(lcv)) / (abs(float(lcv)) + 1e-6) < 1e-4, (tau, float(ls), float(lcv))
    assert _max_rel_grad_err(gs, gc) < 1e-4, f"unify_tau tau={tau} vs canonical"


def test_leg_unify_tau_falls_back_to_lc_tau_use_when_state_absent():
    """unify_tau_state None => uses lc.tau_use (mirrors make_loss_fn's tau_override=None default)."""
    env = _build(2, seed=1020)
    lc = _base_lc(tau_use=0.3, unify_tau_state=None, seg_unify_tau_perpixel=_SEG_UNIFY)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="unify_tau")
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (lb, lm)


def test_leg_unify_tau_missing_callable_refuses_not_silent_ce():
    """NO-FAKE: seg_form=='unify_tau' with no seg_unify_tau_perpixel wired must RAISE, never
    silently fall back to CE (a silent wrong loss)."""
    env = _build(2, seed=1021)
    lc = _base_lc(unify_tau_state={"tau": 0.7}, seg_unify_tau_perpixel=None)
    with pytest.raises(ValueError, match="unify_tau"):
        _batched_and_meanpairs(env, lc, seg_form="unify_tau")


# ---- fail-close narrowing + waterfill projection guard --------------------
def test_validate_logit_adjust_compat_no_longer_refuses_micro_batch():
    """#D15: --logit-adjust-loss-tau is ROUTED -> the trainer validator is now a NO-OP (was a
    fail-close). The genuinely-unrouted levers keep their own precompute-block fail-closes."""
    assert _lvl._validate_logit_adjust_compat(1.0, 8) is None   # ON x batched twin: no longer raises
    assert _lvl._validate_logit_adjust_compat(0.0, 8) is None
    assert _lvl._validate_logit_adjust_compat(1.0, 1) is None


def test_waterfill_still_pins_micro_batch_unmeasured_at_n600():
    """The #294 waterfill re-admits micro-batch memory ONLY on a MEASURED uncontended n600 curve;
    routing the loss legs does NOT flip the knob on-by-default (the trajectory A/B remains the
    inclusion evidence). Guards against inventing a forward-memory curve."""
    import sys as _sys
    _tools = os.path.join(_REPO, "tools")
    if _tools not in _sys.path:
        _sys.path.insert(0, _tools)
    import memory_waterfill_config as mwc
    st = mwc.assess_micro_batch(mwc.DEFAULT_MICRO_BATCH_POINTS, target_n_pairs=600)
    assert not st.measured and st.grid == (1,), (st.measured, st.grid)


# ===========================================================================
# V9 unlock — every active production leg must have a functional batched twin.
# These providers deliberately vary by pair, preventing an accidentally reused row from passing.
# ===========================================================================
def _v9_providers(env):
    chroma_gt, ann, phase_ref, phase_dir, phase_weight, xi = {}, {}, {}, {}, {}, {}
    for k, c1 in enumerate(env["c1_list"]):
        pi = int(c1) // 2
        shape = (1, env["rh"], env["rw"])
        chroma_gt[pi] = mx.array(np.full((*shape, 3), 0.03 * (k + 1), np.float32))
        ann[pi] = mx.array(np.full(shape, 0.5 + 0.1 * k, np.float32))
        phase_ref[pi] = mx.array(np.full(shape, 0.2 + 0.05 * k, np.float32))
        direction = np.zeros(shape, np.float32)
        direction[:, ::2, :] = 1.0
        phase_dir[pi] = mx.array(direction)
        phase_weight[pi] = mx.array(np.full(shape, 0.7 + 0.03 * k, np.float32))
        xi[pi] = mx.zeros((6,), dtype=mx.float32)
    return {"chroma_gt": chroma_gt, "ann": ann, "phase_ref": phase_ref,
            "phase_dir": phase_dir, "phase_weight": phase_weight, "xi": xi}


_PARITY_TOL = 1e-4
_MIN_SIGNAL_MULTIPLE = 10.0


def _assert_lever_signal_is_resolvable(env, lc_on, label, *, lc_off=None, **kwargs):
    """NEGATIVE CONTROL: the lever must move the loss by MUCH more than the parity tolerance.

    ``_assert_parity`` asserts ``|batched - mean_of_pairs| / |mean_of_pairs| < 1e-4``. If the lever
    named in the test contributes LESS than that 1e-4 of the total loss, the assertion cannot
    distinguish "this lever batches correctly" from "this lever is ABSENT" — deleting it entirely
    would still pass. That is vacuity by TOLERANCE, the same failure shape as vacuity by ZERO
    (``[[vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801]]``): the
    instrument's resolution is coarser than the signal it claims to resolve.

    MEASURED 2026-08-01 at the ORIGINAL lever settings, with the ``_build`` determinism fix in
    place so these are reproducible rather than test-order dependent. Lever contribution relative
    to the total loss, and the resulting NEW setting:
        chroma   w=0.4  K=2 3.67e-01, K=4 1.92e-01 -> OK, unchanged (1900-3700x the tolerance)
        phase    w=0.6  K=2 1.11e-04, K=4 9.72e-05 -> MARGINAL at K=2 and BELOW tolerance at
                 K=4; raised to w=60.0 -> 1.09e-02 / 9.62e-03
        area     {0:2.0, 3:0.7}  K=2 1.33e-05, K=4 0.0 EXACT -> VACUOUS. At K=4 the hinge
                 ``max(mean(softmax[c]) - mean(onehot[c]), 0)`` is identically 0 for BOTH
                 configured classes on all 4 pairs, so the parity assert compared a number to
                 itself; widened to all five classes @400 -> 4.72e-03 / 4.29e-02
        temporal w=0.8  K=2 9.33e-08, K=4 7.64e-08 -> VACUOUS, roughly one fp32 ulp of the loss;
                 raised to w=80000 -> 5.53e-03 / 1.07e-02
    The temporal and area terms are NOT defective — both are exactly weight-linear (temporal:
    w=800 -> +0.1027, w=80000 -> +10.27, a clean 100x). They were simply configured at an
    amplitude ~1e-7 of the loss. The fixtures below therefore raise the weights until the signal
    clears the tolerance by >=10x; the batching contract is weight-linear, so this tests exactly
    the same thing at a resolution where a failure can actually be seen.
    """
    off = _batched_and_meanpairs(env, lc_off if lc_off is not None else _base_lc(), **kwargs)[0]
    on = _batched_and_meanpairs(env, lc_on, **kwargs)[0]
    ratio = abs(on - off) / max(abs(on), 1e-30)
    assert ratio > _MIN_SIGNAL_MULTIPLE * _PARITY_TOL, (
        f"{label}: lever contributes {abs(on - off):.6g} = {ratio:.4g} of the total loss "
        f"{on:.6g}, which is below {_MIN_SIGNAL_MULTIPLE}x the {_PARITY_TOL} parity tolerance. "
        "The parity assertion cannot distinguish correct batching from an ABSENT lever — raise "
        "the lever weight or fix the provider, do NOT relax this guard."
    )
    return ratio


def _assert_parity(env, lc, *, receipt_lever=None, backend_receipt="mlx_reference", **kwargs):
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, **kwargs)
    grad_rel = _max_rel_grad_err(gb, gm)
    # Uses _PARITY_TOL so this tolerance and the _assert_lever_signal_is_resolvable guard that
    # protects it cannot drift apart: loosening one without the other silently re-opens vacuity.
    assert abs(lb - lm) / (abs(lm) + 1e-6) < _PARITY_TOL, (lb, lm)
    assert grad_rel < _PARITY_TOL
    if receipt_lever is not None:
        from tac.boundary_math.micro_batch_bit_identity_probe import make_functional_parity_receipt

        fb, fm = dict(tree_flatten(gb)), dict(tree_flatten(gm))
        grad_maxabs = max(float(np.max(np.abs(np.asarray(fb[k]) - np.asarray(fm[k])))) for k in fb)
        receipt = make_functional_parity_receipt(
            lever=receipt_lever, K=len(env["c1_list"]), batched_loss=lb,
            serial_mean_loss=lm, grad_rel_l2=grad_rel, grad_maxabs=grad_maxabs,
            backend_receipt=backend_receipt)
        assert receipt.passed, receipt.as_dict()
    return lb


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("lever", ["chroma", "phase", "area"])
def test_v9_routed_lever_batched_loss_and_grad_parity(K, lever):
    env = _build(K, seed=1100 + K)
    p = _v9_providers(env)
    if lever == "chroma":
        lc = _base_lc(chroma_w=0.4, chroma_gt_prov=p["chroma_gt"],
                      chroma_ann_prov=p["ann"], use_metal_v9_levers=False)
    elif lever == "phase":
        # phase_w RAISED 0.6 -> 60.0 (2026-08-01). MEASURED: at 0.6 this lever moved the loss by
        # 1.1e-4 (K=2) / 9.7e-5 (K=4) relative — at or BELOW the 1e-4 parity tolerance asserted
        # below, so the assertion could not tell correct batching from an absent lever.
        lc = _base_lc(phase_w=60.0, phase_ref_prov=p["phase_ref"],
                      phase_dir_prov=p["phase_dir"], phase_weight_prov=p["phase_weight"],
                      use_metal_v9_levers=False)
    else:
        # area_lambda widened from {0: 2.0, 3: 0.7} to all five classes (2026-08-01). MEASURED:
        # the hinge max(mean(softmax[c]) - mean(onehot[c]), 0) is class- AND K-dependent — at K=2
        # only classes {0,2,4} are active, at K=4 only class {1}. The old two-class map was
        # therefore IDENTICALLY ZERO at K=4 (confirmed: loss unchanged to the last bit across
        # lambda 2.0 -> 500.0), making that parametrization a 0-vs-0 comparison. Covering all
        # five classes keeps the lever active at both K.
        lc = _base_lc(area_lambda=dict.fromkeys(range(5), 400.0))
    _assert_lever_signal_is_resolvable(env, lc, f"v9-routed:{lever}",
                                       seg_form="ce", eik_w=1e-2, len_w=1e-3)
    _assert_parity(env, lc, receipt_lever=lever, seg_form="ce", eik_w=1e-2, len_w=1e-3)


@pytest.mark.parametrize("K", [2, 4])
def test_v9_temporal_screw_batched_loss_and_grad_parity(K, monkeypatch):
    """A deterministic identity warp isolates the temporal loss batching contract."""
    import tac.boundary_math.warp_real_luma_frame0 as warp_mod

    observed_batch_sizes = []

    def compiled_identity(geom):
        def warp(g0, xi):
            observed_batch_sizes.append((int(g0.shape[0]), int(xi.shape[0])))
            return g0

        return warp

    monkeypatch.setattr(warp_mod, "compiled_batch_native_warp", compiled_identity)
    env = _build(K, seed=1200 + K)
    p = _v9_providers(env)
    # temporal_w RAISED 0.8 -> 80000.0 (2026-08-01). MEASURED: at 0.8 the temporal term moved the
    # loss by ~1.3e-4 ABSOLUTE against a ~1220 total, i.e. ~1e-7 relative — one to two fp32 ulp of
    # the loss, 1000x below the 1e-4 parity tolerance asserted below, so the parity assertions
    # passed on a term the instrument could not represent. The term is NOT broken: it is exactly
    # weight-linear (w=800 -> +0.1027, w=80000 -> +10.27). The annulus weighted mean
    # sum(sq*ann)/sum(ann) simply normalizes it to a very small number on this fixture.
    lc = _base_lc(temporal_w=80000.0, temporal_ann_prov=p["ann"],
                  temporal_xi_prov=p["xi"], temporal_geom_mlx=object(),
                  temporal_class_mask=mx.array([1.0, 0.5, 0.25]),
                  use_metal_v9_levers=False)
    _assert_lever_signal_is_resolvable(env, lc, "v9-temporal-screw",
                                       seg_form="ce", eik_w=1e-2, len_w=1e-3)
    _assert_parity(env, lc, receipt_lever="temporal", seg_form="ce", eik_w=1e-2, len_w=1e-3)
    # The batched leg must reach the compiled factory as one K-row warp; serial B=1 reference
    # calls can coexist but may not replace it with a Python pair loop.
    assert (K, K) in observed_batch_sizes


def test_v9_temporal_carrier_live_passes_frame0_code_for_film_mode(monkeypatch):
    """The accepted FiLM carrier mode needs a code vector for every live-xi row at K=2."""
    import tac.boundary_math.warp_real_luma_frame0 as warp_mod

    monkeypatch.setattr(warp_mod, "compiled_batch_native_warp",
                        lambda geom: (lambda g0, xi: g0))
    env = _build(2, seed=1250)
    p = _v9_providers(env)

    class _FilmLikeCarrier:
        def __init__(self):
            self.calls = []

        def xi_effective(self, pair_index, code_vec=None):
            if code_vec is None:
                raise ValueError("film carrier requires code_vec")
            self.calls.append((int(pair_index), code_vec))
            # Retain a differentiable code dependency while making the identity-warp oracle simple.
            return mx.zeros((6,), dtype=code_vec.dtype) + 0.0 * mx.sum(code_vec)

    carrier = _FilmLikeCarrier()
    env["model"].pose_carrier = carrier
    lc = _base_lc(
        temporal_w=0.8, temporal_ann_prov=p["ann"], temporal_xi_source="carrier_live",
        temporal_geom_mlx=object(), temporal_class_mask=mx.array([1.0, 0.5, 0.25]),
        use_metal_v9_levers=False,
    )
    _assert_parity(env, lc, seg_form="ce")
    assert {pair_index for pair_index, _ in carrier.calls} == {
        int(c1) // 2 for c1 in env["c1_list"]
    }
    assert all(np.array_equal(np.asarray(code_vec), np.asarray(env["model"].code[2 * pair_index]))
               for pair_index, code_vec in carrier.calls)


@pytest.mark.parametrize("lever", ["chroma", "phase", "temporal"])
def test_v9_active_levers_fail_closed_on_missing_provider(lever, monkeypatch):
    env = _build(2, seed=1300)
    p = _v9_providers(env)
    if lever == "chroma":
        lc = _base_lc(chroma_w=1.0, chroma_gt_prov=None, chroma_ann_prov=p["ann"])
        match = "chroma_gt_prov"
    elif lever == "phase":
        lc = _base_lc(phase_w=1.0, phase_ref_prov=p["phase_ref"],
                      phase_dir_prov=None, phase_weight_prov=p["phase_weight"])
        match = "phase_dir_prov"
    else:
        import tac.boundary_math.warp_real_luma_frame0 as warp_mod
        monkeypatch.setattr(warp_mod, "compiled_batch_native_warp",
                            lambda geom: (lambda g0, xi: g0))
        lc = _base_lc(temporal_w=1.0, temporal_ann_prov=p["ann"], temporal_xi_prov=None,
                      temporal_geom_mlx=object(), temporal_class_mask=mx.ones((3,)))
        match = "temporal_xi_prov"
    with pytest.raises(ValueError, match=match):
        _batched_and_meanpairs(env, lc)


@pytest.mark.parametrize("lever", ["chroma", "phase", "temporal"])
def test_v9_gate_off_and_zero_mask_are_exact_noops(lever, monkeypatch):
    env = _build(2, seed=1400)
    p = _v9_providers(env)
    base = _batched_and_meanpairs(env, _base_lc())[0]
    if lever == "chroma":
        common = {"chroma_w": 0.9, "chroma_gt_prov": p["chroma_gt"],
                  "chroma_ann_prov": p["ann"], "use_metal_v9_levers": False}
        off = _base_lc(**common, chroma_gate={"on": False})
        zero = _base_lc(**{**common, "chroma_ann_prov": {
            i: mx.zeros_like(v) for i, v in p["ann"].items()}})
    elif lever == "phase":
        common = {"phase_w": 0.9, "phase_ref_prov": p["phase_ref"],
                  "phase_dir_prov": p["phase_dir"], "phase_weight_prov": p["phase_weight"],
                  "use_metal_v9_levers": False}
        off = _base_lc(**common, phase_gate={"on": False})
        zero = _base_lc(**{**common, "phase_weight_prov": {
            i: mx.zeros_like(v) for i, v in p["phase_weight"].items()}})
    else:
        import tac.boundary_math.warp_real_luma_frame0 as warp_mod
        monkeypatch.setattr(warp_mod, "compiled_batch_native_warp",
                            lambda geom: (lambda g0, xi: g0))
        common = {"temporal_w": 0.9, "temporal_ann_prov": p["ann"],
                  "temporal_xi_prov": p["xi"], "temporal_geom_mlx": object(),
                  "temporal_class_mask": mx.ones((3,)), "use_metal_v9_levers": False}
        off = _base_lc(**common, temporal_gate={"on": False})
        zero = _base_lc(**{**common, "temporal_ann_prov": {
            i: mx.zeros_like(v) for i, v in p["ann"].items()}})
    assert abs(_batched_and_meanpairs(env, off)[0] - base) < 1e-9
    assert abs(_batched_and_meanpairs(env, zero)[0] - base) < 1e-9


def test_v9_active_legs_change_loss_and_area_gate_is_empty_mapping(monkeypatch):
    """NEGATIVE CONTROL for every V9 leg: an active lever must MOVE the loss.

    ``temporal`` was ADDED here 2026-08-01. It was the one V9 lever this control omitted, and it
    is exactly the lever whose parity assertions were later found to be vacuous — first passing
    ``0 == 0`` on a saturated mock (ddm_tr6), then still passing on a term ~1e-7 of the loss. A
    negative control that skips a leg cannot notice that leg going missing, which is the same
    empty-scope confound the control exists to prevent
    (``[[vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801]]``). Every leg the
    module routes belongs in this list.
    """
    import tac.boundary_math.warp_real_luma_frame0 as warp_mod
    monkeypatch.setattr(warp_mod, "compiled_batch_native_warp",
                        lambda geom: (lambda g0, xi: g0))
    env = _build(2, seed=1500)
    p = _v9_providers(env)
    base = _batched_and_meanpairs(env, _base_lc())[0]
    configs = [
        _base_lc(chroma_w=0.5, chroma_gt_prov=p["chroma_gt"], chroma_ann_prov=p["ann"],
                 use_metal_v9_levers=False),
        _base_lc(phase_w=0.5, phase_ref_prov=p["phase_ref"], phase_dir_prov=p["phase_dir"],
                 phase_weight_prov=p["phase_weight"], use_metal_v9_levers=False),
        _base_lc(area_lambda={0: 500.0, 1: 500.0, 2: 500.0, 3: 500.0, 4: 500.0}),
        # temporal_w is large for the reason documented on the screw-parity fixture: the annulus
        # weighted mean normalizes this term to ~1e-7 of the loss at w=0.8, below what a float32
        # loss can even represent as a change.
        _base_lc(temporal_w=80000.0, temporal_ann_prov=p["ann"], temporal_xi_prov=p["xi"],
                 temporal_geom_mlx=object(),
                 temporal_class_mask=mx.array([1.0, 0.5, 0.25]), use_metal_v9_levers=False),
    ]
    for lc in configs:
        assert abs(_batched_and_meanpairs(env, lc)[0] - base) > 1e-6
    assert abs(_batched_and_meanpairs(env, _base_lc(area_lambda={}))[0] - base) < 1e-9


@pytest.mark.parametrize("lever", ["chroma", "phase", "temporal", "area"])
def test_v9_single_routed_term_matches_independent_serial_formula_oracle(lever, monkeypatch):
    """B=1 oracle is independent of _batched_v9_map_terms and its fused helper functions."""
    if lever == "temporal":
        import tac.boundary_math.warp_real_luma_frame0 as warp_mod
        monkeypatch.setattr(warp_mod, "compiled_batch_native_warp",
                            lambda geom: (lambda g0, xi: g0))
    env = _build(1, seed=1550)
    p = _v9_providers(env)
    pi = int(env["c1_list"][0]) // 2
    if lever == "chroma":
        weight = 0.4
        lc = _base_lc(chroma_w=weight, chroma_gt_prov=p["chroma_gt"],
                      chroma_ann_prov=p["ann"], use_metal_v9_levers=False)
    elif lever == "phase":
        weight = 0.6
        lc = _base_lc(phase_w=weight, phase_ref_prov=p["phase_ref"],
                      phase_dir_prov=p["phase_dir"], phase_weight_prov=p["phase_weight"],
                      use_metal_v9_levers=False)
    elif lever == "temporal":
        weight = 0.8
        lc = _base_lc(temporal_w=weight, temporal_ann_prov=p["ann"],
                      temporal_xi_prov=p["xi"], temporal_geom_mlx=object(),
                      temporal_class_mask=mx.array([1.0, 0.5, 0.25]),
                      use_metal_v9_levers=False)
    else:
        weight = None
        lc = _base_lc(area_lambda={0: 2.0, 3: 0.7})

    def routed(m):
        return single_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
            env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
            0.0, 0.0, 4.0, 0.5, "ce", 0.0, 0.0, lc)

    def oracle(m):
        f1 = _render_fn(m, env["cf_list"][0], env["c1_list"][0], env["rh"], env["rw"])
        sl1 = env["adapter"].segnet(f1)
        if lever == "chroma":
            luma = (0.299 * f1[..., 0:1] + 0.587 * f1[..., 1:2]
                    + 0.114 * f1[..., 2:3])
            chroma = f1 - luma
            sq = mx.sum(mx.square(chroma - p["chroma_gt"][pi]), axis=-1)
            ann = p["ann"][pi]
            return weight * mx.sum(sq * ann) / (mx.sum(ann) + 1e-6)
        if lever == "phase":
            oh = env["oh_list"][0]
            gt = mx.sum(sl1 * oh, axis=-1)
            runner = mx.max(sl1 + oh * (-1e9), axis=-1)
            margin = mx.maximum(gt - runner, 0.0)
            right = mx.pad(margin[:, :, 1:], [(0, 0), (0, 0), (0, 1)])
            down = mx.pad(margin[:, 1:, :], [(0, 0), (0, 1), (0, 0)])
            partner = mx.where(p["phase_dir"][pi] < 0.5, right, down)
            tie = margin / (margin + partner + lc.phase_eps)
            sq = mx.square(tie - p["phase_ref"][pi])
            phase_weight = p["phase_weight"][pi]
            return weight * mx.sum(sq * phase_weight) / (mx.sum(phase_weight) + 1e-6)
        if lever == "temporal":
            f0 = _render_fn(m, env["cf_list"][0], env["c0_list"][0], env["rh"], env["rw"])
            sl0 = env["adapter"].segnet(f0)
            g1 = mx.softmax(sl1, axis=-1)[..., 0:3]
            g0_identity_warp = mx.softmax(sl0, axis=-1)[..., 0:3]
            sq = mx.sum(mx.square(g1 - g0_identity_warp) * lc.temporal_class_mask, axis=-1)
            ann = p["ann"][pi]
            return weight * mx.sum(sq * ann) / (mx.sum(ann) + 1e-6)
        soft = mx.softmax(sl1, axis=-1)
        area = mx.zeros(())
        for cls, lam in lc.area_lambda.items():
            over = mx.maximum(
                mx.mean(soft[..., int(cls)]) - mx.mean(env["oh_list"][0][..., int(cls)]),
                0.0)
            area = area + 0.5 * float(lam) * over * over
        return area

    routed_value, routed_grad = nn.value_and_grad(env["model"], routed)(env["model"])
    oracle_value, oracle_grad = nn.value_and_grad(env["model"], oracle)(env["model"])
    mx.eval(routed_value, routed_grad, oracle_value, oracle_grad)
    rv, ov = float(routed_value), float(oracle_value)
    assert abs(rv - ov) / (abs(ov) + 1e-6) < 1e-4, (lever, rv, ov)
    assert _max_rel_grad_err(routed_grad, oracle_grad) < 1e-4
    routed_flat, oracle_flat = dict(tree_flatten(routed_grad)), dict(tree_flatten(oracle_grad))
    grad_maxabs = max(float(np.max(np.abs(
        np.asarray(routed_flat[key]) - np.asarray(oracle_flat[key])))) for key in routed_flat)
    assert grad_maxabs < 1e-2, (lever, grad_maxabs)


def test_temporal_uses_raw_f0_provider_while_pose_keeps_general_carrier_render(monkeypatch):
    """Both B=1 and B>1 keep raw temporal f0 separate from the carrier PoseNet f0."""
    import tac.boundary_math.warp_real_luma_frame0 as warp_mod
    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    monkeypatch.setattr(warp_mod, "compiled_batch_native_warp",
                        lambda geom: (lambda g0, xi: g0))
    env = _build(2, seed=1575)
    p = _v9_providers(env)
    pi = int(env["c1_list"][0]) // 2

    def raw_render(model, cf, code_idx, rh, rw):
        return _render_fn(model, cf, code_idx, rh, rw)

    def carrier_render(model, cf, code_idx, rh, rw):
        raw = raw_render(model, cf, code_idx, rh, rw)
        if int(code_idx) % 2:
            return raw
        carrier_delta = mx.array([35.0, -18.0, 9.0], dtype=mx.float32)
        return raw * 0.65 + carrier_delta

    lc = _base_lc(
        temporal_w=0.8, temporal_ann_prov=p["ann"], temporal_xi_prov=p["xi"],
        temporal_geom_mlx=object(), temporal_class_mask=mx.array([1.0, 0.5, 0.25]),
        temporal_render_f0_fn=raw_render, use_metal_v9_levers=False)

    def temporal_routed(m, cfg=lc):
        return single_realized_loss(
            m, env["adapter"], carrier_render, env["rh"], env["rw"],
            env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
            env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
            0.0, 0.0, 4.0, 0.5, "ce", 0.0, 0.0, cfg)

    def temporal_raw_oracle(m):
        f1 = carrier_render(
            m, env["cf_list"][0], env["c1_list"][0], env["rh"], env["rw"])
        f0_raw = raw_render(
            m, env["cf_list"][0], env["c0_list"][0], env["rh"], env["rw"])
        g1 = mx.softmax(env["adapter"].segnet(f1), axis=-1)[..., 0:3]
        g0 = mx.softmax(env["adapter"].segnet(f0_raw), axis=-1)[..., 0:3]
        sq = mx.sum(mx.square(g1 - g0) * lc.temporal_class_mask, axis=-1)
        ann = p["ann"][pi]
        return lc.temporal_w * mx.sum(sq * ann) / (mx.sum(ann) + 1e-6)

    routed_value, routed_grad = nn.value_and_grad(env["model"], temporal_routed)(env["model"])
    oracle_value, oracle_grad = nn.value_and_grad(env["model"], temporal_raw_oracle)(env["model"])
    mx.eval(routed_value, routed_grad, oracle_value, oracle_grad)
    assert abs(float(routed_value) - float(oracle_value)) / (abs(float(oracle_value)) + 1e-6) < 1e-4
    assert _max_rel_grad_err(routed_grad, oracle_grad) < 1e-4

    wrong_lc = _base_lc(
        temporal_w=lc.temporal_w, temporal_ann_prov=p["ann"], temporal_xi_prov=p["xi"],
        temporal_geom_mlx=object(), temporal_class_mask=lc.temporal_class_mask,
        temporal_render_f0_fn=None, use_metal_v9_levers=False)
    wrong_value = temporal_routed(env["model"], wrong_lc)
    mx.eval(wrong_value)
    assert abs(float(wrong_value) - float(oracle_value)) > 1e-6

    def pose_routed(m, render=carrier_render):
        return single_realized_loss(
            m, env["adapter"], render, env["rh"], env["rw"],
            env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
            env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
            0.0, 1.0, 4.0, 0.5, "ce", 0.0, 0.0, _base_lc())

    def pose_general_oracle(m):
        f0 = carrier_render(
            m, env["cf_list"][0], env["c0_list"][0], env["rh"], env["rw"])
        f1 = carrier_render(
            m, env["cf_list"][0], env["c1_list"][0], env["rh"], env["rw"])
        pair = mx.stack([f0[0], f1[0]], axis=0)[None]
        yuv = rgb_to_yuv6_mlx(pair)
        b, t, h2, w2, c6 = yuv.shape
        yuv_nhwc = mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (b, h2, w2, t * c6))
        pose = env["adapter"].posenet(yuv_nhwc)["pose"][0, :env["pt_list"][0].shape[-1]]
        return mx.sqrt(10.0 * mx.mean(mx.square(pose - env["pt_list"][0]))
                       + _base_lc().pose_eps)

    pose_value, pose_grad = nn.value_and_grad(env["model"], pose_routed)(env["model"])
    pose_oracle, pose_oracle_grad = nn.value_and_grad(
        env["model"], pose_general_oracle)(env["model"])
    mx.eval(pose_value, pose_grad, pose_oracle, pose_oracle_grad)
    assert abs(float(pose_value) - float(pose_oracle)) / (abs(float(pose_oracle)) + 1e-6) < 1e-4
    assert _max_rel_grad_err(pose_grad, pose_oracle_grad) < 1e-4
    raw_pose_value = pose_routed(env["model"], raw_render)
    mx.eval(raw_pose_value)
    assert abs(float(raw_pose_value) - float(pose_oracle)) > 1e-6

    # The exact integration risk is the batched path: temporal must use raw f0 while the base
    # PoseNet pair continues to use carrier-composed f0 for every row. Compare B=2 against its
    # correctly routed serial mean, then prove the two tempting wrong aliases change the value.
    temporal_b, _, temporal_serial, _ = _batched_and_meanpairs(
        env, lc, w_seg=0.0, w_pose=0.0, render_fn=carrier_render)
    assert abs(temporal_b - temporal_serial) / (abs(temporal_serial) + 1e-6) < 1e-4
    wrong_b = _batched_and_meanpairs(
        env, wrong_lc, w_seg=0.0, w_pose=0.0, render_fn=carrier_render)[0]
    assert abs(wrong_b - temporal_b) > 1e-6

    pose_b, _, pose_serial, _ = _batched_and_meanpairs(
        env, _base_lc(), w_seg=0.0, w_pose=1.0, render_fn=carrier_render)
    assert abs(pose_b - pose_serial) / (abs(pose_serial) + 1e-6) < 1e-4
    raw_pose_b = _batched_and_meanpairs(
        env, _base_lc(), w_seg=0.0, w_pose=1.0, render_fn=raw_render)[0]
    assert abs(raw_pose_b - pose_b) > 1e-6


def test_v9_live_logit_and_birth_ramp_state_are_reread_without_rebuild():
    env = _build(2, seed=1600)
    offset_state = {"offset": mx.zeros((5,))}
    lc = _base_lc(logit_adjust_state=offset_state,
                  logit_adjust_offset=mx.array(_LA_OFFSET_NP))
    before = _assert_parity(env, lc)
    offset_state["offset"] = mx.array(_LA_OFFSET_NP)
    after = _assert_parity(env, lc)
    assert abs(after - before) > 1e-6

    ramp = {"amp_active": True, "amp_lane": 0.2, "amp_mov": 0.4, "persist_scale": None}
    masks = _island_weight_prov(env, 0.0)
    movable = _island_weight_prov(env, 0.0)
    for pi in masks:
        lane_np = np.zeros((1, env["rh"], env["rw"]), np.float32)
        lane_np[:, :, :env["rw"] // 2] = 1.0
        masks[pi] = mx.array(lane_np)
        movable[pi] = 1.0 - masks[pi]
    lc = _base_lc(amplify_w=0.8, island_weight_mx=_island_weight_prov(env),
                  amplify_ramp_state=ramp, amplify_lane_masks=masks,
                  amplify_movable_masks=movable)
    before = _assert_parity(env, lc)
    ramp["amp_lane"], ramp["amp_mov"] = 1.7, 2.1
    after = _assert_parity(env, lc)
    assert abs(after - before) > 1e-6

    ramp = {"amp_active": False, "amp_lane": 1.0, "amp_mov": 1.0,
            "persist_scale": [0.0, 0.0]}
    lc = _base_lc(persist_gate={"w": 0.5}, persist_classes=(1, 3),
                  persist_cldice_iters=3, persist_recall_w=1.0,
                  amplify_ramp_state=ramp)
    before = _assert_parity(env, lc)
    ramp["persist_scale"] = [2.0, 3.0]
    after = _assert_parity(env, lc)
    assert abs(after - before) > 1e-6


@pytest.mark.parametrize("K", [2, 4])
def test_v9_lane_band_render_composition_is_preserved(K):
    env = _build(K, seed=1700 + K)

    def composed_render(model, cf, code_idx, rh, rw):
        bare = _render_fn(model, cf, code_idx, rh, rw)
        lane = mx.array(np.linspace(0.0, 7.0, rw, dtype=np.float32)[None, None, :, None])
        return mx.clip(bare + lane, 0.0, 255.0)

    composed = _assert_parity(env, _base_lc(), render_fn=composed_render)
    bare = _batched_and_meanpairs(env, _base_lc())[0]
    assert abs(composed - bare) > 1e-6


def test_v9_combined_stack_batched_loss_and_grad_parity(monkeypatch):
    import tac.boundary_math.warp_real_luma_frame0 as warp_mod
    monkeypatch.setattr(warp_mod, "compiled_batch_native_warp",
                        lambda geom: (lambda g0, xi: g0))
    env = _build(4, seed=1800)
    p = _v9_providers(env)
    lc = _base_lc(
        logit_adjust_state={"offset": mx.array(_LA_OFFSET_NP)},
        unify_tau_state={"tau": 0.6}, seg_unify_tau_perpixel=_SEG_UNIFY,
        chroma_w=0.2, chroma_gt_prov=p["chroma_gt"], chroma_ann_prov=p["ann"],
        phase_w=0.3, phase_ref_prov=p["phase_ref"], phase_dir_prov=p["phase_dir"],
        phase_weight_prov=p["phase_weight"],
        temporal_w=0.4, temporal_ann_prov=p["ann"], temporal_xi_prov=p["xi"],
        temporal_geom_mlx=object(), temporal_class_mask=mx.array([1.0, 0.5, 0.25]),
        area_lambda={0: 1.0, 3: 0.5}, use_metal_v9_levers=False,
    )
    _assert_parity(env, lc, seg_form="unify_tau", eik_w=1e-2, len_w=1e-3)


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("kind", ["chroma", "phase", "temporal"])
def test_v9_map_helper_output_and_all_primal_vjps_match_reference(K, kind):
    from tac.local_acceleration import metal_micro_batch_v9_levers as kernels

    rng = np.random.default_rng(1900 + K)
    x = mx.array(rng.standard_normal((K, 5, 7, 3)).astype(np.float32))
    y = mx.array(rng.standard_normal((K, 5, 7, 3)).astype(np.float32))
    if kind == "chroma":
        fast, ref = kernels.chroma_squared_map, kernels.chroma_squared_map_reference
        args = (x, y)
    elif kind == "phase":
        d = mx.array(rng.integers(0, 2, (K, 5, 7)).astype(np.float32))
        r = mx.array(rng.random((K, 5, 7)).astype(np.float32))
        fast, ref, args = kernels.phase_squared_map, kernels.phase_squared_map_reference, (x[..., 0], d, r)
    else:
        mask = mx.array([1.0, 0.5, 0.0])
        fast, ref, args = kernels.temporal_squared_map, kernels.temporal_squared_map_reference, (x, y, mask)

    cotangent = mx.array(rng.standard_normal(tuple(ref(*args).shape)).astype(np.float32))

    def fused_all(*primals):
        return fast(*primals, use_metal=True)

    vf, gf = mx.vjp(fused_all, list(args), [cotangent])
    vr, gr = mx.vjp(ref, list(args), [cotangent])
    mx.eval(vf, vr, *gf, *gr)
    assert np.allclose(np.asarray(vf), np.asarray(vr), rtol=1e-5, atol=1e-5)
    assert len(gf) == len(gr) == len(args)
    # Which primals are theta-bearing is a DECLARATION, not an inference from the data. Only
    # ``phase``'s ``direction`` (index 1) is a non-differentiable selector; every other primal
    # here must carry a nonzero gradient. Without this, an ``allclose`` on two identically-zero
    # fields passes vacuously and a gradient that silently went missing reads as a clean pass
    # (``[[vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801]]``). MEASURED
    # 2026-08-01: ``phase:grad1`` is exactly that case on both this surface and the 384x512 one.
    structurally_zero = {1} if kind == "phase" else set()
    for gi, (fused_grad, reference_grad) in enumerate(zip(gf, gr, strict=True)):
        ref_np = np.asarray(reference_grad)
        peak = float(np.abs(ref_np).max())
        if gi in structurally_zero:
            assert peak == 0.0, (
                f"{kind}:grad{gi} is declared a structurally-zero selector gradient but the "
                f"reference is nonzero (maxabs {peak:.6g}); re-derive the primal roles."
            )
            assert float(np.abs(np.asarray(fused_grad)).max()) == 0.0, (
                f"{kind}:grad{gi} static-provider gradient must be exactly zero in the fused path"
            )
            continue
        assert peak > 0.0, (
            f"{kind}:grad{gi} is theta-bearing but the reference gradient is identically zero — "
            "the quantity under test is absent, so the parity check below would be vacuous."
        )
        assert np.allclose(np.asarray(fused_grad), ref_np, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kind", ["chroma", "phase", "temporal"])
def test_v9_map_helpers_fail_closed_on_malformed_shapes(kind):
    from tac.local_acceleration import metal_micro_batch_v9_levers as kernels

    rgb = mx.zeros((2, 5, 7, 3), dtype=mx.float32)
    maps = mx.zeros((2, 5, 7), dtype=mx.float32)
    with pytest.raises(ValueError, match=r"shape|identical"):
        if kind == "chroma":
            kernels.chroma_squared_map(rgb, mx.zeros((2, 5, 7, 2)), use_metal=True)
        elif kind == "phase":
            kernels.phase_squared_map(maps, maps[:, :, :-1], maps, use_metal=True)
        else:
            kernels.temporal_squared_map(rgb, rgb, mx.ones((2,)), use_metal=True)


def test_v9_gpu_surface_emits_real_metal_backend_receipts():
    from tac.local_acceleration import metal_micro_batch_v9_levers as kernels
    from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

    with temporary_mlx_device("gpu"):
        if not kernels.metal_micro_batch_v9_available():
            pytest.skip("requires an initialized MLX Metal device")
        rng = np.random.default_rng(1950)
        x = mx.array(rng.standard_normal((2, 5, 7, 3)).astype(np.float32))
        y = mx.array(rng.standard_normal((2, 5, 7, 3)).astype(np.float32))
        direction = mx.array(rng.integers(0, 2, (2, 5, 7)).astype(np.float32))
        reference = mx.array(rng.random((2, 5, 7)).astype(np.float32))
        class_mask = mx.array([1.0, 0.5, 0.25], dtype=mx.float32)
        cases = (
            ("chroma", kernels.chroma_squared_map, kernels.chroma_squared_map_reference,
             (x, y)),
            ("phase", kernels.phase_squared_map, kernels.phase_squared_map_reference,
             (x[..., 0], direction, reference)),
            ("temporal", kernels.temporal_squared_map, kernels.temporal_squared_map_reference,
             (x, y, class_mask)),
        )
        for kind, fused, ref, args in cases:
            cotangent = mx.ones(tuple(ref(*args).shape), dtype=mx.float32)
            fast_value, fast_grads = mx.vjp(
                lambda *xs, _fused=fused: _fused(*xs, use_metal=True),
                list(args),
                [cotangent],
            )
            ref_value, ref_grads = mx.vjp(ref, list(args), [cotangent])
            # Graph construction selected the kernel but has not proven a dispatch.
            assert kernels.v9_lever_backend_receipt()[kind] == "metal_planned"
            planned = kernels.v9_lever_backend_details()[kind]
            assert planned["forward_status"] == "planned"
            assert planned["backward_status"] == "planned"
            assert planned["verified_backend"] is None
            verified = kernels.verify_v9_lever_backend_execution(kind)
            assert verified["forward_status"] == "evaluated"
            assert verified["backward_status"] == "evaluated"
            assert verified["verified_backend"] == "metal"
            assert kernels.v9_lever_backend_receipt()[kind] == "metal"
            mx.eval(fast_value, ref_value, *fast_grads, *ref_grads)
            assert np.allclose(
                np.asarray(fast_value), np.asarray(ref_value), rtol=1e-5, atol=1e-5)
            for fast_grad, ref_grad in zip(fast_grads, ref_grads, strict=True):
                assert np.allclose(
                    np.asarray(fast_grad), np.asarray(ref_grad), rtol=1e-4, atol=1e-4)
    assert mx.default_device().type == mx.cpu


def test_v9_faithful_384x512_metal_maps_and_area_value_vjp():
    """Faithful SegNet-grid gate: B=2 fused maps + vectorized area versus serial formulas.

    The tiny matrix above is the fast regression surface. This bounded test keeps the production
    ``(384,512)`` spatial geometry and therefore catches grid-size/indexing/dispatch defects before
    the operator fires the full V9 arm. It is deliberately a Metal-only runtime gate: a headless
    run must skip/refuse rather than certify the fused backend from the MLX fallback.
    """
    from tac.boundary_math.levelset_micro_batch_loss import _batched_v9_map_terms
    from tac.local_acceleration import metal_micro_batch_v9_levers as kernels
    from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

    with temporary_mlx_device("gpu"):
        if not kernels.metal_micro_batch_v9_available():
            pytest.skip("requires an initialized MLX Metal device")
        B, H, W = 2, 384, 512
        rng = np.random.default_rng(1975)
        rgb = mx.array(rng.uniform(0.0, 255.0, (B, H, W, 3)).astype(np.float32))
        target = mx.array(rng.uniform(-80.0, 80.0, (B, H, W, 3)).astype(np.float32))
        signed = mx.array(rng.standard_normal((B, H, W)).astype(np.float32))
        direction = mx.array(rng.integers(0, 2, (B, H, W)).astype(np.float32))
        reference = mx.array(rng.random((B, H, W)).astype(np.float32))
        g0 = mx.array(rng.random((B, H, W, 3)).astype(np.float32))
        # Canonical V9 TemporalScrewConsistency selects all ground classes {0,1,2}.
        class_mask = mx.array([1.0, 1.0, 1.0], dtype=mx.float32)
        cases = (
            ("chroma", kernels.chroma_squared_map, kernels.chroma_squared_map_reference,
             (rgb, target)),
            ("phase", kernels.phase_squared_map, kernels.phase_squared_map_reference,
             (signed, direction, reference)),
            ("temporal", kernels.temporal_squared_map, kernels.temporal_squared_map_reference,
             (rgb / 255.0, g0, class_mask)),
        )
        for kind, fused, ref, args in cases:
            # Non-uniform cotangents cover every primal VJP, including target/reference and the
            # temporal three-class mask reduction, at the production SegNet grid.
            cotangent = mx.array(rng.standard_normal((B, H, W)).astype(np.float32))
            fused_value, fused_grads = mx.vjp(
                lambda *xs, _fused=fused: _fused(*xs, use_metal=True),
                list(args),
                [cotangent],
            )
            ref_value, ref_grads = mx.vjp(ref, list(args), [cotangent])
            assert kernels.v9_lever_backend_receipt()[kind] == "metal_planned"
            kernels.verify_v9_lever_backend_execution(kind)
            assert kernels.v9_lever_backend_receipt()[kind] == "metal"
            mx.eval(fused_value, ref_value, *fused_grads, *ref_grads)
            _assert_fp32_field_parity(f"{kind}:value", fused_value, ref_value,
                                      expect_nonzero=True)
            for gi, (fused_grad, ref_grad) in enumerate(
                zip(fused_grads, ref_grads, strict=True)
            ):
                if kind == "phase" and gi == 0:
                    # dL/dsigned: cancellation-dominated, gets its own DERIVED budget.
                    _assert_phase_signed_grad_parity(
                        f"{kind}:grad{gi}", fused_grad, ref_grad, signed, direction, reference,
                        cotangent, _PHASE_KERNEL_DEFAULT_EPS)
                    continue
                # ``phase:grad1`` is dL/d(direction) — a non-differentiable selector, so its
                # gradient is structurally zero. Every other primal here is theta-bearing.
                _assert_fp32_field_parity(
                    f"{kind}:grad{gi}", fused_grad, ref_grad,
                    expect_nonzero=not (kind == "phase" and gi == 1))

        logits = mx.array(rng.standard_normal((B, H, W, 5)).astype(np.float32))
        labels = rng.integers(0, 5, (B, H, W))
        oh = mx.array(np.eye(5, dtype=np.float32)[labels])
        lc = _base_lc(area_lambda={1: 2.0, 3: 0.7})
        dummy_rgb = mx.zeros((B, H, W, 3), dtype=mx.float32)

        def area_batch(z):
            return _batched_v9_map_terms(
                None, z, z, dummy_rgb, None, oh, [0, 1], lc, include_area=True)

        def area_serial(z):
            rows = []
            for k in range(B):
                soft = mx.softmax(z[k:k + 1], axis=-1)
                row = mx.zeros(())
                for cls, lam in lc.area_lambda.items():
                    over = mx.maximum(
                        mx.mean(soft[..., int(cls)]) - mx.mean(oh[k:k + 1, ..., int(cls)]), 0.0)
                    row = row + 0.5 * float(lam) * over * over
                rows.append(row)
            return mx.mean(mx.stack(rows))

        area_b, grad_b = mx.value_and_grad(area_batch)(logits)
        area_s, grad_s = mx.value_and_grad(area_serial)(logits)
        mx.eval(area_b, grad_b, area_s, grad_s)
        assert np.allclose(np.asarray(area_b), np.asarray(area_s), rtol=1e-5, atol=1e-6)
        assert np.allclose(np.asarray(grad_b), np.asarray(grad_s), rtol=1e-4, atol=1e-6)
    assert mx.default_device().type == mx.cpu


@pytest.mark.parametrize("lever", ["chroma", "phase", "temporal", "area"])
def test_v9_probe_functional_parity_receipt_is_explicit(lever):
    from tac.boundary_math.micro_batch_bit_identity_probe import make_functional_parity_receipt

    receipt = make_functional_parity_receipt(
        lever=lever, K=4, batched_loss=10.000001, serial_mean_loss=10.0,
        grad_rel_l2=2e-6, grad_maxabs=3e-4, backend_receipt="mlx_reference",
    )
    payload = receipt.as_dict()
    assert payload["passed"] is True
    assert payload["loss_abs"] > 0.0 and payload["loss_rel"] > 0.0
    assert payload["grad_rel_l2"] == 2e-6 and payload["grad_maxabs"] == 3e-4
    assert payload["loss_rel_tolerance"] == 1e-4
    assert payload["backend_receipt"] == "mlx_reference"


def test_v9_probe_combined_training_admission_rejects_unchecked_receipts_and_bare_speedup():
    from tac.boundary_math.micro_batch_bit_identity_probe import (
        classify_training_admission,
        make_functional_parity_receipt,
    )

    receipts = [make_functional_parity_receipt(
        lever=lever, K=4, batched_loss=1.0, serial_mean_loss=1.0,
        grad_rel_l2=0.0, grad_maxabs=0.0,
        backend_receipt={
            "chroma": "metal", "phase": "metal", "temporal": "metal",
            "area": "mlx_vectorized", "full_v9": "metal+mlx",
        }[lever],
        config_id="v9_cgauge_432", device="gpu", scorer_surface="real_frozen_v9",
        faithful_scale=True, measurement_artifact="receipt.json",
        measurement_artifact_sha256="a" * 64, scorer_fingerprint_sha256="b" * 64,
    ) for lever in ("chroma", "phase", "temporal", "area", "full_v9")]
    # Diagnostic dataclasses and caller-reported speed telemetry are intentionally non-admitting.
    # Persisted JSON can validate schema/custody only; no disk-authored row can attest that Metal,
    # the frozen scorer, or the full V9 step actually executed.
    refused = classify_training_admission(receipts, reported_end_to_end_speedup=1.25)
    assert refused.functional_parity_passed is False
    assert refused.training_throughput_admitted is False
    assert refused.no_score_authority is True
    assert refused.missing_levers == ("chroma", "phase", "temporal", "area", "full_v9")
