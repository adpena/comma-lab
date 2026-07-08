"""Micro-batch (--micro-batch-pairs B>1) bit-identity DECOMPOSITION probe.

CRUX-ENGINEERING FINDING (2026-07-08, MEASURED): the trajectory-A/B gate on the
``--micro-batch-pairs`` 2-4x speed lever CANNOT be dissolved by fixed-order reduction
engineering, because the DOMINANT divergence between the batched twin and the serial
accumulation enters at the FROZEN-SCORER FORWARD KERNEL (EfficientNet-B2 SegNet /
FastViT PoseNet), which is batch-DEPENDENT on the real MLX backends:

    device   segnet max|Δlogit|   argmax px flipped   posenet max|Δ|
    ------   ------------------   -----------------   --------------
    GPU      2.3e-2               11 / 196608         7.7e-3
    CPU      7.1e-5               0                   2.0e-6

(measured with the real upstream adapter over K=4 random 384x512 frames; see
``tools/micro_batch_bit_identity_probe.py`` to reproduce.) This is UPSTREAM of any loss
or gradient reduction: ``segnet(f1_batch)[k] != segnet(f1_batch[k:k+1])[0]`` bit-for-bit,
so the per-pair loss ``L_k`` computed from the batched forward already differs from the
serial per-pair ``L_k`` before any accumulation happens. It is a GPU/CPU conv/matmul
kernel tiling property (sister of the #348 ``mlx_gpu_crossprocess_nondeterminism_v1``
family), NOT a reordering we can control.

The reduction/accumulation-ORDER sources (what the operator's fix (a)/(b) anticipated)
are SECONDARY and only surface where the scorer IS batch-invariant. This module MEASURES
them in isolation with a batch-INVARIANT mock scorer (a linear per-pixel/per-frame op,
whose batched forward+backward are provably 0.0 batch-dependent), so the residual is
PURELY the loss/grad reduction order:

* the batched twin builds ``L = mean_k L_k`` and takes ONE ``value_and_grad`` -> MLX's
  backward accumulates the K per-pair contributions into the SHARED witness params
  (out_tex / in_proj / code ...) in a graph-internal order that differs from the serial
  explicit left-fold ``accum = g0; accum += g1; ...`` -> ~1e-3..1e-2 max|Δ| on the grad
  tree (hidden by the global-L2 ~1e-7 metric the equivalence tests use).

CONSEQUENCE (the honest verdict): full bit-identity of B>1 to the serial path at any
speedup > 1x is IMPOSSIBLE on the real MLX scorer (GPU or CPU), because the entire win is
the batched scorer forward (GPU ~1.56x / CPU ~1.75x at K=8) which is the exact op that is
not batch-invariant; a bit-identical construction requires a per-pair (batch-1) scorer
forward == the serial path == 1.0x. The ``--micro-batch-pairs`` lever therefore stays
trajectory-A/B-gated; the ONLY paths to admission are (a) a bounded n600 d_seg A/B
measuring whether the ~2.3e-2 forward drift is d_seg-neutral over training (argmax flips
only 11/196608 px = 0.006%, so plausibly neutral -- but MEASURE, do not assume), or
(b) batch-invariant scorer kernels (a large MLX/Metal item, out of scope here).

MEANS, not ends. Nothing here moves the pointer (contest-CPU 0.19110 UNMOVED). Only a
byte-closed n600 ``upstream/evaluate.py`` row does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# MEASURED empirical anchors (2026-07-08, real upstream adapter, K=4, 384x512).
# Recorded for provenance + as the classification reference; the CLI re-measures live.
# ─────────────────────────────────────────────────────────────────────────────
MEASURED_SCORER_FWD_GPU_SEG_MAXABS = 2.259e-2
MEASURED_SCORER_FWD_GPU_ARGMAX_FLIPS = 11        # of 384*512 = 196608 px
MEASURED_SCORER_FWD_GPU_POSE_MAXABS = 7.728e-3
MEASURED_SCORER_FWD_CPU_SEG_MAXABS = 7.105e-5
MEASURED_SCORER_FWD_CPU_ARGMAX_FLIPS = 0
MEASURED_SCORER_FWD_CPU_POSE_MAXABS = 2.027e-6
# scorer-forward microbench speedup (K=8, ONE batched fwd vs K per-pair fwds).
MEASURED_SCORER_FWD_SPEEDUP_GPU = 1.56
MEASURED_SCORER_FWD_SPEEDUP_CPU = 1.75
# argmax-invariance is the load-bearing sub-property: SegNet d_seg is an argmax rate.
# CPU flips 0 px; GPU flips 11 px (0.006%). The reduction-order grad drift on a
# batch-INVARIANT mock scorer (source B, isolated), observed max|Δ| on the grad tree:
MEASURED_REDUCTION_ORDER_GRAD_MAXABS_MOCK = 3.9e-3  # K=4, ce, out_tex leaf


@dataclass(frozen=True)
class ReductionOrderDrift:
    """Pure reduction/accumulation-ORDER drift (source B) measured with a batch-INVARIANT
    mock scorer, so the scorer-forward kernel (source A) contributes exactly 0.0 and the
    residual is only the loss/grad accumulation order.

    ``grad_maxabs`` is the max absolute per-leaf difference between the batched twin grad
    and the serial left-fold mean-of-per-pair grad (the trajectory-relevant metric — the
    global-L2 ``grad_rel_l2`` hides it). ``loss_abs`` is the loss-scalar difference.
    """

    K: int
    seg_form: str
    grad_maxabs: float
    grad_rel_l2: float
    loss_abs: float
    worst_leaf: str


@dataclass(frozen=True)
class BitIdentityVerdict:
    """Honest classification of whether B>1 can be bit-identical + at what speedup."""

    device: str
    scorer_fwd_seg_maxabs: float
    scorer_fwd_argmax_flips: int
    scorer_fwd_pose_maxabs: float
    reduction_order_grad_maxabs: float
    scorer_fwd_speedup: float
    # derived
    scorer_forward_is_batch_invariant: bool
    argmax_is_batch_invariant: bool
    bit_identical_at_speedup_possible: bool
    surviving_speedup_at_bit_identity: float
    dominant_source: str            # "scorer_forward" | "reduction_order" | "none"
    admission_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "scorer_fwd_seg_maxabs": self.scorer_fwd_seg_maxabs,
            "scorer_fwd_argmax_flips": self.scorer_fwd_argmax_flips,
            "scorer_fwd_pose_maxabs": self.scorer_fwd_pose_maxabs,
            "reduction_order_grad_maxabs": self.reduction_order_grad_maxabs,
            "scorer_fwd_speedup": self.scorer_fwd_speedup,
            "scorer_forward_is_batch_invariant": self.scorer_forward_is_batch_invariant,
            "argmax_is_batch_invariant": self.argmax_is_batch_invariant,
            "bit_identical_at_speedup_possible": self.bit_identical_at_speedup_possible,
            "surviving_speedup_at_bit_identity": self.surviving_speedup_at_bit_identity,
            "dominant_source": self.dominant_source,
            "admission_path": self.admission_path,
            "pointer": "0.19110 UNMOVED (means; only a byte-closed n600 evaluate.py row moves it)",
        }


# Tolerance below which a scorer FORWARD delta is treated as bit-invariant (fp32
# machine-eps scale). The real scorer clears this on NEITHER GPU (2.3e-2) nor CPU
# (7e-5) — both exceed it — so batching the scorer forward is never bit-invariant.
_SCORER_FWD_BIT_INVARIANT_TOL = 1e-6


def classify_micro_batch_bit_identity(
    *,
    device: str,
    scorer_fwd_seg_maxabs: float,
    scorer_fwd_argmax_flips: int,
    scorer_fwd_pose_maxabs: float,
    reduction_order_grad_maxabs: float,
    scorer_fwd_speedup: float,
    scorer_fwd_bit_invariant_tol: float = _SCORER_FWD_BIT_INVARIANT_TOL,
) -> BitIdentityVerdict:
    """Classify the micro-batch bit-identity situation from MEASURED inputs.

    The load-bearing logic (NO hidden assumptions): a batched-scorer construction can be
    bit-identical to serial ONLY IF the scorer forward is batch-invariant to fp32 eps.
    The entire measured speedup is the batched scorer forward, so if that forward is not
    bit-invariant, the surviving speedup at bit-identity collapses to 1.0x (the per-pair
    forward == serial). The reduction-order grad drift is a SECONDARY source that only
    matters where the scorer is invariant.
    """
    scorer_fwd_invariant = (
        max(float(scorer_fwd_seg_maxabs), float(scorer_fwd_pose_maxabs))
        <= float(scorer_fwd_bit_invariant_tol)
    )
    argmax_invariant = int(scorer_fwd_argmax_flips) == 0

    if scorer_fwd_invariant:
        # Where the scorer IS bit-invariant, the only residual is the reduction order,
        # which is controllable (fixed-order left-fold) -> bit-identity achievable AND
        # the batched-forward speedup survives.
        bit_possible = True
        surviving = float(scorer_fwd_speedup)
        dominant = "reduction_order" if reduction_order_grad_maxabs > 0.0 else "none"
        admission = (
            "reduction-order fix (fixed-order left-fold) => bit-identical at the batched "
            "speedup; admissible without A/B"
        )
    else:
        # The real scorer case: forward is NOT batch-invariant -> bit-identity requires a
        # per-pair (batch-1) forward == serial == 1.0x. The speedup is inseparable from the
        # non-bit-invariant batched forward.
        bit_possible = False
        surviving = 1.0
        dominant = "scorer_forward"
        admission = (
            "bit-identity IMPOSSIBLE at speedup>1x (scorer forward batch-dependent); "
            "admit via (a) bounded n600 d_seg A/B [argmax-invariant on CPU, 11px flip on "
            "GPU => plausibly d_seg-neutral, MEASURE] or (b) batch-invariant scorer kernels"
        )

    return BitIdentityVerdict(
        device=str(device),
        scorer_fwd_seg_maxabs=float(scorer_fwd_seg_maxabs),
        scorer_fwd_argmax_flips=int(scorer_fwd_argmax_flips),
        scorer_fwd_pose_maxabs=float(scorer_fwd_pose_maxabs),
        reduction_order_grad_maxabs=float(reduction_order_grad_maxabs),
        scorer_fwd_speedup=float(scorer_fwd_speedup),
        scorer_forward_is_batch_invariant=bool(scorer_fwd_invariant),
        argmax_is_batch_invariant=bool(argmax_invariant),
        bit_identical_at_speedup_possible=bool(bit_possible),
        surviving_speedup_at_bit_identity=float(surviving),
        dominant_source=dominant,
        admission_path=admission,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reduction-order isolation (source B) — self-contained tiny witness + a batch-INVARIANT
# mock scorer (so source A == 0 by construction). Imports MLX lazily so the module is
# importable/inspectable without MLX.
# ─────────────────────────────────────────────────────────────────────────────
def _build_tiny_env(K: int, seed: int = 0):
    """A tiny witness + a linear (batch-INVARIANT) mock SegNet/PoseNet + K random pairs.
    Returns the arg bundle the twin's ``batched_realized_loss`` / ``single_realized_loss``
    consume. The mock scorer is linear per-pixel/per-frame so ``segnet(batch)[k]`` is
    bit-identical to ``segnet(batch[k:k+1])[0]`` -> isolates the reduction order."""
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    class _TinyWitness(nn.Module):
        def __init__(self, feat_dim, mod_dim, hidden, n_frames, n_classes=5, s=0):
            super().__init__()
            self.n_hidden = 1
            self.hidden_dim = hidden
            self.in_proj = nn.Linear(feat_dim, hidden)
            self.film = nn.Linear(mod_dim, self.n_hidden * 2 * hidden)
            self.hidden = [nn.Linear(hidden, hidden)]
            self.out_sdf = nn.Linear(hidden, n_classes)
            self.out_tex = nn.Linear(hidden, 3)
            rng = np.random.default_rng(s)
            self.code = mx.array(rng.standard_normal((n_frames, mod_dim)).astype(np.float32) * 0.5)

        def _trunk(self, cf, code_idx):
            h = nn.relu(self.in_proj(cf))
            film = mx.reshape(self.film(self.code[code_idx]), (self.n_hidden, 2, self.hidden_dim))
            for li, layer in enumerate(self.hidden):
                h = nn.relu(layer(h) * (1.0 + film[li, 0]) + film[li, 1])
            return h

        def sdf(self, cf, code_idx):
            return self.out_sdf(self._trunk(cf, code_idx))

        def __call__(self, cf, code_idx):
            return mx.sigmoid(self.out_tex(self._trunk(cf, code_idx))) * 255.0

    class _MockAdapter:
        def __init__(self, s=1):
            rng = np.random.default_rng(s)
            self.seg_w = mx.array(rng.standard_normal((3, 5)).astype(np.float32))
            self.seg_b = mx.array(rng.standard_normal((5,)).astype(np.float32))
            self.pose_w = mx.array(rng.standard_normal((12, 6)).astype(np.float32) * 0.01)

        def segnet(self, f):
            return f @ self.seg_w + self.seg_b

        def posenet(self, yuv):
            return {"pose": mx.mean(yuv, axis=(1, 2)) @ self.pose_w}

    rh, rw = 8, 12
    n_px = rh * rw
    feat_dim, mod_dim, hidden = 7, 4, 6
    rng = np.random.default_rng(seed + 7)
    model = _TinyWitness(feat_dim, mod_dim, hidden, 2 * K, s=seed)
    adapter = _MockAdapter(s=seed + 1)
    cf = mx.array(rng.standard_normal((n_px, feat_dim)).astype(np.float32))
    cf_list = [cf for _ in range(K)]
    c0_list = [2 * p + 0 for p in range(K)]
    c1_list = [2 * p + 1 for p in range(K)]
    oh_list, mg_list, pt_list = [], [], []
    for _ in range(K):
        arg = rng.integers(0, 5, size=(rh, rw))
        oh = np.eye(5, dtype=np.float32)[arg].reshape(1, rh, rw, 5)
        mg = rng.random((1, rh, rw)).astype(np.float32) * 0.5
        oh_list.append(mx.array(oh))
        mg_list.append(mx.array(mg))
        pt_list.append(mx.array(rng.standard_normal(6).astype(np.float32) * 0.01))
    mx.eval(model.parameters(), cf, *oh_list, *mg_list, *pt_list)
    return dict(model=model, adapter=adapter, rh=rh, rw=rw, cf_list=cf_list,
                c0_list=c0_list, c1_list=c1_list, oh_list=oh_list, mg_list=mg_list, pt_list=pt_list)


def _zero_eikonal_length(phi_pk, rh, rw, **kw):
    """Trivial batch-invariant eikonal/length stub (returns constant zeros). Used only with
    eik_w == len_w == 0 so it contributes nothing to the loss/grad; keeps the probe
    self-contained (no trainer/scipy import)."""
    import mlx.core as mx

    z = mx.zeros(())
    return z, z, None


def _zero_nuclear(code, **kw):
    import mlx.core as mx

    return mx.zeros(())


def _render_fn(model, cf, code_idx, rh, rw):
    import mlx.core as mx

    return mx.reshape(model(cf, int(code_idx)), (1, rh, rw, 3))


def measure_reduction_order_drift(K: int = 4, seg_form: str = "ce", seed: int | None = None,
                                  *, w_seg: float = 100.0, w_pose: float = 1.0,
                                  hinge: float = 4.0, mtgt: float = 0.5) -> ReductionOrderDrift:
    """Measure the PURE reduction/accumulation-order drift (source B): the batched twin's
    grad vs the serial explicit left-fold mean-of-per-pair grad, using a batch-INVARIANT
    mock scorer so the scorer-forward kernel contributes exactly 0.0.

    Run on MLX CPU (deterministic) — the caller sets the device. Returns the max absolute
    per-leaf grad difference (the trajectory-relevant metric) + the global-L2 rel err +
    the loss-scalar difference.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten, tree_map

    from tac.boundary_math.levelset_micro_batch_loss import (
        LeverConfig,
        batched_realized_loss,
        single_realized_loss,
    )

    env = _build_tiny_env(K, seed=(K if seed is None else seed))
    model = env["model"]
    lc = LeverConfig(seg_loss_default=seg_form, tau_use=0.3, l7_thr_use=0.42, l7_mult=4.0,
                     score_domain=True, pose_eps=1e-2,
                     eikonal_length=_zero_eikonal_length, nuclear_norm_smooth=_zero_nuclear)

    def _bfn(m):
        return batched_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"], env["c0_list"], env["c1_list"],
            env["oh_list"], env["mg_list"], env["pt_list"],
            w_seg, w_pose, hinge, mtgt, seg_form, 0.0, 0.0, lc)

    lb, gb = nn.value_and_grad(model, _bfn)(model)
    mx.eval(lb, gb)

    def _sfn(m, k):
        return single_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"][k], env["c0_list"][k], env["c1_list"][k],
            env["oh_list"][k], env["mg_list"][k], env["pt_list"][k],
            w_seg, w_pose, hinge, mtgt, seg_form, 0.0, 0.0, lc)

    accum = None
    lsum = 0.0
    for k in range(K):
        ls, gs = nn.value_and_grad(model, _sfn)(model, k)
        mx.eval(ls, gs)
        lsum += float(ls)
        accum = gs if accum is None else tree_map(lambda a, b: a + b, accum, gs)
        mx.eval(accum)
    mean_grad = tree_map(lambda g, c=float(K): g / c, accum)
    mx.eval(mean_grad)

    fb = dict(tree_flatten(gb))
    fm = dict(tree_flatten(mean_grad))
    grad_maxabs = 0.0
    worst = ""
    for key in fb:
        d = float(np.max(np.abs(np.asarray(fb[key], np.float64) - np.asarray(fm[key], np.float64))))
        if d >= grad_maxabs:
            grad_maxabs = d
            worst = key
    diff = np.concatenate([(np.asarray(fb[k], np.float64) - np.asarray(fm[k], np.float64)).ravel()
                           for k in fb])
    ref = np.concatenate([np.asarray(fm[k], np.float64).ravel() for k in fm])
    grad_rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12))
    loss_abs = abs(float(lb) - lsum / K)
    return ReductionOrderDrift(K=int(K), seg_form=str(seg_form), grad_maxabs=grad_maxabs,
                               grad_rel_l2=grad_rel_l2, loss_abs=loss_abs, worst_leaf=worst)
