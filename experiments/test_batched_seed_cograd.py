"""BUILD #293 — BATCHED SEED CO-GRAD equivalence + guards (--micro-batch-pairs x --seed-islands).

The trainer (``experiments/train_levelset_witness_realized_through_R_mlx.py``) now composes the
measured 2-4x --micro-batch-pairs speed lever (#261) with the #224(5) --seed-islands dual co-grad
via ``_dual_vg_batch`` = ONE ``mx.value_and_grad(..., argnums=(0, 1))`` over BOTH param trees
(witness model + seed module) of the SAME ``total_loss_fn_batch`` the single-tree batched path
differentiates. The NO-FAKE gate is the EXECUTED equivalence proof, not the wiring's plausibility:

1. MECHANICS gate (mock scorer, fast, exact): dual-batched(B) loss + witness-grad + seed-grad ==
   mean of per-pair dual grads to fp32 machine precision (<= 1e-4 rel L2), for B in {2, 4} and an
   UNEVEN tail group. Proves the dual-vg wiring math (the accum invariant: per-group MEAN grads x
   pair count summed / nb == the serial per-pair mean, BOTH legs) is exact.
2. REAL-PATH proof (slow, $0, CPU): real gt_n6.npz + the REAL frozen MLX SegNet/PoseNet + the real
   ``render_through_R_mlx`` (bicubic-up -> uint8-STE @ camera -> bilinear-down) + the REAL
   island_protection seed build. MEASURED by THIS test (2026-07-04, MLX CPU, chunk=6, B=4 -> 4+2):
       loss rel                                  9.4e-08
       per-GROUP witness grad rel L2             <= 5.9e-05  per-GROUP seed grad <= 4.4e-05
       accum-step (mean-over-chunk) witness      2.01e-05,  seed 2.85e-05
   A sister scratch realization (different Fourier-basis draw) measured accum-step up to
   2.79e-04 witness / 3.40e-04 seed — the accum-step rel can exceed per-group NOT from added
   error but from denominator cancellation (per-pair grads partially cancel in the mean ->
   smaller ||mean|| denominator; absolute deviation unchanged). Decomposition controls (scratch
   protos, same config): no-seed single-tree 9.8e-06; seed-composited-constant single-tree
   8.5e-06; batched re-run bit-identical (0.0) -> the deviation class is the PRE-EXISTING
   batched-scorer fp reduction noise the --micro-batch-pairs opt-in already acknowledges
   (trajectory-affecting, GPU ~1e-3), NOT a dual-vg defect. Tolerances below cover the measured
   worst case with headroom and sit far under the acknowledged 1e-3 class.
3. SOURCE GUARDS: the 4 remaining micro-batch fail-closes (msal-reachability / seg-spike-reweight /
   seg-subpix / seg-chroma) are PRESERVED; the old seed fail-close is GONE (replaced by the dual
   build); the DEFAULT (--micro-batch-pairs 1) serial path incl. the serial ``_dual_vg`` dispatch
   is byte-untouched.
4. INVARIANT (synthetic, exact): group-weighted mean == per-pair mean on synthetic grad trees.

Run: .venv/bin/pytest experiments/test_batched_seed_cograd.py -v          (mock+guards, ~seconds)
     .venv/bin/pytest experiments/test_batched_seed_cograd.py -v -m ""    (incl. the slow real proof)
Memory note (report-only, NOT a gate): on the real-path smoke the in-process peak RSS after the
serial pass was 4.23 GiB and after the batched(B=4) pass 16.27 GiB (~+12.0 GiB; batched
EfficientNet-B2 backward activations at (4,384,512) + batched camera-res R buffers). The launch-side
memory preflight models the real B separately (tools/witness_memory_preflight.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
# MLX CPU: the equivalence AUTHORITY device (GPU is not bit-identical cross-process per
# [[mlx_gpu_not_bit_identical_crossprocess...]]; the sibling test_levelset_micro_batch_loss.py
# pins the same convention). GPU batched-fp noise (~1e-3) is validated end-to-end by the
# trajectory A/B, not here.
mx.set_default_device(mx.cpu)
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "upstream")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.boundary_math.levelset_micro_batch_loss import (
    LeverConfig,
    batched_realized_loss,
    single_realized_loss,
)

_TRAINER_PATH = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
_GT_N6 = _REPO / "experiments" / "results" / "mlx_fleet_gt_cache" / "gt_n6.npz"
_UPSTREAM = _REPO / "upstream"


# ===========================================================================
# helpers
# ===========================================================================
def _rel_l2(ga, gb) -> float:
    """Global relative L2 error between two grad pytrees (the sibling test's honest fp32
    metric; per-leaf ratios blow up on near-zero leaves via catastrophic cancellation)."""
    fa, fb = dict(tree_flatten(ga)), dict(tree_flatten(gb))
    assert set(fa.keys()) == set(fb.keys()), (set(fa) ^ set(fb))
    d = np.concatenate([(np.asarray(fa[k], np.float64) - np.asarray(fb[k], np.float64)).ravel()
                        for k in fa])
    r = np.concatenate([np.asarray(fb[k], np.float64).ravel() for k in fb])
    return float(np.linalg.norm(d) / (np.linalg.norm(r) + 1e-12))


def _lvl_trainer():
    import experiments.train_levelset_witness_realized_through_R_mlx as lvl

    return lvl


def _base_trainer():
    import experiments.train_witness_realized_through_R_mlx as base

    return base


class _SeedMod(nn.Module):
    """Mirror of the trainer's inline seed module (one 'residual' leaf, (P,H,W,3))."""

    def __init__(self, res):
        super().__init__()
        self.residual = mx.array(res)


def _accum_dual(dvg, model, seed_mod, chunk, group_size, call_args):
    """The trainer's accum-loop arithmetic for the DUAL path (BUILD #293): per group of
    ``group_size`` pairs, ONE dual value_and_grad -> group-MEAN grads (both legs) weighted by
    the group's pair count; summed; / nb. ``group_size=1`` == the serial per-pair reference.
    Returns (accum_step_loss, mean_witness_grads, mean_seed_grads, per_group_rows)."""
    accum = accum_seed = None
    lsum = 0.0
    rows = []
    for ss in range(0, len(chunk), group_size):
        sub = chunk[ss:ss + group_size]
        bn = len(sub)
        loss, (gw, gs) = dvg(model.trainable_parameters(), seed_mod.trainable_parameters(),
                             *call_args(sub))
        mx.eval(loss, gw, gs)
        lsum += float(loss) * bn
        wgw = tree_map(lambda g, c=float(bn): g * c, gw)
        wgs = tree_map(lambda g, c=float(bn): g * c, gs)
        accum = wgw if accum is None else tree_map(lambda a, b: a + b, accum, wgw)
        accum_seed = wgs if accum_seed is None else tree_map(lambda a, b: a + b, accum_seed, wgs)
        mx.eval(accum, accum_seed)
        rows.append((tuple(sub), float(loss), gw, gs))
    nb = float(max(len(chunk), 1))
    mean_w = tree_map(lambda g: g / nb, accum)
    mean_s = tree_map(lambda g: g / nb, accum_seed)
    mx.eval(mean_w, mean_s)
    return lsum / nb, mean_w, mean_s, rows


# ===========================================================================
# mock env (sibling-test pattern: batch-independent linear scorer, tiny witness w/ sdf)
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
            h = nn.relu(layer(h) * (1.0 + film[li, 0]) + film[li, 1])
        return h

    def sdf(self, cf, code_idx):
        return self.out_sdf(self._trunk(cf, code_idx))

    def __call__(self, cf, code_idx):
        return mx.sigmoid(self.out_tex(self._trunk(cf, code_idx))) * 255.0


class _MockAdapter:
    """Batch-independent linear SegNet/PoseNet (the eval-mode batching invariance the lever
    relies on holds EXACTLY here -> isolates the DUAL-vg MATH equivalence)."""

    def __init__(self, pose_dims=6, seed=1):
        rng = np.random.default_rng(seed)
        self.seg_w = mx.array(rng.standard_normal((3, 5)).astype(np.float32))
        self.seg_b = mx.array(rng.standard_normal((5,)).astype(np.float32))
        self.pose_w = mx.array(rng.standard_normal((12, pose_dims)).astype(np.float32) * 0.01)

    def segnet(self, f):
        return f @ self.seg_w + self.seg_b

    def posenet(self, yuv):
        return {"pose": mx.mean(yuv, axis=(1, 2)) @ self.pose_w}


def _mock_env(K, seed=0):
    lvl = _lvl_trainer()
    rh, rw = 8, 12
    n_px = rh * rw
    rng = np.random.default_rng(seed + 7)
    model = _TinyWitness(7, 4, 6, 2 * K, seed=seed)
    adapter = _MockAdapter(pose_dims=6, seed=seed + 1)
    cf = mx.array(rng.standard_normal((n_px, 7)).astype(np.float32))
    oh_l, mg_l, pt_l = [], [], []
    for _k in range(K):
        arg = rng.integers(0, 5, size=(rh, rw))
        oh_l.append(mx.array(np.eye(5, dtype=np.float32)[arg].reshape(1, rh, rw, 5)))
        mg_l.append(mx.array(rng.random((1, rh, rw)).astype(np.float32) * 0.5))
        pt_l.append(mx.array(rng.standard_normal(6).astype(np.float32) * 0.01))
    # seed residual on a sparse "island" mask (nonzero so the co-grad leg is exercised)
    res = (rng.standard_normal((K, rh, rw, 3)).astype(np.float32) * 3.0)
    msk = (rng.random((K, rh, rw, 1)) < 0.15).astype(np.float32)
    seed_mod = _SeedMod(res)
    masks = mx.array(msk)
    mx.eval(model.parameters(), seed_mod.parameters(), masks, cf, *oh_l, *mg_l, *pt_l)

    def compose_fn(rgb, code_idx):
        if int(code_idx) % 2 == 1:  # frame1 (SegNet-scored) only — mirrors _compose_chain
            pi = int(code_idx) // 2
            rgb = rgb + seed_mod.residual[pi] * masks[pi]
        return rgb

    def render_fn(m, cfe, code_idx, rh_, rw_):
        rgb = mx.reshape(m(cfe, int(code_idx)), (1, rh_, rw_, 3))
        return compose_fn(rgb, int(code_idx))

    lc = LeverConfig(seg_loss_default="ce", tau_use=0.3, l7_thr_use=0.42, l7_mult=4.0,
                     score_domain=True, pose_eps=1e-2,
                     eikonal_length=lvl._eikonal_length_mlx,
                     nuclear_norm_smooth=lvl._nuclear_norm_smooth_mlx)
    return {"model": model, "seed_mod": seed_mod, "adapter": adapter, "render_fn": render_fn,
            "rh": rh, "rw": rw, "cf": cf, "oh_l": oh_l, "mg_l": mg_l, "pt_l": pt_l, "lc": lc}


def _dual_fns(env, w_seg=100.0, w_pose=1.0, hinge=4.0, mtgt=0.5, seg_form="ce",
              eik_w=1e-2, len_w=1e-3):
    """Build the dual batched + dual per-pair value_and_grad closures (the trainer's
    _combined_seed_loss_batch pattern: in-place model/seed update, argnums=(0,1))."""
    model, seed_mod, lc = env["model"], env["seed_mod"], env["lc"]

    def _dual_b(model_p, seed_p, sub):
        model.update(model_p)
        seed_mod.update(seed_p)
        return batched_realized_loss(
            model, env["adapter"], env["render_fn"], env["rh"], env["rw"],
            [env["cf"]] * len(sub), [2 * p for p in sub], [2 * p + 1 for p in sub],
            [env["oh_l"][p] for p in sub], [env["mg_l"][p] for p in sub],
            [env["pt_l"][p] for p in sub],
            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc)

    def _dual_s(model_p, seed_p, sub):
        assert len(sub) == 1
        p = sub[0]
        model.update(model_p)
        seed_mod.update(seed_p)
        return single_realized_loss(
            model, env["adapter"], env["render_fn"], env["rh"], env["rw"],
            env["cf"], 2 * p, 2 * p + 1, env["oh_l"][p], env["mg_l"][p], env["pt_l"][p],
            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc)

    return (mx.value_and_grad(_dual_b, argnums=(0, 1)),
            mx.value_and_grad(_dual_s, argnums=(0, 1)))


# ===========================================================================
# TEST 1 — MECHANICS gate (mock, exact): dual-batched == mean of per-pair dual, B in {2,4},
#          uneven tail group included (chunk=6, B=4 -> groups 4+2).
# ===========================================================================
@pytest.mark.parametrize("group_size", [2, 4])
def test_dual_batched_grad_equals_mean_of_per_pair_dual_grads_mock(group_size):
    env = _mock_env(K=6, seed=group_size)
    dvg_b, dvg_s = _dual_fns(env)
    chunk = list(range(6))

    def _args(sub):
        return (sub,)

    lb, mw_b, ms_b, _ = _accum_dual(dvg_b, env["model"], env["seed_mod"], chunk, group_size, _args)
    ls, mw_s, ms_s, _ = _accum_dual(dvg_s, env["model"], env["seed_mod"], chunk, 1, _args)
    assert abs(lb - ls) / (abs(ls) + 1e-6) < 1e-4, (group_size, lb, ls)
    werr = _rel_l2(mw_b, mw_s)
    serr = _rel_l2(ms_b, ms_s)
    assert werr < 1e-4, f"B={group_size} witness dual-grad rel L2 {werr:.2e} >= 1e-4"
    assert serr < 1e-4, f"B={group_size} seed dual-grad rel L2 {serr:.2e} >= 1e-4"
    # the seed leg must actually FLOW (a zero seed grad would be a fake equivalence)
    sg = np.asarray(dict(tree_flatten(ms_s))["residual"], np.float64)
    assert np.linalg.norm(sg) > 0.0, "seed co-grad is identically zero (compose not wired?)"


# ===========================================================================
# TEST 2 — INVARIANT (synthetic, exact): group-weighted mean == per-pair mean, both legs.
# ===========================================================================
def test_group_weighted_mean_equals_per_pair_mean_synthetic():
    rng = np.random.default_rng(11)
    P = 6
    per_pair = [{"w": mx.array(rng.standard_normal((3, 2)).astype(np.float32)),
                 "s": mx.array(rng.standard_normal((4,)).astype(np.float32))} for _ in range(P)]
    # serial reference: sum / P
    acc = None
    for g in per_pair:
        acc = g if acc is None else tree_map(lambda a, b: a + b, acc, g)
    ref = tree_map(lambda g: g / float(P), acc)
    # grouped (4+2): per-group MEAN * count, summed, / P — the accum-loop arithmetic
    acc_g = None
    for ss in range(0, P, 4):
        sub = per_pair[ss:ss + 4]
        bn = len(sub)
        gsum = None
        for g in sub:
            gsum = g if gsum is None else tree_map(lambda a, b: a + b, gsum, g)
        gmean = tree_map(lambda g, c=float(bn): g / c, gsum)          # the group MEAN grad
        wg = tree_map(lambda g, c=float(bn): g * c, gmean)            # * count (trainer weighting)
        acc_g = wg if acc_g is None else tree_map(lambda a, b: a + b, acc_g, wg)
    grouped = tree_map(lambda g: g / float(P), acc_g)
    mx.eval(ref, grouped)
    assert _rel_l2(grouped, ref) < 1e-6


# ===========================================================================
# TEST 3 — REAL-PATH executed proof (slow; $0; CPU): real gt_n6 + real frozen MLX scorer +
#          real render_through_R_mlx + real island_protection seed build, chunk=6, B=4 (4+2).
# ===========================================================================
@pytest.mark.slow
@pytest.mark.timeout(900)  # real frozen-scorer fwd+bwd on CPU: measured ~130s serial+batched
@pytest.mark.skipif(not _GT_N6.exists(), reason="gt_n6.npz cache missing")
@pytest.mark.skipif(not (_UPSTREAM / "modules.py").exists(), reason="upstream snapshot missing")
def test_equivalence_real_gt_real_scorer_seed_islands():
    import resource

    import torch
    import torch.nn.functional as tF

    from tac.boundary_math.island_protection import (
        build_island_masks,
        build_island_seed,
        identify_island_classes,
    )
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )

    base = _base_trainer()
    lvl = _lvl_trainer()

    z = np.load(_GT_N6, allow_pickle=False)
    P = int(z["n_pairs"])
    RH, RW = 384, 512
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(_UPSTREAM, device="cpu")

    model = lvl.build_levelset_rgb_witness(
        num_pairs=P, in_feat=12, hidden_dim=8, n_hidden=1, mod_dim=4, n_classes=5,
        activation="relu", softmax_temp=1.0, wire_w0=8.0, wire_s0=4.0,
        hosc_beta=4.0, hosc_omega=1.0, chroma=True)
    rng = np.random.default_rng(0)
    model.code = mx.array(rng.standard_normal((P * 2, 4)).astype(np.float32) * 0.3)
    mx.eval(model.parameters())

    ys, xs = np.meshgrid(np.linspace(-1, 1, RH, dtype=np.float32),
                         np.linspace(-1, 1, RW, dtype=np.float32), indexing="ij")
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1)
    proj = coords @ (rng.standard_normal((6, 2)).astype(np.float32) * 2.0).T
    cf = mx.array(np.concatenate([np.sin(2 * np.pi * proj), np.cos(2 * np.pi * proj)], axis=1))
    mx.eval(cf)

    # trainer-faithful seed build (REAL lstars / gt_f1; self-detected island classes)
    det = identify_island_classes(
        np.stack([z["lstars"][pi].astype(np.int64) for pi in range(P)], axis=0), n_classes=5)
    seed_res = np.zeros((P, RH, RW, 3), np.float32)
    seed_msk = np.zeros((P, RH, RW, 1), np.float32)
    for pi in range(P):
        im = build_island_masks(z["lstars"][pi].astype(np.int64), det.lane_cls,
                                det.movable_cls, dilate_px=1)
        g1 = tF.interpolate(torch.from_numpy(z["gt_f1"][pi].astype(np.float32)).permute(2, 0, 1)[None],
                            size=(RH, RW), mode="bilinear", align_corners=False
                            )[0].permute(1, 2, 0).numpy()
        s = build_island_seed(g1, im, base_render_segres=None, blend=1.0)
        seed_res[pi] = s.residual
        seed_msk[pi, ..., 0] = im.any_mask.astype(np.float32)
    assert float(np.abs(seed_res).max()) > 0.0, "real seed residual is empty — no co-grad to prove"
    seed_mod = _SeedMod(seed_res)
    masks = mx.array(seed_msk)
    mx.eval(seed_mod.parameters(), masks)

    def compose_fn(rgb, code_idx):
        if int(code_idx) % 2 == 1:
            pi = int(code_idx) // 2
            rgb = rgb + seed_mod.residual[pi] * masks[pi]
        return rgb

    def render_fn(m, cfe, code_idx, rh, rw):
        return base.render_through_R_mlx(m, cfe, int(code_idx), rh, rw, compose_fn=compose_fn)

    lc = LeverConfig(seg_loss_default="ce", tau_use=0.3, l7_thr_use=0.42, l7_mult=4.0,
                     score_domain=True, pose_eps=1e-2,
                     eikonal_length=lvl._eikonal_length_mlx,
                     nuclear_norm_smooth=lvl._nuclear_norm_smooth_mlx)
    oh_l = [mx.array(np.eye(5, dtype=np.float32)[z["lstars"][pi].ravel()].reshape(1, RH, RW, 5))
            for pi in range(P)]
    mg_l = [mx.array(z["margins"][pi][None]) for pi in range(P)]
    pt_l = [mx.array(z["gt_poses"][pi].astype(np.float32)) for pi in range(P)]
    mx.eval(*oh_l, *mg_l, *pt_l)
    W_SEG, W_POSE, HINGE, MTGT = 100.0, 1.0, 4.0, 0.5

    def _dual_b(model_p, seed_p, sub):
        model.update(model_p)
        seed_mod.update(seed_p)
        return batched_realized_loss(
            model, adapter, render_fn, RH, RW,
            [cf] * len(sub), [2 * p for p in sub], [2 * p + 1 for p in sub],
            [oh_l[p] for p in sub], [mg_l[p] for p in sub], [pt_l[p] for p in sub],
            W_SEG, W_POSE, HINGE, MTGT, "ce", 1e-2, 1e-3, lc)

    def _dual_s(model_p, seed_p, sub):
        p = sub[0]
        model.update(model_p)
        seed_mod.update(seed_p)
        return single_realized_loss(
            model, adapter, render_fn, RH, RW, cf, 2 * p, 2 * p + 1,
            oh_l[p], mg_l[p], pt_l[p], W_SEG, W_POSE, HINGE, MTGT, "ce", 1e-2, 1e-3, lc)

    dvg_b = mx.value_and_grad(_dual_b, argnums=(0, 1))
    dvg_s = mx.value_and_grad(_dual_s, argnums=(0, 1))
    chunk = list(range(P))

    def _args(sub):
        return (sub,)

    ls, mw_s, ms_s, rows_s = _accum_dual(dvg_s, model, seed_mod, chunk, 1, _args)
    rss_serial = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    lb, mw_b, ms_b, rows_b = _accum_dual(dvg_b, model, seed_mod, chunk, 4, _args)
    rss_batched = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30

    loss_rel = abs(lb - ls) / (abs(ls) + 1e-12)
    werr = _rel_l2(mw_b, mw_s)
    serr = _rel_l2(ms_b, ms_s)
    # per-GROUP deviation (no mean-cancellation denominator): each batched group vs the mean of
    # its own per-pair rows.
    per_pair = {r[0][0]: r for r in rows_s}
    group_errs = []
    for sub, _gl, gw, gs in rows_b:
        aw = as_ = None
        for p in sub:
            _, _, gpw, gps = per_pair[p]
            aw = gpw if aw is None else tree_map(lambda a, b: a + b, aw, gpw)
            as_ = gps if as_ is None else tree_map(lambda a, b: a + b, as_, gps)
        n = float(len(sub))
        gmw = tree_map(lambda g, c=n: g / c, aw)
        gms = tree_map(lambda g, c=n: g / c, as_)
        mx.eval(gmw, gms)
        group_errs.append((sub, _rel_l2(gw, gmw), _rel_l2(gs, gms)))

    sg_norm = float(np.linalg.norm(np.asarray(dict(tree_flatten(ms_s))["residual"], np.float64)))
    print(f"\n[BUILD293-MEASURED] loss_serial={ls:.8f} loss_batched={lb:.8f} rel={loss_rel:.3e}")
    print(f"[BUILD293-MEASURED] accum-step witness_rel_L2={werr:.3e} seed_rel_L2={serr:.3e} "
          f"seed_grad_norm={sg_norm:.3e}")
    for sub, we, se in group_errs:
        print(f"[BUILD293-MEASURED] group {sub}: witness_rel={we:.3e} seed_rel={se:.3e}")
    print(f"[BUILD293-MEASURED] peakRSS after serial={rss_serial:.2f}GiB after batched(B=4)="
          f"{rss_batched:.2f}GiB (report-only; launch preflight models the real B)")

    assert sg_norm > 0.0, "seed co-grad identically zero on the real path"
    assert loss_rel < 1e-5, f"loss rel {loss_rel:.3e} (measured 9.4e-08)"
    for sub, we, se in group_errs:
        assert we < 2e-4, f"group {sub} witness rel {we:.3e} (measured <= 5.9e-05)"
        assert se < 2e-4, f"group {sub} seed rel {se:.3e} (measured <= 4.4e-05)"
    # accum-step (mean-over-chunk) tolerance is looser ONLY for the denominator-cancellation
    # effect documented in the module docstring (absolute deviation unchanged; sister scratch
    # realization measured up to 2.79e-04 / 3.40e-04).
    assert werr < 2e-3, f"accum-step witness rel {werr:.3e} (measured 2.01e-05..2.79e-04)"
    assert serr < 2e-3, f"accum-step seed rel {serr:.3e} (measured 2.85e-05..3.40e-04)"


# ===========================================================================
# TEST 4 — SOURCE GUARDS: remaining fail-closes preserved; old seed fail-close GONE;
#          default serial path untouched; dual build gated correctly.
# ===========================================================================
def _src() -> str:
    return _TRAINER_PATH.read_text(encoding="utf-8")


def test_fail_close_preserved_for_uncovered_configs():
    src = _src()
    # the 4 lever twins the batched loss does NOT consume remain REFUSED under micro-batch
    assert "--margin-saliency-reachability is not supported with --micro-batch-pairs>1" in src
    assert "--seg-spike-reweight is not supported with --micro-batch-pairs>1" in src
    assert "--seg-subpix-boundary-weight>0 is not supported with --micro-batch-pairs>1" in src
    assert "--seg-chroma-boundary-weight>0 is not supported with --micro-batch-pairs>1" in src
    # the OLD seed-islands fail-close is GONE (replaced by the BUILD #293 dual build)
    assert "is not supported together with --seed-islands" not in src
    # the dual build exists and is gated on BOTH micro-batch engagement AND seed presence
    assert "_dual_vg_batch = mx.value_and_grad(_combined_seed_loss_batch, argnums=(0, 1))" in src
    assert "if _use_micro_batch:" in src
    assert "if seed_mod is not None:" in src


def test_default_serial_path_untouched_source_guard():
    src = _src()
    # default OFF: engaged only at >1; both batched vgs None at default
    assert "_use_micro_batch = _micro_batch_pairs > 1" in src
    assert "value_and_grad_batch = None" in src
    assert "_dual_vg_batch = None" in src
    # the SERIAL per-pair loop incl. its serial dual dispatch is byte-untouched
    assert "if _dual_vg is None:" in src
    assert "loss, (grads, sgrads) = _dual_vg(" in src
    assert ("accum_seed = sgrads if accum_seed is None else "
            "tree_map(lambda a, b: a + b, accum_seed, sgrads)") in src
    # the batched loop dispatches dual vs single-tree and weights BOTH legs by pair count
    assert "if _dual_vg_batch is None:" in src
    assert "loss_b, (grads_b, sgrads_b) = _dual_vg_batch(" in src
    assert "_wsg = tree_map(lambda g, c=float(_bn): g * c, sgrads_b)" in src
    # the downstream seed step (shield -> seed_opt) is untouched
    assert '_sg = _seed_shield(_mean_sg["residual"], seed_mod.residual, seed_spec)' in src


def test_micro_batch_default_is_one_and_help_declares_composition():
    src = _src()
    i = src.index('"--micro-batch-pairs"')
    window = src[i:i + 900]
    assert "default=1" in window
    assert "Composes with" in window and "--seed-islands" in window
    assert "NOT with --seed-islands" not in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-m", "", *sys.argv[1:]]))
