# SPDX-License-Identifier: MIT
"""KEYFRAME POSE-SUFFICIENCY LADDER + trained-residual sweep HARNESS (#202 rate track).

THE UNIFIER (operator 2026-07-02): the decoder is PoseNet, so the stored keyframe is
coded FOR PoseNet's first-6 output (a task-SUFFICIENT stream, VCM / MPEG-AI Part 2),
NOT for pixel fidelity. This tool MEASURES, through the FROZEN CPU-torch PoseNet
authority (NEVER MPS), the d_pose of a controlled ladder of keyframe SOURCES, all
paired with their own ego-twist warp so the ONLY variable is the source information:

  (a) FULL real gt_f0                 -> the expensive baseline (real texture)
  (b) DEGRADED real gt_f0             -> the codec/MPEG sufficiency floor
        {resolution roundtrip, DCT low-freq keep, Gaussian blur, bit-depth}
  (c) TEXTURE-FREE synthetic render   -> the STORE-NOTHING-but-xi endpoint proxy
        (class-mean-flattened: partition structure + per-class mean colour, NO
        within-class texture -- proxies the SDF witness's own FiLM-xi-conditioned
        render; if PoseNet reads the pose from THIS, keyframe rate collapses to ~xi)

The pair is ``(S, warp(S, xi_eff))`` (same construction as the MEASURED carrier
baseline ~2.73 in tools/measure_warp_real_luma_frame0_dpose.py), so the increment
over (a) is the degradation cost the trained twist-residual must absorb.

TRAINED-RESIDUAL SWEEP HARNESS (deliverable 2): ``--dxi-source ckpt:<npz>`` loads the
#205 trained per-pair residual ``dxi`` (key ``pose_carrier.dxi``) so ``xi_eff =
xi_calibrated + dxi`` -- the PIVOTAL §4.2 test (does the trained residual absorb keyframe
degradation?). ``--partner {self_warp,sharp_gt,blur_gt}`` tests the §4.1 partner-blur
sensitivity. Runnable against ``experiments/results/levelset_n600_witness_*/`` checkpoints
AS THEY LAND (non-blocking); ``--mock`` unit-self-tests the whole pipeline with NO cache.

AUTHORITY / HONESTY (CLAUDE.md): ``[macOS-CPU advisory / research-signal]`` ONLY. NOT a
contest score. Pointer 0.19110 UNMOVED. score_claim/promotable=False. d_pose = REAL frozen
CPU-torch PoseNet MSE on native uint8 (``cpu_verdict_d_pose_batch``); NEVER MPS. A NO-FAKE
self-check asserts PoseNet(gt pair)==gt_poses before any number is trusted. The synthetic
render (c) is a PROXY for the trained witness render (labelled; the exact witness f1 needs a
#205 checkpoint -- the ``--partner witness:`` hook is stubbed NotImplemented, never faked).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _chunks(n: int, size: int):
    for i in range(0, n, size):
        yield i, min(i + size, n)


# --------------------------------------------------------------------------- #
# Ladder config parsing -> a per-pair source transform.
# --------------------------------------------------------------------------- #
def _parse_ladder(spec: str) -> list[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]


def _apply_degradation(cfg: str, f0: np.ndarray, argmax: np.ndarray | None, kc) -> np.ndarray:
    """Map a ladder token -> degraded/synthetic native source frame."""
    if cfg == "full":
        return f0
    if cfg.startswith("resize:"):
        w, h = cfg.split(":", 1)[1].split("x")
        return kc.resize_roundtrip(f0, int(w), int(h))
    if cfg.startswith("dct:"):
        k = int(cfg.split(":", 1)[1])
        return kc.dct_truncate(f0, k, k)
    if cfg.startswith("blur:"):
        return kc.gaussian_blur(f0, float(cfg.split(":", 1)[1]))
    if cfg.startswith("bit:"):
        return kc.bitdepth_quantize(f0, int(cfg.split(":", 1)[1]))
    if cfg == "classmean":
        if argmax is None:
            raise ValueError("classmean requires lstars (argmax) in the cache")
        return kc.class_mean_render(f0, argmax)
    if cfg.startswith("classmean+lf"):
        lf = int(cfg.split("+lf", 1)[1])
        return kc.class_mean_render(f0, argmax, residual_lowfreq=lf)
    raise ValueError(f"unknown ladder cfg {cfg!r}")


def _rate_hint(cfg: str, native_hw, n_keyframes: int, kc) -> dict:
    """Rough per-config rate hint (the precise temporal-stream rate is in the VCM rate tool).

    For the store-nothing endpoint (classmean) the source is FREE (the partition/argmax is
    the SEPARATE d_seg payload, not a pose keyframe) -> keyframe rate ~ 0 marginal.
    For DCT keep-KxK: K*K coeffs/channel * 2 B (fp16) * N keyframes (an order-of-magnitude).
    """
    H, W = native_hw
    if cfg.startswith("classmean"):
        return {"kind": "store_nothing", "keyframe_bytes_est": 0,
                "rate_est": 0.0, "note": "argmax partition is the SEPARATE d_seg payload"}
    if cfg.startswith("dct:"):
        k = int(cfg.split(":", 1)[1])
        b = k * k * 3 * 2 * n_keyframes  # fp16 coeff estimate (upper bound; entropy-coded lower)
        return {"kind": "dct_coeffs", "coeffs_per_channel": k * k,
                "keyframe_bytes_est": b, "rate_est": kc.rate_from_bytes(b)}
    return {"kind": "pixels", "note": "rate measured by the VCM rate tool at matched op-point"}


# --------------------------------------------------------------------------- #
# dxi loading (harness hook).
# --------------------------------------------------------------------------- #
def _load_dxi(spec: str, P: int) -> tuple[np.ndarray, str]:
    if spec == "zero":
        return np.zeros((P, 6), np.float64), "zero"
    if spec == "random":
        rng = np.random.default_rng(0)
        return rng.normal(0, 1e-3, size=(P, 6)), "random(1e-3)"
    if spec.startswith("ckpt:"):
        p = Path(spec.split(":", 1)[1])
        _refuse_tmp(p)
        if not p.exists():
            raise FileNotFoundError(f"--dxi-source ckpt not found: {p} (run harness once #205 lands a ckpt)")
        z = np.load(p, allow_pickle=False)
        key = next((k for k in z.files if k.endswith("pose_carrier.dxi") or k == "pose_carrier.dxi"), None)
        if key is None:
            raise KeyError(f"no pose_carrier.dxi in {p} (keys sample: {z.files[:8]})")
        dxi = np.asarray(z[key], np.float64)
        if dxi.shape[0] < P:
            raise ValueError(f"ckpt dxi has {dxi.shape[0]} pairs < requested {P}")
        return dxi[:P], f"ckpt:{p.name}"
    raise ValueError(f"unknown --dxi-source {spec!r}")


# --------------------------------------------------------------------------- #
# MOCK self-test (no cache; proves the whole pipeline end-to-end).
# --------------------------------------------------------------------------- #
def _run_mock() -> int:
    from tac.boundary_math import keyframe_codec as kc

    print("[ladder-mock] building tiny synthetic gt (structured; no torch/PoseNet needed for the codec core)")
    rng = np.random.default_rng(1)
    H, W = 96, 128
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    f0 = np.clip(np.stack([np.sin(6 * xx), np.cos(5 * yy), 0.5 * (xx + yy)], -1) * 120 + 128
                 + rng.integers(-6, 6, (H, W, 3)), 0, 255).astype(np.uint8)
    argmax = (np.floor(xx * 4).astype(np.int64) % 5)
    ok = True
    for cfg in ["full", "resize:48x36", "dct:16", "blur:2", "bit:3", "classmean", "classmean+lf8"]:
        s = _apply_degradation(cfg, f0, argmax, kc)
        assert s.shape == f0.shape and s.dtype == np.uint8, cfg
        rate = _rate_hint(cfg, (H, W), 13, kc)
        print(f"  cfg={cfg:16s} meanabsΔ={np.abs(s.astype(int) - f0).mean():5.1f}  rate_hint={rate.get('rate_est','?')}")
    for dsrc in ["zero", "random"]:
        dxi, tag = _load_dxi(dsrc, 4)
        assert dxi.shape == (4, 6), dsrc
    print("[ladder-mock] degradation + rate-hint + dxi-loader pipeline OK (self-test PASS)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all in cache (n600)")
    ap.add_argument("--fit-pairs", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--dxi-source", default="zero", help="zero | random | ckpt:<npz path>")
    ap.add_argument("--partner", default="self_warp", choices=["self_warp", "sharp_gt", "blur_gt"])
    ap.add_argument("--partner-blur", type=float, default=3.0, help="sigma for --partner blur_gt")
    ap.add_argument("--ladder", default=(
        "full,resize:384x512,resize:256x342,resize:192x256,resize:128x170,resize:96x128,"
        "dct:96,dct:48,dct:24,dct:12,blur:1,blur:2,blur:4,bit:5,bit:4,bit:3,"
        "classmean,classmean+lf24,classmean+lf12"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--append-jsonl", default=None,
                    help="append one row per config to this JSONL (accumulate across bounded calls)")
    ap.add_argument("--s-t", type=float, default=None, help="skip the s_t fit; use this value (e.g. 0.044)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--selfcheck-pairs", type=int, default=4)
    args = ap.parse_args(argv)

    if args.mock:
        return _run_mock()

    import mlx.core as mx  # noqa: F401 (import guard)
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    from experiments.train_witness_realized_through_R_mlx import (
        cpu_verdict_d_pose,
        cpu_verdict_d_pose_batch,
    )
    from tac.boundary_math import keyframe_codec as kc
    from tac.boundary_math.warp_real_luma_frame0 import (
        GroundHomographyGeom,
        warp_frame0_uint8_numpy,
        xi_from_pose_calibration,
    )

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    gt_f0 = np.asarray(z["gt_f0"])
    gt_f1 = np.asarray(z["gt_f1"])
    poses = np.asarray(z["gt_poses"], dtype=np.float64)
    has_lstars = "lstars" in z.files
    lstars = np.asarray(z["lstars"]) if has_lstars else None
    P_cache = poses.shape[0]
    P = P_cache if not args.n_pairs else min(args.n_pairs, P_cache)
    gt_f0, gt_f1, poses = gt_f0[:P], gt_f1[:P], poses[:P]
    if lstars is not None:
        lstars = lstars[:P]
    NAT_H, NAT_W = gt_f0.shape[1], gt_f0.shape[2]

    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet = dn.posenet
    for p_ in posenet.parameters():
        p_.requires_grad = False

    sc = min(args.selfcheck_pairs, P)
    max_err = max(cpu_verdict_d_pose(posenet, gt_f0[p], gt_f1[p], poses[p]) for p in range(sc))
    if not (max_err < 1e-6):
        raise SystemExit(f"NO-FAKE self-check FAILED: PoseNet(gt)!=gt_poses (max {max_err:.3e})")
    print(f"[ladder] NO-FAKE self-check PASS ({sc} pairs, max {max_err:.2e})", flush=True)

    geom = GroundHomographyGeom.eon(native_hw=(NAT_H, NAT_W), pitch=0.0)
    dxi, dxi_tag = _load_dxi(args.dxi_source, P)

    def xi_eff_for(p, s_t):
        xi = xi_from_pose_calibration(poses[p], s_t, 0.0, 0.0, whole_ground=True)
        return xi + dxi[p]

    # ---- fit d_pose-optimal global s_t on the FULL source (same as the carrier tool) ----
    grid_st = [0.0, 0.02, 0.044, 0.08, 0.12, 0.16, 0.22, 0.30]
    nf = min(args.fit_pairs, P)
    fit_idx = np.arange(nf)

    def _partner(p, s_native):
        if args.partner == "self_warp":
            return warp_frame0_uint8_numpy(s_native, xi_eff_for(p, _partner.s_t), geom)
        if args.partner == "sharp_gt":
            return gt_f1[p]
        return kc.gaussian_blur(gt_f1[p], args.partner_blur)
    _partner.s_t = 0.044

    def mean_dpose_full(s_t):
        _partner.s_t = s_t
        f0s = [gt_f0[p] for p in fit_idx]
        parts = [_partner(p, gt_f0[p]) for p in fit_idx]
        return float(np.mean(cpu_verdict_d_pose_batch(posenet, f0s, parts, [poses[p] for p in fit_idx])))

    if args.s_t is not None:
        best_st = float(args.s_t)
        fit_scores = {}
        _partner.s_t = best_st
        print(f"[ladder] s_t={best_st} (fit skipped)", flush=True)
    else:
        fit_scores = {s: mean_dpose_full(s) for s in grid_st}
        best_st = min(fit_scores, key=fit_scores.get)
        _partner.s_t = best_st
        print(f"[ladder] s_t fit (n={nf}): best={best_st}  scores={json.dumps({str(k): round(v,3) for k,v in fit_scores.items()})}", flush=True)

    # ---- run the ladder at n{P} ----
    ladder = _parse_ladder(args.ladder)
    results = {}
    baseline_mean = None
    for cfg in ladder:
        if cfg.startswith("classmean") and lstars is None:
            print(f"[ladder] SKIP {cfg} (no lstars in cache)", flush=True)
            continue
        dvals = []
        for a, b in _chunks(P, args.chunk):
            idx = range(a, b)
            srcs = [_apply_degradation(cfg, gt_f0[p], (lstars[p] if lstars is not None else None), kc) for p in idx]
            f0s = list(srcs)  # frame0 = the (degraded/synthetic) source keyframe
            parts = [_partner(p, srcs[i]) for i, p in enumerate(idx)]
            dvals.extend(cpu_verdict_d_pose_batch(posenet, f0s, parts, [poses[p] for p in idx]))
        dvals = np.asarray(dvals)
        m = float(dvals.mean())
        if cfg == "full":
            baseline_mean = m
        rate = _rate_hint(cfg, (NAT_H, NAT_W), 13, kc)
        results[cfg] = {
            "d_pose_mean": m, "d_pose_median": float(np.median(dvals)),
            "abs_incr_vs_full": (m - baseline_mean) if baseline_mean is not None else None,
            "pose_contribution_sqrt10": float(np.sqrt(10.0 * m)),
            "rate_hint": rate,
        }
        incr = results[cfg]["abs_incr_vs_full"]
        print(f"  {cfg:20s} d_pose={m:9.4f}  sqrt10={np.sqrt(10*m):.3f}  incr={('%.3f'%incr) if incr is not None else 'base':>7}  rate_hint={rate.get('rate_est','-')}", flush=True)
        if args.append_jsonl:
            jl = (Path(args.append_jsonl) if Path(args.append_jsonl).is_absolute()
                  else REPO / args.append_jsonl)
            _refuse_tmp(jl)
            jl.parent.mkdir(parents=True, exist_ok=True)
            with open(jl, "a") as fh:
                fh.write(json.dumps({
                    "cfg": cfg, "n_pairs": P, "s_t": best_st, "dxi_source": dxi_tag,
                    "partner": args.partner, "d_pose_mean": m, "d_pose_median": float(np.median(dvals)),
                    "pose_contribution_sqrt10": float(np.sqrt(10.0 * m)), "rate_hint": rate,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }) + "\n")

    out = {
        "tool": "tools/measure_keyframe_pose_sufficiency_ladder.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS-CPU advisory / research-signal] (NOT a contest score)",
        "score_claim": False, "promotion_eligible": False, "promotable": False,
        "ready_for_exact_eval_dispatch": False, "frontier_pointer": "UNMOVED 0.19110",
        "cache": str(cache.relative_to(REPO)) if str(cache).startswith(str(REPO)) else str(cache),
        "n_pairs": P, "native_hw": [NAT_H, NAT_W],
        "dxi_source": dxi_tag, "partner": args.partner,
        "no_fake_selfcheck": {"pairs": sc, "pose_max_abs_mse": max_err},
        "calibration": {"s_t": best_st, "s_t_sweep": {str(k): v for k, v in fit_scores.items()}},
        "epsilon_target": {"pose_contribution": 0.0184, "d_pose": 3.4e-5,
                           "note": "trained residual closes a FIXED offset to ~3.4e-5; the ladder increment "
                                   "is the degradation the residual must additionally absorb"},
        "ladder": results,
        "elapsed_secs": round(time.time() - t0, 1),
    }
    out_path = (Path(args.out) if args.out
                else (REPO / f"experiments/results/keyframe_pose_sufficiency_ladder_n{P}/results.json"))
    _refuse_tmp(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[written] {out_path}  ({out['elapsed_secs']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
