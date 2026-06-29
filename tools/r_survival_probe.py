# SPDX-License-Identifier: MIT
"""R-SURVIVAL PHYSICS probe (DAG FEED-iw / GAP-2 — the binding d_seg wall).

Isolates the RENDERING-SURVIVAL term of contest ``d_seg``: a *geometrically
correct* partition (the GT SegNet argmax ``L*``) can FLIP under the contest
render operator R::

    render (sub-camera) -> bicubic UP to camera 874x1164 -> uint8 @ camera
                        -> bilinear DOWN to scorer 384x512 -> argmax

R is the EXACT chain in
``tac.local_acceleration.pr95_hnerv_mlx_training.apply_contest_faithful_roundtrip_nhwc``
(verified against ``upstream/evaluate.py`` + ``modules.py:108-113``).  We
reproduce it with PyTorch ``align_corners=False`` bicubic/bilinear (the torch
authority the witness trainer uses, ``_torch_R_to_camera_uint8``).

WHAT THIS MEASURES (and what it does NOT):
  * MEASURES: how a *boundary representation* of a partition survives the
    double-resample + uint8 knife-edge.  The partition is encoded as a K=5
    channel membership carrier in [0,255]; R is applied per channel; the
    recovered partition is ``argmax_k R(C_k)``.  This is the topology-matched
    model of "the partition's continuous representation surviving R".
  * DOES NOT run the frozen SegNet.  This is a SegNet-FREE isolation of the
    *resampling/quantization* survival physics (the Nyquist/Gibbs/uint8 term),
    NOT the SegNet's RGB-reading nonlinearity.  It is an ADVISORY
    ``[macOS research-signal]`` measurement, never a contest score (pointer
    0.19110 unmoved).  The SegNet-in-loop confirmation is the witness trainer's
    realized-through-R verdict (``cpu_verdict_d_seg``).

DECOMPOSITION (the task's "isolate rendering-survival vs pre-R geometric cost"):
  * pre-R geometric cost = d_seg of ``argmax_k C_k`` (carrier at render res,
    nearest-resized to 384) vs L* -> for hard/sdf reps this is ~0 by
    construction (argmax preserves L*), so the ENTIRE measured d_seg is the
    rendering-survival term.
  * rendering-survival cost = d_seg of ``argmax_k R(C_k)`` vs L*.

NO FAKE: every number is a real argmax-disagreement (canonical
``tac.boundary_math.bitmask_dseg.d_seg_reference``) on the real cached GT
partition; reps that reproduce L* exactly pre-R are asserted to do so.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np

# Contest geometry (CLAUDE.md "Exact scorer architectures" + R chain above).
CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
N_CLASSES = 5
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}


def _torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def signed_distance_fields(labels: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Per-class signed distance ``phi_k`` (H,W,K): +EDT inside, -EDT outside.

    Canonical builder (copied from ``boundary_math.lever_b_levelset_generator``).
    ``argmax_k phi_k == labels`` EXACTLY.  1-Lipschitz (|grad phi|=1).
    """

    from scipy import ndimage

    a = np.asarray(labels)
    h, w = a.shape
    out = np.zeros((h, w, int(n_classes)), np.float32)
    for k in range(int(n_classes)):
        inside = a == k
        if inside.all():
            out[..., k] = float(max(h, w))
            continue
        if not inside.any():
            out[..., k] = -float(max(h, w))
            continue
        d_in = ndimage.distance_transform_edt(inside)
        d_out = ndimage.distance_transform_edt(~inside)
        out[..., k] = (d_in - d_out).astype(np.float32)
    return out


def _resize(x_kchw, size_hw, mode):
    """Per-channel resize of (K,H,W) float carrier with align_corners=False."""

    torch, F = _torch()
    t = torch.from_numpy(np.ascontiguousarray(x_kchw)).float().unsqueeze(0)  # (1,K,H,W)
    kw = {} if mode == "nearest" else {"align_corners": False}
    t = F.interpolate(t, size=size_hw, mode=mode, **kw)
    return t.squeeze(0).numpy()


def apply_R(carrier_khw: np.ndarray, render_hw: tuple[int, int]) -> np.ndarray:
    """Contest-EXACT R per channel: [bicubic UP to camera] -> uint8 -> bilinear DOWN to 384.

    ``carrier_khw``: (K, Hr, Wr) float in [0,255].  If render res == camera res,
    the bicubic-up is skipped (native sub-pixel render at 874).  Returns
    (K, 384, 512) float (the scorer-res carrier the argmax reads).
    """

    torch, F = _torch()
    x = carrier_khw
    if render_hw != (CAMERA_H, CAMERA_W):
        x = _resize(x, (CAMERA_H, CAMERA_W), "bicubic")
    # uint8 knife-edge at CAMERA res (the stored video frame).
    x = np.clip(np.round(x), 0.0, 255.0)
    x = _resize(x, (SEG_H, SEG_W), "bilinear")
    return x


# --------------------------------------------------------------------------- #
# Boundary representations: L* (384) -> carrier (K, Hr, Wr) in [0,255].
# --------------------------------------------------------------------------- #
def carrier_hard(lstar_384: np.ndarray, render_hw: tuple[int, int]) -> np.ndarray:
    """Hard 0/1 class indicator (the palette-paint / hard-argmax analog).

    Render@384: indicator at 384.  Render@874: NN-upsample labels to 874 then
    indicator (STAIRCASE boundary, quantized to the 384 grid).
    """

    rh, rw = render_hw
    if (rh, rw) != (SEG_H, SEG_W):
        lab = _resize(lstar_384[None].astype(np.float32), (rh, rw), "nearest")[0].round().astype(np.int64)
    else:
        lab = lstar_384
    K = N_CLASSES
    C = np.zeros((K, rh, rw), np.float32)
    for k in range(K):
        C[k] = 255.0 * (lab == k)
    return C


def carrier_sdf(lstar_384: np.ndarray, render_hw: tuple[int, int], slope: float) -> np.ndarray:
    """1-Lipschitz SDF ramp carrier: C_k = clamp(128 + slope*phi_k, 0, 255).

    Render@384: SDF computed at 384, ramped.  Render@874: SDF computed at 384
    then BICUBIC-upsampled to 874 (sub-pixel boundary placement: the zero-crossing
    of the smooth SDF lands between 384-grid samples), then ramped.

    ``slope`` (per-pixel) sets the boundary ramp half-width ~= 127/slope px.
    """

    rh, rw = render_hw
    phi = signed_distance_fields(lstar_384, N_CLASSES)  # (384,512,K)
    phi = np.transpose(phi, (2, 0, 1))  # (K,384,512)
    if (rh, rw) != (SEG_H, SEG_W):
        phi = _resize(phi, (rh, rw), "bicubic")  # sub-pixel zero-crossing at 874
    C = np.clip(128.0 + slope * phi, 0.0, 255.0).astype(np.float32)
    return C


def carrier_palette(lstar_384: np.ndarray, render_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """SINGLE shared channel: paint class k at gray level ``levels[k]`` (the naive
    palette-paint the witness would do in ONE image).  Returns (carrier (1,Hr,Wr),
    levels).  Recovery = nearest level.  This is the SHARED-channel coupling that
    box-averages a thin minority class into its majority neighbor -> the death.
    """

    rh, rw = render_hw
    levels = np.linspace(0.0, 255.0, N_CLASSES).astype(np.float32)  # evenly spaced gray levels
    if (rh, rw) != (SEG_H, SEG_W):
        lab = _resize(lstar_384[None].astype(np.float32), (rh, rw), "nearest")[0].round().astype(np.int64)
    else:
        lab = lstar_384
    img = levels[lab][None]  # (1,Hr,Wr)
    return img.astype(np.float32), levels.reshape(N_CLASSES, 1)  # protos (K,1)


def carrier_rgb3(lstar_384: np.ndarray, render_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """3-channel RGB, 5 maximally-separated prototype colors (the REAL witness
    constraint: 3 channels, 5 classes).  Recovery = nearest prototype color.
    """

    rh, rw = render_hw
    # 5 maximally-separated colors on the RGB cube (greedy max-min over corners/centroid).
    proto = np.array([
        [0, 0, 0],        # 0 Road
        [255, 255, 255],  # 1 Lane (bright -> still averaged when thin)
        [255, 0, 0],      # 2 Undrivable
        [0, 255, 0],      # 3 Movable
        [0, 0, 255],      # 4 MyCar
    ], np.float32)
    if (rh, rw) != (SEG_H, SEG_W):
        lab = _resize(lstar_384[None].astype(np.float32), (rh, rw), "nearest")[0].round().astype(np.int64)
    else:
        lab = lstar_384
    img = np.transpose(proto[lab], (2, 0, 1))  # (3,Hr,Wr)
    return img.astype(np.float32), proto


def _classify_nearest(field_chw: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """Nearest-prototype classify: field (C,H,W), protos (K,C) -> argmin dist (H,W)."""

    C, H, W = field_chw.shape
    x = field_chw.reshape(C, -1).T  # (HW, C)
    d = ((x[:, None, :] - protos[None, :, :]) ** 2).sum(-1)  # (HW, K)
    return d.argmin(1).reshape(H, W).astype(np.int64)


@dataclass
class RepResult:
    rep: str
    render_hw: tuple[int, int]
    slope: float | None
    pre_r_dseg: float
    survival_dseg: float
    per_class_flip_rate: dict[int, float] = field(default_factory=dict)
    per_class_present_frac: dict[int, float] = field(default_factory=dict)


def _per_class_flip(recovered: np.ndarray, lstar: np.ndarray) -> dict[int, float]:
    """Per-class flip rate = (# pixels whose GT class is k AND recovered != k) / (# GT class-k px)."""

    out = {}
    for k in range(N_CLASSES):
        m = lstar == k
        n = int(m.sum())
        if n == 0:
            out[k] = float("nan")
            continue
        out[k] = float(np.count_nonzero(recovered[m] != k)) / n
    return out


def measure(
    lstars: np.ndarray,
    reps: list[tuple[str, tuple[int, int], float | None]],
) -> dict:
    """Run every (rep, render_hw, slope) config over all frames in ``lstars`` (N,384,512)."""

    from tac.boundary_math.bitmask_dseg import d_seg_reference

    N = lstars.shape[0]
    agg: dict[str, dict] = {}
    for (rep, render_hw, slope) in reps:
        key = f"{rep}@{render_hw[0]}" + (f"|s{slope:g}" if slope is not None else "")
        pre_list, surv_list = [], []
        pc_flip = {k: [] for k in range(N_CLASSES)}
        pc_present = {k: 0 for k in range(N_CLASSES)}
        for i in range(N):
            lstar = lstars[i]
            for k in range(N_CLASSES):
                if (lstar == k).any():
                    pc_present[k] += 1
            protos = None
            if rep == "hard":
                C = carrier_hard(lstar, render_hw)
            elif rep == "sdf":
                C = carrier_sdf(lstar, render_hw, float(slope))
            elif rep == "palette":
                C, protos = carrier_palette(lstar, render_hw)
            elif rep == "rgb3":
                C, protos = carrier_rgb3(lstar, render_hw)
            else:
                raise ValueError(rep)

            def _recover(field):
                if protos is None:  # multi-channel membership -> argmax over class channels
                    return np.argmax(field, axis=0).astype(np.int64)
                # shared-channel / RGB -> nearest prototype
                return _classify_nearest(field, protos)

            # pre-R geometric: recover from carrier nearest-resized to 384.
            pre = _resize(C, (SEG_H, SEG_W), "nearest") if render_hw != (SEG_H, SEG_W) else C
            pre_list.append(d_seg_reference(_recover(pre), lstar))
            # rendering-survival: recover from R(C).
            R = apply_R(C, render_hw)
            rec = _recover(R)
            surv_list.append(d_seg_reference(rec, lstar))
            f = _per_class_flip(rec, lstar)
            for k in range(N_CLASSES):
                if not np.isnan(f[k]):
                    pc_flip[k].append(f[k])
        agg[key] = RepResult(
            rep=rep,
            render_hw=render_hw,
            slope=slope,
            pre_r_dseg=float(np.mean(pre_list)),
            survival_dseg=float(np.mean(surv_list)),
            per_class_flip_rate={k: (float(np.mean(v)) if v else float("nan")) for k, v in pc_flip.items()},
            per_class_present_frac={k: pc_present[k] / N for k in range(N_CLASSES)},
        )
    return agg


def lane_geometry_stats(lstars: np.ndarray, lane_cls: int = 1) -> dict:
    """Lane (class 1) width distribution at 384 (grounds the Nyquist argument)."""

    from scipy import ndimage

    widths = []
    areas = []
    for i in range(lstars.shape[0]):
        m = lstars[i] == lane_cls
        if not m.any():
            continue
        areas.append(float(m.mean()))
        # local width ~ 2 * EDT(inside) at the medial axis; use per-pixel 2*EDT over lane px.
        edt = ndimage.distance_transform_edt(m)
        widths.extend((2.0 * edt[m]).tolist())
    w = np.asarray(widths) if widths else np.zeros(1)
    return {
        "lane_area_frac_mean": float(np.mean(areas)) if areas else 0.0,
        "lane_width_px_median": float(np.median(w)),
        "lane_width_px_mean": float(np.mean(w)),
        "lane_width_px_p90": float(np.percentile(w, 90)),
        "lane_width_le_2px_frac": float(np.mean(w <= 2.0)),
        "lane_width_le_1px_frac": float(np.mean(w <= 1.0)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="R-survival physics probe (advisory [macOS research-signal]).")
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--n", type=int, default=96, help="# frames (subset for speed).")
    ap.add_argument("--slopes", default="255,127,64,32", help="SDF ramp slopes (per px).")
    ap.add_argument("--render-res", default="48,96,192,384,874",
                    help="Witness native render heights (capacity sweep). Width = round(h*512/384).")
    ap.add_argument("--stress-res", type=int, default=192,
                    help="Render height for the SDF ramp-slope sweep (where survival bites).")
    ap.add_argument("--out", default=None, help="JSON output path.")
    args = ap.parse_args(argv)

    d = np.load(args.gt_cache)
    lstars = d["lstars"][: args.n]
    print(f"[r_survival] loaded {lstars.shape[0]} frames; classes present: "
          f"{sorted(np.unique(lstars).tolist())}", flush=True)

    geo = lane_geometry_stats(lstars)
    print(f"[r_survival] lane geometry: {json.dumps(geo)}", flush=True)

    slopes = [float(s) for s in args.slopes.split(",")]
    render_hs = [int(h) for h in args.render_res.split(",")]
    def _hw(h):
        if h == CAMERA_H:
            return (CAMERA_H, CAMERA_W)
        return (h, int(round(h * SEG_W / SEG_H)))
    # Capacity x rep grid: at each render res, compare palette / hard / sdf(best slope).
    best_slope = slopes[len(slopes) // 2] if slopes else 64.0
    reps: list[tuple[str, tuple[int, int], float | None]] = []
    for h in render_hs:
        hw = _hw(h)
        reps.append(("palette", hw, None))
        reps.append(("hard", hw, None))
        reps.append(("sdf", hw, best_slope))
    # SDF ramp-slope sweep at the STRESS render res (where survival bites): which
    # ramp half-width best preserves the thin lane through R?
    stress_hw = _hw(int(args.stress_res))
    for s in slopes:
        reps.append(("sdf", stress_hw, s))

    t0 = time.time()
    agg = measure(lstars, reps)
    secs = time.time() - t0

    # Pretty table.
    print(f"\n[r_survival] === SURVIVAL TABLE (n={lstars.shape[0]}, {secs:.1f}s) ===", flush=True)
    hdr = f"{'config':<18}{'preR':>9}{'survival':>10}  " + "".join(f"{CLASS_NAMES[k][:4]:>7}" for k in range(N_CLASSES))
    print(hdr, flush=True)
    rows = {}
    for key, r in agg.items():
        pc = "".join(
            (f"{r.per_class_flip_rate[k]*100:>6.2f}%" if not np.isnan(r.per_class_flip_rate[k]) else f"{'--':>7}")
            for k in range(N_CLASSES)
        )
        print(f"{key:<18}{r.pre_r_dseg:>9.5f}{r.survival_dseg:>10.5f}  {pc}", flush=True)
        rows[key] = {
            "rep": r.rep, "render_h": r.render_hw[0], "slope": r.slope,
            "pre_r_dseg": r.pre_r_dseg, "survival_dseg": r.survival_dseg,
            "per_class_flip_rate": {str(k): r.per_class_flip_rate[k] for k in range(N_CLASSES)},
            "per_class_present_frac": {str(k): r.per_class_present_frac[k] for k in range(N_CLASSES)},
        }

    payload = {
        "evidence_grade": "macOS research-signal",
        "score_claim": False,
        "promotable": False,
        "n_frames": int(lstars.shape[0]),
        "R_chain": "bicubic_up_874x1164 -> uint8 @ camera -> bilinear_down_384x512 (align_corners=False)",
        "lane_geometry": geo,
        "rows": rows,
        "secs": secs,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[r_survival] wrote {args.out}", flush=True)
    return payload


if __name__ == "__main__":
    main()
