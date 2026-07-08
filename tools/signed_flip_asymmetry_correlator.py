# SPDX-License-Identifier: MIT
"""Signed per-class-pair per-DIRECTION field-vs-flip correlator (the standing ASYMMETRY instrument).

LAW + BAND PROVENANCE (T5 crucible, v5 draft `.omx/research/t5_crucible/DRAFT_OPTIMAL_STACK_v5_20260707.md`
S1.3 / S7b "Q1"; negatives review `.omx/research/t5_crucible/negatives_scale_validity_review_20260707.md` S7;
requirement L separatrix-asymmetry addendum):
  A pooled UNSIGNED correlation between a per-pixel cost/saliency field and flip incidence can
  average one-sided (per class-pair, per flip-direction) effects to zero. This instrument computes
  the point-biserial Pearson rho( field , flip ) SEPARATELY per ORDERED class pair (i -> j), where
  the population is the GT-class-i side of the (i,j) GT boundary annulus and the event is the
  witness actually flipping the pixel to class j. The two directions (i -> j erasure of i vs
  j -> i erasure of j) are DISTINCT populations and are never pooled.

  Pre-registered Q1 bands (BINDING; v5 S1.3 gate law for lever B16):
    FIRE   : |rho| >= 0.3 on ANY side of any major pair  -> B16 (signed one-sided hinge) enters
             the duty-to-measure queue with a real prior.
    KILL   : |rho| <  0.1 on BOTH sides of EVERY major pair -> the pooled null was NOT masking
             asymmetry; the texture/UNIWARD cost FORMULATION is SCALE-ROBUST dead.
             Kill SCOPE (requirement R): FORMULATION-level only (THIS cost field + linear
             point-biserial response + these class-pairs). It does NOT kill asymmetry-aware
             weighting as a FAMILY (untested reformulations: other cost fields, hinge vs linear
             response, per-range conditioning, rank correlation).
  Major pairs (spec minimum): Road<->Lane (0,1), Road<->Undrivable (0,2), Movable pairs
  ((0,3),(3,0),(2,3),(3,2)).

FIELDS (all per-pixel at 384x512, theta-independent, computed from cached artifacts only):
  texprox : the LEVER-4 msal_uni UNIWARD texture-cost proxy 1/(1+beta*tex), tex = per-frame
            max-normalized L1 luma gradient of R(x), x = bilinear-downsample(gt_f1). EXACT
            formula from `experiments/train_levelset_witness_realized_through_R_mlx.py`
            (msal_uni branch) and the proven probe `measure_SR_vs_texproxy.py` (memory
            `msal_uni_texture_proxy_inert_build_exact_sR_reachability_weight_20260703`).
  tex     : the raw normalized texture field (before the 1/(1+beta*tex) squash).
  sR      : the cached through-R fragility-weighted margin-Jacobian reachability
            (tools/precompute_sR_reachability.py sidecar, per-frame p99-normalized [0,1]) --
            POSITIVE-CONTROL sentinel: if sR is ALSO at chance vs flips the instrument is
            untrusted and no verdict is admissible (CLAUDE.md Confound self-protection L3).
  margin  : the cached GT top1-top2 margin -- second positive control (must anti-correlate
            with flips strongly; |rho| well above 0.3 expected).

FRAME ALIGNMENT (verified empirically 2026-07-08, byte-equality asserts below):
  witness maps frame w  ==  strided-cache frame 2w  ==  global n600 pair 6w.

ARTIFACT SCHEMA (stable JSON, full-precision floats; consumed by the T5 ledger + rerunnable
free on any later checkpoint's witness maps):
  { "schema": "signed_flip_asymmetry_correlator.v1", "generated_utc", "argv", "inputs": {...},
    "n_frames", "annulus_radius_px", "beta",
    "pairs": { "<i>-><j>": { "class_from", "class_to", "n_pixels", "n_flips", "flip_rate",
                              "rho": {field: float_full_precision}, "major": bool } },
    "bands": {...}, "verdict": {"fire": bool, "robust_dead": bool, "fired_sides": [...],
               "scope": "FORMULATION"} }

USAGE
  .venv/bin/python tools/signed_flip_asymmetry_correlator.py \
      --out experiments/results/t5_probe_waveB_20260708/q1_signed_asymmetry.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CAMERA_HW = (874, 1164)
SEG_HW = (384, 512)
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
# v5 S1.3 spec-minimum "major" ordered pairs (Road<->Lane, Road<->Undrivable, Movable pairs).
MAJOR_PAIRS = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (2, 3), (3, 2)]
FIRE_BAND = 0.3
KILL_BAND = 0.1


def _now_utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")):
        raise ValueError(f"{path!r} is a /tmp-class durable path; use the repo tier per CLAUDE.md.")


def compute_texture_fields(gt_f1_uint8: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """EXACT lever texture proxy on the R output (torch CPU; pure resampling, no scorer).

    Returns (tex, texprox) each (384,512) float64. Formula verbatim from the msal_uni branch:
    tex = (|dy lum| + |dx lum|) / max, texprox = 1/(1+beta*tex), lum = channel-mean of R(x).
    """
    import torch

    t = torch.from_numpy(np.asarray(gt_f1_uint8)).float().permute(2, 0, 1)[None]
    x = torch.nn.functional.interpolate(t, size=SEG_HW, mode="bilinear", align_corners=False)
    up = torch.nn.functional.interpolate(x, size=CAMERA_HW, mode="bicubic", align_corners=False)
    q = up.clamp(0.0, 255.0).round()  # uint8 quantization at camera res (forward-only; no grads needed)
    rx = torch.nn.functional.interpolate(q, size=SEG_HW, mode="bilinear", align_corners=False)
    lum = rx[0].mean(dim=0)  # (384,512)
    dy = torch.abs(lum[1:, :] - lum[:-1, :])
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    dx = torch.abs(lum[:, 1:] - lum[:, :-1])
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
    tex = (dy + dx).numpy().astype(np.float64)
    tex = tex / (tex.max() + 1e-6)
    texprox = 1.0 / (1.0 + beta * tex)
    return tex, texprox


def chebyshev_dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation with a (2r+1)^2 square structuring element (chebyshev radius r)."""
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, structure=np.ones((2 * r + 1, 2 * r + 1), dtype=bool))


class PairAccumulator:
    """Exact pooled point-biserial Pearson via running sums per (ordered pair, field)."""

    def __init__(self, field_names: list[str]):
        self.fields = field_names
        self.n = 0
        self.n_flip = 0
        self.sy = 0.0
        self.syy = 0.0
        self.sx = dict.fromkeys(field_names, 0.0)
        self.sxx = dict.fromkeys(field_names, 0.0)
        self.sxy = dict.fromkeys(field_names, 0.0)

    def add(self, fields: dict[str, np.ndarray], flip: np.ndarray) -> None:
        y = flip.astype(np.float64)
        self.n += int(y.size)
        self.n_flip += int(y.sum())
        self.sy += float(y.sum())
        self.syy += float((y * y).sum())
        for f in self.fields:
            x = fields[f].astype(np.float64)
            self.sx[f] += float(x.sum())
            self.sxx[f] += float((x * x).sum())
            self.sxy[f] += float((x * y).sum())

    def rho(self, f: str) -> float:
        n = self.n
        if n < 2:
            return float("nan")
        cov = self.sxy[f] - self.sx[f] * self.sy / n
        vx = self.sxx[f] - self.sx[f] ** 2 / n
        vy = self.syy - self.sy ** 2 / n
        if vx <= 0.0 or vy <= 0.0:
            return float("nan")
        return float(cov / np.sqrt(vx * vy))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz")
    ap.add_argument("--sr-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600_sR.npz",
                    help="global-n600 sR sidecar (per-frame p99-normalized [0,1])")
    ap.add_argument("--witness-dir", default="experiments/results/witness_per_stage_attribution")
    ap.add_argument("--witness-stage", default="MuonBest", help="maps_<stage>.npz supplying witness argmax")
    ap.add_argument("--witness-to-gt-stride", type=int, default=2,
                    help="witness frame w corresponds to gt-cache frame stride*w")
    ap.add_argument("--gt-to-global-stride", type=int, default=3,
                    help="gt-cache frame s corresponds to global n600 pair stride*s (sR indexing)")
    ap.add_argument("--n-frames", type=int, default=0, help="0 = all witness frames")
    ap.add_argument("--annulus-radius", type=int, default=2, help="chebyshev radius defining the (i,j) boundary annulus")
    ap.add_argument("--beta", type=float, default=4.0, help="msal_uni UNIWARD beta (lever default)")
    ap.add_argument("--min-pixels", type=int, default=500, help="report pairs with at least this population")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = (REPO / args.out) if not os.path.isabs(args.out) else Path(args.out)
    _refuse_tmp(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(REPO / args.gt_cache, mmap_mode="r")
    lstars, margins, gt_f1 = z["lstars"], z["margins"], z["gt_f1"]
    zr = np.load(REPO / args.sr_cache, mmap_mode="r")
    sR = zr["sR"]
    wdir = REPO / args.witness_dir
    wz = np.load(wdir / f"maps_{args.witness_stage}.npz")
    w_argmax = np.asarray(wz["argmax"])
    gt_arg_sub = np.load(wdir / "_gt_argmax_subset.npy", mmap_mode="r")

    n_w = w_argmax.shape[0] if args.n_frames <= 0 else min(args.n_frames, w_argmax.shape[0])
    fields_list = ["texprox", "tex", "sR", "margin"]
    acc: dict[tuple[int, int], PairAccumulator] = {}

    for w in range(n_w):
        s_idx = args.witness_to_gt_stride * w
        g_idx = args.gt_to_global_stride * s_idx
        gta = np.asarray(lstars[s_idx], dtype=np.int64)
        # ALIGNMENT ASSERT (fail loudly, never silently misalign -- CLAUDE.md no-silent-failures)
        if not np.array_equal(gta.astype(np.int8), np.asarray(gt_arg_sub[w])):
            raise RuntimeError(f"frame alignment broken at witness frame {w} (gt idx {s_idx})")
        wa = w_argmax[w].astype(np.int64)
        tex, texprox = compute_texture_fields(gt_f1[s_idx], args.beta)
        fr_fields = {
            "texprox": texprox,
            "tex": tex,
            "sR": np.asarray(sR[g_idx], dtype=np.float64),
            "margin": np.asarray(margins[s_idx], dtype=np.float64),
        }
        cls_present = np.unique(gta)
        for ci in cls_present:
            mask_i = gta == ci
            for cj in cls_present:
                if cj == ci:
                    continue
                near_j = chebyshev_dilate(gta == cj, args.annulus_radius)
                pop = mask_i & near_j & ((wa == ci) | (wa == cj))
                if not pop.any():
                    continue
                flip = (wa == cj)[pop]
                key = (int(ci), int(cj))
                if key not in acc:
                    acc[key] = PairAccumulator(fields_list)
                acc[key].add({f: fr_fields[f][pop] for f in fields_list}, flip)

    pairs_out: dict[str, dict] = {}
    fired_sides: list[str] = []
    kill_ok = True  # robust-dead requires |rho|<KILL_BAND on both sides of every major pair
    major_evaluated: set[tuple[int, int]] = set()  # a MISSING/NaN major side must BLOCK robust_dead,
    # never silently count as passing the kill (review fix, respawn 2026-07-08)
    for (ci, cj), a in sorted(acc.items()):
        if a.n < args.min_pixels:
            continue
        rho = {f: a.rho(f) for f in fields_list}
        is_major = (ci, cj) in MAJOR_PAIRS
        name = f"{ci}->{cj}"
        pairs_out[name] = {
            "class_from": CLASS_NAMES.get(ci, str(ci)),
            "class_to": CLASS_NAMES.get(cj, str(cj)),
            "n_pixels": a.n,
            "n_flips": a.n_flip,
            "flip_rate": a.n_flip / a.n,
            "rho": rho,
            "major": is_major,
        }
        if is_major and np.isfinite(rho["texprox"]):
            major_evaluated.add((ci, cj))
            if abs(rho["texprox"]) >= FIRE_BAND:
                fired_sides.append(name)
            if abs(rho["texprox"]) >= KILL_BAND:
                kill_ok = False
    missing_major = sorted(set(MAJOR_PAIRS) - major_evaluated)
    if missing_major:
        kill_ok = False  # cannot declare the formulation robust-dead on incomplete major coverage

    result = {
        "schema": "signed_flip_asymmetry_correlator.v1",
        "generated_utc": _now_utc(),
        "argv": sys.argv[1:],
        "inputs": {
            "gt_cache": args.gt_cache,
            "sr_cache": args.sr_cache,
            "witness_stage": args.witness_stage,
            "witness_dir": args.witness_dir,
        },
        "n_frames": n_w,
        "annulus_radius_px": args.annulus_radius,
        "beta": args.beta,
        "field_under_test": "texprox",
        "positive_controls": ["sR", "margin"],
        "bands": {
            "fire_abs_rho_gte": FIRE_BAND,
            "kill_abs_rho_lt_both_sides_all_major": KILL_BAND,
            "provenance": "v5 S1.3/S7b Q1; negatives_scale_validity_review S7",
        },
        "pairs": pairs_out,
        "verdict": {
            "fire": bool(fired_sides),
            "fired_sides": fired_sides,
            "missing_major_sides": [f"{a}->{b}" for a, b in missing_major],
            "robust_dead": bool(kill_ok and not fired_sides),
            "scope": "FORMULATION (this cost field + linear point-biserial response; "
                     "NOT asymmetry-aware weighting as a family -- requirement R)",
        },
        "advisory": "[macOS-numpy advisory . NON-PROMOTABLE] pointer 0.19110 UNMOVED",
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, out)
    print(f"[done] {out}")
    for name, row in pairs_out.items():
        if row["major"]:
            print(f"  MAJOR {name} ({row['class_from']}->{row['class_to']}): n={row['n_pixels']} "
                  f"flips={row['n_flips']} rho_texprox={row['rho']['texprox']:+.6f} "
                  f"rho_sR={row['rho']['sR']:+.6f} rho_margin={row['rho']['margin']:+.6f}")
    print(f"  VERDICT fire={result['verdict']['fire']} robust_dead={result['verdict']['robust_dead']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
