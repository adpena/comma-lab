#!/usr/bin/env python3
"""ddm_rt2 -- MECHANISM decomposition of the render->SegNet round-trip seg loss, and the cure.

rt1 measured WHERE the round trip loses (a 1-px curve, 99.22% on the transmitted label
boundary, salt-and-pepper isolated single pixels, median logit deficit 0.105).  It did not
measure BY WHAT MECHANISM.  This tool does, and builds the one cure the mechanism implies.

THE MECHANISM UNDER TEST
------------------------
The scorer's downsample is `F.interpolate(x, size=(384,512), mode='bilinear')` --
`align_corners=False`, **no antialias** (`upstream/modules.py:109`).  The scale factor is
874/384 = 2.2760 rows and 1164/512 = 2.2734 cols, so each output cell covers ~5.17 camera
pixels but the kernel reads only a 2x2 tap spanning ~1 camera pixel.  That is a near
point-sample of a 2.28x-denser field: textbook ALIASING.  Two consequences, both testable:

  (a) ~22.7% of camera pixels carry exactly zero weight and are never read at all;
  (b) what IS read is a point sample whose value depends on where the taps land relative to
      the render's own sub-pixel edge -- which manufactures isolated single-pixel argmax
      disagreements, exactly the salt-and-pepper signature rt1 measured (mean run 1.110).

THE CURE (zero bytes, scorer-free, deterministic)
-------------------------------------------------
`D` is linear, separable, and SURJECTIVE (rank 384*512 out of 874*1164 camera DOF).  So for
any target T at scorer resolution there is an exact camera-resolution preimage.  Choose
T = a properly antialiased downsample of the render (area / triangle), then emit

    X' = X + R_dag @ (T - R X C^T) @ C_dag^T

which satisfies `D X' = T` exactly, in closed form, with two precomputed Gram solves.  Every
input is the decoded render itself: no scorer, no video-derived side data, no archive bytes.
It is generic algorithm in inflate.py -- FREE under rule 118.  The only loss is the uint8
box+grid at camera resolution, which the tool measures rather than assumes.

axis: [macOS-CPU advisory] frozen CPU-torch SegNet -- NEVER a score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"

FRAMES = 600
SEG_H, SEG_W = 384, 512
CAM_H, CAM_W = 874, 1164
SCORED_PX = FRAMES * SEG_H * SEG_W  # 117,964,800
N_CLASSES = 5
S_PER_FLIP = 100.0 / SCORED_PX  # 8.4771e-07

# --- provenance pins ------------------------------------------------------------------------
BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
D_SEG_CUDA = 2.9611e-04
SCORED_SEG_FLIPS_CUDA = 34930.6
RT1_ROUND_TRIP_FLIPS = 33743
RT1_SCORED_FLIPS = 34938

DEFAULT_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/runs/"
    "base_optimized_n600_r3/output/0.raw"
)
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/retained/coders/"
    "s1p25_c1p0/decoded_spatial_tokens.rc64.bin"
)
DEFAULT_GT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_rt2")

# every leg this tool can score.  `base` is the shipped round trip.
LEGS = ("base", "area", "aa", "area_pre", "aa_pre", "dither", "boxblur",
        "unsharp", "unsharp_pre")


class Rt2Error(RuntimeError):
    """Fail-closed error for custody / instrument violations."""


# ============================================================================================
# the scorer's own resampler, reimplemented exactly (so preimages can be solved in closed form)
# ============================================================================================
def bilinear_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Dense (n_out, n_in) matrix for torch `interpolate(mode='bilinear', align_corners=False)`.

    ATen `area_pixel_compute_source_index(scale, dst, align_corners=False, cubic=False)`:
        src = scale * (dst + 0.5) - 0.5, clamped below at 0
    then a 2-tap linear blend between floor(src) and floor(src)+1.
    """
    scale = n_in / n_out
    src = np.maximum(scale * (np.arange(n_out, dtype=np.float64) + 0.5) - 0.5, 0.0)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    frac = src - i0
    m = np.zeros((n_out, n_in), dtype=np.float64)
    rows = np.arange(n_out)
    np.add.at(m, (rows, i0), 1.0 - frac)
    np.add.at(m, (rows, i1), frac)
    return m


def area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Dense (n_out, n_in) exact box-average matrix -- torch `mode='area'` semantics."""
    m = np.zeros((n_out, n_in), dtype=np.float64)
    edges = np.arange(n_out + 1, dtype=np.float64) * n_in / n_out
    for o in range(n_out):
        lo, hi = edges[o], edges[o + 1]
        i0, i1 = int(np.floor(lo)), int(np.ceil(hi))
        for i in range(i0, min(i1, n_in)):
            m[o, i] = max(0.0, min(hi, i + 1.0) - max(lo, float(i)))
        s = m[o].sum()
        if s > 0:
            m[o] /= s
    return m


def triangle_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Dense (n_out, n_in) antialiased-bilinear matrix -- torch `antialias=True` semantics.

    Support is the linear kernel dilated by the downscale factor: w(t) = max(0, 1 - |t|/s).
    """
    scale = n_in / n_out
    s = max(scale, 1.0)
    centre = scale * (np.arange(n_out, dtype=np.float64) + 0.5) - 0.5
    m = np.zeros((n_out, n_in), dtype=np.float64)
    idx = np.arange(n_in, dtype=np.float64)
    for o in range(n_out):
        w = np.maximum(0.0, 1.0 - np.abs(idx - centre[o]) / s)
        tot = w.sum()
        if tot <= 0:
            raise Rt2Error(f"empty triangle kernel at output {o}")
        m[o] = w / tot
    return m


def min_norm_right_inverse(m: np.ndarray) -> np.ndarray:
    """`m_dag` with `m @ m_dag == I` -- the minimum-Frobenius-norm right inverse of a
    full-row-rank `m` (n_out, n_in).  Exact: `m^T (m m^T)^-1`."""
    gram = m @ m.T
    return np.linalg.solve(gram, m).T


class Resampler:
    """The scorer's `D`, its antialiased siblings, and the exact separable preimage solve."""

    def __init__(self) -> None:
        self.R = bilinear_matrix(CAM_H, SEG_H)
        self.C = bilinear_matrix(CAM_W, SEG_W)
        self.R_area = area_matrix(CAM_H, SEG_H)
        self.C_area = area_matrix(CAM_W, SEG_W)
        self.R_tri = triangle_matrix(CAM_H, SEG_H)
        self.C_tri = triangle_matrix(CAM_W, SEG_W)
        self.R_dag = min_norm_right_inverse(self.R)
        self.C_dag = min_norm_right_inverse(self.C)
        # fail-closed: the preimage identity must hold to machine precision.
        # (Accelerate raises spurious FP flags on these matmuls; the values are exact --
        # the Gram of a 2-tap bilinear matrix at scale > 2 is DIAGONAL, cond ~= 2.)
        self.identity_err = {}
        with np.errstate(all="ignore"):
            for name, (m, d) in {"row": (self.R, self.R_dag),
                                 "col": (self.C, self.C_dag)}.items():
                err = float(np.abs(m @ d - np.eye(m.shape[0])).max())
                self.identity_err[name] = err
                if not np.isfinite(err) or err > 1e-9:
                    raise Rt2Error(f"{name} right-inverse identity failed: max err {err:g}")

    @staticmethod
    def _apply(rm: np.ndarray, cm: np.ndarray, x: np.ndarray) -> np.ndarray:
        """(H,W,3) -> (rm.shape[0], cm.shape[0], 3) via rm @ x @ cm^T, per channel."""
        return np.einsum("oi,ijc,pj->opc", rm, x, cm, optimize=True)

    def down_scorer(self, x: np.ndarray) -> np.ndarray:
        return self._apply(self.R, self.C, x)

    def down_area(self, x: np.ndarray) -> np.ndarray:
        return self._apply(self.R_area, self.C_area, x)

    def down_triangle(self, x: np.ndarray) -> np.ndarray:
        return self._apply(self.R_tri, self.C_tri, x)

    def precompensate(self, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Camera-resolution field whose scorer downsample is exactly `target`.

        X' = X + R_dag (target - D X) C_dag^T.  Closed form, no iteration, no scorer.
        Returned as float64 BEFORE the uint8 box+grid is applied.
        """
        resid = target - self.down_scorer(x)
        # R_dag is (CAM_H, SEG_H) and C_dag is (CAM_W, SEG_W), so this is
        # R_dag @ resid @ C_dag^T -- the min-norm camera-resolution correction.
        return x + self._apply(self.R_dag, self.C_dag, resid)


# ============================================================================================
# instrument
# ============================================================================================
class SegInstrument:
    """Frozen CPU-torch SegNet, batch-1, upstream preprocessing verbatim (rt1's pins)."""

    def __init__(self, threads: int) -> None:
        import torch

        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        from modules import SegNet
        from safetensors.torch import load_file

        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self._torch = torch
        self.threads = threads
        net = SegNet().eval()
        net.load_state_dict(
            load_file(str(UPSTREAM / "models" / "segnet.safetensors"), device="cpu")
        )
        self.net = net

    def argmax_from_camera(self, frame_cam: np.ndarray) -> np.ndarray:
        """Full upstream path: (874,1164,3) -> SegNet.preprocess_input -> argmax (384,512)."""
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frame_cam, dtype=np.float32))
            x = x.permute(2, 0, 1)[None, None]  # b t c h w
            out = self.net(self.net.preprocess_input(x))
            return out.argmax(dim=1)[0].numpy().astype(np.uint8)

    def argmax_from_scorer(self, y: np.ndarray) -> np.ndarray:
        """Bypass the scorer's own resize: score a (384,512,3) field directly.

        Legal ONLY as a diagnostic -- it answers 'what would SegNet have said if it had been
        handed this field', which is exactly the counterfactual a preimage realizes.
        """
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
            x = x.permute(2, 0, 1)[None]
            return self.net(x).argmax(dim=1)[0].numpy().astype(np.uint8)


# ============================================================================================
# io
# ============================================================================================
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def open_raw(raw: Path) -> np.memmap:
    n = raw.stat().st_size // (CAM_H * CAM_W * 3)
    if n * CAM_H * CAM_W * 3 != raw.stat().st_size:
        raise Rt2Error(f"{raw} is not a whole number of camera frames")
    return np.memmap(raw, dtype=np.uint8, mode="r", shape=(n, CAM_H, CAM_W, 3))


def open_tokens(tokens: Path) -> np.memmap:
    if tokens.stat().st_size != FRAMES * SEG_H * SEG_W:
        raise Rt2Error(f"{tokens} is not {FRAMES}x{SEG_H}x{SEG_W} uint8")
    return np.memmap(tokens, dtype=np.uint8, mode="r", shape=(FRAMES, SEG_H, SEG_W))


def write_receipt(work: Path, name: str, payload: dict) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    p = work / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p


def seeded_pairs(n: int, seed: int) -> list[int]:
    """Seeded RANDOM pair ids -- never a prefix (m88/m96: a prefix is a different population)."""
    rng = np.random.default_rng(seed)
    return sorted(int(v) for v in rng.choice(FRAMES, size=n, replace=False))


# ============================================================================================
# legs
# ============================================================================================
def unsharp_target(x: np.ndarray, rs: Resampler, lam: float) -> np.ndarray:
    """Scorer-resolution UNSHARP MASK of the render, built from the render alone.

        Y' = D(X) + lam * (D(X) - A(X))

    `D` is the scorer's 2-tap read (effective support 2.47 camera px), `A` is the exact box
    average over the 5.17-px cell footprint.  `D - A` is therefore the high-frequency detail
    the scorer's narrow read already carries relative to a full-footprint integration, and
    `lam > 0` amplifies it.  Every operand is the decoded render: scorer-free, zero bytes.
    """
    return (1.0 + lam) * rs.down_scorer(x) - lam * rs.down_area(x)


def build_leg(leg: str, frame: np.ndarray, rs: Resampler, rng: np.random.Generator,
              *, quantize: bool, lam: float = 0.5) -> tuple[np.ndarray, dict]:
    """Return (camera-res field to score, diagnostics).

    `base`      the shipped uint8 render, untouched
    `area`      COUNTERFACTUAL: hand SegNet the box-average downsample directly
    `aa`        COUNTERFACTUAL: hand SegNet the triangle (antialias=True) downsample
    `area_pre`  THE CURE: camera-res preimage whose scorer bilinear read equals `area`
    `aa_pre`    THE CURE: same, targeting the triangle downsample
    `dither`    uint8 noise-floor read: re-round with a random sub-LSB offset
    `boxblur`   control: a 2x2 camera-res box blur (a naive "just prefilter" attempt)
    """
    x = frame.astype(np.float64)
    diag: dict = {}
    if leg == "base":
        return frame, diag
    if leg == "dither":
        # UINT8 NOISE-FLOOR READ.  The shipped field is X = round(X_float), so
        # X_float = X + e with e ~ U(-0.5, 0.5) to first order.  Feeding X + e (NOT re-rounded
        # -- re-rounding an integer is a no-op and would make this leg inert) scores a draw of
        # the pre-quantization render, so the flip delta estimates the uint8 mechanism's size.
        return np.clip(x + rng.uniform(-0.5, 0.5, x.shape), 0.0, 255.0), diag
    if leg == "boxblur":
        k = np.zeros_like(x)
        for dy in (0, 1):
            for dx in (0, 1):
                k += np.roll(np.roll(x, dy, axis=0), dx, axis=1)
        return np.clip(np.rint(k / 4.0), 0, 255).astype(np.uint8), diag
    if leg in ("area", "aa", "unsharp"):
        if leg == "area":
            return rs.down_area(x), diag
        if leg == "aa":
            return rs.down_triangle(x), diag
        return unsharp_target(x, rs, lam), diag
    if leg in ("area_pre", "aa_pre", "unsharp_pre"):
        if leg == "area_pre":
            target = rs.down_area(x)
        elif leg == "aa_pre":
            target = rs.down_triangle(x)
        else:
            target = unsharp_target(x, rs, lam)
        pre = rs.precompensate(x, target)
        diag["pre_min"] = float(pre.min())
        diag["pre_max"] = float(pre.max())
        diag["pre_frac_out_of_box"] = float(((pre < 0) | (pre > 255)).mean())
        diag["pre_max_abs_delta_vs_render"] = float(np.abs(pre - x).max())
        diag["pre_rms_delta_vs_render"] = float(np.sqrt(((pre - x) ** 2).mean()))
        if quantize:
            out = np.clip(np.rint(pre), 0, 255).astype(np.uint8)
            got = rs.down_scorer(out.astype(np.float64))
            diag["realized_max_abs_err_vs_target"] = float(np.abs(got - target).max())
            diag["realized_rms_err_vs_target"] = float(np.sqrt(((got - target) ** 2).mean()))
            return out, diag
        clipped = np.clip(pre, 0.0, 255.0)
        got = rs.down_scorer(clipped)
        diag["realized_max_abs_err_vs_target"] = float(np.abs(got - target).max())
        diag["realized_rms_err_vs_target"] = float(np.sqrt(((got - target) ** 2).mean()))
        return clipped, diag
    raise Rt2Error(f"unknown leg {leg!r}")


def score_leg(leg: str, args: argparse.Namespace, pairs: list[int]) -> dict:
    raw = open_raw(args.raw)
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")
    rs = Resampler()
    seg = SegInstrument(args.threads)
    rng = np.random.default_rng(args.seed + 7919)
    scorer_res_leg = leg in ("area", "aa")

    flips_label = flips_gt = 0
    n_px = 0
    per_pair = []
    diag_acc: dict[str, list[float]] = {}
    argmax_out = np.zeros((len(pairs), SEG_H, SEG_W), dtype=np.uint8)
    t0 = time.time()
    for k, t in enumerate(pairs):
        frame = np.asarray(raw[2 * t + 1], dtype=np.uint8)
        field, diag = build_leg(leg, frame, rs, rng,
                                quantize=not args.no_quantize, lam=args.lam)
        for key, val in diag.items():
            diag_acc.setdefault(key, []).append(val)
        pred = (seg.argmax_from_scorer(field) if scorer_res_leg
                else seg.argmax_from_camera(field))
        argmax_out[k] = pred
        lab = np.asarray(tok[t])
        g = np.asarray(gt[t])
        fl = int((pred != lab).sum())
        fg = int((pred != g).sum())
        flips_label += fl
        flips_gt += fg
        n_px += SEG_H * SEG_W
        per_pair.append({"pair": t, "flips_vs_label": fl, "flips_vs_gt": fg})
        if args.progress and (k + 1) % args.progress == 0:
            el = time.time() - t0
            print(f"  [{leg}] {k+1}/{len(pairs)} pairs  {el:.0f}s  "
                  f"gt={flips_gt} label={flips_label}", flush=True)

    scale = SCORED_PX / n_px
    receipt = {
        "schema": "ddm_rt2_leg.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "leg": leg,
        "lam": args.lam if leg.startswith("unsharp") else None,
        "n_pairs": len(pairs),
        "pairs_are_seeded_random_not_prefix": True,
        "seed": args.seed,
        "scored_at_scorer_resolution_bypassing_D": scorer_res_leg,
        "uint8_quantized": (not args.no_quantize),
        "pixels_scored": n_px,
        "flips_vs_label": flips_label,
        "flips_vs_gt": flips_gt,
        "d_seg_vs_gt": flips_gt / n_px,
        "S_vs_gt_extrapolated_n600": 100.0 * flips_gt * scale / SCORED_PX,
        "flips_vs_gt_extrapolated_n600": flips_gt * scale,
        "flips_vs_label_extrapolated_n600": flips_label * scale,
        "diagnostics_mean": {k: float(np.mean(v)) for k, v in diag_acc.items()},
        "diagnostics_max": {k: float(np.max(v)) for k, v in diag_acc.items()},
        "per_pair": per_pair,
        "wall_s": time.time() - t0,
        "instrument": {"backend": "cpu-torch", "batch_pairs": 1, "threads": args.threads},
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
    }
    lam_tag = f"_l{args.lam:g}" if leg.startswith("unsharp") else ""
    tag = (f"{leg}{lam_tag}_n{len(pairs)}_s{args.seed}"
           + ("_noq" if args.no_quantize else ""))
    npy = args.work / f"argmax_{tag}.npy"
    np.save(npy, argmax_out)
    receipt["payload"] = {
        "path": str(npy),
        "bytes": npy.stat().st_size,
        "sha256": sha256_file(npy),
        "shape": list(argmax_out.shape),
    }
    write_receipt(args.work, f"RT2_LEG_{tag}", receipt)
    return receipt


# ============================================================================================
# stages
# ============================================================================================
def stage_operator(args: argparse.Namespace) -> int:
    """Characterise D itself -- scorer-free, no frames touched."""
    rs = Resampler()
    rows_read = int((np.abs(rs.R) > 0).any(axis=0).sum())
    cols_read = int((np.abs(rs.C) > 0).any(axis=0).sum())
    # per-output-cell effective support: 1 / sum(w^2) is the participation ratio
    pr_row = float(np.mean(1.0 / (rs.R ** 2).sum(axis=1)))
    pr_col = float(np.mean(1.0 / (rs.C ** 2).sum(axis=1)))
    pr_row_area = float(np.mean(1.0 / (rs.R_area ** 2).sum(axis=1)))
    pr_col_area = float(np.mean(1.0 / (rs.C_area ** 2).sum(axis=1)))
    # how far the scorer's read is from the box average, as an operator distance
    row_gap = float(np.abs(rs.R - rs.R_area).sum(axis=1).mean())
    col_gap = float(np.abs(rs.C - rs.C_area).sum(axis=1).mean())
    receipt = {
        "schema": "ddm_rt2_operator.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "scale_rows": CAM_H / SEG_H,
        "scale_cols": CAM_W / SEG_W,
        "camera_rows_read": rows_read,
        "camera_cols_read": cols_read,
        "camera_pixels_read": rows_read * cols_read,
        "camera_pixels_total": CAM_H * CAM_W,
        "never_read_fraction": 1.0 - rows_read * cols_read / (CAM_H * CAM_W),
        "cell_footprint_camera_px": (CAM_H / SEG_H) * (CAM_W / SEG_W),
        "effective_taps_scorer_bilinear": pr_row * pr_col,
        "effective_taps_area_average": pr_row_area * pr_col_area,
        "aliasing_ratio_footprint_over_effective_taps":
            ((CAM_H / SEG_H) * (CAM_W / SEG_W)) / (pr_row * pr_col),
        "operator_L1_gap_vs_area_rows": row_gap,
        "operator_L1_gap_vs_area_cols": col_gap,
        "rank_D": SEG_H * SEG_W,
        "camera_dof": CAM_H * CAM_W,
        "preimage_dof_ratio": SEG_H * SEG_W / (CAM_H * CAM_W),
        "D_is_surjective_so_exact_preimage_exists": True,
    }
    write_receipt(args.work, "RT2_OPERATOR", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def stage_leg(args: argparse.Namespace) -> int:
    pairs = (list(range(FRAMES)) if args.n_pairs >= FRAMES
             else seeded_pairs(args.n_pairs, args.seed))
    rec = score_leg(args.leg, args, pairs)
    print(json.dumps({k: v for k, v in rec.items() if k != "per_pair"},
                     indent=2, sort_keys=True))
    return 0


def stage_ledger(args: argparse.Namespace) -> int:
    """Assemble every leg receipt present into one mechanism table."""
    rows = []
    base = None
    for p in sorted(args.work.glob("RT2_LEG_*.json")):
        rec = json.loads(p.read_text())
        if rec["n_pairs"] != args.n_pairs or rec["seed"] != args.seed:
            continue
        rows.append(rec)
        if rec["leg"] == "base":
            base = rec
    if base is None:
        raise Rt2Error(f"no base leg at n={args.n_pairs} seed={args.seed}; run it first")
    table = []
    for rec in sorted(rows, key=lambda r: r["flips_vs_gt"]):
        d_flips = rec["flips_vs_gt"] - base["flips_vs_gt"]
        table.append({
            "leg": rec["leg"] + (f"(lam={rec['lam']:g})" if rec.get("lam") is not None else ""),
            "quantized": rec["uint8_quantized"],
            "flips_vs_gt": rec["flips_vs_gt"],
            "flips_vs_label": rec["flips_vs_label"],
            "delta_flips_vs_base": d_flips,
            "delta_flips_n600_equivalent": d_flips * SCORED_PX / rec["pixels_scored"],
            "delta_S_n600_equivalent": d_flips * SCORED_PX / rec["pixels_scored"] * S_PER_FLIP,
            "pct_of_base_seg": 100.0 * d_flips / base["flips_vs_gt"],
            "realized_rms_err_vs_target":
                rec["diagnostics_mean"].get("realized_rms_err_vs_target"),
        })
    receipt = {
        "schema": "ddm_rt2_ledger.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "n_pairs": args.n_pairs,
        "seed": args.seed,
        "pairs_are_seeded_random_not_prefix": True,
        "base_flips_vs_gt": base["flips_vs_gt"],
        "base_flips_vs_label": base["flips_vs_label"],
        "S_per_flip_n600": S_PER_FLIP,
        "table": table,
    }
    write_receipt(args.work, f"RT2_LEDGER_n{args.n_pairs}_s{args.seed}", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["operator", "leg", "ledger"])
    ap.add_argument("--leg", choices=LEGS, default="base")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--no-quantize", action="store_true",
                    help="keep the precompensated field in float (upper bound on the cure)")
    ap.add_argument("--lam", type=float, default=0.5,
                    help="unsharp amount: Y = (1+lam)*D(X) - lam*A(X)")
    ap.add_argument("--progress", type=int, default=4)
    args = ap.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage == "operator":
        return stage_operator(args)
    if args.stage == "leg":
        return stage_leg(args)
    return stage_ledger(args)


if __name__ == "__main__":
    raise SystemExit(main())
