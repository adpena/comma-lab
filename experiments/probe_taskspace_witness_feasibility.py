# SPDX-License-Identifier: MIT
"""TASK-SPACE WITNESS feasibility probe (the make-or-break, $0 CPU, NO-FAKE).

The non-RGB task-space witness stores a ~8-dim/frame task-sufficient statistic and runs a
DETERMINISTIC parametric program at inflate.py to expand it to the SegNet argmax partition +
the directional boundary tangent field (the -48% d_seg lever, regenerated FREE at decode).
This probe measures whether that vehicle is feasible BEFORE any GPU arm:

  (A) STATIC-BASE / TEMPORAL STABILITY.  The partition is quasi-stationary (yousfi probe:
      identity beats pose-warp on adjacent pairs).  Per-pixel temporal mode -> the d_seg of a
      ZERO-per-frame-byte static base + how much of the frame is "active" (must be carried per
      frame).  This is the rate structure: static base = amortized (free); active band = the
      per-frame coords' job.

  (B) DIRECTIONAL-TANGENT BYTE-CLOSEABILITY (the memo's "one genuinely-new build").  The
      directional/curvelet basis needs the all-class boundary TANGENT FIELD.  Today it reads the
      GT argmax (unavailable at decode) -> NOT byte-closeable.  The fix: derive the tangent from
      a CHEAP partition the decoder can itself produce (the witness's OWN argmax, or a coarse
      parametric partition).  We measure cosine agreement between the GT-fine tangent and the
      tangent of (b1) the trained generator's own argmax, (b2) a coarse median-filtered partition,
      (b3) the per-pixel temporal-mode static base.  High agreement => the lever byte-closes.

  (C) PARAMETRIC COVERAGE (temporal amortization window).  Warp a base partition by a per-frame
      homography to nearby frames (whole-partition ICP+DLT) and measure d_seg vs identity over a
      growing temporal window -> the amortization window length + the residual the per-frame
      coords + a cheap residual coder must close.

EVIDENCE GRADE: [macOS-CPU advisory] / mathematical-derivation -- the EXACT frozen-SegNet GT
argmax (gt_segnet_argmax.u8), NOT the 600-sample contest harness on a rendered witness.  d_seg
here is the PARTITION-DOMAIN argmax-disagreement (the upper bound the realized-through-R witness
inherits), NOT a score claim.  NON-PROMOTABLE.  Frontier pointer UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROBE_SCHEMA = "taskspace_witness_feasibility_probe.v1"
PROVENANCE: dict[str, Any] = {
    "evidence_grade": "macOS-CPU advisory",
    "axis_tag": "[macOS-CPU advisory]",
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "hardware_substrate": "local_macos_cpu",
}

SEG_H, SEG_W = 384, 512
PX_PER_MAP = SEG_H * SEG_W
GOAL_DSEG = 1.12e-3       # operator goal
CAPSTONE_DSEG = 7.2e-4   # operator capstone

_SSD = Path("/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610")
_ARGMAX_U8 = _SSD / "targets_n600" / "gt_segnet_argmax.u8"
_GEN_NPZ = _SSD / "generator_ckpt" / "generator_n600.npz"


def _load_argmax(n_pairs: int) -> np.ndarray:
    """(n_pairs, 384, 512) uint8 GT SegNet argmax from the SSD evidence file."""
    total = 600
    n = min(n_pairs, total)
    raw = np.fromfile(_ARGMAX_U8, dtype=np.uint8, count=n * PX_PER_MAP)
    return raw.reshape(n, SEG_H, SEG_W)


# ---------------------------------------------------------------------------
# (A) static-base / temporal stability
# ---------------------------------------------------------------------------
@dataclass
class StaticBaseRow:
    n_pairs: int
    n_classes: int
    # per-pixel temporal mode (the static base); d_seg if it is used for EVERY frame
    d_seg_static_base: float
    # fraction of pixels that NEVER change class over the clip (free static region)
    frac_pixels_always_stable: float
    # fraction of pixels stable >= 99% of frames
    frac_pixels_stable_99: float
    # the "active band": pixels that change at least once -> carry ALL the per-frame d_seg
    active_band_frac: float
    active_band_carries_flip_frac: float  # = 1.0 by construction; sanity
    # mean per-pair flips vs the static base (how many px/frame the per-frame coords must fix)
    mean_flips_per_pair_vs_static: float
    # d_seg of "previous-frame predicts current" (temporal coherence baseline)
    d_seg_prev_frame_predictor: float


def measure_static_base(arr: np.ndarray, n_classes: int = 5) -> StaticBaseRow:
    n = arr.shape[0]
    # per-pixel class counts over time -> mode
    counts = np.zeros((n_classes, SEG_H, SEG_W), dtype=np.int32)
    for c in range(n_classes):
        counts[c] = (arr == c).sum(axis=0)
    mode = counts.argmax(axis=0).astype(np.uint8)  # (384,512) static base
    mode_count = counts.max(axis=0)                  # times the mode occurs
    stability = mode_count.astype(np.float64) / n    # per-pixel fraction-at-mode

    # d_seg of the static base used for every frame
    neq = arr != mode[None]                           # (n,384,512) bool
    d_seg_static = float(neq.mean())
    flips_per_pair = neq.reshape(n, -1).sum(axis=1)

    ever_changes = neq.any(axis=0)                    # (384,512) pixel changed at least once
    active_frac = float(ever_changes.mean())

    # prev-frame predictor d_seg (temporal coherence)
    if n >= 2:
        d_prev = float((arr[1:] != arr[:-1]).mean())
    else:
        d_prev = 0.0

    return StaticBaseRow(
        n_pairs=n,
        n_classes=n_classes,
        d_seg_static_base=d_seg_static,
        frac_pixels_always_stable=float((stability >= 1.0).mean()),
        frac_pixels_stable_99=float((stability >= 0.99).mean()),
        active_band_frac=active_frac,
        active_band_carries_flip_frac=1.0,
        mean_flips_per_pair_vs_static=float(flips_per_pair.mean()),
        d_seg_prev_frame_predictor=d_prev,
    )


# ---------------------------------------------------------------------------
# (B) directional-tangent byte-closeability
# ---------------------------------------------------------------------------
def _boundary_tangent(argmax_hw: np.ndarray, tau: float = 4.0):
    from tac.boundary_math.lever_b_generator import all_class_boundary_proximity_and_tangent

    return all_class_boundary_proximity_and_tangent(argmax_hw, tau=tau)


def _tangent_cosine_on_band(t_ref, t_cand, band_mask) -> float:
    """Mean |cos angle| between two unit tangent fields on the band (tangents are sign-free)."""
    a = t_ref[band_mask]
    b = t_cand[band_mask]
    if a.shape[0] == 0:
        return float("nan")
    dot = np.abs((a * b).sum(axis=-1))
    return float(np.clip(dot, 0.0, 1.0).mean())


def _coarse_partition(argmax_hw: np.ndarray, k: int = 5) -> np.ndarray:
    """A cheap decoder-reproducible partition proxy: per-pixel majority in a k x k window
    (a stand-in for what a coarse parametric / temporal-mode partition produces at decode)."""
    from scipy import ndimage

    out = np.zeros_like(argmax_hw)
    best = np.zeros(argmax_hw.shape, dtype=np.int32)
    for c in range(int(argmax_hw.max()) + 1):
        cnt = ndimage.uniform_filter((argmax_hw == c).astype(np.float32), size=k, mode="nearest")
        upd = cnt > best
        out[upd] = c
        best[upd] = cnt[upd].astype(np.int32) if False else cnt[upd]
    return out


@dataclass
class TangentRow:
    n_pairs: int
    tau: float
    # cosine agreement of GT-fine tangent vs the byte-closeable tangent sources, on the GT band
    cos_own_generator_argmax: float | None
    cos_coarse_majority_partition: float
    cos_static_mode_partition: float
    # how many boundary px the directional lever acts on (the codim-1 annulus)
    mean_boundary_px_per_pair: float
    byte_closeable_verdict: str


def measure_tangent_byte_closeability(arr: np.ndarray, mode_partition: np.ndarray,
                                      n_eval: int = 8, tau: float = 4.0) -> TangentRow:
    from tac.boundary_math.lever_b_generator import (
        all_class_boundary_mask,
        generator_argmax,
        build_coords,
        load_generator_npz,
    )

    n_eval = min(n_eval, arr.shape[0])
    # try to load the trained generator for the "own argmax" tangent source
    gen_params = gen_cfg = None
    coords = None
    if _GEN_NPZ.exists():
        try:
            gen_params, gen_cfg = load_generator_npz(_GEN_NPZ)
            coords = build_coords(SEG_H, SEG_W)
        except Exception:
            gen_params = None

    cos_own, cos_coarse, cos_mode, bpx = [], [], [], []
    for i in range(n_eval):
        L = arr[i].astype(np.int64)
        band = all_class_boundary_mask(L)
        bpx.append(int(band.sum()))
        _, t_ref = _boundary_tangent(L, tau=tau)

        # (b2) coarse majority partition (decoder-reproducible smoothing)
        Lc = _coarse_partition(L, k=5)
        _, t_coarse = _boundary_tangent(Lc, tau=tau)
        cos_coarse.append(_tangent_cosine_on_band(t_ref, t_coarse, band))

        # (b3) the per-pixel temporal-mode static partition (decoder has it for free)
        _, t_mode = _boundary_tangent(mode_partition.astype(np.int64), tau=tau)
        cos_mode.append(_tangent_cosine_on_band(t_ref, t_mode, band))

        # (b1) the trained generator's OWN argmax (the self-orientation fixed point)
        if gen_params is not None:
            try:
                Lg = generator_argmax(gen_params, gen_cfg, coords, i, SEG_H, SEG_W).astype(np.int64)
                _, t_gen = _boundary_tangent(Lg, tau=tau)
                cos_own.append(_tangent_cosine_on_band(t_ref, t_gen, band))
            except Exception:
                pass

    own = float(np.mean(cos_own)) if cos_own else None
    coarse = float(np.mean(cos_coarse))
    modec = float(np.mean(cos_mode))
    best = max([x for x in (own, coarse, modec) if x is not None])
    if best >= 0.85:
        verdict = "BYTE_CLOSEABLE_directional_lever_unlocked"
    elif best >= 0.7:
        verdict = "PARTIAL_tangent_usable_with_residual"
    else:
        verdict = "WALL_cheap_tangent_too_noisy"
    return TangentRow(
        n_pairs=n_eval,
        tau=tau,
        cos_own_generator_argmax=own,
        cos_coarse_majority_partition=coarse,
        cos_static_mode_partition=modec,
        mean_boundary_px_per_pair=float(np.mean(bpx)),
        byte_closeable_verdict=verdict,
    )


# ---------------------------------------------------------------------------
# (C) parametric coverage: temporal-window homography warp of a base partition
# ---------------------------------------------------------------------------
def _edge_points(argmax_hw: np.ndarray) -> np.ndarray:
    from tac.boundary_math.lever_b_generator import all_class_boundary_mask

    band = all_class_boundary_mask(argmax_hw)
    ys, xs = np.nonzero(band)
    return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)


def _apply_h(H, pts):
    if len(pts) == 0:
        return pts
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = ph @ H.T
        w = out[:, 2:3]
        w = np.where(np.abs(w) < 1e-9, 1e-9, w)
        out = out[:, :2] / w
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def _dlt(src, dst):
    n = len(src)
    if n < 4:
        return None
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def _fit_h(src, dst, iters=6):
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(dst)
    except Exception:
        return np.eye(3)
    H = np.eye(3)
    s = src if len(src) <= 3000 else src[:: max(1, len(src) // 3000)]
    for _ in range(iters):
        warped = _apply_h(H, s)
        _, idx = tree.query(warped, k=1)
        Hn = _dlt(s, dst[idx])
        if Hn is None:
            break
        H = Hn
    return H


def _warp_labels(labels: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp a (384,512) label field by H^{-1} (pull each dst pixel from src), NN sample."""
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return labels
    ys, xs = np.mgrid[0:SEG_H, 0:SEG_W]
    dst = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    src = _apply_h(Hinv, dst)
    sx = np.clip(np.round(src[:, 0]).astype(int), 0, SEG_W - 1)
    sy = np.clip(np.round(src[:, 1]).astype(int), 0, SEG_H - 1)
    return labels[sy, sx].reshape(SEG_H, SEG_W)


@dataclass
class ParamCoverageRow:
    window: int
    d_seg_identity_mean: float       # base partition reused as-is for the next `window` frames
    d_seg_homography_warp_mean: float  # base partition warped by per-frame homography
    homography_beats_identity: bool
    n_evaluated: int


def measure_parametric_coverage(arr: np.ndarray, windows=(1, 4, 16, 64),
                                base_stride: int = 50) -> list[ParamCoverageRow]:
    n = arr.shape[0]
    rows: list[ParamCoverageRow] = []
    for w in windows:
        id_ds, h_ds, cnt = [], [], 0
        base_idxs = list(range(0, n - w, base_stride))
        for b in base_idxs:
            base = arr[b].astype(np.int64)
            src = _edge_points(base)
            t = min(b + w, n - 1)
            tgt = arr[t].astype(np.int64)
            id_ds.append(float((base != tgt).mean()))
            dst = _edge_points(tgt)
            if len(src) >= 8 and len(dst) >= 8:
                H = _fit_h(src, dst)
                warped = _warp_labels(base, H)
                h_ds.append(float((warped != tgt).mean()))
            cnt += 1
        rows.append(ParamCoverageRow(
            window=w,
            d_seg_identity_mean=float(np.mean(id_ds)) if id_ds else 0.0,
            d_seg_homography_warp_mean=float(np.mean(h_ds)) if h_ds else 0.0,
            homography_beats_identity=bool(np.mean(h_ds) < np.mean(id_ds)) if h_ds and id_ds else False,
            n_evaluated=cnt,
        ))
    return rows


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
@dataclass
class FeasReport:
    schema: str = PROBE_SCHEMA
    provenance: dict[str, Any] = field(default_factory=lambda: dict(PROVENANCE))
    static_base: dict[str, Any] = field(default_factory=dict)
    tangent: dict[str, Any] = field(default_factory=dict)
    parametric_coverage: list[dict] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


def run(n_pairs: int, n_tangent: int) -> FeasReport:
    t0 = time.time()
    arr = _load_argmax(n_pairs)

    sb = measure_static_base(arr)
    # rebuild mode partition for tangent reuse
    counts = np.stack([(arr == c).sum(0) for c in range(5)])
    mode = counts.argmax(0).astype(np.uint8)

    tg = measure_tangent_byte_closeability(arr, mode, n_eval=n_tangent)
    pc = measure_parametric_coverage(arr)

    # synthesis: the active-band carries d_seg; the per-frame coords + residual must drive the
    # static-base d_seg down to GOAL/CAPSTONE. report the gap.
    summary = {
        "static_base_d_seg": sb.d_seg_static_base,
        "static_base_vs_goal_ratio": sb.d_seg_static_base / GOAL_DSEG,
        "static_base_vs_capstone_ratio": sb.d_seg_static_base / CAPSTONE_DSEG,
        "active_band_frac": sb.active_band_frac,
        "free_static_region_frac": sb.frac_pixels_stable_99,
        "directional_tangent_verdict": tg.byte_closeable_verdict,
        "best_cheap_tangent_cosine": max(
            x for x in (tg.cos_own_generator_argmax, tg.cos_coarse_majority_partition,
                        tg.cos_static_mode_partition) if x is not None),
        "parametric_best_window_residual": min((r.d_seg_homography_warp_mean for r in pc), default=None),
        "goal_d_seg": GOAL_DSEG,
        "capstone_d_seg": CAPSTONE_DSEG,
        "frontier_pointer_moved": False,
        "authority": "[macOS-CPU advisory] partition-domain d_seg (frozen-SegNet GT argmax), NOT 600-sample render harness",
    }
    return FeasReport(
        static_base=asdict(sb),
        tangent=asdict(tg),
        parametric_coverage=[asdict(r) for r in pc],
        summary=summary,
        seconds=round(time.time() - t0, 2),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--n-tangent", type=int, default=8)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)

    rep = run(args.n_pairs, args.n_tangent)
    blob = json.dumps(asdict(rep), indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(blob)
        print(f"wrote {out}")
    print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
