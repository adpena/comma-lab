#!/usr/bin/env python
"""ddm_pi2 -- attribute the 18.2x advisory-vs-contest-CUDA d_pose offset measured by ddm_rn1.

rn1 (`.omx/research/ddm_rn1_render_boundary_mechanism_20260816.md` s3.2b) measured its advisory
CPU pose instrument at n=96 identity giving d_pose 1.251833e-04 on the hv1 ep0634 object whose
contest-CUDA n600 authority value is 6.880000e-06 -- an 18.2x offset that did NOT converge with n.
rn1 named two candidate causes and could not separate them:

    (A) GT-DECODE-PATH drift  -- upstream/evaluate.py routes GT through DaliVideoDataset (nvdec)
        on cuda and through AVVideoDataset -> frame_utils.yuv420_to_rgb (a CPU reimplementation
        whose docstring only CLAIMS to match nvdec) on cpu.
    (B) CPU/CUDA SCORER-FORWARD drift -- the PoseNet forward itself differing across axes.

A third cause exists that neither rn1 nor its charter named, and it must be carried because it
falsifies the obvious reason for dismissing it: our OWN frames are NOT byte-identical across
axes.  `TensorVideoDataset` is a raw mmap on both axes, but it reads whatever `inflate.sh` wrote,
and ddm_ps1u measured the same archive inflating to raw sha `e5539653...` on cpu and
`9a6b75e5...` on cuda (`DEVICE_DEPENDENT_DECODE_CONFIRMED`).  So:

    (C) OUR-DECODE device dependence -- the shipped receiver rendering different frames per host.

This tool separates all three without buying a single CUDA row.

The decomposition rests on the measurements this tool makes:

  1. `scale`  -- the magnitude |P| of the PoseNet pose vector, and the per-pair d_pose
     distribution.  d_pose is an MSE on the 6-vector, so a relative forward drift eps injects
     roughly `eps * |P| * sqrt(2)` per component.  Inverting the offset gives the relative
     forward drift that would be REQUIRED to explain it; comparing that against the known fp32
     CPU-vs-CUDA drift scale falsifies (B) quantitatively rather than by assertion.
     The per-pair rows also give a bootstrap CI, which is the honest test of "subset draw".

  2. `attrib` -- decode the SAME GT frames under a family of legal decoder conventions
     (chroma-upsample siting, rounding, matrix, range) plus a calibrated noise ladder, and score
     each variant's d_pose AND d_seg against our fixed frames.  This measures how much a decoder
     convention difference is worth in d_pose, and -- critically -- shows whether any convention
     lands at the contest-CUDA value.  d_seg is carried on every row as the CONTROL: it is the
     reason the seg half of the instrument looked sound while the pose half was 18x off.

  3. `n600`   -- the identity baseline over all 600 pairs with per-pair values retained, which
     settles subset-draw versus axis-gap outright instead of extrapolating from n=96.

  4. `crossaxis` -- the decisive leg.  A retained pair of T4 GT caches (job #906) holds the GT
     scorer outputs under BOTH decode paths, computed on ONE host with ONE frozen scorer, so
     scoring our frames against each isolates (A) exactly and leaves (B)+(C) as the residual.

  5. `segaxis` -- the same cross-axis test on the seg half, which is the control everyone leaned
     on when they concluded the instrument was sound.

Every number here is `[macOS-CPU advisory]`.  Nothing in this file is a score, and no row may
promote, rank, or kill.  Instrument pins are rn1-identical (frozen CPU torch scorers from
upstream/models/*.safetensors, batch = 1 pair, upstream preprocess verbatim, threads = 8, seeded
random pairs at seed 20260816) so every row is leg-to-leg comparable with rn1's ledger.
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

# --- provenance pins, inherited from rn1 (do NOT re-derive) ---------------------------------
BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
RAW_SHA_PIN = "e5539653"
D_POSE_CUDA = 6.880000e-06          # hv1 ep0634 [contest-CUDA T4 n600]
D_SEG_CUDA = 2.9611e-04             # hv1 ep0634 [contest-CUDA T4 n600]
RN1_ADVISORY_DPOSE_N96 = 1.251833e-04
BASE_S = 0.15959729295498598

FRAMES = 600
SEG_H, SEG_W = 384, 512
CAM_H, CAM_W = 874, 1164
POSE_DIMS = 6

DEFAULT_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/runs/"
    "base_optimized_n600_r3/output/0.raw"
)
DEFAULT_GT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
DEFAULT_MKV = UPSTREAM / "videos" / "0.mkv"
DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_pi2_pose_axis_attribution_20260816")


class Pi2Error(RuntimeError):
    """Fail-closed error for instrument / custody violations."""


# ============================================================================================
# GT decode conventions
#
# `canon` is upstream/frame_utils.py::yuv420_to_rgb re-expressed so the individual conventions
# become switchable.  It is asserted byte-identical to the upstream function before any variant
# is trusted -- the positive control for this whole stage.
# ============================================================================================
BT601 = (1.402, 0.344136, 0.714136, 1.772)
BT709 = (1.5748, 0.1873, 0.4681, 1.8556)


def decode_variant(planes: tuple[np.ndarray, np.ndarray, np.ndarray], *,
                   chroma: str = "bilinear",
                   align_corners: bool = False,
                   rounding: str = "round",
                   matrix: tuple[float, float, float, float] = BT601,
                   limited_range: bool = True) -> np.ndarray:
    """Decode YUV420 planes to (H,W,3) uint8 under one explicit set of conventions."""
    import torch
    import torch.nn.functional as F

    y, u, v = planes
    H, W = y.shape
    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

    if chroma == "nearest":
        u_up = F.interpolate(u_t, size=(H, W), mode="nearest").squeeze()
        v_up = F.interpolate(v_t, size=(H, W), mode="nearest").squeeze()
    else:
        u_up = F.interpolate(u_t, size=(H, W), mode="bilinear",
                             align_corners=align_corners).squeeze()
        v_up = F.interpolate(v_t, size=(H, W), mode="bilinear",
                             align_corners=align_corners).squeeze()

    if limited_range:
        yf = (y_t - 16.0) * (255.0 / 219.0)
        uf = (u_up - 128.0) * (255.0 / 224.0)
        vf = (v_up - 128.0) * (255.0 / 224.0)
    else:
        yf = y_t
        uf = u_up - 128.0
        vf = v_up - 128.0

    kr, kgu, kgv, kb = matrix
    r = (yf + kr * vf).clamp(0, 255)
    g = (yf - kgu * uf - kgv * vf).clamp(0, 255)
    b = (yf + kb * uf).clamp(0, 255)
    rgb = torch.stack([r, g, b], dim=-1)
    rgb = torch.floor(rgb) if rounding == "trunc" else rgb.round()
    return rgb.to(torch.uint8).numpy()


# name -> (kwargs for decode_variant, extra noise spec)
#
# Every entry is a convention a real decoder could legally hold, EXCEPT the `noise_*` rows,
# which are a deliberate CALIBRATION LADDER: they convert "how many LSB do two decoders differ
# by" into "how much d_pose does that cost", which is what turns this stage into an attribution
# rather than a screen.
VARIANTS: dict[str, dict] = {
    "canon": {},
    "chroma_nearest": {"chroma": "nearest"},
    "chroma_align_corners": {"align_corners": True},
    "round_trunc": {"rounding": "trunc"},
    "matrix_bt709": {"matrix": BT709},
    "range_full": {"limited_range": False},
}
NOISE_LADDER = {
    "noise_lsb_0p25": 0.25,
    "noise_lsb_0p5": 0.5,
    "noise_lsb_1p0": 1.0,
    "noise_lsb_2p0": 2.0,
}


def add_gauss_lsb(frame: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Calibration perturbation: N(0, sigma) LSB, rounded and clipped.  Deterministic.

    The seed deliberately does NOT depend on `sigma`, so every rung of the ladder scales the
    SAME noise field.  That makes the ladder a paired comparison across amplitude rather than
    four independent draws, which is what lets a single monotone curve be read off it at the
    small n these screens run at.
    """
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, sigma, size=frame.shape)
    return np.clip(np.rint(frame.astype(np.float32) + n), 0, 255).astype(np.uint8)


# ============================================================================================
# instrument -- rn1-identical pins
# ============================================================================================
class Instrument:
    """Frozen CPU-torch PoseNet (+ optional SegNet), batch = 1 pair, upstream preprocess."""

    def __init__(self, threads: int, with_seg: bool) -> None:
        import torch

        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        import einops
        from modules import PoseNet, posenet_sd_path
        from safetensors.torch import load_file

        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self._torch, self._einops = torch, einops
        self.threads = threads
        pose = PoseNet().eval()
        pose.load_state_dict(load_file(str(posenet_sd_path), device="cpu"))
        self.pose = pose
        self.seg = None
        if with_seg:
            from modules import SegNet, segnet_sd_path

            seg = SegNet().eval()
            seg.load_state_dict(load_file(str(segnet_sd_path), device="cpu"))
            self.seg = seg

    def _bthwc(self, arr: np.ndarray):
        """(T,H,W,3) uint8 -> (1,T,3,H,W) float."""
        t = self._torch.from_numpy(np.ascontiguousarray(arr))[None]
        return self._einops.rearrange(t, "b t h w c -> b t c h w").float()

    def pose_out(self, pair_cam_u8: np.ndarray):
        torch = self._torch
        with torch.inference_mode():
            return self.pose(self.pose.preprocess_input(self._bthwc(pair_cam_u8)))

    def pose_vec(self, out) -> np.ndarray:
        """The 6 scored pose components, exactly as compute_distortion slices them."""
        head = next(h for h in self.pose.hydra.heads if h.name == "pose")
        return out["pose"][..., : head.out // 2][0].numpy().astype(np.float64)

    def d_pose(self, out_a, out_b) -> float:
        return float(self.pose.compute_distortion(out_a, out_b)[0])

    def seg_argmax(self, frame1_cam_u8: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            x = self._bthwc(frame1_cam_u8[None])
            return self.seg(self.seg.preprocess_input(x)).argmax(dim=1)[0].numpy().astype(np.uint8)


# ============================================================================================
# io helpers
# ============================================================================================
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while (b := fh.read(chunk)):
            h.update(b)
    return h.hexdigest()


def open_raw(raw: Path) -> np.memmap:
    n = raw.stat().st_size // (CAM_H * CAM_W * 3)
    if n != 2 * FRAMES:
        raise Pi2Error(f"{raw} holds {n} frames, expected {2 * FRAMES}")
    return np.memmap(raw, dtype=np.uint8, mode="r", shape=(n, CAM_H, CAM_W, 3))


def seeded_pairs(n: int, seed: int) -> list[int]:
    """SEEDED RANDOM pairs -- never a prefix (m88/m96)."""
    rng = np.random.default_rng(seed)
    return sorted(int(v) for v in rng.choice(FRAMES, size=n, replace=False))


def decode_planes(mkv: Path, wanted: set[int]) -> dict[int, tuple]:
    """Raw YUV420 planes for the wanted frame indices, so conventions stay switchable."""
    import av

    out: dict[int, tuple] = {}
    hi = max(wanted)
    container = av.open(str(mkv))
    for idx, frame in enumerate(container.decode(container.streams.video[0])):
        if idx in wanted:
            H, W = frame.height, frame.width
            y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
                H, frame.planes[0].line_size)[:, :W].copy()
            u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(
                H // 2, frame.planes[1].line_size)[:, :W // 2].copy()
            v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(
                H // 2, frame.planes[2].line_size)[:, :W // 2].copy()
            out[idx] = (y, u, v)
        if idx >= hi:
            break
    container.close()
    return out


def canon_control(mkv: Path, probe_idx: int) -> dict:
    """POSITIVE CONTROL: our switchable `canon` must be byte-identical to upstream's function."""
    import av

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from frame_utils import yuv420_to_rgb

    container = av.open(str(mkv))
    ours = upstream_rgb = None
    for idx, frame in enumerate(container.decode(container.streams.video[0])):
        if idx == probe_idx:
            upstream_rgb = yuv420_to_rgb(frame).numpy()
            H, W = frame.height, frame.width
            y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
                H, frame.planes[0].line_size)[:, :W].copy()
            u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(
                H // 2, frame.planes[1].line_size)[:, :W // 2].copy()
            v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(
                H // 2, frame.planes[2].line_size)[:, :W // 2].copy()
            ours = decode_variant((y, u, v))
            break
    container.close()
    if ours is None or upstream_rgb is None:
        raise Pi2Error(f"probe frame {probe_idx} not decoded")
    diff = int(np.abs(ours.astype(np.int16) - upstream_rgb.astype(np.int16)).max())
    if diff != 0:
        raise Pi2Error(f"canon control FAILED: max |ours - upstream| = {diff}, expected 0")
    return {"probe_frame": probe_idx, "max_abs_diff_vs_upstream": diff, "passed": True}


def write_receipt(work: Path, name: str, payload: dict) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    p = work / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return p


def retain_npz(work: Path, name: str, **arrays) -> dict:
    """ALWAYS KEEP THE PAYLOAD: persist the per-pair rows, not only their aggregate."""
    work.mkdir(parents=True, exist_ok=True)
    p = work / f"{name}.npz"
    np.savez_compressed(p, **arrays)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def bootstrap_ci(vals: np.ndarray, iters: int, seed: int) -> dict:
    """Percentile bootstrap on the MEAN -- the honest test of 'is 18x a subset draw?'."""
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = vals[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    return {
        "mean": float(vals.mean()),
        "ci95_lo": float(np.percentile(means, 2.5)),
        "ci95_hi": float(np.percentile(means, 97.5)),
        "ci99_lo": float(np.percentile(means, 0.5)),
        "ci99_hi": float(np.percentile(means, 99.5)),
        "bootstrap_iters": iters,
    }


#: stages that score the FULL population; a "seeded random pairs" label there would be a lie,
#: and the bootstrap on them measures pair-to-pair variability, not error on the mean.
_FULL_POPULATION_STAGES = {"n600", "crossaxis", "segaxis"}


def base_meta(args: argparse.Namespace, stage: str) -> dict:
    full = stage in _FULL_POPULATION_STAGES
    return {
        "schema": f"ddm_pi2_{stage}.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "verdict_scope": "INSTANCE on the hv1 ep0634 object (archive sha 80d9c8c6...)",
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "d_pose_contest_cuda_n600": D_POSE_CUDA,
        "d_seg_contest_cuda_n600": D_SEG_CUDA,
        "rn1_advisory_dpose_n96": RN1_ADVISORY_DPOSE_N96,
        "threads": args.threads,
        "seed": args.seed,
        "pair_selection": ("ALL 600 pairs (full population, not a sample)" if full
                           else "seeded random subset (never a prefix; m88/m96)"),
        "bootstrap_meaning": (
            "pair-to-pair variability of the population, NOT sampling error on the mean -- all "
            "600 pairs are present; read it as the spread a DIFFERENT pair subset would show"
            if full else
            "percentile bootstrap sampling error on the subset mean"),
    }


# ============================================================================================
# stage: scale -- pose vector magnitudes + per-pair d_pose + the forward-drift bound
# ============================================================================================
def stage_scale(args: argparse.Namespace) -> int:
    raw = open_raw(args.raw)
    pairs = seeded_pairs(args.pairs, args.seed)
    inst = Instrument(args.threads, with_seg=False)

    wanted: set[int] = set()
    for p in pairs:
        wanted |= {2 * p, 2 * p + 1}
    print(f"[pi2] decoding {len(wanted)} GT frames ...", flush=True)
    planes = decode_planes(args.mkv, wanted)

    dpose = np.zeros(len(pairs))
    v_gt = np.zeros((len(pairs), POSE_DIMS))
    v_ours = np.zeros((len(pairs), POSE_DIMS))
    t0 = time.time()
    for i, p in enumerate(pairs):
        gt_pair = np.stack([decode_variant(planes[2 * p]), decode_variant(planes[2 * p + 1])])
        our_pair = np.asarray(raw[2 * p: 2 * p + 2])
        og, oo = inst.pose_out(gt_pair), inst.pose_out(our_pair)
        dpose[i] = inst.d_pose(og, oo)
        v_gt[i] = inst.pose_vec(og)
        v_ours[i] = inst.pose_vec(oo)
        if args.progress and (i + 1) % args.progress == 0:
            print(f"  {i + 1}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)

    # d_pose is an MSE over POSE_DIMS components.  A relative forward drift eps perturbs each of
    # P_gt and P_ours by ~eps*|P|; the difference then picks up ~eps*|P|*sqrt(2) per component.
    # Invert the unexplained offset to get the eps that would be REQUIRED to explain it.
    p_rms = float(np.sqrt((v_gt ** 2).mean()))
    offset = float(dpose.mean()) - D_POSE_CUDA
    eps_required = (float(np.sqrt(offset)) / (p_rms * np.sqrt(2.0))) if offset > 0 else float("nan")

    rec = base_meta(args, "scale") | {
        "n_pairs": len(pairs),
        "pairs": pairs,
        "d_pose_mean_advisory": float(dpose.mean()),
        "d_pose_bootstrap": bootstrap_ci(dpose, args.bootstrap, args.seed),
        "d_pose_ratio_vs_contest_cuda": float(dpose.mean()) / D_POSE_CUDA,
        "d_pose_per_pair_min": float(dpose.min()),
        "d_pose_per_pair_max": float(dpose.max()),
        "d_pose_per_pair_median": float(np.median(dpose)),
        "d_pose_per_pair_span_ratio": float(dpose.max() / max(dpose.min(), 1e-300)),
        "pose_vec_rms_gt": p_rms,
        "pose_vec_rms_ours": float(np.sqrt((v_ours ** 2).mean())),
        "pose_vec_abs_max_gt": float(np.abs(v_gt).max()),
        "pose_diff_rms_per_component": float(np.sqrt(((v_gt - v_ours) ** 2).mean())),
        "unexplained_offset_d_pose": offset,
        "forward_drift_relative_required_to_explain_offset": eps_required,
        "forward_drift_note": (
            "eps_required is the RELATIVE PoseNet forward drift that CPU-vs-CUDA would have to "
            "carry for scorer-forward drift alone to produce the measured offset.  Compare it "
            "against the measured fp32 torch-CPU parity scale (2.29e-05 relative to retained) "
            "and the MLX-PoseNet drift scale (5.5e-03 relative).  If eps_required exceeds both "
            "by orders of magnitude, (B) is falsified as the dominant term."),
        "wall_s": time.time() - t0,
    }
    rec["retained"] = retain_npz(args.work, f"PI2_SCALE_{args.tag}",
                                 pairs=np.array(pairs), d_pose=dpose, v_gt=v_gt, v_ours=v_ours)
    write_receipt(args.work, f"PI2_SCALE_{args.tag}", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


# ============================================================================================
# stage: attrib -- GT decode conventions + calibration ladder, scored on pose AND seg
# ============================================================================================
def stage_attrib(args: argparse.Namespace) -> int:
    control = canon_control(args.mkv, probe_idx=0)
    print(f"[pi2] canon control: {control}", flush=True)

    raw = open_raw(args.raw)
    pairs = seeded_pairs(args.pairs, args.seed)
    want_seg = not args.no_seg
    inst = Instrument(args.threads, with_seg=want_seg)
    gt_cache = np.load(args.gt, mmap_mode="r") if want_seg else None

    wanted: set[int] = set()
    for p in pairs:
        wanted |= {2 * p, 2 * p + 1}
    print(f"[pi2] decoding {len(wanted)} GT frames ...", flush=True)
    planes = decode_planes(args.mkv, wanted)

    names = list(VARIANTS) + list(NOISE_LADDER)
    npair = len(pairs)
    dpose = {k: np.zeros(npair) for k in names}
    flips = {k: np.zeros(npair, dtype=np.int64) for k in names}
    pixdiff = {k: np.zeros(npair) for k in names}       # mean |var - canon| over the pair
    pixmax = {k: np.zeros(npair) for k in names}
    pixfrac = {k: np.zeros(npair) for k in names}       # fraction of channels differing
    canon_vs_cache = np.zeros(npair, dtype=np.int64)

    t0 = time.time()
    for i, p in enumerate(pairs):
        our_pair = np.asarray(raw[2 * p: 2 * p + 2])
        oo = inst.pose_out(our_pair)
        our_am = inst.seg_argmax(our_pair[1]) if want_seg else None
        canon_pair = np.stack([decode_variant(planes[2 * p]), decode_variant(planes[2 * p + 1])])

        for k in names:
            if k in VARIANTS:
                gt_pair = (canon_pair if k == "canon" else
                           np.stack([decode_variant(planes[2 * p], **VARIANTS[k]),
                                     decode_variant(planes[2 * p + 1], **VARIANTS[k])]))
            else:
                s = NOISE_LADDER[k]
                gt_pair = np.stack([add_gauss_lsb(canon_pair[0], s, 1000 * p + 1),
                                    add_gauss_lsb(canon_pair[1], s, 1000 * p + 2)])
            d = np.abs(gt_pair.astype(np.int16) - canon_pair.astype(np.int16))
            pixdiff[k][i] = float(d.mean())
            pixmax[k][i] = float(d.max())
            pixfrac[k][i] = float((d > 0).mean())
            dpose[k][i] = inst.d_pose(inst.pose_out(gt_pair), oo)
            if want_seg:
                am = inst.seg_argmax(gt_pair[1])
                flips[k][i] = int((am != our_am).sum())
                if k == "canon":
                    canon_vs_cache[i] = int((am != np.asarray(gt_cache[p])).sum())
        if args.progress and (i + 1) % args.progress == 0:
            print(f"  {i + 1}/{npair}  {time.time() - t0:.0f}s", flush=True)

    npx = npair * SEG_H * SEG_W
    rows = []
    base_dpose = float(dpose["canon"].mean())
    for k in names:
        m = float(dpose[k].mean())
        row = {
            "variant": k,
            "kind": "convention" if k in VARIANTS else "calibration_noise",
            "spec": (str(VARIANTS[k]) if k in VARIANTS else f"gauss sigma={NOISE_LADDER[k]} LSB"),
            "d_pose": m,
            "d_pose_ratio_vs_contest_cuda": m / D_POSE_CUDA,
            "d_pose_ratio_vs_canon": m / base_dpose if base_dpose else float("nan"),
            "pixel_mean_abs_diff_vs_canon": float(pixdiff[k].mean()),
            "pixel_max_abs_diff_vs_canon": float(pixmax[k].max()),
            "pixel_frac_differing_vs_canon": float(pixfrac[k].mean()),
        }
        if want_seg:
            row |= {
                "flips": int(flips[k].sum()),
                "d_seg": float(flips[k].sum()) / npx,
                "d_seg_ratio_vs_contest_cuda": (float(flips[k].sum()) / npx) / D_SEG_CUDA,
                "d_seg_pct_change_vs_canon": (
                    100.0 * (flips[k].sum() - flips["canon"].sum()) / max(int(flips["canon"].sum()), 1)),
            }
        rows.append(row)
    rows.sort(key=lambda r: r["d_pose"])

    rec = base_meta(args, "attrib") | {
        "n_pairs": npair,
        "pairs": pairs,
        "canon_control": control,
        "seg_measured": want_seg,
        "canon_argmax_vs_cached_gt_px_per_pair_mean": (
            float(canon_vs_cache.mean()) if want_seg else None),
        "canon_argmax_vs_cached_gt_px_total": (int(canon_vs_cache.sum()) if want_seg else None),
        "scored_px_per_frame": SEG_H * SEG_W,
        "results": rows,
        "wall_s": time.time() - t0,
    }
    arrays = {"pairs": np.array(pairs), "canon_vs_cache": canon_vs_cache}
    for k in names:
        arrays[f"dpose__{k}"] = dpose[k]
        arrays[f"flips__{k}"] = flips[k]
        arrays[f"pixdiff__{k}"] = pixdiff[k]
    rec["retained"] = retain_npz(args.work, f"PI2_ATTRIB_{args.tag}", **arrays)
    write_receipt(args.work, f"PI2_ATTRIB_{args.tag}", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


# ============================================================================================
# stage: n600 -- the identity baseline over ALL 600 pairs, per-pair rows retained
# ============================================================================================
def stage_n600(args: argparse.Namespace) -> int:
    raw_sha = sha256_file(args.raw)
    if not raw_sha.startswith(RAW_SHA_PIN):
        raise Pi2Error(f"raw custody FAILED: {args.raw} sha256 {raw_sha} !~ {RAW_SHA_PIN}")
    raw = open_raw(args.raw)
    inst = Instrument(args.threads, with_seg=False)

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import av
    from frame_utils import yuv420_to_rgb

    dpose = np.zeros(FRAMES)
    v_gt = np.zeros((FRAMES, POSE_DIMS))
    t0 = time.time()
    container = av.open(str(args.mkv))
    buf: list[np.ndarray] = []
    pair_idx = 0
    for frame in container.decode(container.streams.video[0]):
        buf.append(yuv420_to_rgb(frame).numpy())
        if len(buf) < 2:
            continue
        gt_pair = np.stack(buf)
        buf = []
        og = inst.pose_out(gt_pair)
        dpose[pair_idx] = inst.d_pose(og, inst.pose_out(
            np.asarray(raw[2 * pair_idx: 2 * pair_idx + 2])))
        v_gt[pair_idx] = inst.pose_vec(og)
        pair_idx += 1
        if args.progress and pair_idx % args.progress == 0:
            print(f"  {pair_idx}/{FRAMES}  d_pose_running={dpose[:pair_idx].mean():.6e}"
                  f"  {time.time() - t0:.0f}s", flush=True)
        if pair_idx >= FRAMES:
            break
    container.close()
    if pair_idx != FRAMES:
        raise Pi2Error(f"decoded {pair_idx} pairs, expected {FRAMES}")

    mean = float(dpose.mean())
    rec = base_meta(args, "n600") | {
        "n_pairs": FRAMES,
        "raw_sha256": raw_sha,
        "d_pose_mean_advisory_n600": mean,
        "d_pose_ratio_vs_contest_cuda": mean / D_POSE_CUDA,
        "d_pose_bootstrap": bootstrap_ci(dpose, args.bootstrap, args.seed),
        "d_pose_per_pair_min": float(dpose.min()),
        "d_pose_per_pair_max": float(dpose.max()),
        "d_pose_per_pair_median": float(np.median(dpose)),
        "pose_S_advisory_n600": float(np.sqrt(10.0 * mean)),
        "pose_S_contest_cuda_n600": float(np.sqrt(10.0 * D_POSE_CUDA)),
        "pose_vec_rms_gt": float(np.sqrt((v_gt ** 2).mean())),
        "subset_vs_axis_note": (
            "If this n600 mean lands at the rn1 n=96 value, the 18.2x is an AXIS property and "
            "not a subset draw.  If it lands at the contest-CUDA value, rn1's n=96 was a draw."),
        "wall_s": time.time() - t0,
    }
    rec["retained"] = retain_npz(args.work, f"PI2_N600_{args.tag}", d_pose=dpose, v_gt=v_gt)
    write_receipt(args.work, f"PI2_N600_{args.tag}", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


# ============================================================================================
# stage: addit -- is the offset ADDITIVE and INCOHERENT?  The test that decides the RULE.
#
# Model.  Write our true pose error as `e`, the instrument's axis offset as `d`, and an
# operator's effect as `D`.  If `d` and `D` are incoherent then
#       measured  = |e + d|^2      ~=  |e|^2 + |d|^2
#       measured' = |e + D + d|^2  ~=  |e|^2 + |D|^2 + |d|^2
# so the ABSOLUTE delta `measured' - measured = |D|^2` is UNBIASED by the offset, while the
# RATIO and the sqrt-mapped `delta S_pose` are both badly biased (the sqrt is evaluated at the
# wrong, inflated baseline).
#
# That model is a claim, so this stage MEASURES it: run the same operator against GT floors that
# differ by ~7x in baseline d_pose and check whether the absolute delta is invariant.
#
# PRE-REGISTERED BAR, written before the run: additivity HOLDS if each operator's absolute
# delta varies by < 20% across floors spanning ~7x.  If it varies more, additivity is REFUTED
# and no conversion of advisory pose numbers is licensed -- only signs survive.
# ============================================================================================
ADDIT_FLOORS = {
    "canon": ("convention", {}),
    "chroma_nearest": ("convention", {"chroma": "nearest"}),
    "noise_lsb_2p0": ("noise", 2.0),
}


def _rn1_ops():
    """Reuse rn1's operator definitions verbatim so the rows stay leg-to-leg comparable."""
    import importlib.util

    src = REPO / "experiments" / "ddm_rn1_render_boundary_mechanism.py"
    spec = importlib.util.spec_from_file_location("ddm_rn1_ops", src)
    if spec is None or spec.loader is None:
        raise Pi2Error(f"cannot load rn1 operators from {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "identity": (mod.op_identity, {}),
        "dither:amp=1": (mod.op_dither, {"amp": 1.0}),
        "gain:g=1.02": (mod.op_gain, {"g": 1.02}),
    }


def stage_addit(args: argparse.Namespace) -> int:
    control = canon_control(args.mkv, probe_idx=0)
    print(f"[pi2] canon control: {control}", flush=True)
    raw = open_raw(args.raw)
    pairs = seeded_pairs(args.pairs, args.seed)
    inst = Instrument(args.threads, with_seg=False)
    ops = _rn1_ops()

    wanted: set[int] = set()
    for p in pairs:
        wanted |= {2 * p, 2 * p + 1}
    print(f"[pi2] decoding {len(wanted)} GT frames ...", flush=True)
    planes = decode_planes(args.mkv, wanted)

    dp = {(f, o): np.zeros(len(pairs)) for f in ADDIT_FLOORS for o in ops}
    t0 = time.time()
    for i, p in enumerate(pairs):
        our_pair = np.asarray(raw[2 * p: 2 * p + 2])
        canon_pair = np.stack([decode_variant(planes[2 * p]), decode_variant(planes[2 * p + 1])])
        gt_out = {}
        for f, (kind, spec) in ADDIT_FLOORS.items():
            if kind == "convention":
                gt_pair = (canon_pair if not spec else
                           np.stack([decode_variant(planes[2 * p], **spec),
                                     decode_variant(planes[2 * p + 1], **spec)]))
            else:
                gt_pair = np.stack([add_gauss_lsb(canon_pair[0], spec, 1000 * p + 1),
                                    add_gauss_lsb(canon_pair[1], spec, 1000 * p + 2)])
            gt_out[f] = inst.pose_out(gt_pair)
        for o, (fn, kw) in ops.items():
            edited = np.stack([fn(our_pair[0], **kw), fn(our_pair[1], **kw)])
            oo = inst.pose_out(edited)
            for f in ADDIT_FLOORS:
                dp[(f, o)][i] = inst.d_pose(gt_out[f], oo)
        if args.progress and (i + 1) % args.progress == 0:
            print(f"  {i + 1}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)

    rows = []
    for o in ops:
        if o == "identity":
            continue
        deltas = {f: float(dp[(f, o)].mean() - dp[(f, "identity")].mean()) for f in ADDIT_FLOORS}
        vals = np.array(list(deltas.values()))
        spread = float((vals.max() - vals.min()) / abs(vals.mean())) if vals.mean() else float("nan")
        # The conversion the rule licenses: add the UNBIASED absolute delta to the AUTHORITY
        # baseline, then take the sqrt there -- never rescale the advisory delta S.
        d_abs = deltas["canon"]
        rows.append({
            "operator": o,
            "delta_d_pose_absolute_per_floor": deltas,
            "floor_baseline_d_pose": {f: float(dp[(f, "identity")].mean()) for f in ADDIT_FLOORS},
            "floor_baseline_span_ratio": float(
                max(dp[(f, "identity")].mean() for f in ADDIT_FLOORS)
                / min(dp[(f, "identity")].mean() for f in ADDIT_FLOORS)),
            "absolute_delta_spread_fraction": spread,
            "additivity_holds_at_20pct_bar": bool(spread < 0.20),
            "delta_S_pose_advisory_naive": float(
                np.sqrt(10.0 * dp[("canon", o)].mean()) - np.sqrt(10.0 * dp[("canon", "identity")].mean())),
            "delta_S_pose_converted_at_authority_baseline": float(
                np.sqrt(10.0 * (D_POSE_CUDA + d_abs)) - np.sqrt(10.0 * D_POSE_CUDA)),
        })

    rec = base_meta(args, "addit") | {
        "n_pairs": len(pairs),
        "pairs": pairs,
        "canon_control": control,
        "pre_registered_bar": ("additivity HOLDS if each operator's absolute delta varies by "
                               "< 20% across GT floors spanning ~7x in baseline d_pose"),
        "results": rows,
        "wall_s": time.time() - t0,
    }
    rec["retained"] = retain_npz(
        args.work, f"PI2_ADDIT_{args.tag}", pairs=np.array(pairs),
        **{f"dpose__{f}__{o.replace(':', '_').replace('=', '_')}": dp[(f, o)]
           for f in ADDIT_FLOORS for o in ops})
    write_receipt(args.work, f"PI2_ADDIT_{args.tag}", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


# ============================================================================================
# stage: crossaxis -- score OUR frames against BOTH GT decode paths, end to end
#
# The separator already exists on disk and cost nothing: on ONE Tesla T4, in ONE container,
# PR130's own gt-cache builder ran BOTH GT decode paths against the SAME frozen scorer
# (`experiments/modal_dali_av_gt_cache_diff.py`, job #906).  So `gt_cache_av.pt["pose"]` and
# `gt_cache_dali.pt["pose"]` differ ONLY by the GT decode path, with the scorer, the host and
# the driver held fixed.  A prior arm measured local-macOS-CPU-AV vs T4-AV at 3.57e-12 pose MSE,
# so our local PoseNet may stand in for the T4's without contaminating the comparison -- and the
# `av` leg here is the POSITIVE CONTROL that proves it, because it must reproduce the locally
# measured advisory baseline.
#
# What each leg then means for OUR object:
#   d_pose(ours_cpu_decode, GT_av)   -> must equal the local advisory baseline   [control]
#   d_pose(ours_cpu_decode, GT_dali) -> the authority GT.  Its excess over the contest-CUDA
#                                       6.88e-06 is the part the GT decode does NOT explain,
#                                       i.e. the pose cost of ps1u's device-dependent inflate.
# ============================================================================================
DEFAULT_GT_CACHE_AV = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_av.pt")  # GT_LINEAGE_OK: pi2 is the instrument that MEASURED the two-lineage split; it binds BOTH caches (DALI on the next line) precisely to difference them -- naming the AV lineage is the measurement
DEFAULT_GT_CACHE_DALI = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt")


def stage_crossaxis(args: argparse.Namespace) -> int:
    import torch

    raw_sha = sha256_file(args.raw)
    if not raw_sha.startswith(RAW_SHA_PIN):
        raise Pi2Error(f"raw custody FAILED: {args.raw} sha256 {raw_sha} !~ {RAW_SHA_PIN}")
    raw = open_raw(args.raw)
    inst = Instrument(args.threads, with_seg=False)

    caches = {}
    for name, path in (("av", args.gt_cache_av), ("dali", args.gt_cache_dali)):
        if not path.exists():
            raise Pi2Error(f"GT cache missing: {path}")
        blob = torch.load(path, map_location="cpu")
        pose = blob["pose"].double().numpy()
        if pose.shape != (FRAMES, POSE_DIMS):
            raise Pi2Error(f"{path} pose shape {pose.shape} != {(FRAMES, POSE_DIMS)}")
        caches[name] = {"pose": pose, "bytes": path.stat().st_size,
                        "sha256": sha256_file(path), "path": str(path)}
        print(f"[pi2] {name}: |P|rms={np.sqrt((pose ** 2).mean()):.6f}", flush=True)

    gt_decode_term = float(((caches["av"]["pose"] - caches["dali"]["pose"]) ** 2).mean())

    v_ours = np.zeros((FRAMES, POSE_DIMS))
    t0 = time.time()
    for p in range(FRAMES):
        v_ours[p] = inst.pose_vec(inst.pose_out(np.asarray(raw[2 * p: 2 * p + 2])))
        if args.progress and (p + 1) % args.progress == 0:
            print(f"  {p + 1}/{FRAMES}  {time.time() - t0:.0f}s", flush=True)

    legs = {}
    for name in ("av", "dali"):
        per_pair = ((caches[name]["pose"] - v_ours) ** 2).mean(axis=1)
        legs[name] = {
            "d_pose": float(per_pair.mean()),
            "d_pose_ratio_vs_contest_cuda": float(per_pair.mean()) / D_POSE_CUDA,
            "pose_S": float(np.sqrt(10.0 * per_pair.mean())),
            "bootstrap": bootstrap_ci(per_pair, args.bootstrap, args.seed),
            "cache": {k: caches[name][k] for k in ("path", "bytes", "sha256")},
            "_per_pair": per_pair,
        }

    resid = legs["dali"]["d_pose"] - D_POSE_CUDA
    rec = base_meta(args, "crossaxis") | {
        "n_pairs": FRAMES,
        "raw_sha256": raw_sha,
        "our_decode_device": "cpu (advisory chain)",
        "gt_decode_term_av_minus_dali_pose_mse": gt_decode_term,
        "leg_av_control": {k: v for k, v in legs["av"].items() if not k.startswith("_")},
        "leg_dali_authority_gt": {k: v for k, v in legs["dali"].items() if not k.startswith("_")},
        "residual_after_gt_decode_attribution": resid,
        "residual_fraction_of_total_offset": resid / (legs["av"]["d_pose"] - D_POSE_CUDA),
        "pose_vec_rms_ours": float(np.sqrt((v_ours ** 2).mean())),
        "reading": (
            "leg_av is the CONTROL: it must reproduce the locally measured advisory n600 "
            "baseline, which validates both the cache convention and the negligibility of the "
            "local-vs-T4 scorer forward.  leg_dali scores OUR cpu-decoded frames against the "
            "AUTHORITY GT decode; its excess over 6.88e-06 is everything the GT decode path "
            "does NOT explain -- principally the pose cost of the device-dependent inflate "
            "that ps1u confirmed (raw sha e5539653 on cpu vs 9a6b75e5 on cuda)."),
        "wall_s": time.time() - t0,
    }
    rec["retained"] = retain_npz(
        args.work, f"PI2_CROSSAXIS_{args.tag}", v_ours=v_ours,
        pose_gt_av=caches["av"]["pose"], pose_gt_dali=caches["dali"]["pose"],
        d_pose_vs_av=legs["av"]["_per_pair"], d_pose_vs_dali=legs["dali"]["_per_pair"])
    write_receipt(args.work, f"PI2_CROSSAXIS_{args.tag}", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


# ============================================================================================
# stage: segaxis -- the same cross-axis test on the SEG half, which is the control everybody
# leaned on.  The two T4 GT caches also carry `seg` argmax fields, so our frames can be scored
# against BOTH decode paths' ground truth with no scorer ambiguity at all.
#
# The pose leg says the GT decode path is worth 1.406e-04 of d_pose.  The same decode path moves
# 20,671 of 117,964,800 GT argmax sites (1.75e-04).  If d_seg is nonetheless axis-stable, that is
# rn1's rho law acting on the INSTRUMENT: an undirected perturbation of the REFERENCE at the
# label boundary is a fair coin, so it cancels under argmax while adding incoherently under MSE.
# ============================================================================================
def stage_segaxis(args: argparse.Namespace) -> int:
    import torch

    raw_sha = sha256_file(args.raw)
    if not raw_sha.startswith(RAW_SHA_PIN):
        raise Pi2Error(f"raw custody FAILED: {args.raw} sha256 {raw_sha} !~ {RAW_SHA_PIN}")
    raw = open_raw(args.raw)
    inst = Instrument(args.threads, with_seg=True)

    segs = {}
    for name, path in (("av", args.gt_cache_av), ("dali", args.gt_cache_dali)):
        blob = torch.load(path, map_location="cpu")
        segs[name] = blob["seg"].numpy().astype(np.uint8)
        if segs[name].shape != (FRAMES, SEG_H, SEG_W):
            raise Pi2Error(f"{path} seg shape {segs[name].shape} unexpected")
    local_gt = np.load(args.gt, mmap_mode="r")
    gt_decode_seg_flips = int((segs["av"] != segs["dali"]).sum())

    ours = np.zeros((FRAMES, SEG_H, SEG_W), dtype=np.uint8)
    t0 = time.time()
    for p in range(FRAMES):
        ours[p] = inst.seg_argmax(np.asarray(raw[2 * p + 1]))
        if args.progress and (p + 1) % args.progress == 0:
            print(f"  {p + 1}/{FRAMES}  {time.time() - t0:.0f}s", flush=True)

    npx = FRAMES * SEG_H * SEG_W
    legs = {}
    for name, ref in (("av", segs["av"]), ("dali", segs["dali"]),
                      ("local_pyav_cache", np.asarray(local_gt).astype(np.uint8))):
        f = int((ours != ref).sum())
        legs[name] = {"flips": f, "d_seg": f / npx, "d_seg_ratio_vs_contest_cuda": (f / npx) / D_SEG_CUDA}

    rec = base_meta(args, "segaxis") | {
        "n_pairs": FRAMES,
        "raw_sha256": raw_sha,
        "scored_sites": npx,
        "gt_decode_seg_flips_av_vs_dali": gt_decode_seg_flips,
        "gt_decode_seg_fraction_av_vs_dali": gt_decode_seg_flips / npx,
        "local_pyav_cache_vs_t4_av_flips": int((np.asarray(local_gt).astype(np.uint8) != segs["av"]).sum()),
        "leg_av": legs["av"],
        "leg_dali_authority_gt": legs["dali"],
        "leg_local_pyav_cache": legs["local_pyav_cache"],
        "reading": (
            "If leg_dali ~= leg_av despite 20,671 GT argmax sites moving between the two decode "
            "paths, the seg instrument is axis-stable BY CANCELLATION, not because the decode "
            "paths agree -- rn1's rho(0.01)=0.985 fair coin applied to the reference."),
        "wall_s": time.time() - t0,
    }
    rec["retained"] = retain_npz(args.work, f"PI2_SEGAXIS_FIELD_{args.tag}", ours_argmax=ours)
    write_receipt(args.work, f"PI2_SEGAXIS_{args.tag}", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["scale", "attrib", "addit", "n600", "crossaxis", "segaxis"])
    ap.add_argument("--gt-cache-av", type=Path, default=DEFAULT_GT_CACHE_AV)
    ap.add_argument("--gt-cache-dali", type=Path, default=DEFAULT_GT_CACHE_DALI)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--mkv", type=Path, default=DEFAULT_MKV)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--no-seg", action="store_true")
    ap.add_argument("--progress", type=int, default=8)
    args = ap.parse_args(argv)
    return {"scale": stage_scale, "attrib": stage_attrib, "addit": stage_addit,
            "n600": stage_n600, "crossaxis": stage_crossaxis,
            "segaxis": stage_segaxis}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
