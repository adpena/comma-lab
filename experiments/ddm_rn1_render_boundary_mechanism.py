#!/usr/bin/env python3
"""ddm_rn1 -- decompose hv1's render->SegNet boundary error by MECHANISM, and race the
zero-byte deterministic post-decode operators that address each mechanism.

Parent: `.omx/research/ddm_rt1_seg_roundtrip_decomposition_20260816.md`.

ddm_rt1 measured WHERE the seg axis lives (99.22% on a one-pixel boundary ring; 33,743
round-trip flips = 0.028604 S) and bounded every post-hoc CARRIER lever.  It did not measure
WHAT MECHANISM in the render manufactures that one-pixel error, and it never tested a global
deterministic operator -- only local paint/repaint/solve, all of which are collateral-limited
by "a local edit is not local to the scorer".

This tool asks the mechanism question and races the cures that cost ~0 archive bytes.

WHY A GLOBAL DETERMINISTIC OPERATOR IS ~FREE (rule 118 / CLAUDE.md "inflate.py is a FREE
interpreter"):  a fixed generic image filter is ALGORITHM, and algorithm in inflate.py is not
sized.  Only its handful of tuned scalars are video-derived and therefore COUNTED -- 2 to 6
floats, i.e. 8-24 B, which is 5e-6 to 1.6e-5 S.  Against the -0.0095973 gap that is free to
three significant figures.  The bytes are still counted honestly in every receipt.

MECHANISMS under test
---------------------
  M-A  uint8 quantization at camera resolution         -> probe: +-1 LSB dither sensitivity
  M-B  global sub-pixel misregistration vs GT          -> cure: global bilinear shift
  M-C  boundary blur / high-frequency deficit vs GT    -> cure: unsharp mask
  M-D  global photometric (per-channel gain/bias) drift-> cure: per-channel affine
  M-E  exact argmax ties                               -> probe: top1==top2 census

THE SCORER'S SAMPLING, read at source (upstream/modules.py:107-109):
    SegNet.preprocess_input = x[:, -1]  then  F.interpolate(size=(384,512), mode='bilinear')
`antialias` is NOT passed, so it defaults False: a 874x1164 -> 384x512 downsample (2.276x,
2.273x) built from 2x2 taps per output pixel.  The scorer therefore sees an ALIASED point
sample of our camera-resolution field, and a large fraction of camera pixels never influence
it at all.  Both our render and GT pass through the identical map, so R is a common operator,
not an error source -- which is why the mechanism must be a property of our RGB, not of R.

Instrument pins, IDENTICAL to ddm_rt1 so every row is leg-to-leg comparable with its ledger
(et4 -- batch shape is part of the forward instrument):
    frozen CPU torch SegNet + PoseNet from upstream/models/*.safetensors, batch = 1 pair,
    torch.set_num_threads(--threads, default 8), upstream preprocess verbatim.

Axis is `[macOS-CPU advisory]` -- a yardstick, NEVER a score.  MPS is never used.
GT frames decode ONLY via frame_utils.yuv420_to_rgb (PyAV rgb24 manufactures ~100x phantom
pose).  Pairs are drawn by SEEDED RANDOM choice, never a prefix (m88/m96: a prefix of a skewed
population is a different population, and the sign of the bias inverts between seg and pose).
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

# --- provenance pins, inherited from the rt1 charter (do NOT re-derive) ---------------------
BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
D_SEG_CUDA = 2.9611e-04            # hv1 ep0634 [contest-CUDA T4 n600]
BASE_FLIPS_ADVISORY = 34938        # rt1 RT1_INSTRUMENT_CHECK, n600 advisory
BASE_S = 0.15959729295498598
GAP_TO_015 = BASE_S - 0.15         # +0.0095973 that must be removed

FRAMES = 600
SEG_H, SEG_W = 384, 512
CAM_H, CAM_W = 874, 1164
SCORED_PX = FRAMES * SEG_H * SEG_W          # 117,964,800
N_CLASSES = 5
S_PER_FLIP = 100.0 / SCORED_PX              # 8.4771e-07, td1's derived law
S_PER_BYTE = 25.0 / 37_545_489              # 6.65834e-07

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
DEFAULT_MKV = UPSTREAM / "videos" / "0.mkv"
DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_rn1_render_boundary_20260816")
RAW_SHA_PIN = "e5539653"           # wc1 custody pin, verified in-tool


class Rn1Error(RuntimeError):
    """Fail-closed error for instrument / custody violations."""


# ============================================================================================
# zero-byte deterministic operators on the decoded camera-resolution frame
#
# Every operator here is a GENERIC ALGORITHM (rule 118: free in inflate.py).  Its tuned
# scalars are the only video-derived content and are counted in `payload_bytes`.
# ============================================================================================
def _gauss1d(sigma: float) -> np.ndarray:
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def _sep_blur(xf: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur on (H,W,3) float32, edge-replicated."""
    from scipy.ndimage import convolve1d

    k = _gauss1d(sigma)
    out = convolve1d(xf, k, axis=0, mode="nearest")
    return convolve1d(out, k, axis=1, mode="nearest")


def op_identity(x: np.ndarray, **_: float) -> np.ndarray:
    return x


def op_unsharp(x: np.ndarray, sigma: float = 2.3, amount: float = 0.6) -> np.ndarray:
    """out = clip(round(x + amount * (x - G_sigma * x))).  2 tuned floats -> 8 B counted."""
    xf = x.astype(np.float32)
    hi = xf - _sep_blur(xf, sigma)
    return np.clip(np.rint(xf + amount * hi), 0.0, 255.0).astype(np.uint8)


def op_blur(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """The SIGN CONTROL for M-C: if blur is the mechanism, blurring must make it worse."""
    xf = x.astype(np.float32)
    return np.clip(np.rint(_sep_blur(xf, sigma)), 0.0, 255.0).astype(np.uint8)


def op_shift(x: np.ndarray, dy: float = 0.0, dx: float = 0.0) -> np.ndarray:
    """Global sub-pixel bilinear translation.  2 tuned floats -> 8 B counted."""
    if dy == 0.0 and dx == 0.0:
        return x
    xf = x.astype(np.float32)
    h, w = xf.shape[:2]
    yy = np.clip(np.arange(h, dtype=np.float32) - dy, 0.0, h - 1.0)
    xx = np.clip(np.arange(w, dtype=np.float32) - dx, 0.0, w - 1.0)
    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yy - y0)[:, None, None]
    wx = (xx - x0)[None, :, None]
    top = xf[y0][:, x0] * (1 - wx) + xf[y0][:, x1] * wx
    bot = xf[y1][:, x0] * (1 - wx) + xf[y1][:, x1] * wx
    return np.clip(np.rint(top * (1 - wy) + bot * wy), 0.0, 255.0).astype(np.uint8)


def op_dither(x: np.ndarray, amp: float = 1.0, seed: int = 0) -> np.ndarray:
    """M-A probe: deterministic +-amp LSB perturbation, seeded and reproducible."""
    rng = np.random.default_rng(seed)
    d = rng.integers(-1, 2, size=x.shape, dtype=np.int16) * round(amp)
    return np.clip(x.astype(np.int16) + d, 0, 255).astype(np.uint8)


def op_affine(x: np.ndarray, g0: float = 1.0, g1: float = 1.0, g2: float = 1.0,
              b0: float = 0.0, b1: float = 0.0, b2: float = 0.0) -> np.ndarray:
    """M-D: per-channel gain/bias.  6 tuned floats -> 24 B counted."""
    xf = x.astype(np.float32)
    g = np.array([g0, g1, g2], dtype=np.float32)
    b = np.array([b0, b1, b2], dtype=np.float32)
    return np.clip(np.rint(xf * g + b), 0.0, 255.0).astype(np.uint8)


def op_gamma(x: np.ndarray, gamma: float = 1.05) -> np.ndarray:
    """Global gamma on 0-255.  1 tuned float -> 4 B counted.  Never measured before rn1."""
    xf = x.astype(np.float32) / 255.0
    return np.clip(np.rint(np.power(xf, gamma) * 255.0), 0.0, 255.0).astype(np.uint8)


def op_gain(x: np.ndarray, g: float = 1.1) -> np.ndarray:
    """Global contrast about the frame's OWN per-channel mean.

    The pivot is computed from the decoded frame itself, so it is free to the receiver; only
    `g` is video-derived (1 float -> 4 B counted).  This is the amplitude probe for the
    task-space witness: if hv1's field is an optimized adversarial input whose flips are
    under-amplitude manufactured features, amplifying its own contrast should recover some.
    """
    xf = x.astype(np.float32)
    m = xf.reshape(-1, 3).mean(0)[None, None, :]
    return np.clip(np.rint(m + g * (xf - m)), 0.0, 255.0).astype(np.uint8)


_OPS = {
    "identity": (op_identity, 0),
    "unsharp": (op_unsharp, 8),
    "blur": (op_blur, 4),
    "shift": (op_shift, 8),
    "dither": (op_dither, 0),      # a PROBE, never shippable -- payload_bytes is meaningless
    "affine": (op_affine, 24),
    "gain": (op_gain, 4),
    "gamma": (op_gamma, 4),
}


def parse_op(spec: str):
    """`unsharp:sigma=2.3,amount=0.6` -> (name, callable, kwargs, payload_bytes)."""
    name, _, rest = spec.partition(":")
    if name not in _OPS:
        raise Rn1Error(f"unknown operator {name!r}; known: {sorted(_OPS)}")
    fn, nbytes = _OPS[name]
    kwargs: dict[str, float] = {}
    if rest:
        for tok in rest.split(","):
            if not tok:
                continue
            k, _, v = tok.partition("=")
            kwargs[k.strip()] = float(v)
    return name, fn, kwargs, nbytes


def apply_op(fn, kwargs: dict, frame: np.ndarray) -> np.ndarray:
    return fn(frame, **kwargs)


# ============================================================================================
# instrument -- rt1-identical pins
# ============================================================================================
class Instrument:
    """Frozen CPU-torch SegNet (+ optional PoseNet), batch-1, upstream preprocess verbatim."""

    def __init__(self, threads: int, with_pose: bool) -> None:
        import torch

        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        import einops
        from modules import SegNet, segnet_sd_path
        from safetensors.torch import load_file

        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self._torch, self._einops = torch, einops
        self.threads = threads
        seg = SegNet().eval()
        seg.load_state_dict(load_file(str(segnet_sd_path), device="cpu"))
        self.seg = seg
        self.pose = None
        if with_pose:
            from modules import PoseNet, posenet_sd_path

            pose = PoseNet().eval()
            pose.load_state_dict(load_file(str(posenet_sd_path), device="cpu"))
            self.pose = pose

    def _bthwc(self, arr: np.ndarray):
        """(T,H,W,3) uint8 -> (1,T,3,H,W) float."""
        t = self._torch.from_numpy(np.ascontiguousarray(arr))[None]
        return self._einops.rearrange(t, "b t h w c -> b t c h w").float()

    def seg_argmax(self, frame1_cam_u8: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            x = self._bthwc(frame1_cam_u8[None])
            return self.seg(self.seg.preprocess_input(x)).argmax(dim=1)[0].numpy().astype(np.uint8)

    def seg_logits(self, frame1_cam_u8: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            x = self._bthwc(frame1_cam_u8[None])
            return self.seg(self.seg.preprocess_input(x))[0].numpy()

    def seg_input(self, frame1_cam_u8: np.ndarray) -> np.ndarray:
        """The exact (3,384,512) float tensor SegNet sees -- R applied, nothing else."""
        torch = self._torch
        with torch.inference_mode():
            x = self._bthwc(frame1_cam_u8[None])
            return self.seg.preprocess_input(x)[0].numpy()

    def pose_out(self, pair_cam_u8: np.ndarray):
        torch = self._torch
        with torch.inference_mode():
            x = self._bthwc(pair_cam_u8)
            return self.pose(self.pose.preprocess_input(x))

    def d_pose(self, out_a, out_b) -> float:
        return float(self.pose.compute_distortion(out_a, out_b)[0])


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
        raise Rn1Error(f"{raw} holds {n} frames, expected {2 * FRAMES}")
    return np.memmap(raw, dtype=np.uint8, mode="r", shape=(n, CAM_H, CAM_W, 3))


def open_tokens(tokens: Path) -> np.memmap:
    if tokens.stat().st_size != SCORED_PX:
        raise Rn1Error(f"{tokens} is {tokens.stat().st_size} B, expected {SCORED_PX}")
    return np.memmap(tokens, dtype=np.uint8, mode="r", shape=(FRAMES, SEG_H, SEG_W))


def write_receipt(work: Path, name: str, payload: dict) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    p = work / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return p


def retain_field(work: Path, name: str, arr: np.ndarray) -> dict:  # MEASURE_ONLY_OK:the n600 `leg` stage retains its full argmax field here with sha256+bytes; the subset `spectrum`/`ties`/`exchange`/`race` stages are scalar-only screens whose per-pair intermediates regenerate in minutes from the retained wc1 decode and whose successor (a training-side objective) consumes none of them
    """ALWAYS KEEP THE PAYLOAD: persist the bytes, not only their measured length."""
    work.mkdir(parents=True, exist_ok=True)
    p = work / f"{name}.npy"
    np.save(p, arr)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def seeded_pairs(n: int, seed: int) -> list[int]:
    """SEEDED RANDOM pairs -- never a prefix (m88/m96)."""
    rng = np.random.default_rng(seed)
    return sorted(int(v) for v in rng.choice(FRAMES, size=n, replace=False))


def decode_gt(mkv: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Canonical GT decode ONLY (frame_utils.yuv420_to_rgb)."""
    import av

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from frame_utils import yuv420_to_rgb

    out, hi = {}, max(wanted)
    container = av.open(str(mkv))
    for idx, frame in enumerate(container.decode(container.streams.video[0])):
        if idx in wanted:
            out[idx] = yuv420_to_rgb(frame).numpy()
        if idx >= hi:
            break
    container.close()
    return out


def boundary_ring0(lab: np.ndarray) -> np.ndarray:
    """Scorer-lattice pixels adjacent to a label change (rt1's ring 0, the free support)."""
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


# ============================================================================================
# stage: spectrum -- the scorer-free mechanism ledger (M-B / M-C / M-D)
# ============================================================================================
def _luma(x: np.ndarray) -> np.ndarray:
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)


def _grad_mag(y: np.ndarray) -> np.ndarray:
    gy = np.zeros_like(y)
    gx = np.zeros_like(y)
    gy[1:-1, :] = 0.5 * (y[2:, :] - y[:-2, :])
    gx[:, 1:-1] = 0.5 * (y[:, 2:] - y[:, :-2])
    return np.sqrt(gy * gy + gx * gx)


def _radial_power(y: np.ndarray, nbin: int = 24) -> np.ndarray:
    f = np.fft.rfft2(y - y.mean())
    p = (f.real ** 2 + f.imag ** 2)
    h, w = y.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2) / 0.7071        # 0..1 over the representable disc
    idx = np.clip((r * nbin).astype(np.int32), 0, nbin - 1)
    out = np.zeros(nbin, dtype=np.float64)
    cnt = np.zeros(nbin, dtype=np.int64)
    np.add.at(out, idx.ravel(), p.ravel())
    np.add.at(cnt, idx.ravel(), 1)
    return out / np.maximum(cnt, 1)


def stage_spectrum(args: argparse.Namespace) -> int:
    """Is our render blurrier / shifted / photometrically off vs GT, in the SCORER's lattice?"""
    raw = open_raw(args.raw)
    tok = open_tokens(args.tokens)
    pairs = seeded_pairs(args.pairs, args.seed)
    wanted = {2 * p + 1 for p in pairs}
    print(f"[rn1] decoding {len(wanted)} GT frames (yuv420_to_rgb) ...", flush=True)
    gtf = decode_gt(args.mkv, wanted)
    inst = Instrument(args.threads, with_pose=False)

    rows = []
    pow_ours = np.zeros(24)
    pow_gt = np.zeros(24)
    for p in pairs:
        ours_cam = np.asarray(raw[2 * p + 1])
        gt_cam = gtf[2 * p + 1]
        # R applied EXACTLY as the scorer does it -- this is what SegNet actually reads.
        d_ours = inst.seg_input(ours_cam)          # (3,384,512) float
        d_gt = inst.seg_input(gt_cam)
        yo = _luma(np.transpose(d_ours, (1, 2, 0)))
        yg = _luma(np.transpose(d_gt, (1, 2, 0)))
        go, gg = _grad_mag(yo), _grad_mag(yg)
        ring = boundary_ring0(np.asarray(tok[p]))
        # camera-resolution gradients, for the pre-R view of the same question
        yoc, ygc = _luma(ours_cam.astype(np.float32)), _luma(gt_cam.astype(np.float32))
        pow_ours += _radial_power(yo)
        pow_gt += _radial_power(yg)
        rows.append({
            "pair": p,
            "scorer_grad_ours_all": float(go.mean()),
            "scorer_grad_gt_all": float(gg.mean()),
            "scorer_grad_ours_ring0": float(go[ring].mean()),
            "scorer_grad_gt_ring0": float(gg[ring].mean()),
            "camera_grad_ours": float(_grad_mag(yoc).mean()),
            "camera_grad_gt": float(_grad_mag(ygc).mean()),
            "scorer_luma_mean_ours": float(yo.mean()),
            "scorer_luma_mean_gt": float(yg.mean()),
            "scorer_luma_std_ours": float(yo.std()),
            "scorer_luma_std_gt": float(yg.std()),
            "camera_rgb_mean_ours": [float(v) for v in ours_cam.reshape(-1, 3).mean(0)],
            "camera_rgb_mean_gt": [float(v) for v in gt_cam.reshape(-1, 3).mean(0)],
            "camera_rgb_std_ours": [float(v) for v in ours_cam.reshape(-1, 3).std(0)],
            "camera_rgb_std_gt": [float(v) for v in gt_cam.reshape(-1, 3).std(0)],
            "scorer_rmse_ours_vs_gt": float(np.sqrt(((d_ours - d_gt) ** 2).mean())),
            # THE witness test: a video RECONSTRUCTION correlates ~+1 with GT luma; a
            # task-space witness optimized only through the frozen head need not correlate
            # at all, because human/photometric fidelity is explicitly non-authority.
            "luma_corr_scorer_lattice": float(
                np.corrcoef(yo.ravel(), yg.ravel())[0, 1]),
            "luma_corr_camera": float(
                np.corrcoef(yoc.ravel(), ygc.ravel())[0, 1]),
        })
        print(f"  pair {p:3d}  ring0 grad ours/gt = "
              f"{rows[-1]['scorer_grad_ours_ring0'] / rows[-1]['scorer_grad_gt_ring0']:.4f}",
              flush=True)

    def agg(k: str) -> float:
        return float(np.mean([r[k] for r in rows]))

    ratio = pow_ours / np.maximum(pow_gt, 1e-12)
    rec = {
        "schema": "ddm_rn1_spectrum.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False, "promotable": False,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "n_pairs": len(pairs), "seed": args.seed, "pairs": pairs,
        "pair_selection": "seeded random (never a prefix; m88/m96)",
        "scorer_grad_ratio_all": agg("scorer_grad_ours_all") / agg("scorer_grad_gt_all"),
        "scorer_grad_ratio_ring0": agg("scorer_grad_ours_ring0") / agg("scorer_grad_gt_ring0"),
        "camera_grad_ratio": agg("camera_grad_ours") / agg("camera_grad_gt"),
        "scorer_luma_std_ratio": agg("scorer_luma_std_ours") / agg("scorer_luma_std_gt"),
        "scorer_luma_mean_delta": agg("scorer_luma_mean_ours") - agg("scorer_luma_mean_gt"),
        "scorer_rmse_ours_vs_gt": agg("scorer_rmse_ours_vs_gt"),
        "luma_corr_scorer_lattice_mean": agg("luma_corr_scorer_lattice"),
        "luma_corr_camera_mean": agg("luma_corr_camera"),
        "witness_reading": (
            "|corr| near 1 => the decode is a video RECONSTRUCTION and GT-referenced "
            "sharpness/registration/photometry questions are well posed. |corr| near 0 or "
            "negative => the decode is a TASK-SPACE WITNESS and those questions are not "
            "well posed at all."),
        "radial_power_ratio_ours_over_gt": [float(v) for v in ratio],
        "radial_power_bins": "24 equal bins of |f|/|f|_max over the scorer 384x512 lattice",
        "rows": rows,
    }
    write_receipt(args.work, "RN1_SPECTRUM", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


# ============================================================================================
# stage: ties -- M-E census (are any flips exact argmax ties?)
# ============================================================================================
def stage_ties(args: argparse.Namespace) -> int:
    raw = open_raw(args.raw)
    gt = np.load(args.gt, mmap_mode="r")
    pairs = seeded_pairs(args.pairs, args.seed)
    inst = Instrument(args.threads, with_pose=False)
    n_flip = n_tie = n_tie_flip = 0
    near = []
    for p in pairs:
        lg = inst.seg_logits(np.asarray(raw[2 * p + 1]))
        srt = np.sort(lg, axis=0)
        gap = srt[-1] - srt[-2]
        am = lg.argmax(0).astype(np.uint8)
        fl = am != np.asarray(gt[p])
        n_flip += int(fl.sum())
        n_tie += int((gap == 0.0).sum())
        n_tie_flip += int((fl & (gap == 0.0)).sum())
        near.append(float(np.median(gap[fl])) if fl.any() else float("nan"))
    rec = {
        "schema": "ddm_rn1_ties.v1", "axis": "[macOS-CPU advisory]",
        "score_claim": False, "promotable": False,
        "n_pairs": len(pairs), "seed": args.seed, "pairs": pairs,
        "flips": n_flip, "exact_ties_all_px": n_tie, "exact_ties_that_are_flips": n_tie_flip,
        "median_top1_top2_gap_at_flips": float(np.nanmedian(near)),
        "verdict": ("M-E exact ties are NOT the mechanism"
                    if n_tie_flip * 100 < max(n_flip, 1) else "M-E exact ties are material"),
    }
    write_receipt(args.work, "RN1_TIES", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


# ============================================================================================
# stage: exchange -- THE decisive number for the whole post-hoc family
#
# rt1 measured the logit deficit AT THE FLIPS (median 0.105, 98.3% runner-up) and read it as
# "the residual is a tie, not a wall" -- a favourable shape for a cheap cure.  It never
# measured the OTHER side of the same trade: how many CORRECT boundary pixels sit at an
# equally thin margin and would BREAK under the same perturbation.
#
# That ratio is the exchange rate for ANY undirected zero-byte operator:
#
#     rho(delta) = #{correct boundary px with margin < delta} / #{flips with deficit < delta}
#
# rho >> 1 means an undirected perturbation of scale delta breaks more than it fixes, no
# matter what the operator is -- which would explain, with one coefficient, why every global
# transform ever measured in this repo raised S (ll1 window solve, L28 channel offset,
# odd_luma_bias +-1, temporal blend, AA-on-witness).  rho is measured here, not modelled.
# ============================================================================================
_DELTAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def stage_exchange(args: argparse.Namespace) -> int:
    raw = open_raw(args.raw)
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")
    pairs = seeded_pairs(args.pairs, args.seed)
    inst = Instrument(args.threads, with_pose=False)

    n_flip = n_ring = n_corr_ring = n_corr_off = 0
    cum_flip = np.zeros(len(_DELTAS), dtype=np.int64)
    cum_corr_ring = np.zeros(len(_DELTAS), dtype=np.int64)
    cum_corr_off = np.zeros(len(_DELTAS), dtype=np.int64)
    # per-GT-class deficit ladder: which classes a training-side margin hinge buys back first
    cls_flip = np.zeros((N_CLASSES, len(_DELTAS)), dtype=np.int64)
    cls_tot = np.zeros(N_CLASSES, dtype=np.int64)
    med_flip: list[float] = []
    med_corr_ring: list[float] = []

    for p in pairs:
        lg = inst.seg_logits(np.asarray(raw[2 * p + 1]))
        srt = np.sort(lg, axis=0)
        gap = (srt[-1] - srt[-2]).astype(np.float64)     # top1 - top2, >= 0 everywhere
        am = lg.argmax(0).astype(np.uint8)
        gtp = np.asarray(gt[p])
        flip = am != gtp
        ring = boundary_ring0(np.asarray(tok[p]))
        corr_ring = (~flip) & ring
        corr_off = (~flip) & (~ring)

        n_flip += int(flip.sum())
        n_ring += int(ring.sum())
        n_corr_ring += int(corr_ring.sum())
        n_corr_off += int(corr_off.sum())
        for i, d in enumerate(_DELTAS):
            cum_flip[i] += int((gap[flip] < d).sum())
            cum_corr_ring[i] += int((gap[corr_ring] < d).sum())
            cum_corr_off[i] += int((gap[corr_off] < d).sum())
        for c in range(N_CLASSES):
            m = flip & (gtp == c)
            cls_tot[c] += int(m.sum())
            for i, d in enumerate(_DELTAS):
                cls_flip[c, i] += int((gap[m] < d).sum())
        if flip.any():
            med_flip.append(float(np.median(gap[flip])))
        med_corr_ring.append(float(np.median(gap[corr_ring])))
        print(f"  pair {p:3d}  flips={int(flip.sum()):5d}  ring0={int(ring.sum()):6d}  "
              f"median gap flip={med_flip[-1]:.4f} corr-ring={med_corr_ring[-1]:.4f}",
              flush=True)

    rho = {f"{d}": (float(cum_corr_ring[i]) / cum_flip[i] if cum_flip[i] else float("inf"))
           for i, d in enumerate(_DELTAS)}
    rho_all = {f"{d}": (float(cum_corr_ring[i] + cum_corr_off[i]) / cum_flip[i]
                        if cum_flip[i] else float("inf"))
               for i, d in enumerate(_DELTAS)}
    rec = {
        "schema": "ddm_rn1_exchange.v1", "axis": "[macOS-CPU advisory]",
        "score_claim": False, "promotable": False, "verdict_scope": "INSTANCE",
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "n_pairs": len(pairs), "seed": args.seed, "pairs": pairs,
        "pair_selection": "seeded random (never a prefix; m88/m96)",
        "definition": ("gap = top1 - top2 SegNet logit. At a flip it is the deficit rt1 "
                       "measured (how far a cure must move the head). At a correct pixel it "
                       "is the headroom before that pixel BREAKS."),
        "counts": {"flips": n_flip, "ring0_px": n_ring,
                   "correct_on_ring0": n_corr_ring, "correct_off_ring0": n_corr_off},
        "median_gap_at_flips": float(np.median(med_flip)) if med_flip else None,
        "median_gap_at_correct_ring0": float(np.median(med_corr_ring)),
        "cum_flips_below_delta": {f"{d}": int(cum_flip[i]) for i, d in enumerate(_DELTAS)},
        "cum_correct_ring0_below_delta": {f"{d}": int(cum_corr_ring[i])
                                          for i, d in enumerate(_DELTAS)},
        "cum_correct_off_ring0_below_delta": {f"{d}": int(cum_corr_off[i])
                                              for i, d in enumerate(_DELTAS)},
        "rho_ring0_only": rho,
        "rho_whole_frame": rho_all,
        "per_gt_class_flips": {str(c): int(cls_tot[c]) for c in range(N_CLASSES)},
        "per_gt_class_share_below_delta": {
            str(c): {f"{d}": (float(cls_flip[c, i]) / cls_tot[c] if cls_tot[c] else None)
                     for i, d in enumerate(_DELTAS)}
            for c in range(N_CLASSES)},
        "class_order": "canonical comma10k: 0 Road, 1 Lane, 2 Undrivable, 3 Movable, 4 MyCar",
        "training_hinge_upper_bound": {
            f"{d}": {
                "share_of_flips_recoverable": (float(cum_flip[i]) / n_flip if n_flip else None),
                "flips_n600_equiv": (float(cum_flip[i]) / n_flip * BASE_FLIPS_ADVISORY
                                     if n_flip else None),
                "S_units_n600_equiv": (float(cum_flip[i]) / n_flip * BASE_FLIPS_ADVISORY
                                       * S_PER_FLIP if n_flip else None),
                "multiple_of_gap_to_015": (float(cum_flip[i]) / n_flip * BASE_FLIPS_ADVISORY
                                           * S_PER_FLIP / GAP_TO_015 if n_flip else None),
            } for i, d in enumerate(_DELTAS)},
        "training_hinge_note": (
            "UPPER BOUND, DERIVED not measured: a SIGNED margin shift of +delta at ring-0 "
            "pixels -- which only TRAINING can apply, because it needs GT to know the sign -- "
            "recovers every flip whose deficit is below delta and breaks no correct pixel, "
            "since the same shift moves correct pixels further from the boundary. A real "
            "trained model produces a distribution, not a uniform shift, so this bounds the "
            "prize rather than predicting it."),
        "reading": ("rho is how many correct pixels an undirected perturbation of scale delta "
                    "puts at risk per flip it could fix. A perfectly undirected operator helps "
                    "~half its reachable flips and breaks ~half its reachable correct pixels, "
                    "so it is a NET WIN only where rho < 1."),
    }
    write_receipt(args.work, f"RN1_EXCHANGE_{args.tag}", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "pairs"}, indent=2, sort_keys=True))
    return 0


# ============================================================================================
# stage: race -- screen zero-byte operators on a seeded-random subset (seg AND pose)
# ============================================================================================
def stage_race(args: argparse.Namespace) -> int:
    raw = open_raw(args.raw)
    gt = np.load(args.gt, mmap_mode="r")
    pairs = seeded_pairs(args.pairs, args.seed)
    need_pose = not args.no_pose
    inst = Instrument(args.threads, with_pose=need_pose)

    gtf = {}
    if need_pose:
        wanted = set()
        for p in pairs:
            wanted |= {2 * p, 2 * p + 1}
        print(f"[rn1] decoding {len(wanted)} GT frames for the pose leg ...", flush=True)
        gtf = decode_gt(args.mkv, wanted)

    specs = list(args.op)
    if "identity" not in [s.partition(":")[0] for s in specs]:
        specs.insert(0, "identity")

    # base argmax per pair, cached once: lets every operator report FIXED vs BROKEN, which is
    # the direct validation of the rho exchange-rate law measured by the `exchange` stage.
    base_am = {p: inst.seg_argmax(np.asarray(raw[2 * p + 1])) for p in pairs}

    results = []
    for spec in specs:
        name, fn, kwargs, nbytes = parse_op(spec)
        t0 = time.time()
        flips = n_fixed = n_broken = 0
        dpose_after: list[float] = []
        dpose_before: list[float] = []
        for p in pairs:
            pair = np.asarray(raw[2 * p: 2 * p + 2])
            edited = np.stack([apply_op(fn, kwargs, pair[0]),
                               apply_op(fn, kwargs, pair[1])])
            am = inst.seg_argmax(edited[1])
            gtp = np.asarray(gt[p])
            was_flip = base_am[p] != gtp
            now_flip = am != gtp
            flips += int(now_flip.sum())
            n_fixed += int((was_flip & ~now_flip).sum())
            n_broken += int((~was_flip & now_flip).sum())
            if need_pose:
                g = np.stack([gtf[2 * p], gtf[2 * p + 1]])
                og = inst.pose_out(g)
                dpose_before.append(inst.d_pose(og, inst.pose_out(pair)))
                dpose_after.append(inst.d_pose(og, inst.pose_out(edited)))
        npx = len(pairs) * SEG_H * SEG_W
        rec = {
            "op": spec, "name": name, "kwargs": kwargs,
            "payload_bytes_counted": nbytes,
            "flips": flips, "d_seg": flips / npx, "seg_S_units": 100.0 * flips / npx,
            "fixed": n_fixed, "broken": n_broken,
            "measured_exchange_rate_broken_per_fixed": (n_broken / n_fixed if n_fixed
                                                        else float("inf")),
            "wall_s": time.time() - t0,
        }
        if need_pose:
            # scorer convention: aggregate d_pose ITSELF, never a mean of per-pair ratios
            # (rt1 s6.2b -- the two disagree in SIGN).
            mb, ma = float(np.mean(dpose_before)), float(np.mean(dpose_after))
            rec |= {"d_pose_before": mb, "d_pose_after": ma,
                    "d_pose_ratio_scorer_convention": ma / mb if mb else float("nan"),
                    "pose_S_before": float(np.sqrt(10.0 * mb)),
                    "pose_S_after": float(np.sqrt(10.0 * ma))}
        results.append(rec)
        print(f"  {spec:34s} flips={flips:7d}  S_seg={rec['seg_S_units']:.6f}"
              f"  fixed={n_fixed:6d} broken={n_broken:6d}"
              + (f"  d_pose x{rec['d_pose_ratio_scorer_convention']:.4f}" if need_pose else "")
              + f"  {rec['wall_s']:.0f}s", flush=True)

    base = results[0]
    for r in results:
        r["delta_flips_vs_identity"] = r["flips"] - base["flips"]
        r["delta_S_seg_vs_identity"] = r["seg_S_units"] - base["seg_S_units"]
        if need_pose:
            r["delta_S_pose_vs_identity"] = r["pose_S_after"] - base["pose_S_after"]
        r["delta_S_rate"] = r["payload_bytes_counted"] * S_PER_BYTE
        r["delta_S_total_subset"] = (r["delta_S_seg_vs_identity"] + r["delta_S_rate"]
                                     + (r.get("delta_S_pose_vs_identity", 0.0)))
    rec = {
        "schema": "ddm_rn1_race.v1", "axis": "[macOS-CPU advisory]",
        "score_claim": False, "promotable": False, "verdict_scope": "INSTANCE",
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "n_pairs": len(pairs), "seed": args.seed, "pairs": pairs,
        "pair_selection": "seeded random (never a prefix; m88/m96)",
        "pose_measured": need_pose,
        "note": ("subset deltas are a SCREEN; only a full n600 `leg` row is quotable, and "
                 "per m96 a seeded subset may refute a bar but may not license a LIVE verdict"),
        "results": results,
    }
    write_receipt(args.work, f"RN1_RACE_{args.tag}", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


# ============================================================================================
# stage: leg -- full n600 seg for one operator, rt1-identical instrument
# ============================================================================================
def stage_leg(args: argparse.Namespace) -> int:
    raw_sha = sha256_file(args.raw)
    if not raw_sha.startswith(RAW_SHA_PIN):
        raise Rn1Error(f"raw custody FAILED: {args.raw} sha256 {raw_sha} !~ {RAW_SHA_PIN}")
    raw = open_raw(args.raw)
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")
    if gt.shape != (FRAMES, SEG_H, SEG_W):
        raise Rn1Error(f"GT cache shape {gt.shape} != {(FRAMES, SEG_H, SEG_W)}")
    spec = args.op[0]
    name, fn, kwargs, nbytes = parse_op(spec)
    inst = Instrument(args.threads, with_pose=False)

    pred = np.zeros((FRAMES, SEG_H, SEG_W), dtype=np.uint8)
    t0 = time.time()
    for t in range(FRAMES):
        pred[t] = inst.seg_argmax(apply_op(fn, kwargs, np.asarray(raw[2 * t + 1])))
        if args.progress and (t + 1) % args.progress == 0:
            el = time.time() - t0
            print(f"  {spec} {t + 1}/{FRAMES}  {el:.0f}s "
                  f"eta {el / (t + 1) * (FRAMES - t - 1):.0f}s", flush=True)
    wall = time.time() - t0
    gt_arr, tok_arr = np.asarray(gt), np.asarray(tok)
    flips = int(np.count_nonzero(pred != gt_arr))
    tag = spec.replace(":", "_").replace(",", "_").replace("=", "")
    payload = retain_field(args.work, f"argmax_{tag}", pred)
    rec = {
        "schema": "ddm_rn1_leg.v1", "axis": "[macOS-CPU advisory]",
        "score_claim": False, "promotable": False, "verdict_scope": "INSTANCE",
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "op": spec, "name": name, "kwargs": kwargs,
        "payload_bytes_counted": nbytes, "delta_S_rate": nbytes * S_PER_BYTE,
        "instrument": {"scorer": "frozen CPU torch SegNet, upstream/models/segnet.safetensors",
                       "batch_pairs": 1, "torch_threads": args.threads,
                       "preprocess": "upstream SegNet.preprocess_input verbatim"},
        "n_pairs": FRAMES, "scored_px": SCORED_PX,
        "flips_vs_gt": flips, "d_seg_vs_gt": flips / SCORED_PX,
        "seg_S_units": 100.0 * flips / SCORED_PX,
        "flips_vs_label": int(np.count_nonzero(pred != tok_arr)),
        "baseline_flips_advisory": BASE_FLIPS_ADVISORY,
        "delta_flips_vs_base": flips - BASE_FLIPS_ADVISORY,
        "delta_S_seg_vs_base": (flips - BASE_FLIPS_ADVISORY) * S_PER_FLIP,
        "gap_to_015_S": GAP_TO_015,
        "share_of_gap_closed_seg_only": -(flips - BASE_FLIPS_ADVISORY) * S_PER_FLIP / GAP_TO_015,
        "payload": payload, "wall_s": wall,
    }
    write_receipt(args.work, f"RN1_LEG_{tag}", rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["spectrum", "ties", "exchange", "race", "leg"])
    ap.add_argument("--op", action="append", default=[],
                    help="operator spec, e.g. 'unsharp:sigma=2.3,amount=0.6'. Repeatable.")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--mkv", type=Path, default=DEFAULT_MKV)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--progress", type=int, default=100)
    args = ap.parse_args(argv)
    if args.stage in {"race", "leg"} and not args.op:
        ap.error("--op is required for the race/leg stages")
    return {"spectrum": stage_spectrum, "ties": stage_ties, "exchange": stage_exchange,
            "race": stage_race, "leg": stage_leg}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
