#!/usr/bin/env python3
"""ddm_sr1 -- price the MANUFACTURED half of the hv1 seg axis into actuators.

Charter: convert ddm_rt1's round-trip decomposition into a priced, actuatable lever on the
LIVE hv1 vehicle, or close it with a measured ceiling.  Everything here is scorer-free and $0.

The measured premise (rt1, do not re-derive):

    scored seg term (advisory)                34,938 flips = 0.029617 S
    transmitted labels vs GT                   1,717 flips = 0.001456 S   (label channel)
    round trip (render argmax vs labels)      33,743 flips = 0.028604 S   (MANUFACTURED, 96.6%)
    99.22% of the axis sits on ring 0 of the transmitted label boundary
    correction channel: 33,235 B coded, eta measured 0.6235, eta required 0.7531 -> non-supplier

rt1 left exactly one stage of the realization path UNMEASURED (its own §2.5b scope note):

    "R is not a supplier for any paint-shaped candidate, and it remains unmeasured for the
     render."

This tool measures it, from the LIVE decoder's own geometry.  The live seg chain is

    tokens (384x512)  ->  SemanticTokenRenderer  ->  master_eval (3, 384, 512) in [0, 255]
      -> F.interpolate bilinear UP to (874, 1164), clamp(0,255), round -> uint8   [inflate.py]
      -> SegNet.preprocess_input: F.interpolate bilinear DOWN to (384, 512)       [upstream]
      -> SegNet -> argmax

so the operator the scorer applies to the renderer's output is the SQUARE round trip

    A = D . U : R^(384x512) -> R^(384x512)

which starts and ends at exactly the resolution the scorer consumes.  A cannot add information.
Any deviation of A from the identity is seg error MANUFACTURED by the archive format, and the
cure is decoder-side and costs ZERO archive bytes (inflate.py is free and unsized, CLAUDE.md
"inflate.py is a FREE interpreter"; a fixed resample matrix is a GENERIC ALGORITHM, rule 118
clean -- it contains no video-derived content).

Stages
------
roperator   exact separable characterisation of A = D.U, with a positive control against the
            real F.interpolate chain, plus the realised blur magnitude on the shipped decode
            and the clipping feasibility of a zero-byte pre-compensation m <- A^-1 m.
a1sign      FO-1, run by ddm_a1s: the DECISIVE argmax row for the zero-byte de-blur.  Synthesise
            the post-fix camera frame from the retained decode at a ladder of strengths, push
            each through the frozen CPU SegNet, and adjudicate against the bands this memo
            pre-registered.  Needs the scorer; every other stage here is scorer-free.
waterfill   re-price rt1's free-band correction channel PER CELL instead of on average.  rt1
            priced the whole band at its mean density 1.359%; the break-even density at a given
            eta is a fixed number, so any free-to-the-receiver sub-support above it already
            pays.  Ideal conditional-entropy limit (no model cost) = a CEILING.
ledger      assemble the ceiling arithmetic in S units against the -0.0095973 gap.

Axis: [macOS-CPU advisory] / scorer-free.  NEVER a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# --- provenance pins ------------------------------------------------------------------------
# hv1 ep0634, the live own-vehicle frontier.
BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
BASE_S = 0.15959729295498598
BASE_ARCHIVE_BYTES = 182_759
TARGET_S = 0.15
GAP_S = BASE_S - TARGET_S  # 0.00959729295498598

# rt1 measured rows (cited, not re-derived).
RT1_SCORED_FLIPS = 34_938
RT1_ROUND_TRIP_FLIPS = 33_743
RT1_LABEL_FLIPS = 1_717
RT1_BAND_FLIPS = 34_666
RT1_BAND_PX = 2_551_464
RT1_CHANNEL_BYTES = 33_235  # M7 mask + target class
RT1_ETA_MEASURED = 0.6235  # n=9 seeded-random, pose-null-constrained
RT1_ETA_REQUIRED = 0.7531

# Geometry, pinned from the live decoder + upstream scorer.
FRAMES = 600
SEG_H, SEG_W = 384, 512
CAM_H, CAM_W = 874, 1164
SCORED_PX = FRAMES * SEG_H * SEG_W  # 117,964,800
N_CLASSES = 5

# Exchange rates.  Both are exact contest arithmetic, not fits.
SEG_DS_PER_FLIP = 100.0 / SCORED_PX  # 8.477105e-07 (the comment read ...116e-07 until 2026-08-17)
RATE_DS_PER_BYTE = 25.0 / 37_545_489  # 6.658590e-07
BYTES_PER_FLIP_BAR = SEG_DS_PER_FLIP / RATE_DS_PER_BYTE  # 1.27312 B/flip

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
DEFAULT_BASE_ARGMAX = Path(
    "/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/argmax_base.npy"
)
DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_sr1_manufactured_seg_recovery_20260816")


class Sr1Error(RuntimeError):
    """Fail-closed error for instrument / custody violations."""


# ============================================================================================
# small helpers
# ============================================================================================
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f"wrote {path}")


def save_payload(path: Path, array: np.ndarray) -> dict:
    """Persist a materialised array and return its custody record (ALWAYS KEEP THE PAYLOAD)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    """H(p) in bits, with the 0log0 = 0 convention, elementwise."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    out = np.zeros_like(p)
    mask = (p > 0.0) & (p < 1.0)
    q = p[mask]
    out[mask] = -q * np.log2(q) - (1.0 - q) * np.log2(1.0 - q)
    return out


def boundary(lab: np.ndarray) -> np.ndarray:
    """4-neighbour label boundary -- rt1 / sq1 / gp1 convention, reproduced exactly."""
    b = np.zeros(lab.shape, dtype=bool)
    d = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    return b


def open_tokens(tokens: Path) -> np.memmap:
    need = FRAMES * SEG_H * SEG_W
    if tokens.stat().st_size != need:
        raise Sr1Error(f"token field {tokens} is {tokens.stat().st_size} B, expected {need}")
    return np.memmap(tokens, dtype=np.uint8, mode="r", shape=(FRAMES, SEG_H, SEG_W))


def open_raw(raw: Path) -> np.memmap:
    n, rem = divmod(raw.stat().st_size, CAM_H * CAM_W * 3)
    if rem or n != 2 * FRAMES:
        raise Sr1Error(f"raw {raw} is not {2 * FRAMES} camera frames")
    return np.memmap(raw, dtype=np.uint8, mode="r", shape=(n, CAM_H, CAM_W, 3))


def seeded_pairs(n_pairs: int, seed: int) -> np.ndarray:
    """Seeded RANDOM pair choice.  NEVER a prefix (memory m88/m96: a prefix of a skewed
    population is a different population, and the bias inverts sign between seg and pose)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(FRAMES, size=min(n_pairs, FRAMES), replace=False))


# ============================================================================================
# stage: roperator -- the square round trip A = D . U, exactly
# ============================================================================================
def resample_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Exact bilinear resample matrix along ONE axis, matching torch F.interpolate.

    Built by pushing the identity through the same call the live code makes, so the matrix IS
    the operator rather than a re-derivation of it.  The identity's second axis is held at
    n_in, where bilinear interpolation to the same size is exactly the identity, so only the
    resampled axis moves.  Column j of the identity is a unit impulse at position j, so the
    interpolated column j reads off that impulse's response.

    Returns W with shape (n_out, n_in): row i holds the weights of the input samples that form
    output sample i.
    """
    import torch
    import torch.nn.functional as F

    eye = torch.eye(n_in, dtype=torch.float64).view(1, 1, n_in, n_in)
    out = F.interpolate(eye, size=(n_out, n_in), mode="bilinear", align_corners=False)
    w = out[0, 0].numpy().astype(np.float64)
    if w.shape != (n_out, n_in):
        raise Sr1Error(f"resample matrix shape {w.shape} != {(n_out, n_in)}")
    return w


def stage_roperator(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(args.threads)
    t0 = time.time()

    # --- the four one-axis operators, exactly as the live chain applies them ----------------
    u_r = resample_matrix(SEG_H, CAM_H)   # (874, 384) up, rows
    d_r = resample_matrix(CAM_H, SEG_H)   # (384, 874) down, rows
    u_c = resample_matrix(SEG_W, CAM_W)   # (1164, 512) up, cols
    d_c = resample_matrix(CAM_W, SEG_W)   # (512, 1164) down, cols

    a_r = d_r @ u_r  # (384, 384)
    a_c = d_c @ u_c  # (512, 512)

    # --- POSITIVE CONTROL: the separable matrices must reproduce the real chain -------------
    # Without this the whole stage is an unvalidated instrument.  A NO measured on a broken
    # operator would be an artifact reported as physics (rt1 §6.1's lesson, applied to myself).
    rng = np.random.default_rng(12345)
    probe = rng.uniform(0.0, 255.0, size=(1, 1, SEG_H, SEG_W))
    probe_t = torch.from_numpy(probe)
    up = F.interpolate(probe_t, size=(CAM_H, CAM_W), mode="bilinear", align_corners=False)
    down = F.interpolate(up, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)
    chain = down[0, 0].numpy()
    separable = a_r @ probe[0, 0] @ a_c.T
    control_max_abs = float(np.abs(chain - separable).max())
    control_rel = float(control_max_abs / max(np.abs(chain).max(), 1e-12))
    control_ok = control_rel < 1e-9
    if not control_ok and not args.allow_control_fail:
        raise Sr1Error(
            f"separable operator does not reproduce the real interpolate chain: "
            f"max_abs={control_max_abs:.3e} rel={control_rel:.3e}"
        )

    # --- how far is A from the identity? ---------------------------------------------------
    s_r = np.linalg.svd(a_r, compute_uv=False)
    s_c = np.linalg.svd(a_c, compute_uv=False)
    eye_r = np.eye(SEG_H)
    eye_c = np.eye(SEG_W)

    # The 2-D operator's singular values are the outer product of the per-axis ones.
    s_2d = np.outer(s_r, s_c).ravel()
    s_2d.sort()

    # Impulse response through the middle row / column: the realised blur kernel.
    mid_r = a_r[SEG_H // 2]
    mid_c = a_c[SEG_W // 2]
    nz_r = np.flatnonzero(np.abs(mid_r) > 1e-6)
    nz_c = np.flatnonzero(np.abs(mid_c) > 1e-6)

    spectrum = {
        "row_singular_min": float(s_r.min()),
        "row_singular_max": float(s_r.max()),
        "row_cond": float(s_r.max() / max(s_r.min(), 1e-300)),
        "col_singular_min": float(s_c.min()),
        "col_singular_max": float(s_c.max()),
        "col_cond": float(s_c.max() / max(s_c.min(), 1e-300)),
        "twod_singular_min": float(s_2d.min()),
        "twod_singular_max": float(s_2d.max()),
        "twod_cond": float(s_2d.max() / max(s_2d.min(), 1e-300)),
        "twod_quantiles": {
            q: float(np.quantile(s_2d, q)) for q in (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.99)
        },
        "twod_share_below_0p9": float(np.mean(s_2d < 0.9)),
        "twod_share_below_0p5": float(np.mean(s_2d < 0.5)),
        "twod_share_below_0p1": float(np.mean(s_2d < 0.1)),
    }
    identity_gap = {
        "row_frob_rel": float(np.linalg.norm(a_r - eye_r) / np.linalg.norm(eye_r)),
        "col_frob_rel": float(np.linalg.norm(a_c - eye_c) / np.linalg.norm(eye_c)),
        "row_max_abs_offdiag": float(np.abs(a_r - np.diag(np.diag(a_r))).max()),
        "col_max_abs_offdiag": float(np.abs(a_c - np.diag(np.diag(a_c))).max()),
        "row_diag_min": float(np.diag(a_r).min()),
        "row_diag_max": float(np.diag(a_r).max()),
        "col_diag_min": float(np.diag(a_c).min()),
        "col_diag_max": float(np.diag(a_c).max()),
        "row_kernel_support_px": int(nz_r.size),
        "col_kernel_support_px": int(nz_c.size),
        "row_kernel_peak": float(np.abs(mid_r).max()),
        "col_kernel_peak": float(np.abs(mid_c).max()),
    }

    # --- the realised blur on the SHIPPED field, and pre-compensation feasibility -----------
    # x_scorer = D(shipped camera frame_1) is exactly what SegNet reads.  The renderer intended
    # m, the scorer sees A m, so the manufactured error is (I - A) m.  m is not retained, but
    # m ~= A^-1 x_scorer, so (I - A) m ~= A^-1 x - x, which is directly computable.  The
    # pre-compensation actuator writes U(A^-1 m) instead of U(m); its only physical constraint
    # is the decoder's clamp(0, 255), because the uint8 rounding happens DOWNSTREAM of U and is
    # averaged away by D -- pre-compensation does not amplify it.
    raw = open_raw(args.raw)
    raw_sha = sha256_file(args.raw) if args.verify_raw else None
    pairs = seeded_pairs(args.n_pairs, args.seed)
    tok = open_tokens(args.tokens)

    # Tikhonov inverses, per axis, at the requested regularisation.
    def reg_inverse(mat: np.ndarray, lam: float) -> np.ndarray:
        u, s, vt = np.linalg.svd(mat)
        return (vt.T * (s / (s * s + lam))) @ u.T

    inv_r = reg_inverse(a_r, args.tikhonov)
    inv_c = reg_inverse(a_c, args.tikhonov)

    rows = []
    band_delta_all = []
    interior_delta_all = []
    for t in pairs:
        cam = np.asarray(raw[2 * int(t) + 1], dtype=np.float64)  # frame_1, the seg surface
        # D applied per channel: (384,874) @ (874,1164) @ (1164,512)
        x = np.empty((SEG_H, SEG_W, 3), dtype=np.float64)
        for ch in range(3):
            x[:, :, ch] = d_r @ cam[:, :, ch] @ d_c.T
        # pre-compensated master estimate
        m = np.empty_like(x)
        for ch in range(3):
            m[:, :, ch] = inv_r @ x[:, :, ch] @ inv_c.T
        delta = m - x  # (I - A) m, the manufactured blur, in RGB levels at the scorer
        mag = np.abs(delta).max(axis=2)

        lab = np.asarray(tok[int(t)])
        band = boundary(lab)
        band_delta_all.append(mag[band].astype(np.float32))
        interior_delta_all.append(mag[~band].astype(np.float32))

        # clipping feasibility: does U(m) leave [0, 255]?
        clipped_lo = float(np.mean(m < 0.0))
        clipped_hi = float(np.mean(m > 255.0))
        rows.append({
            "pair": int(t),
            "band_px": int(band.sum()),
            "blur_band_mean_levels": float(mag[band].mean()),
            "blur_band_p50_levels": float(np.median(mag[band])),
            "blur_band_p95_levels": float(np.quantile(mag[band], 0.95)),
            "blur_interior_mean_levels": float(mag[~band].mean()),
            "precomp_below_0_share": clipped_lo,
            "precomp_above_255_share": clipped_hi,
            "x_range": [float(x.min()), float(x.max())],
            "m_range": [float(m.min()), float(m.max())],
        })
        print(f"  pair {int(t):3d}: band blur mean {rows[-1]['blur_band_mean_levels']:.4f} "
              f"p95 {rows[-1]['blur_band_p95_levels']:.4f} levels, "
              f"interior {rows[-1]['blur_interior_mean_levels']:.4f}", flush=True)

    band_delta = np.concatenate(band_delta_all)
    interior_delta = np.concatenate(interior_delta_all)

    realised = {
        "n_pairs": int(pairs.size),
        "pair_selection": "seeded RANDOM without replacement -- never a prefix (m88/m96)",
        "seed": int(args.seed),
        "pairs": pairs.tolist(),
        "band_blur_mean_levels": float(band_delta.mean()),
        "band_blur_p50_levels": float(np.median(band_delta)),
        "band_blur_p95_levels": float(np.quantile(band_delta, 0.95)),
        "band_blur_p99_levels": float(np.quantile(band_delta, 0.99)),
        "band_blur_max_levels": float(band_delta.max()),
        "interior_blur_mean_levels": float(interior_delta.mean()),
        "band_over_interior": float(band_delta.mean() / max(interior_delta.mean(), 1e-12)),
        "share_band_above_1_level": float(np.mean(band_delta > 1.0)),
        "share_band_above_4_levels": float(np.mean(band_delta > 4.0)),
        "precomp_clip_share_mean": float(np.mean(
            [r["precomp_below_0_share"] + r["precomp_above_255_share"] for r in rows])),
        "tikhonov": float(args.tikhonov),
    }

    payloads = {}
    if args.retain:
        payloads["a_row"] = save_payload(args.work / "A_row_384x384.npy", a_r.astype(np.float64))
        payloads["a_col"] = save_payload(args.work / "A_col_512x512.npy", a_c.astype(np.float64))
        payloads["band_blur_levels"] = save_payload(
            args.work / "band_blur_levels.npy", band_delta)

    record = {
        "schema": "ddm_sr1_roperator.v1",
        "axis": "[macOS-CPU advisory] scorer-free -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "operator": {
            "definition": "A = D . U on R^(384x512); U = bilinear up to (874,1164) as the live "
                          "inflate.py render_video applies it; D = bilinear down to (384,512) as "
                          "upstream SegNet.preprocess_input applies it; both align_corners=False",
            "up_from": [SEG_H, SEG_W], "up_to": [CAM_H, CAM_W],
            "square_round_trip": True,
        },
        "positive_control": {
            "question": "do the separable matrices reproduce the real F.interpolate chain?",
            "max_abs_diff": control_max_abs,
            "rel_diff": control_rel,
            "passed": bool(control_ok),
        },
        "spectrum": spectrum,
        "identity_gap": identity_gap,
        "realised_blur": realised,
        "per_pair": rows,
        "raw": {"path": str(args.raw), "sha256": raw_sha},
        "payloads": payloads,
        "wall_s": time.time() - t0,
    }
    write_json(args.work / "SR1_ROPERATOR.json", record)
    return record


# ============================================================================================
# stage: sign -- does inverting A move the scorer's input TOWARD or AWAY from the GT's?
# ============================================================================================
def stage_sign(args: argparse.Namespace) -> dict:
    """The self-attack on the roperator stage, and the measurement that fixes its SIGN.

    `roperator` establishes that the archive format applies a tridiagonal blur A between the
    renderer's output m and the scorer's input.  That alone does NOT license inverting it.
    There is a live failure mode: if the renderer was fit so that its CAMERA output U(m)
    matches the camera GT, then A m is already the correct scorer view and sharpening would
    OVERSHOOT.  The sign is decided by the GT, not by the operator.

    So compare, on the scorer's own lattice:

        g  = D(GT camera frame_1)      the field whose argmax IS the GT label
        x  = D(shipped camera frame_1) what SegNet actually reads from us
        x_alpha = x + alpha (A^-1 x - x)   the pre-compensation, swept

    and find alpha* minimising ||x_alpha - g|| on the band.  alpha* > 0 means the shipped
    scorer view is measurably UNDER-sharp against the truth and the lever has the right sign;
    alpha* ~ 0 closes the family at $0 with no scorer row spent.
    """
    import torch

    sys.path.insert(0, str(REPO / "upstream"))
    from frame_utils import yuv420_to_rgb

    torch.set_num_threads(args.threads)
    t0 = time.time()

    u_r = resample_matrix(SEG_H, CAM_H)
    d_r = resample_matrix(CAM_H, SEG_H)
    u_c = resample_matrix(SEG_W, CAM_W)
    d_c = resample_matrix(CAM_W, SEG_W)
    a_r, a_c = d_r @ u_r, d_c @ u_c
    inv_r = np.linalg.inv(a_r)
    inv_c = np.linalg.inv(a_c)

    pairs = seeded_pairs(args.n_pairs, args.seed)
    wanted = {2 * int(t) + 1 for t in pairs}
    print(f"[sr1] decoding {len(wanted)} GT frames (canonical yuv420_to_rgb)...", flush=True)
    import av
    gt_frames: dict[int, np.ndarray] = {}
    container = av.open(str(args.gt_mkv))
    stream = container.streams.video[0]
    hi = max(wanted)
    for idx, frame in enumerate(container.decode(stream)):
        if idx in wanted:
            gt_frames[idx] = yuv420_to_rgb(frame).numpy()
        if idx >= hi:
            break
    container.close()
    if len(gt_frames) != len(wanted):
        raise Sr1Error(f"decoded {len(gt_frames)} GT frames, wanted {len(wanted)}")

    raw = open_raw(args.raw)
    tok = open_tokens(args.tokens)
    alphas = np.round(np.arange(0.0, args.alpha_max + 1e-9, args.alpha_step), 4)

    def down(cam: np.ndarray) -> np.ndarray:
        out = np.empty((SEG_H, SEG_W, 3), dtype=np.float64)
        for ch in range(3):
            out[:, :, ch] = d_r @ cam[:, :, ch] @ d_c.T
        return out

    rows = []
    band_sse = np.zeros(alphas.size)
    full_sse = np.zeros(alphas.size)
    band_n = 0
    full_n = 0
    for t in pairs:
        gcam = np.asarray(gt_frames[2 * int(t) + 1], dtype=np.float64)
        if gcam.shape[0] == 3:
            gcam = np.transpose(gcam, (1, 2, 0))
        if gcam.shape[:2] != (CAM_H, CAM_W):
            raise Sr1Error(f"GT frame shape {gcam.shape} is not camera resolution")
        g = down(gcam)
        x = down(np.asarray(raw[2 * int(t) + 1], dtype=np.float64))
        sharp = np.empty_like(x)
        for ch in range(3):
            sharp[:, :, ch] = inv_r @ x[:, :, ch] @ inv_c.T
        step = sharp - x

        band = boundary(np.asarray(tok[int(t)]))
        per_alpha_band = []
        for ai, alpha in enumerate(alphas):
            xa = np.clip(x + alpha * step, 0.0, 255.0)
            err = (xa - g) ** 2
            b = float(err[band].sum())
            f = float(err.sum())
            band_sse[ai] += b
            full_sse[ai] += f
            per_alpha_band.append(b / max(band.sum() * 3, 1))
        band_n += int(band.sum()) * 3
        full_n += SEG_H * SEG_W * 3
        best = int(np.argmin(per_alpha_band))
        rows.append({
            "pair": int(t),
            "band_px": int(band.sum()),
            "alpha_star_band": float(alphas[best]),
            "band_rmse_alpha0": float(np.sqrt(per_alpha_band[0])),
            "band_rmse_best": float(np.sqrt(per_alpha_band[best])),
            "band_rmse_alpha1": float(np.sqrt(
                per_alpha_band[int(np.argmin(np.abs(alphas - 1.0)))])),
        })
        print(f"  pair {int(t):3d}: alpha* {alphas[best]:.2f}  band RMSE "
              f"{np.sqrt(per_alpha_band[0]):.4f} -> {np.sqrt(per_alpha_band[best]):.4f}",
              flush=True)

    band_rmse = np.sqrt(band_sse / max(band_n, 1))
    full_rmse = np.sqrt(full_sse / max(full_n, 1))
    b_best = int(np.argmin(band_rmse))
    f_best = int(np.argmin(full_rmse))
    i_one = int(np.argmin(np.abs(alphas - 1.0)))

    alpha_star = float(alphas[b_best])
    # NO authority-shaped verdict is emitted from this stage.  It is an INVALID instrument for
    # the question and the receipt must say so, because receipts outlive memos: the semantic
    # renderer is ~105 RGB levels from the GT at the band while scoring 0.0296% argmax error,
    # so a ~22-level blur term cannot register against that residual.  alpha_star is reported
    # as a measurement, never as a sign.
    verdict = "INVALID_INSTRUMENT_RGB_RESIDUAL_SWAMPS_THE_SIGNAL"
    record = {
        "schema": "ddm_sr1_sign.v1",
        "axis": "[macOS-CPU advisory] scorer-free -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "question": "does x + alpha (A^-1 x - x) move the scorer's input toward D(GT_cam)?",
        "gt_decode": "canonical upstream frame_utils.yuv420_to_rgb (PyAV rgb24 is FORBIDDEN)",
        "n_pairs": int(pairs.size),
        "pairs": pairs.tolist(),
        "seed": int(args.seed),
        "pair_selection": "seeded RANDOM without replacement -- never a prefix (m88/m96)",
        "alphas": alphas.tolist(),
        "band_rmse_by_alpha": band_rmse.tolist(),
        "full_rmse_by_alpha": full_rmse.tolist(),
        "band": {
            "alpha_star": alpha_star,
            "rmse_at_0": float(band_rmse[0]),
            "rmse_at_star": float(band_rmse[b_best]),
            "rmse_at_1": float(band_rmse[i_one]),
            "rel_improvement_at_star": float(1.0 - band_rmse[b_best] / band_rmse[0]),
            "rel_improvement_at_1": float(1.0 - band_rmse[i_one] / band_rmse[0]),
        },
        "full_frame": {
            "alpha_star": float(alphas[f_best]),
            "rmse_at_0": float(full_rmse[0]),
            "rmse_at_star": float(full_rmse[f_best]),
            "rmse_at_1": float(full_rmse[i_one]),
            "rel_improvement_at_star": float(1.0 - full_rmse[f_best] / full_rmse[0]),
        },
        "per_pair": rows,
        "pairs_with_positive_alpha_star": int(
            sum(1 for r in rows if r["alpha_star_band"] >= args.alpha_step * 2)),
        "verdict": verdict,
        "band_rmse_at_alpha0_levels": float(band_rmse[0]),
        "why_invalid": "the shipped render is photometrically ~105 RGB levels from the GT at "
                       "the band while its argmax error is 0.0296%, so distance-to-GT cannot "
                       "resolve a ~22-level blur term; do NOT cite alpha_star as a sign",
        "verdict_scope": f"INSTANCE -- hv1 ep0634 shipped decode, n={pairs.size} "
                         "seeded-random pairs. This stage is RETAINED AS A NEGATIVE CONTROL "
                         "ONLY; it settles nothing about the argmax.",
        "wall_s": time.time() - t0,
    }
    write_json(args.work / "SR1_SIGN.json", record)
    return record


# ============================================================================================
# stage: emphasis -- the VALID sign test, in the class-decision coordinate
# ============================================================================================
def stage_emphasis(args: argparse.Namespace) -> dict:
    """Settle the pre-compensation SIGN in a coordinate the seg objective actually lives in.

    `sign` tried to settle it by RGB distance to D(GT_cam) and FAILED as an instrument: the
    semantic renderer is ~105 RGB levels from the GT at the band while scoring 0.0296% argmax
    error, so a 22-level blur term is invisible under a 105-level residual.  Photometric
    fidelity is not the objective; class decisiveness is.

    The aligned coordinate: every band pixel sits between two class anchor colours.  Project
    it onto the line joining them,

        t = <v - a_own, a_partner - a_own> / ||a_partner - a_own||^2

    so t = 0 is "unambiguously my own class", t = 1 is "unambiguously the neighbour", and
    t = 0.5 is the coin flip.  Then:

      * if the renderer emitted a CRISP class edge and A blurred it, x sits near 0.5 and
        A^-1 x moves t AWAY from 0.5 toward 0 -- pre-compensation restores decisiveness, and
        the lever is LIVE;
      * if the renderer already PRE-EMPHASISED for the round trip, A^-1 x overshoots past
        t < 0 in bulk -- the blur is already paid for and inverting it double-counts, so the
        lever is CLOSED.

    This is a linear colour-space PROXY for a deep network's argmax, not the argmax.  It fixes
    the sign and sizes the effect; only the frozen SegNet converts it to flips.
    """
    import torch

    torch.set_num_threads(args.threads)
    t0 = time.time()

    u_r = resample_matrix(SEG_H, CAM_H)
    d_r = resample_matrix(CAM_H, SEG_H)
    u_c = resample_matrix(SEG_W, CAM_W)
    d_c = resample_matrix(CAM_W, SEG_W)
    a_r, a_c = d_r @ u_r, d_c @ u_c
    inv_r, inv_c = np.linalg.inv(a_r), np.linalg.inv(a_c)

    raw = open_raw(args.raw)
    tok = open_tokens(args.tokens)
    pairs = seeded_pairs(args.n_pairs, args.seed)

    bins = np.array([-np.inf, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, np.inf])
    hist_x = np.zeros(bins.size - 1, dtype=np.int64)
    hist_s = np.zeros(bins.size - 1, dtype=np.int64)
    stats = {"n": 0, "wrong_x": 0, "wrong_s": 0,
             "dec_x": 0.0, "dec_s": 0.0, "t_x": 0.0, "t_s": 0.0}
    rows = []
    for t in pairs:
        cam = np.asarray(raw[2 * int(t) + 1], dtype=np.float64)
        x = np.empty((SEG_H, SEG_W, 3))
        for ch in range(3):
            x[:, :, ch] = d_r @ cam[:, :, ch] @ d_c.T
        xs = np.empty_like(x)
        for ch in range(3):
            xs[:, :, ch] = inv_r @ x[:, :, ch] @ inv_c.T

        lab = np.asarray(tok[int(t)])
        band = boundary(lab)
        own, partner, _ = cell_features(lab)

        # receiver-legal anchors: per-class mean of the DECODED field over stable interior
        anchors = np.zeros((N_CLASSES, 3))
        have = np.zeros(N_CLASSES, dtype=bool)
        for c in range(N_CLASSES):
            interior = (lab == c) & (~band)
            if interior.sum() < 64:
                interior = lab == c
            if interior.sum() == 0:
                continue
            anchors[c] = x[interior].mean(axis=0)
            have[c] = True

        # Both anchors must actually exist.  This guard must fire when a class is ABSENT, which
        # is exactly when `have` is not all-True -- gating it on `have.all()` would skip it in
        # the only case it matters, so it is applied unconditionally.
        sel = band & (partner != 255)
        sel &= have[own] & have[np.where(partner == 255, 0, partner)]
        idx = np.flatnonzero(sel.ravel())
        if idx.size == 0:
            continue
        o = own.ravel()[idx].astype(np.int64)
        p = partner.ravel()[idx].astype(np.int64)
        a0 = anchors[o]
        a1 = anchors[p]
        d = a1 - a0
        nrm = np.einsum("ij,ij->i", d, d)
        keep = nrm > 1.0  # anchors must actually be distinguishable
        idx, o, p, a0, d, nrm = idx[keep], o[keep], p[keep], a0[keep], d[keep], nrm[keep]
        vx = x.reshape(-1, 3)[idx]
        vs = xs.reshape(-1, 3)[idx]
        tx = np.einsum("ij,ij->i", vx - a0, d) / nrm
        ts = np.einsum("ij,ij->i", vs - a0, d) / nrm

        hist_x += np.histogram(tx, bins=bins)[0]
        hist_s += np.histogram(ts, bins=bins)[0]
        stats["n"] += idx.size
        stats["wrong_x"] += int(np.count_nonzero(tx > 0.5))
        stats["wrong_s"] += int(np.count_nonzero(ts > 0.5))
        stats["dec_x"] += float(np.abs(tx - 0.5).sum())
        stats["dec_s"] += float(np.abs(ts - 0.5).sum())
        stats["t_x"] += float(tx.sum())
        stats["t_s"] += float(ts.sum())
        rows.append({
            "pair": int(t), "band_px_scored": int(idx.size),
            "t_mean_x": float(tx.mean()), "t_mean_sharp": float(ts.mean()),
            "wrong_side_share_x": float(np.mean(tx > 0.5)),
            "wrong_side_share_sharp": float(np.mean(ts > 0.5)),
            "decisiveness_x": float(np.abs(tx - 0.5).mean()),
            "decisiveness_sharp": float(np.abs(ts - 0.5).mean()),
        })
        print(f"  pair {int(t):3d}: wrong-side {rows[-1]['wrong_side_share_x']*100:.3f}% -> "
              f"{rows[-1]['wrong_side_share_sharp']*100:.3f}%   decisiveness "
              f"{rows[-1]['decisiveness_x']:.4f} -> {rows[-1]['decisiveness_sharp']:.4f}",
              flush=True)

    n = max(stats["n"], 1)
    wrong_x = stats["wrong_x"] / n
    wrong_s = stats["wrong_s"] / n
    dec_x = stats["dec_x"] / n
    dec_s = stats["dec_s"] / n
    pairs_better = sum(1 for r in rows
                       if r["wrong_side_share_sharp"] < r["wrong_side_share_x"])
    # This proxy calls ~40% of band pixels "wrong side" where the scorer flips ~1.36% of them --
    # it over-predicts by ~29x, because SegNet reads REGIONS, not pixels.  So it may report the
    # direction-free MAGNITUDE result (does A^-1 sharpen?) but it must NOT emit a sign verdict.
    # Emitting one would be a marker outrunning its measurement.
    verdict = (
        "MAGNITUDE_ONLY_A_INVERSE_SHARPENS" if dec_s > dec_x else
        "MAGNITUDE_ONLY_A_INVERSE_DOES_NOT_SHARPEN"
    )
    over_prediction = wrong_x / (RT1_BAND_FLIPS / RT1_BAND_PX)
    record = {
        "schema": "ddm_sr1_emphasis.v1",
        "axis": "[macOS-CPU advisory] scorer-free -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "coordinate": "t = projection of the band pixel onto the own->partner class-anchor "
                      "line; 0 = own class, 1 = neighbour, 0.5 = the coin flip",
        "anchors": "per-class mean of the DECODED scorer-view over stable interior pixels "
                   "(receiver-legal: decoded RGB + transmitted labels only, no GT, no scorer)",
        "n_pairs": int(pairs.size),
        "pairs": pairs.tolist(),
        "seed": int(args.seed),
        "pair_selection": "seeded RANDOM without replacement -- never a prefix (m88/m96)",
        "band_px_scored": int(stats["n"]),
        "shipped": {
            "t_mean": stats["t_x"] / n,
            "wrong_side_share": wrong_x,
            "decisiveness_mean_abs_t_minus_half": dec_x,
        },
        "pre_compensated": {
            "t_mean": stats["t_s"] / n,
            "wrong_side_share": wrong_s,
            "decisiveness_mean_abs_t_minus_half": dec_s,
        },
        "delta": {
            "wrong_side_share_abs": wrong_s - wrong_x,
            "wrong_side_share_rel": (wrong_s - wrong_x) / max(wrong_x, 1e-12),
            "decisiveness_rel": (dec_s - dec_x) / max(dec_x, 1e-12),
            "pairs_with_fewer_wrong_side": pairs_better,
            "pairs_total": len(rows),
        },
        "histogram_bins": bins.tolist(),
        "histogram_shipped": hist_x.tolist(),
        "histogram_pre_compensated": hist_s.tolist(),
        "overshoot_share_shipped": float(hist_x[0] / max(hist_x.sum(), 1)),
        "overshoot_share_pre_compensated": float(hist_s[0] / max(hist_s.sum(), 1)),
        "per_pair": rows,
        "verdict": verdict,
        "proxy_over_prediction_factor": over_prediction,
        "why_no_sign_verdict": (
            f"this proxy calls {100 * wrong_x:.2f}% of band pixels wrong-side where the scorer "
            f"flips {100 * RT1_BAND_FLIPS / RT1_BAND_PX:.3f}%; it over-predicts "
            f"~{over_prediction:.0f}x because SegNet reads regions, not pixels. The direction "
            "result is NOT licensed; only the magnitude result is."),
        "verdict_scope": "INSTANCE -- hv1 ep0634 shipped decode; a LINEAR colour-space proxy "
                         "for the frozen SegNet's argmax, not the argmax itself. The SIGN of "
                         "the pre-compensation lever is NOT settled by this stage.",
        "wall_s": time.time() - t0,
    }
    write_json(args.work / "SR1_EMPHASIS.json", record)
    return record


# ============================================================================================
# stage: waterfill -- re-price rt1's free-band channel PER CELL
# ============================================================================================
def cell_features(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Free-to-the-receiver cell features for every pixel of one label frame.

    Every feature is a deterministic function of the TRANSMITTED label field, which the
    receiver decodes before it needs the correction channel.  Nothing here is video-derived
    side information: the receiver can recompute all of it, so selecting a sub-support with
    these costs zero archive bytes.

    Returns (own_class, partner_class, degree) where partner is the lowest-indexed differing
    4-neighbour class (255 if the pixel is not on the boundary) and degree counts the distinct
    classes present in the closed 4-neighbourhood.
    """
    own = lab
    present = np.zeros((N_CLASSES, *lab.shape), dtype=bool)
    for c in range(N_CLASSES):
        present[c] = lab == c
    shifted = np.zeros_like(present)
    for c in range(N_CLASSES):
        p = present[c]
        s = np.zeros_like(p)
        s[:-1, :] |= p[1:, :]
        s[1:, :] |= p[:-1, :]
        s[:, :-1] |= p[:, 1:]
        s[:, 1:] |= p[:, :-1]
        shifted[c] = s | p
    degree = shifted.sum(axis=0).astype(np.uint8)
    partner = np.full(lab.shape, 255, dtype=np.uint8)
    for c in range(N_CLASSES - 1, -1, -1):
        hit = shifted[c] & (own != c)
        partner[hit] = c
    return own, partner, degree


def stage_waterfill(args: argparse.Namespace) -> dict:
    t0 = time.time()
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")
    pred = np.load(args.base_argmax, mmap_mode="r")
    if gt.shape != (FRAMES, SEG_H, SEG_W) or pred.shape != (FRAMES, SEG_H, SEG_W):
        raise Sr1Error(f"argmax shapes {gt.shape} / {pred.shape} are not n600 scorer-res")

    row_bands = args.row_bands
    n_cells = N_CLASSES * (N_CLASSES + 1) * (args.max_degree + 1) * row_bands
    band_px = np.zeros(n_cells, dtype=np.int64)
    flip_px = np.zeros(n_cells, dtype=np.int64)
    # target-class cost bookkeeping, conditioned on (own, partner) which are both free
    tgt_counts = np.zeros((N_CLASSES, N_CLASSES + 1, N_CLASSES), dtype=np.int64)

    row_idx = (np.arange(SEG_H) * row_bands // SEG_H).astype(np.int64)
    row_plane = np.repeat(row_idx[:, None], SEG_W, axis=1)

    total_band = 0
    total_flip = 0
    for t in range(args.frames):
        lab = np.asarray(tok[t])
        band = boundary(lab)
        own, partner, degree = cell_features(lab)
        deg = np.minimum(degree, args.max_degree)
        part = np.where(partner == 255, N_CLASSES, partner).astype(np.int64)
        key = (((own.astype(np.int64) * (N_CLASSES + 1) + part)
                * (args.max_degree + 1) + deg.astype(np.int64))
               * row_bands + row_plane)
        flip = (np.asarray(pred[t]) != np.asarray(gt[t])) & band
        band_px += np.bincount(key[band], minlength=n_cells)
        flip_px += np.bincount(key[flip], minlength=n_cells)
        if flip.any():
            np.add.at(
                tgt_counts,
                (own[flip].astype(np.int64), part[flip], np.asarray(gt[t])[flip].astype(np.int64)),
                1,
            )
        total_band += int(band.sum())
        total_flip += int(flip.sum())
        if (t + 1) % 100 == 0:
            print(f"  {t + 1}/{args.frames} frames, band {total_band:,} flips {total_flip:,}",
                  flush=True)

    live = band_px > 0
    n_r = band_px[live].astype(np.float64)
    k_r = flip_px[live].astype(np.float64)
    p_r = k_r / n_r
    bits_r = n_r * binary_entropy_bits(p_r)

    # target-class cost, conditional on the free (own, partner) context
    tot = tgt_counts.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(tot > 0, tgt_counts / np.maximum(tot, 1), 0.0)
        logp = np.where(prob > 0, np.log2(np.maximum(prob, 1e-300)), 0.0)
    target_bits_total = float(-(tgt_counts * logp).sum())
    target_bits_per_flip = target_bits_total / max(total_flip, 1)

    # --- the waterfill: include the densest cells first, stop at the marginal bar -----------
    order = np.argsort(-p_r)
    k_s, b_s, p_s = k_r[order], bits_r[order], p_r[order]
    cum_flips = np.cumsum(k_s)
    cum_bits = np.cumsum(b_s)

    def curve(eta: float) -> dict:
        bytes_mask = cum_bits / 8.0
        bytes_tgt = cum_flips * target_bits_per_flip / 8.0
        total_bytes = bytes_mask + bytes_tgt
        net = -eta * cum_flips * SEG_DS_PER_FLIP + total_bytes * RATE_DS_PER_BYTE
        best = int(np.argmin(net))
        # the marginal (per-cell) inclusion test, which is what "waterfill" means
        marginal_ok = (eta * k_s * SEG_DS_PER_FLIP
                       > (b_s / 8.0 + k_s * target_bits_per_flip / 8.0) * RATE_DS_PER_BYTE)
        m_flips = float(k_s[marginal_ok].sum())
        m_bytes = float((b_s[marginal_ok] / 8.0).sum()
                        + m_flips * target_bits_per_flip / 8.0)
        m_net = -eta * m_flips * SEG_DS_PER_FLIP + m_bytes * RATE_DS_PER_BYTE
        return {
            "eta": eta,
            "prefix_optimum": {
                "cells": best + 1,
                "flips_described": float(cum_flips[best]),
                "bytes": float(total_bytes[best]),
                "net_dS": float(net[best]),
                "min_density_included": float(p_s[best]),
                "bytes_per_recovered_flip": float(
                    total_bytes[best] / max(eta * cum_flips[best], 1e-12)),
            },
            "marginal_waterfill": {
                "cells": int(marginal_ok.sum()),
                "flips_described": m_flips,
                "share_of_band_flips": float(m_flips / max(total_flip, 1)),
                "bytes": m_bytes,
                "net_dS": float(m_net),
                "bytes_per_recovered_flip": float(m_bytes / max(eta * m_flips, 1e-12)),
                "share_of_gap_closed": float(-m_net / GAP_S),
            },
            "describe_everything": {
                "flips_described": float(cum_flips[-1]),
                "bytes": float(total_bytes[-1]),
                "net_dS": float(net[-1]),
            },
        }

    # the break-even density at a given eta: eta * p * BAR = H(p)/8  (target cost folded in)
    grid = np.linspace(1e-5, 0.6, 200_000)
    def breakeven_density(eta: float) -> float:
        lhs = eta * grid * SEG_DS_PER_FLIP
        rhs = (binary_entropy_bits(grid) / 8.0
               + grid * target_bits_per_flip / 8.0) * RATE_DS_PER_BYTE
        ok = lhs > rhs
        return float(grid[ok][0]) if ok.any() else float("nan")

    etas = [RT1_ETA_MEASURED, 0.7531, 0.85, 1.0]
    curves = {f"{e:.4f}": curve(e) for e in etas}
    breakeven = {f"{e:.4f}": breakeven_density(e) for e in etas}

    density_hist = {
        "cells_live": int(live.sum()),
        "band_px_total": int(total_band),
        "band_flips_total": int(total_flip),
        "mean_density": float(total_flip / max(total_band, 1)),
        "density_quantiles_flip_weighted": {
            q: float(np.quantile(np.repeat(p_r, flip_px[live]), q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        } if total_flip else {},
        "flip_share_in_cells_above": {
            f"{thr}": float(k_r[p_r >= thr].sum() / max(total_flip, 1))
            for thr in (0.02, 0.03, 0.05, 0.10, 0.20)
        },
    }

    payloads = {}
    if args.retain:
        payloads["cell_band_px"] = save_payload(args.work / "cell_band_px.npy", band_px)
        payloads["cell_flip_px"] = save_payload(args.work / "cell_flip_px.npy", flip_px)

    record = {
        "schema": "ddm_sr1_waterfill.v1",
        "axis": "[macOS-CPU advisory] scorer-free -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "cell_definition": "(own class) x (lowest differing 4-neighbour class) x "
                           "(min(degree, max_degree)) x (row band) -- every factor is a "
                           "deterministic function of the TRANSMITTED label field, so the "
                           "receiver recomputes the support for zero archive bytes",
        "row_bands": row_bands,
        "max_degree": args.max_degree,
        "frames": args.frames,
        "limit": "IDEAL conditional-entropy limit: empirical per-cell H(p), NO model cost and "
                 "NO coder inefficiency. This is a CEILING on the family, not a coder result.",
        "model_cost_note": {
            "cells_live": int(live.sum()),
            "bytes_if_2B_per_cell_probability": int(2 * live.sum()),
        },
        "target_class": {
            "bits_per_flip": target_bits_per_flip,
            "bytes_total_all_flips": target_bits_total / 8.0,
            "conditioning": "(own label, partner label) -- both free",
        },
        "density": density_hist,
        "breakeven_density_by_eta": breakeven,
        "curves_by_eta": curves,
        "rt1_reference": {
            "band_flips": RT1_BAND_FLIPS,
            "band_px": RT1_BAND_PX,
            "channel_bytes_measured": RT1_CHANNEL_BYTES,
            "eta_measured": RT1_ETA_MEASURED,
            "eta_required_whole_band": RT1_ETA_REQUIRED,
        },
        "payloads": payloads,
        "wall_s": time.time() - t0,
    }
    write_json(args.work / "SR1_WATERFILL.json", record)
    return record


# ============================================================================================
# stage: a1sign -- FO-1, the decisive $0 argmax row for the zero-byte de-blur (ddm_a1s)
# ============================================================================================
# Pre-registered by the sr1 FIRE-ORDER (memo section FO-1) BEFORE any alpha > 0 was scored.
# These four constants are the verdict and MUST NOT be moved by the arm that runs the row.
A1_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
A1_CONTROL_FLIPS = 34_938        # rt1 base leg, `argmax_base.npy` sha 2aeb1e6b...
A1_LIVE_BELOW = 33_251           # >= 5% of the 33,743 round-trip flips recovered
A1_NEUTRAL_LO = 34_589           # ceil(34938 * 0.99)
A1_NEUTRAL_HI = 35_287           # floor(34938 * 1.01)
A1_PIN_THREADS = 8               # et4: batch shape and thread count are part of the instrument

# --- FO-A: where does the A1 perturbation live? ----------------------------------------------
# ddm_a1s section 8 pre-registered ONE follow-on measurement: mask `delta_cam` to the lifted
# label band and re-run the alpha = 0.25 pose leg, because the band is ~2.2% of the camera
# pixels but carries the 22.5-level perturbation while the interior is 97.8% of pixels at ~1.8
# levels.  'off' is the ddm_a1s object EXACTLY -- no mask is built, no token field is opened,
# and the emitted pose6 array is byte-identical to the retained `pose6_by_alpha.npy`.  The
# 'interior' leg is the complement, because "band-driven or interior-driven" cannot be answered
# from the band leg alone: rms drift is not additive across a partition of the perturbation.
A1_DELTA_MASKS = ("off", "band", "interior")
# Pre-registered by ddm_a1s section 8 FO-A BEFORE this row was run.  Both are compared against
# the BAND-only pose drift rms at alpha = 0.25, n600.
A1_FOA_LIVE_BELOW = 0.0026240    # sqrt(HV1_D_POSE): the entire incumbent pose error
A1_FOA_CLOSED_ABOVE = 0.0083     # 3.2x the incumbent -- erases any plausible seg win


def _a1_foa_verdict(mask_mode: str, drift_rms: float) -> tuple[str, str]:
    """Adjudicate FO-A against the thresholds ddm_a1s section 8 pre-registered.

    Only the BAND leg carries the verdict; the interior leg is the complement reference.  The
    band between the two thresholds is reported, never bucketed -- forcing a bucket would be
    the same sin as forcing the seg ladder into a band it did not enter.
    """
    if mask_mode != "band":
        return "COMPLEMENT_LEG", ("interior-only reference leg; the band leg carries the "
                                  "FO-A verdict")
    if drift_rms < A1_FOA_LIVE_BELOW:
        return "POSE_NULL_BRANCH_LIVE", (
            "the band-restricted de-blur costs less pose than the incumbent pose error; "
            "OWES its own seg row before any win is claimed")
    if drift_rms >= A1_FOA_CLOSED_ABOVE:
        return "FAMILY_CLOSED", (
            "FAMILY: the post-hoc de-blur of A on this vehicle, band-restricted or not")
    return "INDETERMINATE_BETWEEN_BANDS", (
        "measured between the pre-registered thresholds; NOT bucketed")


def _a1_delta_mask_lift() -> tuple[np.ndarray, np.ndarray]:
    """rt1's nearest-neighbour camera->scorer lift index, IMPORTED not re-typed.

    Same instrument pin as `stage_a1sign`: the FO-A band must be the identical object a1sign
    already reports its `clip_band_*` diagnostics on, so it comes from the same code.
    """
    parent = str(Path(__file__).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from ddm_rt1_seg_roundtrip_decomposition import nn_lift_index

    return nn_lift_index()


def _a1_apply_delta_mask(delta_cam: np.ndarray, tok_frame: np.ndarray, mode: str,
                         lift: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Restrict the actuator to the lifted label band (or to its complement).

    Returns the masked perturbation and the amplitude split that explains it.  `mode == "off"`
    never reaches here -- callers skip the whole path so the default is bit-for-bit the parent
    row, not a mask that happens to be all-ones.
    """
    if mode not in ("band", "interior"):
        raise Sr1Error(f"unknown delta mask mode {mode!r}")
    band_cam = boundary(tok_frame)[np.ix_(*lift)]
    keep = band_cam if mode == "band" else ~band_cam
    amp = np.abs(delta_cam).max(axis=2)

    def mean_or_zero(sel: np.ndarray) -> float:
        # A degenerate frame (one token class everywhere, or no interior) would otherwise put a
        # NaN in the receipt.  Diagnostics never gate the measurement, so report 0.0 and let the
        # pixel count next to it say the set was empty.
        return float(amp[sel].mean()) if sel.any() else 0.0

    total = float((delta_cam ** 2).sum())
    diag = {
        "band_px_cam": int(band_cam.sum()),
        "band_share_cam": float(band_cam.mean()),
        "delta_absmax_mean_band": mean_or_zero(band_cam),
        "delta_absmax_mean_interior": mean_or_zero(~band_cam),
        "delta_energy_share_band": float((delta_cam[band_cam] ** 2).sum() / max(total, 1e-30)),
        "delta_kept_energy_share": float((delta_cam[keep] ** 2).sum() / max(total, 1e-30)),
    }
    return delta_cam * keep[:, :, None], diag


def _a1_axis_operators(tikhonov: float) -> dict:
    """Rebuild D, U and A = D.U per axis, and fail closed unless A reproduces sr1's retained A.

    sr1 retained only the SQUARE composite A (384x384, 512x512).  FO-1 needs `D(cam)`, which A
    cannot express -- D maps 874 -> 384.  So the four one-axis operators are rebuilt here from
    the same deterministic `resample_matrix`, and the rebuild is tied back to sr1's custody by
    requiring `d @ u` to equal the retained matrices EXACTLY.
    """
    u_r = resample_matrix(SEG_H, CAM_H)   # (874, 384)
    d_r = resample_matrix(CAM_H, SEG_H)   # (384, 874)
    u_c = resample_matrix(SEG_W, CAM_W)   # (1164, 512)
    d_c = resample_matrix(CAM_W, SEG_W)   # (512, 1164)
    a_r = d_r @ u_r
    a_c = d_c @ u_c

    def reg_inverse(mat: np.ndarray) -> np.ndarray:
        u, s, vt = np.linalg.svd(mat)
        return (vt.T * (s / (s * s + tikhonov))) @ u.T

    ops = {"u_r": u_r, "d_r": d_r, "u_c": u_c, "d_c": d_c, "a_r": a_r, "a_c": a_c,
           "inv_r": reg_inverse(a_r), "inv_c": reg_inverse(a_c)}
    for name, mat in ops.items():
        if not np.all(np.isfinite(mat)):
            raise Sr1Error(f"operator {name} is not finite")
    return ops


def _a1_custody_control(ops: dict, work: Path) -> dict:
    """Tie the rebuilt operators to sr1's retained payloads, and to the real interpolate chain."""
    import torch
    import torch.nn.functional as F

    rec: dict = {}
    for axis, fname, key in (("row", "A_row_384x384.npy", "a_r"),
                             ("col", "A_col_512x512.npy", "a_c")):
        path = work / fname
        if not path.exists():
            raise Sr1Error(f"missing sr1 retained operator {path}")
        retained = np.load(path)
        diff = float(np.abs(ops[key] - retained).max())
        rec[f"{axis}_rebuild_vs_retained_max_abs"] = diff
        rec[f"{axis}_retained_sha256"] = sha256_file(path)
        if diff != 0.0:
            raise Sr1Error(
                f"{axis} operator rebuild does not reproduce sr1's retained {fname} "
                f"(max abs diff {diff:.3e}); custody chain broken"
            )

    # sr1's own positive control, re-run: the separable matrices must reproduce the real chain.
    rng = np.random.default_rng(12345)
    probe = rng.uniform(0.0, 255.0, size=(1, 1, SEG_H, SEG_W))
    up = F.interpolate(torch.from_numpy(probe), size=(CAM_H, CAM_W),
                       mode="bilinear", align_corners=False)
    chain = F.interpolate(up, size=(SEG_H, SEG_W),
                          mode="bilinear", align_corners=False)[0, 0].numpy()
    separable = ops["a_r"] @ probe[0, 0] @ ops["a_c"].T
    max_abs = float(np.abs(chain - separable).max())
    rec["separable_vs_interpolate_max_abs"] = max_abs
    rec["separable_vs_interpolate_rel"] = float(max_abs / max(np.abs(chain).max(), 1e-12))
    if rec["separable_vs_interpolate_rel"] >= 1e-9:
        raise Sr1Error("separable operator does not reproduce the real interpolate chain")
    rec["inverse_row_residual"] = float(
        np.abs(ops["inv_r"] @ ops["a_r"] - np.eye(SEG_H)).max())
    rec["inverse_col_residual"] = float(
        np.abs(ops["inv_c"] @ ops["a_c"] - np.eye(SEG_W)).max())
    return rec


def _a1_delta_camera(cam_f64: np.ndarray, ops: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return `(x, delta_cam)`: what the scorer reads from the SHIPPED frame, and the actuator's
    camera-side perturbation whose alpha-scaled sum with `cam` realises the de-blur.

    The shipped decoder writes `cam = round(clamp(U m))` for a renderer master `m`, so the
    scorer reads `x = D(cam) ~= A m`.  The A1 actuator instead writes `U(m + a(A^-1 m - m))`.
    `m` is not retained, but `m_est = A^-1 x` recovers it to 1e-4 RGB levels at the band (this
    unit measured that residual directly), so the actuator's camera-side perturbation is

        delta_cam = U(A^-1 m_est - m_est),      cam'(a) = round(clamp(cam + a * delta_cam))

    which puts `x + a(A^-1 x - x)` at the scorer, because `D U = A` collapses the lift exactly.

    This is an ALGEBRAIC REARRANGEMENT of the FO-1 object, not a different object.  It is used
    instead of the order's literal `cam' = round(clamp(U(x + a(A^-1 x - x))))` because that
    form leaves the scorer reading `A x` at a = 0 -- one EXTRA blur, 14.98 RGB levels from the
    shipped decode -- and reading `x` (the status quo) at a = 1, so its ladder runs from
    double-blur to baseline and never reaches the de-blur.  The delta form is `cam` exactly at
    a = 0, so the pre-registered positive control is bit-exact by construction rather than
    approximate; and it matches a fresh `round(clamp(U(A^-1 m_est)))` render to 0.002 levels,
    so it also drops the order's stated one-extra-uint8-round pessimism.
    """
    x = np.empty((SEG_H, SEG_W, 3), dtype=np.float64)
    m_est = np.empty_like(x)
    delta_seg = np.empty_like(x)
    delta_cam = np.empty((CAM_H, CAM_W, 3), dtype=np.float64)
    for ch in range(3):
        x[:, :, ch] = ops["d_r"] @ cam_f64[:, :, ch] @ ops["d_c"].T
        m_est[:, :, ch] = ops["inv_r"] @ x[:, :, ch] @ ops["inv_c"].T
        delta_seg[:, :, ch] = (
            ops["inv_r"] @ m_est[:, :, ch] @ ops["inv_c"].T - m_est[:, :, ch]
        )
        delta_cam[:, :, ch] = ops["u_r"] @ delta_seg[:, :, ch] @ ops["u_c"].T
    if not (np.all(np.isfinite(delta_cam)) and np.all(np.isfinite(x))):
        raise Sr1Error("non-finite field in the A1 perturbation; refusing to score")
    return x, delta_cam


def _a1_per_class(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Flips charged to the REFERENCE class (rt1 / fl1 charge-by-target convention)."""
    bad = pred != ref
    return {str(c): int(np.count_nonzero(bad & (ref == c))) for c in range(N_CLASSES)}


def _a1_verdict(flips: dict, mask_mode: str = "off") -> dict:
    """Adjudicate the ladder against the bands pre-registered in the sr1 FIRE-ORDER.

    The BANDS never move with `mask_mode` -- the bar is the bar -- but the SCOPE string must,
    because a band-restricted actuator is not the global de-blur the FO-1 scope names.
    """
    positive = {a: f for a, f in flips.items() if a > 0.0}
    best_alpha = min(positive, key=lambda a: positive[a])
    best = positive[best_alpha]
    all_in_band = all(A1_NEUTRAL_LO <= f <= A1_NEUTRAL_HI for f in flips.values())
    all_harmful = all(f > A1_NEUTRAL_HI for f in positive.values())
    actuator = ("global linear de-blur of A" if mask_mode == "off"
                else f"{mask_mode}-restricted linear de-blur of A")
    if best < A1_LIVE_BELOW:
        verdict, scope = "LIVE", "INSTANCE on the hv1 ep0634 vehicle"
    elif all_in_band:
        verdict, scope = "CLOSED_NEUTRAL", f"FORMULATION: {actuator}, this vehicle"
    elif all_harmful:
        verdict, scope = "CLOSED_HARMFUL", f"FORMULATION: {actuator}, this vehicle"
    else:
        verdict, scope = "INDETERMINATE_MIXED", "no pre-registered band fired; reported as shaped"
    return {
        "verdict": verdict,
        "verdict_scope": scope,
        "best_alpha": best_alpha,
        "best_flips": best,
        "flips_recovered_vs_control": A1_CONTROL_FLIPS - best,
        "share_of_round_trip_recovered": (A1_CONTROL_FLIPS - best) / RT1_ROUND_TRIP_FLIPS,
        "delta_S_seg": (best - A1_CONTROL_FLIPS) * SEG_DS_PER_FLIP,
        "share_of_gap_closed": (A1_CONTROL_FLIPS - best) * SEG_DS_PER_FLIP / GAP_S,
        "bands": {"live_below": A1_LIVE_BELOW, "neutral_lo": A1_NEUTRAL_LO,
                  "neutral_hi": A1_NEUTRAL_HI, "control": A1_CONTROL_FLIPS},
    }


def stage_a1sign(args: argparse.Namespace) -> dict:
    """FO-1: does undoing A before the up-sample recover argmax flips?  n600, $0, no dispatch."""
    if args.threads != A1_PIN_THREADS:
        raise Sr1Error(
            f"instrument pin violation: FO-1 requires --threads {A1_PIN_THREADS} "
            f"(rt1's base leg), got {args.threads}"
        )
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # Import the scorer and the NN lift from rt1 rather than re-typing them: the FO-1 pin is
    # that this leg differs from rt1's base leg ONLY in the camera frame, so it must run the
    # same code object, not an equivalent-looking copy.
    from ddm_rt1_seg_roundtrip_decomposition import SegInstrument, nn_lift_index

    t0 = time.time()
    alphas = list(A1_ALPHAS)
    mask_mode = getattr(args, "delta_mask", "off")
    suffix = "" if mask_mode == "off" else f"_{mask_mode}mask"
    ops = _a1_axis_operators(args.tikhonov)
    custody = _a1_custody_control(ops, args.work)
    print(json.dumps(custody, indent=2, sort_keys=True), flush=True)

    raw = open_raw(args.raw)
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")
    if gt.shape != (FRAMES, SEG_H, SEG_W):
        raise Sr1Error(f"GT cache shape {gt.shape} != {(FRAMES, SEG_H, SEG_W)}")

    out_dir = args.work / "a1sign"
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = {a: f"a{a:g}".replace(".", "p") for a in alphas}
    # Resume-from-disk (P0): the argmax fields ARE the checkpoint, and the per-pair diagnostic
    # rows are journalled beside them so a resumed run does not silently drop the rows measured
    # before the interruption.
    prog_path = out_dir / f"A1SIGN{suffix.upper()}_PROGRESS.json"
    rows_path = out_dir / f"A1SIGN{suffix.upper()}_PER_PAIR.jsonl"
    paths = {a: out_dir / f"argmax_alpha_{tags[a]}{suffix}.npy" for a in alphas}
    done = 0
    rows: list[dict] = []
    if prog_path.exists() and all(p.exists() for p in paths.values()):
        prev = json.loads(prog_path.read_text())
        if prev.get("alphas") == alphas and prev.get("frames") == args.frames:
            done = int(prev.get("pairs_done", 0))
    if done:
        if rows_path.exists():
            # Dedupe by pair keeping the newest write: a crash between checkpoints can journal
            # rows for pairs whose argmax bytes were never flushed, and those pairs are re-run.
            seen_rows = {}
            for ln in rows_path.read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    if r["pair"] < done:
                        seen_rows[r["pair"]] = r
            rows = [seen_rows[k] for k in sorted(seen_rows)]
        fields = {a: np.lib.format.open_memmap(paths[a], mode="r+") for a in alphas}
        print(f"resuming from pair {done} with {len(rows)} journalled rows", flush=True)
    else:
        rows_path.unlink(missing_ok=True)
        fields = {a: np.lib.format.open_memmap(
            paths[a], mode="w+", dtype=np.uint8, shape=(args.frames, SEG_H, SEG_W))
            for a in alphas}

    inst = SegInstrument(args.threads)
    lift = _a1_delta_mask_lift() if mask_mode != "off" else None
    diag_pairs = set(seeded_pairs(args.diag_pairs, args.seed).tolist())
    for t in range(done, args.frames):
        cam_u8 = np.asarray(raw[2 * t + 1])
        cam_f = cam_u8.astype(np.float64)
        x, delta_cam = _a1_delta_camera(cam_f, ops)
        band = boundary(np.asarray(tok[t]))
        band_cam = band[np.ix_(*nn_lift_index())]
        row = {"pair": t, "band_px": int(band.sum())}
        if mask_mode != "off":
            # ONE implementation of the mask, shared with `stage_a1pose`: a second inline copy
            # here would drift from the helper the tests actually pin.
            delta_cam, mrow = _a1_apply_delta_mask(delta_cam, np.asarray(tok[t]), mask_mode, lift)
            row.update(mrow)
        deblur = None
        if t in diag_pairs:
            if mask_mode == "off":
                deblur = np.stack(
                    [ops["inv_r"] @ x[:, :, c] @ ops["inv_c"].T for c in range(3)], axis=2) - x
            else:
                # Under a mask the actuator's intent is no longer the global de-blur, so the
                # realisation diagnostic is measured against what the scorer will actually see
                # per unit alpha -- D(masked delta) -- not against the unmasked A^-1 x - x.
                deblur = np.stack(
                    [ops["d_r"] @ delta_cam[:, :, c] @ ops["d_c"].T for c in range(3)], axis=2)
            if t == min(diag_pairs) and args.retain:
                save_payload(out_dir / f"delta_cam_pair{t}{suffix}_f32.npy",
                             delta_cam.astype(np.float32))
        for a in alphas:
            shifted = cam_f + a * delta_cam
            cam_prime = np.clip(shifted, 0.0, 255.0).round().astype(np.uint8)
            if a == 0.0 and not np.array_equal(cam_prime, cam_u8):
                raise Sr1Error(f"pair {t}: alpha=0 is not bit-identical to the shipped frame")
            fields[a][t] = inst.argmax_from_camera(cam_prime)
            clip = (shifted < 0.0) | (shifted > 255.0)
            row[f"clip_all_a{a:g}"] = float(clip.mean())
            row[f"clip_band_a{a:g}"] = float(clip.max(axis=2)[band_cam].mean())
            if deblur is not None:
                seen = np.stack(
                    [ops["d_r"] @ cam_prime[:, :, c].astype(np.float64) @ ops["d_c"].T
                     for c in range(3)], axis=2)
                target = x + a * deblur
                row[f"realise_band_a{a:g}"] = float(
                    np.abs(seen - target).max(axis=2)[band].mean())
                row[f"intent_band_a{a:g}"] = float(
                    np.abs(target - x).max(axis=2)[band].mean())
        rows.append(row)
        with rows_path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        if (t + 1) % args.checkpoint_every == 0 or t + 1 == args.frames:
            for a in alphas:
                fields[a].flush()
            prog_path.write_text(json.dumps(
                {"pairs_done": t + 1, "alphas": alphas, "frames": args.frames}, indent=2))
            el = time.time() - t0
            print(f"  pair {t + 1}/{args.frames}  {el:.0f}s  "
                  f"eta {el / (t + 1 - done) * (args.frames - t - 1):.0f}s", flush=True)

    gt_arr = np.asarray(gt)[: args.frames]
    tok_arr = np.asarray(tok)[: args.frames]
    scored_px = args.frames * SEG_H * SEG_W
    ladder, flips = [], {}
    for a in alphas:
        pred = np.asarray(fields[a])
        f_gt = int(np.count_nonzero(pred != gt_arr))
        flips[a] = f_gt
        path = paths[a]
        ladder.append({
            "alpha": a,
            "flips_vs_gt": f_gt,
            "seg_S_units": 100.0 * f_gt / scored_px,
            "flips_vs_label": int(np.count_nonzero(pred != tok_arr)),
            "delta_flips_vs_control": f_gt - A1_CONTROL_FLIPS,
            "per_class_flips_vs_gt_charged_to_gt": _a1_per_class(pred, gt_arr),
            "payload": {"path": str(path), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)},
        })

    control_flips = flips[0.0]
    control_ok = control_flips == A1_CONTROL_FLIPS and args.frames == FRAMES
    base_identical = None
    if args.base_argmax.exists() and args.frames == FRAMES:
        base_identical = bool(np.array_equal(np.asarray(fields[0.0]),
                                             np.load(args.base_argmax, mmap_mode="r")))
    verdict = _a1_verdict(flips, mask_mode) if control_ok else {
        "verdict": "INVALID_CONTROL_FAILED",
        "verdict_scope": "no alpha row may be scored; the instrument did not reproduce the base",
    }
    record = {
        "schema": "ddm_a1s_a1sign.v1",
        "arm": "ddm_a1s",
        "fire_order": "sr1 FO-1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "instrument": {
            "scorer": "frozen CPU torch SegNet, upstream/models/segnet.safetensors",
            "scorer_module": str(Path(__file__).resolve().parent
                                 / "ddm_rt1_seg_roundtrip_decomposition.py"),
            "batch_pairs": 1,
            "torch_threads": args.threads,
            "preprocess": "upstream SegNet.preprocess_input verbatim",
            "tikhonov": float(args.tikhonov),
        },
        "custody": custody,
        "positive_control": {
            "alpha0_flips_measured": control_flips,
            "alpha0_flips_required": A1_CONTROL_FLIPS,
            "passed": control_ok,
            "alpha0_field_bit_identical_to_rt1_base": base_identical,
            "camera_frame_bit_identity_asserted_every_pair": True,
        },
        "delta_mask": mask_mode,
        "ladder": ladder,
        "verdict": verdict,
        "n_pairs": args.frames,
        "scored_px": scored_px,
        "diagnostics_pairs": sorted(diag_pairs),
        "per_pair": rows,
        "wall_s": time.time() - t0,
    }
    write_json(args.work / f"SR1_A1SIGN{suffix.upper()}.json", record)
    print(json.dumps({k: record[k] for k in ("positive_control", "verdict")},
                     indent=2, sort_keys=True))
    if not control_ok:
        raise Sr1Error(
            f"POSITIVE CONTROL FAILED: alpha=0 gave {control_flips} flips, "
            f"required {A1_CONTROL_FLIPS}; no alpha > 0 row is admissible"
        )
    return record


# ============================================================================================
# stage: a1pose -- the pose half of the A1 row (ddm_a1s)
# ============================================================================================
# FO-1 counts seg flips only, but `PoseNet.preprocess_input` (upstream/modules.py:69-73) keeps
# BOTH frames of the pair, so an actuator that edits frame_1 moves `d_pose` as well.  At the hv1
# operating point d_pose is 6.885642960696714e-06, so the pose contribution sqrt(10*d_pose) has
# marginal 5/sqrt(10*d_pose) = 602.6 per unit d_pose -- a third-of-a-percent pose move can erase
# a whole seg win.  This stage measures the pose half on the SAME synthesised frames.
HV1_D_POSE = 6.885642960696714e-06   # ddm_wc2_hpac_mps_port_20260814.md, hv1 ep0634


class _PoseInstrument:
    """Frozen CPU-torch PoseNet, batch = 1 pair, upstream preprocessing verbatim."""

    def __init__(self, threads: int) -> None:
        import torch

        upstream = REPO / "upstream"
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
        import einops
        from modules import PoseNet
        from safetensors.torch import load_file

        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self._torch, self._einops = torch, einops
        net = PoseNet().eval()
        net.load_state_dict(load_file(str(upstream / "models" / "posenet.safetensors"),
                                      device="cpu"))
        self.net = net

    def pose6(self, frame0_u8: np.ndarray, frame1_u8: np.ndarray) -> np.ndarray:
        """The 6 scored pose dimensions for one pair (upstream takes `[..., :out // 2]`)."""
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(np.stack([frame0_u8, frame1_u8])))[None]
            x = self._einops.rearrange(x, "b t h w c -> b t c h w").float()
            return self.net(self.net.preprocess_input(x))["pose"][0, :6].numpy().astype(
                np.float64)


def stage_a1pose(args: argparse.Namespace) -> dict:
    """Measure how far A1 moves PoseNet, and bound the d_pose it can cost, GT-free."""
    if args.threads != A1_PIN_THREADS:
        raise Sr1Error(f"instrument pin violation: requires --threads {A1_PIN_THREADS}")
    t0 = time.time()
    alphas = list(A1_ALPHAS)
    mask_mode = getattr(args, "delta_mask", "off")
    suffix = "" if mask_mode == "off" else f"_{mask_mode}mask"
    ops = _a1_axis_operators(args.tikhonov)
    custody = _a1_custody_control(ops, args.work)
    raw = open_raw(args.raw)
    inst = _PoseInstrument(args.threads)
    tok = open_tokens(args.tokens) if mask_mode != "off" else None
    lift = _a1_delta_mask_lift() if mask_mode != "off" else None

    out_dir = args.work / "a1sign"
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = np.zeros((args.frames, len(alphas), 6), dtype=np.float64)
    mask_rows: list[dict] = []
    for t in range(args.frames):
        cam_u8 = np.asarray(raw[2 * t + 1])
        frame0 = np.asarray(raw[2 * t])
        cam_f = cam_u8.astype(np.float64)
        _, delta_cam = _a1_delta_camera(cam_f, ops)
        if mask_mode != "off":
            delta_cam, mrow = _a1_apply_delta_mask(
                delta_cam, np.asarray(tok[t]), mask_mode, lift)
            mrow["pair"] = t
            mask_rows.append(mrow)
        for i, a in enumerate(alphas):
            cam_prime = np.clip(cam_f + a * delta_cam, 0.0, 255.0).round().astype(np.uint8)
            if a == 0.0 and not np.array_equal(cam_prime, cam_u8):
                raise Sr1Error(f"pair {t}: alpha=0 is not bit-identical to the shipped frame")
            poses[t, i] = inst.pose6(frame0, cam_prime)
        if (t + 1) % args.checkpoint_every == 0 or t + 1 == args.frames:
            el = time.time() - t0
            print(f"  pose pair {t + 1}/{args.frames}  {el:.0f}s  "
                  f"eta {el / (t + 1) * (args.frames - t - 1):.0f}s", flush=True)

    payload = save_payload(out_dir / f"pose6_by_alpha{suffix}.npy", poses)
    base = poses[:, 0, :]
    # Seg deltas measured by the a1sign stage, so the two halves are priced on one object.
    # The suffix keeps a masked pose leg paired with the SAME-mask seg leg, never the parent's.
    sign_path = args.work / f"SR1_A1SIGN{suffix.upper()}.json"
    seg_flips = {}
    if sign_path.exists():
        for r in json.loads(sign_path.read_text())["ladder"]:
            seg_flips[float(r["alpha"])] = int(r["flips_vs_gt"])

    gt6 = None
    if args.gt_pose is not None and Path(args.gt_pose).exists():
        gt6 = np.load(args.gt_pose).astype(np.float64)
        if gt6.shape != (args.frames, 6):
            gt6 = None

    rows = []
    for i, a in enumerate(alphas):
        drift = poses[:, i, :] - base
        drift_rms = float(np.sqrt(np.mean(drift ** 2)))
        row = {
            "alpha": a,
            "pose_drift_rms_vs_alpha0": drift_rms,
            "pose_drift_max_abs": float(np.abs(drift).max()),
            # Triangle bounds, both directions, GT-free.  With p_a = p_0 + d,
            #   d_pose(a) = E||p_0 - p_gt + d||^2 / 6,
            # so sqrt(d_pose(a)) is within rms||d|| of sqrt(d_pose(0)).  The LOWER bound is the
            # decisive one here: once the drift exceeds the incumbent pose error, d_pose MUST
            # rise no matter which direction the drift points.
            "d_pose_worst_case": float((math.sqrt(HV1_D_POSE) + drift_rms) ** 2),
            "d_pose_best_case": float(max(0.0, drift_rms - math.sqrt(HV1_D_POSE)) ** 2),
        }
        row["d_pose_worst_case_rise"] = row["d_pose_worst_case"] - HV1_D_POSE
        row["d_pose_best_case_rise"] = row["d_pose_best_case"] - HV1_D_POSE
        row["delta_S_pose_worst_case"] = (
            math.sqrt(10.0 * row["d_pose_worst_case"]) - math.sqrt(10.0 * HV1_D_POSE))
        row["delta_S_pose_best_case"] = (
            math.sqrt(10.0 * row["d_pose_best_case"]) - math.sqrt(10.0 * HV1_D_POSE))
        if a in seg_flips:
            row["seg_flips"] = seg_flips[a]
            row["delta_S_seg"] = (seg_flips[a] - A1_CONTROL_FLIPS) * SEG_DS_PER_FLIP
            row["delta_S_total_worst_case"] = (
                row["delta_S_seg"] + row["delta_S_pose_worst_case"])
            # The best case for A1: the largest seg win with the smallest pose cost the
            # geometry permits.  If THIS is positive, A1 is a net loss under every assumption.
            row["delta_S_total_best_case"] = (
                row["delta_S_seg"] + row["delta_S_pose_best_case"])
        if gt6 is not None:
            row["d_pose_vs_cached_gt"] = float(np.mean((poses[:, i, :] - gt6) ** 2))
        rows.append(row)

    record = {
        "schema": "ddm_a1s_a1pose.v1",
        "arm": "ddm_a1s",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "instrument": {
            "scorer": "frozen CPU torch PoseNet, upstream/models/posenet.safetensors",
            "batch_pairs": 1, "torch_threads": args.threads,
            "preprocess": "upstream PoseNet.preprocess_input verbatim, first 6 pose dims",
        },
        "custody": custody,
        "hv1_d_pose_reference": HV1_D_POSE,
        "method": "GT-FREE: the triangle inequality bounds sqrt(d_pose(a)) by sqrt(d_pose(0)) "
                  "plus the measured rms pose drift, so no GT pose target is needed for the "
                  "worst case.  `d_pose_vs_cached_gt` is a SECONDARY advisory read against a "
                  "2026-06-10 cache whose sister seg GT disagrees with the qs3 GT on 20,673 px.",
        "gt_pose_cache": str(args.gt_pose) if gt6 is not None else None,
        "delta_mask": mask_mode,
        "ladder": rows,
        "payload": payload,
        "n_pairs": args.frames,
        "wall_s": time.time() - t0,
    }
    if mask_mode != "off" and mask_rows:
        keys = [k for k in mask_rows[0] if k != "pair"]
        record["delta_mask_geometry"] = {
            k: float(np.mean([r[k] for r in mask_rows])) for k in keys}
        record["delta_mask_per_pair"] = mask_rows
        record["foa_thresholds"] = {
            "live_below": A1_FOA_LIVE_BELOW, "closed_above": A1_FOA_CLOSED_ABOVE,
            "pre_registered_in": "ddm_a1s_alpha_sign_verdict_20260816.md section 8 FO-A",
        }
        at025 = next((r for r in rows if r["alpha"] == 0.25), None)
        if at025 is None:
            raise Sr1Error("FO-A adjudicates at alpha = 0.25; that rung is not on the ladder")
        drift = at025["pose_drift_rms_vs_alpha0"]
        verdict, scope = _a1_foa_verdict(mask_mode, drift)
        record["foa_verdict"] = {
            "verdict": verdict, "verdict_scope": scope,
            "alpha": 0.25, "band_only_pose_drift_rms": drift,
            "vs_incumbent_pose_error": drift / math.sqrt(HV1_D_POSE),
        }
    write_json(args.work / f"SR1_A1POSE{suffix.upper()}.json", record)
    print(json.dumps({"ladder": rows, "foa": record.get("foa_verdict"),
                      "geometry": record.get("delta_mask_geometry")},
                     indent=2, sort_keys=True))
    return record


# ============================================================================================
# stage: ledger
# ============================================================================================
def stage_ledger(args: argparse.Namespace) -> dict:
    def load(name: str) -> dict | None:
        p = args.work / name
        return json.loads(p.read_text()) if p.exists() else None

    rop = load("SR1_ROPERATOR.json")
    wat = load("SR1_WATERFILL.json")

    rows = [
        {"row": "hv1 ep0634 seg term", "flips": RT1_SCORED_FLIPS,
         "S": RT1_SCORED_FLIPS * SEG_DS_PER_FLIP,
         "vs_gap": RT1_SCORED_FLIPS * SEG_DS_PER_FLIP / GAP_S},
        {"row": "label channel (transmitted vs GT)", "flips": RT1_LABEL_FLIPS,
         "S": RT1_LABEL_FLIPS * SEG_DS_PER_FLIP,
         "vs_gap": RT1_LABEL_FLIPS * SEG_DS_PER_FLIP / GAP_S},
        {"row": "MANUFACTURED round trip", "flips": RT1_ROUND_TRIP_FLIPS,
         "S": RT1_ROUND_TRIP_FLIPS * SEG_DS_PER_FLIP,
         "vs_gap": RT1_ROUND_TRIP_FLIPS * SEG_DS_PER_FLIP / GAP_S},
    ]
    ceilings = {
        "zero_byte_actuator_share_of_round_trip_needed_to_close_gap":
            GAP_S / (RT1_ROUND_TRIP_FLIPS * SEG_DS_PER_FLIP),
        "seg_market_bar_bytes_per_flip": BYTES_PER_FLIP_BAR,
        "rt1_channel_realised_bytes_per_recovered_flip":
            RT1_CHANNEL_BYTES / (RT1_ETA_MEASURED * RT1_BAND_FLIPS),
    }
    record = {
        "schema": "ddm_sr1_ledger.v1",
        "axis": "[macOS-CPU advisory] scorer-free -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "base": {"archive_sha256": BASE_ARCHIVE_SHA256, "S": BASE_S,
                 "bytes": BASE_ARCHIVE_BYTES, "gap_to_0p15": GAP_S},
        "exchange_rates": {
            "seg_dS_per_flip": SEG_DS_PER_FLIP,
            "rate_dS_per_byte": RATE_DS_PER_BYTE,
            "bytes_per_flip_break_even": BYTES_PER_FLIP_BAR,
        },
        "attribution": rows,
        "ceilings": ceilings,
        "roperator_present": rop is not None,
        "waterfill_present": wat is not None,
        "roperator_headline": None if rop is None else {
            "positive_control_passed": rop["positive_control"]["passed"],
            "twod_cond": rop["spectrum"]["twod_cond"],
            "twod_share_below_0p9": rop["spectrum"]["twod_share_below_0p9"],
            "band_blur_mean_levels": rop["realised_blur"]["band_blur_mean_levels"],
            "band_over_interior": rop["realised_blur"]["band_over_interior"],
            "precomp_clip_share_mean": rop["realised_blur"]["precomp_clip_share_mean"],
        },
        "waterfill_headline": None if wat is None else {
            "breakeven_density_by_eta": wat["breakeven_density_by_eta"],
            "flip_share_in_cells_above": wat["density"]["flip_share_in_cells_above"],
            "marginal_at_measured_eta": wat["curves_by_eta"][
                f"{RT1_ETA_MEASURED:.4f}"]["marginal_waterfill"],
        },
    }
    write_json(args.work / "SR1_LEDGER.json", record)
    return record


# ============================================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage",
                    choices=["roperator", "sign", "emphasis", "waterfill", "a1sign",
                             "a1pose", "ledger"],
                    required=True)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--base-argmax", type=Path, default=DEFAULT_BASE_ARGMAX)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--n-pairs", type=int, default=24,
                    help="roperator: seeded-random pairs for the realised-blur read")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tikhonov", type=float, default=1e-6,
                    help="roperator: Tikhonov lambda for the regularised A^-1")
    ap.add_argument("--row-bands", type=int, default=8,
                    help="waterfill: number of free row strata in the cell key")
    ap.add_argument("--max-degree", type=int, default=4,
                    help="waterfill: cap on the neighbourhood class-degree feature")
    ap.add_argument("--gt-mkv", type=Path, default=REPO / "upstream" / "videos" / "0.mkv")
    ap.add_argument("--alpha-max", type=float, default=1.5,
                    help="sign: largest pre-compensation strength swept")
    ap.add_argument("--alpha-step", type=float, default=0.05)
    ap.add_argument("--retain", action="store_true",
                    help="persist materialised arrays (ALWAYS KEEP THE PAYLOAD)")
    ap.add_argument("--verify-raw", action="store_true",
                    help="roperator: sha256 the 3.6 GB decode for custody (slow)")
    ap.add_argument("--allow-control-fail", action="store_true",
                    help="roperator: continue past a failed positive control (diagnosis only)")
    ap.add_argument("--diag-pairs", type=int, default=12,
                    help="a1sign: seeded-random pairs carrying the realisation-fidelity read")
    ap.add_argument("--gt-pose", type=Path, default=None,
                    help="a1pose: optional (600, 6) cached GT PoseNet target, advisory only")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="a1sign: flush the argmax fields and the resume marker every N pairs")
    ap.add_argument("--delta-mask", choices=list(A1_DELTA_MASKS), default="off",
                    help="a1sign/a1pose (FO-A): restrict the A1 camera perturbation to the "
                         "lifted label BAND, or to its INTERIOR complement.  Default 'off' is "
                         "byte-identical to the ddm_a1s row -- the mask is never applied and no "
                         "token field is read.")
    args = ap.parse_args(argv)

    # An inert flag is a config-orphan: it looks like it did something and did not.  Refuse it
    # on the stages that cannot consume it rather than silently ignoring it.
    if args.delta_mask != "off" and args.stage not in ("a1sign", "a1pose"):
        ap.error(f"--delta-mask is consumed only by a1sign/a1pose, not by '{args.stage}'")

    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage == "roperator":
        stage_roperator(args)
    elif args.stage == "sign":
        stage_sign(args)
    elif args.stage == "emphasis":
        stage_emphasis(args)
    elif args.stage == "waterfill":
        stage_waterfill(args)
    elif args.stage == "a1sign":
        stage_a1sign(args)
    elif args.stage == "a1pose":
        stage_a1pose(args)
    else:
        stage_ledger(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
