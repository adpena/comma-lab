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
SEG_DS_PER_FLIP = 100.0 / SCORED_PX  # 8.477116e-07
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
                    choices=["roperator", "sign", "emphasis", "waterfill", "ledger"],
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
    args = ap.parse_args(argv)

    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage == "roperator":
        stage_roperator(args)
    elif args.stage == "sign":
        stage_sign(args)
    elif args.stage == "emphasis":
        stage_emphasis(args)
    elif args.stage == "waterfill":
        stage_waterfill(args)
    else:
        stage_ledger(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
