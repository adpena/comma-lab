# SPDX-License-Identifier: MIT
"""blind_coordinate — the BLIND-COORDINATE rate lever (#401).

Camera pixels that BOTH scorers are structurally blind to, filled with a generic
(video-INDEPENDENT) decode-time rule so the encoder stores NOTHING there — a rate cut at
PROVABLY zero Δd_seg / Δd_pose.

THE GEOMETRY (derived, verified — never trusted-from-memory):

    Both scorers put every camera frame through the SAME preprocess resize
    (``upstream/modules.py``):

        SegNet.preprocess_input:  x[:, -1]        -> interpolate(size=(384,512), bilinear)
        PoseNet.preprocess_input: both frames     -> interpolate(size=(384,512), bilinear)
                                                   -> rgb_to_yuv6   (on the 384x512 grid)

    The resize ``(CAMERA_H=874, CAMERA_W=1164) -> (SEG_H=384, SEG_W=512)`` is bilinear,
    ``align_corners=False``, ``antialias=False`` (torch defaults). Downsampling by ~2.28x
    WITHOUT antialias SKIPS input pixels: each output sample touches only its 2 bracketing
    input samples, so input rows/cols not adjacent to ANY output-sample coordinate receive
    weight EXACTLY 0.0 (bilinear, not an epsilon). ``rgb_to_yuv6`` runs AFTER the resize on
    the 384x512 grid — it reads EVERY resized pixel (y00/y10/y01/y11 tile the grid; U/V
    average it) and never reaches back to camera pixels, so it adds NO blindness of its own.

    Therefore blindness is a SINGLE camera-grid mask determined by ONE resize:
      * a camera ROW ``i`` is blind  iff ``down_col[:, i]`` is all-zero   (106 of 874),
      * a camera COL ``l`` is blind  iff ``down_row[:, l]`` is all-zero   (140 of 1164),
      * a camera PIXEL ``(i, l)`` is blind iff  row ``i`` blind  OR  col ``l`` blind.
    The mask is the SAME for frame0 (PoseNet-only) and frame1 (both scorers) because the
    resize is identical, so a pixel blind here is blind through EVERY scored path.

    The RETAINED (non-blind) set is exactly the product ``(non-blind rows) x (non-blind
    cols)`` = a regular ``768 x 1024`` sub-grid (786,432 px); the blind set is the
    complement = ``230,904`` px/frame = ``22.6969%`` of the ``874 x 1164`` camera frame.

THE PROOF is BIT-IDENTITY THROUGH R, not a scorer run (:func:`bit_identity_report`): fill
the blind set with ARBITRARY content, recompute the real torch scorer-input tensors
(``posenet_in`` + ``segnet_in``) for both frames, assert bit-for-bit equality to the
unfilled tensors. Because the scorer INPUT is bit-identical, the scorer OUTPUT — hence
d_seg and d_pose — is bit-identical by construction. Run on n600 or it is not evidence.

THE FILL is a rule-118 GENERIC ALGORITHM (:func:`generic_inpaint_fill`): the receiver, given
ONLY the stored non-blind sub-grid, reconstructs every blind pixel by separable linear
interpolation over the KNOWN retained-index coordinates. It is data-INDEPENDENT (the mask
is a fixed function of the resize kernel, re-derivable inside inflate.py for FREE) and reads
only stored, video-derived, COUNTED pixels — so the generator ALGORITHM is free (rule 118)
while the video-derived payload it consumes is the counted archive content. The encoder
that adopts this lever stores the ``768 x 1024`` retained sub-grid instead of the full
``874 x 1164`` frame: the blind residuals are never emitted.

    ---- rule-118 boundary (BINDING) --------------------------------------------------
    FREE in the receiver (inflate.py): the blind-mask derivation (impulse-probe the resize
      kernel) + the separable linear-interp fill — a generic, deterministic algorithm with
      NO video-derived constants baked in.
    COUNTED in archive.zip: the retained non-blind sub-grid pixel values (the video-derived
      payload). NOTHING blind is stored. Smuggling a blind-region table into the "code" to
      dodge the rate term is the hide-data-in-code FAKE (CLAUDE.md NO-FAKE #6/#7) — the fill
      MUST regenerate blind pixels from the retained payload alone.
    -----------------------------------------------------------------------------------

AUTHORITY: the blind-mask + bit-identity are exact PROOFS (mask is a deterministic kernel
property; bit-identity is torch-verified over n600). The byte delta is a real lossless
byte-close (brotli/zlib) of real camera-resolution video content — ``[macOS-CPU advisory /
derivation]`` NON-PROMOTABLE; a promotable ΔS needs the full n600 byte-close + exact eval on
the chosen chain. The pointer does NOT move here; this is the rate MEANS.

SCOPE (honest): the byte delta is the saving available to ANY camera-resolution-storing
representation (raw frames, per-frame residual, HF sidecar). A pure GENERATOR witness
archive stores no camera pixels, so its DIRECT saving is 0 until it carries a camera-res
residual/sidecar section — then this lever drops 22.7% of that section's spatial content.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from tac.through_r.resolution_chain import (
    CAMERA_H,
    CAMERA_W,
    RGB_CHANNELS,
    SEG_H,
    SEG_W,
)

BLIND_COORD_LABEL = "[macOS-CPU advisory / derivation . BLIND-COORDINATE . NON-PROMOTABLE]"


class BlindCoordinateError(ValueError):
    """Raised on a mis-shaped / non-authority / toy input to the blind-coordinate lever."""


# ======================================================================================
# 1. BLIND MASK DERIVATION  (reuses the #391 exact resize-kernel machinery)
# ======================================================================================
@dataclass(frozen=True)
class BlindMask:
    """The camera-grid blind mask + its separable row/col structure and provenance.

    ``mask[i, l]`` is True iff camera pixel ``(i, l)`` receives weight EXACTLY 0.0 in the
    ``(CAMERA_H, CAMERA_W) -> (SEG_H, SEG_W)`` bilinear resize (hence in EVERY scored path).
    ``blind_rows`` / ``blind_cols`` are the 1-D boolean vectors; the mask is their OR.
    """

    mask: np.ndarray  # (CAMERA_H, CAMERA_W) bool
    blind_rows: np.ndarray  # (CAMERA_H,) bool
    blind_cols: np.ndarray  # (CAMERA_W,) bool

    @property
    def n_blind(self) -> int:
        return int(self.mask.sum())

    @property
    def n_total(self) -> int:
        return int(self.mask.size)

    @property
    def n_retained(self) -> int:
        return self.n_total - self.n_blind

    @property
    def retained_rows(self) -> np.ndarray:
        """Sorted indices of the non-blind camera rows (the retained sub-grid row coords)."""
        return np.where(~self.blind_rows)[0]

    @property
    def retained_cols(self) -> np.ndarray:
        return np.where(~self.blind_cols)[0]

    @property
    def retained_hw(self) -> tuple[int, int]:
        """Dense retained sub-grid shape ``(H', W')`` — the non-blind rows x non-blind cols."""
        return int((~self.blind_rows).sum()), int((~self.blind_cols).sum())

    def fraction(self) -> dict[str, Any]:
        """Machine-readable blind-fraction report (max observability)."""

        rh, rw = self.retained_hw
        return {
            "schema": "blind_coordinate_fraction.v1",
            "camera_hw": [CAMERA_H, CAMERA_W],
            "seg_hw": [SEG_H, SEG_W],
            "n_blind_rows": int(self.blind_rows.sum()),
            "n_blind_cols": int(self.blind_cols.sum()),
            "n_retained_rows": rh,
            "n_retained_cols": rw,
            "n_blind_px": self.n_blind,
            "n_retained_px": self.n_retained,
            "n_total_px": self.n_total,
            "blind_fraction": self.n_blind / self.n_total,
            "retained_subgrid_hw": [rh, rw],
            "inclusion_exclusion_check": (
                int(self.blind_rows.sum()) * CAMERA_W
                + int(self.blind_cols.sum()) * CAMERA_H
                - int(self.blind_rows.sum()) * int(self.blind_cols.sum())
            ),
        }


@lru_cache(maxsize=1)
def build_blind_mask() -> BlindMask:
    """Derive the camera-grid blind mask from the REAL bilinear resize kernel.

    Reuses :func:`tac.through_r.flip_inverse.resize_matrix_1d` (#391) to extract the exact
    ``(SEG_H, CAMERA_H)`` and ``(SEG_W, CAMERA_W)`` bilinear-down operators by impulse-probing
    torch (NOT re-derived from the interpolation formula — the operating-manual
    "validate against the primary artifact"). A camera row/col is blind iff its operator
    column is all-zero. Cached (the mask is a fixed property of the pinned resize).
    """

    from tac.through_r.flip_inverse import resize_matrix_1d

    # down operators: down_col (SEG_H, CAMERA_H), down_row (SEG_W, CAMERA_W)
    down_col = resize_matrix_1d(CAMERA_H, SEG_H, "bilinear", align_corners=False)
    down_row = resize_matrix_1d(CAMERA_W, SEG_W, "bilinear", align_corners=False)
    # a camera index is blind iff it contributes to NO output sample (all-zero kernel column).
    # bilinear gives EXACTLY 0.0 for skipped samples (verified: min nonzero weight ~2.6e-3).
    blind_rows = np.abs(down_col).max(axis=0) == 0.0  # (CAMERA_H,)
    blind_cols = np.abs(down_row).max(axis=0) == 0.0  # (CAMERA_W,)
    mask = blind_rows[:, None] | blind_cols[None, :]
    return BlindMask(
        mask=np.ascontiguousarray(mask),
        blind_rows=np.ascontiguousarray(blind_rows),
        blind_cols=np.ascontiguousarray(blind_cols),
    )


def blind_fraction() -> dict[str, Any]:
    """Convenience: the blind-fraction report for the pinned camera/seg geometry."""

    return build_blind_mask().fraction()


# ======================================================================================
# 2. RETAINED SUB-GRID  (the COUNTED payload)  +  GENERIC FILL  (the FREE algorithm)
# ======================================================================================
def _check_camera_frame(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame)
    if a.shape != (CAMERA_H, CAMERA_W, RGB_CHANNELS):
        raise BlindCoordinateError(
            f"expected camera frame ({CAMERA_H},{CAMERA_W},{RGB_CHANNELS}); got {a.shape}"
        )
    return a


def extract_retained_subgrid(frame: np.ndarray, bm: BlindMask | None = None) -> np.ndarray:
    """The dense ``(H', W', 3)`` non-blind sub-grid — the ONLY pixels an encoder stores.

    Because the blind set is (blind-rows OR blind-cols), the retained set is the exact
    product (non-blind rows) x (non-blind cols) = a REGULAR dense sub-grid. This is the
    COUNTED video-derived payload; the blind complement is regenerated for free downstream.
    """

    a = _check_camera_frame(frame)
    bm = bm or build_blind_mask()
    rr, cc = bm.retained_rows, bm.retained_cols
    return np.ascontiguousarray(a[np.ix_(rr, cc)])


def generic_inpaint_fill(
    retained_subgrid: np.ndarray, bm: BlindMask | None = None
) -> np.ndarray:
    """RECEIVER decode-time GENERIC fill: retained ``(H',W',3)`` -> full ``(874,1164,3)``.

    rule-118 FREE algorithm. Places the retained pixels at their known non-blind
    (row, col) coordinates and reconstructs EVERY blind pixel by SEPARABLE LINEAR
    interpolation over the retained-index coordinates (np.interp along rows, then cols).
    Deterministic + data-INDEPENDENT (depends only on the fixed mask geometry) — reads only
    the stored, video-derived retained payload. Blind pixels outside the retained coordinate
    span are edge-clamped (np.interp constant-extends), keeping the rule total.

    The exact fill VALUES are irrelevant to the score (blind pixels have zero scorer weight
    — that is the whole point); linear interp is chosen because it is smooth and maximally
    codeable, and it makes the receiver's reconstruction deterministic and reviewable.
    """

    sub = np.asarray(retained_subgrid)
    bm = bm or build_blind_mask()
    rr, cc = bm.retained_rows, bm.retained_cols
    rh, rw = bm.retained_hw
    if sub.shape != (rh, rw, RGB_CHANNELS):
        raise BlindCoordinateError(
            f"retained sub-grid must be {(rh, rw, RGB_CHANNELS)}; got {sub.shape}"
        )
    work_dtype = np.float64
    subf = sub.astype(work_dtype)
    # Pass 1: interpolate along COLUMNS to full width at the retained row positions.
    #   rows_full_at_retained: (rh, CAMERA_W, 3)
    full_w = np.empty((rh, CAMERA_W, RGB_CHANNELS), dtype=work_dtype)
    all_cols = np.arange(CAMERA_W)
    for k in range(RGB_CHANNELS):
        for a in range(rh):
            full_w[a, :, k] = np.interp(all_cols, cc, subf[a, :, k])
    # Pass 2: interpolate along ROWS to full height.
    out = np.empty((CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=work_dtype)
    all_rows = np.arange(CAMERA_H)
    for k in range(RGB_CHANNELS):
        for b in range(CAMERA_W):
            out[:, b, k] = np.interp(all_rows, rr, full_w[:, b, k])
    if sub.dtype == np.uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out.astype(sub.dtype)


def apply_blind_fill(
    frame: np.ndarray, fill: np.ndarray | int | None = None, bm: BlindMask | None = None
) -> np.ndarray:
    """Overwrite the blind pixels of ``frame`` (test / proof helper).

    ``fill=None`` -> the generic inpaint reconstruction (the receiver output); an int or a
    broadcastable array -> overwrite blind pixels with that content (used by the bit-identity
    proof to prove ARBITRARY blind content is invisible). Non-blind pixels are untouched.
    """

    a = _check_camera_frame(frame).copy()
    bm = bm or build_blind_mask()
    if fill is None:
        recon = generic_inpaint_fill(extract_retained_subgrid(a, bm), bm)
        a[bm.mask] = recon[bm.mask]
        return a
    if np.isscalar(fill):
        a[bm.mask] = fill
        return a
    f = np.asarray(fill)
    if f.shape == a.shape:
        a[bm.mask] = f[bm.mask]
    else:
        a[bm.mask] = f  # broadcast (e.g. per-blind-pixel (n_blind, 3))
    return a


# ======================================================================================
# 3. BIT-IDENTITY THROUGH R  (the PROOF — no scorer run needed)
# ======================================================================================
def _scorer_inputs(f0: np.ndarray, f1: np.ndarray) -> tuple[Any, Any]:
    """Compute the REAL torch scorer-input tensors (posenet_in, segnet_in) for a pair.

    Op-for-op mirror of ``DistortionNet.preprocess_input`` + the two scorer
    ``preprocess_input`` (``upstream/modules.py``): rearrange to (b,t,c,h,w).float ->
    PoseNet [both frames -> bilinear(384,512) -> rgb_to_yuv6] + SegNet [frame1 ->
    bilinear(384,512)]. Returns the exact tensors fed to the frozen nets; if these are
    bit-identical, d_seg and d_pose are bit-identical BY CONSTRUCTION.
    """

    import sys
    from pathlib import Path

    import torch

    up = str(Path(__file__).resolve().parents[3] / "upstream")
    if up not in sys.path:
        sys.path.insert(0, up)
    import einops
    from frame_utils import rgb_to_yuv6  # type: ignore  # upstream

    x = torch.from_numpy(np.stack([f0, f1])[None]).float()  # (1,2,H,W,3)
    x = einops.rearrange(x, "b t h w c -> b t c h w", b=1, t=2, c=3)
    # PoseNet path (both frames)
    xp = einops.rearrange(x, "b t c h w -> (b t) c h w")
    xp = torch.nn.functional.interpolate(xp, size=(SEG_H, SEG_W), mode="bilinear")
    posenet_in = einops.rearrange(rgb_to_yuv6(xp), "(b t) c h w -> b (t c) h w", b=1, t=2)
    # SegNet path (last frame only)
    xs = x[:, -1]
    segnet_in = torch.nn.functional.interpolate(xs, size=(SEG_H, SEG_W), mode="bilinear")
    return posenet_in, segnet_in


@dataclass
class BitIdentityResult:
    """Per-run bit-identity-through-R proof outcome."""

    n_pairs: int
    all_bit_identical: bool
    max_abs_diff_pose: float
    max_abs_diff_seg: float
    n_failures: int
    failing_pairs: list[int]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": "blind_coordinate_bit_identity.v1",
            "n_pairs": self.n_pairs,
            "all_bit_identical": self.all_bit_identical,
            "max_abs_diff_pose": self.max_abs_diff_pose,
            "max_abs_diff_seg": self.max_abs_diff_seg,
            "n_failures": self.n_failures,
            "failing_pairs": self.failing_pairs,
            "label": BLIND_COORD_LABEL,
        }


def bit_identity_report(
    frames_f0: np.ndarray,
    frames_f1: np.ndarray,
    *,
    seed: int = 0,
    bm: BlindMask | None = None,
    fill_mode: str = "random",
) -> BitIdentityResult:
    """PROOF: filling the blind set with ARBITRARY content leaves both scorer inputs
    bit-identical, over every provided pair.

    ``frames_f0`` / ``frames_f1`` are ``(N, 874, 1164, 3)`` uint8 real camera frames. For each
    pair, blind pixels of BOTH frames are overwritten (``fill_mode`` in
    {'random','max','zero'} — random is the strongest adversarial choice) and the real torch
    scorer inputs are recomputed and compared bit-for-bit to the unfilled inputs.
    """

    import torch

    bm = bm or build_blind_mask()
    a0 = np.asarray(frames_f0)
    a1 = np.asarray(frames_f1)
    if a0.ndim != 4 or a0.shape[1:] != (CAMERA_H, CAMERA_W, RGB_CHANNELS):
        raise BlindCoordinateError(f"frames_f0 must be (N,874,1164,3) uint8; got {a0.shape}")
    if a1.shape != a0.shape:
        raise BlindCoordinateError("frames_f0 and frames_f1 must have the same shape")
    n = a0.shape[0]
    if n == 0:
        raise BlindCoordinateError("bit_identity_report needs at least one pair (no toy empties)")
    rng = np.random.default_rng(seed)
    max_pose = 0.0
    max_seg = 0.0
    failures: list[int] = []
    n_blind = bm.n_blind
    for i in range(n):
        f0 = a0[i]
        f1 = a1[i]
        p0, s0 = _scorer_inputs(f0, f1)
        if fill_mode == "random":
            fill0 = rng.integers(0, 256, size=(n_blind, RGB_CHANNELS), dtype=np.uint8)
            fill1 = rng.integers(0, 256, size=(n_blind, RGB_CHANNELS), dtype=np.uint8)
        elif fill_mode == "max":
            fill0 = fill1 = 255
        elif fill_mode == "zero":
            fill0 = fill1 = 0
        else:
            raise BlindCoordinateError(f"fill_mode must be random|max|zero; got {fill_mode!r}")
        f0f = apply_blind_fill(f0, fill0, bm)
        f1f = apply_blind_fill(f1, fill1, bm)
        p1, s1 = _scorer_inputs(f0f, f1f)
        dp = float((p0 - p1).abs().max().item())
        ds = float((s0 - s1).abs().max().item())
        max_pose = max(max_pose, dp)
        max_seg = max(max_seg, ds)
        if not (torch.equal(p0, p1) and torch.equal(s0, s1)):
            failures.append(i)
    return BitIdentityResult(
        n_pairs=n,
        all_bit_identical=(len(failures) == 0),
        max_abs_diff_pose=max_pose,
        max_abs_diff_seg=max_seg,
        n_failures=len(failures),
        failing_pairs=failures,
    )


# ======================================================================================
# 4. BYTE DELTA  (real lossless byte-close of real camera-res video content)
# ======================================================================================
def _lossless_bytes(arr: np.ndarray) -> tuple[int, str]:
    """Lossless byte-close of a uint8 array; brotli-q11 preferred, zlib-9 fallback."""

    raw = np.ascontiguousarray(arr).tobytes()
    try:
        import brotli  # type: ignore

        return len(brotli.compress(raw, quality=11)), "brotli-q11"
    except Exception:
        import zlib

        return len(zlib.compress(raw, 9)), "zlib-9"


@dataclass
class ByteDeltaResult:
    """Per-frame-averaged byte delta from dropping the blind sub-grid."""

    codec: str
    n_frames: int
    bytes_full_mean: float
    bytes_retained_mean: float
    byte_delta_mean: float
    delta_fraction_mean: float
    blind_fraction: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": "blind_coordinate_byte_delta.v1",
            "codec": self.codec,
            "n_frames": self.n_frames,
            "bytes_full_mean": self.bytes_full_mean,
            "bytes_retained_mean": self.bytes_retained_mean,
            "byte_delta_mean": self.byte_delta_mean,
            "delta_fraction_mean": self.delta_fraction_mean,
            "blind_fraction": self.blind_fraction,
            "label": BLIND_COORD_LABEL,
        }


def measure_byte_delta(
    frames: np.ndarray, *, bm: BlindMask | None = None
) -> ByteDeltaResult:
    """Real byte delta: lossless byte-close of the FULL frame vs the retained sub-grid.

    ``frames`` is ``(N, 874, 1164, 3)`` uint8 real camera content. For each frame:
    ``bytes_full`` = codec(full 874x1164) vs ``bytes_retained`` = codec(768x1024 retained
    sub-grid) — the blind pixels the receiver regenerates for FREE are simply NOT stored. The
    delta is the exact byte saving of the lever on that representation. Averages over frames.
    """

    bm = bm or build_blind_mask()
    a = np.asarray(frames)
    if a.ndim != 4 or a.shape[1:] != (CAMERA_H, CAMERA_W, RGB_CHANNELS):
        raise BlindCoordinateError(f"frames must be (N,874,1164,3) uint8; got {a.shape}")
    if a.shape[0] == 0:
        raise BlindCoordinateError("measure_byte_delta needs at least one frame (no toy empties)")
    full_bytes: list[int] = []
    ret_bytes: list[int] = []
    codec = ""
    for i in range(a.shape[0]):
        bf, codec = _lossless_bytes(a[i])
        br, _ = _lossless_bytes(extract_retained_subgrid(a[i], bm))
        full_bytes.append(bf)
        ret_bytes.append(br)
    fb = float(np.mean(full_bytes))
    rb = float(np.mean(ret_bytes))
    deltas = np.asarray(full_bytes, float) - np.asarray(ret_bytes, float)
    fracs = deltas / np.asarray(full_bytes, float)
    return ByteDeltaResult(
        codec=codec,
        n_frames=int(a.shape[0]),
        bytes_full_mean=fb,
        bytes_retained_mean=rb,
        byte_delta_mean=float(np.mean(deltas)),
        delta_fraction_mean=float(np.mean(fracs)),
        blind_fraction=bm.n_blind / bm.n_total,
    )


__all__ = [
    "BLIND_COORD_LABEL",
    "BitIdentityResult",
    "BlindCoordinateError",
    "BlindMask",
    "ByteDeltaResult",
    "apply_blind_fill",
    "bit_identity_report",
    "blind_fraction",
    "build_blind_mask",
    "extract_retained_subgrid",
    "generic_inpaint_fill",
    "measure_byte_delta",
]
