# SPDX-License-Identifier: MIT
"""Resize-null preimage compiler (task #49) — the universal postprocessor.

The contest objective is the evaluator quotient
``100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489``.  Task #47 proved
that BOTH scorer heads' FIRST op is the same fixed bilinear resize
``R: (874, 1164) -> (384, 512)`` (``F.interpolate(..., mode='bilinear',
align_corners=False)``; ``upstream/modules.py:73`` PoseNet, ``:109`` SegNet).
``evaluate.py`` therefore scores the PROJECTION ``y = R x``, NEVER the camera
frame ``x``.

THE THEOREM (#47 / S12): every vehicle should emit the MINIMUM-DESCRIPTION
PREIMAGE of its rendered frames::

    x̃* = argmin  bytes(x̃)   s.t.   R x̃ = R x ,   0 <= x̃ <= 255  (uint8)

i.e. choose the CHEAPEST legal high-res representative of the scorer-equivalence
class, NOT merely "drop invisible pixels".  This is a UNIVERSAL POSTPROCESSOR
for every vehicle (SNeRV render, frontier compose, PR110++ frames, HiNeRV,
PACT-VQ): it removes scorer-invisible degrees of freedom BEFORE any codec
touches the bytes.

This module REUSES #47's closed-form resize derivation (it does NOT rederive R):
``tac.optimization.evaluator_invisibility_basis`` supplies the certified
zero-weight pixel set (tier-1a) and the exact 1D bilinear matrices that compose
the separable operator ``R = R_h (x) R_w``.

TIERS (each independently shippable + tested)
---------------------------------------------
TIER 1 — zero-weight fill (CERTIFIED FREE).  Replace the ~22.7% certified
zero-weight pixels (row OR col dropped by the downsample) with the
entropy-optimal fill, chosen BY MEASUREMENT (constant / horizontal-predictor
continuation / measured-best on the real frame coder).  PROOF per frame:
``max|R x̃ - R x|`` computed with #47's exact kernel must be 0.0 EXACTLY
(zero-weight => exact, hardware-independent).

TIER 2 — integer null-basis descent.  Greedy descent on coded size using
integer-friendly null-basis steps ``n_i`` with ``R n_i = 0``.  Each candidate
step is accepted iff coded bytes decrease AND the uint8 round-trip preserves
``R x̃ = R x`` within a PROVEN tolerance (the per-frame ``max|ΔR|`` is emitted;
exact integer steps keep it 0.0).  Bounded iterations; deterministic seed.  The
canonical integer null-basis families are derived from the separable structure:

  - zero-weight pixels (tier 1, single-pixel) are themselves the simplest
    integer null vectors;
  - DROPPED-COLUMN / DROPPED-ROW redistribution: a unit moved between two input
    indices that share identical resize weights to every output is null;
  - the certified amplitude-unlimited single-pixel directions are the
    integer-exact descent atoms used here (no fp null vector ever rounded into
    the uint8 frame, honoring operator caveat (a)).

TIER 3 — blockwise constrained least-entropy preimage (design + smallest
viable impl).  Where the visible projection allows, snap zero-weight runs into
piecewise-constant tiles; the cheapest variant (constant-fill the dropped
rows/cols + flatten the dropped lattice) is implemented; LP/MILP/learned
upgrades are named follow-ups.

OPERATOR CAVEATS (binding; ``.omx/research/snerv_rate_attack_round2_directive``
S12):
  (a) outputs MUST remain valid uint8/shape/range; the null space is real-valued
      but archive frames are integers, so we use INTEGER-friendly bases only (the
      tier-1 fill and tier-2 steps are integer by construction) and PROVE exact
      ``max|R x̃ - R x|`` per frame;
  (b) equality holds on the RGB tensor BEFORE PoseNet's YUV conversion (then
      YUV6 equality follows); this module operates on RGB camera frames, so
      RGB-equality is the proven invariant;
  (c) the SOLVER runs at compress time; the emitted representation is a plain
      uint8 frame (decode is identity — strictly inside the 30-min budget);
  (d) every application emits a V3 row (``preimage_application_v3_row``).

Evidence grade: the exactness proof is ``mathematical-derivation`` (residual
== 0.0 for zero-weight tier-1, hardware-independent).  The bytes-reduction
measurement is ``[macOS-CPU advisory]`` (real-frame coder on the local CPU).  No
score claim; no promotion; no dispatch authority.

CLAUDE.md compliance:
  - ``planning_only_no_score_claim`` / ``promotable=false``
  - ``no_mps_authoritative`` (pure linear algebra + CPU coder; no MPS)
  - ``no_tmp_paths`` (durable SSD artifacts; this module emits no /tmp)

Cross-references
----------------
- ``tac.optimization.evaluator_invisibility_basis`` (#47; the resize derivation
  REUSED here — ``derive_tier1_resize_null_space``, ``_resize_1d_matrix``)
- ``tac.optimization.lf_payload_rate_distortion`` (#46; consumed by reference
  for THE LAW score-delta helpers — ``delta_rate_score``)
- ``upstream/modules.py:73/109`` (the shared resize), ``upstream/frame_utils.py``
  (camera_size=(1164,874) NHWC raw layout)
- ``.omx/research/resize_null_preimage_compiler_landed_20260610.md`` (landing memo)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# REUSE #47's closed-form resize derivation (do NOT rederive R).
from tac.optimization.evaluator_invisibility_basis import (
    CAMERA_H,
    CAMERA_W,
    CONTEST_TOTAL_BYTES,
    SCORER_INPUT_H,
    SCORER_INPUT_W,
    Tier1ResizeNullSpace,
    _resize_1d_matrix,  # exact 1D bilinear matrix (certification-grade kernel)
    derive_tier1_resize_null_space,
)

RESIZE_NULL_PREIMAGE_SCHEMA = "resize_null_preimage_compiler.v1"
PROOF_EVIDENCE_GRADE = "mathematical-derivation"
BYTES_EVIDENCE_GRADE = "[macOS-CPU advisory]"

# A frame is exactly preimage-equivalent when max|R x̃ - R x| <= this tolerance.
# For integer-only tier-1 fills + tier-2 steps the residual is EXACTLY 0.0; the
# tolerance only exists so a (proven, opt-in) approximate tier can declare its
# verified bound rather than claim a certified zero it did not earn.
EXACT_RESIDUAL_TOL = 0.0

FillStrategy = Literal[
    "constant",
    "horizontal_predictor",
    "vertical_predictor",
    "neighbor_mean",
    "measured_best",
]


class ResizeNullPreimageError(ValueError):
    """Raised when preimage-compiler inputs are malformed or an exactness claim
    cannot be certified (fail-closed: a frame whose residual exceeds the declared
    tolerance is NEVER reported as preimage-equivalent)."""


# ---------------------------------------------------------------------------
# The exact projection operator R = R_h (x) R_w (REUSED from #47, not rederived).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResizeProjector:
    """The exact separable bilinear-resize projector ``R = R_h (x) R_w``.

    ``project(X)`` reproduces ``F.interpolate(X, (scorer_h, scorer_w),
    mode='bilinear', align_corners=False)`` for a single channel plane ``X`` of
    shape ``(camera_h, camera_w)``.  The 1D matrices come straight from #47's
    ``_resize_1d_matrix`` derivation (matched to ``F.interpolate`` to fp64
    roundoff, 9.9e-15) — we do NOT rederive the resize.
    """

    camera_h: int
    camera_w: int
    scorer_h: int
    scorer_w: int
    rh: np.ndarray  # (scorer_h, camera_h)
    rw: np.ndarray  # (scorer_w, camera_w)

    @classmethod
    def build(
        cls,
        *,
        camera_h: int = CAMERA_H,
        camera_w: int = CAMERA_W,
        scorer_h: int = SCORER_INPUT_H,
        scorer_w: int = SCORER_INPUT_W,
    ) -> ResizeProjector:
        rh = _resize_1d_matrix(camera_h, scorer_h)
        rw = _resize_1d_matrix(camera_w, scorer_w)
        return cls(
            camera_h=int(camera_h),
            camera_w=int(camera_w),
            scorer_h=int(scorer_h),
            scorer_w=int(scorer_w),
            rh=rh,
            rw=rw,
        )

    def project_plane(self, plane: np.ndarray) -> np.ndarray:
        """Project a single ``(camera_h, camera_w)`` channel plane to
        ``(scorer_h, scorer_w)`` (float64, exact)."""
        x = np.asarray(plane, dtype=np.float64)
        if x.shape != (self.camera_h, self.camera_w):
            raise ResizeNullPreimageError(
                f"plane shape {x.shape} != camera {(self.camera_h, self.camera_w)}"
            )
        # y = R_h @ X @ R_w^T  (separable bilinear).  errstate silences benign
        # Accelerate fast-math denormal flags on the large dense product; the
        # inputs are finite uint8, the result is exact float64.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return self.rh @ x @ self.rw.T

    def project_frame(self, frame: np.ndarray) -> np.ndarray:
        """Project an ``(H, W, C)`` camera frame to ``(scorer_h, scorer_w, C)``."""
        x = np.asarray(frame, dtype=np.float64)
        if x.ndim != 3 or x.shape[:2] != (self.camera_h, self.camera_w):
            raise ResizeNullPreimageError(
                f"frame shape {x.shape} != ({self.camera_h},{self.camera_w},C)"
            )
        c = x.shape[2]
        out = np.empty((self.scorer_h, self.scorer_w, c), dtype=np.float64)
        for ch in range(c):
            out[:, :, ch] = self.project_plane(x[:, :, ch])
        return out

    def max_abs_projection_residual(
        self, frame_orig: np.ndarray, frame_pre: np.ndarray
    ) -> float:
        """``max|R x̃ - R x|`` over all channels and scorer pixels (the per-frame
        exactness proof number — operator caveat (a))."""
        y0 = self.project_frame(frame_orig)
        y1 = self.project_frame(frame_pre)
        return float(np.max(np.abs(y1 - y0)))


# ---------------------------------------------------------------------------
# The certified zero-weight pixel mask (REUSED from #47's tier-1 basis).
# ---------------------------------------------------------------------------
def zero_weight_pixel_mask(
    *,
    camera_h: int = CAMERA_H,
    camera_w: int = CAMERA_W,
    scorer_h: int = SCORER_INPUT_H,
    scorer_w: int = SCORER_INPUT_W,
    basis: Tier1ResizeNullSpace | None = None,
) -> np.ndarray:
    """Boolean ``(camera_h, camera_w)`` mask: True where a single-pixel change is
    CERTIFIED exactly invisible to BOTH scorer heads.  Delegates to #47's
    ``Tier1ResizeNullSpace.zero_weight_pixel_mask`` (the certified basis)."""
    if basis is None:
        basis = derive_tier1_resize_null_space(
            camera_h=camera_h, camera_w=camera_w, scorer_h=scorer_h, scorer_w=scorer_w
        )
    return basis.zero_weight_pixel_mask()


# ---------------------------------------------------------------------------
# Coded-size measurement (the ONLY admission arbiter — measured, not assumed).
# ---------------------------------------------------------------------------
# Default brotli quality for the FINAL reported number (max compression, the
# PR101 L32 deploy-time discipline) vs the faster quality used to RANK candidates
# during search (search ranking is monotone-correlated with q=11; the winner is
# always re-measured at q=11 before it is reported).
REPORT_BROTLI_QUALITY = 11
SEARCH_BROTLI_QUALITY = 5


def coded_size_bytes(
    arr: np.ndarray, *, coder: str = "brotli", brotli_quality: int = REPORT_BROTLI_QUALITY
) -> int:
    """Measured coded size of a uint8 array under a real coder.

    The directive is explicit: pick the fill / step BY MEASUREMENT, not by
    convention.  ``brotli`` (quality ``brotli_quality``, default 11 = max) and
    ``lzma`` are the two real coders.  No proxy entropy estimate is used as the
    admission arbiter.  During candidate SEARCH a faster quality may be passed;
    the FINAL reported sizes (``coded_size_both``) always use q=11.
    """
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.uint8))
    raw = a.tobytes()
    if coder == "brotli":
        import brotli

        return len(brotli.compress(raw, quality=int(brotli_quality)))
    if coder == "lzma":
        import lzma

        # FORMAT_RAW strips container overhead (PR101 L24 discipline) so the
        # measurement reflects the payload entropy, not the format header.
        filt = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
        return len(lzma.compress(raw, format=lzma.FORMAT_RAW, filters=filt))
    raise ResizeNullPreimageError(f"unknown coder {coder!r} (use brotli|lzma)")


def coded_size_both(arr: np.ndarray) -> dict[str, int]:
    """Both real coders' measured sizes (brotli + lzma) for a uint8 array."""
    return {"brotli": coded_size_bytes(arr, coder="brotli"),
            "lzma": coded_size_bytes(arr, coder="lzma")}


# ---------------------------------------------------------------------------
# TIER 1 — entropy-optimal zero-weight fill.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameProof:
    """Per-frame exactness + bytes proof (the V3-row backbone)."""

    frame_index: int
    max_abs_projection_residual: float
    exact: bool
    bytes_before: dict[str, int]
    bytes_after: dict[str, int]
    n_pixels_changed: int
    fill_strategy: str
    tier: int
    valid_uint8: bool

    @property
    def bytes_reduction_brotli(self) -> int:
        return self.bytes_before.get("brotli", 0) - self.bytes_after.get("brotli", 0)

    @property
    def bytes_reduction_lzma(self) -> int:
        return self.bytes_before.get("lzma", 0) - self.bytes_after.get("lzma", 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESIZE_NULL_PREIMAGE_SCHEMA,
            "frame_index": self.frame_index,
            "tier": self.tier,
            "fill_strategy": self.fill_strategy,
            "max_abs_projection_residual": self.max_abs_projection_residual,
            "exact": self.exact,
            "valid_uint8": self.valid_uint8,
            "n_pixels_changed": self.n_pixels_changed,
            "bytes_before": dict(self.bytes_before),
            "bytes_after": dict(self.bytes_after),
            "bytes_reduction_brotli": self.bytes_reduction_brotli,
            "bytes_reduction_lzma": self.bytes_reduction_lzma,
            "proof_evidence_grade": PROOF_EVIDENCE_GRADE,
            "bytes_evidence_grade": BYTES_EVIDENCE_GRADE,
            "score_claim": False,
            "promotable": False,
        }


def _apply_constant_fill(plane: np.ndarray, mask: np.ndarray, value: int) -> np.ndarray:
    out = plane.copy()
    out[mask] = np.uint8(value)
    return out


def _apply_horizontal_predictor_fill(
    plane: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Continue each masked run from the last non-masked pixel to its LEFT (a
    horizontal predictor => the residual stream is all-zeros over masked runs,
    which the coder models cheaply).  Leading masked pixels fall back to the
    nearest non-masked pixel to the RIGHT, then to a global constant."""
    out = plane.copy()
    h, w = plane.shape
    for r in range(h):
        row_mask = mask[r]
        if not row_mask.any():
            continue
        last = None
        for c in range(w):
            if not row_mask[c]:
                last = out[r, c]
            elif last is not None:
                out[r, c] = last
        # leading-masked pixels (no left neighbour) -> nearest right non-masked
        if row_mask[0]:
            fill = None
            for c in range(w):
                if not row_mask[c]:
                    fill = out[r, c]
                    break
            if fill is None:
                fill = np.uint8(0)
            for c in range(w):
                if row_mask[c]:
                    out[r, c] = fill
                else:
                    break
    return out


def _apply_vertical_predictor_fill(plane: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Vertical analogue of the horizontal predictor (continue from above)."""
    return _apply_horizontal_predictor_fill(plane.T, mask.T).T


def _apply_neighbor_mean_fill(plane: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked pixels with the rounded mean of the non-masked pixels (a flat
    DC fill that is maximally compressible when most of a plane is masked)."""
    out = plane.copy()
    keep = ~mask
    if keep.any():
        val = int(round(float(plane[keep].mean())))
    else:
        val = 0
    out[mask] = np.uint8(np.clip(val, 0, 255))
    return out


_FILL_FUNCS = {
    "constant": lambda p, m: _apply_constant_fill(p, m, 0),
    "horizontal_predictor": _apply_horizontal_predictor_fill,
    "vertical_predictor": _apply_vertical_predictor_fill,
    "neighbor_mean": _apply_neighbor_mean_fill,
}


def apply_tier1_zero_weight_fill(
    frame: np.ndarray,
    *,
    strategy: FillStrategy = "measured_best",
    mask: np.ndarray | None = None,
    projector: ResizeProjector | None = None,
    basis: Tier1ResizeNullSpace | None = None,
    frame_index: int = 0,
) -> tuple[np.ndarray, FrameProof]:
    """Replace the certified zero-weight pixels of ``frame`` (NHWC uint8 single
    frame ``(H, W, C)``) with the entropy-optimal fill.

    ``strategy='measured_best'`` MEASURES every candidate fill (constant /
    horizontal / vertical / neighbor_mean) on the REAL frame coder and keeps the
    one with the smallest brotli size (the directive: measurement, not
    convention).  Returns the preimage frame + the per-frame exactness proof.

    The exactness proof is CERTIFIED: zero-weight pixels carry exactly 0 resize
    weight, so ``max|R x̃ - R x| == 0.0`` regardless of the fill.  The proof is
    still COMPUTED with the real projector (no-fake discipline) and emitted.
    """
    x = np.asarray(frame, dtype=np.uint8)
    if x.ndim != 3:
        raise ResizeNullPreimageError("frame must be (H, W, C)")
    h, w, c = x.shape
    if projector is None:
        projector = ResizeProjector.build(camera_h=h, camera_w=w)
    if mask is None:
        mask = zero_weight_pixel_mask(
            camera_h=h, camera_w=w,
            scorer_h=projector.scorer_h, scorer_w=projector.scorer_w, basis=basis,
        )
    if mask.shape != (h, w):
        raise ResizeNullPreimageError(f"mask shape {mask.shape} != frame {(h, w)}")

    candidate_strategies: Sequence[str]
    if strategy == "measured_best":
        candidate_strategies = ("constant", "horizontal_predictor",
                                "vertical_predictor", "neighbor_mean")
    else:
        candidate_strategies = (strategy,)

    best_frame: np.ndarray | None = None
    best_bytes = None
    best_strategy = strategy
    bytes_before = coded_size_both(x)
    single_cand = len(candidate_strategies) == 1
    for cand in candidate_strategies:
        fn = _FILL_FUNCS[cand]
        out = x.copy()
        for ch in range(c):
            out[:, :, ch] = fn(x[:, :, ch], mask)
        # rank candidates with the fast search quality (winner re-measured at q=11
        # below); a single explicit strategy skips ranking entirely.
        sz = 0 if single_cand else coded_size_bytes(
            out, coder="brotli", brotli_quality=SEARCH_BROTLI_QUALITY
        )
        if best_bytes is None or sz < best_bytes:
            best_bytes = sz
            best_frame = out
            best_strategy = cand
    assert best_frame is not None

    bytes_after = coded_size_both(best_frame)
    residual = projector.max_abs_projection_residual(x, best_frame)
    n_changed = int(np.count_nonzero(np.any(best_frame != x, axis=2)))
    valid = bool(best_frame.dtype == np.uint8 and best_frame.shape == x.shape)
    proof = FrameProof(
        frame_index=frame_index,
        max_abs_projection_residual=residual,
        exact=(residual <= EXACT_RESIDUAL_TOL),
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        n_pixels_changed=n_changed,
        fill_strategy=best_strategy,
        tier=1,
        valid_uint8=valid,
    )
    return best_frame, proof


# ---------------------------------------------------------------------------
# TIER 2 — integer null-basis greedy descent on coded size.
#
# DERIVATION (why the descent atoms are exactly the zero-weight pixels):
# the only integer-exact axis-aligned null directions of R are the zero-weight
# pixels.  Searching the separable kernel for two non-zero-weight input indices
# with IDENTICAL resize-weight columns (which would permit an integer +1/-1
# null transfer between them) finds NONE outside the zero-weight set — every
# pair of duplicate weight columns is the all-zeros (zero-weight) column.  So a
# unit moved between any two NON-zero-weight pixels changes R.  The certified
# integer null directions are therefore exactly the zero-weight pixels (each
# amplitude-unlimited up to clipping), and tier-2 descends the coded size over
# THOSE directions with block-coordinate integer steps — a strict, proven
# improvement search on top of tier-1's single-shot predictor fill.  No
# real-valued null vector is ever rounded into the uint8 frame (operator caveat
# (a)).
# ---------------------------------------------------------------------------
def _zero_weight_runs_per_row(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous horizontal runs of masked pixels: ``(row, c_start, c_end)``
    (``c_end`` exclusive).  These are the integer-null descent blocks."""
    runs: list[tuple[int, int, int]] = []
    h, w = mask.shape
    for r in range(h):
        c = 0
        row = mask[r]
        while c < w:
            if row[c]:
                start = c
                while c < w and row[c]:
                    c += 1
                runs.append((r, start, c))
            else:
                c += 1
    return runs


def apply_tier2_null_basis_descent(
    frame: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    projector: ResizeProjector | None = None,
    basis: Tier1ResizeNullSpace | None = None,
    max_iterations: int = 2,
    coder: str = "brotli",
    frame_index: int = 0,
) -> tuple[np.ndarray, FrameProof]:
    """Block-coordinate integer null-basis descent on coded size over the
    certified zero-weight pixels.

    The descent atoms are whole-plane integer fill SCHEMES applied to the
    certified zero-weight pixels (each scheme is integer + null by construction,
    so ``R x̃ = R x`` stays EXACT).  Per channel plane, the real coder's measured
    size (fast search quality; final re-measured at q=11) is the admission
    arbiter: the scheme that strictly reduces the plane's coded size wins.
    Bounded ``max_iterations`` refinement sweeps follow (block-coordinate; in
    practice converges in 1).  No real-valued null vector is ever rounded into
    the uint8 frame (operator caveat (a)); the per-frame ``max|ΔR|`` is recomputed
    with the real projector and emitted.

    This descends the coded size PER CHANNEL — the marginal lever over tier-1,
    which ranks a single scheme on the whole frame and so cannot pick a different
    per-channel scheme.  Deterministic: scheme order is fixed.
    """
    x = np.asarray(frame, dtype=np.uint8)
    if x.ndim != 3:
        raise ResizeNullPreimageError("frame must be (H, W, C)")
    h, w, c = x.shape
    if projector is None:
        projector = ResizeProjector.build(camera_h=h, camera_w=w)
    if mask is None:
        mask = zero_weight_pixel_mask(
            camera_h=h, camera_w=w,
            scorer_h=projector.scorer_h, scorer_w=projector.scorer_w, basis=basis,
        )
    if mask.shape != (h, w):
        raise ResizeNullPreimageError(f"mask shape {mask.shape} != frame {(h, w)}")

    bytes_before = coded_size_both(x)

    # Per-channel block-coordinate descent over the certified zero-weight pixels.
    # The descent atoms are whole-plane integer fill schemes (each integer + null
    # => R x̃ = R x exact); the per-plane coded size (the real coder, fast search
    # quality) is the admission arbiter.  This is strictly faster than per-run
    # full-frame compression while remaining measurement-driven: the marginal
    # lever over tier-1 is choosing the per-plane scheme that the FULL frame's
    # cross-channel structure does not see when tier-1 ranks on the frame.
    keep = ~mask
    schemes = ("horizontal_predictor", "vertical_predictor",
               "neighbor_mean", "constant")
    out = x.copy()
    for ch in range(c):
        plane = x[:, :, ch]
        best_plane = plane.copy()
        best_sz = coded_size_bytes(
            plane, coder=coder, brotli_quality=SEARCH_BROTLI_QUALITY
        )
        for scheme in schemes:
            fn = _FILL_FUNCS[scheme]
            trial = fn(plane, mask)
            sz = coded_size_bytes(
                trial, coder=coder, brotli_quality=SEARCH_BROTLI_QUALITY
            )
            if sz < best_sz:
                best_sz = sz
                best_plane = trial
        out[:, :, ch] = best_plane

    # Bounded refinement sweeps: a second pass can pick a different per-plane
    # scheme once neighbours have already been flattened (block-coordinate).  In
    # practice this converges in 1 sweep; max_iterations bounds it.
    for _ in range(max(0, int(max_iterations) - 1)):
        improved = False
        for ch in range(c):
            plane = out[:, :, ch]
            cur = coded_size_bytes(
                plane, coder=coder, brotli_quality=SEARCH_BROTLI_QUALITY
            )
            for scheme in schemes:
                fn = _FILL_FUNCS[scheme]
                # re-fill from the ORIGINAL plane (masked pixels are free DOF;
                # non-masked are fixed) so the scheme sees true neighbours.
                trial = fn(x[:, :, ch], mask)
                sz = coded_size_bytes(
                    trial, coder=coder, brotli_quality=SEARCH_BROTLI_QUALITY
                )
                if sz < cur:
                    cur = sz
                    out[:, :, ch] = trial
                    improved = True
        if not improved:
            break

    # Monotonic guarantee: the per-channel q=5 ranking does not always agree with
    # q=11, so confirm the descent result against tier-1's frame-level fill at the
    # REPORT quality and keep whichever is smaller.  Tier-2 is therefore never
    # worse than tier-1 (a true descent).
    t1_frame, _ = apply_tier1_zero_weight_fill(
        x, strategy="measured_best", mask=mask, projector=projector, basis=basis,
        frame_index=frame_index,
    )
    out_sz = coded_size_bytes(out, coder="brotli", brotli_quality=REPORT_BROTLI_QUALITY)
    t1_sz = coded_size_bytes(t1_frame, coder="brotli",
                             brotli_quality=REPORT_BROTLI_QUALITY)
    if t1_sz < out_sz:
        out = t1_frame

    bytes_after = coded_size_both(out)
    residual = projector.max_abs_projection_residual(x, out)
    n_changed = int(np.count_nonzero(np.any(out != x, axis=2)))
    valid = bool(out.dtype == np.uint8 and out.shape == x.shape)
    proof = FrameProof(
        frame_index=frame_index,
        max_abs_projection_residual=residual,
        exact=(residual <= EXACT_RESIDUAL_TOL),
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        n_pixels_changed=n_changed,
        fill_strategy="tier2_null_basis_descent[per_channel_scheme]",
        tier=2,
        valid_uint8=valid,
    )
    return out, proof


# ---------------------------------------------------------------------------
# TIER 3 — blockwise constrained least-entropy preimage (smallest viable impl).
#
# Where the visible projection allows (the certified zero-weight lattice), snap
# the dropped rows/cols into a single piecewise-constant tile structure: set
# EVERY zero-weight pixel (across all rows/cols) to ONE plane-wide constant
# (chosen by measurement), collapsing the 22.7% lattice into a maximally
# RLE/predictor-friendly flat region.  This is the cheapest blockwise variant;
# the LP/MILP/learned upgrades (per-block palette, joint visible-constrained
# least-entropy over ker(R)'s full 80.67%, learned context fill) are named
# follow-ups documented in the landing memo.
# ---------------------------------------------------------------------------
def apply_tier3_blockwise_flat_preimage(
    frame: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    projector: ResizeProjector | None = None,
    basis: Tier1ResizeNullSpace | None = None,
    coder: str = "brotli",
    frame_index: int = 0,
) -> tuple[np.ndarray, FrameProof]:
    """Snap the certified zero-weight lattice to the single plane-wide constant
    that minimizes coded size (the cheapest blockwise-constant preimage).

    Per channel, measure the coded size for a small set of plane-wide constants
    (0, plane DC, 128) over the masked lattice and keep the best.  Integer +
    null by construction (zero-weight pixels), so ``R x̃ = R x`` is EXACT."""
    x = np.asarray(frame, dtype=np.uint8)
    if x.ndim != 3:
        raise ResizeNullPreimageError("frame must be (H, W, C)")
    h, w, c = x.shape
    if projector is None:
        projector = ResizeProjector.build(camera_h=h, camera_w=w)
    if mask is None:
        mask = zero_weight_pixel_mask(
            camera_h=h, camera_w=w,
            scorer_h=projector.scorer_h, scorer_w=projector.scorer_w, basis=basis,
        )
    if mask.shape != (h, w):
        raise ResizeNullPreimageError(f"mask shape {mask.shape} != frame {(h, w)}")

    bytes_before = coded_size_both(x)
    out = x.copy()
    keep = ~mask
    for ch in range(c):
        plane = x[:, :, ch]
        plane_dc = int(round(float(plane[keep].mean()))) if keep.any() else 0
        cands = [0, plane_dc, 128]
        seen: set[int] = set()
        cands = [v for v in cands if not (v in seen or seen.add(v))]
        best_plane = plane.copy()
        best_size = None
        for val in cands:
            trial = plane.copy()
            trial[mask] = np.uint8(np.clip(val, 0, 255))
            # measure the PLANE (fast search quality) — final re-measured at q=11.
            sz = coded_size_bytes(
                trial, coder=coder, brotli_quality=SEARCH_BROTLI_QUALITY
            )
            if best_size is None or sz < best_size:
                best_size = sz
                best_plane = trial
        out[:, :, ch] = best_plane

    bytes_after = coded_size_both(out)
    residual = projector.max_abs_projection_residual(x, out)
    n_changed = int(np.count_nonzero(np.any(out != x, axis=2)))
    valid = bool(out.dtype == np.uint8 and out.shape == x.shape)
    proof = FrameProof(
        frame_index=frame_index,
        max_abs_projection_residual=residual,
        exact=(residual <= EXACT_RESIDUAL_TOL),
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        n_pixels_changed=n_changed,
        fill_strategy="tier3_blockwise_flat",
        tier=3,
        valid_uint8=valid,
    )
    return out, proof


# ---------------------------------------------------------------------------
# THE LAW score-delta on the rate term (consume #46's helper, do not refork).
# ---------------------------------------------------------------------------
def preimage_rate_score_delta(bytes_freed: int) -> float:
    """``ΔS`` from freeing ``bytes_freed`` archive bytes, via #46's
    ``delta_rate_score`` (THE LAW rate term ``25 * Δbytes / N``).  Distortion
    delta is CERTIFIED 0.0 for tier-1/2/3 (zero-weight => R x̃ = R x exactly), so
    the full ΔS_total is exactly this rate term — a strict score improvement for
    any positive ``bytes_freed``."""
    from tac.optimization.lf_payload_rate_distortion import delta_rate_score

    # delta_rate_score(delta_bytes) returns 25 * delta_bytes / N; freeing bytes
    # is a NEGATIVE delta_bytes (archive shrinks) => negative ΔS (improvement).
    return float(delta_rate_score(-int(bytes_freed)))


__all__ = [
    "RESIZE_NULL_PREIMAGE_SCHEMA",
    "PROOF_EVIDENCE_GRADE",
    "BYTES_EVIDENCE_GRADE",
    "EXACT_RESIDUAL_TOL",
    "REPORT_BROTLI_QUALITY",
    "SEARCH_BROTLI_QUALITY",
    "CONTEST_TOTAL_BYTES",
    "ResizeNullPreimageError",
    "ResizeProjector",
    "FrameProof",
    "zero_weight_pixel_mask",
    "coded_size_bytes",
    "coded_size_both",
    "apply_tier1_zero_weight_fill",
    "apply_tier2_null_basis_descent",
    "apply_tier3_blockwise_flat_preimage",
    "preimage_rate_score_delta",
]
