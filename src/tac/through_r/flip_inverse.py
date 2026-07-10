# SPDX-License-Identifier: MIT
"""flip_inverse — the RESIZE-EXPLOIT flip solver (#391).

The scored argmax lives at the SegNet grid ``(SEG_H=384, SEG_W=512)`` AFTER a KNOWN,
deterministic composite that :mod:`resolution_chain` pins from the upstream source:

    render-grid RGB (h,w,3) float[0,255]
      --[R.up]--   bicubic  UP   -> CAMERA (874,1164) -> round -> clamp[0,255] -> uint8
      --[R.down]-- bilinear DOWN -> SEG (384,512)   (SegNet.preprocess_input, upstream/modules.py:109)
      --> SegNet -> argmax -> d_seg = mean_pixels(argmax != L*)

Both resizes are LINEAR with computable kernels; the uint8 ``round`` is a bounded
rounding non-linearity (a per-camera-pixel dead-zone). This module EXPLOITS that
structure:

  1. EXACT COMPOSITE KERNEL + ADJOINT (:class:`ResizeComposite`) — the 1D resize
     operators are EXTRACTED by impulse-probing the REAL ``torch.nn.functional.interpolate``
     (NEVER re-derived from the cubic-convolution formula: the operating manual's
     "validate against the primary artifact" applied to a kernel). ``down`` is
     bilinear/align_corners=False/antialias=False (matches upstream EXACTLY). The
     adjoint is the transpose (exact for a linear op) — it maps a desired change at a
     SEG pixel back to the CAMERA pixels that control it (the bilinear-down footprint).

  2. uint8 DEAD-ZONE (:func:`uint8_dead_zone`) — the FREE (sub-LSB, inert: no
     downstream change) vs COSTED (survives the round) split, per camera pixel.

  3. FLIP LEDGER (:func:`build_flip_ledger`) — on a candidate's realized argmax vs the
     cached ``lstars`` (n600): per flip {position, class-pair, margin deficit,
     annulus-distance (#333), temporal persistence (flicker vs stable — the "when")}.

  4. PER-FLIP INVERSE SOLVE (:func:`solve_flip_costs`) — via the adjoint + the
     candidate's own signed margin field: the minimal camera-grid perturbation (L2 / L∞,
     uint8-aware) whose SegNet-input footprint supplies the logit swing to flip a pixel
     back. Generalises #149 (camera-res sub-pixel PLACEMENT, COMPLETED) from boundary
     placement to arbitrary targeted flips.

  5. WATERFILL (:func:`waterfill_frontier`) — each fixed flip is worth the SAME ΔS
     (``100/(N·SEG_H·SEG_W)``, DERIVED from :mod:`tac.contest_score`), so "what needs to
     flip for optimal" is: fix the CHEAPEST flips first. The frontier = cumulative
     Δd_seg vs cumulative perturbation norm; the P10 map = free / cheap / costed /
     unreachable buckets.

  6. MEASURE (:func:`verify_targeted_fix`) — apply the solved perturbation for the top-K
     cheapest flips, re-measure through the REAL harness (:func:`measure_through_r`),
     confirm the predicted flips REALIZE (prediction-vs-realized is THE honesty number;
     the phi-argmax over-counting precedent makes realized-argmax verification mandatory).

AUTHORITY: the realized argmax is the FROZEN CPU-torch SegNet (never MPS/MLX — CLAUDE.md).
Every number here is ``[macOS-CPU advisory . through-R . NON-PROMOTABLE]`` until a
byte-closed exact eval moves the pointer. This is a $0 CHARACTERISATION instrument (the
P10 flip-fix map), NOT a score claim — the pointer stays 0.19110.

#149 CAVEAT (load-bearing, quoted): camera-res sub-pixel placement collapses the
boundary-BAND flip 12× (0.176->0.022) but the residual RELOCATES to REAL interior-texture
(the +0.00562 "wall", 99% of the residual) which is NOT a resize artifact and NOT fixable
by a resize perturbation. So the FREE/cheap fraction this instrument measures on a
resize-degradation candidate is an UPPER BOUND on the fraction fixable on a real witness
(whose residual is interior-texture-dominated). Every ledger states its candidate class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.through_r.resolution_chain import (
    CAMERA_H,
    CAMERA_W,
    RGB_CHANNELS,
    SEG_H,
    SEG_W,
    render_grid_to_camera_uint8,
)

# d_seg is a per-pixel mean over N pairs; fixing ONE flipped pixel lowers d_seg by
# 1/(N·SEG_H·SEG_W). ΔS = 100·Δd_seg (upstream/evaluate.py:92 seg term). DERIVED, exact.
SEG_PIXELS = SEG_H * SEG_W
FLIP_INVERSE_LABEL = "[macOS-CPU advisory . RESIZE-EXPLOIT through-R . NON-PROMOTABLE]"


class FlipInverseError(ValueError):
    """Raised on a mis-shaped / non-authority / toy input to the flip solver."""


# ======================================================================================
# 1. EXACT COMPOSITE RESIZE KERNEL + ADJOINT  (extracted from the REAL torch ops)
# ======================================================================================
def resize_matrix_1d(
    n_in: int,
    n_out: int,
    mode: str,
    *,
    align_corners: bool = False,
    antialias: bool = False,
    dtype: Any = np.float64,
) -> np.ndarray:
    """The EXACT ``(n_out, n_in)`` linear operator ``torch.interpolate`` applies along one axis.

    Extracted by impulse-probing the REAL ``torch.nn.functional.interpolate`` (a linear op
    for bilinear/bicubic), NOT re-derived from the cubic-convolution weight formula. Each
    input column ``i`` is a unit impulse; the output row is the weight vector into the
    ``n_out`` output samples. Returns ``M`` with ``M[j, i] = weight(in i -> out j)`` so that
    ``y = M @ x`` reproduces ``interpolate(x)`` bit-for-bit (same dtype).

    ``mode`` in {'bilinear','bicubic'}. ``align_corners`` / ``antialias`` are passed to torch
    UNCHANGED so the operator matches the exact upstream call (down = bilinear, align_corners
    False, antialias False — upstream/modules.py:109).
    """

    import torch

    if int(n_in) <= 0 or int(n_out) <= 0:
        raise FlipInverseError(f"resize_matrix_1d needs positive sizes; got {n_in},{n_out}")
    if mode not in ("bilinear", "bicubic"):
        raise FlipInverseError(f"mode must be bilinear|bicubic; got {mode!r}")
    # (n_in, 1, 1, n_in): sample i is the i-th standard basis vector along width.
    eye = torch.eye(int(n_in), dtype=torch.float64).reshape(int(n_in), 1, 1, int(n_in))
    with torch.inference_mode():
        out = torch.nn.functional.interpolate(
            eye,
            size=(1, int(n_out)),
            mode=mode,
            align_corners=bool(align_corners),
            antialias=bool(antialias),
        )
    # out[i,0,0,j] = weight(in i -> out j)  =>  M[j,i]
    m = out[:, 0, 0, :].transpose(0, 1).contiguous().numpy().astype(dtype)
    return np.ascontiguousarray(m)


@dataclass
class ResizeComposite:
    """The pinned R linear operators + the uint8 round, as EXACT matrices with adjoints.

    ``down_col`` ``(SEG_H, CAMERA_H)`` and ``down_row`` ``(SEG_W, CAMERA_W)`` are the
    separable bilinear-DOWN operator (camera -> SEG). ``up_col`` / ``up_row`` (optional) are
    the bicubic-UP operator (render-grid -> camera) present only when a render grid is used.
    """

    down_col: np.ndarray
    down_row: np.ndarray
    up_col: np.ndarray | None = None
    up_row: np.ndarray | None = None
    render_hw: tuple[int, int] | None = None

    # ---- constructors ----
    @classmethod
    def build(cls, render_hw: tuple[int, int] | None = None) -> ResizeComposite:
        """Build the composite. ``render_hw=None`` => camera-native (the #149 regime: only
        the bilinear-down is in the scored path). Otherwise also build the bicubic-up."""

        down_col = resize_matrix_1d(CAMERA_H, SEG_H, "bilinear", align_corners=False)
        down_row = resize_matrix_1d(CAMERA_W, SEG_W, "bilinear", align_corners=False)
        up_col = up_row = None
        if render_hw is not None:
            h, w = int(render_hw[0]), int(render_hw[1])
            up_col = resize_matrix_1d(h, CAMERA_H, "bicubic", align_corners=False)
            up_row = resize_matrix_1d(w, CAMERA_W, "bicubic", align_corners=False)
        return cls(
            down_col=down_col,
            down_row=down_row,
            up_col=up_col,
            up_row=up_row,
            render_hw=(int(render_hw[0]), int(render_hw[1])) if render_hw is not None else None,
        )

    # ---- forward linear maps (per channel) ----
    @staticmethod
    def _sep_apply(col: np.ndarray, row: np.ndarray, img_hwc: np.ndarray) -> np.ndarray:
        """Apply separable ``col @ X @ row.T`` to each channel of ``(H,W,C)`` -> ``(H',W',C)``."""

        a = np.asarray(img_hwc, dtype=np.float64)
        if a.ndim != 3:
            raise FlipInverseError(f"expected (H,W,C); got {a.shape}")
        c = a.shape[2]
        out = np.empty((col.shape[0], row.shape[0], c), dtype=np.float64)
        # macOS Accelerate sets spurious FP-exception flags on large SIMD matmul even when
        # the result is correct (verified bit-exact vs torch in validate_against_torch);
        # suppress locally and finite-guard so a REAL nan/inf still fails closed.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for k in range(c):
                out[..., k] = col @ a[..., k] @ row.T
        if not np.all(np.isfinite(out)):
            raise FlipInverseError("non-finite value in separable resize apply (real overflow)")
        return out

    def down_apply(self, camera_hwc: np.ndarray) -> np.ndarray:
        """Bilinear-DOWN camera ``(CAMERA_H,CAMERA_W,C)`` float -> SEG ``(SEG_H,SEG_W,C)`` float.

        EXACT match to ``F.interpolate(camera, size=SEG_HW, bilinear)`` per channel (the R
        second half / ``SegNet.preprocess_input``)."""

        a = np.asarray(camera_hwc)
        if a.shape[:2] != (CAMERA_H, CAMERA_W):
            raise FlipInverseError(f"down_apply needs (CAMERA_H,CAMERA_W,C); got {a.shape}")
        return self._sep_apply(self.down_col, self.down_row, a)

    def down_adjoint(self, seg_grad_hwc: np.ndarray) -> np.ndarray:
        """Adjoint of ``down_apply``: SEG-grid quantity -> CAMERA grid via ``downᵀ``.

        For a DESIRED change ``g`` at the SEG grid, ``down_adjoint(g)`` is the camera-grid
        footprint that produces it (the bilinear-down transpose). This is the LEVER MAP: the
        nonzero camera pixels for a SEG pixel are exactly this footprint."""

        a = np.asarray(seg_grad_hwc)
        if a.shape[:2] != (SEG_H, SEG_W):
            raise FlipInverseError(f"down_adjoint needs (SEG_H,SEG_W,C); got {a.shape}")
        return self._sep_apply(self.down_col.T, self.down_row.T, a)

    def up_apply(self, render_hwc: np.ndarray) -> np.ndarray:
        """Bicubic-UP render ``(h,w,C)`` float -> camera ``(CAMERA_H,CAMERA_W,C)`` float (pre-round)."""

        if self.up_col is None or self.up_row is None:
            raise FlipInverseError("up_apply requires a render_hw composite (build(render_hw=...))")
        a = np.asarray(render_hwc)
        if self.render_hw is not None and a.shape[:2] != self.render_hw:
            raise FlipInverseError(f"up_apply needs render_hw {self.render_hw}; got {a.shape}")
        return self._sep_apply(self.up_col, self.up_row, a)

    def render_to_camera_uint8(self, render_hwc: np.ndarray) -> np.ndarray:
        """Full R.up: ``up_apply`` then round+clamp[0,255] -> uint8 camera (matches
        :func:`resolution_chain.render_grid_to_camera_uint8`)."""

        cam = self.up_apply(render_hwc)
        return np.clip(np.round(cam), 0.0, 255.0).astype(np.uint8)

    def render_to_seg_float(self, render_hwc: np.ndarray) -> np.ndarray:
        """Full composite render -> SEG float (through the uint8 round at camera res)."""

        cam_u8 = self.render_to_camera_uint8(render_hwc).astype(np.float64)
        return self.down_apply(cam_u8)

    # ---- footprint (which camera pixels control a SEG pixel) ----
    def seg_pixel_footprint(
        self, py: int, px: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Camera pixels controlling SEG pixel ``(py,px)`` under the bilinear down.

        Returns ``(cy_idx, cx_idx, weights)`` where ``weights[a,b]`` is the bilinear-down
        weight from camera ``(cy_idx[a], cx_idx[b])`` to SEG ``(py,px)`` — the exact outer
        product ``down_col[py, cy] ⊗ down_row[px, cx]``. Antialias=False => ≤2 taps/axis."""

        wr = self.down_col[int(py)]
        wc = self.down_row[int(px)]
        cy = np.nonzero(wr)[0]
        cx = np.nonzero(wc)[0]
        weights = np.outer(wr[cy], wc[cx])
        return cy, cx, weights

    # ---- validation vs the REAL torch ops (the operating-manual re-derive discipline) ----
    def validate_against_torch(self, *, seed: int = 0, atol: float = 1e-9) -> dict[str, float]:
        """Prove the extracted matrices reproduce ``F.interpolate`` bit-for-bit (float64).

        Returns max-abs errors. Raises if any exceeds ``atol`` (fail-closed)."""

        import torch

        rng = np.random.default_rng(seed)
        cam = rng.uniform(0, 255, size=(CAMERA_H, CAMERA_W, RGB_CHANNELS))
        seg_mine = self.down_apply(cam)
        with torch.inference_mode():
            t = torch.from_numpy(cam).permute(2, 0, 1)[None].double()
            seg_torch = (
                torch.nn.functional.interpolate(
                    t, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False
                )[0]
                .permute(1, 2, 0)
                .numpy()
            )
        err_down = float(np.max(np.abs(seg_mine - seg_torch)))
        out = {"down_maxabs": err_down}
        if self.up_col is not None:
            rh = self.render_hw
            assert rh is not None
            x = rng.uniform(0, 255, size=(rh[0], rh[1], RGB_CHANNELS))
            cam_mine = self.up_apply(x)
            with torch.inference_mode():
                tx = torch.from_numpy(x).permute(2, 0, 1)[None].double()
                cam_torch = (
                    torch.nn.functional.interpolate(
                        tx, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
                    )[0]
                    .permute(1, 2, 0)
                    .numpy()
                )
            out["up_maxabs"] = float(np.max(np.abs(cam_mine - cam_torch)))
        worst = max(out.values())
        if worst > atol:
            raise FlipInverseError(
                f"resize matrix drift vs torch: {out} exceeds atol {atol} — the extracted "
                "operator does not match the real interpolate (transposition/convention bug)."
            )
        return out

    def row_sums(self) -> dict[str, float]:
        """Max-abs deviation of each operator's row sums from 1 (partition of unity).

        Bilinear + Keys-bicubic (a=-0.75) are BOTH normalised (weights sum to 1) in the
        interior; near the border torch replicates edge taps so rows still sum to 1. A
        deviation flags a convention error."""

        out = {
            "down_col": float(np.max(np.abs(self.down_col.sum(1) - 1.0))),
            "down_row": float(np.max(np.abs(self.down_row.sum(1) - 1.0))),
        }
        if self.up_col is not None and self.up_row is not None:
            out["up_col"] = float(np.max(np.abs(self.up_col.sum(1) - 1.0)))
            out["up_row"] = float(np.max(np.abs(self.up_row.sum(1) - 1.0)))
        return out


# ======================================================================================
# 2. uint8 DEAD-ZONE  (free vs costed split)
# ======================================================================================
def uint8_dead_zone(camera_float: np.ndarray) -> np.ndarray:
    """Per-pixel headroom to the nearest uint8 rounding boundary: ``0.5 - |frac|`` in [0,0.5].

    A perturbation ``δ`` at a camera pixel is FREE-but-INERT (no downstream change: it stays
    inside the dead-zone) iff ``|δ| <= headroom`` toward the nearer boundary; it SURVIVES the
    round (changes the uint8 => can move the score) iff it CROSSES ``headroom``. For an
    already-uint8 field ``frac==0`` so ``headroom==0.5`` everywhere and the minimal surviving
    step is one LSB (integer). Input may be float (render->camera, pre-round) or uint8."""

    a = np.asarray(camera_float, dtype=np.float64)
    frac = a - np.round(a)
    return (0.5 - np.abs(frac)).astype(np.float64)


def free_flip_fraction(camera_float: np.ndarray, eps: float = 0.05) -> float:
    """Fraction of camera samples within ``eps`` of a rounding boundary (headroom < eps).

    These are the CHEAPEST samples to flip: a sub-``eps`` nudge crosses the uint8 boundary.
    A LAW-ready statistic (the free-flip fraction of the resize field)."""

    hz = uint8_dead_zone(camera_float)
    return float(np.mean(hz < float(eps)))


# ======================================================================================
# 3. FLIP LEDGER  (where / when — n600 real SegNet)
# ======================================================================================
def _boundary_distance(lstar_hw: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) of each pixel to the nearest GT class boundary (#333 annulus).

    Boundary = a pixel whose 4-neighbour differs in class. Interior pixels get their distance
    to the nearest such boundary via the exact EDT."""

    from scipy.ndimage import distance_transform_edt

    lab = np.asarray(lstar_hw)
    b = np.zeros(lab.shape, dtype=bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    if not b.any():
        return np.full(lab.shape, float(max(lab.shape)), dtype=np.float64)
    return distance_transform_edt(~b).astype(np.float64)


def candidate_forward_fields(
    segnet: Any,
    camera_frames: list[np.ndarray],
    lstars: np.ndarray,
    *,
    verdict_batch: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """One CPU-torch SegNet pass -> ``(realized (N,H,W) int64, deficit (N,H,W) float32)``.

    ``deficit[p] = z_top1[p] - z_cgt[p] >= 0`` = the logit swing the GT class needs to WIN at
    ``p`` (0 where the GT class already leads; > 0 at a flip). Mirrors the harness forward
    op-for-op (``(b,1,H,W,3)`` -> permute -> ``preprocess_input`` -> logits), CHUNKED per the
    n600-verdict-OOM law. ``argmax`` is authority; ``deficit`` is a byproduct (no extra pass)."""

    import torch

    n = len(camera_frames)
    lstars = np.asarray(lstars)
    step = n if int(verdict_batch) <= 0 else int(verdict_batch)
    realized = np.empty((n, SEG_H, SEG_W), dtype=np.int64)
    deficit = np.empty((n, SEG_H, SEG_W), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, step):
            block = camera_frames[start : start + step]
            arr = np.stack([np.asarray(f)[None] for f in block], axis=0)  # (b,1,H,W,3)
            xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()
            seg_in = segnet.preprocess_input(xp)
            logits = segnet(seg_in)  # (b,5,H,W)
            am = logits.argmax(dim=1)  # (b,H,W)
            top1 = logits.max(dim=1).values  # (b,H,W)
            b = am.shape[0]
            gt = torch.from_numpy(lstars[start : start + b]).long()  # (b,H,W)
            z_gt = torch.gather(logits, 1, gt[:, None]).squeeze(1)  # (b,H,W)
            realized[start : start + b] = am.cpu().numpy().astype(np.int64)
            deficit[start : start + b] = (top1 - z_gt).clamp_min(0.0).cpu().numpy()
    return realized, deficit


@dataclass
class FlipLedger:
    """The n600 residual-flip ledger (where + when each flip lives)."""

    candidate_class: str
    n_pairs: int
    total_flips: int
    total_pixels: int
    d_seg: float
    # per-flip parallel arrays (length == total_flips):
    pair_idx: np.ndarray
    y: np.ndarray
    x: np.ndarray
    c_wrong: np.ndarray
    c_gt: np.ndarray
    deficit: np.ndarray
    annulus_dist: np.ndarray
    persistence: np.ndarray  # # of the pair's temporal neighbours (±1) also flipped at (y,x)
    is_n600: bool
    subset_reason: str | None = None
    class_pair_counts: dict[str, int] = field(default_factory=dict)
    label: str = FLIP_INVERSE_LABEL
    extra: dict[str, float] = field(default_factory=dict)


def build_flip_ledger(
    camera_frames: list[np.ndarray],
    *,
    lstars: np.ndarray,
    candidate_class: str,
    segnet: Any | None = None,
    verdict_batch: int = 32,
    allow_subset_reason: str | None = None,
) -> FlipLedger:
    """Build the residual-flip ledger for ``camera_frames`` (camera-uint8) vs ``lstars``.

    n600 discipline: refuses ``N != 600`` unless ``allow_subset_reason`` is a non-empty
    string (labelled non-authority subset). ``candidate_class`` names WHAT the frames are
    (e.g. 'resize_degraded_gt(192x256)') so the ledger carries its provenance (the #149
    caveat: a resize-degradation upper-bounds the real-witness free fraction)."""

    frames = [np.asarray(f) for f in camera_frames]
    n = len(frames)
    if n == 0:
        raise FlipInverseError("no candidate frames")
    subset_reason: str | None = None
    if n != 600:
        reason = (allow_subset_reason or "").strip()
        if not reason:
            raise FlipInverseError(
                f"n600 discipline: ledger needs N=600; got N={n}. Pass allow_subset_reason="
                "'...' ONLY for an explicitly non-authority subset."
            )
        subset_reason = reason
    lstars = np.asarray(lstars)[:n]
    if lstars.shape[1:] != (SEG_H, SEG_W):
        raise FlipInverseError(f"lstars grid {lstars.shape[1:]} != SEG_HW; wrong compare grid")

    if segnet is None:
        from tac.through_r.harness import load_frozen_segnet

        segnet = load_frozen_segnet("cpu-torch")
    realized, deficit = candidate_forward_fields(
        segnet, frames, lstars, verdict_batch=verdict_batch
    )
    flip_mask = realized != lstars  # (N,H,W) bool

    # temporal persistence: is (y,x) also flipped in the neighbouring pairs (±1)?
    persistence_stack = np.zeros_like(flip_mask, dtype=np.int16)
    if n >= 2:
        persistence_stack[:-1] += flip_mask[1:]
        persistence_stack[1:] += flip_mask[:-1]

    ann = np.stack([_boundary_distance(lstars[i]) for i in range(n)], axis=0)

    idx = np.nonzero(flip_mask)
    pair_idx, yy, xx = idx[0], idx[1], idx[2]
    c_wrong = realized[idx].astype(np.int16)
    c_gt = lstars[idx].astype(np.int16)
    defi = deficit[idx].astype(np.float32)
    annd = ann[idx].astype(np.float32)
    pers = persistence_stack[idx].astype(np.int16)

    total_flips = int(pair_idx.size)
    total_pixels = int(n * SEG_PIXELS)
    d_seg = float(total_flips / total_pixels)
    # class-pair histogram (top entries)
    cp = {}
    if total_flips:
        keys = (c_gt.astype(np.int64) * 10 + c_wrong.astype(np.int64))
        u, counts = np.unique(keys, return_counts=True)
        for k, ct in zip(u.tolist(), counts.tolist(), strict=True):
            cp[f"gt{k // 10}->wrong{k % 10}"] = int(ct)

    return FlipLedger(
        candidate_class=candidate_class,
        n_pairs=n,
        total_flips=total_flips,
        total_pixels=total_pixels,
        d_seg=d_seg,
        pair_idx=pair_idx.astype(np.int32),
        y=yy.astype(np.int16),
        x=xx.astype(np.int16),
        c_wrong=c_wrong,
        c_gt=c_gt,
        deficit=defi,
        annulus_dist=annd,
        persistence=pers,
        is_n600=n == 600,
        subset_reason=subset_reason,
        class_pair_counts=dict(sorted(cp.items(), key=lambda kv: -kv[1])[:12]),
        extra={
            "deficit_median": float(np.median(defi)) if total_flips else 0.0,
            "annulus_median": float(np.median(annd)) if total_flips else 0.0,
            "frac_in_annulus_le5px": (
                float(np.mean(annd <= 5.0)) if total_flips else 0.0
            ),
            "frac_flicker_persistence0": (
                float(np.mean(pers == 0)) if total_flips else 0.0
            ),
        },
    )


# ======================================================================================
# 4 + 5. PER-FLIP INVERSE SOLVE + WATERFILL FRONTIER
# ======================================================================================
def delta_s_per_flip(n_pairs: int) -> float:
    """ΔS for fixing ONE flipped pixel = ``100 / (n_pairs · SEG_H · SEG_W)`` (DERIVED, exact)."""

    return 100.0 / (int(n_pairs) * SEG_PIXELS)


@dataclass
class FlipCostFrontier:
    """The linearised flip-fix cost frontier (the P10 map: what needs to flip for optimal)."""

    n_pairs: int
    n_flips: int
    order: np.ndarray  # flip indices sorted by ascending cost
    cost: np.ndarray  # per-flip min-L2 camera perturbation (LSB units) to supply the deficit
    cum_perturb: np.ndarray  # cumulative L2 perturbation along ``order``
    cum_delta_dseg: np.ndarray  # cumulative Δd_seg = k/(N·SEG_PIXELS)
    cum_delta_s: np.ndarray  # 100·cum_delta_dseg
    n_free: int  # deficit == 0 (already at the flip boundary: a 1-LSB nudge fixes)
    n_cheap: int  # reachable within one uint8 LSB of camera signal
    n_costed: int  # needs multi-LSB real signal
    n_unreachable: int  # interior-texture (far from annulus): not fixable by resize perturb
    label: str = FLIP_INVERSE_LABEL
    extra: dict[str, float] = field(default_factory=dict)


def solve_flip_costs(
    ledger: FlipLedger,
    composite: ResizeComposite,
    *,
    annulus_unreachable_px: float = 8.0,
) -> FlipCostFrontier:
    """Linearised per-flip minimal camera perturbation + the waterfill frontier.

    THE RESIZE-EXPLOIT MODEL (DERIVED / [prediction]): a flip at SEG pixel ``p`` is fixed when
    the GT-class logit gains ``deficit_p``. The controlling camera pixels are the bilinear-down
    footprint (``seg_pixel_footprint``); a unit camera perturbation there delivers a logit swing
    ``s`` (the SegNet margin-saliency, ~constant across boundary pixels — CALIBRATED empirically
    by :func:`verify_targeted_fix`, here folded into a nominal ``s0=1``). The least-L2 camera
    perturbation supplying ``deficit_p`` is ``deficit_p / (s0·‖footprint‖₂)`` in LSB units.

    Because ``footprint`` weights are ~identical across pixels (interior bilinear), the RANKING
    reduces to ``deficit`` ascending — the exact structure #141 predicts. Each fixed flip is
    worth the SAME ΔS, so "what needs to flip for optimal" = the cheapest-first frontier.

    Buckets: FREE (deficit==0), CHEAP (fixable within one camera LSB, i.e. cost<=1), COSTED
    (needs multi-LSB signal), UNREACHABLE (annulus_dist > ``annulus_unreachable_px`` =>
    interior-texture, the #149 wall: a resize perturbation cannot fix it)."""

    n = ledger.total_flips
    if n == 0:
        empty = np.zeros(0)
        return FlipCostFrontier(
            n_pairs=ledger.n_pairs, n_flips=0, order=empty.astype(int), cost=empty,
            cum_perturb=empty, cum_delta_dseg=empty, cum_delta_s=empty,
            n_free=0, n_cheap=0, n_costed=0, n_unreachable=0,
        )
    # footprint L2 norm is ~constant; compute the true norm for a representative interior SEG
    # pixel (mid grid) and use it (the resize-exact scale). Border pixels differ negligibly.
    _, _, w = composite.seg_pixel_footprint(SEG_H // 2, SEG_W // 2)
    fp_l2 = float(np.sqrt(np.sum(w * w)))  # ‖footprint‖₂
    s0 = 1.0  # nominal saliency (calibrated by the real-SegNet verify; ranking is s0-invariant)
    cost = ledger.deficit.astype(np.float64) / (s0 * fp_l2)  # LSB units

    order = np.argsort(cost, kind="stable")
    cost_sorted = cost[order]
    cum_perturb = np.cumsum(cost_sorted)
    k = np.arange(1, n + 1, dtype=np.float64)
    cum_delta_dseg = k / ledger.total_pixels
    cum_delta_s = 100.0 * cum_delta_dseg

    unreachable = ledger.annulus_dist > float(annulus_unreachable_px)
    free = ledger.deficit == 0.0
    cheap = (~free) & (~unreachable) & (cost <= 1.0)
    costed = (~free) & (~unreachable) & (cost > 1.0)
    return FlipCostFrontier(
        n_pairs=ledger.n_pairs,
        n_flips=n,
        order=order.astype(np.int64),
        cost=cost_sorted,
        cum_perturb=cum_perturb,
        cum_delta_dseg=cum_delta_dseg,
        cum_delta_s=cum_delta_s,
        n_free=int(free.sum()),
        n_cheap=int(cheap.sum()),
        n_costed=int(costed.sum()),
        n_unreachable=int(unreachable.sum()),
        extra={
            "footprint_l2": fp_l2,
            "delta_s_per_flip": delta_s_per_flip(ledger.n_pairs),
            "cum_delta_s_at_free_plus_cheap": float(
                100.0 * (int(free.sum()) + int(cheap.sum())) / ledger.total_pixels
            ),
        },
    )


# ======================================================================================
# 6. MEASURE: targeted fix through the REAL harness (prediction-vs-realized)
# ======================================================================================
def build_targeted_perturbation(
    ledger: FlipLedger,
    frontier: FlipCostFrontier,
    composite: ResizeComposite,
    camera_frames: list[np.ndarray],
    gt_frames_camera: list[np.ndarray],
    *,
    top_k: int,
    step_lsb: float = 12.0,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Construct the camera-perturbed candidate for the top-K CHEAPEST flips.

    For each targeted flip, nudge its bilinear-down footprint camera pixels TOWARD the GT
    appearance (``gt - candidate``, the direction that at full strength yields ``lstars`` = 0
    flips), by ``step_lsb`` scaled by the (uint8-aware) footprint weight. Returns the perturbed
    frames (only affected pairs cloned), the affected pair indices, and the ``(K,)`` array of
    targeted flip indices (into the ledger). This is the RESIZE-EXPLOIT actuator: it injects
    signal at EXACTLY the camera pixels the adjoint says control each SEG flip."""

    k = int(min(top_k, frontier.n_flips))
    take = frontier.order[:k]
    affected = sorted({int(ledger.pair_idx[i]) for i in take})
    pos = {p: j for j, p in enumerate(affected)}
    pert = [np.array(camera_frames[p], dtype=np.float64, copy=True) for p in affected]
    for i in take.tolist():
        p = int(ledger.pair_idx[i])
        py, px = int(ledger.y[i]), int(ledger.x[i])
        cy, cx, w = composite.seg_pixel_footprint(py, px)
        cand = pert[pos[p]]
        gt = np.asarray(gt_frames_camera[p], dtype=np.float64)
        sub_gt = gt[np.ix_(cy, cx)]  # (a,b,3)
        sub_ca = cand[np.ix_(cy, cx)]
        direction = np.sign(sub_gt - sub_ca)  # toward GT appearance
        cand[np.ix_(cy, cx)] += float(step_lsb) * (w[..., None]) * direction
    pert_u8 = [np.clip(np.round(a), 0.0, 255.0).astype(np.uint8) for a in pert]
    return pert_u8, affected, take.astype(np.int64)


def verify_targeted_fix(
    ledger: FlipLedger,
    frontier: FlipCostFrontier,
    composite: ResizeComposite,
    camera_frames: list[np.ndarray],
    gt_frames_camera: list[np.ndarray],
    *,
    lstars: np.ndarray,
    top_k: int = 256,
    step_lsb: float = 12.0,
    segnet: Any | None = None,
    verdict_batch: int = 32,
) -> dict[str, Any]:
    """Apply the top-K targeted resize perturbation and RE-MEASURE realized argmax (real SegNet).

    Returns the honesty numbers: ``predicted`` (=K), ``realized_fixed`` (targeted flips whose
    argmax now equals ``c_gt``), ``prediction_vs_realized`` ratio, ``collateral_new_flips`` (new
    flips introduced in the affected pairs by cross-coupling), and the affected-subset d_seg
    before/after. Runs ONE forward on the AFFECTED PAIRS ONLY (labelled subset)."""

    pert_u8, affected, take = build_targeted_perturbation(
        ledger, frontier, composite, camera_frames, gt_frames_camera,
        top_k=top_k, step_lsb=step_lsb,
    )
    if segnet is None:
        from tac.through_r.harness import load_frozen_segnet

        segnet = load_frozen_segnet("cpu-torch")
    lst_aff = np.stack([np.asarray(lstars)[p] for p in affected], axis=0)
    realized_after, _ = candidate_forward_fields(
        segnet, pert_u8, lst_aff, verdict_batch=verdict_batch
    )
    pos = {p: j for j, p in enumerate(affected)}

    # targeted flips: did each now match c_gt?
    fixed = 0
    for i in take.tolist():
        p = int(ledger.pair_idx[i])
        py, px = int(ledger.y[i]), int(ledger.x[i])
        if int(realized_after[pos[p], py, px]) == int(ledger.c_gt[i]):
            fixed += 1
    # collateral: flips in affected pairs BEFORE vs AFTER (net new flips). Recompute the BEFORE
    # realized on the affected pairs for an apples-to-apples net count.
    realized_before, _ = candidate_forward_fields(
        segnet, [np.asarray(camera_frames[p]) for p in affected], lst_aff,
        verdict_batch=verdict_batch,
    )
    flips_before = int((realized_before != lst_aff).sum())
    flips_after = int((realized_after != lst_aff).sum())
    k = len(take)
    return {
        "predicted": k,
        "realized_fixed": int(fixed),
        "prediction_vs_realized": float(fixed / k) if k else 0.0,
        "affected_pairs": len(affected),
        "affected_flips_before": flips_before,
        "affected_flips_after": flips_after,
        "net_flips_removed": flips_before - flips_after,
        "collateral_new_flips": max(0, (flips_after - (flips_before - fixed))),
        "step_lsb": float(step_lsb),
        "affected_subset_dseg_before": flips_before / (len(affected) * SEG_PIXELS),
        "affected_subset_dseg_after": flips_after / (len(affected) * SEG_PIXELS),
        "label": FLIP_INVERSE_LABEL,
        "authority_note": "affected-pair SUBSET re-measure (real CPU-torch SegNet); NON-PROMOTABLE",
    }


# ======================================================================================
# candidate construction ($0, reproducible)
# ======================================================================================
def make_resize_degraded_candidate(
    gt_frame1_camera: np.ndarray,
    render_hw: tuple[int, int],
    composite: ResizeComposite | None = None,
) -> np.ndarray:
    """A reproducible resize-degradation candidate: GT camera frame -> bilinear DOWN to a
    render grid -> R.up (bicubic + round) back to camera-uint8.

    This produces REAL resize-induced flips vs ``lstars`` (the SegNet argmax of the pristine
    GT frame), exercising the exact composite the exploit targets. LABEL it a resize-degradation
    candidate: its flip-set is resize-loss-dominated, so the FREE/cheap fraction is an UPPER
    BOUND on the fraction fixable on a real witness (whose residual is interior-texture — #149)."""

    if composite is None or composite.render_hw != (int(render_hw[0]), int(render_hw[1])):
        composite = ResizeComposite.build(render_hw=render_hw)
    gt = np.asarray(gt_frame1_camera, dtype=np.float64)
    if gt.shape[:2] != (CAMERA_H, CAMERA_W):
        raise FlipInverseError(f"gt must be camera (CAMERA_H,CAMERA_W,3); got {gt.shape}")
    # down to render grid via torch bilinear (the witness render grid), then R.up.
    import torch

    with torch.inference_mode():
        t = torch.from_numpy(gt).permute(2, 0, 1)[None].double()
        small = (
            torch.nn.functional.interpolate(
                t, size=(int(render_hw[0]), int(render_hw[1])), mode="bilinear",
                align_corners=False,
            )[0]
            .permute(1, 2, 0)
            .numpy()
        )
    return render_grid_to_camera_uint8(small)


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "FLIP_INVERSE_LABEL",
    "SEG_H",
    "SEG_PIXELS",
    "SEG_W",
    "FlipCostFrontier",
    "FlipInverseError",
    "FlipLedger",
    "ResizeComposite",
    "build_flip_ledger",
    "build_targeted_perturbation",
    "candidate_forward_fields",
    "delta_s_per_flip",
    "free_flip_fraction",
    "make_resize_degraded_candidate",
    "resize_matrix_1d",
    "solve_flip_costs",
    "uint8_dead_zone",
    "verify_targeted_fix",
]
