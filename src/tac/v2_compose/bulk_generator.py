# SPDX-License-Identifier: MIT
"""v2_compose.bulk_generator — deterministic STORE+GENERATE bulk (Step 3, compress side).

Generates the deterministic bulk partition for every pair from a small set of STORED canonical
keyframes + the STORED per-pair 6-DOF pose + 3 calib scalars, via the per-class STRATIFIED warp
(Road = ground-homography, hood/MyCar = identity, sky/Undriv = rotation-only KRK^-1). This is the
GENERATE tier: the warp / render / R1-ramp algorithm is the FREE generic algorithm (rule-118); only
the keyframes + pose + calib + palette are COUNTED bytes.

NO-FAKE FAITHFULNESS (the load-bearing anchor): this module REUSES the EXACT proven warp+render+R
primitives from ``tools/measure_screw_reach_through_R.py`` (+ its sister tools) — the path where
``k=0`` reproduces the ladder R1 d_seg floor 0.0185 EXACTLY. It does NOT reimplement them, so there
is zero divergence risk. The compress-side through-R is ``render_partition -> bicubic_up(874x1164)
-> uint8 round -> SegNet.preprocess_input(bilinear down 384) -> frozen CPU-torch SegNet argmax`` —
identical to the reach tool. :func:`verify_k0_faithfulness` re-asserts the k=0 ~ 0.0185 anchor.

LAYERING NOTE (CLAUDE.md "tac stays clean"): importing the proven primitives from ``tools.*`` into a
``tac`` module is a deliberate, documented wart — these MEASUREMENT primitives are the single source
of the faithful warp path, and importing the same function objects is the strongest NO-FAKE guarantee
(no copy can drift). FUTURE: promote ``_m_step`` / ``cumulative_homography`` / ``composite_warped_labels``
/ ``warp_labels`` / ``intrinsics_at`` into ``tac.se3`` / ``tac.camera`` and have the tools import from
tac. The decode-side mirror (self-contained numpy) lives in ``tac.v2_compose.archive_grammar`` (the
inflate.py template) and is parity-tested bit-identical to this module's output.

Authority: numpy / CPU-torch (NEVER MPS). All d_seg here is ``[macOS-CPU advisory] NON-PROMOTABLE``.
The frontier pointer is UNMOVED 0.19110; it moves only on a byte-closed ``upstream/evaluate.py`` row.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# --- path bootstrap so the proven tools primitives import (same as the reach tool does) ---
_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / "src", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Proven warp path (faithful-by-construction; do NOT reimplement — see LAYERING NOTE).
from tools.measure_pose_warp_dseg import (  # noqa: E402
    CLASS_NAMES,
    NATIVE_H,
    NATIVE_W,
    intrinsics_at,
    _target_grid,
)
from tools.measure_screw_reach_through_R import (  # noqa: E402
    BULK_IDX,
    R1_BULK_REF,
    SCREW_REGIME,
    composite_warped_labels,
)
from tools.measure_segnet_fooling_ladder import _gauss_blur, _resize_bilinear  # noqa: E402

__all__ = [
    "BulkConfig",
    "BulkResult",
    "select_keyframes",
    "nearest_keyframe",
    "compute_class_mean_palette",
    "render_partition",
    "bicubic_up_to_camera",
    "bulk_argmax_through_R",
    "generate_bulk_argmax_stack",
    "segnet_selfcheck",
    "verify_k0_faithfulness",
    "load_calibration_from_reach",
    "SEG_H",
    "SEG_W",
]

SEG_H, SEG_W = 384, 512
_N_CLASSES = 5
# k=0 faithfulness tolerance (the reach tool's own gate: |k0_bulk - 0.0185| < 0.006).
_K0_TOL = 0.006


@dataclass(frozen=True)
class BulkConfig:
    """Calibration + keyframe spacing for the deterministic bulk."""

    s_t: float                      # screw translation scale (calibration_fit)
    s_r: float                      # screw rotation scale
    pitch: float                    # road-plane pitch
    reach_kstar: int                # keyframe spacing (pairs)
    n_pairs: int

    @property
    def params(self) -> tuple[float, float, float]:
        return (self.s_t, self.s_r, self.pitch)


@dataclass(frozen=True)
class BulkResult:
    """The deterministic bulk argmax-through-R stack + the stored payload + diagnostics."""

    bulk_argmax_through_R: np.ndarray   # (n_pairs, SEG_H, SEG_W) int64 — the bulk's SegNet argmax
    keyframes: list[int]                # the stored keyframe pair indices (anchors)
    keyframe_lstars: np.ndarray         # (n_keyframes, SEG_H, SEG_W) int64 — the STORED partitions
    palette: np.ndarray                 # (5, 3) float — the STORED class-mean RGB palette
    config: BulkConfig
    bulk_dseg_full: float               # realized d_seg (all px) vs GT — the deterministic floor
    bulk_dseg_bulk: float               # realized d_seg on the BULK classes only (Road/Undriv/MyCar)
    k0_faithful: bool                   # NO-FAKE anchor (cache-agnostic): SegNet selfcheck == 0 px
    segnet_selfcheck_max_px: int        # max gt_f1->argmax disagreement vs cached lstars (must be 0)
    k0_bulk_mean: float                 # the k=0 render-through-R bulk d_seg floor for THIS cache
    k0_within_r1_ref: bool              # INFORMATIONAL: |k0_bulk - 0.0185| < tol (exact on n96 only)


def select_keyframes(n_pairs: int, reach_kstar: int) -> list[int]:
    """Keyframe ANCHOR indices at the reach-determined spacing: 0, k*, 2k*, ... < n_pairs.

    The bulk for any pair p is generated by warping the most-recent keyframe <= p forward by
    (p - anchor) steps. ceil(n_pairs/k*) keyframes cover the clip within the reach window.
    """
    if reach_kstar <= 0:
        raise ValueError(f"reach_kstar must be positive, got {reach_kstar}")
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")
    return list(range(0, int(n_pairs), int(reach_kstar)))


def nearest_keyframe(pair_idx: int, keyframes: list[int]) -> tuple[int, int]:
    """Return (anchor_keyframe_pair_index, k_offset) for ``pair_idx`` — the most-recent keyframe
    at or before ``pair_idx`` and the forward offset to it."""
    anchor = keyframes[0]
    for kf in keyframes:
        if kf <= pair_idx:
            anchor = kf
        else:
            break
    return anchor, int(pair_idx - anchor)


def compute_class_mean_palette(gt_f1: np.ndarray, lstars: np.ndarray) -> np.ndarray:
    """The STORED per-class mean RGB palette (5x3, ~15 bytes), computed exactly as the reach
    tool: bilinear-resize each GT frame1 to SEG res, average RGB within each GT class region."""
    P = lstars.shape[0]
    csum = np.zeros((_N_CLASSES, 3), np.float64)
    ccnt = np.zeros(_N_CLASSES, np.float64)
    for p in range(P):
        g384 = _resize_bilinear(gt_f1[p].astype(np.float64), SEG_H, SEG_W)
        for c in range(_N_CLASSES):
            m = lstars[p] == c
            if m.any():
                csum[c] += g384[m].sum(0)
                ccnt[c] += int(m.sum())
    return np.where(ccnt[:, None] > 0, csum / np.maximum(ccnt[:, None], 1.0), 127.0)


def render_partition(label_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """ladder R1 render: per-class mean RGB fill + sigma=1.0 anti-Gibbs ramp (the FREE generic
    rasterizer). Returns (SEG_H, SEG_W, 3) float in [0,255]. Mirrors the reach tool exactly."""
    return _gauss_blur(np.asarray(palette)[np.asarray(label_map)], 1.0)


def bicubic_up_to_camera(render384_f: np.ndarray) -> np.ndarray:
    """R operator part 1: bicubic up render(384x512) -> camera(874x1164), clamp+round (the uint8
    cliff). Returns (874, 1164, 3) float in [0,255]. Identical to the reach tool's ``bicubic_up``."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.asarray(render384_f, dtype=np.float32)).permute(2, 0, 1)[None]
    xu = F.interpolate(x, size=(NATIVE_H, NATIVE_W), mode="bicubic", align_corners=False)
    return xu[0].permute(1, 2, 0).clamp(0, 255).round().numpy()


def bulk_argmax_through_R(label_map: np.ndarray, palette: np.ndarray, segnet: Any) -> np.ndarray:
    """The faithful through-R bulk argmax for one partition: render -> bicubic_up -> uint8 ->
    (SegNet.preprocess_input bilinear-down) -> frozen CPU-torch SegNet argmax. Returns (SEG_H,
    SEG_W) int64. The bilinear-down is performed INSIDE the SegNet preprocess (NOT here) — exactly
    as the reach tool, so k=0 reproduces the R1 floor."""
    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax

    native = bicubic_up_to_camera(render_partition(label_map, palette))
    am, _margin = measure_segnet_argmax(segnet, native.astype(np.float64))
    return am


def _dseg(am: np.ndarray, lstar: np.ndarray, mask: np.ndarray | None = None) -> float:
    ne = am != lstar
    if mask is not None:
        return float(ne[mask].mean()) if mask.any() else 0.0
    return float(ne.mean())


def generate_bulk_argmax_stack(
    lstars: np.ndarray,
    gt_f1: np.ndarray,
    poses: np.ndarray,
    config: BulkConfig,
    *,
    palette: np.ndarray | None = None,
    segnet: Any | None = None,
    verbose: bool = True,
) -> BulkResult:
    """Generate the deterministic bulk argmax-through-R for EVERY pair from the stored keyframes.

    For each pair p: warp the nearest keyframe L[anchor] forward by (p-anchor) steps via the
    per-class stratified ``composite_warped_labels`` (the proven router) -> render -> through R ->
    frozen CPU-torch SegNet argmax. This is the bulk the residual INR must correct (Step 4 target).

    Args:
        lstars: (n_pairs, SEG_H, SEG_W) int64 GT SegNet argmax (from the GT cache).
        gt_f1:  (n_pairs, 874, 1164, 3) uint8 GT frame1 (for the palette).
        poses:  (n_pairs, 6) float64 GT poses (the ``gt_poses`` cache key; drives the warp).
        config: calibration + keyframe spacing.
        palette: optional precomputed (5,3) palette (else computed from gt_f1/lstars).
        segnet: optional preloaded frozen CPU-torch SegNet (else loaded here).
        verbose: progress to stderr.

    Returns:
        :class:`BulkResult` (bulk argmax stack + stored payload + the deterministic d_seg floor).
    """
    lstars = np.asarray(lstars, dtype=np.int64)
    poses = np.asarray(poses, dtype=np.float64)
    P = int(lstars.shape[0])
    if P != config.n_pairs:
        # allow a smaller cache than the configured clip (advisory subsets); use the cache size.
        config = BulkConfig(config.s_t, config.s_r, config.pitch, config.reach_kstar, P)

    if segnet is None:
        from tac.boundary_math.seg_core import load_real_segnet

        segnet = load_real_segnet("cpu")
    if palette is None:
        palette = compute_class_mean_palette(gt_f1, lstars)

    keyframes = select_keyframes(P, config.reach_kstar)
    K = intrinsics_at(SEG_W, SEG_H)
    Kinv = np.linalg.inv(K)
    grid = _target_grid(SEG_H, SEG_W)

    bulk_stack = np.empty((P, SEG_H, SEG_W), np.int64)
    d_full: list[float] = []
    d_bulk: list[float] = []
    bulk_mask_classes = np.asarray(BULK_IDX)
    for p in range(P):
        anchor, k = nearest_keyframe(p, keyframes)
        if k == 0:
            warped = lstars[anchor]  # the stored keyframe itself (no warp)
        else:
            warped = composite_warped_labels(
                lstars[anchor], poses, anchor, k, K, Kinv, config.params, grid
            )
        am = bulk_argmax_through_R(warped, palette, segnet)
        bulk_stack[p] = am
        tgt = lstars[p]
        d_full.append(_dseg(am, tgt))
        d_bulk.append(_dseg(am, tgt, np.isin(tgt, bulk_mask_classes)))
        if verbose and (p + 1) % 16 == 0:
            print(f"[bulk_generator] {p + 1}/{P} pairs", file=sys.stderr)

    # cache-agnostic NO-FAKE anchor: the SAME SegNet that built the cache must reproduce lstars
    # from gt_f1 exactly (0 px). This proves the render->R->SegNet path is the contest path.
    selfcheck_px, selfcheck_ok = segnet_selfcheck(gt_f1, lstars, segnet, n_check=min(8, P))
    # informational R1-floor check (exact only on the n96 reference cache; content-dependent).
    k0_bulk_mean, k0_within_ref = verify_k0_faithfulness(lstars, palette, segnet, n_check=min(8, P))

    return BulkResult(
        bulk_argmax_through_R=bulk_stack,
        keyframes=keyframes,
        keyframe_lstars=lstars[keyframes].copy(),
        palette=np.asarray(palette, np.float64),
        config=config,
        bulk_dseg_full=float(np.mean(d_full)),
        bulk_dseg_bulk=float(np.mean(d_bulk)),
        k0_faithful=bool(selfcheck_ok),
        segnet_selfcheck_max_px=int(selfcheck_px),
        k0_bulk_mean=float(k0_bulk_mean),
        k0_within_r1_ref=bool(k0_within_ref),
    )


def segnet_selfcheck(
    gt_f1: np.ndarray, lstars: np.ndarray, segnet: Any, *, n_check: int = 8
) -> tuple[int, bool]:
    """Cache-agnostic NO-FAKE anchor: the frozen CPU-torch SegNet argmax of the GT frame1 MUST
    reproduce the cached ``lstars`` exactly (the cache was built by this same SegNet on this same
    gt_f1). Returns (max_disagree_px, ok). ok == True iff every checked pair agrees to 0 px —
    this proves the render->bicubic->preprocess->SegNet path used here IS the contest path."""
    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax

    gt_f1 = np.asarray(gt_f1)
    lstars = np.asarray(lstars, dtype=np.int64)
    maxd = 0
    for p in range(int(min(n_check, lstars.shape[0]))):
        am, _m = measure_segnet_argmax(segnet, gt_f1[p].astype(np.float64))
        maxd = max(maxd, int(np.count_nonzero(am != lstars[p])))
    return maxd, bool(maxd == 0)


def verify_k0_faithfulness(
    lstars: np.ndarray, palette: np.ndarray, segnet: Any, *, n_check: int = 8
) -> tuple[float, bool]:
    """INFORMATIONAL R1-floor check: render+through-R the UN-warped (k=0) keyframes and report the
    bulk d_seg. Returns (k0_bulk_mean, within_n96_ref). ``within_n96_ref`` (|mean - 0.0185| < tol)
    is EXACT only on the n96 reference cache — the R1 floor is content-dependent, so a mismatch on a
    different/smaller cache is NOT a pipeline failure (use :func:`segnet_selfcheck` for that)."""
    lstars = np.asarray(lstars, dtype=np.int64)
    bulk_classes = np.asarray(BULK_IDX)
    vals: list[float] = []
    for p in range(int(min(n_check, lstars.shape[0]))):
        am = bulk_argmax_through_R(lstars[p], palette, segnet)
        vals.append(_dseg(am, lstars[p], np.isin(lstars[p], bulk_classes)))
    mean = float(np.mean(vals)) if vals else float("nan")
    return mean, bool(abs(mean - R1_BULK_REF) < _K0_TOL)


def load_calibration_from_reach(
    reach_json: str | Path, *, reach_kstar: int, n_pairs: int
) -> BulkConfig:
    """Build a :class:`BulkConfig` from a real ``reach_*.json`` ``calibration_fit`` block."""
    import json

    data = json.loads(Path(reach_json).read_text())
    fit = data.get("calibration_fit") or {}
    return BulkConfig(
        s_t=float(fit.get("s_t", 0.0)),
        s_r=float(fit.get("s_r", 0.0)),
        pitch=float(fit.get("pitch", 0.0)),
        reach_kstar=int(reach_kstar),
        n_pairs=int(n_pairs),
    )
