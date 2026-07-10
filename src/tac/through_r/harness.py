# SPDX-License-Identifier: MIT
"""harness — the canonical through-R d_seg measurement (candidate frames -> R -> SegNet -> d_seg).

THE most-rebuilt pattern in the campaign, canonicalized: load ``gt_n600.npz`` -> take a
candidate frame1 (render-grid float OR camera-uint8) per pair -> push through R
(:func:`resolution_chain.render_grid_to_camera_uint8`) -> the FROZEN CPU-torch SegNet
(``upstream/modules.py``, loaded via :func:`tac.boundary_math.seg_core.load_real_segnet`) ->
argmax -> per-class + aggregate d_seg vs the cached ``lstars`` (L*).

REUSE-not-rederive (NO-FAKE):
  * The R operator is the pinned :mod:`resolution_chain` (source-verified constants).
  * The SegNet forward mirrors the campaign authority
    ``train_witness_realized_through_R_mlx.cpu_verdict_d_seg_argmax_batch`` op-for-op
    (``(N,1,H,W,3)`` -> permute -> ``segnet.preprocess_input`` -> forward -> ``argmax(dim=1)``),
    CHUNKED per the n600-verdict-OOM law (full-P batch spikes ~+66 GiB; default chunk 32).
  * The aggregate d_seg is the authority functional
    :func:`tac.boundary_math.bitmask_dseg.d_seg_reference`; the per-class decomposition is the
    canonical sensor :func:`tac.witness_control.perclass_verdict.per_class_flip_stats` /
    :func:`per_class_dseg_fields` (``sum(flips)/sum(pixels) == d_seg`` by construction).

P9 (this IS the thing, no proxy branch): ``backend='cpu-torch'`` is the ONLY authority. MPS /
MLX are NEVER a score (CLAUDE.md) — an unsupported backend RAISES, it does not silently
degrade to a proxy. The realized d_seg is authority-grade-through-R but ``[macOS-CPU advisory .
NON-PROMOTABLE]`` until byte-closed exact eval moves the pointer.

n600 discipline (allergic to non-n600 / toys): ``pairs='n600'`` refuses a subset UNLESS
``allow_subset_reason=`` is an explicit non-empty string (labelled non-authority) — mirrors
:func:`tac.inc1a_harness.mask_dseg_meter.measure_mask_dseg`'s ``require_n600``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tac.boundary_math.bitmask_dseg import d_seg_reference
from tac.through_r.resolution_chain import (
    CAMERA_H,
    CAMERA_W,
    SEG_H,
    SEG_W,
    render_grid_to_camera_uint8,
)
from tac.witness_control.perclass_verdict import (
    CLASS_NAMES,
    N_CLASSES,
    per_class_dseg_fields,
    per_class_flip_stats,
)

N600 = 600
DEFAULT_GT_CACHE = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
SUPPORTED_BACKENDS = ("cpu-torch",)
DEFAULT_VERDICT_BATCH = 32  # n600-verdict-OOM law: chunk the full-P SegNet forward.
THROUGH_R_LABEL = "[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]"


class ThroughRHarnessError(ValueError):
    """Raised when the harness is asked to measure a toy / mis-shaped / non-authority input."""


@dataclass
class ThroughRResult:
    """The measured through-R d_seg for one candidate's per-pair frames vs L*."""

    agg_dseg: float
    per_class_dseg: dict[str, float]
    flip_share_by_class: dict[str, float]
    per_pair_dseg: list[float]
    n_frames: int
    n_classes: int
    is_n600: bool
    backend: str
    input_space: str
    total_flips: int = 0
    total_pixels: int = 0
    label: str = THROUGH_R_LABEL
    subset_reason: str | None = None
    realized: np.ndarray | None = None  # (N, SEG_H, SEG_W) int64, only when return_realized
    extra: dict[str, float] = field(default_factory=dict)


def load_frozen_segnet(backend: str = "cpu-torch"):
    """Load the frozen SegNet authority for ``backend``. Only ``cpu-torch`` is authority.

    Delegates to :func:`tac.boundary_math.seg_core.load_real_segnet` on CPU. MPS / MLX are
    REFUSED (never a score authority) — the refusal is information, not a fallback (P9).
    """

    if backend not in SUPPORTED_BACKENDS:
        raise ThroughRHarnessError(
            f"backend {backend!r} is not an authority; only {SUPPORTED_BACKENDS} may score "
            "(MPS/MLX are NEVER a score authority per CLAUDE.md). No proxy fallback."
        )
    from tac.boundary_math.seg_core import load_real_segnet

    return load_real_segnet("cpu")


def load_gt_lstars(
    gt_cache: str | Path = DEFAULT_GT_CACHE, *, n: int | None = None
) -> np.ndarray:
    """Load the cached GT SegNet argmax ``lstars`` ``(N, SEG_H, SEG_W)`` int64 from the npz.

    ``n=None`` returns ALL cached frames. Fail-closed if the cache or key is missing.
    """

    p = Path(gt_cache)
    if not p.exists():
        raise ThroughRHarnessError(f"gt cache not found: {p}")
    z = np.load(p, mmap_mode="r")
    if "lstars" not in z.files:
        raise ThroughRHarnessError(f"gt cache {p} has no 'lstars' key; has {list(z.files)}")
    lstars = z["lstars"]
    total = int(lstars.shape[0])
    take = total if n is None else min(int(n), total)
    return np.asarray(lstars[:take]).astype(np.int64)


def _to_camera_uint8_frames(
    frames: list[np.ndarray], input_space: str
) -> tuple[list[np.ndarray], str]:
    """Normalize candidate frames to camera-uint8 ``(CAMERA_H, CAMERA_W, 3)`` via R's first half.

    ``input_space``:
      * ``'camera-uint8'`` — frames are already the stored camera recon; validated only.
      * ``'render-grid'`` — frames are render-grid float; each pushed through R (bicubic UP).
      * ``'auto'`` — per-frame: (CAMERA_H,CAMERA_W) uint8 => camera-uint8, else render-grid.
    Returns ``(camera_frames, resolved_space)`` where ``resolved_space`` records what was done
    (``'mixed'`` if auto saw both). NO silent subset — a wrong dtype/shape RAISES.
    """

    if input_space not in ("auto", "camera-uint8", "render-grid"):
        raise ThroughRHarnessError(
            f"input_space must be auto|camera-uint8|render-grid; got {input_space!r}"
        )
    out: list[np.ndarray] = []
    saw_camera = False
    saw_render = False
    for i, f in enumerate(frames):
        a = np.asarray(f)
        if a.ndim != 3 or a.shape[-1] != 3:
            raise ThroughRHarnessError(f"frame {i} must be (H,W,3); got {a.shape}")
        is_camera = a.shape[0] == CAMERA_H and a.shape[1] == CAMERA_W and a.dtype == np.uint8
        if input_space == "camera-uint8":
            if not is_camera:
                raise ThroughRHarnessError(
                    f"frame {i}: input_space='camera-uint8' needs (CAMERA_H,CAMERA_W,3) uint8; "
                    f"got shape {a.shape} dtype {a.dtype}"
                )
            out.append(np.ascontiguousarray(a))
            saw_camera = True
        elif input_space == "render-grid":
            out.append(render_grid_to_camera_uint8(a))
            saw_render = True
        else:  # auto
            if is_camera:
                out.append(np.ascontiguousarray(a))
                saw_camera = True
            else:
                out.append(render_grid_to_camera_uint8(a))
                saw_render = True
    if input_space != "auto":
        resolved = input_space
    elif saw_camera and saw_render:
        resolved = "mixed"
    elif saw_camera:
        resolved = "camera-uint8"
    else:
        resolved = "render-grid"
    return out, resolved


def _segnet_argmax_chunked(segnet, camera_frames: list[np.ndarray], chunk: int) -> np.ndarray:
    """Batched SegNet argmax over N camera-uint8 frames, CHUNKED (n600-verdict-OOM law).

    Op-for-op mirror of ``cpu_verdict_d_seg_argmax_batch``: ``(n,1,H,W,3)`` -> permute to
    ``(n,1,3,H,W)`` -> ``segnet.preprocess_input`` (bilinear DOWN to SEG_HW + last-frame) ->
    ``argmax(dim=1)``. Chunk-size independent (``.eval()`` BatchNorm uses running stats;
    argmax is per-pixel), so the result is bit-identical to any chunking. ``chunk<=0`` =>
    single batch (the pre-fix path, memory-permitting). Returns ``(N, SEG_H, SEG_W)`` int64.
    """

    import torch

    n = len(camera_frames)
    step = n if int(chunk) <= 0 else int(chunk)
    realized = np.empty((n, SEG_H, SEG_W), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, n, step):
            block = camera_frames[start : start + step]
            arr = np.stack([np.asarray(f)[None] for f in block], axis=0)  # (b,1,H,W,3)
            xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()  # (b,1,3,H,W)
            seg_in = segnet.preprocess_input(xp)
            am = segnet(seg_in).argmax(dim=1).cpu().numpy().astype(np.int64)  # (b, SEG_H, SEG_W)
            realized[start : start + am.shape[0]] = am
    return realized


def measure_through_r(
    fields_or_frames: list[np.ndarray],
    *,
    lstars: np.ndarray | None = None,
    gt_cache: str | Path = DEFAULT_GT_CACHE,
    pairs: str = "n600",
    backend: str = "cpu-torch",
    input_space: str = "auto",
    verdict_batch: int = DEFAULT_VERDICT_BATCH,
    n_classes: int = N_CLASSES,
    allow_subset_reason: str | None = None,
    return_realized: bool = False,
    segnet: Any | None = None,
) -> ThroughRResult:
    """Measure through-R d_seg of candidate ``fields_or_frames`` vs L* on the frozen CPU SegNet.

    ``fields_or_frames``: list of per-pair candidate frame1 RGB — EITHER camera-uint8
    ``(CAMERA_H,CAMERA_W,3)`` (the stored recon) OR render-grid float ``(h,w,3)`` (pushed
    through R). ``input_space`` selects/auto-detects (see :func:`_to_camera_uint8_frames`).

    ``lstars``: the GT argmax stack ``(N,SEG_H,SEG_W)``; if None it is loaded from ``gt_cache``
    truncated to ``len(fields_or_frames)``. ``pairs='n600'`` REFUSES ``N != 600`` unless
    ``allow_subset_reason`` is a non-empty string (labelled non-authority subset).

    Deterministic: CPU-torch ``.eval()`` inference is deterministic and chunk-size independent.
    Returns a :class:`ThroughRResult`.
    """

    if backend not in SUPPORTED_BACKENDS:
        raise ThroughRHarnessError(
            f"backend {backend!r} is not an authority; only {SUPPORTED_BACKENDS} may score."
        )
    frames = list(fields_or_frames)
    n = len(frames)
    if n == 0:
        raise ThroughRHarnessError("no candidate frames to measure")

    subset_reason: str | None = None
    if pairs == "n600":
        is_n600 = n == N600
        if not is_n600:
            reason = (allow_subset_reason or "").strip()
            if not reason:
                raise ThroughRHarnessError(
                    f"n600 discipline: a verdict needs N=600; got N={n}. Pass a non-empty "
                    "allow_subset_reason='...' ONLY for an explicitly non-authority subset."
                )
            subset_reason = reason
    else:
        raise ThroughRHarnessError(f"pairs must be 'n600' (the authority protocol); got {pairs!r}")

    if lstars is None:
        lstars = load_gt_lstars(gt_cache, n=n)
    lstars = np.asarray(lstars)
    if lstars.shape[0] < n:
        raise ThroughRHarnessError(
            f"lstars has {lstars.shape[0]} frames < {n} candidates; cache too small."
        )
    lstars = lstars[:n]
    if lstars.shape[1:] != (SEG_H, SEG_W):
        raise ThroughRHarnessError(
            f"lstars grid {lstars.shape[1:]} != SEG_HW ({SEG_H},{SEG_W}); wrong compare grid."
        )

    camera_frames, resolved_space = _to_camera_uint8_frames(frames, input_space)
    if segnet is None:
        segnet = load_frozen_segnet(backend)
    realized = _segnet_argmax_chunked(segnet, camera_frames, int(verdict_batch))

    # Aggregate = mean over pairs of the authority functional (realized != L*).mean().
    gt_list = [lstars[i] for i in range(n)]
    pred_list = [realized[i] for i in range(n)]
    per_pair = [float(d_seg_reference(pred_list[i], gt_list[i])) for i in range(n)]
    # TODO(#389): emit a MeasurementRow per pair (through_r verdict rows) once the sibling
    # tac.verdicts.MeasurementRow schema is wired by the #389 sweep. Do NOT import it here.
    agg = float(np.mean(per_pair))

    flips, pixels = per_class_flip_stats(pred_list, gt_list, n_classes=int(n_classes))
    fields = per_class_dseg_fields(flips, pixels)
    names = CLASS_NAMES if int(n_classes) == N_CLASSES else tuple(
        f"class_{c}" for c in range(int(n_classes))
    )
    per_class = {names[c]: float(fields["d_seg_by_class"][c]) for c in range(int(n_classes))}
    share = {names[c]: float(fields["flip_share_by_class"][c]) for c in range(int(n_classes))}

    return ThroughRResult(
        agg_dseg=agg,
        per_class_dseg=per_class,
        flip_share_by_class=share,
        per_pair_dseg=per_pair,
        n_frames=n,
        n_classes=int(n_classes),
        is_n600=n == N600,
        backend=backend,
        input_space=resolved_space,
        total_flips=int(flips.sum()),
        total_pixels=int(pixels.sum()),
        subset_reason=subset_reason,
        realized=realized if return_realized else None,
        extra={"per_pair_std": float(np.std(per_pair))},
    )


__all__ = [
    "DEFAULT_GT_CACHE",
    "DEFAULT_VERDICT_BATCH",
    "N600",
    "SUPPORTED_BACKENDS",
    "THROUGH_R_LABEL",
    "ThroughRHarnessError",
    "ThroughRResult",
    "load_frozen_segnet",
    "load_gt_lstars",
    "measure_through_r",
]
