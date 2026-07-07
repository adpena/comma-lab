# SPDX-License-Identifier: MIT
"""LEVEL-SET WITNESS through R (MLX) — softmax-of-SDF + curvelet, REALIZED d_seg, pose-legal RGB.

Composes the GO'd level-set representation into the existing realized-through-R witness vehicle
WITHOUT duplicating it: this trainer IMPORTS the RGB-render / R / frozen-MLX-scorer / frozen-
CPU-torch-verdict / EMA / curriculum-loss / byte-close primitives from
``experiments/train_witness_realized_through_R_mlx.py`` (owned by a7660df3 — NOT edited here)
and ``tools/witness_byte_close_and_eval.py`` (byte-close parity), and the SDF/curvelet head +
reg + quantize from ``tac.boundary_math.lever_b_levelset_generator`` (this campaign's module)
and the byte-closeable self-orientation directional basis from ``tac.boundary_math.lever_b_generator``.

THE COMPOSITION (the decisive sub-0.15 vehicle):
  curvelet/shearlet (or self-orientation) front-end  →  FiLM-modulated WIRE/HOSC trunk  →
    (a) K SDF fields phi  (1-Lipschitz level-set partition; argmax_k phi_k = the seg structure)
    (b) per-(pair,frame) RGB texture  (pose-carrying luma+chroma detail)
  RGB = sigmoid( softmax(phi/T) @ palette  +  texture ) * 255      (POSE-LEGAL, not flat palette)
  RGB --R--> frozen SegNet argmax  ==>  REALIZED d_seg  (the SDF makes the COLOR boundary track
  the 1-Lipschitz level set -> the SegNet argmax boundary R-survives, the GO'd -587x lever)
  RGB --R--> frozen PoseNet YUV6   ==>  REALIZED d_pose  (the texture carries pose; the
  stored-pose target is the Quantizr sidecar's GT — pose is solved, witness's job is d_seg)

WHY pose-legal (the coordinator's make-or-break): a flat ``softmax(phi/T)@palette`` frame is
POSE-BLIND (measured S=11.65). The additive per-(pair,frame) ``texture`` head restores the
luma/chroma detail PoseNet's YUV6 needs while the palette term keeps the SegNet argmax pinned to
the SDF partition. d_seg is REALIZED (render -> _torch_R_to_camera_uint8 -> frozen CPU-torch
SegNet argmax), NEVER a field-level proxy.

COMPUTE-SUBSTRATE LAW / NO-FAKE / authority: identical to the imported trainer — MLX (cpu/gpu)
is the fp32 TRAINING-GRADIENT device; the d_seg/d_pose VERDICT is the FROZEN CPU-torch SegNet
argmax + PoseNet MSE (NEVER MLX, NEVER MPS). Evidence ``[macOS-MLX training-gradient]`` /
verdict ``[macOS-CPU advisory]``; promotion_eligible=False; pointer UNMOVED until a byte-closed
exact-eval row (tools/witness_byte_close_and_eval.py) lands sub-0.19110.

BORROWED-SUBSTRATE (NO-FAKE #7): BORROWED = the entire realized-through-R RGB-witness pipeline
(a7660df3), curvelets/shearlets, WIRE/HOSC, FiLM, Eikonal/Chan-Vese, the frozen scorers + CPU
authority. OURS-ORIGINAL = composing the SegNet argmax as a softmax-of-SDF level set whose
1-Lipschitz boundary R-survives, rendered as POSE-LEGAL palette+texture RGB, driven by a generic
(byte-closeable, GT-free) curvelet front-end — the joint R-aliasing + directional-byte-close fix.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── imports from the RGB-witness trainer (a7660df3 — reuse, do NOT edit) ──
from train_witness_realized_through_R_mlx import (  # noqa: E402
    MlxEMA,
    SEG_W,
    _build_render_coords,
    _seed_muon_momentum_from_adam,
    _torch_R_to_camera_uint8,
    cpu_verdict_d_pose_batch,
    cpu_verdict_d_seg_argmax_batch,
    cpu_verdict_d_seg_batch,
    focal_pixel_weight_mlx,
    implied_score_from_verdict,
    load_gt_from_cache,
    make_loss_fn,
    maybe_enable_mx_compile_r,
    precompute_gt,
    r_isolated_microbench,
    render_through_R_mlx,
    set_fused_r_kernel,
)

# ── imports from this campaign's level-set module + the byte-closeable directional basis ──
from tac.boundary_math.lever_b_generator import self_orientation_directional_feats  # noqa: E402
from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    curvelet_directional_B,
    curvelet_feats,
    film_modulation_participation_ratio,
    int8_dequant_params,
    lane_thin_weight_map,
    levelset_rgb_forward_numpy,
    quantize_levelset_blob,
    rebuild_per_pair_feats_in_place,
)
from tac.optimization.muon_finisher_mlx import (  # noqa: E402
    build_muon_finisher_optimizer,
    count_muon_adamw_split,
)
from tac.optimization.md_decoupling import (  # noqa: E402
    stiefel_project_columns,
    stiefel_residual,
)
# SENSE (opt-in --annulus-telemetry): REUSE the pure codim-1 boundary-annulus metric math
# (no reimplementation). Only the low-level pure fns are used (flip split + per-class + GT-margin
# percentiles); the full-margin/gibbs series lives in tools/witness_annulus_convergence.py.
from tac import witness_annulus_metrics as _wam  # noqa: E402

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rss_gib() -> float:
    """Best-effort resident-set-size of THIS process in GiB (psutil, then resource fallback).

    Used only for observability (the #205 OOM instrumentation) -- NEVER read back into training
    (BIT-IDENTICAL). Returns -1.0 when unavailable (NO-FAKE: never a fabricated number)."""
    try:
        import psutil  # noqa: PLC0415

        return float(psutil.Process().memory_info().rss) / (1024.0 ** 3)
    except Exception:
        try:
            import resource  # noqa: PLC0415

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS ru_maxrss is BYTES; Linux is KiB. Heuristic: >1e9 => already bytes.
            return float(ru) / (1024.0 ** 3) if ru > 1e9 else float(ru) / (1024.0 ** 2)
        except Exception:
            return -1.0


def _verdict_dseg_dpose_chunked(
    seg_cpu: Any, posenet_cpu: Any,
    f0s: list, f1s: list, lstars: list, poses: list, *, vbatch: int,
    return_realized: bool = False,
) -> "tuple[float, float] | tuple[float, float, list]":
    """(#205 REAL OOM FIX) mean d_seg / d_pose over N pairs, running SegNet/PoseNet in CHUNKS of
    ``vbatch`` instead of one N-wide torch batch.

    The batched verdict (``cpu_verdict_d_seg_batch`` / ``cpu_verdict_d_pose_batch``) casts a
    ``(N, 2, 3, 874, 1164)`` uint8 stack to **fp32** (= ~14.6 GiB at N=600) and forwards it through
    EfficientNet-B2 / FastViT-T12 in ONE batch -> tens of GiB of activations. That transient spike,
    on top of the resident ~41 GiB self-orient cf_mx_cache, is what tripped the 90 GB safe-run guard
    and killed the n600 launch before its first checkpoint. Chunking bounds the transient to
    ``vbatch`` pairs.

    BIT-IDENTICAL: the scorers run under ``torch.inference_mode()`` in EVAL mode -> BatchNorm uses
    RUNNING stats (batch-size-independent), argmax is per-pixel, MSE is per-pair -> the per-chunk
    concatenation equals the single N-wide batch to the last bit. ``vbatch<=0`` restores the
    single-batch (pre-fix) path for the A/B parity check.

    ``return_realized`` (DEFAULT False => BYTE-IDENTICAL to the sealed #205 verdict): when True, ALSO
    returns the realized SegNet argmax maps (list of (h,w) int64) collected from the SAME forward via
    ``cpu_verdict_d_seg_argmax_batch`` (whose per-pair d_seg is bit-identical to ``cpu_verdict_d_seg_batch``
    -- same preprocess -> forward -> argmax(dim=1) -> per-pixel disagreement). This lets the opt-in
    ``--annulus-telemetry`` row reuse ONE forward instead of a second SegNet pass; the returned scalars
    are unchanged. The default branch below is left EXACTLY as before so the flag-absent path is
    byte-identical."""
    n = len(f1s)
    if not return_realized:
        # ── UNCHANGED default path (byte-identical to the sealed #205 verdict) ──
        if vbatch is None or vbatch <= 0 or vbatch >= n:
            ds = cpu_verdict_d_seg_batch(seg_cpu, f1s, lstars)
            dp = cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, poses)
            return float(np.mean(ds)), float(np.mean(dp))
        ds_all: list[float] = []
        dp_all: list[float] = []
        for s in range(0, n, vbatch):
            e = min(s + vbatch, n)
            ds_all.extend(cpu_verdict_d_seg_batch(seg_cpu, f1s[s:e], lstars[s:e]))
            dp_all.extend(cpu_verdict_d_pose_batch(posenet_cpu, f0s[s:e], f1s[s:e], poses[s:e]))
        return float(np.mean(ds_all)), float(np.mean(dp_all))
    # ── return_realized path (annulus telemetry ON only): bit-identical d_seg via the argmax variant,
    #    plus the realized maps collected from the SAME chunked forward. ──
    vb = n if (vbatch is None or vbatch <= 0 or vbatch >= n) else int(vbatch)
    ds_all = []
    dp_all = []
    realized_all: list = []
    for s in range(0, n, vb):
        e = min(s + vb, n)
        ds_chunk, realized = cpu_verdict_d_seg_argmax_batch(seg_cpu, f1s[s:e], lstars[s:e])
        ds_all.extend(ds_chunk)
        dp_all.extend(cpu_verdict_d_pose_batch(posenet_cpu, f0s[s:e], f1s[s:e], poses[s:e]))
        realized_all.extend([realized[i] for i in range(e - s)])
    return float(np.mean(ds_all)), float(np.mean(dp_all)), realized_all


def _verdict_dseg_dpose_nucleus_chunked(
    seg_cpu: Any, posenet_cpu: Any,
    f0s: list, f1s: list, lstars: list, poses: list, *, vbatch: int,
) -> "tuple[float, float, dict]":
    """(#302 nucleus guard) Same chunked mean d_seg / d_pose as ``_verdict_dseg_dpose_chunked`` PLUS
    the per-class critical-nucleus COUNTS, in the SAME single SegNet forward (no double cost).

    Returns ``(d_seg, d_pose, counts)`` where ``counts`` is the accumulated ``_evt_nucleus_counts``
    dict over all pairs. The d_seg / d_pose are BIT-IDENTICAL to ``_verdict_dseg_dpose_chunked``
    (same ``cpu_verdict_d_seg_argmax_batch`` d_seg + same ``cpu_verdict_d_pose_batch`` MSE). The
    argmax the seg verdict already computed is decomposed by class per chunk and accumulated, so the
    nucleus guard costs ZERO extra SegNet forwards. Called ONLY when the nucleus guard / readiness
    telemetry is engaged (else the OFF path uses ``_verdict_dseg_dpose_chunked`` unchanged =>
    byte-identical). ``vbatch <= 0`` runs the single-batch path (parity check)."""
    n = len(f1s)
    vb = n if (vbatch is None or vbatch <= 0 or vbatch >= n) else int(vbatch)
    ds_all: list[float] = []
    dp_all: list[float] = []
    counts: "dict | None" = None
    for s in range(0, n, vb):
        e = min(s + vb, n)
        ds_chunk, realized = cpu_verdict_d_seg_argmax_batch(seg_cpu, f1s[s:e], lstars[s:e])
        ds_all.extend(ds_chunk)
        dp_all.extend(cpu_verdict_d_pose_batch(posenet_cpu, f0s[s:e], f1s[s:e], poses[s:e]))
        counts = _evt_counts_add(
            counts, _evt_nucleus_counts([realized[i] for i in range(e - s)], lstars[s:e]))
    return (float(np.mean(ds_all)), float(np.mean(dp_all)),
            counts if counts is not None else _evt_nucleus_counts([], []))


# ---------------------------------------------------------------------------
# SENSE (opt-in --annulus-telemetry): in-trainer annulus_convergence telemetry. Pure numpy helpers
# (MLX-free, torch-free), factored out of the verdict closure so they are unit-testable at $0. They
# REUSE tac.witness_annulus_metrics (no metric reimplementation). OBSERVABILITY-ONLY: the row is a
# companion to the {stage:verdict} row, NEVER read back into training/parity/resume => flag-absent
# runs are byte-identical (nothing constructs a row).
# ---------------------------------------------------------------------------
def _annulus_realized_maps(seg_cpu: Any, f1s: list, lstars: list, vbatch: int) -> list:
    """Realized SegNet argmax maps over the frame1's, chunked by ``vbatch`` (reuses
    ``cpu_verdict_d_seg_argmax_batch``). Used ONLY when the annulus row needs a dedicated forward
    (the rare nucleus-ON + annulus-ON combo; the common annulus-ON/nucleus-OFF path reuses the
    verdict's own forward via ``_verdict_dseg_dpose_chunked(return_realized=True)``)."""
    n = len(f1s)
    vb = n if (vbatch is None or vbatch <= 0 or vbatch >= n) else int(vbatch)
    out: list = []
    for s in range(0, n, vb):
        e = min(s + vb, n)
        _ds, realized = cpu_verdict_d_seg_argmax_batch(seg_cpu, f1s[s:e], lstars[s:e])
        out.extend([realized[i] for i in range(e - s)])
    return out


def _annulus_metrics_from_maps(
    realized_list: list, gt_lstars_list: list, gt_margins_list: list, *,
    band: float, bottom_k: float, chunk: int = 32,
) -> dict:
    """Codim-1 boundary-annulus convergence metrics from the realized argmax + GT argmax + GT margin
    (all SegNet-resolution per-pair (h,w) maps). REUSES the mask fns + ``flip_map`` in
    ``tac.witness_annulus_metrics`` and ACCUMULATES the SAME per-region counts + collects the SAME
    annulus GT-margin values the pure ``checkpoint_metrics`` fns compute -- the emitted VALUES are
    NUMERICALLY IDENTICAL to the pre-fix all-at-once computation (integer counts add exactly; the
    percentiles are ``np.percentile`` over the identical value multiset).

    MEMORY (default-ON telemetry per CLAUDE.md "'Off' is a tracked queue"): the pre-fix path
    ``np.stack``-ed ALL n_pairs realized-argmax + GT-argmax (int64) + GT-margin (float32) into three
    (N,h,w) arrays => ~2.25 GiB transient at n600 EVERY --eval-every epoch. This streams in CHUNKS of
    ``chunk`` pairs (like ``_verdict_dseg_dpose_chunked``): the two int64 argmax stacks (~1.9 GiB) are
    NEVER materialized full; the only O(N) buffer is a SINGLE flat ``|GT margin|`` float32 plane
    (~471 MiB) that an EXACT ``np.quantile`` (the bottom-k global threshold) fundamentally requires
    (two-pass), freed before the streaming pass. Peak transient ~= that one plane + O(chunk), a >3x cut.

    DOCUMENTED PARTIAL (NO-FAKE): the verdict scope cheaply carries the realized ARGMAX (one SegNet
    forward) + the FIXED GT top1-top2 margin field (``gt.margins``), but NOT the witness's own seg
    LOGITS. So the two witness-margin-dependent metrics -- the realized-margin p10/p50 and the Gibbs
    ring proxy -- are DELIBERATELY OMITTED here (they would need a second logits forward); the
    ``annulus_gt_margin`` block reports the GT-margin distribution within the annulus instead
    (``margin_source="gt"``). The full witness-margin/gibbs series lives in the offline
    ``tools/witness_annulus_convergence.py`` CLI. Two annulus definitions (fixed |margin|<band +
    bottom-k fraction) mirror ``witness_annulus_metrics.checkpoint_metrics``."""
    n = len(realized_list)
    cs = n if (chunk is None or chunk <= 0 or chunk >= n) else int(chunk)
    n_cls = _wam.N_CLASSES

    # ── Pass 1 (two-pass exact): global bottom-k threshold on |GT margin| over ALL pixels. Build ONLY
    #    the flat |margin| float32 plane (NOT the triple int64+float stack); ``annulus_mask_bottom_k``
    #    is thr=quantile(|gt_margin|, k) over the full pixel population -> exact np.quantile needs the
    #    materialized values. overwrite_input=True avoids np.quantile's internal copy (value identical;
    #    it only permits an in-place partition). We free the plane before the streaming pass. ──
    total_px = int(sum(int(np.asarray(m).size) for m in gt_margins_list))
    _abs_flat = np.empty(total_px, np.float32)
    _off = 0
    for _m in gt_margins_list:
        _a = np.abs(np.asarray(_m, np.float32)).ravel()
        _abs_flat[_off:_off + _a.size] = _a
        _off += _a.size
    thr_bk = float(np.quantile(_abs_flat, float(bottom_k), overwrite_input=True)) if total_px else 0.0
    del _abs_flat

    # ── Pass 2: stream chunks; accumulate integer region counts (exact) + collect ONLY the annulus
    #    GT-margin values needed for the p10/p50 percentiles (the "needed reservoir"). ──
    _defs = ("threshold", "bottom_k_def")
    ann_px = {d: 0 for d in _defs}
    ann_flip = {d: 0 for d in _defs}
    int_px = {d: 0 for d in _defs}
    int_flip = {d: 0 for d in _defs}
    cls_px = {d: [0] * n_cls for d in _defs}
    cls_flip = {d: [0] * n_cls for d in _defs}
    margin_vals: dict[str, list] = {d: [] for d in _defs}
    total_flip = 0

    for s in range(0, n, cs):
        e = min(s + cs, n)
        realized_c = np.stack([np.asarray(r).astype(np.int64) for r in realized_list[s:e]], axis=0)
        gt_arg_c = np.stack([np.asarray(g).astype(np.int64) for g in gt_lstars_list[s:e]], axis=0)
        gt_mar_c = np.stack([np.asarray(m).astype(np.float32) for m in gt_margins_list[s:e]], axis=0)
        flip_c = _wam.flip_map(realized_c, gt_arg_c)
        total_flip += int(flip_c.sum())
        for d in _defs:
            if d == "threshold":
                ann_c = _wam.annulus_mask_threshold(gt_mar_c, band)
            else:
                # SAME per-pixel rule as annulus_mask_bottom_k (|gt_margin| <= global thr), applied
                # per chunk with the ONE global threshold -> identical mask to the whole-array call.
                ann_c = np.abs(np.asarray(gt_mar_c, np.float32)) <= thr_bk
            int_c = ~ann_c
            ann_px[d] += int(ann_c.sum())
            ann_flip[d] += int((flip_c & ann_c).sum())
            int_px[d] += int(int_c.sum())
            int_flip[d] += int((flip_c & int_c).sum())
            for c in range(n_cls):
                region = ann_c & (gt_arg_c == c)
                cls_px[d][c] += int(region.sum())
                cls_flip[d][c] += int((flip_c & region).sum())
            # collect raw (not abs) GT-margin within the annulus, in pair order -> concat == full set.
            margin_vals[d].append(np.asarray(gt_mar_c, np.float32)[ann_c])

    def _frac(num: int, den: int) -> float:
        return (float(num) / den) if den else 0.0

    def _pct(vals_list: list) -> dict:
        vals = np.concatenate(vals_list) if vals_list else np.empty(0, np.float32)
        return {f"p{int(p)}": (float(np.percentile(vals, p)) if vals.size else float("nan"))
                for p in (10.0, 50.0)}

    def _block(d: str) -> dict:
        return {
            "annulus_area_frac": _frac(ann_px[d], total_px),
            "annulus_flip_frac": _frac(ann_flip[d], ann_px[d]),
            "interior_flip_frac": _frac(int_flip[d], int_px[d]),
            "annulus_flip_mass_share": _frac(ann_flip[d], total_flip),
            "annulus_gt_margin": _pct(margin_vals[d]),
            "per_class_annulus_flip_frac": {c: _frac(cls_flip[d][c], cls_px[d][c]) for c in range(n_cls)},
        }

    return {
        "overall_d_seg": _frac(total_flip, total_px),
        "total_px": total_px,
        "n_flips": int(total_flip),
        "n_pairs": int(n),
        "band": float(band),
        "bottom_k": float(bottom_k),
        "margin_source": "gt",
        "threshold": _block("threshold"),
        "bottom_k_def": _block("bottom_k_def"),
    }


def _annulus_convergence_row(metrics: dict, epoch: int, seg_form: "str | None") -> dict:
    """Wrap a ``_annulus_metrics_from_maps`` dict (or the ``{"error": ...}`` fail-safe payload) into
    the companion ``{"stage": "annulus_convergence", ...}`` JSON row emitted at verdict cadence."""
    return {
        "stage": "annulus_convergence",
        "epoch": int(epoch),
        "seg_form": (str(seg_form) if seg_form is not None else None),
        **metrics,
        "axis": "[macOS-numpy advisory] NON-PROMOTABLE",
        "note": "OBSERVABILITY-ONLY (never read into training); companion to {stage:verdict}. "
                "PARTIAL: realized-argmax annulus/interior flip split + per-class + GT-margin "
                "p10/p50 (no witness-logit margin/gibbs -- see tools/witness_annulus_convergence.py "
                "for the full offline series).",
    }


def _mlx_mem_gib(mx: Any) -> dict[str, float]:
    """MLX Metal allocator stats in GiB: active (LIVE arrays), cache (freed-but-pooled buffers),
    peak (high-water since last reset). The active/cache split is the #205 OOM diagnosis instrument:
    a small active + huge cache => the buffer POOL is the leak (fixed by ``mx.clear_cache()`` inside
    the accum loop), NOT the live working set. Pure read; NEVER read back into training."""
    out: dict[str, float] = {}
    for key, fn in (("active", "get_active_memory"), ("cache", "get_cache_memory"),
                    ("peak", "get_peak_memory")):
        try:
            out[key] = float(getattr(mx, fn)()) / (1024.0 ** 3)
        except Exception:
            out[key] = -1.0
    return out


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _git_provenance() -> dict[str, Any]:
    """Best-effort git provenance captured ONCE at launch (deterministic-reproducibility
    non-negotiable: provenance with every result = git hash + seed + config + upstream snapshot sha).

    NO-FAKE: when git is unavailable / not a repo, every field is ``"unknown"`` / ``False`` -- NEVER
    a fabricated sha. ``git_sha`` (repo HEAD) pins the trainer code AND the committed pinned
    ``upstream/`` snapshot (both live in the same tree); ``git_dirty`` flags an uncommitted working
    tree (a run from a dirty tree is NOT reproducible from the sha alone); ``upstream_tree_sha`` is the
    ``upstream/`` subtree object id (the frozen-scorer snapshot the verdict authority runs)."""
    import subprocess

    def _g(*a: str) -> str:
        try:
            r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=8)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    sha = _g("rev-parse", "HEAD") or "unknown"
    dirty = bool(_g("status", "--porcelain"))
    # upstream/ is an UNTRACKED pinned snapshot (not in git HEAD), so the frozen-scorer authority is
    # pinned by the CANONICAL content hash (tac.contest_compliance) -- the same upstream_snapshot_sha256
    # every ledger/anchor carries -- NOT a git tree sha. Best-effort; "unknown" when absent (NO-FAKE).
    try:
        from tac.contest_compliance import compute_upstream_snapshot_sha256
        upstream_sha = compute_upstream_snapshot_sha256(REPO) or "unknown"
    except Exception:
        upstream_sha = "unknown"
    return {"git_sha": sha, "git_dirty": dirty, "upstream_snapshot_sha256": upstream_sha}


# ---------------------------------------------------------------------------
# INTERMEDIATE CHECKPOINT + RESUME (FEED-dz, additive, default-off). The trainer historically saved
# the EMA-shadow npz ONLY at loop-end -> a multi-day n600 run is non-resumable (crash = total loss)
# + no early byte-close. These pure-numpy (MLX-free, unit-testable) helpers let the run loop write
# a deploy EMA checkpoint (the byte-close ONE-CODEPATH consumes it) AND a separate resume-state
# sidecar (live weights + EMA shadow + optimizer + epoch) every --ckpt-every epochs, atomically.
#
# DESIGN (NO-FAKE / EMA non-negotiable / byte-close clean):
#   * ``levelset_witness_ema_mlx.npz`` = the EMA SHADOW (deploy weights, NOT live) + ``__cfg_*`` /
#     ``__bank_*`` / ``__render_hw`` scalars. EXACTLY what tools/levelset_byte_close_and_eval.py
#     reads (params = unprefixed keys; cfg = ``__``-prefixed, read selectively). Adding new ``__cfg_*``
#     provenance keys is harmless (byte-close ``.get(...)``s the ones it knows + ignores the rest).
#   * ``levelset_resume_state.npz`` = SEPARATE sidecar (so the EMA npz stays byte-close-clean). Live
#     model params (``liveP__*``), EMA shadow (``emaP__*``), optimizer state (``optP__*``, best-effort),
#     + ``__resume_epoch``. Self-orient dir-feats are NOT stored (they are O(GBs) at n600 and are
#     deterministically regenerable from the EMA argmax fixed-point at resume -> recompute, no bloat).
#   * Atomic write: tmp + os.replace (no partial/corrupt npz if the process dies mid-write).
# ---------------------------------------------------------------------------
_RESUME_LIVE_PREFIX = "liveP__"
_RESUME_EMA_PREFIX = "emaP__"
_RESUME_OPT_PREFIX = "optP__"


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    """Atomic ``np.savez`` (tmp + os.replace) per the durable-state discipline. Refuses /tmp.

    np.savez given a *file object* writes the zip directly (no implicit ``.npz`` suffix append), so
    the temp path is replaced onto the final path atomically on the same filesystem.
    """
    path = Path(path)
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> Path:
    """Atomic JSON write (tmp + os.replace) per the durable-state discipline. Refuses /tmp.

    Used for the tiny best-checkpoint POINTER (``levelset_best.json``) so a harvester / early-stop
    reads the run's best realized-d_seg artifact WITHOUT re-deriving it from the log."""
    path = Path(path)
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def _is_new_best(d_seg: float, prev_best: float) -> bool:
    """NEW-best promotion rule (NO-FAKE): a FINITE, STRICTLY-better realized d_seg only. NaN/inf
    never win; a tie keeps the EARLIER best (reproducible). The 1e-12 guard avoids float-noise
    churn rewriting the best ckpt for sub-ULP "improvements". Module-level + pure -> unit-tested."""
    return bool(np.isfinite(d_seg)) and (float(d_seg) < float(prev_best) - 1e-12)


def _build_ema_checkpoint_arrays(
    shadow_np: dict[str, np.ndarray], *, args: Any, softmax_temp: float,
    render_h: int, render_w: int, epoch: int, in_feat: int,
    hosc_beta: float | None = None, provenance: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """The deploy (byte-close) npz contents: EMA SHADOW params + cfg scalars. MLX-free.

    Reproduces EVERY key the loop-end save historically wrote (so the byte-close path is unchanged)
    and ADDS provenance keys (``__epoch`` + the self-orient/curriculum/w_pose scalars the trainer
    previously did NOT persist -- the gap flagged in tools/levelset_byte_close_and_eval.py)."""
    flat: dict[str, np.ndarray] = {k: np.asarray(v, np.float32) for k, v in shadow_np.items()}
    # ---- EXISTING keys (loop-end save parity; do NOT change names/encodings) ----
    flat["__cfg_n_hidden"] = np.asarray(args.n_hidden)
    flat["__cfg_hidden_dim"] = np.asarray(args.hidden_dim)
    flat["__cfg_softmax_temp"] = np.asarray(float(softmax_temp))
    flat["__cfg_activation"] = np.asarray(args.activation)
    flat["__cfg_chroma"] = np.asarray(int(bool(args.chroma)))
    flat["__cfg_wire_w0"] = np.asarray(args.wire_w0)
    flat["__cfg_wire_s0"] = np.asarray(args.wire_s0)
    # (FEED-fb) persist the CURRENT (possibly annealed) beta so the byte-close/inflate deploy forward
    # uses the SAME activation sharpness the EMA shadow was trained at (NO-FAKE). When the caller does
    # not thread it (hosc_beta is None) OR anneal is off, this == args.hosc_beta => byte-identical cfg.
    flat["__cfg_hosc_beta"] = np.asarray(args.hosc_beta if hosc_beta is None else float(hosc_beta))
    flat["__cfg_hosc_omega"] = np.asarray(args.hosc_omega)
    flat["__bank_n_scales"] = np.asarray(args.bank_n_scales)
    flat["__bank_n_orient0"] = np.asarray(args.bank_n_orient0)
    flat["__bank_f0"] = np.asarray(args.bank_f0)
    flat["__bank_base"] = np.asarray(args.bank_base)
    flat["__bank_n_iso"] = np.asarray(args.bank_n_iso)
    flat["__render_hw"] = np.asarray([render_h, render_w])
    flat["__cfg_max_bank_freq"] = np.asarray(-1.0 if args.max_bank_freq is None else float(args.max_bank_freq))
    flat["__cfg_lane_edge_weight"] = np.asarray(float(args.lane_edge_weight))
    flat["__cfg_lane_edge_class"] = np.asarray(int(args.lane_edge_class))
    # ---- NEW provenance (additive; closes the self-orient/curriculum trainer-persist gap) ----
    flat["__epoch"] = np.asarray(int(epoch))
    flat["__cfg_in_feat"] = np.asarray(int(in_feat))
    flat["__cfg_self_orient"] = np.asarray(int(bool(args.self_orient)))
    flat["__cfg_n_dir_freqs"] = np.asarray(int(args.n_dir_freqs))
    flat["__cfg_freq_across"] = np.asarray(float(args.freq_across))
    flat["__cfg_freq_along"] = np.asarray(float(args.freq_along))
    flat["__cfg_reorient_every"] = np.asarray(int(args.reorient_every))
    flat["__cfg_w_pose"] = np.asarray(float(args.w_pose))
    flat["__cfg_curriculum"] = np.asarray(int(bool(args.curriculum)))
    flat["__cfg_tau_softplus_start_epoch"] = np.asarray(int(args.tau_softplus_start_epoch))
    flat["__cfg_l7_start_epoch"] = np.asarray(int(args.l7_start_epoch))
    # ---- PROVENANCE (deterministic-reproducibility: git sha + upstream snapshot sha in EVERY
    # per-stage byte-close artifact so a shipped checkpoint traces to the exact code + frozen scorer.
    # Additive + byte-close-ignored (.get()s only keys it knows); default "unknown" = NO-FAKE, never
    # a fabricated sha). ----
    _prov = provenance or {}
    flat["__cfg_git_sha"] = np.asarray(str(_prov.get("git_sha", "unknown")))
    flat["__cfg_git_dirty"] = np.asarray(int(bool(_prov.get("git_dirty", False))))
    flat["__cfg_upstream_snapshot_sha256"] = np.asarray(str(_prov.get("upstream_snapshot_sha256", "unknown")))
    # ---- #224 AA-SDF observation-map render cfg (additive provenance; the exact-eval decode
    # (#202) reconstructs the SAME AA mode deterministically -- NO extra archive bytes: the IPE
    # attenuation is a function of (B, render_hw, footprint) all already in the ckpt, and the
    # supersample grid is deterministic at decode). DEFAULT none/1/1.0 => byte-identical cfg. ----
    flat["__cfg_render_aa"] = np.asarray(str(getattr(args, "render_aa", "none")))
    flat["__cfg_aa_supersample"] = np.asarray(int(getattr(args, "aa_supersample", 1)))
    flat["__cfg_aa_ipe_footprint"] = np.asarray(float(getattr(args, "aa_ipe_footprint", 1.0)))
    return flat


def _build_resume_state_arrays(
    live_np: dict[str, np.ndarray], ema_np: dict[str, np.ndarray],
    opt_np: dict[str, np.ndarray] | None, *, args: Any, epoch: int, in_feat: int,
    recent_losses: "list[float] | None" = None, provenance: dict[str, Any] | None = None,
    evt_curriculum_state: "dict | None" = None,
    closed_loop_state: "dict | None" = None,
) -> dict[str, np.ndarray]:
    """The resume-state sidecar contents (NOT byte-close-read): prefixed live / EMA / optimizer
    tensors + epoch + light cfg provenance. MLX-free (caller converts mx->np)."""
    out: dict[str, np.ndarray] = {}
    for k, v in live_np.items():
        out[_RESUME_LIVE_PREFIX + k] = np.asarray(v, np.float32)
    for k, v in ema_np.items():
        out[_RESUME_EMA_PREFIX + k] = np.asarray(v, np.float32)
    has_opt = bool(opt_np)
    if has_opt:
        for k, v in opt_np.items():
            out[_RESUME_OPT_PREFIX + k] = np.asarray(v)
    out["__resume_epoch"] = np.asarray(int(epoch))
    out["__resume_has_opt"] = np.asarray(int(has_opt))
    out["__cfg_n_hidden"] = np.asarray(args.n_hidden)
    out["__cfg_hidden_dim"] = np.asarray(args.hidden_dim)
    out["__cfg_mod_dim"] = np.asarray(args.mod_dim)
    out["__cfg_self_orient"] = np.asarray(int(bool(args.self_orient)))
    out["__cfg_in_feat"] = np.asarray(int(in_feat))
    out["__cfg_w_pose"] = np.asarray(float(args.w_pose))
    # (F2 fix) #224 render-side LEVER cfg: persist the levers whose engagement CHANGES the loss /
    # render target mid-run so a --resume-from can FAIL-CLOSED (via _resume_lever_divergences) when the
    # resume command silently drops or diverges a lever the run was trained with (a deterministic-repro
    # violation the film-arch guard does NOT cover -- these are loss/render-only, they add no param
    # KEYS so the missing-param guard cannot see them). ZERO archive bytes (resume sidecar is not
    # byte-closed). hosc_beta_end None -> -1.0 sentinel (matches the current-arg encoding in the guard).
    out["__cfg_lane_render_band"] = np.asarray(int(bool(getattr(args, "lane_render_band", False))))
    out["__cfg_lane_band_start_epoch"] = np.asarray(int(getattr(args, "lane_band_start_epoch", 300)))
    # #287 dash comb: render-only lever (changes the band coverage, no param KEYS) -> persist for F2.
    out["__cfg_lane_band_dash_comb"] = np.asarray(int(bool(getattr(args, "lane_band_dash_comb", False))))
    out["__cfg_persistence_loss_weight"] = np.asarray(float(getattr(args, "persistence_loss_weight", 0.0)))
    out["__cfg_amplify_weight"] = np.asarray(float(getattr(args, "amplify_weight", 0.0)))
    # BUILD #300: the seed-absorption levers are loss/render-only (trajectory-affecting, no param KEYS) ->
    # record them so a --resume-from that silently drops/changes them fails closed (deterministic-repro).
    out["__cfg_witness_alone_island_loss"] = np.asarray(int(bool(getattr(args, "witness_alone_island_loss", False))))
    out["__cfg_seg_focal_gamma"] = np.asarray(float(getattr(args, "seg_focal_gamma", 0.0)))
    out["__cfg_boundary_distance_weight"] = np.asarray(float(getattr(args, "boundary_distance_weight", 0.0)))
    # (#218) logit-adjustment is loss-only + trajectory-affecting (no param keys) — same class as
    # focal/boundary-distance above: persist so a --resume-from that silently drops/changes it
    # fails closed via _resume_lever_divergences (deterministic-repro).
    out["__cfg_logit_adjust_loss_tau"] = np.asarray(float(getattr(args, "logit_adjust_loss_tau", 0.0)))
    # (C11 confound fix) persist the stiff eikonal-family weights so a --resume-from that ADDS/raises
    # them onto an opt state trained WITHOUT them is DETECTABLE (the resume-drift loud row + LR
    # re-warmup routing). Loss/render-only (no param keys), so the arch guard cannot see them.
    out["__cfg_eikonal_weight"] = np.asarray(float(getattr(args, "eikonal_weight", 0.0)))
    out["__cfg_eikonal_viscosity"] = np.asarray(float(getattr(args, "eikonal_viscosity", 0.0)))
    out["__cfg_seed_anneal_epochs"] = np.asarray(int(getattr(args, "seed_anneal_epochs", 0)))
    out["__cfg_seed_anneal_shape"] = np.asarray(str(getattr(args, "seed_anneal_shape", "linear")))
    out["__cfg_render_aa"] = np.asarray(str(getattr(args, "render_aa", "none")))
    _hbe = getattr(args, "hosc_beta_end", None)
    out["__cfg_hosc_beta_end"] = np.asarray(-1.0 if _hbe is None else float(_hbe))
    # (review R2a-MED-1) ARCH flags that change the param KEYS / training geometry: persist them in the
    # resume sidecar so a crash-resume from the ckpt dir ALONE can fail-closed if the resume command
    # omits the flag the run was trained with (the silent-param-drop risk -- MLX model.update only
    # touches EXISTING params, so a model rebuilt without film_pl/concat_pl would silently DROP the
    # trained per-layer FiLM params). film_per_layer/film_concat_code add params (film_pl./concat_pl.);
    # film_stiefel constrains the existing film.weight (training-dynamics, no new keys). The resume
    # sidecar is NOT byte-closed -> these provenance scalars cost ZERO archive bytes. Per the
    # resumability + deterministic-reproducibility non-negotiables.
    out["__cfg_film_per_layer"] = np.asarray(int(bool(getattr(args, "film_per_layer", False))))
    out["__cfg_film_concat_code"] = np.asarray(int(bool(getattr(args, "film_concat_code", False))))
    out["__cfg_film_stiefel"] = np.asarray(int(bool(getattr(args, "film_stiefel", False))))
    # SPIKE-GUARD running-median window (the last <=50 batch losses). It GATES step-skipping
    # (loss > spike_factor * median => the optimizer.update is skipped), so it is part of the
    # weight trajectory: a resume with an EMPTY window (median None => never skips) would diverge
    # from a continuous run that WOULD have skipped. Persist it so --resume-from is bit-faithful even
    # across a spike. Empty list => a 0-length array (default-safe; a pre-fix ckpt lacks the key =>
    # the loop's fresh [] is used, i.e. the prior behavior). Per the deterministic-repro non-negotiable.
    out["__recent_losses"] = np.asarray(list(recent_losses or []), np.float64)
    # (#292 build-2) EVENT-TRIGGERED CURRICULUM controller state: persist the resolved stage boundaries
    # + within-stage ep_loss history so a --resume-from of an event-triggered run reproduces the SAME
    # fired transition epochs (deterministic-reproducibility non-negotiable). Default None (event-
    # triggered OFF, incl. #205) => ZERO new keys written => the sidecar is byte-identical to the pre-
    # #292-build-2 path. A pre-feature sidecar lacks these keys => the resume loop falls back to the
    # SAFE cap-resolution (past-cap boundaries -> hardcoded caps) so the stage is never mis-assigned.
    if evt_curriculum_state is not None:
        _bt = evt_curriculum_state.get("tau")
        _bl = evt_curriculum_state.get("l7")
        out["__evt_boundary_tau"] = np.asarray(-1 if _bt is None else int(_bt))
        out["__evt_boundary_l7"] = np.asarray(-1 if _bl is None else int(_bl))
        out["__evt_stage_start"] = np.asarray(int(evt_curriculum_state.get("stage_start", 1)))
        out["__evt_stage_losses"] = np.asarray(
            list(evt_curriculum_state.get("losses", []) or []), np.float64)
        # (#302) MEASURED nucleus-readiness state (nucleus half of the CE->tau trigger). Default True
        # (nucleus guard OFF => never blocks). Persisted so an ON-resume reproduces the SAME fired
        # epoch bit-faithfully (the guard reads the last verdict's nucleus_ready).
        out["__evt_nucleus_ready"] = np.asarray(
            1 if bool(evt_curriculum_state.get("nucleus_ready", True)) else 0)
    # (#292 build-3) CLOSED-LOOP LEVER CONTROL state: persist bump/stop state + the captured verdict
    # history so an ON-run --resume-from is bit-faithful (same classifications => same bumps + stop).
    # Default None (closed-loop OFF, incl. #205) => ZERO new keys => sidecar byte-identical.
    if closed_loop_state is not None:
        out.update(_cl_state_arrays(closed_loop_state, closed_loop_state.get("verdicts", [])))
    # ---- PROVENANCE (git sha + upstream snapshot sha; cost ZERO archive bytes -- the resume sidecar
    # is not byte-closed; makes a --resume-from traceable to the exact code + frozen scorer). ----
    _prov = provenance or {}
    out["__cfg_git_sha"] = np.asarray(str(_prov.get("git_sha", "unknown")))
    out["__cfg_git_dirty"] = np.asarray(int(bool(_prov.get("git_dirty", False))))
    out["__cfg_upstream_snapshot_sha256"] = np.asarray(str(_prov.get("upstream_snapshot_sha256", "unknown")))
    return out


def _load_resume_state(npz_path: Path) -> dict[str, Any]:
    """Parse a resume sidecar OR (fallback) a plain EMA deploy npz. Returns live/ema/opt dicts +
    epoch + has_opt + cfg. NO-FAKE: a missing/garbage file raises. MLX-free."""
    z = np.load(Path(npz_path), allow_pickle=False)
    live: dict[str, np.ndarray] = {}
    ema: dict[str, np.ndarray] = {}
    opt: dict[str, np.ndarray] = {}
    cfg: dict[str, Any] = {}
    for k in z.files:
        if k.startswith(_RESUME_LIVE_PREFIX):
            live[k[len(_RESUME_LIVE_PREFIX):]] = np.asarray(z[k], np.float32)
        elif k.startswith(_RESUME_EMA_PREFIX):
            ema[k[len(_RESUME_EMA_PREFIX):]] = np.asarray(z[k], np.float32)
        elif k.startswith(_RESUME_OPT_PREFIX):
            opt[k[len(_RESUME_OPT_PREFIX):]] = np.asarray(z[k])
        elif k.startswith("__"):
            a = z[k]
            cfg[k] = a.item() if a.size == 1 else a.tolist()
        else:
            # plain EMA deploy npz: unprefixed keys are the EMA-shadow params. Use them as the
            # live-weight fallback (resume from the deploy checkpoint when no sidecar exists).
            live.setdefault(k, np.asarray(z[k], np.float32))
    epoch = int(cfg.get("__resume_epoch", cfg.get("__epoch", 0)))
    return {
        "live": live, "ema": ema, "opt": opt,
        "epoch": epoch, "has_opt": bool(int(cfg.get("__resume_has_opt", 0))), "cfg": cfg,
    }


def _resolve_weights_only_warm_start(
    rs: dict[str, Any], *, warm_start_weights_only: bool, warm_start_epoch: int,
    ckpt_start_epoch: int,
) -> dict[str, Any]:
    """(DE#3 clean warm-start) Decide the weights-only warm-start effects and MUTATE ``rs`` to
    discard the optimizer moments when the flag is set. Pure / MLX-free -> unit-tested.

    Effects when ``warm_start_weights_only`` is True (the poisoned-resume-trap cure): take ONLY the
    trained WEIGHTS (``rs['live']`` / ``rs['ema']`` are untouched), DISCARD ``rs['opt']`` + set
    ``rs['has_opt']=False`` so the caller's optimizer-state restore is SKIPPED (=> fresh AdamW), and
    return the start-epoch override (``warm_start_epoch`` if >=0, else keep the caller's
    ``ckpt_start_epoch``). The spike-guard clear + lever-drift auto-allow are gated on the same flag
    at their own call sites. DEFAULT (flag OFF) => ``rs`` untouched, override None => byte-identical.

    Returns {'discarded_opt': bool, 'clear_spike_guard': bool, 'allow_lever_drift': bool,
             'start_epoch': int, 'ckpt_had_opt': bool}."""
    ckpt_had_opt = bool(rs.get("has_opt", False)) and bool(rs.get("opt"))
    if not warm_start_weights_only:
        return {"discarded_opt": False, "clear_spike_guard": False, "allow_lever_drift": False,
                "start_epoch": int(ckpt_start_epoch), "ckpt_had_opt": ckpt_had_opt}
    rs["opt"] = {}
    rs["has_opt"] = False
    _ep = int(warm_start_epoch) if int(warm_start_epoch) >= 0 else int(ckpt_start_epoch)
    return {"discarded_opt": True, "clear_spike_guard": True, "allow_lever_drift": True,
            "start_epoch": _ep, "ckpt_had_opt": ckpt_had_opt}


def _resume_lever_divergences(resume_cfg: dict[str, Any], args: Any) -> list[str]:
    """(F2) List render-side LEVER cfg keys that DIVERGE between the resume sidecar (what the run was
    trained with) and the current argv (what this resume would run). A non-empty list means a
    ``--resume-from`` would SILENTLY change / drop a lever = a deterministic-reproducibility violation
    the film-arch guard cannot see (these loss/render-only levers add NO param keys). Only keys PRESENT
    in the sidecar are checked, so a pre-F2 sidecar (which lacks them) yields NO spurious divergence.
    Pure / MLX-free -> unit-tested. ``lane_band_start_epoch`` is only flagged when the band is engaged
    in EITHER config (a start-epoch change is inert while the band is OFF in both)."""
    div: list[str] = []
    _hbe = getattr(args, "hosc_beta_end", None)
    cur_hbe = -1.0 if _hbe is None else float(_hbe)
    cur_band = int(bool(getattr(args, "lane_render_band", False)))
    # (key, current value, is_float) — non-float compared as string (int/bool/str all normalize).
    checks: list[tuple[str, object, bool]] = [
        ("__cfg_mod_dim", int(getattr(args, "mod_dim", 0)), False),
        ("__cfg_lane_render_band", cur_band, False),
        # #287 dash comb (render-only, trajectory-affecting when the band is on, no param keys).
        ("__cfg_lane_band_dash_comb", int(bool(getattr(args, "lane_band_dash_comb", False))), False),
        ("__cfg_persistence_loss_weight", float(getattr(args, "persistence_loss_weight", 0.0)), True),
        ("__cfg_amplify_weight", float(getattr(args, "amplify_weight", 0.0)), True),
        # BUILD #300 seed-absorption levers (material, trajectory-affecting).
        ("__cfg_witness_alone_island_loss", int(bool(getattr(args, "witness_alone_island_loss", False))), False),
        ("__cfg_seed_anneal_epochs", int(getattr(args, "seed_anneal_epochs", 0)), False),
        ("__cfg_render_aa", str(getattr(args, "render_aa", "none")), False),
        ("__cfg_hosc_beta_end", cur_hbe, True),
        # focal-gamma + boundary-distance seg-loss levers (council levelset-loss-geometry symposium
        # 2026-07-05; loss-only, trajectory-affecting, no param keys — same class as the #300 levers).
        ("__cfg_seg_focal_gamma", float(getattr(args, "seg_focal_gamma", 0.0)), True),
        ("__cfg_boundary_distance_weight", float(getattr(args, "boundary_distance_weight", 0.0)), True),
        # (#218) logit-adjustment per-class offset (loss-only, trajectory-affecting, no param keys).
        ("__cfg_logit_adjust_loss_tau", float(getattr(args, "logit_adjust_loss_tau", 0.0)), True),
        # (review MED-1) film_stiefel constrains the EXISTING film.weight (training-dynamics only,
        # NO new param keys -> the film-arch/param-key guard cannot see it; the sidecar persists
        # __cfg_film_stiefel [R2a-MED-1 note there] precisely so THIS guard can fail-closed on a
        # resume that silently drops/adds the Stiefel constraint).
        ("__cfg_film_stiefel", int(bool(getattr(args, "film_stiefel", False))), False),
    ]
    for key, cur, is_float in checks:
        if key not in resume_cfg:
            continue
        ckpt = resume_cfg[key]
        if is_float:
            try:
                diverged = abs(float(ckpt) - float(cur)) > 1e-6
            except (TypeError, ValueError):
                diverged = str(ckpt) != str(cur)
        else:
            diverged = str(ckpt) != str(cur)
        if diverged:
            div.append(f"{key[len('__cfg_'):]}: ckpt={ckpt!r} != resume-argv={cur!r}")
    # lane_band_start_epoch: inert while the band is OFF in BOTH -> only flag when engaged in either.
    if "__cfg_lane_band_start_epoch" in resume_cfg:
        ckpt_band = int(resume_cfg.get("__cfg_lane_render_band", cur_band) or 0)
        if (ckpt_band or cur_band):
            ckpt_se = int(resume_cfg["__cfg_lane_band_start_epoch"])
            cur_se = int(getattr(args, "lane_band_start_epoch", 300))
            if ckpt_se != cur_se:
                div.append(f"lane_band_start_epoch: ckpt={ckpt_se} != resume-argv={cur_se}")
    # BUILD #300 seed_anneal_shape: inert while the anneal is OFF (epochs 0) in BOTH -> only flag when
    # engaged in either (mirrors lane_band_start_epoch's engaged-gate).
    if "__cfg_seed_anneal_shape" in resume_cfg:
        ckpt_ep = int(resume_cfg.get("__cfg_seed_anneal_epochs", 0) or 0)
        cur_ep = int(getattr(args, "seed_anneal_epochs", 0))
        if (ckpt_ep or cur_ep):
            ckpt_shape = str(resume_cfg["__cfg_seed_anneal_shape"])
            cur_shape = str(getattr(args, "seed_anneal_shape", "linear"))
            if ckpt_shape != cur_shape:
                div.append(f"seed_anneal_shape: ckpt={ckpt_shape!r} != resume-argv={cur_shape!r}")
    return div


def _validate_aa_compose_compat(
    aa_on: bool, band_active: bool, residual_mode: bool, seed_on: bool,
) -> None:
    """(#224 / review MED-3 -> #220 UNBLOCK 2026-07-07) Compatibility gate: --render-aa supersample
    vs the BASE-grid compose levers. RESOLVED: ``tac.boundary_math.aa_sdf_observation_render`` now
    invokes ``compose_fn`` AFTER ``box_downsample_mlx`` (i.e. at the BASE (H,W) grid), so the
    lane-band coverage, the residual-bulk composition mask, and the island-seed residual — all
    base-grid (H,W) tensors — compose with AA supersample BY CONSTRUCTION (footprint integration
    first, then the base-grid composers, then R; ss=1 remains byte-identical because the identity
    downsample makes compose-before == compose-after bit-for-bit). NO genuinely-incompatible combo
    remains among the three tracked composers, so this guard currently accepts every combination;
    the function + signature + call site are KEPT as the fail-closed home for any FUTURE composer
    that genuinely requires the fine (ss*grid) surface. Pure / MLX-free -> unit-tested.
    NOTE the ORTHOGONAL --self-orient x supersample fine-dir-feats guard (memory/wall-clock, the
    --aa-self-orient-fine-mode refuse default) is a DIFFERENT gate and is unchanged."""
    return None


# ---------------------------------------------------------------------------
# (#310 BUILD, FEED-07b lever #2 sister) FINER/FINER++ variable-periodic FIRST-LAYER bias init.
# ---------------------------------------------------------------------------
# Dedicated-RNG salt: the FINER draw NEVER touches the shared np.random / mx.random streams, so
# --finer-bias-init OFF draws NOTHING (byte-identical) and ON perturbs NO other seeded draw.
_FINER_RNG_SALT = 20260707


def _finer_bias_init_values(seed: int, k: float, n: int) -> np.ndarray:
    """(#310) FINER/FINER++ variable-periodic FIRST-LAYER bias init values (pure numpy).

    ``bias ~ U(-k, k)`` over a WIDE range so each first-layer neuron selects its OWN effective
    frequency/phase of the periodic activation (FINER arXiv 2312.02434 / FINER++ arXiv 2407.19434
    — the published fix for the MEASURED fixed-beta hosc saturation-death, DAG FEED 2026-06-25a +
    FEED-ly: with all first-layer biases ~0 every neuron sits at the SAME point of tanh(beta*sin)
    and saturates TOGETHER as beta rises; the wide bias spreads the ensemble across the period so
    some neurons always live on a high-gradient stretch). DEDICATED
    ``np.random.default_rng(seed + _FINER_RNG_SALT)`` stream: NEVER the shared ``np.random`` /
    ``mx.random`` streams (byte-identity discipline — the OFF path draws nothing; the ON path
    shifts no other seeded draw). Deterministic in (seed, k, n). Fail-closed on k<=0 / n<=0."""
    if not (float(k) > 0.0):
        raise ValueError(f"--finer-bias-k must be > 0, got {k!r}")
    if int(n) <= 0:
        raise ValueError(f"finer bias init needs n > 0 neurons, got {n!r}")
    rng = np.random.default_rng(int(seed) + _FINER_RNG_SALT)
    return rng.uniform(-float(k), float(k), size=int(n)).astype(np.float32)


# ---------------------------------------------------------------------------
# (#218 BUILD, FEED-07b lever #3) class-prior LOGIT ADJUSTMENT (Menon et al. 2021,
# arXiv 2007.07314) — the textbook ZERO-BYTE rare-class cure, at the TRAINING-LOSS surface only.
# ---------------------------------------------------------------------------
def _logit_adjust_offsets_np(
    lstars: "list[np.ndarray]", tau: float, n_classes: int = 5,
) -> "tuple[np.ndarray, np.ndarray]":
    """(#218) Per-class logit-adjustment offsets from the cached GT argmax class areas.

    Returns ``(offsets, priors)``: ``priors_c`` = the GT class-area fraction over ALL given L*
    maps (floored at the canonical equation's prior floor, so an absent class never yields
    log(0) = -inf), ``offsets_c = tau * log(priors_c)`` per the registered law
    ``logit_adjustment_class_prior_law_v1`` (``tac.canonical_equations.
    logit_adjustment_class_prior_20260707:logit_adjust_offsets`` — the equations leg IS the
    callable this delegates to; measured n600 priors anchor ~[0.232, 0.0059, 0.495, 0.0124,
    0.254]). Pure numpy -> unit-tested."""
    from tac.canonical_equations.logit_adjustment_class_prior_20260707 import (
        logit_adjust_offsets,
    )

    counts = np.zeros(int(n_classes), np.float64)
    for ls in lstars:
        counts += np.bincount(
            np.asarray(ls, np.int64).ravel(), minlength=int(n_classes))[: int(n_classes)]
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("--logit-adjust-loss-tau: empty GT L* maps (no pixels) — cannot derive priors")
    priors = counts / total
    return logit_adjust_offsets(priors, float(tau)), priors.astype(np.float64)


def _validate_logit_adjust_compat(tau: float, micro_batch_pairs: int) -> None:
    """(#218) Fail-closed: --logit-adjust-loss-tau is wired into the SERIAL base_loss adapter only —
    NOT into the --micro-batch-pairs>1 batched twin (``tac.boundary_math.
    levelset_micro_batch_loss`` receives the UNWRAPPED adapter). Refuse the combination with an
    actionable message instead of silently training the batched arm WITHOUT the adjustment (the
    same not-yet-routed class as --seg-spike-reweight / --margin-saliency-reachability). Pure /
    MLX-free -> unit-tested. tau == 0.0 (OFF) is always compatible."""
    if float(tau) != 0.0 and int(micro_batch_pairs) > 1:
        raise ValueError(
            "--logit-adjust-loss-tau is not wired into the --micro-batch-pairs>1 batched twin "
            "(the batched loss reads the UNWRAPPED scorer adapter); run with "
            "--micro-batch-pairs 1 (the serial path) or leave --logit-adjust-loss-tau 0.")


class _LogitAdjustSegAdapter:
    """(#218) Class-prior logit-adjustment wrapper around the frozen MLX scorer adapter, applied
    ONLY on the training-LOSS surface (the ``adapter`` closed over by ``make_loss_fn``).

    ``segnet(frames)`` returns ``inner.segnet(frames) + offset`` with ``offset_c =
    tau * log(prior_c)`` (a (K,) constant broadcast over (1,H,W,K)); inside the CE form this IS
    the Menon et al. logit-adjusted CE (rare classes get strongly negative log-priors, so
    under-predicting them costs more), and inside the margin forms (tau_softplus / l7 /
    margin_hinge) it is the additive per-class margin generalization (LDAM-style) of the same
    prior. The focal reweight (--seg-focal-gamma), when active, reads the SAME adjusted logits —
    the standard logit-adjusted-focal composition. ``posenet`` passes through UNTOUCHED (the pose
    term is class-free). BYTE-IDENTITY BOUNDARY (binding): this wrapper exists ONLY inside
    base_loss — the deployed/rendered argmax path (the verdict CPU-torch SegNet, the byte-close
    decode, inflate) reads RAW logits and is UNCHANGED; the witness WEIGHTS absorb the pressure
    through training, the shipped forward stays the plain argmax."""

    def __init__(self, inner: Any, offset: Any) -> None:
        self._inner = inner
        self._offset = offset            # (K,) mx.array = tau * log(prior)
        self.posenet = inner.posenet     # pass-through (pose term unadjusted)

    def segnet(self, frames_nhwc: Any):
        return self._inner.segnet(frames_nhwc) + self._offset


def boundary_distance_band_map(lstar_hw: np.ndarray, band_px: float = 2.0) -> np.ndarray:
    """(--boundary-distance-weight; council levelset-loss-geometry symposium 2026-07-05) The
    theta-INDEPENDENT per-pair TARGET boundary-band weight (H,W) f32, computed ONCE per pair from
    the cached GT argmax (Kervadec-style distance-transform discipline; cacheable).

    Boundary set = pixels straddling any GT inter-class edge (label differs from the RIGHT or DOWN
    neighbor; BOTH straddle pixels marked — the same edge convention as the LEVER-4b straddles).
    Weight = relu(1 - D/band_px) where D = Euclidean distance (px) to the nearest boundary pixel:
    1.0 ON the GT boundary, linear ramp to 0 at band_px (default 2.0 px = the measured 1-2 px flip
    band, #149; matches ``_boundary_band``'s radius-2 default). Pure numpy/scipy — unit-tested."""
    from scipy.ndimage import distance_transform_edt

    ls = np.asarray(lstar_hw)
    bnd = np.zeros(ls.shape, bool)
    dif_r = ls[:, :-1] != ls[:, 1:]
    bnd[:, :-1] |= dif_r
    bnd[:, 1:] |= dif_r
    dif_d = ls[:-1, :] != ls[1:, :]
    bnd[:-1, :] |= dif_d
    bnd[1:, :] |= dif_d
    if not bnd.any():
        return np.zeros(ls.shape, np.float32)  # degenerate single-class frame: no target boundary
    dist = distance_transform_edt(~bnd)
    return np.clip(1.0 - dist / float(band_px), 0.0, 1.0).astype(np.float32)


def boundary_distance_term_mlx(phi_flat, lstar_oh, band_map, render_h: int, render_w: int):
    """(--boundary-distance-weight) SDF-NATIVE boundary-placement loss term (scalar mx).

    At GT-boundary-band pixels the witness partition boundary should PASS THROUGH the target
    boundary, i.e. the SDF decision gap ``phi_[GT] - max_{k != GT} phi_k`` should be ZERO there
    (a partition boundary is a tie of the top two fields). The term is the band-weighted mean of
    |gap| — Mallat's "move the contour" degree of freedom, scored on the SDF head DIRECTLY (the
    DOF the witness owns), not through SegNet. With the eikonal |grad phi|=1 constraint the gap
    is calibrated in ~pixel units, so this is the |phi_pred|-on-the-band Kervadec form stated in
    the symposium memo, generalized to the K-field partition. ``phi_flat`` (P_px, K) from
    ``model.sdf``; ``lstar_oh`` (1,H,W,K); ``band_map`` (1,H,W) theta-independent constant."""
    import mlx.core as mx

    phi = mx.reshape(phi_flat, (1, render_h, render_w, -1))
    gt_phi = mx.sum(phi * lstar_oh, axis=-1)                 # (1,H,W) GT-class field
    run_phi = mx.max(phi + lstar_oh * (-1e9), axis=-1)       # (1,H,W) top competitor field
    gap = mx.abs(gt_phi - run_phi)
    return mx.sum(gap * band_map) / (mx.sum(band_map) + 1e-6)


def _load_decoder_params(npz_path: Path) -> dict[str, np.ndarray]:
    """Load ONLY the SHARED-DECODER params from a level-set EMA/deploy npz (FEED-eo amortization).

    Returns the decoder tensors (in_proj/film/hidden.*/out_sdf/out_tex {weight,bias} + palette) but
    EXCLUDES ``code`` (the per-(pair,frame) latents, which the freeze-decoder-fit-codes mode RE-FITS
    for a different pair count) and the free deterministic bank ``B``/``*_B`` (rule 118) and the
    ``__``-prefixed cfg scalars. NO-FAKE: a missing/garbage file raises. MLX-free."""
    z = np.load(Path(npz_path), allow_pickle=False)
    dec: dict[str, np.ndarray] = {}
    for k in z.files:
        if k.startswith("__"):
            continue
        if k == "code" or k.endswith("code"):
            continue
        if k == "B" or k.endswith("_B"):
            continue
        dec[k] = np.asarray(z[k], np.float32)
    if "in_proj.weight" not in dec:
        raise ValueError(
            f"--freeze-decoder-fit-codes {npz_path} has no 'in_proj.weight' (not a level-set witness "
            "decoder npz?); NO-FAKE: refusing to fit codes against a non-decoder file.")
    return dec


def _resolve_resume_path(p: Path) -> Path:
    """Accept a run dir (prefer the resume sidecar, fall back to the EMA deploy npz) OR an explicit
    npz file. NO-FAKE: nonexistent -> FileNotFoundError (never fabricate a resume)."""
    p = Path(p)
    if p.is_dir():
        for name in ("levelset_resume_state.npz", "levelset_witness_ema_mlx.npz"):
            cand = p / name
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"--resume-from dir {p} has neither levelset_resume_state.npz nor "
            "levelset_witness_ema_mlx.npz (nothing to resume from).")
    if p.exists():
        return p
    raise FileNotFoundError(f"--resume-from path {p} does not exist (NO-FAKE: refusing to fabricate).")


_STAGE_TAGS = {"ce": "stageCE", "tau_softplus": "stageTau", "l7_softplus": "stageL7", "margin_hinge": "stageHinge"}


def _stage_tag(seg_form: str) -> str:
    """Filename-safe stage tag for the PRESERVED per-stage checkpoint (PR95 curriculum stages)."""
    return _STAGE_TAGS.get(str(seg_form), f"stage_{seg_form}")


# ---------------------------------------------------------------------------
# The LEVEL-SET RGB witness (MLX). Interface-compatible with the imported render/loss path:
#   __call__(coord_feats, code_idx) -> (P, 3) RGB    (used by make_loss_fn / _render_rgb_render_res)
#   call_batch(coord_feats, code_indices) -> (K, P, 3)
#   sdf(coord_feats, code_idx) -> (P, K)             (used by the Eikonal/length reg)
# ---------------------------------------------------------------------------
def build_levelset_rgb_witness(
    num_pairs: int,
    in_feat: int,
    hidden_dim: int,
    n_hidden: int,
    mod_dim: int,
    n_classes: int,
    activation: str,
    softmax_temp: float,
    wire_w0: float,
    wire_s0: float,
    hosc_beta: float,
    hosc_omega: float,
    chroma: bool,
    palette_init_logit: np.ndarray | None = None,
    film_per_layer: bool = False,
    film_concat_code: bool = False,
):
    import mlx.core as mx
    import mlx.nn as nn

    class LevelSetRGBWitness(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_pairs = num_pairs
            self.n_hidden = n_hidden
            self.hidden_dim = hidden_dim
            self.n_classes = n_classes
            # LEVER-A (FiLM-rank-fix) toggles (default OFF => the extra submodules are NOT created =>
            # model.parameters() / EMA / checkpoints / byte-close are BYTE-IDENTICAL to the pre-LEVER-A
            # witness, and the forward branches below are skipped).
            self.film_per_layer = bool(film_per_layer)
            self.film_concat_code = bool(film_concat_code)
            self.activation = str(activation)
            self.softmax_temp = float(softmax_temp)
            self.wire_w0 = float(wire_w0)
            self.wire_s0 = float(wire_s0)
            self.hosc_beta = float(hosc_beta)
            self.hosc_omega = float(hosc_omega)
            # periodic_omega exposed for parity with the RGB witness verdict-forward convention.
            self.periodic_omega = float(hosc_omega)
            self.wire_scale = float(wire_s0)
            self.chroma = bool(chroma)
            self.code = mx.zeros((num_pairs * 2, mod_dim))
            self.in_proj = nn.Linear(in_feat, hidden_dim)
            self.film = nn.Linear(mod_dim, 2 * hidden_dim * n_hidden)
            self.hidden = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
            # LEVER-A1 (--film-per-layer): SEPARATE per-layer RESIDUAL FiLM projections, IDENTITY at
            # init (zero weight+bias => the residual scale (+0) and shift (+0) are 0 => the modulation
            # at init == the shared-FiLM-only forward; with the flag ON the per-layer route then learns
            # INDEPENDENT per-pair (scale,shift) modulation, raising the per-pair modulation rank to
            # attack the MEASURED participation-ratio collapse 3.34@CE -> 1.19@l7). siren_init touches
            # ONLY in_proj+hidden, so these stay zero at init.
            if self.film_per_layer:
                self.film_pl = [nn.Linear(mod_dim, 2 * hidden_dim) for _ in range(n_hidden)]
                for _lin in self.film_pl:
                    _lin.weight = mx.zeros_like(_lin.weight)
                    _lin.bias = mx.zeros_like(_lin.bias)
            # LEVER-A2 (--film-concat-code): an ADDITIVE per-pair code-injection route added to each
            # hidden pre-activation. This is the algebraically-FOLDED concat: concat([h, code]) @ W
            # == h @ W_h + code @ W_c, folded into ONE zero-init projection mod_dim->hidden_dim
            # (concat_pl[li]) -- a NON-collapsing per-pair TRANSLATION route alongside the
            # multiplicative FiLM (what a moving lane needs). Zero init => no-op at init
            # (identity-residual); shape-safe (no existing layer dims change).
            if self.film_concat_code:
                self.concat_pl = [nn.Linear(mod_dim, hidden_dim) for _ in range(n_hidden)]
                for _lin in self.concat_pl:
                    _lin.weight = mx.zeros_like(_lin.weight)
                    _lin.bias = mx.zeros_like(_lin.bias)
            self.out_sdf = nn.Linear(hidden_dim, n_classes)     # K SDF fields (LINEAR)
            self.out_tex = nn.Linear(hidden_dim, 3)             # pose-carrying RGB texture
            # (DIAGNOSED FIX) learned per-class palette (K,3), in LOGIT space (sigmoid(palette)*255
            # = the class color). DEFAULT: anchor to the NATURAL per-class mean GT RGB (logit) —
            # the transfer probe hit realized d_seg 0.0049 with this palette; a generic luma-ramp
            # init left SegNet unable to separate classes (witness plateaued ~0.51). The palette
            # stays LEARNABLE (it can move off the anchor) but STARTS in SegNet's distribution.
            if palette_init_logit is not None:
                pal = np.asarray(palette_init_logit, np.float32).reshape(n_classes, 3)
            else:
                pal = np.zeros((n_classes, 3), np.float32)
                for k in range(n_classes):
                    t = (k / max(n_classes - 1, 1)) * 2.0 - 1.0
                    pal[k] = np.array([t, -t, 0.5 * t], np.float32) * 2.0
            self.palette = mx.array(pal)

        def _act(self, u):
            if self.activation == "wire":
                return mx.cos(self.wire_w0 * u) * mx.exp(-((self.wire_s0 * u) ** 2))
            if self.activation == "hosc":
                return mx.tanh(self.hosc_beta * mx.sin(self.hosc_omega * u))
            return nn.relu(u)

        def _trunk(self, coord_feats, code_idx):
            h = self._act(self.in_proj(coord_feats))
            code = self.code[code_idx]
            film = mx.reshape(self.film(code), (self.n_hidden, 2, self.hidden_dim))
            for li, layer in enumerate(self.hidden):
                # DEFAULT-OFF => scale==(1.0+film[li,0]), shift==film[li,1], no concat =>
                # pre == layer(h)*(1.0+film[li,0])+film[li,1] => BYTE-IDENTICAL to pre-LEVER-A.
                scale = 1.0 + film[li, 0]
                shift = film[li, 1]
                if self.film_per_layer:
                    pl = mx.reshape(self.film_pl[li](code), (2, self.hidden_dim))
                    scale = scale + pl[0]
                    shift = shift + pl[1]
                pre = layer(h) * scale + shift
                if self.film_concat_code:
                    pre = pre + self.concat_pl[li](code)
                h = self._act(pre)
            return h  # (P, hidden)

        def sdf(self, coord_feats, code_idx):
            return self.out_sdf(self._trunk(coord_feats, code_idx))  # (P, K)

        def _compose_rgb(self, h):
            phi = self.out_sdf(h)                                   # (..., K)
            tex = self.out_tex(h)                                   # (..., 3)
            soft = mx.softmax(phi / self.softmax_temp, axis=-1)     # (..., K)
            base = soft @ self.palette                             # (..., 3) class color (SDF-pinned)
            rgb = mx.sigmoid(base + tex) * 255.0                   # POSE-LEGAL (texture carries pose)
            if not self.chroma:
                luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
                rgb = mx.concatenate([luma, luma, luma], axis=-1)
            return rgb

        def __call__(self, coord_feats, code_idx):
            return self._compose_rgb(self._trunk(coord_feats, code_idx))  # (P, 3)

        def call_batch(self, coord_feats, code_indices):
            h0 = self._act(self.in_proj(coord_feats))               # (P, hidden) shared
            codes = self.code[code_indices]                        # (K, mod)
            film = mx.reshape(self.film(codes), (-1, self.n_hidden, 2, self.hidden_dim))
            h = mx.broadcast_to(h0[None], (film.shape[0], h0.shape[0], h0.shape[1]))
            for li, layer in enumerate(self.hidden):
                # DEFAULT-OFF => BYTE-IDENTICAL to the pre-LEVER-A batched forward (same expression).
                scale = 1.0 + film[:, li, 0][:, None, :]
                shift = film[:, li, 1][:, None, :]
                if self.film_per_layer:
                    pl = mx.reshape(self.film_pl[li](codes), (-1, 2, self.hidden_dim))
                    scale = scale + pl[:, 0][:, None, :]
                    shift = shift + pl[:, 1][:, None, :]
                pre = layer(h) * scale + shift
                if self.film_concat_code:
                    pre = pre + self.concat_pl[li](codes)[:, None, :]
                h = self._act(pre)
            return self._compose_rgb(h)                            # (K, P, 3)

        # ---- #224 accessors for the analytic-lane render-band (ADDITIVE; only called when
        # --lane-render-band is ON => the default render is byte-identical). ----
        def call_margin(self, coord_feats, code_idx):
            """top1-top2 softmax decision margin (PROB scale) of the witness partition — the
            #141 quantity the analytic-lane uncertainty gate rides. Returns (P,); reshape to
            (H,W) at the call site."""
            soft = mx.softmax(self.out_sdf(self._trunk(coord_feats, code_idx)) / self.softmax_temp, axis=-1)
            s = mx.sort(soft, axis=-1)                              # ascending
            return s[..., -1] - s[..., -2]                          # (P,) top1 - top2

        def render_lane_appearance(self, coord_feats, code_idx, lane_cls: int = 1):
            """The witness's OWN per-pixel lane color = sigmoid(palette[lane_cls] + tex)*255
            (self-consistent, byte-free; gradient flows through tex/palette per the band spec).
            luma-collapsed when not chroma (matches _compose_rgb)."""
            tex = self.out_tex(self._trunk(coord_feats, code_idx))  # (P,3)
            rgb = mx.sigmoid(self.palette[lane_cls] + tex) * 255.0  # (P,3)
            if not self.chroma:
                luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
                rgb = mx.concatenate([luma, luma, luma], axis=-1)
            return rgb

    return LevelSetRGBWitness()


# ---------------------------------------------------------------------------
# MLX level-set regularizers (differentiable twins of the numpy reference). On phi (P,K)
# reshaped to (H,W,K): Eikonal drives |grad phi|->1 (true SDF); length is the Chan-Vese
# boundary-perimeter prior (short, smooth class boundaries). Kept SMALL (topology bias, not
# the driver — the realized seg loss drives d_seg).
# ---------------------------------------------------------------------------
def _eikonal_length_mlx(phi_pk, render_h: int, render_w: int, len_eps: float = 1.0,
                        junction_relax: float = 0.0, junction_tau: float = 0.5):
    """(fix h) Eikonal + Chan-Vese length on the DECISION MARGIN m = phi_top1 - phi_top2 (the
    quantity the argmax boundary lives on), NOT each field's own zero-set. Eikonal drives
    |grad m|->1 (the 1-Lipschitz margin = the R-survival quantity); the length term
    delta_eps(m)*|grad m| penalizes the perimeter of the ACTUAL inter-class boundary {m=0}.

    (THETA* TIER-2 STRETCH-1) ``junction_relax`` (default 0.0 = OFF = BIT-IDENTICAL) down-weights the
    Eikonal |grad m|->1 residual near TRIPLE JUNCTIONS, where 3+ classes meet and the top1-top2 margin
    surface m is genuinely non-smooth (a crease/kink), so forcing |grad m|=1 there fights the geometry
    and injects boundary noise. Triple-junction proximity is the top2-top3 SDF gap g3 =
    sort(phi)[-2]-sort(phi)[-3] (small => near a 3-way meet; needs >=3 classes). The per-pixel weight
    w = 1 - junction_relax*exp(-g3/junction_tau) in [1-relax, 1] multiplies the SQUARED Eikonal residual
    BEFORE the mean. junction_relax=0 => w==1.0 exactly => mean is BIT-IDENTICAL (x*1.0==x for finite
    IEEE floats). The LENGTH term is unchanged (delta_eps already localizes it to the {m=0} boundary)."""
    import mlx.core as mx

    phi = mx.reshape(phi_pk, (render_h, render_w, -1))
    srt = mx.sort(phi, axis=-1)
    m = srt[..., -1] - srt[..., -2]  # (H,W) >=0 decision margin (top1-top2)
    gy = m[1:, :] - m[:-1, :]
    gx = m[:, 1:] - m[:, :-1]
    gmag = mx.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2 + 1e-8)  # (H-1,W-1)
    eik_resid = (gmag - 1.0) ** 2
    if junction_relax > 0.0 and phi.shape[-1] >= 3:
        # (STRETCH-1) triple-junction proximity weight: down-weight the Eikonal where 3 classes nearly
        # meet (small top2-top3 gap). Aligned to the (H-1,W-1) gmag grid by the matching [:-1,:-1] slice.
        g3 = srt[..., -2] - srt[..., -3]                                  # (H,W) top2-top3 gap (>=0)
        w = 1.0 - float(junction_relax) * mx.exp(-g3[:-1, :-1] / float(junction_tau))  # (H-1,W-1)
        eik = mx.mean(w * eik_resid)
    else:
        eik = mx.mean(eik_resid)  # DEFAULT: BIT-IDENTICAL to the pre-theta* `mx.mean((gmag-1.0)**2)`.
    mc = m[:-1, :-1]
    delta = (len_eps / np.pi) / (len_eps * len_eps + mc * mc)  # delta_eps at the {m=0} boundary
    length = mx.mean(delta * gmag)
    return eik, length, mx.mean(gx * gx) + mx.mean(gy * gy)


def _eikonal_margin_interior_mlx(phi_pk, render_h: int, render_w: int):
    """(EIK-STAB build 1) Shared interior-stencil geometry of the decision margin m = top1-top2:
    central first derivatives + second derivatives on the (H-2, W-2) interior grid (h = 1 px).
    Returns (gx, gy, m_xx, m_yy, m_xy), each (H-2, W-2). Used ONLY by the default-OFF stabilizer
    terms below (never on the default path -> zero compute unless a stabilizer flag is set)."""
    import mlx.core as mx

    phi = mx.reshape(phi_pk, (render_h, render_w, -1))
    srt = mx.sort(phi, axis=-1)
    m = srt[..., -1] - srt[..., -2]                                # (H,W) decision margin
    gx = 0.5 * (m[1:-1, 2:] - m[1:-1, :-2])                        # central d/dx
    gy = 0.5 * (m[2:, 1:-1] - m[:-2, 1:-1])                        # central d/dy
    m_xx = m[1:-1, 2:] - 2.0 * m[1:-1, 1:-1] + m[1:-1, :-2]
    m_yy = m[2:, 1:-1] - 2.0 * m[1:-1, 1:-1] + m[:-2, 1:-1]
    m_xy = 0.25 * (m[2:, 2:] - m[2:, :-2] - m[:-2, 2:] + m[:-2, :-2])
    return gx, gy, m_xx, m_yy, m_xy


def _eikonal_steik_mlx(phi_pk, render_h: int, render_w: int):
    """(EIK-STAB build 1a) StEik directional-divergence stabilizer (Yang-Walker-Parkinson et al.,
    NeurIPS 2023, arXiv 2305.18414): L_dir = mean |grad u^T D^2u grad u| — the second-order
    derivative along the (RAW, un-normalized) gradient direction, in ABSOLUTE VALUE (L1), exactly
    the paper's integrand. Differentiating |grad u|=1 gives D^2u·grad u = 0 for a true SDF, so the
    term damps ONLY the normal-direction second-order mode (the anti-diffusive/backward-heat
    instability of the (|grad u|-1)^2 gradient flow the paper proves), leaving tangential curvature
    (our lane dashes / fine geometry) FREE — unlike the DiGS full-Laplacian (Delta u)^2 which
    over-smooths (would eat the dashes, the MCF-erasure enemy).

    HONEST ADAPTATIONS (documented per NO-FAKE; litsweep memo 'exact term to be lifted from the
    paper, never invented'): (1) our field is the DECISION MARGIN m = phi_top1 - phi_top2 — the
    same field our eikonal term drives to |grad m|=1 (fix h in _eikonal_length_mlx), NOT a single
    SDF head; the instability we measured (the eikonal runaway along |grad phi|) lives on m, so the
    damping is applied where the disease is. (2) discrete central stencils on the pixel grid (h=1)
    replace the paper's autograd Hessian (our margin is a grid field, not a coordinate-MLP output
    we can double-differentiate cheaply through R). (3) the paper anneals alpha_l linearly to zero
    mid-training; here the weight is a CONSTANT flag (--eikonal-steik-weight) — the n24 arbitration
    probe runs are short; a schedule can be added if the arm wins. Default weight 0.0 => this
    function is NEVER CALLED => byte-identical."""
    import mlx.core as mx

    gx, gy, m_xx, m_yy, m_xy = _eikonal_margin_interior_mlx(phi_pk, render_h, render_w)
    dir_div = gx * gx * m_xx + 2.0 * gx * gy * m_xy + gy * gy * m_yy   # grad m^T H(m) grad m
    return mx.mean(mx.abs(dir_div))


def _eikonal_steik_normalized_mlx(phi_pk, render_h: int, render_w: int, norm_eps: float = 1e-2):
    """(V6 SYMPOSIUM #317, EIK-STAB build 1a-N) NORMALIZED StEik: the UNIT-NORMAL second-order
    curvature n^T H(m) n, n = grad m / |grad m|, in ABSOLUTE VALUE (L1, matching the raw form). This
    is the theoretically-principled fix for the raw-StEik self-amplification NO-GO measured in the
    FEED-05v arbitration (raw |grad m^T H grad m| = |grad m|^2 * |n^T H n| carries a QUARTIC |grad m|^2
    scaling that self-amplifies at the far-from-SDF resumed state where |grad m| >> 1 -> 575x-1431x
    runaway). Removing the |grad m|^2 factor:

        L_norm = mean | (grad m^T H(m) grad m) / (|grad m|^2 + norm_eps) |   = mean |n^T H n|

    Deep math (why this is the SURGICAL cure, not ViscoReg's isotropic one):
    * StEik proves the (|grad u|-1)^2 gradient flow is anti-diffusive along grad u (the NORMAL
      direction n). The unstable mode is EXACTLY the normal-direction 2nd derivative n^T H n. For a
      true SDF |grad u|=1 everywhere => H(m) grad m = 0 => n^T H n = 0, so penalizing |n^T H n|
      drives the field toward the SDF property along n WITHOUT the raw form's scaling.
    * ANISOTROPY vs ViscoReg: Laplacian Lap m = n^T H n + t^T H t (trace = sum of 2nd derivs along
      ANY orthonormal frame; t = the unit tangent). ViscoReg's viscous residual damps the FULL Lap m
      (normal + tangential = isotropic; risks eroding the lane dashes = tangential geometry).
      n^T H n damps the NORMAL direction ONLY (the proven anti-diffusive mode), tangential curvature
      (dashes / fine boundary geometry) FREE. This is the more-surgical of the two cures by design.
    * norm_eps regularizes n where |grad m| -> 0 (flat argmax-stable interior, where n is undefined
      but also IRRELEVANT: there m_xx=m_yy=m_xy=0 too, so dir_div=0 and the term vanishes). At the
      boundary annulus |grad m| ~ 1 (the eikonal target) so norm_eps=1e-2 leaves the boundary term
      essentially intact (1/(1+0.01) ~ 0.99) while suppressing spurious flat-region amplification.

    HONEST ADAPTATIONS (per NO-FAKE): same (1)/(2) as _eikonal_steik_mlx (decision-margin field;
    discrete central stencils on (H-2,W-2)); the normalization by (|grad m|^2 + eps) is OURS (the
    named FEED-05v follow-up 'normalized variant n^T H n, build only if visco walls' — the paper's
    directional-divergence integrand is the RAW form; we adapt it to remove the measured
    self-amplification). Reuses --eikonal-steik-weight for the weight; selected by
    --eikonal-steik-normalized. Default OFF => this function is NEVER CALLED => byte-identical."""
    import mlx.core as mx

    gx, gy, m_xx, m_yy, m_xy = _eikonal_margin_interior_mlx(phi_pk, render_h, render_w)
    dir_div = gx * gx * m_xx + 2.0 * gx * gy * m_xy + gy * gy * m_yy   # grad m^T H(m) grad m
    gmag2 = gx * gx + gy * gy                                          # |grad m|^2
    n_hess_n = dir_div / (gmag2 + float(norm_eps))                    # n^T H n (unit-normal curvature)
    return mx.mean(mx.abs(n_hess_n))


def _eikonal_visco_mlx(phi_pk, render_h: int, render_w: int, visco_eps: float):
    """(EIK-STAB build 1b) ViscoReg vanishing-viscosity eikonal residual (arXiv 2507.00412):
    L_veik = mean( (|grad m| - 1 - eps*Lap m)^2 )  [the paper's p=2 form]. The inviscid eikonal
    equation |grad u|=1 is ill-posed (infinitely many weak solutions; the (|grad u|-1)^2 flow is
    anti-diffusive along grad u — the measured runaway); the VISCOUS equation |grad u| = 1 + eps*Lap u
    selects the true SDF (the viscosity solution) as eps->0. Training with the viscous residual and
    ANNEALING eps to zero (--eikonal-viscosity-anneal; the paper uses piecewise-linear decay to 0 and
    reports insensitivity to the exact profile) is a continuation in eps — literally our Gamma/GNC
    curriculum philosophy applied to the eikonal term itself.

    HONEST ADAPTATIONS (per NO-FAKE): (1) applied on the decision margin m (same field as the
    legacy eikonal term; see _eikonal_steik_mlx note 1). (2) central interior stencil (H-2, W-2)
    for BOTH |grad m| and Lap m (the legacy term uses forward differences on (H-1, W-1)); when the
    anneal reaches eps == 0.0 exactly, the call site switches BACK to the legacy stencil — the two
    eikonal residuals differ by O(stencil) at that instant (documented discontinuity; by then
    eps ~ 0 so the viscous residual ~= the central eikonal residual). This REPLACES the eikonal
    residual while eps > 0 (it is the same constraint, viscous form — adding both would double-count);
    the Chan-Vese length term is unchanged. Default eps 0.0 => NEVER CALLED => byte-identical."""
    import mlx.core as mx

    gx, gy, m_xx, m_yy, _ = _eikonal_margin_interior_mlx(phi_pk, render_h, render_w)
    gmag = mx.sqrt(gx * gx + gy * gy + 1e-8)
    lap = m_xx + m_yy
    resid = gmag - 1.0 - float(visco_eps) * lap
    raw = mx.mean(resid * resid)
    # (C3 CONFOUND FIX 2026-07-05) UNIT/pi-group RECALIBRATION. The `- eps*Lap m` term makes the
    # viscous residual's raw magnitude O(eps^2 * mean(Lap^2)) which measured ~2490 on a sharpening
    # field -- ~2490x the LEGACY (|grad m|-1)^2 residual (~O(1)). The SAME `--eikonal-weight` (0.05)
    # was tuned for the LEGACY residual, so under the viscous form 0.05*2490 = 124.5 => the eik term
    # DOMINATED the loss at 86-91% (the deepest confound root -- an uncalibrated unit bug, NOT
    # "physics needs strong eik"). Normalize to an O(1) per-pixel-MEAN scale via a STOP-GRADIENT
    # self-normalizer floored at 1.0 (the legacy band): when raw >> 1 the term ~= 1.0 (O(1), so
    # --eikonal-weight has the SAME meaning as for the legacy residual; the descent DIRECTION is
    # preserved -- only the magnitude is rescaled); when raw <= 1 (constraint well-satisfied) the term
    # = raw (legacy-like small). This ONLY touches the viscous branch; the legacy residual is
    # UNCHANGED (byte-identical). The startup regularizer-magnitude log (see _log_regularizer_magnitudes)
    # records the raw pre-weight scale so the recalibration is auditable.
    scale = mx.stop_gradient(mx.maximum(raw, mx.array(1.0)))
    return raw / scale


def _visco_eps_for_epoch(ep: int, eps0: float, anneal_epochs: int) -> float:
    """(EIK-STAB build 1b) Per-epoch vanishing-viscosity schedule: linear decay from eps0 at ep=0
    to 0.0 at ep>=anneal_epochs (ViscoReg's 'many reasonable decays work; decay to zero' finding —
    the simplest monotone-to-zero profile). anneal_epochs<=0 => CONSTANT eps0 (no anneal). Pure /
    unit-tested. NOTE the schedule is in ABSOLUTE epochs (matches the trainer's other anneals: a
    resumed run continues the same schedule bit-faithfully)."""
    e0 = float(eps0)
    n = int(anneal_epochs)
    if e0 <= 0.0:
        return 0.0
    if n <= 0:
        return e0
    return e0 * max(0.0, 1.0 - float(ep) / float(n))


def _adaptive_visco_eps(c_a: float, eta: float, lam_eik: float, margin_factor: float,
                        eps_floor: float, eps_upper: float) -> float:
    """(V6 #320 / DE #318 §4 Arm-2 / symposium #317 §7.4) ADAPTIVE vanishing-viscosity eps tracking
    the CFL LOWER edge eps_lower = |c_a|*sqrt(eta*lambda_eik/8):

        eps(t) = clamp( |c_a(t)| * sqrt(eta * lambda_eik / 8) * (1 + margin_factor), eps_floor, eps_upper )

    The DERIVED mechanism cure for the v5 ep110 re-entry: a FIXED eps eventually falls below the
    RISING lower edge (as progressive sharpening grows |c_a(t)|); this eps TRACKS it.

    (C2 CONFOUND FIX 2026-07-05) The ORIGINAL law `eps = |c_a|*sqrt(eta*lambda_eik/8)*(1+margin)`
    was measured INERT: with eta~1e-3, lambda_eik~0.05 the sqrt prefactor collapses to ~2.5e-3, so
    the edge is ~2.5e-3*|c_a| -- to even REACH the floor 0.3 you need |c_a| >= ~80, but the measured
    |c_a| is O(1) (~0.82). So eps clamped at the floor EVERY epoch (0 change-events; the "adaptive"
    wrapper never adapted -> the "viscosity NO-GO" verdict rested on a CONSTANT eps=floor). The CFL
    edge says the RUN is always safe (required eps ~2.5e-3 << floor 0.3); the floor already over-damps.

    REPARAMETERIZATION (keeps the CFL INTENT -- monotone-increasing in |c_a|, saturating -- but makes
    eps actually RESPOND across [floor,upper] for O(1) |c_a|): map the sharpness proxy through a
    saturating tanh into the [floor,upper] band. `(1+margin_factor)` tilts the response UP (more
    viscosity headroom per unit sharpness). At |c_a|~0.82, margin 0.5 => tanh(1.23)~0.84 => eps~0.64
    (mid-band, RESPONSIVE); |c_a|->large => eps->upper (never exceeds the biharmonic-explosion clamp);
    |c_a|->0 => eps->floor. eta/lam_eik are retained in the SIGNATURE (the launcher passes them) but
    are NO LONGER the collapsed prefactor -- they inform only the ADVISORY pi_eik logged by the caller.
    NOTE (NO-FAKE): this DIVERGES from the numpy `adaptive_visco_eps` reference in
    tac.boundary_math.eikonal_sharpness_proxy_reference (sibling-owned; that reference is ALSO inert
    and must be updated to match -- flagged to the launcher/gates owner). eps_upper raised to floor if
    inverted. Pure / unit-testable."""
    lo = float(eps_floor)
    hi = float(eps_upper)
    if hi < lo:
        hi = lo
    # saturating map of the O(1) sharpness proxy into [floor, upper]; margin_factor tilts up.
    frac = math.tanh(abs(float(c_a)) * (1.0 + float(margin_factor)))
    eps = lo + (hi - lo) * frac
    # (eta, lam_eik retained for signature/advisory pi_eik; not the collapsed prefactor -- see docstring)
    _ = (eta, lam_eik)
    return min(max(eps, lo), hi)


def _ca_from_margin_mlx(m, band: float = 0.0) -> float:
    """(V6 #320) Sharpness proxy |c_a| = mean|(|grad m|-1)/|grad m|| over the decision-margin
    interior of a single (H,W) margin field ``m`` (MLX array). band>0 => restrict to the small-margin
    annulus |m|<band (DE #318 §2 flat regime); band==0 (DEFAULT) => interior mean (symposium §7.4
    exact launch formula). Uses the SAME central-diff interior stencil + 1e-8 gmag floor as
    ``_eikonal_margin_interior_mlx`` / the numpy ``sharpness_proxy_c_a`` reference (byte-parity)."""
    import mlx.core as mx

    gx = 0.5 * (m[1:-1, 2:] - m[1:-1, :-2])   # central d/dx (cols)
    gy = 0.5 * (m[2:, 1:-1] - m[:-2, 1:-1])   # central d/dy (rows)
    gmag = mx.sqrt(gx * gx + gy * gy + 1e-8)
    c_a = mx.abs((gmag - 1.0) / gmag)         # (H-2, W-2)
    if band > 0.0:
        m_int = m[1:-1, 1:-1]
        mask = mx.abs(m_int) < float(band)
        denom = float(mx.sum(mask.astype(mx.float32)))
        num = float(mx.sum(mx.where(mask, c_a, mx.zeros_like(c_a))))
        return num / denom if denom > 0.0 else 0.0
    return float(mx.mean(c_a))


def _measure_ca_mlx(model, pairs, cf_fn, render_h: int, render_w: int, band: float = 0.0) -> float:
    """(V6 #320) No-grad per-epoch |c_a(t)| over a small FIXED deterministic subset of pairs. One
    ``model.sdf`` forward per pair (frame0, matching the loss's phi0) => the WITNESS decision margin
    m = top1-top2 (NOT a SegNet forward — |c_a| is the witness's own field, zero SegNet cost); reshape
    to (H,W); |c_a| via ``_ca_from_margin_mlx``; averaged over the subset. Deterministic (no RNG)."""
    import mlx.core as mx

    acc = 0.0
    n = 0
    for pi in pairs:
        phi = model.sdf(cf_fn(int(pi)), 2 * int(pi) + 0)      # (H*W, K) frame0 SDF logits, no grad
        phi_r = mx.reshape(phi, (render_h, render_w, -1))
        srt = mx.sort(phi_r, axis=-1)
        m = srt[..., -1] - srt[..., -2]                       # (H,W) decision margin (top1-top2)
        acc += _ca_from_margin_mlx(m, band)
        n += 1
    return acc / max(n, 1)


class SpikeGuardRollback:
    """(EIK-STAB build 2; sweep lever #3 + #304) Pure decision state machine for
    ``--spike-guard-mode rollback`` — the physics-informed replacement for the legacy
    skip-with-frozen-median actuator whose absorbing deadlock we measured 3x (#205 runs
    015247Z/083453Z/095728Z: median updates only on ACCEPTED batches, so a persistent
    loss-level shift => 100% skip forever).

    Physics (litsweep DOMAIN 2, contradiction row 3): at an Edge-of-Stability crossing the
    oscillation along the sharp direction IS the mechanism that reduces sharpness
    (Damian-Nichani-Lee self-stabilization); skipping every spiked batch BLOCKS that feedback.
    So this guard: (a) TOLERATES bounded oscillation — single finite spikes are ACCEPTED (stepped),
    only counted; (b) on SUSTAINED runaway (spike fraction > ``frac`` over a FULL ``window`` of
    recent batches) returns "rollback" — the trainer restores the last-good weights/EMA/opt
    snapshot, cuts lr x``lr_cut``, clears the loss window (fresh median re-arm), and continues:
    an actuator that returns to the stable basin AT A STABLE STEP SIZE instead of freezing.
    After ``max_rollbacks`` the budget is spent and the machine returns "exhausted" forever
    (the trainer reverts to legacy skip semantics — bounded actuation, loud in the log).

    Pure python (no MLX): unit-testable with synthetic spike sequences (the induced-runaway test).
    The trainer owns the side effects; this class owns ONLY the decision."""

    def __init__(self, window: int, frac: float, max_rollbacks: int) -> None:
        if int(window) < 1:
            raise ValueError(f"spike-rollback window must be >= 1, got {window}")
        if not (0.0 < float(frac) <= 1.0):
            raise ValueError(f"spike-rollback frac must be in (0, 1], got {frac}")
        if int(max_rollbacks) < 1:
            raise ValueError(f"spike-rollback max must be >= 1, got {max_rollbacks}")
        self.window = int(window)
        self.frac = float(frac)
        self.max_rollbacks = int(max_rollbacks)
        self.rollbacks = 0
        self._events: list[bool] = []

    @property
    def exhausted(self) -> bool:
        return self.rollbacks >= self.max_rollbacks

    def spike_frac(self) -> float:
        if not self._events:
            return 0.0
        return sum(1 for e in self._events if e) / float(len(self._events))

    def rearm(self) -> None:
        """Clear the event window (fresh start after a rollback / a stage re-treat)."""
        self._events.clear()

    def observe(self, spiked: bool) -> str:
        """Record one batch outcome; return the action: 'ok' | 'rollback' | 'exhausted'.
        'rollback' fires ONLY on a FULL window with spike fraction > frac, and self-rearms
        (the next trigger needs a freshly refilled window => bounded trigger frequency)."""
        if self.exhausted:
            return "exhausted"
        self._events.append(bool(spiked))
        if len(self._events) > self.window:
            self._events.pop(0)
        if len(self._events) == self.window and self.spike_frac() > self.frac:
            self.rollbacks += 1
            self.rearm()
            return "rollback"
        return "ok"


def _nuclear_norm_smooth_mlx(code, *, rel_eps: float = 1e-3, ns_iters: int = 25):
    """(THETA* TIER-2 MUST-2) DIFFERENTIABLE smoothed nuclear norm of the per-(pair,frame) FiLM code
    matrix ``code`` (shape (num_pairs*2, mod_dim)) -- a convex low-rank relaxation that drives the
    learned per-pair codes toward a low-rank subspace (-> fewer effective DOF -> lower entropy / rate
    at byte-close). DEFAULT-OFF at the call site (--code-nuclear-weight 0.0 => never invoked => the
    loss is byte-identical).

    WHY smoothed + Newton-Schulz (the differentiable-path choice, documented per NO-FAKE): MLX 0.31
    has NO vjp for ``mx.linalg.svd`` NOR ``mx.linalg.eigvalsh`` ([Primitive::vjp] Not implemented),
    so NEITHER can be a LOSS term (verified on CPU). The nuclear norm = sum of singular values =
    trace(sqrt(C^T C)). The matrix square root is computed by the coupled Newton-Schulz iteration
    (matmuls ONLY -> fully autodiff-able in MLX). Plain NS DIVERGES (->NaN) on exact-zero singular
    values -- exactly the rank-deficient codes the penalty itself produces -- so we compute the
    SMOOTHED nuclear norm ``trace(sqrt(C^T C + eps*||C^T C||_F * I)) = sum_i sqrt(sigma_i^2 +
    eps*||G||_F)`` with a small RELATIVE floor ``eps`` (default 1e-3). This is a standard smoothed
    nuclear-norm surrogate: -> the exact nuclear norm as eps->0; matches it to ~0.3% on well-conditioned
    full-rank inputs (verified, gradient cosine 1.0000 vs the exact U V^T); stays FINITE +
    monotone-in-the-singular-values (still drives low-rank) on rank-deficient inputs; and ->0 as the
    codes ->0. It is NOT the exact nuclear norm (the smoothing floor over-counts near-zero singular
    directions) -- labelled SMOOTHED, not exact, per NO-FAKE. MLX matmuls only (no model/scorer; runs
    + autodiffs on CPU). Empirical anchor: experiments/tests/test_levelset_theta_star_tier2_levers.py."""
    import mlx.core as mx

    G = code.T @ code                              # (mod_dim, mod_dim) Gram, PSD
    n = G.shape[0]
    eye = mx.eye(n)
    normG = mx.sqrt(mx.sum(G * G)) + 1e-20         # ||G||_F (scalar)
    Y0 = G / normG + float(rel_eps) * eye          # eigvals in [rel_eps, 1+rel_eps] -> NS-stable
    s = mx.sqrt(mx.sum(Y0 * Y0)) + 1e-20           # spectral renormalization (NS safety margin)
    Y = Y0 / s
    Z = eye
    for _ in range(int(ns_iters)):
        Tm = 0.5 * (3.0 * eye - Z @ Y)             # coupled Newton-Schulz for the matrix sqrt
        Y = Y @ Tm
        Z = Tm @ Z
    # trace(sqrt(G + eps*||G||_F I)) = sqrt(normG)*sqrt(s)*trace(sqrt(Y))  [sqrt homogeneous deg-1/2]
    return mx.trace(Y) * mx.sqrt(s) * mx.sqrt(normG)


# ---------------------------------------------------------------------------
# MLX-GPU SDF->argmax forward (FEED-eo, the --gpu-reorient core, additive). This is the MLX-GPU
# TWIN of the phi path in ``levelset_rgb_forward_numpy`` (the numpy ONE CODEPATH). It runs the same
# in_proj -> FiLM -> hidden -> out_sdf forward on the dequantized deploy weights, but in fp32 ON THE
# GPU (vs the numpy fp64 accumulation), so it is NOT bit-identical (the GPU vs numpy reduction order
# differs) -> the per-pair argmax it returns is PARITY-GATED, never an authority. Its sole consumer
# is the self-orientation reorient (recompute per-pair directional feats from the EMA argmax), which
# is itself a byte-closeable train-time PRIOR (cos 0.89-0.91 vs GT; the dir feats are a deterministic
# function of the witness's own argmax). Eliminating the 600 GPU-idle numpy CPU forwards (~499s every
# --reorient-every epochs at n600) is the ~6.2% wall-clock lever. NO mx ops touch ema.shadow/model.
# ---------------------------------------------------------------------------
def levelset_sdf_argmax_mlx(
    deploy_mx: dict,
    feats_mx,
    code_row_mx,
    *,
    n_hidden: int,
    hidden_dim: int,
    activation: str,
    wire_w0: float,
    wire_s0: float,
    hosc_beta: float,
    hosc_omega: float,
):
    """Return ``argmax_k phi_k`` (P,) int via the MLX-GPU twin of the numpy deploy forward.

    ``deploy_mx`` are the DEQUANTIZED deploy weights already as ``mx.array`` (in_proj/film/hidden.*/
    out_sdf {weight,bias}); ``feats_mx`` is the (P, in_feat) per-pair coord feature grid (curvelet
    [+ self-orient dir]); ``code_row_mx`` is the (mod_dim,) per-(pair,frame) FiLM code. Mirrors
    ``mlx.nn.Linear`` (``x @ W.T + b``) + ``LevelSetRGBWitness._act`` EXACTLY (only the device +
    fp32-vs-fp64 accumulation differ -> parity-gated, NOT the verdict authority). out_tex/palette/
    softmax are NOT computed (argmax of phi is the only quantity the reorient needs)."""
    import mlx.core as mx

    def _act(u):
        if activation == "wire":
            return mx.cos(wire_w0 * u) * mx.exp(-((wire_s0 * u) ** 2))
        if activation == "hosc":
            return mx.tanh(hosc_beta * mx.sin(hosc_omega * u))
        return mx.maximum(u, 0.0)

    h = _act(feats_mx @ deploy_mx["in_proj.weight"].T + deploy_mx["in_proj.bias"])
    film = (code_row_mx @ deploy_mx["film.weight"].T + deploy_mx["film.bias"]).reshape(n_hidden, 2, hidden_dim)
    # LEVER-A AUTO-DETECT (parity-gated reorient): apply the OPTIONAL per-layer FiLM / code-concat
    # routes when their keys are present so the self-orient reorient argmax reflects the trained
    # witness. ABSENT keys (default-off) => BYTE-IDENTICAL to the pre-LEVER-A twin.
    _has_film_pl = any(str(k).startswith("film_pl.") for k in deploy_mx)
    _has_concat = any(str(k).startswith("concat_pl.") for k in deploy_mx)
    for li in range(n_hidden):
        scale = 1.0 + film[li, 0]
        shift = film[li, 1]
        if _has_film_pl:
            pl = (code_row_mx @ deploy_mx[f"film_pl.{li}.weight"].T + deploy_mx[f"film_pl.{li}.bias"]).reshape(2, hidden_dim)
            scale = scale + pl[0]
            shift = shift + pl[1]
        pre = (h @ deploy_mx[f"hidden.{li}.weight"].T + deploy_mx[f"hidden.{li}.bias"]) * scale + shift
        if _has_concat:
            pre = pre + (code_row_mx @ deploy_mx[f"concat_pl.{li}.weight"].T + deploy_mx[f"concat_pl.{li}.bias"])
        h = _act(pre)
    phi = h @ deploy_mx["out_sdf.weight"].T + deploy_mx["out_sdf.bias"]  # (P, K)
    return mx.argmax(phi, axis=-1)


# ---------------------------------------------------------------------------
# Curriculum seg_form by epoch (PR95 d_seg sequence): ce -> tau_softplus -> l7_softplus.
# OPTIMIZER curriculum (DAG FEED-fi): AdamW for the CE/tau/l7 stages, then an OPTIONAL PR95
# stage-8 MUON FINISHER (--muon-start-epoch, default None=AdamW-throughout=BIT-IDENTICAL). At
# the switch epoch the optimizer becomes mlx.optimizers.MultiOptimizer([Muon(2D hidden weights),
# AdamW(biases/code/out_sdf/out_tex)]) via tac.optimization.muon_finisher_mlx (Newton-Schulz
# orthogonalized momentum = THE measured d_seg drop, CLAUDE.md frontier "Muon is THE drop"). The
# switch is a per-stage TREATMENT boundary (re-treat: spike-guard cleared) and saves a PRESERVED
# stage-encoded ckpt (independently byte-closeable + resumable). NO false claim: this is a build;
# the d_seg verdict is the realized-through-R eval, the score is upstream/evaluate.py only.
# ---------------------------------------------------------------------------
def validate_lane_edge_config(
    *, lane_edge_weight: float, lane_edge_start_epoch: int, epochs: int,
    lane_edge_class: int, n_classes: int = 5,
) -> None:
    """(FEED-df R2) LEVER-3 fail-closed config guard (pure; testable; fail LOUD not silent).

    A lane lever that never engages (start_epoch > epochs) is a silent no-op = a FALSE 'lane-edge
    does not help' verdict; an out-of-range class index would IndexError mid-training (after GPU
    spend). When the lever is OFF (weight<=0) the guard is a NO-OP so the additive default path is
    never gated by a lever that is not in use.
    """
    if lane_edge_weight <= 0.0:
        return
    if lane_edge_start_epoch > epochs:
        raise ValueError(
            f"--lane-edge-weight {lane_edge_weight} > 0 but --lane-edge-start-epoch "
            f"({lane_edge_start_epoch}) > --epochs ({epochs}): the lane hinge would NEVER engage "
            "-> a silent no-op = a FALSE 'lane-edge does not help' verdict. Set "
            "--lane-edge-start-epoch <= --epochs (0 = engage from ep1)."
        )
    if not (0 <= lane_edge_class <= n_classes - 1):
        raise ValueError(
            f"--lane-edge-class ({lane_edge_class}) out of range [0,{n_classes - 1}] for the "
            f"{n_classes}-class comma10k CANONICAL partition [Road0,Lane1,Undrivable2,Movable3,MyCar4]; would "
            "IndexError mid-training. Use 1 for the lane orbit (the d_seg gate)."
        )


def validate_lane_thin_config(
    *, lane_thin_weight: float, lane_thin_start_epoch: int, epochs: int,
    lane_thin_class: int, lane_thin_radius: int, n_classes: int = 5,
) -> None:
    """(LEVER-B) thin-lane dropped-dash prior fail-closed config guard (pure; testable; fail LOUD).

    Mirrors ``validate_lane_edge_config``: a thin-lane lever that never engages (start > epochs) is a
    silent no-op = a FALSE 'thin-lane prior does not help' verdict; an out-of-range class would
    IndexError mid-training; a negative radius is malformed. When OFF (weight<=0) the guard is a
    NO-OP so the additive default path is never gated by a lever that is not in use."""
    if lane_thin_weight <= 0.0:
        return
    if lane_thin_start_epoch > epochs:
        raise ValueError(
            f"--lane-thin-weight {lane_thin_weight} > 0 but --lane-thin-start-epoch "
            f"({lane_thin_start_epoch}) > --epochs ({epochs}): the thin-lane hinge would NEVER engage "
            "-> a silent no-op = a FALSE 'thin-lane prior does not help' verdict. Set "
            "--lane-thin-start-epoch <= --epochs (0 = engage from ep1)."
        )
    if not (0 <= lane_thin_class <= n_classes - 1):
        raise ValueError(
            f"--lane-thin-class ({lane_thin_class}) out of range [0,{n_classes - 1}] for the "
            f"{n_classes}-class comma10k CANONICAL partition [Road0,Lane1,Undrivable2,Movable3,MyCar4]; "
            "would IndexError mid-training. Use 1 for the lane orbit (the d_seg gate)."
        )
    if lane_thin_radius < 0:
        raise ValueError(f"--lane-thin-radius ({lane_thin_radius}) must be >= 0 (window half-width).")


def lever_gate_on_at_epoch(weight: float, start_epoch: int, ep: int) -> bool:
    """Engagement predicate for the additive margin levers (lane-edge / margin-saliency / thin-lane).

    A lever is ENGAGED at training epoch ``ep`` iff its weight is > 0 AND the epoch has reached its
    ``start_epoch``. This is the SINGLE source of truth the epoch loop uses to (re-)flip every
    per-lever engagement gate every epoch. Extracting + unit-testing it is the SELF-PROTECT against
    the C1 silent-no-op class (review FEED-hp/hr): a gate initialized OFF for ``start_epoch>1`` that is
    NEVER re-flipped in the loop -> ``--<lever>-start-epoch>1`` (the help-RECOMMENDED 300) silently
    never engages -> a FALSE '<lever> does nothing' verdict from dead code. The C1 regression is
    EXACTLY ``lever_gate_on_at_epoch(w>0, start>1, ep=start)`` returning False; this predicate returns
    True, and the loop assigns its result, so the bug cannot silently re-emerge while this helper is
    the live decision. Pure + total => unit-testable at $0 (the realized-through-R loop needs MLX + the
    frozen scorer + the GT cache; this predicate does not). Per CLAUDE.md "Bugs must be permanently
    fixed AND self-protected against"."""
    return float(weight) > 0.0 and int(ep) >= int(start_epoch)


def seed_compose_weight_at_epoch(anneal_epochs: int, shape: str, ep: int) -> float:
    """BUILD #300 (b): island-SEED compose-weight anneal (transfer schedule) full(1.0) -> 0.0.

    The island seed (``--seed-islands``) is composited into the SegNet-scored frame1 (via
    ``_compose_chain``) so the witness has a formed island to absorb. Per the seed-absorption fix
    (memo ``plateau_disambiguator_results_20260704.md``) the seed compose weight is ANNEALED to 0
    across the CE stage so that by the anneal end the composed frame == the witness render (the deploy
    surface -- the ``eval_roundtrip`` discipline applied to the island crutch). Pure + total => the
    schedule is $0 unit-testable (the compose itself needs MLX + the frozen scorer; this schedule does
    not).

    ``anneal_epochs <= 0`` (DEFAULT) => constant ``1.0`` => ``_compose_chain`` is BYTE-IDENTICAL to the
    pre-#300 seed compose (the caller multiplies by the weight only when it differs from 1.0). For
    ``anneal_epochs > 0`` the weight is ``1.0`` at ``ep <= 1``, ramps to ``0.0`` at
    ``ep >= anneal_epochs`` (linear or cosine full->0), and clamps to ``0.0`` after. Deterministic in
    ``ep`` (a pure function) => a RESUME reproduces the same weight bit-for-bit; nothing to checkpoint.
    Set ``anneal_epochs ~= --tau-softplus-start-epoch`` so the seed is fully transferred to the witness
    BEFORE the tau/MCF stage erodes sub-critical island structure. Per CLAUDE.md "Bugs must be
    permanently fixed AND self-protected against" (the compose-time-crutch-starves-the-gradient
    meta-pattern: any compose-time assist added to a scored forward MUST anneal to zero OR route the
    absorption gradient through the deploy surface, else it starves the gradient it bootstraps)."""
    if int(anneal_epochs) <= 0:
        return 1.0
    e = int(ep)
    if e <= 1:
        return 1.0
    if e >= int(anneal_epochs):
        return 0.0
    frac = (float(e) - 1.0) / (float(anneal_epochs) - 1.0)   # in (0, 1)
    frac = min(max(frac, 0.0), 1.0)
    if str(shape) == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * frac))         # cosine 1 -> 0
    return 1.0 - frac                                         # linear 1 -> 0


def _adam_bias_correction_for(adam_beta2: float) -> bool:
    """#224 Wave C FIX-1 (LAUNCH-BLOCKER): MLX ``optim.AdamW`` ``bias_correction`` DEFAULTS FALSE.

    Without bias correction, at step ``t`` the second-moment ``v`` is ``(1-beta2) * mean(g^2)`` and is
    NOT divided by ``(1-beta2^t)``, so ``sqrt(v)`` is ``~sqrt(1-beta2)`` too small early. With the
    all-levers small-n beta2 (0.9999999, 1-beta2=1e-7) that is ``sqrt(1e-7)/sqrt(1e-3) ~ 316/31.6 ~
    10`` smaller than the 0.999 default => the step-1 effective LR blows up ~100x (measured ratio 99.99x)
    => AdamW random-walk / divergence. The arXiv small-n derivation (1-beta2 <~ (1-beta1^5)/n^3.5) is
    faithful ONLY with bias correction (which makes vhat = v/(1-beta2^t) => step-1 update ~ lr*sign(g)
    independent of beta2). So bias correction is REQUIRED on the high-beta2 path.

    Gate ON only OFF THE DEFAULT (adam_beta2 != 0.999). At 0.999 (== the MLX/proven_base default) we
    keep ``bias_correction`` at the MLX default (False) so the DEFAULT AdamW construction is BYTE-
    IDENTICAL to the pre-FIX-1 path (the --adam-beta2 default stays 0.999 => byte-identical-off 7/7).
    Pure + total => $0 unit-testable. Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
    against"."""
    return abs(float(adam_beta2) - 0.999) > 1e-9


def _seg_form_for_epoch(ep: int, args) -> str:
    if not args.curriculum:
        return args.seg_loss
    if ep < args.tau_softplus_start_epoch:
        return "ce"
    if ep < args.l7_start_epoch:
        return "tau_softplus"
    return "l7_softplus"


def _stage_converged(
    stage_epochs: "list[int]",
    stage_losses: "list[float]",
    *,
    min_stage_epochs: int,
    plateau_rel_eps: float,
    plateau_windows: int,
) -> bool:
    """(#292 build-2) DETERMINISTIC convergence test for an event-triggered curriculum transition.

    Returns ``True`` iff BOTH hold:
      (a) at least ``min_stage_epochs`` COMPLETED epochs have elapsed in the current stage
          (``len(stage_epochs) >= min_stage_epochs``), AND
      (b) the synchronous per-epoch training loss has PLATEAUED: the least-squares slope of the last
          ``plateau_windows`` within-stage ``ep_loss`` values, NORMALIZED by that window's mean (a
          scale-free RELATIVE slope), has magnitude ``<= plateau_rel_eps``.

    PURE (numpy only; NO MLX, NO async d_seg verdict, NO wall-clock) => identical inputs give the
    identical bool => the fired transition epoch is a deterministic function of the seeded loss
    trajectory. A FALLING loss (slope << 0 relative to the level) does NOT plateau => ``False`` (the
    stage keeps training). Mirrors ``_scheduled_eikonal_weight``'s discipline one layer up: the
    CONTROLLER, not the schedule, is byte-clean and unit-testable. Cf. #292 build 1 (eikonal
    STEP-ramp) + the CE->tau transition analysis (CE terminal convergence ~-4.4e-7/ep => a plateau)."""
    n = len(stage_epochs)
    if n < max(1, int(min_stage_epochs)):
        return False
    w = max(2, int(plateau_windows))
    if len(stage_losses) < w:
        return False
    y = np.asarray(stage_losses[-w:], dtype=np.float64)
    if not np.all(np.isfinite(y)):
        return False
    x = np.arange(w, dtype=np.float64)
    xm = float(x.mean())
    ym = float(y.mean())
    denom = float(((x - xm) ** 2).sum())
    if denom <= 0.0:
        return False
    slope = float(((x - xm) * (y - ym)).sum() / denom)   # ep_loss per epoch (deterministic closed form)
    mean_mag = abs(ym)
    if mean_mag <= 0.0:
        return slope == 0.0                              # degenerate zero-mean window: flat-only
    return abs(slope / mean_mag) <= float(plateau_rel_eps)


def _evt_current_stage_form(state: dict) -> str:
    """(#292 build-2) The seg_form of the stage the event-triggered controller is CURRENTLY in, read
    from resolved boundaries WITHOUT firing (no mutation). CE until ``tau`` resolves, then
    ``tau_softplus`` until ``l7`` resolves, then ``l7_softplus``."""
    if state.get("tau") is None:
        return "ce"
    if state.get("l7") is None:
        return "tau_softplus"
    return "l7_softplus"


def _evt_resolve_seg_form(ep: int, state: dict, args) -> "tuple[str, dict | None]":
    """(#292 build-2) Event-triggered effective seg_form at 1-based epoch ``ep`` + boundary resolution.

    Mutates ``state`` IN PLACE: fills ``state['tau']`` / ``state['l7']`` the FIRST epoch the current
    stage either (i) CONVERGES per ``_stage_converged`` on the within-stage loss history
    ``state['losses']``, OR (ii) reaches its hardcoded cap (``--tau-softplus-start-epoch`` /
    ``--l7-start-epoch``) -- whichever is EARLIER. The cap is a HARD CEILING: the trigger NEVER fires
    LATER than the OFF schedule, so an event-triggered run whose loss never plateaus reproduces the
    hardcoded ``_seg_form_for_epoch`` schedule EXACTLY. On firing, ``stage_start`` is reset to ``ep``
    and ``losses`` cleared so the next stage's convergence is judged on its OWN history only.

    Returns ``(seg_form, event)`` where ``event`` is ``None`` (no transition this epoch) or a JSON-
    ready dict describing the fired transition (deterministic log). PURE w.r.t. ``(ep, state, args)``:
    NO MLX, NO async verdict, NO wall-clock => same seeded loss trajectory (captured in
    ``state['losses']``) => same fired epochs. Only reads PAST epochs' losses (the caller appends the
    current epoch's ``ep_loss`` at the END of the epoch).

    (#302) When ``--curriculum-nucleus-guard`` is on, the CE->tau CONVERGENCE fire ALSO requires the
    MEASURED ``state['nucleus_ready']`` (updated at verdict cadence — the critical-nucleus law: MCF
    erodes a below-nucleus class, never grows it). This makes the CE->tau fired epoch depend on the
    verdict cadence (a MEASURED trigger, not pure-on-losses) — but the CAP still fires
    unconditionally (ceiling fallback => the run never hangs), and ``nucleus_ready`` is persisted in
    the resume sidecar so an ON-resume is bit-faithful. Guard OFF => ``nucleus_ready`` stays True =>
    the fire is pure-on-losses exactly as before (byte-identical)."""
    mse = int(getattr(args, "curriculum_min_stage_epochs", 150))
    reps = float(getattr(args, "curriculum_plateau_rel_eps", 1e-3))
    win = int(getattr(args, "curriculum_plateau_windows", 4))
    n = len(state["losses"])
    stage_epochs = list(range(int(state["stage_start"]), int(state["stage_start"]) + n))

    def _fire(boundary_key: str, cap: int, from_form: str, to_form: str, *,
              nucleus_gate: bool = False):
        converged = _stage_converged(
            stage_epochs, state["losses"],
            min_stage_epochs=mse, plateau_rel_eps=reps, plateau_windows=win)
        # (#302) MEASURED nucleus gate on the CE->tau hand-off: a plateau is NECESSARY but NOT
        # SUFFICIENT — hold in the from-stage until every scored class is above its critical nucleus.
        # The CAP still fires unconditionally below (ceiling fallback). Guard OFF => nucleus_gate
        # False => this never trips => byte-identical.
        nucleus_held = bool(nucleus_gate and converged
                            and not bool(state.get("nucleus_ready", True)))
        if nucleus_held:
            converged = False
        hit_cap = (int(cap) > 0 and ep >= int(cap))
        if not (converged or hit_cap):
            return from_form, None
        state[boundary_key] = int(ep)
        state["stage_start"] = int(ep)
        state["losses"] = []
        return to_form, {
            "stage": "curriculum_transition_fired", "from": from_form, "to": to_form,
            "epoch": int(ep),
            "trigger": ("loss_plateau" if converged else "cap"),
            "nucleus_gated": bool(nucleus_gate),
            "nucleus_ready": bool(state.get("nucleus_ready", True))}

    if state.get("tau") is None:
        return _fire("tau", int(args.tau_softplus_start_epoch), "ce", "tau_softplus",
                     nucleus_gate=bool(getattr(args, "curriculum_nucleus_guard", False)))
    # (C2 SEAL-review guard, 4bf533cab) l7 is a MEASURED DEFECT stage demoted from the default
    # curriculum by setting --l7-start-epoch >= --epochs ("never"). The convergence trigger must
    # HONOR that intent: never converge-fire l7 when its cap says never (else a tau plateau —
    # measured rel slope -4.7e-4 on #205 — would fire the defect stage mid-run).
    if int(args.l7_start_epoch) >= int(args.epochs):
        return "tau_softplus", None
    if state.get("l7") is None:
        return _fire("l7", int(args.l7_start_epoch), "tau_softplus", "l7_softplus")
    return "l7_softplus", None


# ── (#302 curriculum-derivation) PER-CLASS CRITICAL-NUCLEUS GUARD + BOUNDARY RE-ANCHOR ───────────
# The CE->tau hand-off law (T3 symposium 2026-07-05 §B.2; canonical equation
# ``curriculum_handoff_critical_nucleus_v1``): the tau stage is sharp-limit mean-curvature flow, and
# Allen-Cahn's critical-nucleus theorem says any scored class-region BELOW its critical size is
# ERASED, never grown (MEASURED: #205 seeded a lane at part_frac 0 -> d_seg CREPT 0.004752@ep300 ->
# 0.006568@ep400 while smooth-loss fell; Muon/MCF cannot nucleate a zero-mass class). Therefore the
# recalibrated plateau trigger (``_stage_converged``, eps 1e-4) is NECESSARY BUT NOT SUFFICIENT:
# CE->tau is admissible only when EVERY scored class is ALSO above its nucleus (born + partition
# formed), else MCF is handed a half-formed partition to erode. The functions below are the guard.
#
# ALL PURE (numpy on argmax int arrays; NO MLX / model / async verdict / wall-clock) => same argmax
# inputs give the same booleans, unit-testable. DEFAULT-OFF: the guard is consulted by the event
# trigger ONLY when --curriculum-nucleus-guard is set; the readiness telemetry row is emitted
# passively (observability-first, so the NEXT run yields validation data even with the trigger OFF)
# but is NEVER read back into training/parity => byte-identical to the #205 path.
#
# CANONICAL CLASS ORDER (CLAUDE.md NON-NEGOTIABLE; NOT a luma-sort of class_values):
#   0=Road 1=Lane 2=Undrivable(incl sky) 3=Movable(cars) 4=MyCar(ego hood).
N_SCORED_CLASSES = 5


def _evt_nucleus_counts(realized_argmax_list: list, gt_argmax_list: list,
                        n_classes: int = N_SCORED_CLASSES) -> dict:
    """Raw per-class pixel counts over a batch of realized/GT argmax maps (the COUNTS interchange
    format — chunk-additive so the chunked verdict can accumulate without holding all argmax).

    ``realized_argmax_list`` / ``gt_argmax_list``: matching lists of (h,w) int argmax maps (the
    frozen CPU-torch SegNet argmax of the R-rendered witness frame1 and the GT ``lstars`` — the SAME
    surfaces the d_seg verdict consumes). Returns ``{"total_px", "pred_px":[...], "gt_px":[...],
    "wrong_px":[...], "n_classes"}``. VERBATIM the arithmetic of
    ``tools/witness_per_stage_annulus_attribution.stage_stats`` (reused, not reinvented). Pure."""
    import numpy as _np
    pred_px = [0] * n_classes
    gt_px = [0] * n_classes
    wrong_px = [0] * n_classes
    total_px = 0
    for am, g in zip(realized_argmax_list, gt_argmax_list):
        am = _np.asarray(am).astype(_np.int64)
        g = _np.asarray(g).astype(_np.int64)
        wrong = am != g
        total_px += int(g.size)
        for c in range(n_classes):
            pred_px[c] += int((am == c).sum())
            gt_px[c] += int((g == c).sum())
            wrong_px[c] += int((wrong & (g == c)).sum())
    return {"total_px": total_px, "pred_px": pred_px, "gt_px": gt_px,
            "wrong_px": wrong_px, "n_classes": n_classes}


def _evt_counts_add(a: "dict | None", b: dict) -> dict:
    """Accumulate two ``_evt_nucleus_counts`` dicts (chunk-additive). ``a is None`` => a copy of ``b``."""
    if a is None:
        return {"total_px": int(b["total_px"]), "pred_px": list(b["pred_px"]),
                "gt_px": list(b["gt_px"]), "wrong_px": list(b["wrong_px"]),
                "n_classes": int(b["n_classes"])}
    nc = int(b["n_classes"])
    return {"total_px": int(a["total_px"]) + int(b["total_px"]), "n_classes": nc,
            "pred_px": [int(a["pred_px"][c]) + int(b["pred_px"][c]) for c in range(nc)],
            "gt_px": [int(a["gt_px"][c]) + int(b["gt_px"][c]) for c in range(nc)],
            "wrong_px": [int(a["wrong_px"][c]) + int(b["wrong_px"][c]) for c in range(nc)]}


def _evt_nucleus_stats(counts: dict) -> "dict[int, dict]":
    """Per-class partition-fraction + within-class flip rate from accumulated ``counts``.

    For each class ``c``:
      * ``part_frac[c]``   = pred_px[c] / total_px (the witness's PREDICTED partition area for ``c``;
        part_frac == 0 => a zero-mass class MCF cannot nucleate — the binding nucleus gate).
      * ``within_flip[c]`` = wrong_px[c] / gt_px[c] (per-class disagreement; measured against ``c``'s
        true support, not per-frame-averaged, so a rare class's flip rate is honest).
    Returns ``{c: {"part_frac","within_flip","gt_px","pred_px"}}``. Pure."""
    nc = int(counts["n_classes"])
    total = int(counts["total_px"])
    out: dict[int, dict] = {}
    for c in range(nc):
        gt = int(counts["gt_px"][c])
        out[c] = {
            "part_frac": (int(counts["pred_px"][c]) / total) if total else 0.0,
            "within_flip": (int(counts["wrong_px"][c]) / gt) if gt else 0.0,
            "gt_px": gt, "pred_px": int(counts["pred_px"][c]),
        }
    return out


def _evt_nucleus_satisfied(stats: "dict[int, dict]", within_flip_thresh: float,
                           min_part_frac: float = 0.0) -> "tuple[dict[int, bool], bool]":
    """Per-class nucleus satisfaction + the all-classes AND.

    Class ``c`` is ABOVE nucleus iff ``part_frac[c] > min_part_frac`` (BORN — nonzero predicted mass,
    the MCF-cannot-nucleate gate) AND ``within_flip[c] <= within_flip_thresh`` (FORMED — the partition
    for ``c`` is settled below the flip threshold). A class with ``gt_px == 0`` in the batch is
    VACUOUSLY satisfied (not scored on this batch — never blocks the hand-off). Returns
    ``(per_class_bool, all_ok)``."""
    per_class: dict[int, bool] = {}
    for c, s in stats.items():
        if int(s.get("gt_px", 0)) == 0:
            per_class[c] = True                     # unscored on this batch => vacuously above nucleus
            continue
        per_class[c] = (float(s["part_frac"]) > float(min_part_frac)
                        and float(s["within_flip"]) <= float(within_flip_thresh))
    return per_class, all(per_class.values())


def _evt_readiness_row(ep: int, seg_form: str, stats: "dict[int, dict]",
                       satisfied: "dict[int, bool]", nucleus_all_ok: bool, plateau_ok: bool,
                       within_flip_thresh: float, min_part_frac: float,
                       guard_active: bool) -> dict:
    """The deterministic ``handoff_readiness`` telemetry row (JSON-ready). Emitted per verdict even
    when the CE->tau trigger is OFF (observability-first => the NEXT run passively yields the
    per-class validation data the hand-off law needs). PURE telemetry — NEVER read back into
    training/parity/resume. ``ready`` = plateau_ok AND nucleus_all_ok (the full readiness predicate
    of the hand-off law)."""
    return {
        "stage": "handoff_readiness", "epoch": int(ep), "seg_form": str(seg_form),
        "plateau_ok": bool(plateau_ok), "nucleus_all_ok": bool(nucleus_all_ok),
        "ready": bool(plateau_ok and nucleus_all_ok), "guard_active": bool(guard_active),
        "within_flip_thresh": float(within_flip_thresh), "min_part_frac": float(min_part_frac),
        "per_class": {str(c): {"part_frac": round(float(stats[c]["part_frac"]), 6),
                               "within_flip": round(float(stats[c]["within_flip"]), 6),
                               "gt_px": int(stats[c]["gt_px"]),
                               "nucleus_ok": bool(satisfied[c])}
                      for c in sorted(stats)},
    }


def _evt_reanchor_epoch(ep: int, boundary_fired: "int | None",
                        boundary_hardcoded: int) -> int:
    """(#302 M1) Map a real epoch into the schedule frame a TAU-RELATIVE wall-clock lever was
    CALIBRATED in, so its boundary-relative event tracks the ACTUAL FIRED tau boundary.

    A lever calibrated so its key event lands at ``boundary_hardcoded`` (the fixed
    ``--tau-softplus-start-epoch``, default 300) should, under event-triggering, land at the fired
    boundary ``boundary_fired`` instead. Feeding the lever the SHIFTED epoch
    ``ep + (boundary_hardcoded - boundary_fired)`` moves the lever's event to ``boundary_fired`` while
    PRESERVING the schedule's shape + length (a shift, not a rescale). Worked: tau calibrated @300,
    fires @200 => shift +100 => a lever completing at virtual 300 completes at real 200; the analytic
    band starting at virtual 350 (=tau+50) starts at real 250 (=fired+50). ``boundary_fired is None``
    (unfired) OR == ``boundary_hardcoded`` (fired exactly at the cap) => returns ``ep`` UNCHANGED
    (the byte-identity contract: OFF, unfired, and fire-at-cap all leave every lever bit-for-bit as
    the #205 path). Pure; unit-tested. Mirrors ``_scheduled_eikonal_weight``'s ``step_epoch``
    re-anchoring one layer up (eikonal was already re-anchored; this completes the tau-relative set:
    persistence-warmup + seed-anneal + analytic-band). hosc-beta is NOT re-anchored here — its beta=4
    freeze is anchored to the MUON boundary, which stays a fixed cap until the Muon-event-trigger
    BUILD (symposium §C.ii item 5); re-anchoring it to tau would mis-place the freeze point."""
    if boundary_fired is None or int(boundary_fired) == int(boundary_hardcoded):
        return int(ep)
    return int(ep) + (int(boundary_hardcoded) - int(boundary_fired))


def _scheduled_eikonal_weight(ep: int, args, step_epoch: "int | None" = None) -> float:
    """(#292 transition-analysis) Per-epoch eikonal weight — the eikonal STEP-ramp control lever.

    ``step_epoch`` (SEAL fix, 2026-07-04 pre-launch review): the ACTUAL tau/MCF onset epoch to step
    at. Default ``None`` == the hardcoded ``--tau-softplus-start-epoch`` (BYTE-IDENTICAL to the
    original build-1 behavior — the OFF/#205 path). With ``--curriculum-event-triggered`` ON the
    caller passes the RESOLVED boundary (the event-fired epoch, or a large sentinel while unfired)
    so the survival step tracks the REAL transition — without this, an early event-fired tau would
    run mean-curvature flow at BASE eikonal until the hardcoded cap (the exact survival window the
    ramp exists to protect).

    BYTE-IDENTICAL constant ``--eikonal-weight`` unless ``--eikonal-weight-end`` is set != base:
    then STEP base -> end at the tau/MCF onset (``--tau-softplus-start-epoch``, the same boundary
    ``_seg_form_for_epoch`` uses), cosine-eased over ``--stage-transition-rewarmup-epochs``.
    Rationale (MEASURED, gt_n6 survival probe): CE holds a valid SDF at eikonal 0.05; at the tau
    onset mean-curvature flow narrows the interface (half-width tau/2) toward sigma1.5 / 49% lane
    survival, so raise the unit-gradient enforcement (knee ~0.10) to keep the thin lane a valid
    unit-gradient SDF (sigma0.8 / 93% survival). With ``--softmax-temp-end 1.0`` (the measured
    resolution floor) no inverse-tau tracking is needed; only this MCF-onset step is load-bearing.
    Mirrors ``_hosc_beta_for_epoch``'s byte-identical-when-unset contract."""
    base = float(args.eikonal_weight)
    end_raw = getattr(args, "eikonal_weight_end", None)
    if end_raw is None:
        return base
    end = float(end_raw)
    if end == base or not getattr(args, "curriculum", False):
        return base
    step_ep = int(args.tau_softplus_start_epoch) if step_epoch is None else int(step_epoch)
    if step_ep <= 0 or ep < step_ep:
        return base
    ease = int(getattr(args, "stage_transition_rewarmup_epochs", 0) or 0)
    if ease <= 0 or ep >= step_ep + ease:
        return end
    frac = (ep - step_ep) / float(ease)                 # 0..1 across the ease window
    w = 0.5 * (1.0 - float(np.cos(np.pi * frac)))        # cosine 0->1 (same map as hosc-beta cosine)
    return base + (end - base) * w


# ── (#292 build-3) CLOSED-LOOP LEVER CONTROL — pure, deterministic, default-OFF ──────────────────
# Classification sentinels + slope/persistence math MUST MATCH tools/witness_control_monitor.py::
# classify_trajectory (the monitor is the read-only sibling; this is the in-run actuator). Replicated
# INLINE (not imported) because the trainer's entry point runs with sys.path[0]=experiments/ (no
# repo-root `tools` package on the live #205 launch path); parity is regression-guarded by
# experiments/test_closed_loop_control.py which importlib-loads BOTH and cross-checks classifications.
_CL_DIVERGING_ERASING = "diverging_erasing"
_CL_TRANSITION_TRANSIENT = "transition_transient"
_CL_VOLATILE = "volatile"
_CL_PLATEAU = "plateau"
_CL_CONVERGING = "converging"


def _cl_lstsq_slope(xs: "list[float]", ys: "list[float]") -> float:
    """Least-squares slope dy/dx (0.0 if <2 points or zero x-spread). VERBATIM replica of
    tools/witness_control_monitor._lstsq_slope (pure, numpy-free => identical floats)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx_ = sum(xs) / n
    my_ = sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0
    sxy = sum((x - mx_) * (y - my_) for x, y in zip(xs, ys))
    return sxy / sxx


def _cl_classify(
    verdicts: "list[dict]", *, window: int = 5, creep_eps: float = 1e-6,
    plateau_eps: float = 5e-7, volatile_cv: float = 0.5, min_sustained_windows: int = 3,
) -> "dict":
    """(#292 build-3) Classify the CURRENT within-stage d_seg trajectory. The classification core of
    tools/witness_control_monitor.classify_trajectory, replicated EXACTLY (same falling-rule order,
    same within-stage filtering on seg_form, same sustained-vs-transient persistence test): sustained
    DIVERGING_ERASING requires the rise to persist >= min_sustained_windows within-stage verdicts AND
    the net-stage slope > 0 — distinct from a recoverable TRANSITION_TRANSIENT (boundary shock).
    PURE (no numpy/MLX/wall-clock): same verdict history => same classification. Each verdict row:
    {"epoch": int, "seg_form": str, "d_seg": float, "ep_loss": float}."""
    if not verdicts:
        raise ValueError("no verdicts to classify")
    latest_stage = str(verdicts[-1].get("seg_form", ""))
    same = [v for v in verdicts if str(v.get("seg_form", "")) == latest_stage]
    win = same[-int(window):] if window > 0 else same
    eps_x = [float(v["epoch"]) for v in win]
    dsegs = [float(v["d_seg"]) for v in win]
    losses = [float(v.get("ep_loss", 0.0)) for v in win]
    d_slope = _cl_lstsq_slope(eps_x, dsegs)
    l_slope = _cl_lstsq_slope(eps_x, losses)
    mean_ds = sum(dsegs) / len(dsegs)
    if len(dsegs) >= 2 and mean_ds > 0:
        var = sum((d - mean_ds) ** 2 for d in dsegs) / len(dsegs)
        cv = (var ** 0.5) / mean_ds
    else:
        cv = 0.0
    # Persistence: NET slope since the stage started (transient rises recover; erosion doesn't).
    full_slope = _cl_lstsq_slope([float(v["epoch"]) for v in same], [float(v["d_seg"]) for v in same])
    n_stage = len(same)
    if d_slope > creep_eps and l_slope < 0.0:
        sustained = (n_stage >= min_sustained_windows) and (full_slope > 0.0)
        cls = _CL_DIVERGING_ERASING if sustained else _CL_TRANSITION_TRANSIENT
    elif cv > volatile_cv:
        cls = _CL_VOLATILE
    elif abs(d_slope) <= plateau_eps:
        cls = _CL_PLATEAU
    else:
        cls = _CL_CONVERGING
    return {"classification": cls, "d_seg_slope": d_slope, "ep_loss_slope": l_slope,
            "net_stage_slope": full_slope, "n_stage": n_stage, "d_seg_cv": cv}


def _cl_effective_eikonal(scheduled: float, bump_add: float, eikonal_max: float) -> float:
    """(#292 build-3) Effective eikonal weight = the build-1 schedule PLUS the bounded closed-loop
    bump: ``min(scheduled + bump_add, eikonal_max)`` — with the cap floored at ``scheduled`` so a
    mis-set max can NEVER pull the weight BELOW the schedule (bounded ABOVE, never below).
    ``bump_add <= 0.0`` returns ``scheduled`` EXACTLY (the byte-identity contract: OFF, or ON with
    no bump fired, is bit-for-bit ``_scheduled_eikonal_weight``). Pure; unit-tested."""
    if bump_add <= 0.0:
        return float(scheduled)
    return float(min(scheduled + bump_add, max(eikonal_max, scheduled)))


def _cl_step(classification: str, state: dict, ep: int, *,
             bump: float, max_bumps: int, stop_after: int) -> "dict | None":
    """(#292 build-3) ONE deterministic controller transition at an eval point. Mutates ``state``
    IN PLACE; returns a JSON-ready action event or ``None`` (no action).

    Policy (Tier-3 "ramp eikonal on creep" + "early termination"):
      * classification != DIVERGING_ERASING  => NO action; the post-budget stop countdown RESETS
        (erosion must PERSIST consecutively to stop — a recovered window breaks persistence).
      * SUSTAINED DIVERGING_ERASING with bump budget left => BOUNDED eikonal bump: bumps += 1,
        bump_add += bump (total add bounded by max_bumps*bump; the APPLICATION is further capped at
        --closed-loop-eikonal-max by _cl_effective_eikonal).
      * SUSTAINED DIVERGING_ERASING with budget SPENT => count consecutive post-budget erosion
        windows; at >= stop_after, arm ``stop_epoch`` (clean early-stop; best ckpt already preserved
        continuously by _maybe_preserve_best). The controller NEVER launches anything (CONTAINMENT):
        it only mutates the in-run eikonal weight + arms the stop flag.
    PURE in (classification, state, ep, params): same inputs => same mutation => same action."""
    if state.get("stop_epoch") is not None:
        return None                                     # already armed; terminal
    if classification != _CL_DIVERGING_ERASING:
        state["post_budget_windows"] = 0                # persistence broken => reset countdown
        return None
    if state["bumps"] < int(max_bumps) and float(bump) > 0.0:
        state["bumps"] = int(state["bumps"]) + 1
        state["bump_add"] = float(state["bump_add"]) + float(bump)
        state["post_budget_windows"] = 0
        return {"action": "eikonal_bump", "bumps_used": state["bumps"],
                "bump_add": round(state["bump_add"], 6)}
    state["post_budget_windows"] = int(state["post_budget_windows"]) + 1
    if state["post_budget_windows"] >= max(1, int(stop_after)):
        state["stop_epoch"] = int(ep)
        return {"action": "early_stop", "stop_epoch": int(ep),
                "post_budget_windows": state["post_budget_windows"]}
    return {"action": "stop_countdown", "post_budget_windows": state["post_budget_windows"],
            "stop_after": int(stop_after)}


_CL_PEND_SHADOW_PREFIX = "__cl_pend_shadow."


def _cl_state_arrays(state: dict, verdicts: "list[dict]") -> "dict[str, np.ndarray]":
    """(#292 build-3) Closed-loop controller state -> resume-sidecar arrays (mirrors the build-2
    __evt_* pattern). Written ONLY when --closed-loop-control is ON (the caller passes None when OFF
    => ZERO new sidecar keys => byte-identical sidecar, the #205-safe path). Persists the bump/stop
    state AND the captured verdict history so an ON-run --resume-from classifies from the SAME
    within-stage trajectory a continuous run would (bit-faithful resume). MLX-free.

    (M2 fix, decide-on-previous) When ``state["pending"]`` is set — an async verdict is IN FLIGHT at
    sidecar-write time, so its row is absent from ``verdicts`` — also persist the PENDING-VERDICT
    record: the epoch/seg_form/ep_loss the row will carry plus the EXACT point-in-time snapshot the
    worker thread is scoring (fp32 shadow arrays + softmax_temp + hosc_beta, ~hundreds of KB). A
    resume recomputes the verdict SYNCHRONOUSLY from these inputs (the verdict is deterministic
    given them) => the recomputed row is bit-identical to what the continuous run's thread produced
    => post-resume decisions match the continuous run EXACTLY. No pending (None/absent) => ZERO
    ``__cl_pend_*`` keys (the pre-M2 sidecar shape, byte-identical)."""
    _se = state.get("stop_epoch")
    out = {
        "__cl_bumps": np.asarray(int(state.get("bumps", 0))),
        "__cl_bump_add": np.asarray(float(state.get("bump_add", 0.0))),
        "__cl_post_budget_windows": np.asarray(int(state.get("post_budget_windows", 0))),
        "__cl_stop_epoch": np.asarray(-1 if _se is None else int(_se)),
        "__cl_v_epochs": np.asarray([int(v["epoch"]) for v in verdicts], np.int64),
        "__cl_v_dseg": np.asarray([float(v["d_seg"]) for v in verdicts], np.float64),
        "__cl_v_eploss": np.asarray([float(v.get("ep_loss", 0.0)) for v in verdicts], np.float64),
        "__cl_v_segform": np.asarray([str(v.get("seg_form", "")) for v in verdicts]),
    }
    pend = state.get("pending")
    if pend is not None:
        out["__cl_pend_epoch"] = np.asarray(int(pend["epoch"]))
        out["__cl_pend_segform"] = np.asarray(str(pend["seg_form"]))
        out["__cl_pend_eploss"] = np.asarray(float(pend["ep_loss"]), np.float64)
        out["__cl_pend_temp"] = np.asarray(float(pend["softmax_temp"]), np.float64)
        out["__cl_pend_beta"] = np.asarray(float(pend["hosc_beta"]), np.float64)
        for k, v in pend["ema_np"].items():
            out[_CL_PEND_SHADOW_PREFIX + k] = np.asarray(v, np.float32)
    return out


def _cl_restore_from_cfg(cfg: dict) -> "tuple[dict, list[dict]] | None":
    """(#292 build-3) Inverse of _cl_state_arrays through the _load_resume_state cfg parse
    (``a.item() if a.size == 1 else a.tolist()`` => scalars OR lists). Returns (state, verdicts) or
    ``None`` when the sidecar predates the feature / was written with closed-loop OFF (the caller
    then starts fresh — deterministic going forward, mirroring the build-2 cap-fallback honesty)."""
    if "__cl_bumps" not in cfg:
        return None

    def _as_list(x) -> list:
        return x if isinstance(x, list) else [x]

    _se = int(cfg.get("__cl_stop_epoch", -1))
    state = {
        "bumps": int(cfg["__cl_bumps"]),
        "bump_add": float(cfg.get("__cl_bump_add", 0.0)),
        "post_budget_windows": int(cfg.get("__cl_post_budget_windows", 0)),
        "stop_epoch": None if _se < 0 else _se,
    }
    eps_l = _as_list(cfg.get("__cl_v_epochs", []))
    ds_l = _as_list(cfg.get("__cl_v_dseg", []))
    el_l = _as_list(cfg.get("__cl_v_eploss", []))
    sf_l = _as_list(cfg.get("__cl_v_segform", []))
    verdicts = [{"epoch": int(e), "d_seg": float(d), "ep_loss": float(l), "seg_form": str(s)}
                for e, d, l, s in zip(eps_l, ds_l, el_l, sf_l)]
    return state, verdicts


def _cl_pending_from_cfg(cfg: dict) -> dict | None:
    """(M2 fix, decide-on-previous) Parse the PENDING-VERDICT record written by _cl_state_arrays
    through the _load_resume_state cfg parse (``a.item() if a.size == 1 else a.tolist()``). Returns
    ``None`` when no verdict was in flight at sidecar-write time (or a pre-M2/OFF sidecar) — the
    caller then reconciles nothing. Shadow arrays round-trip EXACTLY: fp32 -> tolist (float64,
    exact superset) -> back to fp32; a size-1 array flattens to a scalar through ``.item()`` so the
    reconcile RESHAPES each array against the restored shadow's shapes (lossless). MLX-free."""
    if "__cl_pend_epoch" not in cfg:
        return None
    shadow: dict[str, np.ndarray] = {}
    for k, v in cfg.items():
        if k.startswith(_CL_PEND_SHADOW_PREFIX):
            shadow[k[len(_CL_PEND_SHADOW_PREFIX):]] = np.asarray(v, np.float32)
    return {"epoch": int(cfg["__cl_pend_epoch"]),
            "seg_form": str(cfg["__cl_pend_segform"]),
            "ep_loss": float(cfg["__cl_pend_eploss"]),
            "softmax_temp": float(cfg["__cl_pend_temp"]),
            "hosc_beta": float(cfg["__cl_pend_beta"]),
            "ema_np": shadow}


def _hosc_beta_for_epoch(ep: int, args) -> float | None:
    """(FEED-fb) Annealed hosc ``beta`` at 1-based epoch ``ep``, or ``None`` when NO anneal applies.

    Returns ``None`` (caller leaves ``model.hosc_beta`` UNTOUCHED => BIT-IDENTICAL constant-beta path)
    when: activation != ``hosc``, OR ``--hosc-beta-end`` is unset, OR end == start. Otherwise anneals
    ``beta`` from ``--hosc-beta`` (at ep==1) to ``--hosc-beta-end`` (at ep==args.epochs) on a linear
    (default) or cosine schedule. The step-native L-infinity-optimal lever: ``beta -> inf`` makes
    ``tanh(beta*sin)`` approach a step (the topology-matched chart for the piecewise-constant argmax,
    no Gibbs). Pure (no model/MLX); unit-tested. Mirrors ``_seg_form_for_epoch``.
    """
    if (getattr(args, "activation", None) != "hosc"
            or getattr(args, "hosc_beta_end", None) is None
            or args.hosc_beta_end == args.hosc_beta):
        return None
    # (review C2) same anneal denominator as _softmax_temp_for_epoch: --anneal-epochs (schedule
    # length) NOT --epochs (run length). Default None => args.epochs => BIT-IDENTICAL.
    _ae = getattr(args, "anneal_epochs", None) or args.epochs
    prog = (ep - 1) / max(_ae - 1, 1)
    if getattr(args, "hosc_beta_anneal", "linear") == "cosine":
        return float(args.hosc_beta_end + 0.5 * (args.hosc_beta - args.hosc_beta_end) * (1 + np.cos(np.pi * prog)))
    return float(args.hosc_beta + (args.hosc_beta_end - args.hosc_beta) * prog)


def _softmax_temp_for_epoch(ep: int, args) -> float:
    """(config-review #4) Cosine-annealed softmax temperature at 1-based epoch ``ep`` (hi->lo: soft
    start so gradients flow with no RGB-level Gibbs -> sharp end with the SDF partition pinned). Pure
    (no model/MLX); unit-tested. Mirrors ``_seg_form_for_epoch`` / ``_hosc_beta_for_epoch``. Extracted
    from the inline loop anneal so the MUON FINISHER can FREEZE it at the muon-start value (FEED-fm).
    Returns the EXACT value the pre-extraction inline formula produced (BIT-IDENTICAL) when
    --anneal-epochs is unset.

    (review C2) ANNEAL DENOMINATOR: the cosine progress uses ``--anneal-epochs`` (the SCHEDULE length)
    NOT ``--epochs`` (the run length). Default None => falls back to ``args.epochs`` => BIT-IDENTICAL.
    A WARM-START arm (resume the CE ckpt @ ep299, run 100 epochs => --epochs 399) must set
    --anneal-epochs to the ORIGINAL schedule length (1500) so ep300->400 reproduces the DISEASE
    regime temp (~0.91->0.84), not the schedule tail (~0.19->0.05). ``None or x == x`` and 0 is
    treated as unset, so the default path is the pre-C2 formula bit-for-bit.

    (THETA* TIER-2 MUST-1) ``--tau-anneal-shape`` selects the homotopy/continuation curve tau(ep) walks
    from ``softmax_temp_start`` -> ``softmax_temp_end`` (the anneal denominator stays --anneal-epochs):
      * ``cosine``      (DEFAULT) the pre-theta* cosine. BIT-IDENTICAL to the inline formula.
      * ``geometric``   log-spaced (exponential) decay tau = start*(end/start)**prog == start**(1-prog)
                        * end**prog -> spends MORE epochs at small tau (slows the near-tau->0
                        continuation step that drives the measured late-tau d_seg volatility). Requires
                        start>0, end>0 (guarded in main()).
      * ``cosine_hold`` cosine that reaches the floor at ``--tau-hold-frac`` of the window, then HOLDS
                        at ``softmax_temp_end``. ``--tau-hold-frac 1.0`` (DEFAULT) == NO hold == the
                        cosine branch (BIT-IDENTICAL: prog/1.0==prog exactly for finite IEEE floats and
                        hold_frac>=1.0 routes through the SAME final cosine line below).
    Returns the EXACT value the pre-theta* inline cosine produced when shape=='cosine' (or
    'cosine_hold' with hold_frac>=1.0) -- the #1 bit-identical-when-off gate. ``float(args.x) == args.x``
    for the argparse floats, so the named locals do not perturb the arithmetic."""
    _ae = getattr(args, "anneal_epochs", None) or args.epochs
    prog_t = (ep - 1) / max(_ae - 1, 1)
    shape = str(getattr(args, "tau_anneal_shape", "cosine"))
    start = float(args.softmax_temp_start)
    end = float(args.softmax_temp_end)
    if shape == "geometric":
        # log-spaced (exponential) decay; endpoints are exact at prog 0/1. main() guards start>0,end>0.
        return float(start * (end / start) ** prog_t)
    if shape == "cosine_hold":
        hold_frac = float(getattr(args, "tau_hold_frac", 1.0))
        if hold_frac < 1.0:
            if prog_t >= hold_frac:
                return end                       # held at the floor for the tail of the window
            prog_t = prog_t / hold_frac          # rescale [0,hold_frac)->[0,1); falls through to cosine
        # hold_frac>=1.0: NO hold -> fall through with the ORIGINAL prog_t -> BIT-IDENTICAL cosine.
    # DEFAULT cosine (and cosine_hold w/ hold_frac>=1.0): the pre-theta* inline formula, unchanged.
    return float(end + 0.5 * (start - end) * (1 + np.cos(np.pi * prog_t)))


def _stage_rewarmup_factor(
    ep: int, last_boundary_epoch: "int | None", rewarmup_epochs: int, floor: float, shape: str,
) -> float:
    """(BUILD 1 / FEED-fw) LR re-warmup multiplier in (0, 1] at 1-based epoch ``ep`` after an
    AdamW->AdamW stage boundary. DEFAULT-OFF: ``rewarmup_epochs <= 0`` (or no boundary yet) =>
    returns EXACTLY 1.0 => the LR schedule is BIT-IDENTICAL to the pre-FEED-fw path (x*1.0 == x for
    finite IEEE floats). After a registered stage TRANSITION at ``last_boundary_epoch``, ramp the
    multiplier from ``floor`` (at the boundary epoch, offset 0) back to 1.0 over ``rewarmup_epochs``
    epochs -- linear (default) or cosine.

    Rationale (operator 2026-06-26 "different stages need different treatment ... transitions must
    re-treat"; FEED-ft#3 tau-jump root cause): a loss-landscape change at a boundary, hit with FULL
    LR + stale AdamW momentum, is the instability. Ramping the LR back up gives the (optionally
    reset) optimizer state time to re-warm against the NEW stage's landscape, making the transition
    stable by construction. Pure (no model/MLX); unit-tested. Mirrors the per-epoch schedule helpers
    above."""
    if rewarmup_epochs <= 0 or last_boundary_epoch is None:
        return 1.0
    d = ep - last_boundary_epoch
    if d < 0 or d >= rewarmup_epochs:
        return 1.0
    floor = float(min(max(floor, 0.0), 1.0))
    prog = d / float(rewarmup_epochs)  # 0 at the boundary epoch -> ->1 across the window
    if shape == "cosine":
        return float(floor + (1.0 - floor) * 0.5 * (1.0 - np.cos(np.pi * prog)))
    return float(floor + (1.0 - floor) * prog)


LOSS_TERM_KEYS: tuple[str, ...] = (
    # #304 item 4 per-term loss telemetry -- the canonical row schema. Order matches total_loss_fn's
    # additive composition: base (seg CE-form + pose sqrt-term) then every stacked lever term.
    "seg", "pose", "eikonal", "length", "eik_steik", "boundary_distance", "lane_edge",
    "margin_saliency", "subpix", "chroma_boundary", "island_amplify", "persistence", "rankfloor",
    "code_spectral", "thin_lane", "margin_field_head", "code_nuclear",
    # "eik_steik" (EIK-STAB build 1a): the additive StEik directional-divergence stabilizer; 0.0
    # unless --eikonal-steik-weight > 0. NOTE: when --eikonal-viscosity > 0 the "eikonal" key holds
    # the VISCOUS residual contribution (ViscoReg replaces the residual; same constraint, viscous
    # form) — the schema is unchanged, the semantic is logged by the "eik_stabilizer" stage row.
)


def _loss_terms_row(terms: "dict[str, float]", total: float, ep: int, accum_batch: int,
                    *, gnorm: "float | None" = None, skipped: "bool | None" = None,
                    visco_eps: "float | None" = None, visco_c_a: "float | None" = None,
                    accepted_frac: "float | None" = None, weights_stepped: "bool | None" = None,
                    hosc_beta: "float | None" = None, softmax_temp: "float | None" = None) -> dict:
    """(#304 item 4) Build the canonical machine-readable per-term loss telemetry row. Pure /
    MLX-free / unit-tested. Every LOSS_TERM_KEYS key is present (missing/inactive terms -> 0.0) so
    the row schema is STABLE across configs; ``sum_terms`` and ``sum_minus_total`` make the
    breakdown self-checking (|sum_minus_total| should sit at fp tolerance -- the terms ARE the
    total's addends, recomputed on the same state)."""
    t = {k: float(terms.get(k, 0.0)) for k in LOSS_TERM_KEYS}
    ssum = float(sum(t.values()))
    row: dict[str, object] = {"stage": "loss_terms", "ep": int(ep), "accum_batch": int(accum_batch),
                              "terms": {k: round(v, 6) for k, v in t.items()},
                              "total": round(float(total), 6), "sum_terms": round(ssum, 6),
                              "sum_minus_total": round(ssum - float(total), 8)}
    if gnorm is not None:
        row["gnorm"] = round(float(gnorm), 4) if np.isfinite(gnorm) else "nonfinite"
    if skipped is not None:
        row["spike_skipped"] = bool(skipped)
        # (C6) a loss_terms row recomputed for a spike-SKIPPED chunk carries FROZEN-STATE values (the
        # weights did NOT step); flag it so a reader never treats the numbers as live progress.
        if bool(skipped):
            row["terms_frozen"] = True
    # (C6) LIVENESS STAMP: accepted-batch fraction this epoch + whether THIS batch stepped the weights.
    if accepted_frac is not None:
        row["accepted_frac"] = round(float(accepted_frac), 4)
    if weights_stepped is not None:
        row["weights_stepped"] = bool(weights_stepped)
    # (C6 / H2-F2) record the ANNEAL coefficients so a coefficient-driven eik move (beta-anneal on
    # FROZEN weights) is visible as a coefficient change, not misread as physics/eikonal runaway.
    if hosc_beta is not None:
        row["hosc_beta"] = round(float(hosc_beta), 4)
    if softmax_temp is not None:
        row["softmax_temp"] = round(float(softmax_temp), 4)
    # (V6 #320) adaptive-eps control-state observability: top-level fields (NOT in `terms`, so
    # sum_minus_total stays a clean loss-addend check). Emitted only when adaptive is active.
    if visco_eps is not None:
        row["visco_eps"] = round(float(visco_eps), 6)
    if visco_c_a is not None:
        row["visco_c_a"] = round(float(visco_c_a), 6)
    return row


def _loss_term_log_stride(env_probe: bool, log_every: int, chunks_per_epoch: int) -> int:
    """(#304 item 4) Resolve the loss-term telemetry cadence to an accum-chunk STRIDE.

    * env_probe (TAC_LOSS_TERM_PROBE=1)  -> 1   (every accum chunk = per-batch)
    * log_every > 0 (--loss-term-log-every N) -> N (every N chunks)
    * log_every < 0                       -> 0   (telemetry fully OFF; zero extra forwards)
    * default (log_every == 0)            -> chunks_per_epoch (first chunk of each epoch =
                                              the standing per-epoch summary)
    A chunk logs when ``stride > 0 and (accum_batch % stride == 0)``. Pure; unit-tested."""
    if env_probe:
        return 1
    le = int(log_every)
    if le > 0:
        return le
    if le < 0:
        return 0
    return max(int(chunks_per_epoch), 1)


def _rng_state_arrays(hardness_rng: "np.random.Generator | None") -> dict[str, np.ndarray]:
    """(FEED-fm FIX-1) Snapshot EVERY RNG the TRAINING LOOP advances, so a ``--resume-from`` run
    reproduces the CONTINUOUS draw sequence bit-for-bit (the deterministic-reproducibility
    non-negotiable: resume == continuous). The loop advances exactly TWO streams:

      * the GLOBAL ``np.random`` MT19937 -- the per-epoch ``np.random.permutation(P)`` pair order
        (and the ``permutation(concat)`` when hardness-oversample extras are appended); and
      * the LEVER-5 ``hardness_rng`` PCG64 ``Generator`` -- the ``hardness_rng.choice`` oversample.

    NO OTHER ``np.random.*`` call exists in the loop (verified: verdict/quantize/reorient/hardness-
    precompute touch neither global state), so snapshotting at checkpoint time + restoring at resume
    is exact. Keys are ``__``-prefixed so ``_load_resume_state`` routes them to ``cfg`` (the 624-key
    MT19937 array becomes a list there; the PCG64 dict is JSON-stringified). MLX-free; allow_pickle
    is NOT required to reload (plain arrays + unicode str)."""
    out: dict[str, np.ndarray] = {}
    algo, keys, pos, has_gauss, cached_gauss = np.random.get_state(legacy=True)
    out["__rng_np_algo"] = np.asarray(str(algo))
    out["__rng_np_keys"] = np.asarray(keys, np.uint32)
    out["__rng_np_pos"] = np.asarray(int(pos))
    out["__rng_np_has_gauss"] = np.asarray(int(has_gauss))
    out["__rng_np_cached_gauss"] = np.asarray(float(cached_gauss))
    if hardness_rng is not None:
        out["__rng_hardness_json"] = np.asarray(json.dumps(hardness_rng.bit_generator.state))
    return out


def _restore_rng_state(cfg: dict[str, Any], hardness_rng: "np.random.Generator | None") -> dict[str, bool]:
    """(FEED-fm FIX-1) Restore the RNG snapshot from a resume sidecar's ``cfg`` (the dict
    ``_load_resume_state`` returns). DEFAULT-SAFE / back-compat: a pre-FEED-fm checkpoint lacking the
    ``__rng_*`` keys leaves the freshly-seeded RNGs UNTOUCHED (exactly the pre-fix behavior; no crash)
    -- guarded by presence checks. Returns which streams were restored (observability). NO-FAKE: this
    really sets the global MT19937 + the PCG64 generator state so the next draw matches a continuous
    run; it is not a marker."""
    restored = {"np_global": False, "hardness": False}
    if "__rng_np_keys" in cfg and "__rng_np_pos" in cfg:
        keys = np.asarray(cfg["__rng_np_keys"], dtype=np.uint32)
        np.random.set_state((
            str(cfg.get("__rng_np_algo", "MT19937")), keys, int(cfg["__rng_np_pos"]),
            int(cfg.get("__rng_np_has_gauss", 0)), float(cfg.get("__rng_np_cached_gauss", 0.0)),
        ))
        restored["np_global"] = True
    if hardness_rng is not None and "__rng_hardness_json" in cfg:
        try:
            hardness_rng.bit_generator.state = json.loads(str(cfg["__rng_hardness_json"]))
            restored["hardness"] = True
        except Exception:  # malformed/foreign state: keep the fresh PCG64 (best-effort, no crash).
            pass
    return restored


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    out_dir = Path(args.out_dir)
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # #205 PROVENANCE: capture git sha + upstream snapshot sha ONCE at launch (threaded into result.json
    # AND every per-stage checkpoint cfg so the #205 run + each byte-close artifact is reproducible from
    # provenance). NO-FAKE: "unknown" when git is unavailable, never fabricated.
    _run_provenance = _git_provenance()
    print(json.dumps({"stage": "provenance", **_run_provenance,
                      "seed": int(args.seed)}), flush=True)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # --- GT (frozen CPU authority) ---
    t0 = time.time()
    if args.gt_cache:
        gt, seg_cpu, posenet_cpu = load_gt_from_cache(Path(args.gt_cache), args.num_pairs)
    else:
        gt, seg_cpu, posenet_cpu = precompute_gt(args.num_pairs)
    P = gt.n_pairs
    print(json.dumps({"stage": "gt", "n_pairs": P, "secs": round(time.time() - t0, 1)}), flush=True)

    render_h, render_w = args.render_h, args.render_w
    coords_np = _build_render_coords(render_h, render_w)

    # --- FRONT-END: generic curvelet/shearlet bank (byte-closeable, GT-free) ---
    bank = CurveletBankConfig(
        n_scales=args.bank_n_scales, n_orient0=args.bank_n_orient0,
        f0=args.bank_f0, base=args.bank_base, n_iso=args.bank_n_iso,
    )
    # LEVER-2 (stem-Nyquist) cap (default None = no cap = current behavior). Drops curvelet atoms
    # above the SegNet-stem Nyquist (free byte/alias budget; see stem_nyquist_max_freq_*).
    B = curvelet_directional_B(bank, max_freq=args.max_bank_freq)
    curv_feats_np = curvelet_feats(coords_np, B).astype(np.float32)  # (P, 2*cols)
    in_feat = curv_feats_np.shape[1]
    # SELF-ORIENTATION directional augmentation (byte-closeable; tangent from the witness's OWN
    # argmax, cos 0.89-0.91 vs GT). Recomputed every --reorient-every epochs from the live SDF
    # argmax; concatenated to the curvelet feats. OFF by default (the from-scratch smoke uses
    # curvelet only — self-orientation is a finetune lever needing a roughly-learned partition).
    # SELF-ORIENT (#1 follow-up, WIRED): the byte-closeable -48% directional lever. The tangent is
    # computed from the decoder's OWN cheap-forward argmax (self-orientation FIXED POINT: start with
    # zero-directional = curvelet-only iso pass -> argmax -> tangent -> directional feats -> converge),
    # so it is reconstructible at decode with NO GT leak (cos 0.89-0.91 vs GT). PER-PAIR feats are
    # concatenated to the shared curvelet feats and threaded through train+verdict (ONE codepath).
    use_self_orient = bool(args.self_orient)
    n_dir_freqs = int(args.n_dir_freqs)
    dir_w = 4 * n_dir_freqs
    if use_self_orient:
        in_feat += dir_w
    # per-pair directional feats (zeros until the first reorient -> ep<reorient = pure curvelet).
    dir_feats_per_pair = [np.zeros((curv_feats_np.shape[0], dir_w), np.float32) for _ in range(P)] if use_self_orient else None
    # #224 (Wave B) AA-supersample + self-orient FINE dir-feat state (declared here so the render/
    # reorient closures below see run-scope defaults even when AA/self-orient is OFF). Populated only
    # when --render-aa supersample + --self-orient + --aa-self-orient-fine-mode {batch,full}. The base
    # argmax per pair (H,W int8, ~118MB @ n600 — cheap) is snapshotted at each reorient so the fine
    # dir-feats can be recomputed (NN-upsample argmax -> ss*grid -> fine EDT-tangent -> directional
    # Fourier) without re-running the witness argmax.
    _aa_so_fine = False
    _aa_fine_mode = "refuse"
    _aa_coords_fine = None
    base_argmax_per_pair: list = [None] * P
    _aa_fine_dir_full: list = [None] * P      # full mode: per-pair fine dir-feats (mx), rebuilt @ reorient
    _aa_fine_lru: dict = {}                    # batch mode: bounded FIFO cache of per-pair fine dir-feats

    # ── GROUND-FRAME CHART (#194 / §17.1) placeholders: _gfc_chart is BUILT below (after the
    # render_aa block, where the fail-closed combination guards can see render_aa); the closure
    # here late-binds it. use_gfc=False (the default) leaves every path byte-identical.
    use_gfc = bool(getattr(args, "ground_frame_chart", False))
    _gfc_chart = None

    def _feats_np_for_pair(pi: int) -> np.ndarray:
        if use_gfc:
            # per-pair chart coords -> curvelet feats, recomputed on demand (numpy side; the MLX
            # side caches ONCE via cf_mx_cache — the chart is static, unlike the reorient loop).
            return curvelet_feats(_gfc_chart.coords_for_pair_numpy(coords_np, pi), B).astype(np.float32)
        if not use_self_orient:
            return curv_feats_np
        return np.concatenate([curv_feats_np, dir_feats_per_pair[pi]], axis=-1).astype(np.float32)

    print(json.dumps({"stage": "front_end", "curvelet_cols": int(B.shape[1]), "dir_w": int(dir_w),
                      "in_feat": int(in_feat), "self_orient": use_self_orient,
                      "front_end": ("curvelet+self_orient" if use_self_orient else "generic-curvelet only")}), flush=True)

    # (DEVICE BUG FIX) the adapter LOADS the upstream torch scorers then converts to MLX — the
    # torch .device() must be "cpu" (torch has no "gpu"; args.mlx_device="gpu" crashed here in 3.4s).
    # The MLX render runs on mx.gpu via temporary_mlx_device(args.mlx_device) below; the torch
    # scorer/R/verdict are CPU authority. The device SPLIT: MLX "gpu" -> render; torch -> "cpu".
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
    # ---- #224 AA-SDF observation-map render (aa_sdf_observation_render; MEASURED #1 rep lever,
    # DAG FEED-ly/-ma). DEFAULT --render-aa none => this block is a NO-OP (curv_feats_np unchanged +
    # coord_feats_fine_mx None) => BYTE-IDENTICAL. Two AA modes: (ipe) attenuate the curvelet basis
    # columns by the mip-NeRF cone footprint (analytical, base grid; touches ONLY the curvelet feats,
    # NOT the self-orient dir feats); (supersample) build a SEPARATE fine-grid feats for the render
    # path only -- the BASE-grid coord_feats/_feats_np_for_pair stay base-grid so the eikonal/sdf(cf)
    # reshape to (render_h, render_w) is unaffected; _render_R dispatches to render_aa_through_R_mlx. ----
    render_aa = str(getattr(args, "render_aa", "none"))
    aa_ss = int(getattr(args, "aa_supersample", 1))
    coord_feats_fine_mx = None  # (supersample only) the fine-grid render feats; None => point-sample
    if render_aa == "ipe":
        from tac.boundary_math.aa_sdf_observation_render import (
            apply_ipe_attenuation,
            ipe_curvelet_attenuation,
            ipe_footprint_sigma,
        )
        _aa_sx, _aa_sy = ipe_footprint_sigma(render_h, render_w, float(args.aa_ipe_footprint))
        _aa_att = ipe_curvelet_attenuation(B, _aa_sx, _aa_sy)
        curv_feats_np = apply_ipe_attenuation(curv_feats_np, _aa_att).astype(np.float32)  # (P, 2*cols)
        print(json.dumps({"stage": "render_aa_ipe", "footprint": float(args.aa_ipe_footprint),
                          "sigma_x": round(float(_aa_sx), 4), "sigma_y": round(float(_aa_sy), 4),
                          "note": "curvelet basis attenuated (mip-NeRF cone); base grid; ~0 compute"}), flush=True)
    elif render_aa == "supersample" and aa_ss > 1:
        # Fail-closed on the un-wired combinations (NO-FAKE: no silent wrong result). The fine-grid
        # self-orient per-pair dir-feat recompute + the structured-init render-res==L*-res invariant
        # are not yet wired; refuse rather than render on mismatched feats.
        #
        # #224 Wave D AA CORRECTION (aa_feasibility_reconciliation_20260702.md): supersample is NOT
        # the launch AA — it is train-only (neither shipped inflate applies ss → train/decode
        # observation MISMATCH), its fp64 decode is 41min > the 30min budget, AND it HURTS the witness
        # −49% (the 0.00086 floor is a REAL-FRAME ceiling, not witness-realized). The launch config
        # ships --render-aa none + the analytic coverage-integrated --lane-render-band. This whole
        # supersample+self-orient fail-closed path therefore never fires from the all-levers launch;
        # it stays BUILT + fail-closeable for research only. Memory arithmetic below is RESOLVED
        # (reconciliation Q3) but MOOT for the launch given the decode + witness-harm disqualification.
        #
        # #224 Option-B DECISION (FAIL-CLOSED, precise n600 blocker — NOT a shape/impl gap):
        # AA-supersample + --self-orient needs PER-PAIR fine-grid feats = curvelet(coords_fine) ⊕
        # dir_feats_fine(pair), where dir_feats_fine is the spec's argmax-NN-upsample→fine-EDT→
        # directional-Fourier (docs/aa_sdf_observation_render_wire_in_spec.md). Reconciled n600 memory
        # (Q3), ss=2, n_dir_freqs=2 (the shipped config), 384×512:
        #   (a) The fine CURVELET feats are pair-INDEPENDENT → ONE SHARED tensor (~0.23GB), NOT
        #       per-pair. ONLY the fine DIR-feats are per-pair: 25.2 MB/pair @ ndf2 × 600 = ~14GB
        #       (full mode) — NOT the ~164GB the pre-reconciliation comment feared (that was the NAIVE
        #       full-fine-feats-per-pair @ ndf6 over-estimate). Peak ≈ 63GB (fine 14 + base cf_mx_cache
        #       ~41 [held STEADY via the in-place rebuild, L~2411, not 2×] + fwd ~8) → memory-SAFE on
        #       the 128GB M5 Max, but this is a SCALED EXTRAPOLATION (24MB/pair measured), not a real
        #       n600 allocation.
        #   (b) ON-DEMAND fine feats (no fine cache, memory-safe): recompute the fine EDT per render
        #       call. The base path amortizes P EDTs across --reorient-every (~50) epochs via the
        #       cache; on-demand does P fine (ss^2×) EDTs EVERY epoch (~50× more, 4× larger) =>
        #       minutes/epoch of scipy EDT over thousands of epochs => non-n600-viable wall-clock.
        # Neither the cache-memory budget nor the on-demand wall-clock can be measured under
        # CONTAINMENT (no GPU). Per CLAUDE.md OPERATOR PRIORITY (fail-closed when "can't be verified
        # correct without a GPU run") this lever stays fail-closed with THIS precise blocker rather
        # than shipping an unverified / non-n600-viable path. WIRED self-orient-compatible AA/lane
        # alternatives (use these for the from-scratch launch): --render-aa ipe (basis-level cone AA,
        # touches ONLY the shared curvelet columns, self-orient-compatible, ~0 compute) AND/OR
        # --lane-render-band (class-1 render authority, NOW self-orient-composable per the Option-B
        # lane-band wire-in below). AA-supersample WITHOUT --self-orient also still works.
        _aa_fine_mode = str(getattr(args, "aa_self_orient_fine_mode", "refuse"))
        if use_self_orient and _aa_fine_mode == "refuse":
            # FAIL-CLOSED default (Wave B SHARPENED, MEASURED blocker). The per-pair fine-grid dir-feats
            # (argmax→ss*grid→fine-EDT→directional-Fourier, docs/aa_sdf_observation_render_wire_in_spec.md)
            # face a MEASURED memory↔wall-clock tradeoff that cannot be BOTH-satisfied AND n600-validated
            # under the no-launch CONTAINMENT (measured local-MLX, ss=2, 384x512):
            #   * fine-EDT recompute = ~49 ms/pair; per-pair fine dir-feat = 25.2 MB @ n_dir_freqs=2
            #     (the shipped config) — the older 75.5 MB was the ndf6 figure (reconciliation Q3).
            #   * MEMORY-SAFE (--aa-self-orient-fine-mode batch): a batch-bounded on-demand cache is
            #     ~cap*25MB (0.2 GB @ cap=8 vs ~14 GB all-600 @ ndf2) => memory SOLVED. BUT every pair
            #     renders every epoch, so a batch-bounded cache THRASHES => P fine-EDTs/epoch ~29 s/epoch
            #     @ n600 (50x the base --reorient-every amortization) => wall-clock NON-viable for the
            #     multi-thousand-epoch CE→tau→l7→Muon curriculum.
            #   * WALL-CLOCK-viable (--aa-self-orient-fine-mode full): compute the fine dir-feats ONCE
            #     per --reorient-every (amortized ~0.6 s/epoch) BUT store all P => ~14 GB @ ss=2, ndf2
            #     (the fine curvelet feats are pair-independent → ONE shared ~0.23GB tensor, NOT per-pair);
            #     peak ≈ 63 GB (fine 14 + base cf_mx_cache ~41 held STEADY via the in-place rebuild + fwd
            #     ~8). This is a SCALED EXTRAPOLATION (24MB/pair measured); MOOT for the launch (supersample
            #     is disqualified by the decode-budget + −49% witness-harm per the Wave D header above).
            # Both opt-in modes ARE now BUILT + small-MLX-verified (render finite+shape; memory scales
            # ~batch); the DEFAULT stays fail-closed so no unverified OOM / 50x-slow n600 run fires by
            # accident. This is THE operator's-call item: pick `full` after an n600 memory-fit check, or
            # `batch` if the extra CPU-EDT wall-clock is acceptable. Self-orient-compatible alternatives
            # that ARE fully wired: --render-aa ipe (basis-level cone AA, ~0 compute) and/or
            # --lane-render-band; AA-supersample WITHOUT --self-orient also works.
            raise ValueError(
                "--render-aa supersample + --self-orient is fail-closed by DEFAULT (Wave B). The fine "
                "dir-feat path is BUILT + verified; enable it explicitly with "
                "--aa-self-orient-fine-mode full (wall-clock-viable, ~14GB@ss2n600 @ndf2, peak ~63GB — "
                "validate the n600 memory fit first) OR --aa-self-orient-fine-mode batch (memory-safe ~cap*25MB, but "
                "~P fine-EDTs/epoch ~29s@n600). Or use --render-aa ipe / --lane-render-band (both "
                "self-orient-compatible + fully wired), or AA-supersample WITHOUT --self-orient.")
        # #224 Wave C FIX-2: supersample + --structured-init is NOW WIRED (was fail-closed as
        # "not-yet-wired", NOT proven-incompatible). The two operate on DIFFERENT grids and compose:
        # structured-init pretrains the coord-INR witness weights on the BASE grid against the cached L*
        # (its invariant `(render_h,render_w) == lstar_shape` is checked at the structured-init block
        # below and is UNCHANGED by supersample — aa_ss multiplies only the internal fine render grid,
        # NOT render_h/render_w). The fine render then evaluates the SAME shared weights at fine coords
        # (a coord-INR generalizes across coordinate resolution by construction). Verified at small-MLX
        # n4 (ss=2 + self-orient full + structured-init + lane-prior-phi1: finite render + descent). Per
        # the Wave B precedent (LEVER 3 relaxed the self-orient guard behind an opt-in after BUILD +
        # small-MLX verify). The REAL render==L* invariant stays enforced at the structured-init block.
        print(json.dumps({"stage": "render_aa_supersample_structured_init",
                          "structured_init": bool(args.structured_init),
                          "note": "supersample composes with structured-init: base-grid pretrain + "
                          "shared coord-INR weights evaluated at fine coords (render_h/w == L* unchanged)"}),
              flush=True)
        from tac.boundary_math.aa_sdf_observation_render import build_supersampled_coords
        _coords_fine = build_supersampled_coords(render_h, render_w, aa_ss)          # (ss^2*P, 2)
        coord_feats_fine_mx = mx.array(curvelet_feats(_coords_fine, B).astype(np.float32))
        if use_self_orient:
            # opt-in fine self-orient (batch|full). _cf_fine_mx (below) sources per-pair fine dir-feats;
            # rebuilt/invalidated at each reorient. Pre-first-reorient -> zeros -> pure-curvelet fine.
            _aa_so_fine = True
            _aa_coords_fine = _coords_fine
        print(json.dumps({"stage": "render_aa_supersample", "ss": aa_ss,
                          "fine_grid": [render_h * aa_ss, render_w * aa_ss],
                          "self_orient_fine_mode": (_aa_fine_mode if use_self_orient else "n/a"),
                          "note": "separate fine-grid render feats; base-grid eikonal/sdf unaffected"}), flush=True)
    coord_feats_mx = mx.array(curv_feats_np)

    # ── GROUND-FRAME CHART (#194 / §17.1) build — tac.boundary_math.ground_frame_chart. ──
    # v0 fail-closed combinations (coordinate-system consistency; NO-FAKE: refuse a silent hybrid):
    #   * --self-orient: the directional feats are computed at FRAME coords (EDT tangent on the
    #     render grid); concatenating them to GROUND-chart curvelet feats mixes two coordinate
    #     systems in one feature vector — a designable composition (reorient in chart coords), not v0.
    #   * --render-aa != none: ipe attenuates the SHARED curv_feats_np (the chart recomputes fresh,
    #     un-attenuated feats) and supersample builds a SHARED un-charted fine grid.
    #   * --structured-init with --gfc-ref-pair != 0: the pretrain uses pair-0 feats; chart[ref] is
    #     the exact-identity chart, so ref must be 0 for the pretrain grid to match the static core.
    # When ON: one chart per pair from the STORED pose table (dual-use with the pose sidecar —
    # rule-118 FREE, 0 new archive bytes); MLX per-pair feats cached ONCE (static; no reorient churn).
    if use_gfc:
        if use_self_orient:
            raise ValueError(
                "--ground-frame-chart + --self-orient is fail-closed (v0): the self-orient "
                "directional feats live in FRAME coords while the chart moves the curvelet feats to "
                "GROUND coords — one feature vector, two coordinate systems. The composition "
                "(reorient computed in chart coords) is designed but unbuilt; run the chart arm "
                "with --no-self-orient.")
        if render_aa != "none":
            raise ValueError(
                "--ground-frame-chart + --render-aa != none is fail-closed (v0): ipe attenuates the "
                "shared curvelet feats (the chart recomputes per-pair, un-attenuated) and supersample "
                "uses a shared un-charted fine grid. Run the chart arm with --render-aa none.")
        if bool(args.structured_init) and int(args.gfc_ref_pair) != 0:
            raise ValueError(
                "--ground-frame-chart + --structured-init requires --gfc-ref-pair 0 (the pretrain "
                "uses pair-0 feats; only chart[ref] is the exact identity).")
        from tac.boundary_math.ground_frame_chart import ChartCalibration, GroundFrameChart
        _gfc_chart = GroundFrameChart.build(
            np.stack([np.asarray(gt.gt_poses[pi], np.float64) for pi in range(P)]),
            ref_pair=int(args.gfc_ref_pair),
            calib=ChartCalibration(s_t=float(args.gfc_s_t), s_r=float(args.gfc_s_r),
                                   pitch=float(args.gfc_pitch)),
            grid_hw=(render_h, render_w),
        )
        print(json.dumps({"stage": "ground_frame_chart", "ref_pair": int(args.gfc_ref_pair),
                          "regime": "ground", "s_t": float(args.gfc_s_t), "s_r": float(args.gfc_s_r),
                          "pitch": float(args.gfc_pitch), "n_pairs": P,
                          "note": "#194/§17.1 witness input chart pre-composition (FEED-ll math; "
                          "rule-118 FREE from the stored pose table; STRUCTURAL from ep0)"}), flush=True)

    # (DIAGNOSED FIX) natural per-class palette = mean GT RGB per L* class (the transfer-probe's
    # winning ingredient; logit space). Anchors the learned palette inside SegNet's distribution so
    # the partition is READABLE from epoch 0 (a luma-ramp init plateaued ~0.51). NO GT leak at
    # deploy: the palette is a LEARNED weight counted in the archive (it is a fixed (5,3) tensor,
    # not the per-frame GT). --no-palette-anchor restores the generic ramp (ablation).
    palette_init = None
    if args.palette_anchor:
        import torch
        import torch.nn.functional as F
        sums = np.zeros((5, 3), np.float64); cnts = np.zeros(5, np.float64)
        for pi in range(min(P, 64)):
            f1 = torch.from_numpy(np.asarray(gt.gt_f1[pi], np.float32)).permute(2, 0, 1)[None]
            lr = np.asarray(gt.lstars[pi]); hh, ww = lr.shape
            small = F.interpolate(f1, size=(hh, ww), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
            for k in range(5):
                msk = lr == k
                if msk.any():
                    sums[k] += small[msk].sum(0); cnts[k] += int(msk.sum())
        mean = np.where(cnts[:, None] > 0, sums / np.maximum(cnts[:, None], 1), 127.0)
        palette_init = np.log(np.clip(mean / 255.0, 1e-3, 1 - 1e-3) / (1 - np.clip(mean / 255.0, 1e-3, 1 - 1e-3))).astype(np.float32)
        # (C10 confound fix) an init lever is OVERWRITTEN by a --resume-from (model.update replaces
        # every param, incl. palette). Do NOT print active:true silently -> stamp applied:false.
        _init_applied = not bool(args.resume_from)
        print(json.dumps({"stage": "palette_anchor", "mean_rgb": mean.round(1).tolist(),
                          "applied": _init_applied,
                          **({} if _init_applied else {"reason": "overwritten_by_resume"})}), flush=True)

    model = build_levelset_rgb_witness(
        num_pairs=P, in_feat=in_feat, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
        mod_dim=args.mod_dim, n_classes=5, activation=args.activation, softmax_temp=args.softmax_temp_start,
        wire_w0=args.wire_w0, wire_s0=args.wire_s0, hosc_beta=args.hosc_beta, hosc_omega=args.hosc_omega,
        chroma=args.chroma, palette_init_logit=palette_init,
        film_per_layer=bool(getattr(args, "film_per_layer", False)),
        film_concat_code=bool(getattr(args, "film_concat_code", False)),
    )
    mx.eval(model.parameters())
    # #218 facet-1a — fixed simplex-ETF head (Yang et al. 2022, neural-collapse optimal). Replaces the
    # LEARNED out_sdf weight with a deterministic simplex ETF (equal-norm, max-equiangular K prototypes)
    # and FREEZES it: removes the minority-class NORM COLLAPSE that erases Lane/Movable, AND is
    # regenerable from a fixed seed at inflate => the K x d head weight is FREE (rate win). out_sdf.bias
    # stays trainable. args.head != "etf" (default) => untouched => byte-identical.
    if str(getattr(args, "head", "softmax")) == "etf":
        from tac.boundary_math.laguerre_logit_offset import etf_gram_offdiag, simplex_etf
        _etf_w = simplex_etf(5, args.hidden_dim).astype(np.float32)
        model.out_sdf.weight = mx.array(_etf_w)
        model.out_sdf.freeze(keys=["weight"])
        mx.eval(model.parameters())
        print(json.dumps({"stage": "head_etf", "offdiag_cos": round(float(etf_gram_offdiag(_etf_w)), 4),
                          "target_cos": round(-1.0 / 4.0, 4), "frozen_weight": True}), flush=True)
    # SIREN init (Sitzmann 2020) for the periodic family (hosc/wire) — the canonical from-scratch
    # trainability fix (parent: hosc-without-SIREN-init was d_seg 0.689). Reuses the parent's
    # apply_siren_init on in_proj (first) + hidden (subsequent); out_sdf/out_tex/palette/film keep
    # default init (FiLM must stay nonzero or the code-gradient dies).
    if args.activation in {"hosc", "wire"} and args.siren_init:
        from train_witness_realized_through_R_mlx import apply_siren_init
        omega_init = args.hosc_omega if args.activation == "hosc" else args.wire_w0
        apply_siren_init(model, omega=omega_init)
        mx.eval(model.parameters())
    # (#310 BUILD, FEED-07b lever #2 sister) FINER++ variable-periodic FIRST-LAYER bias init
    # (arXiv 2407.19434; the published fix for the MEASURED fixed-beta hosc saturation-death):
    # overwrite in_proj.bias with U(-k, k) from a DEDICATED np.random.default_rng(seed+salt)
    # stream so each first-layer neuron selects its OWN frequency/phase of the periodic
    # activation. Runs AFTER siren_init (which zeroes the bias) and BEFORE structured-init (the
    # FINER bias is the pretrain's starting point). DEFAULT OFF => this branch never runs => NO
    # RNG draw anywhere => byte-identical (the ON path also perturbs no shared stream — the rng
    # is dedicated). Fail-closed on non-periodic activations (relu has no period to phase into).
    if bool(getattr(args, "finer_bias_init", False)):
        if args.activation not in {"hosc", "wire"}:
            raise ValueError(
                "--finer-bias-init requires a periodic activation (hosc/wire): the wide "
                "first-layer bias selects each neuron's phase of the periodic nonlinearity; "
                f"--activation {args.activation} has no period. Drop the flag or switch activation.")
        model.in_proj.bias = mx.array(_finer_bias_init_values(
            int(args.seed), float(args.finer_bias_k), int(model.in_proj.bias.shape[0])))
        mx.eval(model.parameters())
        _fb_applied = not bool(args.resume_from)  # (C10) an init lever is OVERWRITTEN by --resume-from
        print(json.dumps({"stage": "finer_bias_init", "k": float(args.finer_bias_k),
                          "n": int(model.in_proj.bias.shape[0]),
                          "rng": f"np.random.default_rng(seed+{_FINER_RNG_SALT}) [dedicated stream]",
                          "applied": _fb_applied,
                          **({} if _fb_applied else {"reason": "overwritten_by_resume"}),
                          "note": "FINER++ 2407.19434 wide first-layer bias (fix for fixed-beta "
                                  "hosc saturation-death); OFF => zero RNG draws => byte-identical"}),
              flush=True)
    # STRUCTURED-PRIOR phi INIT (FEED-ef, ADDITIVE, default-off). PRETRAIN phi so argmax(phi) ~= the
    # validated self-detected static-core partition (hood+sky+road[+lane] deep SDFs; FEED-dm/du/dw/dx).
    # The one-shot linear-readout init is broken (the random INR trunk's linear span ~= majority class,
    # disagree ~0.51 across hosc/relu/wire); the trunk must be ADAPTED, so this is a short subsampled
    # Adam pretrain of model.sdf -> the clipped structured SDF target (the network has the capacity:
    # trained mod-32 reaches d_seg 0.00124; pretrain reaches direct disagree ~0.025 in ~600 steps).
    # The static-core is generic same-rig camera geometry (rule-118 FREE; train-time init ships 0 bytes
    # -- the archive ships the TRAINED weights). Built on the cached L* (frozen CPU-torch argmax). EMA
    # is created AFTER so the shadow starts at the structured init. Default OFF => skipped => byte-identical.
    # MEASURED CAVEAT (n24 realized-through-R): NO epoch-0 realized win (the render is texture-dominated
    # at init -> SegNet reads random out_tex, not the partition; structured realized 0.586 ~ random 0.506).
    # Value is a training-trajectory A/B only (UNPROVEN). hosc/SIREN-init-fragile -> loud WARN if it stalls.
    if args.structured_init:
        from tac.boundary_math.lever_b_levelset_generator import build_static_core_phi_target
        lstar_shape = tuple(np.asarray(gt.lstars[0]).shape)
        if (render_h, render_w) != lstar_shape:
            raise ValueError(
                f"--structured-init requires --render-h/--render-w == the L* res {lstar_shape} "
                f"(got {(render_h, render_w)}); the static-core masks are built on the cached L*."
            )
        lst_stack_si = np.stack([np.asarray(gt.lstars[pi], np.int64) for pi in range(P)], axis=0)
        phi_tgt_hwk, sc_roles, sc_meta = build_static_core_phi_target(
            lst_stack_si, n_classes=5, include_lane=args.structured_init_include_lane,
            static_thresh=args.structured_init_thresh,
        )
        # BUILD 2 (FEED-fw): inject the openpilot deg-3 centerline lane SDF into the phi1 channel of
        # the structured target BEFORE the joint pretrain absorbs it. DEFAULT-OFF (--lane-prior-phi1
        # off) => phi_tgt_hwk is UNTOUCHED => the structured-init pretrain is BIT-IDENTICAL. Reuses
        # the standalone-geometry helpers (numpy/scipy, $0 CPU): build_structured_lane_sdf is the
        # ground-plane homography (K @ scorer-res {fx=910*512/1164=400.3,...}) -> deg-3 lane curve ->
        # per-pixel signed distance (FEED-fs separatrix, residual 1.9e-5); inject_lane_sdf writes it
        # into the K-field stack. The fit is from the cached L* (frozen CPU-torch argmax) of the
        # chosen pair. rule-118 FREE generic structure: train-time init only, ships 0 archive bytes.
        if getattr(args, "lane_prior_phi1", False):
            from tac.boundary_math.lane_sdf_component import (
                build_structured_lane_sdf,
                inject_lane_sdf,
            )
            _lp_pair = int(args.lane_prior_phi1_source_pair)
            if not (0 <= _lp_pair < P):
                raise ValueError(
                    f"--lane-prior-phi1-source-pair ({_lp_pair}) out of range [0,{P - 1}].")
            _lp_lstar = np.asarray(gt.lstars[_lp_pair], np.int64)
            phi1_lane, lp_meta = build_structured_lane_sdf(
                _lp_lstar, lane_cls=1, dash_gate=bool(args.lane_prior_phi1_dash_gate),
                centerline_deg=3)
            phi_tgt_hwk = inject_lane_sdf(
                phi_tgt_hwk, phi1_lane, lane_cls=1, mode=args.lane_prior_phi1_mode,
                bias_scale=float(args.lane_prior_phi1_bias_scale))
            # (C18 confound fix) after a PAINT-mode inject, the lane class MUST gain partition mass;
            # if part_frac[lane]==0 the paint did NOT win (a sign/band-side bug -- H4#4 measured 0 even
            # FRESH). Assert loud so a FAKE "lane painted" lever can never pass silently.
            _paint_part_frac_lane = None
            if str(args.lane_prior_phi1_mode) == "paint":
                _pp = phi_tgt_hwk.argmax(-1).reshape(-1)
                _paint_part_frac_lane = float(np.count_nonzero(_pp == 1)) / float(_pp.size)
                if _paint_part_frac_lane <= 0.0:
                    print(json.dumps({"stage": "lane_prior_phi1_PAINT_FAILED",
                                      "part_frac_lane": _paint_part_frac_lane, "source_pair": _lp_pair,
                                      "msg": "(C18) inject_lane_sdf(mode=paint) yielded part_frac[lane]=0 "
                                      "-- the paint did NOT win the argmax (sign/band-side bug); the "
                                      "'lane painted by construction' claim is FALSE for this config"}),
                          flush=True)
                    raise ValueError(
                        f"(C18) --lane-prior-phi1-mode paint did NOT paint the lane: "
                        f"part_frac[lane]={_paint_part_frac_lane} (expected >0). The paint must WIN the "
                        "argmax by construction; a 0 mass means a sign/band-side bug in inject_lane_sdf "
                        "(H4#4). Fix the paint or use --lane-prior-phi1-mode replace/bias.")
            # (C10 confound fix) OVERWRITTEN by --resume-from (this init runs, then model.update replaces it).
            _lp_applied = not bool(args.resume_from)
            print(json.dumps({"stage": "lane_prior_phi1", "active": _lp_applied,
                              "applied": _lp_applied, "source_pair": _lp_pair,
                              "mode": args.lane_prior_phi1_mode,
                              "dash_gate": bool(args.lane_prior_phi1_dash_gate),
                              **({"part_frac_lane": round(_paint_part_frac_lane, 6)}
                                 if _paint_part_frac_lane is not None else {}),
                              **({} if _lp_applied else {"reason": "overwritten_by_resume"}),
                              **{f"lane_{k}": v for k, v in lp_meta.items()},
                              "note": "openpilot deg-3 centerline SDF injected into structured-init "
                              "phi1 target (FEED-fs Road<->Lane separatrix; train-time init, 0 "
                              "archive bytes)"}), flush=True)
        sc_part = phi_tgt_hwk.argmax(-1).reshape(-1)
        sc_feats_np = _feats_np_for_pair(0)  # pair-0 feats (curvelet[+zeros]); all codes 0 at init -> SHARED
        sc_clip = float(args.structured_init_sdf_clip)
        sc_tgt_np = np.clip(phi_tgt_hwk.reshape(render_h * render_w, 5), -sc_clip, sc_clip).astype(np.float32)
        sc_ns = min(int(args.structured_init_subsample), sc_feats_np.shape[0])
        sc_rng = np.random.default_rng(args.seed)

        def _structured_init_loss(m, fb, tb):
            return mx.mean((m.sdf(fb, 0) - tb) ** 2)

        sc_vg = nn.value_and_grad(model, _structured_init_loss)
        sc_opt = optim.AdamW(learning_rate=float(args.structured_init_lr))
        for _s in range(int(args.structured_init_steps)):
            sc_idx = sc_rng.integers(0, sc_feats_np.shape[0], sc_ns)
            _sL, _sg = sc_vg(model, mx.array(sc_feats_np[sc_idx]), mx.array(sc_tgt_np[sc_idx]))
            # FREEZE the per-frame code embedding: pretrain the SHARED trunk (code=0) so EVERY frame
            # (all codes 0 at init) starts at the structured partition, not just frame 0. Without this
            # the loss on sdf(.,0) also adapts code[0] -> only frame 0 is structured (MEASURED: a
            # code=0 frame disagrees 0.67 vs 0.011 frozen). Keeps the init a true SHARED prior.
            if "code" in _sg:
                _sg["code"] = mx.zeros_like(_sg["code"])
            sc_opt.update(model, _sg)
            mx.eval(model.parameters())
        sc_phi = np.asarray(model.sdf(mx.array(sc_feats_np), 0))
        sc_disagree = float(np.count_nonzero(sc_phi.argmax(-1) != sc_part)) / sc_part.size
        mx.eval(model.parameters())
        # (C10 confound fix) structured-init is an INIT-TIME weight shaping lever; a --resume-from
        # OVERWRITES every param at model.update, discarding it. Stamp applied:false (not active:true).
        _si_applied = not bool(args.resume_from)
        print(json.dumps({"stage": "structured_init", "roles": sc_roles.as_dict(),
                          "applied": _si_applied,
                          **({} if _si_applied else {"reason": "overwritten_by_resume"}),
                          "pretrain_direct_argmax_disagree_vs_part": round(sc_disagree, 5),
                          "steps": int(args.structured_init_steps), "lr": float(args.structured_init_lr),
                          **{k: v for k, v in sc_meta.items() if k != "roles"}}), flush=True)
        if sc_disagree > 0.30:
            print(json.dumps({"stage": "structured_init_WARN",
                              "msg": "pretrain did NOT structure the partition (disagree>0.30); init ~ random "
                              "(hosc/SIREN trainability fragility). Try --structured-init-lr/-steps or another --seed.",
                              "disagree": round(sc_disagree, 5)}), flush=True)
    # AMORTIZATION (FEED-eo, --freeze-decoder-fit-codes, ADDITIVE, default-off). The witness factors
    # into a SHARED decoder (in_proj/film/hidden/out_sdf/out_tex/palette) + per-(pair,frame) latent
    # codes (1200 x mod_dim). A full from-scratch n600 row co-fits BOTH (days). This mode LOADS a
    # decoder trained on a SUBSET (n96/n192), FREEZES it, and fits ONLY the ~num_pairs*2*mod_dim
    # codes for all pairs (a small per-pair optimization through the frozen render+R+scorer ->
    # embarrassingly parallel per pair; hours not days) -> the future-row fast path IF the frozen
    # shared decoder generalizes (the small-n estimate measures this). Loaded BEFORE EMA so the EMA
    # shadow (the deploy weights) starts at the frozen decoder; freeze BEFORE value_and_grad so the
    # grad/optimizer/weight-decay only ever touch ``code`` (the decoder cannot drift). Default
    # None => skipped => byte-identical to a normal joint run.
    freeze_decoder = bool(getattr(args, "freeze_decoder_fit_codes", None))
    if freeze_decoder:
        if args.resume_from:
            raise ValueError("--freeze-decoder-fit-codes is incompatible with --resume-from (one "
                             "loads a frozen decoder + FRESH codes; the other restores a full state).")
        if args.structured_init:
            raise ValueError("--freeze-decoder-fit-codes is incompatible with --structured-init "
                             "(the decoder is frozen-from-file, not pretrained).")
        if args.film_stiefel:
            # (review Med2) the freeze invariant is "only `code` trains"; --film-stiefel projects
            # model.film.weight (a FROZEN decoder param) every step, mutating a frozen weight OUTSIDE
            # the optimizer/freeze mechanism = a freeze-invariant violation AND a silent no-op for the
            # cure (the decoder is fixed, so there is nothing to orthonormalize the trajectory of).
            raise ValueError("--film-stiefel is incompatible with --freeze-decoder-fit-codes: the "
                             "Stiefel projection mutates the FROZEN decoder's film.weight every step "
                             "(violates the 'only code trains' freeze invariant). Run the Stiefel cure "
                             "on a joint (unfrozen) run.")
        from mlx.utils import tree_unflatten
        dec = _load_decoder_params(Path(args.freeze_decoder_fit_codes))
        got_in = int(dec["in_proj.weight"].shape[1])
        if got_in != in_feat:
            raise ValueError(
                f"--freeze-decoder-fit-codes in_feat MISMATCH: the decoder's in_proj expects {got_in} "
                f"but the current front-end config yields in_feat={in_feat}. Match the decoder's "
                "training config (--bank-*/--max-bank-freq/--self-orient/--n-dir-freqs) so the curvelet"
                "[+dir] feature width agrees; NO-FAKE: refusing to fit codes against a width-mismatched "
                "decoder.")
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in dec.items()]))
        mx.eval(model.parameters())
        model.freeze(recurse=True)
        model.unfreeze(keys=["code"])
        tnames = sorted(k for k, _ in tree_flatten(model.trainable_parameters()))
        if tnames != ["code"]:
            raise RuntimeError(
                f"--freeze-decoder-fit-codes: expected ONLY 'code' trainable after freeze, got {tnames} "
                "(MLX freeze/unfreeze contract changed); fail-closed so the decoder cannot silently train.")
        print(json.dumps({"stage": "freeze_decoder_fit_codes", "decoder_from": str(args.freeze_decoder_fit_codes),
                          "in_feat": int(in_feat), "trainable": tnames, "n_code_params": int(model.code.size),
                          "note": "shared decoder FROZEN (no weight-decay drift); fitting per-pair codes only "
                          "(amortization fast path -- viability per the small-n generalization estimate)"}), flush=True)
    # ---- #224 (3) warp-real-luma frame0 POSE CARRIER build + CHILD-ATTACH (BEFORE EMA/opt so the
    # EMA shadow + AdamW/Muon state + checkpoints all track the carrier residual through the SAME
    # machinery). DEFAULT OFF (--pose-carrier) => no attach => model.trainable_parameters() unchanged
    # => value_and_grad/opt/ema BYTE-IDENTICAL. The RENDER dispatch (even code=f0->carrier warp,
    # odd=f1->witness) is wired below at the render-fn assembly (replacing the old fail-closed guard).
    # The residual co-grad rides the ONE nn.value_and_grad(model, ...) (probe-verified: child dxi gets
    # a finite grad; the carrier's self.freeze(["xi_stored"]) keeps the stored twist out of the
    # trainable tree under parent recursion, so the optimizer never corrupts it).
    pose_carrier = None
    pose_carrier_geom = None
    pose_carrier_xi_stored = None
    if bool(getattr(args, "pose_carrier", False)):
        if bool(getattr(args, "freeze_decoder_fit_codes", False)):
            raise ValueError(
                "--pose-carrier is incompatible with --freeze-decoder-fit-codes: the decoder freeze "
                "runs BEFORE the carrier attach and its trainable-set assertion (only 'code') would "
                "either fail or freeze the carrier residual. Run them separately.")
        if float(args.w_pose) <= 0.0:
            raise ValueError(
                "--pose-carrier requires --w-pose > 0: the residual dxi trains ONLY on the realized "
                "d_pose term; with w_pose=0 the carrier stays at the stored-twist init (no co-grad).")
        from tac.boundary_math.warp_real_luma_frame0 import (
            GroundHomographyGeom as _PCGeom,
            WarpRealLumaFrame0Carrier as _PCCarrier,
            warp_frame0_uint8_numpy as _pc_warp_uint8_np,
            xi_from_pose_calibration as _pc_xi_from_calib,
        )
        _pc_nat_h, _pc_nat_w = int(np.asarray(gt.gt_f0[0]).shape[0]), int(np.asarray(gt.gt_f0[0]).shape[1])
        pose_carrier_geom = _PCGeom.eon(native_hw=(_pc_nat_h, _pc_nat_w), pitch=float(args.pose_carrier_pitch))
        _pc_sr = float(args.pose_carrier_s_r)
        _pc_pitch = float(args.pose_carrier_pitch)
        if args.pose_carrier_s_t is not None:
            _pc_st = float(args.pose_carrier_s_t)
            _pc_fit = None
        else:
            # self-calibrating s_t fit on the frozen CPU-torch PoseNet d_pose grid (mirrors
            # tools/measure_warp_real_luma_frame0_dpose): deterministic, GT-derived, NEVER MPS.
            _pc_nf = max(1, min(int(args.pose_carrier_fit_pairs), P))
            _pc_grid = [0.0, 0.02, 0.044, 0.08, 0.12, 0.16, 0.22, 0.30]

            def _pc_mean_dpose(_st: float) -> float:
                f0s = [np.asarray(gt.gt_f0[p]) for p in range(_pc_nf)]
                preds = [_pc_warp_uint8_np(
                    f0s[p], _pc_xi_from_calib(np.asarray(gt.gt_poses[p]), _st, _pc_sr, _pc_pitch),
                    pose_carrier_geom) for p in range(_pc_nf)]
                dps = cpu_verdict_d_pose_batch(
                    posenet_cpu, f0s, preds, [np.asarray(gt.gt_poses[p]) for p in range(_pc_nf)])
                return float(np.mean(dps))

            _pc_fit = {s: _pc_mean_dpose(s) for s in _pc_grid}
            _pc_st = float(min(_pc_fit, key=_pc_fit.get))
        pose_carrier_xi_stored = np.stack([
            _pc_xi_from_calib(np.asarray(gt.gt_poses[p]), _pc_st, _pc_sr, _pc_pitch)
            for p in range(P)]).astype(np.float32)
        _pc_code_dim = int(args.mod_dim) if str(args.pose_carrier_residual_mode) == "film" else None
        pose_carrier = _PCCarrier.build(
            pose_carrier_xi_stored, pose_carrier_geom,
            residual_mode=str(args.pose_carrier_residual_mode),
            residual_scale=float(args.pose_carrier_residual_scale),
            code_dim=_pc_code_dim, film_hidden=32)
        mx.eval(pose_carrier.parameters())
        model.pose_carrier = pose_carrier.impl   # child-attach: dxi joins model.trainable_parameters()
        mx.eval(model.parameters())
        print(json.dumps({"stage": "pose_carrier", "residual_mode": str(args.pose_carrier_residual_mode),
                          "source": str(getattr(args, "pose_carrier_source", "real_keyframe")),
                          "s_t": round(_pc_st, 5), "s_r": _pc_sr, "pitch": _pc_pitch,
                          "s_t_fit": ({str(k): round(v, 3) for k, v in _pc_fit.items()} if _pc_fit else None),
                          "native_hw": [_pc_nat_h, _pc_nat_w], "n_pairs": P,
                          "note": (("STORE-NOTHING: frame0 = warp(witness's OWN render, xi); stores ONLY "
                                    "xi/H (~0 marginal bytes)" if str(getattr(args, "pose_carrier_source",
                                    "real_keyframe")) == "generated" else
                                    "frame0 real-luma SE(3)-warp pose carrier (stored keyframe)")
                                   + "; residual co-grad via child-attach (ONE value_and_grad + opt + EMA); "
                                     "advisory; pointer 0.19110 UNMOVED")}),
              flush=True)

    ema = MlxEMA(model, decay=args.ema_decay)
    # (THETA* TIER-2 MUST-3) SWA / wider-finisher EMA. DEFAULT-OFF: --ema-decay-finisher None =>
    # ema_finisher_decay None => the loop NEVER mutates ema.decay => the EMA trajectory is
    # BIT-IDENTICAL to the --ema-decay path. When set, from the resolved finisher-start epoch onward
    # the EMA update uses the WIDER decay (averages over the late oscillation -> a flat-basin center,
    # SWA-style). Start resolves to --ema-decay-finisher-start-epoch, else --muon-start-epoch (the
    # natural finisher boundary). main() guards range + start-resolvability when the decay is set.
    ema_finisher_decay = (float(args.ema_decay_finisher)
                          if getattr(args, "ema_decay_finisher", None) is not None else None)
    ema_finisher_start = (int(args.ema_decay_finisher_start_epoch)
                          if getattr(args, "ema_decay_finisher_start_epoch", None) is not None
                          else (int(args.muon_start_epoch) if args.muon_start_epoch is not None else None))
    # #224 Wave C FIX-1: bias_correction ON only on the high-beta2 all-levers path (0.9999999); at the
    # 0.999 default it stays MLX-default False => BYTE-IDENTICAL. Without it high beta2 => ~100x step-1
    # LR blowup => divergence (see _adam_bias_correction_for).
    _adam_bc = _adam_bias_correction_for(getattr(args, "adam_beta2", 0.999))
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay,
                      betas=[0.9, float(getattr(args, "adam_beta2", 0.999))],
                      bias_correction=_adam_bc)

    # ---- RESIDUAL-ONLY MODE (v2 hybrid; gap #1). Load the FIXED deterministic bulk + the
    # bulk-derived composition mask, and build the compose hooks. The bulk arrays live in CLOSURE
    # SCOPE (NOT model attributes) -> they are NEVER in model.parameters() => the EMA / optimizer /
    # quantized blob / checkpoints see ONLY the INR (the bulk does NOT ship; THAT is the rate win).
    # Every realized render (loss + levers + verdict) routes through ``_render_R`` / ``_compose_np``
    # so the d_seg loss is on the COMPOSED witness (bulk (+) INR). Default OFF => _render_R is the
    # bare render + _compose_np is None => byte-identical to the full-partition witness.
    residual_mode = bool(getattr(args, "residual_mode", False))
    _compose_np = None
    _render_R = render_through_R_mlx
    # BUILD #300 (a): default alias for the witness-alone island render. Re-bound below (in the
    # AA/chain block) to a seed-EXCLUDED render when a compose chain is active; the bare-render default
    # here means the OFF path (no chain) never pays a 2nd forward (== _render_R).
    _render_R_wa = render_through_R_mlx
    if residual_mode:
        from tac.v2_compose.residual_compose import load_residual_training_bundle

        _rb = load_residual_training_bundle(Path(args.residual_target_npz))
        if (_rb.render_h, _rb.render_w) != (render_h, render_w):
            raise ValueError(
                f"--residual-target-npz render res {(_rb.render_h, _rb.render_w)} != "
                f"--render-h/--render-w {(render_h, render_w)}: the composition is elementwise at "
                "render res, so they MUST match.")
        if _rb.n_pairs < P:
            raise ValueError(
                f"--residual-target-npz has {_rb.n_pairs} pairs < --num-pairs {P}: the bundle must "
                "cover every trained pair (a larger bundle is fine -- the first P are used).")
        _bulk_rgb_np = np.asarray(_rb.bulk_rgb_render_res[:P], np.float32)   # (P,H,W,3) pre-R RGB
        _resid_mask_np = np.asarray(_rb.composition_mask[:P], bool)          # (P,H,W) override region
        _bulk_rgb_mx = mx.array(_bulk_rgb_np)                                # (P,H,W,3)
        _resid_mask_mx = mx.array(_resid_mask_np.astype(np.float32))[..., None]  # (P,H,W,1)

        def _compose_mx(rgb_nhwc, code_idx):
            # composed = where(mask, INR, bulk) = bulk*(1-m) + INR*m. ``code_idx`` is the per-frame
            # index; the bulk frame is shared across f0/f1 of a pair => pair = code_idx // 2. The
            # bulk is a CONSTANT (no grad) => gradients flow ONLY through the masked residual region.
            pair = int(code_idx) // 2
            m = _resid_mask_mx[pair][None]                # (1,H,W,1)
            return _bulk_rgb_mx[pair][None] * (1.0 - m) + rgb_nhwc * m

        # (#224) the per-frame _render_R that chains _compose_mx is built in the UNIFIED RENDER
        # PATH block below (so residual bulk composes with the analytic-lane band; AA SUPERSAMPLE
        # now ALSO composes — #220 unblock: the AA render invokes compose_fn AFTER box-downsample,
        # at the base grid, so the (H,W) residual mask composes by construction).
        def _compose_np(rgb_hw3, pi):  # noqa: F811 (residual override)
            m = _resid_mask_np[pi][..., None]             # (H,W,1)
            return np.where(m, np.asarray(rgb_hw3, np.float32), _bulk_rgb_np[pi])

        print(json.dumps({"stage": "residual_mode", "npz": str(args.residual_target_npz),
                          "n_pairs": int(_rb.n_pairs), "learn_classes": list(_rb.learn_classes),
                          "dilate": int(_rb.dilate),
                          "composition_override_frac": float(_resid_mask_np.mean()),
                          "note": "INR trains on the COMPOSED-render d_seg (bulk (+) INR); the bulk "
                          "is OUTSIDE the counted weights (rate win). advisory; pointer UNMOVED 0.19110"}),
              flush=True)

    # =====================================================================================
    # #224 UNIFIED RENDER PATH — compose the render-side levers onto _render_R (the per-frame
    # realized render used by make_loss_fn(render_fn=...) AND the shared seg-lever forward). DEFAULT
    # (no residual / no AA / no lane-band / no pose-carrier) => _render_R stays render_through_R_mlx +
    # render_fn=None => BYTE-IDENTICAL. Each lever is opt-in. Docs: analytic_lane_render_band /
    # aa_sdf_observation_render wire-in specs.
    # -------------------------------------------------------------------------------------
    _aa_on = (render_aa == "supersample" and aa_ss > 1)
    _band_active = bool(getattr(args, "lane_render_band", False))
    # (review MED-3 -> #220 UNBLOCK) the base-grid composers (band / residual bulk / island seed)
    # now COMPOSE with the ss*grid AA render (compose_fn runs AFTER box-downsample, at the base
    # grid). The guard is KEPT (currently accepts everything) as the fail-closed home for any
    # future fine-grid-only composer. Pure fn -> unit-tested.
    _validate_aa_compose_compat(
        _aa_on, _band_active, residual_mode, bool(getattr(args, "seed_islands", False)))
    # (2) analytic-lane render-band compose_fn (FEED-dv #203/#213/#215). Precompute the per-code
    # LaneBandPrior ONCE from the frozen GT class-1 mask; ride the witness margin (#141) as the
    # FP-killer uncertainty gate. compose_fn coverage/u_mask are stop-grad constants; the gradient
    # flows through the witness rgb + the witness-derived lane appearance.
    band_compose_fn = None
    band_gate = {"on": False}
    _band_start = int(getattr(args, "lane_band_start_epoch", 300))
    if _band_active:
        # #224 Option-B WIRE-IN (self-orient composable): BOTH the witness margin provider
        # (call_margin) AND the lane RGB provider (render_lane_appearance) feed the model in_proj,
        # which expects base+dir_w feats when --self-orient is on. The pre-Option-B code hardcoded
        # the shared no-self-orient coord_feats_mx -> MLX matmul shape crash under --self-orient at
        # --lane-band-start-epoch. FIX: feed the PER-PAIR self-orient feats (base curvelet + this
        # pair's live dir feats) via _band_feats(code_idx) below (mirrors _cf_mx). NO-FAKE: when
        # --self-orient is OFF this returns the SAME shared coord_feats_mx object (numerically
        # byte-identical to the pre-Option-B measured no-self-orient band path); when ON it returns
        # mx.array(_feats_np_for_pair(pair)) = base curvelet ⊕ this pair's dir feats (zeros pre-first-
        # reorient -> pure-curvelet width base+dir_w -> correct in_proj shape from epoch 0). Sister
        # of the --render-aa supersample + --self-orient wire-in below.
        def _band_feats(code_idx):
            # base-grid per-pair coord feats for the band providers, via the canonical _cf_mx accessor
            # (late-bound; _cf_mx + cf_mx_cache are defined below at main scope and exist by the time
            # band_compose_fn calls this during training). _cf_mx returns the shared coord_feats_mx
            # when no self-orient (exact-object-identical to the measured path) and the already-synced
            # per-pair cf_mx_cache[pi] when self-orient -- BIT-IDENTICAL to mx.array(_feats_np_for_pair
            # (pi)) (rebuild_per_pair_feats_in_place guarantees it) but REUSES the cache rebuilt after
            # every reorient instead of a fresh full-res np.concatenate + mx.array per call (senior-
            # review efficiency fix: kills ~2400 redundant full-res rebuilds/epoch once the band gate
            # opens at --lane-band-start-epoch; serves the shortest-train / MLX-first discipline).
            return _cf_mx(int(code_idx) // 2)
        from tac.boundary_math.analytic_lane_render_band import (
            build_analytic_lane_band_prior,
            make_lane_band_compose_fn,
        )
        _lane_priors: dict[int, Any] = {}
        if bool(getattr(args, "lane_band_dash_comb", False)):
            # #287 EGO-PHASE DASH COMB (dash_erasure_homogenization_v1 corrector): the per-pair
            # FITTED dash phase (1 float/line/pair) is replaced by the world-static comb — global
            # (period, duty, ego-scale) + per-slot world phase, transported to each pair by the
            # cumulative ego forward distance from gt_poses[:,0] (phase-from-ξ, #215). Render-time
            # only; the comb MODULATES the band coverage, geometry/appearance/margin unchanged.
            from tac.boundary_math.dash_comb import build_combed_lane_band_priors
            _comb_priors, _comb_fit = build_combed_lane_band_priors(
                np.stack([np.asarray(gt.lstars[_pi]) for _pi in range(P)]),
                np.stack([np.asarray(gt.gt_poses[_pi], np.float64) for _pi in range(P)]),
                lane_cls=1, softness=float(args.lane_band_softness),
                dash_forward_max_m=float(args.lane_band_dash_forward_max_m),
                comb_softness_m=float(args.lane_band_comb_softness_m))
            for _pi in range(P):
                _lane_priors[2 * _pi + 1] = _comb_priors[_pi]  # frame1 (SegNet-scored)
                _lane_priors[2 * _pi] = _comb_priors[_pi]      # frame0 seg-free; symmetric
            print(json.dumps({"stage": "lane_band_dash_comb", "n_pairs": P,
                              "period_m": round(float(_comb_fit.period_m), 4),
                              "duty": round(float(_comb_fit.duty), 4),
                              "ego_scale": round(float(_comb_fit.scale), 6),
                              "transported_slots": sorted(
                                  int(_s) for _s, _t in _comb_fit.transported_by_slot.items() if _t),
                              "anchor_floats": int(_comb_fit.n_anchor_floats()),
                              "mean_pairwise_concentration": round(
                                  float(_comb_fit.mean_pairwise_concentration), 4),
                              "concentration_at_zero_scale": round(float(_comb_fit.concentration_at_zero_scale), 4),
                              "n_dashed_fits": int(_comb_fit.n_dashed_fits),
                              "note": "#287 ego-phase comb replaces per-pair fitted dash phase; "
                              "per-slot transported-vs-static + anchored/pairwise concentration "
                              "MEASURE the phase-from-xi transport quality; advisory; "
                              "pointer 0.19110 UNMOVED"}), flush=True)
        else:
            for _pi in range(P):
                _prior = build_analytic_lane_band_prior(
                    np.asarray(gt.lstars[_pi]), lane_cls=1, softness=float(args.lane_band_softness),
                    dash_gate=True, dash_forward_max_m=float(args.lane_band_dash_forward_max_m))
                _lane_priors[2 * _pi + 1] = _prior   # frame1 (the SegNet-scored frame)
                _lane_priors[2 * _pi] = _prior       # frame0 seg-free; keep symmetric
        _u_src = str(args.lane_band_uncertainty_source)
        if _u_src == "witness":
            def _band_margin_provider(code_idx):
                # per-pair self-orient feats (base curvelet ⊕ this pair's dir feats) via _band_feats;
                # == shared coord_feats_mx when --self-orient is OFF (measured no-self-orient config).
                return mx.stop_gradient(
                    model.call_margin(_band_feats(code_idx), int(code_idx))).reshape(render_h, render_w)
            _margin_provider: Any = _band_margin_provider
        elif _u_src == "gt":
            _margin_provider = {c: mx.array(np.asarray(gt.margins[c // 2], np.float32)) for c in _lane_priors}
        else:
            _margin_provider = None

        def _band_lane_rgb(code_idx):
            return model.render_lane_appearance(_band_feats(code_idx), int(code_idx), lane_cls=1).reshape(
                render_h, render_w, 3)

        band_compose_fn = make_lane_band_compose_fn(
            _lane_priors, lane_rgb_provider=_band_lane_rgb, margin_provider=_margin_provider,
            tau=float(args.lane_band_tau), eps=float(args.lane_band_eps),
            weight=float(args.lane_band_weight), use_mlx=True)
        band_gate["on"] = _band_start <= 1
        _band_recalls = [float(_lane_priors[2 * pi + 1].band_recall) for pi in range(P)
                         if np.isfinite(_lane_priors[2 * pi + 1].band_recall)]
        print(json.dumps({"stage": "lane_render_band", "n_pairs": P, "uncertainty_source": _u_src,
                          "start_epoch": _band_start,
                          "band_vs_gt_lane_recall_mean": (round(float(np.mean(_band_recalls)), 4)
                                                          if _band_recalls else None),
                          "note": "class-1 render-time authority composited PRE-R; gated at start_epoch "
                          "(spike-guard re-treat); advisory; pointer 0.19110 UNMOVED"}), flush=True)
    # #224 (5) island SEED compose state (LATE-BOUND; populated at the seed build below, which runs
    # AFTER this chain is defined but BEFORE value_and_grad + the training loop, so _compose_chain
    # reads it at CALL time). The seed is a SEPARATE module (own optimizer group) -> NOT in
    # model.parameters()/EMA/blob/deploy, so the verdict (witness-alone) == the 0-byte-accelerant
    # deploy (NO-FAKE, honestly measured). Default OFF (--seed-islands) => seed_state stays empty =>
    # the seed branch never fires => _compose_chain BYTE-IDENTICAL.
    seed_on = bool(getattr(args, "seed_islands", False))
    # BUILD #300 (SEED-ABSORPTION FIX; root cause of the CE plateau = seed-compose island-gradient
    # starvation, memo plateau_disambiguator_results_20260704.md / memory
    # seed_compose_island_gradient_starvation_the_crutch_that_blocks_learning). Two coupled DEFAULT-OFF
    # mechanisms: (a) --witness-alone-island-loss routes the island-FORMATION levers (island amplify +
    # persistence) through the seed-EXCLUDED render so the witness gets the absorption gradient the seed
    # was starving; (b) --seed-anneal-epochs ramps the seed compose weight full->0 across CE (transfer
    # schedule). Both OFF => byte-identical. (a) requires a seed to exclude (else inert -> fail closed).
    wa_island = bool(getattr(args, "witness_alone_island_loss", False))
    if wa_island and not seed_on:
        raise ValueError(
            "--witness-alone-island-loss requires --seed-islands: it routes the island-formation "
            "levers through the seed-EXCLUDED (witness-alone) render; with no seed there is nothing "
            "to exclude (the composed frame already == the witness render) => the flag is inert.")
    seed_anneal_epochs = int(getattr(args, "seed_anneal_epochs", 0))
    seed_anneal_shape = str(getattr(args, "seed_anneal_shape", "linear"))
    # compose_w: the epoch-annealed island-seed compose weight (1.0 => byte-identical seed compose;
    # ramped full->0 by the epoch loop when --seed-anneal-epochs > 0). Read LIVE by _compose_chain.
    seed_state: dict[str, Any] = {"mod": None, "masks": None, "compose_w": 1.0}
    # assemble the compose chain (residual bulk FIRST, then lane band, then island seed). None => bare.
    _use_chain = residual_mode or _band_active or seed_on

    def _compose_chain(rgb_nhwc, code_idx):
        if residual_mode:
            rgb_nhwc = _compose_mx(rgb_nhwc, code_idx)
        if _band_active and band_gate["on"]:
            rgb_nhwc = band_compose_fn(rgb_nhwc, code_idx)
        if seed_state["mod"] is not None and (int(code_idx) % 2 == 1):
            # frame1 (SegNet-scored) ONLY: add the protected per-pair island seed residual (masked to
            # the self-detected island support). Reads the LIVE seed_mod.residual -> the dual
            # value_and_grad co-differentiates it; the compose flows through the SHARED _f1 -> _slog ->
            # _signed (no 2nd SegNet). frame0 (even) is seg-free -> unseeded.
            # BUILD #300 (b): scale by the epoch-annealed compose weight (transfer schedule full->0).
            # compose_w == 1.0 (DEFAULT / --seed-anneal-epochs 0) => the `!= 1.0` guard is False => the
            # extra multiply is NEVER emitted => BYTE-IDENTICAL to the pre-#300 seed compose.
            _pi = int(code_idx) // 2
            _sd = seed_state["mod"].residual[_pi] * seed_state["masks"][_pi]
            _cw = seed_state.get("compose_w", 1.0)
            if _cw != 1.0:
                _sd = _sd * _cw
            rgb_nhwc = rgb_nhwc + _sd
        return rgb_nhwc

    def _compose_chain_noseed(rgb_nhwc, code_idx):
        # BUILD #300 (a): the WITNESS-ALONE compose chain -- residual bulk + lane band (the deploy-time
        # composers) but WITHOUT the deploy-EXCLUDED island seed. Used to route the island-FORMATION
        # levers (island amplify + persistence) through the seed-EXCLUDED render so the witness gets the
        # absorption gradient the seed was starving (memo plateau_disambiguator_results_20260704.md).
        # It is ONLY invoked when the seed is live AND --witness-alone-island-loss is set (see
        # total_loss_fn); the OFF path never calls it. When there is no seed this is identical to
        # _compose_chain (the seed branch is a no-op), so no correctness gap if it were called.
        if residual_mode:
            rgb_nhwc = _compose_mx(rgb_nhwc, code_idx)
        if _band_active and band_gate["on"]:
            rgb_nhwc = band_compose_fn(rgb_nhwc, code_idx)
        return rgb_nhwc

    # (1) AA supersample render dispatch: IGNORE the passed base-grid coord_feats and use the
    # fine-grid feats (the base-grid eikonal/sdf(cf) in total_loss_fn is unaffected -> still base grid).
    if _aa_on or _use_chain:
        from tac.boundary_math.aa_sdf_observation_render import (  # noqa: E402
            render_aa_through_R_mlx as _render_aa_R,
        )

        def _render_R(witness, coord_feats, code_idx, rh, rw):  # noqa: F811 (unified #224 override)
            _cf = _compose_chain if _use_chain else None
            if _aa_on:
                # per-pair FINE feats when --self-orient (shared curvelet-fine [+ fine dir-feats]);
                # else the shared curvelet-fine tensor. _cf_fine_mx is late-bound (defined below).
                _feats_fine = _cf_fine_mx(int(code_idx) // 2) if _aa_so_fine else coord_feats_fine_mx
                return _render_aa_R(witness, _feats_fine, code_idx, rh, rw, aa_ss, compose_fn=_cf)
            return render_through_R_mlx(witness, coord_feats, code_idx, rh, rw, compose_fn=_cf)

        def _render_R_wa(witness, coord_feats, code_idx, rh, rw):  # noqa: F811 (BUILD #300 wa override)
            # BUILD #300 (a): witness-alone render (island seed EXCLUDED via _compose_chain_noseed) for
            # the island-formation levers. Mirrors _render_R exactly but with the no-seed compose chain
            # so amplify/persistence push the WITNESS (not the seed) to form Lane+Movable itself.
            _cf = _compose_chain_noseed if _use_chain else None
            if _aa_on:
                _feats_fine = _cf_fine_mx(int(code_idx) // 2) if _aa_so_fine else coord_feats_fine_mx
                return _render_aa_R(witness, _feats_fine, code_idx, rh, rw, aa_ss, compose_fn=_cf)
            return render_through_R_mlx(witness, coord_feats, code_idx, rh, rw, compose_fn=_cf)

    render_fn = _render_R if (_aa_on or _use_chain) else None

    # (3) warp-real-luma frame0 pose carrier — the parity-dispatch render_fn (even code=f0 -> the
    # SE(3) ground-homography warp of the REAL keyframe luma; odd=f1 -> witness). The carrier BUILD +
    # child-ATTACH (residual co-grad through the ONE value_and_grad + AdamW/Muon + EMA) happened
    # ABOVE, pre-EMA; here we only WRAP render_fn with the parity dispatch. The measured s_t/s_r/pitch
    # calibration is self-fit at build (or --pose-carrier-s-t); the residual dxi co-grad rides the
    # child-attach. Default OFF (pose_carrier is None) => render_fn unchanged => byte-identical.
    if pose_carrier is not None:
        _pc_witness_render = _render_R if (_aa_on or _use_chain) else render_through_R_mlx
        _pc_source_generated = str(getattr(args, "pose_carrier_source", "real_keyframe")) == "generated"

        _pc_code_provider = None
        if str(args.pose_carrier_residual_mode) == "film":
            def _pc_code_provider(pi: int):
                return model.code[2 * pi + 0]   # frame0 per-pair code for the FiLM residual MLP

        if _pc_source_generated:
            # #205 STORE-NOTHING-but-xi: frame0 = warp(the witness's OWN plain frame0 render, xi_eff).
            # NO stored keyframe -> stores ONLY xi/H (~0 marginal bytes; the render is FREE, rule-118).
            # The plain (no-compose) witness f0 render is up-sampled to camera-native (the R "up" step,
            # == the byte-close store_nothing warp source _R), then the carrier warps it + R-downs to
            # SEG. The dxi residual co-grads THROUGH the witness f0 render (the co-adaptation).
            from tac.local_acceleration.pr95_hnerv_mlx_training import (
                CAMERA_HW as _PC_CAMERA_HW,
                apply_contest_faithful_roundtrip_nhwc as _pc_up_to_camera,
            )
            _pc_impl = pose_carrier.impl

            def render_fn(model, coord_feats, code_idx, rh, rw):
                if int(code_idx) % 2 == 1:  # f1 -> witness render (drives d_seg)
                    return _pc_witness_render(model, coord_feats, code_idx, rh, rw)
                # f0 -> STORE-NOTHING: the witness's OWN plain frame0 render, up to camera-native, warped.
                pair_idx = int(code_idx) // 2
                rgb = mx.reshape(model(coord_feats, code_idx), (1, rh, rw, 3))
                src_native = _pc_up_to_camera(rgb, output_hw=_PC_CAMERA_HW, ste_round=True)[0]
                code_vec = model.code[2 * pair_idx] if (_pc_code_provider is not None) else None
                return _pc_impl.render_f0(src_native, pair_idx, code_vec, ste_round=True)
        else:
            def _pc_gt_f0_provider(pi: int):
                # native-res (H,W,3) REAL keyframe luma as mx float32; per-call (transient, no P-length
                # fp32 cache -> n600-memory-safe; the uint8 GT already resides in gt.gt_f0).
                return mx.array(np.asarray(gt.gt_f0[pi], np.float32))

            render_fn = pose_carrier.make_pair_render_dispatch(
                _pc_witness_render, _pc_gt_f0_provider, code_provider=_pc_code_provider)
        print(json.dumps({"stage": "pose_carrier_render_dispatch", "residual_mode": str(args.pose_carrier_residual_mode),
                          "source": ("generated" if _pc_source_generated else "real_keyframe"),
                          "witness_render": ("aa/chain" if (_aa_on or _use_chain) else "bare"),
                          "note": ("STORE-NOTHING: frame0 = warp(witness's OWN render, xi); stores ONLY xi/H"
                                   if _pc_source_generated else
                                   "parity dispatch (even code=f0->carrier warp of the stored real keyframe, "
                                   "odd=f1->witness)") + "; advisory; pointer 0.19110 UNMOVED"}), flush=True)

    # (--seg-focal-gamma, council levelset-loss-geometry symposium 2026-07-05; DEFAULT 0.0 =>
    # make_loss_fn's focal branch NEVER runs => loss + grads BYTE-IDENTICAL). The reweight applies
    # inside base_loss on the SAME seg_logits surface the base form reads — i.e. the render_fn-
    # composed frame (SEED-COMPOSED when --seed-islands; the island-formation levers' witness-alone
    # #300 routing below is UNTOUCHED). Calibrated gamma* comes from
    # experiments/probe_focal_gamma_calibration.py (measured, never guessed).
    focal_gamma = float(getattr(args, "seg_focal_gamma", 0.0))
    # (#218 BUILD, FEED-07b lever #3) LOGIT-ADJUSTMENT per-class offset (Menon et al. 2021,
    # arXiv 2007.07314 — the textbook ZERO-BYTE rare-class cure). TRAINING-time LOSS surface ONLY:
    # the frozen-SegNet logits base_loss reads get ``logits_c += tau * log(prior_c)`` with priors
    # = the GT class-area fractions from the cached L* (measured n600 ~[0.232, 0.0059, 0.495,
    # 0.0124, 0.254] — Lane/Movable get strongly negative log-priors, so under-predicting them
    # costs more gradient). BYTE-IDENTITY BOUNDARY (binding, documented on _LogitAdjustSegAdapter):
    # the DEPLOYED/rendered argmax path is UNCHANGED — no offset at the verdict CPU-torch SegNet,
    # the byte-close decode, or inflate; the adjustment lives ONLY inside the wrapped loss adapter.
    # tau == 0.0 (DEFAULT) => ``_loss_adapter is adapter`` (the SAME object) => the make_loss_fn
    # closure + graph are BYTE-IDENTICAL. Fails closed with --micro-batch-pairs>1 (not routed into
    # the batched twin — same class as --seg-spike-reweight / --margin-saliency-reachability).
    # Equations leg: ``logit_adjustment_class_prior_law_v1``; DSL leg: ``LogitAdjust``.
    la_tau = float(getattr(args, "logit_adjust_loss_tau", 0.0))
    _validate_logit_adjust_compat(la_tau, int(getattr(args, "micro_batch_pairs", 1)))
    _loss_adapter = adapter
    if la_tau != 0.0:
        _la_off, _la_priors = _logit_adjust_offsets_np(
            [np.asarray(gt.lstars[_pi]) for _pi in range(P)], la_tau, n_classes=5)
        _loss_adapter = _LogitAdjustSegAdapter(adapter, mx.array(_la_off))
        print(json.dumps({"stage": "logit_adjust", "tau": la_tau,
                          "priors": [round(float(x), 5) for x in _la_priors],
                          "offsets": [round(float(x), 4) for x in _la_off],
                          "note": "Menon logit-adjusted seg loss (training-LOSS surface only; "
                                  "deployed argmax/verdict/byte-close read RAW logits); "
                                  "advisory; pointer 0.19110 UNMOVED"}), flush=True)
    base_loss = make_loss_fn(
        _loss_adapter, render_h, render_w, score_domain=args.score_domain_loss, pose_eps=args.pose_eps,
        seg_loss=args.seg_loss, tau_softplus_tau=args.tau_softplus_tau, l7_mult=args.l7_mult,
        l7_threshold=args.l7_threshold,
        render_fn=render_fn,
        focal_gamma=focal_gamma,
    )

    # LEVER-3 (lane-edge fragility weighting) hyperparameters captured from args (static; closure
    # constants, NOT value_and_grad args -> ZERO change to the call site). lane_edge_weight=0.0
    # (default) => the branch below is skipped => behavior IDENTICAL to before (fully additive).
    lane_w = float(args.lane_edge_weight)
    lane_cls = int(args.lane_edge_class)
    lane_tgt = float(args.lane_margin_target)
    lane_start = int(args.lane_edge_start_epoch)
    # OPTIMAL-FORM (recursive review, FEED-df): the lane margin hinge is a margin-SHARPENING loss;
    # running it from ep0 during the COARSE ce stage risks the known margin-from-scratch-starves-
    # interior failure (the partition isn't formed yet). ``lane_gate`` is a python bool RE-READ
    # inside total_loss_fn each value_and_grad call (so the lane branch is included/excluded per
    # epoch); the epoch loop sets it = (ep >= lane_start). Default lane_start=0 => engaged from ep1
    # = IDENTICAL to before (fully additive). When lane_start>1 the engagement epoch RE-TREATS the
    # spike-guard (clears recent_losses) so the loss jump from the added term is NOT silently
    # spike-skipped (operator 2026-06-26 "different stages need different treatment ... transitions
    # must re-treat"; margin-engage spike-skip is the named failure this prevents).
    lane_gate = {"on": lane_start <= 1}

    # LEVER-4 (margin-saliency) closure constants (static; ZERO change to the value_and_grad call
    # site). msal_w=0.0 (default) => the branch is skipped => behavior IDENTICAL (fully additive).
    msal_w = float(args.margin_saliency_weight)
    msal_tau = float(args.margin_saliency_tau)
    msal_tgt = float(args.margin_saliency_target)
    msal_start = int(args.margin_saliency_start_epoch)
    msal_uni = bool(args.margin_saliency_uniward)
    msal_uni_beta = float(args.margin_saliency_uniward_beta)
    msal_gate = {"on": msal_start <= 1}
    # LEVER-4 REACHABILITY (default-off; REPLACES the texture path when on): per-pair through-R
    # margin-Jacobian S_R weight. _sR_provider is a list[mx.array (1,H,W)] indexed by pi==int(c1)//2
    # (same key as island_weight_mx); it is POPULATED after lstar_cache is built (see the build site)
    # ONLY when msal_reach AND msal_w>0. Declared None here so the closure name always exists and the
    # OFF path (msal_reach False) NEVER references it => byte-identical to the pre-reachability code.
    msal_reach = bool(getattr(args, "margin_saliency_reachability", False))
    _sR_provider: Any = None

    # LEVER-4b (SUB-PIXEL BOUNDARY-PLACEMENT `t`, DIRECTIONAL upgrade of the scalar margin-saliency
    # #141; asymmetry probe a8afad40 GREEN 2026-07-03). The cross-boundary GT margin RATIO
    # t = M_GT[p] / (M_GT[p] + M_GT[q]) (p,q = the two straddle pixels across an inter-class edge) is a
    # FREE sub-pixel boundary-POSITION localizer LATENT in the already-computed GT margin field (no
    # SegNet forward; pure numpy from gt.margins/gt.lstars). It upgrades LEVER-4's DIRECTIONLESS
    # per-pixel weight -> a SIGNED sub-pixel placement TARGET: where the GT margin V is genuine, supervise
    # the witness's OWN realized margin ratio t_wit = Mw[p]/(Mw[p]+Mw[q]) toward the GT t (a DENSER,
    # sub-pixel, differentiable signal than the argmax weight). Reuses the SHARED realized through-R
    # margin ``_signed`` (Mw = relu(_signed) = witness GT-class margin, the honest mirror of the GT
    # top1-top2 the target is built from) -- NO 2nd SegNet forward (bit-identical to LEVER-4's forward,
    # ``_seg_levers_on`` gated). subpix_w=0.0 (DEFAULT) => the branch is skipped => byte-identical (fully
    # additive). Providers declared None here (closure binds the cells) so the OFF path never references
    # them; POPULATED after lstar_cache is built (spike-map style, inline -- theta-independent + cheap).
    subpix_w = float(getattr(args, "seg_subpix_boundary_weight", 0.0))
    subpix_start = int(getattr(args, "seg_subpix_boundary_start_epoch", 0))
    subpix_band = float(getattr(args, "seg_subpix_boundary_v_band", 1.0))
    subpix_eps = 1e-6
    subpix_gate = {"on": subpix_start <= 1}
    _subpix_t_prov: Any = None     # list[mx.array (1,H,W)] f32, GT t in [0,1] where active, -1.0 sentinel
    _subpix_dir_prov: Any = None   # list[mx.array (1,H,W)] f32 in {0,1}, dominant-straddle dir (0=right,1=down)

    # LEVER-4c (ANNULUS-DIRECTED CHROMA-SHARPENING; chroma DOF probe a3e9f0bd GREEN 2026-07-03; operator
    # 2026-06-25 "Chroma too"; CLAUDE.md "Chroma is a d_seg lever"). Chroma is a PROVEN INDEPENDENT
    # argmax-boundary d_seg actuator: MEASURED n96 (a3e9f0bd, 100% L*-match to the frozen SegNet) removing
    # chroma (constant-luma) flips 7.54% Lane->Road + 4.38% Movable->Undrivable, 93.4% of chroma-flips in
    # the margin<1 ANNULUS (->33.7% at margin<0.25), proven independent of luma (constant-luma DESAT still
    # flips 3.1% of the annulus; margin-gradient energy 78.8% luma / 21.2% chroma). Chroma is a BOUNDARY
    # SHARPENER (power at the knife-edge large-signal flips, not bulk), ORTHOGONAL to the geometry levers
    # (along-tangent-freq / lane-render-band / sub-pixel-t). The witness UNDER-exploits it: its rendered
    # chroma converges to a near per-class CONSTANT palette (the seg CE only rewards argmax; nothing
    # supervises per-pixel chroma) whose inter-class separation (~2.84 Lane/Road) is SMALLER than the
    # intra-class chroma std -> it cannot PAINT the per-pixel boundary chroma SegNet keys on. FIX (this
    # lever): at the fragile annulus supervise the witness's OWN rendered chroma toward the GT chroma
    # (a realized-through-R chroma-MATCH term) so the per-pixel RGB head (self.out = Linear(hidden,3), which
    # HAS per-pixel chroma CAPACITY -- the constant palette is a convergence habit, not a structural
    # ceiling) learns the boundary chroma the constant palette can't. Chroma := rgb - BT.601-luma (the SAME
    # BT.601 the witness _apply_chroma uses); LUMA-INVARIANT by construction (rgb + c*[1,1,1] leaves chroma
    # unchanged) => ORTHOGONAL to every luma lever (NOT a full-RGB reconstruction). RIDES the SHARED
    # rendered frame ``_f1`` (through R) -- NO 2nd render, NO 2nd SegNet forward (``_seg_levers_on`` gated);
    # ``_signed`` (the margin) and ``_f1`` (the RGB) both come from that ONE shared realized-through-R
    # render. Pose synergy (NOTED, not built): pose rides the stored-target sidecar (solved) => the
    # seg-frame's texture chroma is FREE for d_seg (seg (+) pose, orphan #227). chroma_bnd_w=0.0 (DEFAULT)
    # => the branch is skipped => byte-identical (fully additive). Providers declared None here (closure
    # binds the cells) so the OFF path never references them; POPULATED after lstar_cache is built
    # (spike-map / subpix style, inline -- theta-independent + cheap). Fails CLOSED with micro-batch.
    chroma_bnd_w = float(getattr(args, "seg_chroma_boundary_weight", 0.0))
    chroma_bnd_start = int(getattr(args, "seg_chroma_boundary_start_epoch", 0))
    chroma_bnd_band = float(getattr(args, "seg_chroma_boundary_margin_band", 1.0))
    chroma_bnd_gate = {"on": chroma_bnd_start <= 1}
    _chroma_gt_prov: Any = None    # list[mx.array (1,H,W,3)] f32, GT BT.601 chroma at (SEG_H,SEG_W)
    _chroma_w_prov: Any = None     # list[mx.array (1,H,W)] f32 annulus weight (margin<band) in {0,1}

    # (--boundary-distance-weight, council levelset-loss-geometry symposium 2026-07-05) SDF-native
    # Kervadec-style boundary-placement loss closure constants. bd_w=0.0 (DEFAULT) => the branch in
    # total_loss_fn is skipped AND the provider stays None => byte-identical (fully additive). The
    # per-pair GT-boundary band map (distance transform of the GT inter-class edge set, computed ONCE
    # per pair — cacheable/theta-independent) is POPULATED after lstar_cache is built (spike-map
    # style); the term reads the SDF head DIRECTLY (model.sdf(cf, c1), frame1 = the SegNet-scored
    # frame) so the contour is moved on the DOF the witness owns. Fails CLOSED with micro-batch.
    bd_w = float(getattr(args, "boundary_distance_weight", 0.0))
    _bd_band_prov: Any = None   # list[mx.array (1,H,W)] f32 band weights, keyed by pi == int(c1)//2

    # (THETA* TIER-2 MUST-2) nuclear-norm low-rank code penalty closure constants. code_nuc_w=0.0
    # (DEFAULT) => the branch in total_loss_fn is skipped => L is byte-identical (fully additive).
    code_nuc_w = float(getattr(args, "code_nuclear_weight", 0.0))
    code_nuc_eps = float(getattr(args, "code_nuclear_eps", 1e-3))
    code_nuc_iters = int(getattr(args, "code_nuclear_ns_iters", 25))
    # (THETA* TIER-2 STRETCH-1) junction-aware Eikonal relax closure constants. eik_jrelax=0.0
    # (DEFAULT) => _eikonal_length_mlx takes its BIT-IDENTICAL branch (w==1.0) => unchanged.
    eik_jrelax = float(getattr(args, "eikonal_junction_relax", 0.0))
    eik_jtau = float(getattr(args, "eikonal_junction_tau", 0.5))
    # (EIK-STAB build 1) eikonal-stabilizer closure CELL (mutable dict, read LIVE inside
    # total_loss_fn each value_and_grad call — the same pattern as the lever gate dicts; the epoch
    # loop mutates "visco_eps" per the vanishing-viscosity anneal). BOTH default 0.0 => both
    # branches in total_loss_fn are skipped => the loss graph is BYTE-IDENTICAL (fully additive).
    _eik_stab = {
        "steik_w": float(getattr(args, "eikonal_steik_weight", 0.0)),
        # (V6 #317) normalized unit-normal curvature n^T H n instead of raw |grad m^T H grad m|;
        # default False => raw form => byte-identical to the pre-V6 steik branch.
        "steik_normalized": bool(getattr(args, "eikonal_steik_normalized", False)),
        "steik_norm_eps": float(getattr(args, "eikonal_steik_norm_eps", 1e-2)),
        "visco_eps0": float(getattr(args, "eikonal_viscosity", 0.0)),
        "visco_eps": float(getattr(args, "eikonal_viscosity", 0.0)),
        "visco_anneal": int(getattr(args, "eikonal_viscosity_anneal", 0)),
        # (V6 #320) adaptive-eps CFL-edge tracker config. adaptive=False (DEFAULT) => the linear
        # anneal path runs unchanged => visco_eps is set exactly as before => BYTE-IDENTICAL.
        "visco_adaptive": bool(getattr(args, "eikonal_viscosity_adaptive", False)),
        "visco_eps_floor": float(getattr(args, "eikonal_visco_eps_floor", 0.3)),
        "visco_eps_upper": float(getattr(args, "eikonal_visco_eps_upper", 0.7)),
        "visco_margin_factor": float(getattr(args, "eikonal_visco_margin_factor", 0.5)),
        "visco_ca_band": float(getattr(args, "eikonal_visco_ca_band", 0.0)),
        # last measured |c_a(t)| (telemetry only; 0.0 until the first adaptive epoch).
        "visco_c_a": 0.0,
    }
    # (V6 #320) FIXED deterministic strided pair subset for the per-epoch |c_a| measurement (built
    # once; adaptive OFF => never used). Strided over P so the sample spans the sequence.
    _ca_npairs = max(1, int(getattr(args, "eikonal_visco_ca_pairs", 16)))
    _ca_stride = max(1, P // _ca_npairs)
    _ca_pairs = list(range(0, P, _ca_stride))[:_ca_npairs]

    # LEVER-A (FiLM-rank-fix) loss term closure constants. A SOFT participation-ratio FLOOR on the
    # realized per-pair FiLM modulation M = film(code) so the curriculum cannot funnel it to rank-1
    # (MEASURED collapse PR 3.34@CE -> 1.19@l7). rankfloor_w=0.0 (default) => the branch is skipped =>
    # behavior IDENTICAL (fully additive). Computed over a FIXED deterministic subsample of the
    # per-(pair,frame) codes (<= cap, strided) so the S x S Gram is cheap; the penalty is
    # pair-INDEPENDENT, so accumulating it per-pair then averaging counts it ONCE (correct magnitude;
    # redundant compute bounded by the cap). It penalizes the SHARED film route (the measured-collapse
    # determinant); film_pl residual routes are not directly penalized but the shared route dominates
    # the per-pair modulation rank.
    rankfloor_w = float(getattr(args, "film_rank_floor_weight", 0.0))
    rankfloor_tgt = float(getattr(args, "film_rank_floor_target", 4.0))
    rankfloor_idx = None
    if rankfloor_w > 0.0:
        _ncodes = 2 * P
        _cap = 256
        _stride = max(1, _ncodes // _cap)
        rankfloor_idx = mx.array(np.arange(0, _ncodes, _stride)[:_cap].astype(np.int32))

    # DM1b (code spectral-entropy) loss-term closure. A CAPACITY log-barrier -beta*log(PR(cov(code)))
    # on the per-pair code covariance (keeps all ~mod_dim code directions live). Pair-INDEPENDENT (a
    # function of the whole code matrix), so -- exactly like the rank-floor -- accumulating it per-pair
    # then averaging counts it ONCE. PR is computed via the (D,D) covariance Gram (cheap, no eigh),
    # the EXACT MLX twin of tac...code_spectral_entropy_penalty. code_spec_w=0.0 (default) => the
    # branch is skipped => behavior IDENTICAL (fully additive). Same fixed deterministic subsample as
    # the rank-floor so the Gram is bounded.
    code_spec_w = float(getattr(args, "code_spectral_entropy_weight", 0.0))
    code_spec_idx = None
    if code_spec_w > 0.0:
        _ncodes2 = 2 * P
        _cap2 = 256
        _stride2 = max(1, _ncodes2 // _cap2)
        code_spec_idx = mx.array(np.arange(0, _ncodes2, _stride2)[:_cap2].astype(np.int32))

    # LEVER-B (thin-lane dropped-dash prior) closure constants. Up-weight the realized through-R seg
    # margin hinge on THIN GT-lane structures the unweighted mean loss drops (MEASURED: 52.7% of
    # GT-lane connected components wholesale-missed, miss-fraction monotone in dash size). lane_thin_w
    # =0.0 (default) => the branch is skipped => behavior IDENTICAL (fully additive). The per-pair
    # thin-lane weight map (local lane density in a (2r+1)^2 window) is PRECOMPUTED ONCE from the
    # cached L* (deterministic; NOT recomputed per step) and looked up by pair index inside the loss.
    # When lane_thin_start>1 the engagement epoch RE-TREATS the spike-guard (same as LEVER-3/4).
    lane_thin_w = float(getattr(args, "lane_thin_weight", 0.0))
    lane_thin_tgt = float(getattr(args, "lane_thin_target", 0.5))
    lane_thin_cls = int(getattr(args, "lane_thin_class", 1))
    lane_thin_rad = int(getattr(args, "lane_thin_radius", 4))
    lane_thin_start = int(getattr(args, "lane_thin_start_epoch", 0))
    lane_thin_gate = {"on": lane_thin_start <= 1}
    thin_maps_mx = None
    if lane_thin_w > 0.0:
        thin_maps_mx = {
            pi: mx.array(lane_thin_weight_map(
                np.asarray(gt.lstars[pi]), lane_class=lane_thin_cls, radius=lane_thin_rad)[None])
            for pi in range(P)
        }

    # #218 MARGIN-FIELD HEAD levers (facets 1b + 3, BYTE-FREE). A REALIZED through-R per-class margin
    # hinge composing with LEVER-3/4/B on the SHARED _signed. mfh_w=0.0 (default) => branch skipped =>
    # L is byte-identical. Per-pixel margin TARGET b_c = additive-margin (facet-1b, when head==
    # additive-margin) + facet-3 Menon boost on RARE classes: tau*relu(-log pi_c) mean-centered so ONLY
    # rare classes (Lane/Movable) RAISE their target (common classes stay at base). Priors from cached
    # GT L* (deterministic; this is a TRAIN-TIME loss shape => 0 archive bytes). facet-1a (ETF head) is
    # applied at model build above and is orthogonal to this lever.
    mfh_w = float(getattr(args, "margin_field_head_weight", 0.0))
    mfh_target_mx = None
    if mfh_w > 0.0:
        from tac.boundary_math.laguerre_logit_offset import menon_logit_adjustment_offsets
        _mfh_counts = np.bincount(
            np.concatenate([np.asarray(gt.lstars[pi]).reshape(-1) for pi in range(P)]),
            minlength=5).astype(np.float64)
        _mfh_base = float(getattr(args, "additive_margin", 0.0)) if str(getattr(args, "head", "softmax")) == "additive-margin" else 0.0
        _mfh_tgt = np.full(5, _mfh_base, np.float64)
        if bool(getattr(args, "logit_adjust_per_class", False)):
            _mfh_tgt = _mfh_tgt + float(getattr(args, "logit_adjust_tau", 1.0)) * np.maximum(
                menon_logit_adjustment_offsets(_mfh_counts, tau=1.0), 0.0)
        mfh_target_mx = mx.array(_mfh_tgt.reshape(1, 1, 1, 5).astype(np.float32))
        print(json.dumps({"stage": "margin_field_head", "weight": mfh_w,
                          "per_class_margin_target": [round(float(v), 4) for v in _mfh_tgt]}), flush=True)

    # #224 (4) PERSISTENCE/TOPOLOGY loss setup (persistence_topology_loss; #218/TopologyLossGauge).
    # persist_w=0 (default) => persist_classes=() + persist_gate["w"]=0 => branch inert => byte-identical.
    persist_w = float(getattr(args, "persistence_loss_weight", 0.0))
    persist_recall_w = float(getattr(args, "persistence_recall_weight", 1.0))
    persist_cldice_iters = int(getattr(args, "cldice_iters", 5))
    persist_warmup = int(getattr(args, "persistence_warmup_epochs", 0))
    persist_classes: tuple[int, ...] = ()
    persist_gate = {"w": 0.0}   # epoch-annealed weight (set in the loop); 0 => branch inert
    if persist_w > 0.0:
        from tac.boundary_math.persistence_topology_loss import (
            detect_persistence_tail_classes,
            persistence_anneal_weight,
            persistence_topology_loss_mlx,
        )
        _pc = str(getattr(args, "persistence_classes", "auto")).strip()
        if _pc.lower() == "auto":
            _lst_stack_p = np.stack([np.asarray(gt.lstars[pi], np.int64) for pi in range(P)], axis=0)
            persist_classes, _pev = detect_persistence_tail_classes(_lst_stack_p, n_classes=5)
        else:
            persist_classes = tuple(int(x) for x in _pc.split(",") if x.strip() != "")
        print(json.dumps({"stage": "persistence_loss", "target_classes": list(persist_classes),
                          "weight": persist_w, "recall_weight": persist_recall_w,
                          "cldice_iters": persist_cldice_iters, "warmup_epochs": persist_warmup,
                          "note": "soft-clDice + persistence-weighted island recall on the SHARED "
                          "realized seg forward; annealed; advisory; pointer 0.19110 UNMOVED"}), flush=True)

    # (--cache-gt-skeleton, #260) declared here (in the enclosing scope, BEFORE total_loss_fn) so the
    # closure binds the cell; POPULATED after lstar_cache is built (see the build block below). None
    # (default OFF) => total_loss_fn passes sg_precomputed=None => byte-identical to the pre-flag path.
    cache_gt_skeleton = bool(getattr(args, "cache_gt_skeleton", False))
    _sg_cache: dict[int, Any] | None = None

    # #224 (5) ISLAND AMPLIFICATION setup (island_protection; #208/IslandProtectionGauge.AMPLIFY_ONLY).
    # Rides the SHARED LEVER-4 realized margin _signed (#141) -- NO 2nd saliency / SegNet forward.
    # amplify_w=0 (default) => island_weight_mx None => branch skipped => byte-identical.
    amplify_w = float(getattr(args, "amplify_weight", 0.0))
    amplify_form = str(getattr(args, "amplify_form", "hinge"))
    amplify_mtgt = float(getattr(args, "amplify_margin_target", 1.0))
    island_weight_mx: dict[int, Any] | None = None
    if amplify_w > 0.0:
        from tac.boundary_math.island_protection import (
            build_island_masks,
            eased_island_masks,
            identify_island_classes,
            island_birth_from_signed_mx,
            island_persistence_weight,
        )
        _eased_on = bool(getattr(args, "seed_island_eased", False))
        _mk_masks = eased_island_masks if _eased_on else build_island_masks
        _lst_stack_i = np.stack([np.asarray(gt.lstars[pi], np.int64) for pi in range(P)], axis=0)
        _idet = identify_island_classes(_lst_stack_i, n_classes=5)
        island_weight_mx = {}
        for pi in range(P):
            _im = _mk_masks(np.asarray(gt.lstars[pi], np.int64), _idet.lane_cls,
                            _idet.movable_cls, dilate_px=int(args.island_dilate_px))
            _iw = island_persistence_weight(_im.any_mask, kind=str(args.amplify_persist))
            island_weight_mx[pi] = mx.array(np.asarray(_iw, np.float32)[None])   # (1,H,W)
        print(json.dumps({"stage": "island_amplify", "island_classes": list(_idet.island_classes),
                          "lane_cls": _idet.lane_cls, "movable_cls": _idet.movable_cls,
                          "weight": amplify_w, "form": amplify_form, "margin_target": amplify_mtgt,
                          "persist": str(args.amplify_persist),
                          "note": "island-birth rides the SHARED realized _signed margin (#141); "
                          "advisory; pointer 0.19110 UNMOVED"}), flush=True)
    # #224 (5) island SEED + CONTAINMENT build (SEPARATE protected-seed module + its OWN AdamW group;
    # grad-shield applied to the seed leaf BETWEEN the dual value_and_grad and seed_opt.update — NEVER
    # touching the witness grouped-backward / MD-decoupling grads). The seed is a per-pair RGB residual
    # seeded at ep0 from the GT island appearance (build_island_seed), masked to the self-detected
    # lane+movable island band; composited into the SEGNET-scored frame1 BEFORE R (via _compose_chain
    # above) so it rides the SHARED realized _f1/_signed (no 2nd SegNet). Because it is a SEPARATE
    # module (NOT model.parameters()), it is absent from EMA/blob/deploy => the verdict is witness-alone
    # == the deploy render == the 0-byte training-time ACCELERANT semantics, HONESTLY measured (the
    # verdict d_seg IS the deploy-absorption readout; the containment keeps the seed alive during
    # training so the witness has a formed island to absorb). Default OFF => byte-identical.
    seed_mod = None
    seed_opt = None
    seed_spec = None
    _seed_shield = None
    if seed_on:
        if float(args.w_seg) <= 0.0:
            raise ValueError("--seed-islands requires --w-seg > 0: the seed helps ONLY through the "
                             "realized seg loss on the composed frame1; with w_seg=0 it is inert.")
        import mlx.nn as _seed_nn
        from tac.boundary_math.island_protection import (
            ContainmentSpec as _SeedSpec,
            build_island_masks as _build_isl_masks,
            build_island_seed as _build_isl_seed,
            contain_protected_grad_mx as _contain_grad_mx,
            eased_island_masks as _eased_isl_masks,
            identify_island_classes as _ident_isl,
        )
        _seed_shield = _contain_grad_mx
        _mk_seed_masks = _eased_isl_masks if bool(getattr(args, "seed_island_eased", False)) else _build_isl_masks
        _lst_stack_s = np.stack([np.asarray(gt.lstars[pi], np.int64) for pi in range(P)], axis=0)
        _sdet = _ident_isl(_lst_stack_s, n_classes=5)
        _seed_res_np = np.zeros((P, render_h, render_w, 3), np.float32)
        _seed_msk_np = np.zeros((P, render_h, render_w, 1), np.float32)
        _s_supp = []
        for pi in range(P):
            _im = _mk_seed_masks(np.asarray(gt.lstars[pi], np.int64), _sdet.lane_cls,
                                 _sdet.movable_cls, dilate_px=int(args.island_dilate_px))
            _gt1 = np.asarray(gt.gt_f1[pi], np.float32)
            if _gt1.shape[:2] != (render_h, render_w):
                import torch  # noqa: PLC0415
                import torch.nn.functional as _tF  # noqa: PLC0415
                _gt1 = _tF.interpolate(torch.from_numpy(_gt1).permute(2, 0, 1)[None],
                                       size=(render_h, render_w), mode="bilinear", align_corners=False
                                       )[0].permute(1, 2, 0).numpy()
            _seed = _build_isl_seed(_gt1, _im, base_render_segres=None, blend=float(args.seed_blend))
            _seed_res_np[pi] = _seed.residual
            _seed_msk_np[pi, ..., 0] = np.asarray(_im.any_mask, np.float32)
            _s_supp.append(float(_seed.support_frac))

        class _SeedMod(_seed_nn.Module):
            def __init__(self, res):
                super().__init__()
                self.residual = mx.array(res)

        seed_mod = _SeedMod(_seed_res_np)
        mx.eval(seed_mod.parameters())
        _seed_masks_mx = mx.array(_seed_msk_np)
        seed_state["mod"] = seed_mod
        seed_state["masks"] = [_seed_masks_mx[pi] for pi in range(P)]
        seed_spec = _SeedSpec(mode=str(args.containment_mode), damp=float(args.containment_damp),
                              protected_mask=None)
        seed_opt = optim.AdamW(learning_rate=float(args.seed_lr), weight_decay=0.0)
        print(json.dumps({"stage": "island_seed", "lane_cls": _sdet.lane_cls, "movable_cls": _sdet.movable_cls,
                          "island_classes": list(_sdet.island_classes),
                          "mean_support_frac": round(float(np.mean(_s_supp)), 5),
                          "containment_mode": str(args.containment_mode), "seed_lr": float(args.seed_lr),
                          "n_pairs": P,
                          "note": "SEPARATE protected seed module (own AdamW; NOT in EMA/blob/deploy = "
                          "0-byte accelerant; verdict=witness-alone=deploy=absorption readout); "
                          "shield-grad defends it; advisory; pointer 0.19110 UNMOVED"}), flush=True)

    def total_loss_fn(model, cf, c0, c1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, terms_out=None):
        # ``terms_out`` (#304 item 4 per-term loss telemetry; ADDITIVE, default None => BYTE-IDENTICAL):
        # when given a dict, every ADDITIVE contribution to L is recorded as a (lazy) mx array under
        # its LOSS_TERM_KEYS name. Recording only NAMES the same subexpressions L is built from --
        # the graph/loss/grads are bitwise identical (value_and_grad NEVER passes terms_out; only the
        # no-grad loss-term probe recompute in the epoch loop does).
        # (--seg-spike-reweight) per-pixel spike/coherent map for THIS pair (pi==int(c1)//2); None when
        # the lever is off => base_loss byte-identical. A stop-grad theta-independent constant multiplier.
        _seg_px_w = _spike_w_mx[int(c1) // 2] if _spike_reweight_on else None
        L = base_loss(model, cf, c0, c1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, mtgt, seg_form=seg_form, seg_pixel_w=_seg_px_w, terms_out=terms_out)
        phi0 = model.sdf(cf, c0)
        # (THETA* TIER-2 STRETCH-1) junction relax threaded; eik_jrelax=0.0 (default) => BIT-IDENTICAL.
        eik, length, _ = _eikonal_length_mlx(phi0, render_h, render_w,
                                             junction_relax=eik_jrelax, junction_tau=eik_jtau)
        # (EIK-STAB build 1b) ViscoReg vanishing-viscosity residual REPLACES the eikonal residual
        # while eps > 0 (same constraint, viscous form — the stable object; adding both would
        # double-count). eps==0.0 (default, or the anneal's end) => branch skipped => the legacy
        # residual above is used unchanged (byte-identical; the unused legacy `eik` subgraph is
        # never evaluated by MLX's lazy engine when replaced). The length term is unchanged.
        if _eik_stab["visco_eps"] > 0.0:
            eik = _eikonal_visco_mlx(phi0, render_h, render_w, _eik_stab["visco_eps"])
        L = L + eik_w * eik + len_w * length
        if terms_out is not None:
            terms_out["eikonal"] = eik_w * eik
            terms_out["length"] = len_w * length
        # (EIK-STAB build 1a) StEik directional-divergence damping (ADDITIVE; arXiv 2305.18414):
        # damps ONLY the normal-direction second-order mode (the proven anti-diffusive instability
        # of the eikonal flow), tangential curvature free. Default weight 0.0 => skipped => L is
        # byte-identical.
        if _eik_stab["steik_w"] > 0.0:
            # (V6 #317) NORMALIZED n^T H n (removes the |grad m|^2 self-amplification) when
            # --eikonal-steik-normalized; else the RAW form (byte-identical to the pre-V6 branch).
            if _eik_stab["steik_normalized"]:
                _steik_term = _eikonal_steik_normalized_mlx(
                    phi0, render_h, render_w, _eik_stab["steik_norm_eps"])
            else:
                _steik_term = _eikonal_steik_mlx(phi0, render_h, render_w)
            _steik_contrib = _eik_stab["steik_w"] * _steik_term
            L = L + _steik_contrib
            if terms_out is not None:
                terms_out["eik_steik"] = _steik_contrib
        # (--boundary-distance-weight) SDF-native boundary-placement term on FRAME1 (the SegNet-
        # scored frame): band-weighted mean |phi_GT - phi_runner| on the precomputed GT-boundary
        # band (distance-transform of the GT inter-class edges). One extra sdf trunk forward per
        # pair when ON (flag-gated; the eikonal phi0 above is frame0 — different code, not shared).
        # Default bd_w=0.0 => skipped => byte-identical.
        if bd_w > 0.0 and _bd_band_prov is not None:
            _bd_contrib = bd_w * boundary_distance_term_mlx(
                model.sdf(cf, c1), lstar_oh, _bd_band_prov[int(c1) // 2], render_h, render_w)
            L = L + _bd_contrib
            if terms_out is not None:
                terms_out["boundary_distance"] = _bd_contrib
        # (review R2b-M3) SHARED realized through-R seg forward. LEVER-3 (lane-edge), LEVER-4
        # (margin-saliency) and LEVER-B (thin-lane) all need the SAME realized decision margin
        # ``signed = gt_logit - top_competitor`` from the SAME render(cf,c1)->R->frozen SegNet. The
        # render is deterministic (uint8-STE round; no training noise), so computing it ONCE and
        # reusing it across the stacked levers is BIT-IDENTICAL to the prior 3-separate-forwards code
        # while doing 1 (not up to 3) of the expensive forward. Computed ONLY when >=1 seg-margin lever
        # is engaged; default-off (all weights 0) => _seg_levers_on False => block skipped =>
        # byte-identical to the additive default path. ``_f1`` is also reused for LEVER-4's UNIWARD
        # texture map (same rendered frame).
        # (review R2b-M3 + BUILD #300 a) SHARED realized through-R seg forward(s), split into the
        # SEED-COMPOSED forward the surgical levers read and the WITNESS-ALONE (seed-EXCLUDED) forward the
        # island-FORMATION levers read under --witness-alone-island-loss. Each forward is paid LAZILY,
        # only when a lever that needs it is engaged. Levers that read the SEED-COMPOSED margin/logits:
        _nonwa_levers_on = ((lane_w > 0.0 and lane_gate["on"]) or
                            (msal_w > 0.0 and msal_gate["on"]) or
                            (lane_thin_w > 0.0 and lane_thin_gate["on"]) or
                            (mfh_w > 0.0 and mfh_target_mx is not None) or          # #218 facets 1b/3
                            (subpix_w > 0.0 and subpix_gate["on"] and               # LEVER-4b sub-pixel t
                             _subpix_t_prov is not None) or
                            (chroma_bnd_w > 0.0 and chroma_bnd_gate["on"] and        # LEVER-4c chroma
                             _chroma_gt_prov is not None))
        # island-FORMATION levers (#224 amplify + persistence): read the WITNESS-ALONE margin when the
        # BUILD #300 (a) routing is active, else the seed-composed margin (== the pre-#300 path).
        _island_levers_on = ((amplify_w > 0.0 and island_weight_mx is not None) or  # #224 island amplify
                             (persist_gate["w"] > 0.0 and bool(persist_classes)))   # #224 persistence loss
        _wa_route = wa_island and (seed_state["mod"] is not None) and _island_levers_on
        # the seed-composed forward is needed by the non-wa levers, and by the island levers ONLY when
        # they are NOT wa-routed (wa off, or no seed). When wa-routed the composed forward would be
        # UNUSED -> skip it (saves 1 SegNet forward/step; the live config engages only island levers).
        # BYTE-IDENTITY (wa off): _wa_route False => _need_composed == the pre-#300 _seg_levers_on OR =>
        # _signed/_slog computed under the SAME condition, _signed_wa/_slog_wa ALIAS them (SAME objects).
        _need_composed = _nonwa_levers_on or (_island_levers_on and not _wa_route)
        _signed = None
        _slog = None
        if _need_composed:
            # _render_R composes the FIXED bulk before R in residual mode (else == bare render) so
            # the surgical levers (lane-thin/margin-saliency/lane-edge) weight the COMPOSED-render
            # d_seg -- the residual IS the Lane+Movable annulus, so they are maximally relevant.
            _f1 = _render_R(model, cf, c1, render_h, render_w)  # (1, SEG_H, SEG_W, 3) SEED-COMPOSED
            _slog = adapter.segnet(_f1)                                    # (1, H, W, 5)
            _sig_gt = mx.sum(_slog * lstar_oh, axis=-1)                    # (1, H, W) gt-class logit
            _sig_run = mx.max(_slog + lstar_oh * (-1e9), axis=-1)          # (1, H, W) top competitor
            _signed = _sig_gt - _sig_run                                   # (1, H, W) realized margin
        # BUILD #300 (a) WITNESS-ALONE island render/margin. The seed-composed _signed satisfies the seg
        # loss ON the island (the seed CARRIES it) so dL/d(witness) ~= 0 there and the witness never
        # learns to FORM Lane+Movable itself -> deploy (witness-alone) has ~0 island mass (MEASURED: 71%
        # of the plateau = the 2 seeded classes at 100% within-class flip; memo
        # plateau_disambiguator_results_20260704.md). Routing the island levers through the seed-EXCLUDED
        # (witness-alone == deploy) render restores the missing absorption gradient. Default (no routing)
        # => _signed_wa/_slog_wa ALIAS _signed/_slog (SAME objects) => BYTE-IDENTICAL, no 2nd forward.
        # When routed, the seed is ABSENT from _compose_chain_noseed => d(_signed_wa)/d(seed) == 0, so the
        # seed correctly gets NO gradient from these levers (it still trains via the base-CE composed path).
        _signed_wa = _signed
        _slog_wa = _slog
        if _wa_route:
            _f1_wa = _render_R_wa(model, cf, c1, render_h, render_w)   # seed EXCLUDED (witness-alone)
            _slog_wa = adapter.segnet(_f1_wa)                          # (1, H, W, 5)
            _sig_gt_wa = mx.sum(_slog_wa * lstar_oh, axis=-1)         # (1, H, W)
            _sig_run_wa = mx.max(_slog_wa + lstar_oh * (-1e9), axis=-1)
            _signed_wa = _sig_gt_wa - _sig_run_wa                     # (1, H, W) witness-alone margin
        # LEVER-3 (lane-edge fragility weighting, operator 2026-06-27 Yousfi-grounding): contest
        # SegNet argmax order is the comma10k CANONICAL order (MEASURED 2026-06-27 from the cached
        # argmax; CLAUDE.md NON-NEGOTIABLE): [Road0, Lane1, Undrivable2, Movable3, MyCar4]. The
        # FORBIDDEN luma-sort of class_values [41,76,90,124,161] -> [Road0,Lane1,MyCar2,Undriv3,Movable4]
        # is WRONG for 2/3/4 (bit us 3x); do NOT use it. Class0=Road & Class1=Lane are CONFIRMED in
        # BOTH orders (so this lever, which uses ONLY class 1, is correct regardless). Lane (class 1) is thin
        # all-boundary double-edges (19% of d_seg flips) and UNDER-FIT because the CE baseline has NO
        # class weighting. This ADDITIVE term up-weights the REALIZED (through-R SegNet) margin hinge
        # at GT-lane pixels: it renders f1 -> R -> frozen SegNet logits, takes the live decision
        # margin (gt_logit - top_competitor) ONLY where GT==lane, and penalizes relu(target-margin)
        # there. The hinge fires exactly on SMALL-MARGIN (fragile = boundary) lane pixels, so it
        # adds gradient pressure to widen the lane margin at the lane double-edges. Default-off
        # (lane_w=0). When ON it reuses the SHARED realized seg forward above (review R2b-M3: no
        # longer a separate render -- bit-identical, 1 forward shared across the stacked levers).
        if lane_w > 0.0 and lane_gate["on"]:
            lane_mask = lstar_oh[..., lane_cls]                         # (1, H, W) 1.0 where GT==lane
            hinge_map = mx.maximum(lane_tgt - _signed, 0.0) * lane_mask  # fragile lane pixels only
            lane_term = mx.sum(hinge_map) / (mx.sum(lane_mask) + 1e-6)  # mean hinge over lane px
            L = L + lane_w * lane_term
            if terms_out is not None:
                terms_out["lane_edge"] = lane_w * lane_term
        # LEVER-4 (margin-saliency, all-class generalization of LEVER-3). Same realized through-R
        # decision margin, but the hinge is weighted PER-PIXEL by the GT-margin fragility saliency
        # sal=exp(-gt_margin/tau) over EVERY GT pixel (not a single class mask). The flip-prone band
        # (small GT margin) lives across all classes (Road 47% / Lane 19% / Undriv 14% / ...), so this
        # adds widen-the-margin pressure exactly where d_seg lives. CLASS-AGNOSTIC. Default-off.
        if msal_w > 0.0 and msal_gate["on"]:
            sgn = _signed                                              # (1, H, W) SHARED realized margin (R2b-M3)
            sal = mx.exp(-margin / msal_tau)                            # (1, H, W) fragility weight
            if msal_reach and _sR_provider is not None:
                # REACHABILITY (REPLACES the inert texture proxy; MEASURED Pearson -0.033 for 1/(1+b*tex)):
                # multiply the fragility saliency by the cached THROUGH-R margin-Jacobian S_R for THIS pair
                # (stop-grad [0,1] weight; theta-independent reachability of the CORRECT answer). Product
                # concentrates capacity where the pixel is BOTH fragile (small GT margin) AND reachable
                # (high S_R) = the actionable margin-boundary band. pi==int(c1)//2 (same key as
                # island_weight_mx). The cache is precomputed => this is CHEAPER than the per-step tex recompute.
                sal = sal * _sR_provider[int(c1) // 2]                   # (1, H, W) reachability-weighted
            elif msal_uni:
                # UNIWARD: down-weight textured regions (SegNet-undetectable) -> concentrate on the
                # SMOOTH boundary. Texture energy from the realized frame's spatial gradients, used as
                # a STOP-GRAD weight (a cost map, not a loss path). Reuses the SHARED rendered frame _f1.
                lum = mx.mean(mx.stop_gradient(_f1), axis=-1)            # (1, H, W)
                dy = mx.pad(mx.abs(lum[:, 1:, :] - lum[:, :-1, :]), [(0, 0), (0, 1), (0, 0)])
                dx = mx.pad(mx.abs(lum[:, :, 1:] - lum[:, :, :-1]), [(0, 0), (0, 0), (0, 1)])
                tex = dy + dx
                tex = tex / (mx.max(tex) + 1e-6)                         # [0,1]
                sal = sal / (1.0 + msal_uni_beta * tex)
            hmap = mx.maximum(msal_tgt - sgn, 0.0) * sal                 # fragile pixels weighted
            msal_term = mx.sum(hmap) / (mx.sum(sal) + 1e-6)             # saliency-weighted mean hinge
            L = L + msal_w * msal_term
            if terms_out is not None:
                terms_out["margin_saliency"] = msal_w * msal_term
        # LEVER-4b (SUB-PIXEL BOUNDARY-PLACEMENT `t`; DIRECTIONAL upgrade of LEVER-4, GREEN 2026-07-03).
        # Rides the SAME SHARED realized through-R margin `_signed` (Mw = relu(_signed) = the witness
        # GT-class margin at every pixel, the honest mirror of the GT top1-top2 the target `t` is built
        # from) -- NO 2nd SegNet forward. At each pre-selected genuine-V straddle pixel p (dominant
        # inter-class edge; both GT margins in the flip band) the GT boundary sits at fractional position
        # t = M_GT[p]/(M_GT[p]+M_GT[q]) between p and its cross-edge partner q. We supervise the witness's
        # OWN realized margin ratio t_wit = Mw[p]/(Mw[p]+Mw[q]) toward that GT t: a DENSER, sub-pixel,
        # differentiable placement signal that pulls both Mw[p] and Mw[q] to seat the witness boundary at
        # the correct sub-pixel spot. q is the precomputed dominant direction (0=right (i,j+1), 1=down
        # (i+1,j)); Mw[q] is a pure shift of the SHARED Mw (fully vectorized, both differentiable). Masked
        # to the precomputed active straddle set (sentinel t<0 => weight 0). Default subpix_w=0 => skipped
        # => byte-identical. MODEST 2nd-order refinement (weakest on thin lanes; effect in the 1-2px flip
        # band, #149) -> an A/B arm, NOT a claim. pointer 0.19110 UNMOVED.
        if subpix_w > 0.0 and subpix_gate["on"] and _subpix_t_prov is not None:
            _pi_sp = int(c1) // 2
            _t_tgt = _subpix_t_prov[_pi_sp]                              # (1,H,W) f32, -1 sentinel
            _dir_m = _subpix_dir_prov[_pi_sp]                            # (1,H,W) f32 in {0,1}
            _active = (_t_tgt >= 0.0).astype(_signed.dtype)             # (1,H,W) genuine-V straddle mask
            _Mw = mx.maximum(_signed, 0.0)                              # (1,H,W) witness GT-class margin
            # partner margin via a pure shift of the SHARED Mw (edge-col/row pad is inert: those pixels
            # can never be active in the corresponding direction, so the pad value is masked out).
            _M_right = mx.pad(_Mw[:, :, 1:], [(0, 0), (0, 0), (0, 1)])   # _M_right[i,j] = Mw[i,j+1]
            _M_down = mx.pad(_Mw[:, 1:, :], [(0, 0), (0, 1), (0, 0)])    # _M_down[i,j]  = Mw[i+1,j]
            _Mq = mx.where(_dir_m < 0.5, _M_right, _M_down)             # dominant cross-edge partner
            _t_wit = _Mw / (_Mw + _Mq + subpix_eps)                     # witness sub-pixel boundary ratio
            _t_ref = mx.maximum(_t_tgt, 0.0)                            # sentinel -1 -> 0 (masked anyway)
            _sq = mx.square(_t_wit - _t_ref) * _active                  # placement error on genuine-V px
            subpix_term = mx.sum(_sq) / (mx.sum(_active) + 1e-6)        # mean over active straddles
            L = L + subpix_w * subpix_term
            if terms_out is not None:
                terms_out["subpix"] = subpix_w * subpix_term
        # LEVER-4c (ANNULUS-DIRECTED CHROMA-SHARPENING; chroma DOF probe a3e9f0bd GREEN 2026-07-03).
        # RIDES the SHARED rendered frame ``_f1`` (through R; the SAME render the SegNet forward /
        # ``_signed`` come from) -- NO 2nd render, NO 2nd SegNet forward. At the fragile annulus
        # (precomputed GT margin < band, where MEASURED 93.4% of chroma-flips live) supervise the
        # witness's OWN rendered chroma toward the GT chroma. Chroma := rgb - BT.601-luma (the SAME
        # BT.601 the witness _apply_chroma uses) -> LUMA-INVARIANT (adding a constant luma to all 3
        # channels leaves chroma unchanged) => ORTHOGONAL to every luma lever; this is a per-pixel
        # chroma-MATCH at the boundary, NOT a full-RGB reconstruction. The GT chroma target + annulus
        # weight are precomputed theta-independent constants (stop-grad by construction); the witness
        # chroma is the differentiable loss path that pulls the per-pixel RGB head to paint the boundary
        # chroma the near-per-class-constant palette can't. Default chroma_bnd_w=0 => skipped =>
        # byte-identical. BOUNDARY SHARPENER (weakest in bulk; power at the knife-edge flips) -> an A/B
        # arm, NOT a claim. pointer 0.19110 UNMOVED.
        if chroma_bnd_w > 0.0 and chroma_bnd_gate["on"] and _chroma_gt_prov is not None:
            _pi_ch = int(c1) // 2
            _cgt = _chroma_gt_prov[_pi_ch]                              # (1,H,W,3) GT chroma const
            _cw = _chroma_w_prov[_pi_ch]                                # (1,H,W) annulus weight const
            # witness BT.601 luma (differentiable) -> chroma = rgb - luma (broadcast over 3 channels).
            _lum_w = 0.299 * _f1[..., 0:1] + 0.587 * _f1[..., 1:2] + 0.114 * _f1[..., 2:3]  # (1,H,W,1)
            _cwit = _f1 - _lum_w                                        # (1,H,W,3) witness chroma
            _cdiff2 = mx.sum(mx.square(_cwit - _cgt), axis=-1)          # (1,H,W) 3-chan sq chroma error
            chroma_bnd_term = mx.sum(_cdiff2 * _cw) / (mx.sum(_cw) + 1e-6)  # mean over annulus px
            L = L + chroma_bnd_w * chroma_bnd_term
            if terms_out is not None:
                terms_out["chroma_boundary"] = chroma_bnd_w * chroma_bnd_term
        # CONSUMER B (SPEC ONLY, NOT built here -- for the lane-band render integration): the SAME
        # precomputed theta-independent (_subpix_t_prov, _subpix_dir_prov) maps are a decode-time
        # RENDER-PLACEMENT target. The AA-SDF / analytic-lane-band render (--lane-render-band /
        # tac.boundary_math.{aa_sdf_observation_render,analytic_lane_render_band}) can place the band
        # boundary at the sub-pixel position `t` between p and its dominant partner q (dir), instead of
        # the nearest grid edge, falling back to grid placement where the lane V is NOT genuine
        # (t sentinel <0). That render path is owned by the lane-band lever, not this loss term -- spec
        # noted here so the maps are reused, never re-derived (triality: DSL VectorFieldMarginSaliency).
        # #224 (5) ISLAND AMPLIFICATION — the island-birth term on the SHARED realized _signed margin
        # (island x persistence weight; orthogonal to LEVER-4's fragility x all-class weight). Default
        # amplify_w=0 => skipped => byte-identical. c1 = 2*pi+1 (the SegNet-scored frame) => pi=c1//2.
        if amplify_w > 0.0 and island_weight_mx is not None:
            # BUILD #300 (a): the island-BIRTH term reads the WITNESS-ALONE margin (_signed_wa aliases
            # _signed when --witness-alone-island-loss is off => byte-identical) so it pushes the witness
            # to form the island, not the deploy-excluded seed.
            _amp_contrib = amplify_w * island_birth_from_signed_mx(
                _signed_wa, island_weight_mx[int(c1) // 2], amplify_mtgt, form=amplify_form)
            L = L + _amp_contrib
            if terms_out is not None:
                terms_out["island_amplify"] = _amp_contrib
        # #224 (4) PERSISTENCE/TOPOLOGY loss — soft-clDice + persistence-weighted island recall on the
        # SHARED realized seg logits (_slog). GT-presence-gated inside the module (never hallucinate).
        # Annealed weight persist_gate["w"] (set per-epoch, coarse->fine); 0 => branch inert.
        if persist_gate["w"] > 0.0 and persist_classes:
            # (--cache-gt-skeleton #260) reuse the precomputed CONSTANT GT skeleton for THIS pair
            # (pi == c0//2, the SAME key thin_maps_mx/island_weight_mx use). None => inline recompute
            # (byte-identical default); a cache MISS also falls back to None (still bit-identical).
            _sg_pre = _sg_cache.get(int(c0) // 2) if _sg_cache is not None else None
            # BUILD #300 (a): island RECALL/topology reads the WITNESS-ALONE seg logits (_slog_wa aliases
            # _slog when --witness-alone-island-loss is off => byte-identical) so the recall term drives
            # the witness to reproduce the island skeleton itself, not free-ride the deploy-excluded seed.
            _per_contrib = persist_gate["w"] * persistence_topology_loss_mlx(
                _slog_wa, lstar_oh, persist_classes, cldice_iters=persist_cldice_iters,
                w_cldice=1.0, w_recall=persist_recall_w, sg_precomputed=_sg_pre)
            L = L + _per_contrib
            if terms_out is not None:
                terms_out["persistence"] = _per_contrib
        # LEVER-A (FiLM-rank-fix) soft participation-ratio FLOOR. Pushes the per-pair modulation PR up
        # toward rankfloor_tgt (opposing the measured rank-1 collapse). PR computed Gram-wise (NO
        # eigendecomposition): trace(C)=||Mc||_F^2 (== mx.sum(Mc*Mc)), ||C||_F^2=||Mc Mc^T||_F^2. The
        # numpy reference is tac...film_modulation_participation_ratio / film_rank_floor_penalty.
        # Default-off (rankfloor_w=0). Mirrors the numpy reference EXACTLY (one math, two backends).
        if rankfloor_w > 0.0 and rankfloor_idx is not None:
            M = model.film(model.code[rankfloor_idx])                   # (S, D) modulation
            Mc = M - mx.mean(M, axis=0, keepdims=True)
            tr = mx.sum(Mc * Mc)                                        # trace(Gram) = sum eigenvalues
            G = Mc @ Mc.T                                               # (S, S) Gram
            fro2 = mx.sum(G * G)                                        # sum eigenvalues^2
            pr = (tr * tr) / (fro2 + 1e-12)                            # participation ratio in [1, S]
            _rf_contrib = rankfloor_w * mx.maximum(rankfloor_tgt - pr, 0.0)
            L = L + _rf_contrib
            if terms_out is not None:
                terms_out["rankfloor"] = _rf_contrib
        # DM1b (code spectral-entropy CAPACITY penalty): -beta*log(PR(cov(code))) on the per-pair code
        # covariance C = cov(code). Maximizes PR(cov(code)) => keeps all ~mod_dim code directions live;
        # via the Stiefel identity (--film-stiefel) WᵀW=I => PR(M)=PR(cov(code)) this is the other half
        # of the byte-free DM1 cure. PR via the (D,D) covariance Gram (no eigendecomposition): C=Cc^T Cc
        # (the 1/(S-1) cancels in the ratio). Default-off (code_spec_w=0). EXACT MLX twin of the numpy
        # tac...code_spectral_entropy_penalty (one math, two backends). The gradient flows to the
        # `code` latent (spreading its spectrum); film.weight is handled by the Stiefel projection, so
        # the two halves target DIFFERENT params (no double-count, design memo §3 routing).
        if code_spec_w > 0.0 and code_spec_idx is not None:
            Cm = model.code[code_spec_idx]                              # (S, D) per-pair codes
            Cc = Cm - mx.mean(Cm, axis=0, keepdims=True)
            Cov = Cc.T @ Cc                                            # (D, D) ~ cov(code)
            ctr = mx.sum(Cc * Cc)                                      # trace(Cov) = sum eigenvalues
            cfro2 = mx.sum(Cov * Cov)                                  # sum eigenvalues^2
            cpr = (ctr * ctr) / (cfro2 + 1e-12)                        # PR(cov(code)) in [1, D]
            _cs_contrib = -(code_spec_w * mx.log(cpr + 1e-12))         # -beta*log(PR) => raises PR
            L = L + _cs_contrib
            if terms_out is not None:
                terms_out["code_spectral"] = _cs_contrib
        # LEVER-B (thin-lane dropped-dash prior): realized through-R margin hinge weighted by the
        # PRECOMPUTED thin-lane map (nonzero ONLY on thin GT-lane pixels). Same realized decision
        # margin as LEVER-3 but concentrated on the DROPPED thin dashes (the PC0 residual). c0=2*pi
        # so c0//2 == pi keys the per-pair thin map to THIS pair's lstar_oh. Default-off (lane_thin_w
        # =0). Reuses the SHARED realized seg forward above (review R2b-M3: no separate render --
        # bit-identical, 1 forward shared across the stacked levers).
        if lane_thin_w > 0.0 and lane_thin_gate["on"] and thin_maps_mx is not None:
            tw = thin_maps_mx[int(c0) // 2]                            # (1, H, W) thin-lane weight (>=0)
            hmap_t = mx.maximum(lane_thin_tgt - _signed, 0.0) * tw     # fragile thin-lane pixels only
            _lt_contrib = lane_thin_w * (mx.sum(hmap_t) / (mx.sum(tw) + 1e-6))
            L = L + _lt_contrib
            if terms_out is not None:
                terms_out["thin_lane"] = _lt_contrib
        # #218 facets 1b/3 (MARGIN-FIELD HEAD, byte-free): realized through-R PER-CLASS margin hinge.
        # per-pixel target = additive-margin (facet-1b) + per-class Menon boost on rare classes
        # (facet-3), broadcast to each pixel by its GT class via lstar_oh. Reuses the SHARED _signed
        # (R2b-M3). Default-off (mfh_w=0) => byte-identical. This widens the realized SegNet decision
        # margin MORE for the erasure-prone rare classes (Lane<->Road 57% tail, #209).
        if mfh_w > 0.0 and mfh_target_mx is not None:
            per_pix_tgt = mx.sum(lstar_oh * mfh_target_mx, axis=-1)     # (1,H,W) per-class margin target
            hmap_m = mx.maximum(per_pix_tgt - _signed, 0.0)            # fragile pixels below their target
            _mfh_contrib = mfh_w * mx.mean(hmap_m)
            L = L + _mfh_contrib
            if terms_out is not None:
                terms_out["margin_field_head"] = _mfh_contrib
        # (THETA* TIER-2 MUST-2) nuclear-norm low-rank code penalty. DEFAULT-OFF: code_nuc_w=0.0 =>
        # this branch NEVER runs => L is byte-identical (fully additive). When >0 it adds
        # weight * smoothed_nuclear_norm(model.code) -> drives the per-pair FiLM codes
        # (num_pairs*2 x mod_dim) toward a low-rank subspace (rate). The code matrix is identical for
        # every pair, so the per-pair value_and_grad sees the same term and the mean-over-chunk grad
        # applies it ONCE per opt step (NOT P-scaled). Recomputed per value_and_grad call (a parent
        # fusion could hoist it once-per-step; out of scope for this additive prep).
        if code_nuc_w > 0.0:
            _cn_contrib = code_nuc_w * _nuclear_norm_smooth_mlx(
                model.code, rel_eps=code_nuc_eps, ns_iters=code_nuc_iters)
            L = L + _cn_contrib
            if terms_out is not None:
                terms_out["code_nuclear"] = _cn_contrib
        return L

    value_and_grad = nn.value_and_grad(model, total_loss_fn)

    # #224 (5) DUAL value_and_grad for the island SEED (its OWN param tree + optimizer). Co-differentiate
    # the witness (model) AND the seed (seed_mod) w.r.t. the SAME loss (the seed enters via _compose_chain
    # -> _f1 -> seg_l). Default OFF (seed_mod None) => _dual_vg None => the loop takes the single
    # value_and_grad path (BYTE-IDENTICAL). The witness grad tree (grads[0]) is IDENTICAL to the single
    # path (same loss, same model params); only the extra grads[1] (seed) is new -> the shield acts on
    # grads[1] ONLY, then seed_opt (a DISTINCT AdamW) applies it -> the witness opt.update + MD-decoupling
    # + grouped-backward path is UNTOUCHED.
    _dual_vg = None
    if seed_mod is not None:
        def _combined_seed_loss(model_p, seed_p, cf, c0, c1, oh, mg, ptg, ws, wp, hg, mt, sf, ew, lw):
            model.update(model_p)
            seed_mod.update(seed_p)
            return total_loss_fn(model, cf, c0, c1, oh, mg, ptg, ws, wp, hg, mt, sf, ew, lw)

        _dual_vg = mx.value_and_grad(_combined_seed_loss, argnums=(0, 1))

    # ===================================================================================
    # (--micro-batch-pairs, DAG FEED 2026-07-03c) BATCHED twin of ``total_loss_fn``. OPT-IN
    # (--micro-batch-pairs > 1); the DEFAULT B=1 path NEVER calls this (the accum loop keeps its
    # UNCHANGED serial per-pair value_and_grad). The ONLY batched operations are the EXPENSIVE realized
    # render + FROZEN-SCORER forwards (one segnet over the B f1 frames, one posenet over the B pairs) —
    # the measured bottleneck (single-pair EfficientNet-B2 under-utilizes the GPU). EVERY per-pair loss
    # reduction (base seg-form; the score-domain pose ``sqrt(10*d_pose)`` which is NONLINEAR so
    # sqrt(mean)!=mean(sqrt); and every weighted-mean lever ``sum(x*w)/sum(w)``) is computed PER PAIR on
    # the batched scorer outputs and MEAN-ed over B, so
    #     total_loss_fn_batch(B pairs) == mean_b total_loss_fn(pair_b)
    # WITHIN fp tolerance (batched conv/BN is batch-independent in SegNet/PoseNet eval mode -> per-frame
    # logits are unchanged by batching; the mean-over-B is the only reduction re-order). That EXACT
    # per-pair-mean identity makes the accum-loop grad match the serial mean-over-chunk EXACTLY (the
    # accum loop weights each group's mean-grad by its pair count) and lets the numerical-equivalence
    # test pin batched-grad == mean-of-per-pair-grad. The realized segnet(f1) forward is computed ONCE
    # and SHARED by the base seg-form AND the lever ``_signed`` — bit-identical to total_loss_fn's two
    # deterministic-render forwards ((f'+g')·dS == f'·dS + g'·dS). The once-per-step per-MODEL code
    # penalties (rankfloor / code-spec / code-nuc) are added ONCE (matching the serial mean-over-chunk
    # of an identical-per-pair term). NOT bit-identical to the serial path (batched fp reduction order):
    # a trajectory-affecting opt-in validated by a short A/B. Mirrors total_loss_fn op-for-op.
    # ===================================================================================
    # (--micro-batch-pairs) BATCHED twin of total_loss_fn -> delegates to the importable + unit-tested
    # tac.boundary_math.levelset_micro_batch_loss (the nested closure cannot be reached from a test).
    # The LeverConfig SNAPSHOTS the ~30 lever closures; the gate dicts (lane_gate / msal_gate /
    # lane_thin_gate / persist_gate) + the lever tensor dicts (island_weight_mx / thin_maps_mx) are
    # passed BY REFERENCE and are MUTATED-IN-PLACE by the epoch loop, so the per-epoch gate/anneal
    # changes are seen live -- exactly like total_loss_fn re-reads them each value_and_grad call.
    from tac.boundary_math.levelset_micro_batch_loss import (
        LeverConfig as _MicroBatchLeverConfig,
        batched_realized_loss as _micro_batched_realized_loss,
    )

    _micro_batch_lc = _MicroBatchLeverConfig(
        seg_loss_default=args.seg_loss, tau_use=float(args.tau_softplus_tau),
        l7_thr_use=float(args.l7_threshold), l7_mult=float(args.l7_mult),
        score_domain=bool(args.score_domain_loss), pose_eps=float(args.pose_eps),
        eik_jrelax=eik_jrelax, eik_jtau=eik_jtau,
        eikonal_length=_eikonal_length_mlx, nuclear_norm_smooth=_nuclear_norm_smooth_mlx,
        lane_w=lane_w, lane_gate=lane_gate, lane_cls=lane_cls, lane_tgt=lane_tgt,
        msal_w=msal_w, msal_gate=msal_gate, msal_tau=msal_tau, msal_tgt=msal_tgt,
        msal_uni=msal_uni, msal_uni_beta=msal_uni_beta,
        amplify_w=amplify_w, island_weight_mx=island_weight_mx, amplify_mtgt=amplify_mtgt,
        amplify_form=amplify_form,
        persist_gate=persist_gate, persist_classes=persist_classes,
        persist_cldice_iters=persist_cldice_iters, persist_recall_w=persist_recall_w,
        lane_thin_w=lane_thin_w, lane_thin_gate=lane_thin_gate, thin_maps_mx=thin_maps_mx,
        lane_thin_tgt=lane_thin_tgt,
        mfh_w=mfh_w, mfh_target_mx=mfh_target_mx,
        rankfloor_w=rankfloor_w, rankfloor_idx=rankfloor_idx, rankfloor_tgt=rankfloor_tgt,
        code_spec_w=code_spec_w, code_spec_idx=code_spec_idx,
        code_nuc_w=code_nuc_w, code_nuc_eps=code_nuc_eps, code_nuc_iters=code_nuc_iters,
        # (MB-TWIN #313) newly-ROUTED legs (were fail-closed): focal / boundary-distance /
        # eik-stab (ViscoReg+StEik) / witness-alone-island routing. Callables passed to stay
        # bit-identical to the canonical helpers (avoids a tac<-trainer import cycle). _eik_stab
        # is passed BY REFERENCE so the batched twin reads the per-epoch viscosity anneal live.
        focal_gamma=focal_gamma, focal_pixel_weight=focal_pixel_weight_mlx,
        bd_w=bd_w, bd_band_prov=_bd_band_prov, boundary_distance_term=boundary_distance_term_mlx,
        eik_stab=_eik_stab, eikonal_visco=_eikonal_visco_mlx, eikonal_steik=_eikonal_steik_mlx,
        wa_island=wa_island,
    )

    def total_loss_fn_batch(model, cf_list, c0_list, c1_list, oh_list, mg_list, pose_tgt_list,
                            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w):
        # THIN wrapper: the batched loss body + equivalence contract live in the importable module.
        # render_fn (witness / residual-compose / AA / pose-carrier) or the bare R render, exactly as
        # base_loss picks it in total_loss_fn.
        _render = render_fn if render_fn is not None else render_through_R_mlx
        # (MB-TWIN #313) --witness-alone-island-loss: pass the SEED-EXCLUDED render so the batched twin
        # routes the island levers (amplify/persistence) through the seed-excluded margin, exactly as
        # total_loss_fn's _wa_route does. The twin no-ops it (byte-identical) unless wa_island AND an
        # island lever is engaged. _render_R_wa == _render_R when wa routing is off.
        _render_wa = _render_R_wa if wa_island else None
        return _micro_batched_realized_loss(
            model, adapter, _render, render_h, render_w,
            cf_list, c0_list, c1_list, oh_list, mg_list, pose_tgt_list,
            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, _micro_batch_lc,
            render_fn_wa=_render_wa)

    # OPT-IN batched accum path. Build the batched value_and_grad ONLY when engaged (>1).
    # (BUILD #293) BATCHED SEED CO-GRAD: --micro-batch-pairs > 1 now COMPOSES with --seed-islands via
    # a DUAL value_and_grad over BOTH param trees (model + seed_mod) of the SAME total_loss_fn_batch
    # the single-tree batched path differentiates. The seed enters the batched forward exactly as in
    # the serial path: through render_fn -> _compose_chain (frame1-odd residual*mask compose, PRE-R),
    # which batched_realized_loss invokes PER pair before stacking -> the batched loss already
    # contains the seed term; the dual grad just adds the second argnum. Equivalence (the NO-FAKE
    # gate, executed by experiments/test_batched_seed_cograd.py on real gt_n6 + the real frozen MLX
    # scorer): dual-batched(B) loss == mean_b dual-serial(pair_b) loss AND both grad legs match the
    # mean-of-per-pair-grads within fp32 tolerance (MLX CPU: matmul/reductions batch-independent).
    # DEFAULT (1) => _use_micro_batch False => value_and_grad_batch/_dual_vg_batch None => the accum
    # loop takes the UNCHANGED serial path (BYTE-IDENTICAL; the serial _dual_vg dispatch is untouched).
    _micro_batch_pairs = int(getattr(args, "micro_batch_pairs", 1))
    _use_micro_batch = _micro_batch_pairs > 1
    # (MB-TWIN #313) The witness-alone island routing (#300a), --seg-focal-gamma,
    # --boundary-distance-weight, and the EIK-STAB stabilizers (--eikonal-viscosity /
    # --eikonal-steik-weight) are now ROUTED into the batched twin
    # (tac.boundary_math.levelset_micro_batch_loss): the LeverConfig carries focal / bd /
    # eik_stab (by-ref) / wa_island, total_loss_fn_batch passes _render_R_wa, and the batched
    # amplify/persistence read the seed-excluded margin exactly as the serial _wa_route does. The
    # equivalence (batched grad == mean-of-per-pair grad within fp tol) is pinned per-leg by
    # src/tac/tests/test_levelset_micro_batch_loss.py. The prior four NotImplementedError
    # fail-closes are removed. STILL fail-closed below (not yet routed): --margin-saliency-reachability
    # and --seg-spike-reweight. --seed-anneal-epochs composes via _compose_chain (batched render).
    value_and_grad_batch = None
    _dual_vg_batch = None
    if _use_micro_batch:
        if seed_mod is not None:
            # (BUILD #293) dual co-grad twin of the serial _dual_vg: same in-place model/seed update
            # trick, batched arg lists. argnums=(0,1) -> (witness grads, seed grads); the loop weights
            # each group's MEAN grads (both legs) by its pair count so accum/accum_seed keep the
            # serial mean-over-chunk invariant and everything downstream (mean_grads, clip, opt.update,
            # _seed_shield, seed_opt.update) is UNTOUCHED.
            def _combined_seed_loss_batch(model_p, seed_p, cf_l, c0_l, c1_l, oh_l, mg_l, ptg_l,
                                          ws, wp, hg, mt, sf, ew, lw):
                model.update(model_p)
                seed_mod.update(seed_p)
                return total_loss_fn_batch(model, cf_l, c0_l, c1_l, oh_l, mg_l, ptg_l,
                                           ws, wp, hg, mt, sf, ew, lw)

            _dual_vg_batch = mx.value_and_grad(_combined_seed_loss_batch, argnums=(0, 1))
        else:
            value_and_grad_batch = nn.value_and_grad(model, total_loss_fn_batch)
        print(json.dumps({"stage": "micro_batch_pairs", "B": int(_micro_batch_pairs),
                          "accum_pairs": int(args.accum_pairs),
                          "seed_cograd": bool(seed_mod is not None),
                          "note": "OPT-IN batched scorer forward (B pairs/forward); trajectory-affecting "
                          "(batched fp reduction) but grad == serial mean-over-chunk within fp tol; "
                          "seed_cograd=dual value_and_grad (BUILD #293) when --seed-islands; "
                          "advisory; pointer 0.19110 UNMOVED"}), flush=True)

    # one-hot L* + margin per pair at the SegNet OUTPUT res (gt.lstars/gt.margins are 384x512,
    # matching the realized seg_logits = adapter.segnet(R(rgb))). NOT render res.
    def _lstar_oh(pi: int):
        lr = np.asarray(gt.lstars[pi])  # (384,512)
        oh = np.eye(5, dtype=np.float32)[lr.ravel()].reshape(lr.shape[0], lr.shape[1], 5)
        mg = np.asarray(gt.margins[pi], np.float32)  # (384,512)
        return mx.array(oh[None]), mx.array(mg[None])

    pose_tgts = [mx.array(np.asarray(gt.gt_poses[pi], np.float32)) for pi in range(P)]
    lstar_cache = [_lstar_oh(pi) for pi in range(P)]

    # LEVER-4 REACHABILITY (default-off): POPULATE the per-pair through-R S_R provider ONLY when
    # --margin-saliency-reachability AND msal_w>0. Loads the precomputed 'sR' (P,H,W) [0,1] weight from
    # --gt-cache (lazy npz: inflates ONLY the sR member) -> a list[mx.array (1,H,W)] indexed by pi (==
    # int(c1)//2, the SAME key island_weight_mx / _sR_provider[...] use). Fails CLOSED (never silently)
    # if the flag is set but the cache lacks 'sR' (run tools/precompute_sR_reachability.py) or if
    # micro-batch is on (the batched twin's LEVER-4 does not yet consume S_R). When OFF, _sR_provider
    # stays None (declared above) and the LEVER-4 branch never references it => byte-identical resume.
    if msal_reach and msal_w > 0.0:
        if _use_micro_batch:
            raise ValueError(
                "--margin-saliency-reachability is not supported with --micro-batch-pairs>1 (the batched "
                "LEVER-4 twin does not consume S_R yet); run the reachability arm at --micro-batch-pairs 1.")
        if not args.gt_cache:
            raise ValueError(
                "--margin-saliency-reachability requires --gt-cache (the 'sR' reachability map is cached "
                "there); build it with tools/precompute_sR_reachability.py --gt-cache <path>.")
        _zc = np.load(Path(args.gt_cache), allow_pickle=False)
        _sR_src = str(args.gt_cache)
        if "sR" not in _zc.files:
            # SIDECAR FALLBACK (#268): tools/precompute_sR_reachability.py --mode sidecar writes
            # '<stem>_sR.npz' so a LIVE main cache (e.g. gt_n600 under a running arm) NEVER needs an
            # inplace rewrite. Precedence: main-cache 'sR' > sidecar > fail closed. Inside the
            # msal_reach gate => the default/OFF path is untouched (byte-identity preserved).
            _sidecar = Path(args.gt_cache).with_name(Path(args.gt_cache).stem + "_sR.npz")
            if _sidecar.exists():
                _zc = np.load(_sidecar, allow_pickle=False)
                _sR_src = str(_sidecar)
        if "sR" not in _zc.files:
            raise ValueError(
                f"--gt-cache {args.gt_cache} has no 'sR' key (and no '<stem>_sR.npz' sidecar); "
                f"build it first: tools/precompute_sR_reachability.py --gt-cache {args.gt_cache} "
                f"--num-pairs {P} [--mode sidecar].")
        _sR_all = _zc["sR"]  # (cached_P, H, W) float32 in [0,1]; inflate the sR member ONCE
        if int(_sR_all.shape[0]) < P:
            raise ValueError(
                f"--gt-cache {args.gt_cache} 'sR' has {int(_sR_all.shape[0])} pairs < --num-pairs {P}; "
                "re-run tools/precompute_sR_reachability.py at >= the requested size.")
        _sR_provider = [mx.array(np.asarray(_sR_all[pi], np.float32)[None]) for pi in range(P)]
        print(json.dumps({"stage": "margin_saliency_reachability", "active": True, "n_pairs": int(P),
                          "gt_cache": str(args.gt_cache), "sR_source": _sR_src,
                          "sR_norm_mean": round(float(np.asarray(_sR_all[:P]).mean()), 5),
                          "note": "LEVER-4 saliency weighted by cached through-R margin-Jacobian S_R "
                          "(REPLACES the measured-inert 1/(1+beta*tex) texture path); advisory build, "
                          "A/B owed (needs GO); pointer 0.19110 UNMOVED"}), flush=True)

    # (--seg-spike-reweight, source-split MEASURED 2026-07-03) precompute the theta-INDEPENDENT per-pair
    # spike/coherent weight map from the GT argmax TEMPORAL neighbors (list[mx.array (1,H,W)] indexed by
    # pi == int(c1)//2, the SAME key _sR_provider/island_weight_mx use). A pixel is a SPIKE at pair pi if
    # lstar[pi] differs from BOTH neighbors lstar[pi-1] & lstar[pi+1] (single-frame argmax FLICKER a
    # per-frame witness structurally cannot fit -- MEASURED n600 ~88.6% IRREDUCIBLE appearance change, so
    # smooth-is-optimal there); COHERENT = temporally-UNSTABLE but matches >=1 neighbor (the winnable
    # boundary residual). Map = downweight@spike, upweight@coherent, 1.0 else. Endpoints (pi in {0,P-1},
    # only one neighbor) => all-1.0. Default OFF (_spike_w_mx None) OR both scalars==1.0 (map==1.0) =>
    # base_loss gets seg_pixel_w=None/ones => BYTE-IDENTICAL. Fails CLOSED with micro-batch (serial path
    # only; the batched twin does not consume seg_pixel_w yet). A/B owed (needs GO); pointer 0.19110 UNMOVED.
    _spike_reweight_on = bool(getattr(args, "seg_spike_reweight", False))
    _spike_w_mx = None
    if _spike_reweight_on:
        if _use_micro_batch:
            raise ValueError(
                "--seg-spike-reweight is not supported with --micro-batch-pairs>1 (the batched twin does "
                "not consume the per-pixel seg reweight yet); run this arm at --micro-batch-pairs 1.")
        _sp_dn = float(getattr(args, "seg_spike_downweight", 1.0))
        _sp_up = float(getattr(args, "seg_coherent_upweight", 1.0))
        _sp_H, _sp_W = np.asarray(gt.lstars[0]).shape
        _sp_stack = np.stack([np.asarray(gt.lstars[pi], np.int64) for pi in range(P)])  # (P,H,W)
        _spike_w_mx = []
        _sp_n_spike = 0
        _sp_n_coh = 0
        for pi in range(P):
            wmap = np.ones((_sp_H, _sp_W), np.float32)
            if 0 < pi < P - 1:
                c_, p_, n_ = _sp_stack[pi], _sp_stack[pi - 1], _sp_stack[pi + 1]
                dp, dn = (c_ != p_), (c_ != n_)
                sp = dp & dn                 # differs from BOTH neighbors = unfittable flicker
                coh = (dp | dn) & (~sp)       # unstable but matches >=1 neighbor = winnable boundary
                wmap[coh] = _sp_up
                wmap[sp] = _sp_dn
                _sp_n_spike += int(sp.sum())
                _sp_n_coh += int(coh.sum())
            _spike_w_mx.append(mx.array(wmap[None]))  # (1,H,W)
        _sp_byte_identical = (_sp_dn == 1.0 and _sp_up == 1.0)
        print(json.dumps({"stage": "seg_spike_reweight", "active": True, "n_pairs": int(P),
                          "downweight": _sp_dn, "upweight": _sp_up,
                          "spike_px_total": _sp_n_spike, "coherent_px_total": _sp_n_coh,
                          "byte_identical_scalars": _sp_byte_identical,
                          "note": "per-pixel seg-CE reweight: down-weight single-frame flicker "
                          "(~88.6%% irreducible, smooth-is-optimal), up-weight coherent boundary; "
                          "A/B owed (needs GO); pointer 0.19110 UNMOVED"}), flush=True)

    # (--boundary-distance-weight) PRECOMPUTE the per-pair GT-boundary band maps. theta-INDEPENDENT
    # + computed ONCE per pair from the cached GT argmax (pure numpy/scipy distance transform — NO
    # SegNet forward), spike-map style. Memory when ON: P*H*W*4 B ~= 472 MB at n600 (noted for the
    # launcher preflight); DEFAULT OFF => _bd_band_prov stays None => zero cost, byte-identical.
    if bd_w > 0.0:
        _bd_band_prov = []
        _bd_mass = 0.0
        for pi in range(P):
            _bmap = boundary_distance_band_map(np.asarray(gt.lstars[pi]))
            _bd_mass += float(_bmap.mean())
            _bd_band_prov.append(mx.array(_bmap[None]))  # (1,H,W)
        print(json.dumps({"stage": "boundary_distance", "active": True, "n_pairs": int(P),
                          "weight": bd_w, "band_px": 2.0,
                          "band_px_share_mean": round(_bd_mass / max(P, 1), 5),
                          "note": "SDF-native Kervadec boundary-placement loss on the GT-boundary "
                          "band (distance-transform, once per pair); council levelset-loss-geometry "
                          "symposium 2026-07-05; A/B owed (needs GO); pointer 0.19110 UNMOVED"}),
              flush=True)
        # (MB-TWIN #313) RE-POINT the batched twin's LeverConfig at the populated provider: unlike the
        # gate dicts (mutated in place, reference-captured), ``_bd_band_prov`` is REASSIGNED from its
        # None default to a NEW list HERE — AFTER the _micro_batch_lc snapshot above — so the config's
        # captured None is stale. Without this refresh, --boundary-distance-weight would be SILENTLY
        # dropped under --micro-batch-pairs (NO-FAKE: a silent wrong result is worse than a refused one).
        # The serial total_loss_fn is unaffected (it reads _bd_band_prov live at call time).
        _micro_batch_lc.bd_band_prov = _bd_band_prov

    # LEVER-4b (SUB-PIXEL BOUNDARY `t`) PRECOMPUTE. theta-INDEPENDENT + cheap (pure numpy from the cached
    # gt.margins/gt.lstars -- NO SegNet forward, NO torch autograd), so it is built INLINE here (spike-map
    # style), not by a separate tool like the through-R S_R. For each pair pi, per pixel p=(i,j) we form
    # the two axis-aligned inter-class straddles -- RIGHT (p,(i,j+1)) and DOWN (p,(i+1,j)) -- keep only
    # GENUINE-V straddles (lstar differs AND both GT margins < the flip-band `subpix_band`; MEASURED n96:
    # band 1.0 -> ~2196 active px/frame = 1.12%% of pixels, t mean 0.527 std 0.263 ~ informative Uniform),
    # and assign p its DOMINANT straddle = the one with the SHALLOWER partner margin (== the smaller-sum V,
    # since p's own margin is shared; the sharpest / most-defined boundary). Stored per pair: t_map (1,H,W)
    # f32 = the GT ratio M_GT[p]/(M_GT[p]+M_GT[q]) in [0,1] where active, -1.0 sentinel elsewhere (encodes
    # the active mask); dir_map (1,H,W) f32 in {0,1} = the dominant direction (0=right,1=down) the loss
    # shifts Mw by to gather Mw[q]. Providers stay None unless subpix_w>0 => the OFF path is byte-identical.
    # Fails CLOSED with micro-batch (the batched twin's LeverConfig does not carry this lever yet). Memory
    # ~ 2x the down-weight map (t + dir float maps): P*H*W*4*2 ~= 940 MB at n600 (trivial vs RAM; noted for
    # the launcher preflight). A/B owed (needs GO); pointer 0.19110 UNMOVED.
    if subpix_w > 0.0:
        if _use_micro_batch:
            raise ValueError(
                "--seg-subpix-boundary-weight>0 is not supported with --micro-batch-pairs>1 (the batched "
                "twin does not consume the sub-pixel boundary lever yet); run this arm at "
                "--micro-batch-pairs 1.")
        _sx_H, _sx_W = np.asarray(gt.lstars[0]).shape
        _subpix_t_prov = []
        _subpix_dir_prov = []
        _sx_n_active = 0
        _sx_t_sum = 0.0
        for pi in range(P):
            _lst = np.asarray(gt.lstars[pi], np.int64)
            _mg = np.asarray(gt.margins[pi], np.float32)
            # RIGHT straddles (p,(i,j+1)) live in cols [:, :W-1]; DOWN straddles (p,(i+1,j)) in rows [:H-1, :].
            _dh = _lst[:, :-1] != _lst[:, 1:]
            _mph = _mg[:, :-1]; _mqh = _mg[:, 1:]
            _th = _mph / (_mph + _mqh + subpix_eps)
            _vh = _dh & (_mph < subpix_band) & (_mqh < subpix_band)     # genuine-V RIGHT straddles
            _dv = _lst[:-1, :] != _lst[1:, :]
            _mpv = _mg[:-1, :]; _mqv = _mg[1:, :]
            _tv = _mpv / (_mpv + _mqv + subpix_eps)
            _vv = _dv & (_mpv < subpix_band) & (_mqv < subpix_band)     # genuine-V DOWN straddles
            # per-pixel candidate fields (inf partner margin where no candidate -> loses the min).
            _has_r = np.zeros((_sx_H, _sx_W), bool); _has_r[:, :_sx_W - 1] = _vh
            _qr = np.full((_sx_H, _sx_W), np.inf, np.float32); _qr[:, :_sx_W - 1] = _mqh
            _tr = np.zeros((_sx_H, _sx_W), np.float32); _tr[:, :_sx_W - 1] = _th
            _has_d = np.zeros((_sx_H, _sx_W), bool); _has_d[:_sx_H - 1, :] = _vv
            _qd = np.full((_sx_H, _sx_W), np.inf, np.float32); _qd[:_sx_H - 1, :] = _mqv
            _td = np.zeros((_sx_H, _sx_W), np.float32); _td[:_sx_H - 1, :] = _tv
            # dominant = shallower partner margin (ties -> right). p's own margin is shared, so this is the
            # smaller-sum (sharpest) V.
            _pick_r = _has_r & (~_has_d | (_qr <= _qd))
            _pick_d = _has_d & (~_has_r | (_qd < _qr))
            _t_full = np.full((_sx_H, _sx_W), -1.0, np.float32)
            _dir_full = np.zeros((_sx_H, _sx_W), np.float32)
            _t_full[_pick_r] = _tr[_pick_r]; _dir_full[_pick_r] = 0.0
            _t_full[_pick_d] = _td[_pick_d]; _dir_full[_pick_d] = 1.0
            _act = _pick_r | _pick_d
            _sx_n_active += int(_act.sum())
            _sx_t_sum += float(_t_full[_act].sum()) if _act.any() else 0.0
            _subpix_t_prov.append(mx.array(_t_full[None]))              # (1,H,W)
            _subpix_dir_prov.append(mx.array(_dir_full[None]))          # (1,H,W)
        _sx_t_mean = round(_sx_t_sum / _sx_n_active, 4) if _sx_n_active else 0.0
        print(json.dumps({"stage": "seg_subpix_boundary", "active": True, "n_pairs": int(P),
                          "weight": subpix_w, "v_band": subpix_band, "start_epoch": int(subpix_start),
                          "active_px_total": int(_sx_n_active),
                          "active_px_per_frame": round(_sx_n_active / max(P, 1), 1),
                          "t_target_mean": _sx_t_mean,
                          "note": "sub-pixel boundary-placement target t=M_GT[p]/(M_GT[p]+M_GT[q]) on "
                          "genuine-V straddles; supervises the witness realized margin ratio (DIRECTIONAL "
                          "upgrade of LEVER-4 #141); A/B owed (needs GO); pointer 0.19110 UNMOVED"}), flush=True)

    # LEVER-4c (ANNULUS-DIRECTED CHROMA-SHARPENING) PRECOMPUTE. theta-INDEPENDENT + cheap (pure numpy +
    # the numpy-portable bilinear ``_resize_map`` -- NO SegNet forward, NO torch autograd), so it is
    # built INLINE here (spike-map / subpix style), not by a separate tool. Per pair pi: (1) resize the
    # CAMERA GT frame gt.gt_f1[pi] (874x1164x3, [0,255]) to the SegNet-INPUT (SEG_H,SEG_W)=(384,512) with
    # the SAME bilinear (align_corners=False) that SegNet.preprocess_input uses (upstream/modules.py:109
    # F.interpolate mode='bilinear', no normalization) -> the GT frame AS SegNet reads it; (2) chroma :=
    # rgb - BT.601-luma (the SAME BT.601 the witness _apply_chroma uses) -> the per-pixel GT chroma target
    # (LUMA-INVARIANT); (3) annulus weight = (GT margin < band) as {0,1} (MEASURED gt_n96: band 1.0 =>
    # 93.4% of chroma-flips inside it). Stored per pair: chroma_gt (1,H,W,3) f32 + annulus_w (1,H,W) f32.
    # Providers stay None unless chroma_bnd_w>0 => the OFF path is byte-identical. Fails CLOSED with
    # micro-batch (the batched twin's LeverConfig does not carry this lever yet). Memory ~ P*H*W*4*4 (3
    # chroma channels + 1 mask) ~= 1.9 GB at n600 (trivial vs RAM; noted for the launcher preflight).
    # A/B owed (needs GO); pointer 0.19110 UNMOVED.
    if chroma_bnd_w > 0.0:
        if _use_micro_batch:
            raise ValueError(
                "--seg-chroma-boundary-weight>0 is not supported with --micro-batch-pairs>1 (the batched "
                "twin does not consume the chroma-sharpening lever yet); run this arm at "
                "--micro-batch-pairs 1.")
        from tac.optimization.frame1_seg_safe_pose_atoms import _resize_map as _chroma_resize_map
        _ch_H, _ch_W = np.asarray(gt.lstars[0]).shape                  # (384, 512) SegNet output == input
        _chroma_gt_prov = []
        _chroma_w_prov = []
        _ch_n_active = 0
        for pi in range(P):
            _cam = np.asarray(gt.gt_f1[pi], np.float32)                # (874,1164,3) camera GT [0,255]
            # per-channel bilinear resize (align_corners=False) to SegNet-input res == what SegNet reads.
            _rs = np.stack([_chroma_resize_map(_cam[:, :, ch], _ch_H, _ch_W)
                            for ch in range(3)], axis=-1).astype(np.float32)   # (384,512,3)
            _lum = 0.299 * _rs[:, :, 0] + 0.587 * _rs[:, :, 1] + 0.114 * _rs[:, :, 2]  # (384,512) BT.601
            _chr = _rs - _lum[:, :, None]                              # (384,512,3) GT chroma (luma-inv)
            _mg = np.asarray(gt.margins[pi], np.float32)               # (384,512) GT top1-top2 margin
            _ann = (_mg < chroma_bnd_band).astype(np.float32)          # (384,512) fragile-annulus mask
            _ch_n_active += int(_ann.sum())
            _chroma_gt_prov.append(mx.array(_chr[None].astype(np.float32)))    # (1,H,W,3)
            _chroma_w_prov.append(mx.array(_ann[None]))                        # (1,H,W)
        print(json.dumps({"stage": "seg_chroma_boundary", "active": True, "n_pairs": int(P),
                          "weight": chroma_bnd_w, "margin_band": chroma_bnd_band,
                          "start_epoch": int(chroma_bnd_start),
                          "annulus_px_total": int(_ch_n_active),
                          "annulus_px_per_frame": round(_ch_n_active / max(P, 1), 1),
                          "annulus_frac": round(_ch_n_active / max(P * _ch_H * _ch_W, 1), 4),
                          "note": "GT chroma-match at the fragile annulus (margin<band); chroma=rgb-"
                          "BT.601-luma (luma-invariant) on the SHARED realized _f1; boundary-SHARPENER "
                          "orthogonal to the geometry levers; A/B owed (needs GO); pointer 0.19110 "
                          "UNMOVED"}), flush=True)

    # (--cache-gt-skeleton, #260 SPEED) BUILD the per-pair GT soft-skeleton cache ONCE (each sg is
    # mx.eval'd to a concrete constant, detached from any lazy graph, OUTSIDE any value_and_grad
    # transform -> safe + bit-identical to the inline recompute). Keyed by pair index pi (== c0//2 ==
    # c1//2, the SAME key thin_maps_mx / island_weight_mx use). Gated on persist-on AND not
    # micro-batch (the serial total_loss_fn is the only consumer; the batched twin recomputes). The
    # per-pair sg built here matches persistence_topology_loss_mlx's inline `g` construction op-for-op
    # (precompute_sg_mlx uses the identical stack->reshape->soft_skeleton), so sg_precomputed== inline.
    if cache_gt_skeleton and persist_w > 0.0 and persist_classes and not _use_micro_batch:
        from tac.boundary_math.persistence_topology_loss import precompute_sg_mlx as _precompute_sg_mlx
        _sg_cache = {}
        for _pi in range(P):
            _sg = _precompute_sg_mlx(lstar_cache[_pi][0], persist_classes, persist_cldice_iters)
            mx.eval(_sg)  # materialize as a concrete constant (bit-identical to the inline recompute)
            _sg_cache[_pi] = _sg
        print(json.dumps({"stage": "cache_gt_skeleton", "n_pairs": int(P),
                          "target_classes": list(persist_classes), "cldice_iters": persist_cldice_iters,
                          "note": "precomputed CONSTANT GT soft-skeleton per pair (bit-identical "
                          "speed-only; skips ~half the clDice recompute); pointer 0.19110 UNMOVED"}),
              flush=True)

    # ---- realized CPU-torch verdict over a subset (the AUTHORITY trajectory) ----
    vpairs = list(range(0, P, max(1, P // max(args.verdict_pairs, 1)))) if args.verdict_pairs < P else list(range(P))
    vpairs = vpairs[: args.verdict_pairs] if args.verdict_pairs else list(range(P))

    def _fwd_numpy(deploy: dict[str, np.ndarray], feats_np: np.ndarray, code_row: np.ndarray):
        return levelset_rgb_forward_numpy(
            deploy, feats_np, code_row, n_hidden=args.n_hidden, hidden_dim=args.hidden_dim,
            n_classes=5, activation=args.activation, softmax_temp=float(model.softmax_temp),
            # (FEED-fb) CURRENT (possibly annealed) beta -> the verdict/deploy render uses the SAME
            # beta the model is at now (NO-FAKE). Bit-identical when anneal off: model.hosc_beta == args.hosc_beta.
            wire_w0=args.wire_w0, wire_s0=args.wire_s0, hosc_beta=float(model.hosc_beta), hosc_omega=args.hosc_omega,
            chroma=args.chroma,
        )

    def _pc_verdict_f0_uint8(pi: int, deploy: dict[str, np.ndarray]) -> np.ndarray:
        """#224/#205 pose-carrier NO-FAKE verdict frame0: warp the carrier SOURCE by the DEPLOYED
        (EMA-shadow, int8-dequant) carrier twist xi_eff, so the advisory d_pose measures what the
        carrier actually produces. xi_stored uses the original fp32 table (the stored twist ships fp16,
        not int8; the trained residual dxi rides the deploy dict). Native-res uint8 (874x1164x3)
        matching the witness f1 verdict contract + cpu_verdict_d_pose.

        SOURCE (per --pose-carrier-source): real_keyframe (default) warps the STORED gt_f0;
        generated (STORE-NOTHING) warps the witness's OWN plain frame0 render up-to-camera (== the
        byte-close store_nothing warp source: _fwd_numpy(f0 code) -> _torch_R_to_camera_uint8), so the
        advisory d_pose reflects the store-nothing decode (NOT the real keyframe)."""
        from tac.boundary_math.warp_real_luma_frame0 import warp_frame0_uint8_numpy as _pc_warp_u8
        xi = np.asarray(pose_carrier_xi_stored[pi], np.float64)
        scale = float(args.pose_carrier_residual_scale)
        if str(args.pose_carrier_residual_mode) == "table":
            dxi = np.asarray(deploy.get("pose_carrier.dxi"), np.float64)[pi]
        else:  # film: numpy twin of gelu(film_in(code)) -> film_out (advisory reconstruction)
            from scipy.special import erf
            code = np.asarray(deploy["code"][2 * pi + 0], np.float64)
            w_in = np.asarray(deploy["pose_carrier.film_in.weight"], np.float64)
            b_in = np.asarray(deploy["pose_carrier.film_in.bias"], np.float64)
            w_out = np.asarray(deploy["pose_carrier.film_out.weight"], np.float64)
            b_out = np.asarray(deploy["pose_carrier.film_out.bias"], np.float64)
            h = code @ w_in.T + b_in
            h = 0.5 * h * (1.0 + erf(h / np.sqrt(2.0)))    # exact gelu (matches mlx.nn.gelu)
            dxi = (h @ w_out.T + b_out).reshape(-1)
        xi_eff = xi + scale * dxi
        if str(getattr(args, "pose_carrier_source", "real_keyframe")) == "generated":
            # STORE-NOTHING: warp the witness's OWN plain frame0 render (up to camera-native uint8),
            # not the stored real keyframe -> the same source the store_nothing byte-close decodes.
            rgb, _phi = _fwd_numpy(deploy, _feats_np_for_pair(pi), deploy["code"][2 * pi + 0])
            src_native = _torch_R_to_camera_uint8(rgb.reshape(render_h, render_w, 3))
        else:
            src_native = np.asarray(gt.gt_f0[pi])
        return _pc_warp_u8(src_native, xi_eff, pose_carrier_geom)

    def _render_numpy_deploy(deploy: dict[str, np.ndarray], pi: int, fk: int) -> np.ndarray:
        """THE ONE CODEPATH (fp32 numpy, deploy-faithful) — same forward the byte-close/inflate use.
        Uses the PER-PAIR feats (curvelet [+ self-orient]) so the verdict == the deploy render. In
        residual mode the INR RGB is COMPOSED with the FIXED bulk (where(mask, INR, bulk)) BEFORE R,
        so the advisory d_seg reflects the COMPOSED witness that ships (NO-FAKE). #224 pose-carrier:
        frame0 (fk==0) routes through the carrier warp so the d_pose verdict measures the carrier."""
        if pose_carrier is not None and fk == 0:
            return _pc_verdict_f0_uint8(pi, deploy)
        rgb, _phi = _fwd_numpy(deploy, _feats_np_for_pair(pi), deploy["code"][2 * pi + fk])
        rgb_hw3 = rgb.reshape(render_h, render_w, 3)
        if _compose_np is not None:
            rgb_hw3 = _compose_np(rgb_hw3, pi)
        return _torch_R_to_camera_uint8(rgb_hw3)

    def _dir_feats_from_argmax(argmax: np.ndarray) -> np.ndarray:
        """argmax (H,W) int -> self-orientation directional feats (P, dir_w). SAME numpy/scipy
        tangent->fourier path for BOTH the numpy and GPU reorient (only the argmax SOURCE differs)."""
        return self_orientation_directional_feats(
            coords_np, argmax, n_freqs=n_dir_freqs,
            freq_across=args.freq_across, freq_along=args.freq_along).astype(np.float32)

    def _recompute_self_orient_gpu(deploy: dict[str, np.ndarray]) -> float:
        """FEED-eo --gpu-reorient: the per-pair argmax (the GPU-idle 600-numpy-forward bottleneck,
        ~499s every reorient at n600) is computed on MLX-GPU via the fp32 twin forward instead. The
        downstream tangent->directional-fourier feats stay the SAME numpy/scipy path. PARITY-GATED
        (fp32-GPU vs fp64-numpy argmax differs at boundary px) -> default-off; adopt only after the
        probe shows cos>0.999 + negligible d_seg A/B. The deploy weights are dequantized ONCE to mx;
        per-pair feats are built+freed one-at-a-time (memory-bounded, like the numpy path)."""
        deploy_mx = {k: mx.array(np.asarray(v, np.float32)) for k, v in deploy.items()
                     if k not in ("code",) and not (k == "B" or k.endswith("_B"))}
        codes_np = np.asarray(deploy["code"], np.float32)
        mag = 0.0
        with temporary_mlx_device(args.mlx_device):
            for pi in range(P):
                feats_mx = mx.array(_feats_np_for_pair(pi))
                code_row = mx.array(codes_np[2 * pi + 1])
                amx = levelset_sdf_argmax_mlx(
                    deploy_mx, feats_mx, code_row, n_hidden=args.n_hidden, hidden_dim=args.hidden_dim,
                    activation=args.activation, wire_w0=args.wire_w0, wire_s0=args.wire_s0,
                    hosc_beta=float(model.hosc_beta), hosc_omega=args.hosc_omega)  # FEED-fb current beta
                mx.eval(amx)
                argmax = np.asarray(amx).reshape(render_h, render_w).astype(np.int64)
                df = _dir_feats_from_argmax(argmax)
                dir_feats_per_pair[pi] = df
                if _aa_so_fine:  # snapshot base argmax (int8) for the fine dir-feat recompute
                    base_argmax_per_pair[pi] = argmax.astype(np.int8)
                mag += float(np.abs(df).mean())
                del feats_mx, amx, code_row
            mx.clear_cache()
        return mag / max(P, 1)

    def recompute_self_orient(deploy: dict[str, np.ndarray]) -> float:
        """Self-orientation FIXED-POINT step: from the EMA deploy frame1 argmax (current feats),
        recompute each pair's directional feats. Returns the mean |dir feat| (non-triviality check)."""
        if not use_self_orient:
            return 0.0
        if getattr(args, "gpu_reorient", False):
            return _recompute_self_orient_gpu(deploy)
        mag = 0.0
        for pi in range(P):
            _rgb, phi = _fwd_numpy(deploy, _feats_np_for_pair(pi), deploy["code"][2 * pi + 1])
            argmax = phi.argmax(-1).reshape(render_h, render_w).astype(np.int64)
            df = self_orientation_directional_feats(
                coords_np, argmax, n_freqs=n_dir_freqs, freq_across=args.freq_across, freq_along=args.freq_along)
            dir_feats_per_pair[pi] = df.astype(np.float32)
            if _aa_so_fine:  # snapshot base argmax (int8) for the fine dir-feat recompute
                base_argmax_per_pair[pi] = argmax.astype(np.int8)
            mag += float(np.abs(df).mean())
        return mag / max(P, 1)

    def _project_shadow_film_np(params_np: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """(review Med1) Re-orthonormalize the EMA SHADOW's film.weight for the DEPLOYED artifact.

        The EMA shadow is an arithmetic average of (per-step on-manifold) film.weight matrices, which
        is itself NOT orthonormal -> the shipped/verdicted weight drifts OFF-Stiefel and PR(M)=PR(cov
        code) no longer holds for what actually ships. Re-project film.weight onto orthonormal columns
        so the DEPLOYED (verdict + byte-close) weight is on-manifold. Returns a SHALLOW copy with
        film.weight replaced; the live ``ema.shadow`` is UNTOUCHED so --resume-from stays bit-faithful
        to a continuous run (the resume sidecar keeps the un-projected shadow). No-op unless
        --film-stiefel (default OFF => byte-identical)."""
        if not args.film_stiefel or "film.weight" not in params_np:
            return params_np
        out = dict(params_np)
        out["film.weight"] = np.asarray(
            stiefel_project_columns(mx.array(params_np["film.weight"])), np.float32)
        return out

    # (#302) PER-CLASS CRITICAL-NUCLEUS GUARD gate. ON when the CE->tau nucleus guard is enabled OR
    # the readiness telemetry is requested (the guard consumes the same per-class counts). OFF
    # (default; the #205 path) => the verdict uses _verdict_dseg_dpose_chunked UNCHANGED, no counts,
    # no handoff_readiness row, _evt_state["nucleus_ready"] stays True => the trigger + every lever
    # are byte-identical. The readiness telemetry is OBSERVABILITY-ONLY (never read into training).
    _nucleus_guard_on = bool(getattr(args, "curriculum_nucleus_guard", False))
    _nucleus_on = _nucleus_guard_on or bool(getattr(args, "handoff_readiness_telemetry", False))
    _nucleus_within_flip_thresh = float(getattr(args, "curriculum_nucleus_within_flip", 0.5))
    _nucleus_min_part_frac = float(getattr(args, "curriculum_nucleus_min_part_frac", 0.0))

    # (SENSE) opt-in per-verdict annulus_convergence telemetry gate. OFF (default; the #205 path) =>
    # the verdict uses _verdict_dseg_dpose_chunked UNCHANGED (no realized-map collection, no annulus
    # dict, no row) => BYTE-IDENTICAL. OBSERVABILITY-ONLY (never read into training/parity/resume).
    _annulus_on = bool(getattr(args, "annulus_telemetry", False))
    _annulus_band = float(getattr(args, "annulus_band", 2.0))
    _annulus_bottom_k = float(getattr(args, "annulus_bottom_k", 0.05))

    def _verdict_v(f0s: list, f1s: list) -> dict[str, float]:
        """Shared verdict tail: given rendered f0s/f1s over ``vpairs``, return the {d_seg,d_pose} dict.

        Nucleus OFF + --no-annulus-telemetry => the UNCHANGED _verdict_dseg_dpose_chunked call =>
        byte-identical to the sealed #205 verdict. Annulus ON (the argparse DEFAULT — score-neutral
        observability defaults ON per the orphaned-signal rule) => reuse the realized argmax from the
        SAME forward (return_realized) and stash the annulus metrics under v['annulus'] in a try/except
        that can NEVER crash the verdict. Nucleus ON => +per-class counts in the same forward (unchanged).
        The rare nucleus+annulus combo takes one dedicated realized forward (still try/except-guarded).
        The d_seg/d_pose SCALARS are identical across all branches (argmax-variant d_seg is bit-identical)."""
        lstars_v = [gt.lstars[pi] for pi in vpairs]
        poses_v = [gt.gt_poses[pi] for pi in vpairs]
        _realized: "list | None" = None
        if _nucleus_on:
            d_seg, d_pose, _ncounts = _verdict_dseg_dpose_nucleus_chunked(
                seg_cpu, posenet_cpu, f0s, f1s, lstars_v, poses_v, vbatch=int(args.verdict_batch))
            v: dict[str, float] = {"d_seg": d_seg, "d_pose": d_pose, "nucleus_counts": _ncounts}
        elif _annulus_on:
            d_seg, d_pose, _realized = _verdict_dseg_dpose_chunked(
                seg_cpu, posenet_cpu, f0s, f1s, lstars_v, poses_v,
                vbatch=int(args.verdict_batch), return_realized=True)
            v = {"d_seg": d_seg, "d_pose": d_pose}
        else:
            d_seg, d_pose = _verdict_dseg_dpose_chunked(
                seg_cpu, posenet_cpu, f0s, f1s, lstars_v, poses_v, vbatch=int(args.verdict_batch))
            v = {"d_seg": d_seg, "d_pose": d_pose}
        if _annulus_on:
            try:
                if _realized is None:  # nucleus branch forwarded but discarded maps => dedicated pass
                    _realized = _annulus_realized_maps(seg_cpu, f1s, lstars_v, int(args.verdict_batch))
                v["annulus"] = _annulus_metrics_from_maps(
                    _realized, lstars_v, [gt.margins[pi] for pi in vpairs],
                    band=_annulus_band, bottom_k=_annulus_bottom_k,
                    chunk=int(args.verdict_batch))
            except Exception as exc:  # telemetry MUST NEVER crash or corrupt the verdict path.
                v["annulus"] = {"error": f"{type(exc).__name__}: {exc}"}
            # (2026-07-07 telemetry enhancement) PER-CLASS d_seg decomposition from the SAME realized
            # maps (zero extra scorer cost). Closes the §15 apparatus gap: per-class trajectories for
            # the α_lane<α_road weak-KAM check + powerlaw_meat_exit + per-class λ sensors. Score-
            # neutral OBSERVABILITY-ONLY (popped before history in _emit_verdict_row); fail-open.
            try:
                if _realized is not None:
                    from tac.witness_control.perclass_verdict import (
                        per_class_dseg_fields, per_class_flip_stats)
                    _fl, _px = per_class_flip_stats(_realized, lstars_v)
                    v["per_class"] = per_class_dseg_fields(_fl, _px)
            except Exception as exc:
                v["per_class"] = {"error": f"{type(exc).__name__}: {exc}"}
        return v

    def realized_verdict() -> dict[str, float]:
        # (fix a+b+c) verdict the EMA SHADOW, int8-DEQUANTIZED, via the fp32 numpy ONE CODEPATH
        # (NOT the MLX-GPU reduced-precision forward — the 4th artifact). This IS the deploy render.
        # (review Med1) project the shadow film.weight back onto Stiefel so the advisory d_seg reflects
        # the ON-MANIFOLD deployed weight (no-op unless --film-stiefel => bit-identical).
        ema_np = _project_shadow_film_np({k: np.asarray(v, np.float32) for k, v in ema.shadow.items()})
        deploy = int8_dequant_params(ema_np)
        f0s, f1s = [], []
        for pi in vpairs:
            f0s.append(_render_numpy_deploy(deploy, pi, 0))
            f1s.append(_render_numpy_deploy(deploy, pi, 1))
        # (#205 OOM fix) chunk the CPU-scorer inference (bit-identical; eval-mode BN running stats).
        # pose VERDICT still measured (monitoring) but pose is NOT the witness's job at w_pose=0
        # (default). The deploy d_pose is OPEN on the witness — measured through the byte-closed
        # store_nothing/table carrier (#205 R1), NO ancestor number (the 3.4e-5 was ANCESTOR-RGB,
        # never validated on this vehicle; see CLAUDE.md "Pose is SOLVED" caveat + axis-9).
        # (#302) nucleus path: same bit-identical d_seg/d_pose PLUS per-class counts in the SAME
        # SegNet forward (OFF => the original chunked call, byte-identical, no extra cost).
        # (SENSE) annulus path (opt-in) reuses the realized argmax from the same forward. All routed
        # through the shared _verdict_v tail; the flag-absent path is byte-identical.
        return _verdict_v(f0s, f1s)

    # ---- ASYNC verdict (FEED-em; ADDITIVE, DEFAULT-OFF via --async-verdict). The realized
    # CPU-torch verdict (render fp32 numpy + SegNet/PoseNet) is PURELY OBSERVATIONAL — the
    # training loop NEVER reads its result — so running it in a BACKGROUND THREAD off a
    # POINT-IN-TIME snapshot does NOT change the training trajectory at all (BIT-IDENTICAL
    # weights/checkpoints; only the verdict CADENCE may self-throttle under load). Mirrors the
    # base_ch20 async-CPU-authority pattern in src/tac/torch_vehicle/driver.py. The snapshot is
    # captured on the MAIN thread (cheap) so the worker reads ONLY its own copies + constants
    # (curv_feats_np, gt, frozen scorers) -> RACE-FREE (it never touches ema.shadow / model /
    # dir_feats_per_pair / cf_mx_cache, all of which the main loop keeps mutating). The worker
    # uses NO MLX op (pure numpy+torch) so it cannot race the GPU stream.
    def _capture_verdict_snapshot() -> dict[str, Any]:
        return {
            # (review Med1) project the shadow film.weight on-manifold so the ASYNC verdict matches the
            # deployed (byte-closed) artifact (no-op unless --film-stiefel => bit-identical snapshot).
            "ema_np": _project_shadow_film_np({k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}),
            "softmax_temp": float(model.softmax_temp),
            "hosc_beta": float(model.hosc_beta),  # FEED-fb: snapshot the live (possibly annealed) beta
            "dir": ({pi: dir_feats_per_pair[pi].copy() for pi in vpairs} if use_self_orient else None),
        }

    def _feats_for_snapshot(pi: int, dir_snap) -> np.ndarray:
        if not use_self_orient:
            return curv_feats_np
        return np.concatenate([curv_feats_np, dir_snap[pi]], axis=-1).astype(np.float32)

    def _verdict_from_snapshot(snap: dict[str, Any]) -> dict[str, float]:
        # BIT-IDENTICAL to realized_verdict() on the captured state: same int8 dequant, same
        # fp32 ONE-CODEPATH forward, same softmax_temp, same per-pair feats, same CPU scorers.
        deploy = int8_dequant_params(snap["ema_np"])
        st = snap["softmax_temp"]
        sb = snap["hosc_beta"]  # FEED-fb: the live beta captured at schedule time (anneal-correct, NO-FAKE)
        f0s, f1s = [], []
        for pi in vpairs:
            fnp = _feats_for_snapshot(pi, snap["dir"])
            rgb0, _ = levelset_rgb_forward_numpy(
                deploy, fnp, deploy["code"][2 * pi + 0], n_hidden=args.n_hidden, hidden_dim=args.hidden_dim,
                n_classes=5, activation=args.activation, softmax_temp=st, wire_w0=args.wire_w0,
                wire_s0=args.wire_s0, hosc_beta=sb, hosc_omega=args.hosc_omega, chroma=args.chroma)
            rgb1, _ = levelset_rgb_forward_numpy(
                deploy, fnp, deploy["code"][2 * pi + 1], n_hidden=args.n_hidden, hidden_dim=args.hidden_dim,
                n_classes=5, activation=args.activation, softmax_temp=st, wire_w0=args.wire_w0,
                wire_s0=args.wire_s0, hosc_beta=sb, hosc_omega=args.hosc_omega, chroma=args.chroma)
            _r0 = rgb0.reshape(render_h, render_w, 3)
            _r1 = rgb1.reshape(render_h, render_w, 3)
            if _compose_np is not None:  # residual mode: compose the FIXED bulk before R (NO-FAKE)
                _r0 = _compose_np(_r0, pi)
                _r1 = _compose_np(_r1, pi)
            # #224 pose-carrier: frame0 through the carrier warp (deploy int8-dequant xi_eff) so the
            # ASYNC d_pose verdict measures the carrier too (matches the sync _render_numpy_deploy path).
            if pose_carrier is not None:
                f0s.append(_pc_verdict_f0_uint8(pi, deploy))
            else:
                f0s.append(_torch_R_to_camera_uint8(_r0))
            f1s.append(_torch_R_to_camera_uint8(_r1))
        # (#205 OOM fix) chunk the CPU-scorer inference (bit-identical; eval-mode BN running stats).
        # (#302) nucleus + (SENSE) annulus routed through the SAME shared _verdict_v tail as the sync
        # path, so the ASYNC worker's row is bit-identical and the flag-absent path is byte-identical.
        return _verdict_v(f0s, f1s)

    history: list[dict[str, Any]] = []
    _verdict_lock = threading.Lock()
    _verdict_thread: dict[str, Any] = {"t": None, "ep": None}
    _verdict_skipped = [0]
    # ---- BEST-d_seg checkpoint tracker (EMA non-negotiable + per-stage discipline). The rolling
    # "latest" + per-stage ckpts in _do_checkpoint can DRIFT PAST the best realized d_seg (tau
    # over-trains past its knee; l7/Muon oscillate on the plateau) -> the best EMA shadow would be
    # LOST (the gap that forced a manual ep725 snapshot worse than the ep700 best). Per-ARM scope
    # (each out_dir tracks its own best); the campaign compares arm-bests across arms.
    _best: dict[str, Any] = {"d_seg": float("inf"), "ep": None, "path": None}
    # (#292 build-3) CLOSED-LOOP LEVER CONTROL. _cl_on gates the ENTIRE feature: OFF (default; the
    # #205 path) => no verdict capture, no bump, no early-stop, no sidecar keys => BYTE-IDENTICAL.
    # _cl_verdicts is the deterministic IN-MEMORY capture of (epoch, seg_form, d_seg, ep_loss) the
    # controller classifies on. (M2 fix) ASYNC mode DECIDES-ON-PREVIOUS: at each eval point the
    # PREVIOUS eval's verdict is joined FIRST (it has had a full eval window to run => wait ~ 0
    # instead of the full 2062-2439s verdict wall), the decision classifies the rows ending at the
    # previous eval, and only THEN is this epoch's verdict scheduled (never joined this iteration).
    # The action still depends on deterministic d_seg VALUES, never wall-clock/thread timing.
    # _cl_pending holds the PENDING-VERDICT record for the in-flight verdict (its EXACT snapshot
    # inputs) so a sidecar written before the row lands stays bit-faithfully resumable; cleared
    # under _verdict_lock when the row lands (or the worker fails). Resume restore (ON only)
    # happens next to the _evt_state restore below (where resume_cfg lives).
    _cl_on = bool(getattr(args, "closed_loop_control", False))
    _cl_state: dict[str, Any] = {"bumps": 0, "bump_add": 0.0, "post_budget_windows": 0,
                                 "stop_epoch": None}
    _cl_verdicts: list[dict[str, Any]] = []
    _cl_pending: dict[str, Any] = {"rec": None}

    def _verdict_inflight() -> bool:
        t = _verdict_thread["t"]
        return t is not None and t.is_alive()

    def _emit_handoff_readiness(counts: "dict | None", ep: int, seg_form: str) -> None:
        """(#302) Emit the ``handoff_readiness`` telemetry row from the verdict's per-class counts +
        update ``_evt_state['nucleus_ready']`` (the MEASURED half of the CE->tau trigger). Runs at
        VERDICT cadence (litsweep guard: NO per-step adaptive). ``counts is None`` (nucleus feature
        OFF) => no row, no state change => byte-identical. plateau_ok reads the CE-stage ep_loss
        history in ``_evt_state['losses']`` (populated when the nucleus feature is on) through the
        SAME ``_stage_converged`` the trigger uses, so the passive telemetry reports the FULL
        readiness predicate (plateau AND nucleus) the NEXT run's trigger will consume. Pure
        telemetry (never read back into training/parity/resume). Holds _verdict_lock (thread-safe
        with the async worker)."""
        if counts is None:
            return
        stats = _evt_nucleus_stats(counts)
        satisfied, all_ok = _evt_nucleus_satisfied(
            stats, _nucleus_within_flip_thresh, _nucleus_min_part_frac)
        _losses = list(_evt_state.get("losses", []))
        _ss = int(_evt_state.get("stage_start", 1))
        plateau_ok = bool(_stage_converged(
            list(range(_ss, _ss + len(_losses))), _losses,
            min_stage_epochs=int(getattr(args, "curriculum_min_stage_epochs", 150)),
            plateau_rel_eps=float(getattr(args, "curriculum_plateau_rel_eps", 1e-3)),
            plateau_windows=int(getattr(args, "curriculum_plateau_windows", 4))))
        row = _evt_readiness_row(ep, seg_form, stats, satisfied, all_ok, plateau_ok,
                                 _nucleus_within_flip_thresh, _nucleus_min_part_frac,
                                 guard_active=_nucleus_guard_on)
        with _verdict_lock:
            print(json.dumps(row), flush=True)
            # MEASURED trigger state: the nucleus half of the CE->tau readiness predicate. Default
            # True so a guard-OFF run never blocks (plateau/cap alone decide). Only set when the
            # guard is active; only the CE stage's readiness gates the hand-off downstream.
            if _nucleus_guard_on:
                _evt_state["nucleus_ready"] = bool(all_ok)

    def _emit_verdict_row(v: dict[str, float], ema_np: dict[str, np.ndarray], ep: int,
                          seg_form: str, ep_loss: float, *, async_tag: bool,
                          liveness: "dict[str, Any] | None" = None) -> None:
        # (#302) split per-class counts OUT of ``v`` before it flows to the float-only verdict row /
        # history / closed-loop capture (those consume d_seg/d_pose scalars only), then emit the
        # separate handoff_readiness row. ``v`` has no "nucleus_counts" unless the feature is ON.
        _ncounts = v.pop("nucleus_counts", None) if isinstance(v, dict) else None
        # (SENSE) pop the annulus metrics BEFORE the float-only ``row`` spread below (else round()
        # would hit a dict). None unless --annulus-telemetry is ON => byte-identical when absent.
        _annulus_m = v.pop("annulus", None) if isinstance(v, dict) else None
        # (2026-07-07) pop the per-class decomposition the same way (dict of lists — must not hit
        # the float spread, and must NOT reach history/result.json: observability-only).
        _per_class = v.pop("per_class", None) if isinstance(v, dict) else None
        blob = quantize_levelset_blob(ema_np)
        s = implied_score_from_verdict(v["d_seg"], v["d_pose"], blob["total_quantized_blob_bytes"])
        # (C6) LIVENESS STAMP captured at SCHEDULE time (the async worker runs later, off a snapshot;
        # the live _live dict would by then reflect a LATER epoch). None (sync callers not threading
        # it) => the fields are omitted (the sync path stamps liveness at its own call site).
        _lv = dict(liveness) if liveness else None
        with _verdict_lock:
            row = {"stage": "verdict", "epoch": ep, "seg_form": seg_form,
                   **{k: round(vv, 6) for k, vv in v.items()},
                   "blob_bytes": blob["total_quantized_blob_bytes"], "implied_S": round(s, 4),
                   "ep_loss": round(ep_loss, 3),
                   # ADDITIVE telemetry: UTC emit wall-time so dashboards read verdict
                   # arrival times DIRECTLY (the no-timestamp root cause the self-calibrating
                   # dashboard otherwise self-observes). Purely observational; never read back
                   # into training/resume/parity, not appended to history/result.json.
                   "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
            # (2026-07-07) ADDITIVE observability fields — per-class d_seg (when the annulus branch
            # collected realized maps) + process/MLX memory (#329, every verdict row). Both fail-open
            # + row-only (never history/result.json); consumers read rows by key => additive-safe.
            if isinstance(_per_class, dict) and "error" not in _per_class:
                row["d_seg_by_class"] = _per_class.get("d_seg_by_class")
                row["flip_share_by_class"] = _per_class.get("flip_share_by_class")
            try:
                from tac.witness_control.perclass_verdict import memory_telemetry_fields
                row.update(memory_telemetry_fields())
            except Exception:
                pass
            if _lv is not None:
                _fzn = (int(_lv.get("ep_tot", 0)) > 0 and int(_lv.get("acc", 0)) == 0)
                row["accepted_frac"] = round(float(_lv.get("frac", 1.0)), 4)
                row["weights_stepped"] = bool(_lv.get("stepped", True))
                row["accepted_batches"] = int(_lv.get("acc", 0))
                row["skipped_batches"] = int(_lv.get("skip", 0))
                row["frozen_epoch"] = bool(_fzn)
                if _fzn or (float(ep_loss) == 0.0 and int(_lv.get("ep_tot", 0)) > 0):
                    print(json.dumps({"stage": "confound_alarm", "alarm": "frozen_epoch", "ep": ep,
                                      "ep_loss": round(float(ep_loss), 6), "accepted_batches": int(_lv.get("acc", 0)),
                                      "note": "async verdict on a FROZEN-state epoch (all batches skipped "
                                      "/ ep_loss==0.0): not converged progress"}), flush=True)
            if async_tag:
                row["async"] = True
            print(json.dumps(row), flush=True)
            # (SENSE) companion annulus_convergence row (opt-in; under the same lock as the verdict
            # row). Absent unless --annulus-telemetry => byte-identical. Never raises (json of a plain
            # metrics/error dict); OBSERVABILITY-ONLY, never appended to history/result.json.
            if _annulus_m is not None:
                print(json.dumps(_annulus_convergence_row(_annulus_m, ep, seg_form)), flush=True)
            history.append({"epoch": ep, **v, "implied_S": s})
            # (#292 build-3) deterministic closed-loop capture (ON only; OFF appends nothing =>
            # byte-identical). Under _verdict_lock; (M2 fix) the decision point joins the PREVIOUS
            # eval's thread first, so this row is ALWAYS visible to the controller that decides at
            # the NEXT eval point (decide-on-previous).
            if _cl_on:
                _cl_verdicts.append({"epoch": int(ep), "seg_form": str(seg_form),
                                     "d_seg": float(v["d_seg"]), "ep_loss": float(ep_loss)})
                # (M2 fix) the row landed => drop the pending record for this epoch (sidecar
                # invariant, kept consistent by _cl_sidecar_snapshot's locked read: the pending
                # record is present IFF its verdict row is absent).
                _rec = _cl_pending["rec"]
                if _rec is not None and int(_rec["epoch"]) == int(ep):
                    _cl_pending["rec"] = None
        # (#302) emit the handoff_readiness row + update the nucleus trigger state — OUTSIDE the lock
        # (the helper re-acquires _verdict_lock; threading.Lock is non-reentrant). No-op when counts
        # is None (feature OFF) => byte-identical.
        _emit_handoff_readiness(_ncounts, ep, seg_form)

    def _maybe_preserve_best(d_seg: float, ep: int, shadow_np_proj: dict[str, np.ndarray],
                             softmax_temp: float, hosc_beta: float) -> None:
        """Preserve the EMA SHADOW that achieved a NEW best realized-through-R d_seg, as a DEPLOY
        npz (shadow + cfg) -> byte-close-ready AND warm-startable (resume seeds live<-shadow).

        NO-FAKE: only a FINITE, strictly-better d_seg promotes the best (NaN/inf never wins). The
        ``shadow_np_proj`` is the SAME Stiefel-projected shadow the verdict measured (async: the
        point-in-time snapshot; sync: the current shadow) -> the preserved artifact is EXACTLY what
        produced the score (no drift). Atomic (tmp+os.replace). Thread-safe: holds _verdict_lock,
        and only one async verdict is in flight at a time, so best writes never race."""
        with _verdict_lock:
            if not _is_new_best(d_seg, _best["d_seg"]):  # finite + strictly-better only
                return
            prev = _best["d_seg"]
            ema_arrays = _build_ema_checkpoint_arrays(
                shadow_np_proj, args=args, softmax_temp=float(softmax_temp),
                render_h=render_h, render_w=render_w, epoch=int(ep), in_feat=in_feat,
                hosc_beta=float(hosc_beta))
            _atomic_savez(out_dir / "levelset_witness_ema_BEST.npz", ema_arrays)
            _atomic_write_json(out_dir / "levelset_best.json", {
                "d_seg": float(d_seg), "epoch": int(ep),
                "path": "levelset_witness_ema_BEST.npz",
                "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")})
            _best.update(d_seg=float(d_seg), ep=int(ep), path="levelset_witness_ema_BEST.npz")
            print(json.dumps({"stage": "checkpoint", "kind": "best", "epoch": int(ep),
                              "d_seg": round(float(d_seg), 6),
                              "prev_best": (round(prev, 6) if np.isfinite(prev) else None),
                              "path": "levelset_witness_ema_BEST.npz"}), flush=True)

    def _schedule_async_verdict(ep: int, seg_form: str, ep_loss: float) -> bool:
        # (C6) capture the CURRENT-epoch liveness snapshot at SCHEDULE time (immutable copy); the
        # async worker emits its verdict row later, when the live _live dict would already reflect a
        # LATER epoch. Read _live directly (this fn is called SYNCHRONOUSLY at the epoch end, after
        # the completed-epoch liveness snapshot is stamped) -- keeping the call signature unchanged so
        # the M2 decide-on-previous SOURCE-ORDER guard (test_closed_loop_control) stays intact.
        _lv_snap = dict(_live)
        if _verdict_inflight():
            _verdict_skipped[0] += 1
            with _verdict_lock:
                print(json.dumps({"stage": "verdict_skip", "epoch": ep,
                                  "inflight_epoch": _verdict_thread["ep"],
                                  "total_skipped": _verdict_skipped[0],
                                  "note": "prior async verdict still running; cadence self-throttles "
                                  "(GPU never blocks)"}), flush=True)
            return False
        snap = _capture_verdict_snapshot()  # MAIN thread, cheap, point-in-time
        if _cl_on:
            # (M2 fix) PENDING-VERDICT record: THIS verdict is now in flight, so any resume sidecar
            # written before its row lands persists the EXACT inputs the worker scores (the SAME
            # point-in-time snapshot arrays captured above — snap["ema_np"] IS the materialized
            # numpy copy of the shadow at call time). A --resume-from recomputes the row
            # synchronously from these inputs through the same deterministic CPU-torch path =>
            # bit-identical to what the continuous run's thread produces => post-resume decisions
            # match the continuous run EXACTLY. Cleared under _verdict_lock when the row lands, or
            # on worker failure (a failed verdict never produces a row in the continuous run, so
            # the sidecar must not resurrect it).
            with _verdict_lock:
                _cl_pending["rec"] = {
                    "epoch": int(ep), "seg_form": str(seg_form), "ep_loss": float(ep_loss),
                    "softmax_temp": float(snap["softmax_temp"]),
                    "hosc_beta": float(snap["hosc_beta"]),
                    "ema_np": snap["ema_np"]}
        _verdict_thread["ep"] = ep

        def _worker() -> None:
            t0 = time.time()
            try:
                v = _verdict_from_snapshot(snap)
                _emit_verdict_row(v, snap["ema_np"], ep, seg_form, ep_loss, async_tag=True,
                                  liveness=_lv_snap)
                # HARDENING: preserve the best EMA shadow from the SAME snapshot the verdict scored
                # (snap["ema_np"] is the point-in-time Stiefel-projected shadow; cfg from the snap).
                _maybe_preserve_best(v["d_seg"], ep, snap["ema_np"],
                                     snap["softmax_temp"], snap["hosc_beta"])
                with _verdict_lock:
                    print(json.dumps({"stage": "verdict_async_done", "epoch": ep,
                                      "secs": round(time.time() - t0, 1)}), flush=True)
            except Exception as exc:  # an eval failure must NOT kill training (daemon thread).
                with _verdict_lock:
                    # (M2 fix) a FAILED verdict produces NO row in the continuous run => drop the
                    # pending record so a resume does not resurrect a row that never existed.
                    # HONEST SCOPE (review M2-F2): this clear only helps sidecars written AFTER
                    # the failure; a sidecar written WHILE this verdict was in flight still
                    # carries the pending record, and a later resume recomputes it (row the
                    # continuous run dropped — bounded, deterministic, note-and-carry). The
                    # resume-side fail-safe (resume_pending_verdict_failed) covers the case
                    # where that recompute itself fails.
                    _rec = _cl_pending["rec"]
                    if _cl_on and _rec is not None and int(_rec["epoch"]) == int(ep):
                        _cl_pending["rec"] = None
                    print(json.dumps({"stage": "verdict_async_failed", "epoch": ep,
                                      "err": f"{type(exc).__name__}: {exc}"}), flush=True)

        t = threading.Thread(target=_worker, name=f"async-verdict-ep{ep}", daemon=True)
        _verdict_thread["t"] = t
        t.start()
        return True

    def _join_async_verdict() -> None:
        t = _verdict_thread["t"]
        if t is not None and t.is_alive():
            print(json.dumps({"stage": "verdict_async_join",
                              "note": "waiting for in-flight async verdict before continuing"}), flush=True)
            t.join()
        _verdict_thread["t"] = None

    def _cl_sidecar_snapshot() -> dict[str, Any]:
        """(M2 fix) Consistent (state, verdicts, pending) view for the resume sidecar, under
        _verdict_lock so the async worker can never land a row BETWEEN the two reads (which could
        otherwise lose the row on resume: verdicts snapshotted without it AND pending read after
        the worker's clear). Locked invariant: pending present IFF its verdict row is absent."""
        with _verdict_lock:
            _rec = _cl_pending["rec"]
            return {**_cl_state, "verdicts": list(_cl_verdicts),
                    "pending": (dict(_rec) if _rec is not None else None)}

    def _cl_decide(ep: int) -> bool:
        """(#292 build-3; M2 decide-on-previous reorder) ONE closed-loop decision at an eval point:
        classify the captured verdict rows (same math as tools/witness_control_monitor — sustained
        erosion vs recoverable transient), take the BOUNDED action via _cl_step, emit the
        closed_loop telemetry row. Returns True when the early-stop is armed. ASYNC mode calls this
        AFTER joining the PREVIOUS eval's verdict and BEFORE scheduling this epoch's — the row set
        deterministically ends at the previous eval (1-eval lag = eval_every epochs ≪ the ~100-ep
        erosion timescale) and the GPU never waits the full verdict wall. SYNC mode calls it after
        the inline verdict (current-epoch row; unchanged semantics — sync has no wall problem).
        No rows captured yet => no action => False. Never joins/schedules a verdict itself."""
        if not _cl_verdicts:
            return False
        # (C5 META-CONFOUND FIX 2026-07-05) LIVENESS PRECONDITION FIRST: the controller must NEVER
        # certify a FROZEN run "converging"/"plateau" (and thus never BUMP the exploding eikonal on a
        # dead run -- the meta-confound made concrete: v5 ep125/150 "converging" AFTER the ep113
        # freeze). If the decision window's ep_loss are ALL 0.0 (the frozen tell) OR the run's
        # accepted-batch fraction is below the liveness floor, classify FROZEN + action STOP and arm
        # the clean early-stop. This gate runs BEFORE _cl_classify so a frozen run can never be bumped.
        _win = _cl_verdicts[-max(1, int(args.closed_loop_min_sustained_windows)):]
        _win_eploss = [float(vv.get("ep_loss", 0.0)) for vv in _win]
        _frozen = (len(_win_eploss) > 0 and all(e == 0.0 for e in _win_eploss)) or (
            float(_live["frac"]) < _CL_LIVENESS_MIN_ACCEPTED_FRAC)
        if _frozen:
            _cl_state["stop_epoch"] = int(ep)  # arm the clean early-stop (best EMA-shadow ckpt preserved)
            print(json.dumps({
                "stage": "closed_loop", "epoch": ep, "classification": "frozen",
                "action": "stop", "reason": "liveness_precondition_failed",
                "accepted_frac": round(float(_live["frac"]), 4),
                "window_ep_loss": [round(e, 6) for e in _win_eploss],
                "note": "(C5) FROZEN run -- NEVER certified converging; the controller will not bump "
                "the eikonal on a dead run. Clean early-stop armed (best ckpt preserved)."}), flush=True)
            _emit_confound_alarm("closed_loop_frozen_stop", ep=ep,
                                 accepted_frac=round(float(_live["frac"]), 4),
                                 note="closed-loop refused to classify a frozen run as converging")
            return True
        _clc = _cl_classify(
            _cl_verdicts,
            min_sustained_windows=int(args.closed_loop_min_sustained_windows))
        _cla = _cl_step(
            _clc["classification"], _cl_state, ep,
            bump=float(args.closed_loop_eikonal_bump),
            max_bumps=int(args.closed_loop_max_bumps),
            stop_after=int(args.closed_loop_stop_after_windows))
        print(json.dumps({
            "stage": "closed_loop", "epoch": ep,
            "classification": _clc["classification"],
            "d_seg_slope": _clc["d_seg_slope"],
            "net_stage_slope": _clc["net_stage_slope"],
            "n_stage": _clc["n_stage"],
            "eikonal_bump": round(float(_cl_state["bump_add"]), 6),
            "bumps_used": int(_cl_state["bumps"]),
            "action": (_cla["action"] if _cla is not None else "none"),
            # (C6) LIVENESS STAMP on every closed_loop decision row.
            "accepted_frac": round(float(_live["frac"]), 4),
            "weights_stepped": bool(_live["stepped"]),
            **({k: v for k, v in (_cla or {}).items() if k != "action"})}), flush=True)
        return _cl_state["stop_epoch"] is not None

    # per-pair MLX coord-feats cache: shared curvelet tensor when no self-orient; rebuilt on each
    # reorient when self-orient is on (so the train forward uses the SAME per-pair feats the
    # numpy verdict/deploy uses -> ONE codepath).
    cf_mx_cache: list[Any] | None = None

    def _rebuild_cf_mx_cache() -> None:
        # MEMORY-BOUNDED in-place rebuild (FEED-eh): free each OLD per-pair MLX feats entry BEFORE
        # allocating the new one (the naive list-comprehension held old+new => 2x ~41GB at n600 =>
        # OOM at the ep50 reorient). Peak now ~= ONE cache; BIT-IDENTICAL values.
        nonlocal cf_mx_cache
        cf_mx_cache = rebuild_per_pair_feats_in_place(
            cf_mx_cache, P, _feats_np_for_pair, mx_array=mx.array, mx_eval=mx.eval)

    def _cf_mx(pi: int):
        # per-pair cache path when self-orient OR the ground-frame chart (#194) is on; the shared
        # tensor otherwise (byte-identical default).
        return coord_feats_mx if not (use_self_orient or use_gfc) else cf_mx_cache[pi]

    # #224 (Wave B) FINE self-orient dir-feats (AA-supersample + --self-orient). Recompute from the
    # snapshotted base argmax: NN-upsample to the ss*grid -> fine EDT-tangent -> directional Fourier
    # (the SAME self_orientation_directional_feats path as the base, at fine coords). Pre-first-reorient
    # (argmax None) -> zeros -> pure-curvelet fine (matches the base zeros-until-reorient contract).
    def _fine_dir_feats_np(pi: int) -> np.ndarray:
        cols = _aa_coords_fine.shape[0]
        ba = base_argmax_per_pair[pi]
        if ba is None:
            return np.zeros((cols, dir_w), np.float32)
        arg_fine = np.kron(np.asarray(ba, np.int64), np.ones((aa_ss, aa_ss), np.int64))
        return self_orientation_directional_feats(
            _aa_coords_fine, arg_fine, n_freqs=n_dir_freqs,
            freq_across=args.freq_across, freq_along=args.freq_along).astype(np.float32)

    def _rebuild_fine_dir_cache() -> None:
        # full mode: recompute ALL P fine dir-feats ONCE (amortized across the reorient window).
        # batch mode: just INVALIDATE the bounded LRU (recomputed lazily on next use).
        if not _aa_so_fine:
            return
        _aa_fine_lru.clear()
        if _aa_fine_mode == "full":
            for pi in range(P):
                _aa_fine_dir_full[pi] = mx.array(_fine_dir_feats_np(pi))
            mx.eval([x for x in _aa_fine_dir_full if x is not None])

    def _cf_fine_mx(pi: int):
        # per-pair FINE render feats = shared curvelet-fine (coord_feats_fine_mx) [+ fine dir-feats].
        if not _aa_so_fine:
            return coord_feats_fine_mx
        if _aa_fine_mode == "full":
            df = _aa_fine_dir_full[pi]
            if df is None:                       # pre-first-reorient safety
                df = mx.array(_fine_dir_feats_np(pi))
        else:  # batch: bounded FIFO on-demand cache (memory ~ cap*per-pair)
            df = _aa_fine_lru.get(pi)
            if df is None:
                df = mx.array(_fine_dir_feats_np(pi))
                _aa_fine_lru[pi] = df
                _cap = max(1, int(getattr(args, "aa_self_orient_fine_cache_cap", 16)))
                while len(_aa_fine_lru) > _cap:
                    _aa_fine_lru.pop(next(iter(_aa_fine_lru)))
        return mx.concatenate([coord_feats_fine_mx, df], axis=-1)

    if use_self_orient or use_gfc:
        # self-orient: rebuilt at every reorient. ground-frame chart (#194): built ONCE (static —
        # the chart is a fixed function of the stored pose table; no epoch churn).
        _rebuild_cf_mx_cache()  # ep<reorient: dir feats are zeros -> pure curvelet iso pass
        _rebuild_fine_dir_cache()  # AA fine self-orient (no-op unless --aa-self-orient-fine-mode)
        if os.environ.get("TAC_MEM_PROBE", "0") not in ("", "0", "false", "False"):
            _mm = _mlx_mem_gib(mx)
            print(json.dumps({"stage": "mem_probe", "phase": "after_cf_mx_cache_build",
                              "n_pairs": P, "rss_gib": round(_rss_gib(), 2),
                              "mlx_active_gib": round(_mm["active"], 2),
                              "mlx_cache_gib": round(_mm["cache"], 2)}), flush=True)

    # ---- CHECKPOINT closures (FEED-dz; mx->np snapshot + atomic save of the deploy EMA npz + the
    # resume sidecar). The deploy npz keeps the canonical name so the byte-close tool consumes it
    # as-is; the resume sidecar is separate so the deploy npz stays byte-close-clean. ----
    def _snapshot_numpy_state() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        shadow_np = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
        live_np = {k: np.asarray(v, np.float32) for k, v in tree_flatten(model.parameters())}
        opt_np: dict[str, np.ndarray] = {}
        try:  # best-effort: optimizer moments accelerate resume but a fresh AdamW re-warms in steps.
            for k, v in tree_flatten(opt.state):
                arr = np.asarray(v)
                if arr.dtype.kind in "fiub":
                    opt_np[k] = arr
        except Exception:
            opt_np = {}
        return shadow_np, live_np, opt_np

    def _do_checkpoint(epoch: int, *, stage_tag: str | None = None) -> dict[str, Any]:
        shadow_np, live_np, opt_np = _snapshot_numpy_state()
        # (review Med1) the BYTE-CLOSE deploy npz ships the EMA shadow; re-project its film.weight onto
        # Stiefel so the shipped artifact is ON-MANIFOLD (PR(M)=PR(cov code) holds for what ships). The
        # RESUME sidecar keeps the UN-projected shadow (bit-faithful continuous resume). No-op unless
        # --film-stiefel (default OFF => byte-identical deploy + resume npz).
        deploy_shadow_np = _project_shadow_film_np(shadow_np)
        ema_arrays = _build_ema_checkpoint_arrays(
            deploy_shadow_np, args=args, softmax_temp=float(model.softmax_temp),
            render_h=render_h, render_w=render_w, epoch=epoch, in_feat=in_feat,
            hosc_beta=float(model.hosc_beta),  # FEED-fb: persist CURRENT annealed beta in deploy cfg
            provenance=_run_provenance)        # #205: git sha + upstream snapshot sha in EVERY deploy ckpt
        resume_arrays = _build_resume_state_arrays(
            live_np, shadow_np, opt_np, args=args, epoch=epoch, in_feat=in_feat,
            # #205: persist the spike-guard window (bit-faithful step-skip on resume) + git provenance.
            recent_losses=recent_losses, provenance=_run_provenance,
            # (#292 build-2) persist the event-triggered curriculum controller state ONLY when the
            # feature is ON (closure vars _evt_on/_evt_state, assigned before any _do_checkpoint call,
            # mirroring recent_losses). OFF => None => ZERO new sidecar keys => byte-identical (#205-safe).
            evt_curriculum_state=(_evt_state if _evt_on else None),
            # (#292 build-3) persist the closed-loop controller state ONLY when the feature is ON
            # (closure vars _cl_on/_cl_state/_cl_verdicts/_cl_pending, assigned before any
            # _do_checkpoint call, mirroring _evt_state). OFF => None => ZERO new sidecar keys =>
            # byte-identical (#205-safe). (M2 fix) the snapshot is taken under _verdict_lock and
            # includes the PENDING-VERDICT record when one is in flight (bit-faithful resume).
            closed_loop_state=(_cl_sidecar_snapshot() if _cl_on else None))
        # FEED-fm FIX-1: snapshot the loop's RNG streams (global MT19937 + LEVER-5 hardness PCG64)
        # INTO the resume sidecar so --resume-from is bit-faithful to a continuous run. hardness_rng
        # is a run_train local assigned before any _do_checkpoint call (closure ref; safe).
        resume_arrays.update(_rng_state_arrays(hardness_rng))
        # rolling latest: the byte-close default name + the quick resume target (overwritten atomically).
        _atomic_savez(out_dir / "levelset_witness_ema_mlx.npz", ema_arrays)
        _atomic_savez(out_dir / "levelset_resume_state.npz", resume_arrays)
        written: dict[str, Any] = {
            "epoch": epoch, "ema_latest": "levelset_witness_ema_mlx.npz",
            "resume_latest": "levelset_resume_state.npz", "has_opt": bool(opt_np)}
        if stage_tag is not None:  # PRESERVED stage-encoded ckpt (NOT overwritten -> per-stage A/B).
            ema_pres = f"levelset_ckpt_{stage_tag}_ep{epoch}.npz"
            res_pres = f"levelset_resume_{stage_tag}_ep{epoch}.npz"
            _atomic_savez(out_dir / ema_pres, ema_arrays)
            _atomic_savez(out_dir / res_pres, resume_arrays)
            written["ema_preserved"] = ema_pres
            written["resume_preserved"] = res_pres
        return written

    # ---- RESUME restore (FEED-dz; --resume-from None => fresh start => behavior UNCHANGED). Loads
    # decoder + per-pair codes (live) + EMA shadow + optimizer (best-effort) + the epoch position;
    # self-orient dir feats are regenerated from the restored EMA argmax (not stored -> no GB bloat).
    start_epoch = 1
    resume_cfg: dict[str, Any] | None = None  # FEED-fm FIX-1: holds the sidecar cfg for the RNG
    # restore that must run AFTER hardness_rng is constructed (below); None => fresh start.
    # #205: True when --resume-from lands INSIDE the Muon finisher window (start_epoch > muon_start)
    # -> the resume block rebuilds the Muon MultiOptimizer BEFORE restoring its state, and the loop's
    # muon_switched initializes True so the in-loop switch does NOT re-init a fresh (momentum-lost)
    # optimizer. Default False (fresh start / pre-finisher resume) => BIT-IDENTICAL to the prior path.
    _resume_into_finisher = False
    # (C11) resume-drift LR re-warmup boundary: set to the resume start epoch when a stiff loss term
    # is ADDED at resume + lever-drift is allowed + a re-warmup window is configured, so the existing
    # stage-transition LR ramp softens the entry. None (default / no stiff add) => no re-warmup.
    _resume_lr_rewarmup_boundary: "int | None" = None
    if args.resume_from:
        from mlx.utils import tree_unflatten
        rp = _resolve_resume_path(Path(args.resume_from))
        rs = _load_resume_state(rp)
        resume_cfg = rs["cfg"]
        if not rs["live"]:
            raise ValueError(f"--resume-from {rp} has no live/param tensors (NO-FAKE: cannot resume).")
        # (DE#3 clean warm-start) --warm-start-weights-only: take ONLY the trained weights; DISCARD the
        # checkpoint's optimizer moments (=> fresh AdamW) here so the has_opt restore below is skipped
        # EVEN when the sidecar carries optP__ (the poisoned-resume trap: a DEADLOCKED resume_state has
        # stale ep150 moments + a frozen spike-guard window; the WEIGHTS are clean, the run STATE is not).
        # The spike-guard clear (below) and epoch override are gated on the same flag; lever-drift is
        # auto-allowed (a warm-start is an intentional re-treatment). DEFAULT OFF => rs is untouched =>
        # byte-identical. NOTE: a deploy ema/BEST npz ALREADY has has_opt=False + no __recent_losses, so
        # this flag is a NO-OP for that path (it only bites a full sidecar).
        _warm_start_wo = bool(getattr(args, "warm_start_weights_only", False))
        _ws = _resolve_weights_only_warm_start(
            rs, warm_start_weights_only=_warm_start_wo,
            warm_start_epoch=int(getattr(args, "warm_start_epoch", -1)),
            ckpt_start_epoch=int(rs["epoch"]) + 1)
        if _warm_start_wo:
            print(json.dumps({"stage": "warm_start_weights_only",
                              "note": "DISCARDING checkpoint optimizer moments + spike-guard window "
                              "(fresh AdamW; weights-only clean warm-start)",
                              "ckpt_had_opt": _ws["ckpt_had_opt"], "ckpt_epoch": int(rs["epoch"]),
                              "start_epoch": _ws["start_epoch"]}), flush=True)
        # (review R2a-MED-1) FAIL-CLOSED arch-drift guard BEFORE model.update. MLX model.update only
        # writes params the model ALREADY has, so a resume whose ckpt carries trained params the
        # freshly-built model lacks (e.g. the run trained with --film-per-layer / --film-concat-code but
        # the resume command omitted it) would SILENTLY DROP those trained tensors -> a corrupted,
        # non-reproducible resume discovered only at exact-eval. Refuse loudly instead. The check is
        # arch-general (any missing key), not film-specific; the persisted __cfg_film_* flags name the
        # likely cause + fix. Per CLAUDE.md resumability + deterministic-reproducibility + NO-FAKE.
        _model_param_keys = {k for k, _ in tree_flatten(model.parameters())}
        _missing_in_model = sorted(set(rs["live"]) - _model_param_keys)
        if _missing_in_model:
            _ckpt_pl = bool(int(resume_cfg.get("__cfg_film_per_layer", 0) or 0))
            _ckpt_concat = bool(int(resume_cfg.get("__cfg_film_concat_code", 0) or 0))
            _hint = []
            if _ckpt_pl and not bool(getattr(args, "film_per_layer", False)):
                _hint.append("add --film-per-layer")
            if _ckpt_concat and not bool(getattr(args, "film_concat_code", False)):
                _hint.append("add --film-concat-code")
            raise ValueError(
                f"--resume-from {rp}: the checkpoint carries {len(_missing_in_model)} trained param(s) the "
                f"rebuilt model has NO slot for (first few: {_missing_in_model[:6]}) -> model.update would "
                "SILENTLY DROP them = a corrupted, non-reproducible resume. The resume command's ARCH flags "
                f"must MATCH the trained run. Ckpt arch flags: film_per_layer={_ckpt_pl}, "
                f"film_concat_code={_ckpt_concat}, film_stiefel="
                f"{bool(int(resume_cfg.get('__cfg_film_stiefel', 0) or 0))}. "
                + (f"Fix: {', '.join(_hint)}." if _hint else
                   "Rebuild the model with the SAME architecture the checkpoint was trained with."))
        # (F2) FAIL-CLOSED render-side LEVER-drift guard (BEFORE model.update, like the film guard).
        # The loss/render-only levers add no param KEYS, so the missing-param guard above cannot see
        # them; a resume that silently drops/changes a lever the run was trained with is a
        # deterministic-repro violation. Escape: --resume-allow-lever-drift (explicit warm-start).
        if not (bool(getattr(args, "resume_allow_lever_drift", False)) or _ws["allow_lever_drift"]):
            _lever_div = _resume_lever_divergences(resume_cfg, args)
            if _lever_div:
                raise ValueError(
                    f"--resume-from {rp}: {len(_lever_div)} render-side LEVER(s) DIVERGE between the "
                    "checkpoint's training config and this resume command -> a silent lever drop/change "
                    "= a deterministic-reproducibility violation (these loss/render-only levers add no "
                    "param keys, so the arch guard above cannot catch them). Diverged: "
                    + "; ".join(_lever_div)
                    + ". Fix: MATCH the trained run's lever flags, OR pass --resume-allow-lever-drift "
                    "if this is an INTENTIONAL warm-start re-treatment.")
        # (C9 confound fix) load the MODEL from the CLEAN EMA shadow when --resume-model-from ema
        # (auto for re-treatment resumes; see the post-parse resolution). A crash mid-spike writes
        # DIVERGING live weights while the EMA shadow in the SAME file stays clean; loading live
        # re-enters the divergence. Keys missing from the shadow fall back to live so the model has a
        # FULL param set (the shadow tracks all trained params, but the merge is defensive).
        if str(getattr(args, "resume_model_from", "live")) == "ema" and rs["ema"]:
            _model_src = {**rs["live"], **rs["ema"]}
            _model_src_tag = "ema"
        else:
            _model_src = rs["live"]
            _model_src_tag = "live"
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in _model_src.items()]))
        mx.eval(model.parameters())
        print(json.dumps({"stage": "resume_model_source", "resume_model_from": _model_src_tag,
                          "requested": str(getattr(args, "resume_model_from", "live")),
                          "ema_available": bool(rs["ema"]),
                          "note": "C9: which weights loaded into the model (ema=clean shadow, "
                          "live=possibly-diverging live params)"}), flush=True)
        # (C11 confound fix) STIFF-TERM RESUME-DRIFT loud row + gentle-entry routing. A stiff loss
        # term (eikonal weight, ViscoReg viscosity, boundary-distance) ADDED or raised at resume is
        # attached FULL-WEIGHT onto an optimizer state trained WITHOUT it => a silent loss-composition
        # / level shift at the resume boundary (H5-F3: the added viscosity is exactly what runs away).
        # Enumerate the diffs; when any stiff term was added AND lever-drift is allowed, register the
        # resume epoch as a stage-transition boundary so the EXISTING LR re-warmup ramp softens the
        # entry (per-term eik/visco weight ramp is NOT applied here -- it needs the base schedule
        # machinery; the LR ramp + this loud row are the minimal safe treatment. Flagged.).
        _stiff_checks = [
            ("eikonal_weight", "__cfg_eikonal_weight", float(getattr(args, "eikonal_weight", 0.0))),
            ("eikonal_viscosity", "__cfg_eikonal_viscosity", float(getattr(args, "eikonal_viscosity", 0.0))),
            ("boundary_distance_weight", "__cfg_boundary_distance_weight",
             float(getattr(args, "boundary_distance_weight", 0.0))),
        ]
        _stiff_added = []
        for _nm, _ck, _cur in _stiff_checks:
            _ckpt_v = float(resume_cfg.get(_ck, 0.0) or 0.0) if _ck in resume_cfg else None
            if _ckpt_v is not None and _cur > _ckpt_v + 1e-9:
                _stiff_added.append({"term": _nm, "ckpt": _ckpt_v, "resume_argv": _cur})
        _ckpt_git = str(resume_cfg.get("__cfg_git_sha", "unknown")) if "__cfg_git_sha" in resume_cfg else "unknown"
        _retreatment = bool(getattr(args, "resume_allow_lever_drift", False)
                            or _ws["allow_lever_drift"]
                            or getattr(args, "warm_start_weights_only", False))
        if _stiff_added:
            print(json.dumps({"stage": "resume_stiff_term_drift", "added": _stiff_added,
                              "ckpt_git_sha": _ckpt_git, "retreatment": _retreatment,
                              "rewarmup_epochs": int(getattr(args, "stage_transition_rewarmup_epochs", 0) or 0),
                              "note": "(C11) stiff loss term(s) ADDED/raised at resume onto an opt state "
                              "trained WITHOUT them (H5-F3 level shift). LR re-warmup routes the entry "
                              "when --stage-transition-rewarmup-epochs>0; per-term weight ramp NOT applied "
                              "(needs base-schedule machinery -- see launcher)."}), flush=True)
            if _retreatment and int(getattr(args, "stage_transition_rewarmup_epochs", 0) or 0) > 0:
                _resume_lr_rewarmup_boundary = int(_ws["start_epoch"])
        ema_src = rs["ema"] if rs["ema"] else rs["live"]
        for k in list(ema.shadow.keys()):
            if k in ema_src:
                ema.shadow[k] = mx.array(ema_src[k])
        mx.eval(list(ema.shadow.values()))
        # (DE#3) start epoch: _ws["start_epoch"] == int(rs["epoch"])+1 unless --warm-start-weights-only
        # + --warm-start-epoch overrode it (e.g. 126 to continue just past the ep125 BEST verdict when
        # the sidecar's __resume_epoch is the later DEADLOCK epoch). Non-warm-start => byte-identical.
        if _ws["start_epoch"] != int(rs["epoch"]) + 1:
            print(json.dumps({"stage": "warm_start_epoch_override",
                              "from": int(rs["epoch"]) + 1, "to": _ws["start_epoch"]}), flush=True)
        start_epoch = _ws["start_epoch"]
        # (C16 confound fix) --seed-anneal-epochs is ABSOLUTE; if the resume START epoch is >= the
        # anneal length the seed compose crutch is fully withdrawn before this run begins. Warn with
        # the REAL start_epoch (the post-parse warn only saw --warm-start-epoch). NOTE: the compose
        # math is intentionally NOT reinterpreted relative to start_epoch here -- that would break a
        # bit-faithful continuation resume; set --seed-anneal-epochs relative to the resume epoch in
        # the launcher if you want the post-resume crutch. Seed-FORMATION losses are unaffected.
        if int(getattr(args, "seed_anneal_epochs", 0)) > 0 and int(start_epoch) >= int(args.seed_anneal_epochs):
            print(json.dumps({"stage": "seed_anneal_epochs_WARN", "seed_anneal_epochs": int(args.seed_anneal_epochs),
                              "resume_start_epoch": int(start_epoch),
                              "msg": "(C16) resume start_epoch >= --seed-anneal-epochs: the seed compose "
                              "crutch is already fully withdrawn (off every post-resume epoch). Make it "
                              "RELATIVE to the resume epoch in the launcher if intended."}), flush=True)
        # #205 MUON-FINISHER RESUME: if the resumed epoch is INSIDE the finisher window, the saved
        # optimizer state is the Muon MultiOptimizer's -> rebuild it HERE (before the restore below)
        # and mark _resume_into_finisher so (a) the state restore keys match and (b) the loop's in-line
        # switch is skipped (muon_switched initializes True). Otherwise a resume-into-finisher would
        # re-init a FRESH optimizer at start_epoch, LOSING the Muon+AdamW momentum accumulated since
        # muon_start_epoch = a NON-bit-identical continuation (the deterministic-repro non-negotiable).
        _resume_into_finisher = (args.muon_start_epoch is not None
                                 and start_epoch > int(args.muon_start_epoch))
        if _resume_into_finisher:
            _mlr = float(args.muon_lr) if args.muon_lr is not None else 0.1 * float(args.lr)
            _malr = float(args.muon_adamw_lr) if args.muon_adamw_lr is not None else 0.1 * float(args.lr)
            _mwd = float(args.muon_weight_decay) if args.muon_weight_decay is not None else float(args.weight_decay)
            # GAP 1 (default-off): rebuild with the SAME cosine schedule the switch block built (anchored
            # on muon_start_epoch -> epochs), so the RESTORED opt.step reproduces the bit-faithful finisher
            # LR. WARM-START (GAP 2) is N/A here: the Muon momentum ('v') is restored from the checkpoint
            # below (there is no live outgoing AdamW). final_frac >= 1.0 (default) => scalar LR => the
            # rebuild is byte-identical to the pre-GAP-1 resume construction.
            _r_final_frac = float(getattr(args, "muon_lr_final_frac", 1.0))
            _r_anneal_steps = 0
            if _r_final_frac < 1.0:
                _r_steps_per_ep = max(1, (P + args.accum_pairs - 1) // args.accum_pairs)
                _r_anneal_steps = max(
                    1, (int(args.epochs) - int(args.muon_start_epoch) + 1) * _r_steps_per_ep
                )
            opt = build_muon_finisher_optimizer(
                muon_lr=_mlr, muon_adamw_lr=_malr, muon_momentum=float(args.muon_momentum),
                muon_weight_decay=_mwd, muon_ns_steps=int(args.muon_ns_steps),
                adamw_weight_decay=float(args.weight_decay),
                muon_lr_final_frac=_r_final_frac, muon_anneal_steps=_r_anneal_steps,
            )
            print(json.dumps({"stage": "resume_muon_rebuild", "start_epoch": start_epoch,
                              "muon_start_epoch": int(args.muon_start_epoch),
                              "muon_lr_final_frac": _r_final_frac,
                              "muon_lr_decay_active": bool(_r_final_frac < 1.0),
                              "muon_anneal_steps": _r_anneal_steps,
                              "note": "resuming INSIDE the Muon finisher; rebuilt MultiOptimizer before "
                              "state restore (bit-faithful finisher continuation)"}), flush=True)
        restored_opt = False
        if rs["has_opt"] and rs["opt"]:
            try:
                opt.init(model.trainable_parameters())
                flat_state = dict(tree_flatten(opt.state))
                for k in list(flat_state.keys()):
                    if k in rs["opt"]:
                        flat_state[k] = mx.array(rs["opt"][k])
                opt.state = tree_unflatten(list(flat_state.items()))
                mx.eval(opt.state)
                restored_opt = True
            except Exception as e:  # best-effort: a fresh AdamW re-warms its moments in a few steps.
                print(json.dumps({"stage": "resume_opt_warn",
                                  "note": f"optimizer-state restore failed ({type(e).__name__}: {e}); "
                                  "continuing with fresh AdamW moments (best-effort)"}), flush=True)
        if use_self_orient:
            ema_np = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
            mag = recompute_self_orient(int8_dequant_params(ema_np))
            _rebuild_cf_mx_cache()
            _rebuild_fine_dir_cache()  # AA fine self-orient (no-op unless --aa-self-orient-fine-mode)
            print(json.dumps({"stage": "resume_reorient", "mean_abs_dir_feat": round(mag, 5)}), flush=True)
        print(json.dumps({"stage": "resume", "from": str(rp), "resumed_epoch": int(rs["epoch"]),
                          "start_epoch": start_epoch, "restored_opt": restored_opt,
                          "resumed_into_finisher": bool(_resume_into_finisher)}), flush=True)

    # baseline verdict (epoch 0, or the resumed epoch) -- reflects any restored weights.
    if os.environ.get("TAC_MEM_PROBE", "0") not in ("", "0", "false", "False"):
        _mm = _mlx_mem_gib(mx)
        print(json.dumps({"stage": "mem_probe", "phase": "before_v0_verdict", "n_pairs": P,
                          "verdict_batch": int(args.verdict_batch), "rss_gib": round(_rss_gib(), 2),
                          "mlx_active_gib": round(_mm["active"], 2)}), flush=True)
    v0 = realized_verdict()
    v0.pop("nucleus_counts", None)  # (#302) baseline verdict: drop per-class counts (float-only row;
    #                                  readiness telemetry begins at the first in-loop verdict, after
    #                                  _evt_state exists). No-op when the nucleus feature is OFF.
    # (SENSE) pop the annulus metrics BEFORE the float-only v0 row spread below. None unless
    # --annulus-telemetry is ON => byte-identical when absent.
    _annulus_v0 = v0.pop("annulus", None)
    if os.environ.get("TAC_MEM_PROBE", "0") not in ("", "0", "false", "False"):
        _mm = _mlx_mem_gib(mx)
        print(json.dumps({"stage": "mem_probe", "phase": "after_v0_verdict", "n_pairs": P,
                          "verdict_batch": int(args.verdict_batch), "rss_gib": round(_rss_gib(), 2),
                          "mlx_active_gib": round(_mm["active"], 2), "d_seg": round(v0["d_seg"], 6),
                          "d_pose": round(v0["d_pose"], 6)}), flush=True)
    blob = quantize_levelset_blob({k: np.asarray(v, np.float32) for k, v in ema.shadow.items()})
    s0 = implied_score_from_verdict(v0["d_seg"], v0["d_pose"], blob["total_quantized_blob_bytes"])
    print(json.dumps({"stage": "verdict", "epoch": start_epoch - 1, **{k: round(v, 6) for k, v in v0.items()},
                      "blob_bytes": blob["total_quantized_blob_bytes"], "implied_S": round(s0, 4),
                      # (C6 confound-fix completeness, review R1) liveness stamp on the PRE-LOOP baseline
                      # verdict: weights_stepped=False + phase="baseline_v0" DISAMBIGUATES this legitimate
                      # pre-training baseline from a frozen-mid-training deadlock row (both otherwise read
                      # as "no training happened"). accepted_frac is not yet defined (no batch ran).
                      "weights_stepped": False, "accepted_frac": None, "frozen_epoch": False,
                      "phase": "baseline_v0",
                      "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "axis": "[macOS-CPU advisory] NON-PROMOTABLE"}), flush=True)
    # (SENSE) companion annulus_convergence row for the pre-loop baseline (opt-in; byte-identical when
    # absent). Pre-loop, single-threaded => no lock needed.
    if _annulus_v0 is not None:
        print(json.dumps(_annulus_convergence_row(_annulus_v0, start_epoch - 1, "baseline_v0")), flush=True)
    history.append({"epoch": start_epoch - 1, **v0, "implied_S": s0})

    # (C3 confound fix) STARTUP regularizer-magnitude log: the raw (PRE-weight) scale of each active
    # level-set regularizer on ONE pair, so the eikonal unit-recalibration is auditable and the
    # launcher/gates can assert a magnitude band. The viscous raw ~2490 was the unit bug that let eik
    # dominate the loss (86-91%); the C3-normalized form is O(1). Pure no-grad; advisory; never read back.
    try:
        _rp0 = int(vpairs[0]) if len(vpairs) else 0
        _phi0_lm = model.sdf(_cf_mx(_rp0), 2 * _rp0 + 0)
        _eik_raw, _len_raw, _ = _eikonal_length_mlx(_phi0_lm, render_h, render_w)
        mx.eval(_eik_raw, _len_raw)
        _reg_mags: dict[str, object] = {
            "eikonal_legacy_raw": round(float(_eik_raw), 6),
            "length_raw": round(float(_len_raw), 8),
            "eikonal_weight": float(getattr(args, "eikonal_weight", 0.0)),
            "length_weight": float(getattr(args, "length_weight", 0.0))}
        if _eik_stab["visco_eps0"] > 0.0:
            _ve0 = _eik_stab["visco_eps"] if _eik_stab["visco_eps"] > 0.0 else _eik_stab["visco_eps0"]
            _gx, _gy, _mxx, _myy, _ = _eikonal_margin_interior_mlx(_phi0_lm, render_h, render_w)
            _gm = mx.sqrt(_gx * _gx + _gy * _gy + 1e-8)
            _rv = _gm - 1.0 - float(_ve0) * (_mxx + _myy)
            _rv_raw = mx.mean(_rv * _rv)
            _rv_norm = _eikonal_visco_mlx(_phi0_lm, render_h, render_w, _ve0)
            mx.eval(_rv_raw, _rv_norm)
            _reg_mags["eikonal_viscous_raw_unnorm"] = round(float(_rv_raw), 4)
            _reg_mags["eikonal_viscous_normalized"] = round(float(_rv_norm), 6)
            _reg_mags["visco_eps"] = float(_ve0)
        print(json.dumps({"stage": "regularizer_magnitudes", "epoch": start_epoch - 1, "pair": _rp0,
                          **_reg_mags,
                          "note": "(C3) raw PRE-weight regularizer scales; the C3-normalized viscous "
                          "form is O(1) so --eikonal-weight means the SAME as for the legacy residual"}),
              flush=True)
    except Exception as _regexc:  # advisory only -- a probe failure must never block training.
        print(json.dumps({"stage": "regularizer_magnitudes_skip",
                          "err": f"{type(_regexc).__name__}: {_regexc}"}), flush=True)

    if lane_w > 0.0:
        print(json.dumps({"stage": "lane_edge", "active": True, "weight": lane_w, "lane_class": lane_cls,
                          "margin_target": lane_tgt, "start_epoch": lane_start,
                          "note": "additive realized lane-class margin hinge (2nd seg forward when "
                          "active; default-off; engages at ep>=start_epoch with spike-guard re-treat)"}), flush=True)
    if msal_w > 0.0:
        print(json.dumps({"stage": "margin_saliency", "active": True, "weight": msal_w, "tau": msal_tau,
                          "target": msal_tgt, "start_epoch": msal_start, "uniward": msal_uni,
                          "uniward_beta": (msal_uni_beta if msal_uni else None),
                          # reachability field ADDED to the JSON ONLY when it is on -> the OFF-path print
                          # (incl. any live --resume) is byte-identical to the pre-reachability telemetry.
                          **({"reachability": True} if (msal_reach and _sR_provider is not None) else {}),
                          "note": "LEVER-4 ALL-CLASS GT-margin-saliency-weighted realized margin hinge "
                          "(generalizes class-1 lane-edge to every inter-class edge; class-agnostic)"}), flush=True)

    # LEVER-5 (per-pair hardness) precompute: per-pair sampling probability for the oversampled extras.
    # Default --hardness-oversample 0.0 => n_extra 0 => order == permutation(P) => byte-identical.
    n_extra = int(round(P * max(args.hardness_oversample, 0.0)))
    hardness_prob = None
    hardness_rng = np.random.default_rng(int(args.seed) + 777)
    if n_extra > 0:
        if args.hardness_weighted and args.hardness_source == "realized":
            # one-time per-pair BASELINE realized d_seg over ALL pairs (frozen-decoder reconstruction
            # quality with init codes). CPU-torch authority path (no GPU contention with the daemon).
            ema_np0 = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
            deploy0 = int8_dequant_params(ema_np0)
            f1_all = [_render_numpy_deploy(deploy0, pi, 1) for pi in range(P)]
            ds_pp = np.asarray(cpu_verdict_d_seg_batch(seg_cpu, f1_all, [gt.lstars[pi] for pi in range(P)]),
                               dtype=np.float64).reshape(-1)
            h = ds_pp
            hsrc = "realized_per_pair_dseg"
        else:
            # $0 cached-GT hardness: per-pair fraction of flip-prone (small-GT-margin) pixels.
            band = float(args.hardness_band)
            h = np.asarray([float(np.mean(np.asarray(gt.margins[pi], np.float32) < band)) for pi in range(P)],
                           dtype=np.float64)
            hsrc = "margin_small_frac"
        h = np.clip(h, 1e-12, None) ** float(args.hardness_power)
        if not args.hardness_weighted:
            h = np.ones_like(h)  # uniform extras (the FAIR same-total-steps A/B baseline)
            hsrc = "uniform_oversample"
        hardness_prob = h / h.sum()
        print(json.dumps({"stage": "hardness", "oversample": float(args.hardness_oversample),
                          "n_extra_per_epoch": n_extra, "weighted": bool(args.hardness_weighted),
                          "source": hsrc, "power": float(args.hardness_power),
                          "hard_easy_spread": round(float(hardness_prob.max() / max(hardness_prob.min(), 1e-12)), 3),
                          "top_pairs": [int(i) for i in np.argsort(-hardness_prob)[:6]]}), flush=True)
    if args.max_bank_freq is not None:
        from tac.boundary_math.lever_b_levelset_generator import stem_nyquist_max_freq_cycles_per_unit
        nyq = stem_nyquist_max_freq_cycles_per_unit(scorer_w=SEG_W)
        print(json.dumps({"stage": "stem_nyquist", "max_bank_freq": float(args.max_bank_freq),
                          "stem_nyquist_cycles_per_unit": nyq, "curvelet_cols_after_cap": int(B.shape[1])}), flush=True)

    # FEED-fm FIX-1: RESTORE the RNG streams NOW -- after hardness_rng is built and the (RNG-free)
    # hardness precompute, before the FIRST epoch's permutation draw. Nothing between the resume
    # load and here advances the global MT19937 or hardness_rng (verdict/precompute are RNG-free), so
    # the next permutation/choice continues the CONTINUOUS stream bit-for-bit. DEFAULT-SAFE: no
    # resume, or a pre-FEED-fm sidecar without __rng_* keys => fresh-seeded RNGs untouched.
    if resume_cfg is not None:
        _rng_restored = _restore_rng_state(resume_cfg, hardness_rng)
        print(json.dumps({"stage": "resume_rng", "np_global_restored": _rng_restored["np_global"],
                          "hardness_restored": _rng_restored["hardness"],
                          "note": ("bit-faithful RNG resume" if _rng_restored["np_global"] else
                                   "pre-FEED-fm sidecar (no RNG state); fresh-seeded RNGs (back-compat)")}),
              flush=True)

    # ---- (#304 item 4) PER-TERM LOSS TELEMETRY. A NO-GRAD RECOMPUTE of total_loss_fn with
    # ``terms_out`` on the logged chunk's pairs -- it reads model/gates/caches and mutates NOTHING
    # (no opt/ema/RNG/recent_losses touch; the loss math has no RNG), so the training trajectory is
    # BIT-IDENTICAL with the telemetry on or off (proven by the n1 CPU A/B in the landing memo).
    # Cadence: TAC_LOSS_TERM_PROBE=1 -> every accum chunk (per-batch, the diagnostic mode);
    # --loss-term-log-every N>0 -> every N chunks; N<0 -> OFF; default 0 -> first chunk of each
    # epoch (the standing per-epoch summary). Cost: one extra serial forward per logged pair.
    _lt_chunks_per_epoch = max(1, (P + args.accum_pairs - 1) // args.accum_pairs)
    _lt_stride = _loss_term_log_stride(
        os.environ.get("TAC_LOSS_TERM_PROBE", "0") not in ("", "0", "false", "False"),
        int(getattr(args, "loss_term_log_every", 0)), _lt_chunks_per_epoch)

    def _loss_terms_for_chunk(chunk, seg_form, eik_w_ep) -> "tuple[dict[str, float], float]":
        """Mean per-term breakdown over the chunk's pairs (mirrors the accum loop's batch_loss =
        mean over chunk). Returns (mean terms dict, mean recomputed total)."""
        agg: dict[str, float] = {}
        tot = 0.0
        for _pn in chunk:
            _pi = int(_pn)
            _oh, _mg = lstar_cache[_pi]
            _d: dict[str, Any] = {}
            _Lp = total_loss_fn(model, _cf_mx(_pi), 2 * _pi + 0, 2 * _pi + 1, _oh, _mg,
                                pose_tgts[_pi], args.w_seg, args.w_pose, args.hinge_weight,
                                args.margin_target_end, seg_form, eik_w_ep, args.length_weight,
                                terms_out=_d)
            mx.eval(_Lp, *list(_d.values()))
            tot += float(_Lp)
            for _k, _v in _d.items():
                agg[_k] = agg.get(_k, 0.0) + float(_v)
        _n = max(len(chunk), 1)
        return {k: v / _n for k, v in agg.items()}, tot / _n

    recent_losses: list[float] = []
    # #205: restore the spike-guard window so --resume-from is bit-faithful across a spike-skip
    # (the median gates step-skipping = part of the trajectory). DEFAULT-SAFE: no resume, or a pre-#205
    # sidecar without __recent_losses => the fresh [] is used (prior behavior). MLX-free.
    if resume_cfg is not None and "__recent_losses" in resume_cfg:
        if bool(getattr(args, "resume_clear_spike_guard", False)) or bool(
                getattr(args, "warm_start_weights_only", False)):
            # DEADLOCK RE-TREAT (explicit, flag-gated): discard the frozen window so the median
            # re-anchors to the post-resume loss level (guard re-arms after the FIRST accepted
            # batch — exposure is 1 batch, not 50). Bit-faithful default path below is unchanged.
            _rl = resume_cfg["__recent_losses"]
            _n_disc = len(_rl) if isinstance(_rl, (list, np.ndarray)) else 1
            print(json.dumps({"stage": "resume_spike_guard", "restored_recent_losses": 0,
                              "cleared_frozen_window_len": int(_n_disc),
                              "note": "--resume-clear-spike-guard: frozen median discarded "
                              "(spike-skip deadlock re-treat); re-seeds from first accepted batch"}),
                  flush=True)
        else:
            _rl = resume_cfg["__recent_losses"]
            recent_losses = [float(x) for x in (_rl if isinstance(_rl, list) else [_rl])]
            print(json.dumps({"stage": "resume_spike_guard",
                              "restored_recent_losses": len(recent_losses)}),
                  flush=True)
    last_ep = start_epoch - 1
    stage_ckpts: list[dict[str, Any]] = []
    # (#292 build-2) EVENT-TRIGGERED CURRICULUM controller state. _evt_on gates the ENTIRE feature:
    # OFF (default; requires --curriculum too) => the loop calls _seg_form_for_epoch(ep, args) UNCHANGED
    # => BYTE-IDENTICAL to the hardcoded schedule (the #205-safe path; #205 runs event-triggered OFF).
    # The state dict tracks the two resolved transition epochs (None until fired) + the current stage's
    # start epoch + its within-stage ep_loss history (consumed by _evt_resolve_seg_form). Deterministic
    # in the seeded loss trajectory; NOT the async d_seg verdict.
    _evt_on = bool(getattr(args, "curriculum_event_triggered", False)) and bool(args.curriculum)
    # (#302) "nucleus_ready" is the MEASURED half of the CE->tau trigger (updated at verdict cadence
    # by _emit_handoff_readiness when --curriculum-nucleus-guard is on). Default True => a guard-OFF
    # run never blocks (plateau/cap alone decide) => byte-identical to the pre-#302 event trigger.
    _evt_state = {"tau": None, "l7": None, "stage_start": int(start_epoch), "losses": [],
                  "nucleus_ready": True}
    if _evt_on:
        _rc = resume_cfg if resume_cfg is not None else {}
        if "__evt_boundary_tau" in _rc:
            # bit-faithful restore: the persisted controller state reproduces the SAME fired epochs.
            _bt = int(_rc["__evt_boundary_tau"])
            _bl = int(_rc["__evt_boundary_l7"])
            _evt_state["tau"] = None if _bt < 0 else _bt
            _evt_state["l7"] = None if _bl < 0 else _bl
            _evt_state["stage_start"] = int(_rc.get("__evt_stage_start", start_epoch))
            _sl = _rc.get("__evt_stage_losses", [])
            _evt_state["losses"] = [float(x) for x in (_sl if isinstance(_sl, list) else [_sl])]
            # (#302) restore nucleus-readiness (default True when the sidecar predates the feature).
            _evt_state["nucleus_ready"] = bool(int(_rc.get("__evt_nucleus_ready", 1)))
            print(json.dumps({"stage": "resume_event_curriculum", "restored_tau": _evt_state["tau"],
                              "restored_l7": _evt_state["l7"], "stage_start": _evt_state["stage_start"],
                              "within_stage_losses": len(_evt_state["losses"])}), flush=True)
        elif start_epoch > 1:
            # SAFE cap-fallback: resuming an event-triggered run whose sidecar predates this feature
            # (or an OFF->ON switch) -- the truncated history cannot re-derive the true fired epochs, so
            # resolve any boundary ALREADY PAST its hardcoded cap to that cap (== the OFF stage at the
            # resume point) and detect convergence fresh from here for any not-yet-passed boundary. This
            # guarantees the CORRECT stage assignment on resume (never a mis-stage); it only forfeits the
            # "fire early" benefit for the pre-resume portion. Honest + deterministic going forward.
            _tc = int(args.tau_softplus_start_epoch)
            _lc = int(args.l7_start_epoch)
            if start_epoch > _tc:
                _evt_state["tau"] = _tc
            if start_epoch > _lc:
                _evt_state["l7"] = _lc
            _evt_state["stage_start"] = int(start_epoch)
            print(json.dumps({"stage": "resume_event_curriculum_cap_fallback",
                              "tau": _evt_state["tau"], "l7": _evt_state["l7"],
                              "note": "no persisted event state; past-cap boundaries pinned to hardcoded caps"}),
                  flush=True)
    # (#292 build-3) CLOSED-LOOP resume restore (ON only; mirrors the __evt_* pattern above). A
    # sidecar with the __cl_* keys reproduces the SAME bump/stop state + verdict history a continuous
    # run would hold => bit-faithful ON-resume. A pre-feature / OFF-written sidecar lacks the keys =>
    # fresh state, deterministic going forward (the build-2 cap-fallback honesty). OFF => skipped.
    # (#302 M1) BOUNDARY RE-ANCHOR: map a TAU-RELATIVE wall-clock lever's epoch into the schedule
    # frame it was calibrated in, so its boundary-relative event tracks the FIRED tau boundary
    # (persistence-warmup completion, seed-anneal withdrawal, analytic-band engage). Gated on BOTH
    # --curriculum-reanchor-levers AND event-triggering ON (else there is no fired boundary to track).
    # Unfired / fired-at-cap => _evt_reanchor_epoch returns ``ep`` unchanged => byte-identical.
    _reanchor_on = bool(getattr(args, "curriculum_reanchor_levers", False))

    def _lever_epoch(ep: int) -> int:
        if not (_reanchor_on and _evt_on):
            return int(ep)
        return _evt_reanchor_epoch(ep, _evt_state.get("tau"), int(args.tau_softplus_start_epoch))

    if _cl_on and resume_cfg is not None:
        _clr = _cl_restore_from_cfg(resume_cfg)
        if _clr is not None:
            _cl_state.update(_clr[0])
            _cl_verdicts[:] = _clr[1]
            print(json.dumps({"stage": "resume_closed_loop", "bumps": _cl_state["bumps"],
                              "bump_add": _cl_state["bump_add"],
                              "post_budget_windows": _cl_state["post_budget_windows"],
                              "stop_epoch": _cl_state["stop_epoch"],
                              "restored_verdicts": len(_cl_verdicts)}), flush=True)
        # (M2 fix) PENDING-VERDICT reconcile: the sidecar was written while an async verdict was in
        # flight => its row is ABSENT from the restored verdicts but its EXACT inputs (the
        # point-in-time snapshot the worker was scoring) were persisted. Recompute it SYNCHRONOUSLY
        # through the SAME _verdict_from_snapshot chunked-CPU-torch path => the row is bit-identical
        # to what the continuous run's thread produced => every post-resume decision consumes the
        # SAME row set the continuous run's decisions do (decide-on-previous bit-faithful resume).
        # One-time resume cost ~= one verdict wall. The pending epoch is strictly greater than every
        # restored row's (it was the LAST scheduled) => appending keeps epoch order. SELF-ORIENT
        # caveat (pre-existing resume contract, NOT introduced here): snap["dir"] is rebuilt from
        # the RESTORED shadow (resume_reorient above) exactly like the training forward itself on
        # any self-orient resume — the reconcile inherits that same fidelity envelope. Shapes are
        # restored against the live shadow's (a size-1 sidecar array flattens through .item()).
        # Fail-loud on any mismatch (a corrupted resume must raise, per NO-FAKE).
        _pend = _cl_pending_from_cfg(resume_cfg)
        if _pend is not None and all(int(v["epoch"]) != int(_pend["epoch"]) for v in _cl_verdicts):
            # NOTE: the reshape stays OUTSIDE the failure guard below — a CORRUPTED sidecar
            # (shape/key mismatch) must still raise, per the NO-FAKE fail-loud contract.
            _pshadow = {k: np.asarray(v, np.float32).reshape(np.asarray(ema.shadow[k]).shape)
                        for k, v in _pend["ema_np"].items()}
            try:
                _pv = _verdict_from_snapshot({
                    "ema_np": _pshadow,
                    "softmax_temp": float(_pend["softmax_temp"]),
                    "hosc_beta": float(_pend["hosc_beta"]),
                    "dir": ({pi: dir_feats_per_pair[pi].copy() for pi in vpairs}
                            if use_self_orient else None)})
            except Exception as exc:
                # (M2-F2 fix, throughput review 2026-07-04) FAIL-SAFE: a pending record whose
                # synchronous recompute FAILS is DROPPED — explicitly, logged, deterministically.
                # The continuous run's failed worker produces NO row (verdict_async_failed) and
                # training continues, so appending nothing here is the decision-history-preserving
                # behavior: resurrecting a partial row, or CRASHING a resume the continuous run
                # survived, would diverge the post-resume row set. Never silent — the row below
                # records the drop + the error; classification proceeds on the restored rows only.
                _pv = None
                print(json.dumps({"stage": "resume_pending_verdict_failed",
                                  "epoch": int(_pend["epoch"]), "action": "dropped",
                                  "err": f"{type(exc).__name__}: {exc}",
                                  "note": "pending-verdict recompute FAILED on resume; row DROPPED "
                                  "deterministically (a failed verdict produces NO row in the "
                                  "continuous run — M2-F2 fail-safe, explicit + never silent)"}),
                      flush=True)
            if _pv is not None:
                _cl_verdicts.append({"epoch": int(_pend["epoch"]), "seg_form": str(_pend["seg_form"]),
                                     "d_seg": float(_pv["d_seg"]), "ep_loss": float(_pend["ep_loss"])})
                print(json.dumps({"stage": "resume_pending_verdict", "epoch": int(_pend["epoch"]),
                                  "seg_form": str(_pend["seg_form"]),
                                  "d_seg": round(float(_pv["d_seg"]), 6),
                                  "d_pose": round(float(_pv["d_pose"]), 6),
                                  "note": "in-flight-at-checkpoint verdict recomputed from the persisted "
                                  "snapshot (M2 decide-on-previous bit-faithful resume)"}), flush=True)
    # CURRICULUM stage-transition spike-guard re-treat tracker (operator 2026-06-26 "different
    # stages need different treatment ... transitions must re-treat"). Init to the START epoch's
    # seg_form so a fresh-start / resume does NOT spuriously re-treat (prev == current at ep0). Under
    # event-triggering, read the current stage from the (possibly restored) controller state instead.
    prev_seg_form = (_evt_current_stage_form(_evt_state) if _evt_on
                     else _seg_form_for_epoch(start_epoch, args))
    # MUON FINISHER (FEED-fi) per-stage optimizer switch state. muon_start_epoch None (default) =>
    # muon_switched stays False forever => the switch block + tag suffix never fire => BIT-IDENTICAL
    # to the pre-FEED-fi AdamW-throughout path. Effective LRs default to 0.1*lr (PR95 ~0.1x finetune).
    # #205: initialize True when resuming INSIDE the finisher (the opt was rebuilt as the Muon
    # MultiOptimizer in the resume block above) so the in-loop switch does NOT re-init a fresh
    # optimizer. Default False (fresh start / pre-finisher resume) => the in-loop switch fires normally.
    muon_switched = bool(_resume_into_finisher)
    # BUILD 1 (FEED-fw): stage-transition treatment tracker. None until a registered AdamW->AdamW
    # boundary fires (curriculum seg-form change / lane-edge engage / margin-saliency engage); the LR
    # re-warmup + (optional) AdamW moment reset key off it. DEFAULT-OFF flags
    # (--stage-transition-rewarmup-epochs 0 + no --stage-transition-reset-moments) => this is set but
    # never consumed => BIT-IDENTICAL. NOT persisted across resume (re-derived; None at resume start
    # => no spurious re-warmup until a real boundary).
    last_boundary_epoch: "int | None" = None
    # (C11) seed the LR re-warmup boundary from a resume stiff-term drift so the added stiff term's
    # entry is ramped (not full-weight step-in on a foreign optimizer state). None unless the resume
    # block detected an added stiff term + retreatment + a configured re-warmup window.
    if _resume_lr_rewarmup_boundary is not None:
        last_boundary_epoch = int(_resume_lr_rewarmup_boundary)
    # (EIK-STAB build 2; sweep lever #3 + #304) spike-guard MODE dispatch. "legacy" =>
    # _sg_guard is None => every guard branch below is skipped and _sg_state["lr_scale"] stays 1.0
    # (never multiplied in) => BYTE-IDENTICAL to the pre-build trainer (selectable for the A/B but
    # NO LONGER the default — the C1 confound fix made "rollback" the argparse default AND the
    # getattr fallback below, so the deadlock mode can never re-enter via a missing attr). "rollback"
    # = the physics-informed actuator: tolerate bounded oscillation (EoS self-stabilization is
    # FUNCTIONAL — litsweep contradiction row 3), and on SUSTAINED runaway restore the last-good
    # snapshot + cut lr + re-arm a fresh median (fight the disease, don't freeze).
    _sg_mode = str(getattr(args, "spike_guard_mode", "rollback"))
    _sg_guard = (SpikeGuardRollback(int(args.spike_rollback_window),
                                    float(args.spike_rollback_frac),
                                    int(args.spike_rollback_max))
                 if _sg_mode == "rollback" else None)
    _sg_state: dict[str, Any] = {"lr_scale": 1.0, "snap": None, "snap_epoch": None,
                                 "ep_spikes": 0, "ep_batches": 0, "exhausted_warned": False}

    # ── (C5+C6 confound fix 2026-07-05) RUN-LEVEL LIVENESS + typed confound alarms ──────────────────
    # The #1 self-protect (operator "meta confounds"): stamp a LIVENESS signal (accepted-batch
    # fraction) onto EVERY verdict / loss_terms / closed_loop / eik_stabilizer row so no reader or
    # controller can mistake a FROZEN (all-skip) run for a converging one. `ep_acc`/`ep_tot` are the
    # RUNNING counters for the CURRENT epoch (reset at epoch top); `frac`/`stepped`/`acc`/`skip` snapshot
    # the LAST COMPLETED epoch (what epoch-top rows read). All default to "alive" so a fresh run's first
    # rows are not spuriously flagged.
    _live: dict[str, Any] = {"ep_acc": 0, "ep_tot": 0, "frac": 1.0, "stepped": True, "acc": 0, "skip": 0}

    def _live_running_frac() -> float:
        """Accepted-batch fraction SO FAR this epoch (for mid-epoch loss_terms rows)."""
        return float(_live["ep_acc"]) / float(max(int(_live["ep_tot"]), 1))

    # typed confound-alarm streak state (turn the silent confounds LOUD; non-halting unless egregious).
    _alarm: dict[str, Any] = {"deadlock_streak": 0, "termdom_streak": 0, "gnorm_streak": 0,
                              "adaptive_inert_since": None, "adaptive_last_eps": None}
    _CL_LIVENESS_MIN_ACCEPTED_FRAC = 0.10  # (C5) below this accepted-frac the run is FROZEN, never "converging"
    _TERMDOM_FRAC = 0.40      # (C6) a single reg term > 40% of loss => domination
    _TERMDOM_MIN_ROWS = 3     # (C6) sustained for >= this many loss_terms rows
    _GNORM_HIJACK_MULT = 100.0   # (C6) gnorm > 100x grad_clip
    _GNORM_HIJACK_MIN_BATCHES = 3
    _ADAPTIVE_INERT_EP = 20   # (C6) visco_eps pinned at floor with 0 change-events for > this many ep

    def _emit_confound_alarm(kind: str, **fields: Any) -> None:
        """(C6) Emit a typed confound_alarm telemetry row. Advisory (never halts training); the
        launcher/gates + operator dashboard read these to catch a silent confound going loud."""
        print(json.dumps({"stage": "confound_alarm", "alarm": str(kind), **fields}), flush=True)

    def _sg_take_snapshot(ep: int) -> None:
        """Reference-snapshot of the last-good training state (model + EMA + opt [+ seed]).
        ZERO-COPY: MLX arrays are immutable and every consumer (opt.update / ema.update /
        model.update / attribute rebinds) REBINDS leaves rather than mutating them, so holding
        the current array objects is a faithful point-in-time snapshot (same guarantee the
        resume-sidecar save path relies on)."""
        snap: dict[str, Any] = {
            "live": dict(tree_flatten(model.parameters())),
            "ema": dict(ema.shadow),
            "opt": dict(tree_flatten(opt.state)),
        }
        if seed_mod is not None:
            snap["seed"] = dict(tree_flatten(seed_mod.parameters()))
            snap["seed_opt"] = dict(tree_flatten(seed_opt.state))
        _sg_state["snap"] = snap
        _sg_state["snap_epoch"] = int(ep)

    def _sg_do_rollback(ep: int, batch_loss: float, gnorm: float) -> None:
        """SUSTAINED-runaway response: restore last-good weights/EMA/opt, cut lr x lr_cut,
        clear the spike-guard median window (fresh re-arm). The restored opt state carries the
        snapshot's Adam moments (measured: restored moments DAMP the runaway 6.7x vs fresh 25.3x
        — never reset them) and the snapshot's step count (consistent bias-correction state)."""
        snap = _sg_state["snap"]
        cut = float(args.spike_rollback_lr_cut)
        cur_lr = float(opt.learning_rate)
        model.update(tree_unflatten(list(snap["live"].items())))
        mx.eval(model.parameters())
        for _k, _v in snap["ema"].items():
            ema.shadow[_k] = _v
        mx.eval(list(ema.shadow.values()))
        opt.state = tree_unflatten(list(snap["opt"].items()))
        new_lr = cur_lr * cut
        opt.learning_rate = new_lr        # AFTER the state restore (state carries learning_rate)
        mx.eval(opt.state)
        if seed_mod is not None and "seed" in snap:
            seed_mod.update(tree_unflatten(list(snap["seed"].items())))
            seed_opt.state = tree_unflatten(list(snap["seed_opt"].items()))
            mx.eval(seed_mod.parameters(), seed_opt.state)
        _sg_state["lr_scale"] *= cut      # persists through the per-epoch scheduled-lr assignment
        recent_losses.clear()             # fresh median re-arm (the frozen-median deadlock killer)
        print(json.dumps({"stage": "spike_rollback", "ep": int(ep),
                          "restored_from_epoch": _sg_state["snap_epoch"],
                          "batch_loss": (round(float(batch_loss), 4) if np.isfinite(batch_loss) else "nonfinite"),
                          "gnorm": (round(float(gnorm), 4) if np.isfinite(gnorm) else "nonfinite"),
                          "lr_before": round(cur_lr, 8), "lr_after": round(new_lr, 8),
                          "lr_scale": round(float(_sg_state["lr_scale"]), 6),
                          "rollbacks_used": int(_sg_guard.rollbacks),
                          "rollbacks_max": int(_sg_guard.max_rollbacks),
                          "note": "sustained runaway -> restore last-good weights/EMA/opt (moments "
                          "RESTORED not reset), lr cut, median re-armed (EIK-STAB build 2)"}),
              flush=True)
    # (review C2) anneal SCHEDULE length: --anneal-epochs decouples the cosine denominator (the
    # schedule the temp/LR were designed against) from --epochs (this run's length). Default None =>
    # args.epochs => the LR cosine below is BIT-IDENTICAL. A warm-start arm sets it to the ORIGINAL
    # schedule (e.g. 1500) so resuming the CE ckpt @ ep299 reproduces the DISEASE regime, not the tail.
    anneal_epochs = int(args.anneal_epochs) if getattr(args, "anneal_epochs", None) else int(args.epochs)
    muon_lr_eff = float(args.muon_lr) if args.muon_lr is not None else 0.1 * float(args.lr)
    muon_adamw_lr_eff = float(args.muon_adamw_lr) if args.muon_adamw_lr is not None else 0.1 * float(args.lr)
    muon_wd_eff = float(args.muon_weight_decay) if args.muon_weight_decay is not None else float(args.weight_decay)
    # (#205 OOM instrumentation) env-gated per-accum-batch memory telemetry. Default OFF -> no
    # per-batch prints in production; set TAC_MEM_PROBE=1 to trace active/cache/peak/RSS for the
    # first TAC_MEM_PROBE_EPOCHS epochs (the OOM-diagnosis + fix-verification A/B). Pure observability
    # -> BIT-IDENTICAL training whether on or off.
    _mem_probe_on = os.environ.get("TAC_MEM_PROBE", "0") not in ("", "0", "false", "False")
    _mem_probe_epochs = int(os.environ.get("TAC_MEM_PROBE_EPOCHS", "3"))
    # ── compute-facet #252 activation (DEFAULT-OFF => byte-identical to the pre-#252 path) ──
    _profile_timing = bool(getattr(args, "profile_timing", False))
    set_fused_r_kernel(bool(getattr(args, "fused_r_kernel", False)))
    with temporary_mlx_device(args.mlx_device):
        if getattr(args, "fused_r_kernel", False):
            if args.mlx_device != "gpu":
                raise ValueError("--fused-r-kernel requires --mlx-device gpu (the fused R is a Metal kernel).")
            from tac.local_acceleration.metal_fused_r_operator import assert_metal_matches_cpu_oracle
            _fr_gate = assert_metal_matches_cpu_oracle()  # per-chip parity: FAILS CLOSED if not bit-identical
            print(json.dumps({"stage": "fused_r_kernel", "active": True,
                              "forward_bit_identical": bool(_fr_gate["forward_bit_identical"]),
                              "grad_bit_identical": bool(_fr_gate["grad_bit_identical"]),
                              "note": "fused Metal R roundtrip active; per-chip parity gate PASSED; buys "
                              "SPEED not score (verdict stays numpy/torch-CPU authority); pointer 0.19110 UNMOVED"}),
                  flush=True)
        _mxc = maybe_enable_mx_compile_r(bool(getattr(args, "mx_compile", False)), render_hw=(render_h, render_w))
        if _mxc:
            print(json.dumps({"stage": "mx_compile_r", "active": True, **_mxc,
                              "note": "mx.compile'd R installed (startup bit-identity gate PASSED)"}), flush=True)
        # ══ (EIK-STAB build 4; sweep lever #1) lambda_pre HVP PROBE — measure the Adam-
        # PRECONDITIONED sharpness lambda_max(P^-1/2 H P^-1/2) at the (resumed) start state, print
        # JSON rows, and EXIT before ANY training step. Default --lambda-pre-probe-iters 0 => this
        # whole block is skipped => byte-identical. Method: preconditioned power iteration with
        # FORWARD-DIFFERENCE HVPs over the FULL P-pair batch gradient (H u ~= (g(th+h*u)-g(th))/h),
        # fp64 accumulation in numpy; a final CENTRAL-difference consistency check validates the
        # converged eigenvector. Preconditioner matches the LIVE MLX AdamW exactly: P = diag(
        # sqrt(v_hat)+eps) with v from the RESTORED optimizer state (v_hat = v/(1-beta2^step) only
        # when bias_correction is on, mirroring mlx.optimizers.Adam), so lambda_pre is the quantity
        # the Adam-EoS threshold law lambda_pre* ~= 2(1+b1)/(1-b1)/eta = 38/eta (b1=0.9; Cohen et
        # al. arXiv 2207.14484) actually bounds. Tests the litsweep DOMAIN-2 prediction
        # lambda_pre in [4.2e4, 7.6e4] (= 38/eta at the measured bracket [5e-4 stable, 9.1e-4
        # unstable]) on the ep100 snapshot. ══
        if int(getattr(args, "lambda_pre_probe_iters", 0)) > 0:
            _lp_iters = int(args.lambda_pre_probe_iters)
            _lp_fd = float(args.lambda_pre_probe_fd_eps)
            _lp_ep = int(start_epoch)
            # mirror the loop's per-epoch schedule application for the start epoch (so the loss
            # landscape probed IS the one the first post-resume step would see).
            model.softmax_temp = _softmax_temp_for_epoch(_lp_ep, args)
            _lp_beta = _hosc_beta_for_epoch(_lp_ep, args)
            if _lp_beta is not None:
                model.hosc_beta = _lp_beta
            _lp_seg_form = _seg_form_for_epoch(_lp_ep, args)
            _lp_eik_w = _scheduled_eikonal_weight(_lp_ep, args)
            if args.lr_schedule:
                if _lp_ep <= args.warmup_epochs:
                    _lp_eta = float(args.lr) * _lp_ep / max(args.warmup_epochs, 1)
                else:
                    _lp_prog = (_lp_ep - args.warmup_epochs) / max(anneal_epochs - args.warmup_epochs, 1)
                    _lp_eta = float(args.lr_end + 0.5 * (args.lr - args.lr_end)
                                    * (1 + np.cos(np.pi * _lp_prog)))
            else:
                _lp_eta = float(args.lr)
            # optimizer state: need the v (2nd-moment) tree. Restored resume state already ran
            # opt.init + restore; a fresh run has an UNINITIALIZED state -> init to zeros (the probe
            # then reports moments_norm=0 and the preconditioner is the eps floor — a WARNED,
            # near-meaningless lambda_pre; the probe's design point is RESTORED moments).
            _lp_state_flat = dict(tree_flatten(opt.state))
            if not any(k.endswith(".v") for k in _lp_state_flat):
                opt.init(model.trainable_parameters())
                _lp_state_flat = dict(tree_flatten(opt.state))
            _lp_theta_mx = dict(tree_flatten(model.trainable_parameters()))
            _lp_keys = sorted(_lp_theta_mx.keys())
            _lp_shapes = {k: tuple(np.asarray(_lp_theta_mx[k]).shape) for k in _lp_keys}
            _lp_theta = {k: np.asarray(_lp_theta_mx[k], np.float64) for k in _lp_keys}

            def _lp_vec(tree_np: dict) -> np.ndarray:
                return np.concatenate([np.ravel(np.asarray(tree_np[k], np.float64)) for k in _lp_keys])

            def _lp_unvec_to_model(vec: np.ndarray) -> None:
                out, off = [], 0
                for k in _lp_keys:
                    n = int(np.prod(_lp_shapes[k])) if _lp_shapes[k] else 1
                    out.append((k, mx.array(vec[off:off + n].reshape(_lp_shapes[k]).astype(np.float32))))
                    off += n
                model.update(tree_unflatten(out))
                mx.eval(model.parameters())

            def _lp_full_grad() -> np.ndarray:
                acc = None
                for _pi in range(P):
                    _oh, _mg = lstar_cache[_pi]
                    _, _grads = value_and_grad(
                        model, _cf_mx(_pi), 2 * _pi + 0, 2 * _pi + 1, _oh, _mg, pose_tgts[_pi],
                        args.w_seg, args.w_pose, args.hinge_weight, args.margin_target_end,
                        _lp_seg_form, _lp_eik_w, args.length_weight)
                    mx.eval(_grads)
                    acc = _grads if acc is None else tree_map(lambda a, b: a + b, acc, _grads)
                    mx.eval(acc)
                flat = dict(tree_flatten(acc))
                return np.concatenate([np.ravel(np.asarray(flat[k], np.float64)) for k in _lp_keys]) / float(P)

            # Adam preconditioner diag, matching mlx.optimizers.Adam.apply_single EXACTLY:
            # denom = sqrt(v_hat) + eps; v_hat = v/(1-beta2^step) iff bias_correction else v.
            _lp_eps_adam = float(getattr(opt, "eps", 1e-8))
            _lp_b2 = float(getattr(opt, "betas", [0.9, 0.999])[1])
            _lp_bc = bool(getattr(opt, "bias_correction", False))
            _lp_step = float(np.asarray(_lp_state_flat.get("step", 0)))
            _lp_vparts = []
            for k in _lp_keys:
                _v = _lp_state_flat.get(k + ".v")
                _va = (np.zeros(int(np.prod(_lp_shapes[k])) if _lp_shapes[k] else 1, np.float64)
                       if _v is None else np.ravel(np.asarray(_v, np.float64)))
                _lp_vparts.append(_va)
            _lp_v = np.concatenate(_lp_vparts)
            if _lp_bc and _lp_step > 0:
                _lp_v = _lp_v / (1.0 - _lp_b2 ** _lp_step)
            _lp_denom = np.sqrt(np.maximum(_lp_v, 0.0)) + _lp_eps_adam     # = P diag
            _lp_d = np.sqrt(_lp_denom)                                     # = P^{1/2} diag
            _lp_theta_vec = _lp_vec(_lp_theta)
            _lp_theta_norm = float(np.linalg.norm(_lp_theta_vec))
            print(json.dumps({"stage": "lambda_pre_probe_start", "epoch": _lp_ep, "n_pairs": P,
                              "iters": _lp_iters, "fd_eps": _lp_fd, "eta": _lp_eta,
                              "adam_eps": _lp_eps_adam, "beta2": _lp_b2,
                              "bias_correction": _lp_bc, "opt_step": _lp_step,
                              "dim": int(_lp_theta_vec.size),
                              "v_norm": float(np.linalg.norm(_lp_v)),
                              "moments_restored": bool(np.linalg.norm(_lp_v) > 0.0),
                              "axis": "[n24 advisory -- mechanism probe, NOT n600 evidence]"}),
                  flush=True)
            _lp_g0 = _lp_full_grad()
            _lp_rng = np.random.default_rng(1234)
            _lp_w = _lp_rng.standard_normal(_lp_theta_vec.size)
            _lp_w /= np.linalg.norm(_lp_w)
            _lp_lam = float("nan")
            for _it in range(_lp_iters):
                _u = _lp_w / _lp_d
                _h = _lp_fd * (1.0 + _lp_theta_norm) / max(float(np.linalg.norm(_u)), 1e-20)
                _lp_unvec_to_model(_lp_theta_vec + _h * _u)
                _g1 = _lp_full_grad()
                _lp_unvec_to_model(_lp_theta_vec)          # restore
                _Hu = (_g1 - _lp_g0) / _h
                _r = _Hu / _lp_d
                _lp_lam = float(np.dot(_lp_w, _r))
                _rn = float(np.linalg.norm(_r))
                print(json.dumps({"stage": "lambda_pre_iter", "iter": _it,
                                  "lambda_pre": _lp_lam, "residual_norm": _rn}), flush=True)
                if _rn <= 1e-30:
                    break
                _lp_w = _r / _rn
            # central-difference consistency check on the converged direction (validates the
            # forward-difference HVP: |lam_fwd - lam_central| / |lam_central| should be small).
            _u = _lp_w / _lp_d
            _h = _lp_fd * (1.0 + _lp_theta_norm) / max(float(np.linalg.norm(_u)), 1e-20)
            _lp_unvec_to_model(_lp_theta_vec + _h * _u)
            _gp = _lp_full_grad()
            _lp_unvec_to_model(_lp_theta_vec - _h * _u)
            _gm = _lp_full_grad()
            _lp_unvec_to_model(_lp_theta_vec)
            _lam_c = float(np.dot(_lp_w, ((_gp - _gm) / (2.0 * _h)) / _lp_d))
            _lp_bracket = [4.2e4, 7.6e4]   # 38/eta at the MEASURED lr bracket [9.1e-4 unstable, 5e-4 stable]
            _lp_report = {
                "stage": "lambda_pre", "epoch": _lp_ep, "n_pairs": P,
                "lambda_pre": _lp_lam, "lambda_pre_central_check": _lam_c,
                "fwd_vs_central_rel": (abs(_lp_lam - _lam_c) / abs(_lam_c) if _lam_c else None),
                "eta": _lp_eta, "pi_eos": _lp_eta * _lp_lam / 38.0,
                "eta_max_from_law": (38.0 / _lp_lam if _lp_lam > 0 else None),
                "bracket_38_over_eta": _lp_bracket,
                "in_window": bool(_lp_bracket[0] <= _lp_lam <= _lp_bracket[1]),
                "law": "eos_adam_preconditioned_threshold_v1 (FORMALIZATION_PENDING): stability iff "
                       "eta*lambda_pre <~ 2(1+b1)/(1-b1) = 38 at b1=0.9",
                "axis": "[n24 advisory -- mechanism probe, NOT n600 evidence]",
                "pointer": "0.19110 UNMOVED",
            }
            print(json.dumps(_lp_report), flush=True)
            _atomic_write_json(out_dir / "lambda_pre_probe.json", _lp_report)
            raise SystemExit(0)   # probe mode: NO training steps (default-OFF flag => unreachable)
        for ep in range(start_epoch, args.epochs + 1):
            _prof = {"ep_start": time.perf_counter(), "step_s": 0.0, "verdict_s": 0.0} if _profile_timing else None
            if _mem_probe_on and args.mlx_device == "gpu":
                try:
                    mx.reset_peak_memory()  # per-epoch high-water so mem_probe peak is this-epoch scoped
                except Exception:
                    pass
            # (#292 build-2) EVENT-TRIGGERED CURRICULUM: when ON, the stage form + boundary resolution
            # come from the deterministic loss-plateau controller (caps = the hardcoded epochs). When OFF
            # (default), this is the ORIGINAL hardcoded call, UNCHANGED => byte-identical (the #205 path).
            if _evt_on:
                seg_form, _evt_event = _evt_resolve_seg_form(ep, _evt_state, args)
                if _evt_event is not None:
                    print(json.dumps(_evt_event), flush=True)
            else:
                seg_form = _seg_form_for_epoch(ep, args)
                _evt_event = None
            # (#292 SEAL fix) With event-triggering ON the eikonal STEP tracks the RESOLVED tau
            # boundary (the ACTUAL MCF onset), not the hardcoded cap — else an early-fired tau runs
            # MCF at base eikonal until the cap (the survival window the ramp protects). Unfired ->
            # large sentinel -> base (still CE). OFF -> original call, BYTE-IDENTICAL (#205 path).
            if _evt_on:
                _eik_step_ep = _evt_state["tau"] if _evt_state["tau"] is not None else (1 << 30)
                eik_w_ep = _scheduled_eikonal_weight(ep, args, step_epoch=_eik_step_ep)
            else:
                eik_w_ep = _scheduled_eikonal_weight(ep, args)   # (#292) eikonal STEP-ramp; base if --eikonal-weight-end unset (BYTE-IDENTICAL)
            # (#292 build-3) closed-loop BOUNDED bump composes ON TOP of the build-1 schedule:
            # eff = min(scheduled + bump_add, max(--closed-loop-eikonal-max, scheduled)). Guarded so
            # OFF (or ON with no bump fired) leaves eik_w_ep EXACTLY _scheduled_eikonal_weight
            # (the byte-identity contract; #205 runs closed-loop OFF).
            if _cl_on and _cl_state["bump_add"] > 0.0:
                eik_w_ep = _cl_effective_eikonal(eik_w_ep, float(_cl_state["bump_add"]),
                                                 float(args.closed_loop_eikonal_max))
            # BUILD 1 (FEED-fw): detect an AdamW->AdamW stage boundary at THIS epoch BEFORE the
            # existing transition blocks mutate prev_seg_form / lane_gate / msal_gate. Consumed below
            # (after the Muon block, so muon_switched is current) to register the LR re-warmup anchor
            # + optionally reset the AdamW moments. The Muon switch is intentionally EXCLUDED (it
            # already re-treats with a fresh optimizer per FEED-fi, and the base LR schedule is frozen
            # during the finisher). DEFAULT-OFF flags => these booleans are computed but never
            # consumed => BIT-IDENTICAL (pure-python reads, no MLX/model touch).
            _bnd_curriculum = (seg_form != prev_seg_form)
            _bnd_lane = (lane_w > 0.0 and (ep >= lane_start) and not lane_gate["on"])
            _bnd_msal = (msal_w > 0.0 and (ep >= msal_start) and not msal_gate["on"])
            # (review R3-M1) LEVER-B thin-lane engagement is ALSO an AdamW->AdamW treatment boundary
            # (mirrors _bnd_lane/_bnd_msal). Default lane_thin_w=0.0 => never fires => bit-identical.
            _bnd_lane_thin = (lane_thin_w > 0.0 and (ep >= lane_thin_start) and not lane_thin_gate["on"])
            # LEVER-4b sub-pixel boundary engagement is ALSO an AdamW->AdamW treatment boundary (mirrors
            # _bnd_lane/_bnd_msal). Default subpix_w=0.0 => never fires => bit-identical.
            _bnd_subpix = (subpix_w > 0.0 and (ep >= subpix_start) and not subpix_gate["on"])
            # LEVER-4c chroma-sharpening engagement is ALSO an AdamW->AdamW treatment boundary (mirrors
            # _bnd_subpix). Default chroma_bnd_w=0.0 => never fires => bit-identical.
            _bnd_chroma = (chroma_bnd_w > 0.0 and (ep >= chroma_bnd_start) and not chroma_bnd_gate["on"])
            # (F3 fix) #224 analytic-lane render-band engagement is ALSO an AdamW->AdamW treatment
            # boundary (the band's render-target CHANGES at --lane-band-start-epoch): its sibling levers
            # (lane/margin/thin) already OR into _stage_boundary_now, but the band did NOT, so the
            # LR re-warmup + optional moment-reset never fired on band engagement -> stale AdamW momentum
            # pushed through the render-target change. Mirrors _bnd_lane exactly (computed BEFORE the band
            # gate flips at the engage block below). Default --lane-render-band OFF => _band_active False
            # => never fires => bit-identical.
            # (#302 M1) band engage is TAU-RELATIVE (band@350 = tau@300 + 50); re-anchor to the fired
            # tau via _lever_epoch (byte-identical when re-anchor OFF: _lever_epoch(ep) == ep).
            _bnd_band = (_band_active and (_lever_epoch(ep) >= _band_start) and not band_gate["on"])
            _stage_boundary_now = (_bnd_curriculum or _bnd_lane or _bnd_msal or _bnd_lane_thin
                                   or _bnd_band or _bnd_subpix or _bnd_chroma)
            # CURRICULUM stage-transition RE-TREAT (operator 2026-06-26 "transitions must re-treat";
            # PR95-8-stage generalized). The seg LOSS FORM change (ce -> tau_softplus -> l7_softplus)
            # is a per-stage treatment boundary; clear the spike-guard running median so the new
            # stage's loss scale is NOT judged against the prior stage's median (the named "stage
            # inheriting base-stage treatment" failure). The l7 weight is mean-1-renormalized so the
            # scale jump is small in THIS loss design, but the discipline is binding regardless of
            # carrier. Additive: non-curriculum runs have a constant seg_form => prev == current =>
            # NEVER clears => byte-identical. Non-finite guards are unaffected (still always armed).
            if seg_form != prev_seg_form:
                recent_losses.clear()
                print(json.dumps({"stage": "curriculum_transition", "epoch": ep,
                                  "from_seg_form": prev_seg_form, "to_seg_form": seg_form,
                                  "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
                prev_seg_form = seg_form
            # MUON FINISHER switch (FEED-fi; PR95 stage-8). Fires once at the first epoch >= the
            # start (the >= handles RESUME into the finisher too). DEFAULT-OFF (start is None) =>
            # never fires => byte-identical. The switch is a per-stage TREATMENT boundary (operator
            # 2026-06-26 "transitions must re-treat"): rebuild opt AdamW->MultiOptimizer(Muon 2D
            # weights + AdamW rest), re-init optimizer state, CLEAR the spike-guard (the orthogonalized
            # lower-lr step has a different loss scale; do NOT judge it against the prior AdamW stage's
            # median), and SAVE a PRESERVED stage-encoded ckpt so the Muon-finished decoder is
            # independently byte-closeable + resumable. The Muon momentum re-warms from scratch here
            # (best-effort, like the resume path); the DECODER weights are unchanged at the switch.
            if (args.muon_start_epoch is not None) and (not muon_switched) and (ep >= args.muon_start_epoch):
                n_muon, n_adamw = count_muon_adamw_split(model.trainable_parameters())
                # GAP 2 (default-off): capture the OUTGOING AdamW first-moment (state 'm') BEFORE `opt`
                # is rebound, to warm-start the fresh Muon momentum (state 'v'). Only a plain Adam/AdamW
                # base is transferable; anything else (or the flag off) leaves the Muon at cold zeros.
                _warm_start = bool(getattr(args, "muon_warm_start_momentum", False))
                _old_adam_state = (
                    opt.state if (_warm_start and isinstance(opt, (optim.Adam, optim.AdamW))) else None
                )
                # GAP 1 (default-off): cosine-DECAY the Muon-group LR from muon_lr -> muon_lr*final_frac
                # across the finisher span (muon_start_epoch -> epochs). Anchored on muon_start_epoch (NOT
                # `ep`) so the schedule is deterministic in the config -> a resume rebuilds the SAME
                # schedule. opt_updates_per_epoch == ceil(P / accum_pairs) (one opt.update per accum chunk;
                # spike-skips only shorten it, matching the base trainer's step-count semantics).
                # final_frac >= 1.0 (default) => muon_anneal_steps stays 0 => scalar LR => byte-identical.
                _muon_final_frac = float(getattr(args, "muon_lr_final_frac", 1.0))
                _muon_anneal_steps = 0
                if _muon_final_frac < 1.0:
                    _steps_per_ep = max(1, (P + args.accum_pairs - 1) // args.accum_pairs)
                    _muon_anneal_steps = max(
                        1, (int(args.epochs) - int(args.muon_start_epoch) + 1) * _steps_per_ep
                    )
                opt = build_muon_finisher_optimizer(
                    muon_lr=muon_lr_eff, muon_adamw_lr=muon_adamw_lr_eff,
                    muon_momentum=float(args.muon_momentum), muon_weight_decay=muon_wd_eff,
                    muon_ns_steps=int(args.muon_ns_steps), adamw_weight_decay=float(args.weight_decay),
                    # #224 Wave D (R4 #2): thread the same beta2 as the main AdamW so the finisher
                    # rest-group is consistent (default 0.999 => byte-identical).
                    adamw_beta2=float(getattr(args, "adam_beta2", 0.999)),
                    # GAP 1: default (1.0 / 0) => scalar Muon LR => byte-identical.
                    muon_lr_final_frac=_muon_final_frac, muon_anneal_steps=_muon_anneal_steps,
                )
                opt.init(model.trainable_parameters())
                mx.eval(opt.state)
                # GAP 2 (default-off): seed the fresh Muon child's momentum (v) from the captured AdamW m.
                # The Muon child is opt.optimizers[0] (MultiOptimizer([Muon, AdamW], [filter])); its state
                # flattens to '<path>.v' matching the outgoing AdamW's '<path>.m'. try/except cold-fallback
                # so a mismatch never crashes the run (deterministic-repro: cold zeros is the safe default).
                _warm_seeded = 0
                if _old_adam_state is not None:
                    try:
                        _warm_seeded = _seed_muon_momentum_from_adam(opt.optimizers[0], _old_adam_state)
                    except Exception as _warm_err:  # fall back to cold start; never crash the run
                        _warm_seeded = -1
                        print(json.dumps({
                            "stage": "muon_warm_start_FAILED_cold_fallback", "epoch": ep,
                            "err": str(_warm_err),
                        }), flush=True)
                    mx.eval(opt.state)
                muon_switched = True
                recent_losses.clear()
                print(json.dumps({"stage": "muon_finisher_switch", "epoch": ep,
                                  "muon_start_epoch": int(args.muon_start_epoch), "muon_lr": muon_lr_eff,
                                  "muon_adamw_lr": muon_adamw_lr_eff, "muon_momentum": float(args.muon_momentum),
                                  "muon_ns_steps": int(args.muon_ns_steps), "muon_weight_decay": muon_wd_eff,
                                  "n_muon_params": n_muon, "n_adamw_params": n_adamw,
                                  "muon_lr_final_frac": _muon_final_frac,
                                  "muon_lr_decay_active": bool(_muon_final_frac < 1.0),
                                  "muon_anneal_steps": _muon_anneal_steps,
                                  "muon_warm_start_momentum": _warm_start,
                                  "muon_warm_seeded_leaves": _warm_seeded,
                                  "note": "AdamW->Muon (2D hidden weights; biases/code/heads stay AdamW); "
                                  "spike-guard re-treated; LR schedule frozen for the finisher"}), flush=True)
                if args.stage_checkpoints:
                    _wm = _do_checkpoint(ep, stage_tag="stageMuonStart")
                    stage_ckpts.append(_wm)
                    print(json.dumps({"stage": "checkpoint", "kind": "muon_finisher_start", **_wm}), flush=True)
            # lane-edge engagement gate + transition RE-TREAT (spike-guard reset at the engage epoch
            # so the added margin-hinge term's loss jump is not silently spike-skipped; no-op when
            # lane_start<=1 i.e. the default always-on-from-ep1 path -> zero behavior change).
            if lane_w > 0.0:
                _was_on = lane_gate["on"]
                lane_gate["on"] = lever_gate_on_at_epoch(lane_w, lane_start, ep)
                if lane_gate["on"] and not _was_on:
                    recent_losses.clear()
                    print(json.dumps({"stage": "lane_edge_engage", "epoch": ep, "lane_start": lane_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # LEVER-4 margin-saliency engagement gate + transition RE-TREAT (same discipline as lane).
            if msal_w > 0.0:
                _msal_was = msal_gate["on"]
                msal_gate["on"] = lever_gate_on_at_epoch(msal_w, msal_start, ep)
                if msal_gate["on"] and not _msal_was:
                    recent_losses.clear()
                    print(json.dumps({"stage": "margin_saliency_engage", "epoch": ep, "start": msal_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # LEVER-4b sub-pixel boundary engagement gate + transition RE-TREAT (same discipline as LEVER-4).
            if subpix_w > 0.0:
                _subpix_was = subpix_gate["on"]
                subpix_gate["on"] = lever_gate_on_at_epoch(subpix_w, subpix_start, ep)
                if subpix_gate["on"] and not _subpix_was:
                    recent_losses.clear()
                    print(json.dumps({"stage": "seg_subpix_boundary_engage", "epoch": ep, "start": subpix_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # LEVER-4c chroma-sharpening engagement gate + transition RE-TREAT (same discipline as LEVER-4b).
            if chroma_bnd_w > 0.0:
                _chroma_was = chroma_bnd_gate["on"]
                chroma_bnd_gate["on"] = lever_gate_on_at_epoch(chroma_bnd_w, chroma_bnd_start, ep)
                if chroma_bnd_gate["on"] and not _chroma_was:
                    recent_losses.clear()
                    print(json.dumps({"stage": "seg_chroma_boundary_engage", "epoch": ep, "start": chroma_bnd_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # LEVER-B thin-lane engagement gate + transition RE-TREAT (review R3-M1: the gate was
            # initialized at :lane_thin_gate but NEVER flipped, so --lane-thin-start-epoch > 1 left the
            # gate stuck OFF => the loss branch at `lane_thin_gate["on"]` never fired => a SILENT NO-OP
            # = a FALSE 'thin-lane prior does nothing' verdict). Mirrors the lane/margin-saliency gates.
            # No-op when lane_thin_start<=1 (default-on-from-ep1) => zero behavior change.
            if lane_thin_w > 0.0:
                _lt_was = lane_thin_gate["on"]
                lane_thin_gate["on"] = lever_gate_on_at_epoch(lane_thin_w, lane_thin_start, ep)
                if lane_thin_gate["on"] and not _lt_was:
                    recent_losses.clear()
                    print(json.dumps({"stage": "lane_thin_engage", "epoch": ep, "start": lane_thin_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # #224 (2) analytic-lane render-band engagement gate + transition RE-TREAT (mirrors the
            # lane/margin/thin gates). No-op when --lane-render-band off (band never applies).
            if _band_active:
                _band_was = band_gate["on"]
                band_gate["on"] = _band_start <= _lever_epoch(ep)  # (#302 M1) re-anchor to fired tau
                if band_gate["on"] and not _band_was:
                    recent_losses.clear()
                    print(json.dumps({"stage": "lane_render_band_engage", "epoch": ep, "start": _band_start,
                                      "note": "spike-guard re-treated (recent_losses cleared)"}), flush=True)
            # #224 (4) persistence loss anneal (linear warm-up; coarse->fine). No-op when persist_w=0.
            # (#302 M1) TAU-RELATIVE re-anchor: the warmup was calibrated to COMPLETE at tau@300, so
            # under event-triggering it should complete at the FIRED tau (_lever_epoch; ep when OFF).
            if persist_w > 0.0 and persist_classes:
                persist_gate["w"] = persistence_anneal_weight(_lever_epoch(ep), persist_w, persist_warmup)
            # BUILD #300 (b) island-SEED compose-weight anneal (transfer schedule full->0). Deterministic
            # in ep => RESUME reproduces the same weight (nothing to checkpoint). No spike-guard re-treat:
            # the ramp is SMOOTH (small per-epoch delta), not a discrete engage, so it never trips the
            # jump-detector (unlike the lever-engage boundaries above). No-op unless --seed-anneal-epochs
            # > 0 AND a seed is live => compose_w stays 1.0 => _compose_chain byte-identical.
            # (#302 M1) TAU-RELATIVE re-anchor: the seed crutch is withdrawn BY the tau onset, so under
            # event-triggering it should complete at the FIRED tau (_lever_epoch; ep when re-anchor OFF).
            if seed_on and seed_anneal_epochs > 0:
                seed_state["compose_w"] = seed_compose_weight_at_epoch(
                    seed_anneal_epochs, seed_anneal_shape, _lever_epoch(ep))
            # BUILD 1 (FEED-fw): apply stage-transition TREATMENT for an AdamW->AdamW boundary
            # detected above. Skipped during the Muon finisher (muon_switched True; it re-treats
            # itself + freezes the base LR schedule). The spike-guard re-treat already happened in the
            # blocks above (recent_losses cleared); this adds (1) the LR re-warmup anchor and (2) an
            # OPTIONAL fresh-AdamW moment reset. DEFAULT-OFF: --stage-transition-reset-moments False
            # AND --stage-transition-rewarmup-epochs 0 => only sets last_boundary_epoch (then unused
            # by the gated factor) => BIT-IDENTICAL. The fresh AdamW preserves the current
            # learning_rate; the LR-schedule block below resets it for the epoch anyway. (MLX
            # Optimizer.init only fills MISSING state, so a TRUE moment reset requires a fresh
            # optimizer object -- exactly how the Muon switch resets, FEED-fi.)
            if _stage_boundary_now and not muon_switched:
                last_boundary_epoch = ep
                if args.stage_transition_reset_moments:
                    # #224 Wave C FIX-1: the fresh moment-reset optimizer inherits the SAME bias_correction
                    # gate as the main construction (ON only on the high-beta2 all-levers path). A fresh
                    # AdamW resets step->0, so bias_correction correctly re-warms the reset moments.
                    opt = optim.AdamW(learning_rate=float(opt.learning_rate),
                                      weight_decay=args.weight_decay,
                                      betas=[0.9, float(getattr(args, "adam_beta2", 0.999))],
                                      bias_correction=_adam_bias_correction_for(
                                          getattr(args, "adam_beta2", 0.999)))
                    opt.init(model.trainable_parameters())
                    mx.eval(opt.state)
                    print(json.dumps({"stage": "stage_transition_reset_moments", "epoch": ep,
                                      "from_curriculum": bool(_bnd_curriculum),
                                      "from_lane_engage": bool(_bnd_lane),
                                      "from_margin_saliency_engage": bool(_bnd_msal),
                                      "from_lane_thin_engage": bool(_bnd_lane_thin),
                                      "from_lane_render_band_engage": bool(_bnd_band),
                                      "note": "AdamW m/v zeroed (fresh optimizer); spike-guard already "
                                      "re-treated; stale-momentum-through-landscape-change avoided"}),
                          flush=True)
            # SELF-ORIENT reorient cadence (fixed-point): recompute per-pair directional feats from
            # the EMA deploy argmax every --reorient-every epochs (skip ep1: argmax is random).
            if use_self_orient and ep > 1 and (ep - 1) % max(args.reorient_every, 1) == 0:
                ema_np = {k: np.asarray(v, np.float32) for k, v in ema.shadow.items()}
                mag = recompute_self_orient(int8_dequant_params(ema_np))
                _rebuild_cf_mx_cache()
                _rebuild_fine_dir_cache()  # AA fine self-orient (no-op unless --aa-self-orient-fine-mode)
                print(json.dumps({"stage": "reorient", "epoch": ep, "mean_abs_dir_feat": round(mag, 5)}), flush=True)
            # (config-review #4) ANNEAL softmax-temp hi->lo (cosine): start soft (gradients flow,
            # no RGB-level Gibbs) -> end sharp (the SDF partition pinned). Fixing T=0.1 reintroduces
            # Gibbs at the RGB level per deep-math; anneal like the hosc_beta schedule.
            # FEED-fm FIX-2: FREEZE softmax_temp AND hosc_beta DURING THE MUON FINISHER. At/after the
            # switch (muon_switched True) hold BOTH at their muon-START value -- i.e. the value at
            # epoch == muon_start_epoch (deterministic in muon_start_epoch, NOT the process-local fire
            # epoch, so RESUME-into-finisher reproduces the same frozen target). This mirrors the LR
            # freeze already gated on `not muon_switched` below: the orthogonalized finisher conditions
            # boundary PLACEMENT against a STATIONARY target (clean Eikonal=slope / Muon=placement
            # attribution per FEED-fk). DEFAULT-SAFE: --muon-start-epoch None => muon_switched is
            # always False => _anneal_ep == ep => the _softmax_temp_for_epoch / _hosc_beta_for_epoch
            # calls reproduce the pre-FEED-fm inline formulas exactly => BIT-IDENTICAL.
            _anneal_ep = int(args.muon_start_epoch) if muon_switched else ep
            model.softmax_temp = _softmax_temp_for_epoch(_anneal_ep, args)
            # (FEED-fb) ANNEAL hosc_beta start->end (the step-native L-infinity-optimal lever;
            # beta->inf = step-native tanh(beta*sin)). The model's _act reads self.hosc_beta FRESH
            # each forward, so mutating model.hosc_beta per epoch retunes the activation (exactly how
            # softmax_temp is annealed above). DEFAULT-SAFE: _hosc_beta_for_epoch returns None when
            # --hosc-beta-end is unset (or == --hosc-beta, or activation != hosc) -> model.hosc_beta
            # is NEVER touched => stays at its construction value (== args.hosc_beta) every epoch =>
            # BIT-IDENTICAL to the pre-FEED-fb path (and the finisher freeze is then a no-op too). The
            # verdict/checkpoint/byte-close forwards read float(model.hosc_beta) so realized d_seg is
            # measured (and deploy cfg saved) at the CURRENT beta (NO-FAKE).
            _beta = _hosc_beta_for_epoch(_anneal_ep, args)
            if _beta is not None:
                model.hosc_beta = _beta
            # (EIK-STAB build 1b) vanishing-viscosity anneal: mutate the closure cell so
            # total_loss_fn reads the CURRENT eps live (same pattern as the lever gate dicts).
            # Default eps0 0.0 => never touched => byte-identical. (V6 #320) when
            # --eikonal-viscosity-adaptive is set the LINEAR anneal is SKIPPED here and visco_eps is
            # instead set by the adaptive-eps block BELOW (after the LR is known, so eta(t) is current).
            if _eik_stab["visco_eps0"] > 0.0 and not _eik_stab["visco_adaptive"]:
                _ve = _visco_eps_for_epoch(ep, _eik_stab["visco_eps0"], _eik_stab["visco_anneal"])
                if _ve != _eik_stab["visco_eps"]:
                    print(json.dumps({"stage": "eik_stabilizer", "epoch": ep,
                                      "visco_eps": round(_ve, 6),
                                      "accepted_frac": round(float(_live["frac"]), 4),  # (C6) prev-epoch liveness
                                      "weights_stepped": bool(_live["stepped"]),
                                      "note": "ViscoReg vanishing-viscosity eps annealed (replaces "
                                      "the eikonal residual while eps>0; legacy stencil at eps==0)"}),
                          flush=True)
                _eik_stab["visco_eps"] = _ve
            # (EIK-STAB build 2) last-good snapshot refresh at epoch top: refresh ONLY when the
            # PREVIOUS epoch was healthy (spike fraction strictly below the trigger frac), so a
            # runaway epoch can never overwrite the good basin. First epoch (snap None) => take
            # unconditionally. Legacy mode (_sg_guard None) => skipped => byte-identical.
            if _sg_guard is not None:
                _prev_frac = (_sg_state["ep_spikes"] / float(_sg_state["ep_batches"])
                              if _sg_state["ep_batches"] > 0 else 0.0)
                if _sg_state["snap"] is None or _prev_frac < float(args.spike_rollback_frac):
                    _sg_take_snapshot(ep)
                _sg_state["ep_spikes"] = 0
                _sg_state["ep_batches"] = 0
            # (THETA* TIER-2 MUST-3) SWA / wider-finisher EMA: from the finisher-start epoch onward,
            # widen the EMA decay so the EMA shadow averages over the late oscillation (a flat-basin
            # center). Idempotent per-epoch set (keys off `ep`, not state) => RESUME-safe (ema.decay
            # is not persisted; re-applied on resume into the finisher window). DEFAULT-OFF:
            # ema_finisher_decay None => ema.decay is NEVER touched => the EMA trajectory is
            # BIT-IDENTICAL to the --ema-decay path.
            if (ema_finisher_decay is not None and ema_finisher_start is not None
                    and ep >= ema_finisher_start and ema.decay != ema_finisher_decay):
                _prev_decay = ema.decay
                ema.decay = ema_finisher_decay
                print(json.dumps({"stage": "ema_finisher_widen", "epoch": ep,
                                  "from_decay": float(_prev_decay), "to_decay": float(ema_finisher_decay),
                                  "start_epoch": int(ema_finisher_start),
                                  "note": "SWA-style wider EMA averaging for the finisher (flat-basin "
                                  "center over the late oscillation)"}), flush=True)
            # LR warmup->cosine. Gated OFF once the Muon finisher is active (operator 2026-06-26
            # "different stages need different treatment"): the finisher is a PR95 flat low-LR
            # polish at its own muon_lr/muon_adamw_lr, NOT the base cosine, and the MultiOptimizer's
            # children own their own LRs (setting opt.learning_rate would not reach them). Default
            # (no --muon-start-epoch) => muon_switched False => identical to before (BIT-IDENTICAL).
            if args.lr_schedule and not muon_switched:
                if ep <= args.warmup_epochs:
                    lr = args.lr * ep / max(args.warmup_epochs, 1)
                else:
                    # (review C2) cosine denominator = anneal_epochs (schedule length), NOT args.epochs
                    # (run length). anneal_epochs defaults to args.epochs => BIT-IDENTICAL; a warm-start
                    # arm sets --anneal-epochs to the ORIGINAL schedule so the post-resume LR matches the
                    # disease regime (~0.9*peak at ep300/1500) instead of the run-length tail.
                    prog = (ep - args.warmup_epochs) / max(anneal_epochs - args.warmup_epochs, 1)
                    lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * prog))
                # BUILD 1 (FEED-fw): stage-transition LR re-warmup. DEFAULT-OFF
                # (--stage-transition-rewarmup-epochs 0) => _rw is EXACTLY 1.0 => lr*1.0 == lr =>
                # BIT-IDENTICAL. After a registered AdamW->AdamW boundary, ramp the scheduled LR up
                # from the floor over N epochs so the post-boundary landscape change is not hit at
                # full LR with (possibly reset) momentum (the FEED-ft#3 tau-jump root cause).
                _rw = _stage_rewarmup_factor(
                    ep, last_boundary_epoch, args.stage_transition_rewarmup_epochs,
                    args.stage_transition_rewarmup_floor, args.stage_transition_rewarmup_shape)
                lr = lr * _rw
                # (EIK-STAB build 2) rollback-guard persistent lr cut: fold the accumulated
                # x0.5-per-rollback scale into every scheduled assignment. scale==1.0 (legacy mode,
                # or rollback mode before any rollback) => branch skipped => byte-identical.
                if _sg_state["lr_scale"] != 1.0:
                    lr = lr * _sg_state["lr_scale"]
                opt.learning_rate = float(lr)
            # (V6 #320) ADAPTIVE-eps CFL-edge tracker: eps(t) = clamp(|c_a(t)|*sqrt(eta*lambda_eik/8)
            # *(1+margin), floor, upper). Placed AFTER the LR assignment so eta(t)=opt.learning_rate is
            # the CURRENT epoch's flow time-step and eik_w_ep is the CURRENT lambda_eik. Replaces the
            # linear anneal (skipped above when adaptive). |c_a(t)| = no-grad witness-margin sharpness
            # over a FIXED strided pair subset (witness-only, zero SegNet cost). DEFAULT-OFF
            # (visco_adaptive False) => this whole block is skipped => BYTE-IDENTICAL.
            if _eik_stab["visco_adaptive"] and _eik_stab["visco_eps0"] > 0.0:
                _eta_t = float(opt.learning_rate)
                _ca_t = _measure_ca_mlx(model, _ca_pairs, _cf_mx, render_h, render_w,
                                        band=_eik_stab["visco_ca_band"])
                _ve = _adaptive_visco_eps(_ca_t, _eta_t, float(eik_w_ep),
                                          _eik_stab["visco_margin_factor"],
                                          _eik_stab["visco_eps_floor"], _eik_stab["visco_eps_upper"])
                _eik_stab["visco_c_a"] = _ca_t
                if _ve != _eik_stab["visco_eps"]:
                    print(json.dumps({"stage": "eik_stabilizer_adaptive", "epoch": ep,
                                      "visco_eps": round(_ve, 6), "c_a": round(_ca_t, 6),
                                      "eta": round(_eta_t, 8), "lambda_eik": round(float(eik_w_ep), 6),
                                      "accepted_frac": round(float(_live["frac"]), 4),  # (C6) prev-epoch liveness
                                      "weights_stepped": bool(_live["stepped"]),
                                      "note": "adaptive-eps (C2-reparam) tracks sharpness |c_a| into "
                                      "[floor,upper] via saturating tanh; RESPONSIVE at O(1) |c_a|"}),
                          flush=True)
                _eik_stab["visco_eps"] = _ve
                # (C6) adaptive_eps_INERT alarm: visco_eps PINNED at the floor with 0 change-events for
                # > _ADAPTIVE_INERT_EP epochs = the adaptive wrapper is not adapting (the pre-C2 bug).
                _floor = float(_eik_stab["visco_eps_floor"])
                if abs(float(_ve) - _floor) <= 1e-9:
                    if _alarm["adaptive_last_eps"] is not None and abs(float(_alarm["adaptive_last_eps"]) - _floor) <= 1e-9:
                        if _alarm["adaptive_inert_since"] is None:
                            _alarm["adaptive_inert_since"] = int(ep)
                        elif int(ep) - int(_alarm["adaptive_inert_since"]) == _ADAPTIVE_INERT_EP:
                            _emit_confound_alarm("adaptive_eps_INERT", ep=ep, visco_eps=round(float(_ve), 6),
                                                 floor=_floor, pinned_epochs=int(ep) - int(_alarm["adaptive_inert_since"]),
                                                 c_a=round(float(_ca_t), 6),
                                                 note="adaptive visco_eps pinned at floor with 0 "
                                                 "change-events (the pre-C2 INERT bug); the CFL "
                                                 "reparam should respond at O(1) |c_a| -- investigate")
                    else:
                        _alarm["adaptive_inert_since"] = int(ep)
                else:
                    _alarm["adaptive_inert_since"] = None
                _alarm["adaptive_last_eps"] = float(_ve)
            # LEVER-5: base permutation (every pair >=1 step, never starved) + hardness-allocated
            # extras. n_extra=0 (default) => order == permutation(P) => byte-identical to before.
            order = np.random.permutation(P)
            if n_extra > 0 and hardness_prob is not None:
                extra = hardness_rng.choice(P, size=n_extra, replace=True, p=hardness_prob)
                order = np.random.permutation(np.concatenate([order, extra]))
            ep_loss = 0.0
            _live["ep_acc"] = 0  # (C6 liveness) reset the CURRENT-epoch accepted/total batch counters
            _live["ep_tot"] = 0
            if _prof is not None:
                _prof["_step0"] = time.perf_counter()  # #252 profile: fwd+bwd+opt+ema step start
            for s in range(0, P, args.accum_pairs):
                chunk = order[s:s + args.accum_pairs]
                accum = None
                accum_seed = None   # #224 (5): seed grad accumulator (None unless --seed-islands)
                lsum = 0.0
                if _use_micro_batch:
                    # (--micro-batch-pairs B) sub-batch each accum chunk into B-pair groups; ONE batched
                    # value_and_grad per group. Weight each group's MEAN grad/loss by its pair count so
                    # sum-over-groups / nb == the serial per-pair mean-over-chunk (mean_grads + batch_loss
                    # below are UNCHANGED). (BUILD #293) with --seed-islands the group call is the DUAL
                    # _dual_vg_batch -> (witness grads, seed grads); BOTH legs are group-MEAN grads and
                    # BOTH are weighted by the group's pair count, so accum AND accum_seed preserve the
                    # same invariant (sum-over-groups / nb == serial per-pair mean) and the downstream
                    # seed step (_mean_sg = accum_seed/nb -> shield -> seed_opt) is UNTOUCHED.
                    _B = _micro_batch_pairs
                    for _ss in range(0, len(chunk), _B):
                        _sub = [int(p) for p in chunk[_ss:_ss + _B]]
                        _bn = len(_sub)
                        _vg_args = (
                            [_cf_mx(p) for p in _sub],
                            [2 * p + 0 for p in _sub], [2 * p + 1 for p in _sub],
                            [lstar_cache[p][0] for p in _sub], [lstar_cache[p][1] for p in _sub],
                            [pose_tgts[p] for p in _sub],
                            args.w_seg, args.w_pose, args.hinge_weight, args.margin_target_end, seg_form,
                            eik_w_ep, args.length_weight,
                        )
                        if _dual_vg_batch is None:
                            loss_b, grads_b = value_and_grad_batch(model, *_vg_args)
                        else:
                            loss_b, (grads_b, sgrads_b) = _dual_vg_batch(
                                model.trainable_parameters(), seed_mod.trainable_parameters(), *_vg_args)
                            _wsg = tree_map(lambda g, c=float(_bn): g * c, sgrads_b)  # mean-seed-grad * count
                            accum_seed = _wsg if accum_seed is None else tree_map(lambda a, b: a + b, accum_seed, _wsg)
                            mx.eval(accum_seed)
                        mx.eval(loss_b, grads_b)  # materialize per group (bound the lazy fwd+bwd graph)
                        lsum += float(loss_b) * _bn          # mean-over-group * count = group sum
                        _wg = tree_map(lambda g, c=float(_bn): g * c, grads_b)  # mean-grad * count = group-sum grad
                        accum = _wg if accum is None else tree_map(lambda a, b: a + b, accum, _wg)
                        mx.eval(accum)
                else:
                    for pi_np in chunk:
                        pi = int(pi_np)
                        oh, mg = lstar_cache[pi]
                        if _dual_vg is None:
                            loss, grads = value_and_grad(
                                model, _cf_mx(pi), 2 * pi + 0, 2 * pi + 1, oh, mg, pose_tgts[pi],
                                args.w_seg, args.w_pose, args.hinge_weight, args.margin_target_end, seg_form,
                                eik_w_ep, args.length_weight,
                            )
                        else:
                            # #224 (5) dual co-grad: witness grads[0] (== the single-path grads, same loss/
                            # params) + seed grads[1]. The seed leg is accumulated + shielded separately below.
                            loss, (grads, sgrads) = _dual_vg(
                                model.trainable_parameters(), seed_mod.trainable_parameters(),
                                _cf_mx(pi), 2 * pi + 0, 2 * pi + 1, oh, mg, pose_tgts[pi],
                                args.w_seg, args.w_pose, args.hinge_weight, args.margin_target_end, seg_form,
                                eik_w_ep, args.length_weight,
                            )
                            accum_seed = sgrads if accum_seed is None else tree_map(lambda a, b: a + b, accum_seed, sgrads)
                            mx.eval(accum_seed)
                        mx.eval(loss, grads)  # materialize per pair (bound the lazy fwd+bwd graph)
                        lsum += float(loss)
                        accum = grads if accum is None else tree_map(lambda a, b: a + b, accum, grads)
                        mx.eval(accum)
                nb = max(len(chunk), 1)
                batch_loss = lsum / nb
                mean_grads = tree_map(lambda g, c=float(nb): g / c, accum)
                clipped, total = optim.clip_grad_norm(mean_grads, args.grad_clip if args.grad_clip > 0 else 1e30)
                mx.eval(total)
                gnorm = float(total)
                # (C4 confound fix) PER-GROUP grad clip: when --per-group-grad-clip is ON, re-clip each
                # top-level parameter GROUP to --grad-clip INDEPENDENTLY, so a volatile regularizer
                # gradient dominating the GLOBAL norm cannot throttle the seg/pose gradient on OTHER
                # param groups (film/out_tex) via the shared 1/gnorm scale. `gnorm`/`total` above stay
                # the GLOBAL norm (spike-guard + telemetry reference). DEFAULT OFF => `clipped` above is
                # used unchanged => BYTE-IDENTICAL. No-op when grad_clip<=0 or gnorm nonfinite.
                if (bool(getattr(args, "per_group_grad_clip", False)) and args.grad_clip > 0
                        and np.isfinite(gnorm) and isinstance(mean_grads, dict)):
                    _grp_clipped: dict[str, Any] = {}
                    for _gk, _gsub in mean_grads.items():
                        _gc, _gt = optim.clip_grad_norm(_gsub, args.grad_clip)
                        _grp_clipped[_gk] = _gc
                    clipped = _grp_clipped
                    mx.eval(clipped)
                # spike-guard: skip non-finite / >spike_factor x running median.
                # (EIK-STAB build 2) MODE DISPATCH. legacy (default; _sg_guard None): skip =
                # nonfinite OR spiked — IDENTICAL semantics to the pre-build expression (with a
                # nonfinite loss the original or-chain short-circuited True; _spiked's isfinite
                # guard only avoids a nan-comparison, same result). rollback: tolerate single
                # finite spikes (STEP them — EoS oscillation is the self-stabilization mechanism),
                # skip only nonfinite, and respond to SUSTAINED runaway with rollback+lr-cut+re-arm.
                med = float(np.median(recent_losses)) if recent_losses else None
                _nonfinite = (not np.isfinite(batch_loss)) or (not np.isfinite(gnorm))
                _spiked = bool(med is not None and np.isfinite(batch_loss)
                               and batch_loss > args.spike_factor * med)
                _sg_act = None
                if _sg_guard is not None:
                    _sg_state["ep_batches"] += 1
                    if _spiked or _nonfinite:
                        _sg_state["ep_spikes"] += 1
                    _sg_act = _sg_guard.observe(_spiked or _nonfinite)
                    if _sg_act == "exhausted":
                        if not _sg_state["exhausted_warned"]:
                            print(json.dumps({
                                "stage": "spike_rollback_exhausted", "ep": ep,
                                "rollbacks_used": int(_sg_guard.rollbacks),
                                "note": "rollback budget spent (--spike-rollback-max); guard "
                                "REVERTS to legacy skip semantics from here (bounded actuation)"}),
                                flush=True)
                            _sg_state["exhausted_warned"] = True
                        skip = _nonfinite or _spiked
                    else:
                        skip = _nonfinite or (_sg_act == "rollback")
                else:
                    skip = _nonfinite or _spiked
                # (C6 liveness) count THIS accum-batch; ep_acc incremented only on an accepted step
                # (after opt.update below). Works in ALL modes (legacy + rollback), unlike _sg_state.
                _live["ep_tot"] += 1
                # (C6) gnorm_hijack alarm: a global grad-norm >> the clip budget means one (volatile)
                # gradient is scaling the WHOLE step down (starving the others). Sustained => loud.
                if np.isfinite(gnorm) and args.grad_clip > 0 and gnorm > _GNORM_HIJACK_MULT * args.grad_clip:
                    _alarm["gnorm_streak"] += 1
                    if _alarm["gnorm_streak"] == _GNORM_HIJACK_MIN_BATCHES:
                        _emit_confound_alarm("gnorm_hijack", ep=ep, gnorm=round(float(gnorm), 2),
                                             grad_clip=float(args.grad_clip),
                                             ratio=round(float(gnorm) / float(args.grad_clip), 1),
                                             sustained_batches=int(_alarm["gnorm_streak"]),
                                             per_group_grad_clip=bool(getattr(args, "per_group_grad_clip", False)),
                                             note="global grad-norm >> clip budget: one gradient group "
                                             "is scaling the whole step down (seg starvation risk); C3 "
                                             "de-dominates eik, --per-group-grad-clip bounds per group")
                else:
                    _alarm["gnorm_streak"] = 0
                # (#304 item 4) per-term loss telemetry: BEFORE the skip branch so spike-skipped
                # chunks (the deadlock state) are covered too. Pure no-grad recompute + print.
                _bidx_lt = s // args.accum_pairs
                if _lt_stride and (_bidx_lt % _lt_stride == 0):
                    _t_agg, _t_tot = _loss_terms_for_chunk(chunk, seg_form, eik_w_ep)
                    # (V6 #320) thread the adaptive-eps control state when active (None => omitted =>
                    # byte-identical row schema for non-adaptive runs).
                    _lt_ve = _eik_stab["visco_eps"] if _eik_stab["visco_adaptive"] else None
                    _lt_ca = _eik_stab["visco_c_a"] if _eik_stab["visco_adaptive"] else None
                    print(json.dumps(_loss_terms_row(
                        _t_agg, _t_tot, ep, _bidx_lt, gnorm=gnorm, skipped=skip,
                        visco_eps=_lt_ve, visco_c_a=_lt_ca,
                        accepted_frac=_live_running_frac(), weights_stepped=(not skip),
                        hosc_beta=float(model.hosc_beta), softmax_temp=float(model.softmax_temp))), flush=True)
                    # (C6) term_domination alarm: a single reg term > 40% of the (post-weight) total
                    # for >= N sustained rows. _t_agg values are already post-weight addends.
                    _reg_keys = ("eikonal", "length", "eik_steik", "boundary_distance")
                    _tot_abs = abs(float(_t_tot)) + 1e-12
                    _dom = max(((k, abs(float(_t_agg.get(k, 0.0))) / _tot_abs) for k in _reg_keys),
                               key=lambda kv: kv[1], default=(None, 0.0))
                    if _dom[1] > _TERMDOM_FRAC:
                        _alarm["termdom_streak"] += 1
                        if _alarm["termdom_streak"] == _TERMDOM_MIN_ROWS:
                            _emit_confound_alarm("term_domination", ep=ep, term=_dom[0],
                                                 frac_of_loss=round(float(_dom[1]), 4),
                                                 sustained_rows=int(_alarm["termdom_streak"]),
                                                 note="a single regularizer term dominates the loss "
                                                 "(>40%): the scored seg/pose signal is a passenger "
                                                 "(C3 recalibrates the viscous eik unit scale)")
                    else:
                        _alarm["termdom_streak"] = 0
                # (EIK-STAB build 2) SUSTAINED-runaway rollback: restore last-good + cut lr +
                # re-arm, then skip THIS batch's step too (its gradient came from the diverged
                # state). Legacy mode: _sg_act is None => never fires => byte-identical.
                if _sg_act == "rollback":
                    _sg_do_rollback(ep, batch_loss, gnorm)
                    if (args.mlx_device == "gpu" and args.mlx_cache_clear_accum > 0
                            and ((s // args.accum_pairs) % args.mlx_cache_clear_accum == 0)):
                        mx.clear_cache()
                    continue
                if skip:
                    print(json.dumps({"stage": "spike_skip", "ep": ep,
                                      "batch_loss": (round(batch_loss, 4) if np.isfinite(batch_loss) else "nonfinite"),
                                      "gnorm": (round(gnorm, 4) if np.isfinite(gnorm) else "nonfinite")}), flush=True)
                    # (#205 OOM fix) still return the render/backward buffer POOL to the OS even on a
                    # spike-skipped batch, so a RUN of consecutive skips cannot balloon the Metal cache.
                    if (args.mlx_device == "gpu" and args.mlx_cache_clear_accum > 0
                            and ((s // args.accum_pairs) % args.mlx_cache_clear_accum == 0)):
                        mx.clear_cache()
                    continue
                opt.update(model, clipped)
                mx.eval(model.parameters(), opt.state)
                _live["ep_acc"] += 1  # (C6 liveness) an ACCEPTED (weight-stepping) accum-batch
                # #224 (5) SEED CONTAINMENT step: shield the seed grad (defend the seeded islands from
                # the bulk-CE wash) then apply the SEPARATE seed AdamW. The shield touches ONLY the seed
                # 'residual' leaf; the witness opt.update above + MD-decoupling below + grouped-backward
                # are all UNTOUCHED (distinct optimizer + distinct param tree). Gated by the same spike
                # skip as the witness step (only steps when the batch was not spike-skipped).
                if seed_mod is not None and accum_seed is not None:
                    _mean_sg = tree_map(lambda g, c=float(nb): g / c, accum_seed)
                    _sg = _seed_shield(_mean_sg["residual"], seed_mod.residual, seed_spec)  # leaf-only shield
                    seed_opt.update(seed_mod, {"residual": _sg})
                    mx.eval(seed_mod.parameters(), seed_opt.state)
                # DM1a (Stiefel-W): project the LIVE film.weight onto orthonormal columns AFTER the
                # optimizer step, so PR(M)=PR(cov(code)) holds (to the projection's ~1e-2 residual) for
                # the LIVE weight.
                # Default-off (--film-stiefel) => skipped => byte-identical. The cubic Newton-Schulz
                # polar re-normalizes columns, which also neutralizes the global-magnitude component of
                # AdamW weight-decay on W (the design's WD=0-on-W intent). NOTE: composes with the Muon
                # finisher (the projection runs whichever optimizer produced the step).
                #   (review Med1) The EMA update below averages the (per-step on-manifold) LIVE weight
                #   into the shadow; an arithmetic EMA of orthonormal matrices is NOT itself orthonormal,
                #   so the DEPLOYED shadow drifts OFF-Stiefel. The shipped artifact is re-projected at
                #   verdict + byte-close via _project_shadow_film_np (NOT here -- mutating the shadow
                #   in place would break resume bit-faithfulness). This comment formerly claimed "the
                #   deploy shadow tracks the on-manifold weight" -- FALSE; corrected.
                if args.film_stiefel:
                    model.film.weight = stiefel_project_columns(model.film.weight)
                    mx.eval(model.film.weight)
                ema.update(model)
                mx.eval(list(ema.shadow.values()))
                # (EIK-STAB build 2) median hygiene: in rollback mode an ACCEPTED-but-spiked batch
                # must NOT poison the running median (the spike detector's healthy reference).
                # Legacy mode reaches here only for non-spiked batches => condition True => the
                # append is byte-identical to the pre-build unconditional append.
                if _sg_guard is None or not (_spiked or _nonfinite):
                    recent_losses.append(batch_loss)
                    if len(recent_losses) > 50:
                        recent_losses.pop(0)
                ep_loss += batch_loss
                # (#205 OOM fix) return the Metal buffer POOL to the OS every N accum-batches. The
                # lazy graph is already materialized per-pair; this frees the CACHED (already-freed)
                # render+backward buffers so peak RSS ~= active working set + one batch (NOT a whole
                # epoch's freed-buffer pool). clear_cache never touches LIVE arrays => BIT-IDENTICAL.
                _bidx = s // args.accum_pairs
                if (args.mlx_device == "gpu" and args.mlx_cache_clear_accum > 0
                        and (_bidx % args.mlx_cache_clear_accum == 0)):
                    mx.clear_cache()
                if _mem_probe_on and ep <= _mem_probe_epochs:
                    _mm = _mlx_mem_gib(mx)
                    print(json.dumps({"stage": "mem_probe", "ep": ep, "accum_batch": _bidx,
                                      "rss_gib": round(_rss_gib(), 2),
                                      "mlx_active_gib": round(_mm["active"], 2),
                                      "mlx_cache_gib": round(_mm["cache"], 2),
                                      "mlx_peak_gib": round(_mm["peak"], 2),
                                      "clear_accum": int(args.mlx_cache_clear_accum)}), flush=True)
            if _prof is not None:
                _prof["step_s"] = time.perf_counter() - _prof["_step0"]  # #252 profile: step (fwd+bwd+opt+ema)
            if args.mlx_device == "gpu":
                mx.clear_cache()
            # (C6 liveness) SNAPSHOT the just-completed epoch's accepted-batch fraction so the
            # verdict / closed_loop / next-epoch-top rows read a TRUTHFUL liveness signal. Then the
            # spike_deadlock alarm: a run frozen (skip-frac > 0.9) for >= 2 consecutive epochs.
            _live["acc"] = int(_live["ep_acc"])
            _live["skip"] = int(_live["ep_tot"]) - int(_live["ep_acc"])
            _live["frac"] = float(_live["ep_acc"]) / float(max(int(_live["ep_tot"]), 1))
            _live["stepped"] = bool(_live["ep_acc"] > 0)
            if int(_live["ep_tot"]) > 0 and _live["frac"] < (1.0 - 0.9):  # skip-frac > 0.9
                _alarm["deadlock_streak"] += 1
                if _alarm["deadlock_streak"] >= 2:
                    _emit_confound_alarm("spike_deadlock", ep=ep,
                                         accepted_frac=round(float(_live["frac"]), 4),
                                         accepted_batches=int(_live["acc"]),
                                         skipped_batches=int(_live["skip"]),
                                         consecutive_epochs=int(_alarm["deadlock_streak"]),
                                         spike_guard_mode=str(getattr(args, "spike_guard_mode", "rollback")),
                                         note="spike-guard skip-frac > 0.9 for >=2 ep = the absorbing "
                                         "median-freeze DEADLOCK (all telemetry is FROZEN-state; use "
                                         "--spike-guard-mode rollback)")
            else:
                _alarm["deadlock_streak"] = 0
            # #224 (5) SEED SURVIVAL telemetry: mean |seed residual| ON the island support. The
            # containment shield should keep this ABOVE ~0 (the seeded islands survive the bulk-CE
            # wash); WITHOUT the shield the bulk wash drives it toward 0 (the failure this defends).
            # Purely observational (never read back). Default OFF (seed_mod None) => never fires.
            if seed_mod is not None and (ep % args.eval_every == 0 or ep == args.epochs):
                _sr = np.asarray(seed_mod.residual)                       # (P,H,W,3)
                _sm = np.stack([np.asarray(m) for m in seed_state["masks"]], axis=0)  # (P,H,W,1)
                _mon = float(np.sum(np.abs(_sr) * _sm) / (np.sum(_sm) * 3.0 + 1e-9))
                print(json.dumps({"stage": "seed_survival", "epoch": ep,
                                  "mean_abs_seed_on_island": round(_mon, 5),
                                  "containment_mode": str(args.containment_mode),
                                  "note": "shield keeps seeded islands alive vs bulk-CE wash (advisory)"}),
                      flush=True)
            if _prof is not None:
                _prof["_v0"] = time.perf_counter()  # #252 profile: verdict start
            # (#292 build-3) closed-loop early-stop flag, re-armed each epoch by the decision
            # point below. OFF (default; the #205 path) => stays False => never fires.
            _cl_stop_now = False
            if ep % args.eval_every == 0 or ep == args.epochs:
                if args.async_verdict:
                    # FEED-em: offload the observational verdict to a background thread so the
                    # GPU loop never idles. BIT-IDENTICAL training (verdict is never read back).
                    if _cl_on:
                        # ── (M2 fix) DECIDE-ON-PREVIOUS-VERDICT reorder (closed-loop + async) ──
                        # The pre-fix order (schedule THIS epoch's verdict, then JOIN it at the
                        # decision point) blocked the GPU for the FULL verdict wall at every eval
                        # (measured 2062-2439s ~= the 25-epoch train wall => ~2x total run wall,
                        # ~22h -> ~44h). Airtight reorder:
                        #   1. JOIN the PREVIOUS eval's verdict FIRST — it has had a full eval
                        #      window to run => wait ~= max(0, verdict_wall - window_wall) ~= 0.
                        #      After the join NOTHING is in flight and _cl_verdicts holds ALL rows
                        #      for evals < ep, deterministically.
                        #   2. DECIDE on those rows (decide-on-previous; a pure function of the
                        #      seeded trajectory — the verdict EPOCH SET is fixed because
                        #      join-before-schedule means the skip-throttle can never fire when ON).
                        #   3. THEN schedule THIS epoch's verdict — it is NEVER joined in this
                        #      iteration, so the GPU never blocks on it. It is scheduled even when
                        #      the early-stop just armed: the post-loop _join_async_verdict() lands
                        #      the final row (valuable telemetry) before the final checkpoint +
                        #      result.json. Any sidecar written below while it is in flight carries
                        #      the PENDING-VERDICT record (bit-faithful resume; see
                        #      _schedule_async_verdict + the resume reconcile).
                        _join_async_verdict()
                        _cl_stop_now = _cl_decide(ep)
                    elif ep == args.epochs:
                        # closed-loop OFF async path — the ORIGINAL FEED-em order, unchanged:
                        # at the FINAL epoch, JOIN first so the last row is not skip-throttled.
                        _join_async_verdict()
                    _schedule_async_verdict(ep, seg_form, ep_loss)
                else:
                    v = realized_verdict()
                    # (#302) split per-class counts out of the float-only sync verdict row / history.
                    _ncounts_sync = v.pop("nucleus_counts", None) if isinstance(v, dict) else None
                    # (SENSE) pop annulus BEFORE the float-only row spread. None unless
                    # --annulus-telemetry is ON => byte-identical when absent.
                    _annulus_sync = v.pop("annulus", None) if isinstance(v, dict) else None
                    blob = quantize_levelset_blob({k: np.asarray(v, np.float32) for k, v in ema.shadow.items()})
                    s = implied_score_from_verdict(v["d_seg"], v["d_pose"], blob["total_quantized_blob_bytes"])
                    # (C6) LIVENESS STAMP on the verdict row + frozen_epoch flag/alarm: a verdict d_seg
                    # from a FROZEN (all-skip) epoch is a frozen-state sample, NOT converged progress.
                    _frozen_ep = (int(_live["ep_tot"]) > 0 and int(_live["acc"]) == 0)
                    print(json.dumps({"stage": "verdict", "epoch": ep, "seg_form": seg_form,
                                      **{k: round(vv, 6) for k, vv in v.items()},
                                      "blob_bytes": blob["total_quantized_blob_bytes"], "implied_S": round(s, 4),
                                      "ep_loss": round(ep_loss, 3),
                                      "accepted_frac": round(float(_live["frac"]), 4),
                                      "weights_stepped": bool(_live["stepped"]),
                                      "accepted_batches": int(_live["acc"]), "skipped_batches": int(_live["skip"]),
                                      "frozen_epoch": bool(_frozen_ep),
                                      "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}), flush=True)
                    # (SENSE) companion annulus_convergence row (opt-in; byte-identical when absent).
                    if _annulus_sync is not None:
                        print(json.dumps(_annulus_convergence_row(_annulus_sync, ep, seg_form)), flush=True)
                    if _frozen_ep or (float(ep_loss) == 0.0 and int(_live["ep_tot"]) > 0):
                        _emit_confound_alarm("frozen_epoch", ep=ep, ep_loss=round(float(ep_loss), 6),
                                             accepted_batches=int(_live["acc"]),
                                             note="ep_loss==0.0 / all batches skipped: this verdict is a "
                                             "FROZEN-state sample, not converged progress (do not treat "
                                             "as a plateau or a 'best')")
                    history.append({"epoch": ep, **v, "implied_S": s})
                    _emit_handoff_readiness(_ncounts_sync, ep, seg_form)  # (#302) readiness row; no-op OFF
                    # (#292 build-3) closed-loop capture on the SYNC verdict path too (already
                    # deterministic — no thread). OFF => appends nothing => byte-identical.
                    if _cl_on:
                        _cl_verdicts.append({"epoch": int(ep), "seg_form": str(seg_form),
                                             "d_seg": float(v["d_seg"]), "ep_loss": float(ep_loss)})
                    # HARDENING: preserve the best EMA shadow (sync path = current shadow IS what
                    # realized_verdict just scored; project film.weight on-manifold like the verdict).
                    _maybe_preserve_best(
                        v["d_seg"], ep,
                        _project_shadow_film_np({k: np.asarray(vv, np.float32)
                                                 for k, vv in ema.shadow.items()}),
                        float(model.softmax_temp), float(model.hosc_beta))
            if _prof is not None:
                _prof["verdict_s"] += time.perf_counter() - _prof["_v0"]  # #252 profile: verdict wall-clock
            # ── (#292 build-3) CLOSED-LOOP decision point — SYNC-verdict path ONLY (M2 fix) ──
            # The async+ON path decided ABOVE (decide-on-previous: join the PREVIOUS eval's verdict,
            # decide, THEN schedule — no join of the just-scheduled verdict anywhere, so the GPU
            # never waits the full verdict wall). The SYNC path has no wall problem (the verdict was
            # computed inline this epoch) and keeps the original current-epoch-row semantics
            # unchanged. Placed BEFORE the checkpoint blocks so the resume sidecar written this
            # epoch carries the POST-decision state (+ THIS epoch's row on the sync path; the async
            # path's in-flight row rides the PENDING-VERDICT record instead). CONTAINMENT: only
            # mutates the in-run eikonal bump + arms a clean early-stop; the best EMA-shadow ckpt
            # is preserved continuously by _maybe_preserve_best. OFF (default; the #205 path) =>
            # _cl_stop_now stays False and NOTHING here runs.
            if _cl_on and not args.async_verdict and (ep % args.eval_every == 0 or ep == args.epochs):
                _cl_stop_now = _cl_decide(ep)
            # DM1 telemetry (decisive-smoke signals; design memo §6 firewall). At eval cadence, log
            # PR(M) (per-pair FiLM modulation participation ratio), PR(cov(code)) and the Stiefel
            # residual ‖WᵀW−I‖_F so the A/B can SEPARATE "means fixed" (PR held >~3.0) from "end moved"
            # (advisory d_seg, in the verdict row above).
            #   (review C1) GATE WIDENED to include --dm1-telemetry so the A0 BASELINE (no DM1 lever)
            #     also logs the row -- otherwise the "baseline collapses" half of the firewall is
            #     UNMEASURABLE. Default-off (all three off) => never fires => bit-identical observability.
            #   (review Med1) The DEPLOYED weight is the EMA SHADOW, not live. An arithmetic EMA of
            #     orthonormal matrices is NOT orthonormal => the shadow drifts off-Stiefel. The firewall
            #     must read what SHIPS, so report BOTH the LIVE and the SHADOW PR(M)+residual (shadow
            #     modulation M_shadow = code @ W_shadowᵀ + b_shadow, ISOLATING the W drift on the same
            #     codes). Pure read (no model/grad touch).
            if (args.film_stiefel or code_spec_w > 0.0 or args.dm1_telemetry) and (ep % args.eval_every == 0 or ep == args.epochs):
                _S = min(2 * P, 256)
                _ssub = np.arange(0, 2 * P, max(1, (2 * P) // _S))[:_S].astype(np.int32)
                _codes = model.code[mx.array(_ssub)]
                _M = model.film(_codes)                                # (S, 2*H*L) LIVE modulation
                _pr_m = float(film_modulation_participation_ratio(np.asarray(_M, np.float32)))
                _pr_c = float(film_modulation_participation_ratio(np.asarray(_codes, np.float32)))
                _sres = stiefel_residual(model.film.weight) if args.film_stiefel else None
                # Med1: the SHADOW (deployed) film.weight modulation + its Stiefel residual.
                _Ws = ema.shadow.get("film.weight")
                _bs = ema.shadow.get("film.bias")
                _pr_m_shadow = None
                _sres_shadow = None
                if _Ws is not None:
                    _M_shadow = _codes @ _Ws.T
                    if _bs is not None:
                        _M_shadow = _M_shadow + _bs
                    _pr_m_shadow = float(film_modulation_participation_ratio(np.asarray(_M_shadow, np.float32)))
                    _sres_shadow = stiefel_residual(_Ws) if args.film_stiefel else None
                print(json.dumps({"stage": "dm1_telemetry", "epoch": ep, "seg_form": seg_form,
                                  "pr_film_M": round(_pr_m, 4), "pr_cov_code": round(_pr_c, 4),
                                  "stiefel_residual": (round(_sres, 5) if _sres is not None else None),
                                  "pr_film_M_shadow": (round(_pr_m_shadow, 4) if _pr_m_shadow is not None else None),
                                  "stiefel_residual_shadow": (round(_sres_shadow, 5) if _sres_shadow is not None else None),
                                  "film_stiefel": bool(args.film_stiefel),
                                  "code_spec_w": code_spec_w}), flush=True)
            # (--seg-focal-gamma) Rudin observability: per-epoch MEASURED island-gradient share row
            # (no silent reweighting). One ROTATING pair per epoch: render frame1 through the SAME
            # base-loss surface (render_fn = the composed chain when engaged), stop-grad the frame,
            # and take d(focal-weighted base-CE)/d(frame) — a true (post-R-surface) gradient share,
            # cheap (ONE SegNet fwd+bwd on ONE pair). DEFAULT 0.0 => never fires => bit-identical
            # observability. Islands = GT classes {1,3} (Lane, Movable; canonical comma10k order).
            if focal_gamma > 0.0:
                _fo_pi = (ep - 1) % P
                _fo_c1 = 2 * _fo_pi + 1
                _fo_render = render_fn if render_fn is not None else render_through_R_mlx
                _fo_f1 = mx.stop_gradient(_fo_render(model, _cf_mx(_fo_pi), _fo_c1, render_h, render_w))
                _fo_oh, _fo_mg = lstar_cache[_fo_pi]

                # (round-2 review F4) read through the SAME loss adapter the base loss uses:
                # when --logit-adjust-loss-tau != 0 the loss surface is the WRAPPED
                # _LogitAdjustSegAdapter — raw adapter.segnet here would misattribute the
                # island-gradient share in exactly the runs the lever targets. tau == 0.0 =>
                # ``_loss_adapter is adapter`` (same object) => byte-identical telemetry.
                def _fo_loss(fv, _oh=_fo_oh, _mg=_fo_mg):
                    logits = _loss_adapter.segnet(fv)
                    ce = mx.logsumexp(logits, axis=-1) - mx.sum(logits * _oh, axis=-1)
                    pw = ce * (1.0 + args.hinge_weight * mx.exp(-mx.clip(_mg, 0.0, 1e9)))
                    return mx.mean(pw * focal_pixel_weight_mlx(logits, _oh, focal_gamma))

                _fo_g = mx.grad(_fo_loss)(_fo_f1)
                mx.eval(_fo_g)
                _fo_gm = np.abs(np.asarray(_fo_g))[0].sum(-1)          # (H,W) per-pixel |grad| mass
                _fo_isl = np.isin(np.asarray(gt.lstars[_fo_pi]), (1, 3))
                _fo_share = float(_fo_gm[_fo_isl].sum() / (_fo_gm.sum() + 1e-12))
                print(json.dumps({"stage": "focal", "epoch": ep, "pair": int(_fo_pi),
                                  "gamma": focal_gamma,
                                  "island_grad_share": round(_fo_share, 5)}), flush=True)
            # ---- CHECKPOINTING (FEED-dz; mandatory per operator "never launch non-resumable / save
            # per-stage" rule). PER-STAGE: at every curriculum-stage TRANSITION save a PRESERVED,
            # stage-encoded, byte-close-loadable ckpt (per-stage A/B of which stage moves d_seg).
            # INTRA-STAGE: every --ckpt-every epochs save the rolling latest (crash-resume window).
            if _evt_on:
                # (#292 build-2) event-triggered: the OFF lookahead _seg_form_for_epoch(ep+1) is invalid
                # (the next transition depends on FUTURE losses not yet known), so save the PRESERVED
                # stage-transition ckpt at the epoch the transition ACTUALLY fired (this epoch, the first
                # of the new stage). _evt_event is set at the START of this epoch by _evt_resolve_seg_form.
                is_transition = bool(args.stage_checkpoints and _evt_event is not None)
            else:
                is_transition = (
                    args.stage_checkpoints and ep < args.epochs
                    and _seg_form_for_epoch(ep + 1, args) != seg_form)
            do_periodic = args.ckpt_every > 0 and ep % args.ckpt_every == 0
            if is_transition:
                # FEED-fi: tag the preserved ckpt with the optimizer phase too, so a curriculum
                # transition DURING the Muon finisher is distinctly byte-closeable (suffix "" when
                # the finisher is off => identical filename to the pre-FEED-fi path).
                w = _do_checkpoint(ep, stage_tag=_stage_tag(seg_form) + ("_muon" if muon_switched else ""))
                stage_ckpts.append(w)
                print(json.dumps({"stage": "checkpoint", "kind": "stage_transition", **w}), flush=True)
            elif do_periodic:
                w = _do_checkpoint(ep)
                print(json.dumps({"stage": "checkpoint", "kind": "intra_stage", **w}), flush=True)
            # ── #252 per-epoch timing emit (advisory; at eval cadence so no per-epoch spam). The
            # split is fwd+bwd-step (INR+R+scorer+loss+backward+opt+ema, fused inside value_and_grad)
            # vs verdict vs overhead (gates/reorient/permutation/LR). R is NOT separable inside the
            # fused graph, so its share is measured DIRECTLY by an isolated in-situ R micro-bench at the
            # real render resolution (reference vs fused, fwd + fwd+bwd) -> R_fraction = R_fwdbwd *
            # frames/epoch / step_s, and the realized whole-run speedup follows by Amdahl. Emitted only
            # when --profile-timing (default OFF => this whole block is skipped => byte-identical).
            if _prof is not None and (ep % args.eval_every == 0 or ep == args.epochs):
                _ep_s = time.perf_counter() - _prof["ep_start"]
                _frames = int(2 * len(order))  # 2 frames (f0,f1) per pair-visit this epoch
                _rmb = r_isolated_microbench(render_h=render_h, render_w=render_w, n_frames=2, reps=15)
                _step_s = float(_prof["step_s"])
                _rfwdbwd_ms = _rmb.get("ref_fwdbwd_ms_per_frame")
                _r_share = (
                    (_rfwdbwd_ms / 1e3 * _frames / _step_s) if (_rfwdbwd_ms and _step_s > 0) else None)
                print(json.dumps({
                    "stage": "profile_timing", "epoch": ep,
                    "t_epoch_s": round(_ep_s, 4),
                    "t_step_fwd_bwd_opt_ema_s": round(_step_s, 4),
                    "t_verdict_s": round(float(_prof["verdict_s"]), 4),
                    "t_overhead_s": round(max(_ep_s - _step_s - float(_prof["verdict_s"]), 0.0), 4),
                    "frames_per_epoch": _frames,
                    "R_isolated": _rmb,
                    "R_fraction_of_step_est": (round(_r_share, 4) if _r_share is not None else None),
                    "fused_r_active": bool(getattr(args, "fused_r_kernel", False)),
                    "note": "R fraction from isolated in-situ R fwd+bwd; whole-run speedup by Amdahl "
                    "1/((1-f)+f/su_R); advisory, buys SPEED not score; pointer 0.19110 UNMOVED"}),
                    flush=True)
            if _evt_on or _nucleus_on:
                # (#292 build-2) record THIS epoch's synchronous training loss into the current stage's
                # history, consumed by _evt_resolve_seg_form at the START of the NEXT epoch (so the
                # controller only ever reads PAST epochs -> no lookahead). ep_loss is the seeded per-epoch
                # loss sum (NOT the async d_seg verdict) => the fired epochs stay deterministic. When ONLY
                # --handoff-readiness-telemetry is on (event trigger OFF), the history still feeds the
                # readiness row's plateau_ok (passive validation) but no boundary ever fires => training
                # is byte-identical. Both OFF => skipped entirely => byte-identical (the #205 path).
                _evt_state["losses"].append(float(ep_loss))
            last_ep = ep
            # (#292 build-3) closed-loop EARLY-STOP: break at the END of the loop body (after
            # last_ep + the checkpoint/telemetry blocks above ran for this epoch) so the post-loop
            # final checkpoint + result.json land normally with last_ep == the stop epoch. The stop
            # was ARMED at the decision point above (before the sidecar write => armed state
            # persists). OFF (default) => _cl_stop_now is always False => never fires.
            if _cl_stop_now:
                print(json.dumps({
                    "stage": "closed_loop_early_stop", "epoch": ep,
                    "reason": "sustained diverging_erasing persisted "
                              f"{_cl_state['post_budget_windows']} eval-windows after the "
                              f"bump budget ({int(args.closed_loop_max_bumps)}) was spent",
                    "best_d_seg": (round(float(_best["d_seg"]), 6)
                                   if _best["ep"] is not None else None),
                    "best_epoch": _best["ep"], "best_path": _best["path"],
                    "note": "clean stop; final checkpoint + result.json follow (best EMA "
                            "shadow preserved); structural erosion — don't waste epochs"}),
                    flush=True)
                break

    # FEED-em: JOIN any in-flight async verdict so the final verdict row + history land BEFORE
    # result.json is written (the DONE-marker contract). No-op when --async-verdict is off.
    if args.async_verdict:
        _join_async_verdict()

    # FINAL checkpoint (replaces the historical loop-end-only save, which is now FORBIDDEN). Always
    # writes the rolling latest + a PRESERVED final stage-encoded ckpt -> the run is byte-closeable
    # and resumable from disk at completion. Saves the EMA SHADOW (deploy), NOT live (EMA rule).
    # (#292 build-2) event-triggered: the final stage form is the controller's CURRENT stage (the OFF
    # _seg_form_for_epoch(last_ep) would use the hardcoded caps, not the actually-fired boundaries).
    final_form = (
        _evt_current_stage_form(_evt_state) if (_evt_on and last_ep >= 1)
        else (_seg_form_for_epoch(last_ep, args) if last_ep >= 1 else args.seg_loss))
    # FEED-fi: the FINAL ckpt is the Muon-finished decoder when the finisher ran -> tag it "_muon"
    # so it is distinctly byte-closeable (suffix "" when off => identical to the pre-FEED-fi path).
    _final_tag = (_stage_tag(final_form) + ("_muon" if muon_switched else "")) if args.stage_checkpoints else None
    final = _do_checkpoint(last_ep, stage_tag=_final_tag)
    stage_ckpts.append({**final, "kind": "final"})
    ck = out_dir / "levelset_witness_ema_mlx.npz"
    print(json.dumps({"stage": "checkpoint", "kind": "final", **final}), flush=True)
    result = {
        "utc": _utc(), "n_pairs": P, "epochs": args.epochs, "final_epoch": last_ep,
        # #205 PROVENANCE (deterministic-reproducibility: git sha + upstream snapshot sha + seed).
        "provenance": {**_run_provenance, "seed": int(args.seed)},
        "render_hw": [render_h, render_w],
        "front_end": "curvelet" + ("+self_orient" if use_self_orient else ""),
        "activation": args.activation, "in_feat": int(in_feat),
        "history": history, "checkpoint": str(ck), "stage_checkpoints": stage_ckpts,
        # HARDENING: the BEST realized-d_seg EMA-shadow ckpt (None if no finite verdict landed).
        # The harvester / next-arm warm-start reads this (or levelset_best.json) instead of the
        # rolling "latest", which can have drifted past the best.
        "best": (dict(_best) if _best["ep"] is not None else None),
        "resumable": True, "ckpt_every": int(args.ckpt_every),
        # (review C2) anneal schedule length (deterministic-reproducibility provenance). None default =>
        # records the resolved value (== epochs) so a reader knows the exact cosine denominator used.
        "anneal_epochs": int(anneal_epochs),
        # (review C1/Med1) DM1 telemetry + shadow-projection provenance (all default-OFF paths recorded).
        "dm1_telemetry": bool(getattr(args, "dm1_telemetry", False)),
        "film_stiefel": bool(getattr(args, "film_stiefel", False)),
        "code_spectral_entropy_weight": float(getattr(args, "code_spectral_entropy_weight", 0.0)),
        # BUILD 1/2 (FEED-fw) provenance (deterministic-reproducibility: record config with the
        # result). All default-OFF => these reflect the bit-identical path.
        "stage_transition_rewarmup_epochs": int(getattr(args, "stage_transition_rewarmup_epochs", 0)),
        "stage_transition_rewarmup_floor": float(getattr(args, "stage_transition_rewarmup_floor", 0.1)),
        "stage_transition_rewarmup_shape": str(getattr(args, "stage_transition_rewarmup_shape", "linear")),
        "stage_transition_reset_moments": bool(getattr(args, "stage_transition_reset_moments", False)),
        "lane_prior_phi1": bool(getattr(args, "lane_prior_phi1", False)),
        "lane_prior_phi1_mode": str(getattr(args, "lane_prior_phi1_mode", "replace")),
        # THETA* TIER-2 levers (deterministic-reproducibility: record config with the result). All
        # default-OFF => these values reflect the bit-identical path.
        "tau_anneal_shape": str(getattr(args, "tau_anneal_shape", "cosine")),
        "tau_hold_frac": float(getattr(args, "tau_hold_frac", 1.0)),
        "code_nuclear_weight": float(getattr(args, "code_nuclear_weight", 0.0)),
        "code_nuclear_eps": float(getattr(args, "code_nuclear_eps", 1e-3)),
        "code_nuclear_ns_iters": int(getattr(args, "code_nuclear_ns_iters", 25)),
        "ema_decay_finisher": (float(args.ema_decay_finisher)
                               if getattr(args, "ema_decay_finisher", None) is not None else None),
        "ema_decay_finisher_start_epoch": (int(args.ema_decay_finisher_start_epoch)
                                           if getattr(args, "ema_decay_finisher_start_epoch", None) is not None else None),
        "eikonal_junction_relax": float(getattr(args, "eikonal_junction_relax", 0.0)),
        "eikonal_junction_tau": float(getattr(args, "eikonal_junction_tau", 0.5)),
        # LEVER-A / LEVER-B provenance (deterministic-reproducibility; all default-OFF => the
        # bit-identical path is recorded as off).
        "film_per_layer": bool(getattr(args, "film_per_layer", False)),
        "film_concat_code": bool(getattr(args, "film_concat_code", False)),
        "film_rank_floor_weight": float(getattr(args, "film_rank_floor_weight", 0.0)),
        "film_rank_floor_target": float(getattr(args, "film_rank_floor_target", 4.0)),
        "lane_thin_weight": float(getattr(args, "lane_thin_weight", 0.0)),
        "lane_thin_class": int(getattr(args, "lane_thin_class", 1)),
        "lane_thin_radius": int(getattr(args, "lane_thin_radius", 4)),
        "lane_thin_target": float(getattr(args, "lane_thin_target", 0.5)),
        "lane_thin_start_epoch": int(getattr(args, "lane_thin_start_epoch", 0)),
        "axis": "[macOS-MLX training-gradient]/[macOS-CPU advisory] verdict; promotion_eligible=false; pointer UNMOVED",
    }
    (out_dir / "levelset_train_result.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LEVEL-SET witness through R (MLX): softmax-of-SDF + curvelet, realized d_seg")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-pairs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=1500,
                    help="(fix d) >=1500 for the PR95 d_seg curriculum (ce->tau->l7). Fail-closed asserted vs curriculum boundaries.")
    ap.add_argument("--anneal-epochs", type=int, default=None,
                    help="(review C2) SCHEDULE length for the softmax-temp + hosc-beta + LR cosine anneals "
                    "(the cosine DENOMINATOR), decoupled from --epochs (the RUN length). None (default) "
                    "=> use --epochs => BIT-IDENTICAL. A WARM-START arm (e.g. --resume-from a CE ckpt @ "
                    "ep299, --epochs 399) MUST set this to the ORIGINAL schedule length (e.g. 1500) so "
                    "ep300->400 reproduces the DISEASE regime (temp ~0.91->0.84, LR ~0.9*peak) the lever "
                    "must be tested in -- NOT the schedule tail (temp ~0.19->0.05, LR ~0.15*peak).")
    ap.add_argument("--eval-every", type=int, default=25)
    # RESUMABILITY + CHECKPOINTING (FEED-dz; additive). Per operator "never launch non-resumable /
    # save+preserve a checkpoint at the end of each stage": per-stage PRESERVED ckpts default ON;
    # --ckpt-every adds intra-stage rolling saves (crash window). --resume-from continues a run.
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="save the rolling EMA+resume checkpoint every N epochs (0=off; per-stage + final "
                    "saves always happen). Set e.g. 100 to bound a crash/OOM to <=N epochs of loss "
                    "and enable early byte-close during a multi-day run.")
    ap.add_argument("--stage-checkpoints", action=argparse.BooleanOptionalAction, default=True,
                    help="save a PRESERVED, stage-encoded, byte-close-loadable ckpt at every curriculum "
                    "stage transition + at the final epoch (default ON; --no-stage-checkpoints only for "
                    "throwaway smokes -- loop-end-only is forbidden for real rows).")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="resume a run from a checkpoint: a run DIR (prefers levelset_resume_state.npz, "
                    "falls back to levelset_witness_ema_mlx.npz) OR an explicit npz. Restores decoder + "
                    "per-pair codes + EMA shadow + optimizer (best-effort) + the epoch position.")
    ap.add_argument("--resume-allow-lever-drift", action=argparse.BooleanOptionalAction, default=False,
                    help="(F2) allow a --resume-from whose render-side LEVERS (lane_render_band / "
                    "persistence_loss_weight / amplify_weight / lane_band_start_epoch / render_aa / "
                    "hosc_beta_end / mod_dim) DIFFER from the checkpoint's training config. DEFAULT OFF "
                    "= FAIL-CLOSED (a silent lever drop is a deterministic-repro violation). Set ON only "
                    "for an INTENTIONAL warm-start re-treatment (loss/render-only levers add no params).")
    ap.add_argument("--resume-clear-spike-guard", action=argparse.BooleanOptionalAction, default=False,
                    help="(CE-window pre-stage 2026-07-05) on --resume-from, DISCARD the checkpoint's "
                    "saved spike-guard window (__recent_losses) instead of restoring it, so the guard "
                    "re-seeds its running median from the first post-resume accepted batch. The escape "
                    "for the measured guard DEADLOCK class: the median updates only on ACCEPTED batches, "
                    "so after a persistent >spike_factor loss-level shift EVERY batch skips forever "
                    "(#205 run 20260705T015247Z: 75/75 skips/ep from ep92, frozen median ~6-8 vs loss "
                    "~58-66). DEFAULT OFF = the bit-faithful restore is unchanged.")
    ap.add_argument("--warm-start-weights-only", action=argparse.BooleanOptionalAction, default=False,
                    help="(DE#3 clean warm-start 2026-07-05) on --resume-from, take ONLY the trained "
                    "WEIGHTS from the checkpoint (EMA-shadow preferred, else live params) and DISCARD its "
                    "optimizer moments (=> fresh AdamW), its saved spike-guard window (=> re-seeded), and "
                    "(unless --warm-start-epoch is given) advance the epoch normally. Extincts the "
                    "POISONED-RESUME trap: resuming a DEADLOCKED levelset_resume_state.npz re-enters the "
                    "stale-optimizer + frozen-spike-guard-window deadlock (v5 deadlocked ep110-172; its "
                    "resume_state carries the runaway __recent_losses + ep150 moments), whereas the BEST "
                    "WEIGHTS (ema_BEST, d_seg 0.025) are clean. This flag lets a warm-start take the clean "
                    "weights from EITHER a deploy npz OR the poisoned sidecar. Also auto-allows lever "
                    "drift (a warm-start is an intentional re-treatment). NOTE: --resume-from a deploy "
                    "ema/BEST npz ALREADY yields fresh moments (has_opt=False) + no frozen guard; this "
                    "flag makes the weights-only intent EXPLICIT and safe from a full sidecar too. "
                    "DEFAULT OFF => the resume path is byte-identical.")
    ap.add_argument("--warm-start-epoch", type=int, default=-1,
                    help="(DE#3) with --warm-start-weights-only, set the start epoch explicitly "
                    "(e.g. 126 to continue just past the ep125 BEST verdict when warm-starting from a "
                    "resume_state.npz whose __resume_epoch is the later DEADLOCK epoch). Default -1 => "
                    "use the checkpoint's own epoch + 1 (the deploy-npz path's natural continuation).")
    ap.add_argument("--resume-model-from", type=str, default=None, choices=("live", "ema"),
                    help="(C9 confound fix 2026-07-05) on --resume-from, which weights load into the "
                    "MODEL: 'live' = the checkpoint's live params (the pre-fix behavior); 'ema' = the "
                    "CLEAN EMA shadow (a crash mid-spike writes DIVERGING live weights while the EMA "
                    "shadow in the SAME file is clean; loading live re-enters the divergence). DEFAULT "
                    "resolves to 'live' EXCEPT it auto-defaults to 'ema' for a re-treatment resume "
                    "(--warm-start-weights-only, OR both --resume-clear-spike-guard AND "
                    "--resume-allow-lever-drift set) where the clean shadow is the right warm-start "
                    "source. Explicit value always wins. When 'ema', keys missing from the shadow fall "
                    "back to live so the model has a full param set.")
    ap.add_argument("--freeze-decoder-fit-codes", type=str, default=None,
                    help="FEED-eo AMORTIZATION (days->hours): load the SHARED decoder from this level-set "
                    "EMA/deploy npz (trained on a SUBSET, e.g. n96/n192), FREEZE it, and fit ONLY the "
                    "per-pair codes for all --num-pairs pairs (embarrassingly-parallel per-pair latent "
                    "fit through the frozen render+R+scorer). The front-end config (--bank-*/--max-bank-"
                    "freq/--self-orient/--n-dir-freqs) MUST match the decoder's in_feat. Incompatible "
                    "with --resume-from/--structured-init. DEFAULT None = normal joint train.")
    # ---- RESIDUAL-ONLY MODE (v2 hybrid; gap #1; ADDITIVE, default-OFF => BIT-IDENTICAL). The
    # rate-bearing fix: train the small INR on the RESIDUAL the FIXED deterministic bulk leaves,
    # with the bulk GENERATED at decode (OUTSIDE the counted weights) and COMPOSED before R --
    # NOT baked into the weights via --structured-init (which does NOT shrink the rate). Every
    # realized render becomes ``composed = where(bulk_label_mask, INR, bulk)`` (the mask is
    # bulk-LABEL-derived => regenerated FREE at inflate, 0 counted bytes). The d_seg loss + ALL
    # surgical levers (lane-thin/margin-saliency/hardness) then weight the COMPOSED-render d_seg,
    # so the INR only has to flip the Lane+Movable residual annulus -> it can be SMALL (the rate
    # win). --residual-mode OFF (default) => NONE of this fires => byte-identical to the
    # full-partition witness. See tac.v2_compose.residual_compose + the landing memo.
    ap.add_argument("--residual-mode", action=argparse.BooleanOptionalAction, default=False,
                    help="RESIDUAL-ONLY MODE (v2 hybrid): compose the FIXED deterministic bulk (+) "
                    "the small INR residual before R; train the INR on the COMPOSED-render d_seg. "
                    "Requires --residual-target-npz. DEFAULT OFF => byte-identical full-partition "
                    "witness. The rate win: the bulk is OUTSIDE the counted weights (NOT "
                    "--structured-init, which bakes it IN).")
    ap.add_argument("--residual-target-npz", type=str, default=None,
                    help="RESIDUAL-ONLY MODE input: the residual training bundle "
                    "(tac.v2_compose.residual_compose.save_residual_training_bundle) carrying the "
                    "deterministic bulk RGB (render res, pre-R) + the bulk-derived composition mask "
                    "per pair. Required when --residual-mode. The COUNTED bytes are the INR weights "
                    "this run produces -- NEVER this bundle.")
    # (config-review #1) render-384 is the MEASURED R-survival floor (render-192 pre-caps at
    # 0.00085 d_seg = +0.085 S, mathematically blocking sub-0.15). camera-R + SegNet dominate
    # wall-clock, so 384 is ~free vs 192. The "SDF smooth -> low-res ok" assumption is FALSIFIED.
    ap.add_argument("--render-h", type=int, default=384)
    ap.add_argument("--render-w", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--n-hidden", type=int, default=4)
    # (config-review #2) mod-32 (with hidden-96) -> ~122-130KB at n600 = the RD-optimum B*~122KB
    # (rate 0.081); mod-48/hidden-128 -> 161KB (0.107) overshoots by +0.026 S. n96 = capacity sweep.
    ap.add_argument("--mod-dim", type=int, default=32)
    # (config-review #4) softmax-temp ANNEAL hi->lo (not fixed 0.1, which reintroduces RGB Gibbs).
    ap.add_argument("--softmax-temp-start", type=float, default=1.0, help="anneal START (soft; gradients flow).")
    ap.add_argument("--softmax-temp-end", type=float, default=0.05, help="anneal END (sharp; SDF partition pinned).")
    # (THETA* TIER-2 MUST-1) softmax-temp anneal SHAPE (additive; default 'cosine' == bit-identical to
    # the pre-theta* cosine). 'geometric' = log-spaced decay (more epochs at small tau; damps late-tau
    # d_seg volatility). 'cosine_hold' = cosine to the floor at --tau-hold-frac, then HOLD at the end.
    ap.add_argument("--tau-anneal-shape", choices=["cosine", "geometric", "cosine_hold"], default="cosine",
                    help="THETA* MUST-1: softmax-temp anneal curve. cosine (default, bit-identical) | "
                    "geometric (log-spaced, more epochs at small tau) | cosine_hold (reach floor at "
                    "--tau-hold-frac then hold). geometric requires --softmax-temp-start/-end > 0.")
    ap.add_argument("--tau-hold-frac", type=float, default=1.0,
                    help="THETA* MUST-1: for --tau-anneal-shape cosine_hold, the fraction (0,1] of the "
                    "anneal window at which tau reaches --softmax-temp-end and HOLDS. 1.0 (default) = "
                    "no hold = BIT-IDENTICAL to cosine.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-end", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # (#222 deep-math gap-1) Adam second-moment decay beta2. Default 0.999 == the MLX AdamW default
    # betas=[0.9, 0.999] => BIT-IDENTICAL to the pre-flag path. The n600 accumulated-microbatch regime
    # has only n = P/accum_pairs ~ 75 optimizer steps per epoch's worth of distinct gradient statistics;
    # the arXiv 2603.02092 small-n rule 1-beta2 <~ (1-beta1^5)/n^3.5 => for beta1=0.9, n=75:
    # (1-0.59049)/75^3.5 ~ 1.12e-7 => beta2* ~ 0.99999988. The default 0.999 (1-beta2=1e-3) is ~4 orders
    # ABOVE that floor = under-smoothed for n~75. The launch config (witness_autoconfig all_levers)
    # sets 0.9999999 (1-beta2=1e-7 < 1.12e-7 => clears the threshold). beta1 stays 0.9 (MLX default).
    ap.add_argument("--adam-beta2", type=float, default=0.999,
                    help="#222 AdamW second-moment decay beta2 (beta1 fixed 0.9). Default 0.999 = MLX "
                    "default => bit-identical. Small-n (n~75 accum steps) optimum ~0.9999999 per "
                    "arXiv 2603.02092 (1-beta2 <~ (1-beta1^5)/n^3.5).")
    ap.add_argument("--ema-decay", type=float, default=0.997)
    # (THETA* TIER-2 MUST-3) SWA / wider-finisher EMA (additive; default None == bit-identical to the
    # --ema-decay path). When set, from the resolved finisher-start epoch onward the EMA uses this
    # WIDER decay (averages over the late oscillation -> flat-basin center, SWA-style).
    ap.add_argument("--ema-decay-finisher", type=float, default=None,
                    help="THETA* MUST-3: wider EMA decay applied from the finisher-start epoch onward "
                    "(SWA-style late-oscillation averaging). None (default) = use --ema-decay everywhere "
                    "= BIT-IDENTICAL. Typically > --ema-decay (e.g. 0.999/0.9995). Must be in (0,1).")
    ap.add_argument("--ema-decay-finisher-start-epoch", type=int, default=None,
                    help="THETA* MUST-3: 1-based epoch at which the wider --ema-decay-finisher engages. "
                    "None (default) = fall back to --muon-start-epoch. Required (here or via "
                    "--muon-start-epoch) when --ema-decay-finisher is set.")
    ap.add_argument("--lr-schedule", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--w-seg", type=float, default=100.0)
    # (fix g) DROP pose-from-texture (the COLLAPSED amortized carrier, d_pose 2.67-12.66). Pose is
    # SOLVED by the Quantizr stored-pose sidecar (3.4e-5); the witness's ONLY binding job is d_seg.
    # w_pose=0 by default -> the texture head serves SegNet realism (seg), not pose reconstruction.
    ap.add_argument("--w-pose", type=float, default=0.0)
    ap.add_argument("--score-domain-loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pose-eps", type=float, default=1e-2)
    ap.add_argument("--hinge-weight", type=float, default=4.0)
    ap.add_argument("--accum-pairs", type=int, default=8)
    # (SPEED LEVER, DAG FEED 2026-07-03c) micro-batch B pairs per forward. DEFAULT 1 => the accum loop
    # takes the UNCHANGED serial per-pair value_and_grad path (BYTE-IDENTICAL). B>1 renders + scores B
    # pairs in ONE batched frozen-scorer forward (EfficientNet-B2 SegNet / FastViT PoseNet saturate the
    # GPU far better than single-pair batches) -> the measured ~2-4x speed lever. NOT bit-identical
    # (batched fp reduction order) => a trajectory-affecting opt-in, validated by a short trajectory A/B.
    # The per-pair loss reductions (base seg-form, score-domain pose sqrt, weighted-mean levers) are
    # computed PER PAIR on the batched scorer outputs and MEAN-ed over B, so total_loss_fn_batch(B) ==
    # mean_b total_loss_fn(pair_b) within fp tolerance -> the accum-loop grad matches the serial
    # mean-over-chunk EXACTLY (see the accum loop's per-group `* _bn` weighting). (BUILD #293)
    # COMPOSES with --seed-islands via the batched DUAL co-grad (_dual_vg_batch; equivalence executed
    # by experiments/test_batched_seed_cograd.py). Score-neutral verdict authority is unaffected.
    ap.add_argument("--micro-batch-pairs", type=int, default=1,
                    help="(speed lever) pairs per batched value_and_grad forward (1 = serial "
                    "byte-identical per-pair path; >1 = opt-in batched scorer forward, trajectory-"
                    "affecting, ~2-4x). Sub-batches each --accum-pairs chunk; grads weighted by pair "
                    "count so the accum-step grad == the serial mean-over-chunk. Composes with "
                    "--seed-islands (BUILD #293 dual co-grad).")
    # (--cache-gt-skeleton, #260 SPEED, BIT-IDENTICAL) opt-in per-pair cache of the CONSTANT GT
    # soft-skeleton the persistence loss recomputes every step. sg=soft_skeleton(gt) is a function of
    # the FROZEN GT argmax one-hot ONLY (constant across epochs) + carries NO gradient (it multiplies
    # pred in tsens), so precomputing it once per pair + reusing via sg_precomputed= is BIT-IDENTICAL
    # (a materialized concrete constant == the inline recompute) while skipping ~half the clDice cost.
    # Default OFF => total_loss_fn passes sg_precomputed=None => byte-identical to the pre-flag path.
    # No-op unless --persistence-loss-weight>0 (the only consumer); skipped under --micro-batch-pairs>1.
    ap.add_argument("--cache-gt-skeleton", action="store_true",
                    help="(speed, bit-identical) cache the CONSTANT per-pair GT soft-skeleton for the "
                    "persistence loss (sg=soft_skeleton(gt) is epoch-invariant + gradient-free). "
                    "Default OFF = byte-identical. No-op unless --persistence-loss-weight>0; "
                    "skipped under --micro-batch-pairs>1 (serial total_loss_fn is the only consumer).")
    # (#205 OOM FIX) MLX Metal caching-allocator hygiene. The lazy graph is already materialized
    # per-pair (mx.eval(loss, grads) + mx.eval(accum)); the leak is the Metal buffer POOL (freed
    # render/backward buffers stay CACHED, not returned to the OS) growing across an epoch's ~P/8
    # accum-batches -> a ~15 GiB active working set peaked at 90 GiB and tripped the 90 GB safe-run
    # guard (killed the run before the first checkpoint). Calling mx.clear_cache() every N accum
    # batches returns the pool to the OS -> peak RSS ~= active + one batch. clear_cache frees ONLY
    # pooled (already-freed) buffers, NEVER live arrays, and MLX is lazy-but-deterministic -> WHEN we
    # clear the pool cannot change WHAT is computed => BIT-IDENTICAL loss/d_seg (verified n64 A/B).
    # 1 = clear every accum-batch (safest peak); 0 = never inside the loop (the pre-fix behaviour,
    # for the A/B). GPU-only (no-op on cpu). The existing per-epoch clear at loop-end is preserved.
    ap.add_argument("--mlx-cache-clear-accum", type=int, default=1,
                    help="(#205 OOM fix) mx.clear_cache() every N accum-batches inside the epoch loop "
                    "(GPU only; score-neutral). 0 disables the in-loop clear (pre-fix behaviour).")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--per-group-grad-clip", action=argparse.BooleanOptionalAction, default=False,
                    help="(C4 confound fix 2026-07-05) clip the global grad-norm PER top-level "
                    "parameter GROUP (in_proj / hidden / out_sdf / out_tex / film / palette / code) "
                    "independently to --grad-clip, instead of ONE global-norm clip over the whole "
                    "vector. The single global clip scales the WHOLE gradient by 1/gnorm, so a "
                    "volatile regularizer (eikonal) gradient that dominates gnorm THROTTLES the "
                    "seg/pose gradient on OTHER param groups (film/out_tex) 300-900x (measured "
                    "confound). Per-group clip bounds each group to its own budget so a volatile "
                    "group cannot hijack the shared clip. DEFAULT OFF => the single global clip = "
                    "BYTE-IDENTICAL. NOTE: a TRUE per-LOSS-TERM (eik-vs-seg) clip needs a second "
                    "backward pass (the fused grad sums eik+seg on the shared trunk); C3's viscous "
                    "normalization removes the eik-domination ROOT, so this per-group clip + the "
                    "gnorm_hijack alarm are the shipped defense-in-depth.")
    ap.add_argument("--spike-factor", type=float, default=5.0)
    # (EIK-STAB build 2; sweep lever #3 + #304) spike-guard actuator mode. Default legacy =>
    # BYTE-IDENTICAL skip-with-frozen-median. rollback = tolerate bounded oscillation (single
    # finite spikes STEP; EoS self-stabilization is functional) + on SUSTAINED runaway (> frac of
    # a full window) restore the last-good snapshot, cut lr x lr-cut, re-arm a fresh median.
    ap.add_argument("--spike-guard-mode", type=str, default="rollback",
                    choices=("legacy", "rollback"),
                    help="EIK-STAB build 2: spike-guard actuator (legacy=skip-with-frozen-median; "
                    "rollback=bounded-oscillation tolerance + rollback-to-last-good + lr cut on "
                    "sustained runaway). DEFAULT rollback (C1 confound fix 2026-07-05): the legacy "
                    "accepted-only median FREEZES on a sustained loss-level shift (75/75 skips/ep "
                    "forever; #205 froze v5 ep114 / v6 ep103 -> the 'viscosity NO-GO' verdict was a "
                    "frozen artifact). legacy stays selectable for the A/B but is NO LONGER the "
                    "default. NOTE: legacy is NO LONGER byte-identical-by-default.")
    ap.add_argument("--spike-rollback-window", type=int, default=20,
                    help="rollback mode: batch window for the sustained-runaway trigger.")
    ap.add_argument("--spike-rollback-frac", type=float, default=0.5,
                    help="rollback mode: trigger when spike fraction over a FULL window exceeds this.")
    ap.add_argument("--spike-rollback-lr-cut", type=float, default=0.5,
                    help="rollback mode: multiply lr by this on every rollback (persistent).")
    ap.add_argument("--spike-rollback-max", type=int, default=8,
                    help="rollback mode: max rollbacks per run; after that the guard reverts to "
                    "legacy skip semantics (bounded actuation).")
    # (EIK-STAB build 4; sweep lever #1) lambda_pre HVP probe: measure the Adam-PRECONDITIONED
    # sharpness lambda_max(P^-1/2 H P^-1/2) at the (resumed) start state via preconditioned power
    # iteration with finite-difference HVPs over the FULL n-pair batch, print JSON rows, and EXIT
    # before any training step. Tests the Adam-EoS threshold law lambda_pre* ~= 38/eta
    # (litsweep DOMAIN 2) against the measured lr bracket. Default 0 => OFF => byte-identical.
    ap.add_argument("--lambda-pre-probe-iters", type=int, default=0,
                    help="EIK-STAB build 4: >0 => run N preconditioned power iterations (FD HVP) "
                    "at the start state, print lambda_pre rows, and exit WITHOUT training.")
    ap.add_argument("--lambda-pre-probe-fd-eps", type=float, default=1e-3,
                    help="relative finite-difference step for the HVP probe.")
    ap.add_argument("--loss-term-log-every", type=int, default=0,
                    help="(#304 item 4) per-term loss telemetry cadence in accum chunks: 0 (default) "
                    "= first chunk of each epoch (standing per-epoch summary); N>0 = every N chunks; "
                    "-1 = fully OFF (zero extra forwards). TAC_LOSS_TERM_PROBE=1 overrides to every "
                    "chunk (per-batch diagnostic). Pure no-grad recompute: the training trajectory is "
                    "bit-identical on/off.")
    # (#205 REAL OOM fix) chunk the CPU-scorer verdict inference into vbatch-pair torch batches so
    # the fp32 (N,2,3,874,1164) cast + EfficientNet/FastViT activations do NOT spike ~30-50 GiB at
    # N=600 on top of the resident ~41 GiB self-orient cf_mx_cache (the 90 GB OOM). BIT-IDENTICAL
    # (eval-mode BN running stats). 0 = single N-wide batch (pre-fix, for the A/B parity check).
    ap.add_argument("--verdict-batch", type=int, default=32,
                    help="(#205 OOM fix) CPU-scorer verdict inference chunk size (pairs per torch "
                    "batch); 0 = single N-wide batch (pre-fix). Score-neutral (eval-mode BN).")
    ap.add_argument("--verdict-pairs", type=int, default=0,
                    help="realized fp32-numpy EMA-shadow verdict subset (0=all=n600; DEFAULT 0 per "
                    "C12 confound fix 2026-07-05 -- a 24-pair default violated the n600 non-negotiable "
                    "at the number that DEFINES the goal: best-ckpt selection + ALL d_seg telemetry + "
                    "the closed-loop classifier ran on 24/600). Subsetting stays OPT-IN. ALWAYS fp32 "
                    "one-codepath, never mlx-gpu.")
    ap.add_argument("--annulus-telemetry", action=argparse.BooleanOptionalAction, default=True,
                    help="(SENSE, DEFAULT ON — score-neutral read-only observability per CLAUDE.md \"'Off' is "
                    "a tracked queue\"; it only reads the already-computed verdict argmax + logs, changing NO "
                    "weight/byte/d_seg, so there is no safety reason to gate it; --no-annulus-telemetry opts "
                    "out for a pure byte-identity A/B) emit a companion {stage:annulus_convergence} row per verdict: the codim-1 "
                    "boundary-annulus vs interior d_seg flip split + per-class annulus flip-frac + GT-margin "
                    "p10/p50, computed from the realized argmax + the FIXED GT margin (gt.margins) via "
                    "tac.witness_annulus_metrics. --no-annulus-telemetry opts OUT => BYTE-IDENTICAL (no "
                    "row, no realized-map collection, no extra forward). ON reuses ONE SegNet forward for the realized argmax "
                    "(wrapped in try/except => can never crash/corrupt the verdict). OBSERVABILITY-ONLY "
                    "(never read into training/parity/resume). PARTIAL: no witness-logit margin/gibbs -- see "
                    "tools/witness_annulus_convergence.py for the full offline series.")
    ap.add_argument("--annulus-band", type=float, default=2.0,
                    help="(SENSE) fixed-threshold annulus = {px: |GT margin| < band} (SegNet-logit units). "
                    "Used only when --annulus-telemetry is set.")
    ap.add_argument("--annulus-bottom-k", type=float, default=0.05,
                    help="(SENSE) bottom-k annulus = smallest-k fraction of |GT margin| pixels (0<k<=1). "
                    "Used only when --annulus-telemetry is set.")
    ap.add_argument("--async-verdict", action=argparse.BooleanOptionalAction, default=False,
                    help="FEED-em: run the OBSERVATIONAL CPU-torch verdict in a BACKGROUND THREAD off a "
                    "point-in-time snapshot so the MLX-GPU loop never idles (~4.7%% wall-clock reclaim @ "
                    "n600). BIT-IDENTICAL training (the verdict is never read back); only the verdict "
                    "CADENCE may self-throttle under load (at-most-one in-flight). DEFAULT OFF = the "
                    "current synchronous bit-identical behavior.")
    ap.add_argument("--mlx-device", choices=["gpu", "cpu"], default="gpu")
    # ── compute-facet #252 (MLX + custom Metal). All DEFAULT-OFF + bit-identical when off. ──
    ap.add_argument("--fused-r-kernel", action=argparse.BooleanOptionalAction, default=False,
                    help="(#252) swap the pure-MLX R roundtrip for the fused Metal kernel "
                    "(metal_fused_r_operator; bit-identical fwd to the numpy-fp32 authority, ~1 ULP VJP). "
                    "A startup per-chip parity gate (assert_metal_matches_cpu_oracle) fails CLOSED if the "
                    "kernel is not bit-identical on this GPU. NO-FAKE: buys SPEED, never a score. Default OFF.")
    ap.add_argument("--mx-compile", action=argparse.BooleanOptionalAction, default=False,
                    help="(#252) install an mx.compile'd reference R, GATED by a startup bit-identity check. "
                    "MEASURED 2026-07-03: mx.compile reintroduces fp-contraction that flips the uint8-STE "
                    "d_seg argmax (fwd Δ~4.8e-3, ~1.11x) so this FAILS CLOSED on non-bit-identical hosts. "
                    "Prefer --fused-r-kernel. Default OFF.")
    ap.add_argument("--profile-timing", action=argparse.BooleanOptionalAction, default=False,
                    help="(#252) emit per-epoch wall-clock phase split (fwd+bwd step / opt+ema / verdict / "
                    "overhead) + an isolated R micro-bench (fwd and fwd+bwd, reference vs fused) so the R "
                    "fraction -> realized whole-run speedup is MEASURED, not estimated. Advisory; DEFAULT OFF "
                    "=> zero added work, byte-identical.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-cache", type=str, default=None)
    ap.add_argument("--chroma", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--palette-anchor", action=argparse.BooleanOptionalAction, default=True,
                    help="(DIAGNOSED FIX) init learnable palette to natural per-class mean GT RGB (transfer-probe ingredient; "
                    "breaks the ~0.51 luma-ramp plateau). --no-palette-anchor = generic ramp ablation.")
    # FRONT-END
    ap.add_argument("--bank-n-scales", type=int, default=4)
    ap.add_argument("--bank-n-orient0", type=int, default=6)
    ap.add_argument("--bank-f0", type=float, default=2.0)
    ap.add_argument("--bank-base", type=float, default=2.0)
    ap.add_argument("--bank-n-iso", type=int, default=4)
    # LEVER-2 (stem-Nyquist rate/anti-alias): cap curvelet-bank freqs (cycles/unit) at the SegNet
    # stem Nyquist (default 64 for SEG_W=512, stem-stride-2). None (default) = no cap = current
    # behavior. The DEFAULT curvelet bank (max 16 cyc/unit) is already sub-Nyquist so this is a
    # no-op there; the over-Nyquist waste is in --n-dir-freqs/--freq-across (see the memo). Additive.
    ap.add_argument("--max-bank-freq", type=float, default=None,
                    help="LEVER-2: drop curvelet atoms above this freq (cycles/unit); None=no cap. "
                    "Stem Nyquist = SEG_W/(4*stem_stride) = 64 for the default 512/stride-2.")
    ap.add_argument("--self-orient", action=argparse.BooleanOptionalAction, default=False,
                    help="add byte-closeable self-orientation directional feats (finetune lever; needs a roughly-learned base).")
    ap.add_argument("--n-dir-freqs", type=int, default=6)
    ap.add_argument("--reorient-every", type=int, default=50)
    ap.add_argument("--gpu-reorient", action=argparse.BooleanOptionalAction, default=False,
                    help="FEED-eo: compute the per-pair reorient argmax on MLX-GPU (fp32 twin forward) "
                    "instead of the 600 GPU-idle numpy CPU forwards (~6.2%% wall-clock reclaim @ n600). "
                    "PARITY-GATED (fp32-GPU vs fp64-numpy argmax differs at boundary px): adopt only "
                    "after experiments/probe_levelset_gpu_reorient_parity.py shows cos>0.999 + negligible "
                    "d_seg A/B. DEFAULT OFF = the bit-faithful numpy reorient (current behavior).")
    ap.add_argument("--freq-across", type=float, default=32.0, help="self-orient: HIGH freq across the edge (normal).")
    ap.add_argument("--freq-along", type=float, default=4.0, help="self-orient: LOW freq along the edge (tangent).")
    # ── GROUND-FRAME CHART (#194 / council draft §17.1; tac.boundary_math.ground_frame_chart) ──
    # Define the witness field ONCE in the reference pair's chart; per-pair evaluation PRE-COMPOSES
    # the input coords with the ξ-homography (chart change on INPUT coords, NOT a pixel warp; still
    # trained through R+scorer => does NOT inherit the FEED-ll #190 deterministic-render floor).
    # STRUCTURAL lever: active from ep0 by construction when on. v0 = single GROUND-plane chart
    # (per-class stratified routing = screw_blend's future consumer). DEFAULT OFF = byte-identical.
    ap.add_argument("--ground-frame-chart", action=argparse.BooleanOptionalAction, default=False,
                    help="(#194/§17.1) evaluate the witness in the GROUND frame: pre-compose per-pair "
                    "input coords with the cumulative ξ-homography (FEED-ll math, chart change only; "
                    "rule-118 FREE — derived from the stored pose table, 0 new archive bytes). v0 is "
                    "GROUND-plane-only and fail-closes with --self-orient / --render-aa != none "
                    "(coordinate-system consistency; see the wire-in block). DEFAULT OFF.")
    ap.add_argument("--gfc-ref-pair", type=int, default=0,
                    help="ground-frame-chart reference pair (the canonical chart; chart[ref]==identity "
                    "exactly). Must be 0 when --structured-init is on (the pretrain uses pair-0 feats).")
    ap.add_argument("--gfc-s-t", type=float, default=-0.003224707899359239,
                    help="ground-frame-chart translation scale s_t (default: the MEASURED FEED-ll "
                    "d_seg-optimal reach calibration, experiments/results/screw_reach/reach_n96.json).")
    ap.add_argument("--gfc-s-r", type=float, default=0.0,
                    help="ground-frame-chart rotation scale s_r (FEED-ll fit default 0.0).")
    ap.add_argument("--gfc-pitch", type=float, default=-0.01,
                    help="ground-frame-chart ground-plane pitch (rad; FEED-ll fit default -0.01).")
    # ACTIVATION
    # (config-review #3) HOSC is the ONLY descent evidence (probe 0.0066; A/B 0.221 hosc vs 0.265
    # wire). WIRE was a paper-default guess; default HOSC, run wire as a sweep arm.
    ap.add_argument("--activation", choices=["wire", "hosc", "relu"], default="hosc")
    ap.add_argument("--wire-w0", type=float, default=20.0)
    ap.add_argument("--wire-s0", type=float, default=10.0)
    ap.add_argument("--hosc-beta", type=float, default=4.0)
    # (FEED-fb) BETA-ANNEAL: the named UNSWEPT step-native L-infinity-optimal lever. hosc is
    # tanh(beta*sin(omega*u)); beta->inf => STEP-native (the topology-matched chart for the
    # piecewise-constant argmax target, no Gibbs). --hosc-beta-end is the anneal TARGET; when it is
    # None (default) OR == --hosc-beta, NO anneal occurs and beta stays CONSTANT every epoch =>
    # BIT-IDENTICAL to the pre-FEED-fb path. The optimal-form decoder build sharpens beta start->end
    # (e.g. --hosc-beta 4 --hosc-beta-end 8) so the activation step-sharpens as the SDF partition
    # pins (sister of the softmax-temp anneal at the top of the epoch loop).
    ap.add_argument("--hosc-beta-end", type=float, default=None,
                    help="hosc beta anneal TARGET (None => no anneal, beta constant at --hosc-beta => bit-identical).")
    ap.add_argument("--hosc-beta-anneal", choices=["linear", "cosine"], default="linear",
                    help="hosc beta anneal schedule start->end (only used when --hosc-beta-end is set).")
    ap.add_argument("--hosc-omega", type=float, default=1.0)
    ap.add_argument("--siren-init", action=argparse.BooleanOptionalAction, default=True,
                    help="SIREN init (Sitzmann 2020) for hosc/wire periodic layers (from-scratch trainability fix).")
    # (#310 BUILD, FEED-07b lever #2 sister) FINER++ variable-periodic FIRST-LAYER bias init.
    ap.add_argument("--finer-bias-init", action=argparse.BooleanOptionalAction, default=False,
                    help="#310/FEED-07b: FINER++ (arXiv 2407.19434) variable-periodic FIRST-LAYER bias "
                    "init — in_proj.bias ~ U(-k, k) from a DEDICATED rng stream (seed+salt) so each "
                    "neuron selects its own frequency/phase of the periodic (hosc/wire) activation; "
                    "the published fix for the measured fixed-beta hosc saturation-death. Applied "
                    "AFTER siren-init, from-scratch only (a --resume-from overwrites it; stamped "
                    "applied:false). DEFAULT OFF => zero RNG draws => byte-identical. Fails closed "
                    "on --activation relu (no period).")
    ap.add_argument("--finer-bias-k", type=float, default=10.0,
                    help="#310: FINER++ first-layer bias range k (bias ~ U(-k, k)); wide k spreads "
                    "the neuron ensemble across the activation period (paper-range default 10.0). "
                    "Only read when --finer-bias-init is on; must be > 0.")
    # SEG LOSS / CURRICULUM
    ap.add_argument("--seg-loss", choices=["ce", "tau_softplus", "l7_softplus", "margin_hinge"], default="ce")
    ap.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tau-softplus-start-epoch", type=int, default=300)
    ap.add_argument("--l7-start-epoch", type=int, default=800)
    # (#292 build-2) EVENT-TRIGGERED CURRICULUM: fire the CE->tau_softplus->l7_softplus transitions on
    # loss-plateau CONVERGENCE (deterministic, from the SYNCHRONOUS ep_loss -- never the async d_seg
    # verdict) instead of the hardcoded epochs above, which then act as HARD CAPS (the trigger only
    # fires EARLIER, never later). DEFAULT OFF => _seg_form_for_epoch is called unchanged => BYTE-
    # IDENTICAL to the hardcoded schedule (the #205-safe path). Operator 2026-07-04 "epochs should not
    # be hardcoded but dynamical according to convergence trajectory".
    ap.add_argument("--curriculum-event-triggered", action=argparse.BooleanOptionalAction, default=False,
                    help="#292: fire curriculum stage transitions on ep_loss-plateau convergence (caps = "
                    "--tau-softplus-start-epoch / --l7-start-epoch). Default OFF => byte-identical hardcoded.")
    ap.add_argument("--curriculum-min-stage-epochs", type=int, default=150,
                    help="#292: min COMPLETED epochs in a stage before an event-triggered transition may fire.")
    ap.add_argument("--curriculum-plateau-rel-eps", type=float, default=1e-4,
                    help="#292/#302: |relative least-squares slope| threshold (slope/mean over the window) for "
                    "plateau. RECALIBRATED 1e-3->1e-4 (T3 symposium 2026-07-05 C1): 1e-3 fired ep151 MID-DESCENT "
                    "on #205 (rel slope -8.2e-4 while d_seg still falling 0.0055->0.0048) = 15%% CE-floor loss; 1e-4 "
                    "separates ep275+ (true plateau) from ep150. Consumed ONLY on the event-trigger / readiness "
                    "path (both default OFF) => byte-identical for default runs.")
    ap.add_argument("--curriculum-plateau-windows", type=int, default=4,
                    help="#292: number of trailing within-stage ep_loss values in the plateau slope window.")
    # (#302 curriculum-derivation) PER-CLASS CRITICAL-NUCLEUS GUARD (the CE->tau hand-off law's
    # completion) + BOUNDARY RE-ANCHOR + READINESS TELEMETRY. All DEFAULT-OFF (run-3 targets); the
    # #205 path never engages them => byte-identical.
    ap.add_argument("--curriculum-nucleus-guard", action=argparse.BooleanOptionalAction, default=False,
                    help="#302: gate the CE->tau CONVERGENCE fire on the MEASURED per-class critical nucleus "
                    "(every scored class BORN part_frac>0 AND within-class flip <= --curriculum-nucleus-within-flip), "
                    "updated at verdict cadence. A plateau alone is NECESSARY-NOT-SUFFICIENT (Allen-Cahn: MCF "
                    "erodes a below-nucleus class, never grows it). The cap still fires unconditionally (never "
                    "hangs). Default OFF => nucleus_ready stays True => the trigger is the pure-loss #292 build-2.")
    ap.add_argument("--curriculum-nucleus-within-flip", type=float, default=0.5,
                    help="#302: per-class within-flip threshold a class must be AT/BELOW to count as above nucleus "
                    "(FORMED). Emitted per verdict in the handoff_readiness row so the NEXT run calibrates it "
                    "empirically (the theoretical knee is pi1=w/sigma~5; this is the operational proxy).")
    ap.add_argument("--curriculum-nucleus-min-part-frac", type=float, default=0.0,
                    help="#302: per-class predicted partition-fraction a class must EXCEED to count as BORN "
                    "(the MCF-cannot-nucleate gate). Default 0.0 = strictly nonzero predicted mass.")
    ap.add_argument("--handoff-readiness-telemetry", action=argparse.BooleanOptionalAction, default=False,
                    help="#302: emit the per-class handoff_readiness telemetry row per verdict (part_frac + "
                    "within_flip + nucleus_ok per class + plateau_ok + ready) WITHOUT gating the trigger. "
                    "OBSERVABILITY-FIRST: run it on a normal (event-OFF) run to passively yield the per-class "
                    "validation data the hand-off law needs. Pure telemetry, NEVER read into training => "
                    "byte-identical. Implied ON when --curriculum-nucleus-guard is set (the guard needs the counts).")
    ap.add_argument("--curriculum-reanchor-levers", action=argparse.BooleanOptionalAction, default=False,
                    help="#302 (M1): under event-triggering, re-anchor the TAU-RELATIVE wall-clock levers "
                    "(persistence-warmup completion, seed-anneal withdrawal, analytic-band engage) to the FIRED "
                    "tau boundary instead of their calibrated ep300-relative epochs (a shift, not a rescale). "
                    "Requires --curriculum-event-triggered. Unfired / fired-at-cap / OFF => epochs unchanged => "
                    "byte-identical. hosc-beta is NOT re-anchored (its beta=4 freeze is Muon-anchored; that "
                    "re-anchor waits on the Muon-event-trigger build, symposium C.ii item 5).")
    # (#292 build-3) CLOSED-LOOP LEVER CONTROL: at each eval point, JOIN the async verdict (the
    # deterministic d_seg VALUE, never thread timing), classify the within-stage trend with the SAME
    # sustained-erosion-vs-transient math as tools/witness_control_monitor, and on SUSTAINED
    # DIVERGING_ERASING take BOUNDED action: (a) step the effective eikonal weight up (composes with
    # the build-1 schedule, capped at --closed-loop-eikonal-max, at most --closed-loop-max-bumps
    # times); (b) after the bump budget is spent, if erosion persists --closed-loop-stop-after-windows
    # consecutive eval-windows, EARLY-STOP cleanly (best EMA-shadow ckpt already preserved). DEFAULT
    # OFF => no capture, no bump, no stop, no sidecar keys => BYTE-IDENTICAL (the #205-safe path).
    # Tier-3 operator 2026-07-04: "closed-loop monitor->lever control, ramp eikonal/lane-prior on
    # creep" + "self convergence + early termination using a mathematical system". CONTAINMENT: the
    # loop never launches anything; it only mutates the in-run eikonal + arms a clean stop.
    ap.add_argument("--closed-loop-control", action=argparse.BooleanOptionalAction, default=False,
                    help="#292 build-3: closed-loop d_seg-trend monitor->lever control (bounded eikonal "
                    "bump on sustained erosion + early-stop after budget). Default OFF => byte-identical.")
    ap.add_argument("--closed-loop-eikonal-bump", type=float, default=0.05,
                    help="#292 build-3: eikonal weight ADDED per sustained-erosion bump (bounded step).")
    ap.add_argument("--closed-loop-eikonal-max", type=float, default=0.20,
                    help="#292 build-3: HARD CAP on the effective eikonal weight (schedule + bumps); the "
                    "cap is floored at the scheduled value so it can never pull BELOW the schedule.")
    ap.add_argument("--closed-loop-max-bumps", type=int, default=2,
                    help="#292 build-3: max eikonal bumps per run (the bounded actuation budget).")
    ap.add_argument("--closed-loop-stop-after-windows", type=int, default=3,
                    help="#292 build-3: consecutive post-budget sustained-erosion eval-windows before the "
                    "clean early-stop is armed (best ckpt preserved).")
    ap.add_argument("--closed-loop-min-sustained-windows", type=int, default=3,
                    help="#292 build-3: within-stage verdicts a d_seg rise must persist (with net-stage "
                    "slope > 0) before it counts as EROSION rather than a recoverable boundary transient "
                    "(matches tools/witness_control_monitor classify_trajectory).")
    ap.add_argument("--tau-softplus-tau", type=float, default=0.3)
    ap.add_argument("--l7-mult", type=float, default=4.0)
    ap.add_argument("--l7-threshold", type=float, default=1.0)
    ap.add_argument("--margin-target-end", type=float, default=0.5)
    # LEVER-3 (lane-edge fragility weighting): up-weight class-1 (Lane) flips in the REALIZED margin
    # hinge. Lane is thin all-boundary double-edges (19% of d_seg flips) under-fit by the unweighted
    # CE baseline. Default 0.0 = OFF = current behavior (fully additive). When >0, costs a 2nd
    # realized seg forward (acceptable per operator "score > training time"). SegNet class order is the
    # comma10k CANONICAL order (MEASURED 2026-06-27; CLAUDE.md NON-NEGOTIABLE): [Road0, Lane1,
    # Undrivable2, Movable3, MyCar4]. The luma-sort [Road0,Lane1,MyCar2,Undriv3,Movable4] is FORBIDDEN/
    # WRONG for 2/3/4. Class 0=Road & 1=Lane CONFIRMED in both (the lever uses only class 1). LEVER-4
    # (class-agnostic margin-saliency) is PREFERRED as it sidesteps the class index entirely.
    ap.add_argument("--lane-edge-weight", type=float, default=0.0,
                    help="LEVER-3: weight on the additive realized lane-class margin hinge (0=off).")
    ap.add_argument("--lane-edge-class", type=int, default=1,
                    help="LEVER-3: GT class index to up-weight (1=Lane, CONFIRMED; comma10k CANONICAL "
                    "order [Road0,Lane1,Undrivable2,Movable3,MyCar4] for 2/3/4 -- NOT the forbidden luma-sort).")
    ap.add_argument("--lane-margin-target", type=float, default=0.5,
                    help="LEVER-3: target decision margin for the lane hinge relu(target - margin).")
    # #218 MARGIN-FIELD HEAD levers (facets 1 & 3, BYTE-FREE; see src/tac/boundary_math/laguerre_logit_offset.py
    # + experiments/probe_laguerre_logit_offset_sweep.py). Default (head=softmax, weight 0) => byte-identical.
    ap.add_argument("--head", choices=["softmax", "etf", "additive-margin"], default="softmax",
                    help="#218 facet-1: out_sdf head geometry. 'etf'=fixed simplex-ETF weight (frozen, "
                    "byte-free + rate-win, neural-collapse minority-norm fix). 'additive-margin'=use the AM "
                    "realized-margin hinge target from --additive-margin. 'softmax'=default (byte-identical).")
    ap.add_argument("--additive-margin", type=float, default=0.0,
                    help="#218 facet-1b: AM-softmax margin (target realized SegNet decision margin) fed to the "
                    "margin-field hinge when --head additive-margin.")
    ap.add_argument("--logit-adjust-per-class", action="store_true",
                    help="#218 facet-3 (Menon 2007.07314): raise the realized-margin target for RARE classes "
                    "(Lane/Movable) by tau*relu(-log pi_c). Byte-free. Needs --margin-field-head-weight>0.")
    ap.add_argument("--logit-adjust-tau", type=float, default=1.0, help="#218 facet-3 tau scale.")
    ap.add_argument("--margin-field-head-weight", type=float, default=0.0,
                    help="#218 facets 1b/3 loss weight for the realized through-R per-class margin hinge "
                    "(0.0=off=byte-identical). Composes with LEVER-3/4/B on the shared _signed.")
    ap.add_argument("--lane-edge-start-epoch", type=int, default=0,
                    help="LEVER-3 OPTIMAL-FORM: engage the lane hinge only at ep>=this (0=from ep1=current "
                    "behavior). Gate to the tau_softplus/l7 margin stage (e.g. 300) to avoid the "
                    "margin-from-scratch-starves-interior failure; the engage epoch re-treats the spike-guard.")
    # LEVER-A (FiLM-RANK-FIX, ADDITIVE, ALL DEFAULT-OFF). Attacks the MEASURED per-pair FiLM modulation
    # participation-ratio collapse (3.34@CE -> 1.27@tau -> 1.19@l7: 91.8% of per-pair variation in ONE
    # axis -> the decoder receives ~1 effective per-pair direction -> caps d_seg AND held-out
    # amortization). All-off => byte-identical to the pre-LEVER-A witness (the extra submodules / loss
    # term are not created). See build_levelset_rgb_witness + the rank-floor branch in total_loss_fn.
    ap.add_argument("--film-per-layer", action="store_true",
                    help="LEVER-A1 [CAPACITY, NOT rank -- review M2/FEED-ht]: add SEPARATE per-layer "
                    "RESIDUAL FiLM projections (identity at init). +~25k params (~+0.01 rate). MEASURED "
                    "(M2): does NOT raise modulation rank -- A1/A2/shared-FiLM are all functions of the "
                    "SAME mod_dim code, so PR(M) <= rank(codes) <= mod_dim regardless of capacity. The "
                    "byte-FREE rank lever is --film-stiefel (+ --code-spectral-entropy-weight): PR(M) "
                    "1.19->4.57 at 0 added bytes. Prefer those. Default OFF = shared-FiLM-only.")
    ap.add_argument("--film-concat-code", action="store_true",
                    help="LEVER-A2 [CAPACITY, NOT rank -- review M2/FEED-ht]: add an ADDITIVE per-pair "
                    "code-injection route (folded concat; identity at init). +~12k params. Same mod_dim "
                    "rank ceiling as A1 (cannot raise PR(M) above rank(codes)); use --film-stiefel for "
                    "the byte-free rank fix. Default OFF.")
    ap.add_argument("--film-rank-floor-weight", type=float, default=0.0,
                    help="LEVER-A3 [DOMINATED by --film-stiefel; NOT recommended -- review FEED-ht/M1]: "
                    "weight of a SOFT participation-ratio FLOOR penalty relu(target-PR) on M=film(code). "
                    "0.0 (default) = OFF. CAVEAT (review M1): the PR measure is 0-homogeneous so its grad "
                    "~1/||M|| can blow up at small codes (no warm-in/start-gate here) and proxy-games "
                    "low-gain directions. Prefer the byte-free --film-stiefel (+ --code-spectral-entropy-"
                    "weight), which makes PR(M)=PR(cov(code)) hold by construction. Kept for ablation only.")
    ap.add_argument("--film-rank-floor-target", type=float, default=4.0,
                    help="LEVER-A3: the participation-ratio FLOOR (effective-dim target) the penalty pushes "
                    "M toward (must be > 1 when --film-rank-floor-weight > 0; PR >= 1 always). Default 4.0.")
    # DM1 minimal cure (design memo per_stage_fractal_optimizer_priming_reheat_anneal_20260629 §0/§4).
    # Two byte-free structural moves that make PR(M)=PR(cov(code)) hold to the projection's ~1e-2
    # residual (Stiefel isometry) + keep the code spectrum spread. Both DEFAULT-OFF => no new params,
    # the train step + loss branches are skipped => byte-identical to the pre-DM1 path.
    ap.add_argument("--film-stiefel", action="store_true",
                    help="DM1a: each optimizer step, project film.weight (W) onto the Stiefel manifold of "
                    "ORTHONORMAL COLUMNS (WᵀW=I) via the cubic Newton-Schulz polar W(WᵀW)^-1/2. Then W is "
                    "an isometry => PR(M)=PR(cov(code)) to the projection's ~1e-2 residual (the resonance "
                    "cannot concentrate through W). Re-normalizing columns each step also neutralizes the "
                    "global-magnitude component of AdamW weight-decay on W (the design's 'WD=0 on W' "
                    "intent) WITHOUT touching the optimizer. Default OFF = byte-identical.")
    ap.add_argument("--code-spectral-entropy-weight", type=float, default=0.0,
                    help="DM1b: weight beta of a CAPACITY spectral-entropy penalty -beta*log(PR(cov(code))) "
                    "on the per-pair code covariance, keeping all ~mod_dim code directions live (the other "
                    "half of the byte-free FiLM rank-collapse cure; via WᵀW=I this raises PR(M)). PR is "
                    "(D,D)-Gram-computed (no eigendecomposition). 0.0 (default) = OFF = byte-identical.")
    ap.add_argument("--dm1-telemetry", action="store_true",
                    help="(review C1) FORCE the dm1_telemetry row (PR(M) live+shadow, PR(cov code), "
                    "Stiefel residual) at eval cadence EVEN when no DM1 lever is active -- so the A0 "
                    "BASELINE logs the PR-collapse half of the firewall verdict (else 'baseline "
                    "collapses' is unmeasurable). Pure READ (no model/grad touch); default OFF => "
                    "the row only fires when --film-stiefel/--code-spectral-entropy-weight is on => "
                    "BIT-IDENTICAL observability to the pre-C1 path.")
    # LEVER-B (THIN-LANE DROPPED-DASH PRIOR, ADDITIVE, DEFAULT-OFF). Attacks the MEASURED dominant
    # residual: 57% Road<->Lane confusion, PC0 (34.5% of residual variance) = Lane->Road DROP, 52.7% of
    # GT-lane connected components WHOLESALE-MISSED, miss-fraction monotone in dash size (<5px 93%
    # missed). The unweighted mean seg loss UNDER-fits thin 3px dashes. This up-weights the realized
    # through-R margin hinge on THIN GT-lane pixels (a precomputed local-lane-density weight map). NOTE:
    # distinct from --lane-prior-phi1 (the structured-init lane SDF prior); this is the --lane-thin-*
    # realized-margin prior. Default lane_thin_weight=0.0 = OFF = byte-identical.
    ap.add_argument("--lane-thin-weight", type=float, default=0.0,
                    help="LEVER-B: weight of the realized through-R thin-lane margin hinge (up-weights "
                    "thin/dropped GT-lane dashes). 0.0 (default) = OFF.")
    ap.add_argument("--lane-thin-class", type=int, default=1,
                    help="LEVER-B: the lane class index in the comma10k CANONICAL order "
                    "[Road0,Lane1,Undrivable2,Movable3,MyCar4]. Default 1 (Lane).")
    ap.add_argument("--lane-thin-radius", type=int, default=4,
                    help="LEVER-B: half-width of the (2r+1)^2 window for the local-lane-density thinness "
                    "measure (thin dashes => low local density => high weight). Default 4.")
    ap.add_argument("--lane-thin-target", type=float, default=0.5,
                    help="LEVER-B: the decision-margin target for the thin-lane hinge relu(target-margin). "
                    "Default 0.5 (matching --lane-margin-target).")
    ap.add_argument("--lane-thin-start-epoch", type=int, default=0,
                    help="LEVER-B: engage the thin-lane hinge only at ep>=this (0=from ep1). Gate to the "
                    "tau/l7 margin stage (e.g. 300) to avoid margin-from-scratch starvation; the engage "
                    "epoch re-treats the spike-guard.")
    # LEVER-4 (MARGIN-SALIENCY weighting, DAG FEED-eq, ADDITIVE, DEFAULT-OFF). GENERALIZES LEVER-3
    # from the class-1-only mask to the ALL-CLASS flip-prone band: the realized through-R decision
    # margin hinge is weighted PER-PIXEL by the GT-margin fragility saliency sal=exp(-gt_margin/tau)
    # (small GT margin = near a decision boundary = flip-prone; ~1 at the boundary annulus, ->0 in the
    # confident interior). MEASURED (FEED-eq, gt_n96, band 0.5): the flip-prone band is Road 47% / Lane
    # 19% / Undrivable 14% / Movable 9% / MyCar 11% -> LEVER-3 (class 1) defends only 19% of it; this
    # all-class saliency defends 100%. CLASS-AGNOSTIC (weights by fragility, not class index) so it
    # sidesteps the class-order dispute entirely. Default 0.0=OFF=byte-identical. When >0, costs ONE
    # realized seg forward (a 2nd if LEVER-3 is also on; nobody runs both). Fridrich square-root-law:
    # spread small corrections across the boundary, do not concentrate. NO scorer weights ship (the
    # saliency is computed from the PROVIDED frozen scorer at train time; rule-118 FREE).
    ap.add_argument("--margin-saliency-weight", type=float, default=0.0,
                    help="LEVER-4: weight on the additive ALL-CLASS GT-margin-saliency-weighted realized "
                    "margin hinge (0=off; generalizes --lane-edge-weight to every inter-class edge).")
    ap.add_argument("--margin-saliency-tau", type=float, default=0.5,
                    help="LEVER-4: GT-margin saliency softness sal=exp(-gt_margin/tau); smaller tau = "
                    "tighter focus on the most fragile (smallest-margin) boundary pixels. ~p1 of the "
                    "GT-margin dist (gt_n96 p1~0.38, p5~2.16) keeps the weight on the flip-prone band.")
    ap.add_argument("--margin-saliency-target", type=float, default=0.5,
                    help="LEVER-4: target decision margin for the saliency hinge relu(target - margin).")
    ap.add_argument("--margin-saliency-start-epoch", type=int, default=0,
                    help="LEVER-4 OPTIMAL-FORM: engage only at ep>=this (0=from ep1). Gate to the "
                    "tau_softplus/l7 margin stage to avoid margin-from-scratch-starves-interior; the "
                    "engage epoch re-treats the spike-guard (same discipline as --lane-edge-start-epoch).")
    ap.add_argument("--margin-saliency-uniward", action="store_true",
                    help="LEVER-4 UNIWARD (Fridrich inverse-steganalysis): additionally DOWN-weight the "
                    "saliency in TEXTURED regions (SegNet-undetectable) so capacity concentrates on the "
                    "SMOOTH flip-prone boundary. Texture energy from the realized frame's spatial "
                    "gradients (stop-grad WEIGHT). Default off.")
    ap.add_argument("--margin-saliency-uniward-beta", type=float, default=4.0,
                    help="LEVER-4 UNIWARD: texture down-weight strength sal /= (1 + beta*tex_norm).")
    ap.add_argument("--margin-saliency-reachability", action="store_true",
                    help="LEVER-4 REACHABILITY (REPLACES the UNIWARD texture path when set): multiply the "
                    "fragility saliency by the cached THROUGH-R fragility-weighted margin-Jacobian S_R "
                    "(reachability of the CORRECT answer at the GT target frame) instead of 1/(1+beta*tex). "
                    "The texture proxy was MEASURED inert (Pearson -0.033 vs S_R, top-5%% Jaccard 0.024 = "
                    "statistical chance, mildly misdirects); S_R lives on the fragile margin band where the "
                    "d_seg debt is. Requires an 'sR' key in --gt-cache (build via "
                    "tools/precompute_sR_reachability.py). Default OFF => byte-identical (texture path "
                    "unchanged). NOT supported with --micro-batch-pairs>1 (serial path only; fails closed).")
    # LEVER-4b SUB-PIXEL BOUNDARY-PLACEMENT `t` (asymmetry probe a8afad40 GREEN 2026-07-03; DIRECTIONAL
    # upgrade of LEVER-4 #141; ADDITIVE, DEFAULT-OFF). Supervises the witness's realized margin ratio
    # t_wit = Mw[p]/(Mw[p]+Mw[q]) toward the FREE GT cross-boundary margin ratio t = M_GT[p]/(M_GT[p]+
    # M_GT[q]) on genuine-V straddles (a denser sub-pixel placement signal than the argmax weight). Reuses
    # the SHARED realized through-R margin (no 2nd SegNet forward). subpix_w=0.0 (DEFAULT) => byte-identical.
    ap.add_argument("--seg-subpix-boundary-weight", type=float, default=0.0,
                    help="LEVER-4b: weight on the additive sub-pixel boundary-placement loss "
                    "(t_wit - t_GT)^2 over genuine-V inter-class straddles (0=off). The GT target "
                    "t=M_GT[p]/(M_GT[p]+M_GT[q]) is a FREE sub-pixel localizer latent in the GT margin "
                    "field; supervises the witness's OWN realized margin ratio. Reuses the SHARED "
                    "LEVER-4 through-R margin forward. NOT supported with --micro-batch-pairs>1 (fails closed).")
    ap.add_argument("--seg-subpix-boundary-v-band", type=float, default=1.0,
                    help="LEVER-4b: genuine-V flip-band. A straddle qualifies only when BOTH GT margins "
                    "are < this (t is meaningful only where the margin V is clean). MEASURED gt_n96: "
                    "band 1.0 -> ~2196 active px/frame (1.12%% of px), t mean 0.527 std 0.263 "
                    "(informative ~Uniform); the straddle set saturates by ~2.0 (boundary pixels are "
                    "already low-margin on both sides).")
    ap.add_argument("--seg-subpix-boundary-start-epoch", type=int, default=0,
                    help="LEVER-4b OPTIMAL-FORM: engage only at ep>=this (0=from ep1). Gate to the "
                    "tau_softplus/l7 margin stage (placement is meaningful once the argmax is roughly "
                    "correct); the engage epoch re-treats the spike-guard (same discipline as LEVER-4).")
    # LEVER-4c ANNULUS-DIRECTED CHROMA-SHARPENING (chroma DOF probe a3e9f0bd GREEN 2026-07-03; operator
    # 2026-06-25 "Chroma too"; ADDITIVE, DEFAULT-OFF). At the fragile margin annulus supervise the
    # witness's OWN rendered chroma (rgb - BT.601-luma, LUMA-INVARIANT) toward the GT chroma so the
    # per-pixel RGB head paints the boundary chroma the near-per-class-constant palette can't. Reuses the
    # SHARED realized-through-R rendered frame _f1 (no 2nd render/SegNet). chroma_bnd_w=0 => byte-identical.
    ap.add_argument("--seg-chroma-boundary-weight", type=float, default=0.0,
                    help="LEVER-4c: weight on the additive chroma-MATCH loss ||chroma(_f1)-chroma(GT)||^2 "
                    "over the fragile margin annulus (0=off). chroma := rgb - BT.601-luma (LUMA-INVARIANT, "
                    "so ORTHOGONAL to every luma lever; NOT a full-RGB reconstruction). GT chroma target is "
                    "the camera GT bilinear-resized to SegNet-input res (what SegNet reads). Reuses the "
                    "SHARED realized-through-R render _f1. Chroma is a PROVEN independent d_seg BOUNDARY "
                    "SHARPENER (probe a3e9f0bd: 93.4%% of chroma-flips in the margin<1 annulus). NOT "
                    "supported with --micro-batch-pairs>1 (fails closed).")
    ap.add_argument("--seg-chroma-boundary-margin-band", type=float, default=1.0,
                    help="LEVER-4c: fragile-annulus band. A pixel is supervised only where the GT top1-top2 "
                    "margin is < this (chroma's d_seg power is at the knife-edge). MEASURED gt_n96: band 1.0 "
                    "captures 93.4%% of chroma-flips (->33.7%% at 0.25).")
    ap.add_argument("--seg-chroma-boundary-start-epoch", type=int, default=0,
                    help="LEVER-4c OPTIMAL-FORM: engage only at ep>=this (0=from ep1). Gate to the "
                    "tau_softplus/l7 margin stage (chroma-boundary supervision is meaningful once the "
                    "argmax is roughly seated); the engage epoch re-treats the spike-guard (same "
                    "discipline as LEVER-4b).")
    # SPIKE-AWARE seg REWEIGHT (source-split MEASURED n600 2026-07-03; ADDITIVE, DEFAULT-OFF). Reweight
    # the per-pixel base seg CE by a theta-INDEPENDENT map from the GT argmax TEMPORAL neighbors: a SPIKE
    # pixel (lstar[t] != lstar[t-1] AND != lstar[t+1]) is single-frame argmax FLICKER a per-frame witness
    # structurally CANNOT fit. MEASURED: the flicker is ~88.6%% IRREDUCIBLE appearance-change (spike luma
    # temporal-delta 34 vs 4 stable = 8.4x) -> a SMOOTH witness is PROVABLY optimal there (d_seg=q(1-r)+
    # r(1-q), min r=0 for q<0.5). DOWN-weight the unfittable flicker gradient; UP-weight the COHERENT
    # temporally-consistent boundary (the winnable residual). Default scalars 1.0/1.0 => map==1.0 =>
    # BYTE-IDENTICAL even with --seg-spike-reweight set. MODEST headroom (live residual d_seg ~ the
    # popout floor; benefit is 2nd-order reallocation) -> an A/B arm, NOT a claim. Store-the-flicker is
    # net-NEGATIVE (rate +0.56 > d_seg 0.52) and the REPLICATE alternative is not warranted (predictable
    # fraction ~11.4%%, weak ego-coupling r=0.16). NOT supported with --micro-batch-pairs>1 (fails closed).
    ap.add_argument("--seg-spike-reweight", action="store_true",
                    help="Enable the spike-aware per-pixel seg-CE reweight (DEFAULT OFF => byte-identical). "
                    "Down-weights unfittable single-frame argmax flicker, up-weights the coherent boundary.")
    ap.add_argument("--seg-spike-downweight", type=float, default=1.0,
                    help="Per-pixel seg-loss weight at SPIKE (single-frame flicker) pixels. <1.0 down-weights "
                    "the unfittable flicker gradient. 1.0 (DEFAULT) => no change (byte-identical).")
    ap.add_argument("--seg-coherent-upweight", type=float, default=1.0,
                    help="Per-pixel seg-loss weight at COHERENT (temporally-consistent, unstable) boundary "
                    "pixels. >1.0 concentrates capacity on the winnable residual. 1.0 (DEFAULT) => no change.")
    # FOCAL-GAMMA + BOUNDARY-DISTANCE seg-loss levers (council levelset-loss-geometry symposium
    # 2026-07-05, PROCEED_WITH_REVISIONS; BUILT default-OFF, READY, NOT deployed — the pre-registered
    # fire criterion (ep50->100 witness-alone slope flattening, |d(d_seg)| < 0.02 per 25ep window
    # with islands still >50%% of residual) is the PARENT's decision, never auto-fired here).
    # gamma* comes MEASURED from experiments/probe_focal_gamma_calibration.py (never guessed).
    ap.add_argument("--seg-focal-gamma", type=float, default=0.0,
                    help="Focal reweight (1-p_y)^gamma on the BASE per-pixel seg loss (all seg forms), "
                    "p_y = realized softmax GT prob on the SAME surface the base loss reads (the "
                    "render_fn-composed frame). STOP-GRAD + mean-1 renormalized (gradient-BUDGET "
                    "reallocation; Rudin readback: weight ratio p=0.5 vs p=0.9 is exactly 5^gamma). "
                    "0.0 (DEFAULT) => branch never built => byte-identical. Emits a per-epoch "
                    "{'stage':'focal','island_grad_share':...} observability row when >0. "
                    "NOT supported with --micro-batch-pairs>1 (fails closed).")
    # (#218 BUILD, FEED-07b lever #3) class-prior LOGIT ADJUSTMENT (Menon et al. 2021).
    ap.add_argument("--logit-adjust-loss-tau", type=float, default=0.0,
                    help="#218/FEED-07b: class-prior logit adjustment (Menon et al. 2021, arXiv "
                    "2007.07314) on the TRAINING seg loss — the frozen-SegNet logits base_loss "
                    "reads get logits_c += tau*log(prior_c), priors = GT class-area fractions from "
                    "the cached L* (measured n600 ~[0.232, 0.0059, 0.495, 0.0124, 0.254]); the "
                    "textbook ZERO-BYTE rare-class (Lane/Movable) cure. TRAINING-time LOSS surface "
                    "ONLY: the deployed/rendered argmax path (verdict CPU-torch SegNet, byte-close "
                    "decode, inflate) is UNCHANGED (raw logits). 0.0 (DEFAULT) => the loss adapter "
                    "is the SAME object => byte-identical. tau=1.0 is the canonical Menon setting. "
                    "NOT supported with --micro-batch-pairs>1 (fails closed). SISTER of (do not "
                    "confuse with) the #218 facet-3 pair --logit-adjust-per-class + "
                    "--logit-adjust-tau, which boost the MARGIN-FIELD-HEAD per-class TARGET "
                    "(fires only with --margin-field-head-weight>0); THIS flag adjusts the BASE "
                    "seg-LOSS logits themselves. The two compose.")
    ap.add_argument("--boundary-distance-weight", type=float, default=0.0,
                    help="SDF-native Kervadec-style boundary-placement loss: band-weighted mean "
                    "|phi_GT - phi_runner| on the GT inter-class boundary band (distance transform "
                    "per pair, computed ONCE from the cached GT argmax; band ramp = 2 px, the "
                    "measured 1-2 px flip band). Read off model.sdf on frame1 directly (the contour "
                    "DOF the witness owns; Mallat's move-the-contour). 0.0 (DEFAULT) => provider not "
                    "built, branch skipped => byte-identical. NOT supported with "
                    "--micro-batch-pairs>1 (fails closed).")
    # LEVER-5 (per-pair HARDNESS-weighted code-fit / training, DAG FEED-eq, ADDITIVE, DEFAULT-OFF).
    # WATERFILL the per-epoch pair-iteration budget toward HARD pairs (high d_seg debt). The frozen-
    # decoder code-fit fits independent per-pair codes, so giving a hard pair MORE update STEPS (not a
    # bigger loss scale -- Adam normalizes per-pair loss-scale to ~no-op) converges its codes further.
    # Mechanism: each epoch keeps the full permutation(P) (every pair >=1 step, never starved) PLUS
    # round(P*oversample) EXTRA steps drawn ~ hardness^power. The FAIR A/B at fixed --hardness-oversample
    # is --hardness-weighted on (extras ~ hardness) vs off (extras uniform): SAME total steps, different
    # allocation. Default --hardness-oversample 0.0 => no extras => byte-identical. MEASURED CAVEAT
    # (FEED-eq): per-pair GT-margin hardness spread on gt_n96 is only 1.31x (the fragile band is ~1.3%
    # of pixels per pair, nearly constant) -> margin-source reallocation is modest; --hardness-source
    # realized (per-pair baseline realized d_seg, which varies with the frozen decoder's per-pair
    # reconstruction quality) is the SHARPER signal for the code-fit and is the recommended source.
    ap.add_argument("--hardness-oversample", type=float, default=0.0,
                    help="LEVER-5: extra per-epoch pair-iteration steps as a fraction of P (0=off="
                    "byte-identical; e.g. 0.5 = +50%% steps, allocated by --hardness-weighted).")
    ap.add_argument("--hardness-weighted", action="store_true",
                    help="LEVER-5: draw the --hardness-oversample extra steps ~ per-pair hardness^power "
                    "(on) vs uniformly (off). On = waterfill hard pairs more code-fit budget.")
    ap.add_argument("--hardness-source", choices=["margin", "realized"], default="margin",
                    help="LEVER-5 hardness signal: 'margin' = $0 cached GT small-margin pixel fraction "
                    "(weak 1.31x spread); 'realized' = one-time per-pair baseline realized d_seg over ALL "
                    "pairs (CPU, no GPU contention; sharper; the recommended code-fit source).")
    ap.add_argument("--hardness-power", type=float, default=1.0,
                    help="LEVER-5: sharpness exponent on the per-pair hardness sampling probability.")
    ap.add_argument("--hardness-band", type=float, default=0.5,
                    help="LEVER-5 (margin source): GT-margin threshold defining a flip-prone pixel for "
                    "the per-pair hardness = mean(gt_margin < band).")
    # LEVEL-SET REG
    # (EIK-STAB build 1) the two candidate eikonal-runaway CURES (litsweep DOMAIN 1), both
    # default-OFF => byte-identical; the n24 arbitration probe decides which (if either) rides
    # the GO-gated relaunch. steik = ADDITIVE StEik directional-divergence damping
    # (arXiv 2305.18414: mean |grad m^T H(m) grad m|, damps only the normal-direction unstable
    # mode). viscosity = ViscoReg vanishing-viscosity residual (arXiv 2507.00412: REPLACES
    # (|grad m|-1)^2 with (|grad m|-1-eps*Lap m)^2 while eps>0; anneal eps -> 0).
    ap.add_argument("--eikonal-steik-weight", type=float, default=0.0,
                    help="EIK-STAB build 1a: StEik directional-divergence stabilizer weight "
                    "(additive; 0 = OFF = byte-identical).")
    ap.add_argument("--eikonal-steik-normalized", action="store_true",
                    help="V6 #317 EIK-STAB build 1a-N: use the NORMALIZED unit-normal curvature "
                    "n^T H n = (grad m^T H grad m)/(|grad m|^2+eps) instead of the raw "
                    "|grad m^T H grad m| (removes the quartic |grad m|^2 self-amplification measured "
                    "NO-GO at the far-from-SDF resumed state). Requires --eikonal-steik-weight>0.")
    ap.add_argument("--eikonal-steik-norm-eps", type=float, default=1e-2,
                    help="V6 #317: normal-direction regularizer eps in n^T H n = dir_div/(|grad m|^2+eps) "
                    "(default 1e-2: leaves the |grad m|~1 boundary annulus ~intact, suppresses flat "
                    "argmax-stable interior). Only used when --eikonal-steik-normalized.")
    ap.add_argument("--eikonal-viscosity", type=float, default=0.0,
                    help="EIK-STAB build 1b: ViscoReg viscosity eps (replaces the eikonal residual "
                    "with the viscous form while >0; 0 = OFF = byte-identical).")
    ap.add_argument("--eikonal-viscosity-anneal", type=int, default=0,
                    help="EIK-STAB build 1b: linear eps decay to 0 over this many ABSOLUTE epochs "
                    "(0 = constant eps; ViscoReg-style vanishing viscosity).")
    ap.add_argument("--eikonal-viscosity-adaptive", action="store_true",
                    help="V6 #320 (DE #318 §4 Arm-2 / symposium #317 §7.4): REPLACE the linear "
                    "--eikonal-viscosity-anneal with the ADAPTIVE-eps CFL-edge tracker eps(t) = "
                    "clamp(|c_a(t)|*sqrt(eta*lambda_eik/8)*(1+margin), floor, upper), recomputed "
                    "per-epoch. |c_a(t)| = mean|(|grad m|-1)/|grad m|| measured no-grad on the witness "
                    "decision margin (the DERIVED mechanism cure for the ep110 eikonal re-entry: a "
                    "FIXED eps falls below the RISING lower edge as sharpening grows |c_a|). Requires "
                    "--eikonal-viscosity>0 (the visco term must be active). Default OFF = the linear "
                    "anneal path = BYTE-IDENTICAL.")
    ap.add_argument("--eikonal-visco-eps-floor", type=float, default=0.3,
                    help="V6 #320: adaptive-eps LOWER clamp (never anneal below the FEED-05v measured "
                    "stable floor 0.3). Only used with --eikonal-viscosity-adaptive.")
    ap.add_argument("--eikonal-visco-eps-upper", type=float, default=0.7,
                    help="V6 #320: adaptive-eps UPPER clamp (stay below the eps=1.0 biharmonic "
                    "explosion measured at n24/FEED-05v). Only used with --eikonal-viscosity-adaptive.")
    ap.add_argument("--eikonal-visco-margin-factor", type=float, default=0.5,
                    help="V6 #320: adaptive-eps safety margin above the CFL lower edge "
                    "(eps = edge*(1+margin_factor); DE #318 §7.4 margin~0.5). "
                    "Only used with --eikonal-viscosity-adaptive.")
    ap.add_argument("--eikonal-visco-ca-pairs", type=int, default=16,
                    help="V6 #320: number of FIXED (strided) pairs the per-epoch no-grad |c_a| "
                    "measurement forwards over (cheap; witness-only, no SegNet). "
                    "Only used with --eikonal-viscosity-adaptive.")
    ap.add_argument("--eikonal-visco-ca-band", type=float, default=0.0,
                    help="V6 #320: if >0, restrict |c_a| to the small-margin annulus |m|<band "
                    "(DE #318 §2 flat regime). Default 0.0 = interior mean = symposium §7.4 exact "
                    "launch formula. Only used with --eikonal-viscosity-adaptive.")
    ap.add_argument("--eikonal-weight", type=float, default=0.01, help="Eikonal |grad phi|->1 (topology bias, small).")
    ap.add_argument("--eikonal-weight-end", type=float, default=None,
                    help="(#292 control-system) STEP the eikonal weight from --eikonal-weight (CE) up "
                    "to this value at the tau/MCF onset (--tau-softplus-start-epoch), cosine-eased over "
                    "--stage-transition-rewarmup-epochs. Unset or ==base => BYTE-IDENTICAL constant. "
                    "Fresh run: base 0.05 -> end 0.10 (the MEASURED survival knee; holds the thin lane "
                    "at sigma0.8/93%% vs sigma1.5/49%% as MCF narrows the interface).")
    ap.add_argument("--length-weight", type=float, default=0.001, help="Chan-Vese boundary-length (short smooth boundaries).")
    # (THETA* TIER-2 MUST-2) nuclear-norm low-rank code penalty (additive; default 0.0 == OFF ==
    # bit-identical loss). Drives the per-pair FiLM codes toward a low-rank subspace (rate). Computed
    # as a DIFFERENTIABLE smoothed nuclear norm via Newton-Schulz matrix-sqrt trace (MLX has no svd/
    # eigvalsh vjp); see _nuclear_norm_smooth_mlx.
    ap.add_argument("--code-nuclear-weight", type=float, default=0.0,
                    help="THETA* MUST-2: weight on the smoothed nuclear norm of the per-pair code "
                    "matrix (low-rank -> rate). 0.0 (default) = OFF = bit-identical loss.")
    ap.add_argument("--code-nuclear-eps", type=float, default=1e-3,
                    help="THETA* MUST-2: relative smoothing floor for the nuclear norm (keeps "
                    "Newton-Schulz stable on rank-deficient codes). ~0.3%% bias on well-conditioned "
                    "inputs at 1e-3. Must be > 0.")
    ap.add_argument("--code-nuclear-ns-iters", type=int, default=25,
                    help="THETA* MUST-2: Newton-Schulz iterations for the matrix sqrt (converged by "
                    "~25 for mod_dim<=48). Must be >= 1.")
    # (THETA* TIER-2 STRETCH-1) junction-aware Eikonal relax (additive; default 0.0 == OFF ==
    # bit-identical). Down-weights the Eikonal residual near triple junctions (the margin crease).
    ap.add_argument("--eikonal-junction-relax", type=float, default=0.0,
                    help="THETA* STRETCH-1: down-weight the Eikonal |grad m|->1 residual near triple "
                    "junctions by factor (1 - relax*exp(-g3/tau)). 0.0 (default) = OFF = bit-identical. "
                    "Must be in [0, 1).")
    ap.add_argument("--eikonal-junction-tau", type=float, default=0.5,
                    help="THETA* STRETCH-1: top2-top3 SDF-gap scale for the junction relax weight. "
                    "Must be > 0.")
    # STRUCTURED-PRIOR phi INIT (FEED-ef, ADDITIVE, DEFAULT-OFF). When ON, initialize out_sdf so
    # argmax(phi) ~= the VALIDATED self-detected static-core partition (hood+sky+road[+lane] deep SDFs;
    # FEED-dm/du/dw/dx) instead of random/SIREN -> the row STARTS at the ~0.006 structured floor and
    # LEARNS only the residual (lane wall + Movable). DEFAULT OFF = random/SIREN init = byte-identical
    # to the current row. The static-core is GENERIC same-rig camera geometry (rule-118 FREE); as a
    # TRAIN-TIME init it ships 0 bytes (the archive ships TRAINED weights). Requires render res == the
    # L* res (the static masks are built on the cached frozen CPU-torch L*).
    # MEASURED CAVEAT (FEED-ef, n24 realized-through-R): structuring phi gives NO epoch-0 realized
    # d_seg win — the render is texture-dominated at init (random out_tex), so SegNet reads texture
    # NOT the partition (structured-init realized 0.586 ~ random-init 0.506; even IDEAL flat-palette
    # is 0.125, never the 0.006 DIRECT/field-level floor). The structured prior is field-level only;
    # this flag's sole value is a TRAINING-TRAJECTORY A/B (does a correct partition init converge
    # faster?), UNPROVEN. The one-shot linear-readout init is broken (random trunk can't span the
    # partition, disagree ~0.51); this flag uses a short pretrain (adapts the trunk -> direct
    # disagree ~0.025) which is hosc/SIREN-init-FRAGILE (loud WARN if it stalls). Default OFF.
    ap.add_argument("--structured-init", action=argparse.BooleanOptionalAction, default=False,
                    help="FEED-ef: pretrain phi to the structured static-core partition (DEFAULT OFF=random/SIREN, byte-identical). "
                    "MEASURED: no epoch-0 realized win (texture-gated) -> trajectory A/B only.")
    ap.add_argument("--structured-init-include-lane", action=argparse.BooleanOptionalAction, default=True,
                    help="FEED-ef: include a SHARED static lane band in the structured init (lane is also learned per-frame).")
    ap.add_argument("--structured-init-thresh", type=float, default=0.5,
                    help="FEED-ef: majority-vote threshold for the static-core region masks.")
    ap.add_argument("--structured-init-steps", type=int, default=600,
                    help="FEED-ef: subsampled Adam steps to pretrain phi -> structured target.")
    ap.add_argument("--structured-init-lr", type=float, default=5e-3,
                    help="FEED-ef: LR for the structured-init pretrain (5e-3 converges; 8e-3 stalls).")
    ap.add_argument("--structured-init-subsample", type=int, default=8192,
                    help="FEED-ef: pixels/step for the structured-init pretrain (full-grid is CPU-slow).")
    ap.add_argument("--structured-init-sdf-clip", type=float, default=20.0,
                    help="FEED-ef: clip the SDF target to +/-this (argmax-preserving, well-conditioned).")
    # MUON FINISHER (DAG FEED-fi, PR95 stage-8, ADDITIVE, DEFAULT-OFF). The most-potent measured
    # d_seg stage (CLAUDE.md frontier "Muon is THE drop"); the prior 'Muon NOT yet wired' gap.
    # --muon-start-epoch None (default) => AdamW throughout => BIT-IDENTICAL to the pre-FEED-fi path.
    # When set, at that epoch the 2-D hidden weight matrices (in_proj/film/hidden.*) switch to
    # mlx.optimizers.Muon (Newton-Schulz orthogonalized momentum); biases/1-D + the per-pair code
    # latent + the out_sdf/out_tex final heads stay AdamW (MLX Muon docstring: final FC + embeddings
    # are Muon-suboptimal). Routed via MultiOptimizer in tac.optimization.muon_finisher_mlx.
    ap.add_argument("--muon-start-epoch", type=int, default=None,
                    help="MUON FINISHER (PR95 stage-8): epoch to switch 2-D hidden weights AdamW->Muon "
                    "(default None = AdamW throughout = bit-identical). Set AFTER the l7 stage "
                    "(>= --l7-start-epoch) so the orthogonalized finisher polishes a formed partition.")
    ap.add_argument("--muon-lr", type=float, default=None,
                    help="MUON FINISHER: Muon-group LR (default None => 0.1*--lr, the PR95 ~0.1x-base "
                    "finetune relationship). Muon normalizes its update to ~unit spectral norm, so this "
                    "is a spectral-norm step size; TUNE to the lever's own optimum (OPTIMAL-FORM): a "
                    "typical Muon finisher lr is ~1e-3 to 5e-3.")
    ap.add_argument("--muon-adamw-lr", type=float, default=None,
                    help="MUON FINISHER: AdamW-fallback-group LR for biases/code/heads during the "
                    "finisher (default None => 0.1*--lr).")
    ap.add_argument("--muon-momentum", type=float, default=0.95, help="MUON FINISHER: Muon momentum.")
    ap.add_argument("--muon-weight-decay", type=float, default=None,
                    help="MUON FINISHER: Muon-group decoupled weight decay (default None => --weight-decay).")
    ap.add_argument("--muon-ns-steps", type=int, default=5,
                    help="MUON FINISHER: Newton-Schulz iteration count (Keller Jordan default 5).")
    ap.add_argument(
        "--muon-lr-final-frac", type=float, default=1.0,
        help="(GAP 1, default-off) COSINE-DECAY the Muon-group LR from --muon-lr down to "
        "--muon-lr * this fraction across the Muon-stage span (muon_start_epoch -> --epochs). Muon's "
        "Newton-Schulz fixes update MAGNITUDE so a flat LR cannot self-reduce the step near the minimum "
        "(river-valley Muon 2606.21514); the decay lets the finisher settle. 1.0 (default) = flat/"
        "unchanged = byte-identical; e.g. 0.1 decays to 10%% of --muon-lr by stage end. Only the Muon "
        "group decays (the AdamW fallback self-adapts via its second moment). The schedule is anchored on "
        "--muon-start-epoch so a RESUME into the finisher rebuilds the SAME schedule (bit-faithful). "
        "A/B-ready; no effect until the Muon stage. Must be in (0, 1].",
    )
    ap.add_argument(
        "--muon-warm-start-momentum", action=argparse.BooleanOptionalAction, default=False,
        help="(GAP 2, default-off) WARM-START the fresh Muon momentum buffer (state 'v') from the "
        "OUTGOING AdamW first-moment (state 'm') for the shared param paths at the switch, instead of "
        "cold zeros. Both are gradient EMAs and Newton-Schulz re-normalizes the update, so the "
        "transferred DIRECTION removes the cold-start 'wild unit-norm direction from one noisy gradient' "
        "boundary thrash / d_seg spike. Default OFF = cold zero start = byte-identical. Only a plain "
        "Adam/AdamW base is transferable; non-Adam bases fall back to cold. On a RESUME INTO the "
        "finisher this is N/A (the Muon momentum is restored from the checkpoint). A/B-ready.",
    )
    # ---- BUILD 1 (FEED-fw): STAGE-TRANSITION TREATMENT (ADDITIVE, all default-OFF => BIT-IDENTICAL).
    # "different stages need different treatment" applied to the TRANSITIONS so the AdamW->AdamW stage
    # boundaries (ce->tau, tau->l7) + the lane-edge / margin-saliency re-engage epochs are stable by
    # construction (the l7->Muon switch already re-treats via a fresh optimizer, FEED-fi). The
    # spike-guard re-treat already exists at every boundary; these add (1) LR re-warmup + (2) optional
    # AdamW moment reset. theta*-prereq; NOT a score row.
    ap.add_argument("--stage-transition-rewarmup-epochs", type=int, default=0,
                    help="BUILD 1: N>0 ramps LR from --stage-transition-rewarmup-floor back to the "
                    "scheduled LR over N epochs after each AdamW->AdamW stage boundary (default 0=OFF "
                    "=> bit-identical). Requires --lr-schedule; no effect during the Muon finisher.")
    ap.add_argument("--stage-transition-rewarmup-floor", type=float, default=0.1,
                    help="BUILD 1: LR fraction at the boundary epoch for re-warmup (used only when "
                    "--stage-transition-rewarmup-epochs > 0; must be in [0,1]).")
    ap.add_argument("--stage-transition-rewarmup-shape", choices=["linear", "cosine"], default="linear",
                    help="BUILD 1: re-warmup ramp shape (used only when rewarmup-epochs > 0).")
    ap.add_argument("--stage-transition-reset-moments", action="store_true",
                    help="BUILD 1: at each AdamW->AdamW stage boundary, rebuild the AdamW optimizer so "
                    "the m/v moments are zeroed (stale momentum through a loss-landscape change is the "
                    "FEED-ft#3 tau-jump root cause). Default OFF => bit-identical. No-op during the "
                    "Muon finisher (it already re-inits a fresh optimizer).")
    # ---- BUILD 2 (FEED-fw): LANE-PRIOR phi1 (ADDITIVE, default-OFF => structured-init BIT-IDENTICAL).
    # Initialize the structured-init target's phi1 (lane-class SDF) channel to the signed distance of
    # the openpilot deg-3 centerline curve (FEED-fs: that centerline IS the Road<->Lane separatrix,
    # residual 1.9e-5). REUSES tac.boundary_math.lane_sdf_component (build_structured_lane_sdf: the
    # ground-plane homography K @ scorer-res {fx=910*512/1164=400.3, ...} -> image-space deg-3 lane
    # curve -> per-pixel signed distance; + inject_lane_sdf). rule-118 FREE generic structure: a
    # better TRAINING-TIME starting point that ships 0 archive bytes (only if the centerline coords
    # were SHIPPED would they be COUNTED, ~8 floats/frame -- a SEPARATE archive-side option, NOT this
    # build). Requires --structured-init (the pretrain mechanism that absorbs the target).
    ap.add_argument("--lane-prior-phi1", action=argparse.BooleanOptionalAction, default=False,
                    help="BUILD 2: init the structured-init target's lane (phi1) channel to the "
                    "openpilot deg-3 centerline signed distance (default OFF => bit-identical). "
                    "Requires --structured-init.")
    ap.add_argument("--lane-prior-phi1-mode", choices=["replace", "bias", "paint"], default="replace",
                    help="BUILD 2: inject the centerline SDF by REPLACE (lane channel becomes the "
                    "openpilot fit) or BIAS (add to the static-core lane channel) or PAINT "
                    "(paint-then-SDF #291: paint the lane label into the argmax at band pixels then "
                    "rebuild all K SDFs so the lane WINS by construction — the NUCLEATION fix; "
                    "replace is a MEASURED NO-OP: the thin lane SDF loses argmax to the deep road "
                    "static-core -> part_frac[lane]=0). Default replace (byte-identical); fresh "
                    "seeded run uses paint.")
    ap.add_argument("--lane-prior-phi1-bias-scale", type=float, default=1.0,
                    help="BUILD 2: scale for --lane-prior-phi1-mode bias (unused for replace).")
    ap.add_argument("--lane-prior-phi1-source-pair", type=int, default=0,
                    help="BUILD 2: which cached pair's L* argmax the centerline is fit from (default "
                    "0, matching the structured-init pretrain's pair-0 feats convention).")
    ap.add_argument("--lane-prior-phi1-dash-gate", action=argparse.BooleanOptionalAction, default=True,
                    help="BUILD 2: model the lane dash period (deg-3 centerline + dash). Default on.")
    # =====================================================================================
    # #224 CONSOLIDATED WIRE-IN — the 6 LANDED components. ALL flags DEFAULT-OFF => the
    # default render+loss+init path is BYTE-IDENTICAL to the pre-#224 baseline (the
    # non-negotiable acceptance bar; proven by tools/wire_in_224_byte_identical_smoke.py).
    # Each flag routes to the REAL (tested) module function when ON (NO-FAKE). Nothing here
    # fires unless explicitly enabled. Docs: docs/aa_sdf_observation_render_wire_in_spec.md +
    # docs/analytic_lane_render_band_wire_in_spec.md + the in-module WIRE-IN SPECs.
    # -------- (1) AA-SDF observation-map render (aa_sdf_observation_render; MEASURED #1 rep lever) --
    ap.add_argument("--render-aa", choices=["none", "supersample", "ipe"], default="none",
                    help="#224/#220 AA observation-map render mode (default none = byte-identical "
                    "point-sample). supersample=render at ss*grid+box-down; ipe=mip-NeRF cone "
                    "attenuation of the curvelet basis (analytical, ~0-compute).")
    ap.add_argument("--aa-supersample", type=int, default=1,
                    help="#224 supersample factor ss for --render-aa supersample (ss=1 byte-identical).")
    ap.add_argument("--aa-ipe-footprint", type=float, default=1.0,
                    help="#224 footprint std scale for --render-aa ipe (1.0 = one-pixel box).")
    ap.add_argument("--aa-self-orient-fine-mode", type=str, default="refuse",
                    choices=["refuse", "batch", "full"],
                    help="#224 (Wave B) how --render-aa supersample + --self-orient sources the per-pair "
                    "fine-grid dir-feats (measured: fine-EDT ~49ms/pair @ ss=2; per-pair fine dir-feat "
                    "75.5MB). refuse (default) = fail-closed (memory/wall-clock tradeoff is the operator's "
                    "call, see the guard). batch = MEMORY-SAFE bounded on-demand cache (~batch*75MB, e.g. "
                    "0.6GB @ batch=8 vs 45GB all-600) but wall-clock-heavy (P fine-EDTs/epoch ~29s @ "
                    "n600, since every pair renders every epoch => a batch-bounded cache thrashes). full "
                    "= WALL-CLOCK-viable (fine dir-feats computed ONCE per --reorient-every, amortized) "
                    "but ~45GB @ ss=2 n600 (on top of the ~41GB base cache => ~86GB; needs an n600 "
                    "memory-fit validation the no-launch CONTAINMENT forbids this wave).")
    ap.add_argument("--aa-self-orient-fine-cache-cap", type=int, default=16,
                    help="#224 (Wave B) bounded per-pair fine dir-feat cache size for "
                    "--aa-self-orient-fine-mode batch (memory ~ cap*75MB @ ss=2).")
    # -------- (2) analytic-lane render-band (analytic_lane_render_band; FEED-dv #203/#213/#215) ------
    ap.add_argument("--lane-render-band", action=argparse.BooleanOptionalAction, default=False,
                    help="#224/FEED-dv: composite the analytic-lane render-band via compose_fn "
                    "(class-1 render-time authority). DEFAULT OFF => byte-identical.")
    ap.add_argument("--lane-band-softness", type=float, default=1.0,
                    help="#224 AA-SDF coverage ramp width (px) on the band lateral edge.")
    ap.add_argument("--lane-band-dash-forward-max-m", type=float, default=55.0,
                    help="#224/#215 SegNet-Nyquist: dash-gate ONLY where forward < this (m); continuous beyond.")
    ap.add_argument("--lane-band-uncertainty-source", type=str, default="witness",
                    choices=["witness", "gt", "none"],
                    help="#224 uncertainty margin source for the FP-killer gate (witness margin PROB; "
                    "gt margin LOGIT; none disables the gate).")
    ap.add_argument("--lane-band-tau", type=float, default=0.85,
                    help="#224 uncertainty threshold (witness margin PROB [0,1]; gt margin LOGIT ~[0,13]).")
    ap.add_argument("--lane-band-eps", type=float, default=0.35, help="#224 uncertainty ramp width.")
    ap.add_argument("--lane-band-weight", type=float, default=1.0, help="#224 band strength (curriculum ramp).")
    ap.add_argument("--lane-band-start-epoch", type=int, default=300, help="#224 engage the band at this epoch.")
    ap.add_argument("--lane-band-dash-comb", action=argparse.BooleanOptionalAction, default=False,
                    help="#287: replace the band's per-pair FITTED dash phase with the EGO-PHASE dash "
                    "comb (tac.boundary_math.dash_comb) — global (period, duty, ego-scale) + per-slot "
                    "world phase transported by cumulative ego forward distance (the "
                    "dash_erasure_homogenization_v1 cell-problem corrector, rule-118 FREE at decode). "
                    "Only meaningful with --lane-render-band. DEFAULT OFF => byte-identical.")
    ap.add_argument("--lane-band-comb-softness-m", type=float, default=0.3,
                    help="#287 comb AA ramp width (ground meters) on the dash on/off edge; 0 = hard gate.")
    # -------- (3) warp-real-luma frame0 pose carrier (warp_real_luma_frame0; PoseGauge.WARP_REAL_LUMA) --
    ap.add_argument("--pose-carrier", action=argparse.BooleanOptionalAction, default=False,
                    help="#224: render frame0 THROUGH the SE(3) ground-homography warp of the REAL "
                    "keyframe luma (seg-free f0 -> real-luma pose carrier). Parity-dispatch render_fn "
                    "(even code=f0->carrier, odd=f1->witness). Requires --w-pose>0. DEFAULT OFF => "
                    "byte-identical (the witness's own f0 render).")
    ap.add_argument("--pose-carrier-source", type=str, default="real_keyframe",
                    choices=["real_keyframe", "generated"],
                    help="#205 pose-carrier frame0 SOURCE (Track B store-nothing-but-xi, 18927a1ae). "
                    "real_keyframe (default) = warp the STORED real keyframe luma (gt_f0; COUNTS the "
                    "keyframe in archive.zip). generated = STORE-NOTHING: warp the witness's OWN plain "
                    "frame0 INR render (up to camera-native) by the twist -> stores ONLY xi/H (~0 "
                    "marginal bytes; the render is FREE, rule-118). The dxi residual co-adapts to the "
                    "witness-render warp. Default real_keyframe => byte-identical (unchanged wiring).")
    ap.add_argument("--pose-carrier-residual-mode", type=str, default="table", choices=["table", "film"],
                    help="#224 pose-carrier residual parametrization: table (per-pair (P,6), byte-minimal) "
                    "or film (code-conditioned MLP). Default table.")
    ap.add_argument("--pose-carrier-residual-scale", type=float, default=1.0,
                    help="#224 pose-carrier learnable-residual scale (dxi = scale * residual).")
    ap.add_argument("--pose-carrier-s-t", type=float, default=None,
                    help="#224 pose-carrier ground-homography translation scale s_t for the stored twist "
                    "xi = xi_from_pose_calibration(gt_pose, s_t, s_r, pitch). None (default) => FIT s_t at "
                    "startup on --pose-carrier-fit-pairs via the frozen CPU-torch PoseNet d_pose grid "
                    "(self-calibrating, deterministic; mirrors tools/measure_warp_real_luma_frame0_dpose).")
    ap.add_argument("--pose-carrier-s-r", type=float, default=0.0,
                    help="#224 pose-carrier rotation scale s_r for the stored twist (default 0.0 = the "
                    "measured d_pose-optimal whole-ground calibration).")
    ap.add_argument("--pose-carrier-pitch", type=float, default=0.0,
                    help="#224 pose-carrier ground-plane pitch (rad) for the homography geom (default 0.0).")
    ap.add_argument("--pose-carrier-fit-pairs", type=int, default=24,
                    help="#224 # pairs for the startup s_t fit grid (only when --pose-carrier-s-t is None).")
    # -------- (4) persistence/topology loss (persistence_topology_loss; TopologyLossGauge) -----------
    ap.add_argument("--persistence-loss-weight", type=float, default=0.0,
                    help="#224/#218: weight of the soft-clDice + persistence-weighted island-recall "
                    "term on the SHARED realized-through-R seg forward (births the finest-scale "
                    "erasure-tail the CE drops). 0 (default) => branch skipped => byte-identical.")
    ap.add_argument("--persistence-recall-weight", type=float, default=1.0,
                    help="#224 w_recall inside the persistence class loss (clDice weight fixed 1.0).")
    ap.add_argument("--cldice-iters", type=int, default=5,
                    help="#224 soft-skeleton peeling iterations for the clDice connectivity term.")
    ap.add_argument("--persistence-warmup-epochs", type=int, default=0,
                    help="#224 linear warm-up (epochs) for the persistence weight (coarse->fine; "
                    "0=full weight immediately).")
    ap.add_argument("--persistence-classes", type=str, default="auto",
                    help="#224 target classes: 'auto' self-detects the thin/small erasure-tail classes "
                    "from the cached GT argmax (detect_persistence_tail_classes), or a comma list e.g. '1,3'.")
    # -------- (5) island seed/containment/amplification (island_protection; IslandProtectionGauge) ---
    ap.add_argument("--seed-islands", action=argparse.BooleanOptionalAction, default=False,
                    help="#224/#208: EARLY-SEED the finest-scale islands (self-detected lane+movable) "
                    "as an RGB residual (from GT frame1 appearance) composited into the SegNet-scored "
                    "frame1 (accelerant; ships 0 archive bytes; SEPARATE module, absent from "
                    "EMA/blob/deploy). Independent of --structured-init; requires --w-seg > 0 (the seed "
                    "helps only through the realized seg loss). DEFAULT OFF => byte-identical.")
    ap.add_argument("--island-dilate-px", type=int, default=1, help="#224 annulus dilation of the island masks.")
    ap.add_argument("--seed-island-eased", action=argparse.BooleanOptionalAction, default=False,
                    help="#323 LADDER per-class island homotopy: replace the isotropic --island-dilate-px "
                    "annulus (which is off-manifold for a lane curve = measured NO-GO) with class-aware "
                    "eased masks — movable via SDF forward-Euler dilation (proven-transfer), lane via "
                    "openpilot VP-tangent oriented widening (stays on the ~8-dim lane manifold). Applies "
                    "to BOTH the amplify + seed island-mask builders. DEFAULT OFF => byte-identical.")
    ap.add_argument("--seed-blend", type=float, default=1.0,
                    help="#224 island-seed blend (residual = blend*(gt_island_rgb - base) on the island).")
    ap.add_argument("--seed-lr", type=float, default=0.02,
                    help="#224 learning rate for the SEPARATE island-seed AdamW group (its own optimizer; "
                    "the seed is NOT in the witness EMA/blob/deploy).")
    ap.add_argument("--containment-mode", type=str, default="shield", choices=["freeze", "damp", "shield"],
                    help="#224 how the seeded island grad is protected from the bulk-CE wash "
                    "(shield=zero only the destructive same-sign component).")
    ap.add_argument("--containment-damp", type=float, default=0.1, help="#224 damp factor for --containment-mode damp.")
    ap.add_argument("--amplify-weight", type=float, default=0.0,
                    help="#224/#208: weight of the island-birth term (rides the SHARED LEVER-4 _signed "
                    "margin; NO 2nd saliency/SegNet forward). 0 (default) => skipped => byte-identical.")
    ap.add_argument("--amplify-form", type=str, default="hinge", choices=["hinge", "softplus"],
                    help="#224 island-birth penalty form.")
    ap.add_argument("--amplify-margin-target", type=float, default=1.0,
                    help="#224 the margin the island must WIN its pixels by.")
    ap.add_argument("--amplify-persist", type=str, default="inverse_thickness",
                    choices=["uniform", "inverse_thickness"],
                    help="#224 island birth-weight kind (inverse_thickness up-weights the thinnest tail).")
    # BUILD #300 (SEED-ABSORPTION FIX; memo plateau_disambiguator_results_20260704.md). The island seed
    # (--seed-islands) is composited into the SegNet-scored frame1 and read by EVERY realized-through-R
    # seg lever, so once the seed satisfies the loss on the Lane+Movable island, dL/d(witness) ~= 0 there
    # and the witness never learns to FORM the islands itself -> deploy (witness-alone) has ~0 island mass
    # (MEASURED: 71% of the plateau = the 2 seeded classes at 100% within-class flip). Two coupled,
    # DEFAULT-OFF mechanisms restore the absorption gradient (both OFF => byte-identical current trainer):
    ap.add_argument("--witness-alone-island-loss", action=argparse.BooleanOptionalAction, default=False,
                    help="BUILD #300 (a): score the island-FORMATION levers (--amplify-weight island-birth "
                    "+ --persistence-loss-weight island-recall) on the WITNESS-ALONE render (island seed "
                    "EXCLUDED from the compose chain) so the witness gets the absorption gradient the seed "
                    "was starving. The seed still composes for the OTHER levers (base CE etc.) + nucleation. "
                    "REQUIRES --seed-islands (else there is no seed to exclude => fail closed). SERIAL-only "
                    "(not wired into --micro-batch-pairs; fails closed there). DEFAULT OFF => byte-identical.")
    ap.add_argument("--seed-anneal-epochs", type=int, default=0,
                    help="BUILD #300 (b): ramp the island-seed COMPOSE WEIGHT full(1.0)->0.0 over epochs "
                    "[1, seed-anneal-epochs] (transfer schedule: nucleation early, deploy-surface "
                    "(witness == composed) by the anneal end). 0 (DEFAULT) => constant 1.0 => _compose_chain "
                    "byte-identical. Set ~= --tau-softplus-start-epoch so the seed is fully transferred to "
                    "the witness BEFORE the tau/MCF stage erodes sub-critical island structure.")
    ap.add_argument("--seed-anneal-shape", type=str, default="linear", choices=["linear", "cosine"],
                    help="BUILD #300 (b): island-seed compose-weight anneal shape (full->0). Consulted only "
                    "when --seed-anneal-epochs > 0.")
    args = ap.parse_args(argv)

    # (#254) P0 admission guard: refuse a RAW heavy launch that skipped the governed admission gate
    # (tools/launch_witness_run.py / tools/safe_run.py / spawn_durable_daemon register the footprint
    # with the system memory governor BEFORE spawning — a concurrent >128 GB run CRASHED the box).
    # ADVISORY (warn only) until enforce is armed; when armed, REFUSES (exit 7) unless launched via a
    # governed path (marker set) or carrying TAC_ADMISSION_BYPASS_OK=<reason>. Runs AFTER argparse so
    # --help stays free; before any heavy MLX/scorer allocation.
    from tac.admission_guard import assert_governed_admission
    assert_governed_admission("train_levelset_witness_realized_through_R_mlx")

    # ── CONFOUND-CLEANUP post-parse resolution + fail-closed guards (2026-07-05) ──────────────────
    # (C9) resolve --resume-model-from default: 'live' EXCEPT auto-'ema' for a re-treatment resume
    # (warm-start, or the clear-spike-guard + allow-lever-drift palliative pair) where the clean EMA
    # shadow is the right warm-start source. Explicit value on the CLI always wins (default is None).
    if getattr(args, "resume_model_from", None) is None:
        _retreat = bool(getattr(args, "warm_start_weights_only", False)) or (
            bool(getattr(args, "resume_clear_spike_guard", False))
            and bool(getattr(args, "resume_allow_lever_drift", False)))
        args.resume_model_from = "ema" if _retreat else "live"
        if args.resume_from and args.resume_model_from == "ema":
            print(json.dumps({"stage": "resume_model_from_resolved", "resume_model_from": "ema",
                              "reason": ("warm_start_weights_only" if getattr(args, "warm_start_weights_only", False)
                                         else "clear_spike_guard+allow_lever_drift"),
                              "note": "C9: re-treatment resume auto-loads the CLEAN EMA shadow (a "
                              "crash mid-spike wrote diverging LIVE weights; the shadow is clean)"}),
                  flush=True)
    # (C1) legacy spike-guard + --resume-clear-spike-guard is FAIL-CLOSED: clearing the frozen median
    # in LEGACY mode is a ONE-SHOT reset, not a cure -- it re-anchors the median once from the first
    # accepted batch, then re-enters the sustained spike and RE-FREEZES in 1-13 ep (measured). The
    # actual cure is rollback mode. Refuse the combination with a clear message rather than silently
    # delaying the deadlock.
    if (bool(getattr(args, "resume_clear_spike_guard", False))
            and str(getattr(args, "spike_guard_mode", "rollback")) == "legacy"
            and not bool(getattr(args, "warm_start_weights_only", False))):
        raise ValueError(
            "(C1) --resume-clear-spike-guard with --spike-guard-mode legacy is FAIL-CLOSED: clearing "
            "the frozen median in legacy mode only DELAYS the absorbing-median deadlock (re-anchors "
            "once, then re-freezes in 1-13 ep on the same sustained loss shift; #205 measured). Use "
            "--spike-guard-mode rollback (the DEFAULT, and the actual cure) OR "
            "--warm-start-weights-only (a clean fresh-optimizer re-treatment).")
    # (C14) --eikonal-weight-end must not anneal UP relative to --eikonal-weight without an explicit
    # rationale: the eikonal weight should DECAY post-SDF (once the interface is a valid unit-gradient
    # SDF the unit-gradient enforcement is done), not RISE. An end>base ramp is the wrong direction
    # and was a live confound (it drove the eik term to dominate the loss). Refuse unless the operator
    # sets TAC_EIKONAL_WEIGHT_END_UP_OK=<rationale> (a non-empty, non-placeholder string).
    _ewe = getattr(args, "eikonal_weight_end", None)
    if _ewe is not None and float(_ewe) > float(args.eikonal_weight):
        _up_ok = os.environ.get("TAC_EIKONAL_WEIGHT_END_UP_OK", "").strip()
        if _up_ok in ("", "<rationale>", "<reason>"):
            raise ValueError(
                f"(C14) --eikonal-weight-end ({_ewe}) > --eikonal-weight ({args.eikonal_weight}) "
                "anneals the eikonal weight UP -- the WRONG direction (it should DECAY once the SDF is "
                "valid; an UP-ramp let the eik term dominate the loss, a measured confound). If this is "
                "intentional, set env TAC_EIKONAL_WEIGHT_END_UP_OK=<real rationale>.")
        print(json.dumps({"stage": "eikonal_weight_end_up_WAIVED", "base": float(args.eikonal_weight),
                          "end": float(_ewe), "rationale": _up_ok}), flush=True)
    # (C16) --seed-anneal-epochs is interpreted in ABSOLUTE epochs; on a --resume-from whose start
    # epoch is >= the anneal length, the seed compose weight is ALREADY fully withdrawn before the run
    # begins (the seed crutch never contributes) -- warn loudly (the seed-formation losses may still
    # be live; this only concerns the compose crutch). Emitted here where args are known; the resume
    # start_epoch is validated again at load time.
    if (int(getattr(args, "seed_anneal_epochs", 0)) > 0 and args.resume_from
            and int(getattr(args, "warm_start_epoch", -1)) >= int(getattr(args, "seed_anneal_epochs", 0))):
        print(json.dumps({"stage": "seed_anneal_epochs_WARN",
                          "seed_anneal_epochs": int(args.seed_anneal_epochs),
                          "warm_start_epoch": int(args.warm_start_epoch),
                          "msg": "(C16) --seed-anneal-epochs < the resume start epoch: the seed compose "
                          "crutch is fully withdrawn before this run begins (off every epoch). Make it "
                          "RELATIVE to the resume start (add the resume epoch) if you want the crutch "
                          "post-resume. Seed-FORMATION losses are unaffected."}), flush=True)

    # (review C2) --anneal-epochs guard: must be >= 1 when set (it is a cosine DENOMINATOR). A value
    # < --epochs means the anneal COMPLETES before the run ends (temp/LR clamp past their end values
    # for the tail) -- legal for a warm-start window but usually a mistake otherwise, so WARN (do not
    # fail). None (default) => no guard fires => bit-identical.
    if getattr(args, "anneal_epochs", None) is not None:
        if args.anneal_epochs < 1:
            raise ValueError(f"--anneal-epochs ({args.anneal_epochs}) must be >= 1 (cosine denominator).")
        if args.anneal_epochs < args.epochs:
            print(json.dumps({"stage": "anneal_epochs_WARN", "anneal_epochs": int(args.anneal_epochs),
                              "epochs": int(args.epochs),
                              "msg": "--anneal-epochs < --epochs: the temp/LR anneal completes BEFORE the "
                              "run ends; the tail epochs run at the clamped end values. Intended for a "
                              "WARM-START window (resume mid-schedule); verify this is what you want."}),
                  flush=True)

    # (fix d) curriculum boundaries must be strictly ordered and fit inside the budget, else the
    # tau_softplus / l7 stages silently never run (or run for ~0 epochs) -> untrustworthy d_seg.
    if args.curriculum:
        # (L1 SEAL-review relax, 4bf533cab) l7_start_epoch > epochs is the LEGITIMATE "l7 NEVER runs"
        # form (l7 is a MEASURED DEFECT demoted from the default curriculum; the C2 event-guard and
        # _seg_form_for_epoch both honor >= epochs as never). Note l7_start == epochs is the L1
        # off-by-one (l7 WOULD run on the final epoch) — the fresh config uses epochs+1 (e.g. 1001).
        if not (0 < args.tau_softplus_start_epoch < args.l7_start_epoch):
            raise ValueError(
                f"--curriculum requires 0 < tau_softplus_start_epoch ({args.tau_softplus_start_epoch}) "
                f"< l7_start_epoch ({args.l7_start_epoch}). The d_seg sequence (ce->tau_softplus[->l7]) "
                "needs tau_softplus to actually run (THE primary d_seg drop); l7_start_epoch > epochs "
                "is allowed and means l7 NEVER runs (the demoted-defect form)."
            )

    # (FEED-df R2) LEVER-3 fail-closed config guard (pure helper; fails LOUD before any GPU spend).
    validate_lane_edge_config(
        lane_edge_weight=args.lane_edge_weight, lane_edge_start_epoch=args.lane_edge_start_epoch,
        epochs=args.epochs, lane_edge_class=args.lane_edge_class, n_classes=5,
    )

    # (LEVER-B) thin-lane dropped-dash prior fail-closed config guard (same NO-FAKE silent-no-op class).
    validate_lane_thin_config(
        lane_thin_weight=args.lane_thin_weight, lane_thin_start_epoch=args.lane_thin_start_epoch,
        epochs=args.epochs, lane_thin_class=args.lane_thin_class, lane_thin_radius=args.lane_thin_radius,
        n_classes=5,
    )

    # (LEVER-A) FiLM-rank-fix fail-closed config guards (same NO-FAKE silent-no-op class).
    # A rank-floor with target <= 1 can NEVER penalize (PR >= 1 always) = a silent no-op = a FALSE
    # 'rank-floor does nothing' verdict. The film-per-layer / film-concat-code architecture routes are
    # loaded from a frozen-decoder npz that does NOT contain them, so --freeze-decoder-fit-codes would
    # leave them zero-init AND frozen = never trained = a silent no-op = a FALSE 'film-fix does nothing'.
    if args.film_rank_floor_weight > 0.0 and args.film_rank_floor_target <= 1.0:
        raise ValueError(
            f"--film-rank-floor-weight {args.film_rank_floor_weight} > 0 but "
            f"--film-rank-floor-target ({args.film_rank_floor_target}) <= 1: the participation ratio is "
            ">= 1 by construction, so relu(target - PR) would be 0 always -> a silent no-op = a FALSE "
            "'rank-floor does nothing' verdict. Set --film-rank-floor-target > 1 (e.g. 4).")
    if (args.film_per_layer or args.film_concat_code) and args.freeze_decoder_fit_codes:
        raise ValueError(
            "--film-per-layer / --film-concat-code are incompatible with --freeze-decoder-fit-codes: "
            "the frozen decoder npz has no film_pl/concat_pl keys, so those routes would stay zero-init "
            "AND frozen = never trained = a silent no-op = a FALSE 'film-fix does nothing' verdict. Run "
            "the FiLM-rank-fix on a joint (unfrozen) run.")

    # (FEED-eq) LEVER-4 fail-closed config guard: a saliency lever that never engages (start > epochs)
    # is a silent no-op = a FALSE 'margin-saliency does not help' verdict (same NO-FAKE class the lane
    # validator extincts). Also guard tau>0 so exp(-margin/tau) is well-defined.
    if args.margin_saliency_weight > 0.0:
        if args.margin_saliency_start_epoch > args.epochs:
            raise ValueError(
                f"--margin-saliency-weight {args.margin_saliency_weight} > 0 but "
                f"--margin-saliency-start-epoch ({args.margin_saliency_start_epoch}) > --epochs "
                f"({args.epochs}): the saliency hinge would NEVER engage -> a silent no-op = a FALSE "
                "'margin-saliency does not help' verdict. Set --margin-saliency-start-epoch <= --epochs."
            )
        if args.margin_saliency_tau <= 0.0:
            raise ValueError(f"--margin-saliency-tau ({args.margin_saliency_tau}) must be > 0 "
                             "(sal=exp(-gt_margin/tau)).")

    # (FEED-fi) MUON FINISHER fail-closed config guard (same NO-FAKE class as the lane/saliency
    # validators): a finisher that never engages (start > epochs) is a silent no-op = a FALSE
    # 'Muon does not help d_seg' verdict; a finisher with NO trainable 2-D weights (frozen decoder)
    # routes everything to AdamW = the Muon group is empty = the same false verdict. Fail LOUD.
    if args.muon_start_epoch is not None:
        if not (1 <= args.muon_start_epoch <= args.epochs):
            raise ValueError(
                f"--muon-start-epoch ({args.muon_start_epoch}) must be in [1, --epochs ({args.epochs})]: "
                "outside the budget the Muon finisher would NEVER engage -> a silent no-op = a FALSE "
                "'Muon does not help' verdict. PR95 places it as the FINAL stage (set it >= "
                f"--l7-start-epoch {args.l7_start_epoch} when --curriculum is on)."
            )
        if args.freeze_decoder_fit_codes:
            raise ValueError(
                "--muon-start-epoch is incompatible with --freeze-decoder-fit-codes: the only trainable "
                "param then is the per-pair `code` latent, which is AdamW-routed (Muon-suboptimal for "
                "embeddings) -> the Muon group would be EMPTY = a silent no-op = a FALSE 'Muon does not "
                "help' verdict. Muon finishes the DECODER weight matrices; run it on a joint (unfrozen) run."
            )
        # FEED-fm FIX-3 (RULE-6 freedom): placing the finisher BEFORE the l7_softplus stage (under
        # curriculum) is the PR95-suboptimal placement (Muon polishes a not-yet-formed partition), but
        # it is the operator's CHOICE to make -> WARN loudly, do NOT fail closed. The range [1,epochs]
        # + freeze-decoder guards above STAY hard raises (those are silent-no-op / empty-Muon-group
        # NO-FAKE traps, not placement preferences). Gated on --curriculum: l7_start_epoch only governs
        # a stage that exists under curriculum, so the warning is meaningful only there.
        if args.curriculum and args.muon_start_epoch < args.l7_start_epoch:
            print(json.dumps({"stage": "muon_finisher_WARN",
                              "muon_start_epoch": int(args.muon_start_epoch),
                              "l7_start_epoch": int(args.l7_start_epoch),
                              "msg": "--muon-start-epoch < --l7-start-epoch: the Muon finisher engages "
                              "BEFORE the l7_softplus stage forms the partition. PR95 places Muon as the "
                              "FINAL stage; an orthogonalized finisher on a not-yet-formed partition is "
                              "likely weaker d_seg. ALLOWED (operator freedom); set >= --l7-start-epoch "
                              "for the PR95 placement."}), flush=True)

    # BUILD 1 (FEED-fw) fail-closed config guards (same NO-FAKE silent-no-op class as the lane/muon
    # validators). DEFAULT-OFF (rewarmup-epochs 0, lane-prior off) => none of these fire => unchanged.
    if args.stage_transition_rewarmup_epochs < 0:
        raise ValueError(
            f"--stage-transition-rewarmup-epochs ({args.stage_transition_rewarmup_epochs}) must be "
            ">= 0 (0 = OFF).")
    if args.stage_transition_rewarmup_epochs > 0:
        if not args.lr_schedule:
            raise ValueError(
                "--stage-transition-rewarmup-epochs > 0 requires --lr-schedule: the re-warmup "
                "multiplies the SCHEDULED LR, so with --no-lr-schedule it would be a silent no-op = "
                "a FALSE 're-warmup does nothing' verdict.")
        if not (0.0 <= args.stage_transition_rewarmup_floor <= 1.0):
            raise ValueError(
                f"--stage-transition-rewarmup-floor ({args.stage_transition_rewarmup_floor}) must be "
                "in [0, 1] (the LR fraction at the boundary epoch).")
    # BUILD 2 (FEED-fw) fail-closed guard: the lane prior is injected into the structured-init
    # pretrain target, so without --structured-init it would NEVER be applied = a silent no-op = a
    # FALSE 'lane prior does nothing' verdict.
    if getattr(args, "lane_prior_phi1", False) and not args.structured_init:
        raise ValueError(
            "--lane-prior-phi1 requires --structured-init: the openpilot centerline SDF is injected "
            "into the structured-init pretrain target; without --structured-init the prior would "
            "never be applied = a silent no-op = a FALSE 'lane prior does nothing' verdict.")

    # RESIDUAL-ONLY MODE fail-closed config guards (same NO-FAKE silent-no-op class). --residual-mode
    # without the bundle would be a silent no-op (no composition) = a FALSE 'residual mode does
    # nothing'. --structured-init / --lane-prior-phi1 / --freeze-decoder-fit-codes are the CONTRADICTORY
    # mechanism (they bake the bulk INTO the weights = the opposite of residual mode, which keeps the
    # bulk OUTSIDE the counted weights and composes it deterministically) -> fail LOUD rather than
    # silently ship a non-shrinking INR. The loss-weighting surgical levers (--lane-thin-* /
    # --margin-saliency-* / --hardness-*) ARE compatible (they weight the COMPOSED-render d_seg) and
    # are intentionally NOT forbidden.
    if getattr(args, "residual_mode", False):
        if not args.residual_target_npz:
            raise ValueError(
                "--residual-mode requires --residual-target-npz (the residual training bundle): "
                "without it the composition has no bulk to compose = a silent no-op = a FALSE "
                "'residual mode does nothing' verdict.")
        if args.structured_init or getattr(args, "lane_prior_phi1", False):
            raise ValueError(
                "--residual-mode is incompatible with --structured-init / --lane-prior-phi1: those "
                "BAKE the bulk/static-core INTO the INR weights (a train-time init that ships the "
                "bulk inside the counted weights = NO rate shrink), which is the EXACT mechanism "
                "residual mode replaces (the bulk is GENERATED deterministically OUTSIDE the weights "
                "and COMPOSED before R). Run residual mode WITHOUT --structured-init.")
        if args.freeze_decoder_fit_codes:
            raise ValueError(
                "--residual-mode is incompatible with --freeze-decoder-fit-codes: residual mode "
                "trains the INR's decoder to flip the residual annulus; a frozen decoder cannot "
                "(only the per-pair code would move) = a silent no-op = a FALSE 'residual mode does "
                "nothing' verdict.")
    elif args.residual_target_npz:
        raise ValueError(
            "--residual-target-npz was given but --residual-mode is OFF: the bundle would be "
            "loaded-and-ignored = a silent no-op = a FALSE 'residual bundle does nothing'. Pass "
            "--residual-mode to engage the composition, or drop --residual-target-npz.")

    # (THETA* TIER-2 MUST-1) tau-anneal-shape fail-closed guards (pure; fail LOUD before any GPU spend).
    if args.tau_anneal_shape == "geometric" and not (args.softmax_temp_start > 0.0 and args.softmax_temp_end > 0.0):
        raise ValueError(
            f"--tau-anneal-shape geometric requires --softmax-temp-start ({args.softmax_temp_start}) > 0 "
            f"AND --softmax-temp-end ({args.softmax_temp_end}) > 0: the log-spaced curve "
            "tau=start*(end/start)**prog is undefined / non-positive otherwise.")
    if not (0.0 < args.tau_hold_frac <= 1.0):
        raise ValueError(
            f"--tau-hold-frac ({args.tau_hold_frac}) must be in (0, 1] (the fraction of the anneal "
            "window at which cosine_hold reaches the floor; 1.0 = no hold = bit-identical cosine).")

    # (THETA* TIER-2 MUST-2) nuclear-norm penalty fail-closed guards.
    if args.code_nuclear_weight < 0.0:
        raise ValueError(f"--code-nuclear-weight ({args.code_nuclear_weight}) must be >= 0 (0 = OFF).")
    if args.code_nuclear_weight > 0.0:
        if args.code_nuclear_eps <= 0.0:
            raise ValueError(
                f"--code-nuclear-eps ({args.code_nuclear_eps}) must be > 0 (relative smoothing floor "
                "that keeps Newton-Schulz stable on rank-deficient codes).")
        if args.code_nuclear_ns_iters < 1:
            raise ValueError(
                f"--code-nuclear-ns-iters ({args.code_nuclear_ns_iters}) must be >= 1.")

    # (THETA* TIER-2 MUST-3) SWA / wider-finisher EMA fail-closed guards (same NO-FAKE silent-no-op
    # class as the lane/muon validators): a finisher decay set with no resolvable start would NEVER
    # engage = a FALSE 'wider EMA does nothing' verdict.
    if args.ema_decay_finisher is not None:
        if not (0.0 < args.ema_decay_finisher < 1.0):
            raise ValueError(
                f"--ema-decay-finisher ({args.ema_decay_finisher}) must be in (0, 1).")
        _ema_fin_start = (args.ema_decay_finisher_start_epoch
                          if args.ema_decay_finisher_start_epoch is not None else args.muon_start_epoch)
        if _ema_fin_start is None:
            raise ValueError(
                "--ema-decay-finisher requires a start epoch: set --ema-decay-finisher-start-epoch "
                "(or --muon-start-epoch, which it falls back to). Without one the wider EMA would "
                "NEVER engage = a silent no-op = a FALSE 'wider EMA does nothing' verdict.")
        if not (1 <= _ema_fin_start <= args.epochs):
            raise ValueError(
                f"--ema-decay-finisher start epoch ({_ema_fin_start}) must be in [1, --epochs "
                f"({args.epochs})]: outside the budget the wider EMA would never engage = a silent "
                "no-op.")
    elif args.ema_decay_finisher_start_epoch is not None:
        raise ValueError(
            "--ema-decay-finisher-start-epoch set without --ema-decay-finisher: the start epoch has "
            "no effect = a silent no-op. Set --ema-decay-finisher too, or drop the start flag.")

    # (THETA* TIER-2 STRETCH-1) junction-aware Eikonal relax fail-closed guards.
    if not (0.0 <= args.eikonal_junction_relax < 1.0):
        raise ValueError(
            f"--eikonal-junction-relax ({args.eikonal_junction_relax}) must be in [0, 1) (0 = OFF; "
            "the weight 1-relax*exp(-g3/tau) must stay positive).")
    if args.eikonal_junction_relax > 0.0 and args.eikonal_junction_tau <= 0.0:
        raise ValueError(
            f"--eikonal-junction-tau ({args.eikonal_junction_tau}) must be > 0 (the top2-top3 SDF-gap "
            "scale in exp(-g3/tau)).")

    # (EIK-STAB build 1/2/4) fail-closed config guards (silent-no-op / silent-drop NO-FAKE class).
    if float(args.eikonal_steik_weight) < 0.0:
        raise ValueError(f"--eikonal-steik-weight ({args.eikonal_steik_weight}) must be >= 0.")
    if bool(getattr(args, "eikonal_steik_normalized", False)) and float(args.eikonal_steik_weight) <= 0.0:
        raise ValueError(
            "--eikonal-steik-normalized set without --eikonal-steik-weight > 0: the normalized "
            "n^T H n term has no effect = a silent no-op. Set --eikonal-steik-weight too, or drop "
            "the --eikonal-steik-normalized flag (fail-closed per NO-FAKE).")
    if float(getattr(args, "eikonal_steik_norm_eps", 1e-2)) <= 0.0:
        raise ValueError(
            f"--eikonal-steik-norm-eps ({args.eikonal_steik_norm_eps}) must be > 0 "
            "(regularizes n = grad m/|grad m| where |grad m|->0; <=0 divides by zero).")
    if (bool(getattr(args, "eikonal_steik_normalized", False))
            and int(getattr(args, "micro_batch_pairs", 1)) > 1):
        raise ValueError(
            "--eikonal-steik-normalized is NOT wired into the micro-batch twin "
            "(tac.boundary_math.levelset_micro_batch_loss receives the RAW _eikonal_steik_mlx). "
            "Combining it with --micro-batch-pairs>1 would SILENTLY use the raw self-amplifying form "
            "= NO-FAKE silent-drop. Run --eikonal-steik-normalized with --accum-pairs (serial path) "
            "until the twin threads the normalized fn.")
    if float(args.eikonal_viscosity) < 0.0:
        raise ValueError(f"--eikonal-viscosity ({args.eikonal_viscosity}) must be >= 0.")
    if float(args.eikonal_viscosity) > 0.0 and float(args.eikonal_junction_relax) > 0.0:
        raise ValueError(
            "--eikonal-viscosity > 0 REPLACES the eikonal residual (central interior stencil) and "
            "does NOT carry the junction-relax weight; composing both would SILENTLY drop the "
            "junction relax. Run one or the other (fail-closed per NO-FAKE).")
    if int(args.eikonal_viscosity_anneal) > 0 and float(args.eikonal_viscosity) <= 0.0:
        raise ValueError(
            "--eikonal-viscosity-anneal set without --eikonal-viscosity > 0: the anneal has no "
            "effect = a silent no-op. Set --eikonal-viscosity too, or drop the anneal flag.")
    if args.spike_guard_mode == "rollback":
        if int(args.spike_rollback_window) < 1:
            raise ValueError(f"--spike-rollback-window ({args.spike_rollback_window}) must be >= 1.")
        if not (0.0 < float(args.spike_rollback_frac) <= 1.0):
            raise ValueError(f"--spike-rollback-frac ({args.spike_rollback_frac}) must be in (0, 1].")
        if not (0.0 < float(args.spike_rollback_lr_cut) < 1.0):
            raise ValueError(f"--spike-rollback-lr-cut ({args.spike_rollback_lr_cut}) must be in (0, 1).")
        if int(args.spike_rollback_max) < 1:
            raise ValueError(f"--spike-rollback-max ({args.spike_rollback_max}) must be >= 1.")
    if int(args.lambda_pre_probe_iters) < 0:
        raise ValueError(f"--lambda-pre-probe-iters ({args.lambda_pre_probe_iters}) must be >= 0.")
    if int(args.lambda_pre_probe_iters) > 0 and float(args.lambda_pre_probe_fd_eps) <= 0.0:
        raise ValueError(f"--lambda-pre-probe-fd-eps ({args.lambda_pre_probe_fd_eps}) must be > 0.")

    result = run_train(args)
    print("\n=== LEVEL-SET WITNESS RESULT (realized through R) ===")
    print(json.dumps({"front_end": result["front_end"], "history": result["history"],
                      "axis": result["axis"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
