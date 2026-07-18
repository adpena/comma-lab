# SPDX-License-Identifier: MIT
"""Shared exact-factorization SENSE core for the costate organ (A/B/C upgrades, 2026-07-17).

ONE module holds the REAL inputs the three factorization-grounded organ upgrades share, so
none of them re-derives (or fakes) the physics:

* the EXACT shared resize operator ``A`` (camera (874,1164) -> scorer (384,512); upstream
  ``SegNet.preprocess_input`` = ``F.interpolate(mode='bilinear')``, align_corners default
  = False, no antialias).  ``A`` is separable and SPARSE: each output coordinate reads at
  most 2 input taps per axis, so a closed-form tap table gives the machine-checkable
  ``range(A)`` / ``ker(A)`` split.  The certified-zero-weight (blind) camera fraction this
  module computes is verified in tests against BOTH the live torch operator (one-hot probe)
  and the canonical measured constant
  ``realization_necessity_preimage_per_stratum_v1.CAMERA_SUPPORT_FRAC
  ['certified_free_zero_weight'] = 0.226969``.
* the LIVE witness EMA checkpoint loader (the canonical npz->manifest mapping from
  ``tools/build_witness_showcase._load_witness`` — consumed, NOT re-typed; that mapping
  carries the max_bank_freq sentinel subtlety) + the canonical torch decode
  (``tac.local_acceleration.torch_levelset_inflate.decode_levelset_torch``) so the frames
  scored here are the REAL realized-through-R witness frames.
* the frozen CPU-torch SegNet (sha256-verified against the canonical equation's pinned
  weights hash) and a margin SNAPSHOT: per remaining flip pixel (witness argmax != GT
  argmax on the cached bit-exact ``lstars``) the EXACT pairwise logit margin
  ``m = z_wrong - z_gt`` the correction must cross.  The rank-4 head law
  (``segnet_head_rank4_linear_flipdist_v1``) turns those margins into closed-form
  feature-space flip distances ``m / ||w_c - w_c'||``.

Everything here is READ-ONLY against the live run dir (it only reads npz/log artifacts)
and advisory: ``[macOS-CPU advisory] NON-PROMOTABLE``, ``score_claim=False`` always.
Fail-closed: missing checkpoint / GT cache / weights-hash mismatch RAISES — nothing is
fabricated (NO-FAKE supreme rule).
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tac.canonical_equations.realization_necessity_preimage_20260715 import (
    CAMERA_SUPPORT_FRAC,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    HEAD_PAIR_NORMS,
    SEGNET_WEIGHTS_SHA256,
)

_REPO = Path(__file__).resolve().parents[3]

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)  # (H, W) — upstream segnet_model_input_size == (512, 384) as (W, H)
AXIS_TAG = "[macOS-CPU advisory] NON-PROMOTABLE"

#: fixed log-spaced margin histogram edges (logit units) so snapshot rows persisted to
#: JSONL can be re-consumed by the duty ranking without per-pixel dumps.  Range covers the
#: measured boundary-margin scale (canonical medians 0.08-0.18 logits) out to deep-interior.
MARGIN_HIST_EDGES: np.ndarray = np.geomspace(1e-3, 30.0, 49)


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def pair_key(a: int, b: int) -> str:
    """Canonical UNORIENTED pair name (class-index order, matching HEAD_PAIR_NORMS keys)."""
    lo, hi = (a, b) if a < b else (b, a)
    return f"{CLASS_NAMES[lo]}-{CLASS_NAMES[hi]}"


def oriented_key(wrong: int, gt: int) -> str:
    """ORIENTED flip name ``wrong->gt`` (the correction direction matters for levers)."""
    return f"{CLASS_NAMES[wrong]}->{CLASS_NAMES[gt]}"


def parse_oriented_key(key: str) -> tuple[int, int]:
    w, g = key.split("->", 1)
    return CLASS_NAMES.index(w), CLASS_NAMES.index(g)


def pair_norm_for_oriented(key: str) -> float:
    """||w_c - w_c'|| for an oriented flip key (norm is orientation-independent)."""
    w, g = parse_oriented_key(key)
    return float(HEAD_PAIR_NORMS[pair_key(w, g)])


# ---------------------------------------------------------------------------
# EXACT shared resize operator A (closed form; verified against torch in tests)
# ---------------------------------------------------------------------------
def resize_taps(n_out: int, n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form 1-D bilinear taps of ``F.interpolate(..., mode='bilinear')`` with the
    default ``align_corners=False`` convention: ``src = (o + 0.5) * (n_in/n_out) - 0.5``,
    clamped at 0 (torch's ``area_pixel_compute_source_index``), two taps ``floor(src)`` and
    ``floor(src)+1`` (index-clamped) with weights ``(1-frac, frac)``.

    Returns ``(i0, i1, w0, w1)`` each of shape ``(n_out,)``.  This IS the operator the
    frozen scorer applies per axis — no surrogate."""
    if n_out <= 0 or n_in <= 0:
        raise ValueError("sizes must be positive")
    scale = n_in / n_out
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
    src = np.maximum(src, 0.0)
    i0f = np.floor(src)
    frac = src - i0f
    i0 = np.clip(i0f.astype(np.int64), 0, n_in - 1)
    i1 = np.clip(i0 + 1, 0, n_in - 1)
    return i0, i1, (1.0 - frac), frac


def touched_1d(n_out: int, n_in: int) -> np.ndarray:
    """Boolean (n_in,): True where the axis coordinate carries NONZERO weight in A."""
    i0, i1, w0, w1 = resize_taps(n_out, n_in)
    touched = np.zeros(n_in, dtype=bool)
    touched[i0[w0 > 0.0]] = True
    touched[i1[w1 > 0.0]] = True
    return touched


def ker_a_zero_weight_mask(
    scorer_hw: tuple[int, int] = SCORER_HW, camera_hw: tuple[int, int] = CAMERA_HW
) -> np.ndarray:
    """EXACT certified-zero-weight camera mask (True = ker(A) / scorer-blind).

    A camera pixel is blind iff its ROW is untouched by the height taps OR its COLUMN is
    untouched by the width taps (A is separable, weights are products).  Blind pixels have
    EXACTLY zero influence on every scorer-input value — the zero-marginal theorem the duty
    ranking relies on is this sparsity, not an approximation."""
    rows = touched_1d(scorer_hw[0], camera_hw[0])
    cols = touched_1d(scorer_hw[1], camera_hw[1])
    return ~(rows[:, None] & cols[None, :])


def verify_ker_mask_against_canonical(atol: float = 2e-3) -> dict:
    """Cross-check the closed-form blind fraction against the canonical MEASURED constant
    (realization_necessity: certified_free_zero_weight = 0.226969).  Returns the comparison;
    raises on disagreement (fail-closed: a convention drift would poison every consumer)."""
    mask = ker_a_zero_weight_mask()
    frac = float(mask.mean())
    canon = float(CAMERA_SUPPORT_FRAC["certified_free_zero_weight"])
    if abs(frac - canon) > atol:
        raise AssertionError(
            f"ker(A) zero-weight fraction {frac:.6f} disagrees with canonical "
            f"{canon:.6f} (>|{atol}|) — resize convention drift; refusing"
        )
    return {"closed_form_zero_weight_frac": frac, "canonical": canon, "abs_diff": abs(frac - canon)}


def visible_energy_split(camera_map: np.ndarray, ker_mask: np.ndarray | None = None) -> dict:
    """Energy split of a camera-space map into range(A)-visible vs ker(A)-blind support.

    ``camera_map``: (874,1164) or (874,1164,C) real array (e.g. a witness-vs-GT residual).
    Returns visible/blind energy and the visible fraction.  Exact w.r.t. SUPPORT (a blind
    pixel's value never reaches the scorer input)."""
    m = np.asarray(camera_map, dtype=np.float64)
    if m.shape[:2] != CAMERA_HW:
        raise ValueError(f"camera_map must be {CAMERA_HW}[,C], got {m.shape}")
    k = ker_a_zero_weight_mask() if ker_mask is None else np.asarray(ker_mask, dtype=bool)
    e = m * m
    if e.ndim == 3:
        e = e.sum(axis=-1)
    total = float(e.sum())
    blind = float(e[k].sum())
    visible = total - blind
    return {
        "energy_total": total,
        "energy_visible": visible,
        "energy_blind": blind,
        "visible_frac": (visible / total) if total > 0 else None,
        "ker_zero_weight_frac": float(k.mean()),
    }


# ---------------------------------------------------------------------------
# REAL inputs: live EMA checkpoint, frozen SegNet, cached GT
# ---------------------------------------------------------------------------
def load_witness_ema(ckpt_path: str | Path) -> tuple[dict, dict, np.ndarray]:
    """Load a witness EMA npz -> (manifest, params, code) via the CANONICAL npz->manifest
    mapping in ``tools/build_witness_showcase._load_witness`` (consumed, not re-typed —
    it owns the max_bank_freq sentinel handling).  Fail-closed on a missing file."""
    p = Path(ckpt_path)
    if not p.is_file():
        raise FileNotFoundError(f"witness EMA checkpoint not found: {p}")
    tools_dir = _REPO / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from build_witness_showcase import _load_witness  # canonical mapping

    return _load_witness(p)


def load_frozen_segnet_cpu(upstream_dir: str | Path | None = None):
    """Load the REAL frozen contest SegNet on CPU, sha256-verified against the canonical
    equation's pinned hash.  Raises on missing weights or hash mismatch (NO-FAKE).
    ``TAC_UPSTREAM_DIR`` overrides the default (worktrees lack the big model files)."""
    import os

    up = Path(upstream_dir or os.environ.get("TAC_UPSTREAM_DIR") or (_REPO / "upstream"))
    weights = up / "models" / "segnet.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"frozen SegNet weights not found: {weights}")
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    if sha != SEGNET_WEIGHTS_SHA256:
        raise AssertionError(
            f"segnet.safetensors sha256 {sha} != canonical {SEGNET_WEIGHTS_SHA256} — refusing"
        )
    if str(up) not in sys.path:
        sys.path.insert(0, str(up))
    from modules import SegNet  # upstream (pinned snapshot; read-only import)
    from safetensors.torch import load_file

    net = SegNet()
    net.load_state_dict(load_file(str(weights)), strict=True)
    net.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    return net


def decode_pairs_camera_frames(
    manifest: dict, params: dict, code: np.ndarray, pair_indices: list[int], device: str = "cpu"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decode SELECTED pairs from the real checkpoint through the canonical torch decode
    (realized through R: bicubic up -> round/clamp -> uint8 camera frames).  Slices the
    per-pair code rows; the decode function itself is the canonical one (not forked)."""
    from tac.local_acceleration.torch_levelset_inflate import decode_levelset_torch

    code = np.asarray(code)
    rows: list[int] = []
    for pi in pair_indices:
        if not (0 <= 2 * pi + 1 < code.shape[0]):
            raise IndexError(f"pair {pi} out of range for code shape {code.shape}")
        rows += [2 * pi, 2 * pi + 1]
    m = dict(manifest)
    m["n_pairs"] = len(pair_indices)
    out = decode_levelset_torch(m, params, code[rows], device=device, return_frames=True)
    return out["frames"]


def segnet_logits_for_frames(segnet_cpu, frames1_uint8: list[np.ndarray], batch: int = 4) -> np.ndarray:
    """Frozen-SegNet logits for camera frame1s, through the REAL upstream preprocess
    (``segnet.preprocess_input`` — the exact resize A).  Returns (N,5,384,512) float32.
    Mirrors ``cpu_verdict_d_seg_argmax_batch`` (bit-identical preprocess+forward), but keeps
    the logits so margins are available (the verdict helper only keeps the argmax)."""
    import torch

    chunks: list[np.ndarray] = []
    for s in range(0, len(frames1_uint8), max(int(batch), 1)):
        sub = frames1_uint8[s : s + max(int(batch), 1)]
        arr = np.stack([np.asarray(f)[None] for f in sub], axis=0)  # (n,1,H,W,3)
        xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            seg_in = segnet_cpu.preprocess_input(xp)
            logits = segnet_cpu(seg_in)
        chunks.append(logits.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def load_gt_slices(gt_cache: str | Path, pair_indices: list[int], want_frames: bool = False) -> dict:
    """Slice the bit-exact GT cache for the selected pairs.  Fail-closed on missing keys."""
    p = Path(gt_cache)
    if not p.is_file():
        raise FileNotFoundError(f"GT cache not found: {p}")
    z = np.load(p, allow_pickle=False)
    if "lstars" not in z.files:
        raise KeyError(f"GT cache {p} lacks 'lstars' (frozen SegNet argmax authority)")
    idx = np.asarray(pair_indices, dtype=np.int64)
    lst = np.asarray(z["lstars"])
    if idx.max(initial=-1) >= lst.shape[0]:
        raise IndexError(f"pair index {int(idx.max())} >= cache n_pairs {lst.shape[0]}")
    out: dict = {"lstars": lst[idx].copy()}
    del lst
    if want_frames:
        if "gt_f1" not in z.files:
            raise KeyError(f"GT cache {p} lacks 'gt_f1' (camera frames for the energy split)")
        gtf = np.asarray(z["gt_f1"])
        out["gt_f1"] = gtf[idx].copy()
        del gtf
    return out


# ---------------------------------------------------------------------------
# The margin snapshot (shared by duty ranking A, realization regime B, ingest C)
# ---------------------------------------------------------------------------
@dataclass
class MarginSnapshot:
    """Per-flip-pixel exact pairwise margins of the REAL witness vs the GT argmax."""

    run_ref: str
    ema_epoch: int
    generated_at: str
    pair_indices: tuple[int, ...]
    scorer_hw: tuple[int, int]
    total_px: int                                   # n_sampled_pairs * H * W (d_seg denominator)
    d_seg_sample: float                             # measured on THIS sample (advisory)
    flip_pair_idx: np.ndarray                       # (F,) index into pair_indices
    flip_y: np.ndarray                              # (F,)
    flip_x: np.ndarray                              # (F,)
    flip_wrong: np.ndarray                          # (F,) witness argmax label
    flip_gt: np.ndarray                             # (F,) GT label
    flip_margin: np.ndarray                         # (F,) exact z_wrong - z_gt (>0) logits
    frames1: list[np.ndarray] = field(default_factory=list)      # camera uint8 (optional keep)
    witness_argmax: np.ndarray | None = None        # (N,384,512) realized argmax (optional)
    axis_tag: str = AXIS_TAG
    score_claim: bool = False

    @property
    def n_flips(self) -> int:
        return int(self.flip_margin.size)

    def margins_by_oriented_pair(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for w in range(5):
            for g in range(5):
                if w == g:
                    continue
                sel = (self.flip_wrong == w) & (self.flip_gt == g)
                if np.any(sel):
                    out[oriented_key(w, g)] = np.sort(self.flip_margin[sel])
        return out

    def flipdist_feature_space_by_oriented_pair(self) -> dict[str, np.ndarray]:
        """Head-law flip distances m/||w_c-w_c'|| per oriented pair (exact in feature space,
        per segnet_head_rank4_linear_flipdist_v1)."""
        return {
            k: v / pair_norm_for_oriented(k)
            for k, v in self.margins_by_oriented_pair().items()
        }

    def summary_row(self) -> dict:
        """Compact JSONL-persistable summary: per-oriented-pair counts + margin histograms
        (fixed MARGIN_HIST_EDGES) + flip-distance quantiles.  No per-pixel dumps."""
        edges = MARGIN_HIST_EDGES
        by_pair: dict[str, dict] = {}
        for k, m in self.margins_by_oriented_pair().items():
            hist, _ = np.histogram(m, bins=edges)
            fd = m / pair_norm_for_oriented(k)
            by_pair[k] = {
                "n": int(m.size),
                "margin_hist": hist.astype(int).tolist(),
                "margin_underflow": int(np.count_nonzero(m < edges[0])),
                "margin_overflow": int(np.count_nonzero(m >= edges[-1])),
                "margin_q": {q: float(np.quantile(m, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                "flipdist_feat_q": {q: float(np.quantile(fd, q)) for q in (0.1, 0.5, 0.9)},
            }
        return {
            "schema": "witness_factorized_snapshot.v1",
            "run_ref": self.run_ref,
            "ema_epoch": self.ema_epoch,
            "generated_at": self.generated_at,
            "pair_indices": list(self.pair_indices),
            "n_pairs_sampled": len(self.pair_indices),
            "total_px": self.total_px,
            "d_seg_sample": self.d_seg_sample,
            "n_flips": self.n_flips,
            "margin_hist_edges": [float(e) for e in edges],
            "by_oriented_pair": by_pair,
            "axis_tag": self.axis_tag,
            "score_claim": False,
        }


def default_pair_sample(n_pairs_total: int = 600, n_sample: int = 24) -> list[int]:
    """Deterministic stride sample over the scored pairs (stride-25 at the defaults —
    the same subset convention the necessity solver's A-support stage used, labeled)."""
    n_sample = max(1, min(int(n_sample), int(n_pairs_total)))
    stride = max(1, n_pairs_total // n_sample)
    return list(range(0, n_pairs_total, stride))[:n_sample]


def snapshot_witness_margins(
    ema_ckpt: str | Path,
    gt_cache: str | Path,
    pair_indices: list[int] | None = None,
    *,
    segnet_cpu=None,
    keep_frames: bool = False,
    keep_argmax: bool = False,
    run_ref: str | None = None,
    decode_device: str = "cpu",
) -> MarginSnapshot:
    """Build the REAL margin snapshot: decode the live EMA checkpoint through R, score with
    the frozen SegNet, and extract the exact pairwise margin at every remaining flip pixel.

    Every quantity is computed from the actual bytes on disk (checkpoint + GT cache + frozen
    weights); nothing is estimated.  Advisory, subset-labeled, never a score."""
    manifest, params, code = load_witness_ema(ema_ckpt)
    n_total = int(manifest["n_pairs"])
    pairs = default_pair_sample(n_total) if pair_indices is None else [int(i) for i in pair_indices]
    gt = load_gt_slices(gt_cache, pairs, want_frames=False)
    if segnet_cpu is None:
        segnet_cpu = load_frozen_segnet_cpu()

    frames = decode_pairs_camera_frames(manifest, params, code, pairs, device=decode_device)
    frames1 = [f1 for (_f0, f1) in frames]
    logits = segnet_logits_for_frames(segnet_cpu, frames1)          # (N,5,h,w)
    realized = logits.argmax(axis=1).astype(np.int64)               # (N,h,w)
    lstars = gt["lstars"].astype(np.int64)
    if realized.shape != lstars.shape:
        raise AssertionError(f"argmax/GT shape mismatch: {realized.shape} vs {lstars.shape}")

    flips = realized != lstars
    pidx, ys, xs = np.nonzero(flips)
    wrong = realized[pidx, ys, xs]
    gtl = lstars[pidx, ys, xs]
    # exact pairwise margin the correction must cross: z_wrong - z_gt at the pixel (>0 by
    # construction because wrong is the argmax; assert, don't assume)
    z_w = logits[pidx, wrong, ys, xs].astype(np.float64)
    z_g = logits[pidx, gtl, ys, xs].astype(np.float64)
    margin = z_w - z_g
    if margin.size and float(margin.min()) < 0.0:
        raise AssertionError("pairwise margin z_wrong - z_gt < 0 at a flip pixel — logic error")

    total_px = int(np.prod(lstars.shape))
    d_seg_sample = float(np.count_nonzero(flips)) / float(total_px)
    ep = int(manifest.get("epoch", -1))
    return MarginSnapshot(
        run_ref=run_ref or Path(ema_ckpt).parent.name,
        ema_epoch=ep,
        generated_at=utc_stamp(),
        pair_indices=tuple(pairs),
        scorer_hw=(int(lstars.shape[1]), int(lstars.shape[2])),
        total_px=total_px,
        d_seg_sample=d_seg_sample,
        flip_pair_idx=pidx.astype(np.int32),
        flip_y=ys.astype(np.int32),
        flip_x=xs.astype(np.int32),
        flip_wrong=wrong.astype(np.int8),
        flip_gt=gtl.astype(np.int8),
        flip_margin=margin.astype(np.float64),
        frames1=frames1 if keep_frames else [],
        witness_argmax=realized if keep_argmax else None,
    )


def locked_append_jsonl(path: str | Path, row: dict) -> None:
    """fcntl-locked JSONL append (mirrors the .omx/state discipline; small rows only)."""
    import fcntl

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


__all__ = [
    "AXIS_TAG",
    "CAMERA_HW",
    "CLASS_NAMES",
    "MARGIN_HIST_EDGES",
    "SCORER_HW",
    "MarginSnapshot",
    "decode_pairs_camera_frames",
    "default_pair_sample",
    "ker_a_zero_weight_mask",
    "load_frozen_segnet_cpu",
    "load_gt_slices",
    "load_witness_ema",
    "locked_append_jsonl",
    "oriented_key",
    "pair_key",
    "pair_norm_for_oriented",
    "parse_oriented_key",
    "resize_taps",
    "segnet_logits_for_frames",
    "snapshot_witness_margins",
    "touched_1d",
    "verify_ker_mask_against_canonical",
    "visible_energy_split",
]
