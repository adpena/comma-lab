#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""WITNESS CAPSTONE deep-math descent smoke — improved ScoreNativeSegGenerator (MLX-first).

The lever-B baseline (``tools/lever_b_score_native_argmax_smoke.py``) PROVED the seg term is
generatable (d_seg=0.008257 over 600 pairs at hidden=96/n_hidden=4/n_fourier=16/mod=32, an
85.4K-param net, 63.8 KB quantized blob) but PLATEAUED there. This is the NON-RGB TASK-SPACE
WITNESS capstone (#171/#78): the nonlinear coordinate-INR that AMORTIZES the SegNet argmax
partition directly, spending capacity on the scorer-relevant manifold instead of full RGB.

THE UNIT'S DECISIVE QUESTION: do the deep-math levers DRIVE d_seg DOWN from 0.008257 toward the
sub-0.15 need (~0.001)?

DEEP-MATH LEVERS (grounded in the crux: residual = class-1 lane edges = ~8-dim nonlinear
lane-orbit manifold; flips concentrate at the shallow-margin codim-1 boundary):
  L1 CAPACITY — sweep hidden_dim / n_hidden / n_fourier / mod_dim UP (cheap ∵ non-RGB).
  L2 BOUNDARY-PROXIMITY INPUT FEATURE — a precomputed, per-pair, per-pixel boundary-proximity
     key exp(-d_boundary/tau) (d_boundary from GT class-1 EDT, scipy) concatenated to the coord
     features. This is the KKT-waterfill routing prior: the net SEES where the binding band is
     and can spend capacity there. ZERO archive bytes (a deterministic function of the GT argmax
     the witness already reproduces — see NO-FAKE note below).
  L3 DIRECTIONAL / ANISOTROPIC FOURIER FEATURES — per-pair, per-pixel oriented Fourier features
     whose frequency is HIGH across the lane edge (normal) and LOW along it (tangent), the
     curvelet O(N^-2) >> isotropic O(N^-1) inductive bias for a codim-1 smooth curve. Tangent
     field precomputed from the GT signed-distance gradient (scipy).
  L4 (optional) GAUSSIAN/step activation in place of ReLU.

BORROWED-SUBSTRATE ACCOUNTING (CLAUDE.md NO-FAKE):
  * BORROWED: scipy.ndimage.distance_transform_edt (Felzenszwalb-Huttenlocher / Maurer EDT).
  * BORROWED: isotropic random Fourier features (Tancik 2020); anisotropic/oriented generalization
    (AFPE, Kuckelhaus 2025; steerable filters Freeman-Adelson 1991; curvelets Candes-Donoho 2004);
    FiLM-per-instance modulation (Perez 2018); HNeRV amortized-decoder framing.
  * OURS: using the GT class-1 boundary distance/tangent as the routing KEY + orientation field for
    a SMALL non-RGB argmax witness INR specifically to descend the contest's binding d_seg lane-edge
    band; the boundary-proximity-as-input-feature codec-free design; the per-pair oriented PE.

WITNESS_DSEG_FEASIBILITY_ONLY (operator paranoia 2026-06-26, R3 harness audit):
d_seg here is a PROXY — the argmax-disagreement RATE between the GENERATOR'S OWN per-pixel
argmax and the FROZEN SegNet's GT argmax over 196,608 px x num_pairs. It is FEASIBILITY-ONLY,
NOT a realized score: it SKIPS the contest reconstruction operator R (bicubic-up -> uint8 @
camera -> bilinear-down) AND the SegNet RE-segmentation of the rendered RGB frame. R3 measured
this proxy ~170-350x optimistic vs the realized quantity. The earlier docstring claim ("EXACT
score-native quantity") was the exact FAKE this re-founding extincts. The ONLY witness d_seg
SCORE authority is the realized harness (render -> _torch_R_to_camera_uint8 -> real CPU-torch
SegNet argmax), validated to >=6 decimals vs upstream/evaluate.py by
experiments/validate_realized_harness_vs_oracle.py (see tac.measurement_integrity). A stub that
returns a constant still FAILS (the descent histogram must move) — but a moving proxy descent is
a feasibility upper-bound, never a frontier/promote/kill signal.

MLX-FIRST: forward + train are MLX (MPS gradient OK — bc20 arms are DEAD so MPS is free). The
proxy d_seg references the frozen SegNet argmax (precomputed CPU-torch). EVIDENCE
[macOS-MLX research-signal], promotion_eligible=false, NO score claim. Disk: SSD tier, NEVER /tmp.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
_FOURIER_SEED = 0


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# Deep-math feature precompute (numpy/scipy, portable, train-time prior, 0-byte)
# ---------------------------------------------------------------------------
def _all_class_boundary_mask(argmax_hw: np.ndarray) -> np.ndarray:
    """Pixels on ANY inter-class boundary (argmax differs from a 4-neighbor).

    The flips are NOT lane-only — they concentrate at EVERY class boundary
    (measured: only ~19% of small-margin px are class-1; ~50% class-0). So the
    routing/orientation prior must key on the union of all class edges, not just
    class-1. Pure numpy.
    """
    a = argmax_hw
    b = np.zeros_like(a, dtype=bool)
    b[:-1, :] |= a[:-1, :] != a[1:, :]
    b[1:, :] |= a[:-1, :] != a[1:, :]
    b[:, :-1] |= a[:, :-1] != a[:, 1:]
    b[:, 1:] |= a[:, :-1] != a[:, 1:]
    return b


def boundary_proximity_and_tangent(
    argmax_hw: np.ndarray, lane_class: int, tau: float, *, all_class: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair boundary-proximity key + unit tangent field from the GT argmax.

    Returns:
        prox: (H, W) float32 in [0,1] = exp(-|sdf|/tau) (1 on the lane-edge band).
        tangent: (H, W, 2) unit tangent (tx, ty) of the lane boundary (rotated EDT gradient).

    Borrowed: scipy EDT. The signed distance sdf = EDT(~lane) - EDT(lane); the tangent is the
    90deg rotation of grad(sdf) (the boundary normal). Pure CPU/numpy.
    """
    from scipy import ndimage

    H, W = argmax_hw.shape
    if all_class:
        # Distance to the UNION of all inter-class boundaries (the real flip locus).
        bnd = _all_class_boundary_mask(argmax_hw)
        if not bnd.any():
            diag = float(np.hypot(H, W))
            prox = np.exp(-np.abs(np.full((H, W), diag, np.float32)) / tau).astype(np.float32)
            tang = np.zeros((H, W, 2), np.float32)
            tang[..., 0] = 1.0
            return prox, tang
        # unsigned distance to the boundary curve; sdf for the tangent is the same
        # field (no inside/outside for a union of curves) -> use distance gradient.
        dist = ndimage.distance_transform_edt(~bnd).astype(np.float32)
        prox = np.exp(-dist / tau).astype(np.float32)
        sdf = dist  # gradient of distance-to-curve points along the normal
        gy = np.zeros_like(sdf)
        gx = np.zeros_like(sdf)
        gy[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) * 0.5
        gx[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) * 0.5
        tx = -gy
        ty = gx
        nrm = np.sqrt(tx * tx + ty * ty)
        flat = nrm < 1e-6
        nrm = np.maximum(nrm, 1e-8)
        tx = tx / nrm
        ty = ty / nrm
        tx[flat] = 1.0
        ty[flat] = 0.0
        return prox, np.stack([tx, ty], axis=-1).astype(np.float32)
    lane = argmax_hw == lane_class
    if not lane.any() or lane.all():
        diag = float(np.hypot(H, W))
        sdf = np.full((H, W), diag, np.float32)
        prox = np.exp(-np.abs(sdf) / tau).astype(np.float32)
        tang = np.zeros((H, W, 2), np.float32)
        tang[..., 0] = 1.0
        return prox, tang
    d_out = ndimage.distance_transform_edt(~lane).astype(np.float32)
    d_in = ndimage.distance_transform_edt(lane).astype(np.float32)
    sdf = d_out - d_in
    prox = np.exp(-np.abs(sdf) / tau).astype(np.float32)
    # tangent = rotate grad(sdf) 90deg.
    gy = np.zeros_like(sdf)
    gx = np.zeros_like(sdf)
    gy[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) * 0.5
    gx[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) * 0.5
    tx = -gy
    ty = gx
    nrm = np.sqrt(tx * tx + ty * ty)
    flat = nrm < 1e-6
    nrm = np.maximum(nrm, 1e-8)
    tx = tx / nrm
    ty = ty / nrm
    tx[flat] = 1.0
    ty[flat] = 0.0
    return prox, np.stack([tx, ty], axis=-1).astype(np.float32)


def directional_fourier_feats(
    coords: np.ndarray,  # (P,2) in [-1,1]
    tangent: np.ndarray,  # (P,2) unit tangent
    n_freqs: int,
    freq_across: float,
    freq_along: float,
) -> np.ndarray:
    """Per-pixel oriented Fourier features (P, 4*n_freqs). Deterministic, 0-byte."""
    tx = tangent[:, 0]
    ty = tangent[:, 1]
    nx = ty
    ny = -tx
    cx = coords[:, 0]
    cy = coords[:, 1]
    u_t = cx * tx + cy * ty  # along edge
    u_n = cx * nx + cy * ny  # across edge
    two_pi = 2.0 * np.pi
    feats = []
    for k in range(n_freqs):
        fa = float(freq_along) * (2.0**k)
        fc = float(freq_across) * (2.0**k)
        feats.append(np.sin(two_pi * fa * u_t))
        feats.append(np.cos(two_pi * fa * u_t))
        feats.append(np.sin(two_pi * fc * u_n))
        feats.append(np.cos(two_pi * fc * u_n))
    return np.stack(feats, axis=-1).astype(np.float32)


def deterministic_fourier_B(n_fourier: int, fourier_sigma: float) -> np.ndarray:
    rng = np.random.default_rng(_FOURIER_SEED)
    return (rng.standard_normal((2, n_fourier)) * fourier_sigma).astype(np.float32)


# ---------------------------------------------------------------------------
# Improved generator: coord-INR + isotropic Fourier + optional boundary-prox
# input feature + optional directional Fourier + FiLM-per-pair modulation.
# ---------------------------------------------------------------------------
class ImprovedSegGenerator(nn.Module):
    """g(coords, prox, dirfeat, pair) -> 5-class logits.

    Input feature vector per pixel = [sin/cos(coords @ B_iso)]  (2*n_fourier)
                                   + [prox]                       (1, if use_prox)
                                   + [directional Fourier feats]  (4*n_dir_freqs, if use_dir)
    Base MLP (shared) + per-pair FiLM modulation on each hidden layer.
    """

    def __init__(
        self,
        num_pairs: int,
        n_fourier: int,
        hidden_dim: int,
        n_hidden: int,
        mod_dim: int,
        fourier_sigma: float,
        use_prox: bool,
        use_dir: bool,
        n_dir_freqs: int,
        activation: str,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.num_pairs = num_pairs
        self.n_fourier = n_fourier
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.mod_dim = mod_dim
        self.n_classes = n_classes
        self.use_prox = use_prox
        self.use_dir = use_dir
        self.n_dir_freqs = n_dir_freqs
        self.activation = activation
        B = deterministic_fourier_B(n_fourier, fourier_sigma)
        self._fourier_B = mx.array(B)
        in_feat = 2 * n_fourier
        if use_prox:
            in_feat += 1
        if use_dir:
            in_feat += 4 * n_dir_freqs
        self.in_feat = in_feat
        self.mod = mx.zeros((num_pairs, mod_dim))
        self.in_proj = nn.Linear(in_feat, hidden_dim)
        self.film = nn.Linear(mod_dim, 2 * hidden_dim * n_hidden)
        self.hidden = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        self.out = nn.Linear(hidden_dim, n_classes)

    def _act(self, x: mx.array) -> mx.array:
        if self.activation == "relu":
            return nn.relu(x)
        if self.activation == "gelu":
            return nn.gelu(x)
        if self.activation == "gauss":
            # Gaussian (RBF-like) activation: smooth, localized — step-friendly.
            return mx.exp(-(x * x))
        raise ValueError(f"unknown activation {self.activation!r}")

    def __call__(self, coord_feats: mx.array, pair_idx: int) -> mx.array:
        """coord_feats: (P, in_feat) prebuilt feature batch; returns (P, n_classes)."""
        h = self._act(self.in_proj(coord_feats))
        m = self.mod[pair_idx]
        film = self.film(m).reshape(self.n_hidden, 2, self.hidden_dim)
        for li, layer in enumerate(self.hidden):
            scale = 1.0 + film[li, 0]
            shift = film[li, 1]
            h = self._act(layer(h) * scale + shift)
        return self.out(h)


def hard_pixel_boost(
    pred_argmax: np.ndarray, gt_labels: np.ndarray, error_boost: float
) -> np.ndarray:
    """Per-pixel hard-pixel error boost = 1 + error_boost * (pred != gt).

    ADAPTED FROM PR74 (ph4ntom_drv, error_boost=49) + PR62 (fp4_mask_gen, error_boost=9):
    ``boost = 1.0 + (logits.argmax(1) != gt_mask).float() * error_boost`` applied to a
    per-pixel cross-entropy. The boost is a DYNAMIC, model-aware signal (it weights the
    pixels the CURRENT model gets WRONG), distinct from this witness's existing STATIC
    margin-weight (a GT-fixed proxy for where flips live). The two compose multiplicatively
    here: a pixel that is BOTH shallow-margin AND currently-misclassified gets the most weight,
    which is exactly the binding d_seg residual (the union of all inter-class edges that the
    generator still flips).

    BORROWED-SUBSTRATE ACCOUNTING: the error-driven hard-pixel reweighting is BORROWED from
    PR74/PR62 (their RGB-decoder seg loss). OURS = applying it to a NON-RGB coordinate-INR
    argmax witness, composed with the static margin-weight, on the argmax target only (no
    GT logits needed — works on the argmax/margin targets we already have; KL-on-logits, the
    sister PR62 term, is NOT adaptable without rebuilding 5-class GT logit targets).

    0 archive bytes: train-time only (the boost is computed from the model's own predictions
    and the GT argmax; nothing is stored).

    Args:
        pred_argmax: (N,) int per-pixel predicted class (model argmax at the current step).
        gt_labels: (N,) int per-pixel GT class.
        error_boost: multiplier on misclassified pixels (0.0 = disabled = no behavior change).

    Returns:
        (N,) float32 boost factor; 1.0 where correct, 1.0+error_boost where wrong.
    """
    wrong = (pred_argmax != gt_labels).astype(np.float32)
    return (1.0 + float(error_boost) * wrong).astype(np.float32)


def _np_softplus_act(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    if activation == "gauss":
        return np.exp(-(x * x))
    raise ValueError(activation)


def kd_kl_logits(
    student_logits: mx.array,  # (P, n_classes) raw witness logits
    teacher_logits: mx.array,  # (P, n_classes) raw FROZEN-SegNet teacher logits
    temperature: float,
) -> mx.array:
    """Soft-logit KD loss = T^2 * KL(softmax(teacher/T) || softmax(student/T)), per-pixel.

    Hinton-Vinyals-Dean 2015 ("Distilling the Knowledge in a Neural Network") /
    Quantizr PR62 ``kl_on_logits`` form: the teacher's FULL 5-class probability field is a DENSE
    target (the margin geometry the d_seg flips live on), strictly richer than the hard argmax.
    The T^2 factor restores the gradient magnitude (soft targets have 1/T^2 smaller gradients).

    Direction: KL(teacher || student) — the canonical distillation direction (the student is pulled
    toward the teacher's soft distribution; matches Hinton-2015 + nn KLDivLoss(student_logT, teacher_p)).
    Returns a (P,) per-pixel KL so the caller can apply the same margin/error weighting if desired.

    BORROWED (KD/kl_on_logits/T) accounting + 0-archive-byte (train-time only) are documented in the
    module docstring + tools/build_segnet_teacher_logits.py manifest.
    """
    t = float(temperature)
    log_p_student = student_logits / t - mx.logsumexp(student_logits / t, axis=-1, keepdims=True)
    log_p_teacher = teacher_logits / t - mx.logsumexp(teacher_logits / t, axis=-1, keepdims=True)
    p_teacher = mx.exp(log_p_teacher)
    # KL(teacher || student) = sum_c p_teacher * (log_p_teacher - log_p_student)
    kl = mx.sum(p_teacher * (log_p_teacher - log_p_student), axis=-1)
    return (t * t) * kl


def boundary_affinity_kd(
    stud_a: mx.array,  # (M, n_classes) student logits at pixel A
    stud_b: mx.array,  # (M, n_classes) student logits at adjacent pixel B
    teach_a: mx.array,  # (M, n_classes) teacher logits at pixel A
    teach_b: mx.array,  # (M, n_classes) teacher logits at adjacent pixel B
    temperature: float,
) -> mx.array:
    """Pair-wise boundary-AFFINITY KD: match the student's edge field to the teacher's.

    THE STRUCTURALLY-NEW LEVER (vs per-pixel CE/KD which SATURATE ~0.0024). The measured residual
    map (experiments/probe_witness_residual_localization.py) proved the witness's remaining flips
    are NOT a deep-interior representation failure: 87% hug the GT boundary within 1.5 px, 90% are
    at margin < 2, and 71% land where the FROZEN SegNet teacher is ITSELF near-tie (top2 gap < 1).
    The residual is the EDGE-PLACEMENT band (road<->lane = 47% of the tie-band). Per-pixel signals
    cannot supervise where the EDGE sits — that is a RELATIONAL (pair-wise) property of adjacent
    pixels. This term supplies exactly that.

    Mechanism (Liu et al. CVPR 2019 "Structured Knowledge Distillation", pair-wise scheme, ADAPTED
    to soft affinity): for each adjacent pixel pair (A, B), the affinity = the probability that A
    and B share a class = <softmax(A/T), softmax(B/T)> (the expected agreement; ~1 inside a region,
    drops at an edge). The loss pulls the student affinity toward the teacher affinity. Because the
    teacher's affinity encodes its EXACT (soft) edge geometry, the student learns to put its edge
    where SegNet puts its edge — the thing per-pixel KD is blind to.

    Returns (M,) per-pair affinity-match loss (squared error on the soft affinity). 0 archive
    bytes (train-time only; the teacher logits are the cached frozen authority).

    BORROWED-SUBSTRATE ACCOUNTING (CLAUDE.md NO-FAKE): the pair-wise affinity / inter-pixel
    relation KD is BORROWED from Liu-Chen-Liu-Qin-Luo-Wang CVPR 2019 (Structured Knowledge
    Distillation for Semantic Segmentation; pair-wise + holistic schemes) + the BPKD WACV-2024
    edge-decoupling insight (Liu et al.; edge regions need higher spatial sensitivity than body).
    OURS = (a) applying soft pair-wise affinity to a NON-RGB coordinate-INR argmax witness (it has
    no feature maps to mimic; we mimic the OUTPUT edge field instead), (b) keying it on the FROZEN
    contest SegNet as the self-affinity teacher, (c) the spatially-adjacent-coordinate-pair sampling
    that makes the relational term computable in the witness's random-pixel SGD loop.
    """
    t = float(temperature)
    ps_a = mx.softmax(stud_a / t, axis=-1)
    ps_b = mx.softmax(stud_b / t, axis=-1)
    pt_a = mx.softmax(teach_a / t, axis=-1)
    pt_b = mx.softmax(teach_b / t, axis=-1)
    aff_s = mx.sum(ps_a * ps_b, axis=-1)  # (M,) student soft agreement in [0,1]
    aff_t = mx.sum(pt_a * pt_b, axis=-1)  # (M,) teacher soft agreement in [0,1]
    return (aff_s - aff_t) ** 2


def load_teacher_logits(teacher_dir: Path, n_pairs: int, H: int, W: int) -> np.memmap:
    """Memmap the (P, 5, H, W) fp16 teacher logits, returned reshaped to (P, HW, 5) on access."""
    meta = json.loads((teacher_dir / "teacher_logits_meta.json").read_text())
    n_built = int(meta["num_pairs_built"])
    n_cls = int(meta["n_classes"])
    th, tw = meta["seg_input_hw"]
    if (th, tw) != (H, W):
        raise ValueError(f"teacher hw {(th, tw)} != targets hw {(H, W)}")
    n = min(n_pairs, n_built)
    return np.memmap(teacher_dir / "gt_segnet_logits.f16", dtype=np.float16, mode="r",
                     shape=(n_built, n_cls, th, tw))[:n]


# ---------------------------------------------------------------------------
# Targets + feature cache
# ---------------------------------------------------------------------------
def _build_coords_np(h: int, w: int) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def load_targets(targets_dir: Path, n_pairs: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    meta = json.loads((targets_dir / "targets_meta.json").read_text())
    n_built = int(meta["num_pairs_built"])
    H, W = meta["seg_input_hw"]
    n = min(n_pairs, n_built)
    argmax = np.memmap(targets_dir / "gt_segnet_argmax.u8", dtype=np.uint8, mode="r", shape=(n_built, H, W))
    margin = np.memmap(targets_dir / "gt_segnet_margin.f16", dtype=np.float16, mode="r", shape=(n_built, H, W))
    return np.asarray(argmax[:n]).astype(np.int32), np.asarray(margin[:n]).astype(np.float32), H, W


def precompute_features(
    coords: np.ndarray,  # (HW,2)
    argmax: np.ndarray,  # (P,H,W)
    H: int,
    W: int,
    fourier_B: np.ndarray,
    use_prox: bool,
    use_dir: bool,
    n_dir_freqs: int,
    freq_across: float,
    freq_along: float,
    prox_tau: float,
    lane_class: int,
    all_class: bool = False,
) -> np.ndarray:
    """Build the (P, HW, in_feat) feature tensor (isotropic Fourier + optional prox + dir)."""
    P = argmax.shape[0]
    with np.errstate(over="ignore", invalid="ignore"):
        proj = coords @ fourier_B
    iso = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)  # (HW, 2nf)
    feats_per_pair = []
    for pi in range(P):
        parts = [iso]
        if use_prox or use_dir:
            prox, tang = boundary_proximity_and_tangent(
                argmax[pi], lane_class, prox_tau, all_class=all_class
            )
            if use_prox:
                parts.append(prox.reshape(-1, 1))
            if use_dir:
                dirf = directional_fourier_feats(
                    coords, tang.reshape(-1, 2), n_dir_freqs, freq_across, freq_along
                )
                parts.append(dirf)
        feats_per_pair.append(np.concatenate(parts, axis=-1).astype(np.float32))
    return np.stack(feats_per_pair, axis=0)  # (P, HW, in_feat)


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------
def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    argmax, margin, H, W = load_targets(Path(args.targets_dir), args.num_pairs)
    P = argmax.shape[0]
    n_px = H * W
    coords = _build_coords_np(H, W)
    fourier_B = deterministic_fourier_B(args.n_fourier, args.fourier_sigma)

    t_feat = time.time()
    feats = precompute_features(
        coords, argmax, H, W, fourier_B,
        use_prox=args.use_prox, use_dir=args.use_dir, n_dir_freqs=args.n_dir_freqs,
        freq_across=args.freq_across, freq_along=args.freq_along,
        prox_tau=args.prox_tau, lane_class=args.lane_class, all_class=args.all_class,
    )
    feat_secs = time.time() - t_feat
    in_feat = feats.shape[-1]
    feats_mx = mx.array(feats)  # (P, HW, in_feat)
    labels_flat = argmax.reshape(P, n_px).astype(np.int32)
    margin_flat = margin.reshape(P, n_px).astype(np.float32)

    # --- KD teacher soft-logits (optional; kd_weight=0.0 = exact backward-compat) ---
    teacher_mm = None  # (P, 5, H, W) fp16 memmap; reshaped per-pair to (HW, 5) on sample
    kd_active = float(args.kd_weight) > 0.0
    if kd_active:
        if not args.teacher_logits_dir:
            raise ValueError("--kd-weight > 0 requires --teacher-logits-dir")
        teacher_mm = load_teacher_logits(Path(args.teacher_logits_dir), args.num_pairs, H, W)
        if teacher_mm.shape[0] < P:
            raise ValueError(f"teacher has {teacher_mm.shape[0]} pairs < {P} requested")

    model = ImprovedSegGenerator(
        num_pairs=P, n_fourier=args.n_fourier, hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden, mod_dim=args.mod_dim, fourier_sigma=args.fourier_sigma,
        use_prox=args.use_prox, use_dir=args.use_dir, n_dir_freqs=args.n_dir_freqs,
        activation=args.activation,
    )
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    tau = 1.0
    kd_weight = float(args.kd_weight)
    kd_temp = float(args.kd_temp)
    kd_ce_blend = float(args.kd_ce_blend)  # weight on the hard-argmax weighted-CE alongside KD

    bnd_aff_w = float(args.bnd_affinity_weight)
    bnd_aff_active = bnd_aff_w > 0.0
    if bnd_aff_active and not kd_active:
        raise ValueError("--bnd-affinity-weight > 0 requires --kd-weight > 0 (needs teacher logits)")

    def loss_fn(model, pair_idx, fbatch, labels, wpx, teacher):
        """Composable loss: kd_ce_blend * margin-weighted-CE + kd_weight * margin-weighted-KD.

        When kd_weight == 0 the KD branch is skipped entirely (teacher is None) and the loss is
        exactly the original (ce * wpx).mean() — bit-for-bit backward-compatible.
        """
        logits = model(fbatch, pair_idx)
        ce = nn.losses.cross_entropy(logits, labels, reduction="none")
        ce_term = (ce * wpx).mean()
        if teacher is None:
            return ce_term
        kd = kd_kl_logits(logits, teacher, kd_temp)  # (P,) per-pixel KL
        kd_term = (kd * wpx).mean()
        return kd_ce_blend * ce_term + kd_weight * kd_term

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def loss_fn_aff(model, pair_idx, fb_a, fb_b, lab_a, wpx_a, teach_a, teach_b):
        """Per-pixel CE/KD on A + pair-wise boundary-affinity KD on (A, B-neighbor).

        Used ONLY when --bnd-affinity-weight > 0. The per-pixel branch is the SAME margin-weighted
        CE+KD as loss_fn (so the affinity term is purely ADDITIVE on top of the optimal-form c1
        recipe), plus bnd_aff_w * mean( weight_pair * boundary_affinity_kd(A,B) ).
        """
        logits_a = model(fb_a, pair_idx)
        ce = nn.losses.cross_entropy(logits_a, lab_a, reduction="none")
        ce_term = (ce * wpx_a).mean()
        kd = kd_kl_logits(logits_a, teach_a, kd_temp)
        kd_term = (kd * wpx_a).mean()
        per_pixel = kd_ce_blend * ce_term + kd_weight * kd_term
        logits_b = model(fb_b, pair_idx)
        aff = boundary_affinity_kd(logits_a, logits_b, teach_a, teach_b, kd_temp)  # (M,)
        # weight the affinity pairs by the SAME shallow-margin hinge (A's weight) — the edge band
        # is exactly where the residual lives, so concentrate the relational signal there too.
        aff_term = (aff * wpx_a).mean()
        return per_pixel + bnd_aff_w * aff_term

    loss_and_grad_aff = nn.value_and_grad(model, loss_fn_aff)

    def eval_d_seg() -> float:
        disagree = 0
        total = 0
        for pi in range(P):
            logits = model(feats_mx[pi], pi)
            pred = np.array(mx.argmax(logits, axis=-1)).astype(np.int32)
            gt = labels_flat[pi]
            disagree += int((pred != gt).sum())
            total += pred.size
        return disagree / total

    history: list[dict[str, Any]] = []
    best_d_seg = 1.0
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        ep_loss = 0.0
        order = rng.permutation(P)
        for pi_np in order:
            pi = int(pi_np)
            idx = rng.integers(0, n_px, size=args.px_per_step)
            fb = feats_mx[pi][mx.array(idx)]
            lab_np = labels_flat[pi][idx]
            lab = mx.array(lab_np)
            mg = margin_flat[pi][idx]
            wnp = 1.0 + args.hinge_weight * np.exp(-np.maximum(mg, 0.0) / tau)
            if args.error_boost > 0.0:
                # PR74/PR62 hard-pixel error boost: weight the pixels the CURRENT model gets
                # wrong (a no-grad forward; argmax target only, 0 archive bytes). Composes
                # multiplicatively with the static margin-weight above.
                cur_pred = np.array(mx.argmax(model(fb, int(pi)), axis=-1)).astype(np.int32)
                wnp = wnp * hard_pixel_boost(cur_pred, lab_np, args.error_boost)
            wpx = mx.array(wnp.astype(np.float32))
            teacher_b = None
            tlog = None
            if teacher_mm is not None:
                # sample the SAME pixel indices from the (5,H,W) teacher -> (px_per_step, 5)
                tlog = np.asarray(teacher_mm[pi]).reshape(5, n_px).astype(np.float32)
                teacher_b = mx.array(tlog[:, idx].T)  # (px_per_step, 5)
            if bnd_aff_active:
                # B = the right-neighbor of A within the SAME row (clamp last col to left-neighbor);
                # this makes (A, B) a horizontally-adjacent pixel pair across which an edge may sit.
                col = idx % W
                bidx = np.where(col < W - 1, idx + 1, idx - 1)
                fb_b = feats_mx[pi][mx.array(bidx)]
                teach_b_b = mx.array(tlog[:, bidx].T)  # (px_per_step, 5)
                loss, grads = loss_and_grad_aff(
                    model, int(pi), fb, fb_b, lab, wpx, teacher_b, teach_b_b
                )
            else:
                loss, grads = loss_and_grad(model, int(pi), fb, lab, wpx, teacher_b)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            ep_loss += float(loss)
        if ep % args.eval_every == 0 or ep == 1 or ep == args.epochs:
            d_seg = eval_d_seg()
            best_d_seg = min(best_d_seg, d_seg)
            row = {"epoch": ep, "train_loss": round(ep_loss / P, 5),
                   "d_seg": round(d_seg, 6), "best": round(best_d_seg, 6),
                   "wall_s": round(time.time() - t0, 1)}
            history.append(row)
            print(json.dumps(row), flush=True)

    blob = _quantize_blob_bytes(model)
    result = {
        "subagent": "witness_capstone_deepmath_smoke_20260625",
        "utc": _utc(),
        "evidence_grade": "[macOS-MLX research-signal]",
        "promotion_eligible": False, "score_claim": False, "ready_for_exact_eval_dispatch": False,
        "config": {
            "num_pairs": P, "epochs": args.epochs, "hidden_dim": args.hidden_dim,
            "n_hidden": args.n_hidden, "mod_dim": args.mod_dim, "n_fourier": args.n_fourier,
            "fourier_sigma": args.fourier_sigma, "use_prox": args.use_prox, "use_dir": args.use_dir,
            "n_dir_freqs": args.n_dir_freqs, "freq_across": args.freq_across,
            "freq_along": args.freq_along, "prox_tau": args.prox_tau, "activation": args.activation,
            "all_class": args.all_class,
            "lr": args.lr, "weight_decay": args.weight_decay, "hinge_weight": args.hinge_weight,
            "error_boost": args.error_boost,
            "kd_weight": kd_weight, "kd_temp": kd_temp, "kd_ce_blend": kd_ce_blend,
            "kd_active": kd_active,
            "bnd_affinity_weight": bnd_aff_w, "bnd_affinity_active": bnd_aff_active,
            "teacher_logits_dir": str(args.teacher_logits_dir) if args.teacher_logits_dir else None,
            "px_per_step": args.px_per_step, "seed": args.seed, "in_feat": in_feat,
        },
        "kd_borrowed_substrate": (
            "KD/kl_on_logits/T from Quantizr-PR62 + Hinton-Vinyals-Dean 2015; "
            "OURS=frozen-contest-SegNet self-distillation into a non-RGB coord-INR argmax witness"
        ) if kd_active else None,
        "px_per_map": n_px, "px_pairs_total": n_px * P, "device": "mlx (Apple GPU)",
        "feat_precompute_secs": round(feat_secs, 1),
        "history": history, "final_d_seg": history[-1]["d_seg"], "best_d_seg": best_d_seg,
        "quantized_blob": blob,
        "baseline_n600_d_seg": 0.008257, "baseline_n600_blob_bytes": 63802,
    }
    (out_dir / "smoke_result.json").write_text(json.dumps(result, indent=2))
    return result


def _quantize_blob_bytes(model: ImprovedSegGenerator) -> dict[str, Any]:
    import brotli
    from mlx.utils import tree_flatten

    flat = tree_flatten(model.parameters())
    base_chunks: list[bytes] = []
    mod_chunk = b""
    n_params = 0
    for name, arr in flat:
        a = np.array(arr).astype(np.float32)
        n_params += a.size
        s = float(np.abs(a).max()) + 1e-8
        q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
        if name == "mod":
            mod_chunk = q.tobytes()
        else:
            base_chunks.append(q.tobytes())
    base_raw = b"".join(base_chunks)
    base_brotli = len(brotli.compress(base_raw, quality=11))
    mod_brotli = len(brotli.compress(mod_chunk, quality=11)) if mod_chunk else 0
    return {
        "n_params": int(n_params), "base_int8_raw_bytes": len(base_raw),
        "base_int8_brotli_bytes": base_brotli, "mod_int8_raw_bytes": len(mod_chunk),
        "mod_int8_brotli_bytes": mod_brotli, "total_quantized_blob_bytes": base_brotli + mod_brotli,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WITNESS CAPSTONE deep-math descent smoke (MLX)")
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-pairs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    # capacity
    ap.add_argument("--n-fourier", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--n-hidden", type=int, default=4)
    ap.add_argument("--mod-dim", type=int, default=32)
    ap.add_argument("--fourier-sigma", type=float, default=8.0)
    # deep-math levers
    ap.add_argument("--use-prox", action="store_true")
    ap.add_argument("--use-dir", action="store_true")
    ap.add_argument("--n-dir-freqs", type=int, default=6)
    ap.add_argument("--freq-across", type=float, default=32.0)
    ap.add_argument("--freq-along", type=float, default=4.0)
    ap.add_argument("--prox-tau", type=float, default=4.0)
    ap.add_argument("--activation", choices=["relu", "gelu", "gauss"], default="relu")
    ap.add_argument("--lane-class", type=int, default=1)
    ap.add_argument("--all-class", action="store_true",
                    help="route prox/dir to the UNION of all class boundaries (the real flip locus)")
    # opt
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hinge-weight", type=float, default=4.0)
    ap.add_argument("--error-boost", type=float, default=0.0,
                    help="PR74/PR62 hard-pixel error boost: extra weight on px the current model "
                         "misclassifies (0.0=off; PR62 used 9, PR74 used 49). 0-byte, train-time.")
    # KD (soft-logit knowledge distillation; Quantizr kl_on_logits / Hinton-2015)
    ap.add_argument("--kd-weight", type=float, default=0.0,
                    help="weight on the T^2*KL(teacher||student) soft-logit KD term (0.0=off=exact "
                         "backward-compat). Requires --teacher-logits-dir. 0 archive bytes (train-time).")
    ap.add_argument("--kd-temp", type=float, default=2.0,
                    help="KD softmax temperature T (Quantizr/Hinton default 2.0).")
    ap.add_argument("--kd-ce-blend", type=float, default=0.0,
                    help="weight on the hard-argmax margin-weighted-CE alongside KD (0.0=KD-only "
                         "when kd-weight>0; e.g. 1.0 for an even KD+CE blend).")
    ap.add_argument("--teacher-logits-dir", type=Path, default=None,
                    help="dir with gt_segnet_logits.f16 + teacher_logits_meta.json "
                         "(from tools/build_segnet_teacher_logits.py).")
    ap.add_argument("--bnd-affinity-weight", type=float, default=0.0,
                    help="weight on the pair-wise BOUNDARY-AFFINITY KD term (Liu CVPR-2019 "
                         "structured-KD, adapted): match student adjacent-pixel agreement to the "
                         "teacher's soft edge field. 0.0=off=exact backward-compat. Requires "
                         "--kd-weight>0 (uses teacher logits). 0 archive bytes (train-time). The "
                         "STRUCTURALLY-NEW lever targeting the edge-PLACEMENT residual that "
                         "per-pixel CE/KD saturate.")
    ap.add_argument("--px-per-step", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # WITNESS_DSEG_FEASIBILITY_ONLY: loud fail-closed tag — this harness's d_seg is a
    # generator-argmax proxy (no R, no SegNet re-seg), NOT a realized contest score.
    from tac.measurement_integrity import warn_feasibility_only_dseg

    feasibility_tag = warn_feasibility_only_dseg("witness_capstone_deepmath_smoke")

    result = run_smoke(args)
    if isinstance(result, dict):
        result["d_seg_axis_tag"] = feasibility_tag
        result["score_claim"] = False
        result["promotion_eligible"] = False
    print("\n=== SMOKE RESULT (d_seg %s) ===" % feasibility_tag)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
