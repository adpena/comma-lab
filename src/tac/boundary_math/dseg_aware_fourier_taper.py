# SPDX-License-Identifier: MIT
"""d_seg-aware Fourier-feature amplitude taper (#121 lever, re-validate-at-convergence).

WHAT IT IS (mechanically). The level-set / coord-INR witness feeds a coordinate grid through a
FIXED Fourier / curvelet-directional basis to get per-pixel features ``feats[px, k]`` (k indexes
the Fourier/curvelet columns). The witness's ``in_proj`` linear layer consumes those features.
Every column currently enters with UNIT amplitude — a FLAT taper. A *taper* is a per-column
amplitude envelope ``w[k]`` applied as ``feats[px, k] *= w[k]``: it reallocates the INR's spectral
prior across the Fourier columns WITHOUT adding a single trainable parameter (byte-neutral) and
WITHOUT changing the feature shape.

A d_seg-AWARE taper derives ``w[k]`` from the **d_seg saliency field** (= the GT top1-top2 SegNet
argmax MARGIN field: small |margin| ⇒ near the codim-1 boundary annulus ⇒ where d_seg is actually
decided). Columns whose spatial energy concentrates in the high-saliency boundary annulus are
UP-weighted; columns whose energy sits in the flat argmax-stable interior are DOWN-weighted. This
is the spectral analogue of the vendored ``configurable_taper_decoder.dseg_aware_taper`` capacity
reallocation (move representational precision to where d_seg lives), realised on the witness's own
Fourier basis instead of the HNeRV channel schedule.

WHY it is byte-neutral AND rule-118 FREE. The taper adds ZERO trainable params, so the archive is
unchanged. ``w`` is a deterministic function of (the fixed basis geometry + the GT margin field),
recomputable at decode exactly like the self-orientation directional basis and the openpilot lane
band — a geometry PRIOR, not learned/video-derived weights.

THE TWO DEGENERACIES (both provable, both tested — the NO-FAKE contract). ``strength == 0`` ⇒
``w == 1`` exactly. A UNIFORM saliency field ⇒ ``w == 1`` exactly. In either case the tapered
features are BIT-IDENTICAL to the flat features → the trainer's default-OFF path (which simply does
not call this) is byte-identical, and the ``strength=0`` / uniform-S calls degrade to the flat
taper as a runtime guarantee, not a promise.

VERDICT STATUS (do NOT strip this caveat). #121 lever ledger row: "+18% NO-GO RETRACTED
(under-converged ge300/3000); converged anchors flip sign to -8% ~0.03; RE-VALIDATE at
convergence". So this is a RE-VALIDATE-AT-CONVERGENCE lever — its d_seg effect is
ASSUMED_AWAITING_VERIFICATION until a converged (byte-close) A/B measures it. This module BUILDS the
mechanism (default-OFF, byte-identical); it makes NO score claim. means != ends: pointer UNMOVED.

Pure numpy (no MLX / torch / GPU) so the mechanism is unit-testable at $0. DSL leg:
``tac.witness_dsl.curriculum_dsl.DsegAwareTaper``. Equations leg:
``tac.canonical_equations.dseg_aware_fourier_taper_20260709``.
"""
from __future__ import annotations

import numpy as np


def _nn_resize_float(a: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize a 2-D FLOAT field to (out_h, out_w) — the float twin of the trainer's
    ``_nn_resize_labels`` (same ``np.ix_`` index scheme). NN (no interpolation) so a resized margin
    keeps genuine GT values; used to bring a SEG-grid margin onto the RENDER grid so the per-pixel
    saliency matches the coord/feature grid for ANY render/seg ratio (the same mismatch the
    directional basis fixes). Fail-closed on a non-2-D input."""
    a = np.asarray(a, np.float64)
    if a.ndim != 2:
        raise ValueError(f"_nn_resize_float: expected a 2-D (H,W) field, got shape {a.shape}")
    if a.shape == (out_h, out_w):
        return a
    ys = np.linspace(0, a.shape[0] - 1, out_h).round().astype(np.int64)
    xs = np.linspace(0, a.shape[1] - 1, out_w).round().astype(np.int64)
    return a[np.ix_(ys, xs)]


def saliency_from_margins(
    margins,
    scale: float | None = None,
    *,
    target_hw: tuple[int, int] | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-pixel d_seg saliency (P_px,) from GT top1-top2 argmax MARGIN fields (high near boundary).

    ``margins``: a per-pair GT margin field ``(H, W)`` OR a sequence of them (one per pair). Small
    ``|margin|`` = near the codim-1 argmax boundary annulus = where d_seg is decided → HIGH saliency.

    ``target_hw`` (the RENDER grid ``(H, W)``): when given, each margin is NN-resized to it BEFORE
    flattening, so the saliency P_px matches the coord/feature grid ``render_h*render_w`` even when the
    GT margin lives on a DIFFERENT (SEG) grid — the exact render≠seg mismatch the directional basis
    resolves with ``_nn_resize_labels``. With ``target_hw`` set, each margin MUST be 2-D ``(H, W)``.
    ``None`` (the pure/test path) keeps the raw per-margin flatten.

    ``S_pair(px) = exp(-|margin_pair(px)| / scale)`` ∈ (0, 1]; ``S(px) = mean over pairs``; the result
    is normalised to unit mean (``mean_px S == 1``) so a UNIFORM margin field yields a UNIFORM
    saliency (→ the flat taper via :func:`compute_dseg_aware_fourier_taper`).

    ``scale`` (the exp kernel width, in margin units): ``None`` ⇒ AUTO = the median of ``|margin|``
    over all pixels/pairs (a robust, deterministic, GT-derived width), with a mean fallback and a
    final 1.0 fallback so ``scale`` is always strictly positive (fail-closed, never a /0).
    """
    if margins is None:
        raise ValueError("saliency_from_margins: margins is None")
    # Normalise to a list of per-pair 2-D (or flattenable) margin fields.
    if isinstance(margins, np.ndarray) and margins.ndim <= 2 and not isinstance(margins, (list, tuple)):
        items = [margins]
    else:
        items = list(margins)
    if not items:
        raise ValueError("saliency_from_margins: empty margins sequence")
    flats = []
    for m in items:
        m = np.asarray(m, np.float64)
        if target_hw is not None:
            m = _nn_resize_float(m, int(target_hw[0]), int(target_hw[1]))
        flats.append(np.abs(m).reshape(-1))
    p_px = flats[0].shape[0]
    for i, f in enumerate(flats):
        if f.shape[0] != p_px:
            raise ValueError(
                f"saliency_from_margins: margin[{i}] has {f.shape[0]} px, expected {p_px} "
                "(all pairs must share the pixel grid)")
    abs_mar = np.stack(flats, axis=0)  # (n_pair, P_px)

    if scale is None:
        med = float(np.median(abs_mar))
        if med > eps:
            width = med
        else:
            mean = float(np.mean(abs_mar))
            width = mean if mean > eps else 1.0
    else:
        width = float(scale)
        if width <= eps:
            raise ValueError(f"saliency_from_margins: scale must be > 0, got {scale!r}")

    s_pair = np.exp(-abs_mar / width)          # (n_pair, P_px) in (0,1]
    s = s_pair.mean(axis=0)                      # (P_px,)
    mean_s = float(s.mean())
    if mean_s > eps:
        s = s / mean_s                          # unit-mean → uniform margins ⇒ uniform saliency
    return s.astype(np.float32)


def compute_dseg_aware_fourier_taper(
    feats: np.ndarray,
    saliency: np.ndarray,
    strength: float = 1.0,
    *,
    floor: float = 0.05,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-Fourier-column amplitude taper ``w`` (F,) reweighting the taper by d_seg saliency.

    ``feats``: ``(P_px, F)`` Fourier/curvelet features. ``saliency``: ``(P_px,)`` per-pixel d_seg
    saliency (unit-mean, high near the boundary annulus; from :func:`saliency_from_margins`).

    Per column k:
      ``e_k    = mean_px feats[:,k]^2``                     (overall column energy)
      ``sw_k   = mean_px saliency(px) * feats[:,k]^2``      (boundary-annulus-weighted energy)
      ``r_k    = sw_k / (e_k + eps)``                       (>1 ⇒ column concentrates near boundary)
      ``w_k    = 1 + strength * (r_k / mean_k r_k - 1)``    (reallocate toward boundary-aligned cols)
      ``w_k    = max(w_k, floor)``                          (positivity guard)

    DEGENERACIES (the byte-identity contract): ``strength == 0`` ⇒ ``w == 1`` exactly; a UNIFORM
    ``saliency`` ⇒ ``r_k`` constant ⇒ ``w == 1`` exactly. By construction ``mean_k w_k == 1`` before
    the floor clamp (the reweighting is a REALLOCATION, not a net amplitude change), so the mechanism
    is byte-neutral (no new archived params) AND amplitude-preserving in the mean.
    """
    feats = np.asarray(feats, np.float64)
    saliency = np.asarray(saliency, np.float64)
    if feats.ndim != 2:
        raise ValueError(f"feats must be 2-D (P_px, F); got shape {feats.shape}")
    if saliency.ndim != 1 or saliency.shape[0] != feats.shape[0]:
        raise ValueError(
            f"saliency must be (P_px,) matching feats P_px={feats.shape[0]}; got {saliency.shape}")
    if floor < 0.0:
        raise ValueError(f"floor must be >= 0, got {floor!r}")
    f = float(strength)
    if f == 0.0:
        return np.ones((feats.shape[1],), np.float32)

    e2 = feats * feats                                  # (P_px, F)
    e_k = e2.mean(axis=0)                                # (F,) overall energy
    sw_k = (saliency[:, None] * e2).mean(axis=0)         # (F,) saliency-weighted energy
    r_k = sw_k / (e_k + eps)                             # (F,)
    mean_r = float(r_k.mean())
    if mean_r <= eps:
        # No usable saliency contrast (e.g. all-zero features) → flat taper (fail-closed to no-op).
        return np.ones((feats.shape[1],), np.float32)
    w = 1.0 + f * (r_k / mean_r - 1.0)                   # mean_k w == 1 by construction
    w = np.maximum(w, float(floor))
    return w.astype(np.float32)


def apply_dseg_aware_fourier_taper(feats: np.ndarray, taper: np.ndarray) -> np.ndarray:
    """Apply a per-column taper: ``feats[:, k] *= taper[k]`` (shape-checked; dtype preserved).

    ``taper`` all-ones ⇒ the returned array is numerically identical to ``feats`` (the flat / off
    path). Fail-closed on a column-count mismatch (NO-FAKE: never silently broadcast wrong)."""
    feats_in = np.asarray(feats)
    taper = np.asarray(taper)
    if taper.ndim != 1 or taper.shape[0] != feats_in.shape[-1]:
        raise ValueError(
            f"taper must be (F,) matching feats F={feats_in.shape[-1]}; got {taper.shape}")
    return (feats_in.astype(np.float64) * taper.astype(np.float64)[None, :]).astype(
        feats_in.dtype if feats_in.dtype.kind == "f" else np.float32)


__all__ = [
    "apply_dseg_aware_fourier_taper",
    "compute_dseg_aware_fourier_taper",
    "saliency_from_margins",
]
