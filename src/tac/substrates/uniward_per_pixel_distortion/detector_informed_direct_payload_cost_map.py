# SPDX-License-Identifier: MIT
"""Detector-informed cost-map for the UWD1 sparse-delta DIRECT-PAYLOAD surface.

OPTIMAL-FORM UNIWARD (best-chance re-test per operator directive 2026-05-31):
the prior UNIWARD negatives on NSCS06 v8 were tested at a CARGO-CULTED form
(Catalog #307/#315/#303):

  CARGO-CULT 1 — SegNet-FREE cost-map. Prior wires used a luma-class proxy with
    only 20/80 bins nonempty. UNIWARD/Fridrich is DETECTOR-informed by definition:
    spend distortion where the DETECTOR (SegNet) is blind, protect where it is
    sensitive. A SegNet-free cost-map discards the entire premise.
  CARGO-CULT 2 — wrong surface class. The grayscale finalize (commit 5951a3a02)
    proved a spatial luma RASTER compresses via GLOBAL bit-depth, not spatial
    concentration, so UNIWARD's spatial lever cannot help a globally-entropy-coded
    raster. UNIWARD belongs on a DIRECT-PAYLOAD surface where per-element precision
    IS the rate cost.

This module fixes BOTH. The DIRECT-PAYLOAD surface is the canonical UWD1
sparse-delta sidechannel (``tac.uniward_delta.pack_sparse_delta`` →
``unpack_sparse_delta`` → ``apply_delta_to_frame``; byte-closed, PR98-L28 class):
``pack_sparse_delta`` keeps the top-K pixel-channels of a δ residual ranked by
``rank_score = |δ| * (1 + cost_norm)``, int8-quantizes them, and zlib-packs. The
``cost_map_bhw`` parameter is the EXACT injection point for a detector-informed
ranking — NO surface rebuild is required.

The DETECTOR-informed cost-map composes two REAL signals on REAL rendered frames:

  texture_cost = compute_uniward_cost_map(frames)          # high in textured regions
  boundary_w   = segnet_boundary_band_weights(seg_logits)  # w_i = exp(-margin/τ) ≈ 1
                                                           #   on the decision boundary
The contest seg distortion is the per-pixel argmax-flip RATE
(``upstream.modules.SegNet.compute_distortion``), so a δ entry only MOVES d_seg at a
pixel whose top-2 SegNet-logit margin is small (the boundary band). We provide two
canonical compositions, selected by the sidecar's role:

  ROLE = ``correction`` (δ pushes a degraded render back toward GT; bytes should buy
    score-relevant flips): rank score-relevant boundary-band entries FIRST.
      cost = texture_cost * (eps + boundary_w)
  ROLE = ``attack`` (δ injects detector-invisible distortion to free rate elsewhere;
    bytes should avoid score-relevant flips): rank textured NON-boundary interiors.
      cost = texture_cost * (eps + (1 - boundary_w))

Both are HARD-EARNED Fridrich inverse-steganalysis: ``correction`` spends precision
where the scorer is blind-to-the-FIX-being-needed (small-margin boundary) and
``attack`` spends where the scorer is blind-to-perturbation (textured interior).

NON-PROMOTABLE: this is an MLX/CPU research-signal surface per CLAUDE.md "MPS auth
eval is NOISE" + Catalog #192/#341/#127/#323. The cost-map only RANKS which δ
entries survive the byte budget; it never fabricates a contest score. All persisted
rows carry canonical Provenance with ``score_claim=False`` /
``axis_tag="[macOS-CPU advisory]"``.

[verified-against: upstream/modules.py::SegNet.compute_distortion (argmax-flip rate);
 tac.uniward_delta.pack_sparse_delta (cost_map_bhw ranking injection point);
 tac.multi_granularity_sensitivity.segnet_boundary_band_weights (w=exp(-margin/τ));
 tac.uniward_delta.compute_uniward_cost_map (S-UNIWARD texture)]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "SIDECAR_ROLE_ATTACK",
    "SIDECAR_ROLE_CORRECTION",
    "VALID_SIDECAR_ROLES",
    "DetectorCostMapError",
    "DetectorInformedCostMap",
    "allocation_diff_proof",
    "compose_detector_informed_cost_map",
    "update_from_anchor",
]

SIDECAR_ROLE_CORRECTION = "correction"
SIDECAR_ROLE_ATTACK = "attack"
VALID_SIDECAR_ROLES = frozenset({SIDECAR_ROLE_CORRECTION, SIDECAR_ROLE_ATTACK})

# Additive floor so a zero boundary-weight (or zero non-boundary weight) does not
# annihilate the texture cost — keeps the ranking a strict reweight, not a mask.
_BOUNDARY_FLOOR_EPS = 1e-3
# Catalog #341 canonical non-promotable markers (this surface RANKS; never scores).
_CANONICAL_AXIS_TAG = "[macOS-CPU advisory]"


class DetectorCostMapError(ValueError):
    """Raised on malformed inputs to the detector-informed cost-map composition."""


@dataclass(frozen=True)
class DetectorInformedCostMap:
    """A detector-informed UWD1 ranking cost-map + its observability stats.

    ``cost_bhw`` is fed DIRECTLY to ``tac.uniward_delta.pack_sparse_delta`` as the
    ``cost_map_bhw`` argument. Higher cost ⇒ rank-boosted ⇒ the entry survives a
    tighter byte budget.
    """

    cost_bhw: np.ndarray  # (B, H, W) float32, the ranking signal for pack_sparse_delta
    role: str
    boundary_band_fraction: float  # detector observability: fraction in boundary band
    boundary_mean_weight: float
    texture_mean: float
    cost_gini: float  # concentration of the composed cost (how peaked the ranking is)
    tau: float
    n_pixels: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "boundary_band_fraction": float(self.boundary_band_fraction),
            "boundary_mean_weight": float(self.boundary_mean_weight),
            "texture_mean": float(self.texture_mean),
            "cost_gini": float(self.cost_gini),
            "tau": float(self.tau),
            "n_pixels": int(self.n_pixels),
            "axis_tag": _CANONICAL_AXIS_TAG,
            "score_claim": False,
            "promotable": False,
            "schema": "detector_informed_direct_payload_cost_map_v1",
        }


def _gini(x: np.ndarray) -> float:
    """Gini concentration of a non-negative vector (0 = flat, →1 = peaked)."""
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    a = np.clip(a, 0.0, None)
    s = a.sum()
    if s <= 0.0:
        return 0.0
    a_sorted = np.sort(a)
    n = a_sorted.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((np.sum((2.0 * idx - n - 1.0) * a_sorted)) / (n * s))


def _to_bhw_f32(arr: Any, name: str) -> np.ndarray:
    """Coerce a torch/numpy spatial map to a contiguous (B, H, W) float32 array."""
    try:  # torch tensors expose .detach().cpu().numpy()
        import torch

        if torch.is_tensor(arr):
            arr = arr.detach().to("cpu", dtype=torch.float32).numpy()
    except ImportError:  # pragma: no cover - torch always present in this repo
        pass
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, :, :]
    if a.ndim != 3:
        raise DetectorCostMapError(
            f"{name} must be (H,W) or (B,H,W); got ndim={a.ndim} shape={a.shape}"
        )
    return np.ascontiguousarray(a)


def compose_detector_informed_cost_map(
    texture_cost_bhw: Any,
    boundary_weight_bhw: Any,
    *,
    role: str = SIDECAR_ROLE_CORRECTION,
    tau: float = 1.0,
) -> DetectorInformedCostMap:
    """Compose the detector-informed ranking cost-map for the UWD1 surface.

    Parameters
    ----------
    texture_cost_bhw:
        S-UNIWARD texture cost from ``tac.uniward_delta.compute_uniward_cost_map``
        on the REAL rendered frames (high = textured = safe to perturb). torch or
        numpy ``(B, H, W)`` or ``(H, W)``.
    boundary_weight_bhw:
        SegNet boundary-band weight ``w_i = exp(-margin_i/τ)`` from
        ``tac.multi_granularity_sensitivity.segnet_boundary_band_weights`` on the
        REAL SegNet logits of the SAME frames (≈1 on the decision boundary where
        d_seg is sensitive, →0 in confident interiors). torch or numpy, same shape.
    role:
        ``"correction"`` (boost boundary band — bytes buy score-relevant flips) or
        ``"attack"`` (boost textured non-boundary interiors — detector-invisible).
    tau:
        Recorded for provenance; the actual τ is baked into ``boundary_weight_bhw``
        upstream.

    Returns
    -------
    DetectorInformedCostMap
        ``.cost_bhw`` feeds ``pack_sparse_delta(cost_map_bhw=...)`` directly.

    Raises
    ------
    DetectorCostMapError
        On shape mismatch, negative texture cost, or out-of-[0,1] boundary weight.
    """
    if role not in VALID_SIDECAR_ROLES:
        raise DetectorCostMapError(
            f"role must be one of {sorted(VALID_SIDECAR_ROLES)}; got {role!r}"
        )
    texture = _to_bhw_f32(texture_cost_bhw, "texture_cost_bhw")
    boundary = _to_bhw_f32(boundary_weight_bhw, "boundary_weight_bhw")
    if texture.shape != boundary.shape:
        raise DetectorCostMapError(
            f"shape mismatch: texture={texture.shape} vs boundary={boundary.shape}"
        )
    if np.any(texture < 0.0):
        raise DetectorCostMapError("texture_cost must be non-negative (S-UNIWARD)")
    if np.any(~np.isfinite(texture)) or np.any(~np.isfinite(boundary)):
        raise DetectorCostMapError("inputs must be finite")
    # boundary weight is exp(-margin/tau) ∈ (0, 1]; tolerate tiny fp slack.
    if np.any(boundary < -1e-4) or np.any(boundary > 1.0 + 1e-4):
        raise DetectorCostMapError(
            "boundary_weight must be in [0, 1] (it is exp(-margin/τ))"
        )
    boundary = np.clip(boundary, 0.0, 1.0)

    # Per-frame texture normalization so one bright frame does not dominate the
    # global ranking (mirrors pack_sparse_delta's own cost_norm convention).
    tex_max = np.maximum(texture.reshape(texture.shape[0], -1).max(axis=1), 1e-8)
    tex_norm = texture / tex_max[:, None, None]  # (B, H, W) in [0, 1]

    if role == SIDECAR_ROLE_CORRECTION:
        detector_term = _BOUNDARY_FLOOR_EPS + boundary
    else:  # SIDECAR_ROLE_ATTACK
        detector_term = _BOUNDARY_FLOOR_EPS + (1.0 - boundary)

    cost = (tex_norm * detector_term).astype(np.float32)

    stats = DetectorInformedCostMap(
        cost_bhw=cost,
        role=role,
        boundary_band_fraction=float((boundary > np.exp(-1.0)).mean()),
        boundary_mean_weight=float(boundary.mean()),
        texture_mean=float(texture.mean()),
        cost_gini=_gini(cost),
        tau=float(tau),
        n_pixels=int(cost.size),
    )
    return stats


def allocation_diff_proof(
    cost_a_bhw: Any,
    cost_b_bhw: Any,
    *,
    n_kept: int,
) -> dict[str, Any]:
    """NON-FAKE allocation-diff proof (Catalog #105/#139/#220).

    Proves the cost-map actually CHANGES which δ entries survive a byte budget: it
    simulates ``pack_sparse_delta``'s top-K ranking (``|δ| * (1 + cost_norm)``) with a
    constant |δ|=1 stand-in so the *only* thing distinguishing the two rankings is the
    cost map, then reports the symmetric difference of the kept index sets.

    Returns a dict with ``kept_set_symmetric_difference`` (>0 ⇒ the maps allocate
    differently — the no-op guard) plus the per-map kept-index sets.
    """
    a = _to_bhw_f32(cost_a_bhw, "cost_a_bhw")
    b = _to_bhw_f32(cost_b_bhw, "cost_b_bhw")
    if a.shape != b.shape:
        raise DetectorCostMapError(
            f"shape mismatch: a={a.shape} vs b={b.shape}"
        )
    if n_kept < 0:
        raise DetectorCostMapError(f"n_kept must be non-negative; got {n_kept}")

    def _topk_set(cost: np.ndarray) -> set[int]:
        # Replicate pack_sparse_delta's normalization: |δ|=1 ⇒ rank = 1 + cost_norm.
        cmax = np.maximum(cost.reshape(cost.shape[0], -1).max(axis=1), 1e-8)
        cnorm = cost / cmax[:, None, None]
        rank = (1.0 + cnorm).ravel()
        k = min(n_kept, rank.size)
        if k <= 0:
            return set()
        # argpartition for the top-k; ties broken by lower index (stable) to mirror
        # torch.topk's first-occurrence-on-equal-scores determinism.
        idx = np.argsort(-rank, kind="stable")[:k]
        return {int(i) for i in idx}

    set_a = _topk_set(a)
    set_b = _topk_set(b)
    sym = set_a ^ set_b
    return {
        "kept_set_symmetric_difference": len(sym),
        "n_kept": int(min(n_kept, a.size)),
        "n_kept_a": len(set_a),
        "n_kept_b": len(set_b),
        "allocation_changed": bool(len(sym) > 0),
        "schema": "uniward_detector_allocation_diff_proof_v1",
    }


def update_from_anchor(anchor: Mapping[str, Any]) -> dict[str, Any]:
    """Continual-learning hook 5 (Catalog #125 / #335 contract).

    Observability-only: records that a detector-informed-cost-map empirical anchor
    landed. This surface ranks δ entries; it never mutates a score, so the anchor is
    echoed back with canonical non-promotable markers (Catalog #341). No fabricated
    numbers — only the anchor's own fields are reflected.
    """
    role = str(anchor.get("role", "")) if isinstance(anchor, Mapping) else ""
    return {
        "consumed": True,
        "role": role,
        "axis_tag": _CANONICAL_AXIS_TAG,
        "score_claim": False,
        "promotable": False,
        "predicted_delta_adjustment": 0.0,
        "schema": "detector_informed_direct_payload_cost_map_update_v1",
    }
