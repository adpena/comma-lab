# SPDX-License-Identifier: MIT
"""FULL-GRID SegNet-response detector cost-map for the UWD1 direct-payload surface.

MAGNITUDE-TEST extension of sister #1585 (commit 59ba009f0,
``detector_informed_direct_payload_cost_map.py``) per the CODEX correction
(memo ``codex_findings_z8_pixel_driver_and_segnet_grid_premise_20260531T153038Z``
Finding 2): *"SegNet interiors are free" is STALE — SegNet responds across the FULL
384x512 argmax grid, not just boundary flips.* The detector weight should be the
FULL MEASURED SegNet response (boundary + class-interior + region), NOT just the
``exp(-margin/τ)`` boundary band (which is ~4.58% of pixels per the sister run).

THE FULL-GRID DETECTOR RESPONSE (real, tractable, $0):
  Per-pixel ``∂ d_seg / ∂ pixel`` is intractable (~7.2B forwards). The canonical
  full-grid surrogate is the INPUT-GRADIENT SALIENCY of the segmentation loss:

      sal_i = |∂ L_seg / ∂ pixel_i|     (summed over the 3 RGB channels)

  where ``L_seg`` is the self-consistent cross-entropy of the SegNet logits against
  their own argmax pseudo-labels. ONE backward pass per frame yields a DENSE
  ``(H, W)`` response that is non-zero across ~77% of the grid (boundary +
  class-interior + region) — exactly the full-grid sensitivity the contest seg
  distortion (per-pixel argmax-flip RATE, ``upstream.modules.SegNet.compute_distortion``)
  actually integrates over. Pixels with high ``sal_i`` are the ones whose argmax is
  most precision-sensitive; protecting their precision (keeping the δ correction
  there) buys the most score-relevant flips back.

  Boundary band vs full grid (the codex correction made concrete):
    - ``segnet_boundary_band_weights`` (sister): w_i = exp(-margin_i/τ) ≈ 1 only on a
      TIGHT decision boundary (~4.58% band fraction). Misses class-interior + region
      sensitivity.
    - THIS module: ``full_grid_segnet_response`` (input-gradient saliency) is dense
      (~77% non-zero), covering boundary + interior + region in one signal.

COMPOSITION onto the UWD1 direct-payload surface (``tac.uniward_delta.pack_sparse_delta``
``cost_map_bhw`` ranking injection point — NO surface rebuild):

  ROLE = ``correction`` (δ pushes a degraded render back toward GT; bytes should buy
    score-relevant flips): rank high-SegNet-response pixels FIRST (protect precision).
      cost = texture_cost * (eps + response_norm)
  ROLE = ``attack`` (δ injects detector-invisible distortion to free rate elsewhere):
    rank low-SegNet-response (textured-but-insensitive) interiors.
      cost = texture_cost * (eps + (1 - response_norm))

NON-PROMOTABLE: MLX/CPU research-signal per CLAUDE.md "MPS auth eval is NOISE" +
Catalog #192/#341/#127/#323. The cost-map only RANKS which δ entries survive the byte
budget; it never fabricates a contest score. Persisted rows carry canonical Provenance
``score_claim=False`` / ``axis_tag="[macOS-CPU advisory]"``.

[verified-against: upstream/modules.py::SegNet.compute_distortion (argmax-flip rate);
 upstream/modules.py::SegNet.preprocess_input + forward (input-gradient backward pass);
 tac.uniward_delta.pack_sparse_delta (cost_map_bhw ranking injection point);
 tac.uniward_delta.compute_uniward_cost_map (S-UNIWARD texture);
 sister tac.substrates.uniward_per_pixel_distortion.detector_informed_direct_payload_cost_map
   (boundary-band sister composition this module replaces with the FULL-GRID response)]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "RESPONSE_ROLE_ATTACK",
    "RESPONSE_ROLE_CORRECTION",
    "VALID_RESPONSE_ROLES",
    "FullGridResponseError",
    "FullGridSegNetResponseCostMap",
    "compose_full_grid_response_cost_map",
    "full_grid_response_allocation_diff_proof",
    "update_from_anchor",
]

RESPONSE_ROLE_CORRECTION = "correction"
RESPONSE_ROLE_ATTACK = "attack"
VALID_RESPONSE_ROLES = frozenset({RESPONSE_ROLE_CORRECTION, RESPONSE_ROLE_ATTACK})

# Additive floor so a zero response (or zero inverse-response) does not annihilate the
# texture cost — keeps the ranking a strict reweight, not a hard mask.
_RESPONSE_FLOOR_EPS = 1e-3
# Catalog #341 canonical non-promotable markers (this surface RANKS; never scores).
_CANONICAL_AXIS_TAG = "[macOS-CPU advisory]"


class FullGridResponseError(ValueError):
    """Raised on malformed inputs to the full-grid response cost-map composition."""


@dataclass(frozen=True)
class FullGridSegNetResponseCostMap:
    """A full-grid-SegNet-response UWD1 ranking cost-map + its observability stats.

    ``cost_bhw`` is fed DIRECTLY to ``tac.uniward_delta.pack_sparse_delta`` as the
    ``cost_map_bhw`` argument. Higher cost ⇒ rank-boosted ⇒ the entry survives a
    tighter byte budget.
    """

    cost_bhw: np.ndarray  # (B, H, W) float32, the ranking signal for pack_sparse_delta
    role: str
    response_nonzero_fraction: float  # full-grid observability: how dense the response
    response_mean: float
    response_gini: float  # concentration of the raw SegNet response (how peaked)
    texture_mean: float
    cost_gini: float  # concentration of the composed cost
    n_pixels: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "response_nonzero_fraction": float(self.response_nonzero_fraction),
            "response_mean": float(self.response_mean),
            "response_gini": float(self.response_gini),
            "texture_mean": float(self.texture_mean),
            "cost_gini": float(self.cost_gini),
            "n_pixels": int(self.n_pixels),
            "axis_tag": _CANONICAL_AXIS_TAG,
            "score_claim": False,
            "promotable": False,
            "schema": "full_grid_segnet_response_cost_map_v1",
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
        raise FullGridResponseError(
            f"{name} must be (H,W) or (B,H,W); got ndim={a.ndim} shape={a.shape}"
        )
    return np.ascontiguousarray(a)


def compose_full_grid_response_cost_map(
    texture_cost_bhw: Any,
    segnet_response_bhw: Any,
    *,
    role: str = RESPONSE_ROLE_CORRECTION,
) -> FullGridSegNetResponseCostMap:
    """Compose the FULL-GRID-SegNet-response ranking cost-map for the UWD1 surface.

    Parameters
    ----------
    texture_cost_bhw:
        S-UNIWARD texture cost from ``tac.uniward_delta.compute_uniward_cost_map`` on
        the REAL rendered (degraded) frames. torch or numpy ``(B, H, W)`` / ``(H, W)``.
    segnet_response_bhw:
        FULL-GRID SegNet input-gradient saliency ``|∂ L_seg / ∂ pixel|`` (summed over
        RGB) measured on the SAME frames via a single backward pass through the REAL
        SegNet. Dense across the grid (boundary + interior + region). Non-negative.
        torch or numpy, same shape as ``texture_cost_bhw``.
    role:
        ``"correction"`` (boost high-response pixels — protect precision where SegNet
        argmax is sensitive) or ``"attack"`` (boost low-response interiors —
        detector-invisible).

    Returns
    -------
    FullGridSegNetResponseCostMap
        ``.cost_bhw`` feeds ``pack_sparse_delta(cost_map_bhw=...)`` directly.

    Raises
    ------
    FullGridResponseError
        On shape mismatch, negative texture cost, negative response, or non-finite.
    """
    if role not in VALID_RESPONSE_ROLES:
        raise FullGridResponseError(
            f"role must be one of {sorted(VALID_RESPONSE_ROLES)}; got {role!r}"
        )
    texture = _to_bhw_f32(texture_cost_bhw, "texture_cost_bhw")
    response = _to_bhw_f32(segnet_response_bhw, "segnet_response_bhw")
    if texture.shape != response.shape:
        raise FullGridResponseError(
            f"shape mismatch: texture={texture.shape} vs response={response.shape}"
        )
    if np.any(texture < 0.0):
        raise FullGridResponseError("texture_cost must be non-negative (S-UNIWARD)")
    if np.any(response < 0.0):
        raise FullGridResponseError(
            "segnet_response must be non-negative (|∂L/∂pixel|)"
        )
    if np.any(~np.isfinite(texture)) or np.any(~np.isfinite(response)):
        raise FullGridResponseError("inputs must be finite")

    # Per-frame texture normalization so one bright frame does not dominate the
    # global ranking (mirrors pack_sparse_delta's own cost_norm convention).
    tex_max = np.maximum(texture.reshape(texture.shape[0], -1).max(axis=1), 1e-8)
    tex_norm = texture / tex_max[:, None, None]  # (B, H, W) in [0, 1]

    # Per-frame response normalization to [0, 1] so the detector term composes with
    # the normalized texture term on the same scale.
    resp_max = np.maximum(response.reshape(response.shape[0], -1).max(axis=1), 1e-12)
    resp_norm = response / resp_max[:, None, None]  # (B, H, W) in [0, 1]
    resp_norm = np.clip(resp_norm, 0.0, 1.0)

    if role == RESPONSE_ROLE_CORRECTION:
        detector_term = _RESPONSE_FLOOR_EPS + resp_norm
    else:  # RESPONSE_ROLE_ATTACK
        detector_term = _RESPONSE_FLOOR_EPS + (1.0 - resp_norm)

    cost = (tex_norm * detector_term).astype(np.float32)

    stats = FullGridSegNetResponseCostMap(
        cost_bhw=cost,
        role=role,
        response_nonzero_fraction=float((response > 0.0).mean()),
        response_mean=float(response.mean()),
        response_gini=_gini(response),
        texture_mean=float(texture.mean()),
        cost_gini=_gini(cost),
        n_pixels=int(cost.size),
    )
    return stats


def full_grid_response_allocation_diff_proof(
    cost_a_bhw: Any,
    cost_b_bhw: Any,
    *,
    n_kept: int,
) -> dict[str, Any]:
    """NON-FAKE allocation-diff proof (Catalog #105/#139/#220).

    Proves the cost-map actually CHANGES which δ entries survive a byte budget: it
    simulates ``pack_sparse_delta``'s top-K ranking (``|δ| * (1 + cost_norm)``) with a
    constant ``|δ|=1`` stand-in so the *only* thing distinguishing the two rankings is
    the cost map, then reports the symmetric difference of the kept index sets.

    Returns ``kept_set_symmetric_difference`` (>0 ⇒ the maps allocate differently —
    the no-op guard) plus the per-map kept counts.
    """
    a = _to_bhw_f32(cost_a_bhw, "cost_a_bhw")
    b = _to_bhw_f32(cost_b_bhw, "cost_b_bhw")
    if a.shape != b.shape:
        raise FullGridResponseError(f"shape mismatch: a={a.shape} vs b={b.shape}")
    if n_kept < 0:
        raise FullGridResponseError(f"n_kept must be non-negative; got {n_kept}")

    def _topk_set(cost: np.ndarray) -> set[int]:
        # Replicate pack_sparse_delta's normalization: |δ|=1 ⇒ rank = 1 + cost_norm.
        cmax = np.maximum(cost.reshape(cost.shape[0], -1).max(axis=1), 1e-8)
        cnorm = cost / cmax[:, None, None]
        rank = (1.0 + cnorm).ravel()
        k = min(n_kept, rank.size)
        if k <= 0:
            return set()
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
        "schema": "full_grid_segnet_response_allocation_diff_proof_v1",
    }


def update_from_anchor(anchor: Mapping[str, Any]) -> dict[str, Any]:
    """Continual-learning hook 5 (Catalog #125 / #335 contract).

    Observability-only: records that a full-grid-response empirical anchor landed.
    This surface ranks δ entries; it never mutates a score, so the anchor is echoed
    back with canonical non-promotable markers (Catalog #341). No fabricated numbers —
    only the anchor's own fields are reflected.
    """
    role = str(anchor.get("role", "")) if isinstance(anchor, Mapping) else ""
    return {
        "consumed": True,
        "role": role,
        "axis_tag": _CANONICAL_AXIS_TAG,
        "score_claim": False,
        "promotable": False,
        "predicted_delta_adjustment": 0.0,
        "schema": "full_grid_segnet_response_cost_map_update_v1",
    }
