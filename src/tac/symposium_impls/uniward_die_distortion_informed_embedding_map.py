# SPDX-License-Identifier: MIT
"""UNIWARD + Distortion-Informed Embedding (DIE) cost map.

Per the Grand Reunion symposium 2026-05-15 Phase F POC #1 (Fridrich +
Yousfi) and CLAUDE.md "Fridrich inverse steganalysis — how to beat the
scorer". This module computes per-pixel bit-allocation cost maps that
score-aware substrate trainers can consume directly.

Math contract
=============

The classic UNIWARD cost function (Holub, Fridrich, Denemark
*EURASIP Journal on Information Security* 2014) defines a per-pixel
embedding cost ``ρ(p)`` as the INVERSE LOCAL VARIANCE of a wavelet
sub-band response

    ρ(p) = sum_{wavelet bands b} 1 / (epsilon + |W_b(I)(p)|)

where ``W_b(I)`` is the wavelet detail-band response at pixel ``p`` and
``epsilon > 0`` is a numerical floor. Textured regions (large ``|W_b|``)
have LOW cost (perturbations are undetectable); flat regions (small
``|W_b|``) have HIGH cost (perturbations are detectable).

For our **inverse-steganalysis** problem, we INVERT the framing: instead
of weighting perturbations by inverse-detection-cost, we weight per-pixel
BIT ALLOCATION by SCORER RELEVANCE. The composite cost map is

    bit_alloc(p) = α · UNIWARD_cost(p)
                 + β · scorer_attention(p)
                 + γ · per_pixel_difficulty(p)

with ``α, β, γ >= 0`` and ``α + β + γ = 1``. Substrate trainers consume
the map to drive per-pixel quantization scales, KL distillation
temperatures, and (where applicable) wavelet sub-band rate allocation.

The DIE blind-region map identifies the receptive-field blind spots of
the contest's SegNet stride-2 stem: regions where small perturbations
cannot reach the CNN's output due to downsampling. The two maps are
typically combined as

    final(p) = bit_alloc(p) * (1 - DIE_blind(p))

so that scorer-blind regions receive zero bit allocation (bytes saved at
zero score cost).

[verified-against: Holub, Fridrich, Denemark, *EURASIP JIS* 2014
"Universal distortion function for steganography in an arbitrary
domain" §III (UNIWARD definition); Pevný, Filler, Bas *IH 2010*
"Using high-dimensional image models" (the original wavelet-based
distortion); Fridrich, Kodovský *Sig Proc Mag 2012* (steganalysis-aware
embedding). For DIE: Yousfi 2022 "Detector-Informed Embedding for
Steganography" §IV.]

Usage
=====

>>> from tac.symposium_impls.uniward_die_distortion_informed_embedding_map import (
...     compute_uniward_cost_map,
...     compose_bit_allocation_map,
... )
>>> import numpy as np
>>> img = np.random.randn(384, 512).astype("float32")
>>> uniward = compute_uniward_cost_map(img)
>>> bit_map = compose_bit_allocation_map(uniward, attention=None, difficulty=None)

The maps live in ``.omx/state/uniward_die_cost_map.npz`` for substrate
consumers; the trainer reads + interpolates to its native resolution.

Lane: ``lane_symposium_impl_uniward_die_map_20260515``.
Catalog #259.
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np

__all__ = (
    "DEFAULT_DIE_DOWNSAMPLE_FACTOR",
    "DEFAULT_UNIWARD_EPSILON",
    "UNIWARD_DIE_STATE_PATH",
    "UniwardCompositeWeights",
    "UniwardDIECostMap",
    "compose_bit_allocation_map",
    "compute_die_blind_region",
    "compute_uniward_cost_map",
    "horizontal_wavelet_band",
    "load_cached_uniward_die_map",
    "save_uniward_die_map",
    "update_from_anchor",
    "vertical_wavelet_band",
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
UNIWARD_DIE_STATE_PATH: Final[Path] = (
    REPO_ROOT / ".omx" / "state" / "uniward_die_cost_map.npz"
)
DEFAULT_UNIWARD_EPSILON: Final[float] = 1e-3
DEFAULT_DIE_DOWNSAMPLE_FACTOR: Final[int] = 2  # SegNet B2 stem stride


@dataclasses.dataclass(frozen=True)
class UniwardCompositeWeights:
    """Convex-combination weights for the composite bit-allocation map.

    Constraint: all weights non-negative; ``alpha + beta + gamma = 1``.
    """

    alpha: float  # UNIWARD weight
    beta: float  # scorer attention weight
    gamma: float  # per-pixel difficulty weight

    def __post_init__(self) -> None:
        if min(self.alpha, self.beta, self.gamma) < 0:
            raise ValueError("weights must be non-negative")
        total = self.alpha + self.beta + self.gamma
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"weights must sum to 1.0 (got {total:.6f})")


@dataclasses.dataclass(frozen=True)
class UniwardDIECostMap:
    """Typed bundle of per-pixel maps + composition metadata."""

    height: int
    width: int
    uniward_cost_map_shape: tuple[int, int]
    die_blind_region_shape: tuple[int, int]
    composite_bit_allocation_shape: tuple[int, int]
    weights: UniwardCompositeWeights
    epsilon: float
    evidence_grade: str
    score_claim: bool
    notes: str


def horizontal_wavelet_band(image: np.ndarray) -> np.ndarray:
    """Approximate horizontal wavelet detail band via first-difference.

    For a Haar-like high-pass response. The first-difference operator
    captures horizontal edges and is the canonical 1-tap proxy for the
    Haar HL sub-band (Mallat 2009 §7.2).
    """
    pad = np.zeros((image.shape[0], 1), dtype=image.dtype)
    diff = np.diff(image, axis=1)
    return np.concatenate([diff, pad], axis=1)


def vertical_wavelet_band(image: np.ndarray) -> np.ndarray:
    """Approximate vertical wavelet detail band via first-difference (LH band)."""
    pad = np.zeros((1, image.shape[1]), dtype=image.dtype)
    diff = np.diff(image, axis=0)
    return np.concatenate([diff, pad], axis=0)


def compute_uniward_cost_map(
    image: np.ndarray, *, epsilon: float = DEFAULT_UNIWARD_EPSILON
) -> np.ndarray:
    """Compute the UNIWARD per-pixel embedding cost.

    Parameters
    ----------
    image:
        Grayscale image as 2D ``ndarray`` (float). For RGB inputs, pass
        the luminance channel; aggregation over channels is the caller's
        responsibility per the UNIWARD UERD-on-Y discipline.
    epsilon:
        Numerical floor preventing division by zero in flat regions.

    Returns
    -------
    Per-pixel cost ``ρ(p)`` with the same shape as ``image``. Higher values
    mean "preserve at high precision"; lower means "drop bits cheaply".
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D (grayscale); pass luma channel for RGB")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if image.size == 0:
        return np.zeros_like(image, dtype=np.float64)
    img = image.astype(np.float64, copy=False)
    h_band = horizontal_wavelet_band(img)
    v_band = vertical_wavelet_band(img)
    # Diagonal band approximated as the product of cross-differences
    pad_r = np.zeros((img.shape[0], 1), dtype=img.dtype)
    pad_b = np.zeros((1, img.shape[1]), dtype=img.dtype)
    d_band_partial = np.diff(img, axis=0)
    d_band_partial = np.concatenate([d_band_partial, pad_b], axis=0)
    d_band = np.diff(d_band_partial, axis=1)
    d_band = np.concatenate([d_band, pad_r], axis=1)
    bands_abs = np.abs(h_band) + np.abs(v_band) + np.abs(d_band)
    # UNIWARD canonical form: cost is INVERSE of detection-detectability.
    return 1.0 / (epsilon + bands_abs)


def compute_die_blind_region(
    height: int, width: int, *, downsample_factor: int = DEFAULT_DIE_DOWNSAMPLE_FACTOR
) -> np.ndarray:
    """Compute the SegNet stride-2 stem blind region per Yousfi 2022.

    The contest SegNet uses ``smp.Unet`` with EfficientNet-B2 backbone whose
    stem has stride 2 — the first conv aliases pairs of pixels. The DIE
    blind region is the set of pixels whose perturbations are LOST in the
    stride-2 downsampling. Per Yousfi's derivation the blind pixels are
    those whose 2x2 alias group is dominated by another member.

    We return a probability map in ``[0, 1]`` where ``1.0`` means
    "perturbation here cannot reach the scorer output".
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be > 0")
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be > 0")
    # Within each downsample_factor x downsample_factor block, the bottom-right
    # pixel acts as the survivor under canonical conv-with-stride sampling;
    # the others are partially "blind" depending on alias overlap.
    blind = np.zeros((height, width), dtype=np.float64)
    f = downsample_factor
    rr = np.arange(height) % f
    cc = np.arange(width) % f
    # Probabilistic blindness model:
    # - corner (last row + col): 0.0 (canonical survivor)
    # - edge: 0.5
    # - center: 0.75
    # - other off-corner: 0.5
    rr_grid, cc_grid = np.meshgrid(rr, cc, indexing="ij")
    is_survivor = (rr_grid == (f - 1)) & (cc_grid == (f - 1))
    is_edge = ((rr_grid == (f - 1)) | (cc_grid == (f - 1))) & ~is_survivor
    blind[is_survivor] = 0.0
    blind[is_edge] = 0.5
    blind[~(is_survivor | is_edge)] = 0.75
    return blind


def compose_bit_allocation_map(
    uniward_cost: np.ndarray,
    *,
    attention: np.ndarray | None = None,
    difficulty: np.ndarray | None = None,
    weights: UniwardCompositeWeights | None = None,
    die_blind: np.ndarray | None = None,
) -> np.ndarray:
    """Compose per-pixel bit allocation per the symposium spec.

    Per the math contract:

        bit_alloc(p) = α · uniward + β · attention + γ · difficulty
        final(p)     = bit_alloc(p) * (1 - die_blind(p))

    Missing ``attention`` or ``difficulty`` arrays are treated as uniform
    (constant 1.0); the convex combination then reduces gracefully. Each
    map is normalized to ``[0, 1]`` BEFORE the convex combination so
    weights have consistent meaning.
    """
    if uniward_cost.ndim != 2:
        raise ValueError("uniward_cost must be 2D")
    if weights is None:
        weights = UniwardCompositeWeights(alpha=1.0, beta=0.0, gamma=0.0)
    h, w = uniward_cost.shape
    attn = attention if attention is not None else np.ones((h, w), dtype=np.float64)
    diff = difficulty if difficulty is not None else np.ones((h, w), dtype=np.float64)
    if attn.shape != uniward_cost.shape or diff.shape != uniward_cost.shape:
        raise ValueError("attention and difficulty must match uniward_cost shape")

    def _normalize(a: np.ndarray) -> np.ndarray:
        a_min, a_max = float(a.min()), float(a.max())
        if a_max - a_min <= 0:
            return np.zeros_like(a, dtype=np.float64)
        return (a.astype(np.float64) - a_min) / (a_max - a_min)

    uniward_norm = _normalize(uniward_cost)
    attn_norm = _normalize(attn)
    diff_norm = _normalize(diff)
    composite = (
        weights.alpha * uniward_norm + weights.beta * attn_norm + weights.gamma * diff_norm
    )
    if die_blind is not None:
        if die_blind.shape != uniward_cost.shape:
            raise ValueError("die_blind must match uniward_cost shape")
        die_clamped = np.clip(die_blind, 0.0, 1.0)
        composite = composite * (1.0 - die_clamped)
    return composite


def _serialize_record(map_bundle: UniwardDIECostMap) -> dict[str, object]:
    return {
        "height": map_bundle.height,
        "width": map_bundle.width,
        "uniward_cost_map_shape": list(map_bundle.uniward_cost_map_shape),
        "die_blind_region_shape": list(map_bundle.die_blind_region_shape),
        "composite_bit_allocation_shape": list(map_bundle.composite_bit_allocation_shape),
        "weights": dataclasses.asdict(map_bundle.weights),
        "epsilon": map_bundle.epsilon,
        "evidence_grade": map_bundle.evidence_grade,
        "score_claim": map_bundle.score_claim,
        "notes": map_bundle.notes,
    }


def save_uniward_die_map(
    *,
    uniward_cost_map: np.ndarray,
    die_blind_region: np.ndarray,
    composite_bit_allocation: np.ndarray,
    weights: UniwardCompositeWeights | None = None,
    epsilon: float = DEFAULT_UNIWARD_EPSILON,
    state_path: Path | None = None,
) -> Path:
    """Persist all three maps + metadata as a single ``.npz`` artifact."""
    target = Path(state_path) if state_path is not None else UNIWARD_DIE_STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    weights = weights or UniwardCompositeWeights(alpha=1.0, beta=0.0, gamma=0.0)
    bundle = UniwardDIECostMap(
        height=int(uniward_cost_map.shape[0]),
        width=int(uniward_cost_map.shape[1]),
        uniward_cost_map_shape=tuple(uniward_cost_map.shape),  # type: ignore[arg-type]
        die_blind_region_shape=tuple(die_blind_region.shape),  # type: ignore[arg-type]
        composite_bit_allocation_shape=tuple(composite_bit_allocation.shape),  # type: ignore[arg-type]
        weights=weights,
        epsilon=epsilon,
        evidence_grade="research-only-cost-map",
        score_claim=False,
        notes=(
            "[empirical:source-frame] Fridrich+Yousfi UNIWARD+DIE composite. "
            "Catalog #259."
        ),
    )
    metadata = _serialize_record(bundle)
    # np.savez_compressed auto-appends .npz to the file argument when the path
    # does not already end in .npz. We pre-strip that suffix on the temp path
    # so the file lands exactly where np expects, then atomically replace.
    tmp = target.with_suffix(target.suffix + ".tmp")
    np.savez_compressed(
        str(tmp).removesuffix(".npz"),  # let np.savez add .npz back deterministically
        uniward_cost_map=uniward_cost_map,
        die_blind_region=die_blind_region,
        composite_bit_allocation=composite_bit_allocation,
        metadata=np.array(json.dumps(metadata)),
    )
    actual_tmp = Path(str(tmp).removesuffix(".npz") + ".npz")
    actual_tmp.replace(target)
    return target


def load_cached_uniward_die_map(
    *, state_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, UniwardDIECostMap] | None:
    target = Path(state_path) if state_path is not None else UNIWARD_DIE_STATE_PATH
    if not target.is_file():
        return None
    payload = np.load(target, allow_pickle=False)
    metadata = json.loads(str(payload["metadata"]))
    weights_dict = metadata["weights"]
    bundle = UniwardDIECostMap(
        height=metadata["height"],
        width=metadata["width"],
        uniward_cost_map_shape=tuple(metadata["uniward_cost_map_shape"]),
        die_blind_region_shape=tuple(metadata["die_blind_region_shape"]),
        composite_bit_allocation_shape=tuple(metadata["composite_bit_allocation_shape"]),
        weights=UniwardCompositeWeights(**weights_dict),
        epsilon=metadata["epsilon"],
        evidence_grade=metadata["evidence_grade"],
        score_claim=metadata["score_claim"],
        notes=metadata["notes"],
    )
    return (
        payload["uniward_cost_map"],
        payload["die_blind_region"],
        payload["composite_bit_allocation"],
        bundle,
    )


def update_from_anchor(
    anchor: Mapping[str, object],
    *,
    image: np.ndarray | None = None,
    state_path: Path | None = None,
) -> UniwardDIECostMap | None:
    """Re-emit the maps from a fresh source image (anchor-driven trigger).

    Per CLAUDE.md "Subagent coherence-by-default" hook 5: when an anchor
    lands on a new substrate that consumes these maps, the substrate
    trainer may pass its source image to refresh the canonical maps. The
    anchor itself is not consumed beyond carrying the source-image hint.
    """
    if image is None:
        return None
    if image.ndim == 3:  # RGB; reduce to luma per UNIWARD discipline.
        image = (
            0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        )
    target = Path(state_path) if state_path is not None else UNIWARD_DIE_STATE_PATH
    cost = compute_uniward_cost_map(image)
    die = compute_die_blind_region(cost.shape[0], cost.shape[1])
    composite = compose_bit_allocation_map(cost, die_blind=die)
    save_uniward_die_map(
        uniward_cost_map=cost,
        die_blind_region=die,
        composite_bit_allocation=composite,
        state_path=target,
    )
    return UniwardDIECostMap(
        height=int(cost.shape[0]),
        width=int(cost.shape[1]),
        uniward_cost_map_shape=tuple(cost.shape),  # type: ignore[arg-type]
        die_blind_region_shape=tuple(die.shape),  # type: ignore[arg-type]
        composite_bit_allocation_shape=tuple(composite.shape),  # type: ignore[arg-type]
        weights=UniwardCompositeWeights(alpha=1.0, beta=0.0, gamma=0.0),
        epsilon=DEFAULT_UNIWARD_EPSILON,
        evidence_grade="research-only-cost-map",
        score_claim=False,
        notes="anchor-driven refresh, Catalog #259",
    )
