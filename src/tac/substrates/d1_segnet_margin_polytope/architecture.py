# SPDX-License-Identifier: MIT
"""D1 sidecar architecture — composes with a frozen base substrate.

D1 is a **SIDECAR** like YUCR: it does not own a renderer. The margin-map
computation + polytope allocation logic IS the architecture.
:func:`compose_with_base` wraps any of the recognized base substrates
(A1, PR101, time-traveler, sane_hnerv, YUCR) so the D1 overhead lands as
a small monolithic 0.bin alongside the base archive.

Predicted overhead band: 0.5-3 KB ``[first-principles-bound]`` (smaller
than YUCR because the margin map is non-negative + brotli compresses
high-margin-interior plateaus well). Predicted ΔS contest-CPU:
``[-0.012, -0.005]`` per the deep-math memo §10 D1 derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

# Recognized base substrates that D1 can compose with. Every entry MUST
# have a verified score-aware substrate package under
# ``tac.substrates.<id>``. Default A1: only sub-0.20 anchor in posterior
# as of 2026-05-14. ``yucr`` listed for D1+YUCR cross-axis stacking (the
# frame-0 nullspace + frame-1 polytope bidirectional exploit).
D1POLY_BASE_SUBSTRATE_IDS: tuple[str, ...] = (
    "a1",
    "pr101_lc_v2_clone",
    "time_traveler",
    "time_traveler_l5_autonomy",
    "sane_hnerv",
    "pretrained_driving_prior",
    "yucr",
)
"""Base substrates D1 can compose with. Order = recommended dispatch order
ranked by `feedback_orphan_anchor_backfill_landed_20260513` posterior plus
the YUCR sister-subagent landing 2026-05-14."""

D1POLY_DEFAULT_BASE_SUBSTRATE: str = "a1"
"""A1 is the canonical first composition target — sole sub-0.20 anchor."""

D1POLY_DEFAULT_BUDGET_BITS: int = 8000
"""Default polytope payload budget in bits (~ 1 KB after brotli)."""

D1POLY_OVERHEAD_TARGET_BYTES_MIN: int = 512
"""Minimum D1 overhead — header + small payload."""

D1POLY_OVERHEAD_TARGET_BYTES_MAX: int = 3072
"""Maximum D1 overhead before we exceed the rate-axis Pareto budget."""

EVAL_HW: tuple[int, int] = (384, 512)
"""Scorer-resolution eval grid for margin-map computation."""

SHRUNK_HW: tuple[int, int] = (96, 128)
"""16x-shrunk margin map grid for archive cost reduction.

See :data:`tac.substrates.d1_segnet_margin_polytope.margin_map.MARGIN_MAP_SHRUNK_RESOLUTION`
for the full rationale. Encoder-side area-pool downsample 384x512 → 96x128;
inflate-side bicubic upsample 96x128 → 384x512 via
:func:`tac.substrates.d1_segnet_margin_polytope.margin_map.upsample_margin_map_for_overlay`.
"""

# Conservative empirical Lipschitz bound for SegNet (logit-vs-input-RGB).
# Calibrated offline on a held-out frame batch; recorded in archive
# metadata so the receiver inverts deterministically. Operators should
# override per-deployment if a tighter empirical L is measured.
_DEFAULT_JACOBIAN_LIPSCHITZ: float = 20.0


@dataclass(frozen=True)
class D1PolytopeConfig:
    """D1 substrate configuration.

    Validated at construction so a YAML recipe cannot dispatch with a
    contradictory L/budget/margin-threshold combo.

    Args:
        base_substrate_id: The composable base substrate. Must be in
            :data:`D1POLY_BASE_SUBSTRATE_IDS`.
        margin_map_mode: How to derive the per-pixel margin map.
            ``"segnet_top1_minus_top2"`` uses the canonical SegNet
            top1-minus-top2 logit margin per
            :func:`tac.substrates.d1_segnet_margin_polytope.margin_map.compute_logit_margin_map`.
        polytope_payload_bits: Bit budget allocated to the polytope payload.
            Smaller = tighter Pareto; larger = more reconstruction
            headroom on safe-polytope-interior pixels.
        margin_map_resolution: Spatial resolution of the margin map.
            Defaults to scorer eval resolution (384, 512).
        margin_map_int8_scale: Quantization scale for margin-map int8
            packing. Larger = finer margin resolution but larger archive.
        jacobian_lipschitz: SegNet Jacobian operator-norm upper bound
            ``L``. Per-pixel safe budget = ``margin / L``. Calibrated
            offline. Default 20.0 is conservative for normalized RGB
            inputs on SegNet (tu-efficientnet_b2) per the deep-math memo
            §2.5 + §3.6 derivations.
        margin_threshold: Hinge threshold for the score-aware loss term
            (penalizes pushing pixels below this margin during training
            to preserve argmax stability).
        pose_sqrt_weight: ``sqrt(10)`` per the contest formula. Captured
            in config so it shows up in archive metadata + readiness
            manifest.
        seg_weight: 100.0 per the contest formula.
    """

    base_substrate_id: str = D1POLY_DEFAULT_BASE_SUBSTRATE
    margin_map_mode: str = "segnet_top1_minus_top2"
    polytope_payload_bits: int = D1POLY_DEFAULT_BUDGET_BITS
    margin_map_resolution: tuple[int, int] = EVAL_HW
    margin_map_int8_scale: float = 127.0
    jacobian_lipschitz: float = _DEFAULT_JACOBIAN_LIPSCHITZ
    margin_threshold: float = 0.1
    pose_sqrt_weight: float = 10.0**0.5
    seg_weight: float = 100.0
    eps: float = 1e-6

    def __post_init__(self) -> None:  # noqa: D401
        if self.base_substrate_id not in D1POLY_BASE_SUBSTRATE_IDS:
            raise ValueError(
                f"D1PolytopeConfig.base_substrate_id={self.base_substrate_id!r} "
                f"not in {D1POLY_BASE_SUBSTRATE_IDS}. Add the base to "
                "D1POLY_BASE_SUBSTRATE_IDS before composing."
            )
        if self.margin_map_mode not in {
            "segnet_top1_minus_top2",
            "uniform",
            "dummy_constant",
        }:
            raise ValueError(
                f"D1PolytopeConfig.margin_map_mode={self.margin_map_mode!r} "
                "unsupported. Expected segnet_top1_minus_top2 | uniform | "
                "dummy_constant."
            )
        if self.polytope_payload_bits <= 0 or self.polytope_payload_bits > 1 << 20:
            raise ValueError(
                f"D1PolytopeConfig.polytope_payload_bits="
                f"{self.polytope_payload_bits} out of range [1, 2^20]"
            )
        if (
            self.margin_map_resolution[0] <= 0
            or self.margin_map_resolution[1] <= 0
        ):
            raise ValueError(
                "D1PolytopeConfig.margin_map_resolution must be positive; "
                f"got {self.margin_map_resolution}"
            )
        if (
            self.margin_map_int8_scale <= 0
            or self.margin_map_int8_scale > 1024
        ):
            raise ValueError(
                f"D1PolytopeConfig.margin_map_int8_scale="
                f"{self.margin_map_int8_scale} out of range (0, 1024]"
            )
        if self.jacobian_lipschitz <= 0 or self.jacobian_lipschitz > 1024.0:
            raise ValueError(
                f"D1PolytopeConfig.jacobian_lipschitz="
                f"{self.jacobian_lipschitz} out of range (0, 1024]. The "
                "operator-norm upper bound must be strictly positive; "
                "typical SegNet values are O(10-50)."
            )
        if self.margin_threshold < 0 or self.margin_threshold > 10.0:
            raise ValueError(
                f"D1PolytopeConfig.margin_threshold="
                f"{self.margin_threshold} out of range [0, 10]"
            )
        if self.eps <= 0 or self.eps > 1.0:
            raise ValueError(
                f"D1PolytopeConfig.eps={self.eps} out of range (0, 1]"
            )


@dataclass(frozen=True)
class _BaseArchiveDescriptor:
    """Identifies the base substrate's archive without copying its bytes.

    D1 records ``base_archive_sha256_truncated[:16]`` (per Catalog #157
    SHA-256 discipline) so inflate-time we can verify the consumer is
    paired with the correct base archive. The base bytes themselves are
    NOT inside the D1 overhead; they live in their own zip member.
    """

    base_substrate_id: str
    base_archive_sha256: str
    base_archive_bytes: int

    def __post_init__(self) -> None:  # noqa: D401
        if self.base_substrate_id not in D1POLY_BASE_SUBSTRATE_IDS:
            raise ValueError(
                f"_BaseArchiveDescriptor.base_substrate_id="
                f"{self.base_substrate_id!r} not in "
                f"{D1POLY_BASE_SUBSTRATE_IDS}"
            )
        if len(self.base_archive_sha256) != 64:
            raise ValueError(
                "base_archive_sha256 must be 64-char hex; got len="
                f"{len(self.base_archive_sha256)}"
            )
        if self.base_archive_bytes < 0 or self.base_archive_bytes > 1 << 24:
            raise ValueError(
                f"base_archive_bytes={self.base_archive_bytes} out of "
                "range [0, 2^24]"
            )


@dataclass(frozen=True)
class D1PolytopeSidecar:
    """Top-level D1 substrate handle.

    :class:`D1PolytopeSidecar` is a value object. The actual margin-map +
    polytope-encoder computation happens in :func:`compose_with_base`
    (which returns a packed archive) and in
    :class:`tac.substrates.d1_segnet_margin_polytope.score_aware_loss.D1PolytopeScoreAwareLoss`
    (which is invoked from a trainer). The sidecar handle exists so the
    autopilot ranking + lane registry can carry the typed config without
    instantiating the margin map.
    """

    config: D1PolytopeConfig = field(default_factory=D1PolytopeConfig)


def compose_with_base(
    *,
    base_archive_descriptor: _BaseArchiveDescriptor,
    margin_map: torch.Tensor,
    polytope_payload: bytes,
    config: D1PolytopeConfig,
    extra_meta: Mapping[str, object] | None = None,
) -> bytes:
    """Compose a D1 sidecar with a frozen base archive.

    Returns the packed D1POLY1 0.bin bytes (header + margin_map_int8 +
    polytope_payload + base_archive_id + meta JSON). The base archive bytes
    are NOT bundled here — they live in their own zip member.
    ``base_archive_sha256`` records the binding so inflate-time can
    fail-closed if the operator paired the wrong base.

    The returned blob is the canonical D1 sidecar archive. To produce a
    full submission packet, write this blob alongside the base archive
    into the contest archive zip.

    Args:
        base_archive_descriptor: SHA-256-bound identifier for the base.
        margin_map: ``(H, W)`` float32 tensor with the SegNet logit margin
            map. Will be quantized to int8 inside ``pack_archive``.
        polytope_payload: Bytes from
            :func:`tac.substrates.d1_segnet_margin_polytope.polytope_encoder.encode_polytope_payload`.
        config: :class:`D1PolytopeConfig` used to derive the margin map /
            payload.
        extra_meta: Optional sidecar metadata (e.g. trainer hash, anchor
            sha256, predicted ΔS). Will be JSON-serialized into META
            section.

    Raises:
        ValueError: when margin map shape doesn't match
            ``config.margin_map_resolution``.
    """
    if margin_map.dim() != 2:
        raise ValueError(
            f"compose_with_base expects 2D margin_map (H, W); got "
            f"{tuple(margin_map.shape)}"
        )
    if tuple(margin_map.shape) != tuple(config.margin_map_resolution):
        raise ValueError(
            f"margin_map shape {tuple(margin_map.shape)} != "
            f"config.margin_map_resolution {config.margin_map_resolution}"
        )

    # Local import to avoid cycle (archive.py imports from this module).
    from tac.substrates.d1_segnet_margin_polytope.archive import pack_archive

    return pack_archive(
        margin_map=margin_map,
        polytope_payload=polytope_payload,
        jacobian_lipschitz=float(config.jacobian_lipschitz),
        base_substrate_id=base_archive_descriptor.base_substrate_id,
        base_archive_sha256=base_archive_descriptor.base_archive_sha256,
        base_archive_bytes=base_archive_descriptor.base_archive_bytes,
        config=config,
        extra_meta=extra_meta or {},
    )


def estimate_overhead_bytes(
    *,
    config: D1PolytopeConfig,
    margin_zero_count: int = 0,
) -> int:
    """Conservative overhead estimate for autopilot ranking.

    Predicted overhead = header (~64 B) + margin_map_int8 (~H*W bytes;
    brotli closes 0.35-0.55 ratio for margin maps with plateaus) +
    polytope_payload (~bits/8 after brotli ~0.5x) + meta_json (~256 B).
    The ``margin_zero_count`` parameter lets autopilot estimate the
    post-brotli reduction when the margin map has many boundary pixels
    (typical when most pixels are scorer-sensitive — but the polytope
    encoder makes those small so the brotli ratio is still favorable).
    """
    h, w = config.margin_map_resolution
    margin_size = h * w
    nonzero_frac = max(0.1, 1.0 - margin_zero_count / max(margin_size, 1))
    margin_compressed = int(0.45 * margin_size * nonzero_frac)
    polytope_compressed = int(0.5 * config.polytope_payload_bits / 8)
    return 64 + margin_compressed + polytope_compressed + 256


__all__ = [
    "D1POLY_BASE_SUBSTRATE_IDS",
    "D1POLY_DEFAULT_BASE_SUBSTRATE",
    "D1POLY_DEFAULT_BUDGET_BITS",
    "D1POLY_OVERHEAD_TARGET_BYTES_MAX",
    "D1POLY_OVERHEAD_TARGET_BYTES_MIN",
    "EVAL_HW",
    "SHRUNK_HW",
    "D1PolytopeConfig",
    "D1PolytopeSidecar",
    "_BaseArchiveDescriptor",
    "compose_with_base",
    "estimate_overhead_bytes",
]
