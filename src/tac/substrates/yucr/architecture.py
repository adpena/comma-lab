"""YUCR sidecar architecture — composes with a frozen base substrate.

YUCR is a **SIDECAR**: it does not own a renderer. The cost-map computation
+ STC allocation logic IS the architecture. ``compose_with_base`` wraps any
of the recognized base substrates (A1, PR101, time-traveler, sane_hnerv,
DP1) so the YUCR overhead lands as a small monolithic 0.bin alongside the
base archive.

Predicted overhead band: 1-5 KB ``[time-traveler-prediction]``. Predicted
ΔS contest-CPU: ``[-0.020, -0.040]`` per the synthesis derivation in
``feedback_yousfi_uniward_cooperative_receiver_synthesis_landed_20260513``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

# Recognized base substrates that YUCR can compose with. Every entry MUST
# have a verified score-aware substrate package under ``tac.substrates.<id>``.
# The default (A1) was chosen because A1's verified contest-CPU anchor
# 0.192848 is the only sub-0.20 anchor in the empirical posterior as of
# 2026-05-14, so YUCR's predicted -0.02 to -0.04 lands in the medal band.
YUCR_BASE_SUBSTRATE_IDS: tuple[str, ...] = (
    "a1",
    "pr101_lc_v2_clone",
    "time_traveler",
    "time_traveler_l5_autonomy",
    "sane_hnerv",
    "pretrained_driving_prior",
    "wyner_ziv_cooperative_receiver",
)
"""Base substrates YUCR can compose with. Order = recommended dispatch order
ranked by `feedback_orphan_anchor_backfill_landed_20260513` posterior."""

YUCR_DEFAULT_BASE_SUBSTRATE: str = "a1"
"""A1 is the canonical first composition target — sole sub-0.20 anchor."""

YUCR_DEFAULT_STC_PAYLOAD_BITS: int = 8000
"""Default STC payload budget in bits (~ 1 KB stc_payload after brotli)."""

YUCR_OVERHEAD_TARGET_BYTES_MIN: int = 1024
"""Minimum YUCR overhead — even an empty cost-map carries header bytes."""

YUCR_OVERHEAD_TARGET_BYTES_MAX: int = 5120
"""Maximum YUCR overhead before we exceed the rate-axis Pareto budget."""

EVAL_HW: tuple[int, int] = (384, 512)
"""Scorer-resolution eval grid for cost-map computation."""


@dataclass(frozen=True)
class YUCRConfig:
    """YUCR substrate configuration.

    Validated at construction time so a YAML recipe cannot dispatch with
    a contradictory pose-sqrt + L_inf-cap + STC budget combo.

    Args:
        base_substrate_id: The composable base substrate. Must be in
            :data:`YUCR_BASE_SUBSTRATE_IDS`.
        cost_map_mode: How to derive the per-pixel cost map. ``"score_gradient"``
            uses the canonical Atick-Redlich orthogonal-complement projector
            (:func:`tac.substrates.yucr.cost_map.compute_cost_map`).
        stc_payload_bits: Bit budget allocated to the STC payload. Smaller =
            tighter Pareto; larger = more reconstruction headroom.
        cost_map_resolution: Spatial resolution of the cost map. Defaults to
            scorer eval resolution (384, 512).
        cost_map_int8_scale: Quantization scale for cost-map int8 packing.
            Larger = finer cost resolution but larger archive.
        l_inf_noise_cap: Hard cap on per-pixel STC noise magnitude. The
            UNIWARD water-fill is unbounded mathematically; the cap prevents
            visible artifacts.
        pose_sqrt_weight: ``sqrt(10)`` per the contest formula. Captured in
            the config so it shows up in archive metadata + readiness manifest.
        seg_weight: 100.0 per the contest formula.
    """

    base_substrate_id: str = YUCR_DEFAULT_BASE_SUBSTRATE
    cost_map_mode: str = "score_gradient"
    stc_payload_bits: int = YUCR_DEFAULT_STC_PAYLOAD_BITS
    cost_map_resolution: tuple[int, int] = EVAL_HW
    cost_map_int8_scale: float = 127.0
    l_inf_noise_cap: float = 4.0
    pose_sqrt_weight: float = 10.0**0.5
    seg_weight: float = 100.0
    eps: float = 1e-6

    def __post_init__(self) -> None:  # noqa: D401
        if self.base_substrate_id not in YUCR_BASE_SUBSTRATE_IDS:
            raise ValueError(
                f"YUCRConfig.base_substrate_id={self.base_substrate_id!r} not in "
                f"{YUCR_BASE_SUBSTRATE_IDS}. Add the base to YUCR_BASE_SUBSTRATE_IDS "
                "before composing."
            )
        if self.cost_map_mode not in {"score_gradient", "uniform", "dummy_constant"}:
            raise ValueError(
                f"YUCRConfig.cost_map_mode={self.cost_map_mode!r} unsupported. "
                "Expected score_gradient | uniform | dummy_constant."
            )
        if self.stc_payload_bits <= 0 or self.stc_payload_bits > 1 << 20:
            raise ValueError(
                f"YUCRConfig.stc_payload_bits={self.stc_payload_bits} out of "
                "range [1, 2^20]"
            )
        if self.cost_map_resolution[0] <= 0 or self.cost_map_resolution[1] <= 0:
            raise ValueError(
                f"YUCRConfig.cost_map_resolution must be positive; "
                f"got {self.cost_map_resolution}"
            )
        if self.cost_map_int8_scale <= 0 or self.cost_map_int8_scale > 1024:
            raise ValueError(
                f"YUCRConfig.cost_map_int8_scale={self.cost_map_int8_scale} out of "
                "range (0, 1024]"
            )
        if self.l_inf_noise_cap <= 0 or self.l_inf_noise_cap > 32.0:
            raise ValueError(
                f"YUCRConfig.l_inf_noise_cap={self.l_inf_noise_cap} out of "
                "range (0, 32]"
            )
        if self.eps <= 0 or self.eps > 1.0:
            raise ValueError(f"YUCRConfig.eps={self.eps} out of range (0, 1]")


@dataclass(frozen=True)
class _BaseArchiveDescriptor:
    """Identifies the base substrate's archive without copying its bytes.

    YUCR records ``base_archive_sha256_truncated[:16]`` (per Catalog #157
    SHA-256 discipline) so inflate-time we can verify the consumer is paired
    with the correct base archive. The base bytes themselves are NOT inside
    the YUCR overhead; they live in their own zip member.
    """

    base_substrate_id: str
    base_archive_sha256: str
    base_archive_bytes: int

    def __post_init__(self) -> None:  # noqa: D401
        if self.base_substrate_id not in YUCR_BASE_SUBSTRATE_IDS:
            raise ValueError(
                f"_BaseArchiveDescriptor.base_substrate_id={self.base_substrate_id!r} "
                f"not in {YUCR_BASE_SUBSTRATE_IDS}"
            )
        if len(self.base_archive_sha256) != 64:
            raise ValueError(
                "base_archive_sha256 must be 64-char hex; got len="
                f"{len(self.base_archive_sha256)}"
            )
        if self.base_archive_bytes < 0 or self.base_archive_bytes > 1 << 24:
            raise ValueError(
                f"base_archive_bytes={self.base_archive_bytes} out of range [0, 2^24]"
            )


@dataclass(frozen=True)
class YUCRSubstrate:
    """Top-level YUCR substrate handle.

    ``YUCRSubstrate`` is a value object. The actual cost-map + STC computation
    happens in :func:`compose_with_base` (which returns a packed archive) and
    in :class:`tac.substrates.yucr.score_aware_loss.YUCRScoreAwareLoss`
    (which is invoked from a trainer). The substrate handle exists so the
    autopilot ranking + lane registry can carry the typed config without
    instantiating the cost map.
    """

    config: YUCRConfig = field(default_factory=YUCRConfig)


def compose_with_base(
    *,
    base_archive_descriptor: _BaseArchiveDescriptor,
    cost_map: torch.Tensor,
    stc_payload: bytes,
    config: YUCRConfig,
    extra_meta: Mapping[str, object] | None = None,
) -> bytes:
    """Compose a YUCR sidecar with a frozen base archive.

    Returns the packed YUCR1 0.bin bytes (header + cost_map_int8 + stc_payload
    + base_archive_id + meta JSON). The base archive bytes are NOT bundled
    here — they live in their own zip member. ``base_archive_sha256`` records
    the binding so inflate-time can fail-closed if the operator paired the
    wrong base.

    The returned blob is the canonical YUCR sidecar archive. To produce a
    full submission packet, write this blob alongside the base archive into
    the contest archive zip.

    Args:
        base_archive_descriptor: SHA-256-bound identifier for the base.
        cost_map: ``(H, W)`` float32 tensor with the score-gradient
            sensitivity map. Will be quantized to int8 inside ``pack_archive``.
        stc_payload: Bytes from :func:`tac.substrates.yucr.stc_encoder.encode_stc_payload`.
        config: :class:`YUCRConfig` used to derive the cost map / payload.
        extra_meta: Optional sidecar metadata (e.g. trainer hash, anchor
            sha256, predicted ΔS). Will be JSON-serialized into META section.

    Raises:
        ValueError: when cost map shape doesn't match ``config.cost_map_resolution``.
    """

    if cost_map.dim() != 2:
        raise ValueError(
            f"compose_with_base expects 2D cost_map (H, W); got {tuple(cost_map.shape)}"
        )
    if tuple(cost_map.shape) != tuple(config.cost_map_resolution):
        raise ValueError(
            f"cost_map shape {tuple(cost_map.shape)} != config.cost_map_resolution "
            f"{config.cost_map_resolution}"
        )

    from tac.substrates.yucr.archive import pack_archive  # local import to avoid cycle

    return pack_archive(
        cost_map=cost_map,
        stc_payload=stc_payload,
        base_substrate_id=base_archive_descriptor.base_substrate_id,
        base_archive_sha256=base_archive_descriptor.base_archive_sha256,
        base_archive_bytes=base_archive_descriptor.base_archive_bytes,
        config=config,
        extra_meta=extra_meta or {},
    )


def estimate_overhead_bytes(
    *,
    config: YUCRConfig,
    cost_map_zero_count: int = 0,
) -> int:
    """Conservative overhead estimate for autopilot ranking.

    Predicted overhead = header (~64 B) + cost_map_int8 (~H*W bytes; brotli
    closes typical ratios 0.4-0.6x) + stc_payload (~stc_payload_bits/8 bytes
    after brotli ~0.5x) + meta_json (~256 B). The ``cost_map_zero_count``
    parameter lets autopilot estimate the post-brotli reduction when the
    cost map has many zeros (typical when most pixels are scorer-blind).
    """

    H, W = config.cost_map_resolution
    cost_map_size = H * W
    nonzero_frac = max(0.05, 1.0 - cost_map_zero_count / max(cost_map_size, 1))
    cost_map_compressed = int(0.55 * cost_map_size * nonzero_frac)
    stc_compressed = int(0.5 * config.stc_payload_bits / 8)
    return 64 + cost_map_compressed + stc_compressed + 256


__all__ = [
    "EVAL_HW",
    "YUCR_BASE_SUBSTRATE_IDS",
    "YUCR_DEFAULT_BASE_SUBSTRATE",
    "YUCR_DEFAULT_STC_PAYLOAD_BITS",
    "YUCR_OVERHEAD_TARGET_BYTES_MAX",
    "YUCR_OVERHEAD_TARGET_BYTES_MIN",
    "YUCRConfig",
    "YUCRSubstrate",
    "_BaseArchiveDescriptor",
    "compose_with_base",
    "estimate_overhead_bytes",
]
