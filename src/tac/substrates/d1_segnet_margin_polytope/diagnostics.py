# SPDX-License-Identifier: MIT
"""D1 overlay diagnostics for no-op prevention and payload forensics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tac.substrates.d1_segnet_margin_polytope.overlay import (
    _build_camera_resolution_overlay,
    attenuate_overlay_levels,
    channel_policy_weights,
    normalize_overlay_amplitude_scale,
    overlay_sign_for_pair,
)
from tac.substrates.d1_segnet_margin_polytope.polytope_encoder import (
    decode_polytope_payload,
)

if TYPE_CHECKING:
    from tac.substrates.d1_segnet_margin_polytope.archive import D1PolytopeArchive


@dataclass(frozen=True)
class D1OverlayDiagnostics:
    """Static diagnostics for a D1 sidecar's inflate-time overlay effect."""

    decoded_noise_pixels: int
    decoded_noise_nonzero_pixels: int
    decoded_noise_abs_sum: int
    camera_overlay_nonzero_pixels: int
    camera_overlay_abs_sum: int
    attenuated_overlay_nonzero_pixels: int
    attenuated_overlay_abs_sum: int
    active_channel_count: int
    estimated_changed_bytes_upper_bound_per_pair: int
    estimated_changed_lsb_l1_upper_bound_per_pair: int
    estimated_changed_lsb_l2_energy_upper_bound_per_pair: int
    integer_feasible_pixels: int
    unsafe_nonzero_pixels: int
    pair_mask_active_pairs: int | None
    max_safe_budget_lsb: float
    mean_safe_budget_lsb: float
    payload_jacobian_lipschitz: float
    archive_jacobian_lipschitz: float
    overlay_channel_policy: str
    overlay_amplitude_scale: float
    overlay_sign_policy: str

    @property
    def dispatch_blockers(self) -> list[str]:
        """Return exact-eval dispatch blockers implied by this payload."""
        blockers: list[str] = []
        if self.decoded_noise_nonzero_pixels <= 0:
            blockers.append("d1_decoded_polytope_payload_all_zero")
        if self.camera_overlay_nonzero_pixels <= 0:
            blockers.append("d1_camera_overlay_all_zero")
        if self.attenuated_overlay_nonzero_pixels <= 0:
            blockers.append("d1_overlay_all_zero_after_attenuation")
        if self.active_channel_count <= 0:
            blockers.append("d1_overlay_channel_policy_has_no_active_channels")
        if self.estimated_changed_bytes_upper_bound_per_pair <= 0:
            blockers.append("d1_estimated_changed_bytes_upper_bound_zero")
        if self.integer_feasible_pixels <= 0:
            blockers.append("d1_no_integer_feasible_pixels_under_lipschitz_bound")
        if self.unsafe_nonzero_pixels > 0:
            blockers.append("d1_overlay_exceeds_integer_safe_budget")
        if self.pair_mask_active_pairs == 0:
            blockers.append("d1_pair_mask_has_no_active_pairs")
        return blockers

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "decoded_noise_pixels": self.decoded_noise_pixels,
            "decoded_noise_nonzero_pixels": self.decoded_noise_nonzero_pixels,
            "decoded_noise_abs_sum": self.decoded_noise_abs_sum,
            "camera_overlay_nonzero_pixels": self.camera_overlay_nonzero_pixels,
            "camera_overlay_abs_sum": self.camera_overlay_abs_sum,
            "attenuated_overlay_nonzero_pixels": self.attenuated_overlay_nonzero_pixels,
            "attenuated_overlay_abs_sum": self.attenuated_overlay_abs_sum,
            "active_channel_count": self.active_channel_count,
            "estimated_changed_bytes_upper_bound_per_pair": (
                self.estimated_changed_bytes_upper_bound_per_pair
            ),
            "estimated_changed_lsb_l1_upper_bound_per_pair": (
                self.estimated_changed_lsb_l1_upper_bound_per_pair
            ),
            "estimated_changed_lsb_l2_energy_upper_bound_per_pair": (
                self.estimated_changed_lsb_l2_energy_upper_bound_per_pair
            ),
            "integer_feasible_pixels": self.integer_feasible_pixels,
            "unsafe_nonzero_pixels": self.unsafe_nonzero_pixels,
            "pair_mask_active_pairs": self.pair_mask_active_pairs,
            "max_safe_budget_lsb": self.max_safe_budget_lsb,
            "mean_safe_budget_lsb": self.mean_safe_budget_lsb,
            "payload_jacobian_lipschitz": self.payload_jacobian_lipschitz,
            "archive_jacobian_lipschitz": self.archive_jacobian_lipschitz,
            "overlay_channel_policy": self.overlay_channel_policy,
            "overlay_amplitude_scale": self.overlay_amplitude_scale,
            "overlay_sign_policy": self.overlay_sign_policy,
            "dispatch_blockers": self.dispatch_blockers,
        }


def analyze_d1_overlay_effect(
    archive: D1PolytopeArchive,
    *,
    channel_policy: str | None = None,
    amplitude_scale: float | None = None,
    sign_policy: str | None = None,
    pair_sign_mask: Sequence[int] | None = None,
    min_integer_delta_lsb: float = 1.0,
) -> D1OverlayDiagnostics:
    """Analyze whether a D1 sidecar can change inflate output bytes.

    This is a static no-op guard. It does not load scorers and does not make a
    score claim. It catches the repeated D1 failure class where a nonempty
    sidecar has a valid payload section but decodes to all-zero lattice levels,
    causing auth eval to spend rate bytes without changing frames.
    """
    policy = str(
        channel_policy
        if channel_policy is not None
        else archive.meta.get("overlay_channel_policy", "rgb")
    )
    scale = normalize_overlay_amplitude_scale(
        amplitude_scale
        if amplitude_scale is not None
        else float(archive.meta.get("overlay_amplitude_scale", 1.0))
    )
    sign = str(
        sign_policy
        if sign_policy is not None
        else archive.meta.get("overlay_sign_policy", "payload")
    )
    overlay_sign_for_pair(sign, 0, pair_sign_mask)
    pair_mask_active_pairs: int | None = None
    if sign.strip().lower() == "pair_mask":
        if pair_sign_mask is None:
            raise ValueError("overlay_sign_policy='pair_mask' requires pair_sign_mask")
        pair_mask_active_pairs = 0
        for pair_idx in range(len(pair_sign_mask)):
            pair_mask_active_pairs += int(
                overlay_sign_for_pair(sign, pair_idx, pair_sign_mask) != 0
            )

    decoded = decode_polytope_payload(archive.polytope_payload)
    camera_overlay = _build_camera_resolution_overlay(
        noise_levels_flat=decoded.noise_levels,
        encoder_grid_h=archive.height,
        encoder_grid_w=archive.width,
    )
    attenuated = attenuate_overlay_levels(
        camera_overlay,
        amplitude_scale=scale,
    )
    weights = channel_policy_weights(policy)
    active_channels = int(np.count_nonzero(weights))
    channel_l1 = int(np.abs(weights).sum())
    channel_l2_energy = int(np.square(weights.astype(np.int16)).sum())
    attenuated_abs = np.abs(attenuated.astype(np.int16))
    safe_budget = archive.margin_map_float().reshape(-1).astype(np.float32) / max(
        float(archive.jacobian_lipschitz),
        1e-12,
    )
    max_safe_abs = np.floor(safe_budget + 1e-6).astype(np.int16)
    decoded_abs = np.abs(decoded.noise_levels.astype(np.int16))
    return D1OverlayDiagnostics(
        decoded_noise_pixels=int(decoded.noise_levels.size),
        decoded_noise_nonzero_pixels=int(np.count_nonzero(decoded.noise_levels)),
        decoded_noise_abs_sum=int(decoded_abs.sum()),
        camera_overlay_nonzero_pixels=int(np.count_nonzero(camera_overlay)),
        camera_overlay_abs_sum=int(
            np.abs(camera_overlay.astype(np.int16)).sum()
        ),
        attenuated_overlay_nonzero_pixels=int(np.count_nonzero(attenuated)),
        attenuated_overlay_abs_sum=int(attenuated_abs.sum()),
        active_channel_count=active_channels,
        estimated_changed_bytes_upper_bound_per_pair=int(
            np.count_nonzero(attenuated) * active_channels
        ),
        estimated_changed_lsb_l1_upper_bound_per_pair=int(
            attenuated_abs.sum() * channel_l1
        ),
        estimated_changed_lsb_l2_energy_upper_bound_per_pair=int(
            np.square(attenuated_abs, dtype=np.int64).sum() * channel_l2_energy
        ),
        integer_feasible_pixels=int(
            np.count_nonzero(safe_budget >= float(min_integer_delta_lsb))
        ),
        unsafe_nonzero_pixels=int(np.count_nonzero(decoded_abs > max_safe_abs)),
        pair_mask_active_pairs=pair_mask_active_pairs,
        max_safe_budget_lsb=float(safe_budget.max()) if safe_budget.size else 0.0,
        mean_safe_budget_lsb=float(safe_budget.mean()) if safe_budget.size else 0.0,
        payload_jacobian_lipschitz=float(decoded.jacobian_lipschitz),
        archive_jacobian_lipschitz=float(archive.jacobian_lipschitz),
        overlay_channel_policy=policy,
        overlay_amplitude_scale=scale,
        overlay_sign_policy=sign,
    )


__all__ = [
    "D1OverlayDiagnostics",
    "analyze_d1_overlay_effect",
]
