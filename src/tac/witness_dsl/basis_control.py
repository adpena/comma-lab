"""Canonical runtime names for the curvelet-vs-legacy-Fourier basis A/B.

The historical ``polar_fourier`` CLI/checkpoint spelling remains readable, but
normalizes to an explicitly non-shipping regression-control identity. The
windowed-curvelet treatment remains opt-in until the governed n600 byte-closed
realized-through-R no-d_seg-regression verdict exists.
"""

from __future__ import annotations

LEGACY_FOURIER_AB_CONTROL = "legacy_fourier_ab_control"
WINDOWED_CURVELET = "windowed_curvelet"
COMPACT_SHEARLET = "compact_shearlet"
GENUINE_FRAME_FEATURE_WIDTH = 80
_LEGACY_ALIASES = frozenset(
    {"polar_fourier", "polar_directional_fourier", LEGACY_FOURIER_AB_CONTROL}
)


def normalize_basis_family(value: object) -> str:
    """Return the canonical runtime basis name while accepting the old control alias."""

    raw = str(value)
    if raw in _LEGACY_ALIASES:
        return LEGACY_FOURIER_AB_CONTROL
    if raw == WINDOWED_CURVELET:
        return WINDOWED_CURVELET
    if raw == COMPACT_SHEARLET:
        return COMPACT_SHEARLET
    raise ValueError(f"unknown basis family: {raw!r}")


def is_legacy_fourier_ab_control(value: object) -> bool:
    """Whether ``value`` selects the historical byte-compatible control computation."""

    return normalize_basis_family(value) == LEGACY_FOURIER_AB_CONTROL


def genuine_frame_windowed_curvelet_config():
    """Return the ticket-sealed 40-atom/80-column localized-curvelet frame.

    Keeping this constructor beside the runtime family names gives the trainer and
    byte-close receiver one authority for the static-equality gate. The older
    module default is a larger exploratory dictionary and is intentionally not
    launchable as the equal-value V9 treatment.
    """

    from tac.boundary_math.windowed_curvelet_frame import (
        WindowedCurveletConfig,
        n_atoms,
    )

    config = WindowedCurveletConfig(n_scales=3, n_orient0=10, n_trans=1)
    feature_width = 2 * int(n_atoms(config))
    if feature_width != GENUINE_FRAME_FEATURE_WIDTH:
        raise RuntimeError(
            "genuine-frame windowed-curvelet width drift: "
            f"expected={GENUINE_FRAME_FEATURE_WIDTH}, live={feature_width}"
        )
    return config


def genuine_frame_compact_shearlet_config():
    """Return the ticket-sealed 40-atom/80-column compact-shearlet frame."""

    from tac.boundary_math.compact_shearlet_frame import (
        CompactShearletConfig,
        n_atoms,
    )

    config = CompactShearletConfig(n_scales=4, n_shear=2, n_trans=1)
    feature_width = 2 * int(n_atoms(config))
    if feature_width != GENUINE_FRAME_FEATURE_WIDTH:
        raise RuntimeError(
            "genuine-frame compact-shearlet width drift: "
            f"expected={GENUINE_FRAME_FEATURE_WIDTH}, live={feature_width}"
        )
    return config


def genuine_frame_equal_value_budget(*, num_pairs: int, mod_dim: int = 32) -> dict[str, int]:
    """Derive the static V9 equal-value receipt without importing MLX.

    The fixed decoder shape is the launch ticket's 80 -> 96, four-hidden-layer
    trunk with five SDF outputs, three texture outputs, and a five-by-three
    palette. Per-pair codes have two rows (the two evaluated frames).
    """

    pairs = int(num_pairs)
    latent_width = int(mod_dim)
    if pairs <= 0 or latent_width <= 0:
        raise ValueError("num_pairs and mod_dim must be positive")
    hidden = 96
    n_hidden = 4
    n_classes = 5
    in_projection = (GENUINE_FRAME_FEATURE_WIDTH + 1) * hidden
    film_projection = (latent_width + 1) * (2 * hidden * n_hidden)
    hidden_stack = n_hidden * (hidden + 1) * hidden
    sdf_head = (hidden + 1) * n_classes
    texture_head = (hidden + 1) * 3
    palette = n_classes * 3
    decoder_values = (
        in_projection
        + film_projection
        + hidden_stack
        + sdf_head
        + texture_head
        + palette
    )
    per_pair_code_values = pairs * 2 * latent_width
    return {
        "feature_width": GENUINE_FRAME_FEATURE_WIDTH,
        "hidden_dim": hidden,
        "hidden_layers": n_hidden,
        "mod_dim": latent_width,
        "pair_frame_code_rows": pairs * 2,
        "in_projection": in_projection,
        "film_projection": film_projection,
        "hidden_stack": hidden_stack,
        "sdf_head": sdf_head,
        "texture_head": texture_head,
        "palette": palette,
        "decoder_values": decoder_values,
        "per_pair_code_values": per_pair_code_values,
        "total_trainable_values": decoder_values + per_pair_code_values,
    }


__all__ = [
    "COMPACT_SHEARLET",
    "GENUINE_FRAME_FEATURE_WIDTH",
    "LEGACY_FOURIER_AB_CONTROL",
    "WINDOWED_CURVELET",
    "genuine_frame_compact_shearlet_config",
    "genuine_frame_equal_value_budget",
    "genuine_frame_windowed_curvelet_config",
    "is_legacy_fourier_ab_control",
    "normalize_basis_family",
]
