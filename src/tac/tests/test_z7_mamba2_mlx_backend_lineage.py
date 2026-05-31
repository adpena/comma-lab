# SPDX-License-Identifier: MIT
"""Fail-closed backend-lineage tests for Z7-Mamba-2 MLX artifacts."""

from __future__ import annotations

from tac.substrates.time_traveler_l5_z7_mamba2.archive_candidate import (
    z7_mamba2_meta_from_config,
)
from tac.substrates.time_traveler_l5_z7_mamba2.mlx_native import (
    Z7Mamba2MLXRenderConfig,
)


def test_z7_mlx_config_declares_reference_s6_not_canonical_ssd() -> None:
    cfg = Z7Mamba2MLXRenderConfig(num_pairs=2)

    assert cfg.mamba2_mlx_backend_lineage == "reference_s6_mlx"
    assert cfg.canonical_ssd_mlx_backend_wired is False
    assert cfg.canonical_ssd_mlx_blocker == "canonical_ssd_mlx_backend_not_wired"


def test_z7_mlx_archive_meta_carries_ssd_claim_blocker() -> None:
    meta = z7_mamba2_meta_from_config(Z7Mamba2MLXRenderConfig(num_pairs=2))

    assert meta["mamba2_mlx_backend_lineage"] == "reference_s6_mlx"
    assert meta["canonical_ssd_mlx_backend_wired"] is False
    assert meta["canonical_ssd_mlx_blocker"] == "canonical_ssd_mlx_backend_not_wired"
