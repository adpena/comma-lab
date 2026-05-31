# SPDX-License-Identifier: MIT
"""Fail-closed backend-lineage tests for Z7-Mamba-2 MLX artifacts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tac.substrates.time_traveler_l5_z7_mamba2.archive_candidate import (
    pack_archive_from_exported_state_dict,
    z7_mamba2_meta_from_config,
    z7_mamba2_pytorch_config_from_mlx,
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

    assert meta["use_canonical_ssd_mlx_backend"] is False
    assert meta["ssd_nheads"] is None
    assert meta["ssd_headdim"] is None
    assert meta["mamba2_mlx_backend_lineage"] == "reference_s6_mlx"
    assert meta["canonical_ssd_mlx_backend_wired"] is False
    assert meta["canonical_ssd_mlx_blocker"] == "canonical_ssd_mlx_backend_not_wired"


def test_z7_mlx_config_can_opt_into_canonical_ssd_with_export_blocker() -> None:
    cfg = Z7Mamba2MLXRenderConfig(
        num_pairs=2,
        d_model=8,
        expand=2,
        d_state=4,
        use_canonical_ssd_mlx_backend=True,
        ssd_nheads=2,
    )

    assert cfg.d_inner == 16
    assert cfg.effective_ssd_nheads == 2
    assert cfg.effective_ssd_headdim == 8
    assert cfg.mamba2_mlx_backend_lineage == (
        "canonical_mamba2_ssd_mlx_z7_gated_experimental"
    )
    assert cfg.canonical_ssd_mlx_backend_wired is True
    assert cfg.canonical_ssd_mlx_blocker == (
        "canonical_ssd_mlx_pytorch_bridge_export_not_wired"
    )


def test_z7_mlx_config_rejects_phantom_ssd_provenance() -> None:
    with pytest.raises(ValueError, match="requires use_canonical_ssd"):
        Z7Mamba2MLXRenderConfig(
            num_pairs=2,
            canonical_ssd_mlx_backend_wired=True,
        )

    with pytest.raises(ValueError, match="does not divide d_inner"):
        Z7Mamba2MLXRenderConfig(
            num_pairs=2,
            d_model=8,
            expand=2,
            use_canonical_ssd_mlx_backend=True,
            ssd_nheads=3,
        )


def test_z7_mlx_archive_meta_records_canonical_ssd_without_export_authority() -> None:
    cfg = Z7Mamba2MLXRenderConfig(
        num_pairs=2,
        d_model=8,
        expand=2,
        d_state=4,
        use_canonical_ssd_mlx_backend=True,
        ssd_nheads=2,
    )
    meta = z7_mamba2_meta_from_config(cfg)

    assert meta["use_canonical_ssd_mlx_backend"] is True
    assert meta["ssd_nheads"] == 2
    assert meta["ssd_headdim"] == 8
    assert meta["canonical_ssd_mlx_backend_wired"] is True
    assert meta["canonical_ssd_mlx_blocker"] == (
        "canonical_ssd_mlx_pytorch_bridge_export_not_wired"
    )

    with pytest.raises(NotImplementedError, match="pytorch_bridge_export_not_wired"):
        z7_mamba2_pytorch_config_from_mlx(cfg)
    with pytest.raises(NotImplementedError, match="legacy S6-shaped Z7MCM2"):
        pack_archive_from_exported_state_dict(exported_state_dict={}, mlx_cfg=cfg)


def test_z7_full_mlx_trainer_threads_backend_lineage_into_harness_bundle() -> None:
    trainer = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py"
    )
    tree = ast.parse(trainer.read_text(encoding="utf-8"))

    renderer_bundle_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RendererBundle"
    ]
    assert renderer_bundle_calls, "Z7 trainer must construct a RendererBundle"

    assert any(
        any(
            kw.arg == "substrate_artifact_metadata"
            and isinstance(kw.value, ast.Name)
            and kw.value.id == "z7_substrate_artifact_metadata"
            for kw in call.keywords
        )
        for call in renderer_bundle_calls
    ), "full MLX training bundle must thread backend lineage metadata"

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "z7_substrate_artifact_metadata"
            for target in node.targets
        )
    ]
    assert assignments, "Z7 metadata dict must exist next to cfg construction"
    metadata_keys = {
        key.value
        for assignment in assignments
        if isinstance(assignment.value, ast.Dict)
        for key in assignment.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert {
        "schema",
        "mamba2_mlx_backend_lineage",
        "use_canonical_ssd_mlx_backend",
        "ssd_nheads",
        "ssd_headdim",
        "canonical_ssd_mlx_backend_wired",
        "backend_claim_blockers",
        "math_fidelity_scope",
    } <= metadata_keys
    assert "ready_for_exact_eval_dispatch" not in metadata_keys
