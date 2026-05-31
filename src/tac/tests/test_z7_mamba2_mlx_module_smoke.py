# SPDX-License-Identifier: MIT
"""Regression tests for the Z7-Mamba-2 trainable MLX module smoke surface."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="Z7-Mamba-2 MLX module smoke requires mlx on Apple Silicon",
)


def test_z7_mamba2_mlx_module_forward_shape_and_false_authority() -> None:
    """The trainable MLX module reconstructs pairs but carries no score authority."""
    import mlx.core as mx

    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_module import (
        MLX_EVIDENCE_GRADE,
        SCHEMA_VERSION,
        Z7Mamba2MLXModule,
    )
    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_native import (
        Z7Mamba2MLXRenderConfig,
    )

    cfg = Z7Mamba2MLXRenderConfig(num_pairs=2)
    model = Z7Mamba2MLXModule(cfg, seed=0)
    rgb_0, rgb_1, latents = model(np.array([0, 1], dtype=np.int64))
    mx.eval(rgb_0, rgb_1, latents)

    assert SCHEMA_VERSION == "z7_mamba2_mlx_module_v1_20260528"
    assert MLX_EVIDENCE_GRADE == "[macOS-MLX research-signal]"
    assert tuple(int(x) for x in rgb_0.shape) == (2, 3, 384, 512)
    assert tuple(int(x) for x in rgb_1.shape) == (2, 3, 384, 512)
    assert tuple(int(x) for x in latents.shape) == (2, 24)
    assert model.num_parameters() > 0


def test_z7_mamba2_mlx_smoke_manifest_uses_trainable_module(tmp_path: Path) -> None:
    """The operator smoke exercises Z7Mamba2MLXModule, not the old native-only renderer."""
    output_dir = tmp_path / "z7_mlx_smoke"
    resolved = str(output_dir.resolve())
    if resolved.startswith(("/tmp/", "/private/tmp/")):
        pytest.skip("smoke CLI intentionally refuses persisted /tmp artifacts")

    from experiments.train_substrate_time_traveler_l5_z7_mamba2_mlx_local import (
        _build_parser,
        _smoke_main,
    )

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--num-pairs",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert _smoke_main(args) == 0

    manifest = json.loads((output_dir / "smoke_manifest.json").read_text())
    assert manifest["renderer_module"].endswith(".mlx_module")
    assert manifest["renderer_num_parameters"] > 0
    assert manifest["score_claim"] is False
    assert manifest["promotable"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "BLOCKED pending" not in manifest["canonical_provenance"]["rationale"]
    assert manifest["forward_smoke"]["latents_shape"] == [2, 24]
