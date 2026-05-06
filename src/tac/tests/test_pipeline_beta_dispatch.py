"""Integration test for PARADIGM-β dispatch path in step_compress_weights.

Wired 2026-05-06 (commit 107f6fea). This test exercises the full β path:
1. Build a tiny renderer model
2. Synthesize per-conv sensitivity vectors
3. Save them via tac.sensitivity_map.save_sensitivity_map
4. Save the renderer checkpoint
5. Call step_compress_weights with cfg.use_sensitivity_weighted=True and a
   valid sensitivity_map_path
6. Verify the dispatch produces a non-empty archive at the expected path
7. Verify the .done marker records ``mode: owv3_sensitivity_weighted``

If a future change breaks the wiring (renames encode_owv3_archive, changes
the sensitivity-map format, swaps build_renderer kwargs) this test fails.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from experiments.pipeline import PipelineConfig, step_compress_weights
from tac.renderer import build_renderer
from tac.sensitivity_map import save_sensitivity_map


def _build_tiny_model_and_save_checkpoint(ckpt_dir: Path) -> tuple[torch.nn.Module, Path]:
    """Build a tiny renderer (matches PipelineConfig defaults) and save a
    checkpoint compatible with the standard load_state_dict path."""
    model = build_renderer(
        base_ch=36,
        mid_ch=60,
        motion_hidden=32,
        depth=1,
        pose_dim=6,
        embed_dim=6,
        use_dsconv=False,
        padding_mode="replicate",
        use_dilation=False,
        use_zoom_flow=False,
    )
    model.eval()
    ckpt_path = ckpt_dir / "renderer.pt"
    torch.save(model.state_dict(), ckpt_path)
    return model, ckpt_path


def _save_synthetic_sensitivity_map(
    model: torch.nn.Module, sens_path: Path
) -> dict[str, torch.Tensor]:
    """Synthesize a per-conv sensitivity dict and save it via the canonical
    saver. Mirrors the GPU sensitivity-sweep output format."""
    sensitivities: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            n_out = mod.out_channels
            # Synthesize a plausible sensitivity vector with the right shape.
            # Real sensitivities come from Fisher-diagonal proxy; here we
            # just need shape correctness.
            sensitivities[name] = torch.linspace(1e-4, 1e-2, n_out, dtype=torch.float32)
    save_sensitivity_map(
        sens_path,
        sensitivities,
        metadata={
            "source": "test_pipeline_beta_dispatch",
            "synthetic": True,
        },
    )
    return sensitivities


def test_beta_dispatch_with_valid_artifact_produces_archive(tmp_path: Path) -> None:
    """End-to-end: cfg.use_sensitivity_weighted=True + valid sensitivity_map_path
    → step_compress_weights writes an OWV3 archive at the expected path."""
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    model, ckpt_path = _build_tiny_model_and_save_checkpoint(ckpt_dir)

    sens_path = tmp_path / "sensitivity_map.pt"
    _save_synthetic_sensitivity_map(model, sens_path)

    output_dir = tmp_path / "results"
    cfg = PipelineConfig(
        output_dir=str(output_dir),
        use_sensitivity_weighted=True,
        sensitivity_map_path=str(sens_path),
        owv3_bit_budget_ratio=0.7,
        owv3_protect_threshold=1e-3,
    )

    archive_path = step_compress_weights(cfg, ckpt_path, iteration=0)

    assert archive_path.exists(), (
        f"β dispatch did not produce an archive at {archive_path}"
    )
    assert archive_path.name == "renderer_owv3_sensitivity.bin", (
        f"β archive filename drift: expected renderer_owv3_sensitivity.bin, "
        f"got {archive_path.name!r}"
    )
    assert archive_path.stat().st_size > 0, "β archive is empty"

    # Verify .done marker records the right mode.
    done_path = output_dir / "iter_0" / ".done_compress_weights"
    assert done_path.exists(), "β dispatch did not write .done marker"
    metadata = json.loads(done_path.read_text())
    assert metadata.get("mode") == "owv3_sensitivity_weighted", (
        f"β .done marker mode drift: expected 'owv3_sensitivity_weighted', "
        f"got {metadata.get('mode')!r}"
    )
    assert metadata.get("archive_bytes") == archive_path.stat().st_size


def test_beta_dispatch_falls_through_when_path_missing(
    tmp_path: Path, capsys
) -> None:
    """If cfg.use_sensitivity_weighted=True but sensitivity_map_path is empty
    or non-existent, the dispatch must WARN-and-fall-through to the
    cfg.weight_compression mode (not raise, not silently produce wrong
    archive)."""
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    _model, ckpt_path = _build_tiny_model_and_save_checkpoint(ckpt_dir)

    output_dir = tmp_path / "results"
    cfg = PipelineConfig(
        output_dir=str(output_dir),
        use_sensitivity_weighted=True,
        sensitivity_map_path="",  # empty
        weight_compression="fp4",  # known-wired fallback
    )
    # The β dispatch should fall through to the fp4 path which then attempts
    # to call _infer_asymmetric_config and friends. We don't need that to
    # succeed end-to-end for this test — we just need to confirm the β
    # branch did NOT execute (no β archive at the expected path).
    try:
        step_compress_weights(cfg, ckpt_path, iteration=0)
    except Exception:
        # Downstream fp4 path may fail on this synthetic model; that's fine.
        # The contract is: β didn't run because path was missing.
        pass

    beta_archive = output_dir / "iter_0" / "renderer_owv3_sensitivity.bin"
    assert not beta_archive.exists(), (
        f"β dispatch ran despite missing sensitivity_map_path; "
        f"archive should not exist at {beta_archive}"
    )
    captured = capsys.readouterr()
    # The WARN message names the empty path
    assert "sensitivity_map_path" in (captured.out + captured.err)


def test_beta_dispatch_falls_through_when_path_nonexistent(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    _model, ckpt_path = _build_tiny_model_and_save_checkpoint(ckpt_dir)

    output_dir = tmp_path / "results"
    cfg = PipelineConfig(
        output_dir=str(output_dir),
        use_sensitivity_weighted=True,
        sensitivity_map_path=str(tmp_path / "this_does_not_exist.pt"),
        weight_compression="fp4",
    )
    try:
        step_compress_weights(cfg, ckpt_path, iteration=0)
    except Exception:
        pass

    beta_archive = output_dir / "iter_0" / "renderer_owv3_sensitivity.bin"
    assert not beta_archive.exists(), (
        f"β dispatch ran despite non-existent sensitivity_map_path"
    )
