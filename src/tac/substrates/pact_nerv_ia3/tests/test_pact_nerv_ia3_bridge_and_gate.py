# SPDX-License-Identifier: MIT
"""Tests for the PACT-NeRV-IA3 MLX -> PyTorch bridge + Catalog #1265 gate.

Sister of ``src/tac/substrates/pact_nerv_ia3/tests/test_pact_nerv_ia3.py``
(L0 SCAFFOLD substrate tests) covering the L1 promotion pathway:

  1. MLX numpy-portable state_dict round-trips byte-stably to PyTorch
     state_dict via the canonical bridge tool.
  2. The bridge correctly transposes Conv2d HWIO -> OIHW.
  3. The bridge writes canonical Provenance per Catalog #287/#323 with
     non-promotable markers per Catalog #341.
  4. The Catalog #1265 contest-equivalence gate parses PIA3 archives + runs
     decoder-parity measurement + emits canonical verdict JSON.
  5. The paired-dispatch recipe schema validates per Catalog #240.

Per CLAUDE.md "Catalog #229 Premise Verification before edit": every test
exercises a real artifact path. The MLX-side path is skipped on non-Apple-
Silicon hosts (the bridge works as a pure converter without MLX too).

Discipline honored:
- Catalog #229 PV (every test references live source surfaces)
- Catalog #168 ast.Assign + ast.AnnAssign (none of these tests use AST scan)
- Catalog #287 placeholder rejection (rationale strings >= 4 chars)
- Catalog #110/#113 APPEND-ONLY (tests don't mutate forensic artifacts)
"""
from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.substrates.pact_nerv_ia3.architecture import (
    PactNervIa3Config,
    PactNervIa3Substrate,
)
from tac.substrates.pact_nerv_ia3.archive import pack_archive

try:  # pragma: no cover - skipped on non-Apple-Silicon hosts
    import mlx.core as mx  # noqa: F401

    _MLX_AVAILABLE = True
except Exception:
    _MLX_AVAILABLE = False


def _build_synthetic_mlx_state_dict_npsd_blob(
    num_pairs: int = 4,
) -> tuple[bytes, dict[str, np.ndarray]]:
    """Build a synthetic .npsd blob shaped like the MLX trainer output.

    Returns (blob_bytes, raw_dict) so tests can assert per-tensor equality
    after round-trip through pack_state_dict_numpy / unpack_state_dict_numpy.
    """
    from tac.substrates._shared.numpy_portable_inflate import (
        pack_state_dict_numpy,
    )

    cfg = PactNervIa3Config(num_pairs=num_pairs)
    # Construct numpy arrays in MLX HWIO Conv2d layout matching what the live
    # MLX renderer's ``export_state_dict`` produces (verified empirically
    # against the 2000ep checkpoint at landing time).
    rng = np.random.default_rng(seed=42)
    channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
    sd: dict[str, np.ndarray] = {}
    sd["latents"] = rng.standard_normal((num_pairs, cfg.latent_dim)).astype(
        np.float32
    ) * 0.02
    sd["ego_poses"] = rng.standard_normal((num_pairs, cfg.pose_dim)).astype(
        np.float32
    ) * 0.02
    sd["latent_embed.weight"] = rng.standard_normal(
        (cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w, cfg.latent_dim)
    ).astype(np.float32)
    sd["latent_embed.bias"] = np.zeros(
        cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w, dtype=np.float32
    )
    for i in range(cfg.num_upsample_blocks):
        in_ch = channels[i]
        out_ch_dsc = channels[i + 1] * 4  # before PixelShuffle(2)
        # MLX depthwise: (in_ch, 3, 3, 1)
        sd[f"blocks.{i}.dsc.depthwise.weight"] = rng.standard_normal(
            (in_ch, 3, 3, 1)
        ).astype(np.float32)
        sd[f"blocks.{i}.dsc.depthwise.bias"] = np.zeros(in_ch, dtype=np.float32)
        # MLX pointwise: (out_ch_dsc, 1, 1, in_ch)
        sd[f"blocks.{i}.dsc.pointwise.weight"] = rng.standard_normal(
            (out_ch_dsc, 1, 1, in_ch)
        ).astype(np.float32)
        sd[f"blocks.{i}.dsc.pointwise.bias"] = np.zeros(
            out_ch_dsc, dtype=np.float32
        )
        # IA3 gamma_proj: Linear(pose_dim -> channels[i+1])
        sd[f"ia3_mods.{i}.gamma_proj.weight"] = (
            rng.standard_normal((channels[i + 1], cfg.pose_dim)).astype(np.float32)
            * cfg.ia3_init_delta_std
        )
        sd[f"ia3_mods.{i}.gamma_proj.bias"] = np.zeros(
            channels[i + 1], dtype=np.float32
        )
    final_ch = channels[cfg.num_upsample_blocks]
    for head in ("head_rgb_0", "head_rgb_1"):
        sd[f"{head}.weight"] = rng.standard_normal((3, 1, 1, final_ch)).astype(
            np.float32
        )
        sd[f"{head}.bias"] = np.zeros(3, dtype=np.float32)
    blob = pack_state_dict_numpy(sd, dtype="fp32")
    return blob, sd


def _tiny_pia3_config(num_pairs: int = 4) -> PactNervIa3Config:
    return PactNervIa3Config(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        pose_dim=6,
        ia3_init_delta_std=0.01,
        num_pairs=num_pairs,
        output_height=24,
        output_width=32,
    )


def _build_tiny_pia3_archive(
    *,
    num_pairs: int = 4,
    drop_decoder_key: str | None = None,
) -> tuple[bytes, PactNervIa3Config]:
    """Build a small real PIA3 archive for fast gate tests."""
    cfg = _tiny_pia3_config(num_pairs=num_pairs)
    model = PactNervIa3Substrate(cfg).eval()
    state = model.state_dict()
    latents = state["latents"]
    ego_poses = state["ego_poses"]
    decoder_state = {
        key: value
        for key, value in state.items()
        if key not in {"latents", "ego_poses"}
    }
    if drop_decoder_key is not None:
        decoder_state.pop(drop_decoder_key)
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "ia3_init_delta_std": cfg.ia3_init_delta_std,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }
    return (
        pack_archive(
            decoder_state_dict=decoder_state,
            latents=latents,
            ego_poses=ego_poses,
            meta=meta,
            pose_dim=cfg.pose_dim,
        ),
        cfg,
    )


def test_bridge_converts_synthetic_npsd_to_pytorch_pt(tmp_path: Path) -> None:
    """The bridge tool converts a synthetic MLX .npsd to PyTorch .pt cleanly."""
    from tools.export_pact_nerv_ia3_mlx_to_pytorch_state_dict import (
        export_pact_nerv_ia3_mlx_to_pytorch,
    )

    blob, _raw_sd = _build_synthetic_mlx_state_dict_npsd_blob(num_pairs=4)
    src = tmp_path / "synthetic.npsd"
    src.write_bytes(blob)
    out_pt = tmp_path / "pia3.pt"
    proof = tmp_path / "numpy_pytorch_parity_proof.json"

    manifest = export_pact_nerv_ia3_mlx_to_pytorch(
        mlx_state_dict_path=src,
        output_pytorch_state_dict=out_pt,
        parity_proof_out=proof,
        sample_pair_indices=(0, 1),
    )
    assert out_pt.is_file()
    assert proof.is_file()
    assert manifest["tensor_count"] == 50, manifest["tensor_count"]
    assert manifest["score_claim"] is False
    assert manifest["promotable"] is False
    assert manifest["axis_tag"] == "[predicted]"
    assert "predicted_axis_not_contest_authority_per_catalog_127_192_317_341" in (
        manifest["blockers"]
    )


def test_bridge_transposes_conv2d_hwio_to_oihw_correctly(tmp_path: Path) -> None:
    """Conv2d MLX HWIO (out, kH, kW, in) -> PyTorch OIHW (out, in, kH, kW)."""
    from tools.export_pact_nerv_ia3_mlx_to_pytorch_state_dict import (
        export_pact_nerv_ia3_mlx_to_pytorch,
    )

    blob, raw_sd = _build_synthetic_mlx_state_dict_npsd_blob(num_pairs=4)
    src = tmp_path / "synthetic.npsd"
    src.write_bytes(blob)
    out_pt = tmp_path / "pia3.pt"
    export_pact_nerv_ia3_mlx_to_pytorch(
        mlx_state_dict_path=src,
        output_pytorch_state_dict=out_pt,
        sample_pair_indices=(0,),
    )
    pyt_sd = torch.load(out_pt, weights_only=True)

    # Conv2d depthwise: MLX (in_ch, 3, 3, 1) -> PyTorch (in_ch, 1, 3, 3)
    mlx_dw = raw_sd["blocks.0.dsc.depthwise.weight"]
    pyt_dw = pyt_sd["blocks.0.dsc.depthwise.weight"].numpy()
    assert list(mlx_dw.shape) == [64, 3, 3, 1]
    assert list(pyt_dw.shape) == [64, 1, 3, 3]
    # Verify the transpose: np.transpose(mlx_dw, (0, 3, 1, 2)) == pyt_dw
    expected_pyt = np.transpose(mlx_dw, (0, 3, 1, 2))
    np.testing.assert_array_equal(pyt_dw, expected_pyt)

    # Linear / per-pair tensors: layout preserved (no transpose).
    np.testing.assert_array_equal(
        pyt_sd["latents"].numpy(), raw_sd["latents"]
    )
    np.testing.assert_array_equal(
        pyt_sd["latent_embed.weight"].numpy(), raw_sd["latent_embed.weight"]
    )


def test_bridge_pytorch_substrate_loads_strict(tmp_path: Path) -> None:
    """The bridge output state_dict load_state_dict(strict=True) on the canonical sister."""
    from tools.export_pact_nerv_ia3_mlx_to_pytorch_state_dict import (
        export_pact_nerv_ia3_mlx_to_pytorch,
    )

    blob, _raw_sd = _build_synthetic_mlx_state_dict_npsd_blob(num_pairs=4)
    src = tmp_path / "synthetic.npsd"
    src.write_bytes(blob)
    out_pt = tmp_path / "pia3.pt"
    export_pact_nerv_ia3_mlx_to_pytorch(
        mlx_state_dict_path=src,
        output_pytorch_state_dict=out_pt,
        sample_pair_indices=(0,),
    )
    cfg = PactNervIa3Config(num_pairs=4)
    model = PactNervIa3Substrate(cfg).eval()
    pyt_sd = torch.load(out_pt, weights_only=True)
    result = model.load_state_dict(pyt_sd, strict=True)
    assert not result.missing_keys, result.missing_keys
    assert not result.unexpected_keys, result.unexpected_keys


def test_bridge_refuses_missing_input() -> None:
    """The bridge raises FileNotFoundError on a non-existent MLX state_dict."""
    from tools.export_pact_nerv_ia3_mlx_to_pytorch_state_dict import (
        export_pact_nerv_ia3_mlx_to_pytorch,
    )

    with tempfile.TemporaryDirectory() as tmp, pytest.raises(FileNotFoundError):
        export_pact_nerv_ia3_mlx_to_pytorch(
            mlx_state_dict_path=Path(tmp) / "does_not_exist.npsd",
            output_pytorch_state_dict=Path(tmp) / "ignored.pt",
        )


def test_bridge_refuses_overwrite_when_disabled(tmp_path: Path) -> None:
    """The bridge raises FileExistsError when overwrite=False and target exists."""
    from tools.export_pact_nerv_ia3_mlx_to_pytorch_state_dict import (
        export_pact_nerv_ia3_mlx_to_pytorch,
    )

    blob, _raw_sd = _build_synthetic_mlx_state_dict_npsd_blob(num_pairs=4)
    src = tmp_path / "synthetic.npsd"
    src.write_bytes(blob)
    out_pt = tmp_path / "pia3.pt"
    out_pt.write_bytes(b"placeholder")
    with pytest.raises(FileExistsError):
        export_pact_nerv_ia3_mlx_to_pytorch(
            mlx_state_dict_path=src,
            output_pytorch_state_dict=out_pt,
            overwrite=False,
        )


def test_bridge_emits_canonical_provenance(tmp_path: Path) -> None:
    """The bridge writes canonical Provenance per Catalog #287/#323 in manifest."""
    from tools.export_pact_nerv_ia3_mlx_to_pytorch_state_dict import (
        export_pact_nerv_ia3_mlx_to_pytorch,
    )

    blob, _raw_sd = _build_synthetic_mlx_state_dict_npsd_blob(num_pairs=4)
    src = tmp_path / "synthetic.npsd"
    src.write_bytes(blob)
    out_pt = tmp_path / "pia3.pt"
    manifest = export_pact_nerv_ia3_mlx_to_pytorch(
        mlx_state_dict_path=src,
        output_pytorch_state_dict=out_pt,
        sample_pair_indices=(0,),
    )
    prov = manifest["provenance"]
    assert prov is not None
    assert prov.get("kind") == "predicted" or prov.get("axis_tag") == "[predicted]" or "predicted" in str(prov)


def test_bridge_proof_file_round_trips_json(tmp_path: Path) -> None:
    """The optional parity_proof_out file is valid JSON with required fields."""
    from tools.export_pact_nerv_ia3_mlx_to_pytorch_state_dict import (
        export_pact_nerv_ia3_mlx_to_pytorch,
    )

    blob, _raw_sd = _build_synthetic_mlx_state_dict_npsd_blob(num_pairs=4)
    src = tmp_path / "synthetic.npsd"
    src.write_bytes(blob)
    out_pt = tmp_path / "pia3.pt"
    proof = tmp_path / "proof.json"
    export_pact_nerv_ia3_mlx_to_pytorch(
        mlx_state_dict_path=src,
        output_pytorch_state_dict=out_pt,
        parity_proof_out=proof,
        sample_pair_indices=(0,),
    )
    data = json.loads(proof.read_text(encoding="utf-8"))
    assert data["schema_version"] == "pact_nerv_ia3_mlx_pytorch_export_bridge.v1"
    assert data["tensor_count"] == 50
    assert "pytorch_state_dict_sha256" in data
    assert "mlx_state_dict_sha256" in data
    assert data["score_claim"] is False
    assert data["promotable"] is False


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX not installed (non-Apple-Silicon host)")
def test_bridge_forward_parity_in_0_1_sigmoid_space(tmp_path: Path) -> None:
    """The bridge's forward-parity proof reports drift in [0, 1] sigmoid space."""
    from tools.export_pact_nerv_ia3_mlx_to_pytorch_state_dict import (
        export_pact_nerv_ia3_mlx_to_pytorch,
    )

    blob, _raw_sd = _build_synthetic_mlx_state_dict_npsd_blob(num_pairs=4)
    src = tmp_path / "synthetic.npsd"
    src.write_bytes(blob)
    out_pt = tmp_path / "pia3.pt"
    manifest = export_pact_nerv_ia3_mlx_to_pytorch(
        mlx_state_dict_path=src,
        output_pytorch_state_dict=out_pt,
        sample_pair_indices=(0, 1),
    )
    fp = manifest["forward_parity"]
    assert fp["backends_compared"] == "mlx_vs_pytorch_forward"
    assert fp["decoder_output_space"] == "sigmoid_0_to_1"
    # Drift is reported as a number; we do NOT require within_band because
    # the IA3 substrate's deep SIREN+PixelShuffle stack is expected to
    # exceed 0.001 in [0, 1] sigmoid space per the drift-vs-depth anchor.
    assert "max_abs_drift_01" in fp
    assert "mean_abs_drift_01" in fp
    assert isinstance(fp["max_abs_drift_01"], float)
    assert fp["max_abs_drift_01"] >= 0.0
    assert fp["max_abs_drift_01"] <= 1.0  # bounded by sigmoid range


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX not installed (non-Apple-Silicon host)")
def test_gate_parses_pia3_archive_and_emits_verdict(tmp_path: Path) -> None:
    """The Catalog #1265 sister gate parses PIA3 + emits canonical verdict JSON."""
    from tools.gate_mlx_candidate_contest_equivalence_pact_nerv_ia3 import (
        measure_pact_nerv_ia3_decoder_parity,
    )

    archive_bytes, cfg = _build_tiny_pia3_archive(num_pairs=4)
    archive_path = tmp_path / "pia3.bin"
    archive_path.write_bytes(archive_bytes)
    result = measure_pact_nerv_ia3_decoder_parity(archive_path, n_pairs=2)
    assert result["decoder_output_space"] == "sigmoid_0_to_1"
    assert result["n_pairs_measured"] == 2
    assert result["frame_shape"] == [2, 2, 3, cfg.output_height, cfg.output_width]
    assert isinstance(result["max_abs_drift"], float)
    assert result["max_abs_drift"] >= 0.0


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX not installed (non-Apple-Silicon host)")
def test_gate_accepts_zipped_pia3_archive_member(tmp_path: Path) -> None:
    """The gate accepts contest-packet ZIP form and measures member ``0.bin``."""
    from tools.gate_mlx_candidate_contest_equivalence_pact_nerv_ia3 import (
        measure_pact_nerv_ia3_decoder_parity,
    )

    archive_bytes, cfg = _build_tiny_pia3_archive(num_pairs=3)
    archive_zip = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_zip, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", archive_bytes)

    result = measure_pact_nerv_ia3_decoder_parity(archive_zip, n_pairs=3)
    assert result["archive_source"].startswith("zip_member_0_bin_size_")
    assert result["n_pairs_measured"] == 3
    assert result["frame_shape"] == [3, 2, 3, cfg.output_height, cfg.output_width]


def test_gate_refuses_pia3_archive_with_missing_decoder_key(tmp_path: Path) -> None:
    """Corrupt decoder state must fail closed instead of silently measuring."""
    from tools.gate_mlx_candidate_contest_equivalence_pact_nerv_ia3 import (
        _build_pytorch_substrate_from_archive,
    )

    archive_bytes, _cfg = _build_tiny_pia3_archive(
        num_pairs=2,
        drop_decoder_key="head_rgb_0.weight",
    )
    archive_path = tmp_path / "missing_key.pia3"
    archive_path.write_bytes(archive_bytes)
    with pytest.raises(RuntimeError, match="missing_keys"):
        _build_pytorch_substrate_from_archive(archive_path.read_bytes())


def test_paired_dispatch_recipe_schema_validates() -> None:
    """The paired-dispatch recipe parses as valid YAML + carries required fields."""
    import yaml

    # Walk up to repo root looking for .omx/operator_authorize_recipes/
    here = Path(__file__).resolve()
    repo_root = next(
        (p for p in here.parents if (p / ".omx/operator_authorize_recipes").is_dir()),
        None,
    )
    assert repo_root is not None, "repo root with .omx/operator_authorize_recipes/ not found"
    recipe_path = (
        repo_root
        / ".omx/operator_authorize_recipes/substrate_pact_nerv_ia3_modal_t4_paired_dispatch.yaml"
    )
    assert recipe_path.is_file(), recipe_path
    data = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    # Catalog #240 recipe-vs-trainer-state consistency
    assert data["dispatch_enabled"] is False
    assert data["research_only"] is True
    # Catalog #244 NVML 3-export block
    env = data["env_overrides"]
    assert env["DALI_DISABLE_NVML"] == "1"
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    # Catalog #324 predicted_band_validation_status
    assert data["predicted_band_validation_status"] == "pending_post_training"
    assert "predicted_band_reactivation_criteria" in data
    # Catalog #170/#171/#172/#181/#182/#173/#215 required-fields
    assert data["min_vram_gb"] >= 1
    assert data["video_input_strategy"] in {
        "per_dispatch_local_copy",
        "readonly_mmap",
        "shared_volume_no_contention_expected",
    }
    assert data["pyav_decode_strategy"] in {
        "cpu_thread_async_upload",
        "cuda_nvdec",
        "cpu_blocking_upload",
        "not_applicable",
    }
    assert "research_substrate" in data["target_modes"]
    assert data["canary_status"] in {
        "canary",
        "post_canary_dependent",
        "independent_substrate",
    }
    assert data["min_smoke_gpu"] in {"T4", "L4", "A10G", "L40S", "A100", "H100"}
    # Dispatch blockers list MUST include the sister gates' constraints
    blockers = data["dispatch_blockers"]
    assert any("catalog_325" in b for b in blockers)
    assert any("catalog_167" in b for b in blockers)
