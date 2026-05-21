# SPDX-License-Identifier: MIT
"""MLX-native Selfcomp grayscale_lut tests + canonical export round-trip.

OVERNIGHT-WW Phase 3+4 contract tests. Skipped if MLX is not available on
the host.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.portable_primitives import is_mlx_available, is_pytorch_available
from tac.substrates.grayscale_lut.architecture import (
    GrayscaleLutConfig,
    GrayscaleLutSubstrate,
)


pytestmark = pytest.mark.skipif(
    not is_mlx_available(),
    reason="MLX framework must be installed for MLX-native tests",
)


def _small_cfg() -> GrayscaleLutConfig:
    """Small cfg for fast tests: 4 pairs, 32x32 output, 2 decoder blocks."""
    return GrayscaleLutConfig(
        grayscale_downsample=4,
        decoder_hidden=8,
        decoder_blocks=2,
        embedding_dim=4,
        num_pairs=4,
        output_height=32,
        output_width=32,
        lut_bits=8,
    )


def test_mlx_native_constructs_without_mlx_skips_cleanly() -> None:
    """The pytestmark above handles the skip cleanly when MLX is missing."""
    from tac.substrates.grayscale_lut.mlx_native import GrayscaleLutMLXNative

    cfg = _small_cfg()
    sub = GrayscaleLutMLXNative(cfg)
    assert sub.cfg is cfg


def test_mlx_native_num_parameters_matches_pytorch_sister() -> None:
    """Param count exactly matches the PyTorch sister architecture."""
    from tac.substrates.grayscale_lut.mlx_native import GrayscaleLutMLXNative

    cfg = _small_cfg()
    sub_mlx = GrayscaleLutMLXNative(cfg)
    sub_pt = GrayscaleLutSubstrate(cfg)
    assert sub_mlx.num_parameters() == sub_pt.num_parameters()


def test_mlx_native_forward_produces_valid_rgb_shapes() -> None:
    """Forward returns (B, 3, H, W) tensors in [0, 1]."""
    import mlx.core as mx

    from tac.substrates.grayscale_lut.mlx_native import GrayscaleLutMLXNative

    cfg = _small_cfg()
    sub = GrayscaleLutMLXNative(cfg)

    pair_indices = mx.array(np.array([0, 1, 2], dtype=np.int32))
    rgb_0, rgb_1 = sub(pair_indices)
    mx.eval(rgb_0, rgb_1)

    assert tuple(rgb_0.shape) == (3, 3, cfg.output_height, cfg.output_width)
    assert tuple(rgb_1.shape) == (3, 3, cfg.output_height, cfg.output_width)
    rgb_0_np = np.array(rgb_0)
    rgb_1_np = np.array(rgb_1)
    assert rgb_0_np.min() >= 0.0
    assert rgb_0_np.max() <= 1.0
    assert rgb_1_np.min() >= 0.0
    assert rgb_1_np.max() <= 1.0


def test_mlx_native_export_state_dict_has_pytorch_canonical_keys() -> None:
    """Exported state_dict keys match PyTorch sister's state_dict keys exactly."""
    from tac.substrates.grayscale_lut.mlx_native import GrayscaleLutMLXNative

    cfg = _small_cfg()
    sub_mlx = GrayscaleLutMLXNative(cfg)
    sub_pt = GrayscaleLutSubstrate(cfg)

    mlx_keys = set(sub_mlx.export_state_dict().keys())
    pt_keys = set(sub_pt.state_dict().keys())
    assert mlx_keys == pt_keys, f"key mismatch: mlx={mlx_keys}, pt={pt_keys}"


def test_mlx_native_export_state_dict_shapes_match_pytorch_sister() -> None:
    """Per-tensor shapes match the PyTorch sister exactly."""
    from tac.substrates.grayscale_lut.mlx_native import GrayscaleLutMLXNative

    cfg = _small_cfg()
    sub_mlx = GrayscaleLutMLXNative(cfg)
    sub_pt = GrayscaleLutSubstrate(cfg)

    mlx_sd = sub_mlx.export_state_dict()
    pt_sd = sub_pt.state_dict()
    for name in mlx_sd:
        mlx_shape = mlx_sd[name].shape
        pt_shape = tuple(pt_sd[name].shape)
        assert mlx_shape == pt_shape, f"{name}: mlx={mlx_shape} != pt={pt_shape}"


def test_mlx_native_state_dict_roundtrip_byte_stable() -> None:
    """Export -> load -> export reproduces identical bytes."""
    from tac.substrates.grayscale_lut.mlx_native import GrayscaleLutMLXNative

    cfg = _small_cfg()
    sub = GrayscaleLutMLXNative(cfg)

    sd_before = sub.export_state_dict()
    sub.load_state_dict_from_numpy(sd_before)
    sd_after = sub.export_state_dict()

    for name in sd_before:
        np.testing.assert_allclose(
            sd_before[name], sd_after[name], atol=0, rtol=0
        )


@pytest.mark.skipif(
    not is_pytorch_available(),
    reason="PyTorch required for the MLX-trained-weights -> PyTorch-load test",
)
def test_mlx_trained_weights_load_into_pytorch_substrate() -> None:
    """MLX weights -> numpy -> PyTorch substrate loads cleanly.

    This is the structural pin for the canonical pipeline: MLX-trained
    weights must load into the PyTorch architecture for CUDA T4 eval.
    """
    import torch

    from tac.substrates.grayscale_lut.mlx_native import GrayscaleLutMLXNative

    cfg = _small_cfg()
    sub_mlx = GrayscaleLutMLXNative(cfg)
    mlx_sd_np = sub_mlx.export_state_dict()

    sub_pt = GrayscaleLutSubstrate(cfg)
    torch_sd = {
        name: torch.from_numpy(arr.astype(np.float32).copy())
        for name, arr in mlx_sd_np.items()
    }
    sub_pt.load_state_dict(torch_sd)

    # Sanity: forward pass works after loading MLX weights.
    pair_indices_pt = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0_pt, rgb_1_pt = sub_pt(pair_indices_pt)
    assert tuple(rgb_0_pt.shape) == (2, 3, cfg.output_height, cfg.output_width)
    assert tuple(rgb_1_pt.shape) == (2, 3, cfg.output_height, cfg.output_width)


@pytest.mark.skipif(
    not is_pytorch_available(),
    reason="PyTorch required for the MLX -> PyTorch numerical-equivalence test",
)
def test_mlx_and_pytorch_forward_numerically_equivalent_after_weight_transfer() -> None:
    """After MLX -> PyTorch weight transfer, both produce identical RGB output.

    This is the structural pin for the canonical train-anywhere eval-anywhere
    contract: MLX inference and PyTorch inference must agree within ε.
    """
    import mlx.core as mx
    import torch

    from tac.substrates.grayscale_lut.mlx_native import GrayscaleLutMLXNative

    cfg = _small_cfg()
    sub_mlx = GrayscaleLutMLXNative(cfg)
    mlx_sd_np = sub_mlx.export_state_dict()

    sub_pt = GrayscaleLutSubstrate(cfg)
    torch_sd = {
        name: torch.from_numpy(arr.astype(np.float32).copy())
        for name, arr in mlx_sd_np.items()
    }
    sub_pt.load_state_dict(torch_sd)

    pair_indices_np = np.array([0, 1, 2], dtype=np.int32)
    pair_indices_mlx = mx.array(pair_indices_np)
    pair_indices_pt = torch.from_numpy(pair_indices_np.astype(np.int64))

    rgb_0_mlx, rgb_1_mlx = sub_mlx(pair_indices_mlx)
    mx.eval(rgb_0_mlx, rgb_1_mlx)
    rgb_0_mlx_np = np.array(rgb_0_mlx)
    rgb_1_mlx_np = np.array(rgb_1_mlx)

    with torch.no_grad():
        rgb_0_pt, rgb_1_pt = sub_pt(pair_indices_pt)
    rgb_0_pt_np = rgb_0_pt.numpy()
    rgb_1_pt_np = rgb_1_pt.numpy()

    # Per Phase 1 PV: Metal vs CPU fp32 FMA reordering produces ~1e-3 max abs
    # diff on Conv2d forward; with stacked Conv+FiLM+GELU+sigmoid we expect
    # ε to compound to ~1e-2 worst case but typically ~1e-3.
    np.testing.assert_allclose(rgb_0_mlx_np, rgb_0_pt_np, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(rgb_1_mlx_np, rgb_1_pt_np, atol=2e-2, rtol=2e-2)


def test_export_pt_file_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    """Canonical export pipeline: MLX -> .pt -> PyTorch load."""
    import torch

    from tac.local_acceleration.mlx_to_pytorch_export import (
        export_mlx_state_dict_to_torch_pt,
        load_pytorch_state_dict_from_pt,
    )
    from tac.substrates.grayscale_lut.mlx_native import GrayscaleLutMLXNative

    cfg = _small_cfg()
    sub_mlx = GrayscaleLutMLXNative(cfg)
    sd_np = sub_mlx.export_state_dict()

    pt_path = tmp_path / "mlx_trained.pt"  # type: ignore[attr-defined]
    manifest = export_mlx_state_dict_to_torch_pt(
        sd_np,
        pt_path,
        substrate_id="grayscale_lut",
        run_id="overnight_ww_smoke",
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["training_evidence_grade"] == "macOS-MLX-research-signal"
    assert manifest["tensor_count"] == len(sd_np)
    assert len(manifest["file_sha256"]) == 64

    # Round-trip: load via canonical helper + verify shapes match.
    loaded = load_pytorch_state_dict_from_pt(pt_path)
    for name in sd_np:
        assert name in loaded
        assert tuple(loaded[name].shape) == sd_np[name].shape
        np.testing.assert_allclose(
            loaded[name].numpy(), sd_np[name].astype(np.float32), atol=0, rtol=0
        )


def test_export_refuses_overwrite_by_default(tmp_path) -> None:
    """Export refuses to overwrite an existing .pt file without explicit opt-in."""
    from tac.local_acceleration.mlx_to_pytorch_export import (
        export_mlx_state_dict_to_torch_pt,
    )

    sd_np = {"w": np.zeros((2, 2), dtype=np.float32)}
    pt_path = tmp_path / "test.pt"
    pt_path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        export_mlx_state_dict_to_torch_pt(
            sd_np, pt_path, substrate_id="x", run_id="y"
        )


def test_cuda_t4_eval_pipeline_invocation(tmp_path) -> None:
    """CUDA T4 eval pipeline returns canonical invocation structure."""
    from tac.local_acceleration.cuda_t4_eval_pipeline import (
        build_cuda_t4_eval_invocation,
    )

    archive_path = tmp_path / "archive.zip"
    archive_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal valid zip stub

    result = build_cuda_t4_eval_invocation(
        substrate_id="grayscale_lut",
        archive_path=archive_path,
        archive_sha256="a" * 64,
        expected_cost_usd=0.30,
    )

    assert result["substrate_id"] == "grayscale_lut"
    assert result["cost_band"] == "smoke"
    assert "operator_authorize.py" in result["operator_command"]
    assert "modal" in result["operator_command"]
    assert result["current_evidence_grade"] == "macOS-MLX-research-signal"
    assert "requires_cuda_t4" in " ".join(result["promotion_blockers"])
    assert len(result["next_steps"]) == 7


def test_cuda_t4_pipeline_steps_documented() -> None:
    """describe_pipeline_steps returns 6 canonical steps."""
    from tac.local_acceleration.cuda_t4_eval_pipeline import describe_pipeline_steps

    steps = describe_pipeline_steps()
    assert len(steps) == 6
    for i, s in enumerate(steps, start=1):
        assert s["step"] == str(i)
        assert s["name"]
        assert s["module"]
        assert s["cost"]
        assert s["axis"]
