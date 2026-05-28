# SPDX-License-Identifier: MIT
"""Tests for PACT-NeRV-SELECTOR-V2 MLX renderer + MLX→PyTorch bridge.

Per the canonical IA3 sister test pattern (commit ``bbf11079d``;
``src/tac/substrates/pact_nerv_ia3/tests/test_pact_nerv_ia3.py`` +
``test_pact_nerv_ia3_bridge_and_gate.py``). Tests are skipped on non-MLX
hosts via the canonical pytest skipmark.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import mlx.core as mx  # noqa: F401
    import mlx.nn  # noqa: F401

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

skip_no_mlx = pytest.mark.skipif(
    not _MLX_AVAILABLE,
    reason="MLX not available on this host; PACT-NeRV-SELECTOR-V2 MLX tests "
    "require Apple Silicon with the ``mlx`` package installed.",
)


@skip_no_mlx
def test_mlx_renderer_imports_clean() -> None:
    """Catalog #229 PV — verify imports + canonical schema constants."""
    from tac.substrates.pact_nerv_selector_v2.mlx_renderer import (
        MLX_EVIDENCE_GRADE,
        SCHEMA_VERSION,
        PactNervSelectorV2SubstrateMLX,
    )

    assert SCHEMA_VERSION == "pact_nerv_selector_v2_mlx_renderer_v1"
    assert MLX_EVIDENCE_GRADE == "[macOS-MLX research-signal]"
    assert PactNervSelectorV2SubstrateMLX is not None


@skip_no_mlx
def test_mlx_renderer_parameter_parity_with_pytorch() -> None:
    """Verify the MLX renderer parameter count matches PyTorch sister."""
    from tac.substrates.pact_nerv_selector_v2.architecture import (
        PactNervSelectorV2Config,
        PactNervSelectorV2Substrate,
    )
    from tac.substrates.pact_nerv_selector_v2.mlx_renderer import (
        PactNervSelectorV2SubstrateMLX,
    )

    cfg = PactNervSelectorV2Config(num_pairs=4)
    torch_model = PactNervSelectorV2Substrate(cfg)
    mlx_model = PactNervSelectorV2SubstrateMLX(cfg)
    # Exact match required at init — same fan-in counts + per-pair latent layout.
    assert torch_model.num_parameters() == mlx_model.num_parameters()


@skip_no_mlx
def test_mlx_renderer_forward_shape_b2chw_255() -> None:
    """Verify the call_b2chw_255 forward convention shape contract."""
    import mlx.core as mx

    from tac.substrates.pact_nerv_selector_v2.architecture import (
        PactNervSelectorV2Config,
    )
    from tac.substrates.pact_nerv_selector_v2.mlx_renderer import (
        PactNervSelectorV2SubstrateMLX,
    )

    cfg = PactNervSelectorV2Config(num_pairs=4)
    model = PactNervSelectorV2SubstrateMLX(cfg)
    idx = mx.array([0, 1, 2, 3], dtype=mx.int32)
    output = model(idx)
    mx.eval(output)
    # canonical call_b2chw_255 convention: (B, 2, 3, H, W) in [0, 255].
    expected = (4, 2, 3, cfg.output_height, cfg.output_width)
    assert tuple(int(s) for s in output.shape) == expected
    assert float(mx.min(output)) >= 0.0
    assert float(mx.max(output)) <= 255.0


@skip_no_mlx
def test_mlx_renderer_export_state_dict_shape_layout() -> None:
    """Verify export_state_dict produces PyTorch-layout numpy arrays."""
    from tac.substrates.pact_nerv_selector_v2.architecture import (
        PactNervSelectorV2Config,
    )
    from tac.substrates.pact_nerv_selector_v2.mlx_renderer import (
        PactNervSelectorV2SubstrateMLX,
    )

    cfg = PactNervSelectorV2Config(num_pairs=4)
    model = PactNervSelectorV2SubstrateMLX(cfg)
    sd = model.export_state_dict()
    # latents: per-pair (num_pairs, latent_dim)
    assert sd["latents"].shape == (cfg.num_pairs, cfg.latent_dim)
    # latent_embed: Linear(latent_dim -> embed_dim * H0 * W0)
    expected_embed_out = cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w
    assert sd["latent_embed.weight"].shape == (expected_embed_out, cfg.latent_dim)
    # NO ego_poses in SELECTOR-V2 state_dict (vs IA3 sister).
    assert "ego_poses" not in sd
    # NO ia3_mods in SELECTOR-V2 state_dict (vs IA3 sister).
    assert not any(k.startswith("ia3_mods") for k in sd)
    # Conv2d weights in PyTorch OIHW layout.
    # blocks.0.dsc.depthwise.weight: depthwise (in_ch=embed_dim, out=in_ch, kH=3, kW=3, groups=in_ch)
    # PyTorch nn.Conv2d depthwise weight shape: (in_ch, 1, kH, kW) [groups=in_ch makes in_ch//groups = 1]
    assert sd["blocks.0.dsc.depthwise.weight"].shape == (cfg.embed_dim, 1, 3, 3)
    # RGB heads: 1x1 conv (3, final_ch, 1, 1).
    final_ch = cfg.decoder_channels[cfg.num_upsample_blocks - 1]
    assert sd["head_rgb_0.weight"].shape == (3, final_ch, 1, 1)
    assert sd["head_rgb_1.weight"].shape == (3, final_ch, 1, 1)


@skip_no_mlx
def test_mlx_renderer_selectors_buffer_validation() -> None:
    """Verify the selectors buffer mirrors PyTorch register_buffer semantics."""
    import numpy as np

    from tac.substrates.pact_nerv_selector_v2.architecture import (
        PactNervSelectorV2Config,
    )
    from tac.substrates.pact_nerv_selector_v2.mlx_renderer import (
        PactNervSelectorV2SubstrateMLX,
    )

    cfg = PactNervSelectorV2Config(num_pairs=8)
    model = PactNervSelectorV2SubstrateMLX(cfg)
    # Default: zeros (matches PyTorch sister register_buffer default).
    assert model.selectors.shape == (cfg.num_pairs,)
    assert model.selectors.dtype == np.int64
    assert int(model.selectors.sum()) == 0
    # set_selectors honors palette size + length + dtype.
    valid = np.array([0, 3, 7, 15, 2, 14, 8, 1], dtype=np.int64)
    model.set_selectors(valid)
    assert np.array_equal(model.selectors, valid)
    # Length mismatch raises.
    with pytest.raises(ValueError, match="length"):
        model.set_selectors(np.array([0, 1, 2], dtype=np.int64))
    # Palette range violation raises.
    with pytest.raises(ValueError, match=r"\[0, 16\)"):
        model.set_selectors(np.array([0, 0, 0, 0, 0, 0, 0, 16], dtype=np.int64))
    # Non-integer dtype raises.
    with pytest.raises(ValueError, match="integer"):
        model.set_selectors(np.zeros(cfg.num_pairs, dtype=np.float32))


@skip_no_mlx
def test_bridge_tool_imports_clean() -> None:
    """Catalog #229 PV — verify the bridge tool imports + schema constant."""
    from tools.export_pact_nerv_selector_v2_mlx_to_pytorch_state_dict import (
        PSV2_BRIDGE_SCHEMA,
        export_pact_nerv_selector_v2_mlx_to_pytorch,
    )

    assert PSV2_BRIDGE_SCHEMA == "pact_nerv_selector_v2_mlx_pytorch_export_bridge.v1"
    assert callable(export_pact_nerv_selector_v2_mlx_to_pytorch)


@skip_no_mlx
def test_bridge_pytorch_strict_false_missing_only_selectors_buffer(tmp_path: Path) -> None:
    """Verify the bridge tolerates PyTorch sister's selectors buffer as missing key."""
    import json

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    from tac.substrates._shared.numpy_portable_inflate import pack_state_dict_numpy
    from tac.substrates.pact_nerv_selector_v2.architecture import (
        PactNervSelectorV2Config,
    )
    from tac.substrates.pact_nerv_selector_v2.mlx_renderer import (
        PactNervSelectorV2SubstrateMLX,
    )
    from tools.export_pact_nerv_selector_v2_mlx_to_pytorch_state_dict import (
        export_pact_nerv_selector_v2_mlx_to_pytorch,
    )

    cfg = PactNervSelectorV2Config(num_pairs=4)
    model = PactNervSelectorV2SubstrateMLX(cfg)
    mx.eval(model.parameters())
    # Pack as MLX numpy-portable state_dict.
    flat = tree_flatten(model.parameters())
    sd_np = {name: np.asarray(arr, dtype=np.float32) for name, arr in flat}
    npsd_bytes = pack_state_dict_numpy(sd_np)
    src = tmp_path / "test.npsd"
    src.write_bytes(npsd_bytes)
    out_pt = tmp_path / "test.pt"
    proof = tmp_path / "proof.json"
    manifest = export_pact_nerv_selector_v2_mlx_to_pytorch(
        mlx_state_dict_path=src,
        output_pytorch_state_dict=out_pt,
        parity_proof_out=proof,
        sample_pair_indices=(0, 1),
    )
    # PyTorch state_dict loaded without unexpected keys.
    assert manifest["tensor_count"] == len(sd_np)
    assert out_pt.exists()
    assert proof.exists()
    # Canonical non-promotable markers per Catalog #287/#323.
    assert manifest["promotable"] is False
    assert manifest["score_claim"] is False
    assert manifest["axis_tag"] == "[predicted]"
    # Forward parity reported (band check is informational; promotion gated
    # at sister gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v2).
    parity = manifest["forward_parity"]
    assert parity["backends_compared"] == "mlx_vs_pytorch_forward"
    assert "max_abs_drift_01" in parity
    proof_data = json.loads(proof.read_text())
    assert proof_data["schema_version"] == "pact_nerv_selector_v2_mlx_pytorch_export_bridge.v1"
