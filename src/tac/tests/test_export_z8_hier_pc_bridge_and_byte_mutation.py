# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Z8 MLX->PyTorch export bridge + byte-mutation consumption proof.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Slot EEE Class 2 (tests verify
ACTUAL BEHAVIOR, not constants): every test here exercises the real bridge /
proof code paths against a real (small) trained Z8 MLX renderer state_dict and
a real Z8HPC1 archive. The headline guards FAIL if the bridge stops producing a
byte-stable round-trip or the byte-mutation proof stops detecting wavelet
consumption.
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

# Skip the whole module cleanly on non-Apple-Silicon hosts (MLX-required for
# the renderer forward; the bridge converter + byte-mutation proof are
# numpy-only and could run, but the trained-renderer fixture needs MLX).
_HAS_MLX = importlib.util.find_spec("mlx") is not None

from tools.export_z8_hier_pc_mlx_to_pytorch_state_dict import (  # noqa: E402
    Z8_BRIDGE_SCHEMA,
    export_z8_hier_pc_mlx_to_pytorch,
    infer_z8_config_from_state_dict,
)
from tools.probe_z8_archive_distinguishing_feature_byte_mutation import (  # noqa: E402
    Z8_BYTE_MUTATION_PROOF_SCHEMA,
    probe_z8_archive_distinguishing_feature,
)

mlx_required = pytest.mark.skipif(not _HAS_MLX, reason="MLX (mlx) not available")


# --------------------------------------------------------------------------- #
# Fixtures: a real small trained Z8 MLX renderer state_dict + a real archive.   #
# --------------------------------------------------------------------------- #
def _small_z8_config():
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
    )

    return Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=3,
        deterministic_state_dim=8,
        ego_motion_dim=6,
        gumbel_temperature=1.0,
        use_straight_through=True,
    )


def _extract_z8_renderer_state_dict_numpy(model) -> dict:
    """Extract the Z8 MLX renderer's params as a numpy state_dict (MLX layout).

    Walks the SAME dotted-name surface the trained-checkpoint .npsd carries:
    list-stored ``logits_per_level.<i>`` + the nested nn.Module params via the
    canonical numpy-portable pack helper (it accepts MLX arrays).
    """
    import mlx.core as mx

    sd: dict = {}
    # List-stored categorical posterior (NOT nn.Module attrs).
    for i, arr in enumerate(model.logits_per_level):
        sd[f"logits_per_level.{i}"] = np.asarray(mx.stop_gradient(arr), dtype=np.float32)
    # cat_to_continuous_per_level.<i>.{weight,bias}
    for i, lin in enumerate(model.cat_to_continuous_per_level):
        sd[f"cat_to_continuous_per_level.{i}.weight"] = np.asarray(lin.weight, dtype=np.float32)
        sd[f"cat_to_continuous_per_level.{i}.bias"] = np.asarray(lin.bias, dtype=np.float32)
    sd["deterministic_gate.weight"] = np.asarray(model.deterministic_gate.weight, dtype=np.float32)
    sd["deterministic_gate.bias"] = np.asarray(model.deterministic_gate.bias, dtype=np.float32)
    sd["level_fusion.weight"] = np.asarray(model.level_fusion.weight, dtype=np.float32)
    sd["level_fusion.bias"] = np.asarray(model.level_fusion.bias, dtype=np.float32)
    sd["stem.weight"] = np.asarray(model.stem.weight, dtype=np.float32)
    sd["stem.bias"] = np.asarray(model.stem.bias, dtype=np.float32)
    for i, block in enumerate(model.blocks):
        sd[f"blocks.{i}.conv.weight"] = np.asarray(block.conv.weight, dtype=np.float32)
        sd[f"blocks.{i}.conv.bias"] = np.asarray(block.conv.bias, dtype=np.float32)
        if getattr(block, "skip_conv", None) is not None:
            sd[f"blocks.{i}.skip_conv.weight"] = np.asarray(block.skip_conv.weight, dtype=np.float32)
            sd[f"blocks.{i}.skip_conv.bias"] = np.asarray(block.skip_conv.bias, dtype=np.float32)
    sd["refine0.weight"] = np.asarray(model.refine0.weight, dtype=np.float32)
    sd["refine0.bias"] = np.asarray(model.refine0.bias, dtype=np.float32)
    sd["refine1.weight"] = np.asarray(model.refine1.weight, dtype=np.float32)
    sd["refine1.bias"] = np.asarray(model.refine1.bias, dtype=np.float32)
    sd["rgb_0.weight"] = np.asarray(model.rgb_0.weight, dtype=np.float32)
    sd["rgb_0.bias"] = np.asarray(model.rgb_0.bias, dtype=np.float32)
    sd["rgb_1.weight"] = np.asarray(model.rgb_1.weight, dtype=np.float32)
    sd["rgb_1.bias"] = np.asarray(model.rgb_1.bias, dtype=np.float32)
    return sd


def _write_trained_npsd(tmp_path) -> tuple:
    """Build a real small Z8 MLX renderer, randomize, pack to an .npsd file."""
    import mlx.core as mx

    from tac.substrates._shared.numpy_portable_inflate import pack_state_dict_numpy
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalPredictiveCoderMLX,
    )

    cfg = _small_z8_config()
    model = Z8HierarchicalPredictiveCoderMLX(cfg)
    # Randomize logits so the categorical posterior is non-uniform (a real
    # trained renderer has informative logits; uniform would be a degenerate
    # fixture per Slot EEE Class 3).
    for i in range(len(model.logits_per_level)):
        key = mx.random.key(100 + i)
        model.logits_per_level[i] = mx.random.normal(
            shape=model.logits_per_level[i].shape, key=key
        )
    sd = _extract_z8_renderer_state_dict_numpy(model)
    # fp32 to preserve the byte-stable parity guarantee (fp16 would inject
    # quantization that is irrelevant to the bridge's transpose correctness).
    blob = pack_state_dict_numpy(sd, dtype="fp32")
    npsd_path = tmp_path / "z8_trained.npsd"
    npsd_path.write_bytes(blob)
    return npsd_path, cfg, sd


def _build_small_archive() -> bytes:
    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        build_z8hpc1_archive_bytes_from_canonical_quadruple,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
    )

    cfg = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=2,
        deterministic_state_dim=8,
        eval_size=(32, 32),
    )
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    rng = np.random.RandomState(7)
    f0 = rng.uniform(0, 1, size=(2, 32, 32, 3)).astype(np.float32)
    f1 = rng.uniform(0, 1, size=(2, 32, 32, 3)).astype(np.float32)
    return build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)


# --------------------------------------------------------------------------- #
# Config inference (numpy-only; runs without MLX).                              #
# --------------------------------------------------------------------------- #
def test_infer_config_from_state_dict_matches_known_shapes() -> None:
    # Hand-build a numpy state_dict with the canonical Z8 shapes; the inferred
    # config must match exactly (Catalog #229 premise discipline).
    sd = {
        "logits_per_level.0": np.zeros((3, 4, 16), dtype=np.float32),
        "logits_per_level.1": np.zeros((3, 3, 8), dtype=np.float32),
        "logits_per_level.2": np.zeros((3, 2, 4), dtype=np.float32),
        "cat_to_continuous_per_level.0.weight": np.zeros((12, 64), dtype=np.float32),
        "stem.weight": np.zeros((8 * 6 * 8, 12), dtype=np.float32),  # base_channels=8
        "deterministic_gate.weight": np.zeros((8, 18), dtype=np.float32),  # det=8, in=12+6
    }
    inferred = infer_z8_config_from_state_dict(sd)
    assert inferred["num_levels"] == 3
    assert inferred["num_pairs"] == 3
    assert inferred["num_groups_per_level"] == (4, 3, 2)
    assert inferred["num_categories_per_level"] == (16, 8, 4)
    assert inferred["decoder_latent_dim"] == 12
    assert inferred["base_channels"] == 8
    assert inferred["deterministic_state_dim"] == 8
    assert inferred["ego_motion_dim"] == 6


def test_infer_config_rejects_missing_logits() -> None:
    with pytest.raises(ValueError, match=r"logits_per_level\.0"):
        infer_z8_config_from_state_dict({"stem.weight": np.zeros((1, 1))})


# --------------------------------------------------------------------------- #
# Bridge: real MLX trained renderer -> PyTorch .pt + self-parity proof.          #
# --------------------------------------------------------------------------- #
@mlx_required
def test_bridge_exports_loadable_pytorch_state_dict_all_39_tensors(tmp_path) -> None:
    import torch

    npsd_path, _cfg, sd = _write_trained_npsd(tmp_path)
    out_pt = tmp_path / "z8.pt"
    manifest = export_z8_hier_pc_mlx_to_pytorch(
        mlx_state_dict_path=npsd_path,
        output_pytorch_state_dict=out_pt,
        parity_proof_out=tmp_path / "proof.json",
        sample_pair_indices=(0, 1, 2),
    )
    assert manifest["schema_version"] == Z8_BRIDGE_SCHEMA
    assert out_pt.is_file()
    loaded = torch.load(out_pt, weights_only=True)
    # Every source tensor key is present in the exported PyTorch state_dict.
    assert set(loaded.keys()) == set(sd.keys())
    assert manifest["tensor_count"] == len(sd)


@mlx_required
def test_bridge_conv_weights_transposed_hwio_to_oihw(tmp_path) -> None:
    import torch

    npsd_path, _cfg, sd = _write_trained_npsd(tmp_path)
    out_pt = tmp_path / "z8.pt"
    export_z8_hier_pc_mlx_to_pytorch(
        mlx_state_dict_path=npsd_path,
        output_pytorch_state_dict=out_pt,
        parity_proof_out=None,
    )
    loaded = torch.load(out_pt, weights_only=True)
    # MLX HWIO (out, kh, kw, in) -> PyTorch OIHW (out, in, kh, kw).
    mlx_w = sd["blocks.0.conv.weight"]  # (32, 3, 3, 8)
    pt_w = np.asarray(loaded["blocks.0.conv.weight"], dtype=np.float32)
    assert mlx_w.shape == (32, 3, 3, 8)
    assert pt_w.shape == (32, 8, 3, 3)
    # The transpose must be exact: HWIO[o,h,w,i] == OIHW[o,i,h,w].
    assert np.abs(np.transpose(mlx_w, (0, 3, 1, 2)) - pt_w).max() == 0.0
    # Non-conv tensors (logits, Linear) pass through unchanged.
    assert np.abs(sd["logits_per_level.0"] - np.asarray(loaded["logits_per_level.0"])).max() == 0.0


@mlx_required
def test_bridge_self_parity_is_byte_stable_zero_drift(tmp_path) -> None:
    """HEADLINE GUARD: the exported .pt round-trips back to MLX and reproduces
    the IDENTICAL frames (zero drift). FAILS if the OIHW<->HWIO transpose or the
    renderer reconstruction ever diverges (a real bridge bug)."""
    npsd_path, _cfg, _sd = _write_trained_npsd(tmp_path)
    manifest = export_z8_hier_pc_mlx_to_pytorch(
        mlx_state_dict_path=npsd_path,
        output_pytorch_state_dict=tmp_path / "z8.pt",
        parity_proof_out=tmp_path / "proof.json",
        sample_pair_indices=(0, 1, 2),
    )
    parity = manifest["forward_parity"]
    assert parity["backends_compared"].startswith("mlx_renderer_vs")
    assert parity["drift_within_band"] is True
    # Byte-stable: the round-trip transpose is lossless on fp32.
    assert parity["max_abs_drift_01"] == 0.0
    assert parity["mean_abs_drift_01"] == 0.0


@mlx_required
def test_bridge_manifest_carries_tier_a_non_promotable_markers(tmp_path) -> None:
    """Catalog #341/#192/#323: the bridge output is NOT a score claim."""
    npsd_path, _cfg, _sd = _write_trained_npsd(tmp_path)
    proof = tmp_path / "proof.json"
    export_z8_hier_pc_mlx_to_pytorch(
        mlx_state_dict_path=npsd_path,
        output_pytorch_state_dict=tmp_path / "z8.pt",
        parity_proof_out=proof,
    )
    d = json.loads(proof.read_text())
    assert d["axis_tag"] == "[predicted]"
    assert d["evidence_grade"] == "predicted"
    assert d["score_claim"] is False
    assert d["promotable"] is False
    assert d["ready_for_exact_eval_dispatch"] is False
    assert "provenance" in d


# --------------------------------------------------------------------------- #
# Byte-mutation consumption proof (numpy-only; runs without MLX).               #
# --------------------------------------------------------------------------- #
def test_byte_mutation_proof_wavelet_blob_is_pixel_consumed() -> None:
    """HEADLINE GUARD: mutating a wavelet-coefficient byte CHANGES the
    reconstructed pixels (Catalog #272 operational consumption). FAILS if the
    inflate path ever stops consuming the wavelet_blob (a no-op regression)."""
    archive = _build_small_archive()
    manifest = probe_z8_archive_distinguishing_feature(_write_archive_tmp(archive))
    assert manifest["schema_version"] == Z8_BYTE_MUTATION_PROOF_SCHEMA
    wav = manifest["sections"]["wavelet_blob"]
    assert wav["verdict"] == "PIXEL_CONSUMED"
    assert wav["max_abs_pixel_delta"] > 0.0
    assert wav["n_pixel_changed"] >= 1
    assert manifest["distinguishing_feature_consumed"] is True


def test_byte_mutation_proof_decoder_blob_not_pixel_consumed_honest() -> None:
    """HONEST CHARACTERIZATION (Catalog #220 trap-avoidance): the decoder_blob
    is parse-consumed only, NOT pixel-consumed. The proof reports this so a
    future wave does not mistake the decoder slot for a distinguishing feature."""
    archive = _build_small_archive()
    manifest = probe_z8_archive_distinguishing_feature(_write_archive_tmp(archive))
    dec = manifest["sections"]["decoder_blob"]
    # The decoder slot is a placeholder; it is never PIXEL_CONSUMED by inflate.
    assert dec["verdict"] != "PIXEL_CONSUMED"
    # The honest architecture note must surface the research-substrate-trap warning.
    assert "research-substrate trap" in manifest["honest_architecture_note"]


def test_byte_mutation_proof_is_non_promotable() -> None:
    archive = _build_small_archive()
    manifest = probe_z8_archive_distinguishing_feature(_write_archive_tmp(archive))
    assert manifest["score_claim"] is False
    assert manifest["promotable"] is False
    assert manifest["axis_tag"] == "[macOS-CPU advisory]"


# --------------------------------------------------------------------------- #
# Helpers                                                                        #
# --------------------------------------------------------------------------- #
_TMP_ARCHIVES: list = []


def _write_archive_tmp(archive_bytes: bytes):
    import os
    import tempfile
    from pathlib import Path

    fd, name = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd, "wb") as fh:
        fh.write(archive_bytes)
    _TMP_ARCHIVES.append(name)
    return Path(name)
