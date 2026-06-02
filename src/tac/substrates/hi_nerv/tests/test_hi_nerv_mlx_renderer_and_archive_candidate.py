# SPDX-License-Identifier: MIT
"""HiNeRV MLX renderer bridge and archive-bound bundle tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

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
    reason=(
        "MLX not available on this host; HiNeRV MLX tests require Apple "
        "Silicon with the mlx package installed."
    ),
)


def test_mlx_renderer_uses_canonical_generic_resize_helper() -> None:
    source = (
        REPO_ROOT / "src" / "tac" / "substrates" / "hi_nerv" / "mlx_renderer.py"
    ).read_text(encoding="utf-8")

    assert "bilinear_resize_nhwc" in source
    resize_body = source.split("def _bilinear_resize_nhwc", maxsplit=1)[1].split(
        "def _siren_uniform_bound", maxsplit=1
    )[0]
    assert "NotImplementedError" not in resize_body


def _smoke_cfg():
    from tac.substrates.hi_nerv.architecture import HinervConfig

    return HinervConfig(
        latent_dim_coarse=4,
        latent_dim_mid=6,
        latent_dim_fine=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _exportable_torch_model():
    from tac.substrates.hi_nerv.architecture import HinervSubstrate

    cfg = _smoke_cfg()
    torch.manual_seed(19)
    model = HinervSubstrate(cfg).eval()

    class _ExportableModel:
        def __init__(self) -> None:
            self.cfg = cfg

        def export_state_dict(self) -> dict[str, object]:
            return {
                name: tensor.detach().cpu().numpy().copy()
                for name, tensor in model.state_dict().items()
            }

    return _ExportableModel()


@skip_no_mlx
def test_mlx_renderer_imports_clean() -> None:
    from tac.substrates.hi_nerv.mlx_renderer import (
        MLX_EVIDENCE_GRADE,
        SCHEMA_VERSION,
        HinervSubstrateMLX,
    )

    assert SCHEMA_VERSION == "hi_nerv_mlx_renderer_v1"
    assert MLX_EVIDENCE_GRADE == "[macOS-MLX research-signal]"
    assert HinervSubstrateMLX is not None


@skip_no_mlx
def test_mlx_renderer_parameter_parity_with_pytorch() -> None:
    from tac.substrates.hi_nerv.architecture import HinervSubstrate
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = _smoke_cfg()
    torch_model = HinervSubstrate(cfg)
    mlx_model = HinervSubstrateMLX(cfg)
    assert torch_model.num_parameters() == mlx_model.num_parameters()


@skip_no_mlx
def test_mlx_renderer_forward_shape_b2chw_255() -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    output = model(mx.array([0, 1, 2], dtype=mx.int32))
    mx.eval(output)
    assert tuple(int(s) for s in output.shape) == (
        3,
        2,
        3,
        cfg.output_height,
        cfg.output_width,
    )
    assert float(mx.min(output)) >= 0.0
    assert float(mx.max(output)) <= 255.0


@skip_no_mlx
def test_mlx_renderer_generic_resize_path_matches_pytorch() -> None:
    import mlx.core as mx
    import numpy as np
    import torch.nn.functional as F

    from tac.substrates.hi_nerv.mlx_renderer import _bilinear_resize_nhwc

    rng = np.random.default_rng(17)
    x_np = rng.normal(size=(2, 5, 7, 3)).astype("float32")
    y_mlx = np.asarray(
        _bilinear_resize_nhwc(mx.array(x_np), target_h=13, target_w=17),
        dtype=np.float32,
    )
    y_ref = (
        F.interpolate(
            torch.from_numpy(x_np).permute(0, 3, 1, 2),
            size=(13, 17),
            mode="bilinear",
            align_corners=False,
        )
        .permute(0, 2, 3, 1)
        .numpy()
    )
    assert float(np.max(np.abs(y_mlx - y_ref))) < 1e-5


@skip_no_mlx
def test_mlx_decoder_fake_quant_uses_archive_axis0_scale() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.mlx_renderer import _fake_quant_symmetric_ste

    values = mx.array(
        [
            [1.0, 1.7, 4.0],
            [0.50, 0.20, -0.10],
        ],
        dtype=mx.float32,
    )
    quantized = _fake_quant_symmetric_ste(values, bits=2)
    mx.eval(quantized)

    np.testing.assert_allclose(
        np.asarray(quantized),
        np.asarray(
            [
                [0.0, 0.0, 4.0],
                [0.50, 0.0, -0.0],
            ],
            dtype=np.float32,
        ),
        atol=0.0,
    )


@skip_no_mlx
def test_mlx_decoder_fake_quant_forward_changes_surface_without_mutating_export() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    pair_indices = mx.array([0, 1, 2], dtype=mx.int32)
    baseline = model(pair_indices)
    mx.eval(baseline)
    exported_before = model.export_state_dict()

    model.configure_decoder_fake_quant_forward(enabled=True, quant_bits=2)
    quantized = model(pair_indices)
    mx.eval(quantized)
    exported_after = model.export_state_dict()

    assert tuple(int(s) for s in quantized.shape) == tuple(
        int(s) for s in baseline.shape
    )
    assert np.isfinite(np.asarray(quantized)).all()
    assert float(mx.min(quantized)) >= 0.0
    assert float(mx.max(quantized)) <= 255.0
    assert float(mx.max(mx.abs(quantized - baseline))) > 1.0e-7
    for name, before in exported_before.items():
        np.testing.assert_array_equal(before, exported_after[name])

    model.configure_decoder_fake_quant_forward(enabled=False, quant_bits=2)
    restored = model(pair_indices)
    mx.eval(restored)
    assert float(mx.max(mx.abs(restored - baseline))) < 1.0e-6


@skip_no_mlx
def test_mlx_decoder_fake_quant_rejects_invalid_quant_bits() -> None:
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    model = HinervSubstrateMLX(_smoke_cfg())
    with pytest.raises(ValueError, match="quant_bits"):
        model.configure_decoder_fake_quant_forward(enabled=True, quant_bits=0)


@skip_no_mlx
def test_mlx_exported_state_dict_matches_pytorch_forward() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.architecture import HinervSubstrate
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = _smoke_cfg()
    mlx_model = HinervSubstrateMLX(cfg)
    mx.eval(mlx_model.parameters())
    torch_model = HinervSubstrate(cfg).eval()
    state = {
        name: torch.from_numpy(arr.copy())
        for name, arr in mlx_model.export_state_dict().items()
    }
    load_result = torch_model.load_state_dict(state, strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    pair_indices = [0, 1, 2]
    with torch.no_grad():
        rgb_0, rgb_1 = torch_model(torch.tensor(pair_indices, dtype=torch.long))
    torch_out = torch.stack([rgb_0, rgb_1], dim=1).numpy().astype("float32")
    mlx_out = (
        np.asarray(
            mlx_model(mx.array(np.asarray(pair_indices, dtype=np.int32))),
            dtype=np.float32,
        )
        / 255.0
    )
    drift = np.abs(torch_out - mlx_out)
    assert float(drift.max()) < 0.001
    assert float(drift.mean()) < 1e-4


def test_archive_candidate_int8_decoder_packet_roundtrip() -> None:
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import (
        pack_archive_from_exported_state_dict,
    )

    exportable = _exportable_torch_model()
    blob = pack_archive_from_exported_state_dict(
        exported_state_dict=exportable.export_state_dict(),
        cfg=exportable.cfg,
        decoder_codec="int8_mixed",
    )
    arc = parse_archive(blob)

    assert blob[:4] == b"HIV1"
    assert arc.latents_coarse.shape == (
        exportable.cfg.num_pairs,
        exportable.cfg.latent_dim_coarse,
    )
    assert arc.meta["_decoder_state_codec"]["codec"] == "int8_mixed"
    assert "latents_coarse" not in arc.decoder_state_dict


def test_archive_candidate_rejects_incomplete_exported_decoder_state() -> None:
    from tac.substrates.hi_nerv.archive_candidate import (
        pack_archive_from_exported_state_dict,
    )

    exportable = _exportable_torch_model()
    exported = exportable.export_state_dict()
    exported.pop("head_rgb_1.bias")

    with pytest.raises(ValueError, match="hi_nerv_exported_decoder_state invalid"):
        pack_archive_from_exported_state_dict(
            exported_state_dict=exported,
            cfg=exportable.cfg,
            decoder_codec="int8_mixed",
        )


def test_archive_export_emits_receiver_proof_and_hprc_spine(tmp_path: Path) -> None:
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive

    archive_path, archive_sha, archive_bytes = export_hi_nerv_mlx_archive(
        _exportable_torch_model(),
        tmp_path / "hi_nerv_export",
        repo_root=REPO_ROOT,
        decoder_codec="int8_mixed",
        retain_receiver_proof_output=False,
    )

    assert archive_path.is_file()
    assert len(archive_sha) == 64
    assert archive_bytes == archive_path.stat().st_size

    manifest_path = (
        tmp_path
        / "hi_nerv_export"
        / "hprc_representation_spine_hi_nerv_manifest.json"
    )
    package_path = tmp_path / "hi_nerv_export" / "archive_bound_candidate_adapter_package.json"
    npz_path = tmp_path / "hi_nerv_export" / "hi_nerv_mlx_exported_state.npz"
    npz_manifest_path = (
        tmp_path
        / "hi_nerv_export"
        / "hi_nerv_mlx_exported_state_npz_manifest.json"
    )
    proof_path = (
        tmp_path
        / "hi_nerv_export"
        / "receiver_proof"
        / "hi_nerv_mlx_receiver_proof.json"
    )
    assert manifest_path.is_file()
    assert package_path.is_file()
    assert npz_path.is_file()
    assert npz_manifest_path.is_file()
    assert proof_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    npz_manifest = json.loads(npz_manifest_path.read_text(encoding="utf-8"))
    assert manifest["family"] == "hi_nerv"
    assert proof["runtime_consumption_proof_ready"] is True
    assert proof["receiver_output_kind"] == "file"
    assert proof["receiver_output_retained"] is False
    assert package["receiver_proof"]["receiver_contract_satisfied"] is True
    assert npz_manifest["schema"] == "framework_agnostic_npz_bridge_manifest.v1"
    assert npz_manifest["consumption_recommended"] is True
    assert npz_manifest["artifact_sha256"]
    spine_extra = manifest["manifest"]["representation_spine"]["manifest_extra"]
    assert spine_extra["state_npz_bridge"]["artifact_sha256"] == (
        npz_manifest["artifact_sha256"]
    )
    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    runtime_manifest = row["runtime_adapter_manifest"]
    assert runtime_manifest["state_npz_bridge_manifest"]["artifact_sha256"] == (
        npz_manifest["artifact_sha256"]
    )
    portability = row["runtime_adapter_manifest"][
        "mlx_numpy_portability_contract"
    ]
    assert portability["portability_status"] == (
        "numpy_export_bridge_ready_receiver_not_numpy"
    )
    assert portability["numpy_array_export"] is True
    assert portability["canonical_npz_bridge_used"] is True
    assert portability["pure_numpy_inflate"] is False
    assert "torch" in portability["non_numpy_receiver_dependencies"]
    assert "inflate_runtime_not_pure_numpy" in portability["portability_blockers"]
    assert "canonical_npz_bridge_not_used_or_not_applicable" not in portability[
        "portability_blockers"
    ]
