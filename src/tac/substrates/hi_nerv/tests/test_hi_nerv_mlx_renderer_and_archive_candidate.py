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
    proof_path = (
        tmp_path
        / "hi_nerv_export"
        / "receiver_proof"
        / "hi_nerv_mlx_receiver_proof.json"
    )
    assert manifest_path.is_file()
    assert package_path.is_file()
    assert proof_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    assert manifest["family"] == "hi_nerv"
    assert proof["runtime_consumption_proof_ready"] is True
    assert proof["receiver_output_kind"] == "file"
    assert proof["receiver_output_retained"] is False
    assert package["receiver_proof"]["receiver_contract_satisfied"] is True
    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    portability = row["runtime_adapter_manifest"][
        "mlx_numpy_portability_contract"
    ]
    assert portability["portability_status"] == (
        "numpy_export_bridge_ready_receiver_not_numpy"
    )
    assert portability["numpy_array_export"] is True
    assert portability["pure_numpy_inflate"] is False
    assert "torch" in portability["non_numpy_receiver_dependencies"]
    assert "inflate_runtime_not_pure_numpy" in portability["portability_blockers"]
