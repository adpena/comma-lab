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


def test_mlx_renderer_contains_official_grid_convnext_port() -> None:
    source = (
        REPO_ROOT / "src" / "tac" / "substrates" / "hi_nerv" / "mlx_renderer.py"
    ).read_text(encoding="utf-8")

    assert "class HierarchicalFeatureGridMLX" in source
    assert "class ConvNeXtBlockMLX" in source
    assert "trilinear_upsample_mlx" in source
    assert "feature_grids.{i}.grids.{level}" in source
    assert "convnext_blocks.{i}.dwconv.weight" in source


def test_hi_nerv_inflate_refuses_multi_entry_file_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tac.substrates.hi_nerv import inflate

    archive_dir = tmp_path / "archive"
    output_dir = tmp_path / "out"
    archive_dir.mkdir()
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("0\n1\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inflate.py",
            archive_dir.as_posix(),
            output_dir.as_posix(),
            file_list.as_posix(),
        ],
    )

    assert inflate.main_cli() == 2
    assert "supports exactly one archive-bound video entry" in capsys.readouterr().err


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


def _official_smoke_cfg():
    from tac.substrates.hi_nerv.architecture import HinervConfig

    return HinervConfig(
        latent_dim_coarse=3,
        latent_dim_mid=4,
        latent_dim_fine=5,
        embed_dim=8,
        initial_grid_h=2,
        initial_grid_w=3,
        decoder_channels=(7, 6),
        sin_frequency=10.0,
        num_upsample_blocks=2,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=3,
        output_height=8,
        output_width=12,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=3,
        convnext_mlp_ratio=2,
        convnext_kernel_size=3,
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
def test_mlx_renderer_official_grid_convnext_parameter_parity_with_pytorch() -> None:
    from tac.substrates.hi_nerv.architecture import HinervSubstrate
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = _official_smoke_cfg()
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
def test_mlx_decoder_fake_quant_can_target_named_receiver_tensors() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    model = HinervSubstrateMLX(_smoke_cfg())
    pair_indices = mx.array([0, 1, 2], dtype=mx.int32)
    baseline = model(pair_indices)
    mx.eval(baseline)
    exported_before = model.export_state_dict()

    model.configure_decoder_fake_quant_forward(
        enabled=True,
        quant_bits=None,
        per_tensor_bits={
            "head_rgb_0.weight": 7,
            "head_rgb_1.weight": 0,
        },
    )
    targeted = model(pair_indices)
    mx.eval(targeted)
    exported_after = model.export_state_dict()

    assert float(mx.max(mx.abs(targeted[:, 0] - baseline[:, 0]))) > 1.0e-8
    assert float(mx.max(mx.abs(targeted[:, 1] - baseline[:, 1]))) > 1.0e-7
    for name, before in exported_before.items():
        np.testing.assert_array_equal(before, exported_after[name])


@skip_no_mlx
def test_mlx_decoder_fake_quant_can_consume_waterfill_plan() -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    model = HinervSubstrateMLX(_smoke_cfg())
    report = model.configure_decoder_fake_quant_forward_from_waterfill_plan(
        {
            "schema": "nerv_decoder_weight_waterfill.v1",
            "family": "hi_nerv",
            "candidate_id": "unit",
            "rows": [
                {
                    "group_name": "head_rgb_0.weight",
                    "selected_bits": 6,
                    "selected_action": "int6",
                },
                {
                    "group_name": "head_rgb_1.weight",
                    "selected_bits": 32,
                    "selected_action": "fp32_protect",
                },
            ],
            "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
        }
    )
    pair_indices = mx.array([0, 1, 2], dtype=mx.int32)
    output = model(pair_indices)
    mx.eval(output)

    assert report["configured"] is True
    assert report["configured_per_tensor_bits"] == {"head_rgb_0.weight": 6}
    assert model.decoder_fake_quant_bits_by_name == {"head_rgb_0.weight": 6}
    assert tuple(int(s) for s in output.shape) == (
        3,
        2,
        3,
        model.cfg.output_height,
        model.cfg.output_width,
    )
    assert report["score_claim"] is False


@skip_no_mlx
def test_mlx_decoder_fake_quant_rejects_invalid_quant_bits() -> None:
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    model = HinervSubstrateMLX(_smoke_cfg())
    with pytest.raises(ValueError, match="quant_bits"):
        model.configure_decoder_fake_quant_forward(enabled=True, quant_bits=0)
    with pytest.raises(ValueError, match="per_tensor_bits"):
        model.configure_decoder_fake_quant_forward(
            enabled=True,
            quant_bits=None,
            per_tensor_bits={"head_rgb_1.weight": 3},
        )


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


@skip_no_mlx
def test_mlx_official_grid_convnext_export_matches_pytorch_forward() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.architecture import HinervSubstrate
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = _official_smoke_cfg()
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
    exported = mlx_model.export_state_dict()
    assert "feature_grids.0.grids.0" in exported
    assert "convnext_blocks.0.dwconv.weight" in exported
    assert "convnext_blocks.0.gamma" in exported


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


def test_archive_candidate_pixel_proof_samples_full_video_span() -> None:
    from tac.substrates.hi_nerv.archive_candidate import (
        _sample_pair_indices_for_pixel_proof,
    )

    assert _sample_pair_indices_for_pixel_proof(
        num_pairs=600,
        max_pair_samples=3,
    ).tolist() == [0, 300, 599]
    assert _sample_pair_indices_for_pixel_proof(
        num_pairs=3,
        max_pair_samples=3,
    ).tolist() == [0, 1, 2]


def test_archive_candidate_applies_decoder_waterfill_plan_to_packed_state() -> None:
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import (
        pack_archive_from_exported_state_dict,
    )

    exportable = _exportable_torch_model()
    blob = pack_archive_from_exported_state_dict(
        exported_state_dict=exportable.export_state_dict(),
        cfg=exportable.cfg,
        decoder_codec="fp16_enveloped",
        decoder_weight_waterfill_plan={
            "schema": "nerv_decoder_weight_waterfill.v1",
            "family": "hi_nerv",
            "candidate_id": "unit",
            "compact_runner_launch_custody": {
                "schema": (
                    "compact_hi_nerv_decoder_weight_waterfill_launch_custody.v1"
                ),
                "path": "/Volumes/VertigoDataTier/pact/unit_waterfill.json",
                "sha256": "a" * 64,
                "source_schema": "nerv_decoder_weight_waterfill.v1",
                "family": "hi_nerv",
                "candidate_id": "unit",
                "row_count": 1,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "rows": [
                {
                    "group_name": "head_rgb_1.weight",
                    "selected_bits": 0,
                    "selected_action": "zero_rle",
                }
            ],
            "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
        },
    )
    arc = parse_archive(blob)

    assert torch.count_nonzero(arc.decoder_state_dict["head_rgb_1.weight"]).item() == 0
    waterfill = arc.meta["_hi_nerv_bitstream_preparation"][
        "decoder_weight_waterfill"
    ]
    assert waterfill["plan_attached"] is True
    assert waterfill["method"] == "decoder_weight_waterfill_selected_actions"
    assert waterfill["changed_tensor_count"] == 1
    assert waterfill["applied_rows"][0]["group_name"] == "head_rgb_1.weight"
    assert waterfill["applied_rows"][0]["changed"] is True
    assert waterfill["plan_custody"]["sha256"] == "a" * 64
    assert waterfill["plan_custody"]["path"].endswith("unit_waterfill.json")
    assert waterfill["plan_custody"]["score_claim"] is False
    assert "contest_cpu_cuda_exact_eval_not_executed" in waterfill["blockers"]
    assert waterfill["score_claim"] is False
    proof = waterfill["rendered_pixel_proof"]
    assert proof == arc.meta["_hi_nerv_bitstream_preparation"][
        "decoder_rendered_pixel_proof"
    ]
    assert waterfill["rendered_pixel_proof_status"] == (
        "sampled_rendered_pixels_changed"
    )
    assert proof["proof_kind"] == "sampled_receiver_rendered_pixel_delta"
    assert proof["pair_indices"] == [0, 1, 2]
    assert proof["changed_decoder_tensor_names"] == ["head_rgb_1.weight"]
    assert proof["rendered_pixels_changed"] is True
    assert proof["changed_rendered_pixel_count"] > 0
    assert proof["max_abs_rendered_pixel_delta"] > 0.0
    assert proof["score_claim"] is False


def test_archive_candidate_refuses_decoder_prep_rendered_pixel_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tac.substrates.hi_nerv import archive_candidate

    def _fake_noop_proof(**_: object) -> dict[str, object]:
        return {
            "schema": "hi_nerv_decoder_preparation_rendered_pixel_proof.v1",
            "proof_status": "sampled_rendered_pixels_no_change",
            "decoder_state_changed": True,
            "rendered_pixels_changed": False,
        }

    monkeypatch.setattr(
        archive_candidate,
        "_build_decoder_rendered_pixel_proof",
        _fake_noop_proof,
    )
    exportable = _exportable_torch_model()
    with pytest.raises(ValueError, match="rendered pixels did not change"):
        archive_candidate.pack_archive_from_exported_state_dict(
            exported_state_dict=exportable.export_state_dict(),
            cfg=exportable.cfg,
            decoder_codec="fp16_enveloped",
            decoder_weight_waterfill_plan={
                "schema": "nerv_decoder_weight_waterfill.v1",
                "family": "hi_nerv",
                "candidate_id": "unit",
                "rows": [
                    {
                        "group_name": "head_rgb_1.weight",
                        "selected_bits": 0,
                        "selected_action": "zero_rle",
                    }
                ],
                "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
            },
        )


def test_archive_candidate_refuses_unsafe_decoder_waterfill_plan() -> None:
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import (
        pack_archive_from_exported_state_dict,
    )

    exportable = _exportable_torch_model()
    blob = pack_archive_from_exported_state_dict(
        exported_state_dict=exportable.export_state_dict(),
        cfg=exportable.cfg,
        decoder_codec="fp16_enveloped",
        decoder_weight_waterfill_plan={
            "schema": "nerv_decoder_weight_waterfill.v1",
            "family": "hi_nerv",
            "candidate_id": "unit",
            "rows": [
                {
                    "group_name": "head_rgb_1.weight",
                    "selected_bits": 0,
                    "selected_action": "zero_rle",
                }
            ],
            "blockers": [
                "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin"
            ],
        },
    )
    arc = parse_archive(blob)

    assert torch.count_nonzero(arc.decoder_state_dict["head_rgb_1.weight"]).item() > 0
    waterfill = arc.meta["_hi_nerv_bitstream_preparation"][
        "decoder_weight_waterfill"
    ]
    assert waterfill["method"] == "decoder_weight_waterfill_blocked"
    assert waterfill["changed_tensor_count"] == 0
    assert waterfill["applied_rows"] == []
    assert waterfill["actuation_blockers"] == [
        "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin"
    ]


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
        source_backend="pytorch_test_export",
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
    bitstream_report_path = (
        tmp_path / "hi_nerv_export" / "hi_nerv_bitstream_preparation.json"
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
    assert bitstream_report_path.is_file()
    assert proof_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    npz_manifest = json.loads(npz_manifest_path.read_text(encoding="utf-8"))
    bitstream_report = json.loads(bitstream_report_path.read_text(encoding="utf-8"))
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
    assert spine_extra["hi_nerv_bitstream_preparation"] == bitstream_report
    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    runtime_manifest = row["runtime_adapter_manifest"]
    assert runtime_manifest["state_npz_bridge_manifest"]["artifact_sha256"] == (
        npz_manifest["artifact_sha256"]
    )
    assert runtime_manifest["hi_nerv_bitstream_preparation"] == bitstream_report
    assert runtime_manifest["hi_nerv_bitstream_preparation_path"] == (
        bitstream_report_path.as_posix()
    )
    portability = row["runtime_adapter_manifest"][
        "mlx_numpy_portability_contract"
    ]
    assert portability["portability_status"] == (
        "numpy_export_bridge_ready_receiver_not_numpy"
    )
    assert portability["training_backend"] == "pytorch_test_export"
    assert portability["numpy_array_export"] is True
    assert portability["canonical_npz_bridge_used"] is True
    assert portability["pure_numpy_inflate"] is False
    assert "torch" in portability["non_numpy_receiver_dependencies"]
    assert "training_backend_not_mlx" in portability["portability_blockers"]
    assert "inflate_runtime_not_pure_numpy" in portability["portability_blockers"]
    assert "canonical_npz_bridge_not_used_or_not_applicable" not in portability[
        "portability_blockers"
    ]


def test_archive_export_refuses_over_hard_byte_ceiling_before_receiver_package(
    tmp_path: Path,
) -> None:
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive

    out_dir = tmp_path / "hi_nerv_export_over_cap"
    with pytest.raises(ValueError, match="exceeds hard_byte_ceiling"):
        export_hi_nerv_mlx_archive(
            _exportable_torch_model(),
            out_dir,
            repo_root=REPO_ROOT,
            decoder_codec="int8_mixed",
            retain_receiver_proof_output=False,
            source_backend="pytorch_test_export",
            hard_byte_ceiling=1,
        )

    assert (out_dir / "archive.zip").is_file()
    assert not (out_dir / "archive_bound_candidate_adapter_package.json").exists()
    assert not (out_dir / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json").exists()


def test_archive_export_emits_hprc_spine_for_brotli_latents(tmp_path: Path) -> None:
    from tac.substrates.hi_nerv.archive import parse_archive, split_archive_sections
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive

    archive_path, archive_sha, archive_bytes = export_hi_nerv_mlx_archive(
        _exportable_torch_model(),
        tmp_path / "hi_nerv_export_brotli_latents",
        repo_root=REPO_ROOT,
        decoder_codec="int8_mixed",
        latent_codec="int16_brotli_q11",
        retain_receiver_proof_output=False,
        source_backend="pytorch_test_export",
    )

    manifest_path = (
        tmp_path
        / "hi_nerv_export_brotli_latents"
        / "hprc_representation_spine_hi_nerv_manifest.json"
    )
    proof_path = (
        tmp_path
        / "hi_nerv_export_brotli_latents"
        / "receiver_proof"
        / "hi_nerv_mlx_receiver_proof.json"
    )
    assert archive_path.is_file()
    assert len(archive_sha) == 64
    assert archive_bytes == archive_path.stat().st_size
    assert manifest_path.is_file()
    assert proof_path.is_file()

    inner = (tmp_path / "hi_nerv_export_brotli_latents" / "0.bin").read_bytes()
    sections = split_archive_sections(inner)
    parsed = parse_archive(inner)
    assert parsed.latents_coarse.shape[0] == _exportable_torch_model().cfg.num_pairs
    assert sections.meta["_latent_codec"] == "int16_brotli_q11"
    assert sections.meta["_latent_raw_bytes_coarse"] == parsed.latents_coarse.numel() * 2
    assert sections.meta["_latent_coded_bytes_coarse"] == len(
        sections.latents_coarse_blob
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    header = manifest["manifest"]["representation_spine"]["manifest_extra"]
    assert header["source_payload_kind"] == "hi_nerv_hiv1"
    assert any(row["name"] == "latents_rc" for row in manifest["manifest"]["sections"])
    assert any(
        row["name"] == "receiver_state"
        for row in manifest["manifest"]["sections"]
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert proof["runtime_consumption_proof_ready"] is True
