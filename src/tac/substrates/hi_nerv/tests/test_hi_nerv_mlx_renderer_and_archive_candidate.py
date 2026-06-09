# SPDX-License-Identifier: MIT
"""HiNeRV MLX renderer bridge and archive-bound bundle tests."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    reason=("MLX not available on this host; HiNeRV MLX tests require Apple Silicon with the mlx package installed."),
)


def test_mlx_renderer_uses_canonical_generic_resize_helper() -> None:
    source = (REPO_ROOT / "src" / "tac" / "substrates" / "hi_nerv" / "mlx_renderer.py").read_text(encoding="utf-8")

    assert "bilinear_resize_nhwc" in source
    resize_body = source.split("def _bilinear_resize_nhwc", maxsplit=1)[1].split(
        "def _siren_uniform_bound", maxsplit=1
    )[0]
    assert "NotImplementedError" not in resize_body


def test_mlx_renderer_contains_official_grid_convnext_port() -> None:
    source = (REPO_ROOT / "src" / "tac" / "substrates" / "hi_nerv" / "mlx_renderer.py").read_text(encoding="utf-8")

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
            return {name: tensor.detach().cpu().numpy().copy() for name, tensor in model.state_dict().items()}

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
def test_hinerv_mlx_renderer_init_seed_owns_exported_state() -> None:
    from dataclasses import replace

    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = replace(_official_smoke_cfg(), init_seed=17)
    mx.random.seed(12345)
    first = HinervSubstrateMLX(cfg).export_state_dict()
    mx.random.seed(98765)
    second = HinervSubstrateMLX(cfg).export_state_dict()

    assert set(first) == set(second)
    for name in sorted(first):
        assert np.array_equal(first[name], second[name]), name

    changed = HinervSubstrateMLX(replace(cfg, init_seed=18)).export_state_dict()
    assert any(not np.array_equal(first[name], changed[name]) for name in first)


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
def test_mlx_output_head_target_bias_init_is_archive_exported() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0 = mx.ones((cfg.num_pairs, cfg.output_height, cfg.output_width, 3)) * 0.1
    target1 = mx.ones((cfg.num_pairs, cfg.output_height, cfg.output_width, 3)) * 0.2

    payload = model.initialize_output_head_bias_from_targets(target0, target1)
    exported = model.export_state_dict()

    assert payload["runtime_sidecar_bytes"] == 0
    assert payload["archive_charged_decoder_tensors"] == [
        "head_rgb_0.bias",
        "head_rgb_1.bias",
    ]
    np.testing.assert_allclose(
        exported["head_rgb_0.bias"],
        np.log(np.asarray([0.1, 0.1, 0.1], dtype=np.float32) / 0.9),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        exported["head_rgb_1.bias"],
        np.log(np.asarray([0.2, 0.2, 0.2], dtype=np.float32) / 0.8),
        rtol=1e-5,
        atol=1e-5,
    )


@skip_no_mlx
def test_mlx_output_head_target_contrast_init_scales_archive_head_weights() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    ramp = mx.reshape(
        mx.linspace(0.05, 0.95, cfg.output_height * cfg.output_width),
        (1, cfg.output_height, cfg.output_width, 1),
    )
    target0 = mx.tile(mx.concatenate([ramp, 1.0 - ramp, 0.5 * ramp], axis=-1), (cfg.num_pairs, 1, 1, 1))
    target1 = mx.tile(mx.concatenate([1.0 - ramp, ramp, 0.25 + 0.5 * ramp], axis=-1), (cfg.num_pairs, 1, 1, 1))
    model.initialize_output_head_bias_from_targets(target0, target1)
    model.head_rgb_0.update({"weight": model.head_rgb_0.weight * 0.01})
    model.head_rgb_1.update({"weight": model.head_rgb_1.weight * 0.01})
    mx.eval(model.head_rgb_0.weight, model.head_rgb_1.weight)

    payload = model.initialize_output_head_contrast_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        max_gain=16.0,
    )

    assert payload["runtime_sidecar_bytes"] == 0
    assert payload["archive_charged_decoder_tensors"] == [
        "head_rgb_0.weight",
        "head_rgb_1.weight",
    ]
    before = np.mean(payload["output_rgb_1_std_before"])
    after = np.mean(payload["output_rgb_1_std_after"])
    assert after > before * 2.0
    assert payload["contrast_lift_passed"] is True
    assert payload["blockers"] == []
    assert payload["output_rgb_1_std_lift_ratio"] > 2.0
    assert max(payload["head_rgb_1_weight_gain"]) <= 16.0


@skip_no_mlx
def test_mlx_pair_local_actuator_smoke_is_latent_row_local(tmp_path: Path) -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(12)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    ramp = mx.reshape(
        mx.linspace(0.05, 0.95, cfg.output_height * cfg.output_width),
        (1, cfg.output_height, cfg.output_width, 1),
    )
    target0 = mx.tile(
        mx.concatenate([ramp, 1.0 - ramp, 0.5 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    target1 = mx.tile(
        mx.concatenate([1.0 - ramp, ramp, 0.25 + 0.5 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )

    smoke = model.build_pair_local_actuator_smoke_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        learning_rate=2.0e5,
        artifact_dir=tmp_path / "pair_local_smoke",
    )

    assert smoke["schema"] == "hinerv_pair_local_actuator_smoke.v1"
    assert smoke["execution_completed"] is True
    assert smoke["actuator"]["kind"] == "pair_local_latent_row"
    assert smoke["actuator"]["tensor_name"] == "latents_fine"
    assert smoke["actuator"]["runtime_sidecar_bytes"] == 0
    assert smoke["gradient"]["value_and_grad_checked"] is True
    assert smoke["pair_local_adapter_bytes"] > 0
    assert len(smoke["pair_local_adapter_sha256"]) == 64
    assert smoke["pair_local_grad_norm"] > 0.0
    assert smoke["pair_local_output_delta_l2"] > 0.0
    assert smoke["output_delta"]["pair_locality_verified"] is True
    assert smoke["output_delta"]["non_target_pair_output_delta_l2_max"] <= 1.0e-12
    assert smoke["output_delta"]["receiver_uint8_crossing_potential"] is True
    assert smoke["output_delta"]["pair_local_output_delta_max_abs_uint8"] >= 0.5
    assert smoke["output_delta"]["receiver_uint8_changed"] is True
    assert smoke["output_delta"]["receiver_uint8_changed_count"] > 0
    assert smoke["output_delta"]["receiver_uint8_delta_abs_max"] > 0
    assert smoke["output_delta"]["non_target_pair_receiver_uint8_changed_count"] == 0
    assert smoke["output_delta"]["non_target_pair_receiver_uint8_delta_abs_max"] == 0
    artifact = smoke["pair_local_smoke_artifact"]
    artifact_path = Path(artifact["path"])
    assert artifact_path.is_file()
    artifact_bytes = artifact_path.read_bytes()
    assert artifact["bytes"] == len(artifact_bytes)
    assert artifact["sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
    artifact_payload = json.loads(artifact_bytes.decode("utf-8"))
    assert artifact_payload["schema"] == "hinerv_pair_local_actuator_smoke_artifact.v1"
    assert artifact_payload["payload"]["schema"] == ("hinerv_pair_local_actuator_smoke.v1")
    assert artifact_payload["payload"]["pair_local_adapter_sha256"] == (smoke["pair_local_adapter_sha256"])
    state_restore = smoke["state_restore"]
    assert state_restore["checked_tensor_name"] == "latents_fine"
    assert state_restore["checked_row_indices"] == [0]
    assert state_restore["state_restored_after_smoke"] is True
    assert state_restore["original_row_sha256"] == state_restore["restored_row_sha256"]
    assert state_restore["mutated_row_sha256"] == smoke["pair_local_adapter_sha256"]
    assert state_restore["mutated_row_sha256"] != state_restore["original_row_sha256"]
    summary = smoke["summary_for_pr95_guard"]
    assert summary["pair_local_smoke_schema"] == "hinerv_pair_local_actuator_smoke.v1"
    assert summary["pair_local_adapter_sha256"] == smoke["pair_local_adapter_sha256"]
    assert summary["receiver_uint8_crossing_potential"] is True
    assert summary["pair_local_output_delta_max_abs_uint8"] >= 0.5
    assert summary["receiver_uint8_changed"] is True
    assert summary["receiver_uint8_changed_count"] == smoke["output_delta"]["receiver_uint8_changed_count"]
    assert summary["non_target_pair_receiver_uint8_changed_count"] == 0
    assert summary["pair_local_smoke_artifact_schema"] == ("hinerv_pair_local_actuator_smoke_artifact.v1")
    assert summary["pair_local_smoke_artifact_path"] == artifact["path"]
    assert summary["pair_local_smoke_artifact_sha256"] == artifact["sha256"]
    assert summary["pair_local_smoke_artifact_bytes"] == artifact["bytes"]
    assert summary["state_restored_after_smoke"] is True
    assert (
        summary["pair_local_latents_fine_original_row_sha256"]
        == (summary["pair_local_latents_fine_restored_row_sha256"])
    )
    output_rows = summary["section_output_delta_per_byte_rows"]
    assert output_rows[0]["output_delta_l2_per_byte"] > 0.0
    assert output_rows[0]["value_semantics"] == ("receiver_output_l2_per_byte_not_score_value")
    assert output_rows[0]["score_value_per_byte_measured"] is False
    assert summary["section_value_per_byte_rows"] == []
    assert summary["score_claim"] is False


@skip_no_mlx
def test_mlx_pair_local_actuator_smoke_blocks_subquantum_receiver_delta() -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(12)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    ramp = mx.reshape(
        mx.linspace(0.05, 0.95, cfg.output_height * cfg.output_width),
        (1, cfg.output_height, cfg.output_width, 1),
    )
    target0 = mx.tile(
        mx.concatenate([ramp, 1.0 - ramp, 0.5 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    target1 = mx.tile(
        mx.concatenate([1.0 - ramp, ramp, 0.25 + 0.5 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )

    smoke = model.build_pair_local_actuator_smoke_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        learning_rate=1.0e-2,
        receiver_quantum_line_search=False,
    )

    assert smoke["execution_completed"] is False
    assert "hinerv_pair_local_output_delta_below_uint8_half_step" in smoke["blockers"]
    assert smoke["summary_for_pr95_guard"] is None
    assert smoke["state_restore"]["state_restored_after_smoke"] is True
    assert smoke["output_delta"]["receiver_uint8_crossing_potential"] is False
    assert smoke["output_delta"]["receiver_uint8_changed"] is False
    assert smoke["output_delta"]["receiver_uint8_changed_count"] == 0
    assert smoke["output_delta"]["pair_local_output_delta_max_abs_uint8"] < 0.5


@skip_no_mlx
def test_mlx_pair_local_actuator_smoke_line_search_crosses_receiver_quantum() -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(12)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    ramp = mx.reshape(
        mx.linspace(0.05, 0.95, cfg.output_height * cfg.output_width),
        (1, cfg.output_height, cfg.output_width, 1),
    )
    target0 = mx.tile(
        mx.concatenate([ramp, 1.0 - ramp, 0.5 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    target1 = mx.tile(
        mx.concatenate([1.0 - ramp, ramp, 0.25 + 0.5 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )

    smoke = model.build_pair_local_actuator_smoke_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        learning_rate=100.0,
        receiver_quantum_line_search=True,
        receiver_quantum_line_search_max_scale=1.0e6,
    )

    attempts = smoke["output_delta"]["receiver_quantum_line_search_attempts"]
    assert smoke["execution_completed"] is True
    assert smoke["output_delta"]["receiver_uint8_crossing_potential"] is True
    assert smoke["output_delta"]["pair_local_output_delta_max_abs_uint8"] >= 0.5
    assert smoke["output_delta"]["non_target_pair_receiver_uint8_changed_count"] == 0
    assert smoke["output_delta"]["receiver_quantum_line_search_selected_scale"] > 1.0
    assert smoke["gradient"]["actual_learning_rate"] > smoke["gradient"]["base_learning_rate"]
    assert attempts[0]["scale"] == 1.0
    assert attempts[0]["receiver_uint8_crossing_potential"] is False
    assert any(attempt["accepted"] for attempt in attempts)
    summary = smoke["summary_for_pr95_guard"]
    assert (
        summary["receiver_quantum_line_search_selected_scale"]
        == smoke["output_delta"]["receiver_quantum_line_search_selected_scale"]
    )
    assert summary["receiver_quantum_line_search_attempt_count"] == len(attempts)


@skip_no_mlx
def test_mlx_scorer_domain_bootstrap_reduces_rgb_yuv6_loss() -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(0)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    ramp = mx.reshape(
        mx.linspace(0.05, 0.95, cfg.output_height * cfg.output_width),
        (1, cfg.output_height, cfg.output_width, 1),
    )
    target0 = mx.tile(
        mx.concatenate([0.2 + 0.2 * ramp, 0.1 + 0.4 * ramp, 0.3 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    target1 = mx.tile(
        mx.concatenate([0.4 - 0.2 * ramp, 0.15 + 0.3 * ramp, 0.2 + 0.2 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    model.initialize_output_head_bias_from_targets(target0, target1)
    model.initialize_output_head_contrast_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        max_gain=16.0,
    )

    payload = model.fit_scorer_domain_bootstrap_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        steps=6,
        learning_rate=1.0e-3,
        rgb_weight=1.0,
        yuv6_weight=0.5,
        temporal_delta_weight=0.25,
        grad_clip_max_norm=1.0,
    )

    assert payload["enabled"] is True
    assert payload["runtime_sidecar_bytes"] == 0
    assert payload["human_visual_fidelity_objective"] is False
    assert payload["contrast_floor_preserving_acceptance"] is True
    assert payload["contrast_floor_rejected_step_count"] >= 0
    assert payload["loss_history_last"] <= payload["loss_history_first"]
    assert payload["metrics_after"]["contrast_floor_loss"] <= payload["metrics_before"]["contrast_floor_loss"]
    assert payload["output_rgb_std_ratio_delta"] > 0.0
    assert payload["output_yuv6_temporal_delta_std_ratio_delta"] > 0.0
    assert payload["contrast_floor_weight"] > 0.0
    assert "latents_coarse" in payload["archive_charged_decoder_tensors"]


@skip_no_mlx
def test_mlx_scorer_domain_bootstrap_accepts_exact_target_region_waterfill() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(1)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    ramp = mx.reshape(
        mx.linspace(0.05, 0.95, cfg.output_height * cfg.output_width),
        (1, cfg.output_height, cfg.output_width, 1),
    )
    target0 = mx.tile(
        mx.concatenate([0.15 + 0.2 * ramp, 0.2 + 0.25 * ramp, 0.05 + 0.1 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    target1 = mx.tile(
        mx.concatenate([0.45 - 0.25 * ramp, 0.1 + 0.2 * ramp, 0.2 + 0.35 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    labels_np = np.zeros(
        (cfg.num_pairs, cfg.output_height, cfg.output_width),
        dtype=np.int32,
    )
    labels_np[:, : max(1, cfg.output_height // 4), :] = 1
    labels = mx.array(labels_np)

    model.initialize_output_head_bias_from_targets(target0, target1)
    model.initialize_output_head_contrast_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        max_gain=16.0,
    )

    payload = model.fit_scorer_domain_bootstrap_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=labels,
        target_region_bootstrap_weight=2.0,
        steps=8,
        learning_rate=1.0e-3,
        rgb_weight=0.5,
        yuv6_weight=0.25,
        temporal_delta_weight=0.1,
        grad_clip_max_norm=1.0,
    )

    region = payload["target_region_bootstrap"]
    assert region["enabled"] is True
    assert region["map_source"] == "exact_segnet_target_argmax_frame1"
    assert region["metadata"]["class_count"] == 2
    assert payload["target_region_bootstrap_weight"] == pytest.approx(2.0)
    assert payload["loss_history_last"] <= payload["loss_history_first"]
    assert payload["target_region_rgb_frame1_mse_delta"] >= 0.0
    assert payload["runtime_sidecar_bytes"] == 0
    assert "head_rgb_1.*" in payload["archive_charged_decoder_tensors"]


@skip_no_mlx
def test_mlx_scorer_domain_bootstrap_uses_live_segnet_margin_debt() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    class FakeSegNetTeacher:
        num_classes = 2

        def __init__(self, labels):
            self._labels = labels

        def teacher_argmax_for_indices(self, indices):
            return mx.take(self._labels, indices, axis=0)

        def teacher_logits_for_frames_nhwc01(self, frames):
            red = frames[..., 0]
            green = frames[..., 1]
            class0 = green - red
            class1 = red - green
            return mx.stack([class0, class1], axis=-1)

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    ramp = mx.reshape(
        mx.linspace(0.05, 0.95, cfg.output_height * cfg.output_width),
        (1, cfg.output_height, cfg.output_width, 1),
    )
    target0 = mx.tile(
        mx.concatenate([0.2 + 0.1 * ramp, 0.3 + 0.2 * ramp, 0.1 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    target1 = mx.tile(
        mx.concatenate([0.15 + 0.1 * ramp, 0.55 - 0.1 * ramp, 0.1 + 0.1 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    labels_np = np.zeros(
        (cfg.num_pairs, cfg.output_height, cfg.output_width),
        dtype=np.int32,
    )
    labels_np[:, : max(1, cfg.output_height // 3), :] = 1
    labels = mx.array(labels_np)
    teacher = FakeSegNetTeacher(labels)

    model.initialize_output_head_bias_from_targets(target0, target1)
    model.initialize_output_head_contrast_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        max_gain=16.0,
    )

    payload = model.fit_scorer_domain_bootstrap_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=labels,
        target_region_bootstrap_weight=0.25,
        scorer_teacher=teacher,
        segnet_margin_bootstrap_weight=1.5,
        segnet_margin_bootstrap_floor=0.1,
        segnet_hard_birth_bootstrap_weight=2.0,
        segnet_hard_birth_bootstrap_min_ratio_floor=0.05,
        steps=6,
        learning_rate=5.0e-4,
        rgb_weight=0.5,
        yuv6_weight=0.25,
        temporal_delta_weight=0.1,
        grad_clip_max_norm=1.0,
    )

    margin = payload["segnet_margin_bootstrap"]
    assert margin["enabled"] is True
    assert margin["source"] == (
        "receiver_uint8_roundtrip_ste_live_mlx_segnet_candidate_logits_against_frame1_target_argmax"
    )
    assert payload["segnet_margin_bootstrap_weight"] == pytest.approx(1.5)
    hard_birth = payload["segnet_hard_birth_bootstrap"]
    assert hard_birth["enabled"] is True
    assert hard_birth["source"] == (
        "receiver_uint8_roundtrip_ste_live_mlx_segnet_candidate_logits_worst_target_class_birth"
    )
    assert hard_birth["receiver_surface"] == "clamp_round_uint8_rgb_ste_nhwc01"
    assert hard_birth["worst_loss_selection"] == "score_weighted_unsolved_argmax_mass"
    assert payload["segnet_hard_birth_bootstrap_weight"] == pytest.approx(2.0)
    assert payload["bootstrap_update_scope"] == "live_segnet_scoped_late_feature_grid_fine_latent_head_rgb_1"
    smoke = payload["hinerv_pair_local_actuator_smoke"]
    assert smoke["schema"] == "hinerv_pair_local_actuator_smoke.v1"
    assert smoke["actuator"]["kind"] == "pair_local_latent_row"
    assert smoke["actuator"]["tensor_name"] == "latents_fine"
    assert smoke["gradient"]["updated_tensor_names"] == ["latents_fine"]
    assert smoke["pair_local_adapter_bytes"] > 0
    assert len(smoke["pair_local_adapter_sha256"]) == 64
    assert smoke["pair_local_grad_norm"] > 0.0
    assert smoke["pair_local_output_delta_l2"] > 0.0
    assert smoke["output_delta"]["pair_locality_verified"] is True
    assert smoke["output_delta"]["non_target_pair_output_delta_l2_max"] <= 1.0e-12
    assert smoke["execution_completed"] is False
    assert "hinerv_pair_local_output_delta_below_uint8_half_step" in smoke["blockers"]
    assert smoke["output_delta"]["receiver_uint8_crossing_potential"] is False
    assert smoke["state_restore"]["state_restored_after_smoke"] is True
    assert smoke["state_restore"]["original_row_sha256"] == (smoke["state_restore"]["restored_row_sha256"])
    assert smoke["state_restore"]["mutated_row_sha256"] == (smoke["pair_local_adapter_sha256"])
    assert smoke["section_output_delta_per_byte_rows"][0]["bytes"] == (smoke["pair_local_adapter_bytes"])
    assert smoke["section_output_delta_per_byte_rows"][0]["output_delta_l2_per_byte"] > 0.0
    assert smoke["section_value_per_byte_rows"] == []
    assert payload["pr95_scorer_atom_actuator_execution_evidence"] is None
    assert "latents_coarse" not in payload["archive_charged_decoder_tensors"]
    assert "head_rgb_0.*" not in payload["archive_charged_decoder_tensors"]
    for name in payload["bootstrap_update_applied_tensor_names"]:
        assert (
            name == "latents_fine"
            or name.startswith("latents_fine.")
            or name.startswith("feature_grids.")
            or name.startswith("fine_injector.")
            or name.startswith("head_rgb_1.")
        )
    assert payload["segnet_score_debt_preserving_acceptance"] is True
    assert payload["segnet_score_debt_rejected_step_count"] >= 0
    assert payload["receiver_quantum_acceptance_enabled"] is True
    assert payload["receiver_quantum_attempt_count"] >= payload["accepted_step_count"]
    assert payload["receiver_quantum_growth_attempt_count"] > 0
    assert payload["receiver_quantum_surface"] == "clamp_round_uint8_rgb_frame1"
    assert (
        payload["max_candidate_frame1_receiver_uint8_changed_count"]
        >= (payload["max_accepted_frame1_receiver_uint8_changed_count"])
    )
    assert payload["max_accepted_frame1_receiver_uint8_changed_count"] >= 0.0
    assert (
        payload["receiver_quantum_rejected_step_count"] + payload["hard_birth_argmax_progress_rejected_step_count"]
    ) > 0
    assert payload["hard_birth_argmax_progress_accepted_step_count"] == (payload["accepted_step_count"])
    assert payload["hard_birth_argmax_progress_rejected_step_count"] > 0
    assert payload["max_accepted_segnet_worst_debt_reduction"] >= 0.0
    assert payload["max_candidate_segnet_worst_debt_reduction"] >= (payload["max_accepted_segnet_worst_debt_reduction"])
    assert "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass" in payload["metrics_after"]
    assert "segnet_margin_bootstrap_class_1_score_weighted_unsolved_argmax_mass" in payload["metrics_after"]
    assert "segnet_hard_birth_bootstrap_class_1_seed_prob_deficit" in payload["metrics_after"]
    assert (
        payload["metrics_after"]["segnet_hard_birth_bootstrap_worst_loss_class_index"]
        == payload["metrics_after"]["segnet_hard_birth_bootstrap_worst_class_index"]
    )
    assert payload["runtime_sidecar_bytes"] == 0
    assert "head_rgb_1.*" in payload["archive_charged_decoder_tensors"]


@skip_no_mlx
def test_mlx_scorer_domain_bootstrap_hard_birth_can_actuate_without_margin_weight() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    class FakeSegNetTeacher:
        num_classes = 2

        def __init__(self, labels):
            self._labels = labels

        def teacher_argmax_for_indices(self, indices):
            return mx.take(self._labels, indices, axis=0)

        def teacher_logits_for_frames_nhwc01(self, frames):
            red = frames[..., 0]
            green = frames[..., 1]
            class0 = green - red
            class1 = red - green
            return mx.stack([class0, class1], axis=-1)

    mx.random.seed(11)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    ramp = mx.reshape(
        mx.linspace(0.05, 0.95, cfg.output_height * cfg.output_width),
        (1, cfg.output_height, cfg.output_width, 1),
    )
    target0 = mx.tile(
        mx.concatenate([0.2 + 0.1 * ramp, 0.3 + 0.2 * ramp, 0.1 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    target1 = mx.tile(
        mx.concatenate([0.15 + 0.1 * ramp, 0.55 - 0.1 * ramp, 0.1 + 0.1 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    labels_np = np.zeros(
        (cfg.num_pairs, cfg.output_height, cfg.output_width),
        dtype=np.int32,
    )
    labels_np[:, : max(1, cfg.output_height // 3), :] = 1
    labels = mx.array(labels_np)
    teacher = FakeSegNetTeacher(labels)

    model.initialize_output_head_bias_from_targets(target0, target1)
    payload = model.fit_scorer_domain_bootstrap_from_targets(
        target0,
        target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=labels,
        scorer_teacher=teacher,
        segnet_margin_bootstrap_weight=0.0,
        segnet_hard_birth_bootstrap_weight=2.0,
        segnet_hard_birth_bootstrap_min_ratio_floor=0.05,
        steps=3,
        learning_rate=2.5e-4,
        rgb_weight=0.1,
        yuv6_weight=0.0,
        temporal_delta_weight=0.0,
        grad_clip_max_norm=1.0,
    )

    assert payload["segnet_margin_bootstrap"]["enabled"] is False
    assert payload["segnet_hard_birth_bootstrap"]["enabled"] is True
    assert payload["segnet_hard_birth_bootstrap"]["worst_loss_selection"] == "score_weighted_unsolved_argmax_mass"
    assert payload["bootstrap_update_scope"] == "live_segnet_scoped_late_feature_grid_fine_latent_head_rgb_1"
    assert "latents_coarse" not in payload["archive_charged_decoder_tensors"]
    assert "head_rgb_0.*" not in payload["archive_charged_decoder_tensors"]
    assert payload["segnet_score_debt_preserving_acceptance"] is True
    assert payload["receiver_quantum_acceptance_enabled"] is True
    assert payload["receiver_quantum_attempt_count"] >= payload["accepted_step_count"]
    assert payload["receiver_quantum_growth_attempt_count"] > 0
    assert payload["receiver_quantum_surface"] == "clamp_round_uint8_rgb_frame1"
    assert payload["receiver_quantum_crossing_accepted_step_count"] == (payload["accepted_step_count"])
    assert (
        payload["max_candidate_frame1_receiver_uint8_delta_abs"]
        >= (payload["max_accepted_frame1_receiver_uint8_delta_abs"])
    )
    assert payload["max_accepted_frame1_receiver_uint8_changed_count"] >= 0.0
    assert payload["hard_birth_argmax_progress_accepted_step_count"] == (payload["accepted_step_count"])
    assert payload["hard_birth_argmax_progress_rejected_step_count"] > 0
    assert payload["max_accepted_segnet_worst_debt_reduction"] >= 0.0
    assert payload["max_candidate_segnet_worst_debt_reduction"] >= (payload["max_accepted_segnet_worst_debt_reduction"])
    assert payload["metrics_before"]["segnet_hard_birth_bootstrap_loss"] > 0.0
    assert payload["metrics_before"]["segnet_hard_birth_bootstrap_active_class_count"] > 0.0
    assert payload["segnet_hard_birth_bootstrap_loss_delta"] >= 0.0


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
def test_mlx_trilinear_upsample_matches_pytorch_reference() -> None:
    import mlx.core as mx
    import numpy as np

    from tac.substrates.hi_nerv.architecture import trilinear_upsample
    from tac.substrates.hi_nerv.mlx_renderer import trilinear_upsample_mlx

    rng = np.random.default_rng(41)
    grid_np = rng.normal(size=(4, 3, 3, 2)).astype("float32")
    pair_indices_np = np.asarray([0, 1, 4], dtype=np.int64)
    torch_ref = trilinear_upsample(
        torch.from_numpy(grid_np),
        torch.from_numpy(pair_indices_np),
        num_pairs=5,
        target_h=5,
        target_w=7,
        local_scale=3,
    ).numpy()
    mlx_out = np.asarray(
        trilinear_upsample_mlx(
            mx.array(grid_np),
            mx.array(pair_indices_np.astype(np.int32)),
            num_pairs=5,
            target_h=5,
            target_w=7,
            local_scale=3,
        ),
        dtype=np.float32,
    )

    np.testing.assert_allclose(mlx_out, torch_ref, atol=1.0e-6, rtol=1.0e-6)


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
@pytest.mark.parametrize("bits", [2, 4, 6, 7, 8])
def test_mlx_decoder_fake_quant_matches_decoder_state_codec_roundtrip(bits: int) -> None:
    import mlx.core as mx
    import numpy as np
    import torch

    from tac.substrates._shared.decoder_state_codec import (
        _decode_int8_record,
        _decode_nbit_record,
        _encode_int8_record,
        _encode_nbit_record,
    )
    from tac.substrates.hi_nerv.mlx_renderer import _fake_quant_symmetric_ste

    values_np = np.asarray(
        [
            [[-0.37, -0.19, 0.08], [0.41, -0.12, 0.27]],
            [[-0.31, 0.53, 0.02], [-0.44, 0.16, -0.07]],
            [[0.33, -0.25, 0.49], [-0.58, 0.04, 0.21]],
        ],
        dtype=np.float32,
    )
    tensor = torch.from_numpy(values_np)
    if bits == 8:
        reference = _decode_int8_record(_encode_int8_record(tensor)).numpy()
    else:
        reference = _decode_nbit_record(
            _encode_nbit_record(tensor, bits=bits),
            bits=bits,
        ).numpy()

    quantized = np.asarray(
        _fake_quant_symmetric_ste(mx.array(values_np), bits=bits),
        dtype=np.float32,
    )

    np.testing.assert_allclose(quantized, reference, atol=0.0, rtol=0.0)


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

    assert tuple(int(s) for s in quantized.shape) == tuple(int(s) for s in baseline.shape)
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
def test_mlx_decoder_fake_quant_forward_can_follow_stage_qat_flag() -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    model = HinervSubstrateMLX(_smoke_cfg())
    pair_indices = mx.array([0, 1, 2], dtype=mx.int32)
    baseline = model(pair_indices)
    mx.eval(baseline)

    model.configure_decoder_fake_quant_forward(
        enabled=True,
        quant_bits=2,
        stage_controlled=True,
    )
    assert model.decoder_fake_quant_forward_configured_enabled is True
    assert model.decoder_fake_quant_forward_stage_controlled is True
    assert model.decoder_fake_quant_forward_enabled is False
    inactive = model(pair_indices)
    mx.eval(inactive)
    assert float(mx.max(mx.abs(inactive - baseline))) < 1.0e-6

    model.notify_curriculum_stage(
        17,
        SimpleNamespace(name="qat_on", enable_qat=True),
    )
    active = model(pair_indices)
    mx.eval(active)
    assert model.decoder_fake_quant_forward_enabled is True
    assert model.decoder_fake_quant_forward_last_stage["forward_active"] is True
    assert model.decoder_fake_quant_forward_last_stage["source"] == "canonical_curriculum_stage"
    assert float(mx.max(mx.abs(active - baseline))) > 1.0e-7

    model.notify_pr95_stage_verdict(
        18,
        SimpleNamespace(
            descriptor_id="pr95_stage1",
            stage_index=1,
            qat_active=False,
        ),
    )
    restored = model(pair_indices)
    mx.eval(restored)
    assert model.decoder_fake_quant_forward_enabled is False
    assert model.decoder_fake_quant_forward_last_stage["source"] == ("pr95_faithful_stage_verdict")
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
    state = {name: torch.from_numpy(arr.copy()) for name, arr in mlx_model.export_state_dict().items()}
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
    state = {name: torch.from_numpy(arr.copy()) for name, arr in mlx_model.export_state_dict().items()}
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
    from tac.substrates.hi_nerv.archive import build_archive_section_telemetry, parse_archive
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
    assert build_archive_section_telemetry(blob)["decoder_codec"] == "int8_mixed"
    assert "_decoder_state_codec" not in arc.meta
    assert "_hi_nerv_bitstream_preparation" not in arc.meta
    assert "latents_coarse" not in arc.decoder_state_dict


def test_archive_quantizer_parity_receipt_localizes_action_bound_tensor_groups(
    tmp_path: Path,
) -> None:
    from tac.submission_archive import (
        MINIMAL_SINGLE_MEMBER_NAME,
        build_minimal_single_member_archive_bytes,
    )
    from tac.substrates.hi_nerv.archive_candidate import (
        HI_NERV_ARCHIVE_QUANTIZER_PARITY_RECEIPT_SCHEMA,
        build_hi_nerv_archive_quantizer_parity_receipt,
        pack_archive_from_exported_state_dict,
    )

    exportable = _exportable_torch_model()
    exported_state = exportable.export_state_dict()
    blob = pack_archive_from_exported_state_dict(
        exported_state_dict=exported_state,
        cfg=exportable.cfg,
        decoder_codec="int8_mixed",
        latent_codec="int4_packed",
    )
    archive_zip_bytes, _ = build_minimal_single_member_archive_bytes(blob)
    archive_path = tmp_path / "archive.zip"
    archive_path.write_bytes(archive_zip_bytes)
    npz_path = tmp_path / "hi_nerv_mlx_exported_state.npz"
    np.savez_compressed(npz_path, **exported_state)

    row = build_hi_nerv_archive_quantizer_parity_receipt(
        archive_path=archive_path,
        exported_state_npz_path=npz_path,
        live_receiver_export_parity={
            "schema": "hi_nerv_mlx_live_receiver_export_parity_proof.v1",
            "passed": False,
            "receiver_decode_passed": True,
            "max_abs_delta": 0.75,
        },
        live_to_parseback_audit={
            "schema": "hi_nerv_live_to_parseback_scorer_effect_delta_audit.v1",
            "action_id": "a" * 64,
            "support_sha256": "b" * 64,
            "decoded_action_sha256": "c" * 64,
            "parseback_scorer_effect_survived": False,
            "retention": {
                "live_wrong_to_target": 100,
                "fakequant_wrong_to_target": 90,
                "parseback_wrong_to_target": 2,
                "fakequant_wrong_to_target_retention_ratio": 0.9,
                "parseback_wrong_to_target_retention_ratio": 0.02,
            },
        },
    )

    assert row["schema"] == HI_NERV_ARCHIVE_QUANTIZER_PARITY_RECEIPT_SCHEMA
    assert row["action_id"] == "a" * 64
    assert row["support_sha256"] == "b" * 64
    assert row["parity_ready"] is True
    assert row["archive_member_name"] == MINIMAL_SINGLE_MEMBER_NAME
    groups = {item["tensor_group"]: item for item in row["tensor_group_rows"]}
    assert "latents_fine" in groups
    assert "head_rgb_1" in groups
    assert row["first_failed_surface"].endswith("_quantizer_delta")
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_hiv1_receiver_state_projection_uses_real_pack_parse_path() -> None:
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import (
        pack_archive_from_exported_state_dict,
        project_hi_nerv_hiv1_receiver_state,
    )

    exportable = _exportable_torch_model()
    exported_state = exportable.export_state_dict()
    kwargs = {
        "exported_state_dict": exported_state,
        "cfg": exportable.cfg,
        "decoder_codec": "int8_mixed",
        "latent_codec": "int8_raw",
    }

    projected, report = project_hi_nerv_hiv1_receiver_state(**kwargs)
    parsed = parse_archive(pack_archive_from_exported_state_dict(**kwargs))

    assert report["schema"] == "hi_nerv_hiv1_receiver_state_projection.v1"
    assert projected["latents_fine"].shape == exported_state["latents_fine"].shape
    assert np.array_equal(projected["latents_fine"], parsed.latents_fine.numpy())
    assert np.array_equal(
        projected["head_rgb_1.weight"],
        parsed.decoder_state_dict["head_rgb_1.weight"].numpy(),
    )
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False


def test_archive_candidate_keeps_bitstream_proof_out_of_charged_hiv1_meta() -> None:
    from tac.substrates.hi_nerv.archive import (
        build_archive_section_telemetry,
        pack_archive,
        parse_archive,
    )
    from tac.substrates.hi_nerv.archive_candidate import (
        pack_archive_from_exported_state_dict,
    )

    exportable = _exportable_torch_model()
    blob, bitstream_report = pack_archive_from_exported_state_dict(
        exported_state_dict=exportable.export_state_dict(),
        cfg=exportable.cfg,
        decoder_codec="fp16_enveloped",
        return_bitstream_report=True,
    )
    arc = parse_archive(blob)
    old_style_blob = pack_archive(
        arc.decoder_state_dict,
        arc.latents_coarse,
        arc.latents_mid,
        arc.latents_fine,
        {
            **arc.meta,
            "_hi_nerv_bitstream_preparation": bitstream_report,
        },
        decoder_codec="fp16_enveloped",
    )

    current_rows = {row["name"]: row for row in build_archive_section_telemetry(blob)["sections"]}
    old_rows = {row["name"]: row for row in build_archive_section_telemetry(old_style_blob)["sections"]}
    assert "_hi_nerv_bitstream_preparation" not in arc.meta
    assert old_rows["meta_json"]["bytes"] - current_rows["meta_json"]["bytes"] > 100


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
    blob, bitstream_report = pack_archive_from_exported_state_dict(
        exported_state_dict=exportable.export_state_dict(),
        cfg=exportable.cfg,
        decoder_codec="fp16_enveloped",
        decoder_weight_waterfill_plan={
            "schema": "nerv_decoder_weight_waterfill.v1",
            "family": "hi_nerv",
            "candidate_id": "unit",
            "compact_runner_launch_custody": {
                "schema": ("compact_hi_nerv_decoder_weight_waterfill_launch_custody.v1"),
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
        return_bitstream_report=True,
    )
    arc = parse_archive(blob)

    assert torch.count_nonzero(arc.decoder_state_dict["head_rgb_1.weight"]).item() == 0
    assert "_hi_nerv_bitstream_preparation" not in arc.meta
    waterfill = bitstream_report["decoder_weight_waterfill"]
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
    assert proof == bitstream_report["decoder_rendered_pixel_proof"]
    assert waterfill["rendered_pixel_proof_status"] == ("sampled_rendered_pixels_changed")
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
    blob, bitstream_report = pack_archive_from_exported_state_dict(
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
            "blockers": ["decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin"],
        },
        return_bitstream_report=True,
    )
    arc = parse_archive(blob)

    assert torch.count_nonzero(arc.decoder_state_dict["head_rgb_1.weight"]).item() > 0
    assert "_hi_nerv_bitstream_preparation" not in arc.meta
    waterfill = bitstream_report["decoder_weight_waterfill"]
    assert waterfill["method"] == "decoder_weight_waterfill_blocked"
    assert waterfill["changed_tensor_count"] == 0
    assert waterfill["applied_rows"] == []
    assert waterfill["actuation_blockers"] == ["decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin"]


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

    manifest_path = tmp_path / "hi_nerv_export" / "hprc_representation_spine_hi_nerv_manifest.json"
    package_path = tmp_path / "hi_nerv_export" / "archive_bound_candidate_adapter_package.json"
    npz_path = tmp_path / "hi_nerv_export" / "hi_nerv_mlx_exported_state.npz"
    npz_manifest_path = tmp_path / "hi_nerv_export" / "hi_nerv_mlx_exported_state_npz_manifest.json"
    bitstream_report_path = tmp_path / "hi_nerv_export" / "hi_nerv_bitstream_preparation.json"
    live_receiver_codec_portfolio_selection_path = (
        tmp_path / "hi_nerv_export" / "hi_nerv_live_receiver_codec_portfolio_selection.json"
    )
    live_receiver_export_parity_path = tmp_path / "hi_nerv_export" / "hi_nerv_mlx_live_receiver_export_parity.json"
    archive_section_telemetry_path = tmp_path / "hi_nerv_export" / "hi_nerv_archive_section_telemetry.json"
    proof_path = tmp_path / "hi_nerv_export" / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json"
    assert manifest_path.is_file()
    assert package_path.is_file()
    assert npz_path.is_file()
    assert npz_manifest_path.is_file()
    assert bitstream_report_path.is_file()
    assert live_receiver_codec_portfolio_selection_path.is_file()
    assert live_receiver_export_parity_path.is_file()
    assert archive_section_telemetry_path.is_file()
    assert proof_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    npz_manifest = json.loads(npz_manifest_path.read_text(encoding="utf-8"))
    bitstream_report = json.loads(bitstream_report_path.read_text(encoding="utf-8"))
    live_receiver_codec_portfolio_selection = json.loads(
        live_receiver_codec_portfolio_selection_path.read_text(encoding="utf-8")
    )
    live_receiver_export_parity = json.loads(live_receiver_export_parity_path.read_text(encoding="utf-8"))
    archive_section_telemetry = json.loads(archive_section_telemetry_path.read_text(encoding="utf-8"))
    assert manifest["family"] == "hi_nerv"
    with zipfile.ZipFile(archive_path) as zf:
        assert zf.namelist() == ["x"]
        assert zf.read("x") == (tmp_path / "hi_nerv_export" / "0.bin").read_bytes()
        assert "inflate.py" not in zf.namelist()
        assert "inflate.sh" not in zf.namelist()
        assert not any(name.startswith("src/") for name in zf.namelist())
    assert (tmp_path / "hi_nerv_export" / "submission" / "inflate.sh").is_file()
    assert (tmp_path / "hi_nerv_export" / "submission" / "inflate.py").is_file()
    assert proof["runtime_consumption_proof_ready"] is True
    assert proof["receiver_output_kind"] == "file"
    assert proof["receiver_output_retained"] is False
    assert package["receiver_proof"]["receiver_contract_satisfied"] is True
    assert npz_manifest["schema"] == "framework_agnostic_npz_bridge_manifest.v1"
    assert npz_manifest["consumption_recommended"] is True
    assert npz_manifest["artifact_sha256"]
    spine_extra = manifest["manifest"]["representation_spine"]["manifest_extra"]
    assert spine_extra["state_npz_bridge"]["artifact_sha256"] == (npz_manifest["artifact_sha256"])
    assert bitstream_report["decoder_codec_requested_by_export"] == "int8_mixed"
    assert bitstream_report["decoder_codec_selected_by_export"] == "int8_mixed"
    assert bitstream_report["decoder_codec"] == "int8_mixed"
    assert bitstream_report["requested_decoder_codec"] == "int8_mixed"
    assert bitstream_report["latent_codec"] == "int16_raw"
    assert bitstream_report["live_receiver_codec_portfolio_selection"] == (live_receiver_codec_portfolio_selection)
    assert spine_extra["hi_nerv_bitstream_preparation"] == bitstream_report
    assert (
        spine_extra["hi_nerv_live_receiver_codec_portfolio_selection_path"]
        == live_receiver_codec_portfolio_selection_path.as_posix()
    )
    assert spine_extra["hi_nerv_live_receiver_codec_portfolio_selection"] == (live_receiver_codec_portfolio_selection)
    assert spine_extra["hi_nerv_mlx_live_receiver_export_parity_path"] == live_receiver_export_parity_path.as_posix()
    assert spine_extra["hi_nerv_mlx_live_receiver_export_parity"] == live_receiver_export_parity
    assert live_receiver_export_parity["schema"] == ("hi_nerv_mlx_live_receiver_export_parity_proof.v1")
    assert live_receiver_export_parity["proof_status"] == "not_applicable_non_mlx_source_backend"
    assert live_receiver_export_parity["source_backend"] == "pytorch_test_export"
    assert live_receiver_export_parity["sampled_pair_count"] > 0
    assert "sampled_live_receiver_export_parity_not_full_video" in (live_receiver_export_parity["blockers"])
    assert (
        "hi_nerv_mlx_live_receiver_export_parity_not_applicable_non_mlx_source_backend"
        in live_receiver_export_parity["blockers"]
    )
    assert spine_extra["archive_section_telemetry_path"] == archive_section_telemetry_path.as_posix()
    assert spine_extra["archive_section_telemetry"] == archive_section_telemetry
    assert spine_extra["archive_zip_payload_only"] is True
    assert spine_extra["runtime_source_outside_archive_zip"] is True
    assert spine_extra["archive_zip_build"]["member_names"] == ["x"]
    assert spine_extra["archive_zip_build"]["payload_sha256"]
    assert archive_section_telemetry["archive_zip_bytes"] == archive_bytes
    assert {row["name"] for row in archive_section_telemetry["sections"]} == {
        "hiv1_header",
        "decoder_state",
        "latents_coarse",
        "latents_mid",
        "latents_fine",
        "meta_json",
    }
    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    assert row["runtime_adapter_ready"] is False
    assert row["contest_runtime_decoder_adapter_ready"] is False
    assert "hi_nerv_mlx_live_receiver_export_parity_not_applicable_non_mlx_source_backend" in row["blockers"]
    assert "runtime_adapter_ready_requires_no_extra_blockers" in row["blockers"]
    runtime_manifest = row["runtime_adapter_manifest"]
    assert runtime_manifest["runtime_adapter_ready"] is False
    assert runtime_manifest["contest_runtime_decoder_adapter_ready"] is False
    assert runtime_manifest["state_npz_bridge_manifest"]["artifact_sha256"] == (npz_manifest["artifact_sha256"])
    assert runtime_manifest["hi_nerv_bitstream_preparation"] == bitstream_report
    assert runtime_manifest["hi_nerv_bitstream_preparation_path"] == (bitstream_report_path.as_posix())
    assert runtime_manifest["hi_nerv_live_receiver_codec_portfolio_selection"] == (
        live_receiver_codec_portfolio_selection
    )
    assert (
        runtime_manifest["hi_nerv_live_receiver_codec_portfolio_selection_path"]
        == live_receiver_codec_portfolio_selection_path.as_posix()
    )
    assert runtime_manifest["hi_nerv_mlx_live_receiver_export_parity"] == (live_receiver_export_parity)
    assert runtime_manifest["hi_nerv_mlx_live_receiver_export_parity_path"] == (
        live_receiver_export_parity_path.as_posix()
    )
    assert runtime_manifest["archive_section_telemetry"] == archive_section_telemetry
    assert runtime_manifest["archive_zip_payload_only"] is True
    assert runtime_manifest["runtime_source_outside_archive_zip"] is True
    assert runtime_manifest["archive_zip_build"]["member_names"] == ["x"]
    assert runtime_manifest["archive_section_telemetry_path"] == (archive_section_telemetry_path.as_posix())
    portability = row["runtime_adapter_manifest"]["mlx_numpy_portability_contract"]
    assert portability["portability_status"] == ("numpy_export_bridge_ready_receiver_not_numpy")
    assert portability["training_backend"] == "pytorch_test_export"
    assert portability["numpy_array_export"] is True
    assert portability["canonical_npz_bridge_used"] is True
    assert portability["pure_numpy_inflate"] is False
    assert "torch" in portability["non_numpy_receiver_dependencies"]
    assert "training_backend_not_mlx" in portability["portability_blockers"]
    assert "inflate_runtime_not_pure_numpy" in portability["portability_blockers"]
    assert "canonical_npz_bridge_not_used_or_not_applicable" not in portability["portability_blockers"]


def test_archive_export_rent_gate_drops_unproven_target_region_action(tmp_path: Path) -> None:
    """EXPORT-BOUNDARY rent gate (2026-06-08 incident extinction): a target-region
    action with NO supplied CandidateActionEvaluation is FAIL-CLOSED dropped — the
    archive ships backend-only and the gate row records the drop."""
    import json

    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive
    from tac.substrates.hi_nerv.target_region_actions import (
        TARGET_REGION_ACTION_META_KEY,
        TargetRegionPixelAction,
        encode_target_region_actions_meta,
    )

    model = _exportable_torch_model()
    action_program = encode_target_region_actions_meta(
        [
            TargetRegionPixelAction(
                pair_index=0,
                frame_index=1,
                height=model.cfg.output_height,
                width=model.cfg.output_width,
                yx=np.array([[0, 0], [0, 1]], dtype=np.uint16),
                rgb_u8=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            )
        ]
    )

    out_dir = tmp_path / "hi_nerv_export_target_action"
    archive_path, _archive_sha, _archive_bytes = export_hi_nerv_mlx_archive(
        model,
        out_dir,
        repo_root=REPO_ROOT,
        decoder_codec="int8_mixed",
        retain_receiver_proof_output=False,
        source_backend="pytorch_test_export",
        target_region_action_program_base64=action_program,
    )

    inner = (out_dir / "0.bin").read_bytes()
    parsed = parse_archive(inner)
    assert archive_path.is_file()
    # Fail-closed: the unproven sidecar is dropped, archive ships backend-only.
    assert TARGET_REGION_ACTION_META_KEY not in dict(parsed.meta or {})
    gate = json.loads((out_dir / "hi_nerv_target_region_action_pack_rent_gate.json").read_text())
    assert gate["pack_action"] == "dropped_no_candidate_action_evaluation_supplied"
    assert gate["admit"] is False
    assert gate["action_packed"] is False
    assert gate["promotion_eligible"] is False
    assert gate["score_claim"] is False


def test_archive_export_charges_rent_paying_target_region_action_program(tmp_path: Path) -> None:
    """A rent-PAYING target-region action (exact ΔS < 0) IS packed + charged when a
    proving CandidateActionEvaluation is supplied at the export boundary."""
    from tac.substrates.hi_nerv.archive import (
        build_archive_section_telemetry,
        parse_archive,
    )
    from tac.substrates.hi_nerv.archive_candidate import (
        build_hi_nerv_candidate_action_evaluation,
        export_hi_nerv_mlx_archive,
        pack_archive_from_exported_state_dict,
    )
    from tac.substrates.hi_nerv.target_region_actions import (
        TARGET_REGION_ACTION_META_KEY,
        TargetRegionPixelAction,
        encode_target_region_actions_meta,
    )

    model = _exportable_torch_model()
    action_program = encode_target_region_actions_meta(
        [
            TargetRegionPixelAction(
                pair_index=0,
                frame_index=1,
                height=model.cfg.output_height,
                width=model.cfg.output_width,
                yx=np.array([[0, 0], [0, 1]], dtype=np.uint16),
                rgb_u8=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            )
        ]
    )

    # Build the proving evaluation against backend-only vs with-action payloads
    # packed from the same exported tensors (the action LOWERS d_seg => pays rent).
    exported = model.export_state_dict()
    backend = pack_archive_from_exported_state_dict(
        exported_state_dict=exported, cfg=model.cfg, decoder_codec="int8_mixed"
    )
    with_action = pack_archive_from_exported_state_dict(
        exported_state_dict=exported,
        cfg=model.cfg,
        decoder_codec="int8_mixed",
        target_region_action_program_base64=action_program,
    )
    evaluation = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend,
        with_action_archive=with_action,
        d_seg_base=0.10,
        d_pose_base=0.01,
        d_seg_with_action=0.05,
        d_pose_with_action=0.01,
    )
    assert evaluation.pays_rent is True

    out_dir = tmp_path / "hi_nerv_export_target_action_pays_rent"
    archive_path, _archive_sha, archive_bytes = export_hi_nerv_mlx_archive(
        model,
        out_dir,
        repo_root=REPO_ROOT,
        decoder_codec="int8_mixed",
        retain_receiver_proof_output=False,
        source_backend="pytorch_test_export",
        target_region_action_program_base64=action_program,
        target_region_action_evaluation=evaluation,
    )

    inner = (out_dir / "0.bin").read_bytes()
    parsed = parse_archive(inner)
    telemetry = build_archive_section_telemetry(
        inner,
        archive_zip_bytes=archive_bytes,
    )

    assert archive_path.is_file()
    assert parsed.meta[TARGET_REGION_ACTION_META_KEY] == action_program
    assert telemetry["target_region_actions"]["meta_key"] == TARGET_REGION_ACTION_META_KEY
    assert telemetry["target_region_actions"]["action_count"] == 1
    assert telemetry["target_region_actions"]["pixel_count"] == 2
    assert telemetry["target_region_actions"]["payload_bytes"] > 0
    assert telemetry["target_region_actions"]["charged_as_hiv1_meta_blob"] is True


def test_runner_selects_nested_support_moving_target_region_action() -> None:
    from tools.run_compact_renderer_mlx_spine_runner import (
        _select_target_region_action_program_from_birth_payload,
    )

    payload = {
        "schema": "hi_nerv_target_region_birth_payload.v1",
        "candidate_frontier_telemetry": {
            "masked_residual_oracle": {
                "candidates": [
                    {
                        "schema": "hi_nerv_target_region_masked_residual_oracle_candidate.v1",
                        "target_region_action_program_base64": "unsupported-but-cheap",
                        "target_support_moved": False,
                        "exact_delta_score_nonrate": -10.0,
                        "target_region_action_payload_bytes": 1,
                    },
                    {
                        "schema": "hi_nerv_target_region_masked_residual_oracle_candidate.v1",
                        "target_region_action_program_base64": "support-moving",
                        "target_support_moved": True,
                        "exact_delta_score_nonrate": -0.25,
                        "target_region_action_payload_bytes": 128,
                        "target_region_action_pixel_count": 9,
                        "target_region_action_section_telemetry": {
                            "support_sha256": "same-support",
                            "support_cardinality": 9,
                            "support_encoding": "explicit_yx_u16_coordinates",
                            "support_encoded_bytes": 36,
                        },
                        "direct_seg_wall_oracle": {"support_sha256": "same-support"},
                        "admission_decision": {"exact_score_decision": "accept"},
                    },
                ],
            },
        },
    }

    program, selection = _select_target_region_action_program_from_birth_payload(payload)

    assert program == "support-moving"
    assert selection is not None
    assert selection["candidate_count"] == 2
    assert selection["exact_delta_score_nonrate"] == -0.25
    assert selection["target_region_action_payload_bytes"] == 128
    assert selection["target_region_action_pixel_count"] == 9
    assert selection["target_region_action_support_sha256"] == "same-support"
    assert selection["direct_teacher_support_sha256"] == "same-support"
    assert selection["same_support_as_direct_teacher"] is True
    assert selection["promotion_eligible"] is False
    assert selection["ready_for_exact_eval_dispatch"] is False


def test_runner_refuses_target_region_action_with_direct_teacher_support_mismatch() -> None:
    from tools.run_compact_renderer_mlx_spine_runner import (
        _select_target_region_action_program_from_birth_payload,
    )

    payload = {
        "schema": "hi_nerv_target_region_birth_payload.v1",
        "candidate_frontier_telemetry": {
            "masked_residual_oracle": {
                "candidates": [
                    {
                        "schema": "hi_nerv_target_region_masked_residual_oracle_candidate.v1",
                        "target_region_action_program_base64": "support-moving-but-wrong-support",
                        "target_support_moved": True,
                        "exact_delta_score_nonrate": -10.0,
                        "target_region_action_payload_bytes": 128,
                        "target_region_action_pixel_count": 9,
                        "target_region_action_section_telemetry": {
                            "support_sha256": "sidecar-support",
                            "support_cardinality": 9,
                            "support_encoding": "explicit_yx_u16_coordinates",
                            "support_encoded_bytes": 36,
                        },
                        "direct_seg_wall_oracle": {"support_sha256": "direct-teacher-support"},
                        "admission_decision": {"exact_score_decision": "accept"},
                    },
                ],
            },
        },
    }

    program, selection = _select_target_region_action_program_from_birth_payload(payload)

    assert program is None
    assert selection is not None
    assert selection["selected_for_export"] is False
    assert selection["blockers"] == ["hi_nerv_target_region_action_no_total_score_improving_same_support_candidate"]
    assert selection["best_candidate_exact_accepted"] is True
    assert selection["best_candidate_support_moved"] is True
    assert selection["best_candidate_same_support_as_direct_teacher"] is False
    assert selection["best_candidate_action_support_sha256"] == "sidecar-support"
    assert selection["best_candidate_direct_teacher_support_sha256"] == "direct-teacher-support"
    assert selection["promotion_eligible"] is False
    assert selection["ready_for_exact_eval_dispatch"] is False


def test_runner_refuses_worsening_target_region_action_export() -> None:
    from tools.run_compact_renderer_mlx_spine_runner import (
        _select_target_region_action_program_from_birth_payload,
    )

    payload = {
        "schema": "hi_nerv_target_region_birth_payload.v1",
        "candidate_frontier_telemetry": {
            "masked_residual_oracle": {
                "candidates": [
                    {
                        "schema": "hi_nerv_target_region_masked_residual_oracle_candidate.v1",
                        "target_region_action_program_base64": "worsening-action",
                        "target_support_moved": True,
                        "exact_delta_score_nonrate": 16.4,
                        "target_region_action_section_telemetry": {"payload_bytes": 66348},
                        "admission_decision": {"exact_score_decision": "accept"},
                    }
                ],
            },
        },
    }

    program, selection = _select_target_region_action_program_from_birth_payload(payload)

    assert program is None
    assert selection is not None
    assert selection["selected_for_export"] is False
    assert selection["blockers"] == ["hi_nerv_target_region_action_no_total_score_improving_same_support_candidate"]
    assert selection["best_candidate_estimated_delta_score_total"] > 0.0
    assert selection["promotion_eligible"] is False
    assert selection["ready_for_exact_eval_dispatch"] is False


def test_archive_portfolio_auto_selects_receiver_parity_surviving_codec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tac.substrates.hi_nerv import archive_candidate
    from tac.substrates.hi_nerv.archive import build_archive_section_telemetry

    def _fake_parity_proof(*, archive_bytes, source_backend, **_kwargs):
        emitted = build_archive_section_telemetry(archive_bytes)["decoder_codec"]
        passed = emitted == "fp16_enveloped"
        return {
            "schema": "hi_nerv_mlx_live_receiver_export_parity_proof.v1",
            "source_backend": source_backend,
            "passed": passed,
            "proof_status": (
                "sampled_live_receiver_export_parity_passed" if passed else "sampled_live_receiver_export_parity_failed"
            ),
            "mean_abs_delta": 0.0 if passed else 0.1,
            "max_abs_delta": 0.0 if passed else 0.25,
            "sampled_pair_count": 1,
            "blockers": [] if passed else ["hi_nerv_mlx_live_receiver_export_parity_failed"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        archive_candidate,
        "_build_mlx_live_receiver_export_parity_proof",
        _fake_parity_proof,
    )

    archive_path, _archive_sha, _archive_bytes = archive_candidate.export_hi_nerv_mlx_archive(
        _exportable_torch_model(),
        tmp_path / "hi_nerv_portfolio_auto",
        repo_root=REPO_ROOT,
        decoder_codec="portfolio_auto",
        source_backend="mlx",
        retain_receiver_proof_output=False,
    )

    out_dir = archive_path.parent
    selection = json.loads(
        (out_dir / "hi_nerv_live_receiver_codec_portfolio_selection.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((out_dir / "hprc_representation_spine_hi_nerv_manifest.json").read_text(encoding="utf-8"))
    package = json.loads((out_dir / "archive_bound_candidate_adapter_package.json").read_text(encoding="utf-8"))

    assert selection["requested_decoder_codec"] == "portfolio_auto"
    assert selection["selected_decoder_codec"] == "fp16_enveloped"
    assert selection["selected_decoder_codec_requested"] == "fp16_enveloped"
    assert selection["selected_decoder_codec_effective"] == "fp16_enveloped"
    assert selection["selected_decoder_codec_source"] == "archive_section_telemetry"
    assert selection["selection_mode"] == "cheapest_live_receiver_parity_passing_codec"
    assert selection["parity_passing_candidate_count"] == 1
    assert all(
        row["decoder_codec_requested"] != "int2_mixed" or not row["live_receiver_export_parity_passed"]
        for row in selection["rows"]
        if row["status"] == "measured"
    )

    spine_extra = manifest["manifest"]["representation_spine"]["manifest_extra"]
    assert spine_extra["requested_decoder_codec"] == "portfolio_auto"
    assert spine_extra["decoder_codec"] == "fp16_enveloped"
    assert spine_extra["hi_nerv_bitstream_preparation"]["decoder_codec_requested_by_export"] == "portfolio_auto"
    assert spine_extra["hi_nerv_bitstream_preparation"]["decoder_codec_selected_by_export"] == "fp16_enveloped"
    assert spine_extra["hi_nerv_bitstream_preparation"]["decoder_codec"] == "fp16_enveloped"
    assert spine_extra["hi_nerv_bitstream_preparation"]["requested_decoder_codec"] == "portfolio_auto"
    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    runtime_manifest = row["runtime_adapter_manifest"]
    assert runtime_manifest["requested_decoder_codec"] == "portfolio_auto"
    assert runtime_manifest["decoder_codec"] == "fp16_enveloped"
    assert runtime_manifest["hi_nerv_live_receiver_codec_portfolio_selection"] == selection
    assert "hi_nerv_live_receiver_codec_portfolio_selected_codec_failed_parity" not in row["blockers"]


def test_archive_portfolio_auto_preserves_requested_vs_effective_alias_codec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tac.substrates.hi_nerv import archive_candidate
    from tac.substrates.hi_nerv.archive import build_archive_section_telemetry

    original_pack = archive_candidate.pack_archive_from_exported_state_dict

    def _aliasing_pack_archive_from_exported_state_dict(**kwargs):
        if kwargs.get("decoder_codec") == "int4_mixed":
            kwargs = {**kwargs, "decoder_codec": "int4_scale_bundled"}
        return original_pack(**kwargs)

    def _fake_parity_proof(*, archive_bytes, source_backend, **_kwargs):
        emitted = build_archive_section_telemetry(archive_bytes)["decoder_codec"]
        passed = emitted == "int4_scale_bundled"
        return {
            "schema": "hi_nerv_mlx_live_receiver_export_parity_proof.v1",
            "source_backend": source_backend,
            "passed": passed,
            "proof_status": (
                "sampled_live_receiver_export_parity_passed" if passed else "sampled_live_receiver_export_parity_failed"
            ),
            "mean_abs_delta": 0.0 if passed else 0.1,
            "max_abs_delta": 0.0 if passed else 0.25,
            "sampled_pair_count": 1,
            "blockers": [] if passed else ["hi_nerv_mlx_live_receiver_export_parity_failed"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        archive_candidate,
        "_LIVE_RECEIVER_CODEC_PORTFOLIO_CANDIDATES",
        ("int4_mixed",),
    )
    monkeypatch.setattr(
        archive_candidate,
        "pack_archive_from_exported_state_dict",
        _aliasing_pack_archive_from_exported_state_dict,
    )
    monkeypatch.setattr(
        archive_candidate,
        "_build_mlx_live_receiver_export_parity_proof",
        _fake_parity_proof,
    )

    archive_path, _archive_sha, _archive_bytes = archive_candidate.export_hi_nerv_mlx_archive(
        _exportable_torch_model(),
        tmp_path / "hi_nerv_portfolio_auto_alias",
        repo_root=REPO_ROOT,
        decoder_codec="portfolio_auto",
        source_backend="mlx",
        retain_receiver_proof_output=False,
    )

    out_dir = archive_path.parent
    selection = json.loads(
        (out_dir / "hi_nerv_live_receiver_codec_portfolio_selection.json").read_text(encoding="utf-8")
    )
    bitstream_report = json.loads((out_dir / "hi_nerv_bitstream_preparation.json").read_text(encoding="utf-8"))
    archive_section_telemetry = json.loads(
        (out_dir / "hi_nerv_archive_section_telemetry.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((out_dir / "hprc_representation_spine_hi_nerv_manifest.json").read_text(encoding="utf-8"))
    package = json.loads((out_dir / "archive_bound_candidate_adapter_package.json").read_text(encoding="utf-8"))

    assert selection["requested_decoder_codec"] == "portfolio_auto"
    assert selection["selected_decoder_codec_requested"] == "int4_mixed"
    assert selection["selected_decoder_codec_effective"] == "int4_scale_bundled"
    assert selection["selected_decoder_codec"] == "int4_scale_bundled"
    assert selection["selected_row"]["decoder_codec_requested"] == "int4_mixed"
    assert selection["selected_row"]["decoder_codec_emitted"] == "int4_scale_bundled"
    assert archive_section_telemetry["decoder_codec"] == "int4_scale_bundled"

    assert bitstream_report["requested_decoder_codec"] == "portfolio_auto"
    assert bitstream_report["decoder_codec_requested_by_export"] == "portfolio_auto"
    assert bitstream_report["decoder_codec_selected_by_export"] == ("int4_scale_bundled")
    assert bitstream_report["decoder_codec"] == "int4_scale_bundled"

    spine_extra = manifest["manifest"]["representation_spine"]["manifest_extra"]
    assert spine_extra["requested_decoder_codec"] == "portfolio_auto"
    assert spine_extra["decoder_codec"] == "int4_scale_bundled"
    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    runtime_manifest = row["runtime_adapter_manifest"]
    assert runtime_manifest["requested_decoder_codec"] == "portfolio_auto"
    assert runtime_manifest["decoder_codec"] == "int4_scale_bundled"


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
    from tac.substrates.hi_nerv.archive import (
        build_archive_section_telemetry,
        parse_archive,
        split_archive_sections,
    )
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

    manifest_path = tmp_path / "hi_nerv_export_brotli_latents" / "hprc_representation_spine_hi_nerv_manifest.json"
    proof_path = tmp_path / "hi_nerv_export_brotli_latents" / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json"
    assert archive_path.is_file()
    assert len(archive_sha) == 64
    assert archive_bytes == archive_path.stat().st_size
    assert manifest_path.is_file()
    assert proof_path.is_file()

    inner = (tmp_path / "hi_nerv_export_brotli_latents" / "0.bin").read_bytes()
    sections = split_archive_sections(inner)
    parsed = parse_archive(inner)
    telemetry = build_archive_section_telemetry(inner)
    rows = {row["name"]: row for row in telemetry["sections"]}
    assert parsed.latents_coarse.shape[0] == _exportable_torch_model().cfg.num_pairs
    assert sections.meta["_latent_codec"] == "int16_brotli_q11"
    assert "_latent_raw_bytes_coarse" not in sections.meta
    assert "_latent_coded_bytes_coarse" not in sections.meta
    assert rows["latents_coarse"]["raw_bytes"] == parsed.latents_coarse.numel() * 2
    assert rows["latents_coarse"]["bytes"] == len(sections.latents_coarse_blob)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    header = manifest["manifest"]["representation_spine"]["manifest_extra"]
    assert header["source_payload_kind"] == "hi_nerv_hiv1"
    assert any(row["name"] == "latents_rc" for row in manifest["manifest"]["sections"])
    assert any(row["name"] == "receiver_state" for row in manifest["manifest"]["sections"])
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert proof["runtime_consumption_proof_ready"] is True


def test_archive_export_emits_hprc_spine_for_high_byte_arithmetic_latents(
    tmp_path: Path,
) -> None:
    from tac.substrates.hi_nerv.archive import parse_archive, split_archive_sections
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive

    archive_path, archive_sha, archive_bytes = export_hi_nerv_mlx_archive(
        _exportable_torch_model(),
        tmp_path / "hi_nerv_export_hi_ac_latents",
        repo_root=REPO_ROOT,
        decoder_codec="int8_mixed",
        latent_codec="int16_hi_ac_brotli_q11",
        retain_receiver_proof_output=False,
        source_backend="pytorch_test_export",
    )

    export_dir = tmp_path / "hi_nerv_export_hi_ac_latents"
    manifest_path = export_dir / "hprc_representation_spine_hi_nerv_manifest.json"
    package_path = export_dir / "archive_bound_candidate_adapter_package.json"
    proof_path = export_dir / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json"

    assert archive_path.is_file()
    assert len(archive_sha) == 64
    assert archive_bytes == archive_path.stat().st_size
    inner = (export_dir / "0.bin").read_bytes()
    sections = split_archive_sections(inner)
    parsed = parse_archive(inner)
    assert sections.meta["_latent_codec"] == "int16_hi_ac_brotli_q11"
    assert sections.latents_coarse_blob.startswith(b"HILA1")
    assert parsed.latents_coarse.shape[0] == _exportable_torch_model().cfg.num_pairs

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    extra = manifest["manifest"]["representation_spine"]["manifest_extra"]
    assert extra["latent_codec"] == "int16_hi_ac_brotli_q11"
    runtime_manifest = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0][
        "runtime_adapter_manifest"
    ]
    assert runtime_manifest["latent_codec"] == "int16_hi_ac_brotli_q11"
    portability = runtime_manifest["mlx_numpy_portability_contract"]
    assert "constriction" in portability["non_numpy_receiver_dependencies"]
    assert proof["runtime_consumption_proof_ready"] is True


def test_archive_export_emits_receiver_bound_lower_bit_latents(
    tmp_path: Path,
) -> None:
    from tac.substrates.hi_nerv.archive import (
        build_archive_section_telemetry,
        parse_archive,
        split_archive_sections,
    )
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive

    archive_path, archive_sha, archive_bytes = export_hi_nerv_mlx_archive(
        _exportable_torch_model(),
        tmp_path / "hi_nerv_export_int4_latents",
        repo_root=REPO_ROOT,
        decoder_codec="int8_mixed",
        latent_codec="int4_packed_brotli_q11",
        retain_receiver_proof_output=False,
        source_backend="pytorch_test_export",
    )

    export_dir = tmp_path / "hi_nerv_export_int4_latents"
    manifest_path = export_dir / "hprc_representation_spine_hi_nerv_manifest.json"
    package_path = export_dir / "archive_bound_candidate_adapter_package.json"
    proof_path = export_dir / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json"

    assert archive_path.is_file()
    assert len(archive_sha) == 64
    assert archive_bytes == archive_path.stat().st_size
    inner = (export_dir / "0.bin").read_bytes()
    sections = split_archive_sections(inner)
    parsed = parse_archive(inner)
    telemetry = build_archive_section_telemetry(inner)
    rows = {row["name"]: row for row in telemetry["sections"]}
    assert sections.meta["_latent_codec"] == "int4_packed_brotli_q11"
    assert parsed.latents_coarse.shape[0] == _exportable_torch_model().cfg.num_pairs
    assert rows["latents_coarse"]["quant_bits"] == 4
    assert rows["latents_coarse"]["raw_bytes"] == (parsed.latents_coarse.numel() * 4 + 7) // 8

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    extra = manifest["manifest"]["representation_spine"]["manifest_extra"]
    assert extra["latent_codec"] == "int4_packed_brotli_q11"
    runtime_manifest = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0][
        "runtime_adapter_manifest"
    ]
    assert runtime_manifest["latent_codec"] == "int4_packed_brotli_q11"
    assert runtime_manifest["archive_section_telemetry"]["latent_codec"] == ("int4_packed_brotli_q11")
    portability = runtime_manifest["mlx_numpy_portability_contract"]
    assert "constriction" not in portability["non_numpy_receiver_dependencies"]
    assert "lossy relative to the int16 latent quantizer" in portability["notes"]
    assert proof["runtime_consumption_proof_ready"] is True


def test_lossy_latent_codec_selection_treats_decode_survival_as_measured_delta() -> None:
    from tac.substrates.hi_nerv import archive_candidate

    selection = archive_candidate._build_single_codec_portfolio_selection_report(
        requested_decoder_codec="int4_mixed",
        selected_decoder_codec="int4_mixed",
        latent_codec="int4_packed_brotli_q11",
        archive_bytes=196_951,
        payload_bytes=196_966,
        archive_zip_build={"archive_sha256": "0" * 64},
        live_receiver_export_parity={
            "passed": False,
            "receiver_decode_passed": True,
            "proof_status": "sampled_live_receiver_export_lossy_latent_delta_measured",
            "mean_abs_delta": 0.000954,
            "max_abs_delta": 0.0174,
            "blockers": [
                "sampled_live_receiver_export_parity_not_full_video",
                "contest_cpu_cuda_exact_eval_not_executed",
                "scorer_replay_not_executed",
            ],
        },
        hard_byte_ceiling=285_000,
    )

    row = selection["selected_row"]
    assert row["lossy_latent_codec"] is True
    assert row["live_receiver_export_parity_passed"] is False
    assert row["live_receiver_export_receiver_survived"] is True
    assert selection["receiver_surviving_candidate_count"] == 1
    assert "hi_nerv_live_receiver_codec_portfolio_selected_codec_failed_parity" not in selection["blockers"]
    assert "hi_nerv_live_receiver_codec_portfolio_selected_not_receiver_surviving" not in selection["blockers"]


def test_archive_bound_package_wrapper_preserves_high_byte_latent_codec(
    tmp_path: Path,
) -> None:
    from tac.substrates.hi_nerv.archive_candidate import (
        export_hi_nerv_mlx_archive_bound_candidate_package,
    )

    package = export_hi_nerv_mlx_archive_bound_candidate_package(
        _exportable_torch_model(),
        tmp_path / "hi_nerv_package_hi_ac_latents",
        repo_root=REPO_ROOT,
        decoder_codec="int8_mixed",
        latent_codec="int16_hi_ac_brotli_q11",
        source_backend="pytorch_test_export",
    )

    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    runtime_manifest = row["runtime_adapter_manifest"]
    assert runtime_manifest["latent_codec"] == "int16_hi_ac_brotli_q11"
    assert runtime_manifest["archive_section_telemetry"]["latent_codec"] == ("int16_hi_ac_brotli_q11")
    portability = runtime_manifest["mlx_numpy_portability_contract"]
    assert "constriction" in portability["non_numpy_receiver_dependencies"]


def test_strip_target_region_action_from_archive_payload_is_lossless_double_win() -> None:
    """Canonical remediation for the 2026-06-08 harmful-sidecar incident: stripping
    the target-region action losslessly recovers the backend-only archive (action
    meta removed, decoder/latent tensors byte-identical, bytes reduced)."""
    import numpy as np

    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import (
        pack_archive_from_exported_state_dict,
        strip_target_region_action_from_archive_payload,
    )
    from tac.substrates.hi_nerv.target_region_actions import (
        TARGET_REGION_ACTION_META_KEY,
        TargetRegionPixelAction,
        encode_target_region_actions_meta,
    )

    exportable = _exportable_torch_model()
    exported = exportable.export_state_dict()
    cfg = exportable.cfg
    codecs = {"decoder_codec": "int8_mixed", "latent_codec": "int8_brotli_q11"}

    no_action_blob = pack_archive_from_exported_state_dict(
        exported_state_dict=exported, cfg=cfg, **codecs
    )

    action = TargetRegionPixelAction(
        pair_index=0,
        frame_index=1,
        height=int(cfg.output_height),
        width=int(cfg.output_width),
        yx=np.array([[0, 0], [1, 1], [2, 3]], dtype=np.uint16),
        rgb_u8=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
    )
    program_b64 = encode_target_region_actions_meta([action])
    with_action_blob = pack_archive_from_exported_state_dict(
        exported_state_dict=exported,
        cfg=cfg,
        target_region_action_program_base64=program_b64,
        **codecs,
    )
    assert TARGET_REGION_ACTION_META_KEY in dict(parse_archive(with_action_blob).meta or {})

    stripped = strip_target_region_action_from_archive_payload(with_action_blob, **codecs)

    # (1) action meta removed.
    assert TARGET_REGION_ACTION_META_KEY not in dict(parse_archive(stripped).meta or {})
    # (2) byte DOUBLE-WIN: stripped is smaller than with-action (rate term lowered).
    assert len(stripped) < len(with_action_blob)
    # (3) lossless backend: decoder + latent tensors byte-identical to the
    #     no-action pack (the re-quant is idempotent).
    a_strip = parse_archive(stripped)
    a_none = parse_archive(no_action_blob)
    assert np.array_equal(
        a_strip.decoder_state_dict["head_rgb_1.weight"].numpy(),
        a_none.decoder_state_dict["head_rgb_1.weight"].numpy(),
    )
    assert np.array_equal(a_strip.latents_fine.numpy(), a_none.latents_fine.numpy())
    # (4) no-op on an already-backend-only payload.
    assert strip_target_region_action_from_archive_payload(no_action_blob, **codecs) == no_action_blob


# --- Vehicle 1 (V1-EXPORT): pays-rent law at the export/pack boundary ---


def _rent_gate_fixture():
    """Return (cfg, exported, backend_blob, with_action_blob, codecs) sharing the
    SAME exported tensors so a CandidateActionEvaluation can be bound to real
    archive bytes (no synthetic fixtures, real HIV1 pack/parse)."""
    import numpy as np

    from tac.substrates.hi_nerv.archive_candidate import pack_archive_from_exported_state_dict
    from tac.substrates.hi_nerv.target_region_actions import (
        TargetRegionPixelAction,
        encode_target_region_actions_meta,
    )

    exportable = _exportable_torch_model()
    exported = exportable.export_state_dict()
    cfg = exportable.cfg
    codecs = {"decoder_codec": "int8_mixed", "latent_codec": "int8_brotli_q11"}
    backend_blob = pack_archive_from_exported_state_dict(exported_state_dict=exported, cfg=cfg, **codecs)
    action = TargetRegionPixelAction(
        pair_index=0,
        frame_index=1,
        height=int(cfg.output_height),
        width=int(cfg.output_width),
        yx=np.array([[0, 0], [1, 1], [2, 3]], dtype=np.uint16),
        rgb_u8=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
    )
    program_b64 = encode_target_region_actions_meta([action])
    with_action_blob = pack_archive_from_exported_state_dict(
        exported_state_dict=exported,
        cfg=cfg,
        target_region_action_program_base64=program_b64,
        **codecs,
    )
    return cfg, exported, backend_blob, with_action_blob, codecs


def test_build_hi_nerv_candidate_action_evaluation_binds_real_bytes_and_uses_exact_contest_score() -> None:
    """Invariant (2)+(3): the builder binds base/with-action sha256 + byte counts
    from the REAL archive bytes and the exact contest score comes from
    CandidateActionEvaluation (no hand-approximation)."""
    import math

    from tac.optimization.bayesian_experimental_design import contest_score
    from tac.repo_io import sha256_bytes
    from tac.substrates.hi_nerv.archive_candidate import build_hi_nerv_candidate_action_evaluation

    _cfg, _exported, backend_blob, with_action_blob, _codecs = _rent_gate_fixture()
    evaluation = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend_blob,
        with_action_archive=with_action_blob,
        d_seg_base=0.10,
        d_pose_base=0.02,
        d_seg_with_action=0.06,
        d_pose_with_action=0.02,
    )
    # sha256 + byte counts are derived from the REAL bytes (base binding).
    assert evaluation.base_archive_sha256 == sha256_bytes(backend_blob)
    assert evaluation.with_action_archive_sha256 == sha256_bytes(with_action_blob)
    assert evaluation.bytes_base == len(backend_blob)
    assert evaluation.bytes_with_action == len(with_action_blob)
    # Exact contest score terms match contest_score() — NOT a hand-approximation.
    assert evaluation.score_base == contest_score(0.10, 0.02, len(backend_blob))
    assert evaluation.score_with_action == contest_score(0.06, 0.02, len(with_action_blob))
    # ΔS sign: d_seg dropped 0.04 (=> -4.0 seg term) dominates the tiny byte add.
    assert evaluation.delta_score_total == pytest.approx(
        contest_score(0.06, 0.02, len(with_action_blob)) - contest_score(0.10, 0.02, len(backend_blob))
    )
    assert evaluation.delta_score_total < 0.0
    assert math.isclose(evaluation.delta_score_nonrate, -4.0, abs_tol=1e-9)


def test_candidate_action_evaluation_pays_rent_iff_delta_score_negative() -> None:
    """Invariant (1): pays_rent is True iff delta_score_total < 0 (and survives)."""
    from tac.substrates.hi_nerv.archive_candidate import build_hi_nerv_candidate_action_evaluation

    _cfg, _exported, backend_blob, with_action_blob, _codecs = _rent_gate_fixture()
    # Score-lowering action -> pays rent.
    paying = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend_blob,
        with_action_archive=with_action_blob,
        d_seg_base=0.10,
        d_pose_base=0.01,
        d_seg_with_action=0.05,
        d_pose_with_action=0.01,
    )
    assert paying.pays_rent is True
    assert paying.delta_score_total < 0.0
    # Score-raising action (worse d_seg + extra bytes) -> does NOT pay rent.
    harmful = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend_blob,
        with_action_archive=with_action_blob,
        d_seg_base=0.05,
        d_pose_base=0.01,
        d_seg_with_action=0.12,
        d_pose_with_action=0.01,
    )
    assert harmful.pays_rent is False
    assert harmful.delta_score_total > 0.0
    # A score-lowering action whose scorer effect did NOT survive parse-back is
    # rejected even though delta_score < 0.
    not_survived = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend_blob,
        with_action_archive=with_action_blob,
        d_seg_base=0.10,
        d_pose_base=0.01,
        d_seg_with_action=0.05,
        d_pose_with_action=0.01,
        scorer_effect_survived=False,
    )
    assert not_survived.delta_score_total < 0.0
    assert not_survived.pays_rent is False


def test_pack_gate_keeps_rent_paying_action() -> None:
    """A rent-paying action survives into the packed payload (action meta kept)."""
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import (
        build_hi_nerv_candidate_action_evaluation,
        pack_hi_nerv_target_region_action_payload_requiring_pays_rent,
    )
    from tac.substrates.hi_nerv.target_region_actions import TARGET_REGION_ACTION_META_KEY

    _cfg, _exported, backend_blob, with_action_blob, codecs = _rent_gate_fixture()
    evaluation = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend_blob,
        with_action_archive=with_action_blob,
        d_seg_base=0.10,
        d_pose_base=0.01,
        d_seg_with_action=0.05,
        d_pose_with_action=0.01,
    )
    packed, row = pack_hi_nerv_target_region_action_payload_requiring_pays_rent(
        with_action_payload=with_action_blob,
        evaluation=evaluation,
        **codecs,
    )
    # Action genuinely retained (decode the real payload, not a constant).
    assert TARGET_REGION_ACTION_META_KEY in dict(parse_archive(packed).meta or {})
    assert packed == with_action_blob
    assert row["admit"] is True
    assert row["pays_rent"] is True
    assert row["action_packed"] is True
    assert row["pack_action"] == "kept_action_pays_rent"
    assert row["schema"] == "hi_nerv_target_region_action_pack_rent_gate.v1"
    assert row["candidate_action_evaluation"]["schema"] == "hi_nerv_candidate_action_evaluation.v1"


def test_pack_gate_drops_structurally_valid_action_that_does_not_pay_rent() -> None:
    """Invariant (4): a structurally-VALID action that does not pay rent is REJECTED
    (stripped), so the archive ships backend-only — smaller AND action removed."""
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import (
        build_hi_nerv_candidate_action_evaluation,
        pack_hi_nerv_target_region_action_payload_requiring_pays_rent,
    )
    from tac.substrates.hi_nerv.target_region_actions import TARGET_REGION_ACTION_META_KEY

    _cfg, _exported, backend_blob, with_action_blob, codecs = _rent_gate_fixture()
    # The with-action blob is a structurally-VALID receiver payload (it parses).
    assert TARGET_REGION_ACTION_META_KEY in dict(parse_archive(with_action_blob).meta or {})
    harmful = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend_blob,
        with_action_archive=with_action_blob,
        d_seg_base=0.05,
        d_pose_base=0.01,
        d_seg_with_action=0.12,
        d_pose_with_action=0.01,
    )
    assert harmful.pays_rent is False
    packed, row = pack_hi_nerv_target_region_action_payload_requiring_pays_rent(
        with_action_payload=with_action_blob,
        evaluation=harmful,
        **codecs,
    )
    # Action stripped: meta gone, payload smaller (rate term lowered = double win).
    assert TARGET_REGION_ACTION_META_KEY not in dict(parse_archive(packed).meta or {})
    assert len(packed) < len(with_action_blob)
    assert row["admit"] is False
    assert row["action_packed"] is False
    assert row["pack_action"] == "dropped_action_does_not_pay_rent"
    assert "hi_nerv_target_region_action_does_not_pay_rent" in row["blockers"]


def test_pack_gate_drops_stale_evaluation_against_phantom_base() -> None:
    """Invariant (3) anti-drift: an evaluation measured against a DIFFERENT base is
    STALE and the action is dropped — the phantom-base failure mode is refused."""
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import (
        build_hi_nerv_candidate_action_evaluation,
        pack_hi_nerv_target_region_action_payload_requiring_pays_rent,
    )
    from tac.substrates.hi_nerv.target_region_actions import TARGET_REGION_ACTION_META_KEY

    _cfg, _exported, backend_blob, with_action_blob, codecs = _rent_gate_fixture()
    paying = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend_blob,
        with_action_archive=with_action_blob,
        d_seg_base=0.10,
        d_pose_base=0.01,
        d_seg_with_action=0.05,
        d_pose_with_action=0.01,
    )
    assert paying.pays_rent is True
    # Current base differs from the base the evaluation was measured against.
    packed, row = pack_hi_nerv_target_region_action_payload_requiring_pays_rent(
        with_action_payload=with_action_blob,
        evaluation=paying,
        current_base_archive_sha256="0" * 64,
        **codecs,
    )
    assert TARGET_REGION_ACTION_META_KEY not in dict(parse_archive(packed).meta or {})
    assert row["admit"] is False
    assert row["stale_for_current_base"] is True
    assert row["pack_action"] == "stale_evaluation_for_current_base"
    assert "hi_nerv_target_region_action_evaluation_stale_for_current_base" in row["blockers"]


def test_pack_gate_drops_evaluation_bound_to_different_with_action_bytes() -> None:
    """Anti-drift binding: an evaluation whose with-action sha256 does NOT match the
    bytes being packed is refused (cannot leak its verdict onto other bytes)."""
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import (
        build_hi_nerv_candidate_action_evaluation,
        pack_hi_nerv_target_region_action_payload_requiring_pays_rent,
    )
    from tac.substrates.hi_nerv.target_region_actions import TARGET_REGION_ACTION_META_KEY

    _cfg, _exported, backend_blob, with_action_blob, codecs = _rent_gate_fixture()
    # Evaluation bound to backend_blob as if it were the with-action archive.
    mismatched = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend_blob,
        with_action_archive=backend_blob,
        d_seg_base=0.10,
        d_pose_base=0.01,
        d_seg_with_action=0.05,
        d_pose_with_action=0.01,
    )
    assert mismatched.pays_rent is True
    packed, row = pack_hi_nerv_target_region_action_payload_requiring_pays_rent(
        with_action_payload=with_action_blob,
        evaluation=mismatched,
        bind_with_action_payload_sha256=True,
        **codecs,
    )
    assert TARGET_REGION_ACTION_META_KEY not in dict(parse_archive(packed).meta or {})
    assert row["with_action_payload_sha256_mismatch"] is True
    assert row["pack_action"] == "stale_evaluation_with_action_payload_sha256_mismatch"


def test_pack_gate_no_action_payload_is_passthrough_backend_only() -> None:
    """A backend-only payload (no action) passes through unchanged regardless of the
    evaluation verdict — there is nothing to drop."""
    from tac.substrates.hi_nerv.archive_candidate import (
        build_hi_nerv_candidate_action_evaluation,
        pack_hi_nerv_target_region_action_payload_requiring_pays_rent,
    )

    _cfg, _exported, backend_blob, with_action_blob, codecs = _rent_gate_fixture()
    harmful = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend_blob,
        with_action_archive=with_action_blob,
        d_seg_base=0.05,
        d_pose_base=0.01,
        d_seg_with_action=0.12,
        d_pose_with_action=0.01,
    )
    packed, row = pack_hi_nerv_target_region_action_payload_requiring_pays_rent(
        with_action_payload=backend_blob,
        evaluation=harmful,
        bind_with_action_payload_sha256=False,
        **codecs,
    )
    assert packed == backend_blob
    assert row["with_action_payload_has_action"] is False
    assert row["pack_action"] == "no_action_present"


def test_pack_gate_row_is_false_authority_non_promotable() -> None:
    """Invariant (5): the gate row carries false-authority markers — bytes are
    planning-control evidence, never a score/promotion claim."""
    from tac.substrates.hi_nerv.archive_candidate import (
        build_hi_nerv_candidate_action_evaluation,
        pack_hi_nerv_target_region_action_payload_requiring_pays_rent,
    )

    _cfg, _exported, backend_blob, with_action_blob, codecs = _rent_gate_fixture()
    evaluation = build_hi_nerv_candidate_action_evaluation(
        base_archive=backend_blob,
        with_action_archive=with_action_blob,
        d_seg_base=0.10,
        d_pose_base=0.01,
        d_seg_with_action=0.05,
        d_pose_with_action=0.01,
    )
    _packed, row = pack_hi_nerv_target_region_action_payload_requiring_pays_rent(
        with_action_payload=with_action_blob,
        evaluation=evaluation,
        **codecs,
    )
    assert row["promotion_eligible"] is False
    assert row["score_claim"] is False
    assert row["promotable"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["candidate_action_evaluation"]["promotion_eligible"] is False
    assert (
        "hi_nerv_candidate_action_evaluation_is_planning_control_false_authority" in row["blockers"]
    )


# -----------------------------------------------------------------------------
# F1: PR95 bilinear-skip + terminal refine HF-residual gate (gated default-OFF).
# deep_hinerv_snerv_fidelity_review H1 — the missing residual path that the
# whole NeRV fleet shares; routed through the canonical cross-backend primitive.
# -----------------------------------------------------------------------------


@skip_no_mlx
def test_bilinear_skip_off_is_byte_identical_no_new_params() -> None:
    """Default cfg (use_bilinear_skip=False) creates NO skip/refine modules — the
    historical skip-free carrier is preserved byte-for-byte (zero regression)."""
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = _smoke_cfg()
    assert cfg.use_bilinear_skip is False
    model = HinervSubstrateMLX(cfg)
    assert not hasattr(model.blocks[0], "skip"), "OFF block must not create a skip conv"
    assert not hasattr(model, "refine"), "OFF renderer must not create a refine conv"
    # export works on the default carrier.
    sd = model.export_state_dict()
    assert not any(".skip." in k or k.startswith("refine") for k in sd), "OFF export must not emit skip/refine keys"
    mx.eval(model.parameters())


@skip_no_mlx
def test_bilinear_skip_on_forwards_adds_params_and_variance() -> None:
    """With use_bilinear_skip=True the block gains a skip conv + the renderer a
    refine conv; the forward runs at the SAME output shape, adds parameters, and
    injects spatial variance the skip-free carrier lacks (the mean-field escape)."""
    from dataclasses import replace

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    base = _smoke_cfg()
    off = HinervSubstrateMLX(base)
    on = HinervSubstrateMLX(replace(base, use_bilinear_skip=True))
    mx.eval(off.parameters(), on.parameters())

    def nparams(m):
        return int(sum(int(np.asarray(v).size) for _, v in tree_flatten(m.parameters())))

    assert hasattr(on.blocks[0], "skip") and hasattr(on, "refine")
    assert nparams(on) > nparams(off), "skip + refine must add parameters"

    r0_off, _ = off.reconstruct_pair(mx.array([0]))
    r0_on, _ = on.reconstruct_pair(mx.array([0]))
    mx.eval(r0_off, r0_on)
    assert tuple(r0_on.shape) == tuple(r0_off.shape), "skip must not change output shape"
    # At init the skip-free decoder is ~constant (one-class flat); the skip path
    # injects markedly more spatial variance (the d_seg~0.50 mean-field escape).
    std_off = float(np.asarray(r0_off).std())
    std_on = float(np.asarray(r0_on).std())
    assert std_on > std_off, f"skip-on init variance {std_on} must exceed skip-off {std_off}"


@skip_no_mlx
def test_bilinear_skip_on_export_is_fail_closed_research_only() -> None:
    """Archive export of the skip/refine weights is a gated F1 follow-up (needs
    export layout + PyTorch-oracle parity). Until then it MUST fail closed — no
    silent incomplete archive (NO FAKE IMPLEMENTATIONS)."""
    from dataclasses import replace

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    on = HinervSubstrateMLX(replace(_smoke_cfg(), use_bilinear_skip=True))
    with pytest.raises(NotImplementedError):
        on.export_state_dict()
