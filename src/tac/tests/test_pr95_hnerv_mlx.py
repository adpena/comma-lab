# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    EXACT_READINESS_REFUSAL_BLOCKERS,
    FALSE_AUTHORITY,
    HNeRVDecoderMLX,
    HNeRVSyntheticTrainingBundleMLX,
    bilinear_resize2x_align_corners_false_nhwc,
    partition_pr95_mlx_parameter_names,
    pixel_shuffle_2x_nhwc,
    pytorch_state_dict_from_mlx,
    run_pr95_mlx_synthetic_timing_smoke,
    write_pr95_mlx_byte_closed_smoke_archive,
    zeropower_via_newtonschulz5_mlx,
)
from tac.optimization.local_training_runtime_profile import (  # noqa: E402
    normalize_runtime_profile_observation,
)


def _assert_false_authority(payload: dict) -> None:
    for key in FALSE_AUTHORITY:
        assert payload[key] is False


def test_pixel_shuffle_2x_nhwc_matches_pr95_layout() -> None:
    x = mx.array([[[[0.0, 1.0, 2.0, 3.0]]]])

    y = pixel_shuffle_2x_nhwc(x)
    mx.eval(y)

    assert y.shape == (1, 2, 2, 1)
    np.testing.assert_array_equal(np.asarray(y)[0, :, :, 0], np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_bilinear_resize2x_matches_align_corners_false_scale_two() -> None:
    x = mx.array([[[[0.0], [4.0]]]])

    y = bilinear_resize2x_align_corners_false_nhwc(x)
    mx.eval(y)

    assert y.shape == (1, 2, 4, 1)
    expected = np.array(
        [
            [
                [0.0],
                [1.0],
                [3.0],
                [4.0],
            ],
            [
                [0.0],
                [1.0],
                [3.0],
                [4.0],
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(y)[0], expected, atol=1e-6)


def test_decoder_output_shape_and_pytorch_state_names() -> None:
    model = HNeRVDecoderMLX(base_channels=4)
    z = mx.zeros((1, 28))

    y = model(z)
    mx.eval(y)
    exported = pytorch_state_dict_from_mlx(model)

    assert y.shape == (1, 2, 3, 384, 512)
    assert exported["stem.weight"].shape == (4 * 6 * 8, 28)
    assert exported["blocks.0.weight"].shape == (16, 4, 3, 3)
    assert exported["rgb_0.weight"].shape == (3, 2, 3, 3)
    assert "skips.0.weight" not in exported
    assert "skips.2.weight" in exported


def test_pr95_stage8_partition_keeps_muon_off_latents_stem_and_rgb_heads() -> None:
    bundle = HNeRVSyntheticTrainingBundleMLX(latent_count=2, base_channels=36, seed=3)

    split = partition_pr95_mlx_parameter_names(bundle.parameters())

    assert len(split["muon"]) == 11
    assert "latents" in split["adamw"]
    assert "decoder.stem.weight" in split["adamw"]
    assert "decoder.rgb_0.weight" in split["adamw"]
    assert "decoder.rgb_1.weight" in split["adamw"]
    assert "decoder.blocks.0.conv.weight" in split["muon"]
    assert "decoder.refine0.weight" in split["muon"]
    assert all("rgb_" not in name for name in split["muon"])


def test_newton_schulz5_preserves_shape_and_finite_values() -> None:
    gradient = mx.array(np.arange(12, dtype=np.float32).reshape(3, 4) / 10.0)

    update = zeropower_via_newtonschulz5_mlx(gradient)
    mx.eval(update)

    assert update.shape == gradient.shape
    assert np.isfinite(np.asarray(update)).all()


def test_synthetic_timing_smoke_emits_runtime_profile_and_refusal() -> None:
    manifest = run_pr95_mlx_synthetic_timing_smoke(
        stage_index=8,
        steps=1,
        batch_size=1,
        synthetic_pairs=1,
        seed=5,
        base_channels=4,
    )

    normalized = normalize_runtime_profile_observation(manifest["runtime_profile"])

    assert manifest["schema"] == "pr95_hnerv_mlx_timing_smoke_manifest_v1"
    assert manifest["stage_module"] == "stage8_muon_finetune"
    assert manifest["optimizer_recipe"]["stage_uses_muon"] is True
    assert manifest["exact_readiness_refusal"]["ready"] is False
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in manifest[
        "exact_readiness_refusal"
    ]["blockers"]
    assert set(EXACT_READINESS_REFUSAL_BLOCKERS).issubset(
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert normalized["training_backend"] == "mlx"
    assert normalized["scheduler_resource_kind"] == "local_mlx"
    _assert_false_authority(manifest)


def test_byte_closed_smoke_archive_is_deterministic_and_not_exact_ready(tmp_path: Path) -> None:
    manifest = run_pr95_mlx_synthetic_timing_smoke(
        stage_index=1,
        steps=1,
        batch_size=1,
        synthetic_pairs=1,
        seed=7,
        base_channels=4,
    )

    first = write_pr95_mlx_byte_closed_smoke_archive(manifest, output_dir=tmp_path / "a")
    second = write_pr95_mlx_byte_closed_smoke_archive(manifest, output_dir=tmp_path / "b")

    assert first["sha256"] == second["sha256"]
    assert first["runtime_consumption_proof_present"] is False
    assert first["receiver_proof_present"] is False
    _assert_false_authority(first)
    with zipfile.ZipFile(tmp_path / "a" / "archive.zip") as zf:
        assert zf.namelist() == ["0.bin"]
        payload = json.loads(zf.read("0.bin"))
    assert payload["schema"] == "pr95_hnerv_mlx_byte_closed_smoke_archive_v1"
    assert payload["exact_readiness_refusal"]["ready"] is False
