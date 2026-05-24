# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    EXACT_READINESS_REFUSAL_BLOCKERS,
    FALSE_AUTHORITY,
    PR95_MLX_BACKEND_STATUS_SYNTHETIC_TIMING_ONLY,
    PR95_MLX_SOURCE_FAITHFUL_BLOCKERS,
    PR95_MLX_TRAINING_FIDELITY_SYNTHETIC_TIMING_ONLY,
    HNeRVDecoderMLX,
    HNeRVSyntheticTrainingBundleMLX,
    Pr95HNeRVMlxError,
    bilinear_resize2x_align_corners_false_nhwc,
    build_pr95_public_archive_member,
    compare_pr95_public_archive_forward_with_pytorch,
    load_pytorch_state_dict_into_mlx,
    parse_pr95_public_archive_member,
    parse_pr95_public_archive_zip,
    partition_pr95_mlx_parameter_names,
    pixel_shuffle_2x_nhwc,
    pr95_mlx_parameter_shape_records,
    pytorch_state_dict_from_mlx,
    run_pr95_mlx_synthetic_timing_smoke,
    stage_smoke_config,
    write_pr95_mlx_byte_closed_smoke_archive,
    write_pr95_public_archive_zip,
    zeropower_via_newtonschulz5_mlx,
)
from tac.optimization.local_training_runtime_profile import (  # noqa: E402
    normalize_runtime_profile_observation,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PR95_SOURCE_MODEL = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src/model.py"
)
PR95_RELEASE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto"
    / "archive.zip"
)


def _assert_false_authority(payload: dict) -> None:
    for key in FALSE_AUTHORITY:
        assert payload[key] is False


def _load_public_pr95_model_module():
    if not PR95_SOURCE_MODEL.is_file():
        pytest.skip("public PR95 source model.py is unavailable")
    spec = importlib.util.spec_from_file_location(
        "public_pr95_hnerv_model_for_mlx_parity",
        PR95_SOURCE_MODEL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pixel_shuffle_2x_nhwc_matches_pr95_layout() -> None:
    x = mx.array([[[[0.0, 1.0, 2.0, 3.0]]]])

    y = pixel_shuffle_2x_nhwc(x)
    mx.eval(y)

    assert y.shape == (1, 2, 2, 1)
    np.testing.assert_array_equal(np.asarray(y)[0, :, :, 0], np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_pixel_shuffle_2x_nhwc_matches_pytorch_channel_major_order() -> None:
    x = mx.array(np.arange(8, dtype=np.float32).reshape(1, 1, 1, 8))

    y = pixel_shuffle_2x_nhwc(x)
    mx.eval(y)

    assert y.shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(y)[0],
        np.array(
            [
                [[0.0, 4.0], [1.0, 5.0]],
                [[2.0, 6.0], [3.0, 7.0]],
            ],
            dtype=np.float32,
        ),
    )


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


def test_public_pr95_pytorch_state_load_matches_mlx_forward() -> None:
    torch = pytest.importorskip("torch")
    module = _load_public_pr95_model_module()

    torch.manual_seed(123)
    torch_model = module.HNeRVDecoder(
        latent_dim=28,
        base_channels=4,
        eval_size=(384, 512),
    ).eval()
    mlx_model = HNeRVDecoderMLX(latent_dim=28, base_channels=4)
    load_pytorch_state_dict_into_mlx(mlx_model, torch_model.state_dict())
    z_torch = torch.randn(1, 28)

    with torch.no_grad():
        torch_output = torch_model(z_torch).detach().cpu().numpy()
    mlx_output = mlx_model(mx.array(z_torch.detach().cpu().numpy()))
    mx.eval(mlx_output)

    diff = np.abs(torch_output - np.asarray(mlx_output))
    assert diff.max() <= 1e-4
    assert diff.mean() <= 1e-5


def test_parse_public_pr95_archive_packet_bytes() -> None:
    if not PR95_RELEASE_ARCHIVE.is_file():
        pytest.skip("public PR95 archive.zip is unavailable")

    packet = parse_pr95_public_archive_zip(PR95_RELEASE_ARCHIVE)
    with zipfile.ZipFile(PR95_RELEASE_ARCHIVE) as zf:
        source_member = zf.read("0.bin")
    member_state, member_latents, member_meta = parse_pr95_public_archive_member(
        source_member
    )
    custody = packet.custody_manifest()

    assert packet.archive_zip_sha256 == (
        "e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a"
    )
    assert packet.member_sha256 == (
        "4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4"
    )
    assert packet.member_bytes == 178_309
    assert packet.meta == {
        "n_pairs": 600,
        "latent_dim": 28,
        "base_channels": 36,
        "eval_size": [384, 512],
    }
    assert packet.latents.shape == (600, 28)
    assert len(packet.state_dict) == 28
    assert member_meta == packet.meta
    assert member_latents.shape == packet.latents.shape
    assert member_state.keys() == packet.state_dict.keys()
    assert custody["schema"] == "pr95_hnerv_public_archive_packet.v1"
    assert custody["state_dict_tensor_count"] == 28
    _assert_false_authority(custody)


def test_public_pr95_archive_packet_mlx_cpu_forward_parity_probe() -> None:
    pytest.importorskip("torch")
    if not PR95_RELEASE_ARCHIVE.is_file():
        pytest.skip("public PR95 archive.zip is unavailable")
    module = _load_public_pr95_model_module()
    packet = parse_pr95_public_archive_zip(PR95_RELEASE_ARCHIVE)

    result = compare_pr95_public_archive_forward_with_pytorch(
        packet,
        module.HNeRVDecoder,
        sample_indices=[0],
        mlx_device="cpu",
    )

    assert result["schema"] == "pr95_hnerv_public_archive_mlx_forward_parity.v1"
    assert result["sample_indices"] == [0]
    assert result["mlx_device"] == "cpu"
    assert result["parity"]["passed"] is True
    assert result["parity"]["max_abs"] <= 2e-3
    assert result["parity"]["mean_abs"] <= 1e-4
    assert result["exact_readiness_refusal"]["ready"] is False
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in result[
        "exact_readiness_refusal"
    ]["blockers"]
    _assert_false_authority(result)


def test_build_public_pr95_archive_member_round_trips_packet_grammar(
    tmp_path: Path,
) -> None:
    if not PR95_RELEASE_ARCHIVE.is_file():
        pytest.skip("public PR95 archive.zip is unavailable")
    packet = parse_pr95_public_archive_zip(PR95_RELEASE_ARCHIVE)

    rebuilt_member = build_pr95_public_archive_member(
        packet.state_dict,
        packet.latents,
        meta=packet.meta,
    )
    rebuilt_state, rebuilt_latents, rebuilt_meta = parse_pr95_public_archive_member(
        rebuilt_member
    )
    summary = write_pr95_public_archive_zip(
        packet.state_dict,
        packet.latents,
        meta=packet.meta,
        output_zip_path=tmp_path / "archive.zip",
    )
    summary_again = write_pr95_public_archive_zip(
        packet.state_dict,
        packet.latents,
        meta=packet.meta,
        output_zip_path=tmp_path / "again" / "archive.zip",
    )
    reparsed_zip = parse_pr95_public_archive_zip(tmp_path / "archive.zip")

    assert hashlib.sha256(rebuilt_member).hexdigest() == (
        "4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4"
    )
    assert rebuilt_meta == packet.meta
    assert rebuilt_latents.shape == packet.latents.shape
    assert rebuilt_state.keys() == packet.state_dict.keys()
    assert np.isfinite(rebuilt_latents).all()
    assert abs(float(rebuilt_latents[0, 0]) - float(packet.latents[0, 0])) < 0.05
    assert summary["schema"] == "pr95_hnerv_archive_export.v1"
    assert summary["member_name"] == "0.bin"
    assert summary["member_compress_type"] == zipfile.ZIP_STORED
    assert summary["parsed_latent_shape"] == [600, 28]
    assert summary["archive_zip_sha256"] == summary_again["archive_zip_sha256"]
    assert summary["member_sha256"] == summary_again["member_sha256"]
    assert summary["runtime_consumption_proof_present"] is False
    assert summary["exact_readiness_refusal"]["ready"] is False
    assert reparsed_zip.meta == packet.meta
    with zipfile.ZipFile(tmp_path / "archive.zip") as zf:
        assert zf.namelist() == ["0.bin"]
        assert zf.getinfo("0.bin").compress_type == zipfile.ZIP_STORED
        assert zf.comment == b""
    _assert_false_authority(summary)


def test_public_pr95_archive_export_fails_closed_on_invalid_inputs() -> None:
    if not PR95_RELEASE_ARCHIVE.is_file():
        pytest.skip("public PR95 archive.zip is unavailable")
    packet = parse_pr95_public_archive_zip(PR95_RELEASE_ARCHIVE)

    with pytest.raises(Pr95HNeRVMlxError, match="non-empty"):
        build_pr95_public_archive_member(
            packet.state_dict,
            np.zeros((0, 28), dtype=np.float32),
            meta={"n_pairs": 0, "latent_dim": 28, "base_channels": 36},
        )
    with pytest.raises(Pr95HNeRVMlxError, match="non-finite"):
        bad_latents = packet.latents.copy()
        bad_latents[0, 0] = np.nan
        build_pr95_public_archive_member(packet.state_dict, bad_latents, meta=packet.meta)
    with pytest.raises(Pr95HNeRVMlxError, match="key mismatch"):
        bad_state = dict(packet.state_dict)
        bad_state.pop("rgb_1.bias")
        build_pr95_public_archive_member(bad_state, packet.latents, meta=packet.meta)
    with pytest.raises(Pr95HNeRVMlxError, match="shape"):
        bad_state = dict(packet.state_dict)
        bad_state["rgb_1.bias"] = np.zeros((4,), dtype=np.float32)
        build_pr95_public_archive_member(bad_state, packet.latents, meta=packet.meta)
    with pytest.raises(Pr95HNeRVMlxError, match="non-finite"):
        bad_state = dict(packet.state_dict)
        bad_state["rgb_1.bias"] = bad_state["rgb_1.bias"].copy()
        bad_state["rgb_1.bias"][0] = np.inf
        build_pr95_public_archive_member(bad_state, packet.latents, meta=packet.meta)


def test_public_pr95_runtime_consumption_cli_accepts_native_export(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    module = _load_public_pr95_model_module()
    torch.manual_seed(44)
    model = module.HNeRVDecoder(
        latent_dim=28,
        base_channels=4,
        eval_size=(384, 512),
    ).eval()
    latents = torch.randn(1, 28)
    archive_summary = write_pr95_public_archive_zip(
        model.state_dict(),
        latents,
        meta={
            "n_pairs": 1,
            "latent_dim": 28,
            "base_channels": 4,
            "eval_size": [384, 512],
        },
        output_zip_path=tmp_path / "archive.zip",
    )

    output_json = tmp_path / "runtime_consumption_proof.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/prove_pr95_public_archive_runtime_consumption.py"),
            "--archive-zip",
            str(tmp_path / "archive.zip"),
            "--output-json",
            str(output_json),
            "--max-output-bytes",
            "10000000",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    proof = json.loads(output_json.read_text())
    assert proof["schema"] == "pr95_hnerv_public_runtime_consumption_proof.v1"
    assert proof["runtime_consumption_proven"] is True
    assert proof["expected_raw_bytes"] == 2 * 874 * 1164 * 3
    assert proof["raw_output_bytes"] == proof["expected_raw_bytes"]
    assert proof["raw_output_sha256"]
    assert proof["archive_packet"]["member_sha256"] == archive_summary["member_sha256"]
    assert proof["exact_readiness_refusal"]["ready"] is False
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in proof[
        "exact_readiness_refusal"
    ]["blockers"]
    _assert_false_authority(proof)


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
    assert manifest["training_fidelity"] == (
        PR95_MLX_TRAINING_FIDELITY_SYNTHETIC_TIMING_ONLY
    )
    assert manifest["source_faithful_training"] is False
    assert manifest["source_faithfulness_blockers"] == list(
        PR95_MLX_SOURCE_FAITHFUL_BLOCKERS
    )
    assert "pr95_eval_roundtrip_scorer_preprocess_loss_not_ported_to_mlx" not in (
        manifest["source_faithfulness_blockers"]
    )
    assert (
        "pr95_eval_roundtrip_yuv6_preprocess_ported_but_scorer_loss_not_wired_to_mlx"
        in manifest["source_faithfulness_blockers"]
    )
    assert manifest["optimizer_recipe"]["stage_uses_muon"] is True
    assert manifest["optimizer_recipe"]["source_faithful_training"] is False
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


def test_pr95_optimizer_descriptor_drives_stage8_partition() -> None:
    stage = stage_smoke_config(8)

    assert stage.optimizer_descriptor_id == "pr95_stage8_muon_adamw_mlx"
    assert stage.optimizer.use_muon is True
    assert stage.optimizer_config_sha256
    assert stage.parameter_group_lr_policy_id == "embedding_theta1_hidden_muon_adamw"
    assert stage.optimizer_backend_status == (
        PR95_MLX_BACKEND_STATUS_SYNTHETIC_TIMING_ONLY
    )

    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=2,
        latent_dim=4,
        base_channels=4,
        seed=3,
    )
    fingerprint_records = pr95_mlx_parameter_shape_records(bundle.parameters())
    assert any(record["name"].endswith("latents") for record in fingerprint_records)

    with pytest.raises(Pr95HNeRVMlxError, match="not executable on MLX"):
        stage_smoke_config(
            8,
            optimizer_descriptor_id="pr95_langevin_stage8_polish_descriptor_only",
        )


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
