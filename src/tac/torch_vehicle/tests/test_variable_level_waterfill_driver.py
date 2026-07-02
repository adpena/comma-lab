# SPDX-License-Identifier: MIT
"""Driver wire-in tests for Track-A Item B / D2 variable-level waterfill.

These tests are deliberately driver-facing: the core allocator and codec already have
their own no-fake suites. Here we guard the production adapter contract:

* D2 defaults off and the archive bytes are identical to the vendored path.
* D2 uses the conservative byte target with ``net_stop=False``.
* D2 fails closed on non-base_ch20 or missing/measured RD custody.
* D2 composes with the active FiLM path and remains receiver-parseable.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import brotli
import pytest
import torch

from tac.losses.variable_level_waterfill_allocator import (
    byte_saving_score_value,
    solve_waterfill_allocation,
)
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    _split_three_section_archive,
    import_vendored_bundle,
)
from tac.torch_vehicle.pose_film import (
    PoseFiLMHNeRVWrapper,
    inflate_film_decoder,
    parse_pose_section,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext

REPO_ROOT = Path(__file__).resolve().parents[4]
ADAPTER_INFLATE = (
    REPO_ROOT
    / "experiments/public_runtime_adapters/torch_vehicle_d2_hnerv_adapter/inflate.py"
)
ARCHIVE_ZIP_BUILDER = REPO_ROOT / "tools/build_torch_vehicle_d2_archive_zip.py"


def _archive_meta(archive: bytes) -> dict:
    meta_brotli, _decoder_blob, _latents_brotli, _trailing = _split_three_section_archive(
        archive
    )
    return json.loads(brotli.decompress(meta_brotli))


def _base_state_and_latents(v, *, n_pairs: int = 4, seed: int = 7):
    dec = v.HNeRVDecoder(latent_dim=28, base_channels=20)
    ema_sd = {k: t.detach().cpu().clone() for k, t in dec.state_dict().items()}
    latents = torch.randn(n_pairs, 28, generator=torch.Generator().manual_seed(seed)) * 0.1
    meta = {
        "n_pairs": n_pairs,
        "latent_dim": 28,
        "base_channels": 20,
        "eval_size": [384, 512],
    }
    return ema_sd, latents, meta


def _single_tensor_rd_table(tensor_name: str) -> dict[str, dict[int, tuple[float, float]]]:
    """A measured-shape table where byte_target=2731 picks 96 but net_stop picks 64."""
    bv = byte_saving_score_value(1.0)
    first = 2731.0
    second_total = 29614.0
    return {
        tensor_name: {
            127: (0.0, 0.0),
            96: (first, 0.1 * bv * first),
            64: (second_total, 0.1 * bv * first + 0.2 * bv * (second_total - first)),
        }
    }


def _d2_film_archive(*, n_pairs: int = 1) -> bytes:
    v = import_vendored_bundle()
    dec = v.HNeRVDecoder(latent_dim=28, base_channels=20)
    wrapper = PoseFiLMHNeRVWrapper(dec, n_pairs=n_pairs, film_hidden=8)
    gen = torch.Generator().manual_seed(19)
    for p in wrapper.pose_film.parameters():
        p.data = p.data + torch.randn(p.shape, generator=gen) * 0.05
    pose = torch.randn(n_pairs, 6, generator=torch.Generator().manual_seed(23)) * 0.25
    wrapper.set_stored_pose(pose)
    ema_sd = {k: t.detach().cpu().clone() for k, t in wrapper.state_dict().items()}
    latents = torch.randn(n_pairs, 28, generator=torch.Generator().manual_seed(29)) * 0.1
    meta = {
        "n_pairs": n_pairs,
        "latent_dim": 28,
        "base_channels": 20,
        "eval_size": [384, 512],
    }
    tensor_name = next(k for k in ema_sd if k.startswith("decoder."))
    rd_table = _single_tensor_rd_table(tensor_name.removeprefix("decoder."))

    with tempfile.TemporaryDirectory() as td:
        cfg = TorchVehicleConfig(
            base_channels=20,
            latent_dim=28,
            out_dir=Path(td),
            checkpoint_every_epochs=1,
            device="cpu",
            seed=5,
            pose_film_enabled=True,
            pose_film_hidden=8,
            variable_level_waterfill_enabled=True,
            variable_level_waterfill_rd_table=rd_table,
            variable_level_waterfill_byte_target=2731.0,
        )
        sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=5)
        drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[])
        archive, _eval_dec, _eval_latents = drv._build_archive_and_eval_decoder(
            ema_sd, latents, meta
        )
    return archive


def test_d2_default_off_archive_is_byte_identical_to_vendored() -> None:
    v = import_vendored_bundle()
    with tempfile.TemporaryDirectory() as td:
        cfg = TorchVehicleConfig(
            base_channels=20,
            latent_dim=28,
            out_dir=Path(td),
            checkpoint_every_epochs=1,
            device="cpu",
            seed=5,
        )
        sc = SyntheticScorerContext(n_pairs=4, device="cpu", seed=5)
        drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[])
        ema_sd, latents, meta = _base_state_and_latents(v, n_pairs=4)
        meta_for_legacy = dict(meta)

        archive, eval_dec, eval_latents = drv._build_archive_and_eval_decoder(
            ema_sd, latents, meta
        )
        legacy = v.build_archive(ema_sd, latents, meta_dict=meta_for_legacy)

        assert archive == legacy
        assert type(eval_dec).__name__ == "HNeRVDecoder"
        assert torch.equal(eval_latents, v.parse_archive(legacy)[1])
        assert "variable_level_waterfill" not in _archive_meta(archive)


def test_d2_uses_conservative_byte_target_not_falsified_net_stop() -> None:
    v = import_vendored_bundle()
    ema_sd, latents, meta = _base_state_and_latents(v, n_pairs=4)
    tensor_name = next(iter(ema_sd))
    rd_table = _single_tensor_rd_table(tensor_name)
    target_alloc = solve_waterfill_allocation(
        rd_table, byte_target=2731.0, net_stop=False
    )
    net_stop_alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    assert target_alloc.levels[tensor_name] == 96
    assert net_stop_alloc.levels[tensor_name] == 64

    with tempfile.TemporaryDirectory() as td:
        cfg = TorchVehicleConfig(
            base_channels=20,
            latent_dim=28,
            out_dir=Path(td),
            checkpoint_every_epochs=1,
            device="cpu",
            seed=5,
            variable_level_waterfill_enabled=True,
            variable_level_waterfill_rd_table=rd_table,
            variable_level_waterfill_byte_target=2731.0,
        )
        sc = SyntheticScorerContext(n_pairs=4, device="cpu", seed=5)
        drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[])

        archive, eval_dec, eval_latents = drv._build_archive_and_eval_decoder(
            ema_sd, latents, meta
        )
        got_meta = _archive_meta(archive)
        d2 = got_meta["variable_level_waterfill"]

        assert got_meta["decoder_codec"] == "variable_level_waterfill.v1"
        assert d2["net_stop"] is False
        assert d2["byte_target"] == 2731.0
        assert d2["falsified_path"] == "net_stop"
        assert d2["levels"] == {tensor_name: 96}
        assert d2["levels"][tensor_name] != net_stop_alloc.levels[tensor_name]
        assert d2["decoder_blob_is_variable_format"] is True
        assert type(eval_dec).__name__ == "HNeRVDecoder"
        assert tuple(eval_latents.shape) == tuple(latents.shape)


def test_d2_fail_closed_requires_base_ch20_measured_rd_table(tmp_path: Path) -> None:
    rd_table = {"stem.weight": {127: (0.0, 0.0)}}
    with pytest.raises(ValueError, match="base_ch20"):
        TorchVehicleConfig(
            base_channels=8,
            out_dir=tmp_path / "bad_ch",
            variable_level_waterfill_enabled=True,
            variable_level_waterfill_rd_table=rd_table,
        )
    with pytest.raises(ValueError, match="requires a measured RD table"):
        TorchVehicleConfig(
            base_channels=20,
            out_dir=tmp_path / "missing_table",
            variable_level_waterfill_enabled=True,
            variable_level_waterfill_rd_table=None,
        )

    v = import_vendored_bundle()
    ema_sd, latents, meta = _base_state_and_latents(v, n_pairs=2)
    cfg = TorchVehicleConfig(
        base_channels=20,
        out_dir=tmp_path / "missing_tensor",
        variable_level_waterfill_enabled=True,
        variable_level_waterfill_rd_table={
            "not_in_decoder": {127: (0.0, 0.0), 96: (2731.0, 0.0)}
        },
    )
    sc = SyntheticScorerContext(n_pairs=2, device="cpu", seed=5)
    drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[])
    with pytest.raises(ValueError, match="absent from the archive decoder state dict"):
        drv._build_archive_and_eval_decoder(ema_sd, latents, meta)


def test_d2_composes_with_pose_film_and_receiver_parseback() -> None:
    v = import_vendored_bundle()
    n_pairs = 3
    dec = v.HNeRVDecoder(latent_dim=28, base_channels=20)
    wrapper = PoseFiLMHNeRVWrapper(dec, n_pairs=n_pairs, film_hidden=8)
    gen = torch.Generator().manual_seed(19)
    for p in wrapper.pose_film.parameters():
        p.data = p.data + torch.randn(p.shape, generator=gen) * 0.05
    pose = torch.randn(n_pairs, 6, generator=torch.Generator().manual_seed(23)) * 0.25
    wrapper.set_stored_pose(pose)
    ema_sd = {k: t.detach().cpu().clone() for k, t in wrapper.state_dict().items()}
    latents = torch.randn(n_pairs, 28, generator=torch.Generator().manual_seed(29)) * 0.1
    meta = {
        "n_pairs": n_pairs,
        "latent_dim": 28,
        "base_channels": 20,
        "eval_size": [384, 512],
    }
    tensor_name = next(k for k in ema_sd if k.startswith("decoder."))
    rd_table = _single_tensor_rd_table(tensor_name.removeprefix("decoder."))

    with tempfile.TemporaryDirectory() as td:
        cfg = TorchVehicleConfig(
            base_channels=20,
            latent_dim=28,
            out_dir=Path(td),
            checkpoint_every_epochs=1,
            device="cpu",
            seed=5,
            pose_film_enabled=True,
            pose_film_hidden=8,
            variable_level_waterfill_enabled=True,
            variable_level_waterfill_rd_table=rd_table,
            variable_level_waterfill_byte_target=2731.0,
        )
        sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=5)
        drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[])
        archive, eval_dec, eval_latents = drv._build_archive_and_eval_decoder(
            ema_sd, latents, meta
        )

        got_meta = _archive_meta(archive)
        assert got_meta["variable_level_waterfill"]["net_stop"] is False
        parsed_pose = parse_pose_section(archive, v.parse_archive)
        assert parsed_pose is not None
        assert tuple(parsed_pose.shape) == (n_pairs, 6)
        assert type(eval_dec).__name__ == "_FiLMEvalDecoder"

        # Receiver parse-back must dispatch the variable decoder blob, recover FiLM
        # weights, and render the same pair shape without using the falsified net_stop
        # path or vendored-only decoder parser.
        frames = inflate_film_decoder(
            archive, v.parse_archive, v.HNeRVDecoder, film_hidden=8, device="cpu"
        )
        assert tuple(frames.shape) == (n_pairs, 2, 3, 384, 512)
        assert tuple(eval_latents.shape) == tuple(latents.shape)


def test_d2_public_runtime_adapter_inflates_variable_film_archive(tmp_path: Path) -> None:
    archive = _d2_film_archive(n_pairs=1)
    input_bin = tmp_path / "0.bin"
    output_raw = tmp_path / "output.raw"
    input_bin.write_bytes(archive)

    result = subprocess.run(
        [sys.executable, str(ADAPTER_INFLATE), str(input_bin), str(output_raw)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "saved 2 frames" in result.stdout
    assert output_raw.stat().st_size == 1 * 2 * 874 * 1164 * 3


def test_d2_archive_zip_builder_is_deterministic(tmp_path: Path) -> None:
    archive = _d2_film_archive(n_pairs=1)
    input_bin = tmp_path / "0.bin"
    zip_a = tmp_path / "archive_a.zip"
    zip_b = tmp_path / "archive_b.zip"
    manifest_a = tmp_path / "manifest_a.json"
    manifest_b = tmp_path / "manifest_b.json"
    input_bin.write_bytes(archive)

    for output_zip, manifest in ((zip_a, manifest_a), (zip_b, manifest_b)):
        subprocess.run(
            [
                sys.executable,
                str(ARCHIVE_ZIP_BUILDER),
                "--input-bin",
                str(input_bin),
                "--output-zip",
                str(output_zip),
                "--manifest-json",
                str(manifest),
            ],
            check=True,
            text=True,
            capture_output=True,
        )

    assert zip_a.read_bytes() == zip_b.read_bytes()
    got_manifest = json.loads(manifest_a.read_text(encoding="utf-8"))
    got_manifest_b = json.loads(manifest_b.read_text(encoding="utf-8"))
    for key in (
        "input_bytes",
        "input_sha256",
        "archive_zip_bytes",
        "archive_zip_sha256",
        "member_name",
        "member_compression",
        "deterministic_mtime",
    ):
        assert got_manifest[key] == got_manifest_b[key]
    assert got_manifest["member_compression"] == "stored"
    assert got_manifest["member_name"] == "0.bin"
    with zipfile.ZipFile(zip_a) as zf:
        info = zf.getinfo("0.bin")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert zf.read("0.bin") == archive
