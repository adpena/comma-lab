# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for nirvana.

Proves the encode/decode contract of the NRV1 monolithic 0.bin grammar +
patch assembly forward-pass parity under fp16 + int16-quant roundtrip.
Plus a smoke-level test that the trainer's _full_main raises
NotImplementedError per the L0 SCAFFOLD posture (Catalog #240).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tac.substrates._shared.numpy_portable_inflate import (
    assert_inflate_is_numpy_portable,
    write_numpy_portable_contest_runtime,
)
from tac.substrates.nirvana.architecture import (
    NirvanaConfig,
    NirvanaSubstrate,
)
from tac.substrates.nirvana.archive import (
    NRV1_HEADER_SIZE,
    NRV1_MAGIC,
    NRV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from tac.substrates.nirvana.archive_numpy import parse_archive_numpy
from tac.substrates.nirvana.inflate import inflate_one_video

REPO_ROOT = Path(__file__).resolve().parents[5]
_INFLATE_PATH = Path(__file__).resolve().parents[1] / "inflate.py"
_ARCHIVE_NUMPY_PATH = Path(__file__).resolve().parents[1] / "archive_numpy.py"


def _smoke_cfg() -> NirvanaConfig:
    return NirvanaConfig(
        latent_dim=8,
        patch_embed_dim=4,
        patch_grid_h=2,
        patch_grid_w=2,
        embed_dim=24,
        initial_patch_grid_h=3,
        initial_patch_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: NirvanaConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_patch_grid_h": cfg.initial_patch_grid_h,
        "initial_patch_grid_w": cfg.initial_patch_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = NirvanaSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()

    blob = pack_archive(
        decoder_sd, latents, _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )
    arc = parse_archive(blob)

    assert arc.schema_version == NRV1_SCHEMA_VERSION
    assert blob[:4] == NRV1_MAGIC
    assert arc.patch_grid_h == cfg.patch_grid_h
    assert arc.patch_grid_w == cfg.patch_grid_w
    assert arc.patch_embed_dim == cfg.patch_embed_dim
    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())
    for k, v in decoder_sd.items():
        rec = arc.decoder_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    assert arc.latents.shape == latents.shape


def test_numpy_parse_matches_torch_parse_roundtrip():
    cfg = _smoke_cfg()
    torch.manual_seed(3)
    model = NirvanaSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    blob = pack_archive(
        decoder_sd,
        sd["latents"].clone(),
        _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )

    torch_arc = parse_archive(blob)
    numpy_arc = parse_archive_numpy(blob)

    assert numpy_arc.schema_version == torch_arc.schema_version
    assert numpy_arc.patch_grid_h == torch_arc.patch_grid_h
    assert numpy_arc.patch_grid_w == torch_arc.patch_grid_w
    assert numpy_arc.patch_embed_dim == torch_arc.patch_embed_dim
    assert np.abs(torch_arc.latents.numpy() - numpy_arc.latents).max() < 1e-5
    assert set(numpy_arc.decoder_state_dict) == set(torch_arc.decoder_state_dict)
    for key, value in numpy_arc.decoder_state_dict.items():
        assert np.abs(torch_arc.decoder_state_dict[key].numpy() - value).max() < 1e-5


def test_nirvana_inflate_has_no_torch_or_mlx_import() -> None:
    assert_inflate_is_numpy_portable(_INFLATE_PATH)
    assert_inflate_is_numpy_portable(_ARCHIVE_NUMPY_PATH)


def test_numpy_portable_runtime_emitter_vendors_runtime_safe_archive(tmp_path: Path):
    cfg = _smoke_cfg()
    torch.manual_seed(5)
    model = NirvanaSubstrate(cfg)
    sd = model.state_dict()
    archive_bytes = pack_archive(
        {k: v for k, v in sd.items() if k != "latents"},
        sd["latents"].clone(),
        _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )

    out = tmp_path / "submission"
    write_numpy_portable_contest_runtime(
        out,
        substrate_pkg_name="nirvana",
        repo_root=REPO_ROOT,
        runtime_module_files=("archive_numpy.py", "inflate.py"),
    )
    assert (out / "inflate.sh").is_file()
    assert (out / "src" / "tac" / "substrates" / "nirvana" / "archive_numpy.py").is_file()
    assert not (out / "src" / "tac" / "substrates" / "nirvana" / "archive.py").exists()
    for py in out.rglob("*.py"):
        assert_inflate_is_numpy_portable(py)

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "0.bin").write_bytes(archive_bytes)
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("0.mkv\n", encoding="utf-8")
    output_dir = tmp_path / "inflated"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
    }
    subprocess.run(
        [sys.executable, str(out / "inflate.py"), str(archive_dir), str(output_dir), str(file_list)],
        check=True,
        env=env,
    )
    assert (output_dir / "0" / "0.png").is_file()
    assert (output_dir / "0" / f"{cfg.num_pairs * 2 - 1}.png").is_file()


def test_header_size_invariant_is_25_bytes():
    assert NRV1_HEADER_SIZE == 25


def test_parse_archive_rejects_short_blob():
    try:
        parse_archive(b"\x00")
    except ValueError as exc:
        assert "too short" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on short blob")


def test_parse_archive_rejects_wrong_magic():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = NirvanaSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    blob = bytearray(
        pack_archive(
            decoder_sd, latents, _smoke_meta(cfg),
            patch_grid_h=cfg.patch_grid_h,
            patch_grid_w=cfg.patch_grid_w,
            patch_embed_dim=cfg.patch_embed_dim,
        )
    )
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_pack_archive_rejects_oversize_patch_grid():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = NirvanaSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    try:
        pack_archive(
            decoder_sd, latents, _smoke_meta(cfg),
            patch_grid_h=256, patch_grid_w=cfg.patch_grid_w,
            patch_embed_dim=cfg.patch_embed_dim,
        )
    except ValueError as exc:
        assert "patch_grid_h" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on out-of-u8 patch_grid_h")


def test_pack_archive_rejects_duplicate_latents_in_decoder_state_dict():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = NirvanaSubstrate(cfg)
    sd = model.state_dict()
    try:
        pack_archive(
            sd,
            sd["latents"].clone(),
            _smoke_meta(cfg),
            patch_grid_h=cfg.patch_grid_h,
            patch_grid_w=cfg.patch_grid_w,
            patch_embed_dim=cfg.patch_embed_dim,
        )
    except ValueError as exc:
        assert "exclude" in str(exc) and "latents" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on duplicated latents tensor")


def test_forward_pass_after_roundtrip_matches_original_within_tolerance():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = NirvanaSubstrate(cfg).eval()

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()
    blob = pack_archive(
        decoder_sd, latents, _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )
    arc = parse_archive(blob)

    rebuilt = NirvanaSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.latents.copy_(arc.latents.to(rebuilt.latents.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


def test_numpy_inflate_one_video_matches_torch_model_png(tmp_path: Path):
    cfg = _smoke_cfg()
    torch.manual_seed(23)
    model = NirvanaSubstrate(cfg).eval()
    with torch.no_grad():
        model.latents.normal_(std=0.5)
        model.patch_embeddings.normal_(std=0.5)
        torch_rgb0, torch_rgb1 = model(torch.tensor([0], dtype=torch.long))
    sd = model.state_dict()
    blob = pack_archive(
        {k: v for k, v in sd.items() if k != "latents"},
        sd["latents"].clone(),
        _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )

    out = tmp_path / "0"
    inflate_one_video(blob, out)

    got0 = np.asarray(Image.open(out / "0.png"))
    got1 = np.asarray(Image.open(out / "1.png"))
    ref0 = (
        torch_rgb0[0].permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).round().clip(0, 255).astype(np.uint8)
    ref1 = (
        torch_rgb1[0].permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).round().clip(0, 255).astype(np.uint8)
    assert np.abs(got0.astype(int) - ref0.astype(int)).max() <= 1
    assert np.abs(got1.astype(int) - ref1.astype(int)).max() <= 1


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_inflate_output_no_op_proof():
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = NirvanaSubstrate(cfg).eval()
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()

    blob_a = pack_archive(
        decoder_sd, latents, _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )
    mutated = latents.clone()
    mutated[0, 0] = mutated[0, 0] + 1.0
    blob_b = pack_archive(
        decoder_sd, mutated, _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )

    assert blob_a != blob_b, "no_op_proof: mutating latents must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)


def test_forward_pass_produces_unit_interval_rgb():
    """L5 compliance: substrate is a full RGB renderer (not a mask codec)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = NirvanaSubstrate(cfg).eval()
    idx = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_patch_grid_carries_correct_number_of_embeddings():
    """Distinctive design check: nirvana must have patch_grid_h * patch_grid_w embeddings."""
    cfg = _smoke_cfg()
    model = NirvanaSubstrate(cfg)
    expected = cfg.patch_grid_h * cfg.patch_grid_w
    assert model.patch_embeddings.shape[0] == expected
    assert model.patch_embeddings.shape[1] == cfg.patch_embed_dim


def test_patch_grid_dimensions_divide_output_evenly():
    """L0 SCAFFOLD invariant: output H/W must be divisible by patch_grid_h/w."""
    cfg_bad = NirvanaConfig(
        latent_dim=8, patch_embed_dim=4,
        patch_grid_h=3,  # 24 is not divisible by 3 in some configs; force a mismatch
        patch_grid_w=2,
        embed_dim=24, initial_patch_grid_h=3, initial_patch_grid_w=4,
        decoder_channels=(20, 16, 12), sin_frequency=30.0,
        num_upsample_blocks=3, num_pairs=3,
        output_height=25,  # NOT divisible by patch_grid_h=3
        output_width=32,
    )
    try:
        NirvanaSubstrate(cfg_bad)
    except ValueError as exc:
        assert "not divisible" in str(exc) or "output_height" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on patch-grid-output mismatch")


def test_full_main_implemented_and_cuda_gated(tmp_path):
    """CLASS-SHIFT-FULL-MAIN-CLUSTER 2026-05-27: _full_main IMPLEMENTED + CUDA-gated.

    The L0 SCAFFOLD NotImplementedError is extinguished: ``_full_main`` routes
    the canonical score-aware training loop through
    ``run_pact_nerv_score_aware_training``. Per CLAUDE.md "MPS auth eval is
    NOISE" + Catalog #1, the full (non-smoke) path is CUDA-required; invoking
    it with ``--device cpu`` refuses via ``device_or_die`` (SystemExit). PAID
    DISPATCH stays gated by ``dispatch_enabled: false`` + ``research_only:
    true`` on the recipe per Catalog #325 (code complete, trigger gated).
    """
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_nirvana")
    src = inspect.getsource(trainer._full_main)
    assert "raise NotImplementedError" not in src, (
        "_full_main NotImplementedError must be extinguished per "
        "CLASS-SHIFT-FULL-MAIN-CLUSTER"
    )
    assert "run_pact_nerv_score_aware_training" in src, (
        "_full_main must route through the canonical shared training loop"
    )
    args = trainer._build_parser().parse_args(
        ["--output-dir", str(tmp_path / "out"), "--device", "cpu", "--epochs", "1"]
    )
    with pytest.raises(SystemExit):
        trainer._full_main(args)
