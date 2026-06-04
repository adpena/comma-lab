# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from tac.repo_io import sha256_file
from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate
from tac.substrates.hi_nerv.archive import pack_archive
from tac.substrates.hi_nerv.receiver_cache_quality import (
    HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA,
    write_hi_nerv_receiver_cache_quality_report,
)


def test_hi_nerv_receiver_cache_quality_writes_direct_cache_from_archive(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(tmp_path / "archive.zip")

    report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "quality",
        max_pairs=1,
        batch_pairs=1,
    )

    cache_dir = Path(report["candidate_cache_dir"])
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (cache_dir / "hi_nerv_direct_receiver_render_cache_identity_audit.json").read_text(
            encoding="utf-8"
        )
    )
    pair_indices = np.load(cache_dir / "pair_indices.npy")

    assert report["schema"] == HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA
    assert report["archive_sha256"] == sha256_file(archive)
    assert report["quality_gate"] is None
    assert report["quality_gate_passed"] is False
    assert "hi_nerv_receiver_cache_quality_reference_gate_not_run" in report[
        "blockers"
    ]
    assert manifest["source_kind"] == "hi_nerv_direct_receiver_render"
    assert manifest["pair_count"] == 1
    assert pair_indices.tolist() == [[0, 1]]
    assert audit["source"]["archive_magic"] == "HIV1"
    assert audit["cache"]["raw_sha256"] == manifest["raw_sha256"]
    assert audit["score_claim"] is False


def test_hi_nerv_receiver_cache_quality_attaches_gate_against_reference(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(tmp_path / "archive.zip")
    reference_report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "reference",
        max_pairs=1,
        batch_pairs=1,
    )

    report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "candidate",
        reference_cache_dir=Path(reference_report["candidate_cache_dir"]),
        max_pairs=1,
        batch_pairs=1,
        min_segnet_std=0.0,
        min_segnet_dynamic_range=0.0,
        max_segnet_mae_vs_reference_for_fit_gate=1.0,
        min_posenet_yuv6_std=0.0,
        min_posenet_yuv6_dynamic_range=0.0,
        max_posenet_yuv6_mae_vs_reference_for_fit_gate=1.0,
    )

    assert report["quality_gate"] is not None
    assert report["quality_gate"]["schema"] == "mlx_cache_quality_gate.v1"
    assert report["quality_gate"]["verdict"] == "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY"
    assert report["quality_gate_passed"] is True
    assert report["score_claim"] is False
    assert Path(report["quality_gate_path"]).is_file()


def _write_tiny_hiv1_archive(path: Path) -> Path:
    cfg = HinervConfig(
        latent_dim_coarse=2,
        latent_dim_mid=2,
        latent_dim_fine=2,
        embed_dim=2,
        initial_grid_h=1,
        initial_grid_w=1,
        decoder_channels=(2, 2, 2),
        sin_frequency=3.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=2,
        output_height=8,
        output_width=8,
    )
    torch.manual_seed(7)
    model = HinervSubstrate(cfg).eval()
    decoder_state = {
        key: value
        for key, value in dict(model.state_dict()).items()
        if key not in {"latents_coarse", "latents_mid", "latents_fine"}
    }
    packet = pack_archive(
        decoder_state,
        model.latents_coarse.detach(),
        model.latents_mid.detach(),
        model.latents_fine.detach(),
        {
            "embed_dim": cfg.embed_dim,
            "initial_grid_h": cfg.initial_grid_h,
            "initial_grid_w": cfg.initial_grid_w,
            "decoder_channels": list(cfg.decoder_channels),
            "sin_frequency": cfg.sin_frequency,
            "num_upsample_blocks": cfg.num_upsample_blocks,
            "mid_injection_block_index": cfg.mid_injection_block_index,
            "fine_injection_block_index": cfg.fine_injection_block_index,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
        },
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", packet)
    return path
